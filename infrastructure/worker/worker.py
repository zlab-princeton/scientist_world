"""Base Worker class — concrete base with shared implementation.

Subclasses (ClaudeCodeWorker, CodexWorker) override only the
scaffold-specific methods: CLI invocation, command building,
workspace setup, credential management, env configuration,
and stats accumulation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from infrastructure.worker.parsing import (
    TurnAction,
    parse_current_milestone,
    parse_turn_action,
)
from infrastructure.worker.prompts import (
    build_budget_exhausted_prompt,
    build_container_build_failure_prompt,
    build_job_cancelled_prompt,
    build_job_completion_prompt,
    build_job_progress_prompt,
    build_reproduce_prompt,
)

logger = logging.getLogger(__name__)

ACTION_FIX_PROMPT = """\
The system could not parse your action.yaml file. \
Please edit action.yaml in your workspace root so it contains a valid action. \
Example:

action: submit
gpus: 1
milestone: none
notes: "what this job does"

Or:

action: done
milestone: none
notes: "why you are done"

Edit action.yaml now."""

# Files hidden from the agent inside the container via /dev/null bind.
HIDDEN_FILES = ("manifest.json", "README.md", "protocol.md",
                "environment/container.final.def")
# Directories hidden from the agent (bind-mounted as empty tmpfs).
HIDDEN_DIRS = ("trajectory", ".claude-tmp")
# Files the agent can read and execute but not modify (bind as self:ro).
READONLY_FILES = ("validate.py",)

# Per-container resource limits (memory cgroup, nice value, thread cap) come
# from the worker config keys container_memory_limit, container_nice and
# container_cpu_threads, which Swarm fills from its own configuration.

# Canonical container paths — both agent and GPU job containers present
# the same namespace so the agent never sees host paths.
CONTAINER_WORKSPACE = "/home/user"
CONTAINER_SHARED_DATASETS = "/home/user/shared/datasets"
CONTAINER_SHARED_MODELS = "/home/user/shared/models"
WORKSPACE_CLAUDE_TMPDIR = ".claude-tmp"
CONTAINER_CLAUDE_TMPDIR = "/tmp/claude-code-tmp"

# Secret redaction — applied to all trajectory output before it lands on disk,
# since trajectories are tracked in git and those repos may eventually go public.
# Patterns cover GitHub PATs, HuggingFace tokens, and userinfo in URLs
# (e.g. https://user:token@host/...). Inline replacement preserves line length
# modulo the matched substring, so JSON structure and byte offsets elsewhere
# are unaffected.
_SECRET_PATTERNS = (
    (re.compile(r"ghp_[A-Za-z0-9]{30,}"), "[REDACTED_GITHUB_PAT]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{60,}"), "[REDACTED_GITHUB_PAT]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "[REDACTED_HF_TOKEN]"),
    # https://user:token@host  →  https://[REDACTED]@host
    (re.compile(r"(https?://)[^/\s\"'@]+:[^/\s\"'@]+@"), r"\1[REDACTED]@"),
)


def redact_secrets(text: str) -> str:
    """Strip known credential patterns from a string, preserving structure.

    Cheap — 4 compiled regex passes, microseconds on KB-sized lines.
    Safe to call on JSON/JSONL text: only matched token substrings are
    replaced, surrounding quotes and delimiters stay intact.
    """
    if not text:
        return text
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# Patterns that indicate a transient API error worth retrying.
_RETRYABLE_PATTERNS = (
    "rate limit", "rate_limit", "ratelimit",
    "overloaded", "too many requests", "429",
    "quota exceeded", "capacity", "try again",
    "server error", "500", "502", "503", "504", "529",
    "enotconn", "socket is not connected",
    "connection reset", "broken pipe", "transport endpoint",
    "connection refused", "timeout", "timed out",
    "deadline exceeded", "temporarily unavailable",
)

# Hard-limit patterns that should NOT be retried even if they match
# a retryable pattern.  These indicate quota exhaustion that won't
# clear for hours/days/weeks — retrying just wastes slots. The caller
# rotates API keys or falls back to another profile instead.
_NON_RETRYABLE_PATTERNS = (
    "weekly/monthly limit",
    "limit exhausted",
    "monthly limit",
    "weekly limit",
    "usage limit reached",
    "prompt is too long",                # context overflow — fall back to larger-context profile
    # Account-wide per-minute rate limit as reported by one gateway. It
    # persists under high parallelism, so falling back beats retrying.
    '"code":"1302"',
    "rate limit reached for requests",
)


def _get_descendants(pid: int) -> list[int]:
    """Return all descendant PIDs of *pid* (children, grandchildren, …).

    Uses /proc on Linux — no external dependencies.  Returns an empty
    list on any error (non-Linux, process already dead, permission).
    """
    try:
        children_of: dict[int, list[int]] = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                stat = open(f"/proc/{entry}/stat", "rb").read()
                # Field 4 is ppid: "pid (comm) state ppid ..."
                ppid = int(stat.split(b")")[1].split()[1])
                children_of.setdefault(ppid, []).append(int(entry))
            except (OSError, IndexError, ValueError):
                continue
        # BFS from pid
        result: list[int] = []
        queue = children_of.get(pid, [])
        while queue:
            child = queue.pop()
            result.append(child)
            queue.extend(children_of.get(child, []))
        return result
    except Exception:
        return []


class Worker(ABC):
    """Concrete base class for agent workers.

    Provides all shared lifecycle, event handling, action parsing,
    trajectory management, and container command building.
    Subclasses implement scaffold-specific hooks.
    """

    # ── abstract hooks (subclasses must implement) ─────────────────────────

    @staticmethod
    @abstractmethod
    def _config_defaults() -> dict[str, Any]:
        """Return scaffold-specific default config values."""

    @staticmethod
    @abstractmethod
    def _initial_cumulative_stats() -> dict:
        """Return the initial cumulative stats dict for this scaffold."""

    @abstractmethod
    def _invoke(self, prompt: str) -> str:
        """Run a single CLI turn and return the agent's text response.

        Must update self._session_id, self._trajectory, self._turn_count,
        self._cumulative_stats, and self._status as appropriate.
        Does NOT read action.yaml — that is handled by
        _invoke_and_read_action().
        """

    @abstractmethod
    def _build_command(self) -> list[str]:
        """Build the CLI argument list for this scaffold."""

    @abstractmethod
    def _setup_workspace(self) -> None:
        """Prepare the workspace before the first turn."""

    @abstractmethod
    def _copy_credentials(self, ws: Path) -> None:
        """Copy host credentials into the workspace for container auth."""

    @abstractmethod
    def _add_scaffold_container_binds(self, cmd: list[str]) -> None:
        """Append scaffold-specific bind-mount flags to the container cmd."""

    @abstractmethod
    def _configure_env(self, env: dict[str, str]) -> None:
        """Apply scaffold-specific environment modifications in-place."""

    @abstractmethod
    def _accumulate_stats(self, turn_stats: dict) -> None:
        """Merge a single turn's stats into self._cumulative_stats."""

    # ── lifecycle ──────────────────────────────────────────────────────────

    def __init__(self, scaffold: str, model: str, config: dict[str, Any]):
        self.scaffold = scaffold
        self.model = model
        self.config = {**self._config_defaults(), **config}
        self._api_key_index: int = 0
        self._clear_state()

    def set_paper_and_workspace(self, paper: Any, workspace_path: str) -> None:
        self._paper = paper
        self._workspace_path = workspace_path

    def set_resources(
        self,
        gpu_budget: float,
        max_gpus: int = 4,
        job_time_limit: str = "24:00:00",
        shared_datasets_path: str = "/shared/datasets",
        shared_models_path: str = "/shared/models",
        uv_cache_dir: str = "",
    ) -> None:
        self._gpu_budget = gpu_budget
        self._max_gpus = max_gpus
        self._job_time_limit = job_time_limit
        self._shared_datasets_path = shared_datasets_path
        self._shared_models_path = shared_models_path
        self._uv_cache_dir = uv_cache_dir

    def get_status(self) -> Literal["idle", "working", "waiting", "done", "error"]:
        return self._status

    def get_error(self) -> str | None:
        return self._error_message

    def _clear_state(self) -> None:
        """Reset all mutable state to initial values."""
        self._paper: Any = None
        self._workspace_path: str | None = None
        self._gpu_budget: float = 10.0
        self._max_gpus: int = 4
        self._job_time_limit: str = "24:00:00"
        self._shared_datasets_path: str = "/shared/datasets"
        self._shared_models_path: str = "/shared/models"
        self._uv_cache_dir: str = ""
        self._session_id: str | None = None
        self._status: Literal["idle", "working", "waiting", "done", "error"] = "idle"
        self._error_message: str | None = None
        self._last_action: TurnAction | None = None
        self._trajectory: list[dict] = []
        self._cumulative_stats: dict = self._initial_cumulative_stats()
        self._turn_count: int = 0
        self._active_proc: subprocess.Popen | None = None
        self._cancel_event: threading.Event = threading.Event()

    def request_cancel(self) -> None:
        """Signal the worker to abort retry sleeps and exit promptly."""
        self._cancel_event.set()

    def _interruptible_sleep(self, seconds: float) -> bool:
        """Sleep for *seconds*, returning True if cancelled early."""
        return self._cancel_event.wait(timeout=seconds)

    def terminate(self) -> None:
        """Mark as done and kill any active subprocess."""
        self._status = "done"
        self._cancel_event.set()
        self.kill_subprocess()

    def kill_subprocess(self) -> None:
        """Kill the active CLI subprocess and its entire process tree.

        Sends SIGTERM first (graceful), waits 5s, then SIGKILL.
        Safe to call from any thread; no-op if no subprocess is active.
        """
        proc = self._active_proc
        if proc is None or proc.pid is None:
            return

        pid = proc.pid

        # Discover process group — codex uses start_new_session=True,
        # so its pgid == pid.  Claude Code inherits our pgid, so we
        # must NOT killpg our own group.
        try:
            pgid = os.getpgid(pid)
            own_pgid = os.getpgid(os.getpid())
            use_pgkill = (pgid != own_pgid)
        except OSError:
            use_pgkill = False

        # Collect descendant PIDs (for same-group case where killpg
        # would kill us too — e.g. claude_code Bash(*) tool children).
        descendants = _get_descendants(pid)

        # SIGTERM the tree
        try:
            if use_pgkill:
                os.killpg(pgid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
                for child_pid in descendants:
                    try:
                        os.kill(child_pid, signal.SIGTERM)
                    except OSError:
                        pass
        except OSError:
            pass

        # Grace period
        try:
            proc.wait(timeout=5)
            return  # exited cleanly
        except Exception:
            pass

        # SIGKILL the tree
        try:
            if use_pgkill:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
                for child_pid in descendants:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except OSError:
                        pass
        except OSError:
            pass

        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    # ── path helpers ────────────────────────────────────────────────────────

    def _effective_paths(self) -> tuple[str, str, str]:
        """Return (workspace, datasets, models) as container or host paths."""
        if self.config.get("container_sif"):
            return CONTAINER_WORKSPACE, CONTAINER_SHARED_DATASETS, CONTAINER_SHARED_MODELS
        return self._workspace_path, self._shared_datasets_path, self._shared_models_path

    def _to_container_path(self, host_path: str) -> str:
        """Translate a workspace-relative host path to its container path."""
        if not self.config.get("container_sif") or not self._workspace_path:
            return host_path
        if host_path.startswith(self._workspace_path):
            rel = os.path.relpath(host_path, self._workspace_path)
            return os.path.join(CONTAINER_WORKSPACE, rel)
        return host_path

    def _workspace_claude_tmpdir(self) -> str | None:
        """Return the per-workspace Claude temp dir on the host."""
        if not self._workspace_path:
            return None
        return os.path.join(self._workspace_path, WORKSPACE_CLAUDE_TMPDIR)

    # ── core ───────────────────────────────────────────────────────────────

    def reproduce(self) -> str:
        """Start the reproduction by sending the full system prompt."""
        if self._paper is None or self._workspace_path is None:
            raise RuntimeError("Call set_paper_and_workspace() first")

        self._setup_workspace()
        self._status = "working"
        ws_path, ds_path, md_path = self._effective_paths()
        prompt = build_reproduce_prompt(
            paper_md_path=self._to_container_path(self._paper.get("paper_md_path", "")),
            paper_images_dir=self._to_container_path(self._paper.get("paper_images_dir", "")),
            paper_pdf_path=self._to_container_path(self._paper.get("paper_pdf_path", "")),
            workspace_path=ws_path,
            gpu_budget=self._gpu_budget,
            max_gpus=self._max_gpus,
            job_time_limit=self._job_time_limit,
            shared_datasets_path=ds_path,
            shared_models_path=md_path,
            scaffold=self.scaffold,
        )
        return self._invoke_and_read_action(prompt)

    def start(self, prompt: str) -> str:
        """Start the session with a custom first prompt (no reproduce_start).

        Used by roles (packager, designer) that don't need the full paper
        context and get their own role-specific system prompt instead.
        """
        if self._workspace_path is None:
            raise RuntimeError("Call set_paper_and_workspace() first")
        self._setup_workspace()
        self._status = "working"
        return self._invoke_and_read_action(prompt)

    def chat(self, message: str) -> str:
        """Send a follow-up message and read the updated action.yaml."""
        self._status = "working"
        return self._invoke_and_read_action(message)

    # ── job event handlers ─────────────────────────────────────────────────

    def on_job_progress(
        self, run_id: str, stdout_path: str, stderr_path: str,
        elapsed_min: int = 0,
    ) -> Literal["continue", "cancel"]:
        prompt = build_job_progress_prompt(
            run_id, stdout_path, stderr_path, elapsed_min=elapsed_min,
        )
        response = self._invoke(prompt)
        lower = response.strip().lower()
        if "cancel" in lower:
            return "cancel"
        return "continue"

    def on_job_completion(
        self, run_id: str, stdout_path: str, stderr_path: str, returncode: int,
    ) -> None:
        prompt = build_job_completion_prompt(run_id, stdout_path, stderr_path, returncode)
        self.chat(prompt)

    def on_budget_exhausted(
        self, gpu_hours_used: float, gpu_budget: float,
    ) -> None:
        milestone = parse_current_milestone(self._workspace_path or "")
        prompt = build_budget_exhausted_prompt(gpu_hours_used, milestone)
        self.chat(prompt)

    def on_container_build_failure(
        self,
        build_log_path: str,
        *,
        phase: str = "build",
        failure_class: str = "unknown",
        failure_signature: str = "",
        failure_hint: str = "",
        env_comparison: str = "first_failure",
        same_signature: bool = False,
    ) -> None:
        prompt = build_container_build_failure_prompt(
            build_log_path,
            phase=phase,
            failure_class=failure_class,
            failure_signature=failure_signature,
            failure_hint=failure_hint,
            env_comparison=env_comparison,
            same_signature=same_signature,
        )
        self.chat(prompt)

    def on_job_cancelled(
        self, run_id: str, stdout_path: str, stderr_path: str,
    ) -> None:
        prompt = build_job_cancelled_prompt(run_id, stdout_path, stderr_path)
        self.chat(prompt)

    # ── action / context ─────────────────────────────────────────────────

    def get_last_action(self) -> TurnAction | None:
        return self._last_action

    def get_context(self) -> dict:
        """Return current worker context for external inspection."""
        return {
            "status": self._status,
            "session_id": self._session_id,
            "last_action": self._last_action,
            "turn_count": self._turn_count,
            "stats": dict(self._cumulative_stats),
            "best_milestone": parse_current_milestone(self._workspace_path or ""),
        }

    def save_trajectory(self, logs_dir: str) -> None:
        """Write trajectory.json with all turn records and cumulative stats."""
        logs_path = Path(logs_dir)
        logs_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "scaffold": self.scaffold,
            "model": self.model,
            "turn_count": self._turn_count,
            "cumulative_stats": self._cumulative_stats,
            "best_milestone": parse_current_milestone(self._workspace_path or ""),
            "turns": self._trajectory,
        }
        (logs_path / "trajectory.json").write_text(
            redact_secrets(json.dumps(payload, indent=2, default=str))
        )

    # ── shared internal helpers ───────────────────────────────────────────

    def _invoke_and_read_action(self, prompt: str) -> str:
        """Invoke the CLI, then read action.yaml from the workspace.

        If action.yaml is invalid, retries by asking the agent to fix it
        (up to ``config["action_retries"]`` times).
        """
        text = self._invoke(prompt)

        # Always read action.yaml — the agent may have written a valid
        # milestone before the process ended (e.g. idle-timeout kill).
        action = parse_turn_action(self._workspace_path)
        if action is not None:
            self._last_action = action

        if self._status in ("error", "done"):
            return text
        self._last_action = action

        max_retries = self.config.get("action_retries", 2)
        retries = 0
        while action is None and retries < max_retries:
            retries += 1
            logger.warning(
                "Turn %d: invalid action.yaml, retry %d/%d",
                self._turn_count, retries, max_retries,
            )
            text = self._invoke(ACTION_FIX_PROMPT)
            if self._status in ("error", "done"):
                break
            action = parse_turn_action(self._workspace_path)
            self._last_action = action

        if action and action.action == "done":
            self._status = "done"
        else:
            # "submit" here only means the agent chose to submit.
            # The paper is not actually "waiting" on a Nexus/Hive job until
            # the scheduler finishes build/install and successfully calls
            # nexus.submit_job(). Only the scheduler should transition papers
            # into the real pre-job/job-waiting states.
            self._status = "working"

        return text

    @staticmethod
    def _is_retryable_error(errors: list[str], stderr: str) -> bool:
        """Check if CLI errors indicate a transient rate-limit / overload."""
        combined = " ".join(errors).lower() + " " + stderr.lower()
        if any(pat in combined for pat in _NON_RETRYABLE_PATTERNS):
            return False
        return any(pat in combined for pat in _RETRYABLE_PATTERNS)

    @staticmethod
    def _is_limit_error(errors: list[str], stderr: str) -> bool:
        """Check if errors indicate a hard API usage limit (key rotation candidate)."""
        combined = " ".join(errors).lower() + " " + stderr.lower()
        return any(pat in combined for pat in _NON_RETRYABLE_PATTERNS)

    def _try_rotate_api_key(self) -> bool:
        """Rotate to next API key if available. Returns True if rotated.

        Only rotates once through all keys per session — if we've cycled
        back to the starting key, all keys are exhausted and we give up.
        """
        api_keys = self.config.get("api_keys", [])
        if len(api_keys) <= 1:
            return False
        next_index = (self._api_key_index + 1) % len(api_keys)
        if next_index == 0:
            # Cycled through all keys — all exhausted
            return False
        self._api_key_index = next_index
        new_key = api_keys[self._api_key_index]
        # Copy: env_vars may be the module-level CONFIG_DEFAULTS dict or the
        # caller's dict, shared with other workers.
        env_vars = dict(self.config.get("env_vars", {}))
        env_vars["ANTHROPIC_AUTH_TOKEN"] = new_key
        self.config["env_vars"] = env_vars
        logger.warning("Rotated API key to index %d", self._api_key_index)
        return True

    def _write_git_excludes(self, ws: Path) -> None:
        """Exclude bind-mounted hidden files from git inside the container.

        The base worker bind-mounts /dev/null over HIDDEN_FILES inside the
        container.  If those files are tracked by git, the agent sees them
        as modified (/dev/null vs real content) and ``git add -A`` fails.

        We fix this on the host before the container starts:
        1. Write .git/info/exclude (pure file write — can't fail)
        2. ``git rm --cached`` to untrack them (best-effort — git can be
           slow on a shared filesystem)

        After the agent finishes, the scheduler's _git_commit_and_push
        re-adds these files with ``git add --force``.
        """
        if not (ws / ".git").is_dir():
            return

        # 1. Write exclude file FIRST — even if git commands below fail,
        #    this prevents ``git add -A`` from re-tracking the files.
        exclude_path = ws / ".git" / "info" / "exclude"
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        lines = set()
        if exclude_path.exists():
            lines = set(exclude_path.read_text().splitlines())
        for f in HIDDEN_FILES:
            lines.add(f"/{f}")
        for d in HIDDEN_DIRS:
            lines.add(f"/{d}/")
        exclude_path.write_text("\n".join(sorted(lines)) + "\n")

        # 2. Untrack hidden files from the index (best-effort).
        #    --ignore-unmatch: no error if the file isn't tracked.
        #    try/except: a timeout here must NOT kill the paper.
        try:
            subprocess.run(
                ["git", "rm", "--cached", "--quiet", "--ignore-unmatch", "--"]
                + list(HIDDEN_FILES),
                cwd=str(ws), capture_output=True, timeout=60,
            )
        except Exception:
            pass  # exclude file is sufficient for untracked files

    def _build_container_cmd(self, sif: str) -> list[str]:
        """Build the ``apptainer exec`` prefix for container isolation.

        Provides the shared skeleton: workspace bind, shared dirs, hidden
        files. Subclasses add scaffold-specific binds via the
        ``_add_scaffold_container_binds`` hook.
        """
        ws = self._workspace_path
        nice_val = str(self.config.get("container_nice", 19))
        mem_limit = str(self.config.get("container_memory_limit", "8G"))
        cpu_threads = str(self.config.get("container_cpu_threads", 2))
        cmd = [
            # nice -n N: low scheduler priority — agents yield CPU to
            # interactive work and to each other under contention.
            # Inherited by all subprocesses (bash, python3, pytest, uv).
            "nice", "-n", nice_val,
            "apptainer", "exec",
            "--contain",              # isolate /home, /tmp
            "--writable-tmpfs",       # allow bind mount points at arbitrary paths
            "--no-mount", "bind-paths",  # block admin-configured binds
            "--home", f"{ws}:{CONTAINER_WORKSPACE}",  # HOME=/home/user
            "--pwd", CONTAINER_WORKSPACE,              # cwd inside container
            # Memory cgroup — hard cap enforced across all processes inside
            # the container (claude CLI, bash, python3, subprocesses).
            "--memory", mem_limit,
            # BLAS / threading caps.  A CPU cgroup is often not delegated
            # to user sessions, so cores cannot be hard-capped, but these
            # env vars bound torch/numpy/scipy/numexpr thread fan-out,
            # which is what most agent scripts spawn.  Agents can
            # override them in code.
            "--env", f"OMP_NUM_THREADS={cpu_threads}",
            "--env", f"MKL_NUM_THREADS={cpu_threads}",
            "--env", f"OPENBLAS_NUM_THREADS={cpu_threads}",
            "--env", f"NUMEXPR_NUM_THREADS={cpu_threads}",
            "--env", f"VECLIB_MAXIMUM_THREADS={cpu_threads}",
        ]

        # Scaffold-specific binds (e.g. CLI binary, Node.js prefix, CA certs)
        self._add_scaffold_container_binds(cmd)

        # Bind shared directories at canonical container paths (read-only
        # to prevent agents from corrupting shared resources).
        for host_path, container_path in (
            (self._shared_datasets_path, CONTAINER_SHARED_DATASETS),
            (self._shared_models_path, CONTAINER_SHARED_MODELS),
        ):
            if host_path:
                cmd.extend(["--bind", f"{host_path}:{container_path}:ro"])

        # Bind the shared uv cache so agent pip/uv installs reuse cached
        # packages instead of downloading from scratch every time.
        if self._uv_cache_dir and os.path.isdir(self._uv_cache_dir):
            resolved = str(Path(self._uv_cache_dir).resolve())
            cmd.extend([
                "--bind", f"{resolved}:{resolved}",
                "--env", f"UV_CACHE_DIR={resolved}",
            ])

        # Give Claude a per-workspace temp directory instead of the shared /tmp
        # namespace. This avoids cross-session cwd file poisoning.
        private_tmp = self._workspace_claude_tmpdir()
        if private_tmp:
            os.makedirs(private_tmp, exist_ok=True)
            cmd.extend(["--bind", f"{private_tmp}:{CONTAINER_CLAUDE_TMPDIR}"])

        # Hide swarm-owned files from the agent
        for hidden in HIDDEN_FILES:
            target = os.path.join(ws, hidden)
            if os.path.exists(target):
                cmd.extend(["--bind", f"/dev/null:{CONTAINER_WORKSPACE}/{hidden}:ro"])
        for hidden in HIDDEN_DIRS:
            target = os.path.join(ws, hidden)
            if os.path.isdir(target):
                # Bind an empty dir over it so it appears empty inside.
                # We create a temp empty dir next to the workspace.
                empty = os.path.join(ws, f".empty_{hidden}")
                os.makedirs(empty, exist_ok=True)
                cmd.extend(["--bind", f"{empty}:{CONTAINER_WORKSPACE}/{hidden}:ro"])
        # Read-only files: agent can read/execute but not modify
        for ro_file in READONLY_FILES:
            target = os.path.join(ws, ro_file)
            if os.path.exists(target):
                cmd.extend(["--bind", f"{target}:{CONTAINER_WORKSPACE}/{ro_file}:ro"])

        # Block git push inside containers — the system handles pushing
        # at end-of-iteration.  Create a wrapper that intercepts 'push'
        # subcommand and no-ops it, forwarding everything else to real git.
        git_shim_dir = os.path.join(ws, ".git-shims")
        os.makedirs(git_shim_dir, exist_ok=True)
        git_shim = os.path.join(git_shim_dir, "git")
        with open(git_shim, "w") as f:
            f.write(
                '#!/bin/sh\n'
                'for arg in "$@"; do\n'
                '  case "$arg" in push)\n'
                '    echo "[system] git push is disabled inside containers — the system pushes automatically."\n'
                '    exit 0\n'
                '    ;;\n'
                '  esac\n'
                'done\n'
                'exec /usr/bin/git "$@"\n'
            )
        os.chmod(git_shim, 0o755)
        cmd.extend(["--bind", f"{git_shim_dir}:/opt/git-shim:ro"])

        cmd.append(sif)
        return cmd

    def _build_env(self) -> dict[str, str]:
        """Build the environment dict for the subprocess.

        Applies scaffold-specific cleanup via ``_configure_env``, then
        sets shared git identity and extra env vars.
        """
        env = dict(os.environ)
        self._configure_env(env)
        # Prepend git-push shim dir to PATH inside the container.
        # APPTAINERENV_PREPEND_PATH is handled by Apptainer runtime
        # and prepended to PATH even with --cleanenv.
        env["APPTAINERENV_PREPEND_PATH"] = "/opt/git-shim"
        # Git identity: scaffold + model + effort
        # actual_model overrides self.model when the model is a gateway alias
        git_name = self.scaffold or self.config.get("scaffold", "agent")
        raw_model = self.config.get("actual_model") or self.model or ""
        model_tag = raw_model.rsplit("/", 1)[-1]  # strip provider prefix
        effort = self.config.get("effort")
        parts = [git_name]
        if model_tag:
            parts.append(model_tag)
        if effort:
            parts.append(effort)
        git_name = f"{'-'.join(parts)} worker"
        env.setdefault("GIT_AUTHOR_NAME", git_name)
        env.setdefault("GIT_COMMITTER_NAME", git_name)
        email = f"{self.scaffold or 'agent'}@reproducer"
        env.setdefault("GIT_AUTHOR_EMAIL", email)
        env.setdefault("GIT_COMMITTER_EMAIL", email)
        # Apply extra env vars
        env.update(self.config.get("env_vars", {}))
        return env

    def _save_raw_output(self, turn_num: int, stdout: str) -> None:
        """Save raw JSONL output to trajectory/turn_NNNN_raw.jsonl."""
        if not self._workspace_path:
            return
        traj_dir = Path(self._workspace_path) / "trajectory"
        traj_dir.mkdir(parents=True, exist_ok=True)
        raw_path = traj_dir / f"turn_{turn_num:04d}_raw.jsonl"
        raw_path.write_text(redact_secrets(stdout))

    def _save_session_info(self) -> None:
        """Save session_id to trajectory/session.json for resume support."""
        if not self._workspace_path or not self._session_id:
            return
        traj_dir = Path(self._workspace_path) / "trajectory"
        traj_dir.mkdir(parents=True, exist_ok=True)
        (traj_dir / "session.json").write_text(
            json.dumps({"session_id": self._session_id}, indent=2)
        )
