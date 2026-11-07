"""Scheduler — worker pool, paper queue, and orchestration loop.

Each paper runs in its own thread. ThreadPoolExecutor limits concurrency.
Workers are synchronous (the agent CLI subprocess blocks), so threading is
natural.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional
import fcntl

# Ensure the repo root is on sys.path so that ``from infrastructure.worker``
# imports work regardless of how the daemon process was started.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from infrastructure.worker.worker import CONTAINER_SHARED_DATASETS, CONTAINER_SHARED_MODELS
from swarm.config import (
    PAPERS_DIR,
    SUPPORTED_SCAFFOLDS,
    SWARM_DIR,
    SwarmConfig,
    WorkerConfig,
    ensure_swarm_dir,
    load_config,
)
from swarm.nexus_bridge import NexusBridge
from swarm.paper import PaperInfo, resolve_paper
from swarm.workspace import (
    WORKSPACE_TEMPLATE_DIR,
    clone_workspace,
    create_workspace,
    ensure_github_remote,
    strip_nested_git_dirs,
)

log = logging.getLogger(__name__)

# Scrub Docker Hub credentials from the scheduler process env on import.
#
# Apptainer forwards ``APPTAINER_DOCKER_USERNAME`` /
# ``APPTAINER_DOCKER_PASSWORD`` (and the Singularity-prefixed variants) to
# every docker:// registry it contacts, not only to Docker Hub. Other
# registries (mirror.gcr.io, nvcr.io, quay.io, ghcr.io, public.ecr.aws)
# reject those credentials with "authentication failed". The variables
# would reach three subprocess call sites (``apptainer inspect``,
# ``apptainer pull``, ``apptainer build``), so they are removed once here.
# The base SIF cache limits how often Docker Hub is pulled anonymously.
for _docker_env_key in (
    "APPTAINER_DOCKER_USERNAME",
    "APPTAINER_DOCKER_PASSWORD",
    "SINGULARITY_DOCKER_USERNAME",
    "SINGULARITY_DOCKER_PASSWORD",
):
    os.environ.pop(_docker_env_key, None)
del _docker_env_key

# Milestone ordering — higher index = better.
_MILESTONE_RANK = {
    "not_possible": -1,
    "none": 0,
    "method_runs": 1,
    "core_claim": 2,
    "core_claim_plus": 3,
    "secondary_claims": 4,
    "majority": 5,
    "near_complete": 6,
    "full": 7,
}

_RESTART_RECOVERY_ERROR = "daemon restarted"
_RESTART_RECOVERY_MESSAGE = "daemon restarted — use 'swarm continue-all' to resume"
_RESTART_RECOVERY_STATUS = "paused"
_LOCAL_GIT_TIMEOUT = 300
_GIT_FINALIZE_TIMEOUT = 180  # aggregate wall-clock cap for _git_commit_and_push
# Concurrent pushes are limited to bound the load on a shared filesystem.
_GIT_PUSH_SEMAPHORE = threading.Semaphore(3)
# Default port of the Nexus daemon, used when nexus.url names no port.
_NEXUS_DEFAULT_PORT = 39000

_EXCLUDED_PAPERS_PATH = Path(__file__).parent / "config" / "excluded_papers.yaml"


def _load_excluded_papers() -> dict[str, str]:
    """Return {arxiv_id: reason} for papers the swarm must not process.

    Reads ``swarm/config/excluded_papers.yaml`` on every call (cheap: the
    list is tiny and append-only). Missing file or parse errors are treated
    as an empty exclusion list — the exclusion is opt-in, not required.
    The shipped ``excluded_papers.example.yaml`` is a template and is
    never read.
    """
    try:
        import yaml
        text = _EXCLUDED_PAPERS_PATH.read_text()
        data = yaml.safe_load(text) or {}
    except (FileNotFoundError, OSError):
        return {}
    except Exception:
        log.exception("Failed to parse %s — ignoring", _EXCLUDED_PAPERS_PATH)
        return {}
    if not isinstance(data, dict):
        log.warning("%s has no top-level 'excluded' mapping — ignoring", _EXCLUDED_PAPERS_PATH)
        return {}
    out: dict[str, str] = {}
    for entry in data.get("excluded", []) or []:
        if not isinstance(entry, dict):
            continue
        aid = str(entry.get("arxiv_id", "")).strip()
        if aid:
            out[aid] = str(entry.get("reason", "")).strip() or "(no reason given)"
    return out


def _guard_excluded(arxiv_id: str) -> None:
    """Raise ValueError if ``arxiv_id`` is on the exclude list."""
    excluded = _load_excluded_papers()
    if arxiv_id in excluded:
        raise ValueError(
            f"Paper {arxiv_id} is on the exclude list and will not be processed. "
            f"Reason: {excluded[arxiv_id]}. "
            f"To reverse, remove the entry from {_EXCLUDED_PAPERS_PATH}."
        )


def _higher_milestone(a: str, b: str) -> str:
    """Return whichever milestone is further along the progression."""
    return a if _MILESTONE_RANK.get(a, 0) >= _MILESTONE_RANK.get(b, 0) else b


# Ordered list for milestone demotion (excludes not_possible)
_MILESTONE_ORDER = [
    "none", "method_runs", "core_claim", "core_claim_plus",
    "secondary_claims", "majority", "near_complete", "full",
]
# Full valid-set the agent may declare in action.yaml. Agents that use
# anything else are re-prompted to pick a canonical name.
_MILESTONE_VALID_SET = frozenset(_MILESTONE_ORDER + ["not_possible"])

# Roles that cannot change the paper's milestone.  action.yaml's
# ``milestone:`` field is ignored for these; the paper's previous_milestone
# is preserved verbatim.  These roles also skip the invalid-milestone retry
# gate and the MILESTONES.md fallback.
_MILESTONE_LOCKED_ROLES = frozenset({
    "packager", "designer",
    "designer_stage_1", "designer_stage_2", "designer_stage_3",
})

# Roles whose initial prompt is sent via worker.start(prompt) instead of
# worker.reproduce() (i.e. they do not see reproduce_start.md).
_NO_REPRODUCE_START_ROLES = frozenset({
    "packager", "designer",
    "designer_stage_1", "designer_stage_2", "designer_stage_3",
})


def _drop_milestone(milestone: str) -> str:
    """Drop a milestone one level. Returns 'none' for the lowest or unknown."""
    try:
        idx = _MILESTONE_ORDER.index(milestone)
    except ValueError:
        return "none"
    return _MILESTONE_ORDER[max(0, idx - 1)]


def _is_model_limit_error_text(error_text: str) -> bool:
    """Check if an error string indicates API limit exhaustion (all keys tried)."""
    from infrastructure.worker.worker import _NON_RETRYABLE_PATTERNS
    lower = error_text.lower()
    return any(pat in lower for pat in _NON_RETRYABLE_PATTERNS)


@contextmanager
def _file_lock(lock_path: Path, timeout: float = 3600.0):
    """Acquire an advisory file lock with polling and timeout."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    start = time.time()
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - start >= timeout:
                    raise TimeoutError(f"Timed out waiting for lock {lock_path}")
                time.sleep(1)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _copy_artifact(src: Path, dst: Path) -> None:
    """Copy large sparse/cacheable artifacts efficiently when possible."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        subprocess.run(
            ["cp", "--reflink=auto", "--sparse=always", str(src), str(dst)],
            capture_output=True,
            timeout=600,
            check=True,
        )
    except Exception:
        shutil.copy2(str(src), str(dst))


def _format_iteration_commit_msg(
    phase: str,
    iteration: int,
    milestone: str,
    profile: str | None,
    role: str,
) -> str:
    """Format iteration boundary commit messages consistently."""
    profile_label = profile or "default"
    return f"{phase} of Iteration {iteration}: {milestone} [{profile_label} ({role})]"


def _iteration_completion_topics(
    iteration: int,
    *,
    status: str,
    has_next_iteration: bool,
) -> list[str]:
    """Return lifecycle topics that should be added after an iteration ends."""
    if status != "done":
        return []
    topics = [f"iteration-{iteration}-complete"]
    if not has_next_iteration:
        topics.append("done")
    return topics


def _workflow_step_gpu_budget(
    config: SwarmConfig,
    workflow: str | None,
    step_idx: int,
) -> float | None:
    """Return the configured default GPU budget for a workflow step."""
    if not workflow:
        return None
    steps = config.workflows.get(workflow, [])
    if 0 <= step_idx < len(steps):
        return steps[step_idx].gpu_budget_hours
    return None


def _create_worker(scaffold: str, model: str, config: dict):
    """Factory: instantiate the right Worker subclass for the scaffold."""
    if scaffold == "codex":
        from infrastructure.worker.codex import CodexWorker
        return CodexWorker(scaffold=scaffold, model=model, config=config)
    if scaffold == "claude-code":
        from infrastructure.worker.claude_code import ClaudeCodeWorker
        return ClaudeCodeWorker(scaffold=scaffold, model=model, config=config)
    raise ValueError(
        f"Unknown scaffold '{scaffold}'. Supported: {list(SUPPORTED_SCAFFOLDS)}. "
        f"Set worker.profile, or give the workflow step a profile."
    )


# ── Paper job / result dataclasses ────────────────────────────────────────

@dataclass
class PaperJob:
    """A paper queued or actively being worked on."""
    arxiv_id: str
    paper: PaperInfo
    workspace_path: str
    gpu_budget: float  # current iteration's budget (GPU-hours)
    profile: str | None = None      # model profile name (None = use daemon default)
    # queued, active, done, error, cancelled, stalled, unsuccessful, paused
    status: str = "queued"
    milestone: str = "none"
    worker_status: str = "idle"
    blocked_on: str | None = None
    last_job_resources: str = ""       # e.g. "1 GPU", "CPU", "2 GPU (test)"
    last_job_id: str = ""              # Nexus/Hive run_id for the most recent submission
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    session_id: str | None = None   # agent CLI session id, used to resume
    is_continuation: bool = False   # True if resuming previous work
    is_iteration: bool = False     # True if iterating with a fresh agent
    iteration: int = 1            # which iteration (1-indexed)
    previous_milestone: str = "none"  # milestone claimed by previous agent
    role: str = "reproducer"      # one of swarm.config.WORKFLOW_ROLES
    workflow: str | None = None    # active workflow name (None = manual)
    workflow_step: int = 0         # current step index in workflow
    gpu_hours_used: float = 0.0     # current iteration's GPU-hours
    total_gpu_hours_used: float = 0.0  # cumulative GPU-hours across all runs
    total_duration_sec: float = 0.0  # cumulative wall-clock across all iterations
    date: str | None = None         # paper date (YYYY-MM-DD), used as GitHub topic
    topics: list[str] | None = None  # extra GitHub topics for repo creation
    last_env_failure_container_hash: str | None = None
    last_env_failure_setup_hash: str | None = None
    last_env_failure_signature: str | None = None
    last_env_failure_class: str | None = None
    last_container_hash: str | None = None  # persisted for restart skip
    last_setup_hash: str | None = None      # persisted for restart skip
    # Lifetime count of workflow-gate reverts triggered for this paper.
    # Compared against config.max_revert_attempts; once the cap is hit, the
    # paper lands in status="unsuccessful" instead of reverting again.
    revert_attempts: int = 0
    _worker: Any = None  # active Worker instance (not serialized)
    # State timing — tracks how long the worker spends in each state
    worker_status_since: float = 0.0   # timestamp of last state change
    worker_status_totals: dict = field(default_factory=dict)  # state -> cumulative seconds
    # Time spent blocked waiting for a capacity slot (not doing real work)
    blocked_sec: float = 0.0

    def set_worker_status(self, new_status: str) -> None:
        """Update worker_status and accumulate time in the previous state."""
        now = time.time()
        if self.worker_status_since > 0 and self.worker_status:
            elapsed = now - self.worker_status_since
            prev = self.worker_status
            self.worker_status_totals[prev] = self.worker_status_totals.get(prev, 0.0) + elapsed
        self.worker_status = new_status
        self.worker_status_since = now

    def to_dict(self) -> dict:
        # Compute live current-state duration
        status_elapsed = 0.0
        if self.worker_status_since > 0:
            status_elapsed = time.time() - self.worker_status_since
        return {
            "arxiv_id": self.arxiv_id,
            "workspace_path": self.workspace_path,
            "gpu_budget": self.gpu_budget,
            "gpu_hours_used": self.gpu_hours_used,
            "total_gpu_hours_used": self.total_gpu_hours_used,
            "status": self.status,
            "milestone": self.milestone,
            "worker_status": self.worker_status,
            "blocked_on": self.blocked_on,
            "worker_status_elapsed": status_elapsed,
            "worker_status_totals": dict(self.worker_status_totals),
            "last_job_resources": self.last_job_resources,
            "last_job_id": self.last_job_id,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "session_id": self.session_id,
            "is_continuation": self.is_continuation,
            "is_iteration": self.is_iteration,
            "previous_milestone": self.previous_milestone,
            "profile": self.profile,
            "iteration": self.iteration,
            "role": self.role,
            "workflow": self.workflow,
            "workflow_step": self.workflow_step,
            "total_duration_sec": self.total_duration_sec,
            "date": self.date,
            "topics": self.topics,
            "last_env_failure_container_hash": self.last_env_failure_container_hash,
            "last_env_failure_setup_hash": self.last_env_failure_setup_hash,
            "last_env_failure_signature": self.last_env_failure_signature,
            "last_env_failure_class": self.last_env_failure_class,
            "last_container_hash": self.last_container_hash,
            "last_setup_hash": self.last_setup_hash,
            "revert_attempts": self.revert_attempts,
        }


@dataclass
class PaperResult:
    """Result of a completed paper run."""
    arxiv_id: str
    milestone: str = "none"
    status: str = "done"            # terminal status, see PaperJob.status
    error: str | None = None
    duration_sec: float = 0.0
    total_duration_sec: float = 0.0
    trajectory_path: str | None = None
    profile: str | None = None
    iteration: int = 1
    workflow: str | None = None
    workflow_step: int = 0
    finished_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "milestone": self.milestone,
            "status": self.status,
            "error": self.error,
            "duration_sec": self.duration_sec,
            "total_duration_sec": self.total_duration_sec,
            "finished_at": self.finished_at,
            "trajectory_path": self.trajectory_path,
            "profile": self.profile,
            "iteration": self.iteration,
            "workflow": self.workflow,
            "workflow_step": self.workflow_step,
        }


# ── Scheduler ─────────────────────────────────────────────────────────────

class Scheduler:
    """Manages the worker pool, paper queue, and orchestration loop.

    Papers are enqueued via :meth:`enqueue` and dispatched to threads
    from the :class:`ThreadPoolExecutor`. Each thread runs the full
    lifecycle for one paper: create worker → reproduce → handle actions
    → save trajectory.
    """

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.queue: deque[PaperJob] = deque()
        self.active: dict[str, PaperJob] = {}       # arxiv_id → job
        self._finalizing: dict[str, PaperJob] = {}  # arxiv_id -> job still inside _finish_paper
        self.completed: dict[str, PaperResult] = {}  # arxiv_id → result
        self._lock = threading.Lock()
        self._capacity_cv = threading.Condition(self._lock)
        self._capacity_state: dict[str, str | None] = {}
        # Runtime soft cap overrides, keyed by the names in _CAP_BUCKETS.
        # Missing key → use config.
        # Soft semantics: reducing a cap only refuses new admissions; it
        # does not kill or preempt jobs already in that bucket.
        self._cap_overrides: dict[str, int] = {}
        self._working_priority_waiters: deque[str] = deque()
        self._working_fresh_waiters: deque[str] = deque()
        self._working_waiter_kind: dict[str, str] = {}
        self._executor = ThreadPoolExecutor(max_workers=self._max_total_workers())
        self._shutdown = False
        self._cancelled: set[str] = set()  # arxiv_ids flagged for cancellation
        self._active_workers: dict[str, Any] = {}  # arxiv_id → Worker for cancel signalling
        self._nexus_projects: set[str] = set()  # active project IDs for cleanup
        self._child_pids: set[int] = set()  # track child subprocess PIDs for cleanup
        self._nexus_proc: subprocess.Popen | None = None

        ensure_swarm_dir()

        # Kill apptainer build/pull processes left behind by a previous
        # daemon session. They survive daemon restarts because _child_pids
        # is in-memory only. This matches every such process of the current
        # user, not only the ones Swarm started.
        self._kill_orphaned_build_processes()

        # Clean up stale overlay images left in local_tmp_dir by previous daemon crashes.
        _stale_overlay_dir = Path(config.local_tmp_dir) / "swarm_overlays"
        if _stale_overlay_dir.exists():
            shutil.rmtree(_stale_overlay_dir, ignore_errors=True)
            log.info("Cleaned stale overlay directory %s", _stale_overlay_dir)

        # Copy agent SIF files to node-local disk (local_tmp_dir) so that
        # every agent container mounts its image from local storage. One
        # squashfuse mount per working container against a single file on
        # a shared network filesystem puts heavy load on that filesystem.
        # Keyed by resolved source path so profile-specific SIFs each
        # get their own local copy.
        self._local_sifs: dict[str, str] = {}  # resolved source path → local copy
        self._cache_sif_locally(config.worker.container_sif)

        # Periodic dispatch tick — self-heals the starvation class where
        # every active paper is in a long working/waiting loop and no
        # event (paper finish, admitted-bucket exit) triggers dispatch
        # for pool slots opened by cap raises or worker-level churn.
        # Runs at 60s — cheap, idempotent, returns immediately if pool
        # is already saturated.
        self._dispatch_tick_interval = 60.0
        self._dispatch_tick_thread = threading.Thread(
            target=self._dispatch_tick_loop,
            name="swarm-dispatch-tick",
            daemon=True,
        )
        self._dispatch_tick_thread.start()

        # Start a Nexus daemon if the configured one does not answer a
        # single 3-second probe.
        self.nexus = NexusBridge(config.nexus.url)
        try:
            import urllib.request as _ur
            _ur.urlopen(config.nexus.url + "/api/projects", timeout=3)
            log.info("Nexus daemon reachable at %s", config.nexus.url)
        except Exception:
            log.warning("Nexus daemon not reachable at %s — starting it", config.nexus.url)
            self._restart_nexus_daemon()

        # Recover orphaned papers from previous daemon session.
        # Papers with status active/queued in meta.json have no worker
        # thread. Mark them "paused" with the action needed to resume, so
        # they show on the dashboard and 'swarm continue-all' picks them up.
        if PAPERS_DIR.is_dir():
            for meta_path in PAPERS_DIR.glob("*/meta.json"):
                try:
                    meta = json.loads(meta_path.read_text())
                    if meta.get("status") in ("active", "queued"):
                        original_status = meta.get("status")
                        self._mark_meta_for_recovery(meta, original_status=original_status)
                        meta_path.write_text(json.dumps(meta, indent=2))
                        log.info(
                            "Recovered orphaned paper %s (was %s, recovery=%s, now %s)",
                            meta_path.parent.name, original_status,
                            meta["recovery_action"], _RESTART_RECOVERY_STATUS,
                        )
                except Exception:
                    log.warning("Could not recover paper state from %s", meta_path, exc_info=True)

        # Clean up stale Nexus projects from previous daemon session.
        # Only delete projects with no running/queued jobs — their GPU budget
        # is tracked in swarm's meta.json and will be carried forward when
        # the paper is relaunched with the correct remaining budget.
        try:
            for p in self.nexus.list_projects():
                jc = p.get("job_counts", {})
                if jc.get("running", 0) == 0 and jc.get("queued", 0) == 0:
                    pid = p.get("project_id", "")
                    if pid and pid.startswith("swarm-"):
                        self.nexus.delete_project(pid)
                        log.info("Cleaned up stale Nexus project %s", pid)
        except Exception as e:
            log.debug("Nexus cleanup failed (may not be running yet): %s", e)

        # Set APPTAINER_CACHEDIR so apptainer build/exec don't fill ~/.apptainer
        if config.apptainer_cache_dir:
            os.environ["APPTAINER_CACHEDIR"] = config.apptainer_cache_dir
            Path(config.apptainer_cache_dir).mkdir(parents=True, exist_ok=True)
            log.info("APPTAINER_CACHEDIR=%s", config.apptainer_cache_dir)
            self._validate_apptainer_cache_index(config.apptainer_cache_dir)

    @staticmethod
    def _validate_apptainer_cache_index(cache_dir: str) -> None:
        """Repair the OCI blob-cache ``index.json`` if it's corrupt.

        Concurrent ``apptainer pull/build`` processes can race on the
        shared ``index.json``, producing a file that contains a valid
        JSON object followed by trailing garbage from a second writer.
        Go's ``json.Unmarshal`` then fails with "invalid character …
        after top-level value", which propagates as a fatal build error
        for every subsequent image pull through this cache.

        Detect this at startup and truncate back to the valid prefix so
        the corruption never reaches a live build.
        """
        index_path = Path(cache_dir) / "cache" / "blob" / "index.json"
        if not index_path.exists():
            return
        try:
            raw = index_path.read_bytes()
            json.loads(raw)
            return  # valid
        except json.JSONDecodeError as e:
            corrupt_pos = e.pos
            log.warning(
                "Apptainer cache index.json is corrupt at byte %d: %s — repairing",
                corrupt_pos, e.msg,
            )
        # Truncate to the valid-JSON prefix.
        try:
            valid = raw[:corrupt_pos]
            json.loads(valid)  # verify truncation produces valid JSON
        except Exception:
            # Can't salvage — remove entirely so apptainer recreates it.
            bak = index_path.with_suffix(".json.corrupt.bak")
            index_path.rename(bak)
            log.warning("Could not repair index.json — moved to %s", bak)
            return
        bak = index_path.with_suffix(".json.corrupt.bak")
        try:
            import shutil
            shutil.copy2(index_path, bak)
        except Exception:
            pass
        index_path.write_bytes(valid)
        log.info(
            "Repaired apptainer cache index.json: truncated %d → %d bytes",
            len(raw), len(valid),
        )

    def _nexus_autostart_endpoint(self) -> tuple[str, int]:
        """Return the (host, port) a Swarm-started Nexus daemon binds to.

        The port comes from ``nexus.url`` (Nexus's default when the URL
        names none); the interface from ``nexus.autostart_host``, which
        defaults to loopback because the Nexus API is unauthenticated.
        """
        from urllib.parse import urlparse
        port = urlparse(self.config.nexus.url).port or _NEXUS_DEFAULT_PORT
        host = self.config.nexus.autostart_host or "127.0.0.1"
        return host, port

    @staticmethod
    def _nexus_autostart_script(host: str, port: int) -> str:
        """Python source that serves the Nexus daemon on ``host:port``."""
        return (
            "from nexus.daemon import create_app\n"
            "from nexus.config import load_config, ensure_nexus_dir\n"
            "ensure_nexus_dir()\n"
            "config = load_config()\n"
            "app = create_app(config)\n"
            f"app.run(host={host!r}, port={port}, debug=False, threaded=True)\n"
        )

    def _restart_nexus_daemon(self):
        """Stop whatever listens on the Nexus port and start a Nexus daemon.

        Called at startup when the configured Nexus URL does not answer.
        Every process listening on the port receives SIGTERM, then a new
        daemon is started as a child of the Swarm daemon. The child needs
        the ``nexus`` package in the same Python environment. It is not
        stopped by :meth:`shutdown`.
        """
        host, port = self._nexus_autostart_endpoint()

        # SIGTERM lets a running Nexus daemon finish its file writes.
        try:
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True, text=True, timeout=5,
            )
            for pid_str in result.stdout.strip().split():
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                if pid == os.getpid():
                    continue
                try:
                    os.kill(pid, signal.SIGTERM)
                    log.info("Sent SIGTERM to process on Nexus port %d (PID %d)", port, pid)
                except ProcessLookupError:
                    pass  # already exited
        except Exception as e:
            log.debug("No existing Nexus daemon to kill: %s", e)

        time.sleep(2)  # give SIGTERM time to finish file writes

        # Start fresh Nexus daemon as child process
        swarm_log = SWARM_DIR / "nexus.log"
        nexus_log_fh = open(str(swarm_log), "a")
        self._nexus_proc = subprocess.Popen(
            [sys.executable, "-c", self._nexus_autostart_script(host, port)],
            stdin=subprocess.DEVNULL,
            stdout=nexus_log_fh,
            stderr=nexus_log_fh,
        )
        log.info("Started Nexus daemon (PID %d) on %s:%d", self._nexus_proc.pid, host, port)

        # Wait for it to be reachable at the configured URL
        import urllib.request
        for _ in range(20):
            time.sleep(0.5)
            try:
                urllib.request.urlopen(self.config.nexus.url.rstrip("/") + "/api/projects", timeout=2)
                log.info("Nexus daemon is ready")
                break
            except Exception:
                pass
        else:
            log.warning(
                "Nexus daemon started on %s:%d did not answer at %s within 10s — "
                "continuing anyway. If nexus.url does not point at that "
                "interface, set nexus.autostart_host or start Nexus yourself.",
                host, port, self.config.nexus.url,
            )

    def _kill_orphaned_build_processes(self) -> None:
        """Kill every ``apptainer build`` / ``apptainer pull`` of the current user.

        Intended for processes orphaned by a previous daemon session, but
        the match is by user and command line only, so unrelated builds
        started by the same user are killed too.
        """
        try:
            result = subprocess.run(
                ["pgrep", "-u", os.environ.get("USER", ""), "-f",
                 "apptainer (build|pull)"],
                capture_output=True, text=True, timeout=5,
            )
            killed = 0
            for line in result.stdout.strip().split("\n"):
                pid_str = line.strip()
                if not pid_str:
                    continue
                pid = int(pid_str)
                if pid == os.getpid():
                    continue
                try:
                    # Kill entire process group (mksquashfs children etc.)
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                    killed += 1
                except OSError:
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed += 1
                    except OSError:
                        pass
            if killed:
                log.info("Killed %d orphaned apptainer build/pull processes", killed)
        except Exception as e:
            log.debug("Orphaned build cleanup skipped: %s", e)

    def _cache_sif_locally(self, sif_path: str | None) -> None:
        """Copy a SIF to ``local_tmp_dir`` if not already cached there."""
        if not sif_path:
            return
        src = Path(sif_path)
        if not src.exists():
            return
        key = str(src.resolve())
        if key in self._local_sifs:
            return
        # Include a hash of the source path in the local filename so
        # different SIFs with the same basename don't collide.
        path_hash = hashlib.sha256(key.encode()).hexdigest()[:8]
        local = Path(self.config.local_tmp_dir) / f"swarm_{path_hash}_{src.name}"
        try:
            src_stat = src.stat()
            if not local.exists() or local.stat().st_size != src_stat.st_size \
                    or local.stat().st_mtime < src_stat.st_mtime:
                shutil.copy2(str(src), str(local))
                log.info("Copied SIF to local disk: %s -> %s", src.name, local)
            else:
                log.info("SIF already on local disk: %s", local)
            self._local_sifs[key] = str(local)
        except Exception as e:
            log.warning("Failed to copy SIF to local disk: %s", e)

    def _get_local_sif(self, sif_path: str | None) -> str | None:
        """Return the node-local copy of a SIF, caching on first use.

        Verifies the cached file still exists on disk — other users or
        temp-directory cleanup policies can delete files even while we
        have a path cached in memory. Re-copies if missing.
        """
        if not sif_path:
            return None
        key = str(Path(sif_path).resolve())
        cached = self._local_sifs.get(key)
        if cached and Path(cached).exists():
            return cached
        # Cached entry missing from disk (or never cached) — drop stale
        # entry and re-copy.
        if cached:
            self._local_sifs.pop(key, None)
            log.warning("Local SIF %s missing from disk — re-copying", cached)
        self._cache_sif_locally(sif_path)
        return self._local_sifs.get(key)

    def _resolve_profile(self, profile_name: str | None) -> WorkerConfig:
        """Resolve a profile name to a WorkerConfig.

        If profile_name is None, returns the daemon's default config.
        Otherwise, starts from daemon defaults, applies profile overrides,
        then returns the result.
        """
        # Start with daemon defaults (max_turns, container_sif, etc.)
        wc = WorkerConfig(
            scaffold=self.config.worker.scaffold,
            model=self.config.worker.model,
            actual_model=self.config.worker.actual_model,
            container_sif=self.config.worker.container_sif,
            max_turns=self.config.worker.max_turns,
            effort=self.config.worker.effort,
            env_vars=dict(self.config.worker.env_vars),
            compact_token_limit=self.config.worker.compact_token_limit,
            setup_timeout=self.config.worker.setup_timeout,
            api_keys=list(self.config.worker.api_keys),
        )
        if profile_name is None:
            return self._require_scaffold_and_model(wc, "the default worker")
        # Load profiles from config file
        from swarm.config import _find_config_path
        import yaml as _yaml
        config_path = _find_config_path()
        profiles = {}
        if config_path:
            with open(config_path) as f:
                raw = _yaml.safe_load(f) or {}
            profiles = raw.get("profiles") or {}
        if profile_name not in profiles:
            raise ValueError(
                f"Profile '{profile_name}' not found. "
                f"Available: {list(profiles.keys())}"
            )
        # Clear profile-specific fields before applying the new profile.
        # Otherwise credentials and settings of the daemon's default
        # profile (env_vars, api_keys, ...) would leak into a profile
        # that does not define them, e.g. one for a different provider.
        wc.env_vars = {}
        wc.api_keys = []
        wc.actual_model = None
        wc.compact_token_limit = None
        for k, v in profiles[profile_name].items():
            if hasattr(wc, k):
                setattr(wc, k, v)
        wc.profile = profile_name
        return self._require_scaffold_and_model(wc, f"profile '{profile_name}'")

    @staticmethod
    def _require_scaffold_and_model(wc: WorkerConfig, source: str) -> WorkerConfig:
        """Fail early when neither a profile nor ``worker`` names an agent."""
        if not wc.scaffold or not wc.model:
            raise ValueError(
                f"No scaffold/model configured for {source}: set both in the "
                f"profile, or select a default with worker.profile."
            )
        return wc

    def shutdown(self):
        """Stop accepting new work, clean up Nexus projects, kill children."""
        self._shutdown = True
        with self._capacity_cv:
            self._capacity_cv.notify_all()
        self._executor.shutdown(wait=False)

        # Signal all active workers to abort retry sleeps, then kill subprocesses.
        # Snapshot under lock — worker threads mutate self.active concurrently.
        with self._lock:
            active_snapshot = list(self.active.values())
            for w in self._active_workers.values():
                try:
                    w.request_cancel()
                except Exception:
                    pass
        for job in active_snapshot:
            w = job._worker
            if w is not None:
                try:
                    w.kill_subprocess()
                    log.info("Killed agent subprocess for %s", job.arxiv_id)
                except Exception:
                    pass

        # Kill child subprocesses (apptainer setup/build) to prevent zombies
        import signal
        for pid in list(self._child_pids):
            try:
                os.kill(pid, signal.SIGTERM)
                log.info("Killed child process %d", pid)
            except OSError:
                pass
        self._child_pids.clear()

        # Delete all active Nexus projects
        for project_id in list(self._nexus_projects):
            try:
                self.nexus.delete_project(project_id)
                log.info("Cleaned up Nexus project %s", project_id)
            except Exception as e:
                log.warning("Failed to clean up Nexus project %s: %s", project_id, e)
        self._nexus_projects.clear()

        log.info("Scheduler shutdown initiated")

    _CAP_BUCKETS: ClassVar[tuple[str, ...]] = (
        "total", "working", "overflow", "admitted", "building", "installing",
    )

    def _max_total_workers(self) -> int:
        override = self._cap_overrides.get("total")
        if override is not None:
            return max(1, override)
        return self.config.effective_total_workers()

    def set_cap(self, bucket: str, value: int | None) -> dict:
        """Set or clear a soft cap override for a bucket. Returns the new cap state.

        Soft semantics: lowering a cap only refuses new admissions to that
        bucket; jobs already occupying it keep running. value=None clears the
        override and restores config-defined behavior.
        """
        if bucket not in self._CAP_BUCKETS:
            raise ValueError(
                f"Unknown cap bucket '{bucket}'. Valid: {', '.join(self._CAP_BUCKETS)}"
            )
        if value is not None and value < 0:
            raise ValueError(f"Cap value must be >= 0, got {value}")
        with self._lock:
            if value is None:
                self._cap_overrides.pop(bucket, None)
            else:
                self._cap_overrides[bucket] = int(value)
            # Wake waiters — a cap raise may admit pending work.
            self._capacity_cv.notify_all()
        log.info("Cap override: %s = %s", bucket, value)
        # Resize the thread pool when the total cap grows beyond the
        # current executor size. Without this, `swarm cap total 30`
        # admits 30 papers but only pool_size threads exist to run
        # them — the rest sit in the executor's internal queue as
        # idle papers that never start.
        new_total = self._max_total_workers()
        if new_total > self._executor._max_workers:
            old = self._executor._max_workers
            self._executor._max_workers = new_total
            log.info("Resized executor thread pool: %d -> %d", old, new_total)
        # A cap raise can open pool slots while nobody is blocked on a
        # state transition (steady-state papers in waiting/working loops
        # don't trigger dispatch).  Drain the queue into any slots now
        # available instead of waiting for the next paper-finish event.
        self._drain_dispatch()
        return self.get_caps()

    def _drain_dispatch(self, max_iterations: int = 128) -> int:
        """Call ``_dispatch_next`` repeatedly until the pool is full.

        Returns the number of jobs admitted.  Bounded by ``max_iterations``
        so a buggy ``_dispatch_next`` that always returns True can't spin.
        """
        admitted = 0
        for _ in range(max_iterations):
            if not self._dispatch_next():
                break
            admitted += 1
        return admitted

    def _dispatch_tick_loop(self) -> None:
        """Background heartbeat that drains the queue every 60s.

        Fixes a structural starvation: ``_dispatch_next`` is only called
        from events (enqueue, paper finish, admitted-bucket exit).  When
        all active papers settle into long working/waiting loops on Hive
        jobs, no event fires and queued papers never get admitted even
        though pool slots are free.  This periodic tick makes admission
        eventually consistent.
        """
        while not self._shutdown:
            try:
                admitted = self._drain_dispatch()
                if admitted:
                    log.info("Dispatch tick: admitted %d paper(s)", admitted)
            except Exception as e:
                log.warning("Dispatch tick failed (continuing): %s", e)
            # Sleep in small slices so shutdown is responsive.
            slept = 0.0
            while slept < self._dispatch_tick_interval and not self._shutdown:
                time.sleep(1.0)
                slept += 1.0

    def clear_caps(self) -> dict:
        """Clear all cap overrides. Returns the new cap state."""
        with self._lock:
            self._cap_overrides.clear()
            self._capacity_cv.notify_all()
        log.info("All cap overrides cleared")
        return self.get_caps()

    def get_caps(self) -> dict:
        """Return current cap overrides and effective limits per bucket."""
        with self._lock:
            overrides = dict(self._cap_overrides)
        overflow_ov = self._cap_overrides.get("overflow")
        effective = {
            "total": self._max_total_workers(),
            "working": self._capacity_limit_for_bucket("working"),
            "overflow": max(0, overflow_ov) if overflow_ov is not None else self.config.effective_max_working_return_overflow(),
            "admitted": self._capacity_limit_for_bucket("admitted"),
            "building": self._capacity_limit_for_bucket("building"),
            "installing": self._capacity_limit_for_bucket("installing"),
        }
        return {"overrides": overrides, "effective": effective}

    def _capacity_bucket_for_status(self, status: str) -> str | None:
        if status == "working":
            return "working"
        if self.config.split_runtime_caps_enabled():
            if status == "idle":
                return "admitted"
            if status == "building":
                return "building"
            if status == "installing":
                return "installing"
            return None
        if status in {"idle", "waiting", "building", "installing"}:
            return "runtime"
        return None

    def _capacity_counts_locked(self) -> dict[str, int]:
        counts = Counter(bucket for bucket in self._capacity_state.values() if bucket)
        for bucket in ("working", "runtime", "admitted", "building", "installing"):
            counts.setdefault(bucket, 0)
        return counts

    def _capacity_limit_for_bucket(self, bucket: str) -> int:
        override = self._cap_overrides.get(bucket)
        if override is not None:
            return max(0, override)
        if bucket == "working":
            return self.config.effective_max_working_workers()
        if bucket == "overflow":
            return self.config.effective_max_working_return_overflow()
        if bucket == "admitted":
            return self.config.effective_max_admitted_workers()
        if bucket == "building":
            return self.config.effective_max_building_workers()
        if bucket == "installing":
            return self.config.effective_max_installing_workers()
        return self.config.effective_max_runtime_workers()

    def _working_waiter_priority(self, old_status: str | None) -> str:
        return "priority" if old_status and old_status != "idle" else "fresh"

    def _dequeue_working_waiter_locked(self, arxiv_id: str) -> None:
        kind = self._working_waiter_kind.pop(arxiv_id, None)
        if kind is None:
            return
        queue = (
            self._working_priority_waiters
            if kind == "priority"
            else self._working_fresh_waiters
        )
        try:
            queue.remove(arxiv_id)
        except ValueError:
            pass

    def _enqueue_working_waiter_locked(self, job: PaperJob, old_status: str | None) -> None:
        desired = self._working_waiter_priority(old_status)
        current = self._working_waiter_kind.get(job.arxiv_id)
        if current == desired:
            return
        if current is not None:
            self._dequeue_working_waiter_locked(job.arxiv_id)
        queue = (
            self._working_priority_waiters
            if desired == "priority"
            else self._working_fresh_waiters
        )
        queue.append(job.arxiv_id)
        self._working_waiter_kind[job.arxiv_id] = desired

    def _working_can_enter_locked(self, job: PaperJob, old_status: str | None) -> bool:
        counts = self._capacity_counts_locked()
        base_limit = self._capacity_limit_for_bucket("working")
        overflow_override = self._cap_overrides.get("overflow")
        overflow_limit = max(0, overflow_override) if overflow_override is not None else self.config.effective_max_working_return_overflow()

        kind = self._working_waiter_kind.get(job.arxiv_id)
        if kind == "priority":
            return (
                counts["working"] < base_limit + overflow_limit
                and bool(self._working_priority_waiters)
                and self._working_priority_waiters[0] == job.arxiv_id
            )
        if kind == "fresh":
            return (
                counts["working"] < base_limit
                and bool(self._working_fresh_waiters)
                and self._working_fresh_waiters[0] == job.arxiv_id
            )

        desired = self._working_waiter_priority(old_status or job.worker_status)
        if desired == "priority":
            return (
                counts["working"] < base_limit + overflow_limit
                and not self._working_priority_waiters
            )
        return (
            counts["working"] < base_limit
            and not self._working_fresh_waiters
        )

    def _status_can_enter_locked(self, job: PaperJob, bucket: str | None, *, old_status: str | None = None) -> bool:
        if bucket is None or self._shutdown:
            return True
        if bucket == "working":
            return self._working_can_enter_locked(job, old_status)
        counts = self._capacity_counts_locked()
        limit = self._capacity_limit_for_bucket(bucket)
        return counts[bucket] < limit

    def _dispatch_next_locked(self) -> PaperJob | None:
        """Reserve the next queued paper while holding ``self._lock``."""
        if self._shutdown:
            return None
        if not self.queue:
            return None
        if len(self.active) >= self._max_total_workers():
            return None

        counts = self._capacity_counts_locked()
        initial_bucket = self._capacity_bucket_for_status("idle")
        if initial_bucket and counts[initial_bucket] >= self._capacity_limit_for_bucket(initial_bucket):
            return None

        job = self.queue.popleft()
        job.status = "active"
        job.started_at = time.time()
        self.active[job.arxiv_id] = job
        self._capacity_state[job.arxiv_id] = initial_bucket
        return job

    def _launch_dispatched_job(self, job: PaperJob) -> None:
        """Persist and start a job that was already reserved under lock."""
        self._save_paper_meta(job)
        self._executor.submit(self._run_paper_safe, job)

    def _rollback_reserved_dispatch(self, job: PaperJob) -> None:
        """Undo a queued-job reservation that was never launched."""
        with self._capacity_cv:
            current = self.active.get(job.arxiv_id)
            if current is job:
                self.active.pop(job.arxiv_id, None)
            self._capacity_state.pop(job.arxiv_id, None)
            self._dequeue_working_waiter_locked(job.arxiv_id)
            job.status = "queued"
            job.started_at = 0.0
            job.blocked_on = None
            self.queue.appendleft(job)
            self._capacity_cv.notify_all()

    def _set_worker_status(self, job: PaperJob, new_status: str, *, save_meta: bool = True) -> None:
        if job.worker_status == new_status and job.blocked_on is None:
            if save_meta:
                self._save_paper_meta(job)
            return

        old_bucket = self._capacity_state.get(job.arxiv_id)
        new_bucket = self._capacity_bucket_for_status(new_status)
        initial_bucket = self._capacity_bucket_for_status("idle")
        preserve_initial_slot = old_bucket == initial_bucket and new_status == "working"
        old_status = job.worker_status
        queued_job: PaperJob | None = None
        blocked = False
        with self._capacity_cv:
            if new_status != "working":
                self._dequeue_working_waiter_locked(job.arxiv_id)

            can_enter = old_bucket == new_bucket or self._status_can_enter_locked(
                job, new_bucket, old_status=old_status
            )
            if can_enter:
                if new_bucket is None:
                    self._capacity_state.pop(job.arxiv_id, None)
                else:
                    self._capacity_state[job.arxiv_id] = new_bucket
                job.blocked_on = None
                job.set_worker_status(new_status)
                if new_status == "working":
                    self._dequeue_working_waiter_locked(job.arxiv_id)
                self._capacity_cv.notify_all()
            else:
                blocked = True
                if old_bucket is not None and not preserve_initial_slot:
                    self._capacity_state.pop(job.arxiv_id, None)
                job.blocked_on = new_status
                if new_status == "working":
                    self._enqueue_working_waiter_locked(job, old_status)
                self._capacity_cv.notify_all()

            if old_bucket == initial_bucket and old_bucket != self._capacity_state.get(job.arxiv_id):
                queued_job = self._dispatch_next_locked()

        if save_meta:
            try:
                self._save_paper_meta(job)
            except Exception:
                if queued_job is not None:
                    self._rollback_reserved_dispatch(queued_job)
                raise
        if queued_job is not None:
            self._launch_dispatched_job(queued_job)

        if not blocked:
            return

        block_start = time.time()
        with self._capacity_cv:
            while not self._status_can_enter_locked(job, new_bucket, old_status=old_status):
                self._capacity_cv.wait(timeout=5)
            if new_bucket is None:
                self._capacity_state.pop(job.arxiv_id, None)
            else:
                self._capacity_state[job.arxiv_id] = new_bucket
            job.blocked_on = None
            job.set_worker_status(new_status)
            if new_status == "working":
                self._dequeue_working_waiter_locked(job.arxiv_id)
            if old_bucket == initial_bucket and old_bucket != self._capacity_state.get(job.arxiv_id):
                queued_job = self._dispatch_next_locked()
            self._capacity_cv.notify_all()
        job.blocked_sec += time.time() - block_start

        if save_meta:
            try:
                self._save_paper_meta(job)
            except Exception:
                if queued_job is not None:
                    self._rollback_reserved_dispatch(queued_job)
                raise
        if queued_job is not None:
            self._launch_dispatched_job(queued_job)

    def _run_wrap_up_turn(self, job: PaperJob, worker: Any, prompt: str) -> None:
        """Run a wrap-up agent turn while exposing a distinct active state."""
        self._set_worker_status(job, "wrapping_up")
        try:
            worker.chat(prompt)
        finally:
            actual_status = worker.get_status()
            if actual_status != job.worker_status:
                self._set_worker_status(job, actual_status)

    def cancel_paper(self, arxiv_id: str) -> bool:
        """Cancel an active or queued paper gracefully.

        Sets a flag that the worker thread checks each cycle. The thread
        performs cleanup (setup merge, git push) before marking the paper
        as "cancelled". The paper stays visible on the dashboard.

        Returns True if the paper was found and flagged.
        """
        with self._lock:
            # Remove from queue if not started yet
            for j in list(self.queue):
                if j.arxiv_id == arxiv_id:
                    self.queue.remove(j)
                    j.status = "cancelled"
                    self.completed[arxiv_id] = PaperResult(
                        arxiv_id=arxiv_id,
                        milestone=j.milestone,
                        status="cancelled",
                        finished_at=time.time(),
                        profile=j.profile,
                        iteration=j.iteration,
                        workflow=j.workflow,
                        workflow_step=j.workflow_step,
                    )
                    self._save_paper_meta(j)
                    log.info("Cancelled queued paper %s", arxiv_id)
                    return True

            # Flag active paper for cancellation — the worker thread
            # will pick this up on its next cycle.
            if arxiv_id in self.active:
                self._cancelled.add(arxiv_id)
                worker = self._active_workers.get(arxiv_id)
                if worker is not None:
                    worker.request_cancel()
                log.info("Flagged active paper %s for cancellation", arxiv_id)
                return True

        log.warning("Cannot cancel %s: not active or queued", arxiv_id)
        return False

    def _infer_recovery_action(self, meta: dict, original_status: str) -> str:
        if not meta.get("is_continuation") and not meta.get("workspace_path"):
            workspace_exists = False
            arxiv_id = meta.get("arxiv_id")
            if arxiv_id:
                workspace_exists = (Path(self.config.workspace_base_dir) / arxiv_id).exists()
            if not workspace_exists:
                return "submit"
        if original_status == "queued" and meta.get("is_iteration"):
            return "iterate"
        return "continue"

    def _mark_meta_for_recovery(self, meta: dict, *, original_status: str) -> dict:
        meta["recovery_action"] = self._infer_recovery_action(meta, original_status)
        meta["recovery_status"] = original_status
        meta["status"] = _RESTART_RECOVERY_STATUS
        meta["error"] = _RESTART_RECOVERY_MESSAGE
        return meta

    @staticmethod
    def _mark_cancelled_finalization_for_recovery(meta: dict) -> dict:
        meta["recovery_action"] = "finalize_cancelled"
        meta["recovery_status"] = "cancelled"
        meta["status"] = _RESTART_RECOVERY_STATUS
        meta["error"] = _RESTART_RECOVERY_MESSAGE
        return meta

    def _write_cancelled_recovery_meta(self, job: PaperJob) -> bool:
        with self._lock:
            active_job = self.active.get(job.arxiv_id)
            finalizing_job = self._finalizing.get(job.arxiv_id)
            if active_job is not job and finalizing_job is not job:
                return False
        paper_dir = PAPERS_DIR / job.arxiv_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        meta_path = paper_dir / "meta.json"
        meta = job.to_dict()
        self._mark_cancelled_finalization_for_recovery(meta)
        meta_path.write_text(json.dumps(meta, indent=2))
        return True

    def _write_job_recovery_meta(self, job: PaperJob, *, original_status: str) -> bool:
        if original_status == "active":
            with self._lock:
                active_job = self.active.get(job.arxiv_id)
                finalizing_job = self._finalizing.get(job.arxiv_id)
                if active_job is not job and finalizing_job is not job:
                    return False
        paper_dir = PAPERS_DIR / job.arxiv_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        meta_path = paper_dir / "meta.json"
        meta = job.to_dict()
        self._mark_meta_for_recovery(meta, original_status=original_status)
        meta_path.write_text(json.dumps(meta, indent=2))
        return True

    def prepare_all_for_restart(self) -> dict[str, int]:
        """Mark queued/active papers so continue-all can recover them later."""
        with self._lock:
            queued_jobs = list(self.queue)
            active_jobs = list(self.active.values())
            finalizing_jobs = list(self._finalizing.values())
            queued_ids = {job.arxiv_id for job in queued_jobs}
            active_ids = {job.arxiv_id for job in active_jobs}
            self.queue.clear()
            for job in queued_jobs:
                job.status = _RESTART_RECOVERY_STATUS
                job.error = _RESTART_RECOVERY_MESSAGE
                self.completed[job.arxiv_id] = PaperResult(
                    arxiv_id=job.arxiv_id,
                    milestone=job.milestone,
                    status=_RESTART_RECOVERY_STATUS,
                    error=_RESTART_RECOVERY_MESSAGE,
                    finished_at=time.time(),
                    profile=job.profile,
                    iteration=job.iteration,
                    workflow=job.workflow,
                    workflow_step=job.workflow_step,
                )

        queued_marked = 0
        for job in queued_jobs:
            if self._write_job_recovery_meta(job, original_status="queued"):
                queued_marked += 1
        active_marked = 0
        for job in active_jobs:
            if job.status == "cancelled":
                if self._write_cancelled_recovery_meta(job):
                    active_marked += 1
                continue
            if self._write_job_recovery_meta(job, original_status="active"):
                active_marked += 1
        for job in finalizing_jobs:
            if job.arxiv_id in queued_ids or job.arxiv_id in active_ids:
                continue
            if job.status == "cancelled":
                self._write_cancelled_recovery_meta(job)
                continue
            if self._write_job_recovery_meta(job, original_status="active"):
                active_marked += 1

        log.info(
            "Prepared %d queued and %d active papers for restart recovery",
            queued_marked,
            active_marked,
        )
        return {"queued": queued_marked, "active": active_marked}

    # ── Public API ────────────────────────────────────────────────────

    def enqueue(self, arxiv_id: str, gpu_budget: float | None = None,
                date: str | None = None,
                topics: list[str] | None = None,
                profile: str | None = None,
                workflow: str | None = None,
                dispatch: bool = True) -> PaperJob:
        """Add a paper to the queue and dispatch if workers are free.

        Workspace creation (file copies, git init, GitHub repo) is deferred
        to when a worker picks up the paper.  This keeps enqueue fast and
        avoids GitHub API timeouts blocking the submit endpoint.

        Args:
            arxiv_id: The arxiv identifier.
            gpu_budget: Override per-paper GPU budget (hours).
            date: Paper date (YYYY-MM-DD), added as GitHub topic.
            topics: Extra GitHub topics to add at repo creation.

        Returns:
            The created PaperJob.

        Raises:
            ValueError: If the paper is already queued/active/completed,
                or is on the exclude list.
        """
        _guard_excluded(arxiv_id)
        with self._lock:
            if arxiv_id in self.active:
                raise ValueError(f"Paper {arxiv_id} is already active")
            if arxiv_id in self.completed:
                raise ValueError(f"Paper {arxiv_id} is already completed")
            if any(j.arxiv_id == arxiv_id for j in self.queue):
                raise ValueError(f"Paper {arxiv_id} is already in the queue")

        # Resolve paper paths (validates paper exists, no I/O)
        paper = resolve_paper(arxiv_id, self.config.scout_data_dir)

        # Use default workflow if none specified
        if workflow is None and self.config.default_workflow:
            workflow = self.config.default_workflow

        # If workflow specified, validate and use first step's profile
        effective_profile = profile
        if workflow:
            wf_steps = self.config.workflows.get(workflow)
            if not wf_steps:
                raise ValueError(
                    f"Workflow '{workflow}' not found. "
                    f"Available: {list(self.config.workflows.keys())}"
                )
            if not effective_profile:
                effective_profile = wf_steps[0].profile

        budget = gpu_budget
        if budget is None:
            budget = _workflow_step_gpu_budget(self.config, workflow, 0)
        if budget is None:
            budget = self.config.nexus.gpu_budget_hours

        job = PaperJob(
            arxiv_id=arxiv_id,
            paper=paper,
            workspace_path="",  # set when worker picks it up
            gpu_budget=budget,
            profile=effective_profile,
            total_gpu_hours_used=0.0,
            date=date,
            topics=topics,
            workflow=workflow,
            workflow_step=0,
        )

        # Save paper metadata
        self._save_paper_meta(job)

        with self._lock:
            self.queue.append(job)

        if dispatch:
            self._dispatch_next()
        return job

    def continue_paper(self, arxiv_id: str, profile: str | None = None,
                        no_workflow: bool = False, fresh: bool = False,
                        dispatch: bool = True, skip_github: bool = False,
                        queue_front: bool = False) -> PaperJob:
        """Resume a previously started paper using its existing workspace.

        Reads session_id from meta.json so the agent can ``--resume``
        the same Claude Code session and retain full conversation history.

        Raises:
            ValueError: If the paper is currently active/queued or on the
                exclude list.
            FileNotFoundError: If no workspace or meta.json exists.
        """
        _guard_excluded(arxiv_id)
        with self._lock:
            if arxiv_id in self.active:
                raise ValueError(f"Paper {arxiv_id} is already active")
            if any(j.arxiv_id == arxiv_id for j in self.queue):
                raise ValueError(f"Paper {arxiv_id} is already in the queue")
            # Remove from completed so it can be re-activated
            self.completed.pop(arxiv_id, None)

        workspace = Path(self.config.workspace_base_dir) / arxiv_id
        if not workspace.is_dir():
            # Try cloning from GitHub if the repo exists
            paper_tmp = resolve_paper(arxiv_id, self.config.scout_data_dir)
            if self.config.github_org:
                cloned = clone_workspace(
                    paper_tmp, self.config.workspace_base_dir, self.config.github_org,
                )
                if cloned:
                    workspace = Path(cloned)
            if not workspace.is_dir():
                raise FileNotFoundError(f"No workspace found at {workspace}")

        # Load previous state from meta.json
        meta_path = PAPERS_DIR / arxiv_id / "meta.json"
        session_id = None
        prev_budget = self.config.nexus.gpu_budget_hours
        prev_gpu_hours_used = 0.0
        prev_total_gpu_hours_used = 0.0
        prev_iteration = 1
        prev_total_duration = 0.0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            session_id = meta.get("session_id")
            prev_budget = meta.get("gpu_budget", prev_budget)
            prev_gpu_hours_used = meta.get("gpu_hours_used", 0.0)
            prev_total_gpu_hours_used = meta.get("total_gpu_hours_used", prev_gpu_hours_used)
            prev_iteration = meta.get("iteration", 1)
            prev_total_duration = meta.get("total_duration_sec", 0.0)
        if not session_id:
            traj_session = workspace / "trajectory" / "session.json"
            if traj_session.exists():
                session_id = json.loads(traj_session.read_text()).get("session_id")

        # If the previous run ended below target_milestone, clear the
        # session so the agent starts fresh instead of resuming stale
        # context where it already decided to quit.
        prev_milestone = meta.get("milestone", "none") if meta_path.exists() else "none"
        target_rank = _MILESTONE_RANK.get(self.config.target_milestone, 2)
        if session_id and _MILESTONE_RANK.get(prev_milestone, 0) < target_rank:
            log.info(
                "Paper %s: previous milestone=%s < target=%s, "
                "clearing session for fresh start",
                arxiv_id, prev_milestone, self.config.target_milestone,
            )
            session_id = None
            # Also clear the stale session file so the worker doesn't
            # pick it up from disk.
            traj_session = workspace / "trajectory" / "session.json"
            if traj_session.exists():
                traj_session.unlink()

        # Force fresh start if previous run was interrupted mid-work.
        # Resuming a dirty session can trap the agent in stale tool calls.
        # Check both current status (dirty stop without pause-all) and
        # recovery_status (pause-all or manual recovery marking — agents
        # are killed either way).
        if session_id and meta_path.exists():
            prev_status = meta.get("status", "")
            recovery_status = meta.get("recovery_status", "")
            if prev_status in ("active", "paused") or recovery_status in ("active", "paused"):
                dirty_reason = prev_status if prev_status in ("active", "paused") else recovery_status
                log.info(
                    "Paper %s: previous status=%s (interrupted), "
                    "forcing fresh start to avoid dirty session resume",
                    arxiv_id, dirty_reason,
                )
                import shutil
                session_id = None
                traj_session = workspace / "trajectory" / "session.json"
                if traj_session.exists():
                    traj_session.unlink()
                claude_projects = workspace / ".claude" / "projects"
                if claude_projects.is_dir():
                    shutil.rmtree(claude_projects, ignore_errors=True)

        if fresh:
            # Wipe Claude Code's internal session storage so it can't
            # auto-resume a bloated old session.
            import shutil
            session_id = None
            traj_session = workspace / "trajectory" / "session.json"
            if traj_session.exists():
                traj_session.unlink()
            claude_projects = workspace / ".claude" / "projects"
            if claude_projects.is_dir():
                shutil.rmtree(claude_projects, ignore_errors=True)
            log.info("Paper %s: fresh start — cleared session storage", arxiv_id)

        paper = resolve_paper(arxiv_id, self.config.scout_data_dir)

        # Carry forward workflow state (unless detached)
        if no_workflow:
            prev_workflow = None
            prev_workflow_step = 0
        else:
            prev_workflow = meta.get("workflow") if meta_path.exists() else None
            prev_workflow_step = meta.get("workflow_step", 0) if meta_path.exists() else 0

        # Re-resolve gpu_budget from the workflow step config rather than
        # trusting whatever was stored in meta.json. This avoids stale
        # budgets when a paper is rolled back to an earlier iteration,
        # whose step has a different budget than the one last recorded.
        workflow_budget = _workflow_step_gpu_budget(
            self.config, prev_workflow, prev_workflow_step
        )
        if workflow_budget is not None:
            if abs(workflow_budget - prev_budget) > 1e-6:
                log.info(
                    "Paper %s: gpu_budget %.1f -> %.1f "
                    "(resolved from workflow %s step %d)",
                    arxiv_id, prev_budget, workflow_budget,
                    prev_workflow, prev_workflow_step,
                )
            prev_budget = workflow_budget

        # Use explicit profile, or resolve from workflow step, or fall back to meta
        effective_profile = profile
        effective_role = None
        if prev_workflow:
            wf_steps = self.config.workflows.get(prev_workflow, [])
            if prev_workflow_step < len(wf_steps):
                step_cfg = wf_steps[prev_workflow_step]
                if not effective_profile:
                    effective_profile = step_cfg.profile
                # Always re-resolve role from the current workflow config so
                # the agent runs the right prompt when the workflow ordering
                # changes after the paper was first scheduled.
                effective_role = getattr(step_cfg, "role", None) or "reproducer"
        if not effective_profile:
            effective_profile = meta.get("profile") if meta_path.exists() else None
        if effective_role is None:
            effective_role = meta.get("role", "reproducer") if meta_path.exists() else "reproducer"
        else:
            stored_role = meta.get("role") if meta_path.exists() else None
            if stored_role and stored_role != effective_role:
                log.info(
                    "Paper %s: role %s -> %s (resolved from workflow %s step %d)",
                    arxiv_id, stored_role, effective_role,
                    prev_workflow, prev_workflow_step,
                )

        # Carry forward iteration state so _run_paper sends the right prompt
        prev_is_iteration = meta.get("is_iteration", False) if meta_path.exists() else False
        prev_previous_milestone = (
            (meta.get("previous_milestone") or "none") if meta_path.exists() else "none"
        )

        job = PaperJob(
            arxiv_id=arxiv_id,
            paper=paper,
            workspace_path=str(workspace),
            gpu_budget=prev_budget,
            profile=effective_profile,
            gpu_hours_used=prev_gpu_hours_used,
            total_gpu_hours_used=prev_total_gpu_hours_used,
            milestone=prev_milestone,
            session_id=session_id,
            is_continuation=True,
            is_iteration=prev_is_iteration,
            iteration=prev_iteration,
            previous_milestone=prev_previous_milestone,
            total_duration_sec=prev_total_duration,
            role=effective_role,
            workflow=prev_workflow,
            workflow_step=prev_workflow_step,
            date=meta.get("date") if meta_path.exists() else None,
            topics=meta.get("topics") if meta_path.exists() else None,
            last_env_failure_container_hash=(
                meta.get("last_env_failure_container_hash") if meta_path.exists() else None
            ),
            last_env_failure_setup_hash=(
                meta.get("last_env_failure_setup_hash") if meta_path.exists() else None
            ),
            last_env_failure_signature=(
                meta.get("last_env_failure_signature") if meta_path.exists() else None
            ),
            last_env_failure_class=(
                meta.get("last_env_failure_class") if meta_path.exists() else None
            ),
            last_container_hash=(
                meta.get("last_container_hash") if meta_path.exists() else None
            ),
            last_setup_hash=(
                meta.get("last_setup_hash") if meta_path.exists() else None
            ),
            revert_attempts=(
                meta.get("revert_attempts", 0) if meta_path.exists() else 0
            ),
        )

        # Ensure GitHub remote is set up (skipped during bulk recovery —
        # deferred to _run_paper where it runs in a worker thread)
        if self.config.github_org and not skip_github:
            ensure_github_remote(
                str(workspace), paper, self.config.github_org,
            )

        self._save_paper_meta(job)

        with self._lock:
            if queue_front:
                self.queue.appendleft(job)
            else:
                self.queue.append(job)

        if dispatch:
            self._dispatch_next()
        return job

    def iterate_paper(self, arxiv_id: str, profile: str | None = None,
                      role: str = "reproducer",
                      gpu_budget: float | None = None,
                      workflow_step_override: int | None = None,
                      workflow_override: str | None = None,
                      date: str | None = None,
                      topics: list[str] | None = None,
                      queue_front: bool = False,
                      dispatch: bool = True,
                      skip_github: bool = False,
                      fresh: bool = False,
                      revert_attempts_override: int | None = None) -> "PaperJob":
        """Iterate on a paper with a fresh agent that audits previous work.

        Unlike continue_paper (which resumes the same session), iterate starts
        a new agent session that first audits the existing work, then fixes
        issues or confirms and pushes toward higher milestones.

        Args:
            role: "reproducer" (default) drops milestone and resets workspace.
                  "verifier" keeps milestone and workspace intact for QC only.

        The milestone is dropped one level from the previous claim (to account
        for potential inflation). MILESTONES.md is reset with the dropped level.

        Raises:
            ValueError: If the paper is currently active/queued or is on
                the exclude list.
            FileNotFoundError: If no workspace exists.
        """
        _guard_excluded(arxiv_id)
        with self._lock:
            if arxiv_id in self.active:
                raise ValueError(f"Paper {arxiv_id} is already active")
            if any(j.arxiv_id == arxiv_id for j in self.queue):
                raise ValueError(f"Paper {arxiv_id} is already in the queue")

        workspace = Path(self.config.workspace_base_dir) / arxiv_id
        if not workspace.is_dir():
            paper_tmp = resolve_paper(arxiv_id, self.config.scout_data_dir)
            if self.config.github_org:
                cloned = clone_workspace(
                    paper_tmp, self.config.workspace_base_dir, self.config.github_org,
                )
                if cloned:
                    workspace = Path(cloned)
            if not workspace.is_dir():
                raise FileNotFoundError(f"No workspace found at {workspace}")

        # Load previous state
        meta_path = PAPERS_DIR / arxiv_id / "meta.json"
        prev_gpu_hours_used = 0.0
        prev_total_gpu_hours_used = 0.0
        prev_milestone = "none"
        prev_iteration = 1
        prev_revert_attempts = 0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            prev_gpu_hours_used = meta.get("gpu_hours_used", 0.0)
            prev_total_gpu_hours_used = meta.get("total_gpu_hours_used", prev_gpu_hours_used)
            prev_milestone = meta.get("milestone", "none")
            prev_iteration = meta.get("iteration", 1)
            prev_revert_attempts = meta.get("revert_attempts", 0)

        prev_workflow = meta.get("workflow") if meta_path.exists() else None
        prev_workflow_step = meta.get("workflow_step", 0) if meta_path.exists() else 0
        prev_total_duration = meta.get("total_duration_sec", 0.0) if meta_path.exists() else 0.0

        # Allow the caller to move the paper to a different workflow, for
        # example to re-run only some stages on papers that already finished.
        if workflow_override is not None and workflow_override not in self.config.workflows:
            raise ValueError(
                f"Unknown workflow '{workflow_override}' "
                f"(valid: {sorted(self.config.workflows.keys())})"
            )
        effective_workflow = (
            workflow_override if workflow_override is not None else prev_workflow
        )

        # For workflow-backed papers, iteration is the 1-indexed workflow
        # position, not a running attempt counter: iteration is always
        # workflow_step + 1. Advances raise it, reverts lower it,
        # same-step retries keep it. Papers without a workflow count
        # iterations instead.
        if effective_workflow:
            effective_step = (
                workflow_step_override
                if workflow_step_override is not None
                else prev_workflow_step
            )
            iteration = effective_step + 1
        elif workflow_step_override is None:
            # Non-workflow manual iterate: bump counter.
            iteration = prev_iteration + 1
        else:
            iteration = prev_iteration

        # Determine workflow step and role: explicit override wins.
        # Manual iterate/verify stays on the current workflow step; only
        # explicit overrides (used by workflow auto-advance) move to the next.
        effective_workflow_step = (
            workflow_step_override if workflow_step_override is not None else prev_workflow_step
        )
        wf_steps = self.config.workflows.get(effective_workflow, []) if effective_workflow else []
        if workflow_step_override is not None:
            if not effective_workflow:
                raise ValueError("workflow_step_override requires a workflow-backed paper")
            if effective_workflow_step < 0 or effective_workflow_step >= len(wf_steps):
                raise ValueError(
                    f"Workflow step {effective_workflow_step} is out of range for "
                    f"workflow '{effective_workflow}' (valid: 0-{len(wf_steps) - 1})"
                )

        # Determine role: explicit parameter > workflow step config > default
        step_role = role
        if step_role == "reproducer" and effective_workflow:
            if 0 <= effective_workflow_step < len(wf_steps):
                step_role = wf_steps[effective_workflow_step].role

        effective_profile = profile
        if effective_profile is None and effective_workflow and 0 <= effective_workflow_step < len(wf_steps):
            effective_profile = wf_steps[effective_workflow_step].profile

        if step_role == "verifier" or step_role in _MILESTONE_LOCKED_ROLES:
            # Verifier / packager / designer: keep milestone as-is, don't
            # reset workspace. Packager and designer additionally cannot
            # change the milestone at all (enforced in _run_paper).
            starting_milestone = prev_milestone
            # Designer stages 1 & 2 write files only — no jobs. The
            # previous iteration's action.yaml still sits in the workspace
            # with ``action: submit`` filled in, and the designer agent
            # reads that as a working template and pattern-matches into
            # submitting a job. Reset action.yaml to a neutral state
            # (``pending`` + ``gpus: 0``) so there's no working submit
            # example to copy from. We deliberately do NOT reset
            # MILESTONES.md — these stages must not touch the milestone.
            if step_role in ("designer_stage_1", "designer_stage_2"):
                self._reset_action_yaml_for_design_only(workspace, starting_milestone)
        else:
            # Reproducer: drop milestone one level, reset workspace for fresh audit
            starting_milestone = _drop_milestone(prev_milestone)
            self._reset_workspace_for_iteration(workspace, starting_milestone)

        effective_budget = gpu_budget
        if effective_budget is None:
            effective_budget = _workflow_step_gpu_budget(
                self.config, effective_workflow, effective_workflow_step
            )
        if effective_budget is None:
            effective_budget = self.config.nexus.gpu_budget_hours

        # Always clear session — iteration is a fresh agent
        traj_session = workspace / "trajectory" / "session.json"
        if traj_session.exists():
            traj_session.unlink()
        if fresh:
            # Wipe Claude Code's auto-resume projects dir so nothing stale
            # is picked up when the worker runs without --resume.  Matches
            # continue_paper(fresh=True) semantics.
            import shutil as _shutil
            claude_projects = workspace / ".claude" / "projects"
            if claude_projects.is_dir():
                _shutil.rmtree(claude_projects, ignore_errors=True)

        paper = resolve_paper(arxiv_id, self.config.scout_data_dir)

        job = PaperJob(
            arxiv_id=arxiv_id,
            paper=paper,
            workspace_path=str(workspace),
            gpu_budget=effective_budget,
            profile=effective_profile,
            gpu_hours_used=0.0,
            total_gpu_hours_used=prev_total_gpu_hours_used,
            milestone=starting_milestone,
            session_id=None,  # always fresh
            is_continuation=True,  # reuse workspace
            is_iteration=True,
            iteration=iteration,
            previous_milestone=prev_milestone,
            role=step_role,
            workflow=effective_workflow,
            workflow_step=effective_workflow_step,
            total_duration_sec=prev_total_duration,
            date=date or (meta.get("date") if meta_path.exists() else None),
            topics=topics or (meta.get("topics") if meta_path.exists() else None),
            last_env_failure_container_hash=(
                meta.get("last_env_failure_container_hash") if meta_path.exists() else None
            ),
            last_env_failure_setup_hash=(
                meta.get("last_env_failure_setup_hash") if meta_path.exists() else None
            ),
            last_env_failure_signature=(
                meta.get("last_env_failure_signature") if meta_path.exists() else None
            ),
            last_env_failure_class=(
                meta.get("last_env_failure_class") if meta_path.exists() else None
            ),
            last_container_hash=(
                meta.get("last_container_hash") if meta_path.exists() else None
            ),
            last_setup_hash=(
                meta.get("last_setup_hash") if meta_path.exists() else None
            ),
            revert_attempts=(
                revert_attempts_override
                if revert_attempts_override is not None
                else prev_revert_attempts
            ),
        )

        if self.config.github_org and not skip_github:
            ensure_github_remote(
                str(workspace), paper, self.config.github_org,
            )

        self._save_paper_meta(job)

        with self._lock:
            self.completed.pop(arxiv_id, None)
            if queue_front:
                self.queue.appendleft(job)
            else:
                self.queue.append(job)

        if dispatch:
            self._dispatch_next()
        return job

    def recover_paper(
        self,
        arxiv_id: str,
        profile: str | None = None,
        no_workflow: bool = False,
        fresh: bool = False,
        dispatch: bool = True,
        skip_github: bool = False,
    ) -> PaperJob:
        """Recover a daemon-interrupted paper using persisted recovery intent."""
        with self._lock:
            if arxiv_id in self.active:
                raise ValueError(f"Paper {arxiv_id} is already active")
            if any(j.arxiv_id == arxiv_id for j in self.queue):
                raise ValueError(f"Paper {arxiv_id} is already in the queue")
            self.completed.pop(arxiv_id, None)

        meta_path = PAPERS_DIR / arxiv_id / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No state found for {arxiv_id}")
        meta = json.loads(meta_path.read_text())
        recovery_action = meta.get("recovery_action")
        if not recovery_action:
            raise ValueError(
                f"Paper {arxiv_id} has no recorded recovery action (it was not "
                f"interrupted by a daemon restart); use continue or iterate instead"
            )

        if recovery_action == "continue":
            return self.continue_paper(
                arxiv_id, profile=profile, no_workflow=no_workflow, fresh=fresh,
                dispatch=dispatch, skip_github=skip_github,
            )

        if recovery_action == "submit":
            effective_workflow = "" if no_workflow else meta.get("workflow")
            # Pass gpu_budget=None so enqueue() resolves it from the workflow
            # step config rather than reusing a potentially stale meta value.
            return self.enqueue(
                arxiv_id,
                gpu_budget=None,
                date=meta.get("date"),
                topics=meta.get("topics"),
                profile=profile if profile is not None else meta.get("profile"),
                workflow=effective_workflow,
                dispatch=dispatch,
            )

        if recovery_action == "finalize_cancelled":
            workspace = Path(meta.get("workspace_path") or "")
            if not workspace.is_dir():
                paper_tmp = resolve_paper(arxiv_id, self.config.scout_data_dir)
                if self.config.github_org:
                    cloned = clone_workspace(
                        paper_tmp, self.config.workspace_base_dir, self.config.github_org,
                    )
                    if cloned:
                        workspace = Path(cloned)
                if not workspace.is_dir():
                    raise FileNotFoundError(f"No workspace found at {workspace}")

            env_dir = workspace / "environment"
            _freeze_overlay_into_containerdef(
                workspace, env_dir / "container.def", env_dir / "setup.sh",
                fallback_sif=self.config.worker.container_sif,
            )

            paper = resolve_paper(arxiv_id, self.config.scout_data_dir)
            job = PaperJob(
                arxiv_id=arxiv_id,
                paper=paper,
                workspace_path=str(workspace),
                gpu_budget=meta.get("gpu_budget", self.config.nexus.gpu_budget_hours),
                profile=profile if profile is not None else meta.get("profile"),
                status="cancelled",
                milestone=meta.get("milestone", "none"),
                worker_status=meta.get("worker_status", "idle"),
                last_job_resources=meta.get("last_job_resources", ""),
                error=None,
                started_at=meta.get("started_at", 0.0),
                finished_at=time.time(),
                session_id=meta.get("session_id"),
                is_continuation=meta.get("is_continuation", False),
                is_iteration=meta.get("is_iteration", False),
                iteration=meta.get("iteration", 1),
                previous_milestone=meta.get("previous_milestone", "none"),
                role=meta.get("role", "reproducer"),
                workflow=None if no_workflow else meta.get("workflow"),
                workflow_step=meta.get("workflow_step", 0),
                gpu_hours_used=meta.get("gpu_hours_used", 0.0),
                total_gpu_hours_used=meta.get("total_gpu_hours_used", meta.get("gpu_hours_used", 0.0)),
                total_duration_sec=meta.get("total_duration_sec", 0.0),
                date=meta.get("date"),
                topics=meta.get("topics"),
            )
            self._save_paper_meta(job)
            with self._lock:
                self.completed[arxiv_id] = PaperResult(
                    arxiv_id=arxiv_id,
                    milestone=job.milestone,
                    status="cancelled",
                    finished_at=job.finished_at,
                    profile=job.profile,
                    iteration=job.iteration,
                    workflow=job.workflow,
                    workflow_step=job.workflow_step,
                )
            return job

        if recovery_action != "iterate":
            raise ValueError(f"Unknown recovery_action '{recovery_action}' for {arxiv_id}")

        workspace = Path(meta.get("workspace_path") or "")
        if not workspace.is_dir():
            paper_tmp = resolve_paper(arxiv_id, self.config.scout_data_dir)
            if self.config.github_org:
                cloned = clone_workspace(
                    paper_tmp, self.config.workspace_base_dir, self.config.github_org,
                )
                if cloned:
                    workspace = Path(cloned)
            if not workspace.is_dir():
                raise FileNotFoundError(f"No workspace found at {workspace}")

        if fresh:
            import shutil
            traj_session = workspace / "trajectory" / "session.json"
            if traj_session.exists():
                traj_session.unlink()
            claude_projects = workspace / ".claude" / "projects"
            if claude_projects.is_dir():
                shutil.rmtree(claude_projects, ignore_errors=True)

        paper = resolve_paper(arxiv_id, self.config.scout_data_dir)
        effective_profile = profile if profile is not None else meta.get("profile")
        effective_workflow = None if no_workflow else meta.get("workflow")
        effective_workflow_step = (
            meta.get("workflow_step", 0) if effective_workflow else 0
        )
        role = meta.get("role", "reproducer")
        previous_milestone = meta.get("previous_milestone") or "none"
        if effective_workflow:
            wf_steps = self.config.workflows.get(effective_workflow, [])
            if (
                effective_workflow_step < 0
                or effective_workflow_step >= len(wf_steps)
            ):
                log.warning(
                    "Recovering %s without workflow gating: saved workflow step %s "
                    "is incompatible with current workflow '%s'",
                    arxiv_id,
                    effective_workflow_step,
                    effective_workflow,
                )
                effective_workflow = None
                effective_workflow_step = 0

        # Resolve gpu_budget from the workflow step config rather than
        # trusting meta.json — avoids stale budgets when a paper is
        # rolled back to a different iteration before recovery.
        meta_budget = meta.get("gpu_budget", self.config.nexus.gpu_budget_hours)
        workflow_budget = _workflow_step_gpu_budget(
            self.config, effective_workflow, effective_workflow_step
        )
        effective_iterate_budget = (
            workflow_budget if workflow_budget is not None else meta_budget
        )
        if workflow_budget is not None and abs(workflow_budget - meta_budget) > 1e-6:
            log.info(
                "Paper %s: gpu_budget %.1f -> %.1f "
                "(resolved from workflow %s step %d)",
                arxiv_id, meta_budget, workflow_budget,
                effective_workflow, effective_workflow_step,
            )

        job = PaperJob(
            arxiv_id=arxiv_id,
            paper=paper,
            workspace_path=str(workspace),
            gpu_budget=effective_iterate_budget,
            profile=effective_profile,
            gpu_hours_used=meta.get("gpu_hours_used", 0.0),
            total_gpu_hours_used=meta.get("total_gpu_hours_used", meta.get("gpu_hours_used", 0.0)),
            milestone=meta.get("milestone", "none"),
            session_id=None,
            is_continuation=True,
            is_iteration=True,
            iteration=meta.get("iteration", 1),
            previous_milestone=previous_milestone,
            role=role,
            workflow=effective_workflow,
            workflow_step=effective_workflow_step,
            total_duration_sec=meta.get("total_duration_sec", 0.0),
            date=meta.get("date"),
            topics=meta.get("topics"),
            last_container_hash=meta.get("last_container_hash"),
            last_setup_hash=meta.get("last_setup_hash"),
        )

        if self.config.github_org and not skip_github:
            ensure_github_remote(
                str(workspace), paper, self.config.github_org,
            )

        self._save_paper_meta(job)

        with self._lock:
            self.queue.append(job)

        if dispatch:
            self._dispatch_next()
        return job

    def recover_all_papers(
        self,
        profile: str | None = None,
        no_workflow: bool = False,
        fresh: bool = False,
    ) -> dict:
        """Recover all daemon-interrupted papers in one scheduler pass."""
        interrupted: list[dict] = []
        if PAPERS_DIR.is_dir():
            for meta_path in sorted(PAPERS_DIR.glob("*/meta.json")):
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    continue
                if _RESTART_RECOVERY_ERROR not in str(meta.get("error", "")):
                    continue
                meta["arxiv_id"] = meta.get("arxiv_id") or meta_path.parent.name
                interrupted.append(meta)

        if not interrupted:
            return {"total": 0, "queued": 0, "failed": 0, "results": []}

        results: list[dict] = []
        successes = 0
        failures = 0
        for meta in interrupted:
            arxiv_id = meta["arxiv_id"]
            action_label = meta.get("recovery_action", "recovered")
            try:
                job = self.recover_paper(
                    arxiv_id,
                    profile=profile,
                    no_workflow=no_workflow,
                    fresh=fresh,
                    dispatch=False,
                    skip_github=True,
                )
                results.append({
                    "arxiv_id": job.arxiv_id,
                    "status": job.status,
                    "recovery_action": action_label,
                })
                successes += 1
            except Exception as e:
                results.append({
                    "arxiv_id": arxiv_id,
                    "status": "failed",
                    "recovery_action": action_label,
                    "error": str(e),
                })
                failures += 1

        # Dispatch is handled by the normal scheduler health loop — no need
        # to block here.  Trigger one round so the first batch starts promptly.
        self._dispatch_next()
        return {
            "total": len(interrupted),
            "queued": successes,
            "failed": failures,
            "results": results,
        }

    @staticmethod
    def _reset_workspace_for_iteration(workspace: Path, starting_milestone: str):
        """Reset workspace files for an iteration agent."""
        # Reset MILESTONES.md with dropped milestone
        milestones_path = workspace / "MILESTONES.md"
        milestones_path.write_text(
            f"# Reproduction Milestones\n\n"
            f"**Current: {starting_milestone}**\n\n"
            f"## Progress Log\n\n"
            f"## Stop Justification\n\n"
        )

        # Reset action.yaml — includes a warning because agents sometimes
        # try to sleep/poll after writing it, blocking the turn forever.
        action_path = workspace / "action.yaml"
        action_path.write_text(
            f"# IMPORTANT: Once you set action to 'submit' or 'done', STOP immediately.\n"
            f"# Do NOT sleep, poll, or wait for jobs. The system reads this file\n"
            f"# only after your turn ends. Sleeping blocks the entire pipeline.\n"
            f"action: pending\n"
            f"gpus: 1\n"
            f"test: false\n"
            f"milestone: {starting_milestone}\n"
            f'notes: ""\n'
        )

    @staticmethod
    def _reset_action_yaml_for_design_only(workspace: Path, starting_milestone: str):
        """Neutralize action.yaml for write-only designer stages (1 & 2).

        These stages forbid job submission in their prompts, but the
        previous iteration's action.yaml sits in the workspace with a
        working ``action: submit`` template the agent copies from.
        Overwriting with ``gpus: 0`` removes that template without
        touching MILESTONES.md (which these stages must not modify).
        """
        action_path = workspace / "action.yaml"
        action_path.write_text(
            f"# This iteration writes files only — do NOT submit compute jobs.\n"
            f"# Once you set action to 'done', STOP immediately.\n"
            f"# Do NOT sleep, poll, or wait. The system reads this file\n"
            f"# only after your turn ends.\n"
            f"action: pending\n"
            f"gpus: 0\n"
            f"test: false\n"
            f"milestone: {starting_milestone}\n"
            f'notes: ""\n'
        )

    def get_status(self) -> dict:
        """Return current scheduler status."""
        with self._lock:
            counts = self._capacity_counts_locked()
            worker_status_counts = Counter(job.worker_status for job in self.active.values())
            split_runtime_caps = self.config.split_runtime_caps_enabled()
            if split_runtime_caps:
                admitted_count = counts.get("admitted", 0)
                waiting_count = worker_status_counts.get("waiting", 0)
                building_count = counts.get("building", 0)
                installing_count = counts.get("installing", 0)
                runtime_count = admitted_count + building_count + installing_count
            else:
                admitted_count = 0
                waiting_count = worker_status_counts.get("waiting", 0)
                building_count = worker_status_counts.get("building", 0)
                installing_count = worker_status_counts.get("installing", 0)
                runtime_count = counts.get("runtime", 0)
            # Every ``max_*`` value below must route through
            # ``_capacity_limit_for_bucket`` (or ``_max_total_workers`` for the
            # pool cap) so runtime overrides from ``swarm cap`` are reflected
            # in the dashboard.  Pulling directly from ``self.config`` here
            # would show the static YAML value and make the override look
            # silently ignored.
            return {
                "queue_size": len(self.queue),
                "active_count": len(self.active),
                "completed_count": len(self.completed),
                "pool_size": self._max_total_workers(),
                "max_working_workers": self._capacity_limit_for_bucket("working"),
                "max_working_return_overflow": self._capacity_limit_for_bucket("overflow"),
                "max_admitted_workers": self._capacity_limit_for_bucket("admitted"),
                "max_building_workers": self._capacity_limit_for_bucket("building"),
                "max_installing_workers": self._capacity_limit_for_bucket("installing"),
                "max_runtime_workers": self._capacity_limit_for_bucket("runtime"),
                "working_count": counts["working"],
                "admitted_count": admitted_count,
                "waiting_count": waiting_count,
                "building_count": building_count,
                "installing_count": installing_count,
                "runtime_count": runtime_count,
                "split_runtime_caps": split_runtime_caps,
                "shutdown": self._shutdown,
            }

    def get_papers(self) -> list[dict]:
        """Return all papers with their current status.

        Includes papers from previous sessions (read from disk) that
        aren't currently tracked in memory.
        """
        papers = []
        known_ids: set[str] = set()
        with self._lock:
            for job in self.queue:
                papers.append({**job.to_dict(), "phase": "queued"})
                known_ids.add(job.arxiv_id)
            for job in self.active.values():
                papers.append({**job.to_dict(), "phase": "active"})
                known_ids.add(job.arxiv_id)
            for result in self.completed.values():
                papers.append({**result.to_dict(), "phase": "completed"})
                known_ids.add(result.arxiv_id)

        # Load completed papers from disk that aren't in memory
        if PAPERS_DIR.is_dir():
            for meta_path in PAPERS_DIR.glob("*/meta.json"):
                arxiv_id = meta_path.parent.name
                if arxiv_id in known_ids:
                    continue
                try:
                    meta = json.loads(meta_path.read_text())
                    if meta.get("status") in (
                        "done", "error", "active", "cancelled", "stalled",
                        "unsuccessful", _RESTART_RECOVERY_STATUS,
                    ):
                        papers.append({**meta, "phase": "completed"})
                except Exception:
                    pass

        return papers

    def get_paper(self, arxiv_id: str) -> dict | None:
        """Return a single paper's status."""
        with self._lock:
            for job in self.queue:
                if job.arxiv_id == arxiv_id:
                    return {**job.to_dict(), "phase": "queued"}
            if arxiv_id in self.active:
                return {**self.active[arxiv_id].to_dict(), "phase": "active"}
            if arxiv_id in self.completed:
                return {**self.completed[arxiv_id].to_dict(), "phase": "completed"}
        return None

    # ── Dispatch ──────────────────────────────────────────────────────

    def _dispatch_next(self) -> bool:
        """Submit the next queued paper to the thread pool if capacity allows."""
        with self._lock:
            job = self._dispatch_next_locked()
        if job is None:
            return False
        self._launch_dispatched_job(job)
        return True

    def _run_paper_safe(self, job: PaperJob):
        """Wrapper that catches exceptions, records errors, and handles profile fallback.

        If the worker hits an API limit (all keys exhausted) and the
        workflow step has fallback_profiles configured, the paper is
        retried with the next fallback profile before being marked as
        errored.
        """
        # Build profile list: primary + fallbacks from workflow step
        fallback_profiles: list[str] = []
        if job.workflow:
            wf_steps = self.config.workflows.get(job.workflow, [])
            if job.workflow_step < len(wf_steps):
                fallback_profiles = list(wf_steps[job.workflow_step].fallback_profiles)
        profiles_to_try = [job.profile] + fallback_profiles

        for pi in range(len(profiles_to_try)):
            if pi > 0:
                job.profile = profiles_to_try[pi]
                log.info(
                    "Paper %s: falling back to profile %s (%d/%d)",
                    job.arxiv_id, profiles_to_try[pi], pi + 1, len(profiles_to_try),
                )

            try:
                needs_fallback = self._run_paper(job, is_fallback_retry=(pi > 0))
            except Exception as e:
                log.exception("Fatal error running paper %s", job.arxiv_id)
                job.status = "error"
                job.error = str(e)
                # Also parse MILESTONES.md — the agent may have updated it
                # before the fatal error.
                from infrastructure.worker.parsing import parse_current_milestone
                md_milestone = parse_current_milestone(job.workspace_path)
                if md_milestone:
                    job.milestone = _higher_milestone(job.milestone, md_milestone)
                self._finish_paper(job)
                return

            if not needs_fallback:
                return  # normal completion — _finish_paper already called

            # needs_fallback=True: limit error, _run_paper cleaned up but
            # did NOT call _finish_paper.  Try next profile if available.
            if pi + 1 >= len(profiles_to_try):
                log.warning(
                    "Paper %s: all profiles exhausted (%s), finishing with error",
                    job.arxiv_id, ", ".join(profiles_to_try),
                )
                self._finish_paper(job)
                return

    # ── Core orchestration loop ───────────────────────────────────────

    def _run_paper(self, job: PaperJob, *, is_fallback_retry: bool = False) -> bool | None:
        """Full lifecycle for one paper. Runs in a worker thread.

        Returns True if the worker hit an API limit error and the caller
        should try a fallback profile.  Returns None on normal completion
        (success or non-limit error) — _finish_paper is called internally.

        *is_fallback_retry* is True when this invocation is a fallback-profile
        retry of the same iteration; start-of-iteration commit messages switch
        from "Start" to "Continue" in that case.
        """
        start_phase = "Continue" if is_fallback_retry else "Start"
        from infrastructure.worker.prompts import build_continue_prompt, build_iterate_prompt, build_verify_prompt

        arxiv_id = job.arxiv_id
        project_id = f"swarm-{arxiv_id}"
        log.info("Starting paper %s (budget=%.1f GPU-hrs)", arxiv_id, job.gpu_budget)

        # 0a. Ensure GitHub remote (deferred from enqueue/recovery for speed)
        if self.config.github_org and job.workspace_path:
            try:
                ensure_github_remote(
                    str(job.workspace_path), job.paper, self.config.github_org,
                )
            except Exception:
                log.warning("GitHub remote setup failed for %s (non-fatal)", arxiv_id)

        # 0b. Create workspace if not yet created (deferred from enqueue)
        if not job.workspace_path:
            workspace = create_workspace(
                job.paper,
                self.config.workspace_base_dir,
                github_org=self.config.github_org,
                date=job.date,
                topics=job.topics,
            )
            job.workspace_path = workspace
            self._save_paper_meta(job)

        # 1. Create worker (resolve per-paper profile)
        wc = self._resolve_profile(job.profile)
        worker_config = {
            "max_turns": wc.max_turns,
            "container_sif": self._get_local_sif(wc.container_sif) or wc.container_sif,
            "container_memory_limit": self.config.container_memory_limit,
            "container_nice": self.config.container_nice,
            "container_cpu_threads": self.config.container_cpu_threads,
        }
        if wc.effort:
            worker_config["effort"] = wc.effort
        if wc.env_vars:
            worker_config["env_vars"] = dict(wc.env_vars)
        if wc.actual_model:
            worker_config["actual_model"] = wc.actual_model
        if wc.compact_token_limit:
            worker_config["compact_token_limit"] = wc.compact_token_limit
        if wc.api_keys:
            worker_config["api_keys"] = list(wc.api_keys)

        worker = _create_worker(
            scaffold=wc.scaffold,
            model=wc.model,
            config=worker_config,
        )
        job._worker = worker
        with self._lock:
            self._active_workers[arxiv_id] = worker

        # Pass workspace-relative paper paths so they resolve correctly
        # inside containers (where only the workspace is bind-mounted,
        # not the original Scout data directory).
        paper_dir = str(Path(job.workspace_path) / "paper")
        paper_dict = {
            "paper_md_path": str(Path(paper_dir) / Path(job.paper.paper_md_path).name),
            "paper_images_dir": str(Path(paper_dir) / "images"),
            "paper_pdf_path": str(Path(paper_dir) / Path(job.paper.paper_pdf_path).name),
        }
        worker.set_paper_and_workspace(paper_dict, job.workspace_path)
        worker.set_resources(
            gpu_budget=job.gpu_budget,
            max_gpus=self.config.nexus.max_gpus,
            job_time_limit=self.config.nexus.job_time_limit,
            shared_datasets_path=self.config.shared_datasets_path,
            shared_models_path=self.config.shared_models_path,
            uv_cache_dir=self.config.uv_cache_dir or "",
        )

        # 2. Create Nexus project
        def _ensure_nexus_project():
            """Create (or recreate) the Nexus project for this paper."""
            shared_dirs = []
            if self.config.shared_datasets_path:
                shared_dirs.append(f"{self.config.shared_datasets_path}:{CONTAINER_SHARED_DATASETS}:ro")
            if self.config.shared_models_path:
                shared_dirs.append(f"{self.config.shared_models_path}:{CONTAINER_SHARED_MODELS}:ro")

            hf_cache = self.config.hf_cache_path or ""

            env_dir = Path(job.workspace_path) / "environment"
            existing_sif = env_dir / "container.sif"
            sif = str(existing_sif) if existing_sif.exists() else ""
            existing_overlay = env_dir / "overlay.img"
            overlay = str(existing_overlay) if existing_overlay.exists() else ""

            # Pass only the remaining budget to Nexus
            remaining_budget = max(0, job.gpu_budget - job.gpu_hours_used)

            result = self.nexus.delete_project(project_id)
            if "error" in result:
                log.warning("Nexus delete returned error for %s: %s (proceeding with create)",
                            project_id, result.get("error"))
            time.sleep(1)  # let Hive finish cancelling old jobs
            try:
                self.nexus.create_project(
                    project_id=project_id,
                    workspace_dir=job.workspace_path,
                    sif_path=sif,
                    overlay_path=overlay,
                    hf_cache_path=hf_cache,
                    gpu_budget=remaining_budget,
                    max_gpus=self.config.nexus.max_gpus,
                    allowed_resources=self.config.nexus.resources.all_resources(),
                    shared_dirs=shared_dirs or None,
                )
            except RuntimeError:
                # 409 CONFLICT — stale project; force delete and retry
                log.warning("Nexus create conflict for %s, retrying delete+create", project_id)
                time.sleep(1)
                self.nexus.delete_project(project_id)
                time.sleep(1)
                self.nexus.create_project(
                    project_id=project_id,
                    workspace_dir=job.workspace_path,
                    sif_path=sif,
                    overlay_path=overlay,
                    hf_cache_path=hf_cache,
                    gpu_budget=remaining_budget,
                    max_gpus=self.config.nexus.max_gpus,
                    allowed_resources=self.config.nexus.resources.all_resources(),
                    shared_dirs=shared_dirs or None,
                )

        try:
            _ensure_nexus_project()
            self._nexus_projects.add(project_id)
        except Exception as e:
            log.error("Failed to create Nexus project for %s: %s", arxiv_id, e)
            raise

        # 3–5 wrapped in try/finally so Nexus project is always cleaned up
        nexus_cleaned = False
        try:
            # 3. Start or resume reproduction
            self._set_worker_status(job, "working")

            # Prepare git excludes before any agent interaction — covers
            # reproduce, resume, and iterate paths equally.
            worker._write_git_excludes(Path(job.workspace_path))

            # Determine role for this step (used later for validate gate)
            step_role = job.role
            if step_role == "reproducer" and job.is_iteration and job.workflow:
                wf_steps = self.config.workflows.get(job.workflow, [])
                if job.workflow_step < len(wf_steps):
                    step_role = wf_steps[job.workflow_step].role

            if job.is_iteration:
                log.info("Iteration #%d for %s (role=%s, previous=%s, starting=%s)",
                         job.iteration, arxiv_id, step_role,
                         job.previous_milestone, _drop_milestone(job.previous_milestone))
                try:
                    _ensure_iteration_start_commit(
                        Path(job.workspace_path),
                        iteration=job.iteration,
                        starting_milestone=job.milestone,
                        profile=job.profile,
                        role=step_role,
                        phase=start_phase,
                    )
                    if step_role in _NO_REPRODUCE_START_ROLES:
                        # Packager / designer: custom role-specific prompt
                        # only.  Skip reproduce_start.md entirely.
                        if step_role == "packager":
                            from infrastructure.worker.prompts import (
                                build_packager_prompt,
                            )
                            prompt = build_packager_prompt(
                                workspace_path=worker._to_container_path(
                                    job.workspace_path
                                ) if hasattr(worker, "_to_container_path")
                                else str(job.workspace_path),
                                scaffold=worker.scaffold,
                            )
                        else:  # single-stage designer, or designer_stage_1/2/3
                            from infrastructure.worker.prompts import (
                                build_designer_prompt,
                            )
                            paper_md = ""
                            if job.paper:
                                raw_md = getattr(job.paper, "paper_md_path", "") or ""
                                paper_md = (
                                    worker._to_container_path(raw_md)
                                    if hasattr(worker, "_to_container_path")
                                    else raw_md
                                )
                            designer_stage = None
                            if step_role.startswith("designer_stage_"):
                                try:
                                    designer_stage = int(step_role.rsplit("_", 1)[-1])
                                except ValueError:
                                    designer_stage = None
                            prompt = build_designer_prompt(
                                workspace_path=worker._to_container_path(
                                    job.workspace_path
                                ) if hasattr(worker, "_to_container_path")
                                else str(job.workspace_path),
                                paper_md_path=paper_md,
                                previous_milestone=job.previous_milestone or "none",
                                scaffold=worker.scaffold,
                                stage=designer_stage,
                            )
                        response = worker.start(prompt)
                    else:
                        response = worker.reproduce()
                        if step_role == "verifier":
                            prompt = build_verify_prompt(
                                previous_milestone=job.previous_milestone,
                            )
                        else:
                            prompt = build_iterate_prompt(
                                previous_milestone=job.previous_milestone,
                                starting_milestone=_drop_milestone(job.previous_milestone),
                            )
                        worker.chat(prompt)
                except Exception as e:
                    log.error("Worker iterate failed for %s: %s", arxiv_id, e)
                    raise
            elif job.is_continuation and job.session_id:
                # Resume previous session
                worker._session_id = job.session_id
                log.info("Resuming session %s for %s", job.session_id, arxiv_id)
                try:
                    worker.chat(build_continue_prompt())
                except Exception as e:
                    log.error("Worker.chat() (resume) failed for %s: %s", arxiv_id, e)
                    raise
            else:
                # Commit initial state so GitHub records the first iteration start
                _git_commit_and_push(
                    Path(job.workspace_path),
                    "none",
                    commit_msg=_format_iteration_commit_msg(
                        start_phase,
                        job.iteration,
                        "none",
                        job.profile,
                        step_role,
                    ),
                )
                try:
                    worker.reproduce()
                except Exception as e:
                    log.error("Worker.reproduce() failed for %s: %s", arxiv_id, e)
                    raise

            # Track hashes for incremental builds.
            # Restore from meta.json so daemon restarts skip unchanged
            # container builds and setup.sh re-runs — but only trust
            # persisted hashes when the corresponding artifacts exist
            # on disk (they may have been cleaned or re-cloned away).
            workspace = Path(job.workspace_path)
            last_container_hash = (
                job.last_container_hash
                if job.last_container_hash
                and (workspace / "environment" / "container.sif").exists()
                else None
            )
            last_setup_hash = (
                job.last_setup_hash
                if job.last_setup_hash
                and (workspace / "environment" / "overlay.img").exists()
                else None
            )
            # Sync gated values back to job so a restart during
            # build/setup doesn't save stale hashes to meta.json.
            if last_container_hash != job.last_container_hash or last_setup_hash != job.last_setup_hash:
                job.last_container_hash = last_container_hash
                job.last_setup_hash = last_setup_hash
                self._save_paper_meta(job)
            # Baseline GPU hours from previous runs (Nexus only tracks current session)
            gpu_hours_baseline = job.gpu_hours_used
            total_gpu_hours_baseline = job.total_gpu_hours_used
            # Circuit breakers: consecutive failures
            consecutive_env_failures = 0
            max_env_failures = 10
            consecutive_job_failures = 0
            max_job_failures = 10
            push_forward_count = 0
            submit_count = 0
            circuit_breaker_reason = ""  # non-empty = circuit breaker fired
            budget_terminal = False      # current iteration ended due to budget exhaustion

            # 4. Orchestration loop
            while not self._shutdown:
                # Check per-paper cancellation flag (under lock for thread safety)
                with self._lock:
                    cancel_requested = arxiv_id in self._cancelled
                    if cancel_requested:
                        self._cancelled.discard(arxiv_id)
                if cancel_requested:
                    log.info("Paper %s: cancellation requested, stopping", arxiv_id)
                    worker.terminate()
                    job.status = "cancelled"
                    break

                action = worker.get_last_action()
                new_ws = worker.get_status()
                if new_ws != job.worker_status:
                    self._set_worker_status(job, new_ws, save_meta=False)
                job.session_id = worker._session_id
                self._save_paper_meta(job)

                # Worker entered a terminal state (e.g. API rate limit
                # killed the claude process).  Break before processing
                # the stale action — otherwise the loop spins forever
                # on a "pending" action that will never be updated.
                if new_ws in ("error", "done"):
                    log.warning(
                        "Paper %s: worker status is %s, exiting orchestration loop",
                        arxiv_id, new_ws,
                    )
                    break

                if action is None:
                    log.warning("No action from worker for %s, treating as done", arxiv_id)
                    break

                job.milestone = _higher_milestone(job.milestone, action.milestone)

                # Workflow milestone cap: stop early and advance
                if job.workflow:
                    wf_steps = self.config.workflows.get(job.workflow, [])
                    if job.workflow_step < len(wf_steps):
                        cap = wf_steps[job.workflow_step].milestone_cap
                        cap_rank = _MILESTONE_RANK.get(cap, 7)  # "full" = 7
                        cur_rank = _MILESTONE_RANK.get(job.milestone, 0)
                        if cap_rank < 7 and cur_rank >= cap_rank:
                            log.info("Paper %s: milestone %s reached cap %s — advancing workflow",
                                     arxiv_id, job.milestone, cap)
                            from infrastructure.worker.prompts import build_wrap_up_prompt
                            try:
                                self._run_wrap_up_turn(
                                    job, worker, build_wrap_up_prompt(job.milestone)
                                )
                            except Exception:
                                log.exception("wrap-up turn failed for %s", arxiv_id)
                            break

                if action.action == "done":
                    log.info("Paper %s: agent signalled done (milestone=%s)",
                             arxiv_id, action.milestone)

                    # Push forward if below target milestone
                    target = self.config.target_milestone
                    cur_rank = _MILESTONE_RANK.get(job.milestone, 0)
                    target_rank = _MILESTONE_RANK.get(target, 2)
                    max_pushes = self.config.push_forward_attempts

                    if (cur_rank < target_rank
                            and action.milestone != "not_possible"
                            and push_forward_count < max_pushes):
                        push_forward_count += 1
                        log.info("Paper %s: below %s, pushing forward (%d/%d)",
                                 arxiv_id, target, push_forward_count, max_pushes)
                        from infrastructure.worker.prompts import build_push_forward_prompt
                        try:
                            worker.chat(build_push_forward_prompt(
                                job.milestone, target,
                                attempt=push_forward_count,
                                max_attempts=max_pushes,
                            ))
                        except Exception:
                            log.exception("push-forward turn failed for %s", arxiv_id)
                            break
                        continue  # back to action loop

                    # At or above target, or exhausted push attempts → wrap up
                    if action.milestone != "not_possible":
                        from infrastructure.worker.prompts import build_wrap_up_prompt
                        try:
                            self._run_wrap_up_turn(
                                job, worker, build_wrap_up_prompt(job.milestone)
                            )
                        except Exception:
                            log.exception("wrap-up turn failed for %s", arxiv_id)
                    break

                if action.action == "submit":
                    submit_count += 1
                    (
                        last_container_hash,
                        last_setup_hash,
                        job_rc,
                        last_run_id,
                        submit_budget_exhausted,
                        submit_budget_finalized,
                    ) = self._handle_submit(
                        worker, job, project_id, action,
                        last_container_hash, last_setup_hash,
                        gpu_hours_baseline,
                        total_gpu_hours_baseline,
                        submit_count=submit_count,
                    )
                    # Sync local loop vars back to job (already persisted to
                    # meta.json inside _handle_submit after build/setup).
                    job.last_container_hash = last_container_hash
                    job.last_setup_hash = last_setup_hash

                    if submit_budget_exhausted:
                        if submit_budget_finalized:
                            budget_terminal = True
                            log.info(
                                "Paper %s: iteration budget exhausted at %.3f / %.3f GPU-hours; "
                                "ending iteration without further submissions",
                                arxiv_id, job.gpu_hours_used, job.gpu_budget,
                            )
                        else:
                            circuit_breaker_reason = (
                                "GPU budget exhausted but worker did not finalize with action=done"
                            )
                            log.error(
                                "Paper %s: budget exhausted at %.3f / %.3f GPU-hours, but "
                                "worker failed to produce action=done",
                                arxiv_id, job.gpu_hours_used, job.gpu_budget,
                            )
                        break

                    # Check if worker is done after handling submit
                    status = worker.get_status()
                    if status in ("done", "error"):
                        break

                    # Circuit breaker: track consecutive env failures.
                    # _handle_submit returns job_rc=None when build/setup
                    # failed (never reached job submission).
                    if job_rc is None and last_run_id is None:
                        consecutive_env_failures += 1
                        if consecutive_env_failures >= max_env_failures:
                            # Log last build/setup log tail for post-mortem
                            build_tail = ""
                            for log_name in ("setup.log", "container_build.log"):
                                log_path = Path(job.workspace_path) / "logs" / log_name
                                try:
                                    lines = log_path.read_text().splitlines()
                                    build_tail = "\n".join(lines[-20:])
                                    break
                                except Exception:
                                    continue
                            if not build_tail:
                                build_tail = "(could not read build/setup log)"
                            log.error(
                                "Paper %s: %d consecutive build/setup failures, "
                                "forcing error\n"
                                "--- last build log (tail 20) ---\n%s\n"
                                "--- end build log ---",
                                arxiv_id, consecutive_env_failures, build_tail,
                            )
                            worker.chat(
                                f"Your container build or setup.sh has failed "
                                f"{consecutive_env_failures} times in a row. "
                                f"The system is stopping this paper. Please "
                                f"set action: done in action.yaml."
                            )
                            circuit_breaker_reason = (
                                f"{consecutive_env_failures} consecutive build/setup failures"
                            )
                            break
                    else:
                        # Job was actually submitted and ran — reset env counter
                        if last_run_id is not None:
                            consecutive_env_failures = 0

                        # Track consecutive job execution failures
                        if job_rc is not None and job_rc == 0:
                            consecutive_job_failures = 0
                        elif job_rc is not None and job_rc != 0:
                            consecutive_job_failures += 1
                            if consecutive_job_failures >= max_job_failures:
                                # Log last stderr tail for post-mortem
                                stderr_tail = ""
                                if last_run_id:
                                    stderr_path = Path(job.workspace_path) / "jobs" / last_run_id / "stderr.log"
                                    try:
                                        lines = stderr_path.read_text().splitlines()
                                        stderr_tail = "\n".join(lines[-20:])
                                    except Exception:
                                        stderr_tail = "(could not read stderr)"
                                log.error(
                                    "Paper %s: %d consecutive job failures "
                                    "(last rc=%s), forcing error\n"
                                    "--- last stderr (tail 20) ---\n%s\n"
                                    "--- end stderr ---",
                                    arxiv_id, consecutive_job_failures, job_rc,
                                    stderr_tail,
                                )
                                worker.chat(
                                    f"Your jobs have failed {consecutive_job_failures} "
                                    f"times in a row (last exit code: {job_rc}). "
                                    f"The system is stopping this paper. Please "
                                    f"set action: done in action.yaml."
                                )
                                # Extract last error line for the dashboard
                                last_err_line = ""
                                if stderr_tail:
                                    for line in reversed(stderr_tail.splitlines()):
                                        line = line.strip()
                                        if line and not line.startswith("---"):
                                            last_err_line = line
                                            break
                                circuit_breaker_reason = (
                                    f"{consecutive_job_failures} consecutive job failures"
                                    f" (rc={job_rc})"
                                )
                                if last_err_line:
                                    circuit_breaker_reason += f": {last_err_line}"
                                break
                        else:
                            consecutive_job_failures = 0

            # 5. Finalize status
            # (setup.sh merge into container.def is deferred to _finish_paper
            # when the workflow is fully done — merging mid-workflow causes
            # duplication: the next iteration rebuilds the container from the
            # merged def and also re-runs setup.sh in the overlay.)
            if job.status == "cancelled":
                pass  # preserve cancel status set above
            elif budget_terminal and worker.get_status() != "error":
                job.status = "done"
                job.error = None
            elif circuit_breaker_reason:
                job.status = "error"
                job.error = circuit_breaker_reason
            else:
                job.status = "done" if worker.get_status() != "error" else "error"
                if job.status == "error" and not job.error:
                    job.error = worker.get_error() or "worker exited with error status"
            # Use the worker's latest action (not the stale loop variable)
            # because _handle_submit → worker.chat() may have updated it
            # after the loop variable was captured.
            #
            # Trust the agent's milestone directly — any agent can lower
            # it if it discovers previous work was wrong.
            # action.yaml is the agent's authoritative current declaration.
            # If present, it wins — both for promotions AND intentional
            # demotions (e.g. a verifier iteration auditing prior work
            # and lowering the milestone).  MILESTONES.md is used only
            # as a fallback when action.yaml carries no milestone, since
            # a stale header from an earlier iteration would otherwise
            # silently override the agent's latest call.
            final_action = worker.get_last_action()

            # Milestone-locked roles (packager, designer stages 1/2): the
            # paper's milestone cannot be changed by this iteration.
            # designer_stage_3 is EXCLUDED — it can lower the milestone
            # (e.g. core_claim → method_runs) when it discovers the
            # reproduction scores are too broken to be useful.
            if step_role in _MILESTONE_LOCKED_ROLES \
                    and step_role != "designer_stage_3":
                if final_action and final_action.milestone \
                        and final_action.milestone != job.previous_milestone:
                    log.info(
                        "Paper %s: %s iteration declared milestone=%s but "
                        "role is milestone-locked; keeping %s",
                        arxiv_id, step_role, final_action.milestone,
                        job.previous_milestone,
                    )
                job.milestone = job.previous_milestone or job.milestone
                # Neutralize so downstream wrap-up / error-recovery paths
                # can't accidentally mutate milestone either.
                if final_action:
                    final_action.milestone = job.milestone

            # Invalid-milestone gate: if the agent declared a milestone not
            # in the canonical set (e.g. "extended_results"), wake them and
            # have them pick a valid name. Runs BEFORE wrap-up so the chosen
            # milestone is embedded correctly in MILESTONES.md / manifest.
            elif final_action and final_action.milestone \
                    and final_action.milestone not in _MILESTONE_VALID_SET:
                from infrastructure.worker.prompts import build_invalid_milestone_prompt
                max_milestone_attempts = 3
                for attempt in range(1, max_milestone_attempts + 1):
                    bad = final_action.milestone
                    log.warning(
                        "Paper %s: agent declared invalid milestone %r "
                        "(attempt %d/%d)",
                        arxiv_id, bad, attempt, max_milestone_attempts,
                    )
                    try:
                        worker.chat(build_invalid_milestone_prompt(
                            bad_milestone=bad,
                            attempt=attempt,
                            max_attempts=max_milestone_attempts,
                        ))
                    except Exception:
                        log.exception("invalid-milestone retry chat failed for %s", arxiv_id)
                        break
                    final_action = worker.get_last_action()
                    if not final_action or not final_action.milestone:
                        break
                    if final_action.milestone in _MILESTONE_VALID_SET:
                        log.info(
                            "Paper %s: agent picked valid milestone %s on retry %d",
                            arxiv_id, final_action.milestone, attempt,
                        )
                        break
                else:
                    # Exhausted retries — keep previous_milestone rather than
                    # silently coerce to 'none'.
                    log.error(
                        "Paper %s: agent kept declaring invalid milestone "
                        "after %d attempts; keeping previous milestone %s",
                        arxiv_id, max_milestone_attempts, job.milestone,
                    )
                    # Neutralize the bad final_action milestone so the
                    # fallback logic below doesn't assign it to job.milestone.
                    if final_action and final_action.milestone \
                            and final_action.milestone not in _MILESTONE_VALID_SET:
                        final_action.milestone = job.milestone

            if final_action and final_action.milestone:
                if final_action.milestone != job.milestone:
                    log.info("Paper %s: milestone %s → %s (from action.yaml)",
                             arxiv_id, job.milestone, final_action.milestone)
                job.milestone = final_action.milestone
            elif action and action.milestone:
                job.milestone = action.milestone
            else:
                from infrastructure.worker.parsing import parse_current_milestone
                md_milestone = parse_current_milestone(job.workspace_path)
                if md_milestone:
                    log.info("Paper %s: milestone %s → %s (from MILESTONES.md fallback)",
                             arxiv_id, job.milestone, md_milestone)
                    job.milestone = md_milestone
            # If the agent wrote action=done, trust that over a non-zero
            # exit code (e.g. context exhaustion after writing the action).
            if job.status == "error" and final_action and final_action.action == "done":
                log.info(
                    "Paper %s: overriding error status → done "
                    "(agent wrote action=done at milestone=%s)",
                    arxiv_id, final_action.milestone,
                )
                job.status = "done"
                job.error = None

            # 5aa. Session-end push-forward: the per-turn action loop's
            # push-forward (above, around action.action == "done") only
            # fires when the loop sees a "done" action while the worker
            # is still alive.  If the worker transitioned to status="done"
            # on the same tick the agent wrote action=done, the loop
            # exited at the worker-status check BEFORE reaching the
            # push-forward branch.  Re-check here so push-forward fires
            # regardless of which loop-exit path was taken.
            while (job.status == "done"
                    and final_action
                    and final_action.milestone != "not_possible"
                    and push_forward_count < self.config.push_forward_attempts):
                target = self.config.target_milestone
                cur_rank = _MILESTONE_RANK.get(job.milestone, 0)
                target_rank = _MILESTONE_RANK.get(target, 2)
                if cur_rank >= target_rank:
                    break
                push_forward_count += 1
                log.info(
                    "Paper %s: session ended below %s (milestone=%s), "
                    "pushing forward (%d/%d)",
                    arxiv_id, target, job.milestone,
                    push_forward_count, self.config.push_forward_attempts,
                )
                from infrastructure.worker.prompts import build_push_forward_prompt
                try:
                    worker.chat(build_push_forward_prompt(
                        job.milestone, target,
                        attempt=push_forward_count,
                        max_attempts=self.config.push_forward_attempts,
                    ))
                except Exception:
                    log.exception("session-end push-forward failed for %s", arxiv_id)
                    break
                # Re-read the agent's response.
                new_final_action = worker.get_last_action()
                if new_final_action and new_final_action.milestone \
                        and new_final_action.milestone in _MILESTONE_VALID_SET:
                    if new_final_action.milestone != job.milestone:
                        log.info(
                            "Paper %s: milestone %s → %s "
                            "(after session-end push-forward)",
                            arxiv_id, job.milestone, new_final_action.milestone,
                        )
                    job.milestone = new_final_action.milestone
                final_action = new_final_action or final_action
                # If the worker gave up (status=error), stop pushing.
                if worker.get_status() == "error":
                    break

            # 5b. Profile fallback: if the error is an API limit
            # exhaustion, return early so _run_paper_safe can retry
            # with a fallback profile (if configured).
            if (job.status == "error"
                    and job.error
                    and _is_model_limit_error_text(job.error)):
                log.warning(
                    "Paper %s: profile %s hit API limit (all keys exhausted)",
                    arxiv_id, job.profile,
                )
                # Save trajectory for the failed attempt
                fb_logs = str(PAPERS_DIR / arxiv_id)
                Path(fb_logs).mkdir(parents=True, exist_ok=True)
                try:
                    worker.save_trajectory(fb_logs)
                except Exception:
                    pass
                worker.terminate()
                job._worker = None
                # Clear session so fallback starts fresh (don't resume a
                # session created by a different model/provider).
                job.session_id = None
                with self._lock:
                    self._active_workers.pop(arxiv_id, None)
                # Clean up Nexus (will be re-created on retry)
                nexus_cleaned = True
                self._nexus_projects.discard(project_id)
                try:
                    self.nexus.delete_project(project_id)
                except Exception:
                    pass
                return True  # signal _run_paper_safe to try fallback

            job.finished_at = time.time()

            # 6b. Wrap-up for errored papers that reached target milestone.
            # Papers that error (circuit breaker, timeout, etc.) but already
            # made meaningful progress still deserve a wrap-up turn so their
            # TARGETS.md, CONSTRAINTS.md, and final commit are preserved.
            target = self.config.target_milestone
            if (job.status == "error"
                    and _MILESTONE_RANK.get(job.milestone, 0)
                    >= _MILESTONE_RANK.get(target, 2)):
                log.info(
                    "Paper %s errored at milestone=%s (>= target %s), "
                    "running wrap-up",
                    arxiv_id, job.milestone, target,
                )
                from infrastructure.worker.prompts import build_wrap_up_prompt
                try:
                    self._run_wrap_up_turn(
                        job, worker, build_wrap_up_prompt(job.milestone)
                    )
                except Exception:
                    log.exception("wrap-up turn failed for errored %s", arxiv_id)
                else:
                    post_wrap_action = worker.get_last_action()
                    if post_wrap_action:
                        job.milestone = _higher_milestone(
                            job.milestone, post_wrap_action.milestone
                        )
                    from infrastructure.worker.parsing import parse_current_milestone
                    md_milestone = parse_current_milestone(job.workspace_path)
                    if md_milestone:
                        job.milestone = _higher_milestone(job.milestone, md_milestone)
                    if post_wrap_action and post_wrap_action.action == "done":
                        log.info(
                            "Paper %s: wrap-up completed after error; overriding status → done "
                            "(milestone=%s)",
                            arxiv_id, post_wrap_action.milestone,
                        )
                        job.status = "done"
                        job.error = None

            # 7. Save trajectory
            logs_dir = str(PAPERS_DIR / arxiv_id)
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            try:
                worker.save_trajectory(logs_dir)
            except Exception as e:
                log.warning("Failed to save trajectory for %s: %s", arxiv_id, e)

            # 8. Write manifest.json (swarm-owned, after status is final)
            traj_stats = None
            ctx = worker.get_context()
            if ctx and "stats" in ctx:
                traj_stats = ctx["stats"]
            _fill_manifest(
                Path(job.workspace_path) / "manifest.json",
                job=job,
                worker_config=wc,
                trajectory_stats=traj_stats,
            )

            # Validation gate: after verifier signals done, run validate.py.
            # If it fails, wake the verifier to fix errors (up to 3 attempts).
            # If it still fails after all attempts, the push is blocked below.
            validate_blocks_push = False
            validate_last_output = ""
            if step_role == "verifier":
                validate_py = Path(job.workspace_path) / "validate.py"
                if validate_py.exists():
                    # Verifier runs after designer_stage_3 and packager —
                    # all six split artifacts should exist by this point, so
                    # run in post-design mode.
                    validate_cmd = [sys.executable, str(validate_py), "--post-design"]
                    max_validate_attempts = 3
                    for attempt in range(1, max_validate_attempts + 1):
                        try:
                            result = subprocess.run(
                                validate_cmd,
                                cwd=job.workspace_path,
                                capture_output=True, text=True, timeout=300,
                            )
                        except subprocess.TimeoutExpired:
                            # A slow shared filesystem can push validate.py
                            # past the timeout while it waits on metadata
                            # I/O. A timeout is therefore treated as a pass
                            # instead of failing the whole paper.
                            log.warning(
                                "Paper %s: validate.py hit timeout %ds "
                                "(likely filesystem lag, not a validator failure). "
                                "Treating as passed.",
                                arxiv_id, 300,
                            )
                            break
                        if result.returncode == 0:
                            log.info("Paper %s: validate.py passed (attempt %d)",
                                     arxiv_id, attempt)
                            break
                        log.warning("Paper %s: validate.py failed (attempt %d/%d):\n%s",
                                    arxiv_id, attempt, max_validate_attempts,
                                    result.stdout + result.stderr)
                        if attempt >= max_validate_attempts:
                            log.error("Paper %s: validate.py still failing after %d attempts",
                                      arxiv_id, max_validate_attempts)
                            validate_blocks_push = True
                            validate_last_output = (result.stdout + result.stderr).strip()
                            break
                        from infrastructure.worker.prompts import build_validation_failure_prompt
                        try:
                            worker.chat(build_validation_failure_prompt(
                                validation_output=result.stdout.strip(),
                                attempt=attempt,
                                max_attempts=max_validate_attempts,
                            ))
                        except Exception:
                            log.exception("validate retry chat failed for %s", arxiv_id)
                            validate_blocks_push = True
                            validate_last_output = (result.stdout + result.stderr).strip()
                            break
                        # Re-read action after the fix turn
                        action = worker.get_last_action()
                        if action and action.milestone:
                            job.milestone = _higher_milestone(job.milestone, action.milestone)

            # Run validate.py with retry for packager / designer stages AND
            # for the reproducer. Same loop as the verifier: if validation
            # fails, wake the agent to fix the issue (up to 3 attempts).
            # If it still fails after all attempts, the push is blocked and
            # the iteration ends as "stalled" (see below).
            elif step_role in _MILESTONE_LOCKED_ROLES or step_role == "reproducer":
                validate_py = Path(job.workspace_path) / "validate.py"
                if validate_py.exists():
                    cmd = [sys.executable, str(validate_py)]
                    if step_role in ("designer_stage_1", "designer_stage_2"):
                        cmd.append("--during-design")
                    elif step_role in ("designer_stage_3", "packager"):
                        cmd.append("--post-design")
                    elif step_role == "reproducer":
                        # Gate reproducer-to-designer advancement on the
                        # paper ↔ scores consistency check. A wildly-off
                        # reproduction makes every future agent look
                        # superhuman against a broken baseline — catch it
                        # here before the designer builds on it.
                        cmd.append("--post-reproducer")

                    max_validate_attempts = 3
                    for attempt in range(1, max_validate_attempts + 1):
                        try:
                            result = subprocess.run(
                                cmd,
                                cwd=job.workspace_path,
                                capture_output=True, text=True, timeout=300,
                            )
                        except subprocess.TimeoutExpired:
                            log.warning(
                                "Paper %s: validate.py timeout after %s "
                                "(filesystem lag; treating as pass).",
                                arxiv_id, step_role,
                            )
                            break
                        if result.returncode == 0:
                            log.info("Paper %s: validate.py passed after %s (attempt %d)",
                                     arxiv_id, step_role, attempt)
                            break
                        log.warning(
                            "Paper %s: validate.py failed after %s (attempt %d/%d):\n%s",
                            arxiv_id, step_role, attempt, max_validate_attempts,
                            result.stdout + result.stderr,
                        )
                        if attempt >= max_validate_attempts:
                            log.error(
                                "Paper %s: validate.py still failing after %d attempts",
                                arxiv_id, max_validate_attempts,
                            )
                            validate_blocks_push = True
                            validate_last_output = (result.stdout + result.stderr).strip()
                            break
                        from infrastructure.worker.prompts import build_validation_failure_prompt
                        try:
                            worker.chat(build_validation_failure_prompt(
                                validation_output=result.stdout.strip(),
                                attempt=attempt,
                                max_attempts=max_validate_attempts,
                            ))
                        except Exception:
                            log.exception("validate retry chat failed for %s", arxiv_id)
                            validate_blocks_push = True
                            validate_last_output = (result.stdout + result.stderr).strip()
                            break

            # Final git commit + push so GitHub has the complete state.
            # Push is gated by validate.py — a failing validator means the
            # repo wouldn't be reproducible for anyone cloning it, so we
            # refuse to publish it and mark the paper stalled instead.
            end_msg = _format_iteration_commit_msg(
                "End",
                job.iteration,
                job.milestone,
                job.profile,
                job.role,
            )
            if validate_blocks_push:
                log.warning(
                    "Paper %s: validate.py failed — skipping git push, "
                    "marking stalled. Validator output:\n%s",
                    arxiv_id, validate_last_output,
                )
                job.status = "stalled"
                job.error = (
                    "validate.py failed — push blocked so defective repo "
                    "does not reach GitHub"
                )
            else:
                push_ok = _git_commit_and_push(
                    Path(job.workspace_path), job.milestone,
                    paper=job.paper, github_org=self.config.github_org,
                    commit_msg=end_msg,
                )
                if not push_ok:
                    log.warning("Paper %s: git push failed — marking as stalled (not done)",
                                arxiv_id)
                    job.status = "stalled"
                    job.error = "git push failed — work is committed locally but not on GitHub"

            worker.terminate()
            job._worker = None

            # Clean up Nexus project BEFORE _finish_paper, which may
            # auto-advance and create a new project with the same ID.
            # Wait briefly for Hive to confirm job cancellations so the
            # next iteration doesn't race with dying jobs.
            nexus_cleaned = True
            self._nexus_projects.discard(project_id)
            try:
                result = self.nexus.delete_project(project_id)
                if "error" in result:
                    log.warning("Nexus delete_project returned error for %s: %s",
                                project_id, result["error"])
            except Exception as e:
                log.warning("Failed to delete Nexus project %s: %s", project_id, e)
            time.sleep(2)  # let Hive finish cancelling actors

            self._finish_paper(job, trajectory_path=logs_dir)

        finally:
            # Safety net for exception path only — if explicit cleanup
            # above already ran, do nothing (auto-advance may have
            # created a new project with the same ID).
            if not nexus_cleaned:
                self._nexus_projects.discard(project_id)
                try:
                    self.nexus.delete_project(project_id)
                except Exception:
                    pass

    def _handle_submit(
        self,
        worker,
        job: PaperJob,
        project_id: str,
        action,
        last_container_hash: str | None,
        last_setup_hash: str | None,
        gpu_hours_baseline: float = 0.0,
        total_gpu_hours_baseline: float = 0.0,
        submit_count: int = 0,
    ) -> tuple[str | None, str | None, int | None, str | None, bool, bool]:
        """Handle a submit action: build container if needed, submit job, wait.

        Returns:
            Tuple of (container_hash, setup_hash, job_returncode, run_id,
            budget_exhausted, budget_finalized).
            job_returncode and run_id are None if the job was never submitted
            (build/setup failure).
        """
        workspace = Path(job.workspace_path)
        container_def = workspace / "environment" / "container.def"
        container_sif = workspace / "environment" / "container.sif"
        setup_sh = workspace / "environment" / "setup.sh"
        overlay_img = workspace / "environment" / "overlay.img"
        container_base_hash = last_container_hash

        # 1. Build container if container.def changed
        if container_def.exists():
            current_hash = _hash_file(container_def)
            container_base_hash = current_hash
            if current_hash != last_container_hash:
                log.info("Building container for %s", job.arxiv_id)
                self._set_worker_status(job, "building")
                build_log_path = str(workspace / "logs" / "container_build.log")
                success = self._build_container(
                    container_def, container_sif, build_log_path,
                    arxiv_id=job.arxiv_id,
                )
                if not success:
                    self._set_worker_status(job, "working")
                    failure_ctx = self._record_env_failure_context(
                        job,
                        workspace,
                        phase="build",
                        log_path=Path(build_log_path),
                    )
                    try:
                        worker.on_container_build_failure(
                            build_log_path,
                            phase="build",
                            failure_class=failure_ctx["failure_class"],
                            failure_signature=failure_ctx["failure_signature"],
                            failure_hint=failure_ctx["failure_hint"],
                            env_comparison=failure_ctx["env_comparison"],
                            same_signature=failure_ctx["same_signature"],
                        )
                    except Exception:
                        log.exception("on_container_build_failure failed for %s", job.arxiv_id)
                    return last_container_hash, last_setup_hash, None, None, False, False

                last_container_hash = current_hash
                last_setup_hash = None
                # Persist both hashes atomically: the new container hash
                # AND the cleared setup hash.  Otherwise a restart
                # between this save and the overlay deletion would
                # recover with a stale setup hash that matches the old
                # (now wrong) overlay.
                job.last_container_hash = last_container_hash
                job.last_setup_hash = last_setup_hash
                self._save_paper_meta(job)
                # container.def changed → overlay was built for the old
                # SIF and must be recreated, even if the new SIF came
                # from cache.  (The restart case — same def, hash was
                # just not in memory — is handled by persisted hashes
                # in meta.json: last_container_hash == current_hash so
                # we never enter this block.)
                if overlay_img.exists():
                    overlay_img.unlink()
                    log.info("Removed stale overlay after container change for %s", job.arxiv_id)
                    # If setup.sh won't recreate the overlay, clear the
                    # Nexus path now so jobs don't reference a deleted
                    # file.  When setup.sh exists, the overlay will be
                    # rebuilt and re-registered below — clearing eagerly
                    # here would leave Nexus empty if that re-registration
                    # fails transiently.
                    if not setup_sh.exists():
                        try:
                            self.nexus.update_project(
                                project_id, overlay_path="",
                            )
                        except Exception:
                            pass
                # last_setup_hash already cleared above (before save).

                # Update Nexus project with the built SIF path
                try:
                    self.nexus.update_project(
                        project_id, sif_path=str(container_sif),
                    )
                except Exception as e:
                    log.warning("Could not update SIF path: %s", e)
                self._clear_env_failure_context(job)

        # 2. Snapshot setup.sh before running (append-only history)
        _snapshot_setup_sh(workspace, job.iteration)

        # 2b. Run setup.sh in overlay if it changed (fast path)
        if setup_sh.exists() and container_sif.exists():
            current_setup_hash = _hash_file(setup_sh)
            if current_setup_hash != last_setup_hash:
                log.info("Running setup.sh in overlay for %s", job.arxiv_id)
                self._set_worker_status(job, "installing")
                setup_log_path = str(workspace / "logs" / "setup.log")
                setup_base_hash = container_base_hash or _hash_file(container_sif)
                success = self._prepare_setup_overlay(
                    container_sif,
                    overlay_img,
                    setup_sh,
                    setup_log_path,
                    arxiv_id=job.arxiv_id,
                    setup_cache_key=self._build_setup_cache_key(
                        setup_base_hash,
                        current_setup_hash,
                    ),
                )
                if not success:
                    failure_ctx = self._record_env_failure_context(
                        job,
                        workspace,
                        phase="setup",
                        log_path=Path(setup_log_path),
                    )
                    # Delete the corrupt overlay so the next attempt starts
                    # fresh. Partial installs can fill the overlay, making
                    # all subsequent attempts fail with "No space left".
                    if overlay_img.exists():
                        overlay_img.unlink()
                        log.info("Removed overlay after setup failure for %s",
                                 job.arxiv_id)
                    self._set_worker_status(job, "working")
                    try:
                        worker.on_container_build_failure(
                            setup_log_path,
                            phase="setup",
                            failure_class=failure_ctx["failure_class"],
                            failure_signature=failure_ctx["failure_signature"],
                            failure_hint=failure_ctx["failure_hint"],
                            env_comparison=failure_ctx["env_comparison"],
                            same_signature=failure_ctx["same_signature"],
                        )
                    except Exception:
                        log.exception("on_setup_failure failed for %s", job.arxiv_id)
                    return last_container_hash, last_setup_hash, None, None, False, False

                # Register overlay with Nexus before committing the
                # hash — if registration fails, the job must not
                # proceed without setup-installed dependencies.
                try:
                    self.nexus.update_project(
                        project_id, overlay_path=str(overlay_img),
                    )
                except Exception:
                    log.error(
                        "Paper %s: setup succeeded but Nexus overlay "
                        "registration failed", job.arxiv_id,
                    )
                    return last_container_hash, last_setup_hash, None, None, False, False

                last_setup_hash = current_setup_hash
                # Persist immediately so a restart while waiting for
                # a submitted job doesn't lose the successful setup.
                job.last_setup_hash = last_setup_hash
                self._save_paper_meta(job)
                self._clear_env_failure_context(job)

        # 3. Wait for overlay lock to be released (if applicable)
        # Apptainer ext3 overlays use exclusive write locks. On a network
        # filesystem, locks held by processes that died on another node
        # can persist after the process exits, and fuser does not see
        # them. Test with an actual open-for-write attempt.
        if overlay_img.exists():
            real_overlay = str(overlay_img.resolve())
            for attempt in range(12):
                try:
                    fd = os.open(real_overlay, os.O_RDWR | os.O_NONBLOCK)
                    os.close(fd)
                    break  # writable — lock is clear
                except OSError:
                    log.warning(
                        "Paper %s: overlay is locked, waiting (attempt %d/12)",
                        job.arxiv_id, attempt + 1,
                    )
                    time.sleep(10)
            else:
                # Still locked after 2 minutes — recreate the overlay
                log.warning("Paper %s: overlay stuck, recreating", job.arxiv_id)
                try:
                    overlay_img.unlink()
                    # Clear overlay path in Nexus so the job doesn't reference a deleted file
                    self.nexus.update_project(project_id, overlay_path="")
                except Exception:
                    pass

        # If the overlay disappeared but setup.sh still exists, rebuild
        # it rather than submitting without setup-installed dependencies.
        # Also clear stale Nexus overlay path if no overlay will exist.
        if not overlay_img.exists():
            if setup_sh.exists() and container_sif.exists():
                log.warning(
                    "Paper %s: overlay.img missing, rebuilding from setup.sh",
                    job.arxiv_id,
                )
                last_setup_hash = None
                job.last_setup_hash = None
                current_setup_hash = _hash_file(setup_sh)
                self._set_worker_status(job, "installing")
                setup_log_path = str(workspace / "logs" / "setup.log")
                setup_base_hash = container_base_hash or _hash_file(container_sif)
                success = self._prepare_setup_overlay(
                    container_sif,
                    overlay_img,
                    setup_sh,
                    setup_log_path,
                    arxiv_id=job.arxiv_id,
                    setup_cache_key=self._build_setup_cache_key(
                        setup_base_hash,
                        current_setup_hash,
                    ),
                )
                if success:
                    # Only commit the hash after Nexus accepts the
                    # overlay path — otherwise the job runs without
                    # setup-installed dependencies.
                    try:
                        self.nexus.update_project(
                            project_id, overlay_path=str(overlay_img),
                        )
                    except Exception:
                        log.error(
                            "Paper %s: rebuilt overlay but failed to "
                            "register it with Nexus", job.arxiv_id,
                        )
                        # Remove the unregistered overlay so no retry
                        # path can use it without Nexus knowing.
                        if overlay_img.exists():
                            overlay_img.unlink()
                        return last_container_hash, last_setup_hash, None, None, False, False
                    last_setup_hash = current_setup_hash
                    job.last_setup_hash = last_setup_hash
                    self._save_paper_meta(job)
                else:
                    # Mirror the normal setup failure path: record
                    # context, notify the worker, clean up overlay.
                    failure_ctx = self._record_env_failure_context(
                        job,
                        workspace,
                        phase="setup",
                        log_path=Path(setup_log_path),
                    )
                    if overlay_img.exists():
                        overlay_img.unlink()
                    self._set_worker_status(job, "working")
                    try:
                        worker.on_container_build_failure(
                            setup_log_path,
                            phase="setup",
                            failure_class=failure_ctx["failure_class"],
                            failure_signature=failure_ctx["failure_signature"],
                            failure_hint=failure_ctx["failure_hint"],
                            env_comparison=failure_ctx["env_comparison"],
                            same_signature=failure_ctx["same_signature"],
                        )
                    except Exception:
                        log.exception("on_setup_failure (rebuild) failed for %s", job.arxiv_id)
                    return last_container_hash, last_setup_hash, None, None, False, False
            else:
                try:
                    self.nexus.update_project(project_id, overlay_path="")
                except Exception:
                    pass

        # 4. Build the run command
        # Use container-internal path — workspace is bind-mounted to /home/user
        run_sh = workspace / "scripts" / "run.sh"
        if run_sh.exists():
            command = "bash /home/user/scripts/run.sh"
        else:
            command = "echo 'No scripts/run.sh found'"
            log.warning("No scripts/run.sh in workspace for %s", job.arxiv_id)

        # 5. Submit job to Nexus (retry on transient errors)
        run_id = None
        last_submit_err = None
        max_attempts = 12  # ~3 minutes total with exponential backoff
        run_id = None
        hive_job_id = ""
        for attempt in range(1, max_attempts + 1):
            try:
                run_id, hive_job_id = self.nexus.submit_job(
                    project_id=project_id,
                    command=command,
                    num_gpus=action.gpus,
                    resource=self.config.nexus.resources.pick(action.gpus, action.test),
                )
                break
            except Exception as e:
                last_submit_err = e
                log.warning("Job submission failed for %s (attempt %d/%d): %s",
                            job.arxiv_id, attempt, max_attempts, e)
                if attempt < max_attempts:
                    time.sleep(min(10 * attempt, 30))
        if run_id is None:
            log.error("Job submission failed for %s after %d attempts: %s",
                      job.arxiv_id, max_attempts, last_submit_err)
            raise last_submit_err

        # 6. Wait for job events
        if action.gpus > 0:
            job.last_job_resources = f"{action.gpus} GPU" + (" (test)" if action.test else "")
        else:
            job.last_job_resources = "CPU" + (" (test)" if action.test else "")
        # Expose the hive job id (not the nexus run_id) on the dashboard —
        # that's the id users look up in hive logs and .hive/jobs/<id>/.
        # Nexus also retains run_id internally for its own bookkeeping.
        job.last_job_id = str(hive_job_id) if hive_job_id else ""
        self._set_worker_status(job, "waiting")

        # Git checkpoint: record state before the job runs
        _git_checkpoint(
            workspace,
            f"Job #{submit_count} submission ({job.last_job_resources})",
        )

        job_returncode = None
        budget_exhausted = False
        budget_finalized = False
        while not self._shutdown:
            event = self.nexus.wait_for_job(
                project_id=project_id,
                timeout=600,
                run_id=run_id,
            )

            event_type = event.get("event", "")

            if event_type == "error":
                log.warning("Wait error for %s: %s", job.arxiv_id, event.get("error"))
                break

            # Server-side timeout (returns {"timeout": True}, no "event" key)
            if event.get("timeout"):
                continue

            # Construct log paths from workspace — Nexus writes logs to
            # {workspace}/jobs/{run_id}/stdout.log and stderr.log
            event_run_id = event.get("run_id", run_id)
            job_dir = Path(job.workspace_path) / "jobs" / event_run_id
            stdout_path = str(job_dir / "stdout.log")
            stderr_path = str(job_dir / "stderr.log")

            if event_type == "awaiting_response":
                # Periodic check-in: the agent reads stdout/stderr and
                # answers continue/cancel — a tiny turn (seconds).  This
                # is NOT a real work state, so it bypasses the
                # working-bucket capacity: the hive job's GPU is already
                # reserved on the worker's behalf, and going through
                # _set_worker_status
                # would pointlessly block this check-in behind a full
                # working cap when there's nothing the worker can do
                # meanwhile.  Update the label directly so the
                # dashboard briefly shows "working" during the
                # decision, then restore "waiting" if the job
                # continues.
                job.set_worker_status("working")
                self._save_paper_meta(job)
                # Compute minutes since the hive job entered "running".
                # Nexus's started_at is set at the queued→running
                # transition, so queued time is not counted.
                elapsed_min = 0
                try:
                    nxinfo = self.nexus.get_job(event_run_id) or {}
                    started_at = nxinfo.get("started_at") or 0
                    if started_at:
                        elapsed_min = max(0, int((time.time() - started_at) / 60))
                except Exception:
                    log.debug("Could not read started_at for %s", event_run_id)
                try:
                    decision = worker.on_job_progress(
                        event_run_id, stdout_path, stderr_path,
                        elapsed_min=elapsed_min,
                    )
                except Exception:
                    log.exception("on_job_progress failed for %s", job.arxiv_id)
                    decision = "continue"

                # Persist the progress-turn workspace/session state so later
                # completion checkpoints only capture subsequent run outputs.
                _git_checkpoint(
                    workspace,
                    f"Job #{submit_count} progress ({event_run_id}, decision={decision})",
                    log_files=[stdout_path, stderr_path],
                )

                if decision == "cancel":
                    self.nexus.cancel_job(event_run_id)
                    try:
                        worker.on_job_cancelled(event_run_id, stdout_path, stderr_path)
                    except Exception:
                        log.exception("on_job_cancelled failed for %s", job.arxiv_id)
                    _git_checkpoint(
                        workspace,
                        f"Job #{submit_count} cancelled-follow-up ({event_run_id})",
                        log_files=[stdout_path, stderr_path],
                        stage_workspace=True,  # capture outputs before agent moves on
                    )
                    break

                # Check-in done, job is going to keep running — flip
                # the badge back to "waiting" to reflect reality.
                job.set_worker_status("waiting")
                self._save_paper_meta(job)

                # Resume Nexus monitoring
                self.nexus.continue_job(event_run_id)

            elif event_type in ("completed", "failed", "cancelled"):
                # Get returncode from job details
                job_info = self.nexus.get_job(event_run_id)
                job_returncode = job_info.get("returncode", -1)

                # Git checkpoint: capture the post-run workspace state
                # before the agent reacts to the logs in on_job_completion.
                log.info("Git checkpoint: Job %s %s (rc=%s) for %s",
                         event_run_id, event_type, job_returncode, job.arxiv_id)
                _git_checkpoint(
                    workspace,
                    f"Job #{submit_count} {event_type} ({event_run_id}, rc={job_returncode})",
                    log_files=[stdout_path, stderr_path],
                    stage_workspace=True,  # preserve raw outputs pre-agent-reaction
                )

                self._set_worker_status(job, "working")
                try:
                    worker.on_job_completion(event_run_id, stdout_path, stderr_path, job_returncode)
                except Exception:
                    log.exception("on_job_completion failed for %s", job.arxiv_id)

                # Persist the agent's follow-up turn so clone-based resume can
                # recover the updated session and workspace state.
                _git_checkpoint(
                    workspace,
                    f"Job #{submit_count} follow-up ({event_run_id}, rc={job_returncode})",
                    log_files=[stdout_path, stderr_path],
                )

                # Track GPU usage: Nexus tracks this session, swarm accumulates
                project_info = self.nexus.get_project(project_id)
                session_gpu_hours = project_info.get("gpu_seconds_used", 0) / 3600
                job.gpu_hours_used = gpu_hours_baseline + session_gpu_hours
                job.total_gpu_hours_used = total_gpu_hours_baseline + session_gpu_hours
                self._save_paper_meta(job)

                # Check budget against this iteration's usage
                budget_exhausted = job.gpu_hours_used >= job.gpu_budget
                if budget_exhausted:
                    try:
                        worker.on_budget_exhausted(job.gpu_hours_used, job.gpu_budget)
                    except Exception:
                        log.exception("on_budget_exhausted failed for %s", job.arxiv_id)
                    _git_checkpoint(
                        workspace,
                        f"Job #{submit_count} budget-exhausted ({event_run_id}, used={job.gpu_hours_used:.3f}/{job.gpu_budget:.3f} GPUh)",
                        log_files=[stdout_path, stderr_path],
                        stage_workspace=True,  # preserve partial outputs before repair turn
                    )
                    final_action = worker.get_last_action()
                    budget_finalized = bool(final_action and final_action.action == "done")
                    if not budget_finalized:
                        log.warning(
                            "Paper %s: budget exhausted follow-up did not set action=done; "
                            "requesting explicit finalization",
                            job.arxiv_id,
                        )
                        try:
                            from infrastructure.worker.prompts import (
                                build_budget_exhausted_prompt,
                            )
                            from infrastructure.worker.parsing import (
                                parse_current_milestone,
                            )
                            latest_milestone = job.milestone
                            latest_action = worker.get_last_action()
                            if latest_action:
                                latest_milestone = _higher_milestone(
                                    latest_milestone, latest_action.milestone
                                )
                            md_milestone = parse_current_milestone(job.workspace_path)
                            if md_milestone:
                                latest_milestone = _higher_milestone(
                                    latest_milestone, md_milestone
                                )
                            worker.chat(
                                build_budget_exhausted_prompt(
                                    job.gpu_hours_used,
                                    latest_milestone,
                                )
                            )
                        except Exception:
                            log.exception(
                                "budget exhausted finalization repair failed for %s",
                                job.arxiv_id,
                            )
                        _git_checkpoint(
                            workspace,
                            f"Job #{submit_count} budget-exhausted-repair ({event_run_id}, used={job.gpu_hours_used:.3f}/{job.gpu_budget:.3f} GPUh)",
                            log_files=[stdout_path, stderr_path],
                        )
                        final_action = worker.get_last_action()
                        budget_finalized = bool(final_action and final_action.action == "done")
                break

            else:
                log.debug("Unknown event type %s for %s", event_type, job.arxiv_id)
                break

        return (
            last_container_hash,
            last_setup_hash,
            job_returncode,
            run_id,
            budget_exhausted,
            budget_finalized,
        )

    def _run_setup_in_overlay(
        self,
        container_sif: Path,
        overlay_img: Path,
        setup_sh: Path,
        log_path: str,
    ) -> bool:
        """Run setup.sh inside the container using a persistent overlay.

        Creates the overlay image if it doesn't exist. Packages are
        installed on the host that runs the daemon, which has internet
        access, so compute nodes do not need it.

        The overlay is created and written on node-local disk
        (``local_tmp_dir``) during setup and copied into the workspace
        when done, so the fuse2fs mount (ext3 overlay writes) does not
        hit a shared network filesystem.

        Returns:
            True if setup succeeded.
        """
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        # Create the overlay on node-local disk; after setup completes,
        # copy the finished overlay to its final location in the workspace.
        use_local = not overlay_img.exists()
        local_overlay: Path | None = None
        if use_local:
            local_dir = Path(self.config.local_tmp_dir) / "swarm_overlays"
            local_dir.mkdir(parents=True, exist_ok=True)
            local_overlay = local_dir / f"{overlay_img.stem}_{os.getpid()}_{threading.get_ident()}.img"
            try:
                subprocess.run(
                    ["truncate", "-s", "30G", str(local_overlay)],
                    capture_output=True, timeout=10, check=True,
                )
                subprocess.run(
                    ["mkfs.ext3", "-q", str(local_overlay)],
                    capture_output=True, timeout=60, check=True,
                )
                log.info("Created overlay on local disk: %s", local_overlay)
            except Exception:
                log.exception("Failed to create overlay image")
                if local_overlay and local_overlay.exists():
                    local_overlay.unlink()
                return False

        # Use the local overlay for setup, or the existing workspace one
        active_overlay = local_overlay if local_overlay else overlay_img

        # Resolve symlinks — the fakeroot namespace cannot follow host
        # symlink chains.
        real_overlay = str(active_overlay.resolve())
        real_sif = str(container_sif.resolve())
        real_setup = str(setup_sh.resolve())

        # Bind-mount the host's uv cache into the container so packages
        # downloaded by one paper are reused by subsequent papers.
        uv_cache = self.config.uv_cache_dir or ""
        if not uv_cache:
            uv_cache = os.environ.get("UV_CACHE_DIR", "")
        if not uv_cache:
            uv_cache = str(Path.home() / ".cache" / "uv")
        uv_cache = str(Path(uv_cache).resolve())
        # On a machine that never ran uv the directory does not exist yet;
        # without it nothing is bound and UV_CACHE_DIR stays unset.
        try:
            os.makedirs(uv_cache, exist_ok=True)
        except OSError:
            pass
        cache_bind: list[str] = []
        if os.path.isdir(uv_cache):
            cache_bind = [
                "--bind", f"{uv_cache}:{uv_cache}",
                "--env", f"UV_CACHE_DIR={uv_cache}",
            ]
        # Bind-mount the host's uv binary so every container gets it
        # regardless of what the agent put in container.def.
        # Mount to /opt/uv/bin (NOT /usr/local/bin) to avoid making
        # /usr/local/bin read-only — uv pip installs entry-point scripts
        # there and needs it writable via the overlay.
        uv_bin = shutil.which("uv")
        if uv_bin:
            uv_bin = str(Path(uv_bin).resolve())
            cache_bind.extend(["--bind", f"{uv_bin}:/opt/uv/bin/uv:ro"])

        # Create a wrapper script that redirects all pip variants to uv pip.
        # This catches `pip install`, `python3 -m pip install`, `pip3 install`,
        # and `python -m pip install` — the shim in the SIF only catches the
        # first form.  Also strips --no-cache-dir so the shared cache works.
        #
        # Key design: we save UV_CACHE_DIR into _SWARM_UV_CACHE before
        # sourcing setup.sh.  The uv/pip wrappers always restore it per
        # invocation so agents can't redirect to /tmp or disable caching.
        wrapper = setup_sh.parent / ".setup_wrapper.sh"
        wrapper.write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            "# Prevent host VIRTUAL_ENV from confusing uv inside the container.\n"
            "unset VIRTUAL_ENV\n"
            "export UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1\n"
            "# Use uv from /opt/uv/bin (bind-mounted from host, avoids\n"
            "# making /usr/local/bin read-only which blocks script installs).\n"
            "export PATH=/opt/uv/bin:$PATH\n"
            "export UV_LINK_MODE=copy\n"
            "# Kill hung downloads after 5 min so uv retries instead of\n"
            "# blocking forever on CLOSE-WAIT sockets.\n"
            "export UV_HTTP_TIMEOUT=300\n"
            "# Lock down the cache dir: save it so wrappers can restore it\n"
            "# even if setup.sh overrides UV_CACHE_DIR.\n"
            "export _SWARM_UV_CACHE=\"${UV_CACHE_DIR:-}\"\n"
            "# Run a command with a 10-min timeout; retry up to 3 times.\n"
            "_timed_run() {\n"
            "  local attempt\n"
            "  for attempt in 1 2 3; do\n"
            "    if timeout 600 \"$@\"; then return 0; fi\n"
            "    echo \"[swarm] command timed out or failed (attempt $attempt/3), retrying...\" >&2\n"
            "  done\n"
            "  echo \"[swarm] command failed after 3 attempts: $*\" >&2\n"
            "  return 1\n"
            "}\n"
            "# Redirect all pip variants to uv pip.\n"
            "_fast_pip() {\n"
            "  local args=()\n"
            "  for a in \"$@\"; do\n"
            "    case \"$a\" in\n"
            "      --no-cache-dir|--no-cache) continue ;;\n"
            "    esac\n"
            "    args+=(\"$a\")\n"
            "  done\n"
            "  _timed_run env UV_CACHE_DIR=\"$_SWARM_UV_CACHE\" /opt/uv/bin/uv pip \"${args[@]}\"\n"
            "}\n"
            "pip()  { _fast_pip \"$@\"; }\n"
            "pip3() { _fast_pip \"$@\"; }\n"
            "export -f _timed_run _fast_pip pip pip3\n"
            "# Intercept direct uv calls to strip --no-cache and force shared cache.\n"
            "uv() {\n"
            "  local args=()\n"
            "  for a in \"$@\"; do\n"
            "    case \"$a\" in\n"
            "      --no-cache-dir|--no-cache) continue ;;\n"
            "    esac\n"
            "    args+=(\"$a\")\n"
            "  done\n"
            "  _timed_run env UV_CACHE_DIR=\"$_SWARM_UV_CACHE\" /opt/uv/bin/uv \"${args[@]}\"\n"
            "}\n"
            "export -f uv\n"
            "python3() {\n"
            "  if [ \"${1:-}\" = \"-m\" ] && [ \"${2:-}\" = \"pip\" ]; then\n"
            "    shift 2\n"
            "    _fast_pip \"$@\"\n"
            "  else\n"
            "    command python3 \"$@\"\n"
            "  fi\n"
            "}\n"
            "export -f python3\n"
            f"source {real_setup}\n"
        )
        wrapper.chmod(0o755)
        real_wrapper = str(wrapper.resolve())

        # Bind shared directories read-only so setup.sh can check for
        # existing models/datasets.  setup.sh should only do pip installs —
        # model downloads belong in scripts/download.sh or agent turns.
        shared_bind: list[str] = []
        if self.config.shared_datasets_path and os.path.isdir(self.config.shared_datasets_path):
            real_ds = str(Path(self.config.shared_datasets_path).resolve())
            shared_bind.extend(["--bind", f"{real_ds}:{CONTAINER_SHARED_DATASETS}:ro"])
        if self.config.shared_models_path and os.path.isdir(self.config.shared_models_path):
            real_md = str(Path(self.config.shared_models_path).resolve())
            shared_bind.extend(["--bind", f"{real_md}:{CONTAINER_SHARED_MODELS}:ro"])

        try:
            proc = subprocess.Popen(
                ["apptainer", "exec", "--fakeroot",
                 *cache_bind,
                 *shared_bind,
                 "--overlay", real_overlay,
                 real_sif,
                 "bash", real_wrapper],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,  # own process group for clean kill
            )
            self._child_pids.add(proc.pid)
            try:
                stdout, stderr = proc.communicate(timeout=self.config.worker.setup_timeout)
            except subprocess.TimeoutExpired:
                # Kill the entire process group (apptainer + fakeroot +
                # bash + uv).  proc.kill() only kills the direct child;
                # apptainer's fakeroot runtime parent is a grandchild
                # that gets reparented to PID 1 and keeps running.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
                stdout, stderr = proc.communicate()
                raise
            finally:
                self._child_pids.discard(proc.pid)
            result = subprocess.CompletedProcess(
                proc.args, proc.returncode, stdout, stderr,
            )

            with open(log_path, "w") as f:
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
                f.write(f"\n=== RETURN CODE: {result.returncode} ===\n")

            if result.returncode != 0:
                log.error("setup.sh failed for %s", container_sif)
                if local_overlay and local_overlay.exists():
                    local_overlay.unlink()
                return False

            # Copy the finished overlay from local disk to the workspace.
            # cp --sparse=always transfers only the blocks that hold data,
            # not the full 30 GB of the sparse file.
            if local_overlay:
                try:
                    subprocess.run(
                        ["cp", "--sparse=always", str(local_overlay), str(overlay_img)],
                        capture_output=True, timeout=300, check=True,
                    )
                    local_overlay.unlink()
                    log.info("Copied overlay to workspace (sparse): %s", overlay_img)
                except Exception:
                    log.exception("Failed to copy overlay to workspace")
                    if local_overlay.exists():
                        local_overlay.unlink()
                    if overlay_img.exists():
                        overlay_img.unlink()
                    return False
            log.info("setup.sh completed in overlay: %s", overlay_img)
            return True

        except subprocess.TimeoutExpired:
            with open(log_path, "w") as f:
                timeout = self.config.worker.setup_timeout
                f.write(f"SETUP TIMED OUT after {timeout} seconds ({timeout // 60} min)\n")
            log.error("setup.sh timed out after %ds", self.config.worker.setup_timeout)
            if local_overlay and local_overlay.exists():
                local_overlay.unlink()
            return False
        except Exception:
            log.exception("Unexpected error in setup for %s", container_sif)
            if local_overlay and local_overlay.exists():
                local_overlay.unlink()
            return False

    def _record_env_failure_context(
        self,
        job: PaperJob,
        workspace: Path,
        *,
        phase: str,
        log_path: Path,
    ) -> dict[str, Any]:
        """Capture scheduler hints for a failed build/setup attempt."""
        failure_class, signature, hint = _classify_env_failure_log(log_path, phase)
        container_hash, setup_hash = _env_hashes(workspace)
        env_comparison = "first_failure"
        same_signature = False

        if job.last_env_failure_signature is not None:
            env_comparison = (
                "unchanged"
                if (
                    job.last_env_failure_container_hash == container_hash
                    and job.last_env_failure_setup_hash == setup_hash
                )
                else "changed"
            )
            same_signature = job.last_env_failure_signature == signature

        job.last_env_failure_container_hash = container_hash
        job.last_env_failure_setup_hash = setup_hash
        job.last_env_failure_signature = signature
        job.last_env_failure_class = failure_class

        return {
            "failure_class": failure_class,
            "failure_signature": signature,
            "failure_hint": hint,
            "env_comparison": env_comparison,
            "same_signature": same_signature,
        }

    @staticmethod
    def _clear_env_failure_context(job: PaperJob) -> None:
        """Clear the most recent environment failure context after success."""
        job.last_env_failure_container_hash = None
        job.last_env_failure_setup_hash = None
        job.last_env_failure_signature = None
        job.last_env_failure_class = None

    def _prevalidate_container_source(
        self,
        container_def: Path,
        build_log_path: str,
    ) -> bool:
        """Perform a lightweight image-ref check before a full apptainer build."""
        source_uri = _extract_container_source_uri(container_def)
        if not source_uri:
            return True

        try:
            result = subprocess.run(
                ["apptainer", "inspect", source_uri],
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
        except FileNotFoundError:
            return True
        except subprocess.TimeoutExpired:
            return True
        except Exception:
            return True

        if result.returncode == 0:
            return True

        combined = "\n".join(
            part for part in (result.stdout, result.stderr) if part
        )
        failure_class, signature, _ = _classify_env_failure_text(combined, "build")
        # Only fast-fail definite "image does not exist" type errors.
        # Transient classifications (rate limits, malformed registry
        # responses, name resolution glitches) must reach the build
        # retry loop instead of being turned into hard failures here.
        if failure_class != "invalid_spec":
            return True

        with open(build_log_path, "w") as f:
            f.write("=== PREVALIDATION ===\n")
            f.write(f"Source: {source_uri}\n")
            f.write(f"Classification: {failure_class}\n")
            f.write(f"Signature: {signature}\n")
            f.write("\n=== STDOUT ===\n")
            f.write(result.stdout or "")
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr or "")
            f.write(f"\n=== RETURN CODE: {result.returncode} ===\n")

        log.warning(
            "Container prevalidation failed for %s (%s: %s)",
            container_def,
            failure_class,
            signature,
        )
        return False

    def _get_container_cache_dir(self) -> Path:
        """Return the shared SIF cache directory, creating it if needed."""
        if self.config.container_cache_dir:
            cache = Path(self.config.container_cache_dir)
        else:
            cache = Path(self.config.workspace_base_dir).parent / "shared" / "containers"
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    def _get_base_sif_cache_dir(self) -> Path:
        """Return the cache dir holding pre-pulled docker base SIFs.

        Lives under the container cache dir as a sibling of the full-SIF
        cache so the existing cache root config covers it. The base SIF
        cache is keyed per-canonical-image-ref, not per-container.def.
        """
        cache = self._get_container_cache_dir() / "base_sifs"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / ".locks").mkdir(parents=True, exist_ok=True)
        (cache / "effective_defs").mkdir(parents=True, exist_ok=True)
        return cache

    # Canonical docker ref -> filename of an optional, operator-provided
    # SIF derived from that base image with common dependencies
    # (python/pip/uv and common apt packages) already installed. Looked
    # up in ``<base_sif_cache>/_derived_sifs/``; no such SIFs or build
    # recipes ship with Swarm, and nothing changes when they are absent.
    _DERIVED_BASE_SIFS: dict[str, str] = {
        "docker.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04":
            "cuda12.1-runtime-common.sif",
        "docker.io/library/python:3.11-slim":
            "python3.11-slim-common.sif",
        "docker.io/library/ubuntu:22.04":
            "ubuntu22.04-common.sif",
    }

    def _find_derived_base_sif(self, canonical: str) -> Path | None:
        """Return a pre-built common-deps SIF for ``canonical`` if available.

        Returns ``None`` when no derived SIF is registered for this base
        image or the expected file is missing on disk.
        """
        filename = self._DERIVED_BASE_SIFS.get(canonical)
        if not filename:
            return None
        derived_dir = self._get_base_sif_cache_dir() / "_derived_sifs"
        candidate = derived_dir / filename
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
        return None

    def _ensure_base_sif_cached(
        self,
        canonical: str,
        build_log_path: str,
    ) -> Path | None:
        """Pull ``canonical`` to the base SIF cache if not already present.

        Returns the cached SIF path on success, or None on any failure
        (network, rate limit, apptainer error). Failures are logged but
        not raised — the caller falls through to the original def.

        Concurrency: a per-canonical file lock prevents two builds from
        racing on the same first pull. Re-checks existence under the lock.
        Pulls go to a sibling ``.tmp`` file and rename atomically so a
        crashed pull never leaves a half-written SIF in the cache.
        """
        cache_dir = self._get_base_sif_cache_dir()
        filename = _base_sif_cache_filename(canonical)
        cached_sif = cache_dir / filename
        if cached_sif.exists() and cached_sif.stat().st_size > 0:
            return cached_sif

        # Lock per-image to prevent concurrent first-pulls of the same ref.
        lock_path = cache_dir / ".locks" / f"{filename}.lock"
        try:
            with _file_lock(lock_path, timeout=3900):
                # Re-check inside the lock — another worker may have just pulled.
                if cached_sif.exists() and cached_sif.stat().st_size > 0:
                    return cached_sif

                tmp_sif = cached_sif.with_suffix(
                    cached_sif.suffix + f".tmp.{os.getpid()}"
                )
                if tmp_sif.exists():
                    try:
                        tmp_sif.unlink()
                    except OSError:
                        pass

                cmd = [
                    "apptainer", "pull", "--force", "--disable-cache",
                    str(tmp_sif), f"docker://{canonical}",
                ]
                log.info("Base SIF cache miss — pulling %s", canonical)
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                    )
                    self._child_pids.add(proc.pid)
                    try:
                        stdout, stderr = proc.communicate(timeout=1800)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except OSError:
                            proc.kill()
                        stdout, stderr = proc.communicate()
                        log.warning(
                            "Base SIF pull timed out for %s after 1800s",
                            canonical,
                        )
                        try:
                            with open(build_log_path, "a") as f:
                                f.write(
                                    f"\n=== BASE SIF PULL TIMED OUT: {canonical} ===\n"
                                )
                        except Exception:
                            pass
                        if tmp_sif.exists():
                            try:
                                tmp_sif.unlink()
                            except OSError:
                                pass
                        return None
                    finally:
                        self._child_pids.discard(proc.pid)

                    if proc.returncode != 0:
                        log.warning(
                            "Base SIF pull failed for %s (rc=%s)",
                            canonical, proc.returncode,
                        )
                        try:
                            with open(build_log_path, "a") as f:
                                f.write(
                                    f"\n=== BASE SIF PULL FAILED rc={proc.returncode} ===\n"
                                )
                                f.write(f"Image: {canonical}\n")
                                f.write("--- pull stdout ---\n")
                                f.write(stdout or "")
                                f.write("\n--- pull stderr ---\n")
                                f.write(stderr or "")
                                f.write("\n")
                        except Exception:
                            pass
                        if tmp_sif.exists():
                            try:
                                tmp_sif.unlink()
                            except OSError:
                                pass
                        return None

                    if not tmp_sif.exists() or tmp_sif.stat().st_size == 0:
                        log.warning(
                            "Base SIF pull produced no file for %s",
                            canonical,
                        )
                        return None

                    # Atomic rename — same directory, same filesystem.
                    os.replace(tmp_sif, cached_sif)
                    log.info(
                        "Base SIF cached: %s -> %s (%.1f MB)",
                        canonical, cached_sif.name,
                        cached_sif.stat().st_size / (1024 * 1024),
                    )
                    return cached_sif
                except FileNotFoundError:
                    log.error("apptainer not found on PATH")
                    return None
                except Exception as e:
                    log.warning("Unexpected base SIF pull error for %s: %s",
                                canonical, e)
                    if tmp_sif.exists():
                        try:
                            tmp_sif.unlink()
                        except OSError:
                            pass
                    return None
        except TimeoutError:
            log.warning("Base SIF lock timeout for %s", canonical)
            return None

    def _maybe_rewrite_def_with_cached_base(
        self,
        container_def: Path,
        build_log_path: str,
        *,
        allow_pull: bool = True,
    ) -> Path | None:
        """Try to rewrite ``container_def`` to use a cached base SIF.

        Returns a Path to the rewritten def on success, or None to mean
        "use the original def unchanged". Any failure (parse, pull,
        write) returns None — the rewrite is best-effort and never
        blocks the build path.

        When ``allow_pull`` is False the helper will only succeed if the
        base SIF is already on disk. The pre-prevalidation fast path
        uses this so a cache miss falls through immediately instead of
        triggering a 1800s ``apptainer pull`` ahead of the 45s
        ``apptainer inspect`` fast-fail that catches typoed/unreachable
        From values.
        """
        try:
            text = container_def.read_text(errors="ignore")
        except Exception as e:
            log.debug("Could not read container.def for rewrite: %s", e)
            return None

        safe, canonical = _can_rewrite_def_to_localimage(text)
        if not safe or not canonical:
            return None

        # ── Tier 2 lookup: derived common-deps SIFs ──
        # For base images that many papers share, the operator may place
        # pre-built SIFs with common packages already installed in the
        # cache (see _DERIVED_BASE_SIFS). Rewriting to one of these makes
        # the paper's own ``apt-get install`` lines mostly a no-op and
        # shortens the build. Falls through to Tier 1 (the plain cached
        # base image) if no derived SIF exists.
        cached_sif: Path | None = None
        derived_sif = self._find_derived_base_sif(canonical)
        if derived_sif is not None:
            cached_sif = derived_sif

        if cached_sif is None:
            if allow_pull:
                cached_sif = self._ensure_base_sif_cached(canonical, build_log_path)
            else:
                cache_dir = self._get_base_sif_cache_dir()
                candidate = cache_dir / _base_sif_cache_filename(canonical)
                if candidate.exists() and candidate.stat().st_size > 0:
                    cached_sif = candidate
                else:
                    cached_sif = None
        if cached_sif is None:
            return None

        rewritten = _rewrite_def_to_localimage(text, cached_sif)
        if rewritten is None:
            return None

        # Persist the rewritten def to a per-build file we can pass to
        # apptainer build.
        cache_dir = self._get_base_sif_cache_dir()
        try:
            def_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
            effective_path = (
                cache_dir / "effective_defs"
                / f"{def_hash}_{os.getpid()}.def"
            )
            effective_path.write_text(rewritten)
        except Exception as e:
            log.warning("Could not write effective container.def: %s", e)
            return None

        try:
            with open(build_log_path, "a") as f:
                f.write("\n=== BASE SIF CACHE HIT ===\n")
                f.write(f"Original: docker://{canonical}\n")
                f.write(f"Cached SIF: {cached_sif}\n")
                f.write(f"Effective def: {effective_path}\n")
        except Exception:
            pass
        log.info("Container build using cached base SIF for %s", canonical)
        return effective_path

    def _get_setup_cache_dir(self) -> Path:
        """Return the shared setup-overlay cache directory."""
        if self.config.container_cache_dir:
            cache = Path(self.config.container_cache_dir).parent / "setup_overlays"
        else:
            cache = Path(self.config.workspace_base_dir).parent / "shared" / "setup_overlays"
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    @staticmethod
    def _build_setup_cache_key(container_hash: str, setup_hash: str) -> str:
        return hashlib.sha256(f"{container_hash}:{setup_hash}".encode()).hexdigest()

    def _prepare_setup_overlay(
        self,
        container_sif: Path,
        overlay_img: Path,
        setup_sh: Path,
        log_path: str,
        *,
        arxiv_id: str,
        setup_cache_key: str,
    ) -> bool:
        """Reuse or build a setup overlay keyed by container+setup hashes."""
        cache_dir = self._get_setup_cache_dir()
        cached_overlay = cache_dir / f"{setup_cache_key}.img"
        meta_path = cache_dir / f"{setup_cache_key}.json"
        lock_path = cache_dir / ".locks" / f"{setup_cache_key}.lock"

        try:
            with _file_lock(lock_path, timeout=self.config.worker.setup_timeout + 300):
                if cached_overlay.exists():
                    _copy_artifact(cached_overlay, overlay_img)
                    with open(log_path, "w") as f:
                        f.write("=== CACHE HIT ===\n")
                        f.write(f"Copied setup overlay from {cached_overlay}\n")
                        f.write("=== RETURN CODE: 0 ===\n")
                    log.info("Setup cache hit: %s -> %s", cached_overlay.name, overlay_img)
                    return True

                if overlay_img.exists():
                    overlay_img.unlink()

                success = self._run_setup_in_overlay(
                    container_sif, overlay_img, setup_sh, log_path,
                )
                if not success:
                    return False

                try:
                    _copy_artifact(overlay_img, cached_overlay)
                    meta_path.write_text(json.dumps({
                        "arxiv_id": arxiv_id,
                        "container_sif": str(container_sif),
                        "setup_sh": str(setup_sh),
                        "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }, indent=2) + "\n")
                    log.info("Setup overlay cached: %s", cached_overlay.name)
                except Exception as e:
                    log.warning("Failed to cache setup overlay: %s", e)

                return True
        except TimeoutError as e:
            with open(log_path, "w") as f:
                f.write(f"SETUP CACHE LOCK TIMED OUT: {e}\n")
            log.error("Timed out waiting for setup cache lock for %s", arxiv_id)
            return False

    # Build retry policy for transient_external failures. Backoff is short
    # by design: long retries lock up an agent slot, and Docker Hub rate
    # limits reset on the order of hours, so retries of a few minutes
    # cannot outlast them. The base SIF cache is the main protection
    # against rate limits; the retries only absorb short glitches.
    _BUILD_MAX_ATTEMPTS = 3
    _BUILD_RETRY_BACKOFFS = (30, 120, 300)

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep in 1-second slices, returning early on shutdown."""
        end = time.time() + seconds
        while time.time() < end:
            if getattr(self, "_shutdown", False):
                return
            time.sleep(min(1.0, end - time.time()))

    def _build_container(
        self,
        container_def: Path,
        container_sif: Path,
        build_log_path: str,
        arxiv_id: str = "",
    ) -> bool:
        """Build an Apptainer container from a definition file.

        Uses a shared SIF cache keyed by the SHA-256 of container.def.
        On cache hit, copies the cached SIF instead of rebuilding.
        On cache miss, builds and stores the result for future reuse.

        Build steps for a cache miss:

        1. ``_maybe_rewrite_def_with_cached_base`` rewrites the def to use
           a pre-cached base SIF when possible. Falls through to the
           original def on any failure.
        2. ``apptainer build`` runs against the effective def.
        3. On ``transient_external`` failures (Docker Hub rate limit,
           registry parse errors, etc.) the build is retried up to
           ``_BUILD_MAX_ATTEMPTS`` times with the configured backoff.

        The full-SIF cache key always uses the ORIGINAL def hash so the
        rewrite is invisible to callers.

        Returns:
            True if build succeeded.
        """
        Path(build_log_path).parent.mkdir(parents=True, exist_ok=True)

        # Portability gate: container.final.def is shipped downstream, so
        # the on-disk container.def the agent writes must build on any
        # host. Runs BEFORE the cache lookup so existing cached SIFs
        # built from non-portable defs can't be served either. Failure
        # message contains the literal phrase 'non-portable container
        # base' so _classify_env_failure_text routes it to agent_fixable.
        portable, portable_err = _validate_container_def_portable(container_def)
        if not portable:
            with open(build_log_path, "w") as f:
                f.write(f"=== PORTABILITY CHECK FAILED for {arxiv_id} ===\n")
                f.write(f"container.def: {container_def}\n\n")
                f.write(portable_err or "")
                f.write("\n=== RETURN CODE: 1 ===\n")
            log.warning(
                "Container portability check failed for %s: %s",
                arxiv_id, (portable_err or "").splitlines()[0],
            )
            return False

        # Check cache (key = original def hash, ignores rewriting).
        def_hash = _hash_file(container_def)
        cache_dir = self._get_container_cache_dir()
        cached_sif = cache_dir / f"{def_hash}.sif"
        meta_path = cache_dir / f"{def_hash}.json"
        lock_path = cache_dir / ".locks" / f"{def_hash}.lock"

        try:
            with _file_lock(lock_path, timeout=3900):
                if cached_sif.exists():
                    _copy_artifact(cached_sif, container_sif)
                    with open(build_log_path, "w") as f:
                        f.write("=== CACHE HIT ===\n")
                        f.write(f"Copied from {cached_sif}\n")
                        f.write("=== RETURN CODE: 0 ===\n")
                    log.info("Container cache hit: %s -> %s", cached_sif.name, container_sif)
                    return True

                # Truncate any prior build log content before any helper
                # writes to it. ``build_log_path`` is reused across
                # rebuilds for the same workspace, and a stale signature
                # from a previous attempt would otherwise be picked up
                # by ``_record_env_failure_context`` and reported as the
                # current failure.
                with open(build_log_path, "w") as f:
                    f.write(f"=== BUILD STARTED for {arxiv_id} ===\n")
                    f.write(f"Original container.def: {container_def}\n")

                # Fast-path rewrite: if the base SIF is ALREADY cached,
                # swap to ``Bootstrap: localimage`` now so prevalidation
                # skips the docker.io ``apptainer inspect``, which can be
                # rate-limited. ``allow_pull=False`` matters here: if the
                # base is NOT cached, fall through to the original
                # docker:// def so prevalidation can fast-fail typos and
                # unreachable refs in ~45s instead of waiting up to 1800s
                # for an ``apptainer pull`` that may not even be needed.
                effective_def: Path | None = self._maybe_rewrite_def_with_cached_base(
                    container_def, build_log_path, allow_pull=False,
                )
                effective_def_to_clean: Path | None = effective_def
                prevalidate_target = effective_def if effective_def is not None else container_def
                if not self._prevalidate_container_source(
                    prevalidate_target, build_log_path,
                ):
                    if effective_def_to_clean is not None:
                        try:
                            effective_def_to_clean.unlink()
                        except OSError:
                            pass
                    return False

                last_failure_signature = ""
                try:
                    for attempt in range(1, self._BUILD_MAX_ATTEMPTS + 1):
                        if getattr(self, "_shutdown", False):
                            log.warning(
                                "Container build aborted for %s "
                                "(shutdown signaled before attempt %d)",
                                container_def, attempt,
                            )
                            with open(build_log_path, "a") as f:
                                f.write(
                                    f"\n=== ABORTED before attempt {attempt} "
                                    f"(shutdown) ===\n"
                                )
                            return False

                        with open(build_log_path, "a") as f:
                            f.write(
                                f"\n=== ATTEMPT {attempt}/{self._BUILD_MAX_ATTEMPTS} ===\n"
                            )

                        # Per-attempt rewrite retry: if no rewrite is in
                        # effect (the pre-loop attempt or a previous in-loop
                        # attempt failed to pull), try again. A pull that
                        # failed transiently last time may succeed now.
                        if effective_def is None:
                            effective_def = self._maybe_rewrite_def_with_cached_base(
                                container_def, build_log_path,
                            )
                            if effective_def is not None:
                                effective_def_to_clean = effective_def
                        build_target = effective_def if effective_def is not None else container_def

                        build_env = os.environ.copy()
                        build_env["APPTAINER_TMPDIR"] = self.config.local_tmp_dir
                        # Use zstd level 1 instead of the default gzip for
                        # mksquashfs: compression is many times faster for
                        # a slightly larger SIF. With gzip, compression
                        # dominates CPU load when several builds run in
                        # parallel on the same host.
                        proc = subprocess.Popen(
                            ["apptainer", "build", "--force",
                             "--mksquashfs-args", "-comp zstd -Xcompression-level 1",
                             str(container_sif), str(build_target)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            start_new_session=True,
                            env=build_env,
                        )
                        self._child_pids.add(proc.pid)
                        timed_out = False
                        try:
                            try:
                                stdout, stderr = proc.communicate(timeout=2700)
                            except subprocess.TimeoutExpired:
                                try:
                                    os.killpg(proc.pid, signal.SIGKILL)
                                except OSError:
                                    proc.kill()
                                stdout, stderr = proc.communicate()
                                timed_out = True
                        finally:
                            self._child_pids.discard(proc.pid)

                        rc = proc.returncode
                        with open(build_log_path, "a") as f:
                            f.write("=== STDOUT ===\n")
                            f.write(stdout or "")
                            f.write("\n=== STDERR ===\n")
                            f.write(stderr or "")
                            f.write(
                                f"\n=== RETURN CODE: {rc}"
                                f"{' (TIMEOUT)' if timed_out else ''} ===\n"
                            )

                        if rc == 0 and not timed_out:
                            try:
                                _copy_artifact(container_sif, cached_sif)
                                meta_path.write_text(json.dumps({
                                    "arxiv_id": arxiv_id,
                                    "container_def": str(container_def),
                                    "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                }, indent=2) + "\n")
                                log.info("Container cached: %s", cached_sif.name)
                            except Exception as e:
                                log.warning("Failed to cache SIF: %s", e)
                            log.info("Container built: %s", container_sif)
                            return True

                        # Failure path
                        if timed_out:
                            # A build that hits the timeout is rarely
                            # transient, so it is not retried.
                            log.error(
                                "Container build timed out for %s (attempt %d)",
                                container_def, attempt,
                            )
                            return False

                        failure_class, signature, _ = _classify_env_failure_text(
                            (stdout or "") + "\n" + (stderr or ""),
                            "build",
                        )
                        last_failure_signature = signature
                        if (failure_class != "transient_external"
                                or attempt >= self._BUILD_MAX_ATTEMPTS
                                or getattr(self, "_shutdown", False)):
                            log.error(
                                "Container build failed for %s "
                                "(class=%s, signature=%s, attempt=%d)",
                                container_def, failure_class, signature, attempt,
                            )
                            return False

                        backoff = self._BUILD_RETRY_BACKOFFS[
                            min(attempt - 1, len(self._BUILD_RETRY_BACKOFFS) - 1)
                        ]
                        log.warning(
                            "Container build attempt %d/%d failed for %s "
                            "(transient_external: %s) — retrying in %ds",
                            attempt, self._BUILD_MAX_ATTEMPTS, container_def,
                            signature, backoff,
                        )
                        with open(build_log_path, "a") as f:
                            f.write(
                                f"\n=== RETRY in {backoff}s "
                                f"(transient_external: {signature}) ===\n"
                            )
                        self._interruptible_sleep(backoff)

                    # Loop fell through (shouldn't happen — we always return
                    # inside the loop on the last attempt).
                    log.error(
                        "Container build retry loop fell through for %s "
                        "(last signature: %s)",
                        container_def, last_failure_signature,
                    )
                    return False
                finally:
                    if effective_def_to_clean is not None:
                        try:
                            effective_def_to_clean.unlink()
                        except OSError:
                            pass

        except TimeoutError:
            with open(build_log_path, "w") as f:
                f.write("BUILD CACHE LOCK TIMED OUT\n")
            log.error("Container build lock timed out for %s", container_def)
            return False
        except FileNotFoundError:
            with open(build_log_path, "w") as f:
                f.write("apptainer command not found\n")
            log.error("apptainer not found on PATH")
            return False

    # ── State persistence ─────────────────────────────────────────────

    def _finish_paper(
        self,
        job: PaperJob,
        trajectory_path: str | None = None,
    ):
        """Move a paper from active to completed and dispatch next."""
        job.finished_at = time.time()
        iter_duration = job.finished_at - job.started_at
        iter_active_duration = max(0.0, iter_duration - job.blocked_sec)
        job.total_duration_sec += iter_active_duration
        result = PaperResult(
            arxiv_id=job.arxiv_id,
            milestone=job.milestone,
            status=job.status,
            error=job.error,
            duration_sec=iter_active_duration,
            total_duration_sec=job.total_duration_sec,
            finished_at=job.finished_at,
            trajectory_path=trajectory_path,
            profile=job.profile,
            iteration=job.iteration,
            workflow=job.workflow,
            workflow_step=job.workflow_step,
        )

        with self._capacity_cv:
            self.active.pop(job.arxiv_id, None)
            self._active_workers.pop(job.arxiv_id, None)
            self._capacity_state.pop(job.arxiv_id, None)
            self._dequeue_working_waiter_locked(job.arxiv_id)
            self._finalizing[job.arxiv_id] = job
            self.completed[job.arxiv_id] = result
            self._capacity_cv.notify_all()

        try:
            # If an iteration agent errored before advancing past the
            # starting milestone, restore the previous milestone so an
            # infrastructure crash doesn't permanently demote the paper.
            if (job.is_iteration
                    and job.status == "error"
                    and job.previous_milestone
                    and job.previous_milestone != "none"):
                prev_rank = _MILESTONE_RANK.get(job.previous_milestone, 0)
                cur_rank = _MILESTONE_RANK.get(job.milestone, 0)
                if cur_rank < prev_rank:
                    log.info(
                        "Restoring milestone for %s: %s -> %s "
                        "(iteration agent errored before advancing)",
                        job.arxiv_id, job.milestone, job.previous_milestone,
                    )
                    job.milestone = job.previous_milestone
                    result.milestone = job.previous_milestone

            self._save_paper_meta(job)
            log.info("Paper %s finished: status=%s milestone=%s (%.0fs)",
                     job.arxiv_id, result.status, result.milestone, result.duration_sec)
            has_next_iteration = False

            # Freeze overlay packages for non-workflow papers or cancelled papers
            if not job.workflow or job.status == "cancelled":
                env_dir = Path(job.workspace_path) / "environment"
                _freeze_overlay_into_containerdef(
                    Path(job.workspace_path), env_dir / "container.def", env_dir / "setup.sh",
                    fallback_sif=self.config.worker.container_sif,
                )

            # Auto-advance workflow if there are more steps
            if job.workflow and job.status not in ("error", "cancelled"):
                wf_steps = self.config.workflows.get(job.workflow, [])
                current_step = wf_steps[job.workflow_step] if job.workflow_step < len(wf_steps) else None
                next_step = job.workflow_step + 1

                # Gate: if the current step has min_milestone and we didn't
                # reach it, either revert to an earlier iteration (if the
                # step is configured to) or stop the workflow.
                if current_step and current_step.min_milestone:
                    min_rank = _MILESTONE_RANK.get(current_step.min_milestone, 0)
                    cur_rank = _MILESTONE_RANK.get(job.milestone, 0)
                    if cur_rank < min_rank:
                        revert_target_1indexed = current_step.revert_to_iteration
                        wf_meta = self.config.workflow_meta.get(job.workflow)
                        max_reverts = int(wf_meta.max_revert_attempts) if wf_meta else 0
                        can_revert = (
                            revert_target_1indexed is not None
                            and max_reverts > 0
                            and job.revert_attempts < max_reverts
                        )
                        if can_revert:
                            target_step = revert_target_1indexed - 1  # 1-indexed -> 0-indexed
                            new_revert_count = job.revert_attempts + 1
                            log.warning(
                                "Paper %s reverting: milestone %s < min_milestone %s "
                                "at iteration %d — restart from iteration %d (revert %d/%d)",
                                job.arxiv_id, job.milestone, current_step.min_milestone,
                                job.workflow_step + 1, revert_target_1indexed,
                                new_revert_count, max_reverts,
                            )
                            # Persist the bumped counter before re-queueing so
                            # a crash between here and iterate_paper() doesn't
                            # lose the attempt.
                            job.revert_attempts = new_revert_count
                            self._save_paper_meta(job)
                            try:
                                self.iterate_paper(
                                    job.arxiv_id,
                                    workflow_step_override=target_step,
                                    date=job.date, topics=job.topics,
                                    queue_front=True,
                                    fresh=True,
                                    revert_attempts_override=new_revert_count,
                                )
                                self._dispatch_next()
                                return
                            except Exception as e:
                                log.error(
                                    "Revert failed for %s, falling through to unsuccessful: %s",
                                    job.arxiv_id, e,
                                )
                                # fall through to terminal state below
                        log.warning(
                            "Paper %s unsuccessful: milestone %s < min_milestone %s "
                            "at iteration %d (reverts used: %d/%d).",
                            job.arxiv_id, job.milestone, current_step.min_milestone,
                            job.workflow_step + 1,
                            job.revert_attempts, max_reverts,
                        )
                        job.status = "unsuccessful"
                        result.status = "unsuccessful"
                        # Workflow stops here — freeze overlay now
                        env_dir = Path(job.workspace_path) / "environment"
                        _freeze_overlay_into_containerdef(
                            Path(job.workspace_path), env_dir / "container.def", env_dir / "setup.sh",
                            fallback_sif=self.config.worker.container_sif,
                        )
                        self._save_paper_meta(job)
                        self._dispatch_next()
                        return

                if next_step < len(wf_steps):
                    has_next_iteration = True
                    step = wf_steps[next_step]
                    log.info("Workflow '%s' advancing %s to step %d/%d (profile=%s)",
                             job.workflow, job.arxiv_id, next_step + 1, len(wf_steps), step.profile)
                    try:
                        self.iterate_paper(
                            job.arxiv_id, profile=step.profile,
                            workflow_step_override=next_step,
                            date=job.date, topics=job.topics,
                            queue_front=True,
                        )
                    except Exception as e:
                        log.error("Workflow auto-advance failed for %s: %s", job.arxiv_id, e)
                else:
                    # Workflow complete — freeze overlay packages into container.def
                    # for a self-contained artifact now that no more iterations
                    # will modify either file.
                    env_dir = Path(job.workspace_path) / "environment"
                    _freeze_overlay_into_containerdef(
                        Path(job.workspace_path), env_dir / "container.def", env_dir / "setup.sh",
                        fallback_sif=self.config.worker.container_sif,
                    )

            if job.paper and self.config.github_org:
                _update_github_repo_metadata(
                    Path(job.workspace_path),
                    job.paper,
                    self.config.github_org,
                    job.milestone,
                    extra_topics=_iteration_completion_topics(
                        job.iteration,
                        status=job.status,
                        has_next_iteration=has_next_iteration,
                    ),
                )

            self._dispatch_next()
        finally:
            with self._lock:
                self._finalizing.pop(job.arxiv_id, None)

    def _save_paper_meta(self, job: PaperJob):
        """Persist paper metadata to disk.

        Enforces ``iteration == workflow_step + 1`` for workflow-backed
        papers — the counter tracks the 1-indexed workflow position, not
        a running attempt count.  Any drift is clamped here so stale
        callers can't leak bad numbers to meta.json.
        """
        if job.workflow and job.workflow_step is not None and job.iteration is not None:
            expected = job.workflow_step + 1
            if job.iteration != expected:
                log.warning(
                    "Clamped iteration for %s: %d -> %d (workflow_step=%d)",
                    job.arxiv_id, job.iteration, expected, job.workflow_step,
                )
                job.iteration = expected
        paper_dir = PAPERS_DIR / job.arxiv_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        meta_path = paper_dir / "meta.json"
        meta_path.write_text(json.dumps(job.to_dict(), indent=2))


# ── Helpers ───────────────────────────────────────────────────────────────

def _hash_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _hash_optional_file(path: Path) -> str | None:
    """Return a file hash when the path exists, otherwise None."""
    return _hash_file(path) if path.exists() else None


def _classify_env_failure_text(text: str, phase: str) -> tuple[str, str, str]:
    """Classify a build/setup failure into a coarse, scheduler-facing bucket."""
    lowered = text.lower()
    compact = " ".join(text.split())

    rules: list[tuple[str, str, str, tuple[str, ...]]] = [
        (
            "invalid_spec",
            "container image not found",
            "The base image reference looks invalid or no longer exists.",
            (
                "name_unknown",
                "repository with name",
            ),
        ),
        (
            "transient_external",
            "registry response malformed",
            "The registry or network returned malformed image metadata.",
            (
                "invalid character",
                "top-level value",
                "conveyor failed to get",
            ),
        ),
        (
            "system_capacity",
            "disk quota exceeded",
            "The filesystem refused more writes, so retrying unchanged may not help.",
            ("disk quota exceeded",),
        ),
        (
            "system_capacity",
            "out of space",
            "The filesystem ran out of free space during environment preparation.",
            ("no space left on device",),
        ),
        (
            "system_capacity",
            "memory allocation failure",
            "The install path hit a memory/resource limit outside the workspace logic.",
            ("cannot allocate memory",),
        ),
        (
            "agent_fixable",
            "missing local package directory",
            "The environment references a local package directory that does not exist in the workspace.",
            ("/home/user/pkgs", "no such file or directory"),
        ),
        (
            "agent_fixable",
            "pip install in container build",
            "Python packages appear to be installed in container.def instead of setup.sh.",
            ("pip install", "%post"),
        ),
        (
            "agent_fixable",
            "uv unavailable during container build",
            "The container build references uv during %post, where it may not exist.",
            ("uv: command not found",),
        ),
        (
            "agent_fixable",
            "non-portable container base",
            "container.def must use 'Bootstrap: docker' with a Docker Hub From: ref so the same def builds on any machine.",
            ("non-portable container base",),
        ),
    ]

    for failure_class, signature, hint, needles in rules:
        if all(needle in lowered for needle in needles):
            return failure_class, signature, hint

    if "timed out" in lowered:
        return (
            "transient_external",
            f"{phase} timeout",
            "The environment step timed out before finishing.",
        )
    if phase == "setup" and "return code: 143" in lowered:
        return (
            "transient_external",
            "setup terminated",
            "The setup process was terminated externally before it finished.",
        )

    summary = compact[:200] if compact else f"{phase} failed"
    return (
        "unknown",
        summary or f"{phase} failed",
        "The failure signature is not recognized; inspect the log before retrying.",
    )


def _classify_env_failure_log(log_path: Path, phase: str) -> tuple[str, str, str]:
    """Classify a build/setup failure by reading its log file."""
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        text = ""
    return _classify_env_failure_text(text, phase)


def _validate_container_def_portable(
    container_def: Path,
) -> tuple[bool, str | None]:
    """Return (is_portable, error_message) for a container.def.

    A def is portable when the same file builds on any host: Docker Hub
    is reachable everywhere, but other bootstraps and registries depend
    on host capabilities. ``container.final.def`` is shipped to
    downstream consumers, so the def the agent writes must satisfy the
    portability contract from the very first build attempt — otherwise
    the workflow ships an artifact that only works on its build host.

    Rules:
      - ``Bootstrap:`` must be ``docker``. Rejects ``dnf`` (needs
        fakeroot+devtmpfs, breaks on nodev mounts), ``localimage`` and
        ``oci-archive`` (need a specific cwd or local file), ``library``,
        ``oras``, ``scratch``, etc.
      - ``From:`` must resolve to a ``docker.io`` reference. Rejects
        ``ghcr.io/...``, ``quay.io/...``, ``gcr.io/...``, and any
        path-shaped ref that isn't a valid Docker Hub image.

    The error message includes the literal phrase
    ``non-portable container base`` so ``_classify_env_failure_text``
    surfaces this as ``agent_fixable`` with a hint the agent can act on.
    """
    try:
        text = container_def.read_text(errors="ignore")
    except OSError:
        return True, None  # can't read — let the regular pipeline error

    header_lines, _body = _split_def_header_and_body(text)
    parsed = _parse_def_header_block(header_lines)
    if not parsed:
        return True, None  # parser bailed — defer to apptainer

    bootstrap = parsed.get("bootstrap")
    from_ref = parsed.get("from")

    if bootstrap and bootstrap != "docker":
        return False, (
            f"non-portable container base: Bootstrap: {bootstrap}\n"
            f"\n"
            f"container.def must use 'Bootstrap: docker' so the same def "
            f"builds on any machine. Bootstraps that depend on host "
            f"capabilities (dnf needs fakeroot+devtmpfs, localimage and "
            f"oci-archive need a specific cwd or local file) work where "
            f"they were authored but break on a fresh rebuild — and "
            f"container.final.def is shipped to downstream consumers.\n"
            f"\n"
            f"Replace the header with:\n"
            f"  Bootstrap: docker\n"
            f"  From: <docker hub image>\n"
            f"\n"
            f"Examples: 'almalinux:9', 'fedora:40', 'python:3.10-slim-bookworm', "
            f"'nvidia/cuda:12.1.0-runtime-ubuntu22.04'."
        )

    if from_ref:
        ref_info = _parse_docker_image_ref(from_ref)
        if ref_info is None or ref_info.get("domain") != "docker.io":
            domain_seen = ref_info.get("domain") if ref_info else "(unparseable)"
            return False, (
                f"non-portable container base: From: {from_ref}\n"
                f"\n"
                f"From: must resolve to a Docker Hub image. Got domain "
                f"'{domain_seen}'. Other registries (ghcr.io, quay.io, "
                f"gcr.io, ...) often require authentication or are rate "
                f"limited from the build hosts that consume "
                f"container.final.def.\n"
                f"\n"
                f"Use a Docker Hub equivalent. Examples: "
                f"'python:3.10-slim-bookworm', 'almalinux:9', 'fedora:40', "
                f"'nvidia/cuda:12.1.0-runtime-ubuntu22.04'."
            )

    return True, None


def _extract_container_source_uri(container_def: Path) -> str | None:
    """Return a lightweight OCI/library URI for container prevalidation when possible.

    Parses ONLY the first header block (everything before the first
    ``%section`` line) via the shared header parser, so a ``Registry:``
    or ``Namespace:`` directive that belongs to a later stage of a
    multi-stage def can never get mixed into the first stage's URI.

    For ``Bootstrap: docker`` defs that also carry ``Registry:`` or
    ``Namespace:`` headers, this returns the canonical
    ``docker://<domain>/<namespace>/<name>:<tag>`` form so prevalidation
    queries the actual image rather than mistakenly inspecting
    ``docker://<from>`` against Docker Hub. When anything looks
    ambiguous — multi-stage def, slashed-From combined with a header
    Registry, parse failure — this returns None so prevalidation skips
    entirely instead of validating the wrong image.
    """
    try:
        text = container_def.read_text(errors="ignore")
    except Exception:
        return None

    header_lines, _body = _split_def_header_and_body(text)
    parsed = _parse_def_header_block(header_lines)
    if not parsed:
        return None

    # Multi-stage defs are not safe to prevalidate as a single image —
    # the first stage's image may not even be the one apptainer fails on.
    # Skip rather than risk a false invalid_spec.
    if parsed["bootstrap_count"] != 1:
        return None

    bootstrap = parsed["bootstrap"]
    from_ref = parsed["from"]
    if not bootstrap or not from_ref:
        return None
    if bootstrap not in {"docker", "oras", "library"}:
        return None

    if bootstrap == "docker" and (parsed["registry"] or parsed["namespace"]):
        canonical = _canonical_docker_ref(
            from_ref, parsed["registry"], parsed["namespace"],
        )
        if canonical is None:
            # Ambiguous merge — better to skip prevalidation entirely
            # than to validate the wrong image.
            return None
        return f"docker://{canonical}"

    # Apptainer accepts both ``From: docker://X`` and ``From: X``. If the
    # ref already carries a scheme, return it as-is rather than prepending
    # a second ``docker://`` (which apptainer then rejects as unparseable).
    if "://" in from_ref:
        return from_ref
    return f"{bootstrap}://{from_ref}"


# ── Base SIF cache: pull each docker base image once ──
#
# Container builds for distinct papers re-pull the same base layers from
# Docker Hub on every cache miss, which trips Docker Hub's per-IP pull rate
# limit and surfaces as ``conveyor failed to get: invalid character ...``
# Apptainer build errors. The base SIF cache pre-downloads each base image
# exactly once, then transparently rewrites ``Bootstrap: docker`` to
# ``Bootstrap: localimage`` so subsequent builds reuse the cached SIF.
#
# Safety: rewrite is conservative — anything ambiguous (multi-stage,
# non-docker bootstrap, malformed header) falls through to the original
# def, which apptainer can still build directly.

_DOCKER_REF_INVALID_CHARS = re.compile(r"[^a-zA-Z0-9._\-/:@]")

# Header keys we recognize on the Apptainer header block.  Anything else
# (Stage:, Fingerprint:, etc.) makes us bail out and skip the rewrite.
_KNOWN_HEADER_KEYS = {"bootstrap", "from", "registry", "namespace", "include"}


def _split_def_header_and_body(text: str) -> tuple[list[str], list[str]]:
    """Split a container.def into (header_lines, body_lines).

    The header is everything before the first ``%section`` line. Comments
    and blank lines belong to whichever side they're physically on.
    """
    lines = text.splitlines()
    body_start = len(lines)
    for i, raw in enumerate(lines):
        if raw.lstrip().startswith("%"):
            body_start = i
            break
    return lines[:body_start], lines[body_start:]


def _parse_def_header_block(header_lines: list[str]) -> dict | None:
    """Parse the header block of a container.def.

    Returns a dict with::

        {
            "bootstrap": str,        # lowered
            "from": str,             # raw From value (no comments)
            "registry": str | None,  # raw Registry value if present
            "namespace": str | None,
            "stage": str | None,     # if present, signals multi-stage
            "extra_keys": set[str],  # unknown header keys we saw
            "bootstrap_count": int,  # number of Bootstrap lines (>1 = multi-stage)
        }

    Returns None on parse failure.
    """
    parsed: dict[str, Any] = {
        "bootstrap": None,
        "from": None,
        "registry": None,
        "namespace": None,
        "stage": None,
        "extra_keys": set(),
        "bootstrap_count": 0,
    }
    try:
        for raw in header_lines:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                # Header lines should be ``Key: value`` — anything else
                # is suspicious.  Don't fail outright; just skip the line.
                continue
            key, value = stripped.split(":", 1)
            key = key.strip().lower()
            # Strip trailing inline comments — Apptainer doesn't support
            # them but agents sometimes write them and apptainer ignores
            # them.  Be defensive.
            value = value.split("#", 1)[0].strip()
            if key == "bootstrap":
                parsed["bootstrap_count"] += 1
                if parsed["bootstrap"] is None:
                    parsed["bootstrap"] = value.lower()
            elif key == "from" and parsed["from"] is None:
                parsed["from"] = value
            elif key == "registry" and parsed["registry"] is None:
                parsed["registry"] = value
            elif key == "namespace" and parsed["namespace"] is None:
                parsed["namespace"] = value
            elif key == "stage":
                parsed["stage"] = value
            elif key == "include":
                pass  # benign — apptainer extra packages, doesn't affect base
            else:
                parsed["extra_keys"].add(key)
    except Exception:
        return None
    return parsed


def _parse_docker_image_ref(ref: str) -> dict | None:
    """Parse a docker image reference into ``{domain, name, tag, digest}``.

    Follows the same domain-detection rule as the Docker reference grammar:
    if the first slash-segment contains ``.``, ``:``, or is exactly
    ``localhost``, it's a registry; otherwise the implicit registry is
    ``docker.io``. Single-segment names on docker.io get the implicit
    ``library/`` namespace.

    Returns None on bad input.
    """
    if not ref:
        return None
    ref = ref.strip()
    # Apptainer accepts ``From: docker://nvidia/cuda:...`` and
    # ``From: nvidia/cuda:...`` interchangeably; strip the optional scheme
    # before parsing or the ``docker:`` head would be treated as a domain
    # and we'd emit ``docker://<ref>`` as the canonical — which, once the
    # caller prepends another ``docker://``, becomes a doubly-schemed URI
    # that apptainer rejects with "unable to parse image name".
    if ref.lower().startswith("docker://"):
        ref = ref[len("docker://"):].lstrip()
    if not ref or _DOCKER_REF_INVALID_CHARS.search(ref):
        return None

    # Digest first (separator @).
    digest = None
    if "@" in ref:
        ref_part, digest = ref.rsplit("@", 1)
        if not digest or ":" not in digest:
            return None
    else:
        ref_part = ref

    # Domain vs path.
    if "/" in ref_part:
        head, rest = ref_part.split("/", 1)
        if "." in head or ":" in head or head == "localhost":
            domain = head
            path = rest
        else:
            domain = "docker.io"
            path = ref_part
    else:
        domain = "docker.io"
        path = ref_part

    if not path:
        return None

    # Tag is in the LAST path segment, after the last ``:``.
    last_slash = path.rfind("/")
    last_segment = path[last_slash + 1:]
    if ":" in last_segment:
        colon_in_path = last_slash + 1 + last_segment.rfind(":")
        name = path[:colon_in_path]
        tag = path[colon_in_path + 1:]
        if not name or not tag:
            return None
    else:
        name = path
        tag = None

    # Implicit ``library/`` namespace for single-segment docker.io names.
    if domain == "docker.io" and "/" not in name:
        name = f"library/{name}"

    return {"domain": domain, "name": name, "tag": tag, "digest": digest}


def _canonical_docker_ref(
    from_ref: str,
    registry: str | None,
    namespace: str | None,
) -> str | None:
    """Build a canonical fully-qualified docker reference string.

    Honors ``Registry:`` and ``Namespace:`` header lines, but only when
    From is a "bare" image name (no slashes in the path). When From
    encodes its own domain or namespace AND a Registry/Namespace header
    is also set, the intent is ambiguous and we return None so the
    caller skips the rewrite and lets apptainer build the def directly.

    Returns ``<domain>/<name>:<tag>`` or ``<domain>/<name>@<digest>``
    (tag omitted on digest pins, matching docker reference resolution).
    """
    if not from_ref:
        return None
    from_ref = from_ref.strip()
    if not from_ref:
        return None

    if registry or namespace:
        # Bare-name check on the RAW from_ref. We can only honor an
        # explicit Registry/Namespace header when From is a single-token
        # name with no slash. Any slash means From carries its own
        # domain or namespace — explicit registry (``ghcr.io/foo``),
        # port-qualified registry (``localhost:5000/foo``), explicit
        # namespace (``myorg/foo``), or even the canonical implicit forms
        # (``library/python``, ``docker.io/python``) — and combining
        # those with a header Registry/Namespace is ambiguous.
        #
        # We must NOT route this through ``_parse_docker_image_ref``,
        # because it normalizes ``python``, ``library/python``, and
        # ``docker.io/python`` to the same parsed shape and would lose
        # the explicit-domain signal we need to detect ambiguity here.
        if "/" in from_ref:
            return None

    parsed = _parse_docker_image_ref(from_ref)
    if not parsed:
        return None

    domain = parsed["domain"]
    name = parsed["name"]
    tag = parsed["tag"]
    digest = parsed["digest"]

    if registry:
        clean = registry
        for prefix in ("http://", "https://"):
            if clean.lower().startswith(prefix):
                clean = clean[len(prefix):]
        clean = clean.strip("/").strip()
        if clean:
            domain = clean
            # The implicit ``library/`` namespace only applies on
            # docker.io — drop it when we override the domain.
            if name.startswith("library/"):
                name = name[len("library/"):]

    if namespace:
        ns = namespace.strip().strip("/")
        if ns:
            base = name[len("library/"):] if name.startswith("library/") else name
            name = f"{ns}/{base}"

    if digest:
        return f"{domain}/{name}@{digest}"
    if tag is None:
        tag = "latest"
    return f"{domain}/{name}:{tag}"


def _base_sif_cache_filename(canonical: str) -> str:
    """Return a stable filename for caching ``canonical``'s pulled SIF.

    Format: ``<friendly>_<sha16>.sif`` where ``friendly`` is a sanitized
    suffix of the canonical ref (for human debuggability) and ``sha16``
    is the first 16 hex chars of SHA-256(canonical) (for collision
    safety across registries / digest pins / tags).
    """
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    # Sanitize: keep alnum, dash, dot, underscore.  Replace others with _.
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", canonical)
    if len(safe) > 80:
        safe = safe[-80:]
    return f"{safe}_{digest}.sif"


def _can_rewrite_def_to_localimage(text: str) -> tuple[bool, str | None]:
    """Decide whether a container.def is safe to rewrite to localimage.

    Returns ``(safe, canonical_ref)``. ``safe`` is True only when the
    def is a single-stage docker bootstrap whose base ref we can fully
    canonicalize. ``canonical_ref`` is the resolved canonical reference
    when safe, otherwise None.
    """
    header_lines, _body = _split_def_header_and_body(text)
    parsed = _parse_def_header_block(header_lines)
    if not parsed:
        return False, None
    if parsed["bootstrap_count"] != 1:
        return False, None  # multi-stage or no bootstrap
    if parsed["bootstrap"] != "docker":
        return False, None
    if parsed["stage"]:
        return False, None
    if parsed["extra_keys"]:
        return False, None  # unknown header key — be conservative
    if not parsed["from"]:
        return False, None
    canonical = _canonical_docker_ref(
        parsed["from"], parsed["registry"], parsed["namespace"],
    )
    if not canonical:
        return False, None
    return True, canonical


def _rewrite_def_to_localimage(text: str, sif_path: Path) -> str | None:
    """Rewrite a single-stage docker def to use ``Bootstrap: localimage``.

    Replaces the Bootstrap line, the From line, and removes any
    Registry/Namespace lines (irrelevant for localimage). Adds a comment
    annotating the rewrite. Body sections (``%files``, ``%post``, …) are
    left untouched.

    Returns the rewritten text, or None if the rewrite isn't safe.
    """
    safe, canonical = _can_rewrite_def_to_localimage(text)
    if not safe:
        return None

    header_lines, body_lines = _split_def_header_and_body(text)
    new_header: list[str] = [
        f"# Rewritten by swarm base SIF cache from: docker://{canonical}",
        "Bootstrap: localimage",
        f"From: {sif_path}",
    ]
    # Preserve any pure-comment lines that came BEFORE the first header
    # directive, since they're often agent-authored documentation.
    leading_comments: list[str] = []
    for raw in header_lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            leading_comments.append(raw)
            continue
        break  # first non-comment header line — stop preserving

    out_lines: list[str] = []
    out_lines.extend(leading_comments)
    out_lines.extend(new_header)
    if body_lines:
        out_lines.append("")  # blank line between header and body
        out_lines.extend(body_lines)
    rewritten = "\n".join(out_lines)
    if not rewritten.endswith("\n"):
        rewritten += "\n"
    return rewritten


def _env_hashes(workspace: Path) -> tuple[str | None, str | None]:
    """Return current hashes for environment files."""
    env_dir = workspace / "environment"
    return (
        _hash_optional_file(env_dir / "container.def"),
        _hash_optional_file(env_dir / "setup.sh"),
    )


def _fill_manifest(
    manifest_path: Path,
    job: "PaperJob",
    worker_config: Any,
    trajectory_stats: dict | None = None,
) -> None:
    """Write manifest.json with all swarm-known metadata.

    Called at the end of each paper run.  The swarm owns this file
    entirely — the agent does not need to touch it.

    On continuation runs the manifest is rebuilt from scratch (the file
    is deleted before the agent starts), but previous-run stats are
    carried forward via ``job`` fields and the ``agents`` list is loaded
    from the swarm paper directory so per-run breakdowns are preserved.
    """
    try:
        workspace = manifest_path.parent

        # Detect which workspace components exist
        has = lambda p: (workspace / p).exists()  # noqa: E731

        # --- Collect GPU job history from workspace ---
        gpu_jobs: list[dict] = []
        this_run_cpu_hours = 0.0
        jobs_dir = workspace / "jobs"
        if jobs_dir.is_dir():
            for run_dir in sorted(jobs_dir.iterdir()):
                meta_file = run_dir / "meta.json"
                if meta_file.exists():
                    try:
                        jm = json.loads(meta_file.read_text())
                        duration = None
                        if jm.get("started_at") and jm.get("completed_at"):
                            duration = round(jm["completed_at"] - jm["started_at"], 1)
                        num_cpus = jm.get("num_cpus", 0) or 0
                        if duration and num_cpus:
                            this_run_cpu_hours += num_cpus * duration / 3600
                        gpu_jobs.append({
                            "run_id": jm.get("run_id"),
                            "command": jm.get("command", ""),
                            "num_gpus": jm.get("num_gpus", 0),
                            "num_cpus": num_cpus,
                            "resource": jm.get("actual_resource") or jm.get("resource"),
                            "hive_job_id": jm.get("hive_job_id"),
                            "status": jm.get("status"),
                            "returncode": jm.get("returncode"),
                            "duration_sec": duration,
                            "submitted_at": jm.get("submitted_at"),
                            "started_at": jm.get("started_at"),
                            "completed_at": jm.get("completed_at"),
                        })
                    except Exception:
                        pass

        # --- Build this run's agent entry ---
        effective_model = getattr(worker_config, "actual_model", None) or worker_config.model
        # Determine role from workflow step config
        step_role = "reproducer"
        if job.workflow:
            wf_steps_list = load_config().workflows.get(job.workflow, [])
            if job.workflow_step < len(wf_steps_list):
                step_role = wf_steps_list[job.workflow_step].role

        this_run: dict[str, Any] = {
            "scaffold": worker_config.scaffold,
            "model": effective_model,
            "profile": getattr(worker_config, "profile", None),
            "effort": getattr(worker_config, "effort", None),
            "role": step_role,
            "is_iteration": job.is_iteration,
            "iteration": job.iteration,
            "workflow": job.workflow,
            "workflow_step": job.workflow_step,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "duration_sec": round(max(0.0, (job.finished_at or 0) - (job.started_at or 0) - job.blocked_sec), 1),
            "blocked_sec": round(job.blocked_sec, 1),
            "status": job.status,
            "error": job.error,
            "milestone": job.milestone,
            "gpu_budget": job.gpu_budget,
            "gpu_hours_used": round(job.gpu_hours_used, 3),
            "total_gpu_hours_used": round(job.total_gpu_hours_used, 3),
            "cpu_hours_used": round(this_run_cpu_hours, 3),
            "time_breakdown": dict(job.worker_status_totals) if job.worker_status_totals else {},
        }
        # Always write all stat fields (null if unavailable for this scaffold)
        _ALL_STAT_KEYS = [
            "cost_usd", "input_tokens", "output_tokens",
            "cached_input_tokens", "cache_creation_tokens",
            "reasoning_tokens", "total_input_tokens",
            "num_turns", "duration_api_ms",
        ]
        ts = trajectory_stats or {}
        for key in _ALL_STAT_KEYS:
            this_run[key] = ts.get(key)  # None if missing or unavailable
        this_run["tool_uses"] = ts.get("tool_uses", {})
        this_run["gpu_jobs"] = gpu_jobs

        # Capture Progress Log from MILESTONES.md before it gets wiped
        milestones_path = workspace / "MILESTONES.md"
        if milestones_path.exists():
            content = milestones_path.read_text()
            # Extract the Progress Log section
            progress_log = ""
            in_log = False
            for line in content.splitlines():
                if line.strip().startswith("## Progress Log"):
                    in_log = True
                    continue
                if in_log and line.strip().startswith("## "):
                    break
                if in_log:
                    progress_log += line + "\n"
            this_run["progress_log"] = progress_log.strip()

        # --- Load previous agent entries (if any) ---
        # The manifest is deleted from the workspace before each run,
        # but we keep a copy in the swarm paper directory.
        prev_agents: list[dict] = []
        swarm_manifest = PAPERS_DIR / job.paper.arxiv_id / "manifest.json"
        if swarm_manifest.exists():
            try:
                prev = json.loads(swarm_manifest.read_text())
                prev_agents = prev.get("agents", [])
            except Exception:
                pass

        all_agents = prev_agents + [this_run]

        # --- Aggregate totals across all runs ---
        def _sum(key: str) -> float | None:
            vals = [a[key] for a in all_agents if a.get(key) is not None]
            return round(sum(vals), 3) if vals else None

        # Total active duration = sum of per-run active durations (excludes blocked time)
        total_duration = _sum("duration_sec")

        manifest = {
            "paper": {
                "arxiv_id": job.paper.arxiv_id,
                "title": job.paper.title or "",
                "date": job.date,
                "topics": job.topics,
            },
            "reproduction": {
                "milestone": job.milestone,
                "status": job.status,
                "error": job.error,
                "scaffold": worker_config.scaffold,
                "model": effective_model,
                "profile": getattr(worker_config, "profile", None),
                "effort": getattr(worker_config, "effort", None),
                "gpu_hours_used": round(job.total_gpu_hours_used, 3),
                "current_iteration_gpu_budget": job.gpu_budget,
                "current_iteration_gpu_hours_used": round(job.gpu_hours_used, 3),
                "total_gpu_hours_used": round(job.total_gpu_hours_used, 3),
                "current_iteration_cpu_hours_used": round(this_run_cpu_hours, 3),
                "total_cpu_hours_used": round(
                    sum(a.get("cpu_hours_used", 0) or 0 for a in all_agents), 3
                ),
                "gpu_budget": job.gpu_budget,
                "duration_sec": total_duration,
                "cost_usd": _sum("cost_usd"),
                "input_tokens": _sum("input_tokens"),
                "output_tokens": _sum("output_tokens"),
                "cached_input_tokens": _sum("cached_input_tokens"),
                "cache_creation_tokens": _sum("cache_creation_tokens"),
                "reasoning_tokens": _sum("reasoning_tokens"),
                "total_input_tokens": _sum("total_input_tokens"),
                "num_turns": _sum("num_turns"),
                "duration_api_ms": _sum("duration_api_ms"),
                "num_runs": len(all_agents),
                "iteration_count": sum(1 for a in all_agents if a.get("is_iteration")),
                "workflow": job.workflow,
                "workflow_step": job.workflow_step,
                "total_gpu_jobs": sum(len(a.get("gpu_jobs", [])) for a in all_agents),
                "time_breakdown": job.worker_status_totals,
            },
            "agents": all_agents,
            "components": {
                "has_container": has("environment/container.sif") or has("environment/container.def"),
                "has_method_code": has("method") and any((workspace / "method").iterdir()),
                "has_evaluation": has("scripts/evaluate.sh"),
                "has_download_script": has("scripts/download.sh"),
                "has_reproduce_script": has("scripts/reproduce.sh"),
                "has_method_script": has("scripts/method.sh"),
                "has_baseline": has("scripts/baseline.sh"),
                "has_reference": has("scoring/reference.json"),
                "has_data": has("data") and any((workspace / "data").iterdir()),
            },
        }

        # Write to workspace (for git push) and swarm paper dir (for accumulation)
        manifest_text = json.dumps(manifest, indent=2) + "\n"
        manifest_path.write_text(manifest_text)
        swarm_manifest.parent.mkdir(parents=True, exist_ok=True)
        swarm_manifest.write_text(manifest_text)
    except Exception as e:
        log.warning("Failed to write manifest.json: %s", e)


def _snapshot_setup_sh(workspace: Path, iteration: int) -> None:
    """Save a snapshot of setup.sh before an iteration runs.

    Snapshots are stored in ``.swarm/papers/<id>/setup_snapshots/`` — outside
    the workspace, so agents can't delete them.  Safe to call multiple times
    for the same iteration (overwrites the snapshot).
    """
    setup_sh = workspace / "environment" / "setup.sh"
    if not setup_sh.exists():
        return
    paper_id = workspace.name
    snap_dir = PAPERS_DIR / paper_id / "setup_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    dest = snap_dir / f"setup_iter{iteration:03d}.sh"
    try:
        shutil.copy2(str(setup_sh), str(dest))
    except OSError as e:
        log.warning("Failed to snapshot setup.sh for %s iter %d: %s",
                    paper_id, iteration, e)


_PIP_INSTALL_RE = None  # lazy-compiled below
_PKG_SPEC_RE = None
_PIP_KEYWORDS = frozenset({
    "pip", "uv", "install", "python", "python3", "python3.9", "python3.10",
    "python3.11", "python3.12", "python3.13", "-m",
    "--system", "--break-system-packages", "--user", "--upgrade", "-U",
    "--no-deps", "--no-cache-dir", "--no-cache", "--quiet", "-q", "--verbose",
    "-v", "--force-reinstall", "--pre", "--editable", "-e",
    "--index-url", "--extra-index-url", "--find-links", "-f",
    "--trusted-host", "--no-build-isolation", "--no-binary", "--only-binary",
    "--require-hashes", "--constraint", "-c", "--requirement", "-r",
    "--target", "-t", "--root", "--prefix", "--src", "--compile",
    "--no-compile", "--no-warn-script-location",
})


def _parse_packages_from_setup(text: str) -> dict[str, str]:
    """Extract package==version pairs from setup.sh content.

    Only parses tokens that sit inside an actual ``pip install`` /
    ``uv pip install`` command (including their line-continuation tails).
    Stops at shell operators (``&&``, ``||``, ``;``, ``|``, ``>``, ``<``)
    so text in an ``echo`` or the RHS of a pipe is never mistaken for a
    package.  Tokens must start with a letter or underscore — digits at
    the start are rejected.
    """
    import re as _re
    global _PIP_INSTALL_RE, _PKG_SPEC_RE
    if _PIP_INSTALL_RE is None:
        _PIP_INSTALL_RE = _re.compile(
            r'(?:^|[\s;&|`(])'          # preceded by whitespace/separator/start
            r'(?:python\d*\s+-m\s+)?'   # optional "python -m "
            r'(?:uv\s+)?'               # optional "uv "
            r'pip\s+install\b'
        )
        # Package spec: name[extras][op version]
        #   ^name is [A-Za-z_][A-Za-z0-9_.-]*
        #   optional [extras]
        #   optional op: == >= <= > < ~= != ===
        #   optional version (non-space, non-; non-shell-op)
        _PKG_SPEC_RE = _re.compile(
            r"^([A-Za-z_][A-Za-z0-9_.\-]*)"        # name
            r"(?:\[[^\]]*\])?"                      # optional [extras]
            r"(?:(==|>=|<=|~=|!=|===|>|<)"         # op
            r"([^\s;&|<>]+))?"                      # version
        )

    packages: dict[str, str] = {}

    def _emit_tokens(segment: str) -> None:
        # Stop at shell control operators — anything after them is a
        # different command (echo, &&, ||, etc.), not pip args.
        for stopper in ("&&", "||", ";", " | ", "|&", "`", "$(", "2>", "1>", ">>", ">"):
            idx = segment.find(stopper)
            if idx != -1:
                segment = segment[:idx]
        for raw in segment.split():
            # Strip matching quotes; backslash-escaped spaces fall apart
            # naturally via split(), we don't attempt to undo them.
            token = raw.strip("'\"")
            if not token or token.startswith("-"):
                continue
            if token in _PIP_KEYWORDS:
                continue
            # URLs, paths, VCS specs, or anything with a host-like part —
            # real package names never contain these characters.
            if any(c in token for c in (":", "/", "@")):
                continue
            m = _PKG_SPEC_RE.match(token)
            if not m:
                continue
            name = m.group(1).lower().replace("-", "_")
            if m.group(2) == "==" and m.group(3):
                packages[name] = m.group(3)
            elif name not in packages:
                packages[name] = ""

    in_pip = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        # Strip trailing line continuation for processing; remember if it was there
        had_cont = line.endswith("\\")
        body = line[:-1].rstrip() if had_cont else line.rstrip()
        stripped = body.strip()
        if not stripped or stripped.startswith("#"):
            if not had_cont:
                in_pip = False
            continue
        if not in_pip:
            m = _PIP_INSTALL_RE.search(body)
            if m:
                in_pip = True
                # Parse tokens after the `install` keyword
                tail = body[m.end():]
                _emit_tokens(tail)
        else:
            _emit_tokens(stripped)
        if not had_cont:
            in_pip = False
    return packages


def _collect_setup_history_packages(workspace: Path) -> dict[str, str]:
    """Collect packages from all setup.sh snapshots, latest iteration wins."""
    paper_id = workspace.name
    snap_dir = PAPERS_DIR / paper_id / "setup_snapshots"
    packages: dict[str, str] = {}
    if not snap_dir.is_dir():
        # No snapshots — read current setup.sh as fallback
        setup_sh = workspace / "environment" / "setup.sh"
        if setup_sh.exists():
            packages = _parse_packages_from_setup(setup_sh.read_text())
        return packages
    # Process snapshots in order (iter001, iter002, ...) — later overwrites earlier
    for snap in sorted(snap_dir.iterdir()):
        if not snap.name.endswith(".sh"):
            continue
        try:
            snap_pkgs = _parse_packages_from_setup(snap.read_text())
            packages.update(snap_pkgs)  # later iteration wins
        except OSError:
            continue
    return packages


def _scan_distinfo_packages(
    workspace: Path,
    fallback_sif: str | None = None,
) -> dict[str, str]:
    """Scan the workspace + overlay for all pip-installed packages via .dist-info.

    Mounts the overlay (if present) via apptainer and runs ``find`` + ``cat``
    to discover every ``*.dist-info/METADATA`` file.  Extracts ``Name`` +
    ``Version`` from each.  This finds packages regardless of which Python
    interpreter, site-packages path, or ``--target`` the agent used.

    The overlay is mounted on the paper's own ``container.sif``; when that
    file is missing, ``fallback_sif`` (the agent SIF, ``worker.container_sif``)
    is used instead. Falls back to a direct filesystem walk, which cannot
    see overlay contents, if apptainer, a SIF or the overlay is unavailable.

    Returns dict of ``{normalised_name: version}``.
    """
    env_dir = workspace / "environment"
    sif = env_dir / "container.sif"
    overlay = env_dir / "overlay.img"

    if not sif.exists() and fallback_sif and Path(fallback_sif).exists():
        sif = Path(fallback_sif)

    metadata_text = ""
    if sif.exists() and overlay.exists() and not overlay.is_dir():
        try:
            result = subprocess.run(
                [
                    "apptainer", "exec",
                    "--contain", "--no-mount", "bind-paths",
                    "--overlay", f"{overlay}:ro",
                    "--bind", f"{workspace}:/home/user",
                    str(sif),
                    "bash", "-c",
                    "find /home/user -name '*.dist-info' -type d "
                    "-exec test -f {}/METADATA ';' "
                    "-exec echo '===DISTINFO===' ';' "
                    "-exec head -20 {}/METADATA ';' "
                    "2>/dev/null",
                ],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                metadata_text = result.stdout
        except Exception as e:
            log.warning("dist-info scan via apptainer failed for %s: %s",
                        workspace.name, e)

    if not metadata_text:
        # Fallback: direct filesystem walk (won't see overlay contents)
        parts = []
        try:
            for dist_info in workspace.rglob("*.dist-info"):
                if not dist_info.is_dir():
                    continue
                meta = dist_info / "METADATA"
                if meta.is_file():
                    try:
                        parts.append("===DISTINFO===")
                        lines = meta.read_text(errors="replace").splitlines()[:20]
                        parts.extend(lines)
                    except OSError:
                        pass
        except OSError:
            pass
        metadata_text = "\n".join(parts)

    if not metadata_text:
        return {}

    packages: dict[str, str] = {}
    name = version = None
    for line in metadata_text.splitlines():
        if line == "===DISTINFO===":
            if name and version:
                packages[name.lower().replace("-", "_")] = version
            name = version = None
            continue
        if line.startswith("Name: "):
            name = line[6:].strip()
        elif line.startswith("Version: "):
            version = line[9:].strip()
    if name and version:
        packages[name.lower().replace("-", "_")] = version

    return packages


def _freeze_overlay_into_containerdef(
    workspace: Path,
    container_def: Path,
    setup_sh: Path,
    *,
    force: bool = False,
    fallback_sif: str | None = None,
) -> None:
    """Build container.final.def with the complete package list.

    Combines two sources:
      1. **dist-info scan** — packages actually installed in the overlay
      2. **setup.sh history** — packages declared across all iterations

    Each package is tagged with its source for diagnostics:
      - ``both``    — found in overlay AND setup.sh history (high confidence)
      - ``overlay`` — found in overlay only (agent installed outside setup.sh)
      - ``setup``   — declared in setup.sh only (may have been installed
                       ephemerally or the overlay scan missed it)

    Writes ``environment/container.final.def`` (never touches container.def).
    Atomic write via temp file + rename for crash safety.

    ``fallback_sif`` is passed to :func:`_scan_distinfo_packages`.
    """
    if not container_def.exists():
        return

    final_def = container_def.parent / "container.final.def"

    # Skip if already written (idempotent) unless forced
    if final_def.exists() and not force:
        try:
            content = final_def.read_text()
            if "Frozen environment" in content:
                return
        except OSError:
            pass

    # Source 1: dist-info overlay scan
    distinfo_pkgs = _scan_distinfo_packages(workspace, fallback_sif=fallback_sif)

    # Source 2: setup.sh history
    setup_pkgs = _collect_setup_history_packages(workspace)

    # Merge: overlay wins for version, setup.sh fills gaps
    all_names = set(distinfo_pkgs.keys()) | set(setup_pkgs.keys())

    if not all_names:
        # No packages from either source — copy container.def as-is so
        # container.final.def always exists for every done paper.
        tmp_path = final_def.parent / f".container.final.def.tmp.{os.getpid()}"
        try:
            shutil.copy2(str(container_def), str(tmp_path))
            tmp_path.rename(final_def)
            log.info("No packages found for %s — copied container.def as container.final.def",
                     workspace.name)
        except OSError as e:
            log.warning("Failed to copy container.def for %s: %s", workspace.name, e)
            tmp_path.unlink(missing_ok=True)
        return

    merged: list[tuple[str, str, str]] = []  # (name==ver, source, name)
    for name in sorted(all_names):
        in_overlay = name in distinfo_pkgs
        in_setup = name in setup_pkgs
        if in_overlay and in_setup:
            source = "both"
            version = distinfo_pkgs[name]
        elif in_overlay:
            source = "overlay"
            version = distinfo_pkgs[name]
        else:
            source = "setup"
            version = setup_pkgs[name]
        spec = f"{name}=={version}" if version else name
        merged.append((spec, source, name))

    counts = {"both": 0, "overlay": 0, "setup": 0}
    for _, source, _ in merged:
        counts[source] += 1
    log.info("Froze %d packages for %s: %d both, %d overlay-only, %d setup-only",
             len(merged), workspace.name,
             counts["both"], counts["overlay"], counts["setup"])

    # Build container.final.def from container.def + frozen packages
    lines = container_def.read_text().splitlines()

    # Drop any package block a previous freeze or merge left in the def,
    # so the frozen block is never duplicated.
    clean_lines = []
    skip = False
    for line in lines:
        if re.search(r"# --- (Merged from setup\.sh|Frozen from overlay|Frozen environment)", line):
            skip = True
            continue
        if skip and line.strip().startswith("%"):
            skip = False
        if skip:
            continue
        clean_lines.append(line)
    lines = clean_lines

    # Build injection block
    injection = [
        "",
        f"    # --- Frozen environment ({len(merged)} packages) ---",
        f"    # Sources: {counts['both']} both, {counts['overlay']} overlay-only, {counts['setup']} setup-only",
        "    pip install --break-system-packages uv 2>/dev/null || pip install uv 2>/dev/null || true",
        "    export UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1",
        "    uv pip install --system \\",
    ]
    # NOTE: bash only treats `\` as a line continuation when it's the
    # very last character on the line. An inline `# source` comment
    # after `\` would turn it into an escaped space and truncate the
    # argument list, so source tags live in the header summary above.
    for i, (spec, _source, _) in enumerate(merged):
        suffix = " \\" if i < len(merged) - 1 else ""
        injection.append(f"        '{spec}'{suffix}")
    injection.append("")

    # Find insertion point: end of %post (or create %post if missing)
    in_post = False
    insert_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped == "%post":
            in_post = True
            continue
        if in_post and stripped.startswith("%"):
            insert_idx = i
            break
    if insert_idx is None and in_post:
        insert_idx = len(lines)
    if insert_idx is None:
        # No %post section — create one before %environment or at end
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("%environment"):
                insert_idx = i
                break
        if insert_idx is None:
            insert_idx = len(lines)
        lines.insert(insert_idx, "\n%post")
        insert_idx += 1

    lines[insert_idx:insert_idx] = injection
    final_content = "\n".join(lines) + "\n"

    # Atomic write: temp file + rename
    tmp_path = final_def.parent / f".container.final.def.tmp.{os.getpid()}"
    try:
        tmp_path.write_text(final_content)
        tmp_path.rename(final_def)
    except OSError as e:
        log.warning("Failed to write container.final.def for %s: %s",
                    workspace.name, e)
        tmp_path.unlink(missing_ok=True)


def _git_env() -> dict:
    """Environment for scheduler-owned git commits (author "swarm").

    Agent commits carry their own identity, set by the Worker.
    """
    return {
        **os.environ,
        "GIT_AUTHOR_NAME": "swarm",
        "GIT_COMMITTER_NAME": "swarm",
        "GIT_AUTHOR_EMAIL": "swarm@reproducer",
        "GIT_COMMITTER_EMAIL": "swarm@reproducer",
    }


# Entries that must be in .gitignore to keep multi-GB files, credentials
# and Worker scaffolding out of the repository.  The agent maintains
# .gitignore itself and may drop entries, so these are re-added before
# every scheduler commit.
_REQUIRED_GITIGNORE = [
    # Credentials copied into the workspace, and the git wrapper the
    # Worker installs for the agent container.
    ".git-credentials",
    ".claude/",
    ".git-shims/",
    "environment/container.sif",
    "environment/container.squashfs",
    "*.squashfs",
    "environment/overlay.img",
    # Only the downloads cache is gitignored — first-party code and data
    # under data/ MUST stay tracked so fresh clones can import it.
    "data/downloads/",
    "trajectory/",
    "jobs/",
    "tmp/",
    ".venv/",
    ".local/",
    ".python_packages/",
    ".cache/",
    ".codex/",
    "codex/",
    "claude_code/",
    "protocol.md",
    "AGENTS.md",
    "CLAUDE.md",
    # Agent-created package/build dirs that can contain tens of thousands of files.
    # Agents install packages under many different names; catch the common patterns.
    "pkgs/",
    "pkgs-*/",
    "pylib/",
    "pylib[0-9]*/",
    "pylibs/",
    "pylibs_*/",
    "pypkgs/",
    "pip_packages/",
    "pip-packages/",
    "local_pkgs/",
    "local_packages*/",
    "local_torch/",
    "tmp_pip/",
    "tmp_pylib*/",
    "tmp_pkgs/",
    "site-packages/",
    "*.egg-info",
    "node_modules/",
    "*.so",
    "build/",
    "dist/",
    "*.whl",
    "wandb/",
    ".wandb/",
    # Hidden variants
    ".pylib/",
    ".pylib[0-9]*/",
    ".pylibs/",
    ".deps/",
    ".packages/",
    ".local_pkgs/",
    # Rootfs/container extracts agents sometimes create
    "rootfs*/",
    "*_rootfs/",
    # Apptainer/Singularity caches and sandboxes (10s of GB each)
    ".apptainer/",
    ".apptainer_tmp/",
    ".apptainer_tools/",
    ".singularity/",
    "environment/container_sandbox/",
    "environment/*sandbox*/",
    "environment/sandbox*/",
    "environment/rootfs*/",
    "environment/*-rootfs*",
    "environment/*.tar.gz",
    "environment/*.tar",
    "environment/*.tar.xz",
    # Container image layouts (root-level, not under environment/)
    "oci_layout/",
    "oci-image/",
    "oci-images/",
    "docker_*/",
    "docker_*.tar",
    "docker_*.tar.gz",
    "docker_*.tar.xz",
    "*.tar.xz",
    # Pip caches (beyond .cache/)
    ".pip_cache/",
    ".pip-cache/",
    "pip_cache/",
    "pip-cache/",
    # HuggingFace caches (beyond .cache/)
    ".hf_cache/",
    ".hf-cache/",
    "hf_cache/",
    ".huggingface/",
    "huggingface_hub/",
    # Extracted package trees (e.g. torch25_extracted/)
    "*_extracted/",
    # Nvidia/CUDA lib dumps (underscore variants the hook doesn't catch)
    "nvidia_libs/",
    "nvidia-libs/",
    "nvidia_wheels/",
    "nvidia-wheels/",
    "all_libs/",
    # Wheels directories (beyond *.whl)
    "wheels/",
    "python_wheels/",
    # Git bundles (often multi-GB)
    "*.bundle",
    "*.bundle.lock",
    # Core dumps
    "core.[0-9]*",
    "core.*",
]


def _ensure_gitignore(workspace: Path) -> None:
    """Append missing entries to .gitignore so git add -A stays fast.

    Also untrack any already-committed files that match the ignore patterns
    so future pushes don't transfer tens of thousands of package files.
    """
    gi = workspace / ".gitignore"
    existing = gi.read_text() if gi.exists() else ""
    existing_lines = set(existing.splitlines())
    missing = [e for e in _REQUIRED_GITIGNORE if e not in existing_lines]
    if missing:
        with gi.open("a") as f:
            f.write("\n# Auto-added by swarm\n")
            for entry in missing:
                f.write(entry + "\n")
        # Untrack files that are now gitignored but were already committed.
        # Only target the newly-added ignore patterns to avoid a full re-add.
        try:
            for entry in missing:
                pattern = entry.rstrip("/")
                if not pattern or pattern.startswith("*") or pattern.startswith("#"):
                    continue
                target = workspace / pattern
                if target.is_dir():
                    subprocess.run(
                        ["git", "rm", "-r", "--cached", "--quiet", "--ignore-unmatch", pattern],
                        cwd=str(workspace), capture_output=True, timeout=120,
                    )
        except Exception:
            pass  # best-effort


def _ensure_overlay_sanity(workspace: Path) -> None:
    """Remove a corrupted overlay.img (directory instead of file).

    Apptainer's overlay-create flow transiently expands overlay.img into a
    directory tree and seals it back into a file. If that process is
    interrupted (crash, preemption, SIGKILL) the workspace is left with a
    directory at overlay.img. Every subsequent run then fails with
    ``IsADirectoryError`` because the scheduler opens it as a file. Detect
    and remove so the next run recreates the overlay from scratch.
    """
    overlay = workspace / "environment" / "overlay.img"
    if overlay.is_dir():
        try:
            import shutil as _shutil
            _shutil.rmtree(overlay, ignore_errors=True)
            log.warning(
                "Removed corrupted overlay.img directory at %s — "
                "likely leftover from an interrupted apptainer overlay create",
                overlay,
            )
        except Exception as e:
            log.error("Failed to remove corrupted overlay.img at %s: %s", overlay, e)


def _ensure_validate_py(workspace: Path) -> None:
    """Sync workspace/validate.py from the current template.

    validate.py is the workspace-level validator invoked by the scheduler
    after verifier/packager/designer turns. It must stay current with the
    template so rules added after a workspace was created take effect for
    ongoing papers. Overwrites unconditionally (safe — validate.py is
    not agent-authored; it's template-owned).
    """
    template_vp = WORKSPACE_TEMPLATE_DIR / "validate.py"
    if not template_vp.is_file():
        return
    target = workspace / "validate.py"
    try:
        template_bytes = template_vp.read_bytes()
    except OSError:
        return
    if target.is_file():
        try:
            if target.read_bytes() == template_bytes:
                return  # already current
        except OSError:
            pass
    try:
        target.write_bytes(template_bytes)
    except OSError:
        log.warning("Failed to sync validate.py into %s", workspace)


def _compile_readme(workspace: Path) -> None:
    """Compile README.md from briefing files so GitHub displays a nice landing page."""
    briefing = workspace / "briefing"
    if not briefing.is_dir():
        return

    sections: list[str] = []
    for name in ("overview.md", "problem.md", "method.md", "evaluation.md"):
        path = briefing / name
        if path.exists():
            content = path.read_text().strip()
            # Skip template placeholders the agent never filled in
            if "<!-- " in content and content.count("\n") < 15:
                continue
            sections.append(content)

    # Build structured reproduction log from manifest.json
    repro_log = _build_reproduction_log(workspace)
    if repro_log:
        sections.append(repro_log)

    # Extract milestone status and stop justification from MILESTONES.md.
    # Do NOT include the raw file — its Progress Log is already captured
    # in the Reproduction Log via manifest.json agent entries.
    milestones = workspace / "MILESTONES.md"
    if milestones.exists():
        ms_text = milestones.read_text().strip()
        ms_parts = []
        # Extract the "**Current: X**" line
        for line in ms_text.split("\n"):
            if line.strip().startswith("**Current:"):
                ms_parts.append(f"# Reproduction Milestones\n\n{line.strip()}")
                break
        # Extract Stop Justification section
        stop_marker = "## Stop Justification"
        stop_idx = ms_text.find(stop_marker)
        if stop_idx >= 0:
            stop_content = ms_text[stop_idx + len(stop_marker):].strip()
            if not stop_content or stop_content.startswith("## "):
                ms_parts.append(f"{stop_marker}\n- Not provided.")
            else:
                ms_parts.append(f"{stop_marker}\n{stop_content}")
        if ms_parts:
            sections.append("\n\n".join(ms_parts))

    if not sections:
        return

    readme = workspace / "README.md"
    readme.write_text("\n\n---\n\n".join(sections) + "\n")
    log.debug("Compiled README.md for %s", workspace)


def _build_reproduction_log(workspace: Path) -> str:
    """Build a structured reproduction log from manifest.json agents list."""
    # Try workspace manifest first, then swarm paper dir
    manifest_path = workspace / "manifest.json"
    if not manifest_path.exists():
        paper_id = workspace.name
        manifest_path = PAPERS_DIR / paper_id / "manifest.json"
    if not manifest_path.exists():
        return ""

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return ""

    agents = manifest.get("agents", [])
    if not agents:
        return ""

    lines = ["# Reproduction Log\n"]
    for agent in agents:
        iteration = agent.get("iteration", "?")
        model = agent.get("model", "unknown")
        role = agent.get("role", "reproducer")
        milestone = agent.get("milestone", "none")
        status = agent.get("status", "?")
        gpu_hours = agent.get("gpu_hours_used", 0)

        # Working time from time_breakdown (excludes GPU wait time)
        breakdown = agent.get("time_breakdown", {})
        working_sec = breakdown.get("working", 0)
        if working_sec >= 3600:
            working_str = f"{working_sec/3600:.1f}h"
        elif working_sec > 0:
            working_str = f"{working_sec/60:.0f}m"
        else:
            working_str = ""

        # Header
        role_tag = f" ({role})" if role != "reproducer" else ""
        lines.append(f"### Iteration {iteration}: {model}{role_tag}")
        lines.append(f"- **Milestone**: `{milestone}` | **Status**: {status}")
        stats_parts = []
        if working_str:
            stats_parts.append(f"**Working time**: {working_str}")
        stats_parts.append(f"**GPU**: {gpu_hours:.1f}h")
        lines.append(f"- {' | '.join(stats_parts)}")

        # GPU jobs summary
        gpu_jobs = agent.get("gpu_jobs", [])
        if gpu_jobs:
            completed = sum(1 for j in gpu_jobs if j.get("status") == "completed")
            failed = sum(1 for j in gpu_jobs if j.get("status") == "failed")
            lines.append(f"- **Jobs**: {len(gpu_jobs)} total ({completed} completed, {failed} failed)")

        # Progress log from this iteration
        progress_log = agent.get("progress_log", "")
        if progress_log:
            lines.append("")
            # Indent progress log under a collapsible section
            lines.append("<details>")
            lines.append(f"<summary>Progress Log</summary>\n")
            lines.append(progress_log)
            lines.append("\n</details>")

        lines.append("")

    return "\n".join(lines)


# GitHub warns at 50 MB, hard-blocks at 100 MB.  Keep log files well
# under that so the push never fails on file size alone.
_LOG_TRIM_BYTES = 10 * 1024 * 1024  # 10 MB per file
_TRAJECTORY_TRIM_BYTES = 95 * 1024 * 1024  # 95 MB per file
# Per-file trim for jobs/{run_id}/*.log (stdout/stderr from agent-submitted
# jobs). These are tracked for reproducibility — but a long training run can
# write multi-MB logs, so we keep only the tail (where errors/final metrics
# live) to keep commits small.
_JOBS_TRIM_BYTES = 5 * 1024 * 1024  # 5 MB per log file
# Per-blob cap for ordinary workspace files captured by checkpoint
# `git add -A`.  GitHub rejects any object >100 MB; 50 MB leaves
# headroom for zlib/packfile overhead on barely-compressible blobs
# (images, already-compressed CSVs, binaries).  Anything above this
# limit is unstaged before commit — binaries can't be trimmed mid-file
# the way text logs can.
_CHECKPOINT_BLOB_LIMIT_BYTES = 50 * 1024 * 1024


def _build_trimmed_log_blob(path: Path, limit: int) -> bytes | None:
    """Return a bottom-preserving trimmed blob for git, or None if no trim is needed."""
    size = path.stat().st_size
    if size <= limit:
        return None

    with open(path, "rb") as f:
        f.seek(size - limit)
        tail = f.read(limit)

    # Keep the newest output and discard any partial first line from the tail.
    while True:
        nl = tail.find(b"\n")
        if nl >= 0:
            tail = tail[nl + 1:]

        header = f"[trimmed: showing last {len(tail)} of {size} bytes]\n".encode()
        if len(header) + len(tail) <= limit:
            return header + tail

        tail = tail[-max(0, limit - len(header)):]


def _stage_blob_without_touching_worktree(workspace: Path, rel: str, path: Path, blob: bytes) -> None:
    """Stage blob content for *rel* directly in the git index, leaving the file untouched."""
    hash_result = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=str(workspace),
        input=blob,
        capture_output=True,
        timeout=_LOCAL_GIT_TIMEOUT,
    )
    if hash_result.returncode != 0:
        stderr = hash_result.stderr.decode("utf-8", errors="replace")
        log.warning("Git checkpoint: hash-object failed for %s: %s", rel, stderr[:200])
        return

    object_id = hash_result.stdout.decode("utf-8", errors="replace").strip()
    mode = "100755" if (path.stat().st_mode & 0o111) else "100644"
    update_result = subprocess.run(
        ["git", "update-index", "--add", "--cacheinfo", f"{mode},{object_id},{rel}"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=_LOCAL_GIT_TIMEOUT,
    )
    if update_result.returncode != 0:
        log.warning(
            "Git checkpoint: update-index failed for %s: %s",
            rel,
            update_result.stderr[:200],
        )


def _force_add_dir_with_trim(
    workspace: Path,
    dir_path: Path,
    trim_limit: int,
) -> None:
    """Force-add a directory, replacing oversized files with trimmed index blobs."""
    rel_dir = str(dir_path.relative_to(workspace))
    add_result = subprocess.run(
        ["git", "add", "-A", "--force", rel_dir],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=_LOCAL_GIT_TIMEOUT,
    )
    if add_result.returncode != 0:
        log.warning("Git force-add failed for %s: %s", rel_dir, add_result.stderr[:200])
        return

    for path in dir_path.rglob("*"):
        if not path.is_file():
            continue
        trimmed_blob = _build_trimmed_log_blob(path, trim_limit)
        if trimmed_blob is None:
            continue
        rel = str(path.relative_to(workspace))
        _stage_blob_without_touching_worktree(workspace, rel, path, trimmed_blob)


def _unstage_scheduler_paths(workspace: Path, *paths: str) -> None:
    """Remove scheduler-owned paths from the index after a broad git add."""
    reset_result = subprocess.run(
        ["git", "reset", "HEAD", "--", *paths],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=_LOCAL_GIT_TIMEOUT,
    )
    if reset_result.returncode != 0:
        log.warning(
            "Git reset failed for %s: %s",
            ", ".join(paths),
            reset_result.stderr[:200],
        )


def _unstage_oversized_blobs(workspace: Path, *, limit_bytes: int) -> list[str]:
    """Unstage any index entries whose worktree file exceeds *limit_bytes*.

    GitHub rejects any single object over 100 MB, so a checkpoint that
    stages a huge ordinary artifact (e.g. a multi-hundred-MB results
    dump) would poison the branch and break every subsequent push for
    that paper.  We can't safely trim arbitrary binary/tabular files
    the way we trim text logs, so the safe default is to drop the
    oversized *update* from the commit.  The worktree copy is
    untouched — the agent still sees the file locally, it just
    doesn't enter git history as a new version.

    Correctness notes:
    - Symlinks are skipped.  Git stores a symlink as a blob containing
      the link *text* (a few bytes), so even a symlink pointing at a
      100 GB file is a tiny blob.  Following the link with stat()
      would false-positive here.
    - ``git restore --staged`` is used (not ``git rm --cached``)
      because the path may already be tracked in HEAD.  ``git rm
      --cached`` stages a *deletion* for tracked paths, which would
      remove a valid committed file from the branch tip on the next
      commit.  ``git restore --staged`` resets the index entry to
      whatever HEAD has (for tracked paths) or removes the entry
      (for new untracked paths) — correct behavior in both cases.

    Returns the list of unstaged paths (for logging).
    """
    try:
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "-z"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=_LOCAL_GIT_TIMEOUT,
        )
    except Exception as e:
        log.warning("Could not list staged files: %s", e)
        return []
    if staged.returncode != 0:
        return []

    oversized: list[str] = []
    for rel in staged.stdout.split("\0"):
        if not rel:
            continue
        full = workspace / rel
        try:
            # lstat() so symlinks report the link's own size (tiny),
            # not the target's.  Skip symlinks entirely: their stored
            # blob is just the link text and can never be oversized.
            st = os.lstat(full)
        except OSError:
            continue
        import stat as _stat
        if _stat.S_ISLNK(st.st_mode):
            continue
        if not _stat.S_ISREG(st.st_mode):
            continue
        if st.st_size > limit_bytes:
            oversized.append(rel)
    if not oversized:
        return []

    # Choose the unstaging primitive based on whether HEAD exists.
    #
    # With HEAD: `git restore --staged` handles both cases correctly
    #   — tracked paths reset to the HEAD version (so the committed
    #   copy isn't wiped), untracked paths are removed from the index.
    #
    # Without HEAD (an "unborn" repo — e.g. cloned from an empty
    # remote after a failed initial push): `git restore --staged`
    # fails with "fatal: could not resolve HEAD".  Fall back to
    # `git rm --cached` there, which is safe in that state because
    # nothing is tracked in HEAD to accidentally stage a deletion
    # for.  continue_paper's clone-from-empty-remote recovery path
    # reaches this state.
    has_head = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "HEAD"],
        cwd=str(workspace),
        capture_output=True,
        timeout=_LOCAL_GIT_TIMEOUT,
    ).returncode == 0
    if has_head:
        unstage_cmd = ["git", "restore", "--staged", "--", *oversized]
    else:
        unstage_cmd = ["git", "rm", "--cached", "--quiet", "--force", "--", *oversized]
    result = subprocess.run(
        unstage_cmd,
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=_LOCAL_GIT_TIMEOUT,
    )
    if result.returncode != 0:
        log.warning(
            "Could not unstage oversized blobs (%s): %s",
            len(oversized), result.stderr[:200],
        )
        return []
    preview = ", ".join(oversized[:3]) + (", ..." if len(oversized) > 3 else "")
    log.warning(
        "Git checkpoint: unstaged %d oversized file(s) (>%d MB): %s",
        len(oversized), limit_bytes // (1024 * 1024), preview,
    )
    return oversized


def _tracked_pathspecs(workspace: Path, *pathspecs: str) -> set[str]:
    """Return tracked paths matching the given git pathspecs."""
    result = subprocess.run(
        ["git", "ls-files", "--", *pathspecs],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=_LOCAL_GIT_TIMEOUT,
    )
    if result.returncode != 0:
        log.warning("Git ls-files failed for %s: %s", ", ".join(pathspecs), result.stderr[:200])
        return set()
    return {line for line in result.stdout.splitlines() if line}


def _force_stage_path_change(workspace: Path, path: Path) -> None:
    """Force-stage an existing file or a tracked deletion for that path."""
    rel = str(path.relative_to(workspace))
    if not path.exists() and rel not in _tracked_pathspecs(workspace, rel):
        return
    add_result = subprocess.run(
        ["git", "add", "--force", rel],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=_LOCAL_GIT_TIMEOUT,
    )
    if add_result.returncode != 0:
        log.warning("Git add --force failed for %s: %s", rel, add_result.stderr[:200])


def _remove_stale_git_lock(workspace: Path, max_age_sec: int = 300) -> None:
    """Remove .git/index.lock if the lock is older than *max_age_sec*."""
    lock = workspace / ".git" / "index.lock"
    if not lock.exists():
        return
    try:
        age_sec = time.time() - lock.stat().st_mtime
        if age_sec > max_age_sec:
            lock.unlink()
            log.warning(
                "Removed stale .git/index.lock (age %.0fs) in %s",
                age_sec, workspace.name,
            )
    except OSError:
        pass


def _git_checkpoint(
    workspace: Path,
    commit_msg: str,
    log_files: list[str] | None = None,
    *,
    stage_workspace: bool = False,
) -> None:
    """Lightweight git commit+push at job state transitions.

    Unlike _git_commit_and_push this skips README compilation, metadata
    updates, and trajectory handling — it's just a quick snapshot.

    If *log_files* are given, each oversized file is staged as a trimmed
    bottom-preserving blob without modifying the working-tree file, so the
    agent still sees the full original log locally. Only the listed files
    are force-added (they live under jobs/ which is in .gitignore).

    When *stage_workspace* is True, also runs ``git add -A`` before the
    commit so ordinary workspace artifacts written by the job (result
    JSONs, plots, etc.) are preserved in this checkpoint before the
    agent's follow-up turn can edit or delete them.  This is a
    full-tree walk, which is expensive on a shared filesystem (even
    with .gitignore covering the big dirs) — so pass True only on
    job-terminal checkpoints where
    raw outputs need to be captured before agent reaction:
    completed/failed/timeout/cancelled/budget-exhausted.  Transitional
    checkpoints (submission, progress, follow-up, repair) skip the
    walk; the end-of-iteration _git_commit_and_push catches everything.
    """
    if not (workspace / ".git").is_dir():
        log.warning("Git checkpoint skipped (no .git): %s", workspace)
        return
    _remove_stale_git_lock(workspace)
    _ensure_gitignore(workspace)
    _ensure_validate_py(workspace)
    _ensure_overlay_sanity(workspace)
    env = _git_env()
    log.info("Git checkpoint starting: %s — %s", workspace.name, commit_msg)

    try:
        if stage_workspace:
            # Strip nested .git/ from vendored repos the agent may have
            # cloned (method/foo/.git, tmp/bar/.git, …) so their files
            # become normal tracked content instead of dangling gitlinks.
            strip_nested_git_dirs(workspace)
            # Capture ordinary workspace artifacts (result JSONs, plots,
            # files under method/ eval/ etc.) before the agent's
            # follow-up turn can mutate them.  _ensure_gitignore above
            # keeps this walk bounded to non-ignored paths.
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(workspace), capture_output=True, timeout=_LOCAL_GIT_TIMEOUT,
            )
            # Drop scheduler-owned paths so they're only restaged
            # through the trim-aware explicit path below.
            _unstage_scheduler_paths(workspace, "trajectory/")
            # Trim oversized logs under jobs/ (stdout/stderr from agent-submitted
            # jobs) to keep commits small while preserving the tail where errors
            # and final metrics live.
            jobs_dir = workspace / "jobs"
            if jobs_dir.is_dir() and any(jobs_dir.iterdir()):
                _force_add_dir_with_trim(workspace, jobs_dir, _JOBS_TRIM_BYTES)
            # Drop oversized artifacts before commit — GitHub rejects
            # any object >100 MB and we can't safely trim binaries/CSVs.
            _unstage_oversized_blobs(
                workspace, limit_bytes=_CHECKPOINT_BLOB_LIMIT_BYTES,
            )

        # Preserve checkpoint metadata needed for cloned resumes and
        # final manifest reconstruction.
        _force_stage_path_change(workspace, workspace / "trajectory" / "session.json")
        jobs_dir = workspace / "jobs"
        tracked_job_meta = _tracked_pathspecs(workspace, ":(glob)jobs/*/meta.json")
        existing_job_meta = set()
        if jobs_dir.is_dir():
            existing_job_meta = {
                str(meta_file.relative_to(workspace))
                for meta_file in jobs_dir.glob("*/meta.json")
            }
        for rel in sorted(tracked_job_meta | existing_job_meta):
            _force_stage_path_change(workspace, workspace / rel)

        # Force-add log files from this run. Oversized logs are staged as
        # trimmed blobs in the index so the working-tree copy remains intact.
        if log_files:
            for path_str in log_files:
                p = Path(path_str)
                if not p.exists():
                    log.debug("Git checkpoint: log file missing: %s", path_str)
                    continue
                rel = str(p.relative_to(workspace))
                trimmed_blob = _build_trimmed_log_blob(p, _LOG_TRIM_BYTES)
                if trimmed_blob is not None:
                    _stage_blob_without_touching_worktree(workspace, rel, p, trimmed_blob)
                    continue

                r = subprocess.run(
                    ["git", "add", "--force", rel],
                    cwd=str(workspace), capture_output=True, text=True, timeout=_LOCAL_GIT_TIMEOUT,
                )
                if r.returncode != 0:
                    log.warning("Git checkpoint: git add --force %s failed: %s", rel, r.stderr[:200])

        # Commit (allow empty so the message is recorded even if no
        # files changed — serves as a timestamp marker)
        r = subprocess.run(
            ["git", "commit", "--allow-empty", "-m", commit_msg],
            cwd=str(workspace), capture_output=True, text=True, timeout=_LOCAL_GIT_TIMEOUT, env=env,
        )
        if r.returncode != 0:
            log.warning("Git checkpoint: commit failed: %s", r.stderr[:200])
            return

        # Commit only — push is deferred to end-of-iteration (_git_commit_and_push)
        log.info("Git checkpoint committed: %s — %s", workspace.name, commit_msg)
    except Exception as e:
        log.warning("Git checkpoint failed for %s: %s", workspace, e, exc_info=True)


def _git_push_with_upstream_fallback(
    workspace: Path,
    *,
    env: Optional[dict[str, str]] = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Push local HEAD to remote ``main``, regardless of local branch name.

    Using ``HEAD:main`` (not ``git push``) means the remote's ``main`` always
    tracks the latest iteration, even when local is on a side branch
    (``master``, ``clean-main``, etc.) from a prior manual recovery.

    Acquires ``_GIT_PUSH_SEMAPHORE`` so at most 3 pushes run concurrently;
    many simultaneous git-push processes overload a shared filesystem.
    """
    with _GIT_PUSH_SEMAPHORE:
        return subprocess.run(
            ["git", "push", "origin", "HEAD:main"],
            cwd=str(workspace), capture_output=True, text=True,
            timeout=timeout, env=env,
        )


_GH_BIG_BLOB_MARKERS = (
    "exceeds GitHub's file size limit",
    "GH001: Large files detected",
)
_GH_BIG_BLOB_PATH_RE = re.compile(
    r"File (.+?) is [\d.]+\s*[KMG]?B; this exceeds GitHub's file size limit"
)


def _is_big_blob_push_error(stderr: str) -> bool:
    """True if GitHub rejected the push because a file exceeds 100 MB."""
    return any(marker in (stderr or "") for marker in _GH_BIG_BLOB_MARKERS)


def _parse_big_blob_paths(stderr: str) -> list[str]:
    """Extract file paths GitHub flagged as oversized from push stderr."""
    seen: list[str] = []
    for match in _GH_BIG_BLOB_PATH_RE.finditer(stderr or ""):
        p = match.group(1).strip()
        if p and p not in seen:
            seen.append(p)
    return seen


def _filter_repo_strip_paths(
    workspace: Path,
    paths: list[str],
    *,
    timeout: int = 600,
) -> bool:
    """Strip ``paths`` from all commits in ``workspace`` via ``git-filter-repo``.

    GitHub rejects any push carrying a blob >100 MB, and adding the path to
    ``.gitignore`` afterwards does nothing for commits already in history.
    This is the escape hatch: rewrite history to drop the offending paths,
    then force-push.  Preserves the origin URL (``git-filter-repo`` drops it
    as a safety default).  Returns True on success, False otherwise — caller
    should fall back to the normal "stalled" path when we can't clean up.

    ``git-filter-repo`` is an optional tool looked up on ``PATH``; without
    it this returns False.
    """
    if not paths:
        return False
    if shutil.which("git-filter-repo") is None:
        log.warning(
            "git-filter-repo is not on PATH; cannot strip oversized files "
            "from the history of %s", workspace,
        )
        return False
    url_result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=str(workspace), capture_output=True, text=True, timeout=_LOCAL_GIT_TIMEOUT,
    )
    remote_url = url_result.stdout.strip() if url_result.returncode == 0 else ""

    args = ["git-filter-repo", "--force", "--invert-paths"]
    for path in paths:
        args += ["--path", path]

    try:
        result = subprocess.run(
            args, cwd=str(workspace), capture_output=True, text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning(
            "git-filter-repo could not run or timed out for %s: %s", workspace, e
        )
        return False
    if result.returncode != 0:
        log.warning(
            "git-filter-repo failed for %s: %s",
            workspace, (result.stderr or "")[:300],
        )
        return False

    if remote_url:
        subprocess.run(
            ["git", "remote", "remove", "origin"],
            cwd=str(workspace), capture_output=True, timeout=_LOCAL_GIT_TIMEOUT,
        )
        subprocess.run(
            ["git", "remote", "add", "origin", remote_url],
            cwd=str(workspace), capture_output=True, timeout=_LOCAL_GIT_TIMEOUT,
        )
    log.info(
        "git-filter-repo stripped %d oversized path(s) from %s: %s",
        len(paths), workspace.name, ", ".join(paths[:5]),
    )
    return True


def _fix_stale_stop_justification(workspace: Path, milestone: str) -> None:
    """Replace a stale Stop Justification in MILESTONES.md.

    Agents write the Stop Justification during their turn, but later
    iteration agents may reach a higher milestone without updating it.
    This leaves contradictory text (e.g. "Not stopping" when the paper
    is done).  Rewrite the section to reflect the actual final state.
    """
    milestones_path = workspace / "MILESTONES.md"
    if not milestones_path.exists():
        return
    text = milestones_path.read_text()
    marker = "## Stop Justification"
    idx = text.find(marker)
    if idx < 0:
        return
    # Replace everything from the marker onward
    new_justification = f"{marker}\n- Completed at milestone `{milestone}`.\n"
    milestones_path.write_text(text[:idx] + new_justification)


def _git_commit_and_push(
    workspace: Path,
    milestone: str,
    paper: Optional[Any] = None,
    github_org: str = "",
    commit_msg: str | None = None,
) -> bool | None:
    """Commit all changes and push to GitHub remote.

    Called by the scheduler (not the worker) at iteration boundaries so
    the workspace state, logs and trajectory are synced.
    After pushing, updates GitHub repo metadata (description, website, topics).

    Returns True when the push succeeded, False when the commit or push
    failed (including a workspace with no remote), and None when the
    workspace is not a git repository.
    """
    if not (workspace / ".git").is_dir():
        return

    _remove_stale_git_lock(workspace)
    _fix_stale_stop_justification(workspace, milestone)
    _compile_readme(workspace)
    _ensure_gitignore(workspace)
    _ensure_validate_py(workspace)
    _ensure_overlay_sanity(workspace)

    env = _git_env()  # "swarm" — this is a scheduler commit, not an agent commit

    msg = commit_msg or f"Final state (milestone: {milestone})"

    deadline = time.monotonic() + _GIT_FINALIZE_TIMEOUT

    def _ttl(floor: int = 5) -> int:
        """Seconds left until deadline, at least *floor*."""
        return max(floor, int(deadline - time.monotonic()))

    try:
        # Strip nested .git/ from vendored repos so they're tracked as
        # normal files in the parent repo (see strip_nested_git_dirs).
        strip_nested_git_dirs(workspace)
        # Stage ordinary workspace changes first, then drop scheduler-owned
        # directories so they can be re-added with special handling.
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(workspace), capture_output=True, timeout=_ttl(),
        )
        _unstage_scheduler_paths(workspace, "trajectory/")
        # Drop oversized artifacts so the push to GitHub doesn't get
        # rejected on the 100 MB per-object limit.  Checkpoints
        # already apply this guard; applying it here protects the
        # final push from any large artifact introduced between the
        # last terminal checkpoint and now.
        _unstage_oversized_blobs(
            workspace, limit_bytes=_CHECKPOINT_BLOB_LIMIT_BYTES,
        )

        # Force-add scheduler-owned files that .git/info/exclude hides
        # from the agent. Without --force, git add -A skips them.
        for name in ("manifest.json", "README.md"):
            if (workspace / name).exists():
                subprocess.run(
                    ["git", "add", "--force", name],
                    cwd=str(workspace), capture_output=True, timeout=_LOCAL_GIT_TIMEOUT,
                )
        traj_dir = workspace / "trajectory"
        if traj_dir.is_dir() and any(traj_dir.iterdir()):
            _force_add_dir_with_trim(workspace, traj_dir, _TRAJECTORY_TRIM_BYTES)
        jobs_dir = workspace / "jobs"
        if jobs_dir.is_dir() and any(jobs_dir.iterdir()):
            _force_add_dir_with_trim(workspace, jobs_dir, _JOBS_TRIM_BYTES)

        # Commit — always create the end-of-iteration marker
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(workspace), capture_output=True, timeout=_LOCAL_GIT_TIMEOUT,
        )
        allow_empty = ["--allow-empty"] if (result.returncode == 0 and commit_msg) else []
        committed = False
        if result.returncode != 0 or commit_msg:
            committed = subprocess.run(
                ["git", "commit", *allow_empty, "-m", msg],
                cwd=str(workspace), capture_output=True, timeout=_LOCAL_GIT_TIMEOUT,
                env=env,
            ).returncode == 0
        else:
            log.debug("No changes to commit in %s", workspace)

        # Ensure remote exists — recover from failed initial setup
        has_remote = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(workspace), capture_output=True, timeout=_LOCAL_GIT_TIMEOUT,
        ).returncode == 0
        if not has_remote and paper and github_org:
            from swarm.workspace import _github_create_and_push
            _github_create_and_push(workspace, paper, github_org)

        # Push. If it still fails, strip trajectory from the commit,
        # amend, and retry as a last resort. Only a commit made above may
        # be amended: an older HEAD may already be on the remote.
        result = _git_push_with_upstream_fallback(
            workspace, env=env, timeout=_ttl(),
        )
        if result.returncode != 0 and committed and traj_dir.is_dir():
            log.warning("Push failed, retrying without trajectory: %s",
                     result.stderr[:200])
            subprocess.run(
                ["git", "reset", "HEAD~", "--", "trajectory/"],
                cwd=str(workspace), capture_output=True, timeout=_ttl(),
            )
            subprocess.run(
                ["git", "commit", "--amend", "--no-edit"],
                cwd=str(workspace), capture_output=True, timeout=_ttl(),
                env=env,
            )
            result = _git_push_with_upstream_fallback(
                workspace, env=env, timeout=_ttl(),
            )

        # Big-blob escape hatch: if GitHub rejected the push because history
        # carries a file >100 MB (containers, core dumps, model weights that
        # leaked past ``.gitignore`` in an earlier commit), strip the offending
        # paths from all history with ``git-filter-repo`` and retry. Without
        # this, papers silently accumulate poisoned history and never publish
        # again — the paper lands in ``status=stalled`` forever.
        if result.returncode != 0 and _is_big_blob_push_error(result.stderr):
            bad_paths = _parse_big_blob_paths(result.stderr)
            if bad_paths:
                log.warning(
                    "Push rejected by GitHub size limit for %s (paths=%s); "
                    "attempting history rewrite",
                    workspace, bad_paths[:5],
                )
                if _filter_repo_strip_paths(workspace, bad_paths, timeout=_ttl()):
                    result = _git_push_with_upstream_fallback(
                        workspace, env=env, timeout=_ttl(),
                    )

        if result.returncode != 0:
            log.warning("Final push failed for %s: %s", workspace, result.stderr)
            return False
        else:
            log.info("Final push succeeded for %s", workspace)

        # Update GitHub repo metadata (description, website, topics)
        if paper and github_org:
            _update_github_repo_metadata(
                workspace, paper, github_org, milestone,
            )
        return True
    except Exception as e:
        log.warning("Final git commit/push failed for %s: %s", workspace, e)
        return False


def _ensure_iteration_start_commit(
    workspace: Path,
    *,
    iteration: int,
    starting_milestone: str,
    profile: str | None,
    role: str,
    phase: str = "Start",
) -> None:
    """Record iteration start only when the iteration actually begins running."""
    commit_msg = _format_iteration_commit_msg(
        phase,
        iteration,
        starting_milestone,
        profile,
        role,
    )
    try:
        history = subprocess.run(
            ["git", "log", "--format=%s", "-n", "50"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=_LOCAL_GIT_TIMEOUT,
        )
        if history.returncode == 0 and commit_msg in history.stdout.splitlines():
            return
    except Exception:
        pass

    _git_commit_and_push(
        workspace,
        starting_milestone,
        commit_msg=commit_msg,
    )


def _overview_domain_topics(workspace: Path) -> list[str]:
    """Return GitHub topic slugs from the Domain line of briefing/overview.md."""
    overview = workspace / "briefing" / "overview.md"
    topics: list[str] = []
    if overview.exists():
        for line in overview.read_text().splitlines():
            match = re.match(r"^-\s*\*\*Domain:\*\*\s*(.+)", line)
            if match:
                raw = match.group(1).strip()
                # Split on "/" or "," and slugify each part
                for part in re.split(r"[/,]", raw):
                    slug = re.sub(r"[^a-z0-9]+", "-", part.strip().lower()).strip("-")
                    if slug and len(slug) <= 35:
                        topics.append(slug)
                break
    return topics


def _update_github_repo_metadata(
    workspace: Path,
    paper: Any,
    github_org: str,
    milestone: str,
    *,
    extra_topics: list[str] | None = None,
) -> None:
    """Update GitHub repo description, website, and topics after final push.

    Description:  "MILESTONE: Paper Title"
    Website:      https://arxiv.org/abs/<arxiv_id>
    Topics:       parsed from Domain in briefing/overview.md
    """
    import re as _re
    from swarm.workspace import _github_api, _get_github_token, _sanitize_github_topics

    token = _get_github_token()
    if not token:
        return

    repo_name = paper.repo_name()
    full_name = f"{github_org}/{repo_name}"

    # Build description: "MILESTONE: Paper Title"
    title = paper.title or paper.arxiv_id
    description = f"{milestone}: {title}"[:350]

    # Website: arxiv abs link
    homepage = f"https://arxiv.org/abs/{paper.arxiv_id}"

    # Update repo description and website
    status, resp = _github_api("PATCH", f"/repos/{full_name}", token, {
        "description": description,
        "homepage": homepage,
    })
    if status != 200:
        msg = resp.get("message", "") if resp else "unknown"
        log.warning("Failed to update repo metadata for %s: %s", full_name, msg)
        return

    # Parse topics from Domain in overview.md
    topics = _overview_domain_topics(workspace)

    # Add milestone as a prefixed topic
    milestone_topics = {
        f"milestone-{_re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')}"
        for name in list(_MILESTONE_ORDER) + ["not_possible"]
    }
    if milestone and milestone != "none":
        ms_slug = _re.sub(r"[^a-z0-9]+", "-", milestone.lower()).strip("-")
        topics.insert(0, f"milestone-{ms_slug}")

    topics.extend(extra_topics or [])

    # Preserve existing topics set at repo creation (date-*, custom topics)
    # that aren't generated here (domain/milestone are regenerated each time)
    status, resp = _github_api("GET", f"/repos/{full_name}/topics", token)
    if status == 200 and resp:
        for t in resp.get("names", []):
            if t in milestone_topics:
                continue
            if t not in topics:
                topics.append(t)

    if topics:
        topics, rewritten_topics, dropped_topics = _sanitize_github_topics(topics)
        if rewritten_topics:
            log.warning(
                "Rewriting GitHub topics for %s: %s",
                full_name,
                rewritten_topics,
            )
        if dropped_topics:
            log.warning(
                "Dropping invalid GitHub topics for %s: %s",
                full_name,
                dropped_topics,
            )
        status, resp = _github_api("PUT", f"/repos/{full_name}/topics", token, {
            "names": topics[:20],  # GitHub limit: 20 topics
        })
        if status != 200:
            msg = resp.get("message", "") if resp else "unknown"
            log.warning("Failed to set topics for %s: %s", full_name, msg)

    log.info("Updated GitHub metadata for %s: desc=%r, topics=%s",
             full_name, description, topics)
