"""ClaudeCodeWorker — drives a builder agent via the Claude Code CLI."""

from __future__ import annotations

import json
import logging
import os
import random
import select
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from infrastructure.worker.parsing import (
    ClaudeOutput,
    parse_claude_output,
)
from infrastructure.worker.worker import (
    CONTAINER_CLAUDE_TMPDIR,
    WORKSPACE_CLAUDE_TMPDIR,
    Worker,
    redact_secrets,
)

logger = logging.getLogger(__name__)

CONFIG_DEFAULTS = {
    "max_turns": 200,
    "allowed_tools": "Bash,Read,Write,Edit,WebSearch,WebFetch,Glob,Grep",
    "effort": None,       # CLAUDE_CODE_EFFORT_LEVEL
    "compact_token_limit": None,  # CLAUDE_CODE_AUTO_COMPACT_WINDOW; None = CLI default
    "timeout": None,      # subprocess timeout (seconds)
    "env_vars": {},       # extra env vars passed to subprocess
    "action_retries": 2,  # retries when action.yaml is invalid after a turn
    "container_sif": None,  # path to reproducer SIF; None = run without container
}


class ClaudeCodeWorker(Worker):
    """Worker implementation backed by the Claude Code CLI subprocess."""

    # ── abstract hook implementations ──────────────────────────────────────

    @staticmethod
    def _config_defaults() -> dict[str, Any]:
        return CONFIG_DEFAULTS

    @staticmethod
    def _initial_cumulative_stats() -> dict:
        return {
            "cost_usd": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cache_creation_tokens": 0,
            "reasoning_tokens": None,   # not reported by claude code
            "total_input_tokens": 0,
            "num_turns": 0,
            "duration_api_ms": 0,
            "tool_uses": {},
        }

    def _accumulate_stats(self, turn_stats: dict) -> None:
        """Merge a single turn's stats into cumulative totals."""
        for key in ("cost_usd", "input_tokens", "output_tokens", "cached_input_tokens",
                     "cache_creation_tokens", "num_turns", "duration_api_ms"):
            self._cumulative_stats[key] += turn_stats.get(key, 0)
        # Anthropic input_tokens excludes cache, so total = all three
        self._cumulative_stats["total_input_tokens"] = (
            self._cumulative_stats["input_tokens"]
            + self._cumulative_stats["cached_input_tokens"]
            + self._cumulative_stats["cache_creation_tokens"]
        )
        for tool, count in turn_stats.get("tool_uses", {}).items():
            self._cumulative_stats["tool_uses"][tool] = (
                self._cumulative_stats["tool_uses"].get(tool, 0) + count
            )

    # ── workspace setup ─────────────────────────────────────────────────

    def _setup_workspace(self) -> None:
        ws = Path(self._workspace_path)
        (ws / WORKSPACE_CLAUDE_TMPDIR).mkdir(parents=True, exist_ok=True)
        self._write_settings(ws)
        self._copy_credentials(ws)
        self._link_protocol_as_claude_md(ws)

    def _write_settings(self, ws: Path) -> None:
        """Write .claude/settings.json with scoped permissions."""
        _, shared_datasets, shared_models = self._effective_paths()
        settings = {
            "permissions": {
                "allow": [
                    "Bash(*)", "Read(**)", "Write(**)", "Edit(**)",
                    "Glob(**)", "Grep(**)",
                    f"Read(//{shared_datasets}/**)",
                    f"Write(//{shared_datasets}/**)",
                    f"Read(//{shared_models}/**)",
                    f"Write(//{shared_models}/**)",
                ],
                "deny": [
                    "Read(trajectory/**)",
                    "Glob(trajectory/**)",
                    "Grep(trajectory/**)",
                    f"Read({WORKSPACE_CLAUDE_TMPDIR}/**)",
                    f"Glob({WORKSPACE_CLAUDE_TMPDIR}/**)",
                    f"Grep({WORKSPACE_CLAUDE_TMPDIR}/**)",
                ],
            },
        }
        claude_dir = ws / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        (claude_dir / "settings.json").write_text(json.dumps(settings, indent=2))

    def _copy_credentials(self, ws: Path) -> None:
        if not self.config.get("container_sif"):
            return
        host_creds = Path.home() / ".claude" / ".credentials.json"
        if host_creds.exists():
            claude_dir = ws / ".claude"
            claude_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(host_creds), str(claude_dir / ".credentials.json"))

    @staticmethod
    def _link_protocol_as_claude_md(ws: Path) -> None:
        protocol_src = ws / "protocol.md"
        claude_md = ws / "CLAUDE.md"
        if protocol_src.exists() and not claude_md.exists():
            claude_md.symlink_to("protocol.md")

    # ── container / env ───────────────────────────────────────────────────

    def _add_scaffold_container_binds(self, cmd: list[str]) -> None:
        host_claude = shutil.which("claude")
        if host_claude:
            real_claude = str(Path(host_claude).resolve())
            cmd.extend(["--bind", f"{real_claude}:/usr/local/bin/claude"])
        # Bind host CA certs so HTTPS works inside the container.
        # The container has /usr/lib/ssl/cert.pem → /etc/ssl/certs/ca-certificates.crt.
        # We bind the host cert bundle to BOTH paths so Python, curl, wget all work.
        for host_ca in ("/etc/ssl/certs/ca-bundle.crt",
                        "/etc/ssl/certs/ca-certificates.crt",
                        "/etc/pki/tls/certs/ca-bundle.crt"):
            if Path(host_ca).exists():
                cmd.extend([
                    "--bind", f"{host_ca}:/etc/ssl/certs/ca-certificates.crt:ro",
                    "--bind", f"{host_ca}:/usr/lib/ssl/cert.pem:ro",
                ])
                break

    def _configure_env(self, env: dict[str, str]) -> None:
        # Remove Claude Code parent-session markers
        for key in list(env):
            if key == "CLAUDECODE" or key.startswith("CLAUDE_CODE_"):
                del env[key]
        # Strip API keys so CLI uses OAuth credentials
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("OPENAI_API_KEY", None)
        private_tmp = self._workspace_claude_tmpdir()
        # Fix container paths — host paths don't exist inside
        if self.config.get("container_sif"):
            env["TMPDIR"] = CONTAINER_CLAUDE_TMPDIR
            env["CLAUDE_CODE_TMPDIR"] = CONTAINER_CLAUDE_TMPDIR
            env["SSL_CERT_FILE"] = "/usr/lib/ssl/cert.pem"
            env["SSL_CERT_DIR"] = "/usr/lib/ssl/certs"
        elif private_tmp:
            os.makedirs(private_tmp, exist_ok=True)
            env["TMPDIR"] = private_tmp
            env["CLAUDE_CODE_TMPDIR"] = private_tmp
        # Apply effort level
        effort = self.config.get("effort")
        if effort:
            env["CLAUDE_CODE_EFFORT_LEVEL"] = str(effort)
        # Auto-compaction threshold. The Claude Code CLI reads it from this
        # variable; it has no command-line option for it (``-c`` is the
        # short form of ``--continue``).
        compact_limit = self.config.get("compact_token_limit")
        if compact_limit:
            env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(compact_limit)

    # ── CLI invocation ────────────────────────────────────────────────────

    def _build_command(self) -> list[str]:
        sif = self.config.get("container_sif")
        claude_bin = "claude" if sif else (shutil.which("claude") or "claude")
        claude_cmd = [
            claude_bin,
            "--print", "--verbose",
            "--output-format", "stream-json",
            "--model", self.model,
            "--allowedTools", self.config["allowed_tools"],
            # Load only MCP servers that are configured explicitly. The
            # agent authenticates with the operator's Claude login, and
            # without this flag the CLI also loads that account's
            # connectors (mail, chat, calendar, ...) as tools.
            "--strict-mcp-config",
        ]
        # Bound the number of agentic turns of one invocation, so that a
        # single agent turn cannot run for hours.
        max_turns = self.config.get("max_turns")
        if max_turns:
            claude_cmd.extend(["--max-turns", str(max_turns)])
        if self._session_id:
            claude_cmd.extend(["--resume", self._session_id])
        if not sif:
            return claude_cmd
        cmd = self._build_container_cmd(sif)
        cmd.extend(claude_cmd)
        return cmd

    def _raw_output_path(self, turn_num: int) -> Path | None:
        """Return the path for streaming raw output, creating dirs as needed."""
        if not self._workspace_path:
            return None
        traj_dir = Path(self._workspace_path) / "trajectory"
        traj_dir.mkdir(parents=True, exist_ok=True)
        return traj_dir / f"turn_{turn_num:04d}_raw.jsonl"

    def _invoke(self, prompt: str) -> str:
        # Ensure credentials exist for container mode
        if self._workspace_path:
            self._copy_credentials(Path(self._workspace_path))

        self._turn_count += 1
        turn_num = self._turn_count
        max_retries = self.config.get("api_retries", 8)
        base_delay = 30
        max_total_retry_sec = self.config.get("max_total_retry_sec", 900)  # 15 min cap
        retry_start = time.monotonic()
        returncode = -1
        stdout = ""
        stderr = ""
        elapsed = 0.0
        parsed = ClaudeOutput()

        for attempt in range(max_retries + 1):
            cmd = self._build_command()
            env = self._build_env()

            if attempt == 0:
                logger.info("Turn %d: invoking Claude Code CLI: %s", turn_num, " ".join(cmd[:6]))
            else:
                logger.info("Turn %d: retry %d/%d", turn_num, attempt, max_retries)

            start = time.monotonic()
            raw_path = self._raw_output_path(turn_num)

            try:
                proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True,
                    cwd=self._workspace_path, env=env,
                )
                self._active_proc = proc
                # Feed prompt and close stdin so the CLI can proceed
                try:
                    proc.stdin.write(prompt)
                    proc.stdin.close()
                except BrokenPipeError:
                    # CLI process died before we could write the prompt
                    # (e.g., container SIF missing, immediate crash).
                    # Treat as retryable — collect stderr and let retry loop handle it.
                    stderr = proc.stderr.read() if proc.stderr else ""
                    proc.wait(timeout=30)
                    self._active_proc = None
                    logger.warning(
                        "Turn %d: CLI died before accepting prompt (attempt %d/%d): %s",
                        turn_num, attempt + 1, max_retries, stderr[:300],
                    )
                    if attempt < max_retries:
                        delay = base_delay * (attempt + 1) + random.uniform(0, 5)
                        if self._interruptible_sleep(delay):
                            break
                        continue
                    self._status = "error"
                    self._error_message = f"CLI crashed on startup: {stderr[:300]}"
                    # Fall through to break — exhausted retries
                    returncode = proc.returncode or -1
                    stdout = ""
                    break

                # Stream stdout line-by-line to the trajectory file.
                # Two timeouts:
                #   - inactivity: kill if no output for 20 min (dead/hung process)
                #   - wall-clock: kill if total turn time exceeds turn_wall_timeout
                #     (an agent that keeps producing output without finishing)
                inactivity_timeout = 20 * 60  # 20 min with no output → kill
                turn_wall_timeout = self.config.get("turn_wall_timeout", 60 * 60)  # 1 h default
                turn_start = time.monotonic()
                stdout_lines: list[str] = []
                raw_fh = open(raw_path, "w") if raw_path else None
                stdout_fd = proc.stdout.fileno()
                try:
                    buf = b""
                    while True:
                        if time.monotonic() - turn_start > turn_wall_timeout:
                            logger.warning("Agent turn wall-time exceeded (%ds), killing process",
                                           turn_wall_timeout)
                            proc.kill()
                            break
                        ready, _, _ = select.select([stdout_fd], [], [], inactivity_timeout)
                        if not ready:
                            logger.warning("Agent inactivity timeout (%ds), killing process",
                                           inactivity_timeout)
                            proc.kill()
                            break
                        chunk = os.read(stdout_fd, 65536)
                        if not chunk:
                            break  # EOF
                        buf += chunk
                        while b"\n" in buf:
                            line_bytes, buf = buf.split(b"\n", 1)
                            line = line_bytes.decode("utf-8", errors="replace") + "\n"
                            stdout_lines.append(line)
                            if raw_fh:
                                raw_fh.write(redact_secrets(line))
                                raw_fh.flush()
                    if buf:
                        line = buf.decode("utf-8", errors="replace")
                        stdout_lines.append(line)
                        if raw_fh:
                            raw_fh.write(redact_secrets(line))
                            raw_fh.flush()
                finally:
                    if raw_fh:
                        raw_fh.close()

                stderr = proc.stderr.read()
                proc.wait(timeout=self.config.get("timeout"))
                stdout = "".join(stdout_lines)
                returncode = proc.returncode
                self._active_proc = None
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                self._active_proc = None
                stdout = "".join(stdout_lines)
                stderr = ""
                returncode = -1
                logger.warning("Turn %d: CLI timed out", turn_num)
            except Exception as exc:
                self._active_proc = None
                logger.error("Turn %d: CLI failed: %s", turn_num, exc)
                self._status = "error"
                self._error_message = f"CLI failed: {exc}"[:500]
                self._trajectory.append({
                    "turn": turn_num, "error": str(exc),
                    "duration_s": round(time.monotonic() - start, 2),
                })
                return f"Error invoking Claude Code: {exc}"

            elapsed = round(time.monotonic() - start, 2)
            parsed = parse_claude_output(stdout)

            # Detect instant CLI crash: non-zero exit, no output, very fast.
            # This typically means the container failed to launch (e.g., SIF
            # missing, overlay corrupt).  Retry instead of wasting a turn.
            if returncode != 0 and not stdout.strip() and elapsed < 5.0:
                diag = stderr[:300] if stderr else "(no stderr)"
                logger.warning(
                    "Turn %d: CLI exited instantly (rc=%d, %.1fs, attempt %d/%d): %s",
                    turn_num, returncode, elapsed, attempt + 1, max_retries, diag,
                )
                if attempt < max_retries:
                    delay = base_delay * (attempt + 1) + random.uniform(0, 5)
                    if self._interruptible_sleep(delay):
                        break
                    continue
                self._status = "error"
                self._error_message = f"CLI crashed on launch (rc={returncode}): {diag}"[:500]
                break

            if returncode != 0 and parsed.errors and self._is_retryable_error(parsed.errors, stderr):
                if attempt < max_retries:
                    total_elapsed = time.monotonic() - retry_start
                    if total_elapsed >= max_total_retry_sec:
                        logger.warning(
                            "Turn %d: total retry time %.0fs exceeds cap %ds, giving up",
                            turn_num, total_elapsed, max_total_retry_sec,
                        )
                        break
                    delay = base_delay * (attempt + 1) + random.uniform(0, 10)
                    delay = min(delay, max_total_retry_sec - total_elapsed)
                    logger.warning(
                        "Turn %d: transient error (attempt %d/%d), retrying in %ds: %s",
                        turn_num, attempt + 1, max_retries, delay, parsed.errors[0][:200],
                    )
                    if self._interruptible_sleep(delay):
                        logger.info("Turn %d: retry sleep interrupted by cancel", turn_num)
                        break
                    continue
                else:
                    logger.error(
                        "Turn %d: transient error persists after %d retries: %s",
                        turn_num, max_retries, parsed.errors[0][:200],
                    )

            # API key rotation: if hard limit hit and backup keys exist, swap and retry
            if returncode != 0 and parsed.errors and self._is_limit_error(parsed.errors, stderr):
                if self._try_rotate_api_key():
                    logger.warning(
                        "Turn %d: hard API limit, rotated key — retrying",
                        turn_num,
                    )
                    continue
            break

        # Raw output already streamed to file — no need for _save_raw_output
        if parsed.session_id:
            self._session_id = parsed.session_id
            self._save_session_info()
        self._accumulate_stats(parsed.stats)

        if parsed.errors and returncode != 0:
            self._status = "error"
            self._error_message = f"rc={returncode}: {'; '.join(parsed.errors[:3])}"[:500]

        self._trajectory.append({
            "turn": turn_num, "text": parsed.text, "stats": parsed.stats,
            "errors": parsed.errors, "returncode": returncode,
            "duration_s": elapsed, "session_id": parsed.session_id,
        })

        if parsed.errors:
            logger.warning("Turn %d errors: %s", turn_num, parsed.errors)
        if stderr and returncode != 0:
            logger.warning("Turn %d stderr (rc=%d): %s", turn_num, returncode, stderr[:500])

        return parsed.text
