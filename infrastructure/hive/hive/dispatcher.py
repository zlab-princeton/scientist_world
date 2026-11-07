"""Hive dispatcher: Ray head lifecycle, job records, and job dispatch.

Can run from any process on the head node. Job runners are detached Ray
named actors, so every process that connects to the same Ray cluster sees
the same set of running jobs.
"""

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import ray
from ray.util.placement_group import placement_group, remove_placement_group

from .config import (
    HiveConfig,
    ResourceSpec,
    STATE_PATH,
    JOBS_DIR,
    HIVE_DIR,
    SLURM_TRACKING_DIR,
    DRAINING_NODES_DIR,
    ensure_hive_dir,
    get_launch_pending_grace_sec,
    get_ray_tmpdir,
    load_config,
)
from .pool import PoolManager, SlurmJob

log = logging.getLogger(__name__)

_JOB_HEARTBEAT_FILE = "heartbeat.ts"
_JOB_PROCESS_RESULT_FILE = "process_result.json"
_JOB_EVENTS_FILE = "events.log"
_DRAIN_NOTE = "[hive] Preempted for SLURM worker drain before walltime"


def _write_atomic(path: Path, text: str):
    """Atomically replace a small text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def _job_heartbeat_path(job_id: str) -> Path:
    return JOBS_DIR / job_id / _JOB_HEARTBEAT_FILE


def _job_process_result_path(job_id: str) -> Path:
    return JOBS_DIR / job_id / _JOB_PROCESS_RESULT_FILE


def _job_events_path(job_id: str) -> Path:
    return JOBS_DIR / job_id / _JOB_EVENTS_FILE


def _draining_node_path(node_id: str) -> Path:
    return DRAINING_NODES_DIR / f"{node_id}.json"


def mark_node_draining(
    node_id: str,
    *,
    resource_name: str = "",
    slurm_job_id: str = "",
    started_at: Optional[float] = None,
) -> None:
    """Persist that a worker node is intentionally draining before walltime."""
    if not node_id:
        return
    if started_at is None:
        started_at = time.time()
    payload = {
        "node_id": node_id,
        "resource_name": resource_name,
        "slurm_job_id": slurm_job_id,
        "started_at": started_at,
    }
    _write_atomic(_draining_node_path(node_id), json.dumps(payload, indent=2))


def clear_node_draining(node_id: str) -> None:
    """Remove a persisted drain marker."""
    if not node_id:
        return
    try:
        _draining_node_path(node_id).unlink()
    except FileNotFoundError:
        pass


def is_node_draining(node_id: str) -> bool:
    """Return True when Hive has intentionally started draining a worker node."""
    if not node_id:
        return False
    return _draining_node_path(node_id).exists()


def prune_stale_draining_nodes(alive_node_ids: set[str]) -> None:
    """Drop drain markers for Ray nodes that no longer exist."""
    if not DRAINING_NODES_DIR.exists():
        return
    for path in DRAINING_NODES_DIR.glob("*.json"):
        node_id = path.stem
        if node_id in alive_node_ids:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def find_job_actor(job_id: str):
    """Return the detached job actor handle for a job, if it still exists."""
    prefix = f"{JOB_ACTOR_PREFIX}{job_id}"
    try:
        named = ray.util.list_named_actors(all_namespaces=True)
    except Exception:
        return None
    for entry in named:
        name = entry if isinstance(entry, str) else entry.get("name", "")
        if not name.startswith(prefix):
            continue
        try:
            return ray.get_actor(name)
        except Exception:
            return None
    return None


def job_actor_name_exists(job_id: str) -> bool:
    """Return True when a detached actor name is still registered for the job."""
    prefix = f"{JOB_ACTOR_PREFIX}{job_id}"
    try:
        named = ray.util.list_named_actors(all_namespaces=True)
    except Exception:
        return False
    for entry in named:
        name = entry if isinstance(entry, str) else entry.get("name", "")
        if name.startswith(prefix):
            return True
    return False


def clear_job_actor_name(job_id: str, *, timeout_sec: float = 30.0, poll_interval: float = 0.2) -> None:
    """Wait for any stale detached actor name for the job to disappear."""
    prefix = f"{JOB_ACTOR_PREFIX}{job_id}"
    deadline = time.time() + timeout_sec
    while True:
        try:
            named = ray.util.list_named_actors(all_namespaces=True)
        except Exception as exc:
            raise RuntimeError(f"Cannot inspect Ray actors for job {job_id}: {exc}") from exc

        matching_names = []
        for entry in named:
            name = entry if isinstance(entry, str) else entry.get("name", "")
            if name.startswith(prefix):
                matching_names.append(name)

        if not matching_names:
            return

        for name in matching_names:
            try:
                actor = ray.get_actor(name)
            except Exception:
                continue
            try:
                ray.kill(actor)
            except Exception:
                pass

        if time.time() >= deadline:
            raise RuntimeError(f"Timed out waiting for stale actor name {prefix} to disappear")
        time.sleep(poll_interval)


def build_job_bundles(record: "JobRecord") -> list[dict]:
    """Build placement bundles for a job record."""
    if getattr(record, "dispatch_resource", None):
        resource_list = [record.dispatch_resource]
    else:
        resource_list = [r.strip() for r in (record.resource or "").split(",") if r.strip()]
    base = {"CPU": record.num_cpus, "GPU": record.num_gpus}
    if resource_list:
        return [dict(base, **{f"partition:{res_name}": 0.01}) for res_name in resource_list]
    return [base]


def default_job_workdir() -> str:
    """Best-effort default workdir for re-dispatched jobs."""
    try:
        cfg = load_config()
        if cfg.env.working_dir:
            return cfg.env.working_dir
    except Exception:
        pass
    return os.getcwd()


def launch_job_runner(
    record: "JobRecord",
    *,
    workdir: Optional[str] = None,
    dispatch_resource: Optional[str] = None,
) -> tuple[str, Optional["JobRecord"]]:
    """Launch a detached runner actor for an already-persisted queued job."""
    if workdir is None:
        workdir = default_job_workdir()
    lock_record = JobRecord(job_id=record.job_id, command="", num_gpus=1, status="unknown")
    with lock_record._launch_lock():
        current = JobRecord.load(record.job_id)
        if current is None:
            return "missing", None
        if current.status in ("completed", "failed", "cancelled", "timeout", "preempted"):
            return "terminal", current
        if current.status != "queued":
            return "unexpected", current
        if current.runner_launched:
            return "already_launched", current

        recovering_pending = current.runner_launch_pending_since is not None
        if recovering_pending:
            if find_job_actor(current.job_id) is not None:
                return "already_launched", current
            pending_age = time.time() - current.runner_launch_pending_since
            if pending_age < get_launch_pending_grace_sec():
                return "launch_pending", current

        try:
            clear_job_actor_name(current.job_id)
        except Exception:
            if recovering_pending:
                current = JobRecord.clear_runner_launch_pending(current.job_id) or current
            raise

        if recovering_pending:
            current = JobRecord.clear_runner_launch_pending(current.job_id) or current
            if current.status != "queued":
                return "unexpected", current
            if current.runner_launched:
                return "already_launched", current

        if dispatch_resource:
            current.dispatch_resource = dispatch_resource

        # Mark pending BEFORE creating the actor so _has_actor() never
        # sees a window where both runner_launched and
        # runner_launch_pending_since are unset while an actor exists.
        updated = JobRecord.mark_runner_launch_pending(
            current.job_id,
            dispatch_resource=dispatch_resource,
        )

        actor_name = f"{JOB_ACTOR_PREFIX}{current.job_id}"
        try:
            # Include a code version in runtime_env so Ray creates fresh
            # workers when hive code changes. Without this, Ray reuses
            # workers with stale sys.modules from hours/days ago.
            _code_version = str(int(Path(__file__).stat().st_mtime))
            _AsyncJobRunner.options(
                name=actor_name,
                lifetime="detached",
                runtime_env={"env_vars": {
                    "HIVE_DIR": str(HIVE_DIR),
                    "HIVE_CODE_VERSION": _code_version,
                }},
            ).remote(
                current.job_id,
                current.command,
                current.num_gpus,
                current.num_cpus,
                current.memory_mb,
                workdir,
                build_job_bundles(current),
                current.timeout_sec,
            )
        except Exception:
            # remote() failed synchronously — no actor was created.
            # Roll back the pending mark so the job stays dispatchable
            # instead of blocking the queue for the full grace period.
            JobRecord.clear_runner_launch_pending(current.job_id)
            raise
        return "launched", updated or current


def _append_log_line(text: str, message: str) -> str:
    """Append one line to a log blob, preserving existing content."""
    if not message:
        return text
    if not text:
        return message
    if text.endswith("\n"):
        return text + message
    return text + "\n" + message


def append_job_stderr_message(job_id: str, message: str):
    """Append a Hive infrastructure note to the real stderr stream."""
    stderr_path = JOBS_DIR / job_id / "stderr.log"
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with stderr_path.open("ab+") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")
            f.write(message.encode("utf-8"))
            f.write(b"\n")
    except Exception:
        log.warning("Job %s: failed to append stderr note", job_id)


def buffer_job_stderr_message(job_id: str, message: str):
    """Stage a Hive note to append after live stderr streaming finishes."""
    events_path = _job_events_path(job_id)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with events_path.open("ab+") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")
            f.write(message.encode("utf-8"))
            f.write(b"\n")
    except Exception:
        log.warning("Job %s: failed to buffer stderr note", job_id)


def _append_text_block(path: Path, text: str, *, separate_if_nonempty: bool = False):
    """Append UTF-8 text to a file without rewriting existing content."""
    if not text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    try:
        with path.open("ab+") as f:
            if separate_if_nonempty:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size > 0:
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b"\n" and not payload.startswith(b"\n"):
                        f.write(b"\n")
            f.seek(0, os.SEEK_END)
            f.write(payload)
    except Exception:
        log.warning("Failed appending text to %s", path)


def write_job_heartbeat(job_id: str, ts: Optional[float] = None):
    """Persist the last known liveness timestamp for a running job process."""
    if not job_id:
        return
    if ts is None:
        ts = time.time()
    _write_atomic(_job_heartbeat_path(job_id), f"{ts}\n")


def read_job_heartbeat(job_id: str) -> Optional[float]:
    """Read the last known job-process heartbeat timestamp."""
    try:
        return float(_job_heartbeat_path(job_id).read_text().strip())
    except Exception:
        return None


def write_job_process_result(
    job_id: str,
    *,
    completed_at: float,
    returncode: int,
    timed_out: bool,
    attempt: int = 0,
):
    """Persist the subprocess exit state independently of the Ray actor.

    The ``attempt`` field lets readers distinguish this attempt's result
    from a predecessor's stale file (process_result.json is shared
    across attempts for the same job_id).
    """
    if not job_id:
        return
    payload = {
        "completed_at": completed_at,
        "returncode": returncode,
        "timed_out": timed_out,
        "attempt": attempt,
    }
    _write_atomic(
        _job_process_result_path(job_id),
        json.dumps(payload, indent=2),
    )


def read_job_process_result(job_id: str) -> dict:
    """Read the persisted subprocess exit state, if available."""
    try:
        return json.loads(_job_process_result_path(job_id).read_text())
    except Exception:
        return {}


def clear_job_process_result(job_id: str) -> None:
    """Delete the persisted process result file, if present.

    Called on requeue so a retry never inherits its predecessor's
    terminal state through a stale file on disk.
    """
    if not job_id:
        return
    try:
        _job_process_result_path(job_id).unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass  # best-effort; orphan-reaper attempt-match is the real guard


def _build_job_log_preamble(
    stream_name: str,
    *,
    job_id: str,
    command: str,
    workdir: str,
    timeout_sec: int | None,
) -> str:
    """Return a short Hive marker block written at the top of each log."""
    record = JobRecord.load(job_id) if job_id else None
    hostname = socket.gethostname()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    lines = [
        f"[hive] This {stream_name} log is managed by Hive.",
        f"[hive] job_id: {job_id or 'unknown'}",
        f"[hive] host: {hostname}",
        f"[hive] node_id: {getattr(record, 'node_id', '') or 'unknown'}",
        f"[hive] user: {getattr(record, 'user', None) or os.environ.get('USER', 'unknown')}",
        f"[hive] resource: {getattr(record, 'resource', None) or 'any'}",
        f"[hive] gpus: {getattr(record, 'num_gpus', 'unknown')}",
        f"[hive] cpus: {getattr(record, 'num_cpus', 'unknown')}",
        f"[hive] workdir: {workdir}",
        f"[hive] timeout_sec: {timeout_sec if timeout_sec else 'none'}",
        f"[hive] started_at: {started_at}",
        f"[hive] command: {command}",
        "",
    ]
    return "\n".join(lines)


@dataclass
class JobRecord:
    job_id: str
    command: str
    num_gpus: int
    status: str  # queued, running, completed, failed, timeout, preempted
    result: Optional[dict] = None
    submitted_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    stdout: str = ""
    stderr: str = ""
    returncode: Optional[int] = None
    resource: Optional[str] = None
    user: Optional[str] = None
    on_complete: Optional[str] = None
    num_cpus: int = 1
    memory_mb: int = 0
    timeout_sec: Optional[int] = None
    max_retries: int = 0
    attempt: int = 0
    node_id: str = ""
    priority: str = "normal"  # "normal" or "low"
    runner_launched: bool = False
    runner_launch_pending_since: Optional[float] = None
    dispatch_resource: Optional[str] = None
    missing_actor_since: Optional[float] = None
    missing_actor_checks: int = 0
    node_dead_since: Optional[float] = None
    node_dead_checks: int = 0

    def to_dict(self) -> dict:
        """Return metadata dict (excludes stdout/stderr — those live in files)."""
        d = {
            "job_id": self.job_id,
            "command": self.command,
            "num_gpus": self.num_gpus,
            "num_cpus": self.num_cpus,
            "memory_mb": self.memory_mb,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "returncode": self.returncode,
            "resource": self.resource,
        }
        if self.user:
            d["user"] = self.user
        if self.on_complete:
            d["on_complete"] = self.on_complete
        if self.timeout_sec is not None:
            d["timeout_sec"] = self.timeout_sec
        if self.max_retries > 0:
            d["max_retries"] = self.max_retries
        if self.attempt > 0:
            d["attempt"] = self.attempt
        if self.node_id:
            d["node_id"] = self.node_id
        if self.priority != "normal":
            d["priority"] = self.priority
        d["runner_launched"] = self.runner_launched
        if self.runner_launch_pending_since is not None:
            d["runner_launch_pending_since"] = self.runner_launch_pending_since
        if self.dispatch_resource:
            d["dispatch_resource"] = self.dispatch_resource
        if self.missing_actor_since is not None:
            d["missing_actor_since"] = self.missing_actor_since
        if self.missing_actor_checks > 0:
            d["missing_actor_checks"] = self.missing_actor_checks
        if self.node_dead_since is not None:
            d["node_dead_since"] = self.node_dead_since
        if self.node_dead_checks > 0:
            d["node_dead_checks"] = self.node_dead_checks
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JobRecord":
        return cls(
            job_id=d["job_id"],
            command=d["command"],
            num_gpus=d.get("num_gpus", 1),
            status=d.get("status", "unknown"),
            submitted_at=d.get("submitted_at", 0.0),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            stdout=d.get("stdout", ""),
            stderr=d.get("stderr", ""),
            returncode=d.get("returncode"),
            resource=d.get("resource") or d.get("partition"),
            user=d.get("user"),
            on_complete=d.get("on_complete"),
            num_cpus=d.get("num_cpus", 1),
            memory_mb=d.get("memory_mb", 0),
            timeout_sec=d.get("timeout_sec"),
            max_retries=d.get("max_retries", 0),
            attempt=d.get("attempt", 0),
            node_id=d.get("node_id", ""),
            priority=d.get("priority", "normal"),
            runner_launched=d.get(
                "runner_launched",
                d.get("priority", "normal") != "low",
            ),
            runner_launch_pending_since=d.get("runner_launch_pending_since"),
            dispatch_resource=d.get("dispatch_resource"),
            missing_actor_since=d.get("missing_actor_since"),
            missing_actor_checks=d.get("missing_actor_checks", 0),
            node_dead_since=d.get("node_dead_since"),
            node_dead_checks=d.get("node_dead_checks", 0),
        )

    @property
    def job_dir(self) -> Path:
        return JOBS_DIR / self.job_id

    @property
    def stdout_path(self) -> Path:
        return self.job_dir / "stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self.job_dir / "stderr.log"

    @property
    def meta_path(self) -> Path:
        return self.job_dir / "meta.json"

    @property
    def meta_lock_path(self) -> Path:
        return self.job_dir / ".meta.lock"

    @property
    def launch_lock_path(self) -> Path:
        return self.job_dir / ".launch.lock"

    @contextmanager
    def _meta_lock(self):
        import fcntl

        self.job_dir.mkdir(parents=True, exist_ok=True)
        with self.meta_lock_path.open("a+") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _launch_lock(self):
        import fcntl

        self.job_dir.mkdir(parents=True, exist_ok=True)
        with self.launch_lock_path.open("a+") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    def _write_meta_unlocked(self):
        _write_atomic(self.meta_path, json.dumps(self.to_dict(), indent=2))

    def _write_logs_unlocked(self):
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_path.write_text(self.stdout)
        self.stderr_path.write_text(self.stderr)

    @property
    def combined_stderr(self) -> str:
        return self.stderr.rstrip("\n")

    @classmethod
    def _load_meta_only(cls, job_id: str) -> Optional["JobRecord"]:
        meta_path = JOBS_DIR / job_id / "meta.json"
        if meta_path.exists():
            return cls.from_dict(json.loads(meta_path.read_text()))

        legacy_path = JOBS_DIR / f"{job_id}.json"
        if legacy_path.exists():
            return cls.from_dict(json.loads(legacy_path.read_text()))

        return None

    def _merge_legacy_events_for_display(self):
        """Fold buffered/legacy events.log notes into safe log views."""
        if self.status not in ("completed", "failed", "timeout", "cancelled", "preempted"):
            return
        events_path = self.job_dir / "events.log"
        if not events_path.exists():
            return
        try:
            notes = events_path.read_text()
        except Exception:
            return
        if notes:
            self.stderr = _append_log_line(self.stderr, notes)

    @classmethod
    def load_unmerged(cls, job_id: str) -> Optional["JobRecord"]:
        """Load a job record without folding buffered events.log notes into stderr."""
        meta_path = JOBS_DIR / job_id / "meta.json"
        if meta_path.exists():
            record = cls.from_dict(json.loads(meta_path.read_text()))
            stdout_path = JOBS_DIR / job_id / "stdout.log"
            stderr_path = JOBS_DIR / job_id / "stderr.log"
            if stdout_path.exists():
                record.stdout = stdout_path.read_text()
            if stderr_path.exists():
                record.stderr = stderr_path.read_text()
            return record

        legacy_path = JOBS_DIR / f"{job_id}.json"
        if legacy_path.exists():
            return cls.from_dict(json.loads(legacy_path.read_text()))

        return None

    @classmethod
    def _consume_buffered_notes_unlocked(
        cls,
        job_id: str,
        *,
        materialize_to_stderr: bool,
    ) -> str:
        """Drain any staged events.log notes, optionally appending them to stderr.log."""
        events_path = _job_events_path(job_id)
        if not events_path.exists():
            return ""
        try:
            notes = events_path.read_text()
        except Exception:
            return ""
        if materialize_to_stderr:
            if notes:
                _append_text_block(JOBS_DIR / job_id / "stderr.log", notes, separate_if_nonempty=True)
            events_path.unlink(missing_ok=True)
        return notes

    @classmethod
    def materialize_buffered_notes_for_queued_job(cls, job_id: str) -> Optional["JobRecord"]:
        """Append staged notes into stderr.log once a job is safely back in queue."""
        lock_record = cls(job_id=job_id, command="", num_gpus=1, status="unknown")
        with lock_record._meta_lock():
            current = cls._load_meta_only(job_id)
            if current is None:
                return None
            if current.status == "queued":
                cls._consume_buffered_notes_unlocked(job_id, materialize_to_stderr=True)
            return current

    @classmethod
    def _migrate_legacy_layout_unlocked(cls, record: "JobRecord") -> "JobRecord":
        """Materialize legacy single-file logs before metadata-only updates."""
        legacy_path = JOBS_DIR / f"{record.job_id}.json"
        if record.meta_path.exists() or not legacy_path.exists():
            return record
        record._write_logs_unlocked()
        record._write_meta_unlocked()
        return record

    @classmethod
    def _persist_cancelled_unlocked(
        cls,
        current: "JobRecord",
        *,
        now: float,
        note: str = "[hive] Job cancelled",
    ) -> "JobRecord":
        """Persist cancelled state and note without snapshot-rewriting live logs."""
        current = cls._migrate_legacy_layout_unlocked(current)
        was_running = current.status == "running"
        current.status = "cancelled"
        current.completed_at = now
        current.returncode = -15
        current.runner_launched = False
        current.runner_launch_pending_since = None
        current.dispatch_resource = None
        if note:
            if was_running:
                buffer_job_stderr_message(current.job_id, note)
            else:
                append_job_stderr_message(current.job_id, note)
                current.stderr = _append_log_line(current.stderr, note)
        current._write_meta_unlocked()
        return current

    @classmethod
    def mark_runner_launched(cls, job_id: str) -> Optional["JobRecord"]:
        """Persist that a detached runner exists without overwriting newer state."""
        lock_record = cls(job_id=job_id, command="", num_gpus=1, status="unknown")
        with lock_record._meta_lock():
            current = cls._load_meta_only(job_id)
            if current is None:
                return None
            if current.status in ("queued", "running"):
                current.runner_launched = True
                current.runner_launch_pending_since = None
                current._write_meta_unlocked()
            return current

    @classmethod
    def mark_runner_launch_pending(
        cls,
        job_id: str,
        ts: Optional[float] = None,
        dispatch_resource: Optional[str] = None,
    ) -> Optional["JobRecord"]:
        """Persist that Ray accepted a runner launch which is not yet confirmed started."""
        if ts is None:
            ts = time.time()
        lock_record = cls(job_id=job_id, command="", num_gpus=1, status="unknown")
        with lock_record._meta_lock():
            current = cls._load_meta_only(job_id)
            if current is None:
                return None
            if current.status == "queued" and not current.runner_launched:
                current.runner_launch_pending_since = ts
                current.dispatch_resource = dispatch_resource
                current._write_meta_unlocked()
            return current

    @classmethod
    def clear_runner_launch_pending(cls, job_id: str) -> Optional["JobRecord"]:
        """Clear a stale pending-launch marker without touching other state."""
        lock_record = cls(job_id=job_id, command="", num_gpus=1, status="unknown")
        with lock_record._meta_lock():
            current = cls._load_meta_only(job_id)
            if current is None:
                return None
            if current.status == "queued" and not current.runner_launched:
                current.runner_launch_pending_since = None
                current.dispatch_resource = None
                current._write_meta_unlocked()
            return current

    @classmethod
    def activate_runner(cls, job_id: str) -> Optional["JobRecord"]:
        """Mark the runner as started unless the job was already cancelled."""
        lock_record = cls(job_id=job_id, command="", num_gpus=1, status="unknown")
        with lock_record._meta_lock():
            current = cls._load_meta_only(job_id)
            if current is None:
                return None
            if current.status == "queued":
                current.runner_launched = True
                current.runner_launch_pending_since = None
                current._write_meta_unlocked()
            return current

    @classmethod
    def cancel_queued_if_not_started(
        cls, job_id: str, now: Optional[float] = None,
    ) -> tuple[str, Optional["JobRecord"]]:
        """Atomically cancel a queued job only if no runner has started yet."""
        if now is None:
            now = time.time()
        lock_record = cls(job_id=job_id, command="", num_gpus=1, status="unknown")
        with lock_record._meta_lock():
            current = cls._load_meta_only(job_id)
            if current is None:
                return "missing", None
            if current.status in ("completed", "failed", "cancelled", "timeout", "preempted"):
                return "terminal", current
            if current.status != "queued":
                return "needs_ray", current
            if current.runner_launched:
                return "needs_ray", current

            return "cancelled", cls._persist_cancelled_unlocked(current, now=now)

    @classmethod
    def mark_cancelled(
        cls, job_id: str, now: Optional[float] = None,
    ) -> tuple[str, Optional["JobRecord"]]:
        """Persist ``cancelled`` metadata and append a Hive note to stderr."""
        if now is None:
            now = time.time()
        lock_record = cls(job_id=job_id, command="", num_gpus=1, status="unknown")
        with lock_record._meta_lock():
            current = cls._load_meta_only(job_id)
            if current is None:
                return "missing", None
            if current.status in ("completed", "failed", "cancelled", "timeout", "preempted"):
                return "terminal", current

            return "cancelled", cls._persist_cancelled_unlocked(current, now=now)

    @classmethod
    def mark_preempted(
        cls,
        job_id: str,
        *,
        note: str = _DRAIN_NOTE,
    ) -> tuple[str, Optional["JobRecord"]]:
        """Persist a planned preemption so the dashboard can requeue the job."""
        lock_record = cls(job_id=job_id, command="", num_gpus=1, status="unknown")
        with lock_record._meta_lock():
            current = cls._load_meta_only(job_id)
            if current is None:
                return "missing", None
            if current.status in ("completed", "failed", "cancelled", "timeout"):
                return "terminal", current
            if current.status != "preempted":
                # attempt is bumped in requeue_preempted (single place
                # of truth) so every preempted→queued transition gets
                # exactly one increment regardless of how preemption
                # happened.
                current.status = "preempted"
                current.started_at = None
                current.completed_at = None
                current.returncode = None
                current.runner_launch_pending_since = None
                current.missing_actor_since = None
                current.missing_actor_checks = 0
                current.node_dead_since = None
                current.node_dead_checks = 0
                if note:
                    buffer_job_stderr_message(current.job_id, note)
                    current.stderr = _append_log_line(current.stderr, note)
                current._write_meta_unlocked()
            return "preempted", current

    @classmethod
    def requeue_preempted(
        cls,
        job_id: str,
    ) -> tuple[str, Optional["JobRecord"]]:
        """Return a preempted job to the normal queued state."""
        lock_record = cls(job_id=job_id, command="", num_gpus=1, status="unknown")
        with lock_record._meta_lock():
            current = cls._load_meta_only(job_id)
            if current is None:
                return "missing", None
            if current.status in ("completed", "failed", "cancelled", "timeout"):
                return "terminal", current
            if current.status == "queued":
                cls._consume_buffered_notes_unlocked(job_id, materialize_to_stderr=True)
                return "queued", cls.load_unmerged(job_id) or current
            if current.status != "preempted":
                return "unexpected", current

            # Bump attempt on every preempted→queued transition.  This
            # is the one place that unambiguously kicks off a new
            # attempt, so bumping here (rather than at each preempt
            # site) guarantees every retry gets a fresh attempt number
            # regardless of when or how the job entered "preempted".
            current.attempt += 1
            current.status = "queued"
            current.started_at = None
            current.completed_at = None
            current.returncode = None
            current.node_id = ""
            current.runner_launched = False
            current.runner_launch_pending_since = None
            current.dispatch_resource = None
            current.missing_actor_since = None
            current.missing_actor_checks = 0
            current.node_dead_since = None
            current.node_dead_checks = 0
            current._write_meta_unlocked()
            # Remove the previous attempt's terminal result file so the
            # orphan reaper cannot finalize the retry from stale data.
            # The attempt-match check in _finalize_from_process_result
            # is the primary guard; this is defense in depth.
            clear_job_process_result(job_id)
            cls._consume_buffered_notes_unlocked(job_id, materialize_to_stderr=True)
            return "queued", cls.load_unmerged(job_id) or current

    @classmethod
    def finalize_runner_record(
        cls,
        record: "JobRecord",
        *,
        retried: bool,
        event_message: Optional[str] = None,
    ) -> tuple["JobRecord", bool]:
        """Resolve final runner state under lock before writing it back."""
        lock_record = cls(job_id=record.job_id, command="", num_gpus=1, status="unknown")
        with lock_record._meta_lock():
            current = cls._load_meta_only(record.job_id)
            if current and current.status == "cancelled":
                record = cls.load_unmerged(record.job_id) or current
                retried = False
            elif current and current.status == "preempted":
                if record.status not in ("completed", "failed", "timeout"):
                    record = cls.load_unmerged(record.job_id) or current
                    retried = True

            if record.status != "running":
                record.runner_launched = False
                record.runner_launch_pending_since = None
                record.dispatch_resource = None
                record.missing_actor_since = None
                record.missing_actor_checks = 0
                record.node_dead_since = None
                record.node_dead_checks = 0

            if record.status in ("completed", "timeout"):
                if record.stdout and (
                    not record.stdout_path.exists()
                    or record.stdout_path.stat().st_size == 0
                ):
                    _append_text_block(record.stdout_path, record.stdout)
                if record.stderr and (
                    not record.stderr_path.exists()
                    or record.stderr_path.stat().st_size == 0
                ):
                    _append_text_block(record.stderr_path, record.stderr)

            drained_notes = cls._consume_buffered_notes_unlocked(
                record.job_id,
                materialize_to_stderr=(record.status != "running"),
            )
            if drained_notes:
                record.stderr = _append_log_line(record.stderr, drained_notes.rstrip("\n"))
            if event_message and record.status != "cancelled":
                append_job_stderr_message(record.job_id, event_message)
                record.stderr = _append_log_line(record.stderr, event_message)

            record._write_meta_unlocked()
            return record, retried

    def queued_job_needs_ray_verification(self, now: Optional[float] = None) -> bool:
        """Return True when a queued job has an actor that already started."""
        return self.status == "queued" and self.runner_launched

    def save(self):
        """Persist job record to .hive/jobs/<id>/ directory.

        Writes meta.json (small, fast to scan) plus stdout.log / stderr.log
        as separate plain-text files.
        """
        with self._meta_lock():
            self._write_meta_unlocked()
            self._write_logs_unlocked()

    def save_meta(self):
        """Persist only meta.json without touching stdout/stderr log files."""
        with self._meta_lock():
            self._write_meta_unlocked()

    @classmethod
    def load(cls, job_id: str) -> Optional["JobRecord"]:
        """Load a job record from disk.

        Tries directory format first (<id>/meta.json), falls back to
        legacy <id>.json for backward compatibility.
        """
        record = cls.load_unmerged(job_id)
        if record is not None:
            record._merge_legacy_events_for_display()
        return record

    @classmethod
    def load_all(cls) -> List["JobRecord"]:
        """Load all job records from .hive/jobs/.

        Scans for directory-format jobs (<id>/meta.json) and legacy
        single-file jobs (<id>.json).
        """
        if not JOBS_DIR.exists():
            return []
        records = []
        seen_ids = set()

        # New directory format
        for meta_path in sorted(JOBS_DIR.glob("*/meta.json")):
            try:
                record = cls.from_dict(json.loads(meta_path.read_text()))
                stdout_path = meta_path.parent / "stdout.log"
                stderr_path = meta_path.parent / "stderr.log"
                if stdout_path.exists():
                    record.stdout = stdout_path.read_text()
                if stderr_path.exists():
                    record.stderr = stderr_path.read_text()
                record._merge_legacy_events_for_display()
                records.append(record)
                seen_ids.add(record.job_id)
            except Exception:
                pass

        # Legacy single-file format (skip if already loaded from directory)
        for p in sorted(JOBS_DIR.glob("*.json")):
            try:
                d = json.loads(p.read_text())
                if d.get("job_id") not in seen_ids:
                    record = cls.from_dict(d)
                    record._merge_legacy_events_for_display()
                    records.append(record)
            except Exception:
                pass

        return records

    @classmethod
    def load_active(cls) -> List["JobRecord"]:
        """Load only running/queued job records (lightweight).

        Skips terminal jobs and does NOT read stdout/stderr files,
        making this much faster than load_all() when there are
        thousands of finished jobs on disk.
        """
        if not JOBS_DIR.exists():
            return []
        records = []
        _terminal = ("completed", "failed", "cancelled", "timeout")
        for meta_path in JOBS_DIR.glob("*/meta.json"):
            try:
                d = json.loads(meta_path.read_text())
                if d.get("status") in _terminal:
                    continue
                records.append(cls.from_dict(d))
            except Exception:
                pass
        return records


def _ray_cmd() -> str:
    """Find the ray binary — same venv as this process, or PATH."""
    venv_ray = Path(sys.executable).parent / "ray"
    if venv_ray.exists():
        return str(venv_ray)
    return shutil.which("ray") or "ray"


class Dispatcher:
    """Core dispatcher: owns the Ray head and the SLURM worker pool, and
    dispatches jobs onto the cluster through Ray placement groups.
    """

    def __init__(self, config: HiveConfig):
        self.config = config
        self.pool: Optional[PoolManager] = None
        self.active_jobs: Dict[str, JobRecord] = {}
        self.completed_jobs: Dict[str, JobRecord] = {}
        self._ray_head_address: Optional[str] = None
        self._resources: List[ResourceSpec] = []

    def start(self, resources: List[ResourceSpec], walltime: str):
        """Start the Hive cluster: Ray head + SLURM worker pool.

        Args:
            resources: List of ResourceSpec, each with num_jobs > 0.
            walltime: SLURM walltime string (e.g. "12:00:00").
        """
        ensure_hive_dir()
        log_dir = HIVE_DIR / "logs"

        # Clear previous session artifacts (preserve job logs across restarts)
        if log_dir.exists():
            shutil.rmtree(log_dir)
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        self._resources = resources

        # Start Ray head on login node
        self._start_ray_head()

        # Submit SLURM jobs across all resources
        self.pool = PoolManager(self.config, self._ray_head_address)
        job_ids = self.pool.submit_jobs_multi(resources, walltime)

        if not job_ids:
            log.error("No SLURM jobs were submitted successfully")
            self.stop()
            raise RuntimeError("Failed to submit any SLURM jobs")

        # Save state
        max_gpus = max(p.gpus_per_job for p in resources)
        self._save_state(
            resources=[p.to_dict() for p in resources],
            max_gpus_per_node=max_gpus,
            walltime=walltime,
        )

        part_summary = ", ".join(
            f"{p.name}({p.num_jobs}x{p.gpus_per_job}gpu)" for p in resources
        )
        log.info(
            f"Hive started: Ray head at {self._ray_head_address}, "
            f"{len(job_ids)} SLURM jobs submitted [{part_summary}]. "
            f"Waiting for workers to join..."
        )

    def add_resources(self, resources: List[ResourceSpec], default_walltime: str, state: dict):
        """Add resources to a running cluster.

        Submits new SLURM jobs and updates state file with new resources and job IDs.
        """
        self._resources = [
            ResourceSpec.from_dict(r) for r in state.get("resources", [])
        ]

        self.pool = PoolManager(self.config, self._ray_head_address)
        # Reconstruct existing jobs from state (handles both old list and new dict format)
        slurm_jobs = state.get("slurm_jobs", {})
        if isinstance(slurm_jobs, list):
            for job_id in slurm_jobs:
                self.pool.jobs[job_id] = SlurmJob(job_id=job_id, state="UNKNOWN")
        else:
            for job_id, meta in slurm_jobs.items():
                self.pool.jobs[job_id] = SlurmJob(
                    job_id=job_id, state="UNKNOWN",
                    resource_name=meta.get("resource_name", ""),
                )

        new_job_ids = self.pool.submit_jobs_multi(resources, default_walltime)
        if not new_job_ids:
            raise RuntimeError("Failed to submit any new SLURM jobs")

        # Merge into state
        self._resources.extend(resources)
        existing_resources = state.get("resources", [])
        existing_resources.extend([p.to_dict() for p in resources])

        max_gpus = max(p.gpus_per_job for p in self._resources)
        self._save_state(
            resources=existing_resources,
            max_gpus_per_node=max_gpus,
            walltime=default_walltime,
        )

        return new_job_ids

    def remove_resources(self, resources: List[ResourceSpec], state: dict) -> List[str]:
        """Remove resources from a running cluster.

        Cancels SLURM jobs matching the requested resources, preferring
        PENDING jobs over RUNNING ones to minimize disruption.

        Returns list of cancelled SLURM job IDs.
        """
        self.pool = PoolManager(self.config, self._ray_head_address)

        # Reconstruct pool from state
        slurm_jobs = state.get("slurm_jobs", {})
        if isinstance(slurm_jobs, list):
            for job_id in slurm_jobs:
                self.pool.jobs[job_id] = SlurmJob(job_id=job_id, state="UNKNOWN")
        else:
            for job_id, meta in slurm_jobs.items():
                self.pool.jobs[job_id] = SlurmJob(
                    job_id=job_id, state="UNKNOWN",
                    resource_name=meta.get("resource_name", ""),
                )

        # Include requeued jobs from slurm_tracking/
        if SLURM_TRACKING_DIR.is_dir():
            for f in SLURM_TRACKING_DIR.iterdir():
                if f.is_file() and f.name not in self.pool.jobs:
                    resource_name = f.read_text().strip()
                    self.pool.jobs[f.name] = SlurmJob(
                        job_id=f.name, state="UNKNOWN",
                        resource_name=resource_name,
                    )

        # Get actual SLURM states
        self.pool.refresh_job_states()

        # Query estimated start times for pending jobs so we can keep the
        # ones closest to running and cancel the rest.
        pending_ids = [
            jid for jid, job in self.pool.jobs.items()
            if job.state == "PENDING"
        ]
        est_start_map: dict[str, str] = {}  # job_id -> ISO start time
        if pending_ids:
            try:
                sq = subprocess.run(
                    ["squeue", "-j", ",".join(pending_ids),
                     "--format=%i|%S", "--noheader"],
                    capture_output=True, text=True, timeout=10,
                )
                if sq.returncode == 0:
                    for line in sq.stdout.strip().split("\n"):
                        if "|" not in line:
                            continue
                        jid, start = line.strip().split("|", 1)
                        jid, start = jid.strip(), start.strip()
                        if start and start != "N/A":
                            est_start_map[jid] = start
            except Exception:
                pass

        cancelled_ids = []
        for rspec in resources:
            matching = [
                (jid, job) for jid, job in self.pool.jobs.items()
                if job.resource_name == rspec.name
                and job.state in ("PENDING", "RUNNING", "UNKNOWN")
            ]

            # Cancel order (first cancelled → last cancelled):
            #   1. PENDING with no est_start (SLURM doesn't know → likely furthest out)
            #   2. PENDING with latest est_start (furthest from running)
            #   3. UNKNOWN
            #   4. RUNNING (last resort)
            _state_pri = {"PENDING": 0, "UNKNOWN": 1, "RUNNING": 2}
            _LATE = "9999-99-99T99:99:99"  # sorts after any real timestamp

            def _cancel_order(item):
                jid, job = item
                sp = _state_pri.get(job.state, 99)
                est = est_start_map.get(jid, "")
                if job.state == "PENDING":
                    # No estimate → cancel first (has_est=0 < 1).
                    # With estimate → later start cancelled first: negate by
                    # subtracting from _LATE (we just reverse the string sort).
                    return (sp, 0 if not est else 1, _LATE if not est else est)
                return (sp, 0, "")

            matching.sort(key=_cancel_order)
            # Within group 2 (PENDING + has est), earliest start is at the
            # front but we want latest first → reverse just that subgroup.
            first_est = next((i for i, (jid, job) in enumerate(matching)
                              if job.state == "PENDING" and est_start_map.get(jid)), None)
            if first_est is not None:
                last_est = first_est
                for i in range(first_est, len(matching)):
                    jid, job = matching[i]
                    if job.state == "PENDING" and est_start_map.get(jid):
                        last_est = i
                    else:
                        break
                matching[first_est:last_est + 1] = reversed(matching[first_est:last_est + 1])

            to_cancel = matching[:rspec.num_jobs]
            if len(to_cancel) < rspec.num_jobs:
                log.warning(
                    f"Requested {rspec.num_jobs} {rspec.name} jobs to remove, "
                    f"but only {len(to_cancel)} are active"
                )

            ids = [jid for jid, _ in to_cancel]
            self.pool.cancel_jobs(ids)
            cancelled_ids.extend(ids)

        # Persist updated state (cancelled jobs removed from pool.jobs)
        max_gpus = max(
            (r.gpus_per_job for r in self._resources),
            default=0,
        )
        self._save_state(
            resources=state.get("resources", []),
            max_gpus_per_node=max_gpus or state.get("max_gpus_per_node", 4),
            walltime=state.get("walltime", self.config.slurm.default_walltime),
        )

        return cancelled_ids

    def wait_for_workers(self, min_gpus: int = 1, timeout: int = 600, poll_interval: int = 10):
        """Block until at least min_gpus GPUs are available in the cluster.

        Returns the number of available GPUs.
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                resources = ray.cluster_resources()
                gpu_count = int(resources.get("GPU", 0))
                if gpu_count >= min_gpus:
                    log.info(f"Cluster ready: {gpu_count} GPUs available")
                    return gpu_count
            except Exception as e:
                log.debug(f"Waiting for cluster: {e}")
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Timed out waiting for {min_gpus} GPUs after {timeout}s"
        )

    def submit_job(self, command: str, num_gpus: int = 1, num_cpus: int = 1,
                   memory_mb: int = 0, workdir: Optional[str] = None,
                   timeout: Optional[int] = None, resource: Optional[str] = None) -> dict:
        """Submit a job synchronously. Blocks until completion.

        Returns dict with stdout, stderr, returncode, duration.
        """
        if workdir is None:
            workdir = self.config.env.working_dir

        # Max GPUs per node across all resources
        if self._resources:
            max_node_gpus = max(p.gpus_per_job for p in self._resources)
        else:
            max_node_gpus = self.config.slurm.gpus_per_job

        # Validate: can't request more GPUs than any single node has
        if num_gpus > max_node_gpus:
            return {
                "stdout": "",
                "stderr": f"Error: max {max_node_gpus} GPUs per job (single-node constraint). Requested: {num_gpus}.",
                "returncode": 1,
                "duration": 0,
                "job_id": None,
            }

        # Check per-node capacity
        max_on_single_node = self._max_gpus_on_single_node()
        if num_gpus > max_on_single_node:
            return {
                "stdout": "",
                "stderr": (
                    f"Cannot allocate {num_gpus} GPUs on a single node. "
                    f"Max available on one node: {max_on_single_node}."
                ),
                "returncode": 1,
                "duration": 0,
                "job_id": None,
            }

        job_id = str(uuid.uuid4())[:8]
        record = JobRecord(
            job_id=job_id,
            command=command,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            memory_mb=memory_mb,
            status="running",
            submitted_at=time.time(),
        )
        self.active_jobs[job_id] = record
        pg = None

        pgs = []
        try:
            # Parse resource list
            resource_list = [r.strip() for r in resource.split(",")] if resource else []

            # Build bundles — one per resource, or a single unconstrained bundle
            base = {"CPU": num_cpus, "GPU": num_gpus}
            if resource_list:
                bundles = []
                for res_name in resource_list:
                    b = dict(base)
                    b[f"partition:{res_name}"] = 0.01
                    bundles.append(b)
            else:
                bundles = [base]

            # Create placement groups and race them
            for b in bundles:
                pgs.append(placement_group([b], strategy="STRICT_PACK"))

            ready_futures = [p.ready() for p in pgs]
            try:
                done, _ = ray.wait(ready_futures, num_returns=1, timeout=60)
                if not done:
                    raise RuntimeError(
                        "Timed out waiting for placement group — not enough resources on any single node"
                    )
            except ray.exceptions.GetTimeoutError:
                raise RuntimeError(
                    "Timed out waiting for placement group — not enough resources on any single node"
                )

            # Use the winner, remove the rest
            winner_idx = ready_futures.index(done[0])
            pg = pgs[winner_idx]
            for i, p in enumerate(pgs):
                if i != winner_idx:
                    remove_placement_group(p)

            # Detect which node the PG landed on
            try:
                job_node_id = ray.get(
                    _get_job_node_id.options(
                        placement_group=pg,
                        placement_group_bundle_index=0,
                    ).remote()
                )
                record.node_id = job_node_id
            except Exception:
                job_node_id = None

            # Submit the job
            _rv = str(int(Path(__file__).stat().st_mtime))
            run_opts = dict(
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                placement_group=pg,
                runtime_env={"env_vars": {
                    "RAY_LOG_TO_STDERR": "1",
                    "RAY_ROTATION_MAX_BYTES": "1",
                    "HIVE_CODE_VERSION": _rv,
                }},
            )
            if memory_mb > 0:
                run_opts["memory"] = memory_mb * 1024 * 1024
            future = _run_job.options(**run_opts).remote(
                command, workdir, job_id=job_id, timeout_sec=timeout or 0,
                attempt=record.attempt,
            )

            # Poll instead of blocking on ray.get() so we can detect
            # worker node death.  Without this, ray.get() hangs
            # indefinitely when a SLURM job is killed.
            #
            # ray.nodes() can return stale data, so require multiple
            # consecutive "dead" observations before acting.
            deadline = time.time() + timeout if timeout else None
            _node_dead_consecutive = 0
            _NODE_DEAD_THRESHOLD = 3
            while True:
                poll_timeout = min(30, timeout) if timeout else 30
                ready, _ = ray.wait([future], timeout=poll_timeout)
                if ready:
                    break
                if deadline and time.time() >= deadline:
                    ray.cancel(future, force=True)
                    raise RuntimeError(
                        f"Job timed out after {timeout}s"
                    )
                if job_node_id:
                    try:
                        node_alive = any(
                            n["NodeID"] == job_node_id
                            and n.get("Alive", False)
                            for n in ray.nodes()
                        )
                        if not node_alive:
                            _node_dead_consecutive += 1
                            if _node_dead_consecutive >= _NODE_DEAD_THRESHOLD:
                                ray.cancel(future, force=True)
                                raise RuntimeError(
                                    f"Worker node {job_node_id[:12]} died "
                                    f"({_node_dead_consecutive} consecutive checks)"
                                )
                        else:
                            _node_dead_consecutive = 0
                    except RuntimeError:
                        raise
                    except Exception:
                        pass
            result = ray.get(future)

            _rc = result.get("returncode")
            if result.get("timed_out"):
                record.status = "timeout"
            elif _rc == 0:
                record.status = "completed"
            else:
                # Non-zero exit is a failure, not a completion.  See
                # the matching branch in the async runner above.
                record.status = "failed"
            record.returncode = _rc
            record.result = result
            record.completed_at = time.time()
            result["duration"] = record.completed_at - record.submitted_at
            result["job_id"] = job_id

        except Exception as e:
            # Infrastructure failure (node death, worker crash).
            # Requeue so the job runs on another node.
            # attempt is bumped in requeue_preempted, not here.
            record.status = "preempted"
            record.started_at = None
            record.completed_at = None
            result = {
                "stdout": "",
                "stderr": f"[hive] Worker lost — requeuing: {e}",
                "returncode": -1,
                "duration": 0,
                "job_id": job_id,
            }
            record.result = result

        finally:
            # Clean up all placement groups
            for p in pgs:
                try:
                    remove_placement_group(p)
                except Exception:
                    pass
            self.completed_jobs[job_id] = self.active_jobs.pop(job_id, record)

        return result

    def get_status(self) -> dict:
        """Return full cluster status."""
        try:
            resources = ray.cluster_resources()
            available = ray.available_resources()
        except Exception:
            resources = {}
            available = {}

        total_gpus = int(resources.get("GPU", 0))
        free_gpus = int(available.get("GPU", 0))

        pool_status = self.pool.get_status_summary() if self.pool else {}

        return {
            "ray_head": self._ray_head_address,
            "total_gpus": total_gpus,
            "free_gpus": free_gpus,
            "active_jobs": {
                jid: {"command": j.command, "gpus": j.num_gpus, "status": j.status}
                for jid, j in self.active_jobs.items()
            },
            "pool": pool_status,
        }

    def submit_async(self, command: str, num_gpus: int = 1, num_cpus: int = 1,
                     memory_mb: int = 0, workdir: Optional[str] = None,
                     resource: Optional[str] = None,
                     user: Optional[str] = None,
                     on_complete: Optional[str] = None,
                     timeout_sec: Optional[int] = None,
                     max_retries: int = 0,
                     priority: str = "normal") -> JobRecord:
        """Submit a job asynchronously. Returns immediately with a JobRecord.

        The job is queued as a lightweight Ray task that manages its own
        placement group. It waits for resources to become available, runs the
        command, and writes results to .hive/jobs/<id>.json.

        resource can be a single name or comma-separated list (e.g. "gpu-b,gpu-test").
        When multiple resources are given, the job runs on whichever is available first.
        """
        if workdir is None:
            workdir = self.config.env.working_dir

        job_id = str(uuid.uuid4())[:8]

        # Low-priority jobs get high max_retries so they auto-retry after preemption
        if priority == "low":
            max_retries = max(max_retries, 100)

        record = JobRecord(
            job_id=job_id,
            command=command,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            memory_mb=memory_mb,
            status="queued",
            submitted_at=time.time(),
            resource=resource,
            user=user or os.environ.get("USER"),
            on_complete=on_complete,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            priority=priority,
        )
        record.save()

        if priority == "low":
            # Low-priority jobs: don't create actors at submit time.
            # The dashboard health loop dispatches them one at a time
            # when GPUs become free. This avoids overwhelming Ray with
            # hundreds of actors waiting for placement groups.
            log.info("Job %s: queued (low-pri, dispatched by health loop)", job_id)
            return record

        # Normal priority: launch immediately as a detached actor.
        # If actor creation fails (e.g., dead Ray socket), leave job as
        # queued — the dashboard health loop will retry via _dispatch_queued.
        try:
            launch_job_runner(record, workdir=workdir)
        except Exception as e:
            log.warning("Actor creation failed for %s (will retry via health loop): %s",
                        job_id, e)

        return record

    def stop(self):
        """Tear down everything: job runners, Ray, SLURM jobs."""
        log.info("Stopping Hive...")

        # Kill any detached job runner actors
        try:
            for entry in ray.util.list_named_actors(all_namespaces=True):
                actor_name = entry if isinstance(entry, str) else entry.get("name", "")
                if actor_name.startswith(JOB_ACTOR_PREFIX):
                    try:
                        actor = ray.get_actor(actor_name)
                        ray.kill(actor)
                    except Exception:
                        pass
        except Exception:
            pass

        # Remove state.json FIRST so EXIT traps in SLURM jobs become no-ops
        # (they check for state.json before requeuing)
        if STATE_PATH.exists():
            STATE_PATH.unlink()

        # Cancel SLURM jobs known to the pool (from state.json)
        if self.pool:
            self.pool.cancel_all()

        # Also cancel any requeued jobs tracked in slurm_tracking/
        from .config import SLURM_TRACKING_DIR
        if SLURM_TRACKING_DIR.is_dir():
            tracked_ids = [f.name for f in SLURM_TRACKING_DIR.iterdir() if f.is_file()]
            # Exclude IDs already cancelled by pool.cancel_all()
            pool_ids = set(self.pool.jobs.keys()) if self.pool else set()
            extra_ids = [jid for jid in tracked_ids if jid not in pool_ids]
            if extra_ids:
                try:
                    subprocess.run(
                        ["scancel"] + extra_ids,
                        capture_output=True, text=True, timeout=30,
                    )
                    log.info(f"Cancelled {len(extra_ids)} requeued SLURM jobs")
                except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                    log.warning(f"scancel error for requeued jobs: {e}")
            import shutil
            shutil.rmtree(SLURM_TRACKING_DIR, ignore_errors=True)

        # Stop Ray
        self._stop_ray_head()

        log.info("Hive stopped.")

    def restart(self):
        """Restart Ray head without cancelling SLURM jobs.

        Workers use a reconnect loop in their sbatch scripts, so they will
        automatically rejoin after the head comes back up.  state.json is
        kept intact throughout so workers know the cluster is still alive.

        Handles dead Ray sessions gracefully — if Ray is unreachable, skips
        actor cleanup and proceeds directly to force-stop + restart.
        """
        log.info("Restarting Hive (keeping SLURM jobs)...")

        # Best-effort cleanup of job actors whose jobs already finished.
        # If Ray is dead this fails silently, which is fine: the actors die
        # with the old session anyway.
        if ray.is_initialized():
            try:
                named = ray.util.list_named_actors(all_namespaces=True)
                finished_ids = set()
                for r in JobRecord.load_all():
                    if r.status not in ("running", "queued"):
                        finished_ids.add(r.job_id)
                for entry in named:
                    name = entry if isinstance(entry, str) else entry.get("name", "")
                    if not name.startswith(JOB_ACTOR_PREFIX):
                        continue
                    job_id = name[len(JOB_ACTOR_PREFIX):]
                    if job_id in finished_ids:
                        try:
                            ray.kill(ray.get_actor(name, namespace="hive"))
                        except Exception:
                            pass
            except Exception:
                pass
        else:
            log.info("Ray not initialized — skipping actor cleanup")

        # Write restart signal so sbatch workers detect it and reconnect
        restart_signal = HIVE_DIR / "restart_signal"
        restart_signal.write_text(str(time.time()))

        # Restart Ray head — workers will detect the loss and reconnect
        self._stop_ray_head()
        time.sleep(2)
        self._start_ray_head()

        # Update PID in state.json (preserve everything else like slurm_jobs)
        state = self.load_state() or {}
        state["ray_head_address"] = self._ray_head_address
        state["pid"] = os.getpid()
        STATE_PATH.write_text(json.dumps(state, indent=2))

        log.info("Hive restarted. Workers will reconnect automatically.")

    # ── Ray lifecycle ────────────────────────────────────────────────

    @staticmethod
    def _detect_cluster_ip() -> str:
        """Find the IP on the internal cluster network (non-public, non-loopback).

        Prefers private IPs (172.x, 10.x, 192.168.x) over public IPs,
        so compute nodes can reach the head via the cluster interconnect.
        """
        result = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=5,
        )
        candidates = result.stdout.strip().split() if result.returncode == 0 else []

        # Prefer private/cluster IPs
        for ip in candidates:
            if ip.startswith("172.") or ip.startswith("10.") or ip.startswith("192.168."):
                return ip
        # Fall back to any IP from the list
        if candidates:
            return candidates[0]
        return socket.gethostbyname(socket.gethostname())

    def _start_ray_head(self):
        """Start Ray head process on the login node."""
        port = self.config.ray.head_port
        if self.config.ray.head_ip:
            ip = self.config.ray.head_ip
        else:
            ip = self._detect_cluster_ip()
        self._ray_head_address = f"{ip}:{port}"

        # Stop any existing Ray instance first
        try:
            subprocess.run([_ray_cmd(), "stop", "--force"], capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            log.warning("ray stop timed out — killing Ray processes directly")
            subprocess.run(["pkill", "-9", "-f", "ray::GCS"], capture_output=True)
            subprocess.run(["pkill", "-9", "-f", "raylet"], capture_output=True)
        time.sleep(2)

        result = subprocess.run(
            [
                _ray_cmd(), "start", "--head",
                f"--port={port}",
                f"--node-ip-address={ip}",
                "--include-dashboard=false",
                "--num-gpus=0",
                "--num-cpus=2",
                f"--temp-dir={get_ray_tmpdir()}",
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start Ray head: {result.stderr}")

        # Connect this process to Ray using the explicit address — never "auto",
        # which can pick up another user's Ray cluster via /tmp/ray/session_latest.
        ray.init(address=self._ray_head_address, namespace="hive", ignore_reinit_error=True)
        log.info(f"Ray head started at {self._ray_head_address}")

    def _stop_ray_head(self):
        """Stop Ray head process."""
        try:
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass
        try:
            subprocess.run([_ray_cmd(), "stop", "--force"], capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            # Zombie GCS can make `ray stop` hang — kill it directly
            log.warning("ray stop timed out — killing Ray processes directly")
            subprocess.run(["pkill", "-9", "-f", "ray::GCS"], capture_output=True)
            subprocess.run(["pkill", "-9", "-f", "raylet"], capture_output=True)
            time.sleep(1)

    # ── Cluster introspection ────────────────────────────────────────

    def _max_gpus_on_single_node(self) -> int:
        """Get max GPUs available on any single node."""
        try:
            nodes = ray.nodes()
            max_gpus = 0
            for node in nodes:
                if not node.get("Alive", False):
                    continue
                resources = node.get("Resources", {})
                node_gpus = int(resources.get("GPU", 0))
                if node_gpus > max_gpus:
                    max_gpus = node_gpus
            return max_gpus
        except Exception:
            if self._resources:
                return max(p.gpus_per_job for p in self._resources)
            return self.config.slurm.gpus_per_job

    # ── State persistence ────────────────────────────────────────────

    def _save_state(self, **kwargs):
        """Persist cluster state to disk so other CLI invocations can reconnect.

        slurm_jobs is stored as {job_id: {resource_name: "..."}} so we can
        resubmit expired jobs with the original ResourceSpec.
        """
        state = {
            "ray_head_address": self._ray_head_address,
            "pid": os.getpid(),
            **kwargs,
        }
        if self.pool:
            state["slurm_jobs"] = {
                jid: {"resource_name": job.resource_name}
                for jid, job in self.pool.jobs.items()
            }

        STATE_PATH.write_text(json.dumps(state, indent=2))

    @classmethod
    def load_state(cls) -> Optional[dict]:
        """Load saved state from disk."""
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text())
        return None


def _targeted_preempt(gpus_needed: int, bundles: list):
    """Free GPUs for a normal-priority job by preempting low-priority jobs.

    Picks the node where killing the fewest low-priority jobs makes
    ``gpus_needed`` GPUs available (GPUs that are already idle on the node
    count towards the total) and kills only as many of those jobs as needed.
    Does nothing when no eligible node runs a low-priority job.
    """
    from .config import JOBS_DIR

    # node_id -> running low-priority jobs, and node_id -> GPUs held by any
    # running job (used to work out how many GPUs on that node are idle).
    node_lowpri: dict[str, list] = {}
    node_busy_gpus: dict[str, int] = {}
    for job_dir in JOBS_DIR.iterdir():
        meta_path = job_dir / "meta.json" if job_dir.is_dir() else None
        if not meta_path or not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if meta.get("status") != "running":
            continue
        nid = meta.get("node_id", "")
        if not nid:
            continue
        gpus = meta.get("num_gpus", 1)
        node_busy_gpus[nid] = node_busy_gpus.get(nid, 0) + gpus
        if meta.get("priority") == "low":
            node_lowpri.setdefault(nid, []).append({
                "job_id": meta["job_id"],
                "num_gpus": gpus,
            })

    if not node_lowpri:
        return

    # GPU totals and partition tags of the live GPU nodes.
    node_total_gpus: dict[str, int] = {}
    node_partitions: dict[str, set] = {}
    try:
        for n in ray.nodes():
            if not n.get("Alive", False):
                continue
            res = n.get("Resources", {})
            total = int(res.get("GPU", 0))
            if total <= 0:
                continue
            nid = n["NodeID"]
            node_total_gpus[nid] = total
            node_partitions[nid] = {
                k.split(":", 1)[1] for k in res if k.startswith("partition:")
            }
    except Exception:
        pass

    # Resource constraints requested by the job's bundles.
    required_partitions = set()
    for b in bundles:
        for k in b:
            if k.startswith("partition:"):
                required_partitions.add(k.split(":", 1)[1])

    # Best node: fewest low-priority jobs to kill while still freeing enough.
    best_node = None
    best_kills = float("inf")
    best_idle = 0
    for nid, jobs in node_lowpri.items():
        if required_partitions and not (node_partitions.get(nid, set()) & required_partitions):
            continue
        idle = max(0, node_total_gpus.get(nid, 0) - node_busy_gpus.get(nid, 0))
        freeable = idle + sum(j["num_gpus"] for j in jobs)
        if freeable >= gpus_needed and len(jobs) < best_kills:
            best_kills = len(jobs)
            best_node = nid
            best_idle = idle

    if best_node is None:
        return

    # Kill low-priority jobs on the target node, only as many as needed on
    # top of the GPUs that are already idle there.
    freed = best_idle
    killed_any = False
    for job_info in node_lowpri[best_node]:
        if freed >= gpus_needed:
            break
        job_id = job_info["job_id"]
        actor_name = None
        for entry in ray.util.list_named_actors(all_namespaces=True):
            name = entry if isinstance(entry, str) else entry.get("name", "")
            if name.startswith(f"{JOB_ACTOR_PREFIX}{job_id}"):
                actor_name = name
                break
        if actor_name:
            try:
                ray.kill(ray.get_actor(actor_name))
                freed += job_info["num_gpus"]
                killed_any = True
                log.warning("Preempt: killed low-pri job %s on node %s (freed %d GPUs)",
                            job_id, best_node[:8], job_info["num_gpus"])
            except Exception as e:
                log.warning("Preempt: failed to kill %s: %s", job_id, e)

    if killed_any:
        time.sleep(5)  # give the killed processes time to release GPU memory



@ray.remote(num_gpus=0)
def _get_job_node_id():
    """Return the Ray node ID of the node this task runs on."""
    return ray.get_runtime_context().get_node_id()


@ray.remote
def _run_job(command: str, workdir: str, job_id: str = "", timeout_sec: int | None = 0, attempt: int = 0) -> dict:
    """Ray remote function that executes a shell command on a compute node.

    Streams stdout/stderr to JOBS_DIR/<job_id>/ every ~1s so the dashboard
    can serve incremental output while the job is running.
    """
    import select

    # Build a clean environment: remove Hive's venv from PATH and clear
    # PYTHONPATH so user scripts don't inherit conflicting packages.
    env = os.environ.copy()
    hive_venv = os.path.dirname(sys.executable)
    if hive_venv:
        env["PATH"] = os.pathsep.join(
            p for p in env.get("PATH", "").split(os.pathsep)
            if not p.startswith(hive_venv)
        )
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)

    stdout_preamble = ""
    stderr_preamble = ""
    if job_id:
        stdout_preamble = _build_job_log_preamble(
            "stdout", job_id=job_id, command=command, workdir=workdir, timeout_sec=timeout_sec,
        )
        stderr_preamble = _build_job_log_preamble(
            "stderr", job_id=job_id, command=command, workdir=workdir, timeout_sec=timeout_sec,
        )

    proc = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=workdir, env=env,
        start_new_session=True,  # own process group so we can kill the tree
    )

    # Kill the subprocess tree on SIGTERM (sent by ray.cancel(force=True))
    import signal
    _original_sigterm = signal.getsignal(signal.SIGTERM)
    def _kill_on_sigterm(signum, frame):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        if callable(_original_sigterm):
            _original_sigterm(signum, frame)
    try:
        signal.signal(signal.SIGTERM, _kill_on_sigterm)
    except Exception:
        pass  # not main thread

    try:
        # Set pipes to non-blocking
        os.set_blocking(proc.stdout.fileno(), False)
        os.set_blocking(proc.stderr.fileno(), False)

        stdout_chunks: list[str] = [stdout_preamble] if stdout_preamble else []
        stderr_chunks: list[str] = [stderr_preamble] if stderr_preamble else []
        start = time.time()

        # Resolve log paths once
        timed_out = False

        if job_id:
            job_dir = JOBS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            stdout_log = job_dir / "stdout.log"
            stderr_log = job_dir / "stderr.log"
            _append_text_block(stdout_log, "".join(stdout_chunks), separate_if_nonempty=True)
            _append_text_block(stderr_log, "".join(stderr_chunks), separate_if_nonempty=True)
            write_job_heartbeat(job_id)
        else:
            stdout_log = stderr_log = None

        while proc.poll() is None:
            ready, _, _ = select.select(
                [proc.stdout, proc.stderr], [], [], 1.0,
            )
            for fd in ready:
                try:
                    data = fd.read()
                except Exception:
                    data = None
                if data:
                    if fd is proc.stdout:
                        stdout_chunks.append(data)
                        if stdout_log:
                            _append_text_block(stdout_log, data)
                    else:
                        stderr_chunks.append(data)
                        if stderr_log:
                            _append_text_block(stderr_log, data)

            # Refresh heartbeat periodically while the process is alive.
            if stdout_log:
                write_job_heartbeat(job_id)

            # Timeout check
            if timeout_sec and timeout_sec > 0 and (time.time() - start) > timeout_sec:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
                stderr_chunks.append(
                    f"\n[hive] Process killed: timeout after {timeout_sec}s\n"
                )
                if stderr_log:
                    _append_text_block(
                        stderr_log,
                        f"\n[hive] Process killed: timeout after {timeout_sec}s\n",
                    )
                timed_out = True
                break

        # Drain remaining output after exit
        for fd, chunks in [(proc.stdout, stdout_chunks), (proc.stderr, stderr_chunks)]:
            if fd is None:
                continue
            try:
                remaining = fd.read()
                if remaining:
                    chunks.append(remaining)
                    if stdout_log and fd is proc.stdout:
                        _append_text_block(stdout_log, remaining)
                    elif stderr_log and fd is proc.stderr:
                        _append_text_block(stderr_log, remaining)
            except Exception:
                pass

        proc.wait()

        stdout_full = "".join(stdout_chunks)
        stderr_full = "".join(stderr_chunks)

        if stdout_log:
            completed_at = time.time()
            write_job_heartbeat(job_id, completed_at)
            write_job_process_result(
                job_id,
                completed_at=completed_at,
                returncode=proc.returncode,
                timed_out=timed_out,
                attempt=attempt,
            )

        return {
            "stdout": stdout_full,
            "stderr": stderr_full,
            "returncode": proc.returncode,
            "timed_out": timed_out,
        }
    finally:
        # Always restore SIGTERM handler — Ray reuses worker processes.
        try:
            signal.signal(signal.SIGTERM, _original_sigterm)
        except Exception:
            pass


JOB_ACTOR_PREFIX = "hive_job_"


@ray.remote(num_cpus=0, num_gpus=0)
class _AsyncJobRunner:
    """Detached actor that manages a single async job's lifecycle.

    All work runs in __init__ so it starts immediately on actor creation,
    before the submitting driver can disconnect and cancel pending method
    calls. After the job finishes, the actor self-destructs.
    """

    def __init__(self, job_id: str, command: str, num_gpus: int,
                 num_cpus: int, memory_mb: int, workdir: str, bundles: list,
                 timeout_sec: Optional[int] = None,
                 hive_dir: Optional[str] = None):
        # Ensure this actor uses the correct HIVE_DIR, even if the Ray
        # worker inherited a stale environment from cluster startup.
        if hive_dir:
            import hive.config as _cfg
            from pathlib import Path as _Path
            _new = _Path(hive_dir)
            if _cfg.HIVE_DIR != _new:
                _cfg.HIVE_DIR = _new
                _cfg.JOBS_DIR = _new / "jobs"
                _cfg.STATE_PATH = _new / "state.json"
                _cfg.DRAINING_NODES_DIR = _new / "draining_nodes"
                # Also patch dispatcher module-level import
                import hive.dispatcher as _disp
                _disp.JOBS_DIR = _cfg.JOBS_DIR

        def _dbg(msg):
            log.debug("job %s: %s", job_id[:8], msg)

        _dbg(f"init start, gpus={num_gpus}, bundles={len(bundles)}, hive_dir={hive_dir}")
        record = JobRecord.load(job_id)
        if record is None:
            _dbg("record is None, returning")
            return

        pg = None
        future = None
        retried = False
        event_message = None
        pgs = []
        draining_notice_emitted = False
        try:
            record = JobRecord.activate_runner(job_id)
            if record is None or record.status != "queued":
                _dbg(f"activate_runner: record={record is not None}, status={record.status if record else 'None'}")
                return

            while True:
                # Create one placement group per bundle (one per resource).
                # Race them — first to be ready wins.
                pgs = []
                for b in bundles:
                    pgs.append(placement_group([b], strategy="STRICT_PACK"))
                _dbg(f"created {len(pgs)} PGs")

                # Wait for any placement group to become ready.
                # For normal-priority jobs, if PG isn't ready immediately,
                # do targeted preemption: find a node where killing low-pri
                # jobs would free enough GPUs, kill only those, then wait.
                ready_futures = [p.ready() for p in pgs]
                done, pending = ray.wait(ready_futures, num_returns=1, timeout=5)
                _dbg(f"PG wait(5s): done={len(done)}, pending={len(pending)}")

                if not done and record.priority != "low":
                    # Targeted preemption: find the best node and kill only
                    # the low-pri jobs there.
                    _targeted_preempt(num_gpus, bundles)
                    # Wait again with longer timeout
                    done, pending = ray.wait(ready_futures, num_returns=1, timeout=600)

                if not done and record.priority == "low":
                    # Low-pri: wait up to 10 min, no preemption
                    done, pending = ray.wait(ready_futures, num_returns=1, timeout=600)

                if not done:
                    # Still nothing — return to queue for later dispatch
                    _dbg("PG timeout, returning to queue")
                    for p in pgs:
                        try:
                            remove_placement_group(p)
                        except Exception:
                            pass
                    pgs.clear()
                    record.status = "queued"
                    record.started_at = None
                    record.dispatch_resource = None
                    record.save_meta()
                    log.info("Job %s: no GPUs available, returning to queue",
                             job_id)
                    return

                # Use the winning PG, remove the rest
                winner_idx = ready_futures.index(done[0])
                pg = pgs[winner_idx]
                for i, p in enumerate(pgs):
                    if i != winner_idx:
                        remove_placement_group(p)

                winning_bundle = bundles[winner_idx]
                _dbg(f"PG ready, winner={winning_bundle}")

                # Record which node the placement group landed on.
                try:
                    job_node_id = ray.get(
                        _get_job_node_id.options(
                            placement_group=pg,
                            placement_group_bundle_index=0,
                        ).remote()
                    )
                    record.node_id = job_node_id
                    _dbg(f"node_id={job_node_id[:8]}")
                except Exception as e:
                    log.warning(f"Node detection failed for job {job_id}: {e}")

                draining = record.node_id and is_node_draining(record.node_id)
                _dbg(f"draining check: node={record.node_id[:8] if record.node_id else 'none'} draining={draining}")
                if draining:
                    for p in pgs:
                        try:
                            remove_placement_group(p)
                        except Exception:
                            pass
                    pgs.clear()
                    if not draining_notice_emitted:
                        note = "[hive] Node draining before SLURM walltime; waiting for another node"
                        append_job_stderr_message(job_id, note)
                        record.stderr = _append_log_line(record.stderr, note)
                        draining_notice_emitted = True
                    record.status = "queued"
                    record.node_id = ""
                    record.started_at = None
                    record.dispatch_resource = None
                    record.save_meta()
                    log.info(
                        "Job %s: placement landed on draining node, waiting for another node",
                        job_id,
                    )
                    time.sleep(2)
                    continue

                # Record which resource was actually used only after accepting the node.
                for k in winning_bundle:
                    if k.startswith("partition:"):
                        record.resource = k.split(":", 1)[1]
                        record.dispatch_resource = None
                        break
                break

            record.status = "running"
            record.started_at = time.time()
            record.missing_actor_since = None
            record.missing_actor_checks = 0
            record.node_dead_since = None
            record.node_dead_checks = 0
            record.save_meta()
            _dbg("status=running, launching _run_job")

            # Run the actual command on the placement group
            _rv = os.environ.get("HIVE_CODE_VERSION", "0")
            run_opts = dict(
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                placement_group=pg,
                runtime_env={"env_vars": {
                    "RAY_LOG_TO_STDERR": "1",
                    "RAY_ROTATION_MAX_BYTES": "1",
                    "HIVE_CODE_VERSION": _rv,
                }},
            )
            if memory_mb > 0:
                run_opts["memory"] = memory_mb * 1024 * 1024
            future = _run_job.options(**run_opts).remote(
                command, workdir, job_id=job_id, timeout_sec=timeout_sec,
                attempt=record.attempt,
            )

            # Poll instead of blocking on ray.get() so we can detect
            # worker node death.  Without this, ray.get() hangs
            # indefinitely when a SLURM job is killed (node dies but
            # the detached actor on the head node keeps waiting).
            #
            # ray.nodes() can return stale data (transient GCS
            # inconsistency), so require multiple consecutive "dead"
            # observations before acting.  A single stale response
            # must never kill a running job.
            _node_dead_consecutive = 0
            _NODE_DEAD_THRESHOLD = 3  # 3 × 30s = 90s of confirmed death
            while True:
                ready, _ = ray.wait([future], timeout=30)
                if ready:
                    break
                # Check if the worker node is still alive
                if record.node_id:
                    try:
                        node_alive = any(
                            n["NodeID"] == record.node_id
                            and n.get("Alive", False)
                            for n in ray.nodes()
                        )
                        if not node_alive:
                            _node_dead_consecutive += 1
                            if _node_dead_consecutive >= _NODE_DEAD_THRESHOLD:
                                ray.cancel(future, force=True)
                                raise RuntimeError(
                                    f"Worker node {record.node_id[:12]} died "
                                    f"({_node_dead_consecutive} consecutive checks)"
                                )
                        else:
                            _node_dead_consecutive = 0
                    except RuntimeError:
                        raise
                    except Exception:
                        pass  # node query failed, keep waiting
            result = ray.get(future)

            record.completed_at = time.time()
            record.stdout = result.get("stdout", "")
            record.stderr = result.get("stderr", "")
            record.returncode = result.get("returncode")
            if result.get("timed_out"):
                record.status = "timeout"
            elif record.returncode == 0:
                record.status = "completed"
            else:
                # Non-zero exit — including SIGKILL (137) — is a failure,
                # not a completion.  Marking it "completed" silently
                # hides OOM kills, infra SIGKILLs, and real process
                # errors from downstream consumers.
                record.status = "failed"

        except (SystemExit, KeyboardInterrupt) as e:
            _dbg(f"SystemExit/KeyboardInterrupt: {e}")
            # ray.kill sends SystemExit — if the job never started,
            # keep it queued so the re-dispatch loop can retry it later.
            if record.started_at is None:
                record.status = "queued"
                record.dispatch_resource = None
                for p in pgs:
                    try:
                        remove_placement_group(p)
                    except Exception:
                        pass
                pgs.clear()
                record.save_meta()
                log.info("Job %s: actor killed before job started, keeping queued",
                         job_id)
                return
            # Job was already running — cancel the _run_job task so its
            # subprocess gets killed and GPU memory is released.
            if future is not None:
                try:
                    ray.cancel(future, force=True)
                except Exception:
                    pass
            raise

        except Exception as e:
            _dbg(f"Exception: {type(e).__name__}: {e}")
            # Exceptions here are infrastructure failures (node death, worker
            # crash, object lost) — normal command failures return data from
            # _run_job and never raise.  Always requeue on infra failure so
            # the job runs on another node; the caller is not notified until
            # the job genuinely completes or fails.

            retry_msg = f"Worker lost (attempt {record.attempt + 1}): {e}"
            retried = True
            # attempt is bumped in requeue_preempted; the +1 in the
            # message above reflects the number the retry will run on.
            record.status = "preempted"
            record.started_at = None
            record.completed_at = None
            event_message = f"[hive] {retry_msg}"

        finally:
            # Clean up all placement groups (winner + any losers still alive)
            for p in pgs:
                try:
                    remove_placement_group(p)
                except Exception:
                    pass
            # Four paths reach this finally block:
            # 1. PG timeout: pgs.clear() already called, early return — guard below skips (pgs empty).
            # 2. SystemExit (ray.kill before job started): caught above, pgs cleared, early return.
            # 3. Exception (node death etc): except block sets status to preempted/failed — guard skips.
            # 4. Legitimate preemption (ray.kill during run): status still "queued", pgs non-empty — retry.
            if record.status == "queued" and pgs:
                if record.attempt < record.max_retries:
                    retried = True
                    # attempt is bumped in requeue_preempted, not here.
                    record.status = "preempted"
                    record.started_at = None
                    record.completed_at = None
                    event_message = (
                        f"[hive] Preempted (attempt {record.attempt}"
                        f"/{record.max_retries})"
                    )
                else:
                    record.status = "failed"
                    record.completed_at = time.time()
                    record.returncode = -9
                    event_message = "[hive] Job runner killed before job could start"
            record, retried = JobRecord.finalize_runner_record(
                record,
                retried=retried,
                event_message=event_message,
            )

            # Run on_complete callback only on final resolution.  The
            # return-to-queue paths above leave status "queued" with
            # retried=False, so check for a terminal status explicitly.
            if not retried and record.status in (
                "completed", "failed", "cancelled", "timeout",
            ):
                if record.on_complete:
                    try:
                        env = os.environ.copy()
                        env["HIVE_JOB_ID"] = record.job_id
                        env["HIVE_STATUS"] = record.status
                        env["HIVE_EXIT_CODE"] = str(record.returncode or 0)
                        subprocess.run(record.on_complete, shell=True, env=env)
                    except Exception:
                        pass

            # Self-destruct: detached actor must clean itself up.
            # ray.actor.exit_actor() does not reliably remove detached
            # actors from the registry, so use ray.kill() instead.
            try:
                self_handle = ray.get_runtime_context().current_actor
                ray.kill(self_handle)
            except Exception:
                try:
                    ray.actor.exit_actor()
                except Exception:
                    pass
