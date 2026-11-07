"""Nexus project management: registration, quotas, GPU-time budgets.

A project represents a bounded compute allocation for an AI agent.
Each project has a GPU quota, per-job timeout defaults, and an optional
soft GPU-time budget.  Projects are persisted as JSON files under
PROJECTS_DIR.

Lifecycle:
    registered -> active

A project becomes "active" when its first job is submitted (the daemon
submit route makes the transition).  Deleting a project removes its
record; there is no retired state.

The GPU-time budget accumulates GPU-seconds over finished jobs.  It is
"soft" because a running job is never stopped for exceeding it: once
the budget is used up and the project has no active job, the monitor
sets ``budget_exhausted`` and further submissions are rejected until
the budget is raised or the counter is reset.
"""

import json
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Tuple, List

from .config import PROJECTS_DIR

# Global lock for Project read-modify-write operations.
# Protects against concurrent saves from the monitor thread
# and Flask request handling thread.
_project_lock = threading.Lock()


@dataclass
class Project:
    """A compute project with GPU quotas and soft GPU-time budgets.

    Attributes:
        project_id: Unique identifier for this project.
        max_gpus: Maximum GPUs that can be in use simultaneously.
        allowed_resources: List of Hive resource names this project may target.
        max_cpus: Maximum CPUs allowed per job (0 = unlimited).
        max_memory_mb: Maximum memory in MB allowed per job (0 = unlimited).
        default_job_timeout: Default per-job timeout in seconds.
        soft_gpu_budget_hours: Soft cumulative GPU-time budget (0 = unlimited).
        workspace_dir: Agent's workspace directory (bind-mounted into containers).
        sif_path: Path to the Apptainer SIF image.
        overlay_path: Optional overlay image, mounted read-only into job containers.
        hf_cache_path: Optional host directory mounted as the shared model cache.
        wake_command: Command to invoke when the agent should be woken at checkpoints.
        shared_dirs: Additional directories to bind-mount into containers.
        status: Current project state (registered, active).
        gpus_in_use: Number of GPUs currently allocated to running jobs.
        gpu_seconds_used: Accumulated GPU-seconds across completed jobs.
        budget_exhausted: True when soft budget exceeded and all jobs finished.
        created_at: Unix timestamp when the project was registered.
        checkpoint_interval: Per-project checkpoint interval override (seconds).
        checkpoint_timeout: Per-project checkpoint timeout override (seconds).
    """
    project_id: str
    max_gpus: int = 4
    allowed_resources: List[str] = field(default_factory=list)
    max_cpus: int = 0
    max_memory_mb: int = 0
    default_job_timeout: int = 3600
    soft_gpu_budget_hours: float = 0
    workspace_dir: str = ""
    sif_path: str = ""
    overlay_path: str = ""
    hf_cache_path: str = ""
    wake_command: Optional[str] = None
    shared_dirs: List[str] = field(default_factory=list)
    status: str = "registered"
    gpus_in_use: int = 0
    gpu_seconds_used: float = 0
    budget_exhausted: bool = False
    created_at: float = field(default_factory=time.time)
    checkpoint_interval: Optional[int] = None
    checkpoint_timeout: Optional[int] = None

    def to_dict(self) -> dict:
        """Serialize project to a JSON-safe dictionary."""
        d = {
            "project_id": self.project_id,
            "max_gpus": self.max_gpus,
            "allowed_resources": self.allowed_resources,
            "max_cpus": self.max_cpus,
            "max_memory_mb": self.max_memory_mb,
            "default_job_timeout": self.default_job_timeout,
            "soft_gpu_budget_hours": self.soft_gpu_budget_hours,
            "workspace_dir": self.workspace_dir,
            "sif_path": self.sif_path,
            "overlay_path": self.overlay_path,
            "hf_cache_path": self.hf_cache_path,
            "shared_dirs": self.shared_dirs,
            "status": self.status,
            "gpus_in_use": self.gpus_in_use,
            "gpu_seconds_used": self.gpu_seconds_used,
            "budget_exhausted": self.budget_exhausted,
            "created_at": self.created_at,
        }
        if self.wake_command is not None:
            d["wake_command"] = self.wake_command
        if self.checkpoint_interval is not None:
            d["checkpoint_interval"] = self.checkpoint_interval
        if self.checkpoint_timeout is not None:
            d["checkpoint_timeout"] = self.checkpoint_timeout
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        """Deserialize a project from a dictionary."""
        return cls(
            project_id=d["project_id"],
            max_gpus=d.get("max_gpus", 4),
            allowed_resources=d.get("allowed_resources") or [],
            max_cpus=d.get("max_cpus", 0),
            max_memory_mb=d.get("max_memory_mb", 0),
            default_job_timeout=d.get("default_job_timeout", 3600),
            soft_gpu_budget_hours=d.get("soft_gpu_budget_hours", 0),
            workspace_dir=d.get("workspace_dir", ""),
            sif_path=d.get("sif_path", ""),
            overlay_path=d.get("overlay_path", ""),
            hf_cache_path=d.get("hf_cache_path", ""),
            wake_command=d.get("wake_command"),
            shared_dirs=d.get("shared_dirs", []),
            status=d.get("status", "registered"),
            gpus_in_use=d.get("gpus_in_use", 0),
            gpu_seconds_used=d.get("gpu_seconds_used", 0),
            budget_exhausted=d.get("budget_exhausted", False),
            created_at=d.get("created_at", 0.0),
            checkpoint_interval=d.get("checkpoint_interval"),
            checkpoint_timeout=d.get("checkpoint_timeout"),
        )

    @property
    def _file_path(self) -> Path:
        """Path to this project's JSON file on disk."""
        return Path(PROJECTS_DIR) / f"{self.project_id}.json"

    def save(self):
        """Persist project state to disk (thread-safe)."""
        with _project_lock:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, project_id: str) -> Optional["Project"]:
        """Load a project from disk by ID. Returns None if not found."""
        path = Path(PROJECTS_DIR) / f"{project_id}.json"
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, KeyError):
            return None

    @classmethod
    def load_all(cls) -> List["Project"]:
        """Load all projects from the projects directory."""
        d = Path(PROJECTS_DIR)
        if not d.exists():
            return []
        projects = []
        for path in sorted(d.glob("*.json")):
            try:
                projects.append(cls.from_dict(json.loads(path.read_text())))
            except Exception:
                pass
        return projects

    def delete(self):
        """Remove this project's JSON file from disk."""
        if self._file_path.exists():
            self._file_path.unlink()

    # The three methods below change one counter. They change it on the
    # state on disk and write that back, not this object: a job keeps the
    # Project it was submitted with for as long as it runs, and writing that
    # copy back would undo any setting changed in the meantime and drop the
    # GPU time other jobs of the project recorded.

    def _write(self) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(json.dumps(self.to_dict(), indent=2))

    def _sync_from(self, other: "Project") -> None:
        """Take every field from *other*, the state on disk."""
        for f in fields(self):
            setattr(self, f.name, getattr(other, f.name))

    def is_over_soft_budget(self) -> bool:
        """Check if cumulative GPU-time exceeds the soft budget.

        Returns False if the budget is unlimited (soft_gpu_budget_hours == 0).
        """
        return (self.soft_gpu_budget_hours > 0
                and self.gpu_seconds_used >= self.soft_gpu_budget_hours * 3600)

    def record_gpu_usage(self, num_gpus: int, elapsed_sec: float):
        """Record GPU-time from a completed job. Atomic read-modify-write.

        Like ``release_gpus``, this skips the disk write when the project
        file has been deleted, so that a deleted project is not recreated
        from stale in-memory state.
        """
        with _project_lock:
            fresh = Project.load(self.project_id)
            if fresh is None:
                self.gpu_seconds_used += num_gpus * elapsed_sec
                return
            fresh.gpu_seconds_used += num_gpus * elapsed_sec
            fresh._write()
            self._sync_from(fresh)

    def can_submit(self, num_gpus: int, num_cpus: int = 1,
                   memory_mb: int = 0) -> Tuple[bool, str]:
        """Check whether a job with the given resources can be submitted.

        Validates budget, GPU quota, CPU limit, and memory limit.

        Args:
            num_gpus: Number of GPUs requested.
            num_cpus: Number of CPUs requested.
            memory_mb: Memory in MB requested (0 = no specific request).

        Returns:
            (True, "") if the submission is allowed.
            (False, "error message") if the submission is denied.
        """
        if self.budget_exhausted:
            return False, "soft GPU budget exhausted"
        # A negative count would pass the quota check and lower gpus_in_use.
        if not isinstance(num_gpus, int) or num_gpus < 0:
            return False, f"invalid GPU count: {num_gpus!r}"
        if num_gpus > self.max_gpus:
            return False, f"requested {num_gpus} GPUs exceeds project limit of {self.max_gpus}"
        if self.gpus_in_use + num_gpus > self.max_gpus:
            return (
                False,
                f"insufficient GPU quota: {self.gpus_in_use}/{self.max_gpus} in use, "
                f"requested {num_gpus}",
            )
        if self.max_cpus > 0 and num_cpus > self.max_cpus:
            return False, f"requested {num_cpus} CPUs exceeds project limit of {self.max_cpus}"
        if self.max_memory_mb > 0 and memory_mb > self.max_memory_mb:
            return False, f"requested {memory_mb} MB exceeds project limit of {self.max_memory_mb} MB"
        return True, ""

    def check_and_acquire(self, num_gpus: int, num_cpus: int = 1,
                          memory_mb: int = 0) -> tuple:
        """Run can_submit and reserve the GPUs atomically. Returns (ok, reason)."""
        with _project_lock:
            fresh = Project.load(self.project_id)
            if fresh is not None:
                self._sync_from(fresh)
            ok, reason = self.can_submit(num_gpus, num_cpus, memory_mb)
            if not ok:
                return (False, reason)
            self.gpus_in_use += num_gpus
            self._write()
            return (True, "")

    def release_gpus(self, num_gpus: int):
        """Release GPUs when a job completes. Atomic read-modify-write.

        If the project file no longer exists (the project was deleted
        while the job was finishing), only the in-memory counter is
        updated.  Writing the file would recreate the deleted project
        with a stale ``gpus_in_use`` and block later submissions to a
        new project of the same ID.
        """
        with _project_lock:
            fresh = Project.load(self.project_id)
            if fresh is None:
                self.gpus_in_use = max(0, self.gpus_in_use - num_gpus)
                return
            fresh.gpus_in_use = max(0, fresh.gpus_in_use - num_gpus)
            fresh._write()
            self._sync_from(fresh)
