"""Swarm configuration management.

Mirrors the pattern from nexus/config.py. Configuration is loaded from
YAML files with a priority chain:

    1. SWARM_DIR/config.yaml  (user override, outside version control)
    2. swarm/config/config.yaml  (package-level, gitignored)
    3. Built-in dataclass defaults

All runtime state (paper records, PID files) lives under SWARM_DIR,
which defaults to <project>/.swarm/ to avoid filling the home directory
on shared clusters.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Runtime directory — use SWARM_DIR env var or default to <project>/.swarm/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWARM_DIR = Path(os.environ["SWARM_DIR"]) if os.environ.get("SWARM_DIR") else _PROJECT_ROOT / ".swarm"
PACKAGE_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.yaml"
USER_CONFIG_PATH = SWARM_DIR / "config.yaml"
PAPERS_DIR = SWARM_DIR / "papers"
DAEMON_PID_PATH = SWARM_DIR / "daemon.pid"

# Agent scaffolds the Worker package implements.
SUPPORTED_SCAFFOLDS = ("claude-code", "codex")

# Roles a workflow step may name. "reproducer" is the default.
WORKFLOW_ROLES = (
    "reproducer",
    "designer",
    "designer_stage_1",
    "designer_stage_2",
    "designer_stage_3",
    "packager",
    "verifier",
)

# Milestone ladder, lowest to highest ("not_possible" ranks below "none").
MILESTONES = (
    "not_possible",
    "none",
    "method_runs",
    "core_claim",
    "core_claim_plus",
    "secondary_claims",
    "majority",
    "near_complete",
    "full",
)

# Top-level scalar keys read from config.yaml (the sections ``profiles``,
# ``worker``, ``workflows`` and ``nexus`` are handled separately).
_TOP_LEVEL_KEYS = (
    "pool_size", "max_working_workers", "max_working_return_overflow",
    "max_runtime_workers", "max_admitted_workers",
    "max_building_workers", "max_installing_workers",
    "scout_data_dir", "workspace_base_dir",
    "shared_datasets_path", "shared_models_path", "hf_cache_path",
    "container_cache_dir", "uv_cache_dir",
    "apptainer_cache_dir", "local_tmp_dir",
    "container_memory_limit", "container_nice", "container_cpu_threads",
    "dashboard_host", "dashboard_port", "github_org",
    "target_milestone", "push_forward_attempts", "default_workflow",
)

_LEGACY_POOL_DEFAULT = 4
_SPLIT_CAP_DEFAULT = 50
_ADMITTED_CAP_DEFAULT = 10
_BUILDING_CAP_DEFAULT = 2
_INSTALLING_CAP_DEFAULT = 4
_POOL_SIZE_SENTINEL = object()


class _PoolSizeValue(int):
    """Int-like pool size that preserves whether the value was explicit."""

    explicit: bool

    def __new__(cls, value: int, explicit: bool = False):
        obj = int.__new__(cls, value)
        obj.explicit = explicit
        return obj

    def __reduce_ex__(self, protocol):
        return (self.__class__, (int(self), self.explicit))


def _pool_size_representer(dumper, data):
    return dumper.represent_int(int(data))


yaml.add_representer(_PoolSizeValue, _pool_size_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(_PoolSizeValue, _pool_size_representer, Dumper=yaml.Dumper)


@dataclass
class WorkerConfig:
    """Worker (agent scaffold) settings.

    Attributes:
        scaffold: Agent scaffold, one of ``SUPPORTED_SCAFFOLDS``. Empty
            until a profile (or an explicit ``worker.scaffold``) sets it.
        model: LLM model name passed to the CLI's --model flag.
            For third-party providers using a gateway, this may be an
            alias (e.g. "opus") rather than the real model name.
        actual_model: The real model name for records/manifest.
            Set this when ``model`` is a gateway alias. If None,
            ``model`` is used as-is.
        profile: Name of a predefined profile from the ``profiles`` section.
            Profile values are applied first, then explicit worker fields
            override them.  This lets you switch models by changing one line.
        container_sif: Path to the agent container SIF. If unset the
            agent CLI runs directly on the host, without isolation.
        max_turns: Max agentic turns per worker invocation.
        effort: Optional effort level for the agent.
        env_vars: Extra environment variables for the agent subprocess,
            e.g. the endpoint and token of an API gateway.
        compact_token_limit: Context size (tokens) at which the agent CLI
            compacts its conversation. None keeps the scaffold default.
        setup_timeout: Seconds allowed for one ``setup.sh`` run. Read
            from the ``worker`` section only, not per profile.
        api_keys: Keys to rotate through when the provider reports a
            hard usage limit.
    """
    scaffold: str = ""
    model: str = ""
    actual_model: Optional[str] = None
    profile: Optional[str] = None
    container_sif: Optional[str] = None
    max_turns: int = 200
    effort: Optional[str] = None
    env_vars: dict = field(default_factory=dict)
    compact_token_limit: Optional[int] = None
    setup_timeout: int = 3600
    api_keys: list = field(default_factory=list)


@dataclass
class WorkflowStep:
    """One step in a multi-model workflow.

    Attributes:
        profile: Model profile name for this step.
        gpu_budget_hours: Default GPU-hour budget for this iteration.
            If omitted, falls back to nexus.gpu_budget_hours. This is
            only the default for the step and can still be overridden.
        milestone_cap: Stop the agent when it reaches this milestone
            and advance to the next step. "full" means no cap.
        role: Builder role for this step, one of ``WORKFLOW_ROLES``.
            "reproducer" is the default.
        min_milestone: Gate checked when the step ends. If the paper is
            below this milestone the workflow reverts to
            ``revert_to_iteration`` (while revert attempts remain) or
            stops with status "unsuccessful". Empty string means no gate.
        fallback_profiles: Ordered list of alternative profiles to try if
            the primary profile's API is unavailable (all keys exhausted).
            Tried in order; the step still uses the same role/budget/caps.
        revert_to_iteration: 1-indexed iteration to re-run when
            ``min_milestone`` is not reached. Must be <= this step's own
            iteration; when equal, the step restarts with a fresh session.
            None means no revert.
    """
    profile: str = ""
    gpu_budget_hours: Optional[float] = None
    milestone_cap: str = "full"
    role: str = "reproducer"
    min_milestone: str = ""
    fallback_profiles: list = field(default_factory=list)
    revert_to_iteration: Optional[int] = None


@dataclass
class WorkflowMeta:
    """Per-workflow configuration outside the step list."""
    # Lifetime cap on revert attempts for any paper running this workflow.
    # 0 disables the revert mechanism for this workflow.
    max_revert_attempts: int = 0


@dataclass
class ResourcesConfig:
    """Categorized Hive resource names for job routing.

    The agent specifies ``gpus`` (0 = CPU) and ``test`` (True/False)
    in ``action.yaml``.  The scheduler picks a resource from the
    matching category.

    Attributes:
        gpu: Normal GPU resources.
        gpu_test: Short-lived GPU test resources.
        cpu: Normal CPU resources.
        cpu_test: Short-lived CPU test resources.
    """
    gpu: list = field(default_factory=list)
    gpu_test: list = field(default_factory=list)
    cpu: list = field(default_factory=list)
    cpu_test: list = field(default_factory=list)

    def all_resources(self) -> list:
        """Flat, deduplicated list of all resource names (for Nexus project creation)."""
        seen: set = set()
        result: list = []
        for r in self.gpu + self.gpu_test + self.cpu + self.cpu_test:
            if r not in seen:
                seen.add(r)
                result.append(r)
        return result

    def pick(self, gpus: int, test: bool) -> str:
        """Select resources based on job type.

        Returns a comma-separated string of all resource names in the
        matching category (Nexus accepts comma-separated resources).

        Raises:
            ValueError: If the matching category has no resources configured.
        """
        if gpus > 0:
            category = "gpu_test" if test else "gpu"
            pool = self.gpu_test if test else self.gpu
        else:
            category = "cpu_test" if test else "cpu"
            pool = self.cpu_test if test else self.cpu
        if not pool:
            raise ValueError(
                f"No resources configured for category '{category}'. "
                f"Add resource names to nexus.resources.{category} in config.yaml."
            )
        return ",".join(pool)


@dataclass
class NexusConnectionConfig:
    """Connection settings for the Nexus project manager.

    Attributes:
        url: Nexus daemon URL (e.g. http://localhost:39000).
        autostart_host: Interface a Nexus daemon started by Swarm binds
            to (Swarm starts one only when ``url`` is unreachable).
        gpu_budget_hours: Default GPU-hours budget per iteration, used
            when the workflow step sets none.
        max_gpus: Maximum GPUs per job.
        job_time_limit: Per-job wall time (HH:MM:SS) told to the agent.
            Informational: the limit that is enforced is the one
            configured in Nexus and on the Hive resource.
        resources: Categorized Hive resource names.
    """
    url: str = ""
    autostart_host: str = "127.0.0.1"
    gpu_budget_hours: float = 50.0
    max_gpus: int = 4
    job_time_limit: str = "24:00:00"
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)


@dataclass
class SwarmConfig:
    """Top-level Swarm configuration.

    Concurrency is limited per worker state. Two modes exist:

    * split mode (any of ``max_admitted_workers``,
      ``max_building_workers``, ``max_installing_workers`` is set):
      separate caps for ``working``, ``admitted`` (papers taken from
      the queue, before their first agent turn), ``building`` and
      ``installing``. Papers waiting on a compute job are not capped.
    * shared mode (none of the three is set): one ``working`` cap and
      one ``max_runtime_workers`` cap shared by idle, waiting, building
      and installing papers.

    Attributes:
        max_working_workers: Maximum concurrent agent turns. Defaults
            to ``pool_size`` when that is set explicitly, else to 50 if
            ``max_runtime_workers`` is set, else to 4.
        max_working_return_overflow: Extra ``working`` slots that only
            papers returning from a job, build or install may use.
        max_runtime_workers: Shared-mode cap (see above). Defaults to
            ``pool_size`` when that is set explicitly, else to 50 if
            ``max_working_workers`` is set, else to 4.
        max_admitted_workers: Split-mode cap on papers taken from the
            queue that have not started their first agent turn.
        max_building_workers: Split-mode cap on concurrent container
            builds.
        max_installing_workers: Split-mode cap on concurrent
            ``setup.sh`` runs.
        pool_size: Total number of papers active at once. When set
            explicitly it overrides the sum of the per-state caps and
            seeds the caps that are omitted.
        scout_data_dir: Path to scout/data/ directory.
        workspace_base_dir: Where per-paper workspaces are created.
        shared_datasets_path: Shared datasets mount path.
        shared_models_path: Shared models mount path.
        dashboard_host: Interface the dashboard binds to.
        dashboard_port: Flask dashboard port.
        worker: Worker scaffold configuration.
        nexus: Nexus connection configuration.
    """
    max_working_workers: Optional[int] = None
    max_working_return_overflow: Optional[int] = None
    max_runtime_workers: Optional[int] = None
    max_admitted_workers: Optional[int] = None
    max_building_workers: Optional[int] = None
    max_installing_workers: Optional[int] = None
    pool_size: int = _LEGACY_POOL_DEFAULT
    scout_data_dir: str = ""
    workspace_base_dir: str = ""
    shared_datasets_path: str = "/shared/datasets"
    shared_models_path: str = "/shared/models"
    hf_cache_path: str = ""  # shared HuggingFace cache (writable by all papers)
    container_cache_dir: str = ""  # shared SIF cache; empty = {workspace_base_dir}/../shared/containers
    uv_cache_dir: str = ""  # shared uv cache for pip→uv shim; empty = system default
    apptainer_cache_dir: str = ""  # APPTAINER_CACHEDIR; empty = Apptainer's default (~/.apptainer)
    local_tmp_dir: str = "/tmp"  # node-local disk for agent SIF copies, builds and overlays
    # ── Per-container resource limits for agent containers ──
    # Agent containers run on the host that runs the daemon, next to each
    # other, so each one is limited.
    # Memory cgroup hard cap, enforced across all processes inside the
    # container. 8G fits normal agent work (agent CLI, package installs,
    # tests) but not loading large models or training inline.
    container_memory_limit: str = "8G"
    # Value for the `nice -n N` wrapper around apptainer exec. A CPU
    # cgroup is often not delegated to user sessions, so CPU sharing
    # relies on scheduler priority. Inherited by all subprocesses.
    container_nice: int = 19
    # Thread cap for BLAS/threading libraries (OMP, MKL, OpenBLAS,
    # NumExpr, VecLib), set through env vars inside the container. This
    # bounds the fan-out of torch/numpy/scipy when no CPU quota exists.
    container_cpu_threads: int = 2
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 39100
    github_org: str = ""
    target_milestone: str = "core_claim"  # papers that error at or above this still get a wrap-up
    push_forward_attempts: int = 1  # times to push agent to keep going when done below target_milestone
    default_workflow: str = "standard"  # workflow used when --workflow is not specified
    workflows: dict[str, list[WorkflowStep]] = field(default_factory=dict)
    # Per-workflow metadata (max_revert_attempts, etc.) — parallel dict keyed
    # by workflow name.  Populated alongside ``workflows`` by load_config.
    workflow_meta: dict[str, WorkflowMeta] = field(default_factory=dict)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    nexus: NexusConnectionConfig = field(default_factory=NexusConnectionConfig)
    _pool_size_explicit: bool = field(default=False, repr=False, compare=False)

    def __init__(
        self,
        max_working_workers: Optional[int] = None,
        max_working_return_overflow: Optional[int] = None,
        max_runtime_workers: Optional[int] = None,
        max_admitted_workers: Optional[int] = None,
        max_building_workers: Optional[int] = None,
        max_installing_workers: Optional[int] = None,
        pool_size: object = _POOL_SIZE_SENTINEL,
        scout_data_dir: str = "",
        workspace_base_dir: str = "",
        shared_datasets_path: str = "/shared/datasets",
        shared_models_path: str = "/shared/models",
        hf_cache_path: str = "",
        container_cache_dir: str = "",
        uv_cache_dir: str = "",
        apptainer_cache_dir: str = "",
        local_tmp_dir: str = "/tmp",
        container_memory_limit: str = "8G",
        container_nice: int = 19,
        container_cpu_threads: int = 2,
        dashboard_host: str = "127.0.0.1",
        dashboard_port: int = 39100,
        github_org: str = "",
        target_milestone: str = "core_claim",
        push_forward_attempts: int = 1,
        default_workflow: str = "standard",
        workflows: Optional[dict[str, list[WorkflowStep]]] = None,
        worker: Optional[WorkerConfig] = None,
        nexus: Optional[NexusConnectionConfig] = None,
        _pool_size_explicit: object = _POOL_SIZE_SENTINEL,
    ) -> None:
        self.max_working_workers = max_working_workers
        self.max_working_return_overflow = max_working_return_overflow
        self.max_runtime_workers = max_runtime_workers
        self.max_admitted_workers = max_admitted_workers
        self.max_building_workers = max_building_workers
        self.max_installing_workers = max_installing_workers
        if pool_size is _POOL_SIZE_SENTINEL:
            pool_value = _PoolSizeValue(_LEGACY_POOL_DEFAULT, explicit=False)
        else:
            if isinstance(pool_size, _PoolSizeValue):
                pool_value = pool_size
            else:
                pool_value = _PoolSizeValue(
                    int(pool_size),  # type: ignore[arg-type]
                    explicit=isinstance(pool_size, int) and pool_size > 0,
                )
        self.pool_size = pool_value
        if _pool_size_explicit is _POOL_SIZE_SENTINEL:
            self._pool_size_explicit = bool(getattr(pool_value, "explicit", False))
        else:
            self._pool_size_explicit = bool(_pool_size_explicit)
            if not self._pool_size_explicit and bool(getattr(pool_value, "explicit", False)):
                self._pool_size_explicit = True
        self.scout_data_dir = scout_data_dir
        self.workspace_base_dir = workspace_base_dir
        self.shared_datasets_path = shared_datasets_path
        self.shared_models_path = shared_models_path
        self.hf_cache_path = hf_cache_path
        self.container_cache_dir = container_cache_dir
        self.uv_cache_dir = uv_cache_dir
        self.apptainer_cache_dir = apptainer_cache_dir
        self.local_tmp_dir = local_tmp_dir
        self.container_memory_limit = container_memory_limit
        self.container_nice = container_nice
        self.container_cpu_threads = container_cpu_threads
        self.dashboard_host = dashboard_host
        self.dashboard_port = dashboard_port
        self.github_org = github_org
        self.target_milestone = target_milestone
        self.push_forward_attempts = push_forward_attempts
        self.default_workflow = default_workflow
        self.workflows = dict(workflows) if workflows is not None else {}
        self.workflow_meta = {}
        self.worker = worker if worker is not None else WorkerConfig()
        self.nexus = nexus if nexus is not None else NexusConnectionConfig()

    def _explicit_legacy_pool_size(self) -> int | None:
        if self._pool_size_explicit and isinstance(self.pool_size, int) and self.pool_size > 0:
            return self.pool_size
        return None

    def _effective_legacy_pool_size(self) -> int:
        explicit = self._explicit_legacy_pool_size()
        if explicit is not None:
            return explicit
        return _LEGACY_POOL_DEFAULT

    def split_runtime_caps_enabled(self) -> bool:
        return any(
            cap is not None
            for cap in (
                self.max_admitted_workers,
                self.max_building_workers,
                self.max_installing_workers,
            )
        )

    def effective_max_working_workers(self) -> int:
        if self.max_working_workers is not None:
            return self.max_working_workers
        explicit_pool = self._explicit_legacy_pool_size()
        if explicit_pool is not None:
            return explicit_pool
        if self.max_runtime_workers is not None:
            return _SPLIT_CAP_DEFAULT
        return self._effective_legacy_pool_size()

    def effective_max_runtime_workers(self) -> int:
        if self.split_runtime_caps_enabled():
            return (
                self.effective_max_admitted_workers()
                + self.effective_max_building_workers()
                + self.effective_max_installing_workers()
            )
        if self.max_runtime_workers is not None:
            return self.max_runtime_workers
        explicit_pool = self._explicit_legacy_pool_size()
        if explicit_pool is not None:
            return explicit_pool
        if self.max_working_workers is not None:
            return _SPLIT_CAP_DEFAULT
        return self._effective_legacy_pool_size()

    def effective_max_working_return_overflow(self) -> int:
        if self.max_working_return_overflow is not None:
            return max(0, self.max_working_return_overflow)
        return 0

    def effective_max_admitted_workers(self) -> int:
        if self.max_admitted_workers is not None:
            return self.max_admitted_workers
        explicit_pool = self._explicit_legacy_pool_size()
        if explicit_pool is not None:
            return min(explicit_pool, _ADMITTED_CAP_DEFAULT)
        return _ADMITTED_CAP_DEFAULT

    def effective_max_building_workers(self) -> int:
        if self.max_building_workers is not None:
            return self.max_building_workers
        if self.max_runtime_workers is not None:
            return min(self.max_runtime_workers, _BUILDING_CAP_DEFAULT)
        explicit_pool = self._explicit_legacy_pool_size()
        if explicit_pool is not None:
            return min(explicit_pool, _BUILDING_CAP_DEFAULT)
        return _BUILDING_CAP_DEFAULT

    def effective_max_installing_workers(self) -> int:
        if self.max_installing_workers is not None:
            return self.max_installing_workers
        if self.max_runtime_workers is not None:
            return min(self.max_runtime_workers, _INSTALLING_CAP_DEFAULT)
        explicit_pool = self._explicit_legacy_pool_size()
        if explicit_pool is not None:
            return min(explicit_pool, _INSTALLING_CAP_DEFAULT)
        return _INSTALLING_CAP_DEFAULT

    def effective_total_workers(self) -> int:
        # If pool_size is explicitly set, it overrides the computed sum.
        explicit_pool = self._explicit_legacy_pool_size()
        if explicit_pool is not None:
            return max(1, explicit_pool)
        if self.split_runtime_caps_enabled():
            return max(
                1,
                self.effective_max_working_workers()
                + self.effective_max_working_return_overflow()
                + self.effective_max_admitted_workers()
                + self.effective_max_building_workers()
                + self.effective_max_installing_workers(),
            )
        return max(
            1,
            self.effective_max_working_workers()
            + self.effective_max_working_return_overflow()
            + self.effective_max_runtime_workers(),
        )


def _find_config_path() -> Optional[Path]:
    """Find config file: SWARM_DIR/config.yaml > package/config.yaml."""
    if USER_CONFIG_PATH.exists():
        return USER_CONFIG_PATH
    if PACKAGE_CONFIG_PATH.exists():
        return PACKAGE_CONFIG_PATH
    return None


def load_config() -> SwarmConfig:
    """Load config, searching SWARM_DIR/config.yaml then package/config.yaml.

    Missing keys fall back to dataclass defaults. Unknown keys are ignored.

    Raises:
        ValueError: If a profile, scaffold, workflow role or milestone
            name is not one Swarm knows, or a workflow is malformed.
    """
    config = SwarmConfig()
    config_path = _find_config_path()
    if config_path:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        # Top-level scalar fields
        for k in _TOP_LEVEL_KEYS:
            if k in raw:
                setattr(config, k, raw[k])
        if "pool_size" in raw:
            config._pool_size_explicit = True
        if config.target_milestone not in MILESTONES:
            raise ValueError(
                f"target_milestone '{config.target_milestone}' is not a milestone. "
                f"Valid: {list(MILESTONES)}"
            )

        # Load worker config: profile first, then explicit fields override
        profiles = raw.get("profiles") or {}
        for prof_name, prof in profiles.items():
            scaffold = (prof or {}).get("scaffold")
            if scaffold is not None and scaffold not in SUPPORTED_SCAFFOLDS:
                raise ValueError(
                    f"Profile '{prof_name}' has unknown scaffold '{scaffold}'. "
                    f"Supported: {list(SUPPORTED_SCAFFOLDS)}"
                )
        worker_raw = raw.get("worker") or {}
        profile_name = worker_raw.get("profile")
        if profile_name:
            if profile_name not in profiles:
                raise ValueError(
                    f"Worker profile '{profile_name}' not found in profiles. "
                    f"Available: {list(profiles.keys())}"
                )
            # Apply profile values as base
            for k, v in (profiles[profile_name] or {}).items():
                if hasattr(config.worker, k):
                    setattr(config.worker, k, v)
            config.worker.profile = profile_name
        # Explicit worker fields override profile
        for k, v in worker_raw.items():
            if k == "profile":
                continue  # already handled
            if hasattr(config.worker, k):
                setattr(config.worker, k, v)
        if config.worker.scaffold and config.worker.scaffold not in SUPPORTED_SCAFFOLDS:
            raise ValueError(
                f"worker.scaffold '{config.worker.scaffold}' is unknown. "
                f"Supported: {list(SUPPORTED_SCAFFOLDS)}"
            )
        # Load workflows — accept two shapes:
        #   workflows:
        #     name:
        #       max_revert_attempts: N          # per-workflow settings
        #       steps: [<step>, <step>]
        #   workflows:
        #     name: [<step>, <step>]            # shorthand: steps only, no reverts
        for wf_name, wf_raw in (raw.get("workflows") or {}).items():
            if isinstance(wf_raw, list):
                steps_raw = wf_raw
                wf_max_reverts = 0
            elif isinstance(wf_raw, dict):
                steps_raw = wf_raw.get("steps", [])
                if not isinstance(steps_raw, list):
                    raise ValueError(
                        f"Workflow '{wf_name}': 'steps' must be a list"
                    )
                wf_max_reverts = wf_raw.get("max_revert_attempts", 0)
                if not isinstance(wf_max_reverts, int) or wf_max_reverts < 0:
                    raise ValueError(
                        f"Workflow '{wf_name}': max_revert_attempts must be a "
                        f"non-negative int, got {wf_max_reverts!r}"
                    )
            else:
                raise ValueError(
                    f"Workflow '{wf_name}': must be a list of steps or a dict "
                    f"with 'steps:' — got {type(wf_raw).__name__}"
                )
            steps = []
            for idx, s in enumerate(steps_raw):
                profile = s.get("profile", "")
                if profile and profile not in profiles:
                    raise ValueError(
                        f"Workflow '{wf_name}' step references unknown profile '{profile}'. "
                        f"Available: {list(profiles.keys())}"
                    )
                fallbacks = s.get("fallback_profiles", [])
                for fb in fallbacks:
                    if fb not in profiles:
                        raise ValueError(
                            f"Workflow '{wf_name}' step fallback references unknown profile '{fb}'. "
                            f"Available: {list(profiles.keys())}"
                        )
                role = s.get("role", "reproducer")
                if role not in WORKFLOW_ROLES:
                    raise ValueError(
                        f"Workflow '{wf_name}' step {idx + 1}: unknown role '{role}'. "
                        f"Valid: {list(WORKFLOW_ROLES)}"
                    )
                for key, default in (("milestone_cap", "full"), ("min_milestone", "")):
                    name = s.get(key, default)
                    if name and name not in MILESTONES:
                        raise ValueError(
                            f"Workflow '{wf_name}' step {idx + 1}: {key} '{name}' "
                            f"is not a milestone. Valid: {list(MILESTONES)}"
                        )
                revert_to = s.get("revert_to_iteration")
                if revert_to is not None:
                    if not isinstance(revert_to, int) or revert_to < 1:
                        raise ValueError(
                            f"Workflow '{wf_name}' step {idx + 1}: "
                            f"revert_to_iteration must be a positive int (1-indexed), got {revert_to!r}"
                        )
                    if revert_to > idx + 1:
                        raise ValueError(
                            f"Workflow '{wf_name}' step {idx + 1}: "
                            f"revert_to_iteration={revert_to} must be <= current iteration ({idx + 1})"
                        )
                    if not s.get("min_milestone"):
                        raise ValueError(
                            f"Workflow '{wf_name}' step {idx + 1}: "
                            f"revert_to_iteration requires min_milestone to be set"
                        )
                steps.append(WorkflowStep(
                    profile=profile,
                    gpu_budget_hours=s.get("gpu_budget_hours"),
                    milestone_cap=s.get("milestone_cap", "full"),
                    role=role,
                    min_milestone=s.get("min_milestone", ""),
                    fallback_profiles=fallbacks,
                    revert_to_iteration=revert_to,
                ))
            config.workflows[wf_name] = steps
            config.workflow_meta[wf_name] = WorkflowMeta(
                max_revert_attempts=wf_max_reverts,
            )

        if "nexus" in raw:
            for k, v in (raw["nexus"] or {}).items():
                if k == "resources" and (v is None or isinstance(v, dict)):
                    for rk, rv in (v or {}).items():
                        if hasattr(config.nexus.resources, rk):
                            setattr(config.nexus.resources, rk, rv)
                elif hasattr(config.nexus, k):
                    setattr(config.nexus, k, v)

    # Env var overrides
    if not config.nexus.url:
        config.nexus.url = os.environ.get("NEXUS_URL", "")
    if not config.scout_data_dir:
        config.scout_data_dir = os.environ.get("SCOUT_DATA_DIR", "")

    return config


def ensure_swarm_dir():
    """Create SWARM_DIR and subdirectories if they don't exist."""
    Path(SWARM_DIR).mkdir(parents=True, exist_ok=True)
    Path(PAPERS_DIR).mkdir(parents=True, exist_ok=True)
