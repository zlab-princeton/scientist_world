"""Tests for Swarm configuration."""

import pytest
import yaml

from swarm.config import (
    SwarmConfig,
    WorkerConfig,
    NexusConnectionConfig,
    ResourcesConfig,
    WorkflowStep,
    load_config,
)


class TestSwarmConfig:
    """SwarmConfig dataclass defaults."""

    def test_defaults(self):
        config = SwarmConfig()
        assert config.pool_size == 4
        assert config.dashboard_port == 39100
        assert isinstance(config.worker, WorkerConfig)
        assert isinstance(config.nexus, NexusConnectionConfig)

    def test_worker_defaults(self):
        # No scaffold or model is built in: a profile has to name them.
        w = WorkerConfig()
        assert w.scaffold == ""
        assert w.model == ""
        assert w.max_turns == 200
        assert w.container_sif is None
        assert w.effort is None

    def test_nexus_defaults(self):
        n = NexusConnectionConfig()
        assert n.url == ""
        assert n.autostart_host == "127.0.0.1"
        assert n.gpu_budget_hours == 50.0
        assert n.max_gpus == 4
        assert n.job_time_limit == "24:00:00"
        assert isinstance(n.resources, ResourcesConfig)
        assert n.resources.all_resources() == []

    def test_resources_pick(self):
        r = ResourcesConfig(
            gpu=["gpu-a", "gpu-b"],
            gpu_test=["gpu-a-test", "gpu-test"],
            cpu=["cpu-a"],
            cpu_test=["cpu-test"],
        )
        assert r.pick(gpus=1, test=False) == "gpu-a,gpu-b"
        assert r.pick(gpus=2, test=True) == "gpu-a-test,gpu-test"
        assert r.pick(gpus=0, test=False) == "cpu-a"
        assert r.pick(gpus=0, test=True) == "cpu-test"
        assert r.all_resources() == ["gpu-a", "gpu-b", "gpu-a-test", "gpu-test", "cpu-a", "cpu-test"]

    def test_all_resources_deduplicates(self):
        r = ResourcesConfig(
            gpu=["gpu-a", "gpu-b", "gpu-test"],
            gpu_test=["gpu-test"],
            cpu=["cpu-a"],
            cpu_test=["cpu-a"],
        )
        assert r.all_resources() == ["gpu-a", "gpu-b", "gpu-test", "cpu-a"]

    def test_resources_pick_empty_raises(self):
        r = ResourcesConfig()
        with pytest.raises(ValueError, match="No resources configured"):
            r.pick(gpus=1, test=False)

    def test_split_runtime_caps_sum_into_total_workers(self):
        config = SwarmConfig(
            max_working_workers=50,
            max_admitted_workers=40,
            max_building_workers=2,
            max_installing_workers=4,
        )
        assert config.split_runtime_caps_enabled() is True
        assert config.effective_max_admitted_workers() == 40
        assert config.effective_max_building_workers() == 2
        assert config.effective_max_installing_workers() == 4
        assert config.effective_max_runtime_workers() == 46
        assert config.effective_total_workers() == 96

    def test_working_return_overflow_adds_to_total_capacity(self):
        config = SwarmConfig(
            max_working_workers=50,
            max_working_return_overflow=6,
            max_admitted_workers=10,
            max_building_workers=6,
            max_installing_workers=12,
        )
        assert config.effective_max_working_return_overflow() == 6
        assert config.effective_total_workers() == 84

    def test_explicit_admitted_cap(self):
        config = SwarmConfig(
            max_working_workers=50,
            max_admitted_workers=10,
            max_building_workers=6,
            max_installing_workers=12,
        )
        assert config.split_runtime_caps_enabled() is True
        assert config.effective_max_admitted_workers() == 10
        assert config.effective_total_workers() == 78

    def test_explicit_pool_size_overrides_sum_of_caps(self):
        config = SwarmConfig(
            pool_size=10,
            max_working_workers=8,
            max_working_return_overflow=8,
            max_admitted_workers=3,
            max_building_workers=4,
            max_installing_workers=12,
        )
        assert config.effective_total_workers() == 10
        assert config.effective_max_working_workers() == 8

    def test_container_and_dashboard_defaults(self):
        config = SwarmConfig()
        assert config.dashboard_host == "127.0.0.1"
        assert config.local_tmp_dir == "/tmp"
        assert config.container_memory_limit == "8G"
        assert config.container_nice == 19
        assert config.container_cpu_threads == 2

    def test_container_and_dashboard_overrides(self):
        config = SwarmConfig(
            dashboard_host="0.0.0.0",
            local_tmp_dir="/local/tmp",
            container_memory_limit="16G",
            container_nice=5,
            container_cpu_threads=4,
        )
        assert config.dashboard_host == "0.0.0.0"
        assert config.local_tmp_dir == "/local/tmp"
        assert config.container_memory_limit == "16G"
        assert config.container_nice == 5
        assert config.container_cpu_threads == 4


class TestLoadConfig:
    """Config loading from YAML."""

    def test_load_from_yaml(self, tmp_swarm_dir):
        """Load config from a YAML file in SWARM_DIR."""
        config_yaml = {
            "pool_size": 8,
            "dashboard_port": 39200,
            "worker": {
                "model": "claude-sonnet-4-5-20250929",
                "max_turns": 100,
            },
            "nexus": {
                "gpu_budget_hours": 20.0,
                "max_gpus": 8,
            },
        }

        config_path = tmp_swarm_dir / "config.yaml"
        config_path.write_text(yaml.dump(config_yaml))

        config = load_config()

        assert config.pool_size == 8
        assert config.dashboard_port == 39200
        assert config.worker.model == "claude-sonnet-4-5-20250929"
        assert config.worker.max_turns == 100
        assert config.nexus.gpu_budget_hours == 20.0
        assert config.nexus.max_gpus == 8

    def test_load_resources_from_yaml(self, tmp_swarm_dir):
        """Load categorized resources from YAML."""
        config_yaml = {
            "nexus": {
                "resources": {
                    "gpu": ["gpu-a", "gpu-b"],
                    "gpu_test": ["gpu-a-test"],
                    "cpu": ["cpu-a"],
                    "cpu_test": ["cpu-test"],
                },
            },
        }
        config_path = tmp_swarm_dir / "config.yaml"
        config_path.write_text(yaml.dump(config_yaml))

        config = load_config()
        assert config.nexus.resources.gpu == ["gpu-a", "gpu-b"]
        assert config.nexus.resources.gpu_test == ["gpu-a-test"]
        assert config.nexus.resources.cpu == ["cpu-a"]
        assert config.nexus.resources.cpu_test == ["cpu-test"]
        assert len(config.nexus.resources.all_resources()) == 5

    def test_load_defaults_when_no_file(self, tmp_swarm_dir):
        """Fall back to defaults when no config file exists."""
        config = load_config()
        assert config.pool_size == 4
        assert config.worker.scaffold == ""
        assert config.dashboard_host == "127.0.0.1"

    def test_load_split_runtime_caps_from_yaml(self, tmp_swarm_dir):
        config_yaml = {
            "max_working_workers": 50,
            "max_admitted_workers": 40,
            "max_building_workers": 2,
            "max_installing_workers": 4,
        }
        config_path = tmp_swarm_dir / "config.yaml"
        config_path.write_text(yaml.dump(config_yaml))

        config = load_config()
        assert config.split_runtime_caps_enabled() is True
        assert config.effective_max_admitted_workers() == 40
        assert config.effective_max_building_workers() == 2
        assert config.effective_max_installing_workers() == 4
        assert config.effective_total_workers() == 96

    def test_load_admitted_cap_from_yaml(self, tmp_swarm_dir):
        config_yaml = {
            "max_working_workers": 50,
            "max_admitted_workers": 10,
            "max_building_workers": 6,
            "max_installing_workers": 12,
        }
        config_path = tmp_swarm_dir / "config.yaml"
        config_path.write_text(yaml.dump(config_yaml))

        config = load_config()
        assert config.split_runtime_caps_enabled() is True
        assert config.effective_max_admitted_workers() == 10
        assert config.effective_total_workers() == 78

    def test_load_working_return_overflow_from_yaml(self, tmp_swarm_dir):
        config_yaml = {
            "max_working_workers": 50,
            "max_working_return_overflow": 6,
            "max_admitted_workers": 10,
            "max_building_workers": 6,
            "max_installing_workers": 12,
        }
        config_path = tmp_swarm_dir / "config.yaml"
        config_path.write_text(yaml.dump(config_yaml))

        config = load_config()
        assert config.effective_max_working_return_overflow() == 6
        assert config.effective_total_workers() == 84

    def test_unknown_keys_ignored(self, tmp_swarm_dir):
        """Unknown YAML keys do not cause errors."""
        config_yaml = {
            "pool_size": 2,
            "unknown_key": "should be ignored",
            "worker": {
                "unknown_field": 42,
            },
        }
        config_path = tmp_swarm_dir / "config.yaml"
        config_path.write_text(yaml.dump(config_yaml))

        config = load_config()
        assert config.pool_size == 2

    def test_load_workflow_step_gpu_budget(self, tmp_swarm_dir):
        """Workflow steps can set a default per-iteration GPU budget."""
        config_yaml = {
            "profiles": {
                "fast": {"scaffold": "codex", "model": "gpt-5.4"},
            },
            "workflows": {
                "standard": [
                    {
                        "profile": "fast",
                        "gpu_budget_hours": 12.5,
                        "milestone_cap": "core_claim",
                    }
                ]
            },
        }
        config_path = tmp_swarm_dir / "config.yaml"
        config_path.write_text(yaml.dump(config_yaml))

        config = load_config()
        assert "standard" in config.workflows
        assert len(config.workflows["standard"]) == 1
        step = config.workflows["standard"][0]
        assert isinstance(step, WorkflowStep)
        assert step.profile == "fast"
        assert step.gpu_budget_hours == 12.5

    def test_load_workflow_fallback_profiles(self, tmp_swarm_dir):
        """Workflow steps can have fallback_profiles for API limit failover."""
        config_yaml = {
            "profiles": {
                "primary": {"scaffold": "codex", "model": "gpt-5.4"},
                "backup": {"scaffold": "codex", "model": "gpt-5.4-mini"},
            },
            "workflows": {
                "standard": [
                    {
                        "profile": "primary",
                        "fallback_profiles": ["backup"],
                    }
                ]
            },
        }
        config_path = tmp_swarm_dir / "config.yaml"
        config_path.write_text(yaml.dump(config_yaml))

        config = load_config()
        step = config.workflows["standard"][0]
        assert step.fallback_profiles == ["backup"]

    def test_load_workflow_fallback_unknown_profile_raises(self, tmp_swarm_dir):
        """fallback_profiles referencing unknown profiles should raise."""
        config_yaml = {
            "profiles": {
                "primary": {"scaffold": "codex", "model": "gpt-5.4"},
            },
            "workflows": {
                "standard": [
                    {
                        "profile": "primary",
                        "fallback_profiles": ["nonexistent"],
                    }
                ]
            },
        }
        config_path = tmp_swarm_dir / "config.yaml"
        config_path.write_text(yaml.dump(config_yaml))

        with pytest.raises(ValueError, match="nonexistent"):
            load_config()

    def test_env_override_nexus_url(self, tmp_swarm_dir, monkeypatch):
        """NEXUS_URL env var overrides empty config."""
        monkeypatch.setenv("NEXUS_URL", "http://test:39000")
        config = load_config()
        assert config.nexus.url == "http://test:39000"

    def test_env_override_scout_data_dir(self, tmp_swarm_dir, monkeypatch):
        """SCOUT_DATA_DIR env var overrides empty config."""
        monkeypatch.setenv("SCOUT_DATA_DIR", "/data/scout")
        config = load_config()
        assert config.scout_data_dir == "/data/scout"


class TestConfigValidation:
    """Names that Swarm does not know are rejected when the config loads."""

    def _write(self, tmp_swarm_dir, config_yaml):
        (tmp_swarm_dir / "config.yaml").write_text(yaml.dump(config_yaml))

    def _workflow(self, **step):
        return {
            "profiles": {"p": {"scaffold": "codex", "model": "m"}},
            "workflows": {"w": {"steps": [{"profile": "p", **step}]}},
        }

    def test_load_dashboard_host_and_nexus_autostart_host(self, tmp_swarm_dir):
        self._write(tmp_swarm_dir, {
            "dashboard_host": "0.0.0.0",
            "nexus": {"autostart_host": "0.0.0.0"},
        })
        config = load_config()
        assert config.dashboard_host == "0.0.0.0"
        assert config.nexus.autostart_host == "0.0.0.0"

    def test_unknown_role_raises(self, tmp_swarm_dir):
        self._write(tmp_swarm_dir, self._workflow(role="verifer"))
        with pytest.raises(ValueError, match="unknown role 'verifer'"):
            load_config()

    @pytest.mark.parametrize("role", [
        "reproducer", "designer", "designer_stage_1", "designer_stage_2",
        "designer_stage_3", "packager", "verifier",
    ])
    def test_known_roles_load(self, tmp_swarm_dir, role):
        self._write(tmp_swarm_dir, self._workflow(role=role))
        assert load_config().workflows["w"][0].role == role

    def test_role_defaults_to_reproducer(self, tmp_swarm_dir):
        self._write(tmp_swarm_dir, self._workflow())
        assert load_config().workflows["w"][0].role == "reproducer"

    @pytest.mark.parametrize("key", ["min_milestone", "milestone_cap"])
    def test_unknown_milestone_raises(self, tmp_swarm_dir, key):
        self._write(tmp_swarm_dir, self._workflow(**{key: "core_claims"}))
        with pytest.raises(ValueError, match="is not a milestone"):
            load_config()

    def test_unknown_target_milestone_raises(self, tmp_swarm_dir):
        self._write(tmp_swarm_dir, {"target_milestone": "core-claim"})
        with pytest.raises(ValueError, match="target_milestone"):
            load_config()

    def test_unknown_profile_scaffold_raises(self, tmp_swarm_dir):
        self._write(tmp_swarm_dir, {
            "profiles": {"p": {"scaffold": "claude_code", "model": "m"}},
        })
        with pytest.raises(ValueError, match="unknown scaffold 'claude_code'"):
            load_config()

    def test_unknown_worker_scaffold_raises(self, tmp_swarm_dir):
        self._write(tmp_swarm_dir, {"worker": {"scaffold": "other", "model": "m"}})
        with pytest.raises(ValueError, match="worker.scaffold"):
            load_config()

    def test_revert_requires_min_milestone(self, tmp_swarm_dir):
        self._write(tmp_swarm_dir, self._workflow(revert_to_iteration=1))
        with pytest.raises(ValueError, match="requires min_milestone"):
            load_config()

    def test_revert_cannot_point_forward(self, tmp_swarm_dir):
        self._write(
            tmp_swarm_dir,
            self._workflow(min_milestone="core_claim", revert_to_iteration=2),
        )
        with pytest.raises(ValueError, match="must be <= current iteration"):
            load_config()

    def test_workflow_meta_and_shorthand(self, tmp_swarm_dir):
        self._write(tmp_swarm_dir, {
            "profiles": {"p": {"scaffold": "codex", "model": "m"}},
            "workflows": {
                "with_meta": {"max_revert_attempts": 2, "steps": [{"profile": "p"}]},
                "shorthand": [{"profile": "p"}],
            },
        })
        config = load_config()
        assert config.workflow_meta["with_meta"].max_revert_attempts == 2
        assert config.workflow_meta["shorthand"].max_revert_attempts == 0
        assert len(config.workflows["shorthand"]) == 1

    def test_empty_sections_are_tolerated(self, tmp_swarm_dir):
        # "profiles:" / "workflows:" / "worker:" with nothing below parse as None.
        (tmp_swarm_dir / "config.yaml").write_text(
            "profiles:\nworkflows:\nworker:\npool_size: 3\n"
        )
        config = load_config()
        assert config.pool_size == 3
        assert config.workflows == {}

    def test_empty_nexus_and_profile_bodies_are_tolerated(self, tmp_swarm_dir):
        from swarm.config import ResourcesConfig

        path = tmp_swarm_dir / "config.yaml"
        path.write_text("nexus:\npool_size: 3\n")
        assert load_config().pool_size == 3

        path.write_text("nexus:\n  resources:\n")
        assert load_config().nexus.resources == ResourcesConfig()

        path.write_text("profiles:\n  p:\nworker:\n  profile: p\n")
        assert load_config().worker.profile == "p"

    def test_milestone_names_match_scheduler_and_worker(self):
        from infrastructure.worker.parsing import VALID_MILESTONES
        from swarm.config import MILESTONES, WORKFLOW_ROLES
        from swarm.scheduler import (
            _MILESTONE_LOCKED_ROLES,
            _MILESTONE_RANK,
            _NO_REPRODUCE_START_ROLES,
        )

        assert set(MILESTONES) == set(_MILESTONE_RANK) == set(VALID_MILESTONES)
        # The ladder order in the config module matches the scheduler's ranks.
        assert sorted(MILESTONES, key=_MILESTONE_RANK.get) == list(MILESTONES)
        assert _MILESTONE_LOCKED_ROLES <= set(WORKFLOW_ROLES)
        assert _NO_REPRODUCE_START_ROLES <= set(WORKFLOW_ROLES)
