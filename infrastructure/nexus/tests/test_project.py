"""Tests for nexus.project: Project CRUD, quota enforcement, GPU budget."""

from nexus.project import Project


class TestProjectSerialization:
    """Project dataclass serialization round-trip."""

    def test_to_dict_from_dict_roundtrip(self):
        p = Project(
            project_id="test_001",
            max_gpus=4,
            allowed_resources=["gpu", "cpu"],
            default_job_timeout=1800,
            soft_gpu_budget_hours=10.0,
            workspace_dir="/workspaces/test",
            sif_path="/containers/test.sif",
            wake_command="echo {project_id}",
            shared_dirs=["/shared:/shared:ro"],
            checkpoint_interval=1800,
        )
        d = p.to_dict()
        restored = Project.from_dict(d)
        assert restored.project_id == "test_001"
        assert restored.max_gpus == 4
        assert restored.allowed_resources == ["gpu", "cpu"]
        assert restored.default_job_timeout == 1800
        assert restored.soft_gpu_budget_hours == 10.0
        assert restored.workspace_dir == "/workspaces/test"
        assert restored.sif_path == "/containers/test.sif"
        assert restored.wake_command == "echo {project_id}"
        assert restored.shared_dirs == ["/shared:/shared:ro"]
        assert restored.status == "registered"
        assert restored.checkpoint_interval == 1800
        assert restored.gpu_seconds_used == 0
        assert restored.budget_exhausted is False

    def test_from_dict_defaults(self):
        p = Project.from_dict({"project_id": "minimal"})
        assert p.max_gpus == 4
        assert p.default_job_timeout == 3600
        assert p.soft_gpu_budget_hours == 0
        assert p.status == "registered"
        assert p.gpus_in_use == 0
        assert p.gpu_seconds_used == 0
        assert p.budget_exhausted is False
        assert p.wake_command is None
        assert p.shared_dirs == []


class TestProjectPersistence:
    """Save, load, delete operations."""

    def test_save_and_load(self, tmp_nexus_dir):
        p = Project(project_id="persist01", max_gpus=2)
        p.save()

        loaded = Project.load("persist01")
        assert loaded is not None
        assert loaded.project_id == "persist01"
        assert loaded.max_gpus == 2

    def test_load_nonexistent_returns_none(self, tmp_nexus_dir):
        assert Project.load("nonexistent") is None

    def test_load_all_empty(self, tmp_nexus_dir):
        assert Project.load_all() == []

    def test_load_all_multiple(self, tmp_nexus_dir):
        for i in range(3):
            Project(project_id=f"proj{i}", max_gpus=i + 1).save()

        projects = Project.load_all()
        assert len(projects) == 3
        ids = {p.project_id for p in projects}
        assert ids == {"proj0", "proj1", "proj2"}

    def test_save_overwrites(self, tmp_nexus_dir):
        p = Project(project_id="upd", max_gpus=2)
        p.save()

        p.max_gpus = 8
        p.save()

        loaded = Project.load("upd")
        assert loaded.max_gpus == 8

    def test_delete(self, tmp_nexus_dir):
        p = Project(project_id="del01", max_gpus=1)
        p.save()
        assert Project.load("del01") is not None

        p.delete()
        assert Project.load("del01") is None


class TestProjectQuotas:
    """GPU quota enforcement."""

    def test_can_submit_within_quota(self, tmp_nexus_dir):
        p = Project(project_id="q01", max_gpus=4, status="registered")
        p.save()
        ok, msg = p.can_submit(2)
        assert ok is True
        assert msg == ""

    def test_can_submit_exceeds_quota(self, tmp_nexus_dir):
        p = Project(project_id="q02", max_gpus=4, gpus_in_use=3, status="active")
        p.save()
        ok, msg = p.can_submit(2)
        assert ok is False
        assert "GPU" in msg

    def test_can_submit_exactly_at_quota(self, tmp_nexus_dir):
        p = Project(project_id="q03", max_gpus=4, gpus_in_use=2, status="active")
        p.save()
        ok, msg = p.can_submit(2)
        assert ok is True

    def test_can_submit_single_job_exceeds_total(self, tmp_nexus_dir):
        p = Project(project_id="q04", max_gpus=4, status="registered")
        p.save()
        ok, msg = p.can_submit(5)
        assert ok is False

    def test_can_submit_budget_exhausted(self, tmp_nexus_dir):
        p = Project(
            project_id="q05", max_gpus=4, status="active",
            budget_exhausted=True,
        )
        p.save()
        ok, msg = p.can_submit(1)
        assert ok is False
        assert "budget" in msg.lower()

    def test_quota_error_names_usage(self, tmp_nexus_dir):
        """Clients retry on this message, so its wording is part of the API."""
        p = Project(project_id="q06", max_gpus=2, gpus_in_use=2, status="active")
        p.save()
        ok, msg = p.can_submit(1)
        assert ok is False
        assert "2/2 in use" in msg

    def test_check_and_acquire_then_release(self, tmp_nexus_dir):
        p = Project(project_id="ar01", max_gpus=4)
        p.save()

        ok, reason = p.check_and_acquire(2)
        assert (ok, reason) == (True, "")
        assert p.gpus_in_use == 2

        loaded = Project.load("ar01")
        assert loaded.gpus_in_use == 2

        p.release_gpus(1)
        assert p.gpus_in_use == 1
        assert Project.load("ar01").gpus_in_use == 1

        p.release_gpus(5)  # should clamp to 0
        assert p.gpus_in_use == 0

    def test_check_and_acquire_rejects_negative_gpus(self, tmp_nexus_dir):
        p = Project(project_id="ar03", max_gpus=4, gpus_in_use=2)
        p.save()

        ok, reason = p.check_and_acquire(-4)
        assert not ok
        assert "-4" in reason
        assert p.gpus_in_use == 2
        assert Project.load("ar03").gpus_in_use == 2

    def test_check_and_acquire_uses_state_on_disk(self, tmp_nexus_dir):
        """A stale in-memory copy cannot exceed the quota."""
        Project(project_id="ar02", max_gpus=2).save()
        first = Project.load("ar02")
        stale = Project.load("ar02")

        assert first.check_and_acquire(2)[0] is True
        ok, reason = stale.check_and_acquire(1)
        assert ok is False
        assert "in use" in reason
        assert Project.load("ar02").gpus_in_use == 2

    def test_a_setting_changed_while_a_job_runs_survives_its_end(self, tmp_nexus_dir):
        """The monitor keeps the Project a job was submitted with. Ending
        the job wrote that copy back and undid the change."""
        Project(project_id="ar04", max_gpus=2, soft_gpu_budget_hours=1).save()
        held = Project.load("ar04")
        assert held.check_and_acquire(1)[0] is True

        changed = Project.load("ar04")
        changed.max_gpus = 8
        changed.soft_gpu_budget_hours = 5
        changed.save()

        held.release_gpus(1)
        held.record_gpu_usage(1, 100)

        on_disk = Project.load("ar04")
        assert (on_disk.max_gpus, on_disk.soft_gpu_budget_hours) == (8, 5)
        assert (on_disk.gpus_in_use, on_disk.gpu_seconds_used) == (0, 100)
        assert (held.max_gpus, held.gpu_seconds_used) == (8, 100)   # the copy follows

    def test_gpu_time_of_overlapping_jobs_adds_up(self, tmp_nexus_dir):
        """Each job holds its own copy of the project. The second one to end
        used to write its stale counter over what the first had recorded."""
        Project(project_id="ar05", max_gpus=4).save()
        first, second = Project.load("ar05"), Project.load("ar05")
        assert first.check_and_acquire(1)[0] and second.check_and_acquire(1)[0]

        first.release_gpus(1)
        first.record_gpu_usage(1, 100)
        second.release_gpus(1)
        second.record_gpu_usage(1, 50)

        on_disk = Project.load("ar05")
        assert (on_disk.gpus_in_use, on_disk.gpu_seconds_used) == (0, 150)

    def test_release_does_not_recreate_deleted_project(self, tmp_nexus_dir):
        p = Project(project_id="ar03", max_gpus=4, gpus_in_use=2)
        p.save()
        p.delete()

        p.release_gpus(2)
        p.record_gpu_usage(2, 100)

        assert p.gpus_in_use == 0
        assert Project.load("ar03") is None


class TestGPUBudget:
    """Soft GPU-time budget tracking."""

    def test_record_gpu_usage(self, tmp_nexus_dir):
        p = Project(project_id="gb01", max_gpus=4, soft_gpu_budget_hours=1.0)
        p.save()

        p.record_gpu_usage(2, 100)  # 2 GPUs * 100s = 200 gpu-seconds
        assert p.gpu_seconds_used == 200

        loaded = Project.load("gb01")
        assert loaded.gpu_seconds_used == 200

    def test_record_gpu_usage_accumulates(self, tmp_nexus_dir):
        p = Project(project_id="gb02", max_gpus=4, soft_gpu_budget_hours=1.0)
        p.save()

        p.record_gpu_usage(1, 100)
        p.record_gpu_usage(2, 50)
        assert p.gpu_seconds_used == 200  # 100 + 100

    def test_is_over_soft_budget_under(self, tmp_nexus_dir):
        p = Project(
            project_id="gb03", max_gpus=4,
            soft_gpu_budget_hours=1.0,
            gpu_seconds_used=1000,  # ~0.28h, under 1h budget
        )
        assert p.is_over_soft_budget() is False

    def test_is_over_soft_budget_over(self, tmp_nexus_dir):
        p = Project(
            project_id="gb04", max_gpus=4,
            soft_gpu_budget_hours=1.0,
            gpu_seconds_used=3600,  # exactly at budget
        )
        assert p.is_over_soft_budget() is True

    def test_is_over_soft_budget_unlimited(self, tmp_nexus_dir):
        p = Project(
            project_id="gb05", max_gpus=4,
            soft_gpu_budget_hours=0,  # unlimited
            gpu_seconds_used=999999,
        )
        assert p.is_over_soft_budget() is False

    def test_budget_exhausted_blocks_submit(self, tmp_nexus_dir):
        p = Project(
            project_id="gb06", max_gpus=4, status="active",
            budget_exhausted=True,
        )
        ok, msg = p.can_submit(1)
        assert ok is False
        assert "budget" in msg.lower()

    def test_budget_not_exhausted_allows_submit(self, tmp_nexus_dir):
        p = Project(
            project_id="gb07", max_gpus=4, status="active",
            soft_gpu_budget_hours=1.0,
            gpu_seconds_used=3500,  # under budget
            budget_exhausted=False,
        )
        ok, msg = p.can_submit(1)
        assert ok is True


class TestCPUMemoryQuotas:
    """CPU and memory limit enforcement."""

    def test_cpu_within_limit(self, tmp_nexus_dir):
        p = Project(project_id="cm01", max_gpus=4, max_cpus=4, status="active")
        p.save()
        ok, msg = p.can_submit(num_gpus=0, num_cpus=3)
        assert ok is True

    def test_cpu_exceeds_limit(self, tmp_nexus_dir):
        p = Project(project_id="cm02", max_gpus=4, max_cpus=4, status="active")
        p.save()
        ok, msg = p.can_submit(num_gpus=0, num_cpus=8)
        assert ok is False
        assert "8 CPUs" in msg
        assert "4" in msg

    def test_memory_within_limit(self, tmp_nexus_dir):
        p = Project(project_id="cm03", max_gpus=4, max_memory_mb=2048, status="active")
        p.save()
        ok, msg = p.can_submit(num_gpus=0, num_cpus=1, memory_mb=1024)
        assert ok is True

    def test_memory_exceeds_limit(self, tmp_nexus_dir):
        p = Project(project_id="cm04", max_gpus=4, max_memory_mb=2048, status="active")
        p.save()
        ok, msg = p.can_submit(num_gpus=0, num_cpus=1, memory_mb=4096)
        assert ok is False
        assert "4096 MB" in msg
        assert "2048" in msg

    def test_unlimited_cpus(self, tmp_nexus_dir):
        """max_cpus=0 means unlimited."""
        p = Project(project_id="cm05", max_gpus=4, max_cpus=0, status="active")
        p.save()
        ok, msg = p.can_submit(num_gpus=0, num_cpus=999)
        assert ok is True

    def test_unlimited_memory(self, tmp_nexus_dir):
        """max_memory_mb=0 means unlimited."""
        p = Project(project_id="cm06", max_gpus=4, max_memory_mb=0, status="active")
        p.save()
        ok, msg = p.can_submit(num_gpus=0, num_cpus=1, memory_mb=999999)
        assert ok is True

    def test_serialization_roundtrip(self, tmp_nexus_dir):
        """max_cpus and max_memory_mb survive save/load."""
        p = Project(project_id="cm07", max_gpus=2, max_cpus=8, max_memory_mb=4096)
        p.save()
        loaded = Project.load("cm07")
        assert loaded.max_cpus == 8
        assert loaded.max_memory_mb == 4096

    def test_from_dict_defaults(self):
        """Old projects without max_cpus/max_memory_mb get defaults (0 = unlimited)."""
        p = Project.from_dict({"project_id": "old01"})
        assert p.max_cpus == 0
        assert p.max_memory_mb == 0
