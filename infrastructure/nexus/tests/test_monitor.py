"""Tests for nexus.monitor: JobMonitor background polling, checkpoints, GPU budget."""

import time
from unittest.mock import patch, MagicMock

import pytest

from nexus.config import NexusConfig, MonitorConfig, HiveConnectionConfig, DaemonConfig, ProjectDefaults
from nexus.hive_bridge import HiveBridge
from nexus.jobs import NexusJob
from nexus.monitor import JobMonitor
from nexus.project import Project

from .conftest import FAKE_HIVE_URL


@pytest.fixture
def monitor_config():
    return NexusConfig(
        hive=HiveConnectionConfig(url=FAKE_HIVE_URL),
        daemon=DaemonConfig(),
        monitor=MonitorConfig(
            status_poll_interval=1,
            checkpoint_interval=10,
            checkpoint_timeout=5,
        ),
        project_defaults=ProjectDefaults(),
    )


@pytest.fixture
def mock_bridge():
    bridge = MagicMock(spec=HiveBridge)
    bridge.poll.return_value = {"status": "running", "stdout": "", "stderr": ""}
    return bridge


@pytest.fixture
def monitor(mock_bridge, monitor_config):
    m = JobMonitor(mock_bridge, monitor_config)
    return m


class TestMonitorAddJob:
    """Adding jobs for tracking."""

    def test_add_job(self, monitor, tmp_nexus_dir):
        project = Project(project_id="p1", max_gpus=4)
        project.save()
        job = NexusJob(run_id="j1", project_id="p1", hive_job_id="hive1")
        job.save()

        monitor.add_job(job, project)
        assert "j1" in monitor.tracked_jobs()

    def test_add_multiple_jobs(self, monitor, tmp_nexus_dir):
        project = Project(project_id="p1", max_gpus=4)
        project.save()
        for i in range(3):
            job = NexusJob(run_id=f"j{i}", project_id="p1", hive_job_id=f"hive{i}")
            job.save()
            monitor.add_job(job, project)
        assert len(monitor.tracked_jobs()) == 3


class TestMonitorContinue:
    """Continue after checkpoint."""

    def test_continue_job(self, monitor, tmp_nexus_dir):
        project = Project(project_id="p1", max_gpus=4)
        project.save()
        job = NexusJob(
            run_id="j1", project_id="p1", hive_job_id="hive1",
            status="awaiting_response",
        )
        job.save()
        monitor.add_job(job, project)

        monitor.continue_job("j1")
        loaded = NexusJob.load("j1")
        assert loaded.status == "running"

    def test_continue_nonexistent_job(self, monitor):
        """Should not raise."""
        monitor.continue_job("nonexistent")


class TestMonitorCancel:
    """Cancelling jobs."""

    def test_cancel_job(self, monitor, mock_bridge, tmp_nexus_dir):
        project = Project(project_id="p1", max_gpus=4, gpus_in_use=2,
                         workspace_dir=str(tmp_nexus_dir / "ws"))
        project.save()
        job = NexusJob(
            run_id="j1", project_id="p1", hive_job_id="hive1",
            status="running", num_gpus=2,
        )
        job.save()
        monitor.add_job(job, project)

        mock_bridge.poll.return_value = {"status": "running", "stdout": "out", "stderr": ""}
        monitor.cancel_job("j1")

        loaded_job = NexusJob.load("j1")
        assert loaded_job.status == "cancelled"
        assert loaded_job.completed_at is not None

        loaded_project = Project.load("p1")
        assert loaded_project.gpus_in_use == 0

        assert "j1" not in monitor.tracked_jobs()
        mock_bridge.cancel.assert_called_once_with("hive1")

    def test_cancel_marks_cancelled_even_if_hive_cancel_fails(
            self, monitor, mock_bridge):
        """A failed Hive cancel still frees the slot in Nexus."""
        project = Project(project_id="p1", max_gpus=4, gpus_in_use=1)
        project.save()
        job = NexusJob(run_id="j1", project_id="p1", hive_job_id="hive1",
                       status="running", num_gpus=1)
        job.save()
        monitor.add_job(job, project)

        mock_bridge.cancel.return_value = False
        monitor.cancel_job("j1")

        assert NexusJob.load("j1").status == "cancelled"
        assert Project.load("p1").gpus_in_use == 0

    def test_cancel_writes_final_logs(self, monitor, mock_bridge, tmp_nexus_dir):
        workspace = tmp_nexus_dir / "ws"
        project = Project(project_id="p1", max_gpus=4, gpus_in_use=1,
                          workspace_dir=str(workspace))
        project.save()
        job = NexusJob(run_id="j1", project_id="p1", hive_job_id="hive1",
                       status="running", num_gpus=1)
        job.save()
        job.write_logs(str(workspace), "at checkpoint\n", "")
        monitor.add_job(job, project)

        mock_bridge.poll.return_value = {
            "status": "cancelled", "stdout": "at checkpoint\nmore\n", "stderr": ""}
        monitor.cancel_job("j1")

        log = workspace / "jobs" / "j1" / "stdout.log"
        assert log.read_text() == "at checkpoint\nmore\n"

    def test_cancel_keeps_logs_when_hive_is_unreachable(
            self, monitor, mock_bridge, tmp_nexus_dir):
        """A failed poll must not replace the last known logs by empty files."""
        workspace = tmp_nexus_dir / "ws"
        project = Project(project_id="p1", max_gpus=4, gpus_in_use=1,
                          workspace_dir=str(workspace))
        project.save()
        job = NexusJob(run_id="j1", project_id="p1", hive_job_id="hive1",
                       status="awaiting_response", num_gpus=1)
        job.save()
        job.write_logs(str(workspace), "at checkpoint\n", "warning\n")
        monitor.add_job(job, project)

        mock_bridge.cancel.return_value = False
        mock_bridge.poll.return_value = {"status": "unknown"}
        monitor.cancel_job("j1")

        job_dir = workspace / "jobs" / "j1"
        assert (job_dir / "stdout.log").read_text() == "at checkpoint\n"
        assert (job_dir / "stderr.log").read_text() == "warning\n"
        assert NexusJob.load("j1").status == "cancelled"

    def test_cancel_untracked_job_is_noop(self, monitor, mock_bridge):
        """Cancelling a job the monitor no longer tracks changes nothing."""
        project = Project(project_id="p1", max_gpus=4, gpus_in_use=0)
        project.save()
        NexusJob(run_id="done1", project_id="p1", hive_job_id="hive1",
                 status="completed", num_gpus=1).save()

        monitor.cancel_job("done1")

        assert NexusJob.load("done1").status == "completed"
        mock_bridge.cancel.assert_not_called()


class TestMonitorCheckJob:
    """The _check_job state machine."""

    def test_queued_to_running(self, monitor, mock_bridge, tmp_nexus_dir):
        project = Project(project_id="p1", max_gpus=4,
                         workspace_dir=str(tmp_nexus_dir / "ws"))
        project.save()
        job = NexusJob(
            run_id="j1", project_id="p1", hive_job_id="hive1",
            status="queued",
        )
        job.save()
        monitor.add_job(job, project)

        mock_bridge.poll.return_value = {"status": "running", "stdout": "", "stderr": ""}
        monitor._tick()

        loaded = NexusJob.load("j1")
        assert loaded.status == "running"
        assert loaded.started_at is not None

    def test_completion(self, monitor, mock_bridge, tmp_nexus_dir):
        project = Project(project_id="p1", max_gpus=4, gpus_in_use=2,
                         workspace_dir=str(tmp_nexus_dir / "ws"))
        project.save()
        job = NexusJob(
            run_id="j1", project_id="p1", hive_job_id="hive1",
            status="running", num_gpus=2, started_at=time.time(),
        )
        job.save()
        monitor.add_job(job, project)

        mock_bridge.poll.return_value = {
            "status": "completed", "stdout": "done\n", "stderr": "",
            "returncode": 0,
        }
        monitor._tick()

        loaded_job = NexusJob.load("j1")
        assert loaded_job.status == "completed"
        assert loaded_job.returncode == 0

        loaded_project = Project.load("p1")
        assert loaded_project.gpus_in_use == 0

        assert "j1" not in monitor.tracked_jobs()

    def test_failure(self, monitor, mock_bridge, tmp_nexus_dir):
        project = Project(project_id="p1", max_gpus=4, gpus_in_use=1,
                         workspace_dir=str(tmp_nexus_dir / "ws"))
        project.save()
        job = NexusJob(
            run_id="j1", project_id="p1", hive_job_id="hive1",
            status="running", num_gpus=1, started_at=time.time(),
        )
        job.save()
        monitor.add_job(job, project)

        mock_bridge.poll.return_value = {
            "status": "failed", "stdout": "", "stderr": "OOM\n",
            "returncode": 1,
        }
        monitor._tick()

        loaded_job = NexusJob.load("j1")
        assert loaded_job.status == "failed"
        assert loaded_job.returncode == 1

    def test_checkpoint_triggered(self, monitor, mock_bridge, tmp_nexus_dir):
        """Job past checkpoint_interval should trigger checkpoint."""
        project = Project(project_id="p1", max_gpus=4,
                         workspace_dir=str(tmp_nexus_dir / "ws"))
        project.save()
        job = NexusJob(
            run_id="j1", project_id="p1", hive_job_id="hive1",
            status="running", num_gpus=1,
            started_at=time.time() - 20,  # 20s ago, interval is 10s
        )
        job.save()
        monitor.add_job(job, project)

        mock_bridge.poll.return_value = {
            "status": "running", "stdout": "epoch 10\n", "stderr": "",
        }
        monitor._tick()

        loaded = NexusJob.load("j1")
        assert loaded.status == "awaiting_response"
        assert loaded.last_checkpoint_at is not None
        assert loaded.checkpoint_count == 1

    def test_checkpoint_auto_continue(self, monitor, mock_bridge, tmp_nexus_dir):
        """Awaiting response past checkpoint_timeout should auto-continue."""
        project = Project(project_id="p1", max_gpus=4,
                         workspace_dir=str(tmp_nexus_dir / "ws"))
        project.save()
        job = NexusJob(
            run_id="j1", project_id="p1", hive_job_id="hive1",
            status="awaiting_response", num_gpus=1,
            started_at=time.time() - 100,
            last_checkpoint_at=time.time() - 10,  # 10s ago, timeout is 5s
        )
        job.save()
        monitor.add_job(job, project)

        monitor._tick()

        loaded = NexusJob.load("j1")
        assert loaded.status == "running"

    def test_a_continued_job_is_not_checkpointed_again_at_once(
            self, monitor, mock_bridge, tmp_nexus_dir):
        """The automatic continue comes later than one interval after the
        checkpoint, so the next tick used to raise another checkpoint and an
        unattended job spent nearly all its time waiting for an answer."""
        project = Project(project_id="p1", max_gpus=4,
                          workspace_dir=str(tmp_nexus_dir / "ws"))
        project.save()
        job = NexusJob(
            run_id="j1", project_id="p1", hive_job_id="hive1",
            status="awaiting_response", num_gpus=1,
            started_at=time.time() - 100,
            last_checkpoint_at=time.time() - 30,  # timeout is 5s, interval 10s
            checkpoint_count=1,
        )
        job.save()
        monitor.add_job(job, project)
        mock_bridge.poll.return_value = {"status": "running", "stdout": "", "stderr": ""}

        monitor._tick()            # automatic continue
        assert NexusJob.load("j1").status == "running"
        monitor._tick()            # still running: one interval has not passed
        loaded = NexusJob.load("j1")
        assert (loaded.status, loaded.checkpoint_count) == ("running", 1)

    def test_unknown_status_ignored(self, monitor, mock_bridge, tmp_nexus_dir):
        """Transient Hive error (unknown status) should not change job state."""
        project = Project(project_id="p1", max_gpus=4,
                         workspace_dir=str(tmp_nexus_dir / "ws"))
        project.save()
        job = NexusJob(
            run_id="j1", project_id="p1", hive_job_id="hive1",
            status="running", num_gpus=1, started_at=time.time(),
        )
        job.save()
        monitor.add_job(job, project)

        mock_bridge.poll.return_value = {"status": "unknown"}
        monitor._tick()

        loaded = NexusJob.load("j1")
        assert loaded.status == "running"  # unchanged


class TestMonitorGPUBudget:
    """GPU-time tracking and soft budget enforcement."""

    def test_completion_records_gpu_usage(self, monitor, mock_bridge, tmp_nexus_dir):
        """Completed job should record GPU-time on the project."""
        started = time.time() - 60  # 60 seconds ago
        project = Project(
            project_id="p1", max_gpus=4, gpus_in_use=2,
            soft_gpu_budget_hours=10.0,
            workspace_dir=str(tmp_nexus_dir / "ws"),
        )
        project.save()
        job = NexusJob(
            run_id="j1", project_id="p1", hive_job_id="hive1",
            status="running", num_gpus=2, started_at=started,
        )
        job.save()
        monitor.add_job(job, project)

        mock_bridge.poll.return_value = {
            "status": "completed", "stdout": "done\n", "stderr": "",
            "returncode": 0,
        }
        monitor._tick()

        loaded_project = Project.load("p1")
        # 2 GPUs * ~60s = ~120 gpu-seconds (allow some tolerance)
        assert loaded_project.gpu_seconds_used > 100
        assert loaded_project.gpu_seconds_used < 200

    def test_budget_exhausted_after_last_job(self, monitor, mock_bridge, tmp_nexus_dir):
        """When soft budget is exceeded and no jobs remain, budget_exhausted should be set."""
        started = time.time() - 3600  # 1 hour ago
        project = Project(
            project_id="p1", max_gpus=4, gpus_in_use=1,
            soft_gpu_budget_hours=0.5,  # 0.5h = 1800 gpu-seconds
            workspace_dir=str(tmp_nexus_dir / "ws"),
        )
        project.save()
        job = NexusJob(
            run_id="j1", project_id="p1", hive_job_id="hive1",
            status="running", num_gpus=1, started_at=started,
        )
        job.save()
        monitor.add_job(job, project)

        mock_bridge.poll.return_value = {
            "status": "completed", "stdout": "done\n", "stderr": "",
            "returncode": 0,
        }
        monitor._tick()

        loaded_project = Project.load("p1")
        assert loaded_project.budget_exhausted is True
        # Should have recorded ~3600 gpu-seconds (1 GPU * 1h)
        assert loaded_project.gpu_seconds_used > 3500

    def test_budget_not_exhausted_when_unlimited(self, monitor, mock_bridge, tmp_nexus_dir):
        """With unlimited budget (0), budget_exhausted should never be set."""
        started = time.time() - 3600
        project = Project(
            project_id="p1", max_gpus=4, gpus_in_use=1,
            soft_gpu_budget_hours=0,  # unlimited
            workspace_dir=str(tmp_nexus_dir / "ws"),
        )
        project.save()
        job = NexusJob(
            run_id="j1", project_id="p1", hive_job_id="hive1",
            status="running", num_gpus=1, started_at=started,
        )
        job.save()
        monitor.add_job(job, project)

        mock_bridge.poll.return_value = {
            "status": "completed", "stdout": "done\n", "stderr": "",
            "returncode": 0,
        }
        monitor._tick()

        loaded_project = Project.load("p1")
        assert loaded_project.budget_exhausted is False

    def test_budget_not_exhausted_while_jobs_running(self, monitor, mock_bridge, tmp_nexus_dir):
        """Budget_exhausted should not be set while other jobs still running."""
        started = time.time() - 3600
        project = Project(
            project_id="p1", max_gpus=4, gpus_in_use=2,
            soft_gpu_budget_hours=0.5,
            workspace_dir=str(tmp_nexus_dir / "ws"),
        )
        project.save()

        # Two jobs for this project
        job1 = NexusJob(
            run_id="j1", project_id="p1", hive_job_id="hive1",
            status="running", num_gpus=1, started_at=started,
        )
        job1.save()
        job2 = NexusJob(
            run_id="j2", project_id="p1", hive_job_id="hive2",
            status="running", num_gpus=1, started_at=started,
        )
        job2.save()
        monitor.add_job(job1, project)
        monitor.add_job(job2, project)

        # Only j1 completes
        def poll_side_effect(hive_id):
            if hive_id == "hive1":
                return {"status": "completed", "stdout": "done\n", "stderr": "",
                        "returncode": 0}
            return {"status": "running", "stdout": "", "stderr": ""}

        mock_bridge.poll.side_effect = poll_side_effect
        monitor._tick()

        loaded_project = Project.load("p1")
        # j2 is still running, so budget_exhausted should NOT be set yet
        assert loaded_project.budget_exhausted is False
        # But GPU usage should be recorded
        assert loaded_project.gpu_seconds_used > 3500


class TestMonitorWake:
    """Per-project wake command."""

    def test_wake_command_template(self, monitor):
        project = Project(
            project_id="p1", workspace_dir="/ws/p1",
            wake_command="notify {project_id} {run_id} {event} {workspace} {jobs_dir}",
        )
        job = NexusJob(run_id="j1", project_id="p1")
        with patch("nexus.monitor.subprocess.run") as run:
            monitor._wake_agent(project, job, "checkpoint")
        run.assert_called_once()
        assert run.call_args[0][0] == "notify p1 j1 checkpoint /ws/p1 /ws/p1/jobs/j1"

    def test_no_wake_command_runs_nothing(self, monitor):
        project = Project(project_id="p1")
        job = NexusJob(run_id="j1", project_id="p1")
        with patch("nexus.monitor.subprocess.run") as run:
            monitor._wake_agent(project, job, "completed")
        run.assert_not_called()

    def test_unknown_template_variable_is_ignored(self, monitor):
        project = Project(project_id="p1", wake_command="notify {nope}")
        job = NexusJob(run_id="j1", project_id="p1")
        with patch("nexus.monitor.subprocess.run") as run:
            monitor._wake_agent(project, job, "completed")
        run.assert_not_called()

    @pytest.mark.parametrize("template", ["notify {}", "notify {0}", "echo {", "notify {run_id.x}"])
    def test_malformed_template_is_ignored(self, monitor, template):
        project = Project(project_id="p1", wake_command=template)
        job = NexusJob(run_id="j1", project_id="p1")
        with patch("nexus.monitor.subprocess.run") as run:
            monitor._wake_agent(project, job, "completed")
        run.assert_not_called()

    def test_malformed_template_does_not_block_completion(
        self, monitor, mock_bridge, tmp_nexus_dir,
    ):
        project = Project(project_id="p1", max_gpus=4, gpus_in_use=1,
                          workspace_dir=str(tmp_nexus_dir / "ws"),
                          wake_command="echo {")
        project.save()
        job = NexusJob(run_id="j1", project_id="p1", hive_job_id="hive1",
                       status="running", num_gpus=1, started_at=time.time())
        job.save()
        monitor.add_job(job, project)

        mock_bridge.poll.return_value = {
            "status": "completed", "stdout": "", "stderr": "", "returncode": 0,
        }
        with patch("nexus.monitor.subprocess.run"):
            monitor._tick()

        assert NexusJob.load("j1").status == "completed"
        assert "j1" not in monitor.tracked_jobs()


class TestMonitorStartStop:
    """Thread lifecycle and recovery of in-flight jobs on start.

    The Hive mock answers "unknown" here, so the tick that the monitor
    thread runs right after start() leaves every reloaded job as it is.
    """

    @pytest.fixture(autouse=True)
    def _hive_unreachable(self, mock_bridge):
        mock_bridge.poll.return_value = {"status": "unknown"}

    def test_start_and_stop(self, monitor):
        monitor.start()
        assert monitor.is_running
        monitor.stop()
        # After stop, monitor should not be running
        assert not monitor.is_running

    def test_start_reloads_active_jobs(self, monitor):
        Project(project_id="p1", max_gpus=4, gpus_in_use=1).save()
        NexusJob(run_id="active1", project_id="p1", hive_job_id="h1",
                 status="running", num_gpus=1).save()
        # Completed job (should NOT be reloaded)
        NexusJob(run_id="done1", project_id="p1", hive_job_id="h2",
                 status="completed", num_gpus=1).save()

        monitor.start()
        try:
            tracked = monitor.tracked_jobs()
            assert "active1" in tracked
            assert "done1" not in tracked
        finally:
            monitor.stop()

    def test_start_fails_jobs_of_deleted_projects(self, monitor):
        NexusJob(run_id="orphan", project_id="gone", hive_job_id="h1",
                 status="running").save()

        monitor.start()
        try:
            assert "orphan" not in monitor.tracked_jobs()
            loaded = NexusJob.load("orphan")
            assert loaded.status == "failed"
            assert loaded.completed_at is not None
        finally:
            monitor.stop()

    def test_start_reconciles_gpus_in_use(self, monitor):
        """gpus_in_use is recomputed from the active jobs on disk."""
        # Counter too high: one active 2-GPU job, but 4 recorded.
        Project(project_id="high", max_gpus=4, gpus_in_use=4).save()
        NexusJob(run_id="h1", project_id="high", hive_job_id="x1",
                 status="running", num_gpus=2).save()
        # Counter leaked: only a finished job, but 1 recorded.
        Project(project_id="idle", max_gpus=4, gpus_in_use=1).save()
        NexusJob(run_id="i1", project_id="idle", hive_job_id="x2",
                 status="failed", num_gpus=1).save()

        monitor.start()
        try:
            assert Project.load("high").gpus_in_use == 2
            assert Project.load("idle").gpus_in_use == 0
        finally:
            monitor.stop()
