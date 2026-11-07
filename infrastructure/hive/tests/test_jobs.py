"""Tests for job record persistence and lifecycle."""

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from hive.dispatcher import (
    Dispatcher,
    JobRecord,
    _run_job,
    _AsyncJobRunner,
    _build_job_log_preamble,
    launch_job_runner,
    is_node_draining,
    mark_node_draining,
    prune_stale_draining_nodes,
    write_job_heartbeat,
    write_job_process_result,
)
from hive.config import JOBS_DIR


class TestJobRecord:
    """JobRecord serialization, persistence, and loading."""

    def test_to_dict_excludes_stdout_stderr(self):
        r = JobRecord(
            job_id="abc123",
            command="python train.py",
            num_gpus=2,
            status="running",
            stdout="epoch 1 loss 0.5",
            stderr="warning: lr too high",
        )
        d = r.to_dict()
        assert "stdout" not in d
        assert "stderr" not in d
        assert d["job_id"] == "abc123"
        assert d["num_gpus"] == 2
        assert d["runner_launched"] is False

    def test_from_dict_roundtrip(self):
        r = JobRecord(
            job_id="abc123",
            command="python train.py",
            num_gpus=2,
            status="running",
            submitted_at=1000.0,
            started_at=1001.0,
            completed_at=None,
            stdout="epoch 1 loss 0.5",
            stderr="",
            returncode=None,
        )
        d = {**r.to_dict(), "stdout": r.stdout, "stderr": r.stderr}
        restored = JobRecord.from_dict(d)
        assert restored.job_id == "abc123"
        assert restored.command == "python train.py"
        assert restored.num_gpus == 2
        assert restored.status == "running"
        assert restored.submitted_at == 1000.0
        assert restored.started_at == 1001.0
        assert restored.completed_at is None
        assert restored.stdout == "epoch 1 loss 0.5"
        assert restored.returncode is None

    def test_from_dict_defaults(self):
        r = JobRecord.from_dict({"job_id": "x", "command": "echo hi"})
        assert r.num_gpus == 1
        assert r.status == "unknown"
        assert r.stdout == ""
        assert r.stderr == ""

    def test_save_creates_directory(self, tmp_hive_dir):
        r = JobRecord(
            job_id="test001",
            command="nvidia-smi",
            num_gpus=1,
            status="completed",
            submitted_at=time.time(),
            returncode=0,
            stdout="GPU 0: NVIDIA A100",
            stderr="",
        )
        r.save()

        job_dir = tmp_hive_dir / "jobs" / "test001"
        assert job_dir.is_dir()
        assert (job_dir / "meta.json").exists()
        assert (job_dir / "stdout.log").exists()
        assert (job_dir / "stderr.log").exists()

        # meta.json should NOT contain stdout/stderr
        meta = json.loads((job_dir / "meta.json").read_text())
        assert "stdout" not in meta
        assert "stderr" not in meta
        assert meta["status"] == "completed"

        # stdout/stderr in separate files
        assert "A100" in (job_dir / "stdout.log").read_text()
        assert (job_dir / "stderr.log").read_text() == ""

    def test_save_and_load(self, tmp_hive_dir):
        r = JobRecord(
            job_id="test001",
            command="nvidia-smi",
            num_gpus=1,
            status="completed",
            submitted_at=time.time(),
            returncode=0,
            stdout="GPU 0: NVIDIA A100",
        )
        r.save()

        loaded = JobRecord.load("test001")
        assert loaded is not None
        assert loaded.job_id == "test001"
        assert loaded.command == "nvidia-smi"
        assert loaded.returncode == 0
        assert "A100" in loaded.stdout

    def test_load_nonexistent_returns_none(self, tmp_hive_dir):
        assert JobRecord.load("nonexistent") is None

    def test_load_all_empty(self, tmp_hive_dir):
        records = JobRecord.load_all()
        assert records == []

    def test_load_all_multiple(self, tmp_hive_dir):
        for i in range(3):
            JobRecord(
                job_id=f"job{i}",
                command=f"echo {i}",
                num_gpus=1,
                status="completed",
            ).save()

        records = JobRecord.load_all()
        assert len(records) == 3
        ids = {r.job_id for r in records}
        assert ids == {"job0", "job1", "job2"}

    def test_save_overwrites(self, tmp_hive_dir):
        r = JobRecord(job_id="upd", command="echo", num_gpus=1, status="queued")
        r.save()

        r.status = "running"
        r.started_at = time.time()
        r.save()

        loaded = JobRecord.load("upd")
        assert loaded.status == "running"
        assert loaded.started_at is not None

    def test_mark_runner_launch_pending_updates_queued_record(self, tmp_hive_dir):
        job = JobRecord(job_id="launch01", command="echo hi", num_gpus=1, status="queued")
        job.save()

        updated = JobRecord.mark_runner_launch_pending("launch01", ts=1234.0)

        assert updated is not None
        assert updated.status == "queued"
        assert updated.runner_launched is False
        assert updated.runner_launch_pending_since == 1234.0

        loaded = JobRecord.load("launch01")
        assert loaded.status == "queued"
        assert loaded.runner_launched is False
        assert loaded.runner_launch_pending_since == 1234.0

    def test_mark_runner_launch_pending_does_not_revert_terminal_record(self, tmp_hive_dir):
        job = JobRecord(
            job_id="launch02",
            command="echo hi",
            num_gpus=1,
            status="completed",
            submitted_at=time.time(),
            started_at=time.time(),
            completed_at=time.time(),
            returncode=0,
            runner_launched=False,
        )
        job.save()

        updated = JobRecord.mark_runner_launch_pending("launch02", ts=1234.0)

        assert updated is not None
        assert updated.status == "completed"
        assert updated.runner_launched is False
        assert updated.runner_launch_pending_since is None

        loaded = JobRecord.load("launch02")
        assert loaded.status == "completed"
        assert loaded.returncode == 0
        assert loaded.runner_launched is False
        assert loaded.runner_launch_pending_since is None

    def test_legacy_json_fallback(self, tmp_hive_dir):
        """Old single-file .json records should still be loadable."""
        legacy_data = {
            "job_id": "old001",
            "command": "echo legacy",
            "num_gpus": 1,
            "status": "completed",
            "submitted_at": 1000.0,
            "stdout": "legacy output",
            "stderr": "",
            "returncode": 0,
        }
        legacy_path = tmp_hive_dir / "jobs" / "old001.json"
        legacy_path.write_text(json.dumps(legacy_data))

        loaded = JobRecord.load("old001")
        assert loaded is not None
        assert loaded.job_id == "old001"
        assert loaded.stdout == "legacy output"

    def test_mark_cancelled_migrates_legacy_logs_without_loss(self, tmp_hive_dir):
        """Cancelling a legacy record should preserve embedded stdout/stderr."""
        legacy_data = {
            "job_id": "legacycancel01",
            "command": "echo legacy",
            "num_gpus": 1,
            "status": "queued",
            "submitted_at": 1000.0,
            "stdout": "legacy stdout",
            "stderr": "legacy stderr",
            "returncode": None,
        }
        legacy_path = tmp_hive_dir / "jobs" / "legacycancel01.json"
        legacy_path.write_text(json.dumps(legacy_data))

        result, record = JobRecord.mark_cancelled("legacycancel01", now=1002.0)

        assert result == "cancelled"
        assert record is not None
        loaded = JobRecord.load("legacycancel01")
        assert loaded is not None
        assert loaded.status == "cancelled"
        assert loaded.completed_at == 1002.0
        assert loaded.stdout == "legacy stdout"
        assert loaded.stderr == "legacy stderr\n[hive] Job cancelled\n"
        assert loaded.combined_stderr == "legacy stderr\n[hive] Job cancelled"

    def test_mark_cancelled_leaves_live_output_files_untouched(self, tmp_hive_dir):
        """Cancelling a running job must not rewrite stdout/stderr snapshots."""
        job = JobRecord(
            job_id="cancelnote01",
            command="python train.py",
            num_gpus=1,
            status="running",
            submitted_at=1000.0,
            started_at=1001.0,
            stdout="live stdout\n",
            stderr="live stderr\n",
        )
        job.save()
        job.stdout_path.write_text("live stdout\nmore output\n")
        job.stderr_path.write_text("live stderr\nmore err\n")

        result, record = JobRecord.mark_cancelled("cancelnote01", now=1002.0)

        assert result == "cancelled"
        assert record is not None
        assert job.stdout_path.read_text() == "live stdout\nmore output\n"
        assert job.stderr_path.read_text() == "live stderr\nmore err\n"
        assert (job.job_dir / "events.log").read_text() == "[hive] Job cancelled\n"
        loaded = JobRecord.load("cancelnote01")
        assert loaded.status == "cancelled"
        assert loaded.combined_stderr == "live stderr\nmore err\n[hive] Job cancelled"

    def test_mark_preempted_marks_running_job_for_retry(self, tmp_hive_dir):
        job = JobRecord(
            job_id="drainpreempt01",
            command="python train.py",
            num_gpus=1,
            status="running",
            submitted_at=1000.0,
            started_at=1001.0,
            node_id="node-drain-01",
            runner_launched=True,
        )
        job.save()

        result, record = JobRecord.mark_preempted("drainpreempt01")

        assert result == "preempted"
        assert record is not None
        loaded = JobRecord.load("drainpreempt01")
        assert loaded.status == "preempted"
        assert loaded.started_at is None
        assert loaded.completed_at is None
        assert loaded.returncode is None
        assert loaded.combined_stderr.endswith("[hive] Preempted for SLURM worker drain before walltime")

    def test_requeue_preempted_restores_job_to_queue(self, tmp_hive_dir):
        job = JobRecord(
            job_id="requeuedrain01",
            command="python train.py",
            num_gpus=1,
            status="preempted",
            submitted_at=1000.0,
            node_id="node-drain-01",
            runner_launched=True,
            stderr="existing stderr\n",
        )
        job.save()
        (job.job_dir / "events.log").write_text("[hive] Preempted for SLURM worker drain before walltime\n")
        # Previous attempt left its process_result.json on disk.
        write_job_process_result(
            job_id="requeuedrain01",
            completed_at=500.0,
            returncode=137,
            timed_out=False,
            attempt=0,
        )

        result, record = JobRecord.requeue_preempted("requeuedrain01")

        assert result == "queued"
        assert record is not None
        loaded = JobRecord.load("requeuedrain01")
        assert loaded.status == "queued"
        assert loaded.node_id == ""
        assert loaded.runner_launched is False
        assert loaded.runner_launch_pending_since is None
        assert loaded.combined_stderr.endswith("[hive] Preempted for SLURM worker drain before walltime")
        assert not (job.job_dir / "events.log").exists()
        # The stale process_result file must be cleared so the retry
        # doesn't inherit its predecessor's terminal state.
        from hive.dispatcher import read_job_process_result
        assert read_job_process_result("requeuedrain01") == {}

    def test_launch_job_runner_waits_for_unresolvable_stale_actor_name_to_disappear(self, tmp_hive_dir):
        record = JobRecord(
            job_id="stalerelaunch01",
            command="python train.py",
            num_gpus=1,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=1000.0,
        )
        record.save()

        with patch("hive.dispatcher.ray") as mock_ray, \
            patch("hive.dispatcher._AsyncJobRunner") as mock_runner, \
            patch("hive.dispatcher.time.sleep"):
            mock_ray.util.list_named_actors.side_effect = [
                [{"name": "hive_job_stalerelaunch01"}],
                [{"name": "hive_job_stalerelaunch01"}],
                [],
            ]
            mock_ray.get_actor.side_effect = ValueError("gone")

            result, updated = launch_job_runner(record, workdir="/tmp/relaunch")

        assert result == "launched"
        assert updated is not None
        mock_runner.options.return_value.remote.assert_called_once()
        assert mock_ray.util.list_named_actors.call_count == 3
        reloaded = JobRecord.load("stalerelaunch01")
        assert reloaded.runner_launch_pending_since is not None

    def test_launch_job_runner_recovers_stale_pending_launch_without_actor(self, tmp_hive_dir):
        record = JobRecord(
            job_id="stalerelaunch02",
            command="python train.py",
            num_gpus=1,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=1000.0,
            runner_launch_pending_since=1001.0,
        )
        record.save()

        with patch("hive.dispatcher.ray") as mock_ray, \
             patch("hive.dispatcher.get_launch_pending_grace_sec", return_value=30), \
             patch("hive.dispatcher._AsyncJobRunner") as mock_runner, \
             patch("hive.dispatcher.time.sleep"):
            mock_ray.util.list_named_actors.return_value = []
            result, updated = launch_job_runner(record, workdir="/tmp/relaunch")

        assert result == "launched"
        assert updated is not None
        mock_runner.options.return_value.remote.assert_called_once()
        reloaded = JobRecord.load("stalerelaunch02")
        assert reloaded.runner_launch_pending_since is not None

    def test_launch_job_runner_binds_dispatch_resource_when_requested(self, tmp_hive_dir):
        record = JobRecord(
            job_id="stalerelaunch02bind",
            command="python train.py",
            num_gpus=1,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=1000.0,
            resource="gpu-a,gpu-b",
        )
        record.save()

        with patch("hive.dispatcher.ray") as mock_ray, \
             patch("hive.dispatcher._AsyncJobRunner") as mock_runner, \
             patch("hive.dispatcher.time.sleep"):
            mock_ray.util.list_named_actors.return_value = []
            result, updated = launch_job_runner(
                record,
                workdir="/tmp/relaunch",
                dispatch_resource="gpu-b",
            )

        assert result == "launched"
        assert updated is not None
        mock_runner.options.return_value.remote.assert_called_once()
        bundles = mock_runner.options.return_value.remote.call_args.args[6]
        assert bundles == [{"CPU": 1, "GPU": 1, "partition:gpu-b": 0.01}]
        reloaded = JobRecord.load("stalerelaunch02bind")
        assert reloaded.dispatch_resource == "gpu-b"
        assert reloaded.runner_launch_pending_since is not None

    def test_launch_job_runner_does_not_recover_recent_pending_launch_without_actor(self, tmp_hive_dir):
        record = JobRecord(
            job_id="stalerelaunch02b",
            command="python train.py",
            num_gpus=1,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=1000.0,
            runner_launch_pending_since=time.time(),
        )
        record.save()

        with patch("hive.dispatcher.get_launch_pending_grace_sec", return_value=30), \
             patch("hive.dispatcher.find_job_actor", return_value=None), \
             patch("hive.dispatcher.clear_job_actor_name") as mock_clear, \
             patch("hive.dispatcher._AsyncJobRunner") as mock_runner:
            result, updated = launch_job_runner(record, workdir="/tmp/relaunch")

        assert result == "launch_pending"
        assert updated is not None
        mock_clear.assert_not_called()
        mock_runner.options.return_value.remote.assert_not_called()
        reloaded = JobRecord.load("stalerelaunch02b")
        assert reloaded.runner_launch_pending_since is not None

    def test_launch_job_runner_skips_if_pending_launch_has_live_actor(self, tmp_hive_dir):
        record = JobRecord(
            job_id="stalerelaunch03",
            command="python train.py",
            num_gpus=1,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=1000.0,
            runner_launch_pending_since=1001.0,
        )
        record.save()

        with patch("hive.dispatcher.find_job_actor", return_value=object()), \
             patch("hive.dispatcher.get_launch_pending_grace_sec", return_value=30), \
             patch("hive.dispatcher._AsyncJobRunner") as mock_runner:
            result, updated = launch_job_runner(record, workdir="/tmp/relaunch")

        assert result == "already_launched"
        assert updated is not None
        mock_runner.options.return_value.remote.assert_not_called()

    def test_launch_job_runner_clears_stuck_pending_marker_when_name_cleanup_fails(self, tmp_hive_dir):
        record = JobRecord(
            job_id="stalerelaunch04",
            command="python train.py",
            num_gpus=1,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=1000.0,
            runner_launch_pending_since=1001.0,
        )
        record.save()

        with patch("hive.dispatcher.find_job_actor", return_value=None), \
             patch("hive.dispatcher.get_launch_pending_grace_sec", return_value=30), \
             patch("hive.dispatcher.clear_job_actor_name", side_effect=RuntimeError("still stale")), \
             pytest.raises(RuntimeError, match="still stale"):
            launch_job_runner(record, workdir="/tmp/relaunch")

        reloaded = JobRecord.load("stalerelaunch04")
        assert reloaded.runner_launch_pending_since is None

    def test_draining_node_markers_prune_stale_entries(self, tmp_hive_dir):
        mark_node_draining("node-live", resource_name="gpu-test", slurm_job_id="123")
        mark_node_draining("node-stale", resource_name="gpu-test", slurm_job_id="456")

        assert is_node_draining("node-live")
        assert is_node_draining("node-stale")

        prune_stale_draining_nodes({"node-live"})

        assert is_node_draining("node-live")
        assert not is_node_draining("node-stale")

    def test_load_all_mixed_formats(self, tmp_hive_dir):
        """load_all should find both directory and legacy JSON records."""
        # New format
        JobRecord(
            job_id="new01", command="echo new", num_gpus=1, status="completed",
        ).save()

        # Legacy format
        legacy_data = {
            "job_id": "old01",
            "command": "echo old",
            "num_gpus": 1,
            "status": "completed",
            "submitted_at": 1000.0,
            "stdout": "old output",
            "stderr": "",
            "returncode": 0,
        }
        legacy_path = tmp_hive_dir / "jobs" / "old01.json"
        legacy_path.write_text(json.dumps(legacy_data))

        records = JobRecord.load_all()
        assert len(records) == 2
        ids = {r.job_id for r in records}
        assert ids == {"new01", "old01"}

    def test_finalize_runner_record_flushes_buffered_cancel_note(self, tmp_hive_dir):
        """Runner teardown should flush a buffered cancel note after live stderr ends."""
        cancelled = JobRecord(
            job_id="finalcancel01",
            command="python train.py",
            num_gpus=1,
            status="cancelled",
            submitted_at=1.0,
            completed_at=2.0,
            stdout="existing stdout",
            stderr="existing stderr",
            returncode=-15,
        )
        cancelled.save()
        (cancelled.job_dir / "events.log").write_text("[hive] Job cancelled\n")

        stale = JobRecord(
            job_id="finalcancel01",
            command="python train.py",
            num_gpus=1,
            status="failed",
            submitted_at=1.0,
            completed_at=3.0,
            stdout="stale stdout",
            stderr="stale stderr",
            returncode=-1,
            runner_launched=True,
            missing_actor_checks=2,
        )

        final, retried = JobRecord.finalize_runner_record(stale, retried=True)

        assert retried is False
        assert final.status == "cancelled"
        assert final.returncode == -15
        assert final.stdout == "existing stdout"
        assert final.stderr == "existing stderr\n[hive] Job cancelled"
        assert final.combined_stderr == "existing stderr\n[hive] Job cancelled"
        loaded = JobRecord.load("finalcancel01")
        assert loaded.status == "cancelled"
        assert loaded.stderr == "existing stderr\n[hive] Job cancelled\n"
        assert loaded.combined_stderr == "existing stderr\n[hive] Job cancelled"
        assert not (cancelled.job_dir / "events.log").exists()

    def test_materialize_buffered_notes_for_queued_job(self, tmp_hive_dir):
        """Queued jobs should absorb legacy/buffered notes into stable stderr.log."""
        queued = JobRecord(
            job_id="queueevents01",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=1.0,
            stderr="existing stderr",
        )
        queued.save()
        (queued.job_dir / "events.log").write_text("[hive] Preempted for retry\n")

        JobRecord.materialize_buffered_notes_for_queued_job("queueevents01")

        loaded = JobRecord.load("queueevents01")
        assert loaded.stderr == "existing stderr\n[hive] Preempted for retry\n"
        assert loaded.combined_stderr == "existing stderr\n[hive] Preempted for retry"
        assert not (queued.job_dir / "events.log").exists()

    def test_load_preempted_job_surfaces_buffered_notes(self, tmp_hive_dir):
        """Preempted jobs should expose buffered notes before requeue."""
        preempted = JobRecord(
            job_id="preemptednotes01",
            command="python train.py",
            num_gpus=1,
            status="preempted",
            submitted_at=1.0,
            stderr="existing stderr\n",
        )
        preempted.save()
        (preempted.job_dir / "events.log").write_text("[hive] Preempted for retry\n")

        loaded = JobRecord.load("preemptednotes01")

        assert loaded.status == "preempted"
        assert loaded.stderr == "existing stderr\n[hive] Preempted for retry\n"
        assert loaded.combined_stderr == "existing stderr\n[hive] Preempted for retry"

    def test_stdout_stderr_paths(self, tmp_hive_dir):
        r = JobRecord(job_id="paths01", command="echo test", num_gpus=1, status="queued")
        assert r.stdout_path == tmp_hive_dir / "jobs" / "paths01" / "stdout.log"
        assert r.stderr_path == tmp_hive_dir / "jobs" / "paths01" / "stderr.log"


class TestJobLifecycle:
    """Simulate a job going through queued -> running -> completed."""

    def test_full_lifecycle(self, tmp_hive_dir):
        # Submit
        job = JobRecord(
            job_id="lifecycle01",
            command="python -c 'print(42)'",
            num_gpus=1,
            status="queued",
            submitted_at=time.time(),
        )
        job.save()
        assert JobRecord.load("lifecycle01").status == "queued"

        # Start
        job.status = "running"
        job.started_at = time.time()
        job.save()
        assert JobRecord.load("lifecycle01").status == "running"

        # Complete
        job.status = "completed"
        job.completed_at = time.time()
        job.returncode = 0
        job.stdout = "42\n"
        job.save()

        final = JobRecord.load("lifecycle01")
        assert final.status == "completed"
        assert final.returncode == 0
        assert final.stdout == "42\n"
        assert final.completed_at >= final.started_at

        # Verify file structure
        job_dir = tmp_hive_dir / "jobs" / "lifecycle01"
        assert job_dir.is_dir()
        assert (job_dir / "stdout.log").read_text() == "42\n"

    def test_failed_job(self, tmp_hive_dir):
        job = JobRecord(
            job_id="fail01",
            command="python -c 'raise ValueError()'",
            num_gpus=1,
            status="failed",
            submitted_at=time.time(),
            completed_at=time.time(),
            returncode=1,
            stderr="ValueError\n",
        )
        job.save()

        loaded = JobRecord.load("fail01")
        assert loaded.status == "failed"
        assert loaded.returncode == 1
        assert "ValueError" in loaded.stderr


class TestOrphanDetection:
    """Dashboard orphan handling should be conservative and recoverable."""

    def test_logs_endpoint_does_not_fail_missing_actor_job(self, tmp_hive_dir, sample_config):
        """Read-only logs API must not fail a job during a transient actor miss."""
        job = JobRecord(
            job_id="orphan01",
            command="sleep 9999",
            num_gpus=1,
            status="running",
            submitted_at=time.time(),
            started_at=time.time(),
            stderr="",
        )
        job.save()

        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.get_actor.side_effect = ValueError("actor not found")
            mock_ray.init = MagicMock()

            from hive.dashboard import create_app
            app = create_app(sample_config, start_health_thread=False)
            client = app.test_client()

            resp = client.get(f"/api/jobs/orphan01/logs")
            data = resp.get_json()

        assert data["status"] == "running"
        assert data["returncode"] is None

        reloaded = JobRecord.load("orphan01")
        assert reloaded.status == "running"
        assert reloaded.completed_at is None

    def test_jobs_list_does_not_mutate_missing_actor_job(self, tmp_hive_dir, sample_config):
        """Read-only job listing must not mutate missing-actor jobs."""
        for i, status in enumerate(["running", "queued", "completed"]):
            JobRecord(
                job_id=f"list{i:02d}",
                command="echo hi",
                num_gpus=1,
                status=status,
                submitted_at=time.time(),
                started_at=time.time() if status == "running" else None,
                stderr="",
            ).save()

        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.get_actor.side_effect = ValueError("actor not found")
            mock_ray.init = MagicMock()

            from hive.dashboard import create_app
            app = create_app(sample_config, start_health_thread=False)
            client = app.test_client()

            resp = client.get("/api/jobs")
            jobs = resp.get_json()

        by_id = {j["job_id"]: j for j in jobs}
        assert by_id["list00"]["status"] == "running"
        assert by_id["list01"]["status"] == "queued"
        assert by_id["list02"]["status"] == "completed"

        reloaded = JobRecord.load("list00")
        assert reloaded.status == "running"

    def test_live_actor_not_marked_orphaned(self, tmp_hive_dir, sample_config):
        """A running job with a live actor stays running."""
        job = JobRecord(
            job_id="alive01",
            command="python train.py",
            num_gpus=1,
            status="running",
            submitted_at=time.time(),
            started_at=time.time(),
            stderr="",
        )
        job.save()

        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.get_actor.return_value = MagicMock()  # actor exists
            mock_ray.util.list_named_actors.return_value = [
                {"name": "hive_job_alive01"},
            ]
            mock_ray.init = MagicMock()

            from hive.dashboard import create_app
            app = create_app(sample_config, start_health_thread=False)
            client = app.test_client()

            resp = client.get(f"/api/jobs/alive01/logs")
            data = resp.get_json()

        assert data["status"] == "running"
        assert data["returncode"] is None

    def test_reap_orphaned_job_waits_for_repeated_misses(self, tmp_hive_dir):
        """Background orphan reaper should tolerate transient actor loss."""
        from hive.dashboard import reap_orphaned_job

        job = JobRecord(
            job_id="orphan02",
            command="sleep 9999",
            num_gpus=1,
            status="running",
            submitted_at=1000.0,
            started_at=1000.0,
        )
        job.save()
        write_job_heartbeat("orphan02", 1100.0)

        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = []
            assert reap_orphaned_job(job, now=1105.0) is True

        reloaded = JobRecord.load("orphan02")
        assert reloaded.status == "running"
        assert reloaded.missing_actor_checks == 1
        assert reloaded.missing_actor_since == 1105.0

    def test_reap_orphaned_job_preserves_logs_on_missing_actor_observation(self, tmp_hive_dir):
        """Missing-actor bookkeeping must not truncate streamed stdout/stderr."""
        from hive.dashboard import reap_orphaned_job

        job = JobRecord(
            job_id="orphan02_logs",
            command="sleep 9999",
            num_gpus=1,
            status="running",
            submitted_at=1000.0,
            started_at=1000.0,
            stdout="live stdout\n",
            stderr="live stderr\n",
        )
        job.save()
        write_job_heartbeat("orphan02_logs", 1100.0)

        active = JobRecord.load_active()
        record = next(r for r in active if r.job_id == "orphan02_logs")
        assert record.stdout == ""
        assert record.stderr == ""

        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = []
            assert reap_orphaned_job(record, now=1105.0) is True

        reloaded = JobRecord.load("orphan02_logs")
        assert reloaded.stdout == "live stdout\n"
        assert reloaded.stderr == "live stderr\n"

    def test_reap_orphaned_job_recovers_completed_process(self, tmp_hive_dir):
        """If the subprocess finished and wrote its result, recover completion."""
        from hive.dashboard import reap_orphaned_job

        job = JobRecord(
            job_id="orphan03",
            command="python train.py",
            num_gpus=1,
            status="running",
            submitted_at=1000.0,
            started_at=1000.0,
        )
        job.save()
        write_job_process_result(
            "orphan03",
            completed_at=1200.0,
            returncode=0,
            timed_out=False,
        )

        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = []
            assert reap_orphaned_job(job, now=1205.0) is True

        reloaded = JobRecord.load("orphan03")
        assert reloaded.status == "completed"
        assert reloaded.returncode == 0
        assert reloaded.completed_at == 1200.0

    def test_reap_orphaned_job_process_recovery_preserves_logs(self, tmp_hive_dir):
        """Process-result recovery must keep stdout/stderr already flushed to disk."""
        from hive.dashboard import reap_orphaned_job

        job = JobRecord(
            job_id="orphan03_logs",
            command="python train.py",
            num_gpus=1,
            status="running",
            submitted_at=1000.0,
            started_at=1000.0,
            stdout="finished stdout\n",
            stderr="finished stderr\n",
        )
        job.save()
        write_job_process_result(
            "orphan03_logs",
            completed_at=1200.0,
            returncode=0,
            timed_out=False,
        )

        active = JobRecord.load_active()
        record = next(r for r in active if r.job_id == "orphan03_logs")
        assert record.stdout == ""
        assert record.stderr == ""

        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = []
            assert reap_orphaned_job(record, now=1205.0) is True

        reloaded = JobRecord.load("orphan03_logs")
        assert reloaded.status == "completed"
        assert reloaded.stdout == "finished stdout\n"
        assert reloaded.stderr == "finished stderr\n"

    def test_append_stderr_message_does_not_reread_whole_log(self, tmp_hive_dir, monkeypatch):
        """Infrastructure notes should append in bounded fashion."""
        from hive.dashboard import _append_stderr_message

        job = JobRecord(
            job_id="stderrappend01",
            command="sleep 1",
            num_gpus=1,
            status="running",
        )
        job.save()
        job.stderr_path.write_text("existing stderr")
        orig_read_text = Path.read_text
        orig_write_text = Path.write_text

        def _forbid(*args, **kwargs):
            raise AssertionError("unbounded read/write helper used")

        monkeypatch.setattr(Path, "read_text", _forbid)
        monkeypatch.setattr(Path, "write_text", _forbid)

        _append_stderr_message(job, "[hive] appended note")

        assert orig_read_text(job.stderr_path) == "existing stderr\n[hive] appended note\n"

    def test_reap_orphaned_job_fails_after_prolonged_stale_heartbeat(self, tmp_hive_dir):
        """A truly stale missing actor should fail, but only after the grace window."""
        from hive.dashboard import reap_orphaned_job

        job = JobRecord(
            job_id="orphan04",
            command="sleep 9999",
            num_gpus=1,
            status="running",
            submitted_at=1000.0,
            started_at=1000.0,
            missing_actor_since=1100.0,
            missing_actor_checks=2,
        )
        job.save()
        write_job_heartbeat("orphan04", 1100.0)

        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = []
            assert reap_orphaned_job(job, now=2050.0) is True

        reloaded = JobRecord.load("orphan04")
        assert reloaded.status == "failed"
        assert reloaded.returncode == -9
        assert "orphaned" in reloaded.combined_stderr.lower()

    def test_reap_orphaned_job_fails_if_node_was_reassigned(self, tmp_hive_dir):
        """Use the specific node's GPU capacity, not the cluster-wide max."""
        from hive.dashboard import reap_orphaned_job

        old_job = JobRecord(
            job_id="orphan04b",
            command="sleep 9999",
            num_gpus=4,
            status="running",
            submitted_at=1000.0,
            started_at=1000.0,
            node_id="node-1",
            missing_actor_since=1100.0,
            missing_actor_checks=2,
        )
        old_job.save()
        write_job_heartbeat("orphan04b", 1100.0)

        new_job = JobRecord(
            job_id="replacement01",
            command="python serve.py",
            num_gpus=4,
            status="running",
            submitted_at=1300.0,
            started_at=1300.0,
            node_id="node-1",
        )
        new_job.save()
        write_job_heartbeat("replacement01", 1999.0)

        replacement_actor = MagicMock()
        with patch("hive.dashboard.Dispatcher.load_state", return_value={"max_gpus_per_node": 8}), \
             patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.side_effect = [
                [{"name": "hive_job_replacement01"}],
                [{"name": "hive_job_replacement01"}],
            ]
            mock_ray.get_actor.return_value = replacement_actor
            mock_ray.nodes.return_value = [
                {"NodeID": "node-1", "Alive": True, "Resources": {"GPU": 4}},
            ]
            assert reap_orphaned_job(old_job, now=2000.0) is True

        reloaded = JobRecord.load("orphan04b")
        assert reloaded.status == "failed"
        assert reloaded.returncode == -9
        assert "reassigned" in reloaded.combined_stderr.lower()

    def test_reap_orphaned_job_reassignment_requires_live_replacement_actor(self, tmp_hive_dir):
        """Do not fail early if the supposed replacement actor is not resolvable."""
        from hive.dashboard import reap_orphaned_job

        old_job = JobRecord(
            job_id="orphan04c",
            command="sleep 9999",
            num_gpus=4,
            status="running",
            submitted_at=1000.0,
            started_at=1000.0,
            node_id="node-1",
            missing_actor_since=1100.0,
            missing_actor_checks=2,
        )
        old_job.save()
        write_job_heartbeat("orphan04c", 1800.0)

        replacement = JobRecord(
            job_id="replacement02",
            command="python serve.py",
            num_gpus=4,
            status="running",
            submitted_at=1300.0,
            started_at=1300.0,
            node_id="node-1",
        )
        replacement.save()
        write_job_heartbeat("replacement02", 1999.0)

        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.side_effect = [
                [{"name": "hive_job_replacement02"}],
                [{"name": "hive_job_replacement02"}],
            ]
            mock_ray.get_actor.side_effect = ValueError("gone")
            mock_ray.nodes.return_value = [
                {"NodeID": "node-1", "Alive": True, "Resources": {"GPU": 4}},
            ]
            assert reap_orphaned_job(old_job, now=2000.0) is True

        reloaded = JobRecord.load("orphan04c")
        assert reloaded.status == "running"
        assert reloaded.missing_actor_checks == 3

    def test_reap_orphaned_job_sums_multiple_newer_jobs_on_same_node(self, tmp_hive_dir):
        """Multiple newer live jobs should count together for reassignment proof."""
        from hive.dashboard import reap_orphaned_job

        old_job = JobRecord(
            job_id="orphan04d",
            command="sleep 9999",
            num_gpus=1,
            status="running",
            submitted_at=1000.0,
            started_at=1000.0,
            node_id="node-2",
            missing_actor_since=1100.0,
            missing_actor_checks=2,
        )
        old_job.save()
        write_job_heartbeat("orphan04d", 1800.0)

        newer_a = JobRecord(
            job_id="replacement03",
            command="python serve_a.py",
            num_gpus=2,
            status="running",
            submitted_at=1300.0,
            started_at=1300.0,
            node_id="node-2",
        )
        newer_a.save()
        write_job_heartbeat("replacement03", 1999.0)

        newer_b = JobRecord(
            job_id="replacement04",
            command="python serve_b.py",
            num_gpus=2,
            status="running",
            submitted_at=1350.0,
            started_at=1350.0,
            node_id="node-2",
        )
        newer_b.save()
        write_job_heartbeat("replacement04", 1998.0)

        replacement_actor = MagicMock()
        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.side_effect = [
                [{"name": "hive_job_replacement03"}, {"name": "hive_job_replacement04"}],
                [{"name": "hive_job_replacement03"}, {"name": "hive_job_replacement04"}],
            ]
            mock_ray.get_actor.return_value = replacement_actor
            mock_ray.nodes.return_value = [
                {"NodeID": "node-2", "Alive": True, "Resources": {"GPU": 4}},
            ]
            assert reap_orphaned_job(old_job, now=2000.0) is True

        reloaded = JobRecord.load("orphan04d")
        assert reloaded.status == "failed"
        assert reloaded.returncode == -9
        assert "reassigned" in reloaded.combined_stderr.lower()

    def test_reap_orphaned_job_requeues_on_confirmed_node_death_no_actor(self, tmp_hive_dir):
        """Missing actor + confirmed dead node + stale heartbeat → preempted (requeue)."""
        from hive.dashboard import reap_orphaned_job

        job = JobRecord(
            job_id="orphan05",
            command="sleep 9999",
            num_gpus=1,
            status="running",
            submitted_at=0.0,
            started_at=0.0,  # old enough to exhaust missing-actor grace
            node_id="dead-node",
        )
        job.save()

        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = []
            mock_ray.nodes.return_value = [
                {"NodeID": "other-node", "Alive": True},
            ]
            # now = 1000, started_at = 0 → missing-actor grace exhausted.
            assert reap_orphaned_job(job, now=1000.0) is True

        reloaded = JobRecord.load("orphan05")
        # Missing actor + confirmed dead node → preempted + requeued
        # (no actor means nothing to kill; node confirmed dead; job
        # can't be running here).  See dashboard.reap_orphaned_job
        # "actor not found" branch.
        assert reloaded.status == "preempted"
        assert "worker node died" in reloaded.combined_stderr.lower()

    def test_reap_orphaned_job_kills_live_actor_only_when_heartbeat_stale(self, tmp_hive_dir):
        """Live head-side actor is only killed when BOTH heartbeat is stale AND node confirmed dead."""
        from hive.dashboard import reap_orphaned_job, _NODE_DEAD_CHECKS_REQUIRED
        from hive.dispatcher import write_job_heartbeat

        job = JobRecord(
            job_id="orphan06",
            command="sleep 9999",
            num_gpus=1,
            status="running",
            submitted_at=0.0,
            started_at=0.0,
            node_id="dead-node",
        )
        job.save()
        # Heartbeat well older than grace — stale.
        write_job_heartbeat("orphan06", 10.0)

        actor_handle = MagicMock()
        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = [{"name": "hive_job_orphan06"}]
            mock_ray.get_actor.return_value = actor_handle
            mock_ray.nodes.return_value = [
                {"NodeID": "other-node", "Alive": True},
            ]
            # Need _NODE_DEAD_CHECKS_REQUIRED consecutive confirmations
            # before the kill fires.  Each call advances the counter.
            for i in range(_NODE_DEAD_CHECKS_REQUIRED):
                assert reap_orphaned_job(job, now=10000.0 + i) is True
                job = JobRecord.load("orphan06")  # refresh counter state

        mock_ray.kill.assert_called_once_with(actor_handle)
        reloaded = JobRecord.load("orphan06")
        assert reloaded.status == "preempted"
        assert "worker node died" in reloaded.combined_stderr.lower()

    def test_reap_orphaned_job_leaves_live_actor_alone_with_fresh_heartbeat(self, tmp_hive_dir):
        """Even if ray.nodes() reports dead node, a fresh heartbeat means job is alive — do not kill."""
        from hive.dashboard import reap_orphaned_job
        from hive.dispatcher import write_job_heartbeat

        job = JobRecord(
            job_id="orphan07",
            command="sleep 9999",
            num_gpus=1,
            status="running",
            submitted_at=0.0,
            started_at=0.0,
            node_id="dead-node",
        )
        job.save()
        # Fresh heartbeat — job is definitely alive.
        write_job_heartbeat("orphan07", 9999.0)

        actor_handle = MagicMock()
        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = [{"name": "hive_job_orphan07"}]
            mock_ray.get_actor.return_value = actor_handle
            # ray.nodes() lies and says the node is dead.
            mock_ray.nodes.return_value = [
                {"NodeID": "other-node", "Alive": True},
            ]
            # Even across many calls, must never kill.
            for i in range(10):
                reap_orphaned_job(job, now=10000.0 + i)
                job = JobRecord.load("orphan07")

        mock_ray.kill.assert_not_called()
        reloaded = JobRecord.load("orphan07")
        assert reloaded.status == "running"

    def test_reap_orphaned_job_persists_dead_node_reset_on_fresh_heartbeat(self, tmp_hive_dir):
        """Clearing accumulated dead-node evidence must survive to the next tick.

        If the reset is only in memory, the next health tick reloads the
        stale counter from disk and a single later dead-node sample can
        satisfy _NODE_DEAD_CHECKS_REQUIRED, killing a job whose fresh
        heartbeat just proved it was alive.
        """
        from hive.dashboard import reap_orphaned_job
        from hive.dispatcher import write_job_heartbeat

        job = JobRecord(
            job_id="orphan08",
            command="sleep 9999",
            num_gpus=1,
            status="running",
            submitted_at=0.0,
            started_at=0.0,
            node_id="flaky-node",
            node_dead_checks=2,  # simulate earlier accumulated evidence
            node_dead_since=100.0,
        )
        job.save()
        # Fresh heartbeat — job is definitely alive now.
        write_job_heartbeat("orphan08", 9999.0)

        actor_handle = MagicMock()
        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = [{"name": "hive_job_orphan08"}]
            mock_ray.get_actor.return_value = actor_handle
            mock_ray.nodes.return_value = [
                {"NodeID": "other-node", "Alive": True},
            ]
            reap_orphaned_job(job, now=10000.0)

        reloaded = JobRecord.load("orphan08")
        assert reloaded.node_dead_checks == 0
        assert reloaded.node_dead_since is None
        # And ensure no kill happened.
        mock_ray.kill.assert_not_called()

    def test_reap_orphaned_job_ignores_stale_process_result_from_previous_attempt(self, tmp_hive_dir):
        """A retry must not be finalized by its predecessor's process_result.

        Identity is established by matching the attempt counter between
        record and result file — not by wall-clock comparisons, which
        fail under clock skew (worker vs head) and for fast retries.
        """
        from hive.dashboard import reap_orphaned_job

        # Current attempt is 3; stale result on disk is from attempt 2.
        job = JobRecord(
            job_id="orphan10",
            command="python train.py",
            num_gpus=1,
            status="running",
            submitted_at=0.0,
            started_at=5000.0,
            node_id="fresh-node",
            attempt=3,
        )
        job.save()
        write_job_process_result(
            job_id="orphan10",
            completed_at=500.0,
            returncode=137,
            timed_out=False,
            attempt=2,  # stale — predecessor attempt
        )

        actor_handle = MagicMock()
        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = [{"name": "hive_job_orphan10"}]
            mock_ray.get_actor.return_value = actor_handle
            mock_ray.nodes.return_value = [
                {"NodeID": "fresh-node", "Alive": True},
            ]
            write_job_heartbeat("orphan10", 5500.0)
            reap_orphaned_job(job, now=5501.0)

        mock_ray.kill.assert_not_called()
        reloaded = JobRecord.load("orphan10")
        assert reloaded.status == "running"

    def test_reap_orphaned_job_accepts_result_without_attempt_on_first_attempt(self, tmp_hive_dir):
        """A result file that carries no attempt number is accepted while the
        job is still on its first attempt, and finalizes the job.
        """
        from hive.dashboard import reap_orphaned_job
        from hive.dispatcher import _job_process_result_path
        import json as _json

        job = JobRecord(
            job_id="orphan13",
            command="python train.py",
            num_gpus=1,
            status="running",
            submitted_at=0.0,
            started_at=5000.0,
            node_id="fresh-node",
            attempt=0,
        )
        job.save()

        # The worker completes and writes a result without an attempt field.
        _job_process_result_path("orphan13").write_text(_json.dumps({
            "completed_at": 5500.0,
            "returncode": 0,
            "timed_out": False,
        }))

        actor_handle = MagicMock()
        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = [{"name": "hive_job_orphan13"}]
            mock_ray.get_actor.return_value = actor_handle
            mock_ray.nodes.return_value = [
                {"NodeID": "fresh-node", "Alive": True},
            ]
            assert reap_orphaned_job(job, now=5501.0) is True

        reloaded = JobRecord.load("orphan13")
        assert reloaded.status == "completed"
        assert reloaded.returncode == 0

    def test_requeue_preempted_bumps_attempt_counter(self, tmp_hive_dir):
        """attempt is bumped at the preempted→queued transition, not at
        preempt time.  Centralizing here guarantees every retry —
        including those whose preempt happened under pre-change code
        that didn't bump — gets a fresh attempt number before dispatch.
        """
        job = JobRecord(
            job_id="attempt_bump01",
            command="sleep 1",
            num_gpus=1,
            status="preempted",  # already preempted, e.g. from old code
            submitted_at=0.0,
            started_at=None,
            attempt=0,  # pre-change code didn't bump on preempt
        )
        job.save()

        result, updated = JobRecord.requeue_preempted("attempt_bump01")
        assert result == "queued"
        reloaded = JobRecord.load("attempt_bump01")
        assert reloaded.status == "queued"
        assert reloaded.attempt == 1  # bumped from 0 to 1 at requeue time

    def test_mark_preempted_does_not_bump_attempt(self, tmp_hive_dir):
        """mark_preempted (drain path) should NOT bump — requeue_preempted
        owns that responsibility.  Otherwise drained-then-requeued jobs
        would double-count max_retries budget.
        """
        job = JobRecord(
            job_id="drain_nobump01",
            command="sleep 1",
            num_gpus=1,
            status="running",
            submitted_at=0.0,
            started_at=100.0,
            attempt=2,
        )
        job.save()

        JobRecord.mark_preempted("drain_nobump01", note="drained")
        reloaded = JobRecord.load("drain_nobump01")
        assert reloaded.status == "preempted"
        assert reloaded.attempt == 2  # unchanged — bump is deferred to requeue

    def test_reap_orphaned_job_accepts_current_attempt_result_despite_clock_skew(self, tmp_hive_dir):
        """A result written on a worker with a skewed clock must still be accepted
        as long as its attempt number matches the record's.

        This exercises the regression the previous timestamp-based
        heuristic introduced: on clusters with clock skew > 1s between
        head and worker, a current attempt's own completed_at could
        be earlier than started_at and get rejected as "stale".
        """
        from hive.dashboard import reap_orphaned_job

        job = JobRecord(
            job_id="orphan11",
            command="python train.py",
            num_gpus=1,
            status="running",
            submitted_at=0.0,
            started_at=5000.0,  # head clock
            node_id="skewed-node",
            attempt=1,
        )
        job.save()
        # Worker-side clock is 10 seconds behind — completed_at from the
        # worker precedes started_at from the head by a wide margin.
        # Attempts match → must be accepted.
        write_job_process_result(
            job_id="orphan11",
            completed_at=4990.0,
            returncode=0,
            timed_out=False,
            attempt=1,
        )

        actor_handle = MagicMock()
        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = [{"name": "hive_job_orphan11"}]
            mock_ray.get_actor.return_value = actor_handle
            mock_ray.nodes.return_value = [
                {"NodeID": "skewed-node", "Alive": True},
            ]
            assert reap_orphaned_job(job, now=5001.0) is True

        reloaded = JobRecord.load("orphan11")
        assert reloaded.status == "completed"
        assert reloaded.returncode == 0

    def test_reap_orphaned_job_finalizes_from_process_result_when_actor_still_registered(self, tmp_hive_dir):
        """Persisted process_result is terminal — finalize, don't requeue.

        If the worker flushes its result file but the detached head-side
        actor is still registered, reap_orphaned_job must read that file
        and mark the record terminal, not accumulate dead-node evidence
        and eventually kill/requeue work that already completed.
        """
        from hive.dashboard import reap_orphaned_job

        job = JobRecord(
            job_id="orphan09",
            command="python train.py",
            num_gpus=1,
            status="running",
            submitted_at=0.0,
            started_at=0.0,
            node_id="dead-node",
        )
        job.save()
        # Worker wrote its result (success).
        write_job_process_result(
            job_id="orphan09",
            completed_at=999.0,
            returncode=0,
            timed_out=False,
        )

        actor_handle = MagicMock()
        with patch("hive.dashboard.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = [{"name": "hive_job_orphan09"}]
            mock_ray.get_actor.return_value = actor_handle
            # ray.nodes() reports the node dead — but process_result is
            # terminal, so reap must finalize before considering node death.
            mock_ray.nodes.return_value = [
                {"NodeID": "other-node", "Alive": True},
            ]
            assert reap_orphaned_job(job, now=10000.0) is True

        mock_ray.kill.assert_not_called()
        reloaded = JobRecord.load("orphan09")
        assert reloaded.status == "completed"
        assert reloaded.returncode == 0


class TestStreamingOutput:
    """Test that _run_job streams output to disk and respects timeout."""

    def test_build_job_log_preamble_contains_hive_marker(self, tmp_hive_dir):
        """The log preamble should include Hive marker and basic job metadata."""
        job = JobRecord(
            job_id="preamble01",
            command="python train.py",
            num_gpus=4,
            num_cpus=2,
            status="running",
            submitted_at=time.time(),
            user="tester",
            resource="gpu-a",
            node_id="node-123",
        )
        job.save()

        preamble = _build_job_log_preamble(
            "stdout",
            job_id="preamble01",
            command="python train.py",
            workdir="/tmp/work",
            timeout_sec=1800,
        )

        assert "managed by Hive" in preamble
        assert "job_id: preamble01" in preamble
        assert "resource: gpu-a" in preamble
        assert "gpus: 4" in preamble
        assert "cpus: 2" in preamble
        assert "workdir: /tmp/work" in preamble

    def test_run_job_writes_hive_preamble_to_both_logs(self, tmp_hive_dir):
        """_run_job should prepend a Hive marker block to stdout and stderr logs."""
        job = JobRecord(
            job_id="streampreamble01",
            command="echo hello-out && echo hello-err >&2",
            num_gpus=2,
            num_cpus=1,
            status="running",
            submitted_at=time.time(),
            user="tester",
            resource="gpu-a",
            node_id="node-abc",
        )
        job.save()

        result = _run_job._function(
            "echo hello-out && echo hello-err >&2",
            workdir="/tmp",
            job_id="streampreamble01",
            timeout_sec=1800,
        )

        stdout = (tmp_hive_dir / "jobs" / "streampreamble01" / "stdout.log").read_text()
        stderr = (tmp_hive_dir / "jobs" / "streampreamble01" / "stderr.log").read_text()

        assert result["returncode"] == 0
        assert "[hive] This stdout log is managed by Hive." in stdout
        assert "[hive] This stderr log is managed by Hive." in stderr
        assert "[hive] job_id: streampreamble01" in stdout
        assert "[hive] gpus: 2" in stdout
        assert "hello-out" in stdout
        assert "hello-err" in stderr

    def test_run_job_appends_to_existing_logs_instead_of_overwriting(self, tmp_hive_dir):
        """Retry/preemption notes should survive when a new _run_job attempt starts."""
        job = JobRecord(
            job_id="streamappend01",
            command="echo newer-out && echo newer-err >&2",
            num_gpus=1,
            num_cpus=1,
            status="running",
            submitted_at=time.time(),
        )
        job.save()
        job.stdout_path.write_text("older stdout\n")
        job.stderr_path.write_text("older stderr\n[hive] Preempted for retry\n")

        _run_job._function(
            "echo newer-out && echo newer-err >&2",
            workdir="/tmp",
            job_id="streamappend01",
            timeout_sec=1800,
        )

        stdout = job.stdout_path.read_text()
        stderr = job.stderr_path.read_text()
        assert stdout.startswith("older stdout\n")
        assert stderr.startswith("older stderr\n[hive] Preempted for retry\n")
        assert "newer-out" in stdout
        assert "newer-err" in stderr

    def test_output_written_to_files(self, tmp_hive_dir):
        """_run_job writes stdout/stderr to JOBS_DIR/<id>/ when job_id is provided."""
        import subprocess as sp

        # Simulate what _run_job does: write a job with known output
        job_id = "stream01"
        job_dir = tmp_hive_dir / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Run a simple command that produces known output
        proc = sp.run(
            "echo hello-stdout && echo hello-stderr >&2",
            shell=True, capture_output=True, text=True,
        )

        # Write output as _run_job would
        (job_dir / "stdout.log").write_text(proc.stdout)
        (job_dir / "stderr.log").write_text(proc.stderr)

        assert "hello-stdout" in (job_dir / "stdout.log").read_text()
        assert "hello-stderr" in (job_dir / "stderr.log").read_text()

    def test_timeout_kills_process(self, tmp_hive_dir):
        """Verify that a long-running process can be killed via timeout logic."""
        import os
        import select
        import subprocess as sp

        job_id = "timeout01"
        job_dir = tmp_hive_dir / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        timeout = 2  # seconds
        proc = sp.Popen(
            "sleep 60 && echo done",
            shell=True,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            text=True,
        )

        os.set_blocking(proc.stdout.fileno(), False)
        os.set_blocking(proc.stderr.fileno(), False)

        stderr_chunks = []
        start = time.time()

        while proc.poll() is None:
            select.select([proc.stdout, proc.stderr], [], [], 0.5)
            if time.time() - start > timeout:
                proc.kill()
                stderr_chunks.append(f"[hive] Process killed: timeout after {timeout}s\n")
                break

        proc.wait()
        stderr_full = "".join(stderr_chunks)
        (job_dir / "stderr.log").write_text(stderr_full)

        assert proc.returncode != 0  # killed
        assert "timeout" in (job_dir / "stderr.log").read_text()


class TestRetry:
    """Tests for max_retries / attempt auto-retry on infrastructure failure."""

    # ------------------------------------------------------------------
    # JobRecord serialization
    # ------------------------------------------------------------------

    def test_max_retries_roundtrip(self):
        """max_retries and attempt survive to_dict / from_dict."""
        r = JobRecord(
            job_id="r1", command="echo", num_gpus=1, status="queued",
            max_retries=3, attempt=1,
        )
        d = r.to_dict()
        assert d["max_retries"] == 3
        assert d["attempt"] == 1

        restored = JobRecord.from_dict(d)
        assert restored.max_retries == 3
        assert restored.attempt == 1

    def test_default_retries_omitted_from_dict(self):
        """Default values (0) should NOT appear in to_dict output."""
        r = JobRecord(job_id="r2", command="echo", num_gpus=1, status="queued")
        assert r.max_retries == 0
        assert r.attempt == 0
        d = r.to_dict()
        assert "max_retries" not in d
        assert "attempt" not in d

    def test_max_retries_persists_to_disk(self, tmp_hive_dir):
        """max_retries and attempt are saved to meta.json and reloaded."""
        r = JobRecord(
            job_id="r3", command="echo", num_gpus=1, status="queued",
            max_retries=2, attempt=1,
        )
        r.save()

        meta = json.loads((tmp_hive_dir / "jobs" / "r3" / "meta.json").read_text())
        assert meta["max_retries"] == 2
        assert meta["attempt"] == 1

        loaded = JobRecord.load("r3")
        assert loaded.max_retries == 2
        assert loaded.attempt == 1

    def test_from_dict_backwards_compatible(self):
        """Old records without max_retries/attempt should default to 0."""
        d = {"job_id": "old", "command": "echo", "num_gpus": 1, "status": "completed"}
        r = JobRecord.from_dict(d)
        assert r.max_retries == 0
        assert r.attempt == 0

    # ------------------------------------------------------------------
    # Preempted status lifecycle
    # ------------------------------------------------------------------

    def test_preempted_status_lifecycle(self, tmp_hive_dir):
        """Simulate: running → preempted → queued → running → completed."""
        job = JobRecord(
            job_id="retry01", command="python train.py", num_gpus=2,
            status="running", submitted_at=time.time(), started_at=time.time(),
            max_retries=2, attempt=0,
        )
        job.save()

        # Simulate infrastructure failure → preempted
        job.attempt += 1
        job.status = "preempted"
        job.started_at = None
        job.completed_at = None
        job.stderr = "Worker lost (attempt 1/2): NodeDiedError"
        job.save()

        loaded = JobRecord.load("retry01")
        assert loaded.status == "preempted"
        assert loaded.attempt == 1
        assert loaded.max_retries == 2
        assert "Worker lost" in loaded.stderr

        # Simulate retry → queued → running → completed
        job.status = "queued"
        job.save()
        job.status = "running"
        job.started_at = time.time()
        job.save()
        job.status = "completed"
        job.completed_at = time.time()
        job.returncode = 0
        job.stdout = "training done"
        job.stderr = ""
        job.save()

        final = JobRecord.load("retry01")
        assert final.status == "completed"
        assert final.attempt == 1
        assert final.returncode == 0

    def test_preempted_not_terminal_for_client_wait(self):
        """client.wait() must NOT treat 'preempted' as terminal."""
        from unittest.mock import MagicMock
        from hive.client import Client

        client = Client("http://localhost:9999")

        # Simulate: running → preempted → queued → running → completed
        client.logs = MagicMock(side_effect=[
            {"status": "running"},
            {"status": "preempted"},
            {"status": "queued"},
            {"status": "running"},
            {"status": "completed", "stdout": "done", "stderr": "", "returncode": 0},
        ])

        with patch("time.sleep"):
            result = client.wait("fake-job-id", poll_interval=0.01)

        assert result["status"] == "completed"
        assert client.logs.call_count == 5

    def test_cancelled_terminal_for_client_wait(self):
        """client.wait() should stop when a job becomes cancelled."""
        from unittest.mock import MagicMock
        from hive.client import Client

        client = Client("http://localhost:9999")
        client.logs = MagicMock(side_effect=[
            {"status": "running"},
            {"status": "cancelled", "stdout": "", "stderr": "[hive] Job cancelled", "returncode": -15},
        ])

        with patch("time.sleep"):
            result = client.wait("fake-job-id", poll_interval=0.01)

        assert result["status"] == "cancelled"
        assert result["returncode"] == -15
        assert client.logs.call_count == 2

    def test_cancel_job_local_marks_job_cancelled(self, tmp_hive_dir):
        """CLI local cancel helper should kill the actor and persist cancelled."""
        from hive.cli import _cancel_job_local

        record = JobRecord(
            job_id="cancel01",
            command="python train.py",
            num_gpus=1,
            status="running",
            submitted_at=time.time(),
            started_at=time.time(),
        )
        record.save()

        with patch("hive.cli.Dispatcher.load_state", return_value={"ray_head_address": "10.0.0.1:6379"}), \
             patch("hive.cli.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = [{"name": "hive_job_cancel01"}]
            result = _cancel_job_local("cancel01")

        assert result["status"] == "cancelled"
        assert result["actor_killed"] is True
        mock_ray.init.assert_called_once()
        mock_ray.kill.assert_called_once()
        mock_ray.shutdown.assert_called_once()

        reloaded = JobRecord.load("cancel01")
        assert reloaded.status == "cancelled"
        assert reloaded.returncode == -15
        assert reloaded.combined_stderr == "[hive] Job cancelled"
        assert (reloaded.job_dir / "events.log").read_text() == "[hive] Job cancelled\n"

    def test_cancel_job_local_noop_for_terminal_job(self, tmp_hive_dir):
        """CLI local cancel helper should not rewrite completed jobs."""
        from hive.cli import _cancel_job_local

        record = JobRecord(
            job_id="cancel02",
            command="python train.py",
            num_gpus=1,
            status="completed",
            submitted_at=time.time(),
            completed_at=time.time(),
            returncode=0,
        )
        record.save()

        with patch("hive.cli.ray") as mock_ray:
            result = _cancel_job_local("cancel02")

        assert result["status"] == "completed"
        assert "already completed" in result["message"]
        mock_ray.init.assert_not_called()

        reloaded = JobRecord.load("cancel02")
        assert reloaded.status == "completed"

    def test_cancel_job_local_refuses_running_job_when_ray_unreachable(self, tmp_hive_dir):
        """A running job must not be marked cancelled unless Ray is reachable."""
        from hive.cli import _cancel_job_local

        record = JobRecord(
            job_id="cancel03",
            command="python train.py",
            num_gpus=1,
            status="running",
            submitted_at=time.time(),
            started_at=time.time(),
        )
        record.save()

        with patch("hive.cli.ray.init", side_effect=RuntimeError("ray down")):
            result = _cancel_job_local("cancel03")

        assert "error" in result
        assert "Cannot reach Ray" in result["error"]

        reloaded = JobRecord.load("cancel03")
        assert reloaded.status == "running"
        assert reloaded.returncode is None

    def test_cancel_job_local_refuses_queued_job_when_ray_unreachable(self, tmp_hive_dir):
        """A queued job must not be cancelled unless Ray confirms actor state."""
        from hive.cli import _cancel_job_local

        record = JobRecord(
            job_id="cancel04",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=time.time(),
            priority="normal",
            runner_launched=True,
        )
        record.save()

        with patch("hive.cli.ray.init", side_effect=RuntimeError("ray down")):
            result = _cancel_job_local("cancel04")

        assert "error" in result
        assert "Cannot reach Ray" in result["error"]

        reloaded = JobRecord.load("cancel04")
        assert reloaded.status == "queued"
        assert reloaded.returncode is None

    def test_cancel_job_local_fails_closed_if_atomic_check_detects_started_runner(self, tmp_hive_dir):
        """A stale queued snapshot must not bypass Ray once activation wins the lock."""
        from hive.cli import _cancel_job_local

        initial = JobRecord(
            job_id="cancel04a",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=time.time(),
            priority="normal",
            runner_launched=False,
        )
        activated = JobRecord(
            job_id="cancel04a",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=initial.submitted_at,
            priority="normal",
            runner_launched=True,
        )

        with patch("hive.cli.JobRecord.load", side_effect=[initial, activated]), \
             patch("hive.cli.JobRecord.cancel_queued_if_not_started", return_value=("needs_ray", activated)), \
             patch("hive.cli.ray.init", side_effect=RuntimeError("ray down")):
            result = _cancel_job_local("cancel04a")

        assert "error" in result
        assert "Cannot reach Ray" in result["error"]

    def test_cancel_job_local_allows_recent_pending_launch_when_ray_unreachable(self, tmp_hive_dir):
        """A launch-pending queued job can cancel offline before the actor starts."""
        from hive.cli import _cancel_job_local

        record = JobRecord(
            job_id="cancel04b",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=time.time(),
            priority="normal",
            runner_launched=False,
            runner_launch_pending_since=time.time(),
        )
        record.save()

        with patch("hive.cli.ray.init", side_effect=RuntimeError("ray down")):
            result = _cancel_job_local("cancel04b")

        assert result["status"] == "cancelled"
        assert result["actor_killed"] is False

        reloaded = JobRecord.load("cancel04b")
        assert reloaded.status == "cancelled"
        assert reloaded.returncode == -15
        assert reloaded.runner_launched is False
        assert reloaded.runner_launch_pending_since is None

    def test_cancel_job_local_allows_stale_pending_launch_when_ray_unreachable(self, tmp_hive_dir):
        """A stale launch-pending queued job can also cancel offline."""
        from hive.cli import _cancel_job_local

        record = JobRecord(
            job_id="cancel04c",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=time.time(),
            priority="normal",
            runner_launched=False,
            runner_launch_pending_since=time.time() - 120,
        )
        record.save()

        with patch("hive.cli.ray.init", side_effect=RuntimeError("ray down")):
            result = _cancel_job_local("cancel04c")

        assert result["status"] == "cancelled"
        assert result["actor_killed"] is False

        reloaded = JobRecord.load("cancel04c")
        assert reloaded.status == "cancelled"
        assert reloaded.returncode == -15
        assert reloaded.runner_launched is False
        assert reloaded.runner_launch_pending_since is None

    def test_cancel_job_local_allows_low_priority_queue_without_runner(self, tmp_hive_dir):
        """A low-priority queued job with no launched runner can cancel offline."""
        from hive.cli import _cancel_job_local

        record = JobRecord(
            job_id="cancel05",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=time.time(),
            priority="low",
            runner_launched=False,
        )
        record.save()

        with patch("hive.cli.ray.init", side_effect=RuntimeError("ray down")):
            result = _cancel_job_local("cancel05")

        assert result["status"] == "cancelled"
        assert result["actor_killed"] is False

        reloaded = JobRecord.load("cancel05")
        assert reloaded.status == "cancelled"
        assert reloaded.returncode == -15
        assert reloaded.combined_stderr.endswith("[hive] Job cancelled")

    def test_cancel_job_local_allows_normal_queued_job_without_runner(self, tmp_hive_dir):
        """A queued job with runner_launched=False can cancel offline."""
        from hive.cli import _cancel_job_local

        record = JobRecord(
            job_id="cancel05b",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=time.time(),
            priority="normal",
            runner_launched=False,
        )
        record.save()

        with patch("hive.cli.ray.init", side_effect=RuntimeError("ray down")):
            result = _cancel_job_local("cancel05b")

        assert result["status"] == "cancelled"
        assert result["actor_killed"] is False

        reloaded = JobRecord.load("cancel05b")
        assert reloaded.status == "cancelled"
        assert reloaded.returncode == -15
        assert reloaded.combined_stderr.endswith("[hive] Job cancelled")

    def test_cancel_job_local_allows_queued_job_when_actor_disappears_mid_lookup(self, tmp_hive_dir):
        """A queued job can cancel if its actor vanishes between list and lookup."""
        from hive.cli import _cancel_job_local

        record = JobRecord(
            job_id="cancel06",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=time.time(),
            priority="normal",
            runner_launched=True,
        )
        record.save()

        with patch("hive.cli.Dispatcher.load_state", return_value={"ray_head_address": "10.0.0.1:6379"}), \
             patch("hive.cli.ray") as mock_ray:
            mock_ray.util.list_named_actors.return_value = [{"name": "hive_job_cancel06"}]
            mock_ray.get_actor.side_effect = ValueError("gone")
            result = _cancel_job_local("cancel06")

        assert result["status"] == "cancelled"
        assert result["actor_killed"] is False
        mock_ray.kill.assert_not_called()

        reloaded = JobRecord.load("cancel06")
        assert reloaded.status == "cancelled"
        assert reloaded.returncode == -15
        assert reloaded.combined_stderr.endswith("[hive] Job cancelled")

    # ------------------------------------------------------------------
    # _AsyncJobRunner retry logic (mocked Ray, real code path)
    #
    # Access the undecorated class via __ray_actor_class__ to instantiate
    # it directly, mocking all Ray calls.
    # ------------------------------------------------------------------

    def test_async_runner_retries_on_infrastructure_failure(self, tmp_hive_dir):
        """When ray.get raises (node death), runner should set status=preempted
        and spawn a new actor for the retry."""
        from unittest.mock import MagicMock, patch as _patch, call

        # Create a job record with max_retries=2
        record = JobRecord(
            job_id="infra01", command="python train.py", num_gpus=1,
            num_cpus=1, memory_mb=0, status="queued",
            submitted_at=time.time(), max_retries=2, attempt=0,
        )
        record.save()

        # Mock all Ray internals
        mock_pg = MagicMock()
        mock_pg.ready.return_value = "ready_future"

        mock_runner_cls = MagicMock()  # for the recursive spawn

        with _patch("hive.dispatcher.placement_group", return_value=mock_pg) as mock_pg_fn, \
             _patch("hive.dispatcher.remove_placement_group") as mock_rm_pg, \
             _patch("hive.dispatcher.ray") as mock_ray, \
             _patch("hive.dispatcher._run_job") as mock_run_job, \
             _patch("hive.dispatcher._get_job_node_id") as mock_get_node, \
             _patch("hive.dispatcher._AsyncJobRunner", mock_runner_cls):

            # ray.wait picks the winning PG, ray.get calls: node detection, then node death
            mock_run_job.options.return_value.remote.return_value = "job_future"
            mock_get_node.options.return_value.remote.return_value = "node_future"
            mock_ray.wait.return_value = (["ready_future"], [])
            mock_ray.get.side_effect = [
                "fake_node_id_12345678",                 # ray.get(_get_job_node_id) — node detection
                RuntimeError("NodeDiedError: worker crashed"),  # ray.get(future) — node died
            ]

            # Instantiate the undecorated class directly
            RawRunner = _AsyncJobRunner.__ray_actor_class__
            RawRunner(
                job_id="infra01", command="python train.py",
                num_gpus=1, num_cpus=1, memory_mb=0,
                workdir="/tmp", bundles=[{"CPU": 1, "GPU": 1}],
            )

        # Verify: record should be preempted.  attempt is bumped at
        # requeue_preempted, not at preempt time, so it stays at 0
        # here — the bump happens when the dashboard transitions the
        # record back to "queued" for retry.
        loaded = JobRecord.load("infra01")
        assert loaded.status == "preempted"
        assert loaded.attempt == 0
        assert loaded.max_retries == 2
        assert loaded.runner_launched is False
        assert "Worker lost" in loaded.combined_stderr
        assert "NodeDiedError" in loaded.combined_stderr

        # After requeue_preempted, attempt advances to 1.
        JobRecord.requeue_preempted("infra01")
        reloaded = JobRecord.load("infra01")
        assert reloaded.attempt == 1
        assert reloaded.status == "queued"

        # Verify: no retry actor spawned (health loop handles re-dispatch)
        mock_runner_cls.options.assert_not_called()

        # Verify: placement group was cleaned up
        mock_rm_pg.assert_called_once_with(mock_pg)

    def test_submit_async_does_not_overwrite_newer_state_after_remote(self, tmp_hive_dir, sample_config):
        """submit_async must not revert a job after the actor writes newer state."""
        dispatcher = Dispatcher(sample_config)

        captured = {}

        def fake_remote(job_id, *_args, **_kwargs):
            captured["job_id"] = job_id
            newer = JobRecord.load(job_id)
            newer.status = "completed"
            newer.started_at = time.time()
            newer.completed_at = time.time()
            newer.returncode = 0
            newer.runner_launched = False
            newer.save()

        with patch("hive.dispatcher._AsyncJobRunner") as mock_runner, \
             patch("hive.dispatcher.ray.util.list_named_actors", return_value=[]):
            mock_runner.options.return_value.remote.side_effect = fake_remote
            record = dispatcher.submit_async("echo done", num_gpus=1, priority="normal")

        loaded = JobRecord.load(record.job_id)
        assert loaded.job_id == captured["job_id"]
        assert loaded.status == "completed"
        assert loaded.returncode == 0
        assert loaded.runner_launched is False

    def test_submit_async_marks_runner_launch_pending_after_remote(self, tmp_hive_dir, sample_config):
        """submit_async should persist a pending launch marker after remote()."""
        dispatcher = Dispatcher(sample_config)

        with patch("hive.dispatcher._AsyncJobRunner") as mock_runner, \
             patch("hive.dispatcher.ray.util.list_named_actors", return_value=[]):
            mock_runner.options.return_value.remote.return_value = None
            record = dispatcher.submit_async("echo done", num_gpus=1, priority="normal")

        loaded = JobRecord.load(record.job_id)
        assert loaded.status == "queued"
        assert loaded.runner_launched is False
        assert loaded.runner_launch_pending_since is not None

    def test_async_runner_does_not_resurrect_cancelled_job(self, tmp_hive_dir):
        """A delayed actor must exit if the queued job was already cancelled."""
        record = JobRecord(
            job_id="cancelledrunner01",
            command="python train.py",
            num_gpus=1,
            num_cpus=1,
            memory_mb=0,
            status="cancelled",
            submitted_at=time.time(),
            completed_at=time.time(),
            returncode=-15,
            runner_launched=False,
            runner_launch_pending_since=time.time(),
        )
        record.save()

        with patch("hive.dispatcher.placement_group") as mock_pg:
            RawRunner = _AsyncJobRunner.__ray_actor_class__
            RawRunner(
                job_id="cancelledrunner01", command="python train.py",
                num_gpus=1, num_cpus=1, memory_mb=0,
                workdir="/tmp", bundles=[{"CPU": 1, "GPU": 1}],
            )

        mock_pg.assert_not_called()
        loaded = JobRecord.load("cancelledrunner01")
        assert loaded.status == "cancelled"
        assert loaded.returncode == -15
        assert loaded.runner_launched is False

    def test_async_runner_retries_when_placement_hits_draining_node(self, tmp_hive_dir):
        from unittest.mock import MagicMock, patch as _patch

        record = JobRecord(
            job_id="drainingrunner01",
            command="python train.py",
            num_gpus=1,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=time.time(),
        )
        record.save()

        first_pg = MagicMock()
        first_pg.ready.return_value = "ready_future_1"
        second_pg = MagicMock()
        second_pg.ready.return_value = "ready_future_2"

        with _patch("hive.dispatcher.placement_group", side_effect=[first_pg, second_pg]), \
             _patch("hive.dispatcher.remove_placement_group") as mock_rm_pg, \
             _patch("hive.dispatcher.ray") as mock_ray, \
             _patch("hive.dispatcher._get_job_node_id") as mock_get_node, \
             _patch("hive.dispatcher._run_job") as mock_run_job, \
             _patch("hive.dispatcher.is_node_draining", side_effect=[True, False]), \
             _patch("hive.dispatcher.time.sleep"):

            mock_get_node.options.return_value.remote.return_value = "node_future"
            mock_ray.wait.side_effect = [
                (["ready_future_1"], []),
                (["ready_future_2"], []),
                (["job_future"], []),
            ]
            mock_ray.get.side_effect = [
                "fake_node_id_12345678",
                "safe_node_id_12345678",
                {"stdout": "ok", "stderr": "", "returncode": 0, "timed_out": False},
            ]
            mock_run_job.options.return_value.remote.return_value = "job_future"

            RawRunner = _AsyncJobRunner.__ray_actor_class__
            RawRunner(
                job_id="drainingrunner01",
                command="python train.py",
                num_gpus=1,
                num_cpus=1,
                memory_mb=0,
                workdir="/tmp",
                bundles=[{"CPU": 1, "GPU": 1}],
            )

        loaded = JobRecord.load("drainingrunner01")
        assert loaded.status == "completed"
        assert loaded.node_id == "safe_node_id_12345678"
        assert loaded.runner_launched is False
        assert "[hive] Node draining before SLURM walltime; waiting for another node" in loaded.combined_stderr
        mock_run_job.options.assert_called_once()
        assert mock_rm_pg.call_count == 2

    def test_node_drain_requeues_jobs_without_dashboard(self, tmp_hive_dir):
        from unittest.mock import patch as _patch
        from hive.node_drain import drain_current_node

        record = JobRecord(
            job_id="drainhandoff01",
            command="python train.py",
            num_gpus=1,
            num_cpus=2,
            memory_mb=0,
            status="running",
            submitted_at=time.time(),
            started_at=time.time(),
            node_id="node-drain-1234",
            runner_launched=True,
        )
        record.save()

        launch_calls = []

        def fake_launch(record, *, workdir=None):
            launch_calls.append((record.job_id, workdir))
            JobRecord.mark_runner_launch_pending(record.job_id)
            return "launched", JobRecord.load(record.job_id)

        actor_handle = object()

        with _patch("hive.node_drain.ray.init"), \
             _patch("hive.node_drain.ray.shutdown"), \
             _patch("hive.node_drain.ray.kill") as mock_kill, \
             _patch("hive.node_drain._resolve_local_node", return_value={"NodeID": "node-drain-1234"}), \
             _patch("hive.node_drain.find_job_actor", side_effect=[actor_handle, None]), \
             _patch("hive.node_drain.launch_job_runner", side_effect=fake_launch), \
             _patch("hive.node_drain.load_config") as mock_load_config:
            mock_load_config.return_value.env.working_dir = "/tmp/retry-workdir"
            result = drain_current_node(
                "10.0.0.1:6379",
                resource_name="gpu-test",
                slurm_job_id="12345",
            )

        assert result["preempted_jobs"] == ["drainhandoff01"]
        assert result["requeued_jobs"] == ["drainhandoff01"]
        assert set(result) == {"node_id", "preempted_jobs", "requeued_jobs", "actor_kills"}
        assert result["actor_kills"] == 1
        mock_kill.assert_called_once_with(actor_handle)
        assert launch_calls == [("drainhandoff01", "/tmp/retry-workdir")]

        reloaded = JobRecord.load("drainhandoff01")
        assert reloaded.status == "queued"
        assert reloaded.runner_launched is False
        assert reloaded.runner_launch_pending_since is not None
        assert reloaded.combined_stderr.endswith("[hive] Preempted for SLURM worker drain before walltime")

    def test_node_drain_does_not_clobber_concurrent_redispatch(self, tmp_hive_dir):
        from unittest.mock import patch as _patch
        from hive.node_drain import drain_current_node

        record = JobRecord(
            job_id="drainhandoff02",
            command="python train.py",
            num_gpus=1,
            num_cpus=2,
            memory_mb=0,
            status="running",
            submitted_at=time.time(),
            started_at=time.time(),
            node_id="node-drain-5678",
            runner_launched=True,
        )
        record.save()

        actor_handle = object()
        pending_record = JobRecord(
            job_id="drainhandoff02",
            command="python train.py",
            num_gpus=1,
            num_cpus=2,
            memory_mb=0,
            status="queued",
            submitted_at=record.submitted_at,
            runner_launch_pending_since=time.time(),
        )

        with _patch("hive.node_drain.ray.init"), \
             _patch("hive.node_drain.ray.shutdown"), \
             _patch("hive.node_drain.ray.kill") as mock_kill, \
             _patch("hive.node_drain._resolve_local_node", return_value={"NodeID": "node-drain-5678"}), \
             _patch("hive.node_drain.find_job_actor", side_effect=[actor_handle, None]), \
             _patch("hive.node_drain.launch_job_runner", return_value=("already_launched", pending_record)) as mock_launch, \
             _patch("hive.node_drain.load_config") as mock_load_config:
            mock_load_config.return_value.env.working_dir = "/tmp/retry-workdir"
            result = drain_current_node(
                "10.0.0.1:6379",
                resource_name="gpu-test",
                slurm_job_id="12346",
            )

        assert result["preempted_jobs"] == ["drainhandoff02"]
        assert result["requeued_jobs"] == ["drainhandoff02"]
        mock_kill.assert_called_once_with(actor_handle)
        mock_launch.assert_called_once()

    def test_node_drain_accepts_recent_pending_relaunch(self, tmp_hive_dir):
        from unittest.mock import patch as _patch
        from hive.node_drain import drain_current_node

        record = JobRecord(
            job_id="drainhandoff03",
            command="python train.py",
            num_gpus=1,
            num_cpus=2,
            memory_mb=0,
            status="running",
            submitted_at=time.time(),
            started_at=time.time(),
            node_id="node-drain-9012",
            runner_launched=True,
        )
        record.save()

        actor_handle = object()
        pending_record = JobRecord(
            job_id="drainhandoff03",
            command="python train.py",
            num_gpus=1,
            num_cpus=2,
            memory_mb=0,
            status="queued",
            submitted_at=record.submitted_at,
            runner_launch_pending_since=time.time(),
        )

        with _patch("hive.node_drain.ray.init"), \
             _patch("hive.node_drain.ray.shutdown"), \
             _patch("hive.node_drain.ray.kill") as mock_kill, \
             _patch("hive.node_drain._resolve_local_node", return_value={"NodeID": "node-drain-9012"}), \
             _patch("hive.node_drain.find_job_actor", side_effect=[actor_handle, None]), \
             _patch("hive.node_drain.launch_job_runner", return_value=("launch_pending", pending_record)) as mock_launch, \
             _patch("hive.node_drain.load_config") as mock_load_config:
            mock_load_config.return_value.env.working_dir = "/tmp/retry-workdir"
            result = drain_current_node(
                "10.0.0.1:6379",
                resource_name="gpu-test",
                slurm_job_id="12347",
            )

        assert result["preempted_jobs"] == ["drainhandoff03"]
        assert result["requeued_jobs"] == ["drainhandoff03"]
        mock_kill.assert_called_once_with(actor_handle)
        mock_launch.assert_called_once()

    def test_async_runner_requeues_infrastructure_failure_past_max_retries(self, tmp_hive_dir):
        """A lost worker is an infrastructure failure, not a job failure: the
        job goes back to the queue even when max_retries is used up."""
        from unittest.mock import MagicMock, patch as _patch

        # Already on last attempt
        record = JobRecord(
            job_id="exhaust01", command="python train.py", num_gpus=1,
            num_cpus=1, memory_mb=0, status="queued",
            submitted_at=time.time(), max_retries=1, attempt=1,
        )
        record.save()

        mock_pg = MagicMock()
        mock_pg.ready.return_value = "ready_future"

        with _patch("hive.dispatcher.placement_group", return_value=mock_pg), \
             _patch("hive.dispatcher.remove_placement_group"), \
             _patch("hive.dispatcher.ray") as mock_ray, \
             _patch("hive.dispatcher._run_job") as mock_run_job, \
             _patch("hive.dispatcher._get_job_node_id") as mock_get_node, \
             _patch("hive.dispatcher._AsyncJobRunner") as mock_runner_cls:

            mock_run_job.options.return_value.remote.return_value = "job_future"
            mock_get_node.options.return_value.remote.return_value = "node_future"
            mock_ray.wait.return_value = (["ready_future"], [])
            mock_ray.get.side_effect = [
                "fake_node_id_12345678",                  # _get_job_node_id
                RuntimeError("NodeDiedError: worker lost"),  # node died
            ]

            RawRunner = _AsyncJobRunner.__ray_actor_class__
            RawRunner(
                job_id="exhaust01", command="python train.py",
                num_gpus=1, num_cpus=1, memory_mb=0,
                workdir="/tmp", bundles=[{"CPU": 1, "GPU": 1}],
            )

        loaded = JobRecord.load("exhaust01")
        assert loaded.status == "preempted"
        assert loaded.attempt == 1  # bumped later, when the job is requeued
        assert loaded.runner_launched is False
        assert "Worker lost" in loaded.combined_stderr
        assert "NodeDiedError" in loaded.combined_stderr

        # The runner itself does not spawn a replacement actor; the health
        # loop requeues preempted jobs.
        mock_runner_cls.options.assert_not_called()

    def test_async_runner_no_retry_on_success(self, tmp_hive_dir):
        """Normal completion should not trigger retry logic."""
        from unittest.mock import MagicMock, patch as _patch

        record = JobRecord(
            job_id="success01", command="echo done", num_gpus=1,
            num_cpus=1, memory_mb=0, status="queued",
            submitted_at=time.time(), max_retries=3, attempt=0,
        )
        record.save()

        mock_pg = MagicMock()
        mock_pg.ready.return_value = "ready_future"

        with _patch("hive.dispatcher.placement_group", return_value=mock_pg), \
             _patch("hive.dispatcher.remove_placement_group"), \
             _patch("hive.dispatcher.ray") as mock_ray, \
             _patch("hive.dispatcher._run_job") as mock_run_job, \
             _patch("hive.dispatcher._get_job_node_id") as mock_get_node, \
             _patch("hive.dispatcher._AsyncJobRunner") as mock_runner_cls:

            mock_run_job.options.return_value.remote.return_value = "job_future"
            mock_get_node.options.return_value.remote.return_value = "node_future"
            mock_ray.wait.return_value = (["ready_future"], [])
            mock_ray.get.side_effect = [
                "fake_node_id_12345678",                            # _get_job_node_id
                {"stdout": "done\n", "stderr": "", "returncode": 0, "timed_out": False},
            ]

            RawRunner = _AsyncJobRunner.__ray_actor_class__
            RawRunner(
                job_id="success01", command="echo done",
                num_gpus=1, num_cpus=1, memory_mb=0,
                workdir="/tmp", bundles=[{"CPU": 1, "GPU": 1}],
            )

        # Verify: completed, not retried
        loaded = JobRecord.load("success01")
        assert loaded.status == "completed"
        assert loaded.attempt == 0
        assert loaded.returncode == 0
        assert loaded.runner_launched is False
        assert loaded.stdout == "done\n"

        # Verify: NO retry actor spawned
        mock_runner_cls.options.assert_not_called()

    def test_async_runner_clears_runner_flag_when_returning_to_queue(self, tmp_hive_dir):
        """If placement never becomes ready, the queued record should clear runner_launched."""
        from unittest.mock import MagicMock, patch as _patch

        record = JobRecord(
            job_id="queueback01", command="sleep 10", num_gpus=1,
            num_cpus=1, memory_mb=0, status="queued",
            submitted_at=time.time(), max_retries=0, attempt=0,
        )
        record.save()

        mock_pg = MagicMock()
        mock_pg.ready.return_value = "ready_future"

        with _patch("hive.dispatcher.placement_group", return_value=mock_pg), \
             _patch("hive.dispatcher.remove_placement_group") as mock_rm_pg, \
             _patch("hive.dispatcher.ray") as mock_ray, \
             _patch("hive.dispatcher._targeted_preempt"):

            mock_ray.wait.side_effect = [([], []), ([], [])]

            RawRunner = _AsyncJobRunner.__ray_actor_class__
            RawRunner(
                job_id="queueback01", command="sleep 10",
                num_gpus=1, num_cpus=1, memory_mb=0,
                workdir="/tmp", bundles=[{"CPU": 1, "GPU": 1}],
            )

        loaded = JobRecord.load("queueback01")
        assert loaded.status == "queued"
        assert loaded.started_at is None
        assert loaded.runner_launched is False
        assert mock_rm_pg.call_count == 1

    def test_async_runner_skips_on_complete_when_returning_to_queue(self, tmp_hive_dir):
        """on_complete must not fire when the job is put back in the queue."""
        from unittest.mock import MagicMock, patch as _patch

        record = JobRecord(
            job_id="queueback02", command="sleep 10", num_gpus=1,
            num_cpus=1, memory_mb=0, status="queued",
            submitted_at=time.time(), max_retries=0, attempt=0,
            on_complete="echo done",
        )
        record.save()

        mock_pg = MagicMock()
        mock_pg.ready.return_value = "ready_future"

        with _patch("hive.dispatcher.placement_group", return_value=mock_pg), \
             _patch("hive.dispatcher.remove_placement_group"), \
             _patch("hive.dispatcher.ray") as mock_ray, \
             _patch("hive.dispatcher._targeted_preempt"), \
             _patch("hive.dispatcher.subprocess.run") as mock_sub_run:

            mock_ray.wait.side_effect = [([], []), ([], [])]

            RawRunner = _AsyncJobRunner.__ray_actor_class__
            RawRunner(
                job_id="queueback02", command="sleep 10",
                num_gpus=1, num_cpus=1, memory_mb=0,
                workdir="/tmp", bundles=[{"CPU": 1, "GPU": 1}],
            )

        assert JobRecord.load("queueback02").status == "queued"
        mock_sub_run.assert_not_called()

    def test_async_runner_long_job_records_node_id(self, tmp_hive_dir):
        """Jobs should record which node they ran on, even for longer runs."""
        from unittest.mock import MagicMock, patch as _patch

        record = JobRecord(
            job_id="long01", command="python train.py --epochs 100", num_gpus=2,
            num_cpus=4, memory_mb=8192, status="queued",
            submitted_at=time.time(), max_retries=1, attempt=0,
        )
        record.save()

        mock_pg = MagicMock()
        mock_pg.ready.return_value = "ready_future"
        fake_node_id = "ff991122aabbccdd"

        with _patch("hive.dispatcher.placement_group", return_value=mock_pg), \
             _patch("hive.dispatcher.remove_placement_group"), \
             _patch("hive.dispatcher.ray") as mock_ray, \
             _patch("hive.dispatcher._run_job") as mock_run_job, \
             _patch("hive.dispatcher._get_job_node_id") as mock_get_node:

            mock_run_job.options.return_value.remote.return_value = "job_future"
            mock_get_node.options.return_value.remote.return_value = "node_future"
            mock_ray.wait.return_value = (["ready_future"], [])
            mock_ray.get.side_effect = [
                fake_node_id,             # _get_job_node_id
                {"stdout": "loss: 0.01\n", "stderr": "", "returncode": 0, "timed_out": False},
            ]

            RawRunner = _AsyncJobRunner.__ray_actor_class__
            RawRunner(
                job_id="long01", command="python train.py --epochs 100",
                num_gpus=2, num_cpus=4, memory_mb=8192,
                workdir="/tmp", bundles=[{"CPU": 4, "GPU": 2}],
            )

        loaded = JobRecord.load("long01")
        assert loaded.status == "completed"
        assert loaded.node_id == fake_node_id
        assert loaded.num_gpus == 2
        assert loaded.stdout == "loss: 0.01\n"

    def test_async_runner_cpu_only_job(self, tmp_hive_dir):
        """CPU-only jobs (num_gpus=0) run to completion and record their node."""
        from unittest.mock import MagicMock, patch as _patch

        record = JobRecord(
            job_id="cpu01", command="python preprocess.py", num_gpus=0,
            num_cpus=8, memory_mb=4096, status="queued",
            submitted_at=time.time(), max_retries=0, attempt=0,
        )
        record.save()

        mock_pg = MagicMock()
        mock_pg.ready.return_value = "ready_future"
        fake_node_id = "deadbeef12345678"

        with _patch("hive.dispatcher.placement_group", return_value=mock_pg), \
             _patch("hive.dispatcher.remove_placement_group"), \
             _patch("hive.dispatcher.ray") as mock_ray, \
             _patch("hive.dispatcher._run_job") as mock_run_job, \
             _patch("hive.dispatcher._get_job_node_id") as mock_get_node:

            mock_run_job.options.return_value.remote.return_value = "job_future"
            mock_get_node.options.return_value.remote.return_value = "node_future"
            mock_ray.wait.return_value = (["ready_future"], [])
            mock_ray.get.side_effect = [
                fake_node_id,             # node detection
                {"stdout": "processed\n", "stderr": "", "returncode": 0, "timed_out": False},
            ]

            RawRunner = _AsyncJobRunner.__ray_actor_class__
            RawRunner(
                job_id="cpu01", command="python preprocess.py",
                num_gpus=0, num_cpus=8, memory_mb=4096,
                workdir="/tmp", bundles=[{"CPU": 8}],
            )

        loaded = JobRecord.load("cpu01")
        assert loaded.status == "completed"
        assert loaded.node_id == fake_node_id
        assert loaded.num_gpus == 0
