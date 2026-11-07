"""Round-trip tests of the daemon REST API.

The requests go through Flask's test client (no sockets) and Hive is a
mock, so these tests cover the routes, the persistence behind them and
the monitor together.  See the ``daemon`` fixture in conftest.py.
"""

import ast
import json
import logging
import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import nexus.daemon as daemon_module
from nexus.jobs import NexusJob
from nexus.project import Project

from .conftest import FAKE_HIVE_URL


class FakeClock:
    """Stand-in for the ``time`` module: each time() call is one second later."""

    def __init__(self):
        self.now = 0

    def time(self):
        self.now += 1
        return self.now

    def sleep(self, seconds):
        pass


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def register(daemon, workspace, project_id="p1", **fields):
    """Register a project and return the response."""
    body = {
        "project_id": project_id,
        "max_gpus": 2,
        "workspace_dir": str(workspace),
        "sif_path": "/images/p.sif",
        "allowed_resources": ["gpu-a"],
    }
    body.update(fields)
    return daemon.client.post("/api/projects", json=body)


def submit(daemon, project_id="p1", **fields):
    """Submit a job and return the response."""
    body = {"command": "echo hi", "num_gpus": 1}
    body.update(fields)
    return daemon.client.post(f"/api/projects/{project_id}/jobs", json=body)


class TestProjects:
    def test_create_get_list_delete(self, daemon, workspace):
        resp = register(daemon, workspace, soft_gpu_budget_hours=5)
        assert resp.status_code == 201
        assert resp.get_json() == {"project_id": "p1", "status": "registered"}

        project = daemon.client.get("/api/projects/p1").get_json()
        assert project["status"] == "registered"
        assert project["max_gpus"] == 2
        assert project["allowed_resources"] == ["gpu-a"]
        assert project["soft_gpu_budget_hours"] == 5
        assert project["gpus_in_use"] == 0
        assert project["job_counts"]["total"] == 0

        listing = daemon.client.get("/api/projects").get_json()
        assert [p["project_id"] for p in listing] == ["p1"]
        assert listing[0]["job_counts"]["running"] == 0

        resp = daemon.client.delete("/api/projects/p1")
        assert resp.get_json() == {"project_id": "p1", "status": "deleted"}
        assert daemon.client.get("/api/projects/p1").status_code == 404
        assert daemon.client.get("/api/projects").get_json() == []

    def test_create_requires_project_id(self, daemon):
        resp = daemon.client.post("/api/projects", json={"max_gpus": 1})
        assert resp.status_code == 400

    def test_create_rejects_path_separator(self, daemon, workspace, tmp_nexus_dir):
        resp = register(daemon, workspace, project_id="../escape")
        assert resp.status_code == 400
        assert not (tmp_nexus_dir / "escape.json").exists()

    def test_create_duplicate_is_conflict(self, daemon, workspace):
        assert register(daemon, workspace).status_code == 201
        assert register(daemon, workspace).status_code == 409

    def test_create_applies_config_defaults(self, daemon):
        daemon.client.post("/api/projects", json={"project_id": "bare"})
        project = daemon.client.get("/api/projects/bare").get_json()
        assert project["max_gpus"] == 4
        assert project["default_job_timeout"] == 3600
        assert project["soft_gpu_budget_hours"] == 0
        assert project["allowed_resources"] == []

    def test_unknown_project_is_404(self, daemon):
        assert daemon.client.get("/api/projects/nope").status_code == 404
        assert daemon.client.patch("/api/projects/nope", json={"max_gpus": 1}).status_code == 404
        assert daemon.client.delete("/api/projects/nope").status_code == 404
        assert daemon.client.get("/api/projects/nope/jobs").status_code == 404
        assert daemon.client.get("/api/projects/nope/wait?timeout=0").status_code == 404
        assert submit(daemon, "nope").status_code == 404

    def test_patch_updates_fields(self, daemon, workspace):
        register(daemon, workspace)
        resp = daemon.client.patch("/api/projects/p1", json={
            "max_gpus": 8,
            "allowed_resources": ["gpu-a", "gpu-b"],
            "sif_path": "/images/new.sif",
            "overlay_path": "/images/overlay.img",
            "default_job_timeout": 60,
        })
        assert resp.status_code == 200
        project = daemon.client.get("/api/projects/p1").get_json()
        assert project["max_gpus"] == 8
        assert project["allowed_resources"] == ["gpu-a", "gpu-b"]
        assert project["sif_path"] == "/images/new.sif"
        assert project["overlay_path"] == "/images/overlay.img"
        assert project["default_job_timeout"] == 60

    def test_patch_rejects_empty_allowed_resources(self, daemon, workspace):
        register(daemon, workspace)
        resp = daemon.client.patch("/api/projects/p1", json={"allowed_resources": []})
        assert resp.status_code == 400

    def test_budget_reset_and_extension_clear_exhausted_flag(self, daemon, workspace):
        register(daemon, workspace, soft_gpu_budget_hours=1)
        project = Project.load("p1")
        project.gpu_seconds_used = 7200
        project.budget_exhausted = True
        project.save()

        resp = submit(daemon)
        assert resp.status_code == 403
        assert "budget" in resp.get_json()["error"]

        # Raising the budget above the usage lifts the block ...
        patched = daemon.client.patch(
            "/api/projects/p1", json={"soft_gpu_budget_hours": 3}).get_json()
        assert patched["budget_exhausted"] is False

        # ... and so does resetting the usage counter.
        project = Project.load("p1")
        project.soft_gpu_budget_hours = 1
        project.budget_exhausted = True
        project.save()
        patched = daemon.client.patch(
            "/api/projects/p1", json={"gpu_seconds_used": 0}).get_json()
        assert patched["budget_exhausted"] is False
        assert patched["gpu_seconds_used"] == 0
        assert submit(daemon).status_code == 201

    def test_delete_cancels_active_jobs_and_removes_records(self, daemon, workspace):
        register(daemon, workspace)
        run_id = submit(daemon).get_json()["run_id"]

        assert daemon.client.delete("/api/projects/p1").status_code == 200

        daemon.bridge.cancel.assert_called_once_with("hive-1")
        assert daemon.monitor.tracked_jobs() == {}
        assert daemon.client.get(f"/api/jobs/{run_id}").status_code == 404
        assert NexusJob.load_all() == []
        assert Project.load_all() == []


class TestSubmit:
    def test_submit_roundtrip(self, daemon, workspace):
        register(daemon, workspace)
        resp = submit(daemon)
        assert resp.status_code == 201
        body = resp.get_json()
        run_id = body["run_id"]
        assert body == {
            "run_id": run_id,
            "hive_job_id": "hive-1",
            "project_id": "p1",
            "command": "echo hi",
            "num_gpus": 1,
            "resource": "gpu-a",
            "status": "queued",
            "timeout_sec": 3600,
            "project_gpus_in_use": 1,
            "project_max_gpus": 2,
            "gpu_seconds_used": 0,
            "soft_gpu_budget_hours": 0,
        }

        # Forwarded to Hive with the project's default timeout and resources.
        args, kwargs = daemon.bridge.submit_for_project.call_args
        assert args[0].project_id == "p1"
        assert args[1:] == ("echo hi", 1, 1, 3600)
        assert kwargs == {"resource": "gpu-a"}

        # Persisted, tracked, mirrored into the workspace.
        assert run_id in daemon.monitor.tracked_jobs()
        job = daemon.client.get(f"/api/jobs/{run_id}").get_json()
        assert job["status"] == "queued"
        assert job["hive_job_id"] == "hive-1"
        meta = json.loads((workspace / "jobs" / run_id / "meta.json").read_text())
        assert meta["run_id"] == run_id

        # The first job activates the project.
        assert daemon.client.get("/api/projects/p1").get_json()["status"] == "active"

    def test_missing_command_is_rejected(self, daemon, workspace):
        register(daemon, workspace)
        resp = daemon.client.post("/api/projects/p1/jobs", json={"num_gpus": 1})
        assert resp.status_code == 400
        daemon.bridge.submit_for_project.assert_not_called()

    def test_concurrency_cap(self, daemon, workspace):
        """The wording of this rejection is relied upon by retrying clients."""
        register(daemon, workspace)
        assert submit(daemon, num_gpus=2).status_code == 201
        resp = submit(daemon, num_gpus=1)
        assert resp.status_code == 403
        assert "2/2 in use" in resp.get_json()["error"]
        assert daemon.bridge.submit_for_project.call_count == 1

    def test_single_job_larger_than_cap(self, daemon, workspace):
        register(daemon, workspace)
        resp = submit(daemon, num_gpus=3)
        assert resp.status_code == 403
        assert Project.load("p1").gpus_in_use == 0

    def test_resource_must_be_allowed(self, daemon, workspace):
        register(daemon, workspace, allowed_resources=["gpu-a", "gpu-b"])
        resp = submit(daemon, resource="gpu-c")
        assert resp.status_code == 403
        assert Project.load("p1").gpus_in_use == 0

        resp = submit(daemon, resource="gpu-b")
        assert resp.status_code == 201
        assert resp.get_json()["resource"] == "gpu-b"

    def test_default_resource_is_every_allowed_one(self, daemon, workspace):
        register(daemon, workspace, allowed_resources=["gpu-a", "gpu-b"])
        assert submit(daemon).get_json()["resource"] == "gpu-a,gpu-b"

    def test_project_without_allowed_resources_accepts_any(self, daemon, workspace):
        register(daemon, workspace, allowed_resources=[])
        assert submit(daemon).get_json()["resource"] == ""
        assert submit(daemon, resource="anything").get_json()["resource"] == "anything"

    def test_explicit_timeout_overrides_project_default(self, daemon, workspace):
        register(daemon, workspace, default_job_timeout=100)
        assert submit(daemon).get_json()["timeout_sec"] == 100
        assert submit(daemon, timeout_sec=7).get_json()["timeout_sec"] == 7

    def test_hive_failure_releases_quota(self, daemon, workspace):
        register(daemon, workspace)
        daemon.bridge.submit_for_project.side_effect = RuntimeError("Hive submit error: boom")

        resp = submit(daemon)

        assert resp.status_code == 500
        assert "boom" in resp.get_json()["error"]
        assert Project.load("p1").gpus_in_use == 0
        assert daemon.monitor.tracked_jobs() == {}
        assert daemon.client.get("/api/projects/p1/jobs").get_json() == []

    def test_retry_with_same_request_id_is_replayed(self, daemon, workspace):
        register(daemon, workspace)
        first = submit(daemon, client_request_id="req-1")
        second = submit(daemon, client_request_id="req-1")

        assert first.status_code == second.status_code == 201
        assert first.get_json() == second.get_json()
        assert daemon.bridge.submit_for_project.call_count == 1
        assert Project.load("p1").gpus_in_use == 1
        assert len(daemon.client.get("/api/projects/p1/jobs").get_json()) == 1

    def test_different_request_ids_are_separate_jobs(self, daemon, workspace):
        register(daemon, workspace)
        first = submit(daemon, client_request_id="req-1").get_json()
        second = submit(daemon, client_request_id="req-2").get_json()
        assert first["run_id"] != second["run_id"]
        assert Project.load("p1").gpus_in_use == 2

    def test_retry_waits_for_an_original_that_is_still_in_flight(
            self, daemon, workspace):
        """A retry must not overtake its original while Hive is slow.

        Request "a" hangs in the Hive submit.  An unrelated request "b"
        completes meanwhile, and then "a" is retried.  The retry has to
        wait for the original and replay its response; it must not submit
        a second job or reserve a second GPU.
        """
        register(daemon, workspace, max_gpus=4)
        original_in_hive = threading.Event()
        release_original = threading.Event()
        duplicate_submitted = threading.Event()
        calls = []

        def hive_submit(project, command, *args, **kwargs):
            calls.append(command)
            if command == "a":
                if calls.count("a") == 1:
                    original_in_hive.set()
                    release_original.wait(timeout=30)
                else:
                    duplicate_submitted.set()
            return f"hive-{len(calls)}"

        daemon.bridge.submit_for_project.side_effect = hive_submit
        responses = {}

        def post(name, request_id):
            client = daemon.app.test_client()
            responses[name] = client.post("/api/projects/p1/jobs", json={
                "command": request_id, "client_request_id": request_id})

        original = threading.Thread(target=post, args=("original", "a"))
        original.start()
        assert original_in_hive.wait(timeout=10)
        post("other", "b")
        retry = threading.Thread(target=post, args=("retry", "a"))
        retry.start()
        # A correct retry is now blocked behind the original; give a wrong
        # one the time to reach Hive.
        overtook = duplicate_submitted.wait(timeout=0.5)
        release_original.set()
        original.join(timeout=30)
        retry.join(timeout=30)

        assert not overtook
        assert calls == ["a", "b"]
        assert responses["retry"].status_code == 201
        assert responses["retry"].get_json() == responses["original"].get_json()
        assert Project.load("p1").gpus_in_use == 2


class TestJobLifecycle:
    def test_run_to_completion(self, daemon, workspace):
        register(daemon, workspace, soft_gpu_budget_hours=10)
        run_id = submit(daemon).get_json()["run_id"]

        # Hive reports the job running: live logs are served.
        daemon.bridge.poll.return_value = {
            "status": "running", "stdout": "epoch 1\n", "stderr": ""}
        daemon.monitor._tick()
        job = daemon.client.get(f"/api/jobs/{run_id}").get_json()
        assert job["status"] == "running"
        assert job["started_at"] is not None
        assert job["hive_status"] == "running"
        assert job["stdout"] == "epoch 1\n"

        # Hive reports completion.
        daemon.bridge.poll.return_value = {
            "status": "completed", "stdout": "epoch 1\ndone\n", "stderr": "",
            "returncode": 0, "resource_name": "gpu-a",
        }
        daemon.monitor._tick()
        job = daemon.client.get(f"/api/jobs/{run_id}").get_json()
        assert job["status"] == "completed"
        assert job["returncode"] == 0
        assert job["actual_resource"] == "gpu-a"
        assert job["completed_at"] >= job["started_at"]
        # Output of a finished job is not served by the API ...
        assert "stdout" not in job
        # ... it is in the workspace.
        job_dir = workspace / "jobs" / run_id
        assert (job_dir / "stdout.log").read_text() == "epoch 1\ndone\n"
        result = json.loads((job_dir / "result.json").read_text())
        assert result["status"] == "completed"
        assert result["returncode"] == 0

        project = daemon.client.get("/api/projects/p1").get_json()
        assert project["gpus_in_use"] == 0
        assert project["gpu_seconds_used"] >= 0
        assert project["budget_exhausted"] is False
        assert project["job_counts"]["completed"] == 1
        assert daemon.monitor.tracked_jobs() == {}

    def test_nonzero_exit_is_failed(self, daemon, workspace):
        register(daemon, workspace)
        run_id = submit(daemon).get_json()["run_id"]
        daemon.bridge.poll.return_value = {
            "status": "completed", "stdout": "", "stderr": "boom\n", "returncode": 2}
        daemon.monitor._tick()
        job = daemon.client.get(f"/api/jobs/{run_id}").get_json()
        assert job["status"] == "failed"
        assert job["returncode"] == 2
        assert Project.load("p1").gpus_in_use == 0

    def test_budget_is_exhausted_after_last_job(self, daemon, workspace):
        register(daemon, workspace, soft_gpu_budget_hours=0.5)
        run_id = submit(daemon).get_json()["run_id"]
        daemon.monitor._tick()  # running
        # Pretend the job started an hour ago.
        daemon.monitor.tracked_jobs()[run_id].job.started_at -= 3600
        daemon.bridge.poll.return_value = {
            "status": "completed", "stdout": "", "stderr": "", "returncode": 0}
        daemon.monitor._tick()

        project = daemon.client.get("/api/projects/p1").get_json()
        assert project["budget_exhausted"] is True
        assert project["gpu_seconds_used"] >= 3600
        resp = submit(daemon)
        assert resp.status_code == 403
        assert "budget" in resp.get_json()["error"]

    def test_checkpoint_continue_cancel(self, daemon, workspace):
        register(daemon, workspace, checkpoint_interval=600)
        run_id = submit(daemon).get_json()["run_id"]
        daemon.bridge.poll.return_value = {
            "status": "running", "stdout": "step 1\n", "stderr": ""}
        daemon.monitor._tick()  # queued -> running

        # After checkpoint_interval seconds of run time the job is held
        # for review.
        daemon.monitor.tracked_jobs()[run_id].job.started_at -= 601
        daemon.monitor._tick()
        job = daemon.client.get(f"/api/jobs/{run_id}").get_json()
        assert job["status"] == "awaiting_response"
        assert job["checkpoint_count"] == 1
        checkpoint = json.loads(
            (workspace / "jobs" / run_id / "checkpoints" / "001.json").read_text())
        assert checkpoint["stdout"] == "step 1\n"

        # While a job awaits a decision the monitor does not poll Hive.
        polls = daemon.bridge.poll.call_count
        daemon.monitor._tick()
        assert daemon.bridge.poll.call_count == polls

        resp = daemon.client.post(f"/api/jobs/{run_id}/continue")
        assert resp.get_json() == {"run_id": run_id, "status": "running"}
        assert daemon.client.get(f"/api/jobs/{run_id}").get_json()["status"] == "running"

        resp = daemon.client.post(f"/api/jobs/{run_id}/cancel")
        assert resp.get_json() == {"run_id": run_id, "status": "cancelled"}
        daemon.bridge.cancel.assert_called_once_with("hive-1")
        assert daemon.client.get(f"/api/jobs/{run_id}").get_json()["status"] == "cancelled"
        assert Project.load("p1").gpus_in_use == 0

    def test_continue_and_cancel_report_status_of_finished_job(self, daemon, workspace):
        register(daemon, workspace)
        run_id = submit(daemon).get_json()["run_id"]
        daemon.bridge.poll.return_value = {
            "status": "completed", "stdout": "", "stderr": "", "returncode": 0}
        daemon.monitor._tick()
        assert daemon.client.get(f"/api/jobs/{run_id}").get_json()["status"] == "completed"

        resp = daemon.client.post(f"/api/jobs/{run_id}/continue")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "completed"
        assert "message" in resp.get_json()

        resp = daemon.client.post(f"/api/jobs/{run_id}/cancel")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "completed"
        assert "message" in resp.get_json()
        daemon.bridge.cancel.assert_not_called()

    def test_list_jobs_puts_active_first(self, daemon, workspace):
        register(daemon, workspace)
        first = submit(daemon).get_json()["run_id"]
        daemon.bridge.poll.return_value = {
            "status": "failed", "stdout": "", "stderr": "", "returncode": 1}
        daemon.monitor._tick()
        second = submit(daemon).get_json()["run_id"]

        jobs = daemon.client.get("/api/projects/p1/jobs").get_json()
        assert [j["run_id"] for j in jobs] == [second, first]
        assert [j["status"] for j in jobs] == ["queued", "failed"]

    def test_unknown_job_is_404(self, daemon):
        assert daemon.client.get("/api/jobs/nope").status_code == 404
        assert daemon.client.post("/api/jobs/nope/continue").status_code == 404
        assert daemon.client.post("/api/jobs/nope/cancel").status_code == 404


class TestWait:
    """GET /api/projects/<id>/wait."""

    def test_timeout_shape(self, daemon, workspace):
        register(daemon, workspace)
        resp = daemon.client.get("/api/projects/p1/wait?timeout=0")
        assert resp.status_code == 200
        assert resp.get_json() == {"timeout": True}

    def test_watched_job_already_ended_returns_at_once(self, daemon, workspace):
        register(daemon, workspace)
        NexusJob(run_id="old", project_id="p1", status="completed").save()
        body = daemon.client.get("/api/projects/p1/wait?timeout=600&run_id=old").get_json()
        assert body == {"run_id": "old", "status": "completed", "event": "completed"}

    def test_checkpoint_is_an_event(self, daemon, workspace):
        register(daemon, workspace)
        NexusJob(run_id="j1", project_id="p1", status="awaiting_response").save()
        body = daemon.client.get("/api/projects/p1/wait?timeout=600&run_id=j1").get_json()
        assert body == {
            "run_id": "j1", "status": "awaiting_response", "event": "awaiting_response"}

    def test_job_ended_before_the_wait_is_not_reported_without_run_id(
            self, daemon, workspace):
        register(daemon, workspace)
        NexusJob(run_id="old", project_id="p1", status="completed").save()
        with patch("nexus.daemon.time", FakeClock()):
            body = daemon.client.get("/api/projects/p1/wait?timeout=3").get_json()
        assert body == {"timeout": True}

    def test_a_job_that_was_at_a_checkpoint_and_then_ends_is_reported(
            self, daemon, workspace):
        """It was already waiting for an answer when the wait began, so it
        used to be skipped for the whole wait."""
        register(daemon, workspace)
        waiting = NexusJob(run_id="j1", project_id="p1", status="awaiting_response")
        running = NexusJob(run_id="j1", project_id="p1", status="running")
        done = NexusJob(run_id="j1", project_id="p1", status="completed")
        with patch.object(NexusJob, "load_by_project",
                          side_effect=[[waiting], [waiting], [running], [done]]), \
                patch("nexus.daemon.time", FakeClock()):
            body = daemon.client.get("/api/projects/p1/wait?timeout=30").get_json()
        assert body == {"run_id": "j1", "status": "completed", "event": "completed"}

    def test_event_during_the_wait_is_reported(self, daemon, workspace):
        register(daemon, workspace)
        running = NexusJob(run_id="j1", project_id="p1", status="running")
        failed = NexusJob(run_id="j1", project_id="p1", status="failed")
        with patch.object(NexusJob, "load_by_project",
                          side_effect=[[running], [failed]]):
            body = daemon.client.get("/api/projects/p1/wait?timeout=30").get_json()
        assert body == {"run_id": "j1", "status": "failed", "event": "failed"}

    def test_wait_is_capped_at_ten_minutes(self, daemon, workspace):
        register(daemon, workspace)
        clock = FakeClock()
        with patch("nexus.daemon.time", clock):
            body = daemon.client.get("/api/projects/p1/wait?timeout=100000").get_json()
        assert body == {"timeout": True}
        # The deadline was set at t=1 and the loop gave up at t=601:
        # 600 seconds after the start, not 100000.
        assert clock.now == 601


class TestStatus:
    def test_status_shape(self, daemon, workspace):
        register(daemon, workspace)
        register(daemon, workspace, project_id="p2")
        submit(daemon)

        body = daemon.client.get("/api/status").get_json()

        assert set(body) == {"daemon", "hive", "projects", "jobs"}
        assert body["daemon"]["pid"] == os.getpid()
        assert body["daemon"]["uptime"] >= 0
        assert body["hive"] == {
            "url": FAKE_HIVE_URL,
            "connected": True,
            "total_gpus": 8,
            "available_gpu": 6,
            "num_gpu_nodes": 2,
            "num_cpu_nodes": 1,
        }
        assert body["projects"] == {"total": 2, "active": 1, "budget_exhausted": 0}
        assert body["jobs"] == {"running": 1, "completed": 0, "failed": 0}

    def test_status_when_hive_is_unreachable(self, daemon):
        daemon.bridge.cluster_status.return_value = {
            "error": "connection refused", "total_gpus": 0, "free_gpus": 0}
        body = daemon.client.get("/api/status").get_json()
        assert body["hive"] == {"url": FAKE_HIVE_URL, "connected": False}

    def test_dashboard_page(self, daemon):
        resp = daemon.client.get("/")
        assert resp.status_code == 200
        assert b"<title>Nexus Dashboard</title>" in resp.data


class TestEntryPoint:
    """create_app, serve and ``python -m nexus.daemon``."""

    def test_requires_hive_url(self):
        from nexus.config import NexusConfig
        with pytest.raises(ValueError, match="hive.url"):
            daemon_module.create_app(NexusConfig())

    def test_main_applies_host_and_port_overrides(self, monkeypatch):
        """python -m nexus.daemon --host/--port override the config."""
        monkeypatch.setenv("HIVE_URL", FAKE_HIVE_URL)
        served = []
        with patch.object(daemon_module, "serve", served.append):
            daemon_module.main(["--host", "0.0.0.0", "--port", "40123"])
        assert served[0].daemon.host == "0.0.0.0"
        assert served[0].daemon.port == 40123
        assert served[0].hive.url == FAKE_HIVE_URL

    def test_main_guard_is_the_last_statement(self):
        """Under -m, main() must run only after the whole module is defined."""
        tree = ast.parse(Path(daemon_module.__file__).read_text())
        last = tree.body[-1]
        assert isinstance(last, ast.If)
        assert "__main__" in ast.dump(last.test)

    def test_serve_runs_a_threaded_server(self, sample_config, caplog):
        app = MagicMock()
        with patch.object(daemon_module, "create_app", return_value=app) as create, \
             patch.object(daemon_module.logging, "basicConfig") as basic_config:
            daemon_module.serve(sample_config)
        create.assert_called_once_with(sample_config)
        app.run.assert_called_once_with(
            host="127.0.0.1", port=39000, debug=False, threaded=True)
        assert basic_config.call_args[1]["level"] == logging.INFO
        assert "no authentication" not in caplog.text

    def test_serve_warns_when_exposed(self, sample_config, caplog):
        sample_config.daemon.host = "0.0.0.0"
        with patch.object(daemon_module, "create_app", return_value=MagicMock()), \
             patch.object(daemon_module.logging, "basicConfig"):
            daemon_module.serve(sample_config)
        assert "Listening on 0.0.0.0: the API has no authentication" in caplog.text
