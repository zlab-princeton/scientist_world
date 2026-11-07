"""Tests for nexus.cli, the admin CLI.

Commands that talk to the daemon are run against the real Flask app
through its test client (see the ``cli`` fixture), so a mismatch between
what the CLI reads and what the API returns fails here.  ``nexus start``
is tested with the daemon subprocess and the TCP probe replaced by fakes;
nothing is spawned and no port is touched.
"""

import io
import json
import os
import signal
import sys
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from nexus import cli
from nexus.jobs import NexusJob
from nexus.project import Project

from .conftest import FAKE_HIVE_URL

# Above the largest PID Linux can assign, so it can never name a process.
IMPOSSIBLE_PID = 4194304 + 1000


@pytest.fixture
def api(daemon, monkeypatch):
    """Route the CLI's HTTP helper to the daemon app's test client."""

    def _api_request(base_url, path, method="GET", data=None):
        resp = daemon.client.open(path, method=method, json=data)
        body = resp.get_json()
        if resp.status_code >= 400:
            print(f"Error: {body.get('error')}", file=sys.stderr)
            sys.exit(1)
        return body

    monkeypatch.setattr(cli, "_api_request", _api_request)
    return daemon


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


def register_args(workspace, project_id="p1", *extra):
    return ["register", "--project-id", project_id, "--max-gpus", "2",
            "--workspace", str(workspace), "--sif", "/images/p.sif",
            "--resource", "gpu-a", *extra]


class TestStatus:
    def test_status_reads_the_api_response(self, api, capsys):
        assert cli.main(["status"]) == 0
        out = capsys.readouterr().out
        assert f"Nexus daemon: running (pid {os.getpid()}, uptime " in out
        assert "pid ?" not in out
        assert f"Hive connection: {FAKE_HIVE_URL} (OK)" in out
        assert "GPUs: 6/8 available (2 gpu node, 1 cpu node)" in out
        assert "Projects: none" in out
        assert "Jobs: 0 running, 0 completed, 0 failed" in out

    def test_status_counts_projects_and_jobs(self, api, workspace, capsys):
        cli.main(register_args(workspace, "p1"))
        cli.main(register_args(workspace, "p2"))
        api.client.post("/api/projects/p1/jobs", json={"command": "echo hi"})
        capsys.readouterr()

        assert cli.main(["status"]) == 0
        out = capsys.readouterr().out
        assert "Projects: 1 active, 1 registered" in out
        assert "Jobs: 1 running, 0 completed, 0 failed" in out

    def test_status_when_hive_is_unreachable(self, api, capsys):
        api.bridge.cluster_status.return_value = {
            "error": "refused", "total_gpus": 0, "free_gpus": 0}
        assert cli.main(["status"]) == 0
        assert f"Hive connection: {FAKE_HIVE_URL} (ERROR: unreachable)" in capsys.readouterr().out

    def test_connect_reports_pid_and_uptime(self, api, tmp_nexus_dir, capsys):
        status = api.client.get("/api/status").get_data()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(status)) as urlopen:
            assert cli.main(["connect", "nexus.invalid:39000"]) == 0
        request = urlopen.call_args[0][0]
        assert request.full_url == "http://nexus.invalid:39000/api/status"
        out = capsys.readouterr().out
        assert f"PID: {os.getpid()}, uptime: " in out
        remote = json.loads((tmp_nexus_dir / "remote.json").read_text())
        assert remote == {"daemon_url": "http://nexus.invalid:39000"}

        assert cli.main(["disconnect"]) == 0
        assert not (tmp_nexus_dir / "remote.json").exists()

    def test_unreachable_remote_daemon_is_disconnected(self, tmp_nexus_dir, capsys):
        (tmp_nexus_dir / "remote.json").write_text(
            json.dumps({"daemon_url": "http://nexus.invalid:39000"}))
        refused = urllib.error.URLError("connection refused")
        with patch("urllib.request.urlopen", side_effect=refused):
            with pytest.raises(SystemExit) as exc:
                cli.main(["status"])
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "cannot reach Nexus daemon at http://nexus.invalid:39000" in err
        assert not (tmp_nexus_dir / "remote.json").exists()

    def test_unreachable_local_daemon(self, capsys):
        refused = urllib.error.URLError("connection refused")
        with patch("urllib.request.urlopen", side_effect=refused):
            with pytest.raises(SystemExit) as exc:
                cli.main(["status"])
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "cannot reach Nexus daemon at http://localhost:39000" in err
        assert "nexus start" in err


class TestProjectCommands:
    def test_register_and_show(self, api, workspace, capsys):
        rc = cli.main(register_args(
            workspace, "p1", "--gpu-budget", "10", "--job-timeout", "120",
            "--checkpoint-interval", "600", "--checkpoint-timeout", "60",
            "--shared-bind", "/data:/data:ro", "--resource", "gpu-b"))
        assert rc == 0
        assert "Project 'p1' registered." in capsys.readouterr().out

        project = Project.load("p1")
        assert project.max_gpus == 2
        assert project.allowed_resources == ["gpu-a", "gpu-b"]
        assert project.workspace_dir == str(workspace)
        assert project.sif_path == "/images/p.sif"
        assert project.soft_gpu_budget_hours == 10
        assert project.default_job_timeout == 120
        assert project.checkpoint_interval == 600
        assert project.checkpoint_timeout == 60
        assert project.shared_dirs == ["/data:/data:ro"]

        assert cli.main(["projects"]) == 0
        listing = capsys.readouterr().out
        assert "p1" in listing and "registered" in listing and "0/2" in listing

        assert cli.main(["project", "p1"]) == 0
        detail = capsys.readouterr().out
        assert "Resources:           gpu-a, gpu-b" in detail
        assert "0.0 / 10.0 gpu-hours used (soft limit)" in detail
        assert "Checkpoint interval: 600s" in detail

    def test_register_requires_a_resource(self, workspace, capsys):
        argv = ["register", "--project-id", "p1", "--max-gpus", "2",
                "--workspace", str(workspace), "--sif", "/images/p.sif"]
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 2
        assert "--resource" in capsys.readouterr().err

    def test_register_duplicate_fails(self, api, workspace, capsys):
        assert cli.main(register_args(workspace)) == 0
        with pytest.raises(SystemExit) as exc:
            cli.main(register_args(workspace))
        assert exc.value.code == 1
        assert "already exists" in capsys.readouterr().err

    def test_extend_and_modify(self, api, workspace, capsys):
        cli.main(register_args(workspace, "p1", "--gpu-budget", "1"))
        assert cli.main(["extend", "p1", "--gpu-hours", "2.5"]) == 0
        assert Project.load("p1").soft_gpu_budget_hours == 3.5
        assert cli.main(["modify", "p1", "--max-gpus", "6"]) == 0
        assert Project.load("p1").max_gpus == 6
        assert cli.main(["modify", "p1"]) == 1  # nothing to change

    def test_deregister_deletes_project_and_jobs(self, api, workspace):
        cli.main(register_args(workspace))
        api.client.post("/api/projects/p1/jobs", json={"command": "echo hi"})
        assert cli.main(["deregister", "p1"]) == 0
        assert Project.load("p1") is None
        assert NexusJob.load_all() == []
        api.bridge.cancel.assert_called_once_with("hive-1")


class TestJobCommands:
    def test_jobs_logs_cancel(self, api, workspace, capsys):
        cli.main(register_args(workspace))
        run_id = api.client.post(
            "/api/projects/p1/jobs", json={"command": "echo hi"}).get_json()["run_id"]
        api.bridge.poll.return_value = {
            "status": "running", "stdout": "hello\n", "stderr": ""}
        api.monitor._tick()
        capsys.readouterr()

        assert cli.main(["jobs"]) == 0
        table = capsys.readouterr().out
        assert run_id in table and "running" in table and "echo hi" in table

        assert cli.main(["jobs", "--status", "completed"]) == 0
        assert "No jobs found." in capsys.readouterr().out

        assert cli.main(["logs", run_id]) == 0
        logs = capsys.readouterr().out
        assert "-- stdout --\nhello\n" in logs
        assert "(job is still running)" in logs

        assert cli.main(["cancel", run_id]) == 0
        assert NexusJob.load(run_id).status == "cancelled"

        # Output of an ended job is not served by the daemon.
        assert cli.main(["logs", run_id]) == 0
        logs = capsys.readouterr().out
        assert "-- stdout --" not in logs
        assert f"<workspace>/jobs/{run_id}/" in logs

    def test_cancel_all_jobs_of_a_project(self, api, workspace, capsys):
        cli.main(register_args(workspace))
        for _ in range(2):
            api.client.post("/api/projects/p1/jobs", json={"command": "echo hi"})
        assert cli.main(["cancel", "--project", "p1"]) == 0
        assert "Cancelled 2 job(s)" in capsys.readouterr().out
        assert Project.load("p1").gpus_in_use == 0

    def test_cancel_needs_a_target(self, capsys):
        assert cli.main(["cancel"]) == 1


class FakeProcess:
    """Stand-in for the Popen object of the daemon subprocess."""

    pid = IMPOSSIBLE_PID

    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


class TestStart:
    @pytest.fixture(autouse=True)
    def _hive_url(self, monkeypatch):
        monkeypatch.setenv("HIVE_URL", FAKE_HIVE_URL)

    @staticmethod
    def _start(argv, process, port_open):
        """Run ``nexus start`` with a fake subprocess and a fake TCP probe.

        *port_open* lists the successive answers of the probe.
        """
        with patch.object(cli.subprocess, "Popen", return_value=process) as popen, \
             patch.object(cli, "_port_open", side_effect=port_open), \
             patch.object(cli.time, "sleep"):
            rc = cli.main(["start", *argv])
        return rc, popen

    def test_start_succeeds_when_daemon_accepts_connections(
            self, tmp_nexus_dir, capsys):
        rc, popen = self._start([], FakeProcess(), port_open=[False, False, True])

        assert rc == 0
        argv = popen.call_args[0][0]
        assert argv == [sys.executable, "-m", "nexus.daemon",
                        "--host", "127.0.0.1", "--port", "39000"]
        assert popen.call_args[1]["start_new_session"] is True
        assert (tmp_nexus_dir / "daemon.pid").read_text() == str(IMPOSSIBLE_PID)
        out = capsys.readouterr().out
        assert f"Nexus daemon started (PID: {IMPOSSIBLE_PID})" in out
        assert "URL: http://127.0.0.1:39000" in out

    def test_start_reports_a_daemon_that_died(self, tmp_nexus_dir, capsys):
        rc, _ = self._start([], FakeProcess(returncode=1), port_open=[False])

        assert rc == 1
        captured = capsys.readouterr()
        assert "started" not in captured.out
        assert "daemon exited during startup (exit code 1)" in captured.err
        assert str(tmp_nexus_dir / "daemon.log") in captured.err
        assert not (tmp_nexus_dir / "daemon.pid").exists()

    def test_start_warns_when_daemon_is_slow(self, tmp_nexus_dir, capsys):
        with patch.object(cli, "_STARTUP_TIMEOUT_SEC", 0):
            rc, _ = self._start([], FakeProcess(), port_open=[False])
        assert rc == 0
        assert "not accepting connections" in capsys.readouterr().out
        assert (tmp_nexus_dir / "daemon.pid").exists()

    def test_start_passes_host_and_port_overrides(self, capsys):
        rc, popen = self._start(
            ["--host", "0.0.0.0", "--port", "40123"], FakeProcess(),
            port_open=[False, True])
        assert rc == 0
        assert popen.call_args[0][0][-4:] == ["--host", "0.0.0.0", "--port", "40123"]

    def test_start_refuses_a_port_that_is_in_use(self, tmp_nexus_dir, capsys):
        rc, popen = self._start([], FakeProcess(), port_open=[True])
        assert rc == 1
        popen.assert_not_called()
        assert "already in use" in capsys.readouterr().err
        assert not (tmp_nexus_dir / "daemon.pid").exists()

    def test_start_requires_hive_url(self, monkeypatch, tmp_nexus_dir, capsys):
        monkeypatch.delenv("HIVE_URL")
        rc, popen = self._start([], FakeProcess(), port_open=[False])
        assert rc == 1
        popen.assert_not_called()
        assert "hive.url is not configured" in capsys.readouterr().err
        assert not (tmp_nexus_dir / "daemon.pid").exists()

    def test_start_refuses_when_already_running(self, tmp_nexus_dir, capsys):
        (tmp_nexus_dir / "daemon.pid").write_text(str(os.getpid()))
        rc, popen = self._start([], FakeProcess(), port_open=[False])
        assert rc == 1
        popen.assert_not_called()
        assert "already running" in capsys.readouterr().out

    def test_start_foreground_serves_in_process(self, tmp_nexus_dir):
        import nexus.daemon as daemon_module
        served = []

        def fake_serve(config):
            served.append(config)
            assert (tmp_nexus_dir / "daemon.pid").read_text() == str(os.getpid())

        with patch.object(daemon_module, "serve", fake_serve):
            rc = cli.main(["start", "--foreground", "--port", "40123"])
        assert rc == 0
        assert served[0].daemon.port == 40123
        assert served[0].daemon.host == "127.0.0.1"
        assert not (tmp_nexus_dir / "daemon.pid").exists()

    def test_wait_for_daemon_uses_poll_not_a_signal(self):
        """An exited child that was not reaped still answers kill(pid, 0)."""
        with patch.object(cli, "_port_open", return_value=False), \
             patch.object(cli.os, "kill") as kill:
            assert cli._wait_for_daemon(FakeProcess(returncode=2), "127.0.0.1", 39000) is False
        kill.assert_not_called()


class TestStop:
    def test_stop_without_pid_file(self, capsys):
        assert cli.main(["stop"]) == 0
        assert "No Nexus daemon running" in capsys.readouterr().out

    def test_stop_refuses_remote_daemon(self, tmp_nexus_dir, capsys):
        (tmp_nexus_dir / "remote.json").write_text(
            json.dumps({"daemon_url": "http://nexus.invalid:39000"}))
        assert cli.main(["stop"]) == 1
        assert "cannot stop a remote daemon" in capsys.readouterr().err

    def test_stop_deletes_all_project_and_job_records(self, tmp_nexus_dir, capsys):
        """``nexus stop`` wipes the state; the README documents this."""
        Project(project_id="p1").save()
        NexusJob(run_id="j1", project_id="p1", status="running").save()
        (tmp_nexus_dir / "daemon.pid").write_text(str(IMPOSSIBLE_PID))

        # os.kill is replaced, so no signal is sent: the fake daemon is
        # alive for the liveness check and for SIGTERM, then gone.
        kill = MagicMock(side_effect=[None, None, ProcessLookupError(),
                                      ProcessLookupError()])
        with patch.object(cli.os, "kill", kill):
            assert cli.main(["stop"]) == 0

        assert kill.call_args_list[1][0] == (IMPOSSIBLE_PID, signal.SIGTERM)
        out = capsys.readouterr().out
        assert "Nexus daemon stopped." in out
        assert "Cleared all project and job state." in out
        assert Project.load_all() == []
        assert NexusJob.load_all() == []
        assert not (tmp_nexus_dir / "daemon.pid").exists()

    def test_stop_with_pid_reused_by_another_user(self, tmp_nexus_dir, capsys):
        """A PID now owned by someone else's process means our daemon is gone."""
        (tmp_nexus_dir / "daemon.pid").write_text(str(IMPOSSIBLE_PID))
        kill = MagicMock(side_effect=PermissionError())
        with patch.object(cli.os, "kill", kill):
            assert cli.main(["stop"]) == 0

        kill.assert_called_once_with(IMPOSSIBLE_PID, 0)
        assert "is not running" in capsys.readouterr().out
        assert not (tmp_nexus_dir / "daemon.pid").exists()


class TestParser:
    def test_no_command_prints_help(self, capsys):
        assert cli.main([]) == 1
        assert "usage: nexus" in capsys.readouterr().out

    def test_format_duration(self):
        assert cli._format_duration(None) == "-"
        assert cli._format_duration(45) == "45s"
        assert cli._format_duration(125) == "2m05s"
        assert cli._format_duration(3725) == "1h02m"

    def test_probe_host(self):
        assert cli._probe_host("0.0.0.0") == "127.0.0.1"
        assert cli._probe_host("::") == "::1"
        assert cli._probe_host("127.0.0.1") == "127.0.0.1"
        assert cli._probe_host("10.1.2.3") == "10.1.2.3"
