"""Tests for the swarm command line: config checks, daemon start, logs."""

import argparse
import json

import pytest

from swarm import cli
from swarm.config import (
    NexusConnectionConfig,
    ResourcesConfig,
    SwarmConfig,
    WorkerConfig,
    WorkflowStep,
)


def _valid_config(tmp_path, **overrides) -> SwarmConfig:
    kwargs = dict(
        scout_data_dir=str(tmp_path / "data"),
        workspace_base_dir=str(tmp_path / "workspaces"),
        github_org="example-org",
        default_workflow="standard",
        workflows={"standard": [WorkflowStep(profile="p")]},
        worker=WorkerConfig(scaffold="codex", model="m", container_sif="/path/agent.sif"),
        nexus=NexusConnectionConfig(
            url="http://localhost:39000",
            resources=ResourcesConfig(gpu=["gpu-a"]),
        ),
    )
    kwargs.update(overrides)
    return SwarmConfig(**kwargs)


# ── _validate_config ──────────────────────────────────────────────────


def test_validate_config_accepts_a_complete_config(tmp_path):
    assert cli._validate_config(_valid_config(tmp_path)) is None


@pytest.mark.parametrize("overrides, fragment", [
    ({"nexus": NexusConnectionConfig(url="", resources=ResourcesConfig(gpu=["gpu-a"]))}, "nexus.url"),
    ({"scout_data_dir": ""}, "scout_data_dir"),
    ({"workspace_base_dir": ""}, "workspace_base_dir"),
    ({"nexus": NexusConnectionConfig(url="http://localhost:39000")}, "nexus.resources"),
    ({"workflows": {}}, "default_workflow 'standard' is not defined"),
])
def test_validate_config_reports_missing_settings(tmp_path, overrides, fragment):
    error = cli._validate_config(_valid_config(tmp_path, **overrides))
    assert error is not None and fragment in error


def test_validate_config_allows_running_without_a_default_workflow(tmp_path):
    config = _valid_config(tmp_path, default_workflow="", workflows={})
    assert cli._validate_config(config) is None


# ── _config_warnings ──────────────────────────────────────────────────


def test_config_warnings_none_for_a_complete_setup(tmp_path, monkeypatch):
    monkeypatch.setattr("swarm.workspace._get_github_token", lambda: "token")
    assert cli._config_warnings(_valid_config(tmp_path)) == []


def test_config_warnings_explain_missing_github_org(tmp_path, monkeypatch):
    monkeypatch.setattr("swarm.workspace._get_github_token", lambda: "token")
    warnings = cli._config_warnings(_valid_config(tmp_path, github_org=""))
    assert len(warnings) == 1
    assert "github_org is not set" in warnings[0] and "stalled" in warnings[0]


def test_config_warnings_explain_missing_github_token(tmp_path, monkeypatch):
    monkeypatch.setattr("swarm.workspace._get_github_token", lambda: None)
    warnings = cli._config_warnings(_valid_config(tmp_path))
    assert len(warnings) == 1
    assert "no GitHub token" in warnings[0] and "stalled" in warnings[0]


def test_config_warnings_flag_missing_container_and_default_worker(tmp_path, monkeypatch):
    monkeypatch.setattr("swarm.workspace._get_github_token", lambda: "token")
    warnings = cli._config_warnings(_valid_config(tmp_path, worker=WorkerConfig()))
    assert any("container_sif is not set" in w for w in warnings)
    assert any("no default worker" in w for w in warnings)


def test_config_warnings_flag_a_missing_local_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("swarm.workspace._get_github_token", lambda: "token")
    config = _valid_config(tmp_path, local_tmp_dir=str(tmp_path / "missing"))
    warnings = cli._config_warnings(config)
    assert len(warnings) == 1
    assert "local_tmp_dir" in warnings[0] and "does not exist" in warnings[0]


def test_get_github_token_reads_git_credentials(tmp_path, monkeypatch):
    from swarm import workspace

    monkeypatch.setattr(workspace.Path, "home", classmethod(lambda cls: tmp_path))
    assert workspace._get_github_token() is None
    (tmp_path / ".git-credentials").write_text(
        "https://someone:other-token@example.com\n"
        "https://someone:fake-token-value@github.com\n"
    )
    assert workspace._get_github_token() == "fake-token-value"


# ── daemon start ──────────────────────────────────────────────────────


def test_daemon_script_binds_the_given_host():
    script = cli._daemon_script("127.0.0.1", 39100)
    compile(script, "<swarm-daemon>", "exec")
    assert "app.run(host='127.0.0.1', port=39100" in script
    assert "0.0.0.0" not in script


class _FakeApp:
    def __init__(self):
        self.run_kwargs = None

    def run(self, **kwargs):
        self.run_kwargs = kwargs


class _FakeScheduler:
    def __init__(self, config):
        self.config = config
        self.shutdown_called = False

    def shutdown(self):
        self.shutdown_called = True


def _start_foreground(tmp_path, monkeypatch, config, port=None):
    app = _FakeApp()
    monkeypatch.setattr(cli, "_is_running", lambda: False)
    monkeypatch.setattr(cli, "ensure_swarm_dir", lambda: None)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "DAEMON_PID_PATH", tmp_path / "daemon.pid")
    monkeypatch.setattr(cli.signal, "signal", lambda *a: None)
    monkeypatch.setattr(cli, "setup_logging", lambda verbose=False: None)
    monkeypatch.setattr("swarm.workspace._get_github_token", lambda: "token")
    monkeypatch.setattr("swarm.scheduler.Scheduler", _FakeScheduler)
    monkeypatch.setattr("swarm.dashboard.create_app", lambda config, scheduler: app)
    args = argparse.Namespace(port=port, foreground=True, verbose=False)
    return cli.cmd_start(args), app


def test_stop_keeps_pid_file_when_daemon_outlives_the_wait(tmp_path, monkeypatch, capsys):
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("4242")
    monkeypatch.setattr(cli, "DAEMON_PID_PATH", pid_path)
    monkeypatch.setattr(cli, "_read_pid", lambda: 4242)
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: None)  # never exits
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)

    assert cli.cmd_stop(argparse.Namespace()) == 1
    assert pid_path.exists()
    assert "Swarm stopped" not in capsys.readouterr().out


def test_start_binds_dashboard_to_loopback_by_default(tmp_path, monkeypatch, capsys):
    rc, app = _start_foreground(tmp_path, monkeypatch, _valid_config(tmp_path))

    assert rc == 0
    assert app.run_kwargs["host"] == "127.0.0.1"
    assert app.run_kwargs["port"] == 39100
    assert "http://127.0.0.1:39100" in capsys.readouterr().out
    assert not (tmp_path / "daemon.pid").exists()  # removed on exit


def test_start_exposes_dashboard_only_when_configured(tmp_path, monkeypatch):
    config = _valid_config(tmp_path, dashboard_host="0.0.0.0", dashboard_port=39111)
    rc, app = _start_foreground(tmp_path, monkeypatch, config, port=39222)

    assert rc == 0
    assert app.run_kwargs["host"] == "0.0.0.0"
    assert app.run_kwargs["port"] == 39222  # --port overrides dashboard_port


def test_start_refuses_an_incomplete_config(tmp_path, monkeypatch, capsys):
    rc, app = _start_foreground(tmp_path, monkeypatch, _valid_config(tmp_path, workflows={}))

    assert rc == 1
    assert app.run_kwargs is None
    assert "default_workflow" in capsys.readouterr().out


def test_start_needs_the_worker_of_the_checkout(tmp_path, monkeypatch, capsys):
    """A regular `pip install` copies Swarm away from infrastructure/worker."""
    assert cli._worker_missing() is None  # this checkout has it

    monkeypatch.setattr(cli, "__file__", str(tmp_path / "site-packages" / "swarm" / "cli.py"))
    rc, app = _start_foreground(tmp_path, monkeypatch, _valid_config(tmp_path))

    assert rc == 1
    assert app.run_kwargs is None
    assert "pip install -e infrastructure/swarm" in capsys.readouterr().out


def test_start_prints_warnings(tmp_path, monkeypatch, capsys):
    rc, _app = _start_foreground(tmp_path, monkeypatch, _valid_config(tmp_path, github_org=""))

    assert rc == 0
    assert "Warning: github_org is not set" in capsys.readouterr().out


# ── client subcommands ────────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_requests(monkeypatch, tmp_path, response):
    requests = []

    def fake_urlopen(req, timeout=None):
        requests.append(req)
        return _FakeResponse(response)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(cli, "load_config", lambda: _valid_config(tmp_path))
    return requests


def test_verify_does_not_assume_a_profile_name(tmp_path, monkeypatch, capsys):
    requests = _capture_requests(monkeypatch, tmp_path, {"arxiv_id": "2512.01152", "status": "queued"})

    assert cli.main(["verify", "2512.01152"]) == 0

    assert requests[0].full_url == "http://localhost:39100/api/papers/2512.01152/iterate"
    assert json.loads(requests[0].data) == {"role": "verifier"}
    assert "workflow step or daemon default" in capsys.readouterr().out


def test_verify_passes_explicit_options(tmp_path, monkeypatch):
    requests = _capture_requests(monkeypatch, tmp_path, {"arxiv_id": "2512.01152", "status": "queued"})

    assert cli.main(["verify", "2512.01152", "--profile", "codex-high",
                     "--gpu-budget", "2.5", "--workflow-step", "7"]) == 0

    assert json.loads(requests[0].data) == {
        "role": "verifier", "profile": "codex-high", "gpu_budget": 2.5, "workflow_step": 7,
    }


def test_submit_sends_only_given_options(tmp_path, monkeypatch):
    requests = _capture_requests(monkeypatch, tmp_path, {"arxiv_id": "2512.01152", "status": "queued"})

    assert cli.main(["submit", "2512.01152", "--date", "2025-12-01", "--topic", "nlp"]) == 0

    assert requests[0].full_url == "http://localhost:39100/api/papers"
    assert json.loads(requests[0].data) == {
        "arxiv_id": "2512.01152", "date": "2025-12-01", "topics": ["nlp"],
    }


def test_removed_alias_and_cap_bucket_are_rejected(capsys):
    with pytest.raises(SystemExit):
        cli.main(["stop-all"])
    with pytest.raises(SystemExit):
        cli.main(["cap", "waiting", "3"])
    capsys.readouterr()
    assert cli._CAP_BUCKETS == ("total", "working", "overflow", "admitted", "building", "installing")


def test_cap_buckets_match_the_scheduler():
    from swarm.scheduler import Scheduler

    assert cli._CAP_BUCKETS == Scheduler._CAP_BUCKETS


# ── logs ──────────────────────────────────────────────────────────────


def test_logs_reads_the_trajectory_written_by_the_worker(tmp_path, monkeypatch, capsys):
    from infrastructure.worker.claude_code import ClaudeCodeWorker

    paper_dir = tmp_path / "papers" / "2512.01152"
    paper_dir.mkdir(parents=True)
    (paper_dir / "meta.json").write_text(json.dumps({
        "arxiv_id": "2512.01152", "status": "stalled", "milestone": "core_claim",
        "iteration": 3, "role": "reproducer", "error": "git push failed",
    }))
    # Use the worker's own writer so the test follows the real file format.
    worker = ClaudeCodeWorker("claude-code", "m", {})
    worker._turn_count = 2
    worker._trajectory = [
        {"turn": 1, "text": "ok", "stats": {}, "errors": [], "returncode": 0,
         "duration_s": 12.34, "session_id": "s"},
        {"turn": 2, "text": "", "stats": {}, "errors": ["rate limit"], "returncode": 1,
         "duration_s": 3.0, "session_id": "s"},
    ]
    worker.save_trajectory(str(paper_dir))
    monkeypatch.setattr(cli, "PAPERS_DIR", tmp_path / "papers")

    assert cli.main(["logs", "2512.01152"]) == 0

    out = capsys.readouterr().out
    assert "Status:    stalled" in out
    assert "Iteration: 3 (reproducer)" in out
    assert "Error:     git push failed" in out
    assert "Trajectory: 2 turns" in out
    assert "Turn 1: rc=0 (12.3s)" in out
    assert "Turn 2: rc=1 (3.0s) errors: rate limit" in out


def test_logs_without_trajectory_or_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PAPERS_DIR", tmp_path / "papers")
    assert cli.main(["logs", "missing"]) == 1
    assert "No logs found" in capsys.readouterr().out

    (tmp_path / "papers" / "p1").mkdir(parents=True)
    assert cli.main(["logs", "p1"]) == 0
    assert "No trajectory file found" in capsys.readouterr().out
