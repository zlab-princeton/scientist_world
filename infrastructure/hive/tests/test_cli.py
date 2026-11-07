"""Tests for Hive CLI relaunch helpers."""

from types import SimpleNamespace
from unittest.mock import patch

from hive.config import DASHBOARD_PID_PATH


def test_build_preserved_relaunch_state_recovers_live_tracked_jobs(tmp_hive_dir, sample_config, monkeypatch):
    """Missing state.json should still preserve live tracked jobs from slurm_tracking/."""
    import hive.cli as cli
    import hive.config as config_mod

    tracking_dir = tmp_hive_dir / "slurm_tracking"
    tracking_dir.mkdir()
    (tracking_dir / "111").write_text("test-gpu")
    (tracking_dir / "222").write_text("prod-gpu")

    monkeypatch.setattr(cli, "SLURM_TRACKING_DIR", tracking_dir)
    monkeypatch.setattr(config_mod, "SLURM_TRACKING_DIR", tracking_dir)

    with patch("hive.cli.subprocess.run") as mock_run, \
         patch("hive.cli.Dispatcher._detect_cluster_ip", return_value="10.0.0.1"):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "111\n"

        preserved = cli._build_preserved_relaunch_state(sample_config)

    assert preserved is not None
    assert preserved["slurm_jobs"] == {"111": "test-gpu"}
    assert preserved["ray_head_address"] == "10.0.0.1:6379"
    assert [r["name"] for r in preserved["resources"]] == ["test-gpu"]
    assert preserved["max_gpus_per_node"] == 2


def test_build_preserved_relaunch_state_ignores_unreadable_state(
    tmp_hive_dir, sample_config, monkeypatch
):
    """Unreadable state.json should fall back to tracked live jobs on disk."""
    import hive.cli as cli
    import hive.config as config_mod
    import hive.dispatcher as dispatcher_mod

    tracking_dir = tmp_hive_dir / "slurm_tracking"
    tracking_dir.mkdir()
    (tracking_dir / "111").write_text("test-gpu")
    (tmp_hive_dir / "state.json").write_text("{not-json")

    monkeypatch.setattr(cli, "SLURM_TRACKING_DIR", tracking_dir)
    monkeypatch.setattr(config_mod, "SLURM_TRACKING_DIR", tracking_dir)
    monkeypatch.setattr(dispatcher_mod, "STATE_PATH", tmp_hive_dir / "state.json")

    with patch("hive.cli.subprocess.run") as mock_run, \
         patch("hive.cli.Dispatcher._detect_cluster_ip", return_value="10.0.0.1"):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "111\n"

        preserved = cli._build_preserved_relaunch_state(sample_config)

    assert preserved is not None
    assert preserved["slurm_jobs"] == {"111": "test-gpu"}
    assert [r["name"] for r in preserved["resources"]] == ["test-gpu"]


def test_cmd_relaunch_rebuilds_state_before_restart(tmp_hive_dir, sample_config, monkeypatch):
    """Clean relaunch should rebuild state and preserve tracked jobs before restart."""
    import hive.cli as cli
    import hive.config as config_mod
    import hive.dispatcher as dispatcher_mod
    import hive.pool as pool_mod

    tracking_dir = tmp_hive_dir / "slurm_tracking"
    tracking_dir.mkdir()
    (tracking_dir / "111").write_text("test-gpu")

    dashboard_pid = tmp_hive_dir / "dashboard.pid"

    monkeypatch.setattr(cli, "HIVE_DIR", tmp_hive_dir)
    monkeypatch.setattr(cli, "SLURM_TRACKING_DIR", tracking_dir)
    monkeypatch.setattr(config_mod, "SLURM_TRACKING_DIR", tracking_dir)
    monkeypatch.setattr(config_mod, "DASHBOARD_PID_PATH", dashboard_pid)
    monkeypatch.setattr(dispatcher_mod, "STATE_PATH", tmp_hive_dir / "state.json")
    monkeypatch.setattr(pool_mod, "HIVE_DIR", tmp_hive_dir)
    monkeypatch.setattr(pool_mod, "STATE_PATH", tmp_hive_dir / "state.json")
    monkeypatch.setattr(pool_mod, "SLURM_TRACKING_DIR", tracking_dir)

    with patch("hive.cli.load_config", return_value=sample_config), \
         patch("hive.cli.ensure_hive_dir"), \
         patch("hive.cli._stop_dashboard") as mock_stop_dashboard, \
         patch("hive.cli._start_dashboard_background") as mock_start_dashboard, \
         patch("hive.cli.subprocess.run") as mock_run, \
         patch("hive.cli.Dispatcher._detect_cluster_ip", return_value="10.0.0.1"), \
         patch("hive.cli.Dispatcher.restart") as mock_restart:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "111\n"

        rc = cli.cmd_relaunch(SimpleNamespace(clean=True))

    assert rc == 0
    assert mock_stop_dashboard.called
    assert mock_restart.called
    assert mock_start_dashboard.called

    state = (tmp_hive_dir / "state.json").read_text()
    assert '"ray_head_address": "10.0.0.1:6379"' in state
    assert '"111"' in state
    assert '"resource_name": "test-gpu"' in state
    assert '"name": "test-gpu"' in state
    sbatch_path = tmp_hive_dir / "sbatch" / "test-gpu.sbatch"
    assert sbatch_path.exists()
    assert "Auto-requeued test-gpu worker" in sbatch_path.read_text()


def test_cmd_jobs_status_waiting_keeps_other_requested_statuses(capsys):
    """`--status waiting,running` must list running jobs as well as waiting ones."""
    import hive.cli as cli
    from hive.dispatcher import JobRecord

    records = [
        JobRecord(job_id="waitjob01", command="big", num_gpus=8, status="queued", submitted_at=1.0),
        JobRecord(job_id="queuejob1", command="small", num_gpus=1, status="queued", submitted_at=2.0),
        JobRecord(job_id="runjob001", command="run", num_gpus=1, status="running", submitted_at=3.0),
    ]

    with patch("hive.cli._get_remote_url", return_value=None), \
         patch("hive.cli.JobRecord.load_all", return_value=records), \
         patch("hive.cli.ray"), \
         patch("hive.cli._ray_head_address", return_value="10.0.0.1:6379"), \
         patch("hive.cli._get_max_gpus_per_node", return_value=4):
        rc = cli.cmd_jobs(SimpleNamespace(status="waiting,running"))

    out = capsys.readouterr().out
    assert rc == 0
    assert "waitjob01" in out
    assert "runjob001" in out
    assert "queuejob1" not in out
