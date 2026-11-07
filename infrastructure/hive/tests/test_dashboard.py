"""Tests for the dashboard API: node counting, GPU availability, CPU stats, dispatch."""

import json
import types
import time
from unittest.mock import patch, MagicMock

import pytest


def _make_node(node_id, gpus=0, cpus=16, alive=True, partition=None, name=None):
    """Build a fake Ray node dict."""
    res = {"CPU": cpus}
    if gpus > 0:
        res["GPU"] = gpus
    if partition:
        res[f"partition:{partition}"] = 100.0
    return {
        "NodeID": node_id,
        "NodeName": name or f"node-{node_id[:8]}",
        "NodeManagerAddress": f"10.0.0.{hash(node_id) % 254 + 1}",
        "Alive": alive,
        "Resources": res,
    }


HEAD = _make_node("head0000aaaa", cpus=64, name="head-node")
GPU1 = _make_node("gpu1111bbbb", gpus=1, cpus=16, partition="gpu-a", name="gpu-node-1")
GPU2 = _make_node("gpu2222cccc", gpus=2, cpus=32, partition="gpu-test", name="gpu-node-2")
CPU1 = _make_node("cpu3333dddd", gpus=0, cpus=48, partition="cpu-pool", name="cpu-node-1")


def _ray_data(nodes, total_gpus=None, free_gpus=None):
    """Build fake _safe_ray_query result."""
    if total_gpus is None:
        total_gpus = sum(int(n["Resources"].get("GPU", 0)) for n in nodes)
    if free_gpus is None:
        free_gpus = total_gpus
    return {
        "nodes": nodes,
        "total_gpus": total_gpus,
        "total_cpus": sum(int(n["Resources"].get("CPU", 0)) for n in nodes),
        "free_gpus": free_gpus,
        "free_cpus": 100,
    }


@pytest.fixture
def app():
    """Create dashboard app with mocked Ray."""
    from hive.dashboard import create_app
    from hive.config import HiveConfig, SlurmConfig, RayConfig, EnvConfig

    config = HiveConfig(
        slurm=SlurmConfig(default_walltime="1:00:00", resources=[]),
        ray=RayConfig(head_port=6379),
        env=EnvConfig(),
    )
    with patch("hive.dashboard.ray"):
        application = create_app(config, start_health_thread=False)
        application.config["TESTING"] = True
        yield application


def test_start_dashboard_background_uses_safe_module_launcher(tmp_path, monkeypatch):
    """The normal start path must not inline a direct ``ray.init`` dashboard script."""
    import hive.cli as cli
    import hive.config as config_mod

    fake_pid_path = tmp_path / "dashboard.pid"
    fake_log_dir = tmp_path / "logs"
    fake_log_dir.mkdir()

    monkeypatch.setattr(cli, "HIVE_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "DASHBOARD_PID_PATH", fake_pid_path)

    launched = {}

    class FakeProc:
        pid = 424242

    def fake_popen(args, **kwargs):
        launched["args"] = args
        launched["kwargs"] = kwargs
        return FakeProc()

    with patch("subprocess.Popen", side_effect=fake_popen):
        ok = cli._start_dashboard_background(cli.HiveConfig(), port=38581)

    assert ok is True
    assert launched["args"] == [
        cli.sys.executable,
        "-m",
        "hive.dashboard",
        "--host",
        "127.0.0.1",
        "--port",
        "38581",
    ]
    assert fake_pid_path.read_text() == "424242"


def test_start_dashboard_background_honours_configured_host(tmp_path, monkeypatch):
    """ray.dashboard_host (or an explicit host) decides the bind address."""
    import hive.cli as cli
    import hive.config as config_mod

    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(cli, "HIVE_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "DASHBOARD_PID_PATH", tmp_path / "dashboard.pid")

    launched = []

    class FakeProc:
        pid = 1

    def fake_popen(args, **kwargs):
        launched.append(args)
        return FakeProc()

    config = cli.HiveConfig()
    config.ray.dashboard_host = "0.0.0.0"
    with patch("subprocess.Popen", side_effect=fake_popen):
        cli._start_dashboard_background(config, port=1234)
        (tmp_path / "dashboard.pid").unlink()
        cli._start_dashboard_background(config, port=1234, host="10.1.2.3")

    assert launched[0][launched[0].index("--host") + 1] == "0.0.0.0"
    assert launched[1][launched[1].index("--host") + 1] == "10.1.2.3"


# ---------------------------------------------------------------
# /api/nodes — head detection and CPU node inclusion
# ---------------------------------------------------------------

class TestApiNodes:
    def test_head_node_excluded(self, app):
        """Head node (no partition resource) should not appear in nodes list."""
        with patch("hive.dashboard._safe_ray_query", return_value=_ray_data([HEAD, GPU1])):
            with app.test_client() as c:
                resp = c.get("/api/nodes")
                data = resp.get_json()

        names = [n["name"] for n in data]
        assert "head-node" not in names
        assert "gpu-node-1" in names

    def test_cpu_worker_included(self, app):
        """CPU-only worker nodes (with partition) should appear in nodes list."""
        with patch("hive.dashboard._safe_ray_query", return_value=_ray_data([HEAD, GPU1, CPU1])):
            with app.test_client() as c:
                resp = c.get("/api/nodes")
                data = resp.get_json()

        names = [n["name"] for n in data]
        assert "cpu-node-1" in names
        assert "gpu-node-1" in names
        assert "head-node" not in names
        assert len(data) == 2

    def test_cpu_node_has_zero_gpus(self, app):
        """CPU-only nodes should report 0 GPUs."""
        with patch("hive.dashboard._safe_ray_query", return_value=_ray_data([HEAD, CPU1])):
            with app.test_client() as c:
                resp = c.get("/api/nodes")
                data = resp.get_json()

        cpu = [n for n in data if n["name"] == "cpu-node-1"][0]
        assert cpu["gpus"] == 0
        assert cpu["cpus"] == 48
        assert cpu["resource"] == "cpu-pool"

    def test_all_node_types(self, app):
        """Mix of head, GPU, and CPU nodes — only workers appear."""
        nodes = [HEAD, GPU1, GPU2, CPU1]
        with patch("hive.dashboard._safe_ray_query", return_value=_ray_data(nodes)):
            with app.test_client() as c:
                resp = c.get("/api/nodes")
                data = resp.get_json()

        assert len(data) == 3  # 2 GPU + 1 CPU, no head
        gpu_nodes = [n for n in data if n["gpus"] > 0]
        cpu_nodes = [n for n in data if n["gpus"] == 0]
        assert len(gpu_nodes) == 2
        assert len(cpu_nodes) == 1


# ---------------------------------------------------------------
# /api/status — node counts and available GPUs
# ---------------------------------------------------------------

class TestApiStatus:
    def _get_status(self, app, nodes, free_gpus=None, jobs=None):
        """Helper: call /api/status with mocked data."""
        if jobs is None:
            jobs = []
        ray_data = _ray_data(nodes, free_gpus=free_gpus)

        with patch("hive.dashboard._safe_ray_query", return_value=ray_data), \
             patch("hive.dashboard._ray_cache", {"data": ray_data, "ts": 1e18}), \
             patch("hive.dashboard.Dispatcher") as mock_disp, \
             patch("hive.dashboard.JobRecord") as mock_jr:
            mock_disp.load_state.return_value = {"ray_head_address": "10.0.0.1:6379"}
            mock_jr.load_all.return_value = jobs
            mock_jr.load_active.return_value = [j for j in jobs if getattr(j, 'status', '') not in ('completed', 'failed', 'cancelled', 'timeout')]
            with app.test_client() as c:
                resp = c.get("/api/status")
                return resp.get_json()

    def test_head_not_counted(self, app):
        """Head node should not be counted as GPU or CPU node."""
        data = self._get_status(app, [HEAD, GPU1])
        assert data["num_gpu_nodes"] == 1
        assert data["num_cpu_nodes"] == 0

    def test_cpu_node_counted(self, app):
        """CPU-only worker should be counted as cpu node."""
        data = self._get_status(app, [HEAD, GPU1, CPU1])
        assert data["num_gpu_nodes"] == 1
        assert data["num_cpu_nodes"] == 1

    def test_gpu_and_cpu_counts(self, app):
        """Full cluster: 2 GPU nodes + 1 CPU node + head."""
        data = self._get_status(app, [HEAD, GPU1, GPU2, CPU1])
        assert data["num_gpu_nodes"] == 2
        assert data["num_cpu_nodes"] == 1
        assert data["total_gpus"] == 3

    def test_available_gpu_equals_free(self, app):
        """available_gpu is exactly the number of free GPUs Ray reports."""
        data = self._get_status(app, [HEAD, GPU1, GPU2], free_gpus=2)
        assert data["total_gpus"] == 3
        assert data["free_gpus"] == 2
        assert data["available_gpu"] == 2

    def test_available_gpu_zero_when_all_busy(self, app):
        data = self._get_status(app, [HEAD, GPU1, GPU2], free_gpus=0)
        assert data["available_gpu"] == 0

    def test_status_fields(self, app):
        """The status payload is a fixed, documented set of fields."""
        data = self._get_status(app, [HEAD, GPU1, GPU2, CPU1], free_gpus=3)
        assert set(data) == {
            "total_gpus", "free_gpus", "available_gpu", "available_cpu",
            "num_gpu_nodes", "num_cpu_nodes", "ray_head",
        }

    def test_cpu_only_cluster(self, app):
        """Cluster with only CPU workers and head — no GPUs."""
        data = self._get_status(app, [HEAD, CPU1])
        assert data["total_gpus"] == 0
        assert data["num_gpu_nodes"] == 0
        assert data["num_cpu_nodes"] == 1
        assert data["available_gpu"] == 0
        assert data["available_cpu"] == 1

    def test_multiple_cpu_nodes(self, app):
        """Multiple CPU-only workers should all be counted."""
        cpu2 = _make_node("cpu4444eeee", gpus=0, cpus=64, partition="cpu-pool", name="cpu-node-2")
        data = self._get_status(app, [HEAD, GPU1, CPU1, cpu2])
        assert data["num_gpu_nodes"] == 1
        assert data["num_cpu_nodes"] == 2
        assert data["available_cpu"] == 2

    def test_running_cpu_job_reduces_available(self, app):
        """Running CPU-only job should reduce available_cpu."""
        cpu_job = MagicMock(status="running", num_gpus=0)
        data = self._get_status(app, [HEAD, GPU1, CPU1], jobs=[cpu_job])
        assert data["num_cpu_nodes"] == 1
        assert data["available_cpu"] == 0

    def test_gpu_job_does_not_reduce_available_cpu(self, app):
        """Running GPU job should not affect available_cpu."""
        gpu_job = MagicMock(status="running", num_gpus=1)
        data = self._get_status(app, [HEAD, GPU1, CPU1], jobs=[gpu_job])
        assert data["available_cpu"] == 1

    def test_completed_cpu_job_not_counted(self, app):
        """Completed CPU-only jobs should not reduce available_cpu."""
        done_job = MagicMock(status="completed", num_gpus=0)
        data = self._get_status(app, [HEAD, GPU1, CPU1], jobs=[done_job])
        assert data["available_cpu"] == 1

    def test_dead_nodes_not_counted(self, app):
        """Dead nodes should be excluded from all counts."""
        dead_gpu = _make_node("dead1111aaaa", gpus=2, partition="gpu-test", alive=False)
        dead_cpu = _make_node("dead2222bbbb", gpus=0, partition="cpu-pool", alive=False)
        data = self._get_status(app, [HEAD, GPU1, dead_gpu, CPU1, dead_cpu])
        assert data["num_gpu_nodes"] == 1
        assert data["num_cpu_nodes"] == 1

    def test_status_does_not_attach_to_ray(self, app):
        """/api/status must stay on the subprocess query path."""
        ray_data = _ray_data([HEAD, GPU1], free_gpus=1)
        with patch("hive.dashboard._safe_ray_query", return_value=ray_data), \
             patch("hive.dashboard._ray_cache", {"data": ray_data, "ts": 1e18}), \
             patch("hive.dashboard.Dispatcher") as mock_disp, \
             patch("hive.dashboard.JobRecord") as mock_jr, \
             patch("hive.dashboard.ray") as mock_ray:
            mock_disp.load_state.return_value = {"ray_head_address": "10.0.0.1:6379"}
            mock_jr.load_all.return_value = []
            mock_jr.load_active.return_value = []
            with app.test_client() as c:
                resp = c.get("/api/status")
                data = resp.get_json()

        assert resp.status_code == 200
        assert data["available_gpu"] == 1
        mock_ray.init.assert_not_called()
        mock_ray.util.list_named_actors.assert_not_called()


class TestDispatchQueued:
    def test_recent_pending_launch_reserves_gpus_for_current_dispatch_pass(self, tmp_hive_dir, tmp_path):
        """A recent pending launch should block over-dispatch in the same health-loop pass."""
        from hive.dashboard import create_app
        from hive.config import HiveConfig, SlurmConfig, RayConfig, EnvConfig
        from hive.dispatcher import JobRecord

        config = HiveConfig(
            slurm=SlurmConfig(default_walltime="1:00:00", resources=[]),
            ray=RayConfig(head_port=6379),
            env=EnvConfig(),
        )

        raylet = tmp_path / "session_latest" / "sockets" / "raylet"
        raylet.parent.mkdir(parents=True, exist_ok=True)
        raylet.write_text("")

        first = JobRecord(
            job_id="dispatchpending01",
            command="python a.py",
            num_gpus=4,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=1.0,
        )
        second = JobRecord(
            job_id="dispatchpending02",
            command="python b.py",
            num_gpus=4,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=2.0,
        )
        records = [first, second]

        captured = {}

        class DummyThread:
            def __init__(self, target=None, daemon=None):
                captured["target"] = target

            def start(self):
                return None

        load_active_calls = {"count": 0}

        def fake_load_active():
            load_active_calls["count"] += 1
            if load_active_calls["count"] in (3, 5):
                return records
            return []

        sleep_calls = {"count": 0}

        def fake_sleep(_secs):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                raise RuntimeError("stop loop")

        launch_calls = []

        def fake_launch(record, workdir=None, dispatch_resource=None, **kwargs):
            launch_calls.append(record.job_id)
            if record.job_id == "dispatchpending01":
                return "launch_pending", record
            return "launched", record

        with patch("hive.dashboard.threading.Thread", DummyThread), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard.Dispatcher") as mock_dispatcher:
            mock_dispatcher.load_state.return_value = {
                "ray_head_address": "10.0.0.1:6379",
                "resources": [],
            }
            app = create_app(config)

        target = captured["target"]
        assert target is not None
        app.config["TESTING"] = True

        with patch("hive.dashboard.JobRecord.load_active", side_effect=fake_load_active), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard._ray_cache", {"data": None, "ts": 0.0}), \
             patch("hive.dashboard._safe_ray_query", return_value=_ray_data([HEAD], total_gpus=4, free_gpus=4)), \
             patch("hive.dashboard.launch_job_runner", side_effect=fake_launch), \
             patch("hive.dashboard.ray") as mock_ray, \
             patch("hive.dashboard.time.sleep", side_effect=fake_sleep):
            mock_ray.util.list_named_actors.return_value = []
            with pytest.raises(RuntimeError, match="stop loop"):
                target()

        assert launch_calls
        assert set(launch_calls) == {"dispatchpending01"}

    def test_cpu_job_without_resource_dispatches_to_cpu_node(self, tmp_hive_dir, tmp_path):
        """A queued CPU-only job with no resource should fall back to any CPU pool."""
        from hive.dashboard import create_app
        from hive.config import HiveConfig, SlurmConfig, RayConfig, EnvConfig
        from hive.dispatcher import JobRecord

        config = HiveConfig(
            slurm=SlurmConfig(default_walltime="1:00:00", resources=[]),
            ray=RayConfig(head_port=6379),
            env=EnvConfig(),
        )

        raylet = tmp_path / "session_latest" / "sockets" / "raylet"
        raylet.parent.mkdir(parents=True, exist_ok=True)
        raylet.write_text("")

        cpu_job = JobRecord(
            job_id="dispatchcpu01",
            command="python prep.py",
            num_gpus=0,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=1.0,
        )
        records = [cpu_job]

        captured = {}

        class DummyThread:
            def __init__(self, target=None, daemon=None):
                captured["target"] = target

            def start(self):
                return None

        load_active_calls = {"count": 0}

        def fake_load_active():
            load_active_calls["count"] += 1
            if load_active_calls["count"] in (3, 5):
                return records
            return []

        sleep_calls = {"count": 0}

        def fake_sleep(_secs):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                raise RuntimeError("stop loop")

        launch_calls = []

        def fake_launch(record, workdir=None, dispatch_resource=None, **kwargs):
            launch_calls.append((record.job_id, dispatch_resource))
            return "launched", record

        with patch("hive.dashboard.threading.Thread", DummyThread), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard.Dispatcher") as mock_dispatcher:
            mock_dispatcher.load_state.return_value = {
                "ray_head_address": "10.0.0.1:6379",
                "resources": [],
            }
            app = create_app(config)

        target = captured["target"]
        assert target is not None
        app.config["TESTING"] = True

        with patch("hive.dashboard.JobRecord.load_active", side_effect=fake_load_active), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard._ray_cache", {"data": None, "ts": 0.0}), \
             patch("hive.dashboard._safe_ray_query", return_value=_ray_data([HEAD, CPU1])), \
             patch("hive.dashboard.launch_job_runner", side_effect=fake_launch), \
             patch("hive.dashboard.ray") as mock_ray, \
             patch("hive.dashboard.time.sleep", side_effect=fake_sleep):
            mock_ray.util.list_named_actors.return_value = []
            with pytest.raises(RuntimeError, match="stop loop"):
                target()

        assert launch_calls
        assert launch_calls[0] == ("dispatchcpu01", "cpu-pool")

    def test_pending_launch_only_blocks_compatible_resource_pool(self, tmp_hive_dir, tmp_path):
        """A pending launch on one partition should not throttle a different partition."""
        from hive.dashboard import create_app
        from hive.config import HiveConfig, SlurmConfig, RayConfig, EnvConfig
        from hive.dispatcher import JobRecord

        config = HiveConfig(
            slurm=SlurmConfig(default_walltime="1:00:00", resources=[]),
            ray=RayConfig(head_port=6379),
            env=EnvConfig(),
        )

        raylet = tmp_path / "session_latest" / "sockets" / "raylet"
        raylet.parent.mkdir(parents=True, exist_ok=True)
        raylet.write_text("")

        first = JobRecord(
            job_id="dispatchpool01",
            command="python a.py",
            num_gpus=4,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=1.0,
            resource="gpu-a",
        )
        second = JobRecord(
            job_id="dispatchpool02",
            command="python b.py",
            num_gpus=4,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=2.0,
            resource="gpu-b",
        )
        records = [first, second]

        captured = {}

        class DummyThread:
            def __init__(self, target=None, daemon=None):
                captured["target"] = target

            def start(self):
                return None

        load_active_calls = {"count": 0}

        def fake_load_active():
            load_active_calls["count"] += 1
            if load_active_calls["count"] in (3, 5):
                return records
            return []

        sleep_calls = {"count": 0}

        def fake_sleep(_secs):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                raise RuntimeError("stop loop")

        launch_calls = []

        def fake_launch(record, workdir=None, dispatch_resource=None, **kwargs):
            launch_calls.append(record.job_id)
            if record.job_id == "dispatchpool01":
                record.dispatch_resource = dispatch_resource
                return "launch_pending", record
            return "launched", record

        gpu_a = _make_node("poolaaaa1111", gpus=4, partition="gpu-a", name="gpu-a-node")
        gpu_b = _make_node("poolbbbb2222", gpus=4, partition="gpu-b", name="gpu-b-node")

        with patch("hive.dashboard.threading.Thread", DummyThread), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard.Dispatcher") as mock_dispatcher:
            mock_dispatcher.load_state.return_value = {
                "ray_head_address": "10.0.0.1:6379",
                "resources": [],
            }
            app = create_app(config)

        target = captured["target"]
        assert target is not None
        app.config["TESTING"] = True

        with patch("hive.dashboard.JobRecord.load_active", side_effect=fake_load_active), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard._ray_cache", {"data": None, "ts": 0.0}), \
             patch("hive.dashboard._safe_ray_query", return_value=_ray_data([HEAD, gpu_a, gpu_b], total_gpus=8, free_gpus=8)), \
             patch("hive.dashboard.launch_job_runner", side_effect=fake_launch), \
             patch("hive.dashboard.ray") as mock_ray, \
             patch("hive.dashboard.time.sleep", side_effect=fake_sleep):
            mock_ray.util.list_named_actors.return_value = []
            with pytest.raises(RuntimeError, match="stop loop"):
                target()

        assert launch_calls[:2] == ["dispatchpool01", "dispatchpool02"]

    def test_bound_flexible_pending_launch_only_reserves_chosen_pool(self, tmp_hive_dir, tmp_path):
        """A dashboard-bound multi-resource launch should only reserve its chosen pool."""
        from hive.dashboard import create_app
        from hive.config import HiveConfig, SlurmConfig, RayConfig, EnvConfig
        from hive.dispatcher import JobRecord

        config = HiveConfig(
            slurm=SlurmConfig(default_walltime="1:00:00", resources=[]),
            ray=RayConfig(head_port=6379),
            env=EnvConfig(),
        )

        raylet = tmp_path / "session_latest" / "sockets" / "raylet"
        raylet.parent.mkdir(parents=True, exist_ok=True)
        raylet.write_text("")

        first = JobRecord(
            job_id="dispatchpool03",
            command="python any.py",
            num_gpus=4,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=1.0,
            resource="gpu-a,gpu-b",
        )
        second = JobRecord(
            job_id="dispatchpool04",
            command="python b.py",
            num_gpus=4,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=2.0,
            resource="gpu-b",
        )
        records = [first, second]

        captured = {}

        class DummyThread:
            def __init__(self, target=None, daemon=None):
                captured["target"] = target

            def start(self):
                return None

        load_active_calls = {"count": 0}

        def fake_load_active():
            load_active_calls["count"] += 1
            if load_active_calls["count"] in (3, 5):
                return records
            return []

        sleep_calls = {"count": 0}

        def fake_sleep(_secs):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                raise RuntimeError("stop loop")

        launch_calls = []

        chosen_resources = []

        def fake_launch(record, workdir=None, dispatch_resource=None, **kwargs):
            launch_calls.append(record.job_id)
            if record.job_id == "dispatchpool03":
                chosen_resources.append(dispatch_resource)
                record.dispatch_resource = dispatch_resource
                return "launch_pending", record
            return "launched", record

        gpu_a = _make_node("poolcccc3333", gpus=4, partition="gpu-a", name="gpu-a-node")
        gpu_b = _make_node("pooldddd4444", gpus=4, partition="gpu-b", name="gpu-b-node")

        with patch("hive.dashboard.threading.Thread", DummyThread), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard.Dispatcher") as mock_dispatcher:
            mock_dispatcher.load_state.return_value = {
                "ray_head_address": "10.0.0.1:6379",
                "resources": [],
            }
            app = create_app(config)

        target = captured["target"]
        assert target is not None
        app.config["TESTING"] = True

        with patch("hive.dashboard.JobRecord.load_active", side_effect=fake_load_active), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard._ray_cache", {"data": None, "ts": 0.0}), \
             patch("hive.dashboard._safe_ray_query", return_value=_ray_data([HEAD, gpu_a, gpu_b], total_gpus=8, free_gpus=8)), \
             patch("hive.dashboard.launch_job_runner", side_effect=fake_launch), \
             patch("hive.dashboard.ray") as mock_ray, \
             patch("hive.dashboard.time.sleep", side_effect=fake_sleep):
            mock_ray.util.list_named_actors.return_value = []
            with pytest.raises(RuntimeError, match="stop loop"):
                target()

        assert chosen_resources
        assert chosen_resources[0] == "gpu-a"
        assert launch_calls[:2] == ["dispatchpool03", "dispatchpool04"]

    def test_flexible_launch_uses_node_feasible_pool(self, tmp_hive_dir, tmp_path):
        """A flexible launch should only bind to pools with a single-node fit."""
        from hive.dashboard import create_app
        from hive.config import HiveConfig, SlurmConfig, RayConfig, EnvConfig
        from hive.dispatcher import JobRecord

        config = HiveConfig(
            slurm=SlurmConfig(default_walltime="1:00:00", resources=[]),
            ray=RayConfig(head_port=6379),
            env=EnvConfig(),
        )

        raylet = tmp_path / "session_latest" / "sockets" / "raylet"
        raylet.parent.mkdir(parents=True, exist_ok=True)
        raylet.write_text("")

        first = JobRecord(
            job_id="dispatchpool07",
            command="python any.py",
            num_gpus=4,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=1.0,
            resource="gpu-a,gpu-b",
        )
        records = [first]

        captured = {}

        class DummyThread:
            def __init__(self, target=None, daemon=None):
                captured["target"] = target

            def start(self):
                return None

        load_active_calls = {"count": 0}

        def fake_load_active():
            load_active_calls["count"] += 1
            if load_active_calls["count"] in (3, 5):
                return records
            return []

        sleep_calls = {"count": 0}

        def fake_sleep(_secs):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                raise RuntimeError("stop loop")

        chosen_resources = []

        def fake_launch(record, workdir=None, dispatch_resource=None, **kwargs):
            chosen_resources.append(dispatch_resource)
            record.dispatch_resource = dispatch_resource
            return "launch_pending", record

        gpu_a_1 = _make_node("poolgggg7777", gpus=2, partition="gpu-a", name="gpu-a-node-1")
        gpu_a_2 = _make_node("poolhhhh8888", gpus=2, partition="gpu-a", name="gpu-a-node-2")
        gpu_b = _make_node("pooliiii9999", gpus=4, partition="gpu-b", name="gpu-b-node")

        with patch("hive.dashboard.threading.Thread", DummyThread), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard.Dispatcher") as mock_dispatcher:
            mock_dispatcher.load_state.return_value = {
                "ray_head_address": "10.0.0.1:6379",
                "resources": [],
            }
            app = create_app(config)

        target = captured["target"]
        assert target is not None
        app.config["TESTING"] = True

        with patch("hive.dashboard.JobRecord.load_active", side_effect=fake_load_active), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard._ray_cache", {"data": None, "ts": 0.0}), \
             patch("hive.dashboard._safe_ray_query", return_value=_ray_data([HEAD, gpu_a_1, gpu_a_2, gpu_b], total_gpus=8, free_gpus=8)), \
             patch("hive.dashboard.launch_job_runner", side_effect=fake_launch), \
             patch("hive.dashboard.ray") as mock_ray, \
             patch("hive.dashboard.time.sleep", side_effect=fake_sleep):
            mock_ray.util.list_named_actors.return_value = []
            with pytest.raises(RuntimeError, match="stop loop"):
                target()

        assert chosen_resources
        assert chosen_resources[0] == "gpu-b"

    def test_reservation_simulation_packs_larger_jobs_first(self, tmp_hive_dir, tmp_path):
        """Reservation simulation should not depend on active-record order."""
        from hive.dashboard import create_app
        from hive.config import HiveConfig, SlurmConfig, RayConfig, EnvConfig
        from hive.dispatcher import JobRecord

        config = HiveConfig(
            slurm=SlurmConfig(default_walltime="1:00:00", resources=[]),
            ray=RayConfig(head_port=6379),
            env=EnvConfig(),
        )

        raylet = tmp_path / "session_latest" / "sockets" / "raylet"
        raylet.parent.mkdir(parents=True, exist_ok=True)
        raylet.write_text("")

        first = JobRecord(
            job_id="dispatchpool08",
            command="python small.py",
            num_gpus=4,
            num_cpus=1,
            memory_mb=0,
            status="running",
            submitted_at=1.0,
            resource="gpu-a",
        )
        second = JobRecord(
            job_id="dispatchpool09",
            command="python large.py",
            num_gpus=8,
            num_cpus=1,
            memory_mb=0,
            status="running",
            submitted_at=2.0,
            resource="gpu-a",
        )
        third = JobRecord(
            job_id="dispatchpool10",
            command="python queued.py",
            num_gpus=4,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=3.0,
            resource="gpu-a",
        )
        records = [first, second, third]

        captured = {}

        class DummyThread:
            def __init__(self, target=None, daemon=None):
                captured["target"] = target

            def start(self):
                return None

        load_active_calls = {"count": 0}

        def fake_load_active():
            load_active_calls["count"] += 1
            if load_active_calls["count"] in (3, 5):
                return records
            return []

        sleep_calls = {"count": 0}

        def fake_sleep(_secs):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                raise RuntimeError("stop loop")

        launch_calls = []

        def fake_launch(record, workdir=None, dispatch_resource=None, **kwargs):
            launch_calls.append(record.job_id)
            return "launched", record

        gpu_a_1 = _make_node("pooljjjj0000", gpus=8, partition="gpu-a", name="gpu-a-node-1")
        gpu_a_2 = _make_node("poolkkkk1111", gpus=4, partition="gpu-a", name="gpu-a-node-2")
        gpu_a_3 = _make_node("poolllll2222", gpus=4, partition="gpu-a", name="gpu-a-node-3")

        with patch("hive.dashboard.threading.Thread", DummyThread), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard.Dispatcher") as mock_dispatcher:
            mock_dispatcher.load_state.return_value = {
                "ray_head_address": "10.0.0.1:6379",
                "resources": [],
            }
            app = create_app(config)

        target = captured["target"]
        assert target is not None
        app.config["TESTING"] = True

        with patch("hive.dashboard.JobRecord.load_active", side_effect=fake_load_active), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard._ray_cache", {"data": None, "ts": 0.0}), \
             patch("hive.dashboard._safe_ray_query", return_value=_ray_data([HEAD, gpu_a_1, gpu_a_2, gpu_a_3], total_gpus=16, free_gpus=16)), \
             patch("hive.dashboard.launch_job_runner", side_effect=fake_launch), \
             patch("hive.dashboard.ray") as mock_ray, \
             patch("hive.dashboard.time.sleep", side_effect=fake_sleep):
            mock_ray.util.list_named_actors.return_value = []
            with pytest.raises(RuntimeError, match="stop loop"):
                target()

        assert launch_calls
        assert launch_calls[0] == "dispatchpool10"

    def test_reservation_simulation_finds_non_greedy_fit(self, tmp_hive_dir, tmp_path):
        """Reservation simulation should accept feasible heterogeneous packings."""
        from hive.dashboard import create_app
        from hive.config import HiveConfig, SlurmConfig, RayConfig, EnvConfig
        from hive.dispatcher import JobRecord

        config = HiveConfig(
            slurm=SlurmConfig(default_walltime="1:00:00", resources=[]),
            ray=RayConfig(head_port=6379),
            env=EnvConfig(),
        )

        raylet = tmp_path / "session_latest" / "sockets" / "raylet"
        raylet.parent.mkdir(parents=True, exist_ok=True)
        raylet.write_text("")

        first = JobRecord(
            job_id="dispatchpool11",
            command="python four.py",
            num_gpus=4,
            num_cpus=1,
            memory_mb=0,
            status="running",
            submitted_at=1.0,
            resource="gpu-a",
        )
        second = JobRecord(
            job_id="dispatchpool12",
            command="python six.py",
            num_gpus=6,
            num_cpus=1,
            memory_mb=0,
            status="running",
            submitted_at=2.0,
            resource="gpu-a",
        )
        third = JobRecord(
            job_id="dispatchpool13",
            command="python queued.py",
            num_gpus=4,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=3.0,
            resource="gpu-a",
        )
        records = [first, second, third]

        captured = {}

        class DummyThread:
            def __init__(self, target=None, daemon=None):
                captured["target"] = target

            def start(self):
                return None

        load_active_calls = {"count": 0}

        def fake_load_active():
            load_active_calls["count"] += 1
            if load_active_calls["count"] in (3, 5):
                return records
            return []

        sleep_calls = {"count": 0}

        def fake_sleep(_secs):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                raise RuntimeError("stop loop")

        launch_calls = []

        def fake_launch(record, workdir=None, dispatch_resource=None, **kwargs):
            launch_calls.append(record.job_id)
            return "launched", record

        gpu_a_1 = _make_node("poolpppp6666", gpus=8, partition="gpu-a", name="gpu-a-node-1")
        gpu_a_2 = _make_node("poolqqqq7777", gpus=6, partition="gpu-a", name="gpu-a-node-2")

        with patch("hive.dashboard.threading.Thread", DummyThread), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard.Dispatcher") as mock_dispatcher:
            mock_dispatcher.load_state.return_value = {
                "ray_head_address": "10.0.0.1:6379",
                "resources": [],
            }
            app = create_app(config)

        target = captured["target"]
        assert target is not None
        app.config["TESTING"] = True

        with patch("hive.dashboard.JobRecord.load_active", side_effect=fake_load_active), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard._ray_cache", {"data": None, "ts": 0.0}), \
             patch("hive.dashboard._safe_ray_query", return_value=_ray_data([HEAD, gpu_a_1, gpu_a_2], total_gpus=14, free_gpus=14)), \
             patch("hive.dashboard.launch_job_runner", side_effect=fake_launch), \
             patch("hive.dashboard.ray") as mock_ray, \
             patch("hive.dashboard.time.sleep", side_effect=fake_sleep):
            mock_ray.util.list_named_actors.return_value = []
            with pytest.raises(RuntimeError, match="stop loop"):
                target()

        assert launch_calls
        assert launch_calls[0] == "dispatchpool13"

    def test_flexible_launch_preserves_later_node_feasible_pool(self, tmp_hive_dir, tmp_path):
        """A flexible launch should avoid taking the only node-feasible pool for later work."""
        from hive.dashboard import create_app
        from hive.config import HiveConfig, SlurmConfig, RayConfig, EnvConfig
        from hive.dispatcher import JobRecord

        config = HiveConfig(
            slurm=SlurmConfig(default_walltime="1:00:00", resources=[]),
            ray=RayConfig(head_port=6379),
            env=EnvConfig(),
        )

        raylet = tmp_path / "session_latest" / "sockets" / "raylet"
        raylet.parent.mkdir(parents=True, exist_ok=True)
        raylet.write_text("")

        first = JobRecord(
            job_id="dispatchpool08",
            command="python any.py",
            num_gpus=4,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=1.0,
            resource="gpu-a,gpu-b",
        )
        second = JobRecord(
            job_id="dispatchpool09",
            command="python big.py",
            num_gpus=8,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=2.0,
            resource="gpu-a,gpu-b",
        )
        records = [first, second]

        captured = {}

        class DummyThread:
            def __init__(self, target=None, daemon=None):
                captured["target"] = target

            def start(self):
                return None

        load_active_calls = {"count": 0}

        def fake_load_active():
            load_active_calls["count"] += 1
            if load_active_calls["count"] in (3, 5):
                return records
            return []

        sleep_calls = {"count": 0}

        def fake_sleep(_secs):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                raise RuntimeError("stop loop")

        launch_calls = []
        chosen_resources = []

        def fake_launch(record, workdir=None, dispatch_resource=None, **kwargs):
            launch_calls.append(record.job_id)
            if record.job_id == "dispatchpool08":
                chosen_resources.append(dispatch_resource)
                record.dispatch_resource = dispatch_resource
                return "launch_pending", record
            return "launched", record

        gpu_a = _make_node("poolmmmm3333", gpus=8, partition="gpu-a", name="gpu-a-node")
        gpu_b_1 = _make_node("poolnnnn4444", gpus=4, partition="gpu-b", name="gpu-b-node-1")
        gpu_b_2 = _make_node("pooloooo5555", gpus=4, partition="gpu-b", name="gpu-b-node-2")

        with patch("hive.dashboard.threading.Thread", DummyThread), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard.Dispatcher") as mock_dispatcher:
            mock_dispatcher.load_state.return_value = {
                "ray_head_address": "10.0.0.1:6379",
                "resources": [],
            }
            app = create_app(config)

        target = captured["target"]
        assert target is not None
        app.config["TESTING"] = True

        with patch("hive.dashboard.JobRecord.load_active", side_effect=fake_load_active), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard._ray_cache", {"data": None, "ts": 0.0}), \
             patch("hive.dashboard._safe_ray_query", return_value=_ray_data([HEAD, gpu_a, gpu_b_1, gpu_b_2], total_gpus=16, free_gpus=16)), \
             patch("hive.dashboard.launch_job_runner", side_effect=fake_launch), \
             patch("hive.dashboard.ray") as mock_ray, \
             patch("hive.dashboard.time.sleep", side_effect=fake_sleep):
            mock_ray.util.list_named_actors.return_value = []
            with pytest.raises(RuntimeError, match="stop loop"):
                target()

        assert chosen_resources
        assert chosen_resources[0] == "gpu-b"
        assert launch_calls[:2] == ["dispatchpool08", "dispatchpool09"]

    def test_unbound_flexible_pending_launch_blocks_every_compatible_pool(self, tmp_hive_dir, tmp_path):
        """A still-unbound flexible pending launch must remain conservative."""
        from hive.dashboard import create_app
        from hive.config import HiveConfig, SlurmConfig, RayConfig, EnvConfig
        from hive.dispatcher import JobRecord

        config = HiveConfig(
            slurm=SlurmConfig(default_walltime="1:00:00", resources=[]),
            ray=RayConfig(head_port=6379),
            env=EnvConfig(),
        )

        raylet = tmp_path / "session_latest" / "sockets" / "raylet"
        raylet.parent.mkdir(parents=True, exist_ok=True)
        raylet.write_text("")

        first = JobRecord(
            job_id="dispatchpool05",
            command="python any.py",
            num_gpus=4,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=1.0,
            resource="gpu-a,gpu-b",
            runner_launch_pending_since=time.time(),
        )
        second = JobRecord(
            job_id="dispatchpool06",
            command="python b.py",
            num_gpus=4,
            num_cpus=1,
            memory_mb=0,
            status="queued",
            submitted_at=2.0,
            resource="gpu-b",
        )
        records = [first, second]

        captured = {}

        class DummyThread:
            def __init__(self, target=None, daemon=None):
                captured["target"] = target

            def start(self):
                return None

        load_active_calls = {"count": 0}

        def fake_load_active():
            load_active_calls["count"] += 1
            if load_active_calls["count"] in (3, 5):
                return records
            return []

        sleep_calls = {"count": 0}

        def fake_sleep(_secs):
            sleep_calls["count"] += 1
            if sleep_calls["count"] >= 2:
                raise RuntimeError("stop loop")

        launch_calls = []

        def fake_launch(record, workdir=None, dispatch_resource=None, **kwargs):
            launch_calls.append(record.job_id)
            if record.job_id == "dispatchpool05":
                return "launch_pending", record
            return "launched", record

        gpu_a = _make_node("pooleeee5555", gpus=4, partition="gpu-a", name="gpu-a-node")
        gpu_b = _make_node("poolffff6666", gpus=4, partition="gpu-b", name="gpu-b-node")

        with patch("hive.dashboard.threading.Thread", DummyThread), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard.Dispatcher") as mock_dispatcher:
            mock_dispatcher.load_state.return_value = {
                "ray_head_address": "10.0.0.1:6379",
                "resources": [],
            }
            app = create_app(config)

        target = captured["target"]
        assert target is not None
        app.config["TESTING"] = True

        with patch("hive.dashboard.JobRecord.load_active", side_effect=fake_load_active), \
             patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard._ray_cache", {"data": None, "ts": 0.0}), \
             patch("hive.dashboard._safe_ray_query", return_value=_ray_data([HEAD, gpu_a, gpu_b], total_gpus=8, free_gpus=8)), \
             patch("hive.dashboard.launch_job_runner", side_effect=fake_launch), \
             patch("hive.dashboard.ray") as mock_ray, \
             patch("hive.dashboard.time.sleep", side_effect=fake_sleep):
            mock_ray.util.list_named_actors.return_value = []
            with pytest.raises(RuntimeError, match="stop loop"):
                target()

        assert launch_calls
        assert set(launch_calls) == {"dispatchpool05"}


class TestApiCancel:
    def test_running_job_not_cancelled_when_ray_unreachable(self, app, tmp_hive_dir, tmp_path):
        """Cancel must fail closed if Ray cannot be reached for a running job."""
        from hive.dispatcher import JobRecord

        record = JobRecord(
            job_id="run123",
            command="python train.py",
            num_gpus=1,
            status="running",
            submitted_at=1.0,
            started_at=2.0,
        )
        record.save()

        with patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)):
            with app.test_client() as c:
                resp = c.post("/api/jobs/run123/cancel")
                data = resp.get_json()

        assert resp.status_code == 503
        assert "Cannot reach Ray" in data["error"]
        assert JobRecord.load("run123").status == "running"

    def test_queued_job_not_cancelled_when_ray_unreachable(self, app, tmp_hive_dir, tmp_path):
        """Queued jobs must fail closed when Ray cannot confirm actor state."""
        from hive.dispatcher import JobRecord

        record = JobRecord(
            job_id="queue123",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=1.0,
            priority="normal",
            runner_launched=True,
        )
        record.save()

        with patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)):
            with app.test_client() as c:
                resp = c.post("/api/jobs/queue123/cancel")
                data = resp.get_json()

        assert resp.status_code == 503
        assert "Cannot reach Ray" in data["error"]
        reloaded = JobRecord.load("queue123")
        assert reloaded.status == "queued"
        assert reloaded.returncode is None

    def test_queued_job_fails_closed_if_atomic_check_detects_started_runner(self, app, tmp_path):
        """A stale queued snapshot must not bypass Ray once the runner starts."""
        initial = MagicMock()
        initial.job_id = "queue123b"
        initial.status = "queued"

        activated = MagicMock()
        activated.job_id = "queue123b"
        activated.status = "queued"
        activated.queued_job_needs_ray_verification.return_value = True

        with patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard.JobRecord.load", side_effect=[initial, activated]), \
             patch("hive.dashboard.JobRecord.cancel_queued_if_not_started", return_value=("needs_ray", activated)):
            with app.test_client() as c:
                resp = c.post("/api/jobs/queue123b/cancel")
                data = resp.get_json()

        assert resp.status_code == 503
        assert "Cannot reach Ray" in data["error"]

    def test_low_priority_queued_job_can_cancel_when_ray_unreachable(self, app, tmp_hive_dir, tmp_path):
        """Low-priority queued jobs with no runner may cancel offline."""
        from hive.dispatcher import JobRecord

        record = JobRecord(
            job_id="queue124",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=1.0,
            priority="low",
            runner_launched=False,
        )
        record.save()

        with patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)):
            with app.test_client() as c:
                resp = c.post("/api/jobs/queue124/cancel")
                data = resp.get_json()

        assert resp.status_code == 200
        assert data["status"] == "cancelled"
        assert data["actor_killed"] is False
        reloaded = JobRecord.load("queue124")
        assert reloaded.status == "cancelled"
        assert reloaded.returncode == -15
        assert reloaded.combined_stderr.endswith("[hive] Job cancelled")

    def test_normal_queued_job_with_no_runner_can_cancel_when_ray_unreachable(self, app, tmp_hive_dir, tmp_path):
        """Queued jobs with runner_launched=False may cancel offline."""
        from hive.dispatcher import JobRecord

        record = JobRecord(
            job_id="queue124b",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=1.0,
            priority="normal",
            runner_launched=False,
        )
        record.save()

        with patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)):
            with app.test_client() as c:
                resp = c.post("/api/jobs/queue124b/cancel")
                data = resp.get_json()

        assert resp.status_code == 200
        assert data["status"] == "cancelled"
        assert data["actor_killed"] is False
        reloaded = JobRecord.load("queue124b")
        assert reloaded.status == "cancelled"
        assert reloaded.returncode == -15
        assert reloaded.combined_stderr.endswith("[hive] Job cancelled")

    def test_pending_launch_can_cancel_when_ray_unreachable(self, app, tmp_hive_dir, tmp_path):
        """A pending launch can cancel offline before the actor starts."""
        from hive.dispatcher import JobRecord

        record = JobRecord(
            job_id="queue124c",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=1.0,
            priority="normal",
            runner_launched=False,
            runner_launch_pending_since=10_000.0,
        )
        record.save()

        with patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)):
            with app.test_client() as c:
                resp = c.post("/api/jobs/queue124c/cancel")
                data = resp.get_json()

        assert resp.status_code == 200
        assert data["status"] == "cancelled"
        assert data["actor_killed"] is False
        reloaded = JobRecord.load("queue124c")
        assert reloaded.status == "cancelled"
        assert reloaded.runner_launch_pending_since is None
        assert reloaded.combined_stderr.endswith("[hive] Job cancelled")

    def test_stale_pending_launch_can_cancel_when_ray_unreachable(self, app, tmp_hive_dir, tmp_path):
        """A stale pending launch can also cancel offline."""
        from hive.dispatcher import JobRecord

        record = JobRecord(
            job_id="queue124d",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=1.0,
            priority="normal",
            runner_launched=False,
            runner_launch_pending_since=1.0,
        )
        record.save()

        with patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)):
            with app.test_client() as c:
                resp = c.post("/api/jobs/queue124d/cancel")
                data = resp.get_json()

        assert resp.status_code == 200
        assert data["status"] == "cancelled"
        assert data["actor_killed"] is False
        reloaded = JobRecord.load("queue124d")
        assert reloaded.status == "cancelled"
        assert reloaded.runner_launch_pending_since is None
        assert reloaded.combined_stderr.endswith("[hive] Job cancelled")

    def test_queued_job_can_cancel_when_actor_disappears_mid_lookup(self, app, tmp_hive_dir, tmp_path):
        """A queued cancel should succeed if the actor vanishes during lookup."""
        from hive.dispatcher import JobRecord

        raylet = tmp_path / "session_latest" / "sockets" / "raylet"
        raylet.parent.mkdir(parents=True, exist_ok=True)
        raylet.write_text("")

        record = JobRecord(
            job_id="queue125",
            command="python train.py",
            num_gpus=1,
            status="queued",
            submitted_at=1.0,
            priority="normal",
            runner_launched=True,
        )
        record.save()

        with patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)), \
             patch("hive.dashboard.ray.util.list_named_actors", return_value=[{"name": "hive_job_queue125"}]), \
             patch("hive.dashboard.ray.get_actor", side_effect=ValueError("gone")), \
             patch("hive.dashboard.ray.kill") as mock_kill:
            with app.test_client() as c:
                resp = c.post("/api/jobs/queue125/cancel")
                data = resp.get_json()

        assert resp.status_code == 200
        assert data["status"] == "cancelled"
        assert data["actor_killed"] is False
        mock_kill.assert_not_called()
        reloaded = JobRecord.load("queue125")
        assert reloaded.status == "cancelled"
        assert reloaded.returncode == -15
        assert reloaded.combined_stderr.endswith("[hive] Job cancelled")


# ---------------------------------------------------------------
# /api/jobs/<id>/logs
# ---------------------------------------------------------------

class TestApiJobLogs:
    def test_logs_endpoint_returns_stable_stderr_stream(self, app, tmp_hive_dir, tmp_path):
        """Dashboard log reads should return the real stderr stream as written."""
        from hive.dispatcher import JobRecord

        record = JobRecord(
            job_id="logs123",
            command="python train.py",
            num_gpus=1,
            status="cancelled",
            submitted_at=1.0,
            completed_at=2.0,
            stdout="stdout body\n",
            stderr="stderr body\n[hive] Job cancelled\n",
            returncode=-15,
        )
        record.save()

        with patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)):
            with app.test_client() as c:
                resp = c.get("/api/jobs/logs123/logs")
                data = resp.get_json()

        assert resp.status_code == 200
        assert data["stdout"] == "stdout body\n"
        assert data["stderr"] == "stderr body\n[hive] Job cancelled\n"
        assert data["status"] == "cancelled"

    def test_logs_endpoint_shows_buffered_cancel_note_for_cancelled_job(self, app, tmp_hive_dir, tmp_path):
        """Cancelled jobs should surface buffered notes before finalize flushes them."""
        from hive.dispatcher import JobRecord

        record = JobRecord(
            job_id="logs124",
            command="python train.py",
            num_gpus=1,
            status="cancelled",
            submitted_at=1.0,
            completed_at=2.0,
            stdout="stdout body\n",
            stderr="stderr body\n",
            returncode=-15,
        )
        record.save()
        (record.job_dir / "events.log").write_text("[hive] Job cancelled\n")

        with patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)):
            with app.test_client() as c:
                resp = c.get("/api/jobs/logs124/logs")
                data = resp.get_json()

        assert resp.status_code == 200
        assert data["stderr"] == "stderr body\n[hive] Job cancelled\n"
        assert data["status"] == "cancelled"

    def test_logs_endpoint_shows_buffered_preempt_note_while_preempted(self, app, tmp_hive_dir, tmp_path):
        """Preempted jobs should surface buffered notes before they are re-queued."""
        from hive.dispatcher import JobRecord

        record = JobRecord(
            job_id="logs125",
            command="python train.py",
            num_gpus=1,
            status="preempted",
            submitted_at=1.0,
            stdout="stdout body\n",
            stderr="stderr body\n",
        )
        record.save()
        (record.job_dir / "events.log").write_text("[hive] Preempted for normal-priority job\n")

        with patch("hive.dashboard.get_ray_tmpdir", return_value=str(tmp_path)):
            with app.test_client() as c:
                resp = c.get("/api/jobs/logs125/logs")
                data = resp.get_json()

        assert resp.status_code == 200
        assert data["stderr"] == "stderr body\n[hive] Preempted for normal-priority job\n"
        assert data["status"] == "preempted"


# ---------------------------------------------------------------
# /api/cpu_stats
# ---------------------------------------------------------------

class TestApiCpuStats:
    def test_cpu_stats_returns_json(self, app):
        """cpu_stats endpoint should return JSON from subprocess."""
        fake_output = json.dumps({
            "cpu-node-1": {"cpu_util": 45, "mem_used_gb": 12.3, "mem_total_gb": 64.0}
        })
        with patch("hive.dashboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=fake_output, stderr=""
            )
            with app.test_client() as c:
                resp = c.get("/api/cpu_stats")
                data = resp.get_json()

        assert "cpu-node-1" in data
        assert data["cpu-node-1"]["cpu_util"] == 45
        assert data["cpu-node-1"]["mem_total_gb"] == 64.0

    def test_cpu_stats_empty_on_no_cpu_nodes(self, app):
        """No CPU nodes → empty dict."""
        with patch("hive.dashboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="{}", stderr=""
            )
            with app.test_client() as c:
                resp = c.get("/api/cpu_stats")
                data = resp.get_json()

        assert data == {}

    def test_cpu_stats_subprocess_failure(self, app):
        """Subprocess failure → empty dict."""
        with patch("hive.dashboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="error"
            )
            with app.test_client() as c:
                resp = c.get("/api/cpu_stats")
                data = resp.get_json()

        assert data == {}

    def test_cpu_stats_timeout(self, app):
        """Subprocess timeout → error response."""
        import subprocess
        with patch("hive.dashboard.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
            with app.test_client() as c:
                resp = c.get("/api/cpu_stats")
                data = resp.get_json()

        assert "error" in data


# ---------------------------------------------------------------
# /api/nodes — CPU node details
# ---------------------------------------------------------------

class TestApiNodesCpuDetails:
    def test_cpu_node_resource_shown(self, app):
        """CPU node should show its resource."""
        with patch("hive.dashboard._safe_ray_query", return_value=_ray_data([HEAD, CPU1])):
            with app.test_client() as c:
                resp = c.get("/api/nodes")
                data = resp.get_json()

        assert len(data) == 1
        assert data[0]["resource"] == "cpu-pool"
        assert data[0]["gpus"] == 0
        assert data[0]["alive"] is True

    def test_mixed_gpu_cpu_nodes_all_present(self, app):
        """Both GPU and CPU worker nodes should be in the response."""
        cpu2 = _make_node("cpu5555ffff", gpus=0, cpus=32, partition="batch", name="cpu-node-2")
        nodes = [HEAD, GPU1, GPU2, CPU1, cpu2]
        with patch("hive.dashboard._safe_ray_query", return_value=_ray_data(nodes)):
            with app.test_client() as c:
                resp = c.get("/api/nodes")
                data = resp.get_json()

        assert len(data) == 4  # 2 GPU + 2 CPU
        names = {n["name"] for n in data}
        assert names == {"gpu-node-1", "gpu-node-2", "cpu-node-1", "cpu-node-2"}
