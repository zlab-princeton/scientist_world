"""Hive CLI — agent-facing interface."""

import argparse
import copy
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

# Suppress Ray FutureWarning about accelerator env var override with num_gpus=0
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import ray

from .config import (
    HiveConfig,
    ResourceSpec,
    STATE_PATH,
    HIVE_DIR,
    SLURM_TRACKING_DIR,
    load_config,
    ensure_hive_dir,
    load_runtime_ray_head,
)
from .dispatcher import (
    Dispatcher,
    JobRecord,
    JOB_ACTOR_PREFIX,
)

# Path for remote connection config (user-level, not per-project)
REMOTE_CONFIG_PATH = HIVE_DIR / "remote.json"
log = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _ray_head_address() -> str:
    """Return the Ray GCS address for THIS hive cluster from state.json.

    Never uses 'auto' — on shared nodes multiple users may have Ray clusters
    running, and 'auto' discovers via /tmp/ray/session_latest which can point
    to a different user's cluster (possibly a different Ray version).
    """
    return load_runtime_ray_head(STATE_PATH) or "auto"


def _get_remote_url() -> str | None:
    """Return the dashboard URL if connected to a remote Hive cluster."""
    if REMOTE_CONFIG_PATH.exists():
        try:
            data = json.loads(REMOTE_CONFIG_PATH.read_text())
            return data.get("dashboard_url")
        except Exception:
            pass
    return None


def _auto_disconnect():
    """Remove remote config and print a message when the dashboard is unreachable."""
    if REMOTE_CONFIG_PATH.exists():
        REMOTE_CONFIG_PATH.unlink()
        print("Dashboard unreachable. Cluster may have stopped. Disconnected.")


def _remote_get(url: str) -> dict:
    """GET a JSON endpoint on the remote dashboard."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except (ConnectionError, urllib.error.URLError, OSError) as e:
        if isinstance(e, urllib.error.HTTPError):
            raise  # HTTP errors (4xx/5xx) are not connection failures
        _auto_disconnect()
        raise


def _remote_post(url: str, data: dict) -> dict:
    """POST JSON to a remote dashboard endpoint."""
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except (ConnectionError, urllib.error.URLError, OSError) as e:
        if isinstance(e, urllib.error.HTTPError):
            raise  # HTTP errors (4xx/5xx) are not connection failures
        _auto_disconnect()
        raise


def _tracked_slurm_jobs_from_state(state: dict | None) -> dict[str, str]:
    """Extract tracked SLURM jobs as {job_id: resource_name} from saved state."""
    if not state:
        return {}
    slurm_jobs = state.get("slurm_jobs", {})
    if isinstance(slurm_jobs, list):
        return {str(job_id): "" for job_id in slurm_jobs}
    if isinstance(slurm_jobs, dict):
        return {
            str(job_id): (meta or {}).get("resource_name", "")
            for job_id, meta in slurm_jobs.items()
        }
    return {}


def _tracked_slurm_jobs_from_disk() -> dict[str, str]:
    """Read tracked SLURM job ids from slurm_tracking/ if present."""
    jobs: dict[str, str] = {}
    if not SLURM_TRACKING_DIR.is_dir():
        return jobs
    for entry in SLURM_TRACKING_DIR.iterdir():
        if not entry.is_file():
            continue
        try:
            jobs[entry.name] = entry.read_text().strip()
        except OSError:
            continue
    return jobs


def _filter_live_slurm_jobs(tracked_jobs: dict[str, str]) -> dict[str, str]:
    """Keep only SLURM jobs that still exist according to squeue.

    If ``squeue`` is unavailable or fails, return the original tracked set to
    avoid accidentally dropping healthy allocations from the preserved state.
    """
    if not tracked_jobs:
        return {}
    job_ids = sorted(tracked_jobs)
    try:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%A", "-j", ",".join(job_ids)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return dict(tracked_jobs)
    if result.returncode != 0:
        return dict(tracked_jobs)
    live_ids = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return {job_id: tracked_jobs[job_id] for job_id in job_ids if job_id in live_ids}


def _resource_dict_map(config: HiveConfig, state: dict | None) -> dict[str, dict]:
    """Build a resource-name → resource-dict map from state first, then config."""
    resources: dict[str, dict] = {}
    if state:
        for entry in state.get("resources", []) or []:
            if isinstance(entry, dict) and entry.get("name"):
                resources[entry["name"]] = copy.deepcopy(entry)
    for spec in config.slurm.resources:
        resources.setdefault(spec.name, spec.to_dict())
    return resources


def _load_relaunch_state() -> dict:
    """Best-effort state load for relaunch recovery.

    Relaunch must survive truncated or partially-written ``state.json`` and
    fall back to tracked SLURM jobs on disk.
    """
    try:
        return Dispatcher.load_state() or {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        log.warning("Ignoring unreadable Hive state during relaunch recovery: %s", exc)
        return {}


def _build_preserved_relaunch_state(config: HiveConfig) -> dict | None:
    """Reconstruct a safe local state snapshot for a clean relaunch.

    This preserves currently tracked SLURM worker jobs by rebuilding the minimum
    state needed for workers to reconnect after the local Ray head restarts.
    """
    state = _load_relaunch_state()
    tracked_jobs = _tracked_slurm_jobs_from_state(state)
    tracked_jobs.update(_tracked_slurm_jobs_from_disk())
    live_jobs = _filter_live_slurm_jobs(tracked_jobs)

    if not state and not live_jobs:
        return None

    resource_map = _resource_dict_map(config, state)
    resources: list[dict] = []
    if state.get("resources"):
        seen = set()
        for entry in state.get("resources", []):
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name or name in seen:
                continue
            resources.append(copy.deepcopy(entry))
            seen.add(name)
    else:
        seen = set()
        for resource_name in live_jobs.values():
            if not resource_name or resource_name in seen:
                continue
            entry = resource_map.get(resource_name)
            if entry is not None:
                resources.append(copy.deepcopy(entry))
                seen.add(resource_name)

    walltime = state.get("walltime", config.slurm.default_walltime)
    max_gpus = state.get("max_gpus_per_node")
    if max_gpus is None:
        max_gpus = max(
            (entry.get("gpus_per_job", 0) for entry in resources if isinstance(entry, dict)),
            default=0,
        )

    dispatcher = Dispatcher(config)
    prior_head = state.get("ray_head_address")
    if not prior_head:
        head_ip = config.ray.head_ip or dispatcher._detect_cluster_ip()
        prior_head = f"{head_ip}:{config.ray.head_port}"

    return {
        "ray_head_address": prior_head,
        "resources": resources,
        "slurm_jobs": live_jobs,
        "walltime": walltime,
        "max_gpus_per_node": max_gpus,
    }


def _rebuild_preserved_sbatch_templates(
    dispatcher: Dispatcher,
    resources: list[dict],
    default_walltime: str,
) -> None:
    """Regenerate persistent sbatch templates for preserved worker resources."""
    sbatch_dir = HIVE_DIR / "sbatch"
    sbatch_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    for entry in resources:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        pspec = ResourceSpec.from_dict(entry)
        walltime = pspec.walltime or default_walltime
        template = dispatcher.pool._build_sbatch_script(pspec=pspec, walltime=walltime)
        tpath = sbatch_dir / f"{pspec.name}.sbatch"
        tpath.write_text(template)
        tpath.chmod(0o755)
        if tpath not in dispatcher.pool._sbatch_template_paths:
            dispatcher.pool._sbatch_template_paths.append(tpath)


def _cancel_job_local(job_id: str) -> dict:
    """Cancel a queued or running local job.

    Running jobs are only marked ``cancelled`` once the detached runner was
    verifiably stopped. Queued jobs only require a live Ray query when Hive
    has already launched a detached runner for them.
    """
    record = JobRecord.load(job_id)
    if record is None:
        return {"error": f"Job '{job_id}' not found"}

    if record.status not in ("running", "queued"):
        return {
            "job_id": job_id,
            "status": record.status,
            "message": f"Job already {record.status}",
        }

    if record.status == "queued":
        queued_result, queued_record = JobRecord.cancel_queued_if_not_started(job_id)
        if queued_result == "cancelled":
            return {
                "job_id": job_id,
                "status": "cancelled",
                "actor_killed": False,
            }
        if queued_result == "terminal" and queued_record is not None:
            return {
                "job_id": job_id,
                "status": queued_record.status,
                "message": f"Job already {queued_record.status}",
            }
        record = JobRecord.load(job_id) or queued_record or record

    needs_ray_verification = record.status == "running" or record.queued_job_needs_ray_verification()

    killed = False
    ray_connected = False
    try:
        ray.init(address=_ray_head_address(), namespace="hive", ignore_reinit_error=True)
        ray_connected = True
    except Exception as e:
        if needs_ray_verification:
            return {"error": f"Cannot reach Ray to verify job cancellation: {e}"}

    try:
        if ray_connected and needs_ray_verification:
            prefix = f"{JOB_ACTOR_PREFIX}{record.job_id}"
            try:
                for entry in ray.util.list_named_actors(all_namespaces=True):
                    name = entry if isinstance(entry, str) else entry.get("name", "")
                    if name.startswith(prefix):
                        try:
                            actor = ray.get_actor(name)
                        except ValueError:
                            continue
                        ray.kill(actor)
                        killed = True
            except Exception as e:
                return {"error": f"Cannot inspect Ray actors for job '{job_id}': {e}"}

        fresh = JobRecord.load(job_id)
        if fresh and fresh.status in ("completed", "failed", "cancelled", "timeout", "preempted"):
            return {
                "job_id": job_id,
                "status": fresh.status,
                "message": f"Job finished as {fresh.status} before cancel took effect",
            }

        if record.status == "running" and not killed:
            return {
                "error": (
                    f"Job '{job_id}' is still marked running, but its Ray actor "
                    "could not be stopped. Refusing to mark it cancelled."
                )
            }

        cancelled_result, cancelled_record = JobRecord.mark_cancelled(job_id)
        if cancelled_result == "terminal" and cancelled_record is not None:
            return {
                "job_id": job_id,
                "status": cancelled_record.status,
                "message": f"Job finished as {cancelled_record.status} before cancel took effect",
            }
        if cancelled_result == "missing":
            return {"error": f"Job '{job_id}' disappeared during cancellation"}
        return {
            "job_id": job_id,
            "status": "cancelled",
            "actor_killed": killed,
        }
    finally:
        if ray_connected:
            ray.shutdown()


def _cancel_job_remote(job_id: str, remote_url: str) -> dict:
    """Cancel a job via the remote dashboard API."""
    try:
        return _remote_post(f"{remote_url}/api/jobs/{job_id}/cancel", {})
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            err = json.loads(body).get("error", body)
        except Exception:
            err = body
        return {"error": err or str(e), "job_id": job_id}
    except Exception as e:
        return {"error": str(e), "job_id": job_id}


def cmd_connect(args):
    """Connect to a remote Hive cluster's dashboard."""
    target = args.target
    if target is None:
        config = load_config()
        target = f"localhost:{config.ray.dashboard_port}"
    if "://" not in target:
        target = f"http://{target}"

    # Ping /api/status to verify
    try:
        status = _remote_get(f"{target}/api/status")
    except Exception as e:
        print(f"Cannot reach dashboard at {target}: {e}")
        return 1

    REMOTE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    REMOTE_CONFIG_PATH.write_text(json.dumps({"dashboard_url": target}, indent=2))

    gpus = status.get("total_gpus", "?")
    nodes = status.get("num_gpu_nodes", 0) + status.get("num_cpu_nodes", 0)
    print(f"Connected to Hive cluster at {target}")
    print(f"  {gpus} GPUs across {nodes} nodes")
    print("Use 'hive submit', 'hive jobs', 'hive queue', 'hive logs' as usual.")
    return 0


def cmd_disconnect(args):
    """Disconnect from a remote Hive cluster."""
    if REMOTE_CONFIG_PATH.exists():
        REMOTE_CONFIG_PATH.unlink()
        print("Disconnected from remote Hive cluster.")
    else:
        print("Not connected to any remote cluster.")
    return 0


def cmd_cancel(args):
    """Cancel one or more queued/running jobs."""
    remote_url = _get_remote_url()
    had_error = False

    for job_id in args.job_ids:
        result = _cancel_job_remote(job_id, remote_url) if remote_url else _cancel_job_local(job_id)

        if "error" in result:
            print(f"{job_id}: error: {result['error']}")
            had_error = True
            continue

        status = result.get("status", "unknown")
        message = result.get("message")
        if status == "cancelled":
            print(f"{job_id}: cancelled")
        elif message:
            print(f"{job_id}: {message}")
        else:
            print(f"{job_id}: {status}")

    return 1 if had_error else 0


def _resolve_resources(args, config: HiveConfig) -> list:
    """Build the list of ResourceSpecs to launch.

    Modes:
      1. -r gpu-a 3 -r gpu-b 2    (normal: pick resources by name, set num_jobs)
      2. --full                  (all non-test resources at max_jobs)
      3. --full --include-test   (all resources at max_jobs)
      4. --test-only             (only test resources at max_jobs)
      5. No flags                (error)
    """
    config_resources = {p.name: p for p in config.slurm.resources}

    if args.full or args.test_only:
        resources = []
        for p in config.slurm.resources:
            if args.test_only and not p.test:
                continue
            if args.full and p.test and not args.include_test:
                continue
            spec = copy.copy(p)
            spec.num_jobs = spec.max_jobs
            resources.append(spec)
        if not resources:
            label = "test" if args.test_only else "non-test"
            raise ValueError(f"No {label} resources found in config")
        return resources

    if not args.resource:
        raise ValueError(
            "Specify resources to launch:\n"
            "  hive start -r gpu-a 3 -r gpu-b 2\n"
            "  hive start --full\n"
            "  hive start --test-only"
        )

    # Normal mode: --resource name num_jobs [--resource name num_jobs ...]
    resources = []
    for name, num_jobs_str in args.resource:
        if name not in config_resources:
            available = ", ".join(config_resources.keys())
            raise ValueError(f"Unknown resource '{name}'. Available: {available}")
        spec = copy.copy(config_resources[name])
        spec.num_jobs = int(num_jobs_str)
        # CLI overrides
        if args.time:
            spec.walltime = args.time
        resources.append(spec)

    return resources


def cmd_start(args):
    """Start the Hive compute pool."""
    if _get_remote_url():
        print("Cannot start a cluster in remote mode. Run 'hive disconnect' first.")
        return 1
    config = load_config()

    if args.venv:
        config.env.venv_path = args.venv

    try:
        resources = _resolve_resources(args, config)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Check for existing cluster
    state = Dispatcher.load_state()
    if state:
        print(f"Hive cluster already running (Ray head: {state['ray_head_address']})")
        print("Run 'hive stop' first, or use 'hive status' to check.")
        return 1

    dispatcher = Dispatcher(config)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nInterrupted -- stopping Hive...")
        dispatcher.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        part_summary = ", ".join(
            f"{p.name}({p.num_jobs}x{p.gpus_per_job}gpu, {p.walltime or config.slurm.default_walltime})"
            for p in resources
        )
        print(f"Starting Hive: [{part_summary}]")
        dispatcher.start(resources=resources, walltime=config.slurm.default_walltime)

        if args.wait:
            min_gpus = min(p.gpus_per_job for p in resources)
            print("Waiting for workers to join the cluster...")
            gpu_count = dispatcher.wait_for_workers(
                min_gpus=min_gpus,
                timeout=args.wait_timeout,
            )
            print(f"Hive ready: {gpu_count} GPUs available")

        # Auto-launch dashboard
        if not args.no_dashboard:
            _start_dashboard_background(config)

        if args.foreground:
            print("Running in foreground. Press Ctrl+C to stop.")
            while True:
                time.sleep(60)
                status = dispatcher.get_status()
                logging.getLogger(__name__).debug(
                    f"Status: {status['total_gpus']} GPUs, "
                    f"{status['free_gpus']} free"
                )
        else:
            # Disconnect client before exit to avoid Ray threading cleanup errors
            ray.shutdown()
            print("SLURM jobs submitted. Workers will join as they get scheduled.")
            print("Use 'hive status' to check, 'hive submit' to run jobs, 'hive stop' to shut down.")

    except TimeoutError as e:
        print(f"Error: {e}")
        print("Cleaning up...")
        dispatcher.stop()
        return 1
    except Exception as e:
        print(f"Error starting Hive: {e}")
        dispatcher.stop()
        return 1

    return 0


def cmd_add(args):
    """Add resources to a running Hive cluster."""
    if _get_remote_url():
        print("Cannot add resources in remote mode. Run 'hive disconnect' first.")
        return 1
    state = Dispatcher.load_state()
    if not state:
        print("No Hive cluster running. Start one with 'hive start'.")
        return 1

    config = load_config()

    try:
        resources = _resolve_resources(args, config)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    try:
        ray.init(address=_ray_head_address(), namespace="hive", ignore_reinit_error=True)
    except Exception as e:
        print(f"Cannot connect to Ray cluster: {e}")
        return 1

    state = Dispatcher.load_state() or state

    ray_address = state.get("ray_head_address")
    dispatcher = Dispatcher(config)
    dispatcher._ray_head_address = ray_address

    try:
        part_summary = ", ".join(
            f"{p.name}({p.num_jobs}x{p.gpus_per_job}gpu, {p.walltime or config.slurm.default_walltime})"
            for p in resources
        )
        print(f"Adding to Hive: [{part_summary}]")
        new_job_ids = dispatcher.add_resources(
            resources=resources,
            default_walltime=config.slurm.default_walltime,
            state=state,
        )
        print(f"Submitted {len(new_job_ids)} new SLURM jobs: {', '.join(new_job_ids)}")

        if args.wait:
            min_gpus = min(p.gpus_per_job for p in resources)
            gpu_before = int(ray.cluster_resources().get("GPU", 0))
            print("Waiting for new workers to join...")
            gpu_count = dispatcher.wait_for_workers(
                min_gpus=gpu_before + min_gpus,
                timeout=args.wait_timeout,
            )
            print(f"Done. Cluster now has {gpu_count} GPUs.")
        else:
            print("Workers will join as they get scheduled. Check 'hive status'.")

    except TimeoutError as e:
        print(f"Warning: {e}")
        print("Jobs were submitted but workers haven't joined yet. Check 'hive status' later.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    ray.shutdown()
    return 0


def cmd_remove(args):
    """Remove resources from a running Hive cluster."""
    if _get_remote_url():
        print("Cannot remove resources in remote mode. Run 'hive disconnect' first.")
        return 1
    state = Dispatcher.load_state()
    if not state:
        print("No Hive cluster running.")
        return 1

    config = load_config()

    try:
        resources = _resolve_resources(args, config)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    try:
        ray.init(address=_ray_head_address(), namespace="hive", ignore_reinit_error=True)
    except Exception as e:
        print(f"Cannot connect to Ray cluster: {e}")
        return 1

    dispatcher = Dispatcher(config)
    dispatcher._ray_head_address = state.get("ray_head_address")
    dispatcher._resources = [
        ResourceSpec.from_dict(r) for r in state.get("resources", [])
    ]

    try:
        part_summary = ", ".join(f"{p.name}({p.num_jobs})" for p in resources)
        print(f"Removing from Hive: [{part_summary}]")
        cancelled_ids = dispatcher.remove_resources(
            resources=resources,
            state=state,
        )
        if cancelled_ids:
            print(f"Cancelled {len(cancelled_ids)} SLURM jobs: {', '.join(cancelled_ids)}")
        else:
            print("No matching jobs found to remove.")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    ray.shutdown()
    return 0


def cmd_status(args):
    """Show cluster status."""
    # Remote mode: fetch from dashboard API
    remote_url = _get_remote_url()
    if remote_url:
        try:
            status = _remote_get(f"{remote_url}/api/status")
        except Exception as e:
            print(f"Cannot reach dashboard: {e}")
            return 1
        print()
        print("Hive Cluster Status (remote)")
        print("=" * 50)
        print(f"  Dashboard:    {remote_url}")
        print(f"  Ray Head:     {status.get('ray_head', 'unknown')}")
        n_nodes = status.get("num_gpu_nodes", 0) + status.get("num_cpu_nodes", 0)
        print(f"  Total GPUs:   {status.get('total_gpus', 0)} ({n_nodes} nodes)")
        print(f"  Free GPUs:    {status.get('free_gpus', 0)}")
        print(f"  Available:    {status.get('available_gpu', 0)} GPUs, "
              f"{status.get('available_cpu', 0)} CPU nodes")
        print()
        return 0

    state = Dispatcher.load_state()
    if not state:
        print("No Hive cluster running. Start one with 'hive start'.")
        return 1

    config = load_config()
    ray_head = state.get("ray_head_address") or "auto"
    try:
        ray.init(address=ray_head, namespace="hive", ignore_reinit_error=True)

        resources = ray.cluster_resources()
        available = ray.available_resources()
        nodes = ray.nodes()
    except Exception as e:
        print(f"Cannot connect to Ray cluster: {e}")
        print(f"Saved state: {json.dumps(state, indent=2)}")
        return 1


    total_gpus = int(resources.get("GPU", 0))
    free_gpus = int(available.get("GPU", 0))
    alive_nodes = [n for n in nodes if n.get("Alive", False) and int(n.get("Resources", {}).get("GPU", 0)) > 0]

    # Check if dashboard is running
    from .config import DASHBOARD_PID_PATH
    dashboard_url = None
    if DASHBOARD_PID_PATH.exists():
        try:
            pid = int(DASHBOARD_PID_PATH.read_text().strip())
            os.kill(pid, 0)
            dashboard_url = f"http://localhost:{config.ray.dashboard_port}"
        except (ProcessLookupError, ValueError):
            pass

    print()
    print("Hive Cluster Status")
    print("=" * 50)
    print(f"  Ray Head:     {state.get('ray_head_address', 'unknown')}")
    if dashboard_url:
        print(f"  Dashboard:    {dashboard_url}")
    print(f"  Total GPUs:   {total_gpus} ({len(alive_nodes)} nodes)")
    print(f"  Free GPUs:    {free_gpus}")
    print()

    if alive_nodes:
        print("  Nodes:")
        for node in alive_nodes:
            res = node.get("Resources", {})
            node_gpus = int(res.get("GPU", 0))
            node_name = node.get("NodeName", "unknown")
            # Extract partition label from custom resource keys
            parts = [k.split(":", 1)[1] for k in res if k.startswith("partition:")]
            part_str = f"  ({', '.join(parts)})" if parts else ""
            print(f"    {node_name:20s}  {node_gpus} GPUs{part_str}")
        print()

    if state.get("resources"):
        print("  Resources:")
        for p in state["resources"]:
            acct = f" (account: {p['account']})" if p.get("account") else ""
            print(f"    {p['name']:12s}  {p['gpus_per_job']} GPUs/node, {p['num_jobs']} jobs{acct}")
        print()

    if state.get("slurm_jobs"):
        slurm_jobs = state["slurm_jobs"]
        if isinstance(slurm_jobs, list):
            print(f"  SLURM Jobs:   {', '.join(slurm_jobs)}")
        else:
            print("  SLURM Jobs:")
            for jid, meta in slurm_jobs.items():
                rname = meta.get("resource_name", "unknown")
                print(f"    {jid}  ({rname})")
        print()

    ray.shutdown()
    return 0


def _symlink_job_logs(job_id: str, log_dir: str):
    """Create symlinks for a job's stdout/stderr in the given directory."""
    from pathlib import Path
    from .config import JOBS_DIR
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    job_dir = JOBS_DIR / job_id
    for name in ("stdout.log", "stderr.log"):
        src = job_dir / name
        dst = log_path / f"{job_id}.{name}"
        dst.unlink(missing_ok=True)
        dst.symlink_to(src)


def cmd_submit(args):
    """Submit a job to the cluster.

    By default, jobs are submitted asynchronously (returns immediately).
    Use --wait for synchronous execution (blocks until completion).
    """
    # Remote mode: submit via dashboard API
    remote_url = _get_remote_url()
    if remote_url:
        return _remote_submit(args, remote_url)

    state = Dispatcher.load_state()
    if not state:
        print("No Hive cluster running. Start one with 'hive start'.")
        return 1

    config = load_config()

    try:
        ray.init(address=_ray_head_address(), namespace="hive", ignore_reinit_error=True)
    except Exception as e:
        print(f"Cannot connect to Ray cluster: {e}")
        return 1


    # Create a minimal dispatcher connected to the running cluster.
    # Reconstruct resource specs from saved state for proper validation.
    dispatcher = Dispatcher(config)
    dispatcher._ray_head_address = state.get("ray_head_address")
    if state.get("resources"):
        dispatcher._resources = [ResourceSpec.from_dict(p) for p in state["resources"]]
    elif state.get("max_gpus_per_node"):
        config.slurm.gpus_per_job = state["max_gpus_per_node"]

    workdir = args.workdir or config.env.working_dir or os.getcwd()
    num_gpus = args.gpus
    num_cpus = args.cpus
    resource = args.resource
    resource_info = f", resource: {resource}" if resource else ""

    if args.wait:
        # Synchronous mode (old behavior)
        print(f"Submitting (sync): {args.command}")
        print(f"  GPUs: {num_gpus}, CPUs: {num_cpus}, workdir: {workdir}{resource_info}")

        result = dispatcher.submit_job(
            command=args.command,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            workdir=workdir,
            timeout=args.timeout,
            resource=resource,
        )

        if result["returncode"] == 0:
            if result["stdout"]:
                print(result["stdout"], end="")
        else:
            print(f"Job failed (exit code {result['returncode']})")
            if result["stderr"]:
                print(result["stderr"], end="")
            if result["stdout"]:
                print(result["stdout"], end="")

        if result.get("duration"):
            print(f"\n[completed in {result['duration']:.1f}s]")

        if args.log_dir and result.get("job_id"):
            _symlink_job_logs(result["job_id"], args.log_dir)

        ray.shutdown()
        return result["returncode"]
    else:
        # Async mode (default)
        if not args.quiet:
            print(f"Submitting (async): {args.command}")
            print(f"  GPUs: {num_gpus}, CPUs: {num_cpus}, workdir: {workdir}{resource_info}")

        record = dispatcher.submit_async(
            command=args.command,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            workdir=workdir,
            resource=resource,
            on_complete=args.on_complete,
            timeout_sec=args.timeout_sec,
            max_retries=args.max_retries,
            priority=args.priority,
        )

        if args.log_dir:
            _symlink_job_logs(record.job_id, args.log_dir)

        if args.quiet:
            print(record.job_id)
        else:
            print(f"  Job ID: {record.job_id}")
            print(f"Use 'hive jobs' to check status, 'hive logs {record.job_id}' for output.")

        ray.shutdown()
        return 0


def _remote_submit(args, remote_url: str) -> int:
    """Submit a job via the remote dashboard API."""
    payload = {
        "command": args.command,
        "num_gpus": args.gpus,
        "num_cpus": args.cpus,
        "user": os.environ.get("USER") or "unknown",
    }
    if args.workdir:
        payload["workdir"] = args.workdir
    if args.resource:
        payload["resource"] = args.resource
    if args.on_complete:
        payload["on_complete"] = args.on_complete
    if args.timeout_sec is not None:
        payload["timeout_sec"] = args.timeout_sec
    if args.max_retries > 0:
        payload["max_retries"] = args.max_retries
    if args.priority != "normal":
        payload["priority"] = args.priority

    if not args.quiet:
        resource_info = f", resource: {args.resource}" if args.resource else ""
        print(f"Submitting (remote): {args.command}")
        print(f"  GPUs: {args.gpus}, CPUs: {args.cpus}{resource_info}")

        if args.wait:
            print("  Note: --wait is not supported in remote mode; job submitted async.")

    try:
        result = _remote_post(f"{remote_url}/api/submit", payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            err = json.loads(body).get("error", body)
        except Exception:
            err = body
        print(f"Error: {err}")
        return 1
    except Exception as e:
        print(f"Cannot reach dashboard: {e}")
        return 1

    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    if getattr(args, "log_dir", None) and result.get("job_id"):
        _symlink_job_logs(result["job_id"], args.log_dir)

    if args.quiet:
        print(result["job_id"])
    else:
        print(f"  Job ID: {result['job_id']}")
        print(f"Use 'hive jobs' to check status, 'hive logs {result['job_id']}' for output.")
    return 0


def _get_max_gpus_per_node() -> int:
    """Best-effort check: max GPUs on any single alive node. Returns 0 if Ray unavailable."""
    try:
        nodes = ray.nodes()
        return max(
            (int(n.get("Resources", {}).get("GPU", 0))
             for n in nodes if n.get("Alive", False)),
            default=0,
        )
    except Exception:
        return 0


def _display_status(record, max_node_gpus: int) -> str:
    """Resolve display status: 'queued' vs 'waiting' based on cluster capacity."""
    if record.status != "queued":
        return record.status
    if max_node_gpus > 0 and record.num_gpus > max_node_gpus:
        return "waiting"
    return "queued"


def cmd_jobs(args):
    """List all async jobs."""
    # Remote mode: fetch from dashboard API
    remote_url = _get_remote_url()
    if remote_url:
        return _remote_jobs(args, remote_url)

    records = JobRecord.load_all()
    if not records:
        print("No jobs found.")
        return 0

    # Filter by status if requested (supports comma-separated values)
    # "waiting" is a display alias for "queued", so include queued when filtering
    filter_statuses = None
    if args.status:
        filter_statuses = set(args.status.split(","))
        if "waiting" in filter_statuses:
            filter_statuses.add("queued")
        records = [r for r in records if r.status in filter_statuses]
        if not records:
            print(f"No jobs with status '{args.status}'.")
            return 0

    # Best-effort Ray check for queued jobs that can't be satisfied
    has_queued = any(r.status == "queued" for r in records)
    max_node_gpus = 0
    if has_queued:
        try:
            ray.init(address=_ray_head_address(), namespace="hive", ignore_reinit_error=True)
            max_node_gpus = _get_max_gpus_per_node()
        except Exception:
            pass

    # Sort by submission time (newest first)
    records.sort(key=lambda r: r.submitted_at, reverse=True)

    # Apply display filter for "waiting" if needed: drop plain-queued jobs,
    # but keep any other statuses that were requested alongside "waiting".
    requested_statuses = set(args.status.split(",")) if args.status else set()
    if "waiting" in requested_statuses and "queued" not in requested_statuses:
        records = [
            r for r in records
            if r.status in requested_statuses
            or _display_status(r, max_node_gpus) == "waiting"
        ]

    # Show USER column only when multiple distinct users exist
    users = {r.user for r in records if r.user}
    multi_user = len(users) > 1

    print()
    if multi_user:
        print(f"{'ID':10s} {'STATUS':12s} {'USER':10s} {'GPUS':6s} {'RESOURCE':12s} {'SUBMITTED':20s} {'COMMAND'}")
        print("-" * 100)
    else:
        print(f"{'ID':10s} {'STATUS':12s} {'GPUS':6s} {'RESOURCE':12s} {'SUBMITTED':20s} {'COMMAND'}")
        print("-" * 90)
    for r in records:
        submitted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.submitted_at)) if r.submitted_at else "\u2014"
        cmd_display = r.command[:40] + "..." if len(r.command) > 43 else r.command
        res = r.resource or "\u2014"
        status = _display_status(r, max_node_gpus)
        if multi_user:
            user = r.user or "\u2014"
            print(f"{r.job_id:10s} {status:12s} {user:10s} {r.num_gpus:<6d} {res:12s} {submitted:20s} {cmd_display}")
        else:
            print(f"{r.job_id:10s} {status:12s} {r.num_gpus:<6d} {res:12s} {submitted:20s} {cmd_display}")
    print()
    return 0


def _remote_jobs(args, remote_url: str) -> int:
    """List jobs from the remote dashboard API."""
    try:
        jobs = _remote_get(f"{remote_url}/api/jobs")
    except Exception as e:
        print(f"Cannot reach dashboard: {e}")
        return 1

    if not jobs:
        print("No jobs found.")
        return 0

    # Filter by status
    if args.status:
        filter_statuses = set(args.status.split(","))
        jobs = [j for j in jobs if j.get("status") in filter_statuses]
        if not jobs:
            print(f"No jobs with status '{args.status}'.")
            return 0

    # Sort by submitted_at descending
    jobs.sort(key=lambda j: j.get("submitted_at", 0), reverse=True)

    # Show USER column only when multiple distinct users exist
    users = {j.get("user") for j in jobs if j.get("user")}
    multi_user = len(users) > 1

    print()
    if multi_user:
        print(f"{'ID':10s} {'STATUS':12s} {'USER':10s} {'GPUS':6s} {'RESOURCE':12s} {'SUBMITTED':20s} {'COMMAND'}")
        print("-" * 100)
    else:
        print(f"{'ID':10s} {'STATUS':12s} {'GPUS':6s} {'RESOURCE':12s} {'SUBMITTED':20s} {'COMMAND'}")
        print("-" * 90)
    for j in jobs:
        submitted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(j["submitted_at"])) if j.get("submitted_at") else "\u2014"
        cmd = j.get("command", "")
        cmd_display = cmd[:40] + "..." if len(cmd) > 43 else cmd
        res = j.get("resource") or j.get("partition") or "\u2014"
        status = j.get("status", "unknown")
        num_gpus = j.get("num_gpus", 1)
        if multi_user:
            user = j.get("user") or "\u2014"
            print(f"{j['job_id']:10s} {status:12s} {user:10s} {num_gpus:<6d} {res:12s} {submitted:20s} {cmd_display}")
        else:
            print(f"{j['job_id']:10s} {status:12s} {num_gpus:<6d} {res:12s} {submitted:20s} {cmd_display}")
    print()
    return 0


def cmd_queue(args):
    """Show queued, waiting, and running jobs."""
    args.status = "queued,running"
    return cmd_jobs(args)


def cmd_logs(args):
    """Show stdout/stderr for a job."""
    # Remote mode: fetch from dashboard API
    remote_url = _get_remote_url()
    if remote_url:
        return _remote_logs(args, remote_url)

    record = JobRecord.load(args.job_id)
    if record is None:
        print(f"Job '{args.job_id}' not found.")
        return 1

    print(f"Job {record.job_id} ({record.status})")
    print(f"  Command:   {record.command}")
    print(f"  GPUs:      {record.num_gpus}")
    if record.user:
        print(f"  User:      {record.user}")
    if record.submitted_at:
        print(f"  Submitted: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.submitted_at))}")
    if record.started_at:
        print(f"  Started:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.started_at))}")
    if record.completed_at:
        print(f"  Completed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.completed_at))}")
    if record.returncode is not None:
        print(f"  Exit code: {record.returncode}")
    print()

    stdout = record.stdout
    stderr = record.stderr

    if stdout:
        print("\u2500\u2500 stdout \u2500\u2500")
        print(stdout, end="")
        if not stdout.endswith("\n"):
            print()
    if stderr:
        print("\u2500\u2500 stderr \u2500\u2500")
        print(stderr, end="")
        if not stderr.endswith("\n"):
            print()

    if record.status == "queued":
        print("(job is still queued \u2014 waiting for GPUs)")
    elif record.status == "running":
        print("(job is still running)")

    return 0


def _remote_logs(args, remote_url: str) -> int:
    """Show logs for a job via the remote dashboard API."""
    try:
        data = _remote_get(f"{remote_url}/api/jobs/{args.job_id}/logs")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"Job '{args.job_id}' not found.")
        else:
            print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Cannot reach dashboard: {e}")
        return 1

    if "error" in data:
        print(f"Error: {data['error']}")
        return 1

    status = data.get("status", "unknown")
    print(f"Job {data['job_id']} ({status})")
    print(f"  Command:   {data.get('command', '')}")
    print(f"  GPUs:      {data.get('num_gpus', 1)}")
    if data.get("user"):
        print(f"  User:      {data['user']}")
    if data.get("submitted_at"):
        print(f"  Submitted: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['submitted_at']))}")
    if data.get("started_at"):
        print(f"  Started:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['started_at']))}")
    if data.get("completed_at"):
        print(f"  Completed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['completed_at']))}")
    if data.get("returncode") is not None:
        print(f"  Exit code: {data['returncode']}")
    print()

    stdout = data.get("stdout", "")
    stderr = data.get("stderr", "")
    if stdout:
        print("\u2500\u2500 stdout \u2500\u2500")
        print(stdout, end="")
        if not stdout.endswith("\n"):
            print()
    if stderr:
        print("\u2500\u2500 stderr \u2500\u2500")
        print(stderr, end="")
        if not stderr.endswith("\n"):
            print()

    if status == "queued":
        print("(job is still queued \u2014 waiting for GPUs)")
    elif status == "running":
        print("(job is still running)")

    return 0


def _start_dashboard_background(config: HiveConfig, port: int = None, host: str = None):
    """Launch the dashboard as a background process. Returns True on success."""
    from .config import DASHBOARD_PID_PATH

    if port is None:
        port = config.ray.dashboard_port
    if host is None:
        host = config.ray.dashboard_host

    # Already running?
    if DASHBOARD_PID_PATH.exists():
        try:
            pid = int(DASHBOARD_PID_PATH.read_text().strip())
            os.kill(pid, 0)
            return True  # already running
        except (ProcessLookupError, ValueError):
            DASHBOARD_PID_PATH.unlink()

    try:
        from .dashboard import create_app  # noqa: F401
    except ImportError:
        print("Dashboard requires Flask. Install it with: pip install flask")
        return False

    import subprocess as _sp
    log_path = HIVE_DIR / "logs" / "dashboard.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(str(log_path), "w")
    try:
        proc = _sp.Popen(
            [sys.executable, "-m", "hive.dashboard", "--host", host, "--port", str(port)],
            stdin=_sp.DEVNULL,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,
        )
    finally:
        log_fh.close()

    DASHBOARD_PID_PATH.write_text(str(proc.pid))
    print(f"Dashboard started at http://{host}:{port} (PID: {proc.pid})")
    return True


def cmd_serve(args):
    """Start the Hive web dashboard.

    Runs in the background by default. Use --foreground to block.
    Use --stop to kill a running dashboard.
    """
    if _get_remote_url():
        print("Cannot manage dashboard in remote mode. Run 'hive disconnect' first.")
        return 1
    from .config import DASHBOARD_PID_PATH

    # Handle --stop
    if args.stop:
        if not DASHBOARD_PID_PATH.exists():
            print("No dashboard running.")
            return 0
        try:
            pid = int(DASHBOARD_PID_PATH.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            DASHBOARD_PID_PATH.unlink()
            print(f"Dashboard stopped (PID: {pid}).")
        except ProcessLookupError:
            DASHBOARD_PID_PATH.unlink()
            print("Dashboard process already gone. Cleaned up PID file.")
        except Exception as e:
            print(f"Error stopping dashboard: {e}")
            return 1
        return 0

    # Check if already running
    if DASHBOARD_PID_PATH.exists():
        try:
            pid = int(DASHBOARD_PID_PATH.read_text().strip())
            os.kill(pid, 0)  # check if process exists
            print(f"Dashboard already running (PID: {pid}). Use 'hive serve --stop' first.")
            return 1
        except (ProcessLookupError, ValueError):
            DASHBOARD_PID_PATH.unlink()  # stale PID file

    state = Dispatcher.load_state()
    if not state:
        print("No Hive cluster running. Start one with 'hive start'.")
        return 1

    config = load_config()
    port = args.port or config.ray.dashboard_port

    try:
        from .dashboard import create_app
    except ImportError:
        print("Dashboard requires Flask. Install it with: pip install flask")
        return 1

    if args.foreground:
        # Blocking mode
        try:
            ray.init(address=_ray_head_address(), namespace="hive", ignore_reinit_error=True)
        except Exception as e:
            print(f"Cannot connect to Ray cluster: {e}")
            return 1

        app = create_app(config)
        host = getattr(args, "host", None) or config.ray.dashboard_host
        print(f"Hive dashboard running at http://{host}:{port}")
        print("Press Ctrl+C to stop.")
        app.run(host=host, port=port, debug=False, threaded=True)
        return 0

    # Background mode
    _start_dashboard_background(config, port, getattr(args, "host", None))
    print("Use 'hive serve --stop' to shut it down.")
    os._exit(0)


def _stop_dashboard():
    """Kill the dashboard process if running."""
    from .config import DASHBOARD_PID_PATH
    if not DASHBOARD_PID_PATH.exists():
        return
    try:
        pid = int(DASHBOARD_PID_PATH.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        DASHBOARD_PID_PATH.unlink()
        print(f"Dashboard stopped (PID: {pid}).")
    except ProcessLookupError:
        DASHBOARD_PID_PATH.unlink()
    except Exception:
        pass


def cmd_stop(args):
    """Stop the Hive cluster."""
    if _get_remote_url():
        print("Cannot stop a cluster in remote mode. Run 'hive disconnect' first.")
        return 1

    state = Dispatcher.load_state()

    if not getattr(args, "yes", False):
        n_slurm = len(state.get("slurm_jobs", {})) if state else 0
        print("\033[1;31m" + "=" * 60)
        print("  WARNING: This will destroy the entire Hive cluster!")
        print("=" * 60 + "\033[0m")
        print(f"  - Cancel ALL {n_slurm} SLURM jobs (may take DAYS to re-queue)")
        print("  - Stop Ray head and all workers")
        print("  - Delete all job logs and sbatch scripts")
        print()
        print("\033[1;33m  IF YOU ARE AN AI AGENT: DO NOT PROCEED.")
        print("  You do NOT have permission to cancel SLURM jobs.")
        print("  Only a human operator can authorize this.\033[0m")
        print()
        try:
            answer = input("  Type 'yes' to confirm: ")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        if answer.strip().lower() != "yes":
            print("Aborted.")
            return 1

    _stop_dashboard()
    if not state:
        print("No Hive cluster running.")
        # Still try to clean up Ray in case state file was lost
        from .dispatcher import _ray_cmd
        import subprocess
        subprocess.run([_ray_cmd(), "stop", "--force"], capture_output=True)
        return 0

    config = load_config()
    dispatcher = Dispatcher(config)
    dispatcher._ray_head_address = state.get("ray_head_address")

    # Reconstruct pool with saved SLURM job IDs (handles list and dict formats)
    if state.get("slurm_jobs"):
        from .pool import PoolManager, SlurmJob
        dispatcher.pool = PoolManager(config, dispatcher._ray_head_address)
        slurm_jobs = state["slurm_jobs"]
        if isinstance(slurm_jobs, list):
            for job_id in slurm_jobs:
                dispatcher.pool.jobs[job_id] = SlurmJob(job_id=job_id, state="UNKNOWN")
        else:
            for job_id, meta in slurm_jobs.items():
                dispatcher.pool.jobs[job_id] = SlurmJob(
                    job_id=job_id, state="UNKNOWN",
                    resource_name=meta.get("resource_name", ""),
                )

    try:
        ray.init(address=_ray_head_address(), namespace="hive", ignore_reinit_error=True)
    except Exception:
        pass

    dispatcher.stop()

    # Clear session artifacts (jobs, logs, sbatch scripts, tracking)
    import shutil
    from .config import JOBS_DIR, SLURM_TRACKING_DIR
    for d in [JOBS_DIR, HIVE_DIR / "logs", HIVE_DIR / "sbatch", SLURM_TRACKING_DIR]:
        if d.exists():
            shutil.rmtree(d)

    # Remove remote config (auto-disconnect if self-connected)
    if REMOTE_CONFIG_PATH.exists():
        REMOTE_CONFIG_PATH.unlink()

    print("Hive stopped.")
    return 0


def cmd_restart(args):
    """Restart Ray head and dashboard without cancelling SLURM jobs."""
    if _get_remote_url():
        print("Cannot restart in remote mode. Run 'hive disconnect' first.")
        return 1

    state = Dispatcher.load_state()
    if not state:
        print("No Hive cluster running. Use 'hive start' instead.")
        return 1

    config = load_config()
    _stop_dashboard()

    dispatcher = Dispatcher(config)
    dispatcher._ray_head_address = state.get("ray_head_address")

    # Check if Ray head is reachable before trying ray.init() —
    # connecting to a dead session causes a C-level segfault that
    # Python's try/except cannot catch.
    from .dispatcher import _ray_cmd
    ray_reachable = False
    try:
        result = subprocess.run(
            [_ray_cmd(), "status"],
            capture_output=True, text=True, timeout=15,
        )
        ray_reachable = result.returncode == 0
    except Exception:
        pass

    if ray_reachable:
        try:
            ray.init(address=_ray_head_address(), namespace="hive",
                     ignore_reinit_error=True)
        except Exception:
            pass

    dispatcher.restart()

    # Re-launch dashboard
    _start_dashboard_background(config)

    print("Hive restarted. SLURM workers will reconnect automatically.")
    return 0


def cmd_relaunch(args):
    """Cleanly relaunch the local Hive control plane while preserving SLURM jobs."""
    if _get_remote_url():
        print("Cannot relaunch in remote mode. Run 'hive disconnect' first.")
        return 1

    config = load_config()
    ensure_hive_dir()

    preserved = _build_preserved_relaunch_state(config)
    if preserved is None:
        print("No tracked Hive state or SLURM jobs found. Use 'hive start' instead.")
        return 1

    _stop_dashboard()

    dispatcher = Dispatcher(config)
    dispatcher._ray_head_address = preserved["ray_head_address"]
    dispatcher._resources = [ResourceSpec.from_dict(r) for r in preserved["resources"]]

    from .pool import PoolManager, SlurmJob
    dispatcher.pool = PoolManager(config, dispatcher._ray_head_address)
    for job_id, resource_name in preserved["slurm_jobs"].items():
        dispatcher.pool.jobs[job_id] = SlurmJob(
            job_id=job_id,
            state="UNKNOWN",
            resource_name=resource_name,
        )

    _rebuild_preserved_sbatch_templates(
        dispatcher,
        preserved["resources"],
        preserved["walltime"],
    )

    # Rebuild a consistent local state file before restarting the head so
    # worker reconnect loops do not interpret the relaunch as a full stop.
    dispatcher._save_state(
        resources=preserved["resources"],
        max_gpus_per_node=preserved["max_gpus_per_node"],
        walltime=preserved["walltime"],
    )

    dispatcher.restart()
    _start_dashboard_background(config)

    print(
        "Hive relaunched cleanly. "
        f"Preserved {len(preserved['slurm_jobs'])} tracked SLURM jobs."
    )
    return 0


def cmd_clear(args):
    """Clear job history and logs."""
    import shutil
    from .config import JOBS_DIR

    cleared = []
    clear_all = getattr(args, "all", False)
    if not getattr(args, "keep_jobs", False) and JOBS_DIR.exists():
        if clear_all:
            count = sum(1 for _ in JOBS_DIR.iterdir() if _.is_dir())
            shutil.rmtree(JOBS_DIR)
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
            cleared.append(f"{count} jobs")
        else:
            # Default: only remove finished (non-running, non-queued) job dirs
            count = 0
            for job_dir in list(JOBS_DIR.iterdir()):
                if not job_dir.is_dir():
                    continue
                meta_path = job_dir / "meta.json"
                if not meta_path.exists():
                    shutil.rmtree(job_dir)
                    count += 1
                    continue
                try:
                    meta = json.loads(meta_path.read_text())
                except (json.JSONDecodeError, ValueError):
                    # Genuinely corrupt JSON — safe to remove
                    shutil.rmtree(job_dir)
                    count += 1
                    continue
                except OSError:
                    # Transient read error — skip, don't delete
                    continue
                if meta.get("status") in ("running", "queued"):
                    continue
                shutil.rmtree(job_dir)
                count += 1
            if count:
                cleared.append(f"{count} finished jobs")

    log_dir = HIVE_DIR / "logs"
    if log_dir.exists():
        shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        cleared.append("logs")

    if cleared:
        print(f"Cleared: {', '.join(cleared)}.")
    else:
        print("Nothing to clear.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="hive",
        description="Hive: Persistent GPU compute pool for AI coding agents",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- hive start
    p_start = subparsers.add_parser("start", help="Start the compute pool")
    p_start.add_argument("--resource", "-r", nargs=2, action="append", metavar=("NAME", "NUM_JOBS"),
                         help="Resource name and number of jobs. Can be repeated. "
                              "E.g. -r gpu-a 3 -r gpu-b 2")
    p_start.add_argument("--full", action="store_true",
                         help="Launch all non-test resources at max_jobs from config")
    p_start.add_argument("--include-test", action="store_true",
                         help="Include test resources when using --full")
    p_start.add_argument("--test-only", action="store_true",
                         help="Launch only test resources at max_jobs")
    p_start.add_argument("--time", type=str, help="Override walltime for all resources")
    p_start.add_argument("--venv", type=str, help="Path to virtual environment")
    p_start.add_argument("--wait", action="store_true", help="Block until at least one worker joins")
    p_start.add_argument("--foreground", action="store_true", help="Run in foreground (don't background)")
    p_start.add_argument("--no-dashboard", action="store_true", help="Don't auto-launch the web dashboard")
    p_start.add_argument("--wait-timeout", type=int, default=600, help="Max seconds to wait for workers with --wait (default: 600)")
    p_start.set_defaults(func=cmd_start)

    # -- hive add
    p_add = subparsers.add_parser("add", help="Add resources to a running cluster")
    p_add.add_argument("--resource", "-r", nargs=2, action="append", metavar=("NAME", "NUM_JOBS"),
                       help="Resource name and number of jobs. Can be repeated. "
                            "E.g. -r gpu-a 3 -r gpu-b 2")
    p_add.add_argument("--full", action="store_true",
                       help="Add all non-test resources at max_jobs from config")
    p_add.add_argument("--include-test", action="store_true",
                       help="Include test resources when using --full")
    p_add.add_argument("--test-only", action="store_true",
                       help="Add only test resources at max_jobs")
    p_add.add_argument("--time", type=str, help="Override walltime for all resources")
    p_add.add_argument("--wait", action="store_true", help="Block until at least one new worker joins")
    p_add.add_argument("--wait-timeout", type=int, default=600, help="Max seconds to wait for workers with --wait (default: 600)")
    p_add.set_defaults(func=cmd_add)

    # -- hive remove
    p_remove = subparsers.add_parser("remove", help="Remove resources from a running cluster")
    p_remove.add_argument("--resource", "-r", nargs=2, action="append", metavar=("NAME", "NUM_JOBS"),
                          help="Resource name and number of jobs to remove. Can be repeated. "
                               "E.g. -r gpu-b 2 -r gpu 1")
    p_remove.add_argument("--full", action="store_true",
                          help="Remove all non-test resources")
    p_remove.add_argument("--include-test", action="store_true",
                          help="Include test resources when using --full")
    p_remove.add_argument("--test-only", action="store_true",
                          help="Remove only test resources")
    p_remove.add_argument("--time", type=str, help=argparse.SUPPRESS)  # unused, for _resolve_resources compat
    p_remove.set_defaults(func=cmd_remove)

    # -- hive status
    p_status = subparsers.add_parser("status", help="Show cluster status")
    p_status.set_defaults(func=cmd_status)

    # -- hive submit
    p_submit = subparsers.add_parser("submit", help="Submit a job to the cluster")
    p_submit.add_argument("command", type=str, help="Command to run")
    p_submit.add_argument("--gpus", type=int, default=1, help="Number of GPUs (default: 1)")
    p_submit.add_argument("--cpus", type=int, default=1, help="Number of CPUs (default: 1)")
    p_submit.add_argument("--resource", "-r", type=str,
                          help="Target resource(s), comma-separated (e.g. -r gpu-b,gpu-test). "
                               "Job runs on first available. Omit for any node.")
    p_submit.add_argument("--workdir", type=str, help="Working directory on compute node")
    p_submit.add_argument("--timeout-sec", type=int, dest="timeout_sec",
                          help="Maximum wall-clock seconds the job may run before being killed")
    p_submit.add_argument("--max-retries", type=int, default=0, dest="max_retries",
                          help="Auto-retry on infrastructure failure (e.g. SLURM worker death). Default: 0")
    p_submit.add_argument("--timeout", type=int, help="Seconds to wait for result (only with --wait)")
    p_submit.add_argument("--wait", action="store_true", help="Block until job completes (default: async)")
    p_submit.add_argument("--quiet", "-q", action="store_true", help="Print only the job ID (for scripting)")
    p_submit.add_argument("--on-complete", type=str, help="Shell command to run when job finishes (async only)")
    p_submit.add_argument("--log-dir", type=str,
                          help="Symlink job stdout/stderr into this directory (e.g. --log-dir ./logs)")
    p_submit.add_argument("--priority", type=str, default="normal", choices=["normal", "low"],
                          help="Job priority. 'low' jobs yield GPUs to normal jobs and auto-retry (default: normal)")
    p_submit.set_defaults(func=cmd_submit)

    # -- hive jobs
    p_jobs = subparsers.add_parser("jobs", help="List async jobs")
    p_jobs.add_argument("--status", type=str,
                        help="Filter by job status (queued, running, completed, failed; comma-separated)")
    p_jobs.set_defaults(func=cmd_jobs)

    # -- hive queue
    p_queue = subparsers.add_parser("queue", help="Show queued and running jobs")
    p_queue.set_defaults(func=cmd_queue)

    # -- hive logs
    p_logs = subparsers.add_parser("logs", help="Show stdout/stderr for a job")
    p_logs.add_argument("job_id", type=str, help="Job ID to show logs for")
    p_logs.set_defaults(func=cmd_logs)

    # -- hive cancel
    p_cancel = subparsers.add_parser("cancel", help="Cancel queued or running jobs")
    p_cancel.add_argument("job_ids", nargs="+", help="Job ID(s) to cancel")
    p_cancel.set_defaults(func=cmd_cancel)

    # -- hive serve
    p_serve = subparsers.add_parser("serve", help="Start the Hive web dashboard")
    p_serve.add_argument("--port", type=int, help="Dashboard port (default: ray.dashboard_port from config)")
    p_serve.add_argument("--host", type=str, default=None,
                         help="Bind address (default: ray.dashboard_host from config, 127.0.0.1). "
                              "The dashboard has no authentication: use 0.0.0.0 only on a trusted network.")
    p_serve.add_argument("--foreground", action="store_true", help="Run in foreground (block)")
    p_serve.add_argument("--stop", action="store_true", help="Stop a running dashboard")
    p_serve.set_defaults(func=cmd_serve)

    # -- hive connect
    p_connect = subparsers.add_parser("connect", help="Connect to a remote Hive dashboard")
    p_connect.add_argument("target", type=str, nargs="?", default=None,
                           help="Dashboard address (host:port). Defaults to localhost:<dashboard_port>")
    p_connect.set_defaults(func=cmd_connect)

    # -- hive disconnect
    p_disconnect = subparsers.add_parser("disconnect", help="Disconnect from remote Hive cluster")
    p_disconnect.set_defaults(func=cmd_disconnect)

    # -- hive clear
    p_clear = subparsers.add_parser("clear", help="Clear job history and logs")
    p_clear.add_argument("--keep-jobs", action="store_true", default=False,
                         help="Preserve job logs, only clear Ray/SLURM logs")
    p_clear.add_argument("--all", action="store_true", default=False,
                         help="Clear all jobs including running/queued (default: only finished)")
    p_clear.set_defaults(func=cmd_clear)

    # -- hive restart
    p_restart = subparsers.add_parser("restart", help="Restart Ray head + dashboard (keeps SLURM jobs)")
    p_restart.set_defaults(func=cmd_restart)

    # -- hive relaunch
    p_relaunch = subparsers.add_parser(
        "relaunch",
        help="Cleanly relaunch local Hive control-plane state (preserves tracked SLURM jobs)",
    )
    p_relaunch.add_argument(
        "--clean",
        action="store_true",
        help="Accepted for clarity; relaunch always performs local control-plane cleanup.",
    )
    p_relaunch.set_defaults(func=cmd_relaunch)

    # -- hive stop
    p_stop = subparsers.add_parser("stop", help="Stop the cluster")
    p_stop.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt")
    p_stop.set_defaults(func=cmd_stop)

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
