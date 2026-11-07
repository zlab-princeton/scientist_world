"""Hive web dashboard: Flask REST API, single-page monitoring UI, and the
background health loop that dispatches queued jobs.

The dashboard shows node status, SLURM worker jobs and the job queue.
Remote clients (``hive connect``, ``hive.client``, Nexus) talk to its REST
API.

Usage:
    app = create_app()
    app.run(host="127.0.0.1", port=38581)
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Optional

os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.setdefault("RAY_LOG_TO_STDERR", "1")

import ray
from flask import Flask, jsonify, render_template_string, request

from .config import HiveConfig, ResourceSpec, load_config, get_ray_tmpdir, get_launch_pending_grace_sec
from .dispatcher import (
    Dispatcher,
    JobRecord,
    JOB_ACTOR_PREFIX,
    append_job_stderr_message,
    buffer_job_stderr_message,
    is_node_draining,
    launch_job_runner,
    prune_stale_draining_nodes,
    read_job_heartbeat,
    read_job_process_result,
)

# Grace tracker for zombie-actor kills.  Maps actor name -> first time it
# was observed as a kill candidate.  Requires two consecutive sightings
# (separated by at least _ZOMBIE_MIN_AGE_SEC) before ray.kill is called —
# this closes the race where reap_orphaned_job flips a record to
# "preempted" in the same health tick that a fresh actor is being
# launched for a requeue.
_zombie_observations: dict = {}
_ZOMBIE_MIN_AGE_SEC = 45.0

# ---------------------------------------------------------------------------
# Safe Ray queries — run in a subprocess so gRPC segfaults don't kill Flask
# ---------------------------------------------------------------------------

_RAY_QUERY_SCRIPT = '''
import json, ray, os, sys
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.pop("RAY_LOG_TO_STDERR", None)
sys.stderr = open(os.devnull, "w")
addr = "auto"
try:
    from hive.config import STATE_PATH
    _s = json.loads(STATE_PATH.read_text())
    addr = _s.get("ray_head_address") or "auto"
except Exception:
    pass
ray.init(address=addr, namespace="hive", ignore_reinit_error=True)
nodes = ray.nodes()
resources = ray.cluster_resources()
available = ray.available_resources()
print(json.dumps({
    "nodes": nodes,
    "total_gpus": int(resources.get("GPU", 0)),
    "total_cpus": int(resources.get("CPU", 0)),
    "free_gpus": int(available.get("GPU", 0)),
    "free_cpus": int(available.get("CPU", 0)),
}))
'''


def _collect_ray_data():
    """Return the cluster snapshot using the current in-process Ray client."""
    nodes = ray.nodes()
    resources = ray.cluster_resources()
    available = ray.available_resources()
    return {
        "nodes": nodes,
        "total_gpus": int(resources.get("GPU", 0)),
        "total_cpus": int(resources.get("CPU", 0)),
        "free_gpus": int(available.get("GPU", 0)),
        "free_cpus": int(available.get("CPU", 0)),
    }

_GPU_STATS_SCRIPT = '''
import json, os, sys, ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.pop("RAY_LOG_TO_STDERR", None)
sys.stderr = open(os.devnull, "w")
addr = "auto"
try:
    from hive.config import STATE_PATH
    _s = json.loads(STATE_PATH.read_text())
    addr = _s.get("ray_head_address") or "auto"
except Exception:
    pass
ray.init(address=addr, namespace="hive", ignore_reinit_error=True)

@ray.remote(num_cpus=0.01, num_gpus=0)
def _query_nvidia_smi():
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().split("\\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "utilization": int(parts[1]),
                    "mem_used": int(parts[2]),
                    "mem_total": int(parts[3]),
                    "temperature": int(parts[4]),
                })
        return gpus
    except Exception:
        return []

nodes = ray.nodes()
futures = {}
for node in nodes:
    if not node.get("Alive", False):
        continue
    res = node.get("Resources", {})
    if int(res.get("GPU", 0)) == 0:
        continue
    node_id = node["NodeID"]
    node_name = node.get("NodeName", "unknown")
    try:
        strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        future = _query_nvidia_smi.options(
            scheduling_strategy=strategy,
            runtime_env={"env_vars": {"RAY_LOG_TO_STDERR": "1", "RAY_ROTATION_MAX_BYTES": "1"}},
        ).remote()
        futures[node_name] = future
    except Exception:
        pass

results = {}
for name, future in futures.items():
    try:
        results[name] = ray.get(future, timeout=10)
    except Exception:
        results[name] = []

print(json.dumps(results))
'''


_CPU_STATS_SCRIPT = '''
import json, os, sys, time, ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.pop("RAY_LOG_TO_STDERR", None)
sys.stderr = open(os.devnull, "w")
addr = "auto"
try:
    from hive.config import STATE_PATH
    _s = json.loads(STATE_PATH.read_text())
    addr = _s.get("ray_head_address") or "auto"
except Exception:
    pass
ray.init(address=addr, namespace="hive", ignore_reinit_error=True)

@ray.remote(num_cpus=0.01, num_gpus=0)
def _query_cpu_stats():
    import time as _time
    result = {}
    # CPU utilization from /proc/stat (two samples)
    try:
        def read_cpu():
            with open("/proc/stat") as f:
                line = f.readline()  # first line: cpu  user nice system idle ...
            parts = line.split()
            # user nice system idle iowait irq softirq steal
            vals = [int(x) for x in parts[1:]]
            idle = vals[3] + vals[4]  # idle + iowait
            total = sum(vals)
            return idle, total

        idle1, total1 = read_cpu()
        _time.sleep(0.5)
        idle2, total2 = read_cpu()
        d_idle = idle2 - idle1
        d_total = total2 - total1
        cpu_util = round((1.0 - d_idle / d_total) * 100) if d_total > 0 else 0
        result["cpu_util"] = cpu_util
    except Exception:
        result["cpu_util"] = -1

    # Memory from /proc/meminfo
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
        total_kb = meminfo.get("MemTotal", 0)
        avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        result["mem_total_gb"] = round(total_kb / 1048576, 1)
        result["mem_used_gb"] = round((total_kb - avail_kb) / 1048576, 1)
    except Exception:
        result["mem_total_gb"] = 0
        result["mem_used_gb"] = 0

    return result

nodes = ray.nodes()
futures = {}
for node in nodes:
    if not node.get("Alive", False):
        continue
    res = node.get("Resources", {})
    has_partition = any(k.startswith("partition:") for k in res)
    if not has_partition:
        continue  # skip head node
    if int(res.get("GPU", 0)) > 0:
        continue  # skip GPU nodes (covered by gpu_stats)
    node_id = node["NodeID"]
    node_name = node.get("NodeName", "unknown")
    try:
        strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        future = _query_cpu_stats.options(
            scheduling_strategy=strategy,
            runtime_env={"env_vars": {"RAY_LOG_TO_STDERR": "1", "RAY_ROTATION_MAX_BYTES": "1"}},
        ).remote()
        futures[node_name] = future
    except Exception:
        pass

results = {}
for name, future in futures.items():
    try:
        results[name] = ray.get(future, timeout=10)
    except Exception:
        results[name] = {"cpu_util": -1, "mem_total_gb": 0, "mem_used_gb": 0}

print(json.dumps(results))
'''


def _safe_ray_query(timeout=15):
    """Query Ray in a subprocess. Returns cached data on failure/segfault."""
    if ray.is_initialized():
        try:
            data = _collect_ray_data()
            _ray_cache["data"] = data
            _ray_cache["ts"] = time.time()
            return data
        except Exception:
            pass

    try:
        result = subprocess.run(
            [sys.executable, "-c", _RAY_QUERY_SCRIPT],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            _ray_cache["data"] = data
            _ray_cache["ts"] = time.time()
            return data
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        pass
    # Fall back to cache, or empty structure if cache is None
    return _ray_cache.get("data") or {"nodes": [], "total_gpus": 0, "total_cpus": 0, "free_gpus": 0, "free_cpus": 0}


# Cache for Ray data (updated by _safe_ray_query, survives child segfaults)
_ray_cache = {"data": None, "ts": 0.0}
_RAY_CACHE_TTL = 5.0  # seconds

log = logging.getLogger(__name__)
_ORPHAN_MISSING_ACTOR_GRACE_SEC = 180
_ORPHAN_MISSING_ACTOR_CHECKS = 3
_ORPHAN_HEARTBEAT_GRACE_SEC = 180
_ORPHAN_UNCERTAIN_FAILURE_GRACE_SEC = 900
_NODE_DEAD_CHECKS_REQUIRED = 3   # consecutive health-loop checks (~3 min)


def _clear_missing_actor_state(record: JobRecord) -> bool:
    """Clear orphan-suspicion bookkeeping when the job looks healthy again."""
    changed = (
        record.missing_actor_since is not None
        or record.missing_actor_checks != 0
    )
    if changed:
        record.missing_actor_since = None
        record.missing_actor_checks = 0
    return changed


def _find_job_actor(job_id: str):
    """Return (actor_handle_or_none, query_ok)."""
    prefix = f"{JOB_ACTOR_PREFIX}{job_id}"
    try:
        named = ray.util.list_named_actors(all_namespaces=True)
    except Exception as e:
        log.warning("Job %s: could not query named actors: %s", job_id, e)
        return None, False

    for entry in named:
        name = entry if isinstance(entry, str) else entry.get("name", "")
        if not name.startswith(prefix):
            continue
        try:
            return ray.get_actor(name), True
        except ValueError as e:
            # The actor may legitimately disappear between list and lookup.
            log.info("Job %s: actor %s disappeared during lookup: %s", job_id, name, e)
            return None, True
        except Exception as e:
            log.warning("Job %s: could not resolve actor %s: %s", job_id, name, e)
            return None, False
    return None, True


def _queued_job_needs_ray_verification(record: JobRecord) -> bool:
    """Return True when a queued job may already have a detached runner."""
    return record.queued_job_needs_ray_verification()


def _get_node_liveness(node_id: str):
    """Return (node_alive_or_none, query_ok, cluster_has_alive_nodes)."""
    if not node_id:
        return None, False, False
    try:
        nodes = ray.nodes()
    except Exception as e:
        log.warning("Job node %s: could not query Ray nodes: %s", node_id[:12], e)
        return None, False, False

    cluster_has_alive_nodes = any(n.get("Alive", False) for n in nodes)
    for node in nodes:
        if node.get("NodeID") == node_id:
            return bool(node.get("Alive", False)), True, cluster_has_alive_nodes
    return False, True, cluster_has_alive_nodes


def _persist_missing_actor_observation(record: JobRecord, now: float):
    """Record a suspicious missing-actor observation without failing the job."""
    if record.missing_actor_since is None:
        record.missing_actor_since = now
        record.missing_actor_checks = 1
    else:
        record.missing_actor_checks += 1
    record.save_meta()


def _append_stderr_message(record: JobRecord, message: str):
    """Append an infrastructure note to stderr without truncating existing logs."""
    append_job_stderr_message(record.job_id, message)


def _list_live_job_actor_ids() -> tuple[Optional[set[str]], bool]:
    """Return job ids whose detached actors are currently resolvable in Ray."""
    try:
        named = ray.util.list_named_actors(all_namespaces=True)
    except Exception as e:
        log.warning("Could not query named actors for reassignment check: %s", e)
        return None, False

    live_ids = set()
    for entry in named:
        name = entry if isinstance(entry, str) else entry.get("name", "")
        if not name.startswith(JOB_ACTOR_PREFIX):
            continue
        try:
            ray.get_actor(name)
        except ValueError:
            continue
        except Exception as e:
            log.warning("Could not resolve actor %s during reassignment check: %s", name, e)
            return None, False
        suffix = name[len(JOB_ACTOR_PREFIX):]
        live_ids.add(suffix.split("_")[0])

    return live_ids, True


def _get_node_gpu_capacity(node_id: str) -> tuple[Optional[float], bool]:
    """Return the GPU capacity of a specific live Ray node."""
    if not node_id:
        return None, False
    try:
        nodes = ray.nodes()
    except Exception as e:
        log.warning("Could not query Ray nodes for capacity on %s: %s", node_id[:12], e)
        return None, False

    for node in nodes:
        if node.get("NodeID") != node_id:
            continue
        resources = node.get("Resources") or {}
        try:
            return float(resources.get("GPU", 0) or 0), True
        except Exception:
            return None, False

    return None, True


def _has_conflicting_newer_running_job(record: JobRecord, now: float) -> bool:
    """Return True when the node has already been reassigned incompatibly.

    The reassignment proof must be conservative:
    - the original job already looks stale enough to be suspicious
    - the newer replacement jobs still have live Ray actors
    - summed newer GPU demand plus the stale job exceeds this node's capacity
    """
    if not record.node_id or record.num_gpus <= 0:
        return False

    live_actor_ids, actor_query_ok = _list_live_job_actor_ids()
    if not actor_query_ok or not live_actor_ids:
        return False

    node_gpu_capacity, capacity_query_ok = _get_node_gpu_capacity(record.node_id)
    if not capacity_query_ok or node_gpu_capacity is None or node_gpu_capacity <= 0:
        return False

    current_started = record.started_at or record.submitted_at or 0.0
    newer_live_gpu_demand = 0
    for other in JobRecord.load_active():
        if other.job_id == record.job_id:
            continue
        if other.status != "running" or other.node_id != record.node_id:
            continue
        other_started = other.started_at or other.submitted_at or 0.0
        if other_started <= current_started:
            continue
        other_heartbeat_at = read_job_heartbeat(other.job_id)
        if other_heartbeat_at is None:
            continue
        if (now - other_heartbeat_at) > _ORPHAN_HEARTBEAT_GRACE_SEC:
            continue
        if other.job_id not in live_actor_ids:
            continue
        newer_live_gpu_demand += other.num_gpus

    return (record.num_gpus + newer_live_gpu_demand) > node_gpu_capacity


def _finalize_from_process_result(record: JobRecord, result: dict) -> bool:
    """Recover a terminal job state from the worker-side process result file."""
    completed_at = result.get("completed_at")
    returncode = result.get("returncode")
    if completed_at is None or returncode is None:
        return False

    # Don't clobber a terminal status already set by another path
    # (e.g. reap_orphaned_job marked this preempted/failed/cancelled).
    if record.status in ("preempted", "failed", "cancelled", "timeout"):
        return False

    # Reject stale results from a previous attempt.
    # process_result.json is shared across attempts for the same
    # job_id.  The worker embeds its attempt number in the file; the
    # record holds the current attempt.  A mismatch means the file is
    # from a predecessor and must not finalize the live retry.
    #
    # Results with an attempt field must attempt-match the record.
    # Results without one are accepted only on the first attempt: a
    # requeue always bumps the attempt and clears the old file, so a
    # result without an attempt number on a later attempt is stale.
    # Timestamps are deliberately not compared: started_at is stamped by
    # the runner actor and the file's mtime by the worker's filesystem,
    # and the two clocks need not agree.
    result_attempt = result.get("attempt")
    if result_attempt is not None:
        if result_attempt != record.attempt:
            return False
    else:
        if record.attempt != 0:
            return False

    record.completed_at = completed_at
    record.returncode = returncode
    if result.get("timed_out"):
        record.status = "timeout"
    elif returncode == 0:
        record.status = "completed"
    else:
        # Non-zero exit — including SIGKILL (137) — is NOT success.
        # Marking these "completed" silently corrupts downstream
        # consumers that treat completed as proof of good output.
        record.status = "failed"
    _clear_missing_actor_state(record)
    record.save_meta()
    return True


def reap_orphaned_job(record: JobRecord, *, now: Optional[float] = None) -> bool:
    """Best-effort orphan reaper used only by background maintenance.

    Returns True when the job record was mutated.
    """
    if now is None:
        now = time.time()

    if record.status != "running":
        if _clear_missing_actor_state(record):
            record.save_meta()
            return True
        return False

    actor, actor_query_ok = _find_job_actor(record.job_id)

    # ── Actor found ──
    # The detached head-side actor can remain registered with Ray after
    # the worker-side job has actually died (Hive documents this in the
    # runner loop; see dispatcher._AsyncJobRunner).  We therefore cannot
    # treat "actor resolvable" alone as proof the job is alive — but we
    # also cannot kill on raw ray.nodes() signal alone, because that
    # returns stale data under load and would tear down live jobs.
    #
    # Require BOTH signals to act: (1) the job has stopped producing
    # heartbeats for > _ORPHAN_HEARTBEAT_GRACE_SEC, AND (2) the worker
    # node is confirmed dead across _NODE_DEAD_CHECKS_REQUIRED
    # consecutive health cycles.  A recent heartbeat short-circuits to
    # "leave it alone" immediately.
    if actor is not None:
        # A persisted process_result is terminal evidence that the
        # worker finished writing its outcome — even if the detached
        # head-side actor is still registered.  Finalize from it
        # before considering heartbeat staleness or node liveness;
        # otherwise we could kill+requeue a job that already completed
        # (or legitimately failed) on the worker side.
        process_result = read_job_process_result(record.job_id)
        if _finalize_from_process_result(record, process_result):
            return True

        heartbeat_at = read_job_heartbeat(record.job_id)
        heartbeat_fresh = (
            heartbeat_at is not None
            and (now - heartbeat_at) <= _ORPHAN_HEARTBEAT_GRACE_SEC
        )
        # Heartbeat absence doesn't count as "stale" until the job has
        # had time to start writing them.  This protects freshly-launched
        # jobs whose first heartbeat hasn't been flushed yet.
        heartbeat_stale = (
            heartbeat_at is not None
            and (now - heartbeat_at) > _ORPHAN_HEARTBEAT_GRACE_SEC
        ) or (
            heartbeat_at is None
            and record.started_at is not None
            and (now - record.started_at) > _ORPHAN_MISSING_ACTOR_GRACE_SEC
        )

        if heartbeat_fresh:
            # Job is alive — clear any accumulated dead-node evidence.
            # Persist the reset so the next health tick doesn't reload
            # stale counters from disk and act on them.
            mutated = False
            if record.node_dead_checks or record.node_dead_since is not None:
                record.node_dead_since = None
                record.node_dead_checks = 0
                mutated = True
            if _clear_missing_actor_state(record):
                mutated = True
            if mutated:
                record.save_meta()
                return True
            return False

        if not heartbeat_stale or not record.node_id:
            # Not enough evidence yet — the actor exists and heartbeat
            # isn't confirmed stale.  Wait for next cycle.
            return False

        node_alive, node_query_ok, cluster_has_alive_nodes = _get_node_liveness(record.node_id)
        if not node_query_ok or not cluster_has_alive_nodes:
            # Can't trust ray.nodes() answer — wait.
            return False
        if node_alive is not False:
            # Node appears alive — reset counter, wait for next cycle.
            if record.node_dead_checks or record.node_dead_since is not None:
                record.node_dead_since = None
                record.node_dead_checks = 0
                record.save_meta()
                return True
            return False

        # Heartbeat stale AND node confirmed dead — accumulate evidence.
        record.node_dead_checks = (record.node_dead_checks or 0) + 1
        if record.node_dead_since is None:
            record.node_dead_since = now
        if record.node_dead_checks < _NODE_DEAD_CHECKS_REQUIRED:
            record.save_meta()
            return True

        # About to cross the kill threshold.  Re-check the process
        # result one more time — if the worker wrote its outcome while
        # we were accumulating evidence, finalize instead of killing.
        late_result = read_job_process_result(record.job_id)
        if _finalize_from_process_result(record, late_result):
            return True

        # Confirmed dead across enough consecutive cycles — kill the
        # zombie actor and requeue the job.
        try:
            ray.kill(actor)
        except Exception as e:
            log.warning(
                "Job %s: failed to kill actor after node death: %s",
                record.job_id, e,
            )
        # attempt is bumped in requeue_preempted (single place of truth).
        record.status = "preempted"
        record.started_at = None
        record.completed_at = None
        _requeue_msg = (
            f"[hive] Worker node died — requeuing "
            f"(stale heartbeat, {record.node_dead_checks} node-dead checks)"
        )
        _append_stderr_message(record, _requeue_msg)
        _clear_missing_actor_state(record)
        record.node_dead_since = None
        record.node_dead_checks = 0
        record.save_meta()
        return True

    # ── Actor not found ──

    # Ray could not answer actor queries. This is exactly the scenario that
    # previously produced false failures, so fail closed and wait.
    if not actor_query_ok:
        return False

    # Check if the job already wrote its result file (normal completion
    # race: actor self-destructed after writing the result).
    process_result = read_job_process_result(record.job_id)
    if _finalize_from_process_result(record, process_result):
        return True

    # Heartbeat age — a recent heartbeat proves the job was alive
    # recently regardless of what ray.nodes() says.
    heartbeat_at = read_job_heartbeat(record.job_id)
    heartbeat_age = (now - heartbeat_at) if heartbeat_at is not None else None

    if heartbeat_age is not None and heartbeat_age <= _ORPHAN_HEARTBEAT_GRACE_SEC:
        _persist_missing_actor_observation(record, now)
        return True

    if heartbeat_at is None and record.started_at is not None \
            and (now - record.started_at) <= _ORPHAN_MISSING_ACTOR_GRACE_SEC:
        _persist_missing_actor_observation(record, now)
        return True

    # Node liveness — only checked AFTER heartbeat grace is exhausted.
    # Requeue (not fail) on confirmed node death so the job runs on
    # another node and the caller is not notified.
    node_alive = None
    node_query_ok = False
    cluster_has_alive_nodes = False
    if record.node_id:
        node_alive, node_query_ok, cluster_has_alive_nodes = _get_node_liveness(record.node_id)
        if node_query_ok and cluster_has_alive_nodes and node_alive is False:
            # attempt is bumped in requeue_preempted (single place of truth).
            record.status = "preempted"
            record.started_at = None
            record.completed_at = None
            _append_stderr_message(record, "[hive] Worker node died — requeuing")
            _clear_missing_actor_state(record)
            record.node_dead_since = None
            record.node_dead_checks = 0
            record.save_meta()
            return True

    if _has_conflicting_newer_running_job(record, now):
        record.status = "failed"
        record.completed_at = now
        _append_stderr_message(
            record,
            "[hive] Job orphaned: node reassigned to a newer running job",
        )
        record.returncode = -9
        _clear_missing_actor_state(record)
        record.save_meta()
        return True

    first_missing_at = record.missing_actor_since if record.missing_actor_since is not None else now
    missing_checks = record.missing_actor_checks + 1 if record.missing_actor_since is not None else 1
    missing_for = now - first_missing_at

    record.missing_actor_since = first_missing_at
    record.missing_actor_checks = missing_checks

    if missing_checks < _ORPHAN_MISSING_ACTOR_CHECKS:
        record.save_meta()
        return True

    if missing_for < _ORPHAN_MISSING_ACTOR_GRACE_SEC:
        record.save_meta()
        return True

    # If the worker process was still heartbeating recently enough to avoid the
    # earlier fast-path, treat the state as uncertain until it is stale for much
    # longer. This prevents a short Ray/GCS outage from killing the job record.
    if heartbeat_age is not None and heartbeat_age < _ORPHAN_UNCERTAIN_FAILURE_GRACE_SEC:
        record.save_meta()
        return True

    if not node_query_ok and missing_for < _ORPHAN_UNCERTAIN_FAILURE_GRACE_SEC:
        record.save_meta()
        return True

    record.status = "failed"
    record.completed_at = now
    _append_stderr_message(record, "[hive] Job orphaned: Ray actor no longer exists")
    record.returncode = -9
    _clear_missing_actor_state(record)
    record.save_meta()
    return True


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------

def create_app(config: HiveConfig = None, start_health_thread: bool = True) -> Flask:
    """Create and configure the Flask dashboard application.

    Args:
        config: Optional HiveConfig. If None, loads from default config path.
        start_health_thread: Start the background health loop (orphan
            reaping, queued-job dispatch). Tests pass False so no daemon
            thread outlives the test.

    Returns:
        Configured Flask app with API endpoints and HTML dashboard.
    """
    if config is None:
        config = load_config()

    app = Flask(__name__)
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True

    # Connect to Ray lazily. The connection is used for job dispatch only;
    # ray.nodes() runs in a child process via _safe_ray_query.
    _ray_initialized = {"done": False}

    def _get_ray_head_address() -> str:
        """Read the specific Ray head address from state.json to avoid
        accidentally connecting to or starting an unrelated Ray cluster."""
        try:
            state = Dispatcher.load_state()
            return (state.get("ray_head_address") if state else None) or "auto"
        except Exception:
            return "auto"

    def _ensure_ray():
        if not _ray_initialized["done"]:
            # Check raylet socket BEFORE calling ray.init().
            # ray.init() calls into the C++ core which will abort() the
            # entire process if the socket is missing — no Python exception,
            # just a fatal crash.  Checking first avoids killing the dashboard.
            import pathlib as _pathlib
            ray_tmp = get_ray_tmpdir()
            latest_socket = _pathlib.Path(f"{ray_tmp}/session_latest/sockets/raylet")
            if not latest_socket.exists():
                log.warning("Raylet socket missing (%s) — skipping ray.init", latest_socket)
                return
            try:
                ray.init(address=_get_ray_head_address(), namespace="hive", ignore_reinit_error=True)
                _ray_initialized["done"] = True
            except Exception as e:
                log.warning(f"Could not connect to Ray: {e}")
                # Do NOT set done=True on failure — retry next call

    def _get_ray_data():
        """Get Ray cluster data, using cache if fresh enough."""
        now = time.time()
        if _ray_cache["data"] is not None and now - _ray_cache["ts"] < _RAY_CACHE_TTL:
            return _ray_cache["data"]
        return _safe_ray_query()

    def _check_orphaned(record):
        """Background-only orphan reaper wrapper."""
        reap_orphaned_job(record)

    def _has_actor(job_id: str, record: Optional[JobRecord] = None) -> bool:
        """Check if a Ray actor exists for this job (including retry suffixes).

        Fails closed during Ray control-plane errors to avoid duplicate
        dispatch while actor visibility is uncertain — UNLESS the record
        shows the actor was definitively never created: both
        ``runner_launched=False`` AND ``runner_launch_pending_since``
        is None.  A job with a pending launch (dispatch in progress but
        actor not yet confirmed) must still go through the Ray query to
        avoid GPU overcommit.

        Low-priority jobs are excluded from the short-circuit because
        legacy records (predating ``runner_launched``) default to
        ``runner_launched=False`` via ``from_dict`` even when an actor
        exists.
        """
        if (record is not None
                and not record.runner_launched
                and record.runner_launch_pending_since is None
                and getattr(record, "priority", "normal") != "low"):
            return False
        actor, query_ok = _find_job_actor(job_id)
        if actor is not None:
            return True
        if not query_ok:
            return True
        return False

    def _cleanup_zombie_actors():
        """Kill detached job actors that shouldn't exist.

        Catches: actors for finished jobs, actors with _re/_r suffixes
        from old dispatch attempts, actors for queued jobs whose runner
        never confirmed (stale pending launch), and duplicate actors.
        """
        grace_sec = get_launch_pending_grace_sec()
        now = time.time()
        try:
            named = ray.util.list_named_actors(all_namespaces=True)
            records = {r.job_id: r for r in JobRecord.load_active()}
            live_names: set = set()
            for entry in named:
                name = entry if isinstance(entry, str) else entry.get("name", "")
                if not name.startswith(JOB_ACTOR_PREFIX):
                    continue
                live_names.add(name)
                # Extract base job_id (strip _reN, _rN suffixes)
                suffix = name[len(JOB_ACTOR_PREFIX):]
                base_id = suffix.split("_")[0]
                record = records.get(base_id)
                should_kill = False
                stale_submit_threshold = max(grace_sec * 10, 180.0)
                if record is None:
                    should_kill = True  # no record at all
                elif record.status not in ("running", "queued"):
                    should_kill = True  # job finished
                elif (record.status == "queued"
                      and not record.runner_launched
                      and record.runner_launch_pending_since is not None
                      and (now - record.runner_launch_pending_since) > grace_sec * 4):
                    # Actor exists but runner never confirmed start and
                    # pending launch is well past grace period — zombie
                    # from a failed self-destruct or stale dispatch.
                    should_kill = True
                elif (record.status == "queued"
                      and not record.runner_launched
                      and record.runner_launch_pending_since is None
                      and record.submitted_at
                      and (now - record.submitted_at) > stale_submit_threshold):
                    # Stranded ghost actor: record says "no launch pending
                    # and runner never started", yet the actor name is
                    # still registered in Ray.  This happens when Ray
                    # registers the detached actor name at .remote() time
                    # but never schedules __init__ (e.g. runtime_env setup
                    # wedged).  The record's pending flag may have been
                    # cleared by a later recovery path, leaving the actor
                    # orphaned.  _has_actor sees the registered name and
                    # keeps excluding the job from dispatch forever.
                    should_kill = True
                if not should_kill:
                    # No longer a candidate — forget any prior sighting
                    _zombie_observations.pop(name, None)
                    continue
                # Require two consecutive sightings (>= _ZOMBIE_MIN_AGE_SEC
                # apart) before killing.  This closes the race where
                # reap_orphaned_job just flipped a record to "preempted"
                # and requeue is about to re-launch an actor with the
                # same name — killing it here would tear down the fresh
                # retry.
                first_seen = _zombie_observations.get(name)
                if first_seen is None:
                    _zombie_observations[name] = now
                    continue
                if (now - first_seen) < _ZOMBIE_MIN_AGE_SEC:
                    continue
                kill_exc = None
                try:
                    ray.kill(ray.get_actor(name))
                    log.info(
                        f"Cleaned up zombie actor {name} "
                        f"(observed for {now - first_seen:.0f}s)"
                    )
                except Exception as _kill_exc:
                    kill_exc = _kill_exc
                    log.warning(
                        f"ray.kill failed for zombie {name}: {_kill_exc!r} — "
                        f"resetting record so dispatch can recover"
                    )
                finally:
                    _zombie_observations.pop(name, None)
                    # If the kill was triggered by stale pending-launch
                    # state, clear that now so the record doesn't stay
                    # wedged.
                    if record is not None and record.runner_launch_pending_since is not None:
                        try:
                            JobRecord.clear_runner_launch_pending(base_id)
                        except Exception:
                            pass
                    # Even when ray.kill failed, make the record
                    # dispatchable again.  The short-circuit in
                    # _has_actor (runner_launched=False AND
                    # runner_launch_pending_since=None) overrides the
                    # Ray registry, so _dispatch_queued will pick the
                    # job up on the next tick even if the zombie name
                    # lingers in Ray.
                    if (kill_exc is not None
                            and record is not None
                            and record.status == "queued"
                            and not record.runner_launched):
                        try:
                            JobRecord.clear_runner_launch_pending(base_id)
                        except Exception:
                            pass
            # Forget sightings for actors that no longer exist
            for stale in [n for n in _zombie_observations if n not in live_names]:
                _zombie_observations.pop(stale, None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # API: cluster overview
    # ------------------------------------------------------------------

    @app.route("/api/status")
    def api_status():
        try:
            ray_data = _get_ray_data()
            total_gpus = ray_data["total_gpus"]
            free_gpus = ray_data["free_gpus"]
            raw_nodes = ray_data["nodes"]

            state = Dispatcher.load_state()
            ray_head = state.get("ray_head_address") if state else None

            # Count nodes by type — worker nodes have partition:<name> custom
            # resources (set by ray start --resources in sbatch); head does not.
            num_gpu_nodes = 0
            num_cpu_nodes = 0
            for n in raw_nodes:
                if not n.get("Alive", False):
                    continue
                res = n.get("Resources", {})
                has_partition = any(k.startswith("partition:") for k in res)
                if not has_partition:
                    continue  # head node
                if int(res.get("GPU", 0)) > 0:
                    num_gpu_nodes += 1
                else:
                    num_cpu_nodes += 1

            # GPUs available for jobs = free GPUs as reported by Ray.
            available_gpu = free_gpus

            # CPU-only nodes available = total - running CPU-only jobs
            try:
                running_cpu_jobs = sum(
                    1 for r in JobRecord.load_active()
                    if r.status == "running" and r.num_gpus == 0
                )
            except Exception:
                running_cpu_jobs = 0
            available_cpu = max(num_cpu_nodes - running_cpu_jobs, 0)

            return jsonify({
                "total_gpus": total_gpus,
                "free_gpus": free_gpus,
                "available_gpu": available_gpu,
                "available_cpu": available_cpu,
                "num_gpu_nodes": num_gpu_nodes,
                "num_cpu_nodes": num_cpu_nodes,
                "ray_head": ray_head,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # API: node list
    # ------------------------------------------------------------------

    @app.route("/api/nodes")
    def api_nodes():
        try:
            ray_data = _get_ray_data()
            raw_nodes = ray_data["nodes"]
            nodes = []

            for node in raw_nodes:
                res = node.get("Resources", {})
                gpu_count = int(res.get("GPU", 0))

                # Skip head node — workers have partition:<name> resources
                has_partition = any(k.startswith("partition:") for k in res)
                if not has_partition:
                    continue

                # Extract resource name from Ray custom resource key
                res_names = [k.split(":", 1)[1] for k in res if k.startswith("partition:")]
                resource_name = res_names[0] if res_names else ""

                cpu_count = int(res.get("CPU", 0))

                nodes.append({
                    "name": node.get("NodeName", node.get("NodeManagerAddress", "unknown")),
                    "alive": node.get("Alive", False),
                    "gpus": gpu_count,
                    "cpus": cpu_count,
                    "resource": resource_name,
                    "address": node.get("NodeManagerAddress", ""),
                })

            return jsonify(nodes)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # API: per-node GPU stats (nvidia-smi)
    # ------------------------------------------------------------------

    @app.route("/api/gpu_stats")
    def api_gpu_stats():
        try:
            result = subprocess.run(
                [sys.executable, "-c", _GPU_STATS_SCRIPT],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return jsonify(json.loads(result.stdout))
            return jsonify({})
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # API: per-node CPU stats (/proc/stat + /proc/meminfo)
    # ------------------------------------------------------------------

    @app.route("/api/cpu_stats")
    def api_cpu_stats():
        try:
            result = subprocess.run(
                [sys.executable, "-c", _CPU_STATS_SCRIPT],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return jsonify(json.loads(result.stdout))
            return jsonify({})
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # API: SLURM jobs
    # ------------------------------------------------------------------

    @app.route("/api/slurm")
    def api_slurm():
        try:
            from .config import SLURM_TRACKING_DIR

            # Source of truth: tracking files in .hive/slurm_tracking/.
            # Each file is named by SLURM job ID, contains the resource name.
            # Created by pool.py on submit, by the sbatch requeue trap on
            # auto-requeue, and by the worker script on startup.
            result = []
            job_id_list = []

            if SLURM_TRACKING_DIR.is_dir():
                for entry in SLURM_TRACKING_DIR.iterdir():
                    if entry.is_file():
                        jid = entry.name
                        resource_name = entry.read_text().strip()
                        result.append({
                            "id": jid,
                            "resource_name": resource_name,
                            "state": "UNKNOWN",
                            "node": "",
                            "elapsed": "",
                        })
                        job_id_list.append(jid)

            if not result:
                return jsonify([])

            # Enrich with sacct data
            if job_id_list:
                try:
                    sacct_result = subprocess.run(
                        [
                            "sacct", "-j", ",".join(job_id_list),
                            "--format=JobID,State,NodeList,Elapsed,TimelimitRaw",
                            "--noheader", "--parsable2",
                        ],
                        capture_output=True, text=True, timeout=30,
                    )
                    if sacct_result.returncode == 0:
                        sacct_map = {}
                        for line in sacct_result.stdout.strip().split("\n"):
                            if not line:
                                continue
                            parts = line.split("|")
                            if len(parts) < 4:
                                continue
                            jid = parts[0].strip()
                            # Skip sub-job entries like "12345.batch"
                            if "." in jid:
                                continue
                            sacct_map[jid] = {
                                "state": parts[1].strip(),
                                "node": parts[2].strip(),
                                "elapsed": parts[3].strip(),
                            }

                        for entry in result:
                            info = sacct_map.get(entry["id"])
                            if info:
                                entry["state"] = info["state"]
                                entry["node"] = info["node"]
                                entry["elapsed"] = info["elapsed"]
                except FileNotFoundError:
                    log.debug("sacct not available")
                except subprocess.TimeoutExpired:
                    log.debug("sacct timed out")
                except Exception as e:
                    log.debug(f"sacct error: {e}")

            # Hide non-active jobs — only show RUNNING and PENDING.
            # Also remove stale tracking files so they don't accumulate.
            _inactive = ("CANCELLED", "COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_ME",
                         "NODE_FAIL", "BOOT_FAIL", "DEADLINE")
            active = []
            for r in result:
                if any(r["state"].startswith(s) for s in _inactive):
                    try:
                        (SLURM_TRACKING_DIR / r["id"]).unlink(missing_ok=True)
                    except Exception:
                        pass
                else:
                    active.append(r)
            result = active

            # Enrich pending jobs with estimated start time from squeue
            pending_ids = [r["id"] for r in result if r["state"] == "PENDING"]
            if pending_ids:
                try:
                    sq = subprocess.run(
                        ["squeue", "-j", ",".join(pending_ids),
                         "--format=%i|%S", "--noheader"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if sq.returncode == 0:
                        start_map = {}
                        for line in sq.stdout.strip().split("\n"):
                            if "|" not in line:
                                continue
                            jid, start = line.strip().split("|", 1)
                            jid = jid.strip()
                            start = start.strip()
                            if start and start != "N/A":
                                start_map[jid] = start
                        for entry in result:
                            entry["est_start"] = start_map.get(entry["id"], "")
                except Exception:
                    pass

            # Sort: running on top, then pending by est_start (earliest first)
            state_order = {"RUNNING": 0, "PENDING": 1, "UNKNOWN": 2}
            result.sort(key=lambda r: (
                state_order.get(r["state"], 9),
                r.get("est_start") or "9999",
            ))
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # API: async jobs
    # ------------------------------------------------------------------

    @app.route("/api/jobs")
    def api_jobs():
        try:
            _ensure_ray()
            records = JobRecord.load_all()

            # Build node_id -> IP lookup
            node_ip_map = {}
            try:
                for n in ray.nodes():
                    node_ip_map[n["NodeID"]] = n.get("NodeName", "")
            except Exception:
                pass

            jobs = []
            for r in records:
                d = r.to_dict()
                nid = d.get("node_id", "")
                d["node_name"] = node_ip_map.get(nid, "")
                jobs.append(d)
            # Sort: running/queued first, normal before low-priority, then newest
            # Running/queued first, then failed+completed mixed by time, then other states
            status_order = {"running": 0, "queued": 1, "failed": 2, "completed": 2,
                            "timeout": 2, "preempted": 3, "cancelled": 3}
            priority_order = {"normal": 0, "low": 1}
            jobs.sort(key=lambda j: (
                status_order.get(j["status"], 9),
                priority_order.get(j.get("priority", "normal"), 2),
                -j.get("submitted_at", 0),
            ))
            return jsonify(jobs)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # API: submit a job (remote clients)
    # ------------------------------------------------------------------

    @app.route("/api/submit", methods=["POST"])
    def api_submit():
        _ensure_ray()
        try:
            data = request.get_json(force=True)
            if not data or "command" not in data:
                return jsonify({"error": "Missing 'command' field"}), 400

            command = data["command"]
            num_gpus = data.get("num_gpus", 1)
            num_cpus = data.get("num_cpus", 1)
            memory_mb = data.get("memory_mb", 0)
            workdir = data.get("workdir", config.env.working_dir or os.getcwd())
            resource = data.get("resource") or data.get("partition")
            user = data.get("user", "unknown")
            user = f"{user} (remote)"
            on_complete = data.get("on_complete")
            timeout_sec = data.get("timeout_sec")
            max_retries = data.get("max_retries", 0)
            priority = data.get("priority", "normal")

            dispatcher = Dispatcher(config)
            state = Dispatcher.load_state()
            if state:
                dispatcher._ray_head_address = state.get("ray_head_address")
                if state.get("resources"):
                    dispatcher._resources = [ResourceSpec.from_dict(p) for p in state["resources"]]

            record = dispatcher.submit_async(
                command=command,
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                memory_mb=memory_mb,
                workdir=workdir,
                resource=resource,
                user=user,
                on_complete=on_complete,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                priority=priority,
            )

            return jsonify({"job_id": record.job_id, "status": record.status})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # API: job logs
    # ------------------------------------------------------------------

    @app.route("/api/jobs/<job_id>/logs")
    def api_job_logs(job_id):
        try:
            _ensure_ray()
            record = JobRecord.load(job_id)
            if record is None:
                return jsonify({"error": f"Job '{job_id}' not found"}), 404

            result = {
                "job_id": record.job_id,
                "status": record.status,
                "command": record.command,
                "num_gpus": record.num_gpus,
                "num_cpus": record.num_cpus,
                "resource_name": record.resource,
                "submitted_at": record.submitted_at,
                "started_at": record.started_at,
                "completed_at": record.completed_at,
                "returncode": record.returncode,
                "stdout": record.stdout,
                "stderr": record.stderr,
            }
            if record.user:
                result["user"] = record.user
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
    def api_cancel_job(job_id):
        """Cancel a running or queued job by killing its Ray actor."""
        try:
            _ensure_ray()
            record = JobRecord.load(job_id)
            if record is None:
                return jsonify({"error": f"Job '{job_id}' not found"}), 404

            if record.status not in ("running", "queued"):
                return jsonify({
                    "job_id": job_id,
                    "status": record.status,
                    "message": f"Job already {record.status}",
                })

            if record.status == "queued":
                queued_result, queued_record = JobRecord.cancel_queued_if_not_started(job_id)
                if queued_result == "cancelled":
                    return jsonify({
                        "job_id": job_id,
                        "status": "cancelled",
                        "actor_killed": False,
                    })
                if queued_result == "terminal" and queued_record is not None:
                    return jsonify({
                        "job_id": job_id,
                        "status": queued_record.status,
                        "message": f"Job already {queued_record.status}",
                    })
                record = JobRecord.load(job_id) or queued_record or record

            # Kill the Ray actor if it exists
            prefix = f"{JOB_ACTOR_PREFIX}{record.job_id}"
            killed = False
            needs_ray_verification = (
                record.status == "running" or _queued_job_needs_ray_verification(record)
            )
            if not _ray_initialized.get("done") and needs_ray_verification:
                return jsonify({
                    "error": "Cannot reach Ray to verify job cancellation",
                    "job_id": job_id,
                }), 503
            if _ray_initialized.get("done") and needs_ray_verification:
                try:
                    for entry in ray.util.list_named_actors(all_namespaces=True):
                        name = entry if isinstance(entry, str) else entry.get("name", "")
                        if name.startswith(prefix):
                            try:
                                actor = ray.get_actor(name)
                            except ValueError:
                                log.info(
                                    "Job %s: actor %s disappeared during cancel lookup",
                                    job_id, name,
                                )
                                continue
                            ray.kill(actor)
                            killed = True
                            log.info("Cancelled job %s: killed actor %s", job_id, name)
                except Exception as e:
                    log.warning("Error killing actor for job %s: %s", job_id, e)
                    return jsonify({
                        "error": f"Cannot inspect Ray actors for job '{job_id}': {e}",
                        "job_id": job_id,
                    }), 503

            # Re-read from disk: the actor's finally block may have saved
            # a terminal status (completed/failed) between our initial load
            # and now.
            fresh = JobRecord.load(job_id)
            if fresh and fresh.status in ("completed", "failed", "cancelled", "timeout", "preempted"):
                return jsonify({
                    "job_id": job_id,
                    "status": fresh.status,
                    "message": f"Job finished as {fresh.status} before cancel took effect",
                })

            if record.status == "running" and not killed:
                return jsonify({
                    "error": (
                        f"Job '{job_id}' is still marked running, but its Ray actor "
                        "could not be stopped. Refusing to mark it cancelled."
                    ),
                    "job_id": job_id,
                }), 409

            cancelled_result, cancelled_record = JobRecord.mark_cancelled(job_id)
            if cancelled_result == "terminal" and cancelled_record is not None:
                return jsonify({
                    "job_id": job_id,
                    "status": cancelled_record.status,
                    "message": f"Job finished as {cancelled_record.status} before cancel took effect",
                })
            if cancelled_result == "missing":
                return jsonify({
                    "error": f"Job '{job_id}' disappeared during cancellation",
                    "job_id": job_id,
                }), 404

            return jsonify({
                "job_id": job_id,
                "status": "cancelled",
                "actor_killed": killed,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # HTML dashboard
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    # ── Restart preempted low-priority jobs when GPUs are idle ──────
    def _restart_preempted_jobs():
        """Re-queue preempted jobs (any priority) if GPUs are free."""
        from .config import JOBS_DIR
        records = JobRecord.load_active()

        # Count free GPUs
        running_gpus = sum(
            getattr(r, "num_gpus", 1) for r in records if r.status == "running"
        )
        data = _get_ray_data() or {}
        total_gpus = data.get("total_gpus", 0)
        free_gpus = total_gpus - running_gpus
        if free_gpus <= 0:
            return

        # Check if normal-pri jobs are queued (low-pri preempted should yield)
        normal_queued = any(
            r.status == "queued" and getattr(r, "priority", "normal") != "low"
            for r in records
        )

        # Mark preempted jobs as queued so _dispatch_queued picks them up
        for r in records:
            if free_gpus <= 0:
                break
            if r.status != "preempted":
                continue
            # Low-pri preempted yields to queued normal-pri jobs
            if getattr(r, "priority", "normal") == "low" and normal_queued:
                continue

            result, updated = JobRecord.requeue_preempted(r.job_id)
            if result != "queued" or updated is None:
                log.warning("Could not re-queue preempted job %s: %s", r.job_id, result)
                continue
            free_gpus -= r.num_gpus
            log.info("Re-queued preempted low-pri job %s", r.job_id)

    def _dispatch_queued():
        """Dispatch queued jobs that have no actor.

        Called by the health loop. Creates one actor per free GPU for
        the oldest queued jobs. Low-pri jobs are queued without actors
        at submit time and dispatched here. Preempts running low-pri
        jobs if normal-pri jobs are waiting.
        """
        records = JobRecord.load_active()

        def _parse_resource_names(resource_value: str | None) -> frozenset[str]:
            return frozenset(
                r.strip() for r in (resource_value or "").split(",") if r.strip()
            )

        # Count GPUs in use (running + queued-with-actor)
        used_gpus = 0
        for r in records:
            if r.status == "running":
                used_gpus += r.num_gpus
            elif r.status == "queued" and _has_actor(r.job_id, r):
                used_gpus += r.num_gpus

        data = _get_ray_data() or {}
        total_gpus = data.get("total_gpus", 0)
        free_gpus = total_gpus - used_gpus
        node_resource_names = {}
        node_free_gpus = {}
        resource_node_ids = {}
        for node in data.get("nodes", []):
            if not node.get("Alive", False):
                continue
            resources = node.get("Resources", {})
            gpu_count = int(resources.get("GPU", 0))
            if gpu_count <= 0:
                continue
            res_names = frozenset(
                k.split(":", 1)[1] for k in resources if k.startswith("partition:")
            )
            if not res_names:
                continue
            node_id = node.get("NodeID", "")
            if is_node_draining(node_id):
                continue  # don't dispatch to nodes draining before SLURM walltime
            node_resource_names[node_id] = res_names
            if len(res_names) == 1:
                res_name = next(iter(res_names))
                node_free_gpus[node_id] = gpu_count
                resource_node_ids.setdefault(res_name, []).append(node_id)

        def _record_resource_names(record: JobRecord) -> frozenset[str]:
            if record.node_id and record.node_id in node_resource_names:
                return node_resource_names[record.node_id]
            if getattr(record, "dispatch_resource", None):
                return frozenset([record.dispatch_resource])
            return _parse_resource_names(record.resource)

        def _candidate_resource_names(record: JobRecord) -> frozenset[str]:
            return frozenset(
                name for name in (_record_resource_names(record) or all_dispatch_resources)
                if name in all_dispatch_resources
            )

        all_dispatch_resources = frozenset(resource_node_ids.keys())
        active_pool_reservations: list[tuple[frozenset[str], int]] = []

        for r in records:
            if r.status != "running" and not (r.status == "queued" and _has_actor(r.job_id, r)):
                continue
            if r.node_id and r.node_id in node_free_gpus:
                node_free_gpus[r.node_id] = max(0, node_free_gpus.get(r.node_id, 0) - r.num_gpus)
                continue
            res_names = _candidate_resource_names(r)
            if res_names:
                active_pool_reservations.append((res_names, r.num_gpus))

        current_pass_reservations: list[tuple[frozenset[str], int]] = []

        def _resource_node_caps_after_reservations(
            res_name: str,
            extra_reservations: Optional[list[tuple[frozenset[str], int]]] = None,
        ) -> list[int]:
            caps = sorted(
                (node_free_gpus.get(node_id, 0) for node_id in resource_node_ids.get(res_name, [])),
                reverse=True,
            )
            if not caps:
                return []

            one_gpu_reservations = 0
            larger_reservations: list[int] = []
            for reservations in (active_pool_reservations, current_pass_reservations, extra_reservations or []):
                for res_names, num_gpus in reservations:
                    if not res_names or res_name in res_names:
                        if num_gpus <= 1:
                            one_gpu_reservations += num_gpus
                        else:
                            larger_reservations.append(num_gpus)

            if not larger_reservations:
                if sum(caps) < one_gpu_reservations:
                    return []
                residual = sorted(caps)
                remaining_ones = one_gpu_reservations
                for idx, cap in enumerate(residual):
                    if remaining_ones <= 0:
                        break
                    used = min(cap, remaining_ones)
                    residual[idx] = cap - used
                    remaining_ones -= used
                residual.sort(reverse=True)
                return residual

            max_cap = max(caps)
            initial_hist = [0] * (max_cap + 1)
            for cap in caps:
                initial_hist[cap] += 1

            states = {tuple(initial_hist)}
            for num_gpus in sorted(larger_reservations, reverse=True):
                next_states = set()
                for hist in states:
                    for cap in range(num_gpus, max_cap + 1):
                        if hist[cap] <= 0:
                            continue
                        next_hist = list(hist)
                        next_hist[cap] -= 1
                        next_hist[cap - num_gpus] += 1
                        next_states.add(tuple(next_hist))
                if not next_states:
                    return []
                states = next_states

            best_caps: Optional[list[int]] = None
            for hist in states:
                residual = []
                total_free = 0
                for cap, count in enumerate(hist):
                    if cap <= 0 or count <= 0:
                        continue
                    residual.extend([cap] * count)
                    total_free += cap * count
                if total_free < one_gpu_reservations:
                    continue
                residual.sort()
                remaining_ones = one_gpu_reservations
                for idx, cap in enumerate(residual):
                    if remaining_ones <= 0:
                        break
                    used = min(cap, remaining_ones)
                    residual[idx] = cap - used
                    remaining_ones -= used
                candidate = sorted(residual, reverse=True)
                if best_caps is None or candidate > best_caps:
                    best_caps = candidate

            return best_caps or []

        def _resource_can_fit(
            res_name: str,
            num_gpus: int,
            extra_reservations: Optional[list[tuple[frozenset[str], int]]] = None,
        ) -> bool:
            return any(cap >= num_gpus for cap in _resource_node_caps_after_reservations(res_name, extra_reservations))

        def _resource_max_free_node(res_name: str) -> int:
            return max(_resource_node_caps_after_reservations(res_name), default=0)

        def _resource_total_free(res_name: str) -> int:
            return sum(_resource_node_caps_after_reservations(res_name))

        def _feasible_resources(
            record: JobRecord,
            extra_reservations: Optional[list[tuple[frozenset[str], int]]] = None,
        ) -> frozenset[str]:
            return frozenset(
                res_name for res_name in _candidate_resource_names(record)
                if _resource_can_fit(res_name, record.num_gpus, extra_reservations)
            )

        def _pick_dispatch_resource(record: JobRecord, remaining_candidates: list[JobRecord]) -> Optional[str]:
            if getattr(record, "dispatch_resource", None):
                return record.dispatch_resource

            res_names = _parse_resource_names(record.resource)
            if len(res_names) == 1:
                return next(iter(res_names))

            if not resource_node_ids:
                return None

            candidate_resources = sorted(_feasible_resources(record))
            if not candidate_resources:
                return None

            def _later_impact(res_name: str) -> tuple[int, int]:
                hypothetical = [(frozenset([res_name]), record.num_gpus)]
                blocked = 0
                constrained = 0
                for later in remaining_candidates:
                    feasible_before = _feasible_resources(later)
                    if not feasible_before:
                        continue
                    feasible_after = _feasible_resources(later, hypothetical)
                    if not feasible_after:
                        blocked += later.num_gpus
                    elif len(feasible_before) > 1 and len(feasible_after) == 1:
                        constrained += later.num_gpus
                return blocked, constrained

            best_resource = min(
                candidate_resources,
                key=lambda res_name: (
                    *_later_impact(res_name),
                    -_resource_max_free_node(res_name),
                    -_resource_total_free(res_name),
                    res_name,
                ),
            )
            return best_resource

        def _can_dispatch_candidate(record: JobRecord) -> bool:
            if free_gpus < record.num_gpus:
                return False

            if not resource_node_ids:
                return True

            candidate_resources = _candidate_resource_names(record)
            if not candidate_resources:
                return False
            for res_name in candidate_resources:
                if _resource_can_fit(res_name, record.num_gpus):
                    return True
            return False

        # If no free GPUs but normal-pri GPU jobs are queued, preempt low-pri running jobs
        if free_gpus <= 0:
            normal_queued = [
                r for r in records
                if r.status == "queued" and not _has_actor(r.job_id, r)
                and getattr(r, "priority", "normal") != "low"
                and r.num_gpus > 0
            ]
            if normal_queued:
                # Find low-pri running jobs to preempt — free enough for the first normal-pri job
                gpus_needed = normal_queued[0].num_gpus
                low_pri_running = [
                    r for r in records
                    if r.status == "running" and getattr(r, "priority", "normal") == "low"
                ]
                for lp in low_pri_running:
                    if free_gpus >= gpus_needed:
                        break
                    # Try to kill the actor; if it's already gone (e.g. after
                    # Ray restart), just mark the job as cancelled directly.
                    prefix = f"{JOB_ACTOR_PREFIX}{lp.job_id}"
                    actor_found = False
                    try:
                        for entry in ray.util.list_named_actors(all_namespaces=True):
                            name = entry if isinstance(entry, str) else entry.get("name", "")
                            if name.startswith(prefix):
                                ray.kill(ray.get_actor(name))
                                actor_found = True
                                break
                    except Exception as e:
                        log.warning("Error killing actor for low-pri job %s: %s", lp.job_id, e)
                    # Whether we killed the actor or it was already gone,
                    # mark the job as preempted so the GPU is freed.
                    # attempt is bumped in requeue_preempted (single place of truth).
                    lp.status = "preempted"
                    lp.started_at = None
                    lp.completed_at = None
                    lp.returncode = None
                    buffer_job_stderr_message(lp.job_id, "[hive] Preempted for normal-priority job")
                    lp.save_meta()
                    free_gpus += lp.num_gpus
                    log.info("Preempted low-pri job %s (actor_found=%s) to free GPUs for normal-pri",
                             lp.job_id, actor_found)

        # Find queued jobs without actors, normal-pri first, then oldest.
        # If ANY normal-pri jobs are queued, skip low-pri entirely —
        # otherwise low-pri re-fills freed GPUs every cycle, forcing
        # normal-pri to wait for the next preemption round (60s).
        all_queued = [
            r for r in records
            if r.status == "queued" and not _has_actor(r.job_id, r)
        ]
        has_normal_queued = any(
            getattr(r, "priority", "normal") != "low" for r in all_queued
        )
        if has_normal_queued:
            candidates = [r for r in all_queued if getattr(r, "priority", "normal") != "low"]
        else:
            candidates = all_queued
        candidates.sort(key=lambda r: r.submitted_at)

        # Split into GPU and CPU-only candidates
        gpu_candidates = [r for r in candidates if r.num_gpus > 0]
        cpu_candidates = [r for r in candidates if r.num_gpus == 0]

        # ── GPU job dispatch ──
        for idx, r in enumerate(gpu_candidates):
            if free_gpus <= 0:
                break
            if not _can_dispatch_candidate(r):
                continue

            workdir = config.env.working_dir or os.getcwd()
            try:
                dispatch_resource = _pick_dispatch_resource(r, gpu_candidates[idx + 1 :])
                log.debug("dispatch %s: gpus=%s resource=%s free=%s",
                          r.job_id[:8], r.num_gpus, dispatch_resource, free_gpus)
                launch_result, updated_record = launch_job_runner(
                    r,
                    workdir=workdir,
                    dispatch_resource=dispatch_resource,
                )
                log.debug("dispatch %s: result=%s", r.job_id[:8], launch_result)
                if launch_result in ("launched", "already_launched", "launch_pending"):
                    reserved_record = updated_record or r
                    current_pass_reservations.append((_record_resource_names(reserved_record), r.num_gpus))
                    free_gpus -= r.num_gpus
                    if launch_result == "launch_pending":
                        log.info("Queued job %s still has a recent pending launch", r.job_id)
                    else:
                        log.info("Dispatched queued job %s (%s)", r.job_id, launch_result)
                elif launch_result not in ("terminal", "unexpected", "missing"):
                    log.warning("Dispatch skipped for %s: %s", r.job_id, launch_result)
            except Exception as e:
                log.warning("Dispatch failed for %s: %s", r.job_id, e, exc_info=True)

        # ── CPU-only job dispatch ──
        # CPU-only nodes are invisible to the GPU capacity system, so
        # we handle them separately: count available CPU nodes per
        # resource, subtract running CPU jobs, dispatch one per free slot.
        log.debug("cpu dispatch: %d candidate(s)", len(cpu_candidates))
        if cpu_candidates:
            cpu_node_counts: dict[str, int] = {}
            for node in data.get("nodes", []):
                if not node.get("Alive", False):
                    continue
                if is_node_draining(node.get("NodeID", "")):
                    continue
                resources = node.get("Resources", {})
                if int(resources.get("GPU", 0)) > 0:
                    continue
                for k in resources:
                    if k.startswith("partition:"):
                        res_name = k.split(":", 1)[1]
                        cpu_node_counts[res_name] = cpu_node_counts.get(res_name, 0) + 1

            cpu_used: dict[str, int] = {}
            for r in records:
                if r.num_gpus == 0 and (r.status == "running" or (r.status == "queued" and _has_actor(r.job_id, r))):
                    for res_name in _parse_resource_names(r.resource):
                        cpu_used[res_name] = cpu_used.get(res_name, 0) + 1

            log.debug("cpu dispatch: nodes=%s used=%s", cpu_node_counts, cpu_used)
            workdir = config.env.working_dir or os.getcwd()
            for r in cpu_candidates:
                res_names = _parse_resource_names(r.resource)
                dispatch_res = None
                for res_name in res_names:
                    available = cpu_node_counts.get(res_name, 0)
                    used = cpu_used.get(res_name, 0)
                    if available > used:
                        dispatch_res = res_name
                        break
                if dispatch_res is None:
                    log.debug("cpu dispatch %s: no free node for %s", r.job_id[:8], res_names)
                    continue
                try:
                    launch_result, updated_record = launch_job_runner(
                        r, workdir=workdir, dispatch_resource=dispatch_res,
                    )
                    log.debug("cpu dispatch %s: resource=%s result=%s",
                              r.job_id[:8], dispatch_res, launch_result)
                    if launch_result in ("launched", "already_launched", "launch_pending"):
                        cpu_used[dispatch_res] = cpu_used.get(dispatch_res, 0) + 1
                        log.info("Dispatched CPU-only job %s to %s (%s)", r.job_id, dispatch_res, launch_result)
                    elif launch_result not in ("terminal", "unexpected", "missing"):
                        log.warning("CPU dispatch skipped for %s: %s", r.job_id, launch_result)
                except Exception as e:
                    log.warning("CPU dispatch failed for %s: %s", r.job_id, e)

    # ── Background health-check thread ──────────────────────────────
    # Runs every 60s regardless of whether anyone is viewing the dashboard:
    # reaps orphaned jobs whose worker nodes died (the runner actor's own
    # poll loop can miss node death), dispatches queued jobs, restarts
    # preempted jobs, and prunes stale drain markers.
    def _health_loop():
        import glob
        import pathlib

        # On startup, kill stale job-runner actors left by a previous
        # dashboard process.  These detached actors survive process
        # restarts and carry old code — they must be replaced.
        try:
            _ensure_ray()
            if _ray_initialized["done"]:
                named = ray.util.list_named_actors(all_namespaces=True)
                for entry in named:
                    name = entry if isinstance(entry, str) else entry.get("name", "")
                    if name.startswith(JOB_ACTOR_PREFIX):
                        try:
                            actor = ray.get_actor(name)
                            ray.kill(actor)
                            log.info("Killed stale job actor on startup: %s", name)
                        except Exception:
                            pass
        except Exception as e:
            log.warning("Stale actor cleanup failed: %s", e)

        while True:
            time.sleep(60)

            # Step 0: keep the Ray session directory fresh. Some systems run a
            # periodic cleaner over temp directories that deletes files not
            # touched for several days, which removes Ray's sockets from
            # under a long-lived head.
            try:
                ray_tmp = get_ray_tmpdir()
                for d_path in glob.glob(f"{ray_tmp}/session_*/sockets"):
                    os.utime(d_path, None)
                    for f in glob.glob(d_path + "/*"):
                        os.utime(f, None)
                    os.utime(os.path.dirname(d_path), None)
                os.utime(ray_tmp, None)
                state = Dispatcher.load_state()
                if state and state.get("ray_head_address"):
                    latest = pathlib.Path(f"{ray_tmp}/session_latest/sockets/raylet")
                    if not latest.exists():
                        log.warning(
                            "Ray head socket %s is missing; the Ray head is "
                            "probably down. Run `hive restart` to bring it back "
                            "(SLURM workers reconnect on their own).", latest,
                        )
            except Exception:
                pass

            # Step 1: Connect to Ray — retry a few times before skipping
            _ensure_ray()
            if not _ray_initialized["done"]:
                for _retry in range(3):
                    time.sleep(5)
                    _ensure_ray()
                    if _ray_initialized["done"]:
                        break
                if not _ray_initialized["done"]:
                    continue

            # Step 2: Reap orphaned jobs
            try:
                records = JobRecord.load_active()
                for r in records:
                    _check_orphaned(r)
            except Exception as e:
                log.warning(f"Orphan check failed: {e}")

            # Step 2b: Unstick queued jobs whose runner died.
            # When a node drains mid-launch, the _AsyncJobRunner actor
            # gets killed by Ray but the job record still has
            # runner_launched=True / status=queued.  Detect this by
            # checking if the actor is actually gone, then reset the
            # record so _dispatch_queued can re-launch it.
            try:
                for r in records:
                    if (r.status == "queued"
                            and r.runner_launched
                            and r.runner_launch_pending_since is None):
                        actor, query_ok = _find_job_actor(r.job_id)
                        if actor is None and query_ok:
                            # Atomic reload+reset under the meta lock so we
                            # never overwrite a state that the actor wrote
                            # after our load_active() snapshot.
                            lock_record = JobRecord(
                                job_id=r.job_id, command="",
                                num_gpus=1, status="unknown",
                            )
                            with lock_record._meta_lock():
                                fresh = JobRecord.load(r.job_id)
                                if (fresh is not None
                                        and fresh.status == "queued"
                                        and fresh.runner_launched):
                                    log.info(
                                        "Job %s: runner_launched=True but actor gone "
                                        "— resetting to dispatchable", r.job_id,
                                    )
                                    fresh.runner_launched = False
                                    fresh.dispatch_resource = None
                                    fresh._write_meta_unlocked()
            except Exception as e:
                log.warning(f"Stuck-runner check failed: {e}")

            # Step 3: Cleanup zombie actors
            try:
                _cleanup_zombie_actors()
            except Exception as e:
                log.warning(f"Zombie cleanup failed: {e}")

            # Step 4: Dispatch queued jobs (first pass)
            try:
                _dispatch_queued()
            except Exception as e:
                log.warning(f"Dispatch failed: {e}", exc_info=True)

            # Step 5: Restart preempted jobs
            try:
                _restart_preempted_jobs()
            except Exception as e:
                log.warning(f"Preempted restart failed: {e}")

            # Step 6: Dispatch again, so jobs that step 5 just moved back to
            # the queue start in this tick instead of the next one.
            try:
                _dispatch_queued()
            except Exception as e:
                log.warning(f"Second dispatch pass failed: {e}")

            # Step 7: Drop drain markers of nodes that have left the cluster.
            # Never prune on a failed or empty snapshot: that would clear
            # every marker and let jobs be relaunched onto draining nodes.
            try:
                nodes = ray.nodes()
                if nodes:
                    alive = {
                        n["NodeID"] for n in nodes
                        if n.get("Alive", False) and n.get("NodeID")
                    }
                    prune_stale_draining_nodes(alive)
            except Exception as e:
                log.warning(f"Drain-marker prune skipped: {e}")

    if start_health_thread:
        _health_thread = threading.Thread(target=_health_loop, daemon=True)
        _health_thread.start()

    return app


# ---------------------------------------------------------------------------
# CLI entry point — ``python -m hive.dashboard --port 8080``
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    parser = argparse.ArgumentParser(description="Hive dashboard server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, use_reloader=False)

# ---------------------------------------------------------------------------
# Dashboard HTML (inline CSS + JS, no external dependencies)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hive Dashboard</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg: #f1f5f9;
    --card: #ffffff;
    --border: #e2e8f0;
    --accent: #3b82f6;
    --success: #16a34a;
    --warning: #ca8a04;
    --error: #dc2626;
    --pending: #7c3aed;
    --text: #0f172a;
    --text2: #475569;
    --text3: #94a3b8;
    --radius: 10px;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
}

body {
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.5;
    min-height: 100vh;
}

/* Header */
.header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    background: var(--card);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 3px;
    color: var(--accent);
    font-family: 'Courier New', monospace;
}

.logo-sub {
    font-size: 12px;
    color: var(--text2);
    font-weight: 400;
}

.header-right {
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 13px;
    color: var(--text2);
}

.refresh-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

.pulse-dot.error {
    background: var(--error);
    animation: none;
}

/* Error banner */
.error-banner {
    display: none;
    padding: 10px 24px;
    background: rgba(239, 68, 68, 0.15);
    border-bottom: 1px solid rgba(239, 68, 68, 0.3);
    color: var(--error);
    font-size: 13px;
}

.error-banner.visible { display: block; }

/* Main */
.main {
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px;
}

/* Utilization bars row */
.util-bars-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
}

/* GPU utilization bar */
.gpu-bar-container {
}

.gpu-bar-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 8px;
}

.gpu-bar-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text2);
}

.gpu-bar-label {
    font-size: 13px;
    color: var(--text2);
    font-family: 'Courier New', monospace;
}

.gpu-bar-track {
    width: 100%;
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
}

.gpu-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), #60a5fa);
    transition: width 0.6s ease;
}

.cpu-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #22c55e, #4ade80);
    transition: width 0.6s ease;
}

/* Overview cards */
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: var(--shadow);
}

.card-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text2);
    margin-bottom: 8px;
}

.card-value {
    font-size: 32px;
    font-weight: 700;
    font-family: 'Courier New', monospace;
    color: var(--text);
    line-height: 1.1;
}

.card-value-secondary {
    font-size: 20px;
    font-weight: 600;
    color: var(--text2);
}

.card-value .tag {
    font-size: 12px;
    font-weight: 400;
    color: var(--text3);
    vertical-align: middle;
}

.card-sub {
    font-size: 12px;
    color: var(--text3);
    margin-top: 4px;
}

/* Sections */
.section {
    margin-bottom: 32px;
}

.section-header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 12px;
}

.section-title {
    font-size: 15px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text2);
}

.section-count {
    font-size: 13px;
    color: var(--text3);
    font-family: 'Courier New', monospace;
}

/* Tables */
.table-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
}

table {
    width: 100%;
    border-collapse: collapse;
}

thead th {
    text-align: left;
    padding: 12px 16px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text3);
    border-bottom: 1px solid var(--border);
    background: rgba(0,0,0,0.02);
}

tbody td {
    padding: 10px 16px;
    font-size: 13px;
    border-bottom: 1px solid var(--border);
    font-family: 'Courier New', monospace;
}

tbody tr:last-child td {
    border-bottom: none;
}

tbody tr:hover {
    background: rgba(59, 130, 246, 0.06);
}

/* Badges */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-running { background: rgba(34,197,94,0.15); color: var(--success); }
.badge-pending { background: rgba(139,92,246,0.15); color: var(--pending); }
.badge-completed { background: rgba(59,130,246,0.15); color: var(--accent); }
.badge-failed { background: rgba(239,68,68,0.15); color: var(--error); }
.badge-timeout { background: rgba(239,68,68,0.15); color: var(--error); }
.badge-cancelled { background: rgba(234,179,8,0.15); color: var(--warning); }
.badge-preempted { background: rgba(234,179,8,0.15); color: var(--warning); }
.badge-queued { background: rgba(139,92,246,0.15); color: var(--pending); }
.badge-unknown { background: rgba(100,116,139,0.15); color: var(--text3); }

.badge-partition {
    background: rgba(59,130,246,0.12);
    color: #2563eb;
    border: 1px solid rgba(59,130,246,0.3);
}

/* Status dot */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}

.status-dot.alive { background: var(--success); }
.status-dot.dead { background: var(--error); }

/* Empty state */
.empty-state {
    padding: 32px;
    text-align: center;
    color: var(--text3);
    font-size: 13px;
}

/* Job detail (expandable stdout/stderr) */
.job-row { cursor: pointer; }
.job-row:hover { background: rgba(59,130,246,0.06); }

.job-detail {
    display: none;
}

.job-detail.open {
    display: table-row;
}

.job-detail td {
    padding: 0;
    background: #f8fafc;
}

.job-output {
    padding: 12px 16px;
    max-height: 600px;
    overflow-y: auto;
}

.job-output-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text3);
    margin-bottom: 4px;
    font-family: system-ui, sans-serif;
}

.job-output pre {
    font-size: 12px;
    color: var(--text2);
    white-space: pre-wrap;
    word-break: break-all;
    margin-bottom: 12px;
    max-height: 250px;
    overflow-y: auto;
    background: #f1f5f9;
    padding: 8px;
    border-radius: 4px;
}

/* Truncated command */
.cmd-truncated {
    max-width: 320px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* GPU mini utilization bars */
.gpu-bars {
    display: flex;
    flex-direction: column;
    gap: 3px;
}

.gpu-mini {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-family: 'Courier New', monospace;
}

.gpu-mini-idx {
    color: var(--text3);
    min-width: 14px;
}

.gpu-mini-track {
    width: 80px;
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    overflow: hidden;
    flex-shrink: 0;
}

.gpu-mini-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

.gpu-mini-pct {
    min-width: 30px;
    text-align: right;
    font-weight: 600;
    color: var(--text);
}

.gpu-mini-mem {
    color: var(--text3);
    font-size: 10px;
}

.gpu-mini-temp {
    color: var(--text3);
    font-size: 10px;
}

/* Responsive */
@media (max-width: 768px) {
    .main { padding: 12px; }
    .cards { grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .card { padding: 14px; }
    .card-value { font-size: 24px; }
    .card-value-secondary { font-size: 16px; }
    .header { padding: 12px 16px; }
    thead th, tbody td { padding: 8px 10px; }
    .cmd-truncated { max-width: 160px; }
}

@media (max-width: 480px) {
    .cards { grid-template-columns: 1fr 1fr; }
}
</style>
</head>
<body>

<div class="header">
    <div class="header-left">
        <div class="logo">HIVE <span class="logo-sub">GPU Cluster</span></div>
    </div>
    <div class="header-right">
        <div class="refresh-indicator">
            <div class="pulse-dot" id="pulseDot"></div>
            <span>Auto-refresh</span>
        </div>
        <span id="lastUpdated">--</span>
    </div>
</div>

<div class="error-banner" id="errorBanner"></div>

<div class="main">

    <!-- Utilization bars -->
    <div class="util-bars-row">
        <div class="gpu-bar-container">
            <div class="gpu-bar-header">
                <span class="gpu-bar-title">GPU Utilization</span>
                <span class="gpu-bar-label" id="gpuBarLabel">-- / -- GPUs</span>
            </div>
            <div class="gpu-bar-track">
                <div class="gpu-bar-fill" id="gpuBarFill" style="width: 0%"></div>
            </div>
        </div>
        <div class="gpu-bar-container">
            <div class="gpu-bar-header">
                <span class="gpu-bar-title">CPU Utilization</span>
                <span class="gpu-bar-label" id="cpuBarLabel">-- / -- CPU nodes</span>
            </div>
            <div class="gpu-bar-track">
                <div class="cpu-bar-fill" id="cpuBarFill" style="width: 0%"></div>
            </div>
        </div>
    </div>

    <!-- Overview cards -->
    <div class="cards">
        <div class="card">
            <div class="card-label">Resources</div>
            <div class="card-value"><span id="totalGpus">--</span> <span class="tag">gpu</span></div>
            <div class="card-value card-value-secondary"><span id="totalCpuNodes">--</span> <span class="tag">cpu node</span></div>
            <div class="card-sub" id="nodeCount">-- nodes</div>
        </div>
        <div class="card">
            <div class="card-label">Available</div>
            <div class="card-value"><span id="availableGpus">--</span> <span class="tag">gpu</span></div>
            <div class="card-value card-value-secondary"><span id="availableCpus">--</span> <span class="tag">cpu node</span></div>
            <div class="card-sub" id="gpuDetail">-- idle</div>
        </div>
        <div class="card">
            <div class="card-label">Active Jobs</div>
            <div class="card-value"><span id="gpuJobs">--</span> <span class="tag">gpu</span></div>
            <div class="card-value card-value-secondary"><span id="cpuJobs">--</span> <span class="tag">cpu</span></div>
            <div class="card-sub" id="jobBreakdown">-- queued</div>
        </div>
    </div>

    <!-- Nodes -->
    <div class="section">
        <div class="section-header">
            <span class="section-title">Nodes</span>
            <span class="section-count" id="nodesCount">(0)</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Node Name</th>
                        <th>Partition</th>
                        <th>Utilization</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="nodesBody">
                    <tr><td colspan="4" class="empty-state">Loading...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- SLURM Jobs -->
    <div class="section">
        <div class="section-header">
            <span class="section-title">SLURM Jobs</span>
            <span class="section-count" id="slurmCount">(0)</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Job ID</th>
                        <th>Resource</th>
                        <th>State</th>
                        <th>Node</th>
                        <th>Elapsed</th>
                        <th>Est. Start</th>
                    </tr>
                </thead>
                <tbody id="slurmBody">
                    <tr><td colspan="6" class="empty-state">Loading...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- Users (only shown when 2+ distinct users have jobs) -->
    <div class="section" id="usersSection" style="display:none">
        <div class="section-header">
            <span class="section-title">Users</span>
            <span class="section-count" id="usersCount">(0)</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>User</th>
                        <th>Running</th>
                        <th>Queued</th>
                        <th>Completed</th>
                        <th>Failed</th>
                        <th>Total</th>
                    </tr>
                </thead>
                <tbody id="usersBody">
                </tbody>
            </table>
        </div>
    </div>

    <!-- Job Queue -->
    <div class="section">
        <div class="section-header">
            <span class="section-title">Job Queue</span>
            <span class="section-count" id="jobsCount">(0)</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Job ID</th>
                        <th>Command</th>
                        <th>Resources</th>
                        <th>Status</th>
                        <th>Submitted</th>
                        <th>Duration</th>
                    </tr>
                </thead>
                <tbody id="jobsBody">
                    <tr><td colspan="6" class="empty-state">Loading...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

</div>

<script>
(function() {
    "use strict";

    // --- EMA smoothing for GPU stats ---
    // Smooths nvidia-smi point-in-time snapshots to avoid jumpy readings.
    var EMA_ALPHA = 0.35;  // higher = more responsive, lower = smoother
    var gpuEma = {};       // key: "nodeName:gpuIdx" -> {util, mem_used, temperature}

    function emaSmooth(key, field, rawValue) {
        var k = key + ":" + field;
        if (!(k in gpuEma)) {
            gpuEma[k] = rawValue;
        } else {
            gpuEma[k] = EMA_ALPHA * rawValue + (1 - EMA_ALPHA) * gpuEma[k];
        }
        return Math.round(gpuEma[k]);
    }

    // --- Utilities ---

    function relativeTime(ts) {
        if (!ts) return "--";
        var now = Date.now() / 1000;
        var diff = now - ts;
        if (diff < 0) diff = 0;
        if (diff < 60) return Math.floor(diff) + "s ago";
        if (diff < 3600) return Math.floor(diff / 60) + "m ago";
        if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
        return Math.floor(diff / 86400) + "d ago";
    }

    function formatDuration(startTs, endTs) {
        if (!startTs) return "--";
        var end = endTs || (Date.now() / 1000);
        var diff = end - startTs;
        if (diff < 0) diff = 0;
        var h = Math.floor(diff / 3600);
        var m = Math.floor((diff % 3600) / 60);
        var s = Math.floor(diff % 60);
        if (h > 0) return h + "h " + m + "m " + s + "s";
        if (m > 0) return m + "m " + s + "s";
        return s + "s";
    }

    function escapeHtml(str) {
        if (!str) return "";
        var d = document.createElement("div");
        d.textContent = str;
        return d.innerHTML;
    }

    function slurmBadgeClass(state) {
        if (!state) return "badge-unknown";
        var s = state.toUpperCase().replace(/\s.*/,"");
        if (s === "RUNNING") return "badge-running";
        if (s === "PENDING") return "badge-pending";
        if (s === "COMPLETED") return "badge-completed";
        if (s === "FAILED") return "badge-failed";
        if (s === "TIMEOUT") return "badge-timeout";
        if (s === "CANCELLED" || s.indexOf("CANCEL") !== -1) return "badge-cancelled";
        return "badge-unknown";
    }

    function gpuUtilColor(pct) {
        if (pct < 40) return "var(--success)";
        if (pct < 75) return "var(--warning)";
        return "var(--error)";
    }

    function jobBadgeClass(status) {
        if (!status) return "badge-unknown";
        var s = status.toLowerCase();
        if (s === "running") return "badge-running";
        if (s === "queued") return "badge-queued";
        if (s === "completed") return "badge-completed";
        if (s === "failed") return "badge-failed";
        if (s === "preempted") return "badge-preempted";
        if (s === "cancelled") return "badge-cancelled";
        if (s === "timeout") return "badge-timeout";
        return "badge-unknown";
    }

    // --- State ---
    var lastStatus = null;
    var lastNodes = null;
    var lastSlurm = null;
    var lastJobs = null;
    var lastJobsFingerprint = "";
    var lastGpuStats = {};
    var lastCpuStats = {};
    var expandedJobs = {};
    var errorCount = 0;

    // --- DOM refs ---
    var $totalGpus = document.getElementById("totalGpus");
    var $totalCpuNodes = document.getElementById("totalCpuNodes");
    var $availableGpus = document.getElementById("availableGpus");
    var $availableCpus = document.getElementById("availableCpus");
    var $nodeCount = document.getElementById("nodeCount");
    var $gpuDetail = document.getElementById("gpuDetail");
    var $gpuJobs = document.getElementById("gpuJobs");
    var $cpuJobs = document.getElementById("cpuJobs");
    var $jobBreakdown = document.getElementById("jobBreakdown");
    var $nodesBody = document.getElementById("nodesBody");
    var $nodesCount = document.getElementById("nodesCount");
    var $slurmBody = document.getElementById("slurmBody");
    var $slurmCount = document.getElementById("slurmCount");
    var $jobsBody = document.getElementById("jobsBody");
    var $jobsCount = document.getElementById("jobsCount");
    var $usersSection = document.getElementById("usersSection");
    var $usersBody = document.getElementById("usersBody");
    var $usersCount = document.getElementById("usersCount");
    var $lastUpdated = document.getElementById("lastUpdated");
    var $gpuBarLabel = document.getElementById("gpuBarLabel");
    var $gpuBarFill = document.getElementById("gpuBarFill");
    var $cpuBarLabel = document.getElementById("cpuBarLabel");
    var $cpuBarFill = document.getElementById("cpuBarFill");
    var $pulseDot = document.getElementById("pulseDot");
    var $errorBanner = document.getElementById("errorBanner");

    // --- Fetchers ---

    function fetchJSON(url, cb) {
        fetch(url)
            .then(function(r) {
                if (!r.ok) throw new Error("HTTP " + r.status);
                return r.json();
            })
            .then(function(data) {
                errorCount = 0;
                $errorBanner.classList.remove("visible");
                $pulseDot.classList.remove("error");
                cb(data);
            })
            .catch(function(err) {
                errorCount++;
                if (errorCount >= 2) {
                    $errorBanner.textContent = "Failed to reach API (" + url + "): " + err.message;
                    $errorBanner.classList.add("visible");
                    $pulseDot.classList.add("error");
                }
            });
    }

    // --- Renderers ---

    function renderStatus(data) {
        if (data.error) return;
        lastStatus = data;

        var totalGpu = data.total_gpus || 0;
        var availGpu = data.available_gpu || 0;
        var availCpu = data.available_cpu || 0;
        var gpuNodes = data.num_gpu_nodes || 0;
        var cpuNodes = data.num_cpu_nodes || 0;
        var freeGpu = data.free_gpus || 0;
        var jobUsed = totalGpu - availGpu;

        $totalGpus.textContent = totalGpu;
        $totalCpuNodes.textContent = cpuNodes;
        var totalNodes = gpuNodes + cpuNodes;
        $nodeCount.textContent = totalNodes + " node" + (totalNodes !== 1 ? "s" : "");
        $availableGpus.textContent = availGpu;
        $availableCpus.textContent = availCpu;
        $gpuDetail.textContent = freeGpu + " idle";

        $gpuBarLabel.textContent = jobUsed + " / " + totalGpu + " GPUs in use by jobs";
        var pct = totalGpu > 0 ? Math.round((jobUsed / totalGpu) * 100) : 0;
        $gpuBarFill.style.width = pct + "%";

        var cpuJobsUsed = cpuNodes - availCpu;
        $cpuBarLabel.textContent = cpuJobsUsed + " / " + cpuNodes + " CPU nodes in use by jobs";
        var cpuPct = cpuNodes > 0 ? Math.round((cpuJobsUsed / cpuNodes) * 100) : 0;
        $cpuBarFill.style.width = cpuPct + "%";
    }

    var jobsDisplayLimit = 20;  // initial number of jobs to show

    function renderJobs(data) {
        if (data.error) return;
        lastJobs = data;

        var queued = 0, running = 0, gpuRunning = 0, cpuRunning = 0, queuedLowPri = 0;
        var fp = "";
        for (var i = 0; i < data.length; i++) {
            var s = data[i].status;
            if (s === "queued") {
                queued++;
                if (data[i].priority === "low") queuedLowPri++;
            }
            if (s === "running") {
                running++;
                if ((data[i].num_gpus || 0) > 0) gpuRunning++;
                else cpuRunning++;
            }
            fp += data[i].job_id + ":" + s + ",";
        }

        // Skip table rebuild if jobs haven't changed (prevents scroll/detail reset)
        var fpFull = fp + "|limit=" + jobsDisplayLimit;
        var jobsChanged = (fpFull !== lastJobsFingerprint);
        lastJobsFingerprint = fpFull;

        $gpuJobs.textContent = gpuRunning;
        $cpuJobs.textContent = cpuRunning;
        var breakdown = queued + " queued";
        if (queuedLowPri > 0) breakdown += " (" + queuedLowPri + " low-pri)";
        $jobBreakdown.textContent = breakdown;
        $jobsCount.textContent = "(" + data.length + ")";

        // Render users section
        renderUsers(data);

        if (!jobsChanged) return;

        // Detect multi-user: show USER column only when >1 distinct user
        var userSet = {};
        for (var u = 0; u < data.length; u++) {
            if (data[u].user) userSet[data[u].user] = true;
        }
        var multiUser = Object.keys(userSet).length > 1;
        var colSpan = multiUser ? 8 : 7;

        // Update table header
        var thead = $jobsBody.parentElement.querySelector("thead tr");
        if (multiUser) {
            thead.innerHTML = '<th>Job ID</th><th>User</th><th>Command</th><th>Resources</th><th>Node</th><th>Status</th><th>Submitted</th><th>Duration</th>';
        } else {
            thead.innerHTML = '<th>Job ID</th><th>Command</th><th>Resources</th><th>Node</th><th>Status</th><th>Submitted</th><th>Duration</th>';
        }

        if (data.length === 0) {
            $jobsBody.innerHTML = '<tr><td colspan="' + colSpan + '" class="empty-state">No jobs submitted</td></tr>';
            return;
        }

        // Render up to jobsDisplayLimit rows (data is already sorted: running/queued first, then newest)
        var html = "";
        var endIdx = Math.min(data.length, jobsDisplayLimit);
        for (var j = 0; j < endIdx; j++) {
            var job = data[j];
            var isOpen = !!expandedJobs[job.job_id];
            var cmdFull = escapeHtml(job.command || "");
            var cmdShort = job.command || "";
            if (cmdShort.length > 60) cmdShort = cmdShort.substring(0, 57) + "...";

            var dur = "--";
            if (job.status === "running" && job.started_at) {
                dur = formatDuration(job.started_at, null);
            } else if (job.completed_at && job.submitted_at) {
                dur = formatDuration(job.submitted_at, job.completed_at);
            }

            html += '<tr class="job-row" data-jobid="' + escapeHtml(job.job_id) + '">';
            html += '<td>' + escapeHtml(job.job_id) + '</td>';
            if (multiUser) {
                html += '<td>' + escapeHtml(job.user || "--") + '</td>';
            }
            html += '<td><span class="cmd-truncated" title="' + cmdFull + '">' + escapeHtml(cmdShort) + '</span></td>';
            var ngpu = (job.num_gpus != null) ? job.num_gpus : 1;
            html += '<td>' + (ngpu > 0 ? ngpu + ' GPU' : (job.num_cpus || 1) + ' CPU') + '</td>';
            html += '<td>' + escapeHtml(job.node_name || "--") + '</td>';
            var statusHtml = '<span class="badge ' + jobBadgeClass(job.status) + '">' + escapeHtml(job.status) + '</span>';
            if (job.priority === "low") statusHtml += ' <span style="font-size:10px;color:var(--text3)">(low)</span>';
            html += '<td>' + statusHtml + '</td>';
            html += '<td>' + relativeTime(job.submitted_at) + '</td>';
            html += '<td>' + dur + '</td>';
            html += '</tr>';

            // Detail row (content loaded on click)
            html += '<tr class="job-detail' + (isOpen ? ' open' : '') + '" data-detailfor="' + escapeHtml(job.job_id) + '">';
            html += '<td colspan="' + colSpan + '"><div class="job-output">';
            html += '<div class="job-output-label">Loading...</div>';
            html += '</div></td></tr>';
        }

        // "Show more" button if there are hidden jobs
        var hidden = data.length - jobsDisplayLimit;
        if (hidden > 0) {
            html += '<tr id="showMoreRow"><td colspan="' + colSpan + '" style="text-align:center;padding:8px">';
            html += '<button id="showMoreBtn" style="background:var(--accent);color:#fff;border:none;padding:6px 18px;border-radius:4px;cursor:pointer;font-size:13px">';
            html += 'Show more (' + hidden + ' older jobs)</button></td></tr>';
        }

        $jobsBody.innerHTML = html;

        function lastLines(text, n) {
            if (!text) return "";
            var lines = text.split("\n");
            return lines.slice(-n).join("\n");
        }

        // Load logs into an expanded detail row.  Used both by the
        // click handler (first open) AND by the auto-refresh path
        // below, because rebuilding $jobsBody.innerHTML wipes any
        // previously-fetched log body and resets data-loaded.  Without
        // this re-fetch, an open row would flash to "Loading..." on
        // every refresh tick.
        function loadJobLogs(jid) {
            var detail = $jobsBody.querySelector('[data-detailfor="' + jid + '"]');
            if (!detail) return;
            detail.setAttribute("data-loaded", "1");
            fetch("/api/jobs/" + jid + "/logs")
                .then(function(r) { return r.json(); })
                .then(function(d) {
                    var out = detail.querySelector(".job-output");
                    if (!out) return;
                    var h = "";
                    if (d.stdout) {
                        h += '<div class="job-output-label">stdout (last 30 lines)</div>';
                        h += '<pre>' + escapeHtml(lastLines(d.stdout, 30)) + '</pre>';
                    }
                    if (d.stderr) {
                        h += '<div class="job-output-label">stderr (last 30 lines)</div>';
                        h += '<pre>' + escapeHtml(lastLines(d.stderr, 30)) + '</pre>';
                    }
                    if (!d.stdout && !d.stderr) {
                        h += '<div class="job-output-label">No output available</div>';
                    }
                    if (d.returncode !== null && d.returncode !== undefined) {
                        h += '<div class="job-output-label">Exit code: ' + d.returncode + '</div>';
                    }
                    out.innerHTML = h;
                })
                .catch(function() {
                    var out = detail.querySelector(".job-output");
                    if (out) out.innerHTML = '<div class="job-output-label">Failed to load logs</div>';
                });
        }

        // Attach click handlers for job rows
        var rows = $jobsBody.querySelectorAll(".job-row");
        for (var k = 0; k < rows.length; k++) {
            rows[k].addEventListener("click", function() {
                var jid = this.getAttribute("data-jobid");
                expandedJobs[jid] = !expandedJobs[jid];
                var detail = $jobsBody.querySelector('[data-detailfor="' + jid + '"]');
                if (detail) {
                    detail.classList.toggle("open");
                    if (detail.classList.contains("open") && !detail.getAttribute("data-loaded")) {
                        loadJobLogs(jid);
                    }
                }
            });
        }

        // Re-populate any row that was already expanded before the
        // rebuild — the innerHTML reset cleared its log body back to
        // "Loading...".
        for (var jid in expandedJobs) {
            if (expandedJobs[jid]) {
                loadJobLogs(jid);
            }
        }

        // "Show more" handler
        var showMoreBtn = document.getElementById("showMoreBtn");
        if (showMoreBtn) {
            showMoreBtn.addEventListener("click", function(e) {
                e.stopPropagation();
                jobsDisplayLimit += 50;
                renderJobs(lastJobs);
            });
        }
    }

    function renderUsers(data) {
        // Aggregate per-user stats from job records
        var users = {};
        for (var i = 0; i < data.length; i++) {
            var job = data[i];
            var u = job.user || "--";
            if (!users[u]) users[u] = {running: 0, queued: 0, completed: 0, failed: 0, total: 0};
            users[u].total++;
            if (job.status === "running") users[u].running++;
            else if (job.status === "queued") users[u].queued++;
            else if (job.status === "completed") users[u].completed++;
            else if (job.status === "failed") users[u].failed++;
        }

        var userNames = Object.keys(users);
        if (userNames.length < 2) {
            $usersSection.style.display = "none";
            return;
        }

        $usersSection.style.display = "";
        $usersCount.textContent = "(" + userNames.length + ")";

        // Sort: users with active jobs first, then alphabetical
        userNames.sort(function(a, b) {
            var aActive = users[a].running + users[a].queued;
            var bActive = users[b].running + users[b].queued;
            if (bActive !== aActive) return bActive - aActive;
            return a.localeCompare(b);
        });

        var html = "";
        for (var j = 0; j < userNames.length; j++) {
            var name = userNames[j];
            var s = users[name];
            html += "<tr>";
            html += "<td>" + escapeHtml(name) + "</td>";
            html += '<td>' + (s.running > 0 ? '<span class="badge badge-running">' + s.running + '</span>' : '<span style="color:var(--text3)">0</span>') + '</td>';
            html += '<td>' + (s.queued > 0 ? '<span class="badge badge-queued">' + s.queued + '</span>' : '<span style="color:var(--text3)">0</span>') + '</td>';
            html += '<td>' + (s.completed > 0 ? '<span class="badge badge-completed">' + s.completed + '</span>' : '<span style="color:var(--text3)">0</span>') + '</td>';
            html += '<td>' + (s.failed > 0 ? '<span class="badge badge-failed">' + s.failed + '</span>' : '<span style="color:var(--text3)">0</span>') + '</td>';
            html += "<td>" + s.total + "</td>";
            html += "</tr>";
        }
        $usersBody.innerHTML = html;
    }

    function renderNodes(data) {
        if (data && !data.error) lastNodes = data;
        if (!lastNodes) return;
        data = lastNodes;

        $nodesCount.textContent = "(" + data.length + ")";

        if (data.length === 0) {
            $nodesBody.innerHTML = '<tr><td colspan="4" class="empty-state">No worker nodes connected</td></tr>';
            return;
        }

        var html = "";
        for (var i = 0; i < data.length; i++) {
            var n = data[i];
            var dotClass = n.alive ? "alive" : "dead";
            var statusText = n.alive ? "Online" : "Offline";
            var gpuStats = lastGpuStats[n.name] || [];

            html += "<tr>";
            html += "<td>" + escapeHtml(n.name) + "</td>";
            html += '<td>' + (n.resource ? '<span class="badge badge-partition">' + escapeHtml(n.resource) + '</span>' : '<span style="color:var(--text3)">--</span>') + '</td>';

            // Utilization column — GPU bars for GPU nodes, CPU bar for CPU nodes
            html += '<td>';
            if (n.gpus > 0) {
                if (gpuStats.length > 0) {
                    html += '<div class="gpu-bars">';
                    for (var g = 0; g < gpuStats.length; g++) {
                        var gpu = gpuStats[g];
                        var emaKey = n.name + ":" + gpu.index;
                        var util = emaSmooth(emaKey, "util", gpu.utilization || 0);
                        var memUsed = emaSmooth(emaKey, "mem", gpu.mem_used || 0);
                        var memTotal = gpu.mem_total || 0;
                        var temp = emaSmooth(emaKey, "temp", gpu.temperature || 0);
                        var memGB = Math.round(memUsed / 1024 * 10) / 10;
                        var memTotalGB = Math.round(memTotal / 1024 * 10) / 10;
                        var barColor = gpuUtilColor(util);

                        html += '<div class="gpu-mini">';
                        html += '<span class="gpu-mini-idx">' + gpu.index + '</span>';
                        html += '<div class="gpu-mini-track"><div class="gpu-mini-fill" style="width:' + util + '%;background:' + barColor + '"></div></div>';
                        html += '<span class="gpu-mini-pct">' + util + '%</span>';
                        html += '<span class="gpu-mini-mem">' + memGB + '/' + memTotalGB + 'G</span>';
                        html += '<span class="gpu-mini-temp">' + temp + '\u00B0C</span>';
                        html += '</div>';
                    }
                    html += '</div>';
                } else {
                    html += '<span style="color:var(--text3)">' + n.gpus + ' GPU' + (n.gpus !== 1 ? 's' : '') + '</span>';
                }
            } else {
                // CPU-only node
                var cpuInfo = lastCpuStats[n.name];
                if (cpuInfo && cpuInfo.cpu_util >= 0) {
                    var cpuUtil = cpuInfo.cpu_util;
                    var cpuColor = gpuUtilColor(cpuUtil);
                    var memUsedCpu = cpuInfo.mem_used_gb || 0;
                    var memTotalCpu = cpuInfo.mem_total_gb || 0;
                    html += '<div class="gpu-bars">';
                    html += '<div class="gpu-mini">';
                    html += '<span class="gpu-mini-idx">CPU</span>';
                    html += '<div class="gpu-mini-track"><div class="gpu-mini-fill" style="width:' + cpuUtil + '%;background:' + cpuColor + '"></div></div>';
                    html += '<span class="gpu-mini-pct">' + cpuUtil + '%</span>';
                    html += '<span class="gpu-mini-mem">' + memUsedCpu + '/' + memTotalCpu + 'G</span>';
                    html += '</div>';
                    html += '</div>';
                } else {
                    html += '<span style="color:var(--text3)">' + n.cpus + ' CPU' + (n.cpus !== 1 ? 's' : '') + '</span>';
                }
            }
            html += '</td>';

            html += '<td><span class="status-dot ' + dotClass + '"></span>' + statusText + '</td>';
            html += "</tr>";
        }
        $nodesBody.innerHTML = html;
    }

    function renderGpuStats(data) {
        if (data && !data.error) {
            lastGpuStats = data;
            renderNodes(null);  // re-render nodes with new GPU data
        }
    }

    function renderCpuStats(data) {
        if (data && !data.error) {
            lastCpuStats = data;
            renderNodes(null);  // re-render nodes with new CPU data
        }
    }

    function renderSlurm(data) {
        if (data.error) return;
        lastSlurm = data;
        $slurmCount.textContent = "(" + data.length + ")";

        if (data.length === 0) {
            $slurmBody.innerHTML = '<tr><td colspan="6" class="empty-state">No SLURM jobs tracked</td></tr>';
            return;
        }

        var html = "";
        for (var i = 0; i < data.length; i++) {
            var j = data[i];
            var stateClean = (j.state || "UNKNOWN").replace(/\s.*/,"");
            var estStart = j.est_start || "";
            if (estStart && estStart.indexOf("T") !== -1) {
                var parts = estStart.split("T");
                var date = parts[0].substring(5); // MM-DD
                var time = parts[1].substring(0, 5); // HH:MM
                estStart = date + " " + time;
            }
            var estCell;
            if (estStart) {
                estCell = escapeHtml(estStart);
            } else if (stateClean === "RUNNING") {
                estCell = "--";
            } else {
                estCell = "N/A";
            }
            html += "<tr>";
            html += "<td>" + escapeHtml(j.id) + "</td>";
            html += "<td>" + (j.resource_name ? '<span class="badge badge-partition">' + escapeHtml(j.resource_name) + '</span>' : '--') + "</td>";
            html += '<td><span class="badge ' + slurmBadgeClass(j.state) + '">' + escapeHtml(stateClean) + '</span></td>';
            html += "<td>" + escapeHtml(j.node || "--") + "</td>";
            html += "<td>" + escapeHtml(j.elapsed || "--") + "</td>";
            html += "<td>" + estCell + "</td>";
            html += "</tr>";
        }
        $slurmBody.innerHTML = html;
    }

    // --- Refresh ---

    function refreshAll() {
        fetchJSON("/api/status", renderStatus);
        fetchJSON("/api/nodes", renderNodes);
        fetchJSON("/api/slurm", renderSlurm);
        fetchJSON("/api/jobs", renderJobs);
        fetchJSON("/api/gpu_stats", renderGpuStats);
        fetchJSON("/api/cpu_stats", renderCpuStats);

        var now = new Date();
        var hh = String(now.getHours()).padStart(2, "0");
        var mm = String(now.getMinutes()).padStart(2, "0");
        var ss = String(now.getSeconds()).padStart(2, "0");
        $lastUpdated.textContent = hh + ":" + mm + ":" + ss;
    }

    // Initial load
    refreshAll();

    // Auto-refresh every 5 seconds
    setInterval(refreshAll, 5000);

})();
</script>

</body>
</html>
"""


if __name__ == "__main__":
    _cli()
