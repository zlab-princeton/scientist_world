"""Nexus daemon: Flask HTTP server and web dashboard.

Central process of Nexus. Provides a REST API for project management and
job submission, a long-poll wait endpoint, and a single-page HTML
dashboard for monitoring. The API has no authentication; see the
security section of the README before binding it to a non-loopback
address.

Usage:
    python -m nexus.daemon [--host HOST] [--port PORT]

or, embedded:
    app = create_app()
    app.run(host="127.0.0.1", port=39000, threaded=True)
"""

import logging
import os
import threading
import time
import uuid
from typing import Optional

from flask import Flask, jsonify, render_template_string, request

from .config import NexusConfig, load_config, ensure_nexus_dir
from .project import Project
from .jobs import NexusJob
from .hive_bridge import HiveBridge
from .monitor import JobMonitor

log = logging.getLogger(__name__)

# Daemon start time, set once in create_app and used for uptime reporting.
_daemon_start_time: float = 0.0


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------

def create_app(config: NexusConfig = None) -> Flask:
    """Create and configure the Flask application.

    Sets up the HiveBridge, JobMonitor, all REST API routes, and the
    inline HTML dashboard.

    Args:
        config: Optional NexusConfig. If None, loads from default config path.

    Returns:
        Configured Flask app with API endpoints and HTML dashboard.
    """
    global _daemon_start_time
    _daemon_start_time = time.time()

    if config is None:
        config = load_config()

    app = Flask(__name__)
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True

    # Validate required config
    if not config.hive.url:
        raise ValueError(
            "hive.url is not configured. Set it in config.yaml or via $HIVE_URL."
        )

    # Core services
    bridge = HiveBridge(config.hive.url)
    monitor = JobMonitor(bridge, config)
    monitor.start()

    # ── Submit-job idempotency cache ──────────────────────────────────
    # Maps client_request_id to (timestamp, response_dict). A client whose
    # submit request timed out can retry with the same id: the original
    # response is replayed and GPU quota is not acquired a second time.
    # Without it, the original request and its retry would each reserve a
    # slot for what is one logical job.
    #
    # The cache is in memory and entries expire after _IDEM_TTL_SEC, which
    # matches the 10-minute retry budget of a Hive submit.
    _idem_cache: dict[str, tuple[float, dict]] = {}
    _idem_lock = threading.Lock()
    _IDEM_TTL_SEC = 600

    # One mutex per client_request_id. The cache alone is not enough: it
    # is filled only after bridge.submit_for_project returns, and that
    # call can block for minutes while it retries a slow Hive. A client
    # with a shorter request timeout retries (same id) while the original
    # handler is still in flight, finds the cache empty and would acquire
    # quota again. Holding the lock across cache check, quota acquire,
    # Hive submit and cache fill makes the retry wait for the original
    # and then replay its response.
    _request_locks: dict[str, threading.Lock] = {}
    _request_locks_lock = threading.Lock()

    def _idem_get(rid: str) -> Optional[dict]:
        if not rid:
            return None
        now = time.time()
        with _idem_lock:
            entry = _idem_cache.get(rid)
            if entry is None:
                return None
            ts, response = entry
            if now - ts > _IDEM_TTL_SEC:
                _idem_cache.pop(rid, None)
                return None
            return response

    def _idem_put(rid: str, response: dict) -> None:
        if not rid:
            return
        now = time.time()
        with _idem_lock:
            _idem_cache[rid] = (now, response)
            # Prune entries older than the TTL.
            stale = [k for k, (t, _) in _idem_cache.items()
                     if now - t > _IDEM_TTL_SEC]
            for k in stale:
                _idem_cache.pop(k, None)
        # Drop the per-request locks whose id has no cached response. A
        # lock that is held belongs to a request still in flight and must
        # stay: a retry of that request has to find and wait on it.
        with _request_locks_lock:
            stale_locks = [k for k, lock in _request_locks.items()
                           if k not in _idem_cache and not lock.locked()]
            for k in stale_locks:
                _request_locks.pop(k, None)

    def _get_request_lock(rid: str) -> threading.Lock:
        """Get-or-create the per-client_request_id mutex."""
        with _request_locks_lock:
            lock = _request_locks.get(rid)
            if lock is None:
                lock = threading.Lock()
                _request_locks[rid] = lock
            return lock

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _job_counts(project_id: str) -> dict:
        """Count jobs by status for a project."""
        jobs = NexusJob.load_by_project(project_id)
        counts = {"running": 0, "completed": 0, "failed": 0, "total": 0,
                  "queued": 0, "cancelled": 0, "awaiting_response": 0}
        for j in jobs:
            counts["total"] += 1
            if j.status in counts:
                counts[j.status] += 1
        return counts

    def _project_summary(p: Project) -> dict:
        """Build a summary dict for a project listing."""
        counts = _job_counts(p.project_id)
        return {
            "project_id": p.project_id,
            "status": p.status,
            "created_at": p.created_at,
            "max_gpus": p.max_gpus,
            "allowed_resources": p.allowed_resources,
            "gpus_in_use": p.gpus_in_use,
            "soft_gpu_budget_hours": p.soft_gpu_budget_hours,
            "gpu_seconds_used": p.gpu_seconds_used,
            "budget_exhausted": p.budget_exhausted,
            "job_counts": counts,
        }

    # ------------------------------------------------------------------
    # API: Project endpoints (admin)
    # ------------------------------------------------------------------

    @app.route("/api/projects", methods=["POST"])
    def api_create_project():
        try:
            data = request.get_json(force=True)
            if not data:
                return jsonify({"error": "Request body is required"}), 400

            project_id = data.get("project_id")
            if not project_id:
                return jsonify({"error": "Missing 'project_id' field"}), 400
            # The id becomes a file name under PROJECTS_DIR and a URL path
            # segment, so it must not contain a path separator.
            if not isinstance(project_id, str) or "/" in project_id:
                return jsonify({
                    "error": "'project_id' must be a string without '/'",
                }), 400

            # Check for duplicate
            existing = Project.load(project_id)
            if existing is not None:
                return jsonify({"error": f"Project '{project_id}' already exists"}), 409

            workspace_dir = data.get("workspace_dir", "")
            sif_path = data.get("sif_path", "")
            overlay_path = data.get("overlay_path", "")
            hf_cache_path = data.get("hf_cache_path", "")

            # allowed_resources is optional; empty means "any resource"
            allowed_resources = data.get("allowed_resources", [])

            project = Project(
                project_id=project_id,
                max_gpus=data.get("max_gpus", config.project_defaults.max_gpus),
                allowed_resources=allowed_resources,
                max_cpus=data.get("max_cpus", 0),
                max_memory_mb=data.get("max_memory_mb", 0),
                default_job_timeout=data.get(
                    "default_job_timeout",
                    config.project_defaults.default_job_timeout,
                ),
                soft_gpu_budget_hours=data.get(
                    "soft_gpu_budget_hours",
                    config.project_defaults.soft_gpu_budget_hours,
                ),
                workspace_dir=workspace_dir,
                sif_path=sif_path,
                overlay_path=overlay_path,
                hf_cache_path=hf_cache_path,
                wake_command=data.get("wake_command"),
                shared_dirs=data.get("shared_dirs", []),
                checkpoint_interval=data.get("checkpoint_interval"),
                checkpoint_timeout=data.get("checkpoint_timeout"),
            )
            project.save()

            return jsonify({"project_id": project_id, "status": "registered"}), 201
        except Exception as e:
            log.exception("Error creating project")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/projects", methods=["GET"])
    def api_list_projects():
        try:
            projects = Project.load_all()
            result = [_project_summary(p) for p in projects]
            # Sort by created_at descending (newest first)
            result.sort(key=lambda x: x.get("created_at", 0), reverse=True)
            return jsonify(result)
        except Exception as e:
            log.exception("Error listing projects")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/projects/<project_id>", methods=["GET"])
    def api_get_project(project_id):
        try:
            project = Project.load(project_id)
            if project is None:
                return jsonify({"error": f"Project '{project_id}' not found"}), 404

            result = project.to_dict()
            result["job_counts"] = _job_counts(project_id)
            return jsonify(result)
        except Exception as e:
            log.exception("Error getting project")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/projects/<project_id>", methods=["PATCH"])
    def api_update_project(project_id):
        try:
            project = Project.load(project_id)
            if project is None:
                return jsonify({"error": f"Project '{project_id}' not found"}), 404

            data = request.get_json(force=True, silent=True)
            if not data:
                return jsonify({"error": "Request body is required"}), 400
            if not isinstance(data, dict):
                return jsonify({"error": "Request body must be a JSON object"}), 400

            try:
                if "max_gpus" in data:
                    project.max_gpus = int(data["max_gpus"])
                if "allowed_resources" in data:
                    resources = data["allowed_resources"]
                    if not isinstance(resources, list) or not resources:
                        return jsonify({"error": "'allowed_resources' must be a non-empty list"}), 400
                    project.allowed_resources = resources
                if "max_cpus" in data:
                    project.max_cpus = int(data["max_cpus"])
                if "max_memory_mb" in data:
                    project.max_memory_mb = int(data["max_memory_mb"])
                if "soft_gpu_budget_hours" in data:
                    project.soft_gpu_budget_hours = float(data["soft_gpu_budget_hours"])
                    # If budget was extended, clear budget_exhausted so new
                    # submissions are allowed again.
                    if not project.is_over_soft_budget():
                        project.budget_exhausted = False
                if "gpu_seconds_used" in data:
                    # Reset (or adjust) the accumulated GPU-time counter, for
                    # a caller that starts a new phase whose budget should
                    # not include what earlier phases used.
                    project.gpu_seconds_used = float(data["gpu_seconds_used"])
                    if not project.is_over_soft_budget():
                        project.budget_exhausted = False
                if "default_job_timeout" in data:
                    project.default_job_timeout = int(data["default_job_timeout"])
                if "sif_path" in data:
                    project.sif_path = str(data["sif_path"])
                if "overlay_path" in data:
                    project.overlay_path = str(data["overlay_path"])
            except (TypeError, ValueError) as e:
                return jsonify({"error": f"Invalid field value: {e}"}), 400

            project.save()
            result = project.to_dict()
            return jsonify(result)
        except Exception as e:
            log.exception("Error updating project")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/projects/<project_id>", methods=["DELETE"])
    def api_delete_project(project_id):
        try:
            project = Project.load(project_id)
            if project is None:
                return jsonify({"error": f"Project '{project_id}' not found"}), 404

            # Cancel all running jobs
            jobs = NexusJob.load_by_project(project_id)
            for job in jobs:
                if job.status in ("queued", "running", "awaiting_response"):
                    try:
                        monitor.cancel_job(job.run_id)
                    except Exception as e:
                        log.warning(f"Failed to cancel job {job.run_id}: {e}")

            # Delete all job records for this project
            for job in jobs:
                job.delete()

            # Delete the project itself
            project.delete()

            return jsonify({"project_id": project_id, "status": "deleted"})
        except Exception as e:
            log.exception("Error deleting project")
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # API: Job endpoints (agent + admin)
    # ------------------------------------------------------------------

    @app.route("/api/projects/<project_id>/jobs", methods=["POST"])
    def api_submit_job(project_id):
        try:
            project = Project.load(project_id)
            if project is None:
                return jsonify({"error": f"Project '{project_id}' not found"}), 404

            data = request.get_json(force=True)
            if not data or "command" not in data:
                return jsonify({"error": "Missing 'command' field"}), 400

            command = data["command"]
            num_gpus = data.get("num_gpus", 1)
            num_cpus = data.get("num_cpus", 1)
            memory_mb = data.get("memory_mb", 0)
            timeout_sec = data.get("timeout_sec", project.default_job_timeout)
            client_request_id = data.get("client_request_id") or ""

            # Per-id mutex: keeps a retry from running while the original
            # request is still in flight (see the comment at
            # _request_locks). A request without a client_request_id
            # cannot be matched to a retry and proceeds without locking.
            req_lock = (
                _get_request_lock(client_request_id)
                if client_request_id else None
            )
            if req_lock is not None:
                req_lock.acquire()
            try:
                # Idempotency: if the client retried after a timeout but
                # the original request already produced a job, replay
                # the cached response without acquiring quota again.
                # The check runs inside the per-id lock so that a retry
                # which arrived while the original was in flight sees
                # the cache entry written by the original handler.
                if client_request_id:
                    cached = _idem_get(client_request_id)
                    if cached is not None:
                        log.info(
                            "Idempotent replay for client_request_id=%s "
                            "(project=%s): run_id=%s",
                            client_request_id, project_id,
                            cached.get("run_id", "?"),
                        )
                        return jsonify(cached), 201

                # Resolve resource(s): optional, defaults to all allowed
                raw_resource = data.get("resource")
                if raw_resource:
                    # The caller named resource(s): check them against the allowed list (if set)
                    requested = [r.strip() for r in raw_resource.split(",") if r.strip()]
                    if project.allowed_resources:
                        bad = [r for r in requested if r not in project.allowed_resources]
                        if bad:
                            return jsonify({
                                "error": f"Resource(s) {bad} not in allowed list: "
                                         f"{project.allowed_resources}",
                            }), 403
                    resource = ",".join(requested)
                else:
                    # Default to all allowed resources (empty = any)
                    resource = ",".join(project.allowed_resources)

                # Atomic check + acquire GPU quota
                ok, reason = project.check_and_acquire(num_gpus, num_cpus, memory_mb)
                if not ok:
                    return jsonify({"error": reason}), 403

                # Everything from here through monitor.add_job() is inside
                # one try, so that the GPUs acquired above are released on
                # any failure (Hive submission, saving the job record,
                # registering with the monitor). Once add_job() has
                # returned, the monitor owns the job and releases the GPUs
                # when it ends.
                try:
                    # Create the job
                    run_id = str(uuid.uuid4())[:12]
                    job = NexusJob(
                        run_id=run_id,
                        project_id=project_id,
                        command=command,
                        num_gpus=num_gpus,
                        num_cpus=num_cpus,
                        timeout_sec=timeout_sec,
                        resource=resource,
                        status="queued",
                    )

                    # Activate the project on first job
                    if project.status == "registered":
                        project.status = "active"
                        project.save()

                    hive_job_id = bridge.submit_for_project(
                        project, command, num_gpus, num_cpus, timeout_sec,
                        resource=resource,
                    )
                    job.hive_job_id = hive_job_id
                    job.save()
                    if project.workspace_dir:
                        try:
                            job.write_meta(project.workspace_dir)
                        except OSError as exc:
                            log.warning(
                                "Could not write job meta to workspace: %s", exc,
                            )

                    # Register with the monitor
                    monitor.add_job(job, project)
                except Exception:
                    project.release_gpus(num_gpus)
                    raise

                response = {
                    "run_id": run_id,
                    "hive_job_id": hive_job_id,
                    "project_id": project_id,
                    "command": command,
                    "num_gpus": num_gpus,
                    "resource": resource,
                    "status": "queued",
                    "timeout_sec": timeout_sec,
                    "project_gpus_in_use": project.gpus_in_use,
                    "project_max_gpus": project.max_gpus,
                    "gpu_seconds_used": project.gpu_seconds_used,
                    "soft_gpu_budget_hours": project.soft_gpu_budget_hours,
                }
                # Cache for idempotent replay if the client times out and retries.
                _idem_put(client_request_id, response)
                return jsonify(response), 201
            finally:
                if req_lock is not None:
                    req_lock.release()
        except Exception as e:
            log.exception("Error submitting job")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/projects/<project_id>/jobs", methods=["GET"])
    def api_list_project_jobs(project_id):
        try:
            project = Project.load(project_id)
            if project is None:
                return jsonify({"error": f"Project '{project_id}' not found"}), 404

            jobs = NexusJob.load_by_project(project_id)
            result = [j.to_dict() for j in jobs]
            # Sort: active jobs first, then by submitted_at descending
            status_order = {"running": 0, "queued": 1, "awaiting_response": 2,
                            "completed": 3, "failed": 4, "cancelled": 5}
            result.sort(key=lambda j: (
                status_order.get(j.get("status", ""), 9),
                -(j.get("submitted_at") or 0),
            ))
            return jsonify(result)
        except Exception as e:
            log.exception("Error listing project jobs")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/jobs/<run_id>", methods=["GET"])
    def api_get_job(run_id):
        try:
            job = NexusJob.load(run_id)
            if job is None:
                return jsonify({"error": f"Job '{run_id}' not found"}), 404

            result = job.to_dict()

            # If the job is still active and has a Hive job ID, fetch live logs
            if job.hive_job_id and job.status in ("queued", "running",
                                                   "awaiting_response"):
                try:
                    poll_data = bridge.poll(job.hive_job_id)
                    if poll_data:
                        result["hive_status"] = poll_data.get("status")
                        result["stdout"] = poll_data.get("stdout", "")
                        result["stderr"] = poll_data.get("stderr", "")
                except Exception as e:
                    log.debug(f"Failed to poll Hive for job {run_id}: {e}")

            return jsonify(result)
        except Exception as e:
            log.exception("Error getting job")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/jobs/<run_id>/continue", methods=["POST"])
    def api_continue_job(run_id):
        try:
            job = NexusJob.load(run_id)
            if job is None:
                return jsonify({"error": f"Job '{run_id}' not found"}), 404

            monitor.continue_job(run_id)
            return jsonify({"run_id": run_id, "status": "running"})
        except Exception as e:
            log.exception("Error continuing job")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/jobs/<run_id>/cancel", methods=["POST"])
    def api_cancel_job(run_id):
        try:
            job = NexusJob.load(run_id)
            if job is None:
                return jsonify({"error": f"Job '{run_id}' not found"}), 404

            monitor.cancel_job(run_id)
            return jsonify({"run_id": run_id, "status": "cancelled"})
        except Exception as e:
            log.exception("Error cancelling job")
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # API: Wait endpoint (agent long-poll)
    # ------------------------------------------------------------------

    @app.route("/api/projects/<project_id>/wait", methods=["GET"])
    def api_wait(project_id):
        try:
            project = Project.load(project_id)
            if project is None:
                return jsonify({"error": f"Project '{project_id}' not found"}), 404

            timeout = request.args.get("timeout", 300, type=int)
            timeout = min(timeout, 600)  # Cap at 10 minutes
            # Optional: caller can specify a run_id to watch.  If the job
            # already reached a terminal state before we start polling,
            # return immediately instead of blocking for the full timeout.
            watch_run_id = request.args.get("run_id", "", type=str)
            poll_interval = 2
            deadline = time.time() + timeout

            # Track which terminal/actionable states we've seen at start
            terminal_statuses = {"awaiting_response", "completed", "failed",
                                 "cancelled"}

            # Jobs already in one of these states when the wait starts,
            # with that state. Such a job is reported only when its state
            # changes: a job waiting at a checkpoint that is continued and
            # then ends during this wait must not be missed.
            initial_jobs = NexusJob.load_by_project(project_id)
            seen_terminal = {}
            for j in initial_jobs:
                if j.status in terminal_statuses:
                    # If the caller is watching this specific run_id and
                    # it's already terminal, return immediately.
                    if watch_run_id and j.run_id == watch_run_id:
                        return jsonify({
                            "run_id": j.run_id,
                            "status": j.status,
                            "event": j.status,
                        })
                    seen_terminal[j.run_id] = j.status

            while time.time() < deadline:
                jobs = NexusJob.load_by_project(project_id)
                for j in jobs:
                    if j.status not in terminal_statuses:
                        seen_terminal.pop(j.run_id, None)
                    elif seen_terminal.get(j.run_id) != j.status:
                        return jsonify({
                            "run_id": j.run_id,
                            "status": j.status,
                            "event": j.status,
                        })
                time.sleep(poll_interval)

            return jsonify({"timeout": True})
        except Exception as e:
            log.exception("Error in wait endpoint")
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # API: Status endpoint
    # ------------------------------------------------------------------

    @app.route("/api/status", methods=["GET"])
    def api_status():
        try:
            # Daemon info
            uptime = time.time() - _daemon_start_time
            pid = os.getpid()

            # Hive info
            hive_connected = False
            hive_data = {}
            try:
                hive_status = bridge.cluster_status()
                if hive_status and not hive_status.get("error"):
                    hive_connected = True
                    total_gpus = hive_status.get("total_gpus", 0)
                    hive_data = {
                        "total_gpus": total_gpus,
                        "available_gpu": min(
                            hive_status.get("available_gpu",
                                            hive_status.get("free_gpus", 0)),
                            total_gpus,
                        ),
                        "num_gpu_nodes": hive_status.get("num_gpu_nodes", 0),
                        "num_cpu_nodes": hive_status.get("num_cpu_nodes", 0),
                    }
            except Exception as e:
                log.debug(f"Failed to query Hive status: {e}")

            # Project stats
            projects = Project.load_all()
            total_projects = len(projects)
            active_projects = sum(1 for p in projects if p.status == "active")
            budget_exhausted = sum(
                1 for p in projects if p.budget_exhausted
            )

            # Job stats
            all_jobs = NexusJob.load_all()
            running_jobs = sum(1 for j in all_jobs
                               if j.status in ("queued", "running",
                                                "awaiting_response"))
            completed_jobs = sum(1 for j in all_jobs
                                 if j.status == "completed")
            failed_jobs = sum(1 for j in all_jobs if j.status == "failed")

            return jsonify({
                "daemon": {
                    "uptime": uptime,
                    "pid": pid,
                },
                "hive": dict(
                    url=config.hive.url,
                    connected=hive_connected,
                    **hive_data,
                ),
                "projects": {
                    "total": total_projects,
                    "active": active_projects,
                    "budget_exhausted": budget_exhausted,
                },
                "jobs": {
                    "running": running_jobs,
                    "completed": completed_jobs,
                    "failed": failed_jobs,
                },
            })
        except Exception as e:
            log.exception("Error getting status")
            return jsonify({"error": str(e)}), 500

    # ------------------------------------------------------------------
    # HTML dashboard
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def serve(config: NexusConfig) -> None:
    """Serve the daemon on ``config.daemon.host:port`` until interrupted.

    Configures logging to stderr, creates the state directories and runs
    the Flask app with one thread per request (the wait endpoint holds
    its connection open).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if config.daemon.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "Listening on %s: the API has no authentication, so every "
            "client that can reach port %d has full control.",
            config.daemon.host, config.daemon.port,
        )
    ensure_nexus_dir()
    app = create_app(config)
    app.run(host=config.daemon.host, port=config.daemon.port,
            debug=False, threaded=True)


def main(argv=None):
    """Run the Nexus daemon as a standalone server."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m nexus.daemon",
        description="Nexus daemon (REST API and web dashboard)",
    )
    parser.add_argument("--port", type=int,
                        help="TCP port (default: daemon.port from the config)")
    parser.add_argument("--host",
                        help="Bind address (default: daemon.host from the config)")
    args = parser.parse_args(argv)

    config = load_config()
    if args.port:
        config.daemon.port = args.port
    if args.host:
        config.daemon.host = args.host

    serve(config)


# ---------------------------------------------------------------------------
# Dashboard HTML (inline CSS + JS, no external dependencies)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nexus Dashboard</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg: #ede9f6;
    --card: #faf8ff;
    --border: #d4cce8;
    --accent: #7c3aed;
    --accent-light: #a78bfa;
    --success: #16a34a;
    --warning: #d97706;
    --error: #dc2626;
    --purple: #7c3aed;
    --blue: #2563eb;
    --amber: #d97706;
    --gray: #64748b;
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
    gap: 16px;
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

.header-pills {
    display: flex;
    gap: 10px;
    font-size: 12px;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 12px;
    font-weight: 500;
    font-family: 'Courier New', monospace;
}

.pill-ok { background: rgba(22,163,74,0.12); color: var(--success); }
.pill-err { background: rgba(220,38,38,0.12); color: var(--error); }
.pill-neutral { background: rgba(99,102,241,0.1); color: var(--accent); }

.pill-dot {
    width: 6px; height: 6px; border-radius: 50%;
}
.pill-ok .pill-dot { background: var(--success); }
.pill-err .pill-dot { background: var(--error); }

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
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--success);
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

.pulse-dot.error { background: var(--error); animation: none; }

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

/* Overview cards */
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
    box-shadow: var(--shadow);
}

.card-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text2);
    margin-bottom: 6px;
}

.card-value {
    font-size: 28px;
    font-weight: 700;
    font-family: 'Courier New', monospace;
}

.card-value-secondary {
    font-size: 18px;
    font-weight: 600;
    font-family: 'Courier New', monospace;
    color: var(--text2);
    margin-top: 2px;
}

.tag {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
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
    margin-bottom: 28px;
}

.section-header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 12px;
}

.section-title {
    font-size: 14px;
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

/* Project cards */
.project-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 16px;
}

.project-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
    box-shadow: var(--shadow);
}

.project-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.project-name {
    font-size: 15px;
    font-weight: 600;
    font-family: 'Courier New', monospace;
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

.badge-registered { background: rgba(100,116,139,0.15); color: var(--gray); }
.badge-active { background: rgba(22,163,74,0.15); color: var(--success); }
.badge-queued { background: rgba(124,58,237,0.15); color: var(--purple); }
.badge-running { background: rgba(37,99,235,0.15); color: var(--blue); }
.badge-awaiting_response { background: rgba(217,119,6,0.15); color: var(--amber); }
.badge-completed { background: rgba(22,163,74,0.15); color: var(--success); }
.badge-failed { background: rgba(220,38,38,0.15); color: var(--error); }
.badge-cancelled { background: rgba(100,116,139,0.15); color: var(--gray); }

/* Progress bars */
.progress-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}

.progress-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text3);
    min-width: 48px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.progress-track {
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}

.progress-fill-gpu { background: linear-gradient(90deg, var(--accent), var(--accent-light)); }
.progress-fill-time { background: linear-gradient(90deg, var(--success), #4ade80); }
.progress-fill-time-warn { background: linear-gradient(90deg, var(--warning), #fbbf24); }
.progress-fill-time-crit { background: linear-gradient(90deg, var(--error), #f87171); }

.progress-value {
    font-size: 11px;
    font-family: 'Courier New', monospace;
    color: var(--text2);
    min-width: 56px;
    text-align: right;
}

/* Job counts inline */
.job-counts {
    display: flex;
    gap: 8px;
    margin-top: 10px;
    flex-wrap: wrap;
}

.job-count-item {
    font-size: 11px;
    font-family: 'Courier New', monospace;
    color: var(--text3);
}

.job-count-item strong {
    font-weight: 700;
    color: var(--text2);
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
    padding: 10px 14px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text3);
    border-bottom: 1px solid var(--border);
    background: rgba(0,0,0,0.02);
}

tbody td {
    padding: 8px 14px;
    font-size: 12px;
    border-bottom: 1px solid var(--border);
    font-family: 'Courier New', monospace;
}

tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: rgba(99,102,241,0.04); }

.cmd-truncated {
    max-width: 260px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: inline-block;
}

.empty-state {
    padding: 24px;
    text-align: center;
    color: var(--text3);
    font-size: 13px;
}

/* Responsive */
@media (max-width: 768px) {
    .main { padding: 12px; }
    .cards { grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .project-grid { grid-template-columns: 1fr; }
    .header { padding: 12px 16px; flex-wrap: wrap; gap: 8px; }
    .header-pills { display: none; }
}
</style>
</head>
<body>

<div class="header">
    <div class="header-left">
        <div class="logo">NEXUS <span class="logo-sub">Project Manager</span></div>
        <div class="header-pills">
            <span class="pill pill-neutral" id="pillHive">
                <span class="pill-dot"></span>
                Hive: --
            </span>
            <span class="pill pill-neutral" id="pillGpus">GPUs: --/--</span>
        </div>
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

    <!-- Overview cards -->
    <div class="cards">
        <div class="card">
            <div class="card-label">Projects</div>
            <div class="card-value" id="totalProjects">--</div>
            <div class="card-sub" id="projectsSub">-- active</div>
        </div>
        <div class="card">
            <div class="card-label">Running Jobs</div>
            <div class="card-value" id="runningJobs">--</div>
            <div class="card-sub" id="jobsSub">-- completed, -- failed</div>
        </div>
        <div class="card">
            <div class="card-label">Hive Resources</div>
            <div class="card-value"><span id="hiveGpus">--</span> <span class="tag">gpu</span></div>
            <div class="card-value card-value-secondary"><span id="hiveCpuNodes">--</span> <span class="tag">cpu node</span></div>
            <div class="card-sub" id="hiveSub">-- available</div>
        </div>
        <div class="card">
            <div class="card-label">Daemon Uptime</div>
            <div class="card-value" id="uptime">--</div>
            <div class="card-sub" id="daemonPid">PID --</div>
        </div>
    </div>

    <!-- Projects -->
    <div class="section">
        <div class="section-header">
            <span class="section-title">Projects</span>
            <span class="section-count" id="projectsCount">(0)</span>
        </div>
        <div class="project-grid" id="projectsGrid">
            <div class="empty-state">Loading...</div>
        </div>
    </div>

    <!-- All Jobs -->
    <div class="section">
        <div class="section-header">
            <span class="section-title">Recent Jobs</span>
            <span class="section-count" id="jobsCount">(0)</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Run ID</th>
                        <th>Project</th>
                        <th>Command</th>
                        <th>GPUs</th>
                        <th>Status</th>
                        <th>Elapsed</th>
                        <th>Checkpoint</th>
                    </tr>
                </thead>
                <tbody id="jobsBody">
                    <tr><td colspan="7" class="empty-state">Loading...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

</div>

<script>
(function() {
    "use strict";

    // --- Utilities ---

    function formatDuration(seconds) {
        if (seconds === null || seconds === undefined || seconds < 0) return "--";
        seconds = Math.floor(seconds);
        var h = Math.floor(seconds / 3600);
        var m = Math.floor((seconds % 3600) / 60);
        var s = seconds % 60;
        if (h > 0) return h + "h " + m + "m";
        if (m > 0) return m + "m " + s + "s";
        return s + "s";
    }

    function formatUptime(seconds) {
        if (!seconds) return "--";
        seconds = Math.floor(seconds);
        var d = Math.floor(seconds / 86400);
        var h = Math.floor((seconds % 86400) / 3600);
        var m = Math.floor((seconds % 3600) / 60);
        if (d > 0) return d + "d " + h + "h";
        if (h > 0) return h + "h " + m + "m";
        return m + "m";
    }

    function elapsed(startTs, endTs) {
        if (!startTs) return "--";
        var end = endTs || (Date.now() / 1000);
        return formatDuration(end - startTs);
    }

    function relativeTime(ts) {
        if (!ts) return "--";
        var diff = (Date.now() / 1000) - ts;
        if (diff < 0) diff = 0;
        if (diff < 60) return Math.floor(diff) + "s ago";
        if (diff < 3600) return Math.floor(diff / 60) + "m ago";
        if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
        return Math.floor(diff / 86400) + "d ago";
    }

    function escapeHtml(str) {
        if (!str) return "";
        var d = document.createElement("div");
        d.textContent = str;
        // Quotes too: the result is also placed inside attribute values.
        return d.innerHTML.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }

    function badgeClass(status) {
        var map = {
            "registered": "badge-registered",
            "active": "badge-active",
            "queued": "badge-queued",
            "running": "badge-running",
            "awaiting_response": "badge-awaiting_response",
            "completed": "badge-completed",
            "failed": "badge-failed",
            "cancelled": "badge-cancelled"
        };
        return map[status] || "badge-registered";
    }

    function timeFillClass(pct) {
        if (pct > 90) return "progress-fill-time-crit";
        if (pct > 70) return "progress-fill-time-warn";
        return "progress-fill-time";
    }

    // --- State ---
    var errorCount = 0;

    // --- DOM refs ---
    var $totalProjects = document.getElementById("totalProjects");
    var $projectsSub = document.getElementById("projectsSub");
    var $runningJobs = document.getElementById("runningJobs");
    var $jobsSub = document.getElementById("jobsSub");
    var $hiveGpus = document.getElementById("hiveGpus");
    var $hiveCpuNodes = document.getElementById("hiveCpuNodes");
    var $hiveSub = document.getElementById("hiveSub");
    var $uptime = document.getElementById("uptime");
    var $daemonPid = document.getElementById("daemonPid");
    var $projectsCount = document.getElementById("projectsCount");
    var $projectsGrid = document.getElementById("projectsGrid");
    var $jobsCount = document.getElementById("jobsCount");
    var $jobsBody = document.getElementById("jobsBody");
    var $lastUpdated = document.getElementById("lastUpdated");
    var $pulseDot = document.getElementById("pulseDot");
    var $errorBanner = document.getElementById("errorBanner");
    var $pillHive = document.getElementById("pillHive");
    var $pillGpus = document.getElementById("pillGpus");

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
                if (errorCount >= 3) {
                    $errorBanner.textContent = "Failed to reach API (" + url + "): " + err.message;
                    $errorBanner.classList.add("visible");
                    $pulseDot.classList.add("error");
                }
            });
    }

    // --- Renderers ---

    function renderStatus(data) {
        if (data.error) return;

        // Cards
        $totalProjects.textContent = data.projects.total;
        $projectsSub.textContent = data.projects.active + " active, " + data.projects.budget_exhausted + " budget exhausted";

        $runningJobs.textContent = data.jobs.running;
        $jobsSub.textContent = data.jobs.completed + " completed, " + data.jobs.failed + " failed";

        var totalGpus = data.hive.total_gpus || 0;
        var availGpu = data.hive.available_gpu || 0;
        var gpuNodes = data.hive.num_gpu_nodes || 0;
        var cpuNodes = data.hive.num_cpu_nodes || 0;
        $hiveGpus.textContent = totalGpus;
        $hiveCpuNodes.textContent = cpuNodes;
        $hiveSub.textContent = availGpu + "/" + totalGpus + " gpu available";

        $uptime.textContent = formatUptime(data.daemon.uptime);
        $daemonPid.textContent = "PID " + data.daemon.pid;

        // Header pills
        if (data.hive.connected) {
            $pillHive.className = "pill pill-ok";
            $pillHive.innerHTML = '<span class="pill-dot"></span>Hive: connected';
        } else {
            $pillHive.className = "pill pill-err";
            $pillHive.innerHTML = '<span class="pill-dot"></span>Hive: disconnected';
        }
        $pillGpus.textContent = "GPUs: " + availGpu + "/" + totalGpus + " available";
    }

    function renderProjects(data) {
        if (data.error) return;

        $projectsCount.textContent = "(" + data.length + ")";

        if (data.length === 0) {
            $projectsGrid.innerHTML = '<div class="empty-state">No projects registered</div>';
            return;
        }

        var html = "";
        for (var i = 0; i < data.length; i++) {
            var p = data[i];
            var gpuPct = p.max_gpus > 0 ? Math.round((p.gpus_in_use / p.max_gpus) * 100) : 0;

            // GPU budget progress
            var budgetPct = 0;
            var budgetLabel = "unlimited";
            var budgetHours = p.soft_gpu_budget_hours || 0;
            var usedHours = (p.gpu_seconds_used || 0) / 3600;
            if (budgetHours > 0) {
                budgetPct = Math.min(100, Math.round((usedHours / budgetHours) * 100));
                budgetLabel = usedHours.toFixed(1) + " / " + budgetHours.toFixed(1) + " gpu-h";
            } else if (usedHours > 0) {
                budgetLabel = usedHours.toFixed(1) + " gpu-h used";
            }

            var jc = p.job_counts || {};

            html += '<div class="project-card">';
            html += '<div class="project-card-header">';
            html += '<span class="project-name">' + escapeHtml(p.project_id) + '</span>';
            html += '<span class="badge ' + badgeClass(p.status) + '">' + escapeHtml(p.status) + '</span>';
            html += '</div>';

            // GPU bar
            html += '<div class="progress-row">';
            html += '<span class="progress-label">GPUs</span>';
            html += '<div class="progress-track"><div class="progress-fill progress-fill-gpu" style="width:' + gpuPct + '%"></div></div>';
            html += '<span class="progress-value">' + p.gpus_in_use + ' / ' + p.max_gpus + '</span>';
            html += '</div>';

            // GPU budget bar
            html += '<div class="progress-row">';
            html += '<span class="progress-label">Budget</span>';
            html += '<div class="progress-track"><div class="progress-fill ' + timeFillClass(budgetPct) + '" style="width:' + budgetPct + '%"></div></div>';
            html += '<span class="progress-value">' + budgetLabel + '</span>';
            html += '</div>';

            // Job counts
            html += '<div class="job-counts">';
            html += '<span class="job-count-item"><strong>' + (jc.running || 0) + '</strong> running</span>';
            html += '<span class="job-count-item"><strong>' + (jc.queued || 0) + '</strong> queued</span>';
            html += '<span class="job-count-item"><strong>' + (jc.completed || 0) + '</strong> done</span>';
            html += '<span class="job-count-item"><strong>' + (jc.failed || 0) + '</strong> fail</span>';
            html += '<span class="job-count-item"><strong>' + (jc.total || 0) + '</strong> total</span>';
            html += '</div>';

            html += '</div>';
        }
        $projectsGrid.innerHTML = html;
    }

    function renderJobs(data) {
        if (data.error) return;

        // Collect all jobs from all projects, flatten
        var allJobs = [];
        for (var i = 0; i < data.length; i++) {
            allJobs = allJobs.concat(data[i]);
        }

        // Sort: active first, then by submitted_at descending
        var statusOrder = {"running": 0, "queued": 1, "awaiting_response": 2,
                           "completed": 3, "failed": 4, "cancelled": 5};
        allJobs.sort(function(a, b) {
            var oa = statusOrder[a.status] !== undefined ? statusOrder[a.status] : 9;
            var ob = statusOrder[b.status] !== undefined ? statusOrder[b.status] : 9;
            if (oa !== ob) return oa - ob;
            return (b.submitted_at || 0) - (a.submitted_at || 0);
        });

        // Limit to 50 most recent
        allJobs = allJobs.slice(0, 50);

        $jobsCount.textContent = "(" + allJobs.length + ")";

        if (allJobs.length === 0) {
            $jobsBody.innerHTML = '<tr><td colspan="7" class="empty-state">No jobs submitted</td></tr>';
            return;
        }

        var html = "";
        for (var j = 0; j < allJobs.length; j++) {
            var job = allJobs[j];

            var cmdShort = job.command || "";
            if (cmdShort.length > 50) cmdShort = cmdShort.substring(0, 47) + "...";
            var cmdFull = escapeHtml(job.command || "");

            var dur = elapsed(job.submitted_at, job.completed_at);

            var ckpt = "--";
            if (job.last_checkpoint_at) {
                ckpt = relativeTime(job.last_checkpoint_at);
                if (job.checkpoint_count) ckpt += " (#" + job.checkpoint_count + ")";
            }

            html += "<tr>";
            html += '<td>' + escapeHtml(job.run_id) + '</td>';
            html += '<td>' + escapeHtml(job.project_id) + '</td>';
            html += '<td><span class="cmd-truncated" title="' + cmdFull + '">' + escapeHtml(cmdShort) + '</span></td>';
            html += '<td>' + (job.num_gpus || 0) + '</td>';
            html += '<td><span class="badge ' + badgeClass(job.status) + '">' + escapeHtml(job.status) + '</span></td>';
            html += '<td>' + dur + '</td>';
            html += '<td>' + ckpt + '</td>';
            html += "</tr>";
        }
        $jobsBody.innerHTML = html;
    }

    // --- Data collection for jobs across projects ---

    var projectIds = [];

    function fetchProjectJobs() {
        if (projectIds.length === 0) {
            renderJobs({error: false});
            return;
        }

        var pending = projectIds.length;
        var allData = [];

        for (var i = 0; i < projectIds.length; i++) {
            (function(pid) {
                fetchJSON("/api/projects/" + encodeURIComponent(pid) + "/jobs", function(data) {
                    if (!data.error) allData.push(data);
                    pending--;
                    if (pending <= 0) renderJobs(allData);
                });
            })(projectIds[i]);
        }
    }

    // --- Refresh ---

    function refreshAll() {
        fetchJSON("/api/status", renderStatus);
        fetchJSON("/api/projects", function(data) {
            if (!data.error) {
                renderProjects(data);
                projectIds = [];
                for (var i = 0; i < data.length; i++) {
                    projectIds.push(data[i].project_id);
                }
                fetchProjectJobs();
            }
        });

        var now = new Date();
        var hh = String(now.getHours()).padStart(2, "0");
        var mm = String(now.getMinutes()).padStart(2, "0");
        var ss = String(now.getSeconds()).padStart(2, "0");
        $lastUpdated.textContent = hh + ":" + mm + ":" + ss;
    }

    refreshAll();
    setInterval(refreshAll, 5000);

})();
</script>

</body>
</html>
"""


if __name__ == "__main__":
    main()
