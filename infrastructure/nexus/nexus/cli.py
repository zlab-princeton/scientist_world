"""Nexus admin CLI: manage the daemon, projects, and jobs."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

from .config import load_config, ensure_nexus_dir, DAEMON_PID_PATH, NEXUS_DIR, REMOTE_CONFIG_PATH, NexusConfig

# How long `nexus start` waits for a background daemon to accept connections.
_STARTUP_TIMEOUT_SEC = 15.0


# ---------------------------------------------------------------------------
# HTTP helpers (urllib, talking to the Nexus daemon REST API)
# ---------------------------------------------------------------------------

def _get_daemon_url(config: NexusConfig) -> str:
    """URL of the local daemon: localhost on the configured port."""
    return f"http://localhost:{config.daemon.port}"


def _get_remote_url() -> Optional[str]:
    """Read remote daemon URL from remote.json, if it exists."""
    if not REMOTE_CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(REMOTE_CONFIG_PATH.read_text())
        return data.get("daemon_url")
    except (json.JSONDecodeError, OSError):
        return None


def _auto_disconnect():
    """Remove remote.json and print a message."""
    if REMOTE_CONFIG_PATH.exists():
        REMOTE_CONFIG_PATH.unlink()
        print("Auto-disconnected from remote daemon (removed remote.json).", file=sys.stderr)


def _resolve_base_url() -> str:
    """Resolve the daemon URL: remote.json first, then localhost from config."""
    remote_url = _get_remote_url()
    if remote_url:
        return remote_url
    config = load_config()
    return _get_daemon_url(config)


def _api_request(base_url: str, path: str, method: str = "GET",
                 data: dict = None) -> dict:
    """Send an HTTP request to the daemon API and return parsed JSON.

    Raises SystemExit with a friendly message on connection errors.
    """
    url = f"{base_url}{path}"
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        # Try to extract a JSON error body from the daemon
        try:
            err_body = json.loads(e.read().decode())
            msg = err_body.get("error", str(e))
        except Exception:
            msg = str(e)
        print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        if _get_remote_url():
            _auto_disconnect()
        print(
            f"Error: cannot reach Nexus daemon at {base_url}\n"
            f"  ({e})\n"
            f"  Is the daemon running?  Start it with:  nexus start",
            file=sys.stderr,
        )
        sys.exit(1)


def _api_get(base_url: str, path: str) -> dict:
    """GET JSON from daemon API."""
    return _api_request(base_url, path, method="GET")


def _api_post(base_url: str, path: str, data: dict = None) -> dict:
    """POST JSON to daemon API."""
    return _api_request(base_url, path, method="POST", data=data or {})


def _api_patch(base_url: str, path: str, data: dict) -> dict:
    """PATCH JSON to daemon API."""
    return _api_request(base_url, path, method="PATCH", data=data)


def _api_delete(base_url: str, path: str) -> dict:
    """DELETE to daemon API."""
    return _api_request(base_url, path, method="DELETE")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Format seconds as 'Xh Ym' or 'Xm Ys'.

    Returns a compact human-readable duration string.
    """
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _daemon_pid() -> Optional[int]:
    """Read the daemon PID from the PID file. Returns None if not found."""
    if not DAEMON_PID_PATH.exists():
        return None
    try:
        pid = int(DAEMON_PID_PATH.read_text().strip())
        os.kill(pid, 0)  # check if process is alive
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        # PermissionError: the PID now belongs to someone else's process.
        return None


def _probe_host(bind_host: str) -> str:
    """Address at which a daemon bound to *bind_host* can be reached locally."""
    if bind_host in ("0.0.0.0", ""):
        return "127.0.0.1"
    if bind_host == "::":
        return "::1"
    return bind_host


def _port_open(host: str, port: int) -> bool:
    """Return True if something accepts TCP connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _wait_for_daemon(proc, host: str, port: int) -> Optional[bool]:
    """Wait for a freshly spawned daemon process to come up.

    Returns True once it accepts connections, False if the process has
    exited, and None if it is still running but not reachable after
    ``_STARTUP_TIMEOUT_SEC``.  ``proc.poll()`` is used rather than a
    signal probe because an exited child that has not been reaped still
    answers ``os.kill(pid, 0)``.
    """
    deadline = time.monotonic() + _STARTUP_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if _port_open(host, port):
            return True
        time.sleep(0.2)
    return False if proc.poll() is not None else None


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_connect(args):
    """Connect to a remote Nexus daemon."""
    target = args.url
    # Normalize URL
    if not target.startswith("http://") and not target.startswith("https://"):
        target = f"http://{target}"
    target = target.rstrip("/")

    # Ping to verify reachability
    try:
        req = urllib.request.Request(f"{target}/api/status")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error: cannot reach Nexus daemon at {target}\n  ({e})", file=sys.stderr)
        return 1

    # Save remote config
    ensure_nexus_dir()
    REMOTE_CONFIG_PATH.write_text(json.dumps({"daemon_url": target}))

    daemon = data.get("daemon", {})
    pid = daemon.get("pid", "?")
    uptime = _format_duration(daemon.get("uptime"))
    print(f"Connected to remote Nexus daemon at {target}")
    print(f"  PID: {pid}, uptime: {uptime}")
    print("All nexus/nx commands now target this remote daemon.")
    print("Use 'nexus disconnect' to return to local mode.")
    return 0


def cmd_disconnect(args):
    """Disconnect from a remote Nexus daemon."""
    if REMOTE_CONFIG_PATH.exists():
        REMOTE_CONFIG_PATH.unlink()
        print("Disconnected from remote Nexus daemon.")
    else:
        print("Not connected to a remote daemon.")
    return 0


def cmd_start(args):
    """Start the Nexus daemon."""
    ensure_nexus_dir()
    config = load_config()

    # Starting a local daemon means going local: clear stale remote state
    if REMOTE_CONFIG_PATH.exists():
        REMOTE_CONFIG_PATH.unlink()

    port = args.port or config.daemon.port
    host = args.host or config.daemon.host

    # Check if daemon is already running
    existing_pid = _daemon_pid()
    if existing_pid is not None:
        print(f"Nexus daemon is already running (PID: {existing_pid}).")
        print(f"Stop it first with: nexus stop")
        return 1

    # The daemon cannot do anything without Hive; fail here with a clear
    # message instead of in the daemon's log.
    if not config.hive.url:
        print("Error: hive.url is not configured.\n"
              "  Set it in config.yaml or export HIVE_URL=http://<hive-host>:<port>",
              file=sys.stderr)
        return 1

    if args.foreground:
        # Foreground mode: serve from this process
        config.daemon.host = host
        config.daemon.port = port
        print(f"Starting Nexus daemon at http://{host}:{port}")
        print("Press Ctrl+C to stop.")
        DAEMON_PID_PATH.write_text(str(os.getpid()))
        try:
            from .daemon import serve
            serve(config)
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            if DAEMON_PID_PATH.exists():
                DAEMON_PID_PATH.unlink()
        return 0
    else:
        # Background mode (default): spawn the daemon as a detached
        # process, record its PID, send its output to the log file.
        log_path = NEXUS_DIR / "daemon.log"
        probe_host = _probe_host(host)
        if _port_open(probe_host, port):
            print(f"Error: port {port} on {probe_host} is already in use.",
                  file=sys.stderr)
            return 1

        log_fh = open(str(log_path), "w")
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "nexus.daemon",
                 "--host", host, "--port", str(port)],
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
                cwd=str(NEXUS_DIR.parent),
            )
        finally:
            log_fh.close()

        DAEMON_PID_PATH.write_text(str(proc.pid))

        ready = _wait_for_daemon(proc, probe_host, port)
        if ready is False:
            print(f"Error: daemon exited during startup (exit code "
                  f"{proc.returncode}). Check the log at:", file=sys.stderr)
            print(f"  {log_path}", file=sys.stderr)
            if DAEMON_PID_PATH.exists():
                DAEMON_PID_PATH.unlink()
            return 1

        print(f"Nexus daemon started (PID: {proc.pid})")
        print(f"  URL: http://{host}:{port}")
        print(f"  Log: {log_path}")
        if ready is None:
            print(f"  Warning: not accepting connections after "
                  f"{int(_STARTUP_TIMEOUT_SEC)}s; check the log.")
        print("Use 'nexus status' to check, 'nexus stop' to shut down.")
        return 0


def cmd_stop(args):
    """Stop the local Nexus daemon and delete all project and job records.

    Jobs that are still running on Hive are not cancelled.
    """
    # Cannot stop a remote daemon
    if _get_remote_url():
        print("Error: cannot stop a remote daemon. Use 'nexus disconnect' first,\n"
              "  then stop it from the node where it's running.", file=sys.stderr)
        return 1

    if not DAEMON_PID_PATH.exists():
        print("No Nexus daemon running (no PID file found).")
        return 0

    try:
        pid = int(DAEMON_PID_PATH.read_text().strip())
    except ValueError:
        print("Invalid PID file. Removing it.")
        DAEMON_PID_PATH.unlink()
        return 0

    # Check if process exists
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        print(f"Daemon process {pid} is not running. Cleaning up PID file.")
        DAEMON_PID_PATH.unlink()
        return 0

    # Send SIGTERM and wait
    os.kill(pid, signal.SIGTERM)
    print(f"Sent SIGTERM to daemon (PID: {pid}). Waiting...")

    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.2)
        except ProcessLookupError:
            break

    # Check if it exited
    try:
        os.kill(pid, 0)
        print(f"Warning: daemon (PID: {pid}) did not exit within 5 seconds.")
        print("You may need to kill it manually: kill -9", pid)
    except ProcessLookupError:
        print("Nexus daemon stopped.")

    if DAEMON_PID_PATH.exists():
        DAEMON_PID_PATH.unlink()

    # Delete all state (projects, jobs): the next start begins empty
    import shutil
    from .config import PROJECTS_DIR, JOBS_DIR
    for state_dir in (PROJECTS_DIR, JOBS_DIR):
        if state_dir.exists():
            shutil.rmtree(state_dir)
            state_dir.mkdir(parents=True)
    print("Cleared all project and job state.")

    # Also remove remote.json if it exists
    if REMOTE_CONFIG_PATH.exists():
        REMOTE_CONFIG_PATH.unlink()

    return 0


def cmd_status(args):
    """Show daemon status, Hive connection, and summary stats."""
    base_url = _resolve_base_url()
    data = _api_get(base_url, "/api/status")

    # Daemon info
    daemon = data.get("daemon", {})
    pid = daemon.get("pid", "?")
    uptime = _format_duration(daemon.get("uptime"))
    print(f"Nexus daemon: running (pid {pid}, uptime {uptime})")
    print(f"  Dashboard: {base_url}")

    # Hive connection
    hive = data.get("hive", {})
    hive_url = hive.get("url", "?")
    if not hive.get("connected"):
        print(f"Hive connection: {hive_url} (ERROR: unreachable)")
    else:
        total = hive.get("total_gpus", 0)
        avail = hive.get("available_gpu", 0)
        gpu_nodes = hive.get("num_gpu_nodes", 0)
        cpu_nodes = hive.get("num_cpu_nodes", 0)
        print(f"Hive connection: {hive_url} (OK)")
        print(f"  GPUs: {avail}/{total} available ({gpu_nodes} gpu node, {cpu_nodes} cpu node)")

    # Project summary. A project is "registered" until its first job is
    # submitted and "active" afterwards, so registered = total - active.
    projects = data.get("projects", {})
    active = projects.get("active", 0)
    budget_exhausted = projects.get("budget_exhausted", 0)
    registered = projects.get("total", 0) - active
    parts = []
    if active:
        parts.append(f"{active} active")
    if registered:
        parts.append(f"{registered} registered")
    if budget_exhausted:
        parts.append(f"{budget_exhausted} budget exhausted")
    print(f"Projects: {', '.join(parts) if parts else 'none'}")

    # Job summary
    jobs = data.get("jobs", {})
    running = jobs.get("running", 0)
    completed = jobs.get("completed", 0)
    failed = jobs.get("failed", 0)
    print(f"Jobs: {running} running, {completed} completed, {failed} failed")

    return 0


def cmd_register(args):
    """Register a new project."""
    base_url = _resolve_base_url()

    payload = {
        "project_id": args.project_id,
        "max_gpus": args.max_gpus,
        "workspace_dir": args.workspace,
        "sif_path": args.sif,
        "allowed_resources": args.resources,
    }
    if args.job_timeout is not None:
        payload["default_job_timeout"] = args.job_timeout
    if args.gpu_budget is not None:
        payload["soft_gpu_budget_hours"] = args.gpu_budget
    if args.wake_command:
        payload["wake_command"] = args.wake_command
    if args.shared_bind:
        payload["shared_dirs"] = args.shared_bind
    if args.checkpoint_interval is not None:
        payload["checkpoint_interval"] = args.checkpoint_interval
    if args.checkpoint_timeout is not None:
        payload["checkpoint_timeout"] = args.checkpoint_timeout

    _api_post(base_url, "/api/projects", payload)
    print(f"Project '{args.project_id}' registered.")
    return 0


def cmd_projects(args):
    """List all projects."""
    base_url = _resolve_base_url()
    data = _api_get(base_url, "/api/projects")

    if isinstance(data, list):
        projects = data
    else:
        projects = data.get("projects", [])

    if not projects:
        print("No projects registered.")
        return 0

    # Header
    print()
    print(f"{'PROJECT':<24s} {'STATUS':<12s} {'GPUs (in-use/max)':<20s} "
          f"{'GPU BUDGET':<24s} {'JOBS (run/done/fail)'}")
    print("-" * 110)

    for p in projects:
        project_id = p.get("project_id", "?")
        status = p.get("status", "?")
        gpus_in_use = p.get("gpus_in_use", 0)
        max_gpus = p.get("max_gpus", 0)
        gpu_str = f"{gpus_in_use}/{max_gpus}"

        # GPU budget
        budget = p.get("soft_gpu_budget_hours", 0)
        used_hours = p.get("gpu_seconds_used", 0) / 3600
        if budget > 0:
            budget_str = f"{used_hours:.1f} / {budget:.1f} gpu-h"
        elif used_hours > 0:
            budget_str = f"{used_hours:.1f} gpu-h (unlimited)"
        else:
            budget_str = "unlimited"

        # Job counts
        job_counts = p.get("job_counts", {})
        running = job_counts.get("running", 0)
        completed = job_counts.get("completed", 0)
        failed = job_counts.get("failed", 0)
        jobs_str = f"{running}/{completed}/{failed}"

        print(f"{project_id:<24s} {status:<12s} {gpu_str:<20s} "
              f"{budget_str:<24s} {jobs_str}")

    print()
    return 0


def cmd_project(args):
    """Show detailed project info."""
    base_url = _resolve_base_url()
    data = _api_get(base_url, f"/api/projects/{args.project_id}")

    print()
    print(f"Project: {data.get('project_id', '?')}")
    print(f"  Status:              {data.get('status', '?')}")
    print(f"  Max GPUs:            {data.get('max_gpus', '?')}")
    print(f"  GPUs in use:         {data.get('gpus_in_use', 0)}")
    resources = data.get("allowed_resources", [])
    if resources:
        print(f"  Resources:           {', '.join(resources)}")
    print(f"  Job timeout:         {data.get('default_job_timeout', 3600)}s")

    # GPU budget
    budget = data.get("soft_gpu_budget_hours", 0)
    used_hours = data.get("gpu_seconds_used", 0) / 3600
    if budget > 0:
        print(f"  GPU budget:          {used_hours:.1f} / {budget:.1f} gpu-hours used (soft limit)")
    else:
        if used_hours > 0:
            print(f"  GPU budget:          {used_hours:.1f} gpu-hours used (unlimited)")
        else:
            print(f"  GPU budget:          unlimited")
    if data.get("budget_exhausted"):
        print(f"  Budget exhausted:    YES")

    print(f"  Workspace:           {data.get('workspace_dir') or '-'}")
    print(f"  SIF path:            {data.get('sif_path') or '-'}")
    if data.get("wake_command"):
        print(f"  Wake command:        {data['wake_command']}")
    if data.get("shared_dirs"):
        for bind in data["shared_dirs"]:
            print(f"  Shared bind:         {bind}")
    if data.get("checkpoint_interval"):
        print(f"  Checkpoint interval: {data['checkpoint_interval']}s")
    if data.get("checkpoint_timeout"):
        print(f"  Checkpoint timeout:  {data['checkpoint_timeout']}s")

    created_at = data.get("created_at")
    if created_at:
        print(f"  Created:             {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_at))}")

    print()
    return 0


def cmd_extend(args):
    """Extend a project's soft GPU budget."""
    base_url = _resolve_base_url()

    # Get current project to compute new budget
    project = _api_get(base_url, f"/api/projects/{args.project_id}")
    current_budget = project.get("soft_gpu_budget_hours", 0)
    # 0 means unlimited; adding to it would turn it into a small finite
    # budget that the project may already have used up.
    if current_budget <= 0:
        print(f"Project '{args.project_id}' has an unlimited GPU budget; nothing to extend.")
        return 1
    new_budget = current_budget + args.gpu_hours

    _api_patch(base_url, f"/api/projects/{args.project_id}", {
        "soft_gpu_budget_hours": new_budget,
    })
    print(f"Project '{args.project_id}' GPU budget extended: {current_budget}h -> {new_budget}h (+{args.gpu_hours}h)")
    return 0


def cmd_modify(args):
    """Modify a project's settings."""
    base_url = _resolve_base_url()

    payload = {}
    if args.max_gpus is not None:
        payload["max_gpus"] = args.max_gpus

    if not payload:
        print("Nothing to modify. Specify at least one option (e.g., --max-gpus).")
        return 1

    _api_patch(base_url, f"/api/projects/{args.project_id}", payload)
    changes = ", ".join(f"{k}={v}" for k, v in payload.items())
    print(f"Project '{args.project_id}' modified: {changes}")
    return 0


def cmd_deregister(args):
    """Deregister a project."""
    base_url = _resolve_base_url()

    _api_delete(base_url, f"/api/projects/{args.project_id}")
    print(f"Project '{args.project_id}' deregistered.")
    return 0


def cmd_jobs(args):
    """List jobs across projects."""
    base_url = _resolve_base_url()

    if args.project:
        # Jobs for a specific project
        data = _api_get(base_url, f"/api/projects/{args.project}/jobs")
        if isinstance(data, list):
            jobs = data
        else:
            jobs = data.get("jobs", [])
    else:
        # All jobs across all projects
        projects_data = _api_get(base_url, "/api/projects")
        if isinstance(projects_data, list):
            projects = projects_data
        else:
            projects = projects_data.get("projects", [])

        jobs = []
        for p in projects:
            pid = p.get("project_id")
            if pid:
                proj_data = _api_get(base_url, f"/api/projects/{pid}/jobs")
                if isinstance(proj_data, list):
                    proj_jobs = proj_data
                else:
                    proj_jobs = proj_data.get("jobs", [])
                jobs.extend(proj_jobs)

    # Filter by status
    if args.status:
        filter_statuses = set(args.status.split(","))
        jobs = [j for j in jobs if j.get("status") in filter_statuses]

    if not jobs:
        print("No jobs found.")
        return 0

    # Sort by submitted_at descending
    jobs.sort(key=lambda j: j.get("submitted_at", 0), reverse=True)

    print()
    print(f"{'RUN_ID':<12s} {'PROJECT':<24s} {'STATUS':<18s} {'GPUs':<6s} "
          f"{'ELAPSED':<10s} {'COMMAND'}")
    print("-" * 110)

    for j in jobs:
        run_id = j.get("run_id", "?")
        project_id = j.get("project_id", "?")
        status = j.get("status", "?")
        num_gpus = j.get("num_gpus", 0)
        cmd = j.get("command", "")
        cmd_display = cmd[:40] + "..." if len(cmd) > 43 else cmd

        # Elapsed time
        started_at = j.get("started_at")
        completed_at = j.get("completed_at")
        if started_at:
            end = completed_at or time.time()
            elapsed_str = _format_duration(end - started_at)
        else:
            elapsed_str = "-"

        # Add exit code for completed jobs
        returncode = j.get("returncode")
        if status == "completed" and returncode is not None:
            status_display = f"{status} (exit {returncode})"
        else:
            status_display = status

        print(f"{run_id:<12s} {project_id:<24s} {status_display:<18s} {num_gpus:<6d} "
              f"{elapsed_str:<10s} {cmd_display}")

    print()
    return 0


def cmd_logs(args):
    """Show logs for a specific job."""
    base_url = _resolve_base_url()

    data = _api_get(base_url, f"/api/jobs/{args.run_id}")

    print(f"Job {data.get('run_id', '?')} ({data.get('status', '?')})")
    print(f"  Project:   {data.get('project_id', '?')}")
    print(f"  Command:   {data.get('command', '?')}")
    print(f"  GPUs:      {data.get('num_gpus', '?')}")

    submitted_at = data.get("submitted_at")
    if submitted_at:
        print(f"  Submitted: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(submitted_at))}")
    started_at = data.get("started_at")
    if started_at:
        print(f"  Started:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_at))}")
    completed_at = data.get("completed_at")
    if completed_at:
        print(f"  Completed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(completed_at))}")
    if data.get("returncode") is not None:
        print(f"  Exit code: {data['returncode']}")
    print()

    stdout = data.get("stdout", "")
    stderr = data.get("stderr", "")
    if stdout:
        print("-- stdout --")
        print(stdout, end="")
        if not stdout.endswith("\n"):
            print()
    if stderr:
        print("-- stderr --")
        print(stderr, end="")
        if not stderr.endswith("\n"):
            print()

    status = data.get("status", "")
    if status == "queued":
        print("(job is queued, waiting for resources)")
    elif status == "running":
        print("(job is still running)")
    elif status == "awaiting_response":
        print("(job is at a checkpoint, awaiting a continue/cancel decision)")
    else:
        # The daemon serves output only while a job is active.
        print("(job has ended; its final output is in "
              f"<workspace>/jobs/{data.get('run_id', args.run_id)}/)")

    return 0


def cmd_cancel(args):
    """Cancel a job or all jobs in a project."""
    base_url = _resolve_base_url()

    if args.run_id:
        # Cancel a specific job
        _api_post(base_url, f"/api/jobs/{args.run_id}/cancel")
        print(f"Job '{args.run_id}' cancelled.")
        return 0

    if args.project:
        # Cancel all running jobs in a project
        data = _api_get(base_url, f"/api/projects/{args.project}/jobs")
        if isinstance(data, list):
            jobs = data
        else:
            jobs = data.get("jobs", [])

        cancelled = 0
        for j in jobs:
            if j.get("status") in ("queued", "running", "awaiting_response"):
                run_id = j.get("run_id")
                if run_id:
                    _api_post(base_url, f"/api/jobs/{run_id}/cancel")
                    print(f"  Cancelled job '{run_id}'")
                    cancelled += 1

        if cancelled == 0:
            print(f"No running jobs found in project '{args.project}'.")
        else:
            print(f"Cancelled {cancelled} job(s) in project '{args.project}'.")
        return 0

    print("Error: specify a run_id or --project.", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the ``nexus`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="Nexus admin CLI: manage the daemon, projects and jobs",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- nexus connect -----------------------------------------------------
    p_connect = subparsers.add_parser("connect", help="Connect to a remote Nexus daemon")
    p_connect.add_argument("url", help="Remote daemon address (e.g. nexus-host:39000)")
    p_connect.set_defaults(func=cmd_connect)

    # -- nexus disconnect --------------------------------------------------
    p_disconnect = subparsers.add_parser("disconnect", help="Disconnect from remote daemon")
    p_disconnect.set_defaults(func=cmd_disconnect)

    # -- nexus start -------------------------------------------------------
    p_start = subparsers.add_parser("start", help="Start the Nexus daemon")
    p_start.add_argument("--foreground", action="store_true",
                         help="Run in foreground (default: background daemon)")
    p_start.add_argument("--port", type=int, default=None,
                         help="HTTP port (default: from config)")
    p_start.add_argument("--host", type=str, default=None,
                         help="Bind address (default: from config)")
    p_start.set_defaults(func=cmd_start)

    # -- nexus stop --------------------------------------------------------
    p_stop = subparsers.add_parser(
        "stop",
        help="Stop the local daemon and delete all project and job records")
    p_stop.set_defaults(func=cmd_stop)

    # -- nexus status ------------------------------------------------------
    p_status = subparsers.add_parser("status", help="Show daemon and cluster status")
    p_status.set_defaults(func=cmd_status)

    # -- nexus register ----------------------------------------------------
    p_register = subparsers.add_parser("register", help="Register a new project")
    p_register.add_argument("--project-id", required=True, help="Unique project identifier")
    p_register.add_argument("--max-gpus", type=int, required=True,
                            help="Maximum concurrent GPUs for this project")
    p_register.add_argument("--workspace", required=True,
                            help="Agent workspace directory, on a filesystem "
                                 "shared with the compute nodes")
    p_register.add_argument("--sif", required=True,
                            help="Path to the Apptainer SIF image")
    p_register.add_argument("--resource", action="append", required=True,
                            dest="resources",
                            help="Allowed Hive resource name (repeatable, at least one required)")
    p_register.add_argument("--job-timeout", type=int, default=None,
                            help="Default per-job timeout in seconds (default: from config)")
    p_register.add_argument("--gpu-budget", type=float, default=None,
                            help="Soft GPU-time budget in hours (0 = unlimited, default: from config)")
    p_register.add_argument("--wake-command", type=str, default=None,
                            help="Shell command to wake the agent at checkpoints")
    p_register.add_argument("--shared-bind", action="append", default=None,
                            help="Additional bind mount (repeatable, e.g. /shared:/shared:ro)")
    p_register.add_argument("--checkpoint-interval", type=int, default=None,
                            help="Seconds between checkpoints (overrides global config)")
    p_register.add_argument("--checkpoint-timeout", type=int, default=None,
                            help="Seconds to wait for agent checkpoint response")
    p_register.set_defaults(func=cmd_register)

    # -- nexus projects ----------------------------------------------------
    p_projects = subparsers.add_parser("projects", help="List all projects")
    p_projects.set_defaults(func=cmd_projects)

    # -- nexus project <id> ------------------------------------------------
    p_project = subparsers.add_parser("project", help="Show project details")
    p_project.add_argument("project_id", help="Project ID to show")
    p_project.set_defaults(func=cmd_project)

    # -- nexus extend <id> -------------------------------------------------
    p_extend = subparsers.add_parser("extend", help="Extend a project's GPU budget")
    p_extend.add_argument("project_id", help="Project ID to extend")
    p_extend.add_argument("--gpu-hours", type=float, required=True,
                          help="GPU-hours to add to the soft budget")
    p_extend.set_defaults(func=cmd_extend)

    # -- nexus modify <id> -------------------------------------------------
    p_modify = subparsers.add_parser("modify", help="Change a project's GPU limit")
    p_modify.add_argument("project_id", help="Project ID to modify")
    p_modify.add_argument("--max-gpus", type=int, default=None,
                          help="New maximum concurrent GPUs")
    p_modify.set_defaults(func=cmd_modify)

    # -- nexus deregister <id> ---------------------------------------------
    p_deregister = subparsers.add_parser(
        "deregister",
        help="Delete a project: cancel its active jobs, remove its records")
    p_deregister.add_argument("project_id", help="Project ID to deregister")
    p_deregister.set_defaults(func=cmd_deregister)

    # -- nexus jobs --------------------------------------------------------
    p_jobs = subparsers.add_parser("jobs", help="List jobs")
    p_jobs.add_argument("--project", type=str, default=None,
                        help="Filter by project ID")
    p_jobs.add_argument("--status", type=str, default=None,
                        help="Filter by status (comma-separated)")
    p_jobs.set_defaults(func=cmd_jobs)

    # -- nexus logs <run_id> -----------------------------------------------
    p_logs = subparsers.add_parser("logs", help="Show job logs")
    p_logs.add_argument("run_id", help="Job run ID")
    p_logs.set_defaults(func=cmd_logs)

    # -- nexus cancel ------------------------------------------------------
    p_cancel = subparsers.add_parser("cancel", help="Cancel a job or all jobs in a project")
    p_cancel.add_argument("run_id", nargs="?", default=None,
                          help="Job run ID to cancel")
    p_cancel.add_argument("--project", type=str, default=None,
                          help="Cancel all running jobs in this project")
    p_cancel.set_defaults(func=cmd_cancel)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
