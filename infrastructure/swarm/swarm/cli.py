"""Swarm CLI — start, stop, submit, status, papers, logs.

Entry point for the ``swarm`` command. Follows the argparse subcommand
pattern from hive/cli.py and nexus/cli.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from swarm.config import (
    DAEMON_PID_PATH,
    PAPERS_DIR,
    SWARM_DIR,
    ensure_swarm_dir,
    load_config,
)

log = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Daemon helpers ────────────────────────────────────────────────────────

def _read_pid() -> int | None:
    """Read the daemon PID file. Returns None if not running."""
    if not DAEMON_PID_PATH.exists():
        return None
    try:
        pid = int(DAEMON_PID_PATH.read_text().strip())
        os.kill(pid, 0)  # check if alive
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        DAEMON_PID_PATH.unlink(missing_ok=True)
        return None


def _is_running() -> bool:
    return _read_pid() is not None


# ── Commands ──────────────────────────────────────────────────────────────

def _worker_missing() -> str | None:
    """Explain a Swarm that was installed away from the Worker, or return None.

    The scheduler runs the Worker package of the repository checkout
    (``infrastructure/worker``). An editable install from a clone keeps the
    two together; a regular ``pip install`` copies Swarm alone.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if (repo_root / "infrastructure" / "worker" / "worker.py").is_file():
        return None
    return (
        "the Worker package (infrastructure/worker) is not next to this installation "
        "of Swarm. Install Swarm in editable mode from a clone of the repository: "
        "pip install -e infrastructure/swarm"
    )


def _validate_config(config) -> str | None:
    """Check required config fields. Returns error message or None."""
    if not config.nexus.url:
        return "nexus.url is not configured. Set it in config.yaml or $NEXUS_URL."
    if not config.scout_data_dir:
        return "scout_data_dir is not configured. Set it in config.yaml or $SCOUT_DATA_DIR."
    if not config.workspace_base_dir:
        return "workspace_base_dir is not configured."
    if not config.nexus.resources.all_resources():
        return "nexus.resources has no resource names. Configure at least one resource category."
    if config.default_workflow and config.default_workflow not in config.workflows:
        return (
            f"default_workflow '{config.default_workflow}' is not defined under workflows. "
            f"Define it, or set default_workflow: \"\" to submit papers without a workflow."
        )
    return None


def _config_warnings(config) -> list[str]:
    """Describe settings that let the daemon start but limit what it can do."""
    from swarm.workspace import _get_github_token

    warnings = []
    if not config.github_org:
        warnings.append(
            "github_org is not set. Workspaces stay local; every iteration then "
            "ends with status 'stalled' (git push failed)."
        )
    elif not _get_github_token():
        warnings.append(
            "no GitHub token found in ~/.git-credentials. Repositories cannot be "
            "created; every iteration then ends with status 'stalled' (git push failed)."
        )
    if not config.worker.container_sif:
        warnings.append(
            "worker.container_sif is not set. Agents run directly on this host, "
            "without container isolation."
        )
    if not config.worker.scaffold or not config.worker.model:
        warnings.append(
            "no default worker (worker.profile) is set. Papers that are neither "
            "given --profile nor run a workflow will fail."
        )
    if not Path(config.local_tmp_dir).is_dir():
        warnings.append(
            f"local_tmp_dir {config.local_tmp_dir} does not exist. Container "
            "builds and overlay setup will fail."
        )
    return warnings


def _daemon_script(host: str, port: int) -> str:
    """Python source run by the background daemon process.

    It rebuilds the scheduler from the config on disk and serves the
    dashboard on ``host:port``.
    """
    return (
        "from swarm.dashboard import create_app\n"
        "from swarm.scheduler import Scheduler\n"
        "from swarm.config import load_config, ensure_swarm_dir, DAEMON_PID_PATH\n"
        "import logging, os, signal, sys\n"
        "logging.basicConfig(level=logging.DEBUG,"
        " format='%(asctime)s %(levelname)-7s %(name)s  %(message)s',"
        " datefmt='%H:%M:%S')\n"
        "ensure_swarm_dir()\n"
        "config = load_config()\n"
        "scheduler = Scheduler(config)\n"
        "app = create_app(config, scheduler)\n"
        "def _shutdown(sig, frame):\n"
        "    scheduler.shutdown()\n"
        "    DAEMON_PID_PATH.unlink(missing_ok=True)\n"
        "    os._exit(0)\n"
        "signal.signal(signal.SIGTERM, _shutdown)\n"
        "signal.signal(signal.SIGINT, _shutdown)\n"
        f"app.run(host={host!r}, port={port}, debug=False, use_reloader=False)\n"
    )


def cmd_start(args) -> int:
    """Start the Swarm daemon (dashboard + scheduler)."""
    if _is_running():
        print(f"Swarm is already running (PID {_read_pid()})")
        return 1

    err = _worker_missing()
    if err:
        print(f"Error: {err}")
        return 1

    ensure_swarm_dir()
    config = load_config()

    err = _validate_config(config)
    if err:
        print(f"Error: {err}")
        return 1
    for warning in _config_warnings(config):
        print(f"Warning: {warning}")

    host = config.dashboard_host
    port = args.port or config.dashboard_port

    if args.foreground:
        # Foreground mode: run directly (useful for debugging)
        setup_logging(args.verbose)
        from swarm.dashboard import create_app
        from swarm.scheduler import Scheduler

        scheduler = Scheduler(config)
        app = create_app(config, scheduler)

        DAEMON_PID_PATH.write_text(str(os.getpid()))

        def signal_handler(sig, frame):
            print("\nShutting down Swarm...")
            scheduler.shutdown()
            DAEMON_PID_PATH.unlink(missing_ok=True)
            os._exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print(f"Swarm started on http://{host}:{port}")
        print(f"  Nexus: {config.nexus.url}")
        print(f"  Working workers: {config.effective_max_working_workers()}")
        if config.effective_max_working_return_overflow() > 0:
            print(f"  Working return overflow: +{config.effective_max_working_return_overflow()}")
        if config.split_runtime_caps_enabled():
            print(f"  Admitted workers: {config.effective_max_admitted_workers()}")
            print(f"  Building workers: {config.effective_max_building_workers()}")
            print(f"  Installing workers: {config.effective_max_installing_workers()}")
        else:
            print(f"  Runtime workers: {config.effective_max_runtime_workers()}")
        print("Press Ctrl+C to stop.")

        try:
            app.run(host=host, port=port, debug=False, use_reloader=False)
        finally:
            scheduler.shutdown()
            DAEMON_PID_PATH.unlink(missing_ok=True)
        return 0
    else:
        # Background mode (default): fork a subprocess
        log_path = SWARM_DIR / "daemon.log"
        script = _daemon_script(host, port)
        # Append mode so daemon restarts keep the earlier log.
        # Add a marker line so restart boundaries are visible in the log.
        log_fh = open(str(log_path), "a")
        log_fh.write(f"\n=== swarm daemon starting at {time.strftime('%Y-%m-%d %H:%M:%S %Z')} (pid pending) ===\n")
        log_fh.flush()
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
                cwd=str(SWARM_DIR.parent),
            )
        finally:
            log_fh.close()

        DAEMON_PID_PATH.write_text(str(proc.pid))

        # Wait briefly and verify it started
        time.sleep(1)
        try:
            os.kill(proc.pid, 0)
        except ProcessLookupError:
            print("Error: daemon failed to start. Check logs at:", file=sys.stderr)
            print(f"  {log_path}", file=sys.stderr)
            DAEMON_PID_PATH.unlink(missing_ok=True)
            return 1

        print(f"Swarm daemon started (PID: {proc.pid})")
        print(f"  Dashboard: http://{host}:{port}")
        print(f"  Nexus:     {config.nexus.url}")
        print(f"  Working workers: {config.effective_max_working_workers()}")
        if config.effective_max_working_return_overflow() > 0:
            print(f"  Working return overflow: +{config.effective_max_working_return_overflow()}")
        if config.split_runtime_caps_enabled():
            print(f"  Admitted workers: {config.effective_max_admitted_workers()}")
            print(f"  Building workers: {config.effective_max_building_workers()}")
            print(f"  Installing workers: {config.effective_max_installing_workers()}")
        else:
            print(f"  Runtime workers: {config.effective_max_runtime_workers()}")
        print(f"  Log:       {log_path}")
        print("Use 'swarm status' to check, 'swarm stop' to shut down.")
        return 0


def cmd_stop(args) -> int:
    """Stop the Swarm daemon."""
    pid = _read_pid()
    if pid is None:
        print("Swarm is not running")
        return 1

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to Swarm (PID {pid})")
        # Wait briefly for clean shutdown
        for _ in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except ProcessLookupError:
                break
        else:
            # Still alive: keep the PID file so `swarm start` does not
            # launch a second daemon next to it.
            print(f"Swarm (PID {pid}) is still shutting down; "
                  f"run `swarm stop` again or check `swarm status`")
            return 1
        DAEMON_PID_PATH.unlink(missing_ok=True)
        print("Swarm stopped")
    except ProcessLookupError:
        DAEMON_PID_PATH.unlink(missing_ok=True)
        print("Swarm was not running (stale PID file removed)")
    except PermissionError:
        print(f"Cannot stop Swarm (PID {pid}): permission denied")
        return 1

    return 0


def cmd_submit(args) -> int:
    """Submit a paper to the Swarm queue via the dashboard API."""
    import urllib.error
    import urllib.request

    config = load_config()
    port = config.dashboard_port
    url = f"http://localhost:{port}"

    payload = {"arxiv_id": args.arxiv_id}
    if args.gpu_budget is not None:
        payload["gpu_budget"] = args.gpu_budget
    if args.date is not None:
        payload["date"] = args.date
    if args.topic:
        payload["topics"] = args.topic
    if args.profile is not None:
        payload["profile"] = args.profile
    if args.workflow is not None:
        payload["workflow"] = args.workflow

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/api/papers",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
        print(f"Submitted: {result.get('arxiv_id')} (status: {result.get('status')})")
        return 0
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode())
            print(f"Error: {err.get('error', e)}")
        except Exception:
            print(f"Error: HTTP {e.code}")
        return 1
    except (urllib.error.URLError, OSError) as e:
        print(f"Error: cannot reach Swarm at {url} ({e})")
        print("Is the Swarm daemon running? Start it with: swarm start")
        return 1


def cmd_continue(args) -> int:
    """Continue a previously started paper (resume session)."""
    import urllib.error
    import urllib.request

    config = load_config()
    port = config.dashboard_port
    url = f"http://localhost:{port}"

    payload = {}
    if getattr(args, "profile", None) is not None:
        payload["profile"] = args.profile
    if getattr(args, "no_workflow", False):
        payload["no_workflow"] = True
    if getattr(args, "fresh", False):
        payload["fresh"] = True
    req = urllib.request.Request(
        f"{url}/api/papers/{args.arxiv_id}/continue",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
        print(f"Continuing: {result.get('arxiv_id')} (status: {result.get('status')})")
        return 0
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode())
            print(f"Error: {err.get('error', e)}")
        except Exception:
            print(f"Error: HTTP {e.code}")
        return 1
    except (urllib.error.URLError, OSError) as e:
        print(f"Error: cannot reach Swarm at {url} ({e})")
        return 1


def cmd_continue_all(args) -> int:
    """Continue all papers that were interrupted by a daemon restart or pause-all."""
    import urllib.error
    import urllib.request

    config = load_config()
    port = config.dashboard_port
    url = f"http://localhost:{port}"

    continue_payload = {}
    if getattr(args, "profile", None) is not None:
        continue_payload["profile"] = args.profile
    if getattr(args, "no_workflow", False):
        continue_payload["no_workflow"] = True
    if getattr(args, "fresh", False):
        continue_payload["fresh"] = True

    req = urllib.request.Request(
        f"{url}/api/recover-all",
        data=json.dumps(continue_payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode())
            print(f"Error: {err.get('error', f'HTTP {e.code}')}")
        except Exception:
            print(f"Error: HTTP {e.code}")
        return 1
    except (urllib.error.URLError, OSError) as e:
        print(f"Error: cannot reach Swarm at {url} ({e})")
        return 1

    total = result.get("total", 0)
    if total == 0:
        print("No daemon-interrupted papers found.")
        return 0

    print(f"Recovering {total} daemon-interrupted papers...")
    for item in result.get("results", []):
        if item.get("status") == "failed":
            print(f"  FAIL {item.get('arxiv_id')}: {item.get('error', 'unknown error')}")
        else:
            print(
                f"  OK   {item.get('arxiv_id')} "
                f"({item.get('recovery_action', 'recovered')}, {item.get('status')})"
            )

    failures = result.get("failed", 0)
    successes = result.get("queued", 0)
    print(f"Done: {successes} queued, {failures} failed.")
    return 0 if failures == 0 else 1


def cmd_pause_all(args) -> int:
    """Pause all queued/active papers for continue-all recovery, then stop Swarm."""
    import urllib.error
    import urllib.request

    config = load_config()
    port = config.dashboard_port
    url = f"http://localhost:{port}"

    req = urllib.request.Request(
        f"{url}/api/pause-all",
        data=b"{}",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode())
            print(f"Error: {err.get('error', e)}")
        except Exception:
            print(f"Error: HTTP {e.code}")
        return 1
    except (urllib.error.URLError, OSError) as e:
        print(f"Error: cannot reach Swarm at {url} ({e})")
        print("Is the Swarm daemon running? Start it with: swarm start")
        return 1

    print(
        "Prepared papers for restart recovery "
        f"(queued: {result.get('queued', 0)}, active: {result.get('active', 0)})"
    )
    print("Stopping Swarm daemon...")
    return cmd_stop(args)


def cmd_cancel(args) -> int:
    """Cancel an active or queued paper gracefully."""
    import urllib.error
    import urllib.request

    config = load_config()
    port = config.dashboard_port
    url = f"http://localhost:{port}"

    req = urllib.request.Request(
        f"{url}/api/papers/{args.arxiv_id}/cancel",
        data=b"{}",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
        print(f"Cancelling: {args.arxiv_id} (status: {result.get('status')})")
        return 0
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode())
            print(f"Error: {err.get('error', e)}")
        except Exception:
            print(f"Error: HTTP {e.code}")
        return 1
    except (urllib.error.URLError, OSError) as e:
        print(f"Error: cannot reach Swarm at {url} ({e})")
        return 1


def cmd_iterate(args) -> int:
    """Iterate on a paper with a fresh agent that audits previous work."""
    import urllib.error
    import urllib.request
    config = load_config()
    port = config.dashboard_port
    url = f"http://localhost:{port}"

    payload = {}
    if getattr(args, "profile", None) is not None:
        payload["profile"] = args.profile
    if getattr(args, "gpu_budget", None) is not None:
        payload["gpu_budget"] = args.gpu_budget
    if getattr(args, "workflow_step", None) is not None:
        payload["workflow_step"] = args.workflow_step
    req = urllib.request.Request(
        f"{url}/api/papers/{args.arxiv_id}/iterate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
        print(f"Iterating: {result.get('arxiv_id')} (status: {result.get('status')})")
        return 0
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode())
            print(f"Error: {err.get('error', e)}")
        except Exception:
            print(f"Error: HTTP {e.code}")
        return 1
    except (urllib.error.URLError, OSError) as e:
        print(f"Error: cannot reach Swarm at {url} ({e})")
        return 1


def cmd_verify(args) -> int:
    """Run a verifier agent on a paper (final quality check)."""
    import urllib.error
    import urllib.request
    config = load_config()
    port = config.dashboard_port
    url = f"http://localhost:{port}"

    payload = {"role": "verifier"}
    profile = getattr(args, "profile", None)
    if profile is not None:
        payload["profile"] = profile
    if getattr(args, "gpu_budget", None) is not None:
        payload["gpu_budget"] = args.gpu_budget
    if getattr(args, "workflow_step", None) is not None:
        payload["workflow_step"] = args.workflow_step
    req = urllib.request.Request(
        f"{url}/api/papers/{args.arxiv_id}/iterate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
        print(f"Verifying: {result.get('arxiv_id')} (profile: {profile or 'workflow step or daemon default'})")
        return 0
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode())
            print(f"Error: {err.get('error', e)}")
        except Exception:
            print(f"Error: HTTP {e.code}")
        return 1
    except (urllib.error.URLError, OSError) as e:
        print(f"Error: cannot reach Swarm at {url} ({e})")
        return 1


def cmd_status(args) -> int:
    """Show Swarm status."""
    import urllib.error
    import urllib.request

    config = load_config()
    port = config.dashboard_port
    url = f"http://localhost:{port}"

    try:
        req = urllib.request.Request(f"{url}/api/status")
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = json.loads(resp.read().decode())

        working_cap = str(status.get("max_working_workers", "?"))
        overflow = status.get("max_working_return_overflow", 0)
        if isinstance(overflow, int) and overflow > 0:
            working_cap = f"{working_cap}+{overflow}"

        print(f"Queue:     {status.get('queue_size', '?')}")
        print(f"Active:    {status.get('active_count', '?')} / {status.get('pool_size', '?')}")
        print(
            f"Working:   {status.get('working_count', '?')} / "
            f"{working_cap}"
        )
        print(
            f"Runtime:   {status.get('runtime_count', '?')} / "
            f"{status.get('max_runtime_workers', '?')}"
        )
        if status.get("split_runtime_caps"):
            print(
                f"Building:  {status.get('building_count', '?')} / "
                f"{status.get('max_building_workers', '?')}"
            )
            print(
                f"Installing:{status.get('installing_count', '?')} / "
                f"{status.get('max_installing_workers', '?')}"
            )
            print(
                f"Wait:      {status.get('waiting_count', '?')}"
            )
            print(
                f"Admitted:  {status.get('admitted_count', '?')} / "
                f"{status.get('max_admitted_workers', '?')}"
            )
        print(f"Completed: {status.get('completed_count', '?')}")

        if status.get("shutdown"):
            print("Status:    SHUTTING DOWN")

        return 0
    except (urllib.error.URLError, OSError) as e:
        # Fallback: check PID
        pid = _read_pid()
        if pid:
            print(f"Swarm daemon is running (PID {pid}) but dashboard is not responding")
        else:
            print("Swarm is not running")
        return 1


_CAP_BUCKETS = ("total", "working", "overflow", "admitted", "building", "installing")


def cmd_cap(args) -> int:
    """Show/set/clear runtime soft caps on worker buckets."""
    import urllib.error
    import urllib.request

    config = load_config()
    url = f"http://localhost:{config.dashboard_port}/api/caps"

    try:
        if args.clear:
            # clear all or one
            if args.bucket:
                req = urllib.request.Request(
                    f"{url}?bucket={args.bucket}", method="DELETE"
                )
            else:
                req = urllib.request.Request(url, method="DELETE")
            with urllib.request.urlopen(req, timeout=10) as resp:
                state = json.loads(resp.read().decode())
        elif args.bucket is None:
            # no args: show current
            with urllib.request.urlopen(url, timeout=10) as resp:
                state = json.loads(resp.read().decode())
        else:
            # set
            if args.value is None:
                print("Error: value required when setting a cap", file=sys.stderr)
                return 2
            payload = json.dumps(
                {"bucket": args.bucket, "value": args.value}
            ).encode()
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                state = json.loads(resp.read().decode())

        overrides = state.get("overrides", {})
        effective = state.get("effective", {})
        print(f"{'bucket':<12} {'effective':>10} {'override':>10}")
        for b in _CAP_BUCKETS:
            ov = overrides.get(b)
            ov_str = str(ov) if ov is not None else "-"
            print(f"{b:<12} {effective.get(b, '?'):>10} {ov_str:>10}")
        return 0
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, "read") else ""
        try:
            msg = json.loads(body).get("error", body)
        except Exception:
            msg = body or str(e)
        print(f"Error: {msg}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError) as e:
        print(f"Swarm dashboard not reachable: {e}", file=sys.stderr)
        return 1


def cmd_papers(args) -> int:
    """List all papers with status."""
    import urllib.error
    import urllib.request

    config = load_config()
    port = config.dashboard_port
    url = f"http://localhost:{port}"

    try:
        req = urllib.request.Request(f"{url}/api/papers")
        with urllib.request.urlopen(req, timeout=10) as resp:
            papers = json.loads(resp.read().decode())

        if not papers:
            print("No papers submitted yet")
            return 0

        # Group by phase
        for phase in ("active", "queued", "completed"):
            group = [p for p in papers if p.get("phase") == phase]
            if not group:
                continue

            print(f"\n{phase.upper()} ({len(group)})")
            print("-" * 60)
            for p in group:
                arxiv_id = p.get("arxiv_id", "?")
                milestone = p.get("milestone", "-")
                status = p.get("status", "-")
                line = f"  {arxiv_id:<20s} milestone={milestone:<15s}"
                if phase == "completed":
                    dur = p.get("duration_sec", 0)
                    dur_str = f"{dur / 3600:.1f}h" if dur > 0 else "-"
                    line += f" status={status:<8s} duration={dur_str}"
                elif phase == "active":
                    line += f" worker={p.get('worker_status', '-')}"
                print(line)

        return 0
    except (urllib.error.URLError, OSError) as e:
        print(f"Error: cannot reach Swarm ({e})")
        return 1


def cmd_reset(args) -> int:
    """Remove all state for a paper (workspace + swarm history)."""
    import shutil

    config = load_config()
    arxiv_id = args.arxiv_id

    removed = []

    # Remove workspace
    workspace = Path(config.workspace_base_dir) / arxiv_id
    if workspace.exists():
        shutil.rmtree(str(workspace))
        removed.append(f"workspace: {workspace}")

    # Remove swarm paper state
    paper_state = PAPERS_DIR / arxiv_id
    if paper_state.exists():
        shutil.rmtree(str(paper_state))
        removed.append(f"state:     {paper_state}")

    if removed:
        print(f"Reset paper {arxiv_id}:")
        for r in removed:
            print(f"  removed {r}")
    else:
        print(f"Nothing to reset for {arxiv_id} (no workspace or state found)")

    return 0


def cmd_reset_all(args) -> int:
    """Remove all workspaces, swarm paper state, and logs."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor, as_completed

    config = load_config()

    # List what will be removed
    workspace_base = Path(config.workspace_base_dir)
    workspaces = list(workspace_base.iterdir()) if workspace_base.is_dir() else []
    paper_states = list(PAPERS_DIR.iterdir()) if PAPERS_DIR.is_dir() else []
    log_files = [
        SWARM_DIR / "daemon.log",
    ]

    if not workspaces and not paper_states:
        print("Nothing to reset.")
        return 0

    print("Will remove:")
    print(f"  {len(workspaces)} workspaces in {workspace_base}")
    print(f"  {len(paper_states)} paper records in {PAPERS_DIR}")
    print("  Swarm log (daemon.log)")
    print()

    if not args.yes:
        answer = input("Type 'RESET' to proceed: ")
        if answer != "RESET":
            print("Aborted.")
            return 1

    def _remove(path: Path, label: str) -> str:
        shutil.rmtree(str(path))
        return f"  removed {label} {path.name}"

    # Delete workspaces and paper states in parallel
    all_tasks = [(ws, "workspace") for ws in workspaces] + [(ps, "state") for ps in paper_states]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_remove, p, l): p for p, l in all_tasks}
        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                print(f"  FAILED {futures[future]}: {e}")

    for lf in log_files:
        if lf.exists():
            lf.unlink()
            print(f"  removed {lf.name}")

    print(f"\nReset complete: {len(workspaces)} workspaces, {len(paper_states)} paper records.")
    return 0


def cmd_logs(args) -> int:
    """Show the recorded state and the last iteration's turns for a paper.

    Reads ``SWARM_DIR/papers/<id>/`` directly, so it works without a
    running daemon. ``trajectory.json`` is rewritten at the end of every
    iteration and holds only that iteration's agent turns.
    """
    paper_dir = PAPERS_DIR / args.arxiv_id

    if not paper_dir.exists():
        print(f"No logs found for {args.arxiv_id}")
        return 1

    # Show meta.json
    meta_path = paper_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"Paper:     {meta.get('arxiv_id', '?')}")
        print(f"Status:    {meta.get('status', '?')}")
        print(f"Milestone: {meta.get('milestone', '?')}")
        print(f"Iteration: {meta.get('iteration', '?')} ({meta.get('role', '?')})")
        if meta.get("error"):
            print(f"Error:     {meta['error']}")
        print()

    # Show trajectory if exists
    trajectory_path = paper_dir / "trajectory.json"
    if trajectory_path.exists():
        trajectory = json.loads(trajectory_path.read_text())
        turns = trajectory.get("turns", []) if isinstance(trajectory, dict) else []
        print(f"Trajectory: {len(turns)} turns")
        for turn in turns:
            line = (f"  Turn {turn.get('turn', '?')}: rc={turn.get('returncode', '?')} "
                    f"({float(turn.get('duration_s') or 0):.1f}s)")
            problems = turn.get("errors") or ([turn["error"]] if turn.get("error") else [])
            if problems:
                line += f" errors: {str(problems[0])[:120]}"
            print(line)
    else:
        print("No trajectory file found")

    return 0


# ── Main ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="swarm",
        description="Swarm: builds research environments from papers by "
                    "coordinating builder agents",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start
    p_start = subparsers.add_parser("start", help="Start Swarm daemon")
    p_start.add_argument("--port", type=int, default=None,
                         help="Dashboard port (default: dashboard_port from config; "
                              "the other subcommands always use the config value)")
    p_start.add_argument("--foreground", action="store_true",
                         help="Run in foreground (default: background daemon)")
    p_start.set_defaults(func=cmd_start)

    # stop
    p_stop = subparsers.add_parser("stop", help="Stop Swarm daemon")
    p_stop.set_defaults(func=cmd_stop)

    # pause-all
    p_pause_all = subparsers.add_parser(
        "pause-all",
        help="Pause all queued/active papers for continue-all recovery, then stop Swarm",
    )
    p_pause_all.set_defaults(func=cmd_pause_all)

    # submit
    p_submit = subparsers.add_parser("submit", help="Submit a paper")
    p_submit.add_argument("arxiv_id", type=str, help="arXiv ID (e.g. 2512.01152)")
    p_submit.add_argument("--gpu-budget", type=float, default=None,
                          help="GPU-hours budget override")
    p_submit.add_argument("--date", type=str, default=None,
                          help="Paper date (YYYY-MM-DD), added as GitHub topic")
    p_submit.add_argument("--topic", type=str, action="append", default=[],
                          help="GitHub topic to add (repeatable, e.g. --topic nlp --topic transformer)")
    p_submit.add_argument("--profile", type=str, default=None,
                          help="Profile for the first iteration, a name from the "
                               "config's profiles section (default: the workflow step's profile)")
    p_submit.add_argument("--workflow", type=str, default=None,
                          help="Workflow name from the config (default: default_workflow)")
    p_submit.set_defaults(func=cmd_submit)

    # continue
    p_continue = subparsers.add_parser("continue", help="Resume a previously started paper")
    p_continue.add_argument("arxiv_id", type=str, help="arXiv ID to resume")
    p_continue.add_argument("--profile", type=str, default=None,
                            help="Profile override, a name from the config's profiles section")
    p_continue.add_argument("--no-workflow", action="store_true", dest="no_workflow",
                            help="Detach from workflow — continue as a standalone run")
    p_continue.add_argument("--fresh", action="store_true",
                            help="Clear session storage and start the agent with a new session")
    p_continue.set_defaults(func=cmd_continue)

    # continue-all
    p_continue_all = subparsers.add_parser(
        "continue-all",
        help="Resume all papers interrupted by a daemon restart",
    )
    p_continue_all.add_argument("--profile", type=str, default=None,
                                help="Model profile override applied to every resumed paper")
    p_continue_all.add_argument("--no-workflow", action="store_true", dest="no_workflow",
                                help="Detach all resumed papers from workflows")
    p_continue_all.add_argument("--fresh", action="store_true",
                                help="Clear session storage for every resumed paper")
    p_continue_all.set_defaults(func=cmd_continue_all)

    # cancel
    p_cancel = subparsers.add_parser("cancel", help="Cancel an active or queued paper gracefully")
    p_cancel.add_argument("arxiv_id", type=str, help="arXiv ID to cancel")
    p_cancel.set_defaults(func=cmd_cancel)

    # iterate
    p_iterate = subparsers.add_parser("iterate", help="Iterate on a paper with a fresh agent that audits previous work")
    p_iterate.add_argument("arxiv_id", type=str, help="arXiv ID to iterate on")
    p_iterate.add_argument("--profile", type=str, default=None,
                            help="Profile override, a name from the config's profiles section")
    p_iterate.add_argument("--gpu-budget", type=float, default=None,
                            help="Override the GPU-hours budget for this iteration")
    p_iterate.add_argument("--workflow-step", type=int, default=None,
                            help="Workflow step index (0-based) to run, for recovery or manual advancement")
    p_iterate.set_defaults(func=cmd_iterate)

    # verify
    p_verify = subparsers.add_parser("verify", help="Run a verifier agent for final quality check")
    p_verify.add_argument("arxiv_id", type=str, help="arXiv ID to verify")
    p_verify.add_argument("--profile", type=str, default=None,
                           help="Profile to use (default: the paper's current workflow step, "
                                "or the daemon's default worker)")
    p_verify.add_argument("--gpu-budget", type=float, default=None,
                           help="Override the GPU-hours budget for this iteration")
    p_verify.add_argument("--workflow-step", type=int, default=None,
                           help="Workflow step index (0-based) to run, for recovery or manual advancement")
    p_verify.set_defaults(func=cmd_verify)

    # status
    p_status = subparsers.add_parser("status", help="Show queue and worker status")
    p_status.set_defaults(func=cmd_status)

    # papers
    p_papers = subparsers.add_parser("papers", help="List all papers with status")
    p_papers.set_defaults(func=cmd_papers)

    # cap — runtime soft caps
    p_cap = subparsers.add_parser(
        "cap",
        help="Show/set/clear runtime soft caps on worker buckets "
             "(total, working, overflow, admitted, building, installing)",
    )
    p_cap.add_argument(
        "bucket", nargs="?", choices=_CAP_BUCKETS,
        help="Bucket to set/clear. Omit to show all.",
    )
    p_cap.add_argument(
        "value", nargs="?", type=int,
        help="New cap value. Required when setting.",
    )
    p_cap.add_argument(
        "--clear", action="store_true",
        help="Clear override (for given bucket, or all if bucket omitted).",
    )
    p_cap.set_defaults(func=cmd_cap)

    # reset
    p_reset = subparsers.add_parser("reset", help="Remove all state for a paper (workspace + history)")
    p_reset.add_argument("arxiv_id", type=str, help="arXiv ID to reset")
    p_reset.set_defaults(func=cmd_reset)

    # reset-all
    p_reset_all = subparsers.add_parser("reset-all", help="Remove ALL workspaces, paper state, and logs")
    p_reset_all.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    p_reset_all.set_defaults(func=cmd_reset_all)

    # logs
    p_logs = subparsers.add_parser(
        "logs", help="Show a paper's recorded state and the turns of its last iteration")
    p_logs.add_argument("arxiv_id", type=str, help="arXiv ID")
    p_logs.set_defaults(func=cmd_logs)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command != "start":
        setup_logging(args.verbose)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
