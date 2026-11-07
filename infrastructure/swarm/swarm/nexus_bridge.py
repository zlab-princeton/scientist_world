"""HTTP client for the Nexus project manager.

Stdlib-only (urllib) — no external dependencies. Mirrors the pattern
from nexus/hive_bridge.py but talks to the Nexus daemon API.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger(__name__)


class NexusBridge:
    """Client for the Nexus daemon REST API.

    Args:
        nexus_url: Nexus daemon URL, e.g. ``http://localhost:39000``.
    """

    def __init__(self, nexus_url: str):
        self.url = nexus_url.rstrip("/")

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, timeout: int = 15) -> dict:
        """GET a JSON endpoint."""
        req = urllib.request.Request(f"{self.url}{path}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _post(self, path: str, data: dict, timeout: int = 30) -> dict:
        """POST JSON to an endpoint."""
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _patch(self, path: str, data: dict, timeout: int = 15) -> dict:
        """PATCH JSON to an endpoint."""
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="PATCH",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _delete(self, path: str, timeout: int = 15) -> dict:
        """DELETE a JSON endpoint."""
        req = urllib.request.Request(f"{self.url}{path}", method="DELETE")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    # ------------------------------------------------------------------
    # Project operations
    # ------------------------------------------------------------------

    def create_project(
        self,
        project_id: str,
        workspace_dir: str,
        sif_path: str,
        gpu_budget: float,
        max_gpus: int,
        allowed_resources: list[str] | None = None,
        shared_dirs: list[str] | None = None,
        overlay_path: str = "",
        hf_cache_path: str = "",
    ) -> dict:
        """Register a new project with Nexus.

        Returns:
            Nexus API response dict.

        Raises:
            RuntimeError: If Nexus returns an error.
        """
        if not allowed_resources:
            raise ValueError(
                "allowed_resources must be a non-empty list "
                "(Nexus requires at least one Hive resource name)"
            )
        payload = {
            "project_id": project_id,
            "workspace_dir": workspace_dir,
            "sif_path": sif_path,
            "soft_gpu_budget_hours": gpu_budget,
            "max_gpus": max_gpus,
            "allowed_resources": allowed_resources,
        }
        if overlay_path:
            payload["overlay_path"] = overlay_path
        if hf_cache_path:
            payload["hf_cache_path"] = hf_cache_path
        if shared_dirs:
            payload["shared_dirs"] = shared_dirs

        try:
            result = self._post("/api/projects", payload)
        except urllib.error.HTTPError as exc:
            # Nexus reports errors (e.g. 409 project exists) as non-2xx
            # responses, which urlopen raises instead of returning.
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"Nexus create_project error (HTTP {exc.code}): {body or exc}"
            ) from exc
        if "error" in result:
            raise RuntimeError(f"Nexus create_project error: {result['error']}")
        log.info("Created Nexus project %s", project_id)
        return result

    def list_projects(self) -> list:
        """List all Nexus projects."""
        try:
            return self._get("/api/projects")
        except (urllib.error.URLError, OSError):
            return []

    def delete_project(self, project_id: str) -> dict:
        """Deregister a project.

        Returns:
            Nexus API response dict.
        """
        try:
            return self._delete(f"/api/projects/{project_id}")
        except (urllib.error.URLError, OSError) as exc:
            log.warning("Failed to delete project %s: %s", project_id, exc)
            return {"error": str(exc)}

    def update_project(self, project_id: str, **kwargs) -> dict:
        """Update project fields via PATCH.

        Supported fields: sif_path, max_gpus, allowed_resources, etc.
        """
        return self._patch(f"/api/projects/{project_id}", kwargs)

    def get_project(self, project_id: str) -> dict:
        """Get project details.

        Returns:
            Project info dict, or dict with ``error`` key on failure.
        """
        try:
            return self._get(f"/api/projects/{project_id}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"error": "not found"}
            raise
        except (urllib.error.URLError, OSError) as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Job operations
    # ------------------------------------------------------------------

    def submit_job(
        self,
        project_id: str,
        command: str,
        num_gpus: int = 1,
        timeout_sec: int | None = None,
        resource: str | None = None,
    ) -> tuple[str, str]:
        """Submit a job through Nexus, idempotently riding out Hive backpressure.

        Under heavy concurrent submit load, Hive's synchronous submit path
        can take longer than this client's HTTP timeout. Nexus surfaces
        that as HTTP 5xx (or urllib timeout). A plain retry would
        re-acquire GPU quota on top of the in-flight original, leaking
        ``gpus_in_use`` and producing spurious 403 "max_gpus in use"
        errors.

        One ``client_request_id`` is therefore generated per logical
        submit and sent on every retry. The Nexus daemon deduplicates
        retries with the same id and replays the original response
        without acquiring quota twice.

        Retry classes:
          - HTTP 5xx / urllib errors / OSError: transient, retry
          - HTTP 4xx (except 403 "budget"): permanent, raise
          - 403 with "budget" in body: caller-fixable, retry

        Returns:
            Tuple of ``(run_id, hive_job_id)``. ``hive_job_id`` is "" when
            the daemon's response does not include it.
        """
        import time
        import urllib.error
        import uuid

        client_request_id = str(uuid.uuid4())
        payload: dict = {
            "command": command,
            "num_gpus": num_gpus,
            "client_request_id": client_request_id,
        }
        if timeout_sec is not None:
            payload["timeout_sec"] = timeout_sec
        if resource:
            payload["resource"] = resource

        deadline = time.monotonic() + 600.0  # 10-minute wall cap
        backoff_schedule = [5, 10, 20, 30, 60]
        attempt = 0
        last_exc: Exception | None = None

        while True:
            try:
                result = self._post(
                    f"/api/projects/{project_id}/jobs", payload, timeout=120,
                )
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    body = ""
                    try:
                        body = exc.read().decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    # 403 "budget exhausted" can clear once the caller
                    # bumps soft_gpu_budget_hours; treat as retryable.
                    # Other 4xx (project missing, malformed payload,
                    # truly-exhausted max_gpus) are permanent.
                    if exc.code == 403 and "budget" in body.lower():
                        last_exc = exc
                    else:
                        raise RuntimeError(
                            f"Nexus submit_job rejected "
                            f"(HTTP {exc.code}): {body or exc}"
                        ) from exc
                else:
                    last_exc = exc
            except (urllib.error.URLError, OSError) as exc:
                last_exc = exc
            else:
                if "error" in result:
                    raise RuntimeError(f"Nexus submit error: {result['error']}")
                run_id = result.get("run_id")
                if not run_id:
                    raise RuntimeError(
                        f"Nexus submit response missing run_id: {result}"
                    )
                hive_job_id = result.get("hive_job_id") or ""
                log.info(
                    "Submitted job %s (hive=%s) for project %s (gpus=%d)",
                    run_id, hive_job_id or "?", project_id, num_gpus,
                )
                return run_id, hive_job_id

            now = time.monotonic()
            if now >= deadline:
                break
            delay = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
            delay = min(delay, max(1.0, deadline - now))
            attempt += 1
            log.warning(
                "Nexus submit_job for project %s retry %d after %.0fs "
                "(last: %s)", project_id, attempt, delay, last_exc,
            )
            time.sleep(delay)

        raise RuntimeError(
            f"Nexus submit_job for project {project_id} failed after "
            f"{attempt + 1} attempts over ~{int(time.monotonic() - (deadline - 600.0))}s "
            f"(last error: {last_exc!r})"
        )

    def wait_for_job(
        self,
        project_id: str,
        timeout: int = 600,
        run_id: str = "",
    ) -> dict:
        """Long-poll Nexus for job events (checkpoint, completion).

        Args:
            project_id: Nexus project identifier.
            timeout: Server-side timeout in seconds (max 600).
            run_id: Optional run_id to watch. If the job already finished
                before we start polling, the server returns immediately.

        Returns:
            Event dict with ``event``, ``run_id``, ``status`` keys.
        """
        path = f"/api/projects/{project_id}/wait?timeout={timeout}"
        if run_id:
            path += f"&run_id={run_id}"

        try:
            return self._get(path, timeout=timeout + 30)
        except (urllib.error.URLError, OSError) as exc:
            log.warning("Wait failed for project %s: %s", project_id, exc)
            return {"event": "error", "error": str(exc)}

    def cancel_job(self, run_id: str) -> bool:
        """Cancel a running job.

        Returns:
            True if cancellation succeeded.
        """
        try:
            result = self._post(f"/api/jobs/{run_id}/cancel", {})
            if "error" in result:
                log.warning("Cancel returned error for %s: %s", run_id, result["error"])
                return False
            log.info("Cancelled job %s", run_id)
            return True
        except (urllib.error.URLError, OSError) as exc:
            log.warning("Could not reach Nexus to cancel %s: %s", run_id, exc)
            return False

    def continue_job(self, run_id: str) -> dict:
        """Tell Nexus to resume monitoring a job after checkpoint review.

        Args:
            run_id: The Nexus run ID.

        Returns:
            Nexus API response dict.
        """
        try:
            return self._post(f"/api/jobs/{run_id}/continue", {})
        except (urllib.error.URLError, OSError) as exc:
            log.warning("Could not continue job %s: %s", run_id, exc)
            return {"error": str(exc)}

    def get_job(self, run_id: str) -> dict:
        """Get job status and details.

        Returns:
            Job info dict, or dict with ``error`` key on failure.
        """
        try:
            return self._get(f"/api/jobs/{run_id}")
        except (urllib.error.URLError, OSError) as exc:
            log.debug("Could not fetch job %s: %s", run_id, exc)
            return {"status": "unknown", "error": str(exc)}
