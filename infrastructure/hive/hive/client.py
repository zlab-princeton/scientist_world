"""Hive Python Client — programmatic access to a Hive dashboard.

Thin HTTP client wrapping the dashboard API. No external dependencies
beyond the Python standard library.

Usage:
    from hive.client import Client

    client = Client("http://head-node:38581")
    job_id = client.submit("python train.py", on_complete=lambda r: print("Done!", r["status"]))
    # Script continues — callback fires in background when job completes
"""

import json
import os
import threading
import time
import urllib.request
import urllib.error


class Client:
    """Client for a remote Hive dashboard."""

    def __init__(self, url: str):
        """Connect to a Hive dashboard.

        Args:
            url: Dashboard URL, e.g. 'http://head-node:38581'
        """
        self.url = url.rstrip("/")
        self._threads: list[threading.Thread] = []

    def _get(self, path: str) -> dict:
        """GET a JSON endpoint."""
        req = urllib.request.Request(f"{self.url}{path}")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    def _post(self, path: str, data: dict) -> dict:
        """POST JSON to an endpoint."""
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self.url}{path}", data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            # The dashboard reports errors as {"error": ...} with a 4xx/5xx
            # status; return that body so callers can raise its message.
            try:
                body = json.loads(e.read().decode())
            except Exception:
                body = None
            if isinstance(body, dict) and "error" in body:
                return body
            raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e

    def submit(self, command: str, num_gpus: int = 1, num_cpus: int = 1,
               memory_mb: int = 0, workdir: str = None, resource: str = None,
               on_complete: callable = None, timeout_sec: int = None,
               max_retries: int = 0) -> str:
        """Submit a job. Returns the job ID immediately.

        Args:
            command: Shell command to run on the compute node.
            num_gpus: Number of GPUs to allocate (default: 1).
            num_cpus: Number of CPUs to allocate (default: 1).
            memory_mb: Memory limit in MB (0 = no limit).
            workdir: Working directory on the compute node.
            resource: Target resource(s), comma-separated (e.g. "gpu-b,gpu-test").
            on_complete: Callback function called with the result dict
                when the job finishes. Runs in a background thread.
            timeout_sec: Maximum wall-clock seconds before the job is killed.
                None means no limit.
            max_retries: Number of automatic retries on infrastructure failure
                (e.g. SLURM worker death). 0 means no retry.

        Returns:
            The job ID string.
        """
        payload = {
            "command": command,
            "num_gpus": num_gpus,
            "num_cpus": num_cpus,
            "memory_mb": memory_mb,
            "user": os.environ.get("USER", "unknown"),
        }
        if workdir:
            payload["workdir"] = workdir
        if resource:
            payload["resource"] = resource
        if timeout_sec is not None:
            payload["timeout_sec"] = timeout_sec
        if max_retries > 0:
            payload["max_retries"] = max_retries

        result = self._post("/api/submit", payload)
        if "error" in result:
            raise RuntimeError(result["error"])

        job_id = result["job_id"]

        if on_complete is not None:
            t = threading.Thread(
                target=self._poll_and_callback,
                args=(job_id, on_complete, 5.0),
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        return job_id

    def jobs(self, status: str = None) -> list[dict]:
        """List all jobs, optionally filtered by status.

        Args:
            status: Filter by status (e.g. 'running', 'completed').

        Returns:
            List of job metadata dicts.
        """
        all_jobs = self._get("/api/jobs")
        if status:
            return [j for j in all_jobs if j.get("status") == status]
        return all_jobs

    def logs(self, job_id: str) -> dict:
        """Get full job details including stdout/stderr.

        Args:
            job_id: The job ID.

        Returns:
            Dict with job_id, status, command, stdout, stderr, etc.
        """
        return self._get(f"/api/jobs/{job_id}/logs")

    def status(self) -> dict:
        """Get cluster status.

        Returns:
            Dict with total_gpus, free_gpus, num_gpu_nodes, num_cpu_nodes, etc.
        """
        return self._get("/api/status")

    def wait(self, job_id: str, poll_interval: float = 5.0,
             timeout: float = None) -> dict:
        """Block until a job completes. Returns the result dict.

        Args:
            job_id: The job ID to wait for.
            poll_interval: Seconds between status checks (default: 5).
            timeout: Maximum seconds to wait before returning regardless
                of job status (None = wait forever).

        Returns:
            Full job result dict (same as logs()).
        """
        deadline = time.time() + timeout if timeout is not None else None
        while True:
            time.sleep(poll_interval)
            result = self.logs(job_id)
            if result["status"] in ("completed", "failed", "timeout", "cancelled"):
                return result
            if deadline is not None and time.time() >= deadline:
                return result

    def _poll_and_callback(self, job_id: str, callback: callable,
                           poll_interval: float):
        """Background thread: polls until job is done, then calls callback."""
        while True:
            time.sleep(poll_interval)
            try:
                result = self.logs(job_id)
                if result["status"] in ("completed", "failed", "timeout", "cancelled"):
                    callback(result)
                    return
            except Exception:
                pass  # Transient error — keep polling
