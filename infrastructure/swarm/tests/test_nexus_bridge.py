"""Tests for the Nexus HTTP client used by the swarm scheduler."""

import io
import urllib.error

import pytest

from swarm.nexus_bridge import NexusBridge


def test_create_project_raises_runtime_error_on_http_error(monkeypatch):
    # The scheduler's 409 delete-and-retry path catches RuntimeError, so
    # an HTTP error from Nexus must surface as RuntimeError, not HTTPError.
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 409, "CONFLICT", {},
            io.BytesIO(b'{"error": "Project \'p1\' already exists"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    bridge = NexusBridge("http://nexus.test")

    with pytest.raises(RuntimeError, match="already exists"):
        bridge.create_project(
            project_id="p1", workspace_dir="/ws", sif_path="",
            gpu_budget=1.0, max_gpus=1, allowed_resources=["gpu"],
        )
