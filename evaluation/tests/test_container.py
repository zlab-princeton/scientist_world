"""rebuild_if_changed: which layers are rebuilt after container.def /
setup.sh edits (builds stubbed out)."""
from __future__ import annotations

from pathlib import Path

import pytest

from evaluation import container


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    env_dir = tmp_path / "run-eval"
    sub = env_dir / "environment"
    sub.mkdir(parents=True)
    (sub / "container.def").write_text("Bootstrap: docker\nFrom: base:1\n")
    (sub / "setup.sh").write_text("pip install a\n")
    (sub / "container.sif").write_text("sif-1")
    (sub / "overlay.img").write_text("overlay-1")
    container.snapshot_hashes(env_dir)

    calls = {"container": 0, "overlay": 0, "overlay_fails": False}

    def fake_ensure_container(env_dir, **kwargs):
        sif = env_dir / "environment" / "container.sif"
        if not sif.exists():
            calls["container"] += 1
            sif.write_text("sif-rebuilt")
        return sif

    def fake_ensure_overlay(env_dir, container_sif, **kwargs):
        overlay = env_dir / "environment" / "overlay.img"
        if overlay.exists():
            return overlay
        calls["overlay"] += 1
        if calls["overlay_fails"]:
            raise RuntimeError("setup.sh failed (rc=1)")
        overlay.write_text("overlay-rebuilt")
        return overlay

    monkeypatch.setattr(container, "ensure_container", fake_ensure_container)
    monkeypatch.setattr(container, "ensure_overlay", fake_ensure_overlay)
    return env_dir, calls


def test_failed_setup_does_not_rebuild_the_new_container_again(env):
    env_dir, calls = env
    sub = env_dir / "environment"
    (sub / "container.def").write_text("Bootstrap: docker\nFrom: base:2\n")
    calls["overlay_fails"] = True

    with pytest.raises(RuntimeError, match="setup.sh failed"):
        container.rebuild_if_changed(env_dir)
    assert calls["container"] == 1

    # The agent fixes setup.sh; only the overlay must be rebuilt.
    (sub / "setup.sh").write_text("pip install b\n")
    calls["overlay_fails"] = False
    sif, overlay, did_rebuild = container.rebuild_if_changed(env_dir)

    assert calls["container"] == 1
    assert did_rebuild
    assert overlay.read_text() == "overlay-rebuilt"
