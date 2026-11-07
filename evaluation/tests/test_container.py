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

    def fake_ensure_container(env_dir, **kwargs):
        return env_dir / "environment" / "container.sif"

    def fake_ensure_overlay(env_dir, container_sif, **kwargs):
        overlay = env_dir / "environment" / "overlay.img"
        if overlay.exists():
            return overlay
        if not (env_dir / "environment" / "setup.sh").exists():
            return None
        overlay.write_text("overlay-rebuilt")
        return overlay

    monkeypatch.setattr(container, "ensure_container", fake_ensure_container)
    monkeypatch.setattr(container, "ensure_overlay", fake_ensure_overlay)
    return env_dir


def test_deleting_setup_sh_drops_the_overlay(env):
    (env / "environment" / "setup.sh").unlink()

    sif, overlay, did_rebuild = container.rebuild_if_changed(env)

    assert overlay is None
    assert did_rebuild
    assert not (env / "environment" / "overlay.img").exists()
    # Nothing changed since: no further rebuild.
    assert container.rebuild_if_changed(env) == (sif, None, False)
