"""Tests for namespace-package imports in workspace_template/validate.py."""

import importlib.util
import subprocess

import pytest

from .conftest import TEMPLATE_DIR


@pytest.fixture
def validate(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "workspace_validate", TEMPLATE_DIR / "validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WORKSPACE_DIR", tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "method").mkdir()
    return module


def _track(root, *paths):
    subprocess.run(["git", "-C", str(root), "add", *paths], check=True)


def test_from_namespace_package_import_tracked_module(validate, tmp_path):
    (tmp_path / "method" / "model.py").write_text("")
    (tmp_path / "method" / "train.py").write_text("from method import model\n")
    _track(tmp_path, "method/model.py", "method/train.py")

    assert validate.check_imports_not_gitignored() == []


def test_from_namespace_package_import_untracked_module(validate, tmp_path):
    (tmp_path / "method" / "model.py").write_text("")
    (tmp_path / "method" / "train.py").write_text("from method import model\n")
    _track(tmp_path, "method/train.py")

    errors = validate.check_imports_not_gitignored()
    assert len(errors) == 1
    assert errors[0].startswith("method/train.py:1: imports method ")
