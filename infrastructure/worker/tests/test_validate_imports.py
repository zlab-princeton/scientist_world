"""Tests for workspace_template/validate.py import checks."""

import importlib.util

import pytest

from .conftest import TEMPLATE_DIR


@pytest.fixture
def validate(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "workspace_validate", TEMPLATE_DIR / "validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WORKSPACE_DIR", tmp_path)
    return module


def test_imports_check_skipped_outside_git_repo(validate, tmp_path, monkeypatch):
    # Stop git from finding a repository above tmp_path.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    (tmp_path / "method").mkdir()
    (tmp_path / "method" / "model.py").write_text("")
    (tmp_path / "method" / "train.py").write_text("import method.model\n")

    assert validate.check_imports_not_gitignored() == []
