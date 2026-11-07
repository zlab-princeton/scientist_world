"""Tests for the summary printed by workspace_template/validate.py main()."""

import importlib.util
import json
import sys

import pytest

from .conftest import TEMPLATE_DIR

REFERENCE = {
    "experiments": {
        "exp_a": {
            "description": "Table 1",
            "weight": 0.5,
            "primary_metric": "acc",
            "metrics": {"acc": {"higher_is_better": True, "coefficient": 1.0}},
            "results": {
                "baseline": {"type": "baseline", "acc": 70.0},
                "ours": {"type": "proposed", "acc": 80.0},
            },
        },
        "exp_b": {
            "description": "Table 2",
            "weight": 0.5,
            "primary_metric": "acc",
            "metrics": {"acc": {"higher_is_better": True, "coefficient": 1.0}},
            "results": {
                "baseline": {"type": "baseline", "acc": 60.0},
                "ours": {"type": "proposed", "acc": 65.0},
            },
        },
    },
}


@pytest.fixture
def validate(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "workspace_validate", TEMPLATE_DIR / "validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(module, "SCORING_DIR", tmp_path / "scoring")
    (tmp_path / "scoring").mkdir()
    (tmp_path / "scoring" / "reference.json").write_text(json.dumps(REFERENCE))
    return module


def test_summary_counts_wrapped_scores_format(validate, tmp_path, monkeypatch, capsys):
    scores = {"experiments": {
        name: {"results": {m: {"acc": v["acc"]} for m, v in exp["results"].items()}}
        for name, exp in REFERENCE["experiments"].items()
    }}
    (tmp_path / "scoring" / "scores.json").write_text(json.dumps(scores))
    monkeypatch.setattr(sys, "argv", ["validate.py"])

    with pytest.raises(SystemExit):
        validate.main()

    out = capsys.readouterr().out
    assert "reference.json — OK (2 experiments, 4 methods)" in out
    assert "scores.json — OK (2 experiments, 4 methods)" in out


def test_summary_counts_flat_scores_format(validate, tmp_path, monkeypatch, capsys):
    scores = {
        name: {m: {"acc": v["acc"]} for m, v in exp["results"].items()}
        for name, exp in REFERENCE["experiments"].items()
    }
    (tmp_path / "scoring" / "scores.json").write_text(json.dumps(scores))
    monkeypatch.setattr(sys, "argv", ["validate.py"])

    with pytest.raises(SystemExit):
        validate.main()

    assert "scores.json — OK (2 experiments, 4 methods)" in capsys.readouterr().out
