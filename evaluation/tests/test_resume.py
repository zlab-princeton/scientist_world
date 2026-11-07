"""Unit tests for resume-in-place eligibility (runner._resumable)."""
import json
from pathlib import Path

import pytest

from evaluation.runner import _resumable


def _env(tmp_path: Path, meta: dict | None, scoring_files: list[str] = ()) -> Path:
    env = tmp_path / "env"
    env.mkdir(exist_ok=True)
    if meta is not None:
        (env / ".evaluation.json").write_text(json.dumps(meta))
    for name in scoring_files:
        (env / "scoring").mkdir(exist_ok=True)
        (env / "scoring" / name).write_text("{}")
    return env


def test_none_and_missing_meta(tmp_path):
    assert _resumable(None) is False
    assert _resumable(_env(tmp_path, None)) is False


def test_completed_not_resumable(tmp_path):
    env = _env(tmp_path, {"status": "completed", "current_turn": 30})
    assert _resumable(env) is False


def test_no_progress_not_resumable(tmp_path):
    env = _env(tmp_path, {"status": "running", "current_turn": 0})
    assert _resumable(env) is False


def test_interrupted_with_progress_resumable(tmp_path):
    env = _env(tmp_path, {"status": "running", "current_turn": 7})
    assert _resumable(env) is True


def test_failed_with_progress_resumable(tmp_path):
    env = _env(tmp_path, {"status": "failed", "current_turn": 12})
    assert _resumable(env) is True


def test_partial_train_scores_still_resumable(tmp_path):
    # scores_train.json is written incrementally during the main phase —
    # it must not seal the session.
    env = _env(tmp_path, {"status": "running", "current_turn": 7},
               ["scores_train.json"])
    assert _resumable(env) is True


def test_test_artifacts_seal_the_session(tmp_path):
    env = _env(tmp_path, {"status": "running", "current_turn": 7},
               ["scores_test.json"])
    assert _resumable(env) is False
    (tmp_path / "b").mkdir()
    env2 = _env(tmp_path / "b", {"status": "running", "current_turn": 7},
                ["verification.json"])
    (env2 / "scoring" / "verification.json").write_text(
        json.dumps({"status": "verified"}))
    assert _resumable(env2) is False


@pytest.mark.parametrize("status", ["in_progress", "incomplete"])
def test_non_verdict_verification_does_not_seal(tmp_path, status):
    """A grader lock is not a verdict — it must not block resume.

    A grader session that dies (quota, auth, container build) leaves its
    "in_progress" marker behind forever. Reading that as a verdict strands
    the run: the agent may not resume and the grader will not retry.
    """
    env = _env(tmp_path, {"status": "running", "current_turn": 7},
               ["verification.json"])
    (env / "scoring" / "verification.json").write_text(
        json.dumps({"status": status}))
    assert _resumable(env) is True


@pytest.mark.parametrize("status", ["verified", "unfixable", "exhausted",
                                    "audit_clean"])
def test_grader_verdict_seals(tmp_path, status):
    env = _env(tmp_path, {"status": "running", "current_turn": 7},
               ["verification.json"])
    (env / "scoring" / "verification.json").write_text(
        json.dumps({"status": status}))
    assert _resumable(env) is False


def test_validation_rounds_continue_turn_numbering(tmp_path, monkeypatch):
    """Validation-retry rounds must number their turns after the main
    session's, or their trajectory files overwrite turn_0001... ."""
    from evaluation import runner, validate
    from evaluation.loop import LoopResult

    class _Gaps:
        ok, phase, n_missing = False, "train", 1

        def summary(self):
            return "train: 1 cell(s) missing"

        def format_for_prompt(self):
            return "- exp / ours / acc"

    class _Ok:
        ok = True

    results = iter([_Gaps(), _Gaps(), _Ok()])
    monkeypatch.setattr(validate, "validate_train", lambda env_dir: next(results))
    offsets = []

    def fake_run_loop(*args, turn_offset=0, **kwargs):
        offsets.append(turn_offset)
        return LoopResult(turns=turn_offset + 3, last_action="done", reason="done")

    monkeypatch.setattr(runner, "run_loop", fake_run_loop)
    env = _env(tmp_path, {})
    ok = runner._validate_with_retries(
        env, None, None, "proj", extra_binds=None,
        cfg={"nexus": {"gpu_budget_hours": 1.0}}, turn_offset=12,
    )
    assert ok is True
    assert offsets == [12, 15]
