"""The three `codex exec` passes (clean_reasoning, rephrase, judge_craft) and
their batch drivers. The CLI is never run: subprocess.run is replaced."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile

import pytest

import clean_reasoning
import clean_reasoning_batch
import judge_craft
import judge_craft_batch
import rephrase
import rephrase_batch
from conftest import make_row

SCORES = {"novelty": 70, "soundness": 60, "overall": 65, "reason": "r"}

# (module, call, a stdout the pass accepts, the value it then returns)
PASSES = {
    "clean_reasoning": (
        clean_reasoning,
        lambda **kw: clean_reasoning.clean_reasoning("THE TASK", ["v1", "v2"], **kw),
        'progress line\n{"reasoning": " because "}\n', "because"),
    "rephrase": (
        rephrase,
        lambda **kw: rephrase.rephrase("THE TASK", **kw),
        '{"reworded": " new text "}\n', "new text"),
    "judge_craft": (
        judge_craft,
        lambda **kw: judge_craft.judge("THE TASK", "an idea", **kw),
        json.dumps(SCORES) + "\n", SCORES),
}


class FakeCodex:
    """Stands in for subprocess.run and records each call."""

    def __init__(self, stdout="", error=None):
        self.stdout, self.error, self.calls = stdout, error, []

    def __call__(self, argv, **kwargs):
        schema_path = argv[argv.index("--output-schema") + 1]
        with open(schema_path) as f:
            schema = json.load(f)
        self.calls.append({"argv": argv, "schema_path": schema_path,
                           "schema": schema, **kwargs})
        if self.error:
            raise self.error
        return subprocess.CompletedProcess(argv, 0, stdout=self.stdout, stderr="")


@pytest.fixture(params=sorted(PASSES))
def codex_pass(request, monkeypatch):
    module, call, stdout, expected = PASSES[request.param]
    monkeypatch.setattr(module, "DEFAULT_MODEL", "gpt-5.5")
    monkeypatch.setattr(module, "DEFAULT_EFFORT", "medium")

    def run(fake, **kw):
        monkeypatch.setattr(module.subprocess, "run", fake)
        return call(**kw)

    return run, stdout, expected


def test_pass_invokes_codex_with_the_published_defaults(codex_pass):
    run, stdout, expected = codex_pass
    fake = FakeCodex(stdout)
    assert run(fake) == expected
    (call,) = fake.calls
    assert call["argv"] == [
        "codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only",
        "--model", "gpt-5.5", "-c", "reasoning_effort=medium",
        "--output-schema", call["schema_path"], "-"]
    assert "THE TASK" in call["input"]                    # the prompt goes on stdin
    assert call["cwd"] == tempfile.gettempdir()
    assert call["schema"]["type"] == "object" and call["schema"]["required"]
    assert not os.path.exists(call["schema_path"])        # the schema file is removed


def test_pass_takes_model_and_effort(codex_pass):
    run, stdout, _ = codex_pass
    fake = FakeCodex(stdout)
    run(fake, model="some-model", effort="high")
    argv = fake.calls[0]["argv"]
    assert argv[argv.index("--model") + 1] == "some-model"
    assert argv[argv.index("-c") + 1] == "reasoning_effort=high"


@pytest.mark.parametrize("fake", [
    FakeCodex(error=FileNotFoundError("codex")),          # CLI not installed
    FakeCodex(error=subprocess.TimeoutExpired("codex", 1)),
    FakeCodex(stdout="no json here\n"),
], ids=["missing-cli", "timeout", "no-json"])
def test_pass_returns_none_on_failure(codex_pass, fake):
    run, _, _ = codex_pass
    fake.calls.clear()
    assert run(fake) is None
    assert not os.path.exists(fake.calls[-1]["schema_path"])


def test_passes_skip_empty_input(monkeypatch):
    def refuse(*a, **kw):
        raise AssertionError("codex must not be called")

    for module in (clean_reasoning, rephrase, judge_craft):
        monkeypatch.setattr(module.subprocess, "run", refuse)
    assert clean_reasoning.clean_reasoning("task", ["only one version"]) is None
    assert rephrase.rephrase("   ") is None
    assert judge_craft.judge("task", "") is None


@pytest.mark.parametrize("module", [clean_reasoning, rephrase, judge_craft])
def test_defaults_come_from_the_environment(module, monkeypatch):
    try:
        monkeypatch.delenv("SW_CODEX_MODEL", raising=False)
        monkeypatch.delenv("SW_CODEX_EFFORT", raising=False)
        importlib.reload(module)
        assert (module.DEFAULT_MODEL, module.DEFAULT_EFFORT) == ("gpt-5.5", "medium")
        monkeypatch.setenv("SW_CODEX_MODEL", "other-model")
        monkeypatch.setenv("SW_CODEX_EFFORT", "low")
        importlib.reload(module)
        assert (module.DEFAULT_MODEL, module.DEFAULT_EFFORT) == ("other-model", "low")
    finally:
        monkeypatch.undo()
        importlib.reload(module)


# ---------------------------------------------------------------- batch drivers

def _records(path):
    return [json.loads(ln) for ln in path.read_text().splitlines()]


def _point_at(monkeypatch, batch, tmp_path, name, argv):
    monkeypatch.setattr(batch, "_OUT", tmp_path)
    monkeypatch.setattr(batch, "_CKPT", tmp_path / name)
    monkeypatch.setattr(batch.shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    monkeypatch.setattr(sys, "argv", [batch.__name__ + ".py", *argv])
    return tmp_path / name


def test_clean_reasoning_batch_is_resumable(monkeypatch, tmp_path):
    batch = clean_reasoning_batch
    rows = [make_row("p1", "innovate", "multi", versions=2),
            make_row("p1", "innovate", "single", versions=1),
            make_row("p1", "improve", "fails", versions=3)]
    seen = []

    def fake_clean(task, versions, model=None, effort=None):
        seen.append((len(versions), model, effort))
        return None if len(versions) == 3 else "trace"

    out = _point_at(monkeypatch, batch, tmp_path, "reasoning.jsonl",
                    ["--workers", "1", "--model", "m", "--effort", "low"])
    monkeypatch.setattr(batch, "_load_rows", lambda source=None: rows)
    monkeypatch.setattr(batch, "clean_reasoning", fake_clean)

    batch.main()
    assert {r["run_id"]: r.get("reasoning") for r in _records(out)} == {
        "multi": "trace", "fails": None}                 # single-version run skipped
    assert sorted(seen) == [(2, "m", "low"), (3, "m", "low")]

    seen.clear()
    batch.main()                                         # only the failure is retried
    assert seen == [(3, "m", "low")]


def test_rephrase_batch_rewords_the_best_scoring_examples(monkeypatch, tmp_path):
    batch = rephrase_batch
    examples = [
        {"run_id": f"r{score}", "task_id": "p1", "test_score": score,
         "conversation": [{}, {}, {"content": f"completion {score}"}]}
        for score in (1.0, 3.0, None, 2.0)]
    seen = []

    def fake_rephrase(completion, model=None, effort=None):
        seen.append((completion, model, effort))
        return completion.upper()

    out = _point_at(monkeypatch, batch, tmp_path, "rephrased.jsonl",
                    ["--n", "2", "--workers", "1"])
    monkeypatch.setattr(batch, "sft_examples", lambda **kw: examples)
    monkeypatch.setattr(batch, "rephrase", fake_rephrase)

    batch.main()
    assert {r["run_id"]: r["rephrased_completion"] for r in _records(out)} == {
        "r3.0": "COMPLETION 3.0", "r2.0": "COMPLETION 2.0"}
    assert {(m, e) for _, m, e in seen} == {(batch.DEFAULT_MODEL, batch.DEFAULT_EFFORT)}


def test_rephrase_batch_skips_non_finite_scores(monkeypatch, tmp_path):
    # Reasoning examples keep their score as-is, so a missing score can be NaN;
    # it must not take a top-N slot.
    batch = rephrase_batch
    examples = [
        {"run_id": f"r{score}", "task_id": "p1", "test_score": score,
         "conversation": [{}, {}, {"content": f"completion {score}"}]}
        for score in (1.0, float("nan"), 3.0, 2.0)]
    out = _point_at(monkeypatch, batch, tmp_path, "rephrased.jsonl",
                    ["--n", "2", "--workers", "1"])
    monkeypatch.setattr(batch, "sft_examples", lambda **kw: examples)
    monkeypatch.setattr(batch, "rephrase",
                        lambda completion, model=None, effort=None: completion)

    batch.main()
    assert sorted(r["run_id"] for r in _records(out)) == ["r2.0", "r3.0"]


def test_judge_craft_batch_scores_every_final_idea(monkeypatch, tmp_path):
    batch = judge_craft_batch
    rows = [make_row("p1", "innovate", "a1"), make_row("p1", "improve", "a2"),
            make_row("p1", "improve", "a3", idea=None)]
    out = _point_at(monkeypatch, batch, tmp_path, "craft_judge.jsonl",
                    ["--workers", "1", "--model", "m"])
    monkeypatch.setattr(batch, "_load_rows", lambda source=None: rows)
    monkeypatch.setattr(batch, "judge", lambda task, idea, model=None, effort=None:
                        dict(SCORES, reason=model))

    batch.main()
    records = _records(out)
    assert sorted(r["run_id"] for r in records) == ["a1", "a2"]
    assert all(r["overall"] == 65 and r["reason"] == "m" for r in records)


@pytest.mark.parametrize("batch, name", [
    (clean_reasoning_batch, "reasoning.jsonl"),
    (judge_craft_batch, "craft_judge.jsonl"),
])
def test_batch_stops_when_codex_is_missing(batch, name, monkeypatch, tmp_path):
    out = _point_at(monkeypatch, batch, tmp_path, name, [])
    monkeypatch.setattr(batch.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(batch, "_load_rows",
                        lambda source=None: [make_row("p1", "innovate", "a1", versions=2)])
    with pytest.raises(SystemExit, match="codex"):
        batch.main()
    assert not out.exists()                              # nothing was recorded
