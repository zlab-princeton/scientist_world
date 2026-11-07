"""Dataset builder: extraction, model ids, row assembly, upload arguments
and the judge wrappers (with the judge CLI stubbed)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from dataset_builder import (
    build_dataset, consolidate, extract, grade_axes, grade_axes_batch,
    grade_grounded, model_ids, push_to_hf,
)
from evaluation import config as _config
from evaluation import normalization as nz

PAPER = "2501.00001"
FAKE_KEY = "sk-ant-" + "k" * 60
FAKE_PAT = "ghp_" + "g" * 40


@pytest.fixture(autouse=True)
def _fresh_model_id_caches():
    for fn in (model_ids._profiles, model_ids.resolve_model_id,
               model_ids.combined_model_id):
        fn.cache_clear()
    yield
    for fn in (model_ids._profiles, model_ids.resolve_model_id,
               model_ids.combined_model_id):
        fn.cache_clear()


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data))


def _scores(acc: float, loss: float) -> dict:
    return {"experiments": {"exp1": {"results": {"ours": {"acc": acc, "loss": loss}}}}}


@pytest.fixture
def corpus(eval_config) -> Path:
    """One paper with three graded improve runs and its coefficient files.

    ``m1-xhigh`` is the run the row tests look at: a claude-code
    trajectory with thinking blocks, two idea versions and a reflection.
    """
    data_dir = eval_config(profiles={
        "m1-xhigh": {"scaffold": "claude-code", "model": "alias",
                     "actual_model": "real-model-1"},
        "m2": {"scaffold": "codex", "model": "model-2"},
        "m3": {"scaffold": "codex", "model": "model-3"},
    })
    paper = data_dir / "envs" / PAPER
    source = paper / "source"
    _write(source / "scoring" / "reference.json", {"experiments": {"exp1": {"metrics": {
        "acc": {"higher_is_better": True}, "loss": {"higher_is_better": False}}}}})
    for phase in ("train", "test"):
        _write(source / "scoring" / f"scores_{phase}.json", _scores(0.70, 2.0))
    _write(source / "manifest.json", {"paper": {"title": "A Paper", "topics": ["t1", "t2"]}})
    _write(source / "briefing" / "problem.md", "# Problem\n\nSolve it.\n")
    _write(source / "briefing" / "overview.md",
           "# Overview\n\nContext.\n\n## Key Results\n\n92.1% accuracy\n\n## Setup\n\nDetails.\n")
    _write(source / "scoring" / "DIRECTION.md", "Go this way.\n")

    for profile, scores in (("m1-xhigh", _scores(0.80, 1.5)),
                            ("m2", _scores(0.70, 2.5)), ("m3", _scores(0.60, 2.0))):
        run = paper / f"{profile}-improve-eval"
        _write(run / ".evaluation.json", {
            "run_id": f"run-{profile}", "paper_id": PAPER, "model": profile,
            "mode": "improve", "status": "completed", "gpu_hours_used": 1.23456,
            "current_turn": 9,
            "usage": {"scaffold": "claude-code" if profile == "m1-xhigh" else "codex",
                      "cost_usd": 0.5, "turn_count": 4, "output_tokens": 100,
                      "reasoning_tokens": 7},
            "publish": {"status": "pushed", "repo": "example-org/repo",
                        "branch": f"improve-{profile}"},
        })
        for phase in ("train", "test"):
            _write(run / "scoring" / f"scores_{phase}.json", scores)
        _write(run / "scoring" / "verification.json", {
            "status": "verified", "grader_profile": "grader",
            "scores_train": scores, "scores_test": scores})

    run = paper / "m1-xhigh-improve-eval"
    _write(run / "idea" / "idea_v1.md", "First idea.\n")
    _write(run / "idea" / "idea_v2.md", f"Second idea, as in v1 but with key {FAKE_KEY}.\n")
    _write(run / "idea" / "idea_v10.md", "Tenth idea.\n")
    _write(run / "reflection.md", f"I pushed with https://user:{FAKE_PAT}@github.com/x/y.git\n")
    _write(run / "trajectory" / "turn_0001_raw.jsonl", "\n".join(json.dumps(e) for e in (
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": f"The token is {FAKE_PAT}, do not print it."},
            {"type": "text", "text": "not reasoning"}]}},
        {"type": "user", "message": {"content": "tool result"}},
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "Second thought."}]}},
    )) + "\nnot json\n")
    _write(run / "trajectory" / "turn_0002_raw.jsonl", "")  # empty turn file: skipped
    nz.recompute_paper(PAPER)
    return data_dir


# ── extract ──────────────────────────────────────────────────────────────


def test_strip_leaky_sections() -> None:
    md = ("# Title\n\nIntro.\n\n## Key Results\n\nsecret 92%\n\n### Sub\n\nmore\n\n"
          "## Method\n\nkept\n\n## Reported results\nx\n")
    out = extract.strip_leaky_sections(md)
    assert "secret" not in out and "more" not in out
    assert "Intro." in out and "## Method" in out and "kept" in out
    assert "Reported results" not in out
    assert extract.strip_leaky_sections(None) is None
    assert extract.strip_leaky_sections("## Target numbers\nonly this\n") is None


def test_strip_leaky_sections_ignores_comments_in_code_fences() -> None:
    md = ("## Key Results\n\n```python\n# accuracy on the test split\nacc = 92.1\n```\n"
          "~~~\n# loss\nloss = 0.31\n~~~\n\n## Method\n\n```\n# kept code\n```\n")
    out = extract.strip_leaky_sections(md)
    assert "92.1" not in out and "0.31" not in out
    assert out.startswith("## Method") and "# kept code" in out


def test_extract_inputs_and_manifest(corpus: Path) -> None:
    source = corpus / "envs" / PAPER / "source"
    inputs = extract.extract_inputs(source)
    assert set(inputs) == {"problem", "direction", "constraint", "target",
                           "experiments", "overview", "evaluation"}
    assert inputs["problem"].startswith("# Problem")
    assert inputs["direction"] == "Go this way.\n"
    assert inputs["constraint"] is None
    assert "92.1%" not in inputs["overview"] and "## Setup" in inputs["overview"]
    assert extract.extract_paper_manifest(source) == {
        "paper_title": "A Paper", "paper_topics": ["t1", "t2"]}
    assert extract.extract_paper_manifest(corpus / "nowhere") == {
        "paper_title": None, "paper_topics": None}


def test_extract_ideas_orders_by_version_number(corpus: Path) -> None:
    run = corpus / "envs" / PAPER / "m1-xhigh-improve-eval"
    versions, final = extract.extract_ideas(run)
    assert [v.split()[0] for v in versions] == ["First", "Second", "Tenth"]
    assert final == "Tenth idea.\n"
    assert extract.extract_ideas(corpus / "nowhere") == ([], None)


def test_extract_reflection_ignores_the_unfilled_template(tmp_path: Path) -> None:
    assert extract.extract_reflection(tmp_path) is None
    (tmp_path / "reflection.md").write_text("<!-- write your reflection here -->\n")
    assert extract.extract_reflection(tmp_path) is None
    (tmp_path / "reflection.md").write_text("too short")
    assert extract.extract_reflection(tmp_path) is None
    (tmp_path / "reflection.md").write_text("A real reflection with enough text.\n")
    assert extract.extract_reflection(tmp_path).startswith("A real reflection")


def _trajectory(run: Path, *events: dict) -> None:
    _write(run / "trajectory" / "turn_0001_raw.jsonl",
           "\n".join(json.dumps(e) for e in events) + "\n")


def test_trace_per_scaffold(tmp_path: Path) -> None:
    codex = tmp_path / "codex"
    _trajectory(codex,
                {"type": "item.completed", "item": {"type": "agent_message", "text": " narrated "}},
                {"type": "item.completed", "item": {"type": "command_execution", "text": "ls"}})
    assert extract.extract_thinking_trace(codex, "codex") == (["narrated"], False)

    kimi = tmp_path / "kimi"
    _trajectory(kimi, {"role": "assistant", "content": [
        {"type": "think", "think": "kimi thought"}, {"type": "text", "text": "answer"}]})
    assert extract.extract_thinking_trace(kimi, "kimi") == (["kimi thought"], True)

    claude = tmp_path / "claude"
    _trajectory(claude, {"type": "assistant", "message": {"content": [
        {"type": "thinking", "thinking": "claude thought"}]}})
    assert extract.extract_thinking_trace(claude, "claude-code") == (["claude thought"], True)
    assert extract.extract_thinking_trace(claude, "claude-tmux") == (["claude thought"], True)

    # No parser for gemini, unknown scaffolds, or a run without a trajectory.
    assert extract.extract_thinking_trace(claude, "gemini") == ([], False)
    assert extract.extract_thinking_trace(claude, None) == ([], False)
    assert extract.extract_thinking_trace(tmp_path / "none", "codex") == ([], False)


# ── model ids ────────────────────────────────────────────────────────────


def test_model_id_resolution(eval_config) -> None:
    eval_config(profiles={
        "gateway-xhigh": {"scaffold": "claude-code", "model": "opus",
                          "actual_model": "vendor-model-2"},
        "plain-high": {"scaffold": "codex", "model": "model-x"},
        "tier-in-model-low": {"scaffold": "codex", "model": "model-y-low"},
        "no-tier": {"scaffold": "kimi", "model": "model-z"},
    })
    assert model_ids.resolve_model_id("gateway-xhigh") == "vendor-model-2"
    assert model_ids.combined_model_id("gateway-xhigh") == "vendor-model-2-xhigh"
    assert model_ids.combined_model_id("plain-high") == "model-x-high"
    assert model_ids.combined_model_id("tier-in-model-low") == "model-y-low"
    assert model_ids.combined_model_id("no-tier") == "model-z"
    assert model_ids.scaffold_for("no-tier") == "kimi"
    # A profile that is no longer configured: its name minus the tier.
    assert model_ids.resolve_model_id("retired-medium") == "retired"
    assert model_ids.combined_model_id("retired-medium") == "retired-medium"
    assert model_ids.resolve_model_id("unknown") == "unknown"
    assert model_ids.scaffold_for("unknown") is None


def test_model_ids_have_no_built_in_profile_table() -> None:
    assert not hasattr(model_ids, "_PROFILE_OVERRIDES")


# ── build_dataset ────────────────────────────────────────────────────────


def test_build_row(corpus: Path, monkeypatch) -> None:
    monkeypatch.setattr(build_dataset, "_CONSOLIDATED", {})
    row = build_dataset.build_row(corpus / "envs" / PAPER / "m1-xhigh-improve-eval")

    assert row["task_id"] == PAPER and row["mode"] == "improve"
    assert row["paper_title"] == "A Paper" and row["paper_topics"] == ["t1", "t2"]
    assert row["model"] == "real-model-1-xhigh"
    assert row["scaffold"] == "claude-code"
    assert row["run_id"] == "run-m1-xhigh"
    assert row["branch_url"] == "https://github.com/example-org/repo/tree/improve-m1-xhigh"
    assert row["problem"].startswith("# Problem")

    assert row["n_trace_segments"] == 2 and row["has_real_cot"] is True
    assert row["thinking_segments"][1] == "Second thought."
    assert row["n_idea_versions"] == 3
    assert row["idea"] == "First idea.\n"
    assert row["final_idea"] is None  # multi-version run, nothing consolidated yet
    assert row["has_reflection"] is True

    # m1 leads on both cells by one spread each: 1 / paper_baseline = 1.5
    assert row["train_score"] == pytest.approx(1.5)
    assert row["test_score"] == pytest.approx(1.5)
    assert (row["fni_train"], row["fni_test"], row["fni"]) == pytest.approx((150.0, 150.0, 150.0))
    assert row["train_score_reason"] is None
    assert row["score_locked"] is False and row["n_models_cohort"] == 3
    assert row["grader_profile"] == "grader"
    assert "grader_verified" not in row and "verification_status" not in row
    assert json.loads(row["train_score_raw"]) == _scores(0.80, 1.5)

    assert row["gpu_hours"] == 1.2346
    assert (row["cost_usd"], row["turn_count"], row["output_tokens"],
            row["reasoning_tokens"]) == (0.5, 4, 100, 7)


def test_build_row_redacts_credentials_in_text_columns(corpus: Path, monkeypatch) -> None:
    monkeypatch.setattr(build_dataset, "_CONSOLIDATED", {
        "run-m1-xhigh": f"Consolidated idea using {FAKE_KEY}."})
    row = build_dataset.build_row(corpus / "envs" / PAPER / "m1-xhigh-improve-eval")
    blob = json.dumps(row)
    assert FAKE_KEY not in blob and FAKE_PAT not in blob
    assert "ghp_REDACTED" in row["thinking_trace"]
    assert "ghp_REDACTED" in row["thinking_segments"][0]
    assert "sk-ant-REDACTED" in row["idea_versions"][1]
    assert "sk-ant-REDACTED" in row["final_idea"]
    assert "REDACTED@" in row["reflection"]
    # Non-text values and ordinary text are untouched.
    assert row["thinking_segments"][1] == "Second thought."
    assert row["n_idea_versions"] == 3 and row["has_real_cot"] is True


def test_build_row_single_version_idea_is_its_own_final_idea(corpus: Path, monkeypatch) -> None:
    monkeypatch.setattr(build_dataset, "_CONSOLIDATED", {})
    run = corpus / "envs" / PAPER / "m2-improve-eval"
    _write(run / "idea" / "idea_v1.md", "Only idea.\n")
    row = build_dataset.build_row(run)
    assert row["final_idea"] == "Only idea.\n" == row["idea"]
    assert row["model"] == "model-2"
    assert row["thinking_trace"] is None and row["has_real_cot"] is False


def test_build_row_takes_only_verified_runs(corpus: Path) -> None:
    run = corpus / "envs" / PAPER / "m3-improve-eval"
    verdict = run / "scoring" / "verification.json"
    for status in ("incomplete", "unfixable", "exhausted", "in_progress", "audit_clean"):
        verdict.write_text(json.dumps({**json.loads(verdict.read_text()), "status": status}))
        assert build_dataset.build_row(run) is None, status
    verdict.unlink()
    assert build_dataset.build_row(run) is None


def test_build_row_scores_what_the_grader_measured(corpus: Path, monkeypatch) -> None:
    """The agent's own score file is a claim; the verdict holds the measurement.
    A verified run counts even when its status was left at something else."""
    monkeypatch.setattr(build_dataset, "_CONSOLIDATED", {})
    run = corpus / "envs" / PAPER / "m1-xhigh-improve-eval"
    _write(run / "scoring" / "scores_test.json", _scores(9.99, 99.0))
    meta = json.loads((run / ".evaluation.json").read_text())
    meta["status"] = "running"
    (run / ".evaluation.json").write_text(json.dumps(meta))
    row = build_dataset.build_row(run)
    assert row["fni_test"] == pytest.approx(150.0)
    assert json.loads(row["test_score_raw"]) == _scores(0.80, 1.5)
    assert build_dataset.build_row(corpus / "envs" / PAPER / "source") is None


def test_iter_run_dirs(corpus: Path) -> None:
    names = [p.name for p in build_dataset.iter_run_dirs(corpus / "envs")]
    assert names == ["m1-xhigh-improve-eval", "m2-improve-eval", "m3-improve-eval"]


# ── batch graders: which runs ────────────────────────────────────────────


def test_completed_cells(corpus: Path) -> None:
    envs = corpus / "envs"
    paper = envs / PAPER
    # a variant workspace and an unfinished run are not graded
    _write(paper / "m2-improve-impl-sft-eval" / ".evaluation.json", {
        "paper_id": PAPER, "model": "m2", "mode": "improve", "status": "completed"})
    _write(paper / "m9-innovate-eval" / ".evaluation.json", {
        "paper_id": PAPER, "model": "m9", "mode": "innovate", "status": "running"})

    cells = grade_axes_batch.completed_cells(envs)
    assert cells == [(PAPER, "improve", "m1-xhigh"), (PAPER, "improve", "m2"),
                     (PAPER, "improve", "m3")]
    assert grade_axes_batch.completed_cells(envs, models={"m2", "zzz"}) == [
        (PAPER, "improve", "m2")]
    assert grade_axes_batch.completed_cells(envs, papers={"other"}) == []
    assert grade_axes_batch.split_arg(None) is None
    assert grade_axes_batch.split_arg(" a, b ,,") == {"a", "b"}


def test_graders_have_no_built_in_paper_or_model_lists() -> None:
    from dataset_builder import grade_grounded_batch
    for module in (grade_axes_batch, grade_grounded_batch):
        assert not hasattr(module, "TOP10") and not hasattr(module, "MODELS")


# ── judge wrappers (codex CLI stubbed) ───────────────────────────────────


def _judge(stdout: str):
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        return mock.Mock(returncode=0, stdout=stdout, stderr="")

    return fake_run, calls


def test_consolidate_single_version_needs_no_call(monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("no judge call expected")

    monkeypatch.setattr(consolidate.subprocess, "run", boom)
    assert consolidate.consolidate({}, []) is None
    assert consolidate.consolidate({}, ["only"]) == "only"


def test_consolidate_reads_the_structured_answer(monkeypatch) -> None:
    fake_run, calls = _judge('session noise\n{"final_idea": " merged idea "}\n')
    monkeypatch.setattr(consolidate.subprocess, "run", fake_run)
    out = consolidate.consolidate({"problem": "P"}, ["v1", "v2"])
    assert out == "merged idea"
    (cmd, kwargs), = calls
    assert cmd[:2] == ["codex", "exec"]
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert cmd[cmd.index("--model") + 1] == consolidate._MODEL
    assert "===== idea_v2 =====" in kwargs["input"]

    fake_run, _ = _judge("no json here\n")
    monkeypatch.setattr(consolidate.subprocess, "run", fake_run)
    assert consolidate.consolidate({}, ["v1", "v2"]) is None


def test_grade_axis(corpus: Path, monkeypatch) -> None:
    run = corpus / "envs" / PAPER / "m1-xhigh-improve-eval"
    fake_run, calls = _judge('{"score": 71, "reason": "solid"}\n')
    monkeypatch.setattr(grade_axes.subprocess, "run", fake_run)

    assert grade_axes.grade_axis(run, "improve", "rigor") == {"score": 71, "reason": "solid"}
    (cmd, kwargs), = calls
    assert cmd[cmd.index("--model") + 1] == grade_axes._MODEL
    assert "RIGOR" in kwargs["input"] and "idea_v10.md" in kwargs["input"]
    assert grade_axes.grade_axis(run, "improve", "not-an-axis") is None
    # nothing to judge: no call
    assert grade_axes.grade_axis(corpus / "nowhere", "improve", "rigor") is None
    assert len(calls) == 1


def test_grounded_judge_sees_the_measured_score(corpus: Path, monkeypatch) -> None:
    text = grade_grounded._measured(PAPER, "m1-xhigh", "improve")
    assert "test delta_norm = 1.50" in text and "train delta_norm = 1.50" in text
    assert "could not be normalized" in grade_grounded._measured(PAPER, "m1-xhigh", "innovate")

    fake_run, calls = _judge('{"score": 55, "reason": "ok"}\n')
    monkeypatch.setattr(grade_grounded.subprocess, "run", fake_run)
    assert grade_grounded.grade_axis(PAPER, "m1-xhigh", "improve", "depth")["score"] == 55
    assert "test delta_norm = 1.50" in calls[0][1]["input"]


def test_grounded_module_imports_without_a_configured_data_dir() -> None:
    """The data directory is looked up when a run is graded, not at import."""
    import importlib
    assert not _config.CONFIG_PATH.exists()
    importlib.reload(grade_grounded)
    with pytest.raises(_config.ConfigError):
        grade_grounded.gather(PAPER, "m1", "improve")


# ── push_to_hf arguments ─────────────────────────────────────────────────


def test_push_requires_a_repository(monkeypatch, capsys) -> None:
    monkeypatch.delenv("SW_TRACES_REPO", raising=False)
    with pytest.raises(SystemExit) as exc:
        push_to_hf._parse_args([])
    assert exc.value.code == 2
    assert "SW_TRACES_REPO" in capsys.readouterr().err


def test_push_repository_from_flag_or_environment(monkeypatch) -> None:
    monkeypatch.setenv("SW_TRACES_REPO", "env-org/env-name")
    assert push_to_hf._parse_args([]).repo == "env-org/env-name"
    assert push_to_hf._parse_args(["--repo", "org/name"]).repo == "org/name"


def test_push_is_private_unless_asked_otherwise(monkeypatch) -> None:
    monkeypatch.delenv("SW_TRACES_REPO", raising=False)
    assert push_to_hf._parse_args(["--repo", "o/n"]).private is True
    assert push_to_hf._parse_args(["--repo", "o/n", "--private"]).private is True
    assert push_to_hf._parse_args(["--repo", "o/n", "--no-private"]).private is False


def test_push_takes_the_token_from_the_environment_only(monkeypatch, capsys) -> None:
    with pytest.raises(SystemExit):
        push_to_hf._parse_args(["--repo", "o/n", "--token", "hf_x"])
    capsys.readouterr()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert push_to_hf._token() is None
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "b")
    assert push_to_hf._token() == "b"
    monkeypatch.setenv("HF_TOKEN", "a")
    assert push_to_hf._token() == "a"
