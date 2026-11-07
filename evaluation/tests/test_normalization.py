"""Normalization (FNI) on small synthetic cohorts.

The expected numbers are worked out by hand from the rules in
``docs/scoring.md``; the tests pin the behaviour of
``evaluation.normalization`` as it is.

Base example (one experiment, weight 2.0; the proposed method ``ours``
and a baseline ``base``):

    cell   source  direction  prior   m1     m2     m3    improvements      spread
    acc    0.70    higher     2.0     0.80   0.70   0.60  +0.1  0.0  -0.1   0.1
    loss   2.0     lower      1.0     1.5    2.5    2.0   +0.5 -0.5   0.0   0.5

    typical_spread   = median(0.1, 0.5) = 0.3
    multipliers      = 0.5 + 0.1/0.4 = 0.75      0.5 + 0.5/0.8 = 1.125
    effective weight = 1.5/2.625 = 4/7           1.125/2.625 = 3/7
    paper_score_raw  = m1: 1.0   m2: -3/7   m3: -4/7
    paper_baseline   = (1 + 3/7 + 4/7) / 3 = 2/3
    coefficient      = (4/7)/(0.1 * 2/3) = 60/7  (3/7)/(0.5 * 2/3) = 9/7
    paper_score      = m1: 1.5   m2: -9/14   m3: -6/7     (mean |score| = 1)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from evaluation import normalization as nz

PAPER = "2501.00001"

REFERENCE = {"experiments": {"exp1": {
    "weight": 2.0,
    "metrics": {
        "acc": {"coefficient": 1.0, "higher_is_better": True},
        "loss": {"coefficient": -0.5, "higher_is_better": False},
    },
    "results": {"ours": {"type": "proposed"}, "base": {"type": "baseline"}},
}}}

SOURCE = {"experiments": {"exp1": {"results": {
    "ours": {"acc": 0.70, "loss": 2.0},
    "base": {"acc": 0.50, "loss": 3.0},
}}}}


def _scores(acc: float, loss: float, method: str = "ours") -> dict:
    return {"experiments": {"exp1": {"results": {method: {"acc": acc, "loss": loss}}}}}


COHORT = {
    "m1": _scores(0.80, 1.5),
    "m2": _scores(0.70, 2.5),
    "m3": _scores(0.60, 2.0),
}


def _profiles(*names: str, **extra: dict) -> dict:
    out = {n: {"scaffold": "codex", "model": f"{n}-model"} for n in names}
    for name, fields in extra.items():
        out[name] = {"scaffold": "codex", "model": f"{name}-model", **fields}
    return out


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _paper(data_dir: Path, *, reference: dict = REFERENCE,
           train: dict | None = SOURCE, test: dict | None = None) -> Path:
    paper_dir = data_dir / "envs" / PAPER
    scoring = paper_dir / "source" / "scoring"
    _write(scoring / "reference.json", reference)
    if train is not None:
        _write(scoring / "scores_train.json", train)
    if test is not None:
        _write(scoring / "scores_test.json", test)
    return paper_dir


def _run(paper_dir: Path, profile: str, *, mode: str = "improve",
         status: str | None = "verified", graded: dict | None = None,
         agent: dict | None = None, phases: tuple[str, ...] = ("train",)) -> Path:
    """One run directory. ``graded`` goes into verification.json (the
    grader's re-run), ``agent`` into scores_<phase>.json (the agent's file)."""
    run_dir = paper_dir / f"{profile}-{mode}-eval"
    (run_dir / "scoring").mkdir(parents=True, exist_ok=True)
    if status is not None:
        ver = {"status": status}
        if graded is not None:
            for phase in phases:
                ver[f"scores_{phase}"] = graded
        _write(run_dir / "scoring" / "verification.json", ver)
    if agent is not None:
        for phase in phases:
            _write(run_dir / "scoring" / f"scores_{phase}.json", agent)
    return run_dir


def _coeffs(phase: str = "train", mode: str = "improve") -> dict:
    data = nz.load_coeffs(PAPER, phase, mode)
    assert data is not None, f"no coefficient file for {mode}/{phase}"
    return data


def _cell(coeffs: dict, metric: str) -> dict:
    return next(c for c in coeffs["cells"] if c["metric"] == metric)


@pytest.fixture
def base_paper(eval_config) -> Path:
    """The base example: three graded cohort runs in improve mode."""
    data_dir = eval_config(profiles=_profiles("m1", "m2", "m3"))
    paper_dir = _paper(data_dir)
    for name, scores in COHORT.items():
        _run(paper_dir, name, graded=scores)
    nz.recompute_paper(PAPER)
    return paper_dir


# ── The worked example ───────────────────────────────────────────────────


def test_cells_improvement_and_spread(base_paper: Path) -> None:
    co = _coeffs()
    assert co["excluded"] is False
    assert co["mode"] == "improve" and co["phase"] == "train"
    assert co["models_present"] == ["m1", "m2", "m3"]
    assert co["n_models"] == 3
    # Only the proposed method's cells: the baseline row is not scored.
    assert [(c["method"], c["metric"]) for c in co["cells"]] == [
        ("ours", "acc"), ("ours", "loss")]

    acc, loss = _cell(co, "acc"), _cell(co, "loss")
    # improvement = (agent - source) x direction, not divided by source
    assert acc["cohort_imp"] == pytest.approx({"m1": 0.1, "m2": 0.0, "m3": -0.1})
    assert loss["cohort_imp"] == pytest.approx({"m1": 0.5, "m2": -0.5, "m3": 0.0})
    # spread = sample standard deviation across the cohort
    assert acc["spread"] == pytest.approx(0.1)
    assert loss["spread"] == pytest.approx(0.5)
    assert co["typical_spread"] == pytest.approx(0.3)
    assert acc["normalizer"] == pytest.approx(0.1)
    assert loss["normalizer"] == pytest.approx(0.5)


def test_prior_weight_is_abs_coefficient_times_experiment_weight(base_paper: Path) -> None:
    co = _coeffs()
    assert _cell(co, "acc")["prior_weight"] == pytest.approx(2.0)
    # the sign of the reference coefficient is ignored: direction comes
    # from higher_is_better only
    assert _cell(co, "loss")["prior_weight"] == pytest.approx(1.0)
    assert _cell(co, "loss")["higher_is_better"] is False


def test_effective_weights_baseline_and_coefficients(base_paper: Path) -> None:
    co = _coeffs()
    acc, loss = _cell(co, "acc"), _cell(co, "loss")
    assert acc["effective_weight"] == pytest.approx(4 / 7)
    assert loss["effective_weight"] == pytest.approx(3 / 7)
    assert sum(c["effective_weight"] for c in co["cells"]) == pytest.approx(1.0)
    assert co["cohort_raw"] == pytest.approx({"m1": 1.0, "m2": -3 / 7, "m3": -4 / 7})
    assert co["paper_baseline"] == pytest.approx(2 / 3)
    assert acc["coefficient"] == pytest.approx(60 / 7)
    assert loss["coefficient"] == pytest.approx(9 / 7)
    assert co["pruned_cells"] == []


def test_cohort_mean_absolute_score_is_one(base_paper: Path) -> None:
    co = _coeffs()
    scores = {name: nz.normalize_delta(co, blob) for name, blob in COHORT.items()}
    assert scores["m1"]["delta_norm"] == pytest.approx(1.5)
    assert scores["m2"]["delta_norm"] == pytest.approx(-9 / 14)
    assert scores["m3"]["delta_norm"] == pytest.approx(-6 / 7)
    mean_abs = sum(abs(s["delta_norm"]) for s in scores.values()) / 3
    assert mean_abs == pytest.approx(1.0)
    assert scores["m1"]["n_cells_used"] == 2
    assert scores["m1"]["n_cells_missing"] == 0
    assert scores["m1"]["n_cells_pruned"] == 0
    assert scores["m1"]["n_models"] == 3


def test_any_model_is_scored_with_the_cohort_coefficients(base_paper: Path) -> None:
    """A model outside the cohort uses the same stored coefficients."""
    co = _coeffs()
    res = nz.normalize_delta(co, _scores(0.75, 2.0))
    assert res["delta_norm"] == pytest.approx(60 / 7 * 0.05)


def test_missing_cells_are_skipped_not_penalised(base_paper: Path) -> None:
    co = _coeffs()
    only_acc = {"experiments": {"exp1": {"results": {"ours": {"acc": 0.80}}}}}
    res = nz.normalize_delta(co, only_acc)
    assert res["delta_norm"] == pytest.approx(60 / 7 * 0.1)
    assert res["n_cells_used"] == 1 and res["n_cells_missing"] == 1


def test_modes_are_accounted_separately(base_paper: Path) -> None:
    """The three improve runs give innovate nothing."""
    inn = _coeffs(mode="innovate")
    assert inn["excluded"] is True
    assert inn["reason"] == "cohort_too_small"
    assert inn["n_models"] == 0


def test_tied_cell_uses_typical_spread_at_half_weight(eval_config) -> None:
    reference = json.loads(json.dumps(REFERENCE))
    reference["experiments"]["exp1"]["metrics"]["f1"] = {
        "coefficient": 1.0, "higher_is_better": True}
    source = json.loads(json.dumps(SOURCE))
    source["experiments"]["exp1"]["results"]["ours"]["f1"] = 0.4
    paper_dir = _paper(eval_config(profiles=_profiles("m1", "m2", "m3")),
                       reference=reference, train=source)
    for name, blob in COHORT.items():
        blob = json.loads(json.dumps(blob))
        blob["experiments"]["exp1"]["results"]["ours"]["f1"] = 0.5  # all tied
        _run(paper_dir, name, graded=blob)
    nz.recompute_paper(PAPER)

    co = _coeffs()
    f1 = _cell(co, "f1")
    assert f1["spread"] == 0.0
    assert co["typical_spread"] == pytest.approx(0.3)  # zero spreads are left out
    assert f1["normalizer"] == pytest.approx(0.3)
    # multiplier 0.5: raw weights 1.5 (acc), 1.125 (loss), 2.0 * 0.5 (f1)
    assert f1["effective_weight"] == pytest.approx(1.0 / 3.625)
    assert sum(c["effective_weight"] for c in co["cells"]) == pytest.approx(1.0)


# ── Which runs count ─────────────────────────────────────────────────────


def test_fewer_than_three_graded_models_gives_no_coefficients(eval_config) -> None:
    paper_dir = _paper(eval_config(profiles=_profiles("m1", "m2", "m3")))
    _run(paper_dir, "m1", graded=COHORT["m1"])
    _run(paper_dir, "m2", graded=COHORT["m2"])
    nz.recompute_paper(PAPER)

    co = _coeffs()
    assert co["excluded"] is True
    assert co["reason"] == "cohort_too_small"
    assert co["min_required"] == nz.MIN_COHORT_PRESENT == 3
    assert co["cells"] == []
    res = nz.normalize_delta(co, COHORT["m1"])
    assert res["delta_norm"] is None
    assert res["reason"] == "cohort_too_small"


@pytest.mark.parametrize("status", [None, "unfixable", "exhausted", "incomplete",
                                    "in_progress"])
def test_runs_without_a_clean_verdict_are_ignored(eval_config, status) -> None:
    paper_dir = _paper(eval_config(profiles=_profiles("m1", "m2", "m3", "m4")))
    for name, scores in COHORT.items():
        _run(paper_dir, name, graded=scores)
    # A fourth run with scores on disk but no clean verdict.
    _run(paper_dir, "m4", status=status, graded=_scores(0.99, 0.1),
         agent=_scores(0.99, 0.1))
    nz.recompute_paper(PAPER)

    co = _coeffs()
    assert co["models_present"] == ["m1", "m2", "m3"]
    assert co["paper_baseline"] == pytest.approx(2 / 3)


def test_grader_scores_win_over_the_agents_file(eval_config) -> None:
    paper_dir = _paper(eval_config(profiles=_profiles("m1", "m2", "m3")))
    _run(paper_dir, "m1", graded=COHORT["m1"], agent=_scores(0.99, 0.1))
    _run(paper_dir, "m2", graded=COHORT["m2"])
    _run(paper_dir, "m3", graded=COHORT["m3"])
    nz.recompute_paper(PAPER)
    assert _cell(_coeffs(), "acc")["cohort_imp"]["m1"] == pytest.approx(0.1)


def test_audit_clean_run_uses_the_agents_file(eval_config) -> None:
    paper_dir = _paper(eval_config(profiles=_profiles("m1", "m2", "m3")))
    _run(paper_dir, "m1", status="audit_clean", agent=COHORT["m1"])
    _run(paper_dir, "m2", graded=COHORT["m2"])
    _run(paper_dir, "m3", graded=COHORT["m3"])
    nz.recompute_paper(PAPER)
    co = _coeffs()
    assert co["models_present"] == ["m1", "m2", "m3"]
    assert co["paper_baseline"] == pytest.approx(2 / 3)


def test_profiles_outside_the_cohort_do_not_shape_coefficients(eval_config) -> None:
    data_dir = eval_config(profiles=_profiles("m1", "m2", "m3", "m4"),
                           profile_cohort_exclude=["m4"])
    paper_dir = _paper(data_dir)
    for name, scores in COHORT.items():
        _run(paper_dir, name, graded=scores)
    _run(paper_dir, "m4", graded=_scores(0.72, 1.9))
    nz.recompute_paper(PAPER)

    co = _coeffs()
    assert co["cohort"] == ["m1", "m2", "m3"]
    assert co["models_present"] == ["m1", "m2", "m3"]
    assert _cell(co, "acc")["coefficient"] == pytest.approx(60 / 7)


def test_cohort_defaults_to_every_profile(eval_config) -> None:
    eval_config(profiles=_profiles("a", "b", "c"))
    assert nz._cohort() == ["a", "b", "c"]
    eval_config(profile_cohort_exclude=["b"])
    assert nz._cohort() == ["a", "c"]
    eval_config(profile_cohort=["c", "b", "zzz"])
    assert nz._cohort() == ["c", "zzz"]


def test_profiles_sharing_publish_as_count_once(eval_config) -> None:
    """Two profiles with one published identity are one cohort model, and
    the profile named like the identity provides the scores."""
    data_dir = eval_config(profiles=_profiles(
        "m1", "m2", "m3", **{"m1-alt": {"publish_as": "m1"}}))
    paper_dir = _paper(data_dir)
    for name, scores in COHORT.items():
        _run(paper_dir, name, graded=scores)
    _run(paper_dir, "m1-alt", graded=_scores(0.99, 0.1))
    nz.recompute_paper(PAPER)

    co = _coeffs()
    assert co["models_present"] == ["m1", "m2", "m3"]
    assert _cell(co, "acc")["cohort_imp"]["m1"] == pytest.approx(0.1)


def test_alias_profile_alone_fills_the_identity_slot(eval_config) -> None:
    data_dir = eval_config(profiles=_profiles(
        "m2", "m3", **{"m1-alt": {"publish_as": "m1"}}))
    paper_dir = _paper(data_dir)
    _run(paper_dir, "m1-alt", graded=COHORT["m1"])
    _run(paper_dir, "m2", graded=COHORT["m2"])
    _run(paper_dir, "m3", graded=COHORT["m3"])
    nz.recompute_paper(PAPER)
    assert _coeffs()["models_present"] == ["m1", "m2", "m3"]


# ── Exclusions ───────────────────────────────────────────────────────────


def test_all_cells_tied(eval_config) -> None:
    paper_dir = _paper(eval_config(profiles=_profiles("m1", "m2", "m3")))
    for name in COHORT:
        _run(paper_dir, name, graded=_scores(0.8, 1.5))
    nz.recompute_paper(PAPER)
    co = _coeffs()
    assert co["excluded"] is True and co["reason"] == "all_cells_tied"


def test_all_priors_zero(eval_config) -> None:
    reference = json.loads(json.dumps(REFERENCE))
    for meta in reference["experiments"]["exp1"]["metrics"].values():
        meta["coefficient"] = 0
    paper_dir = _paper(eval_config(profiles=_profiles("m1", "m2", "m3")),
                       reference=reference)
    for name, scores in COHORT.items():
        _run(paper_dir, name, graded=scores)
    nz.recompute_paper(PAPER)
    co = _coeffs()
    assert co["excluded"] is True and co["reason"] == "all_priors_zero"


def test_paper_baseline_zero(eval_config) -> None:
    """Every cohort model's two cells cancel exactly."""
    reference = {"experiments": {"exp1": {"metrics": {
        "a": {"higher_is_better": True}, "b": {"higher_is_better": True}}}}}
    source = {"experiments": {"exp1": {"results": {"ours": {"a": 0.0, "b": 0.0}}}}}
    paper_dir = _paper(eval_config(profiles=_profiles("m1", "m2", "m3")),
                       reference=reference, train=source)
    for name, (a, b) in {"m1": (1.0, -1.0), "m2": (-1.0, 1.0), "m3": (0.0, 0.0)}.items():
        _run(paper_dir, name, graded={"experiments": {"exp1": {"results": {
            "ours": {"a": a, "b": b}}}}})
    nz.recompute_paper(PAPER)
    co = _coeffs()
    assert co["excluded"] is True and co["reason"] == "paper_baseline_zero"


def test_no_cells_when_the_cohort_reports_none(eval_config) -> None:
    paper_dir = _paper(eval_config(profiles=_profiles("m1", "m2", "m3")))
    for name in COHORT:
        _run(paper_dir, name, graded={"experiments": {"other_exp": {"results": {
            "ours": {"acc": 0.9}}}}})
    nz.recompute_paper(PAPER)
    co = _coeffs()
    assert co["excluded"] is True and co["reason"] == "no_cells"


def test_recompute_skips_papers_without_reference(eval_config) -> None:
    data_dir = eval_config(profiles=_profiles("m1"))
    assert nz.recompute_paper("missing") == {
        "paper_id": "missing", "skipped": "paper_dir_missing"}
    (data_dir / "envs" / PAPER).mkdir(parents=True)
    assert nz.recompute_paper(PAPER) == {"paper_id": PAPER, "skipped": "no_reference"}


# ── Cell enumeration ─────────────────────────────────────────────────────


def test_enumerate_cells_skips_baselines_and_non_numbers() -> None:
    source = {"experiments": {"exp1": {"results": {
        "ours": {"acc": 0.7, "note": "text", "flag": True, "loss": 2},
        "base": {"acc": 0.5},
        "not_a_dict": 3,
    }}}}
    cells = nz._enumerate_proposed_cells(source, REFERENCE)
    assert [(c["method"], c["metric"], c["source"]) for c in cells] == [
        ("ours", "acc", 0.7), ("ours", "loss", 2.0)]


def test_enumerate_cells_defaults() -> None:
    """Missing metadata: weight 1.0, coefficient 1.0, higher is better.
    A method the reference does not list counts as proposed."""
    source = {"exp1": {"other": {"m": 1.0}}}  # flat layout, no "results" level
    cells = nz._enumerate_proposed_cells(source, {})
    assert cells == [{
        "experiment": "exp1", "method": "other", "metric": "m", "source": 1.0,
        "higher_is_better": True, "prior_weight": 1.0,
    }]


def test_enumerate_cells_zero_weight_and_zero_coefficient() -> None:
    """An experiment weight of 0 is read as 1.0; a metric coefficient of 0
    gives the cell a prior weight of 0."""
    reference = {"experiments": {"exp1": {
        "weight": 0,
        "metrics": {"a": {"coefficient": 0}, "b": {"coefficient": 3}},
    }}}
    source = {"experiments": {"exp1": {"results": {"ours": {"a": 1.0, "b": 1.0}}}}}
    priors = {c["metric"]: c["prior_weight"]
              for c in nz._enumerate_proposed_cells(source, reference)}
    assert priors == {"a": 0.0, "b": 3.0}


# ── Method-name lookup ───────────────────────────────────────────────────


def test_cell_value_exact_name() -> None:
    assert nz._cell_value(_scores(0.8, 1.5), "exp1", "ours", "acc") == 0.8
    assert nz._cell_value(_scores(0.8, 1.5), "exp1", "ours", "missing") is None
    assert nz._cell_value(_scores(0.8, 1.5), "nope", "ours", "acc") is None


def test_cell_value_accepts_flat_layouts() -> None:
    flat = {"exp1": {"ours": {"acc": 0.8}}}
    assert nz._cell_value(flat, "exp1", "ours", "acc") == 0.8


def test_cell_value_rejects_non_numbers() -> None:
    blob = {"experiments": {"exp1": {"results": {"ours": {"acc": True, "f1": "0.9"}}}}}
    assert nz._cell_value(blob, "exp1", "ours", "acc") is None
    assert nz._cell_value(blob, "exp1", "ours", "f1") is None


def test_cell_value_rejects_non_finite_numbers(eval_config) -> None:
    blob = {"experiments": {"exp1": {"results": {"ours": {
        "acc": float("nan"), "loss": float("inf")}}}}}
    assert nz._cell_value(blob, "exp1", "ours", "acc") is None
    assert nz._cell_value(blob, "exp1", "ours", "loss") is None

    # One NaN cohort value must not poison every coefficient.
    data_dir = eval_config(profiles=_profiles("m1", "m2", "m3", "m4"))
    paper_dir = _paper(data_dir)
    for name, scores in COHORT.items():
        _run(paper_dir, name, graded=scores)
    _run(paper_dir, "m4", graded=_scores(float("nan"), 2.0))
    nz.recompute_paper(PAPER)
    co = _coeffs()
    assert co["excluded"] is False
    assert all(math.isfinite(c["coefficient"]) for c in co["cells"])


def test_alias_fallback_single_other_row() -> None:
    """An agent that files its numbers under its own method name."""
    blob = _scores(0.8, 1.5, method="my_new_method")
    assert nz._cell_value(blob, "exp1", "ours", "acc") == 0.8


def test_alias_fallback_needs_exactly_one_candidate() -> None:
    blob = {"experiments": {"exp1": {"results": {
        "mine_a": {"acc": 0.8}, "mine_b": {"acc": 0.9}}}}}
    assert nz._cell_value(blob, "exp1", "ours", "acc") is None


def test_alias_fallback_never_takes_another_scored_method() -> None:
    blob = {"experiments": {"exp1": {"results": {"ours_large": {"acc": 0.9}}}}}
    known = frozenset({"ours_large"})
    assert nz._cell_value(blob, "exp1", "ours", "acc", known_methods=known) is None
    # ... and a known name does not count towards ambiguity either
    blob["experiments"]["exp1"]["results"]["mine"] = {"acc": 0.8}
    assert nz._cell_value(blob, "exp1", "ours", "acc", known_methods=known) == 0.8


def test_known_methods_are_the_other_scored_methods_of_the_experiment() -> None:
    cells = [
        {"experiment": "exp1", "method": "ours", "metric": "acc"},
        {"experiment": "exp1", "method": "ours_large", "metric": "acc"},
        {"experiment": "exp2", "method": "elsewhere", "metric": "acc"},
    ]
    assert nz._known_methods(cells, cells[0]) == frozenset({"ours_large"})


def test_alias_fallback_accepts_a_lone_baseline_row(base_paper: Path) -> None:
    """Known limitation (docs/scoring.md): the names excluded from the
    fallback are the paper's other *scored* methods. Baseline methods are
    not scored cells, so a file whose only row for the experiment is a
    baseline row is read as the agent's result."""
    only_baseline = {"experiments": {"exp1": {"results": {"base": {"acc": 0.5}}}}}
    res = nz.normalize_delta(_coeffs(), only_baseline)
    assert res["n_cells_used"] == 1
    assert res["delta_norm"] == pytest.approx(60 / 7 * (0.5 - 0.7))


# ── Dominant-cell pruning ────────────────────────────────────────────────


def test_dominant_cell_is_pruned_for_every_model(eval_config) -> None:
    """A model outside the cohort is far off on one cell: that cell's
    contribution exceeds one whole normal score and all its other cells
    together, so the cell is dropped for all models."""
    data_dir = eval_config(profiles=_profiles("m1", "m2", "m3"))
    paper_dir = _paper(data_dir)
    for name, scores in COHORT.items():
        _run(paper_dir, name, graded=scores)
    _run(paper_dir, "outlier", graded=_scores(1.20, 2.0))  # acc piece = 60/7 * 0.5
    nz.recompute_paper(PAPER)

    co = _coeffs()
    assert co["models_present"] == ["m1", "m2", "m3"]  # not a cohort model
    assert co["pruned_cells"] == [
        {"experiment": "exp1", "method": "ours", "metric": "acc"}]
    # Coefficients are not re-derived after pruning.
    assert _cell(co, "loss")["coefficient"] == pytest.approx(9 / 7)

    res = nz.normalize_delta(co, COHORT["m1"])
    assert res["n_cells_pruned"] == 1 and res["n_cells_used"] == 1
    assert res["delta_norm"] == pytest.approx(9 / 7 * 0.5)
    out = nz.normalize_delta(co, _scores(1.20, 2.0))
    assert out["delta_norm"] == pytest.approx(0.0)


def test_unverified_outlier_does_not_trigger_pruning(eval_config) -> None:
    data_dir = eval_config(profiles=_profiles("m1", "m2", "m3"))
    paper_dir = _paper(data_dir)
    for name, scores in COHORT.items():
        _run(paper_dir, name, graded=scores)
    _run(paper_dir, "outlier", status="unfixable", graded=_scores(1.20, 2.0),
         agent=_scores(1.20, 2.0))
    nz.recompute_paper(PAPER)
    assert _coeffs()["pruned_cells"] == []


def test_large_but_not_dominant_cell_is_kept() -> None:
    """Both conditions are needed: bigger than the rest AND above 1.0."""
    cells = [
        {"experiment": "e", "method": "m", "metric": "a", "coefficient": 1.0,
         "source": 0.0, "higher_is_better": True},
        {"experiment": "e", "method": "m", "metric": "b", "coefficient": 1.0,
         "source": 0.0, "higher_is_better": True},
    ]

    def agent(a: float, b: float) -> dict:
        return {"e": {"m": {"a": a, "b": b}}}

    # 5 is above 1.0 but does not outweigh the other cell
    assert nz._detect_pruned_cells(cells, {"x": agent(5.0, 5.0)}) == []
    # 0.9 outweighs the rest but is below one normal score
    assert nz._detect_pruned_cells(cells, {"x": agent(0.9, 0.1)}) == []
    # both conditions hold
    assert nz._detect_pruned_cells(cells, {"x": agent(5.0, 0.1)}) == [
        {"experiment": "e", "method": "m", "metric": "a"}]


def test_pruning_is_capped() -> None:
    """Each cell dominates the ones after it; without the cap seven of the
    eight would go."""
    cells = [
        {"experiment": "e", "method": "m", "metric": f"c{i}", "coefficient": 1.0,
         "source": 0.0, "higher_is_better": True}
        for i in range(8)
    ]
    agent = {"e": {"m": {f"c{i}": float(10 ** (7 - i)) for i in range(8)}}}
    pruned = nz._detect_pruned_cells(cells, {"x": agent})
    assert len(pruned) == nz.MAX_PRUNED_CELLS == 5
    assert [p["metric"] for p in pruned] == ["c0", "c1", "c2", "c3", "c4"]


# ── Locking ──────────────────────────────────────────────────────────────


def _four_model_paper(eval_config) -> Path:
    data_dir = eval_config(profiles=_profiles("m1", "m2", "m3", "m4", "m5"))
    paper_dir = _paper(data_dir, test=SOURCE)
    runs = dict(COHORT, m4=_scores(0.75, 1.8))
    for name, scores in runs.items():
        _run(paper_dir, name, graded=scores, phases=("train", "test"))
    return paper_dir


def test_three_models_do_not_lock(eval_config) -> None:
    data_dir = eval_config(profiles=_profiles("m1", "m2", "m3"))
    paper_dir = _paper(data_dir, test=SOURCE)
    for name, scores in COHORT.items():
        _run(paper_dir, name, graded=scores, phases=("train", "test"))
    nz.update_for_paper(PAPER)
    assert _coeffs("test")["excluded"] is False
    assert not _coeffs("test").get("locked")
    assert nz.is_mode_locked(PAPER, "improve") is False


def test_four_models_lock_both_phases(eval_config) -> None:
    _four_model_paper(eval_config)
    nz.update_for_paper(PAPER)
    for phase in ("train", "test"):
        co = _coeffs(phase)
        assert co["locked"] is True
        assert co["locked_n_models"] == 4 == nz.AUTO_LOCK_MIN_MODELS
        assert co["locked_at"]
    assert nz.is_mode_locked(PAPER, "improve") is True
    assert nz.is_mode_locked(PAPER, "innovate") is False


def test_locked_coefficients_are_never_recomputed(eval_config) -> None:
    paper_dir = _four_model_paper(eval_config)
    nz.update_for_paper(PAPER)
    files = {ph: nz._paper_coeffs_path(PAPER, ph, "improve") for ph in ("train", "test")}
    before = {ph: p.read_text() for ph, p in files.items()}

    # A fifth cohort run arrives: the frozen files must not move ...
    _run(paper_dir, "m5", graded=_scores(0.95, 0.5), phases=("train", "test"))
    summary = nz.recompute_paper(PAPER)
    assert {ph: p.read_text() for ph, p in files.items()} == before
    assert summary["modes"]["improve"]["test"]["locked"] is True
    # ... and the new run is scored against them.
    res = nz.normalize_delta(_coeffs("test"), _scores(0.95, 0.5))
    assert res["n_models"] == 4
    assert res["delta_norm"] is not None


def test_mode_without_a_test_phase_never_locks(eval_config) -> None:
    data_dir = eval_config(profiles=_profiles("m1", "m2", "m3", "m4"))
    paper_dir = _paper(data_dir)  # no source scores_test.json
    for name, scores in dict(COHORT, m4=_scores(0.75, 1.8)).items():
        _run(paper_dir, name, graded=scores)
    nz.update_for_paper(PAPER)
    assert nz.load_coeffs(PAPER, "test", "improve") is None
    assert not _coeffs("train").get("locked")


def test_excluded_mode_cannot_be_locked(eval_config) -> None:
    paper_dir = _paper(eval_config(profiles=_profiles("m1", "m2", "m3")), test=SOURCE)
    _run(paper_dir, "m1", graded=COHORT["m1"], phases=("train", "test"))
    nz.recompute_paper(PAPER)
    res = nz.lock_paper_mode(PAPER, "improve", force=True)
    assert res["locked"] is False and res["reason"] == "excluded"


# ── normalize_delta: reasons ─────────────────────────────────────────────


def test_normalize_delta_reasons(base_paper: Path) -> None:
    co = _coeffs()
    assert nz.normalize_delta(None, COHORT["m1"]) == {
        "delta_norm": None, "reason": "no_coeffs"}
    assert nz.normalize_delta(co, None)["reason"] == "no_score"
    assert nz.normalize_delta({"excluded": True}, COHORT["m1"])["reason"] == "paper_excluded"
    assert nz.normalize_delta({"cells": []}, COHORT["m1"])["reason"] == "no_cells"
    unrelated = {"experiments": {"other": {"results": {"x": {"y": 1.0}}}}}
    res = nz.normalize_delta(co, unrelated)
    assert res["delta_norm"] is None and res["reason"] == "no_cells_used"
    assert res["n_cells_missing"] == 2


# ── Per-mode summary ─────────────────────────────────────────────────────


def test_global_summary(base_paper: Path) -> None:
    summary = nz.recompute_global_sd()
    imp = summary["improve"]
    assert imp["n_papers_train"] == 1 and imp["n_excluded_train"] == 0
    assert imp["median_baseline_train"] == pytest.approx(2 / 3)
    assert imp["n_papers_test"] == 0  # the source has no test scores
    assert summary["innovate"]["n_excluded_train"] == 1
    assert nz.load_global_sd("improve")["median_baseline_train"] == pytest.approx(2 / 3)
    assert set(imp) == {
        "mode", "n_papers_train", "n_papers_test", "n_excluded_train",
        "n_excluded_test", "median_baseline_train", "median_baseline_test",
        "last_updated"}
