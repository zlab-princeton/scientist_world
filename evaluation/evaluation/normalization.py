"""Per-paper, cohort-relative score normalization (FNI).

``docs/scoring.md`` is the full description; this is the summary.

For one (paper, mode, phase), over the cohort models that have a clean
grader verdict:

1. For each scoring cell (experiment, method, metric), every cohort
   model's signed *raw* improvement against the paper's own
   ``source/scoring/scores_<phase>.json``:
   ``(agent - source) × direction``, no division by source, no clamp.
2. Per cell, the sample standard deviation across the cohort is the
   cell's *spread*. Per paper, ``typical_spread`` is the **median** of
   the non-zero cell spreads.
3. Per cell ``normalizer``: the cell's own spread if > 0, else
   ``typical_spread``.
4. Per cell ``effective_weight = prior × (0.5 + spread / (spread + typical_spread))``,
   renormalized over the cells to sum to 1. The multiplier ranges over
   [0.5, 1.5).
5. Per cohort model, ``paper_score_raw = Σ effective_weight × (improvement / normalizer)``.
6. ``paper_baseline = mean |paper_score_raw|`` across the cohort: the
   per-paper scale that makes the cohort's mean |paper_score| equal 1
   on every paper.
7. Stored per-cell ``coefficient = effective_weight / (normalizer × paper_baseline)``.

Scoring any model (cohort or not):
``paper_score = Σ coefficient × ((agent - source) × direction)`` over
the cells it reports. Displayed × 100, this is the Frontier-Normalized
Improvement.

On top of that: no coefficients with fewer than three cohort models
(the file is written with ``excluded: true``); cells whose contribution
dominates a model's score are pruned for every model
(``_detect_pruned_cells``); and once a mode's test-phase file has four
cohort models its coefficients are locked and never recomputed.

Storage: ``<data_dir>/envs/<paper>/normalization/coeffs_<mode>_<phase>.json``
for mode in {innovate, improve} and phase in {train, test}, recomputed
at the end of every run of the paper until locked.

``reference.json`` is read only for the per-experiment ``weight``, the
per-metric ``coefficient`` and ``higher_is_better``, and the per-method
``type``. It is never a source of compared values: the anchor is always
the source's own ``scores_*.json``. Run workspaces are only read here.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path

from evaluation import config as _config

log = logging.getLogger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────

# Default for ``profile_cohort_exclude`` (see _cohort).
DEFAULT_COHORT_EXCLUDE: tuple[str, ...] = ()

# Phases: one coefficient file per scoring split.
PHASES = ("train", "test")

# Modes. Each mode has its own per-paper coefficients, because runs
# from different modes are not comparable.
MODES = ("innovate", "improve")

# Minimum number of cohort models (with a clean grader verdict) before
# coefficients are computed for a (paper, mode, phase). With fewer, the
# per-cell spread is unreliable: the sample standard deviation of two
# numbers is |a-b|/√2, and a near-tie gives a tiny spread, hence a huge
# 1/spread coefficient and out-of-range scores for other models.
MIN_COHORT_PRESENT = 3

# Dominant-cell pruning. A cell is pruned (for ALL models, recorded in
# the coefficient file) when, for any model scored on the paper, that
# single cell's contribution (coefficient × raw improvement) both:
#   (a) outweighs all the paper's other cells combined, AND
#   (b) exceeds one full normal score in magnitude (|piece| > 1.0; the
#       cohort is calibrated so its mean |paper_score| is 1.0, so a
#       single cell larger than that is a pathology, not signal).
# This catches saturated metrics, where a near-zero cohort spread
# inflates a coefficient so that a small miss by another model becomes
# a score orders of magnitude outside the normal range. Both conditions
# are data-derived; the only fixed number is the cap below.
PRUNE_DOMINANCE_SCORE = 1.0
# Never prune more than this many cells from one paper (per mode ×
# phase), so that a paper whose scoring is broken cannot have most of
# its cells removed.
MAX_PRUNED_CELLS = 5

# Lock-in: once a (paper, mode) has at least this many cohort models
# scored, its coefficients are frozen so the score of any later
# submission is stable. A locked mode is never recomputed; both its
# phases (train + test) freeze together.
AUTO_LOCK_MIN_MODELS = 4

_LOCK = threading.Lock()


# ── Config helpers ────────────────────────────────────────────────────

def _cohort() -> list[str]:
    """Profile names that form the normalization cohort.

    ``profile_cohort`` from the config when it is set, otherwise every
    configured profile (the keys of ``profiles``); in both cases minus
    ``profile_cohort_exclude``. With the default (empty) settings every
    new profile is part of the cohort automatically.
    """
    cfg = _config.load()
    profiles = list((cfg.get("profiles") or {}).keys())
    exclude = set(cfg.get("profile_cohort_exclude") or DEFAULT_COHORT_EXCLUDE)
    explicit = cfg.get("profile_cohort")
    if explicit:
        return [p for p in explicit if p not in exclude]
    return [p for p in profiles if p not in exclude]


# ── Lock helpers (coefficient freeze) ─────────────────────────────────

def _is_locked(data: dict | None) -> bool:
    return bool(data and data.get("locked"))


def is_mode_locked(paper_id: str, mode: str) -> bool:
    """A (paper, mode) is locked iff its test-phase coeffs are locked.

    We always lock both phases of a mode together, so the test phase is
    an authoritative single check.
    """
    return _is_locked(_read_json(_paper_coeffs_path(paper_id, "test", mode)))


def lock_paper_mode(
    paper_id: str, mode: str,
    min_models: int = 0, force: bool = False,
) -> dict:
    """Freeze the current coefficients for both phases of one mode.

    Sets ``locked: true`` (+ ``locked_at`` / ``locked_n_models``) on the
    existing ``coeffs_<mode>_<phase>.json`` files in place — it does NOT
    recompute. After this, ``recompute_paper`` skips the mode entirely so
    the published score stops drifting as new submissions arrive.

    Refuses to lock when:
      - the coeffs don't exist or the mode is ``excluded`` (not enough
        cohort spread to score); ``force`` does not override this, it
        only bypasses ``min_models``;
      - ``n_models < min_models`` and not ``force``.

    Returns a status dict ``{locked: bool, n_models, reason?}``.
    """
    with _LOCK:
        result = {"paper_id": paper_id, "mode": mode, "locked": False}
        paths = {ph: _paper_coeffs_path(paper_id, ph, mode) for ph in PHASES}
        datas = {ph: _read_json(p) for ph, p in paths.items()}
        test = datas.get("test")
        if not test:
            result["reason"] = "no_coeffs"
            return result
        if test.get("excluded"):
            # An excluded mode has no usable coefficients to freeze.
            result["reason"] = "excluded"
            return result
        if _is_locked(test):
            result["locked"] = True
            result["reason"] = "already_locked"
            result["n_models"] = int(test.get("n_models", 0) or 0)
            return result
        n_models = int(test.get("n_models", 0) or 0)
        if not force and n_models < min_models:
            result["reason"] = f"n_models {n_models} < {min_models}"
            result["n_models"] = n_models
            return result
        stamp = datetime.now(timezone.utc).isoformat()
        for ph, d in datas.items():
            if not d:
                continue
            d["locked"] = True
            d["locked_at"] = stamp
            d["locked_n_models"] = int(d.get("n_models", 0) or 0)
            _atomic_write(paths[ph], d)
        result["locked"] = True
        result["n_models"] = n_models
        return result


# ── Path helpers ──────────────────────────────────────────────────────

def _paper_coeffs_path(paper_id: str, phase: str, mode: str) -> Path:
    """Per-paper variance-norm coeffs path, scoped by mode and phase.

    The ``mode`` segment in the filename keeps innovate vs. improve
    accounting separate — runs from different modes are not comparable
    and must not be averaged into the same coefficient set.
    """
    return (
        _config.envs_dir() / paper_id / "normalization"
        / f"coeffs_{mode}_{phase}.json"
    )


def _global_sd_path(mode: str) -> Path:
    return _config.data_dir() / "normalization" / f"global_sd_{mode}.json"


# ── JSON helpers ──────────────────────────────────────────────────────

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


# ── Cell extraction ───────────────────────────────────────────────────

def _flatten_experiments(score: dict | None) -> dict:
    """Experiments of a score file, in either layout: wrapped
    (``{"experiments": {...}}``) or flat (top-level keys are experiments)."""
    if not isinstance(score, dict):
        return {}
    exps = score.get("experiments")
    if isinstance(exps, dict):
        return exps
    return {k: v for k, v in score.items() if isinstance(v, dict)}


def _known_methods(cells: list, current: dict) -> frozenset:
    """Other method names the paper's scoring knows for this experiment —
    excluded as alias-fallback candidates so a flexible lookup can never
    grab a different scored method's row."""
    exp = current.get("experiment")
    return frozenset(
        c.get("method") for c in (cells or [])
        if c.get("experiment") == exp and c.get("method") != current.get("method")
    )


def _cell_value(
    score_blob: dict, exp: str, method: str, metric: str,
    known_methods: frozenset = frozenset(),
) -> float | None:
    """Return the agent's float value for a given cell, or None if absent.

    Name-flexible lookup: innovate-mode agents build methods whose own
    name differs from the paper's method name that keys the cell, and an
    exact-name-only lookup would drop their results (or push agents to
    file their numbers under the paper's method name). When the exact
    key is missing, fall back to the agent's method row for this metric,
    but only when it is unambiguous: exactly one candidate row that
    carries the metric and is not one of the paper's other known method
    names. Ambiguity (e.g. baseline rows recorded alongside) keeps the
    strict behaviour.
    """
    exps = _flatten_experiments(score_blob)
    e = exps.get(exp)
    if not isinstance(e, dict):
        return None
    results = e.get("results") if isinstance(e.get("results"), dict) else e
    if not isinstance(results, dict):
        return None

    def _num(d: dict) -> float | None:
        v = d.get(metric)
        if (isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v)):
            return float(v)
        return None

    m = results.get(method)
    if isinstance(m, dict):
        v = _num(m)
        if v is not None:
            return v
    # Unambiguous-alias fallback.
    cands = [
        k for k, d in results.items()
        if isinstance(d, dict) and k != method
        and k not in known_methods and _num(d) is not None
    ]
    if len(cands) == 1:
        return _num(results[cands[0]])
    return None


def _raw_improvement(agent: float, source: float, higher_is_better: bool) -> float:
    """Sign-preserved raw improvement.

    No division by source: variance normalization (cohort spread) absorbs
    metric scale. Cells with source = 0 work unchanged. No clamp.
    """
    return (agent - source) if higher_is_better else (source - agent)


def _enumerate_proposed_cells(
    source_score: dict,
    reference: dict,
) -> list[dict]:
    """List proposed-method cells with their REFERENCE prior weights.

    Cells are skipped when:
    - the source value isn't numeric;
    - the method's REFERENCE type is ``baseline`` (out of scope for the agent);
    - source has no entry for the (exp, method, metric) (we only score
      cells the source itself reproduces).
    """
    out: list[dict] = []
    exps_src = _flatten_experiments(source_score)
    exps_ref = (reference or {}).get("experiments") or {}
    if not isinstance(exps_ref, dict):
        exps_ref = {}
    for exp_name, src_exp in exps_src.items():
        if not isinstance(src_exp, dict):
            continue
        ref_exp = exps_ref.get(exp_name) or {}
        exp_weight = float(ref_exp.get("weight", 1.0) or 1.0)
        metrics_meta = ref_exp.get("metrics") or {}
        ref_methods = ref_exp.get("results") or {}
        src_results = src_exp.get("results") if isinstance(src_exp.get("results"), dict) else src_exp
        for method, src_metrics in (src_results or {}).items():
            if not isinstance(src_metrics, dict):
                continue
            ref_method_entry = ref_methods.get(method) if isinstance(ref_methods, dict) else None
            method_type = (
                ref_method_entry.get("type")
                if isinstance(ref_method_entry, dict) else None
            )
            if method_type == "baseline":
                continue  # paper baseline; agent isn't responsible
            for metric, src_val in src_metrics.items():
                if not isinstance(src_val, (int, float)) or isinstance(src_val, bool):
                    continue
                meta = metrics_meta.get(metric) if isinstance(metrics_meta, dict) else None
                if not isinstance(meta, dict):
                    meta = {}
                coeff = float(meta.get("coefficient", 1.0) or 0.0)
                higher = bool(meta.get("higher_is_better", True))
                prior_weight = abs(coeff) * exp_weight
                out.append({
                    "experiment": exp_name,
                    "method": method,
                    "metric": metric,
                    "source": float(src_val),
                    "higher_is_better": higher,
                    "prior_weight": prior_weight,
                })
    return out


# ── Per-paper recompute ───────────────────────────────────────────────

#: Verdicts that make a run's numbers usable for normalization. A run
#: without one of these was never checked, and a spread built from
#: unchecked numbers is a spread built from whatever the agent chose to
#: write. ``verified`` means the grader re-ran the frozen method.
#: ``audit_clean`` marks a run that was inspected by hand and found
#: clean but not re-executed, so it carries no scores of its own; the
#: harness never writes this status itself.
_CLEAN_VERDICTS = frozenset({"verified", "audit_clean"})


def _run_scores(run_dir: Path, phase: str) -> dict | None:
    """The authoritative scores for one run, or None if it has none.

    Prefers the grader's re-run values in ``scoring/verification.json``
    over the agent's own ``scoring/scores_<phase>.json``. The two agree
    for most cells, but where they disagree the agent's file is the
    claim and the grader's is the measurement. Runs whose verdict is not
    clean return None.

    The verdict is also what decides that a run *finished*, rather than
    ``.evaluation.json``'s ``state``: that field tracks liveness during
    a run (working, waiting, done) and can be left stale, whereas a
    graded run is finished by construction.
    """
    verdict = _read_json(run_dir / "scoring" / "verification.json") or {}
    if verdict.get("status") not in _CLEAN_VERDICTS:
        return None
    graded = verdict.get(f"scores_{phase}")
    if graded is not None:
        return graded
    # audit_clean: inspected, not re-executed, so the agent's file stands.
    return _read_json(run_dir / "scoring" / f"scores_{phase}.json")


def _canonical(prof: str) -> str:
    """The published identity of a profile (``publish_as``, else itself).

    Two profiles that serve the same model through different endpoints
    have separate run directories but one identity. Counting both would
    enter the same model into the spread twice.
    """
    from evaluation.models import load_profiles
    prof_cfg = load_profiles().get(prof)
    return (getattr(prof_cfg, "publish_as", None) or prof) if prof_cfg else prof


def _gather_cohort_scores(
    paper_dir: Path, phase: str, cohort: list[str], mode: str,
) -> dict[str, dict]:
    """Authoritative scores per cohort model, keyed by published identity.

    One entry per *identity*, not per profile: a model that ran under
    two profiles with the same ``publish_as`` contributes once. When
    both have clean runs, the profile whose name equals the identity
    wins; otherwise the last one in cohort order.

    Innovate and improve are accounted separately because their score
    scales aren't comparable.
    """
    out: dict[str, dict] = {}
    chosen: dict[str, str] = {}
    for prof in cohort:
        # Run dir naming: <profile>-<mode>-eval.
        run_dir = paper_dir / f"{prof}-{mode}-eval"
        if not run_dir.is_dir():
            continue
        scores = _run_scores(run_dir, phase)
        if scores is None:
            continue
        ident = _canonical(prof)
        if ident in out and chosen.get(ident) == ident:
            continue            # canonical profile already took this slot
        out[ident] = scores
        chosen[ident] = prof
    return out


def _compute_phase_coeffs(
    paper_id: str,
    phase: str,
    cohort: list[str],
    cell_specs: list[dict],
    cohort_scores: dict[str, dict],
) -> dict:
    """Coefficient computation for one (paper, phase).

    Returns a dict ready to be written as the coefficient file. When no
    coefficients can be produced (too few cohort models, no cells, all
    cells tied, all prior weights zero, or paper_baseline = 0) the file
    still records the cohort and diagnostics but is flagged
    ``"excluded": true`` with a ``reason``, so every consumer skips the
    paper uniformly.
    """
    present = sorted(cohort_scores.keys())
    n_present = len(present)
    base = {
        "paper_id": paper_id,
        "phase": phase,
        "cohort": list(cohort),
        "models_present": present,
        "n_models": n_present,
        "n_cells": len(cell_specs),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }

    # Cohort-size gate (see MIN_COHORT_PRESENT): no coefficients until
    # enough cohort models have been graded. Consumers show no score
    # until then, as for any other excluded paper.
    if n_present < MIN_COHORT_PRESENT:
        return {**base, "excluded": True,
                "reason": "cohort_too_small",
                "min_required": MIN_COHORT_PRESENT,
                "typical_spread": 0.0, "cells": []}

    # Step 1: per cell, gather cohort raw improvements + spread.
    enriched_cells: list[dict] = []
    for spec in cell_specs:
        cohort_imp: dict[str, float] = {}
        for prof, ascore in cohort_scores.items():
            av = _cell_value(ascore, spec["experiment"],
                             spec["method"], spec["metric"],
                             known_methods=_known_methods(cell_specs, spec))
            if av is None:
                continue
            cohort_imp[prof] = _raw_improvement(
                av, spec["source"], spec["higher_is_better"])
        if not cohort_imp:
            # No cohort model produced this cell; can't compute spread.
            continue
        imps = list(cohort_imp.values())
        spread = statistics.stdev(imps) if len(imps) >= 2 else 0.0
        enriched_cells.append({**spec, "spread": float(spread),
                               "cohort_imp": cohort_imp})

    if not enriched_cells:
        return {**base, "excluded": True, "reason": "no_cells",
                "cells": []}

    # Step 2: typical_spread = median of nonzero spreads.
    nonzero_spreads = [c["spread"] for c in enriched_cells if c["spread"] > 0]
    typical_spread = statistics.median(nonzero_spreads) if nonzero_spreads else 0.0

    if typical_spread <= 0:
        # All spreads are zero: the cohort tied on every cell (the
        # "< 2 models" case cannot occur past the cohort-size gate
        # above). No coefficients can be produced; exclude until that
        # changes.
        return {**base, "excluded": True,
                "reason": ("cohort_too_small" if n_present < 2
                           else "all_cells_tied"),
                "typical_spread": 0.0, "cells": []}

    # Step 3: per cell normalizer (spread if >0 else typical_spread).
    # Step 4: effective_weight = prior × (0.5 + spread / (spread + typical_spread)).
    #         Renormalized to sum to 1.
    raw_weights: list[float] = []
    for c in enriched_cells:
        spread = c["spread"]
        normalizer = spread if spread > 0 else typical_spread
        # spread = 0 → multiplier = 0.5 (tied cells still contribute,
        # but at half weight). spread = typical → 1.0. extreme → ~1.5.
        multiplier = 0.5 + spread / (spread + typical_spread)
        raw_w = float(c["prior_weight"]) * multiplier
        c["normalizer"] = float(normalizer)
        c["multiplier"] = float(multiplier)
        c["raw_weight"] = float(raw_w)
        raw_weights.append(raw_w)
    weight_sum = sum(raw_weights)
    if weight_sum <= 0:
        return {**base, "excluded": True, "reason": "all_priors_zero",
                "typical_spread": float(typical_spread), "cells": []}
    for c in enriched_cells:
        c["effective_weight"] = c["raw_weight"] / weight_sum

    # Step 5: paper_score_raw per cohort model.
    cohort_raw: dict[str, float] = {}
    for prof in present:
        total = 0.0
        for c in enriched_cells:
            imp = c["cohort_imp"].get(prof)
            if imp is None:
                continue
            total += c["effective_weight"] * (imp / c["normalizer"])
        cohort_raw[prof] = total

    # Step 6: paper_baseline = mean |paper_score_raw| across cohort.
    raws = [abs(v) for v in cohort_raw.values()]
    paper_baseline = statistics.fmean(raws) if raws else 0.0
    if paper_baseline <= 0:
        # Cohort all matched source on every cell, or canceled
        # exactly. Either way no scale information available.
        return {**base, "excluded": True, "reason": "paper_baseline_zero",
                "typical_spread": float(typical_spread),
                "cohort_raw": cohort_raw, "cells": []}

    # Step 7: per-cell coefficient absorbs everything frozen.
    out_cells: list[dict] = []
    for c in enriched_cells:
        coefficient = c["effective_weight"] / (c["normalizer"] * paper_baseline)
        out_cells.append({
            "experiment": c["experiment"],
            "method": c["method"],
            "metric": c["metric"],
            "source": c["source"],
            "higher_is_better": c["higher_is_better"],
            "coefficient": coefficient,
            # Diagnostics (not used at runtime, helpful for debugging /
            # eyeballing why a cell weighs what it weighs):
            "prior_weight": c["prior_weight"],
            "spread": c["spread"],
            "normalizer": c["normalizer"],
            "effective_weight": c["effective_weight"],
            "cohort_imp": c["cohort_imp"],
        })

    return {
        **base,
        "excluded": False,
        "typical_spread": float(typical_spread),
        "paper_baseline": float(paper_baseline),
        "cohort_raw": cohort_raw,
        "cells": out_cells,
    }


def _gather_all_scores(
    paper_dir: Path, phase: str, mode: str,
) -> dict[str, dict]:
    """Read EVERY model's scores_<phase>.json for this mode (done runs).

    Unlike ``_gather_cohort_scores`` (cohort only, used to build
    coefficients), this returns all models, cohort or not. Pruning
    detection must see non-cohort models too: a cell can dominate only
    for a model that is not in the cohort (the cohort models all matched
    the source there, so the cell looks tied among them).

    Same authority rule as the cohort gather: graded scores where the
    grader produced them, and nothing at all from a run that was never
    checked. Pruning decides which cells count for every model on the
    paper, so it must not be driven by an unverified outlier.
    """
    out: dict[str, dict] = {}
    suffix = f"-{mode}-eval"
    if not paper_dir.is_dir():
        return out
    for run_dir in paper_dir.iterdir():
        if not (run_dir.is_dir() and run_dir.name.endswith(suffix)):
            continue
        prof = run_dir.name[: -len(suffix)]
        scores = _run_scores(run_dir, phase)
        if scores is None:
            continue
        out[prof] = scores
    return out


def _detect_pruned_cells(
    cells: list[dict], all_scores: dict[str, dict],
) -> list[dict]:
    """Return the list of cells to prune for ALL models on this paper.

    A cell is pruned if, for ANY scored model, its contribution both
    dominates the paper (|piece| > sum of all other |pieces|) and is
    larger than one full normal score (|piece| > PRUNE_DOMINANCE_SCORE).
    Iterated so a paper with several pathological cells is fully cleaned,
    but capped at MAX_PRUNED_CELLS. The gate (b) stops the cascade once
    the remaining cells are normal-sized, so it never strips healthy
    metrics (the "biggest of small leftovers" bug a pure-dominance rule
    would hit).

    Returns cell-key dicts ``{experiment, method, metric}`` — the audit
    record proving the same set was dropped uniformly for every model.
    """
    if not cells or not all_scores:
        return []
    keys = [(c["experiment"], c["method"], c["metric"]) for c in cells]
    by_key = {k: c for k, c in zip(keys, cells)}

    def _piece(cell: dict, agent: dict) -> float | None:
        av = _cell_value(agent, cell["experiment"], cell["method"], cell["metric"],
                         known_methods=_known_methods(cells, cell))
        if av is None:
            return None
        return cell["coefficient"] * _raw_improvement(
            av, float(cell.get("source", 0.0) or 0.0),
            bool(cell.get("higher_is_better", True)))

    excluded: set = set()
    while len(excluded) < MAX_PRUNED_CELLS:
        active = [k for k in keys if k not in excluded]
        found = None
        for agent in all_scores.values():
            pieces = []
            for k in active:
                p = _piece(by_key[k], agent)
                if p is not None:
                    pieces.append((k, p))
            if len(pieces) < 2:
                continue
            pieces.sort(key=lambda kp: abs(kp[1]), reverse=True)
            top_key, top_val = pieces[0]
            rest = sum(abs(v) for _, v in pieces[1:])
            if abs(top_val) > rest and abs(top_val) > PRUNE_DOMINANCE_SCORE:
                found = top_key
                break
        if found is None:
            break
        excluded.add(found)
    return [{"experiment": e, "method": m, "metric": me}
            for (e, m, me) in keys if (e, m, me) in excluded]


def recompute_paper(paper_id: str) -> dict:
    """Rebuild per-cell coefficients for one paper, all modes × phases.

    Writes ``coeffs_<mode>_<phase>.json`` (one per mode × phase) under
    ``<envs>/<paper>/normalization/``. Returns a summary dict keyed by
    mode.

    Until at least ``MIN_COHORT_PRESENT`` cohort models have a clean
    grader verdict in a given mode, that mode's coefficient file is
    written with ``"excluded": true``. As more cohort runs are graded
    the file is rewritten, until the mode is locked.
    """
    with _LOCK:
        envs = _config.envs_dir()
        paper_dir = envs / paper_id
        if not paper_dir.is_dir():
            return {"paper_id": paper_id, "skipped": "paper_dir_missing"}
        source_dir = paper_dir / "source" / "scoring"
        reference = _read_json(source_dir / "reference.json")
        if reference is None:
            return {"paper_id": paper_id, "skipped": "no_reference"}

        cohort = _cohort()
        summary: dict = {"paper_id": paper_id, "modes": {}}
        for mode in MODES:
            mode_summary: dict = {}
            # A locked mode's coefficients are final: never recompute
            # them, no matter how many new runs arrive. Both phases lock
            # together, so check once per mode.
            if is_mode_locked(paper_id, mode):
                for phase in PHASES:
                    existing = _read_json(_paper_coeffs_path(paper_id, phase, mode))
                    if existing is not None:
                        mode_summary[phase] = {
                            "n_cells": len(existing.get("cells") or []),
                            "n_models": existing.get("n_models", 0),
                            "excluded": bool(existing.get("excluded")),
                            "locked": True,
                        }
                summary["modes"][mode] = mode_summary
                continue
            for phase in PHASES:
                src = _read_json(source_dir / f"scores_{phase}.json")
                if src is None:
                    continue
                cell_specs = _enumerate_proposed_cells(src, reference)
                cohort_scores = _gather_cohort_scores(
                    paper_dir, phase, cohort, mode)
                data = _compute_phase_coeffs(
                    paper_id, phase, cohort, cell_specs, cohort_scores)
                data["mode"] = mode
                # Dominant-metric pruning: detect across ALL models (not
                # just cohort), record the dropped set in the coeffs file
                # so normalize_delta skips the same cells for every model.
                if not data.get("excluded") and data.get("cells"):
                    all_scores = _gather_all_scores(paper_dir, phase, mode)
                    data["pruned_cells"] = _detect_pruned_cells(
                        data["cells"], all_scores)
                else:
                    data["pruned_cells"] = []
                _atomic_write(
                    _paper_coeffs_path(paper_id, phase, mode), data)
                mode_summary[phase] = {
                    "n_cells": len(data.get("cells") or []),
                    "n_models": data.get("n_models", 0),
                    "excluded": bool(data.get("excluded")),
                }
            summary["modes"][mode] = mode_summary
        return summary


# ── Project-level recompute (informational only) ─────────────────────

def recompute_global_sd() -> dict:
    """Per-mode summary over all papers: how many have coefficients,
    how many are excluded, and the median ``paper_baseline``.

    Informational only (nothing in the scoring reads it). Writes one
    ``global_sd_<mode>.json`` per mode and returns a dict keyed by mode.
    """
    with _LOCK:
        envs = _config.envs_dir()
        result: dict[str, dict] = {}
        for mode in MODES:
            n_papers: dict[str, int] = {p: 0 for p in PHASES}
            n_excluded: dict[str, int] = {p: 0 for p in PHASES}
            baselines: dict[str, list[float]] = {p: [] for p in PHASES}
            if envs.exists():
                for paper_dir in envs.iterdir():
                    if not paper_dir.is_dir():
                        continue
                    for phase in PHASES:
                        coeffs = _read_json(
                            _paper_coeffs_path(paper_dir.name, phase, mode))
                        if not coeffs:
                            continue
                        n_papers[phase] += 1
                        if coeffs.get("excluded"):
                            n_excluded[phase] += 1
                            continue
                        pb = coeffs.get("paper_baseline")
                        if isinstance(pb, (int, float)) and pb > 0:
                            baselines[phase].append(float(pb))
            data = {
                "mode": mode,
                "n_papers_train": n_papers["train"],
                "n_papers_test": n_papers["test"],
                "n_excluded_train": n_excluded["train"],
                "n_excluded_test": n_excluded["test"],
                "median_baseline_train": (statistics.median(baselines["train"])
                                          if baselines["train"] else 0.0),
                "median_baseline_test": (statistics.median(baselines["test"])
                                         if baselines["test"] else 0.0),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_write(_global_sd_path(mode), data)
            result[mode] = data
        return result


def maybe_autolock_and_sync(
    paper_id: str, min_models: int = AUTO_LOCK_MIN_MODELS,
) -> list[str]:
    """Lock any not-yet-locked mode that has reached ``min_models`` cohort
    models, then push the paper's frozen coefficients to git.

    The first time a (paper, mode) accumulates enough scored cohort
    models, its coefficients are frozen so that scores computed from
    them are stable from then on, and (when publishing is enabled) the
    frozen files are pushed to the paper's repository. Returns the list
    of modes newly locked by this call.
    """
    newly_locked: list[str] = []
    for mode in MODES:
        if is_mode_locked(paper_id, mode):
            continue
        test = _read_json(_paper_coeffs_path(paper_id, "test", mode))
        if not test or test.get("excluded"):
            continue
        if int(test.get("n_models", 0) or 0) < min_models:
            continue
        if lock_paper_mode(paper_id, mode, min_models=min_models).get("locked"):
            newly_locked.append(mode)
    if newly_locked:
        # Push the frozen coefficients to the paper's repository
        # (dedicated branch; a no-op unless publishing is enabled).
        # Best-effort: a push failure must not break the run path.
        try:
            from evaluation import git_publish
            git_publish.publish_coeffs(paper_id)
        except Exception:
            log.exception("publish_coeffs failed for %s (locked %s)",
                          paper_id, newly_locked)
    return newly_locked


def update_for_paper(paper_id: str) -> None:
    """Recompute one paper, refresh the per-mode summary, lock if due.

    Called at the end of every run and after every grading. Errors are
    logged, never raised: normalization must not break a run's
    finalization.
    """
    try:
        recompute_paper(paper_id)
        recompute_global_sd()
        maybe_autolock_and_sync(paper_id)
    except Exception:
        log.exception("Normalization recompute failed for %s", paper_id)


def recompute_all() -> dict:
    """Rebuild every paper's coefficients and the per-mode summary.

    Use after changing the cohort or after deleting the normalization
    files (locked modes are left untouched). CLI: ``renormalize``.
    """
    envs = _config.envs_dir()
    n = 0
    if envs.exists():
        for paper_dir in sorted(envs.iterdir()):
            if not paper_dir.is_dir():
                continue
            recompute_paper(paper_dir.name)
            n += 1
    g = recompute_global_sd()
    return {"papers_recomputed": n, "global_sd": g}


# ── Read path (for the dashboard) ─────────────────────────────────────

def load_coeffs(paper_id: str, phase: str, mode: str = "improve") -> dict | None:
    """Load a paper's coefficient file for a given phase and mode."""
    return _read_json(_paper_coeffs_path(paper_id, phase, mode))


def load_global_sd(mode: str = "improve") -> dict | None:
    """Load the per-mode summary written by :func:`recompute_global_sd`."""
    return _read_json(_global_sd_path(mode))


def normalize_delta(
    coeffs: dict | None,
    agent_score: dict | None,
) -> dict:
    """Compute the normalized ``paper_score`` for one (paper, model, phase).

    For any model, cohort or not: walks the stored coefficients, looks
    up the model's value per cell in ``agent_score``, computes the raw
    improvement against the source value, multiplies by the stored
    coefficient, and sums. The cohort's mean absolute score on a paper
    is 1.0 by construction (before pruning).

    Returns a dict with ``delta_norm`` (the paper_score; ``None`` when
    it cannot be computed, with a ``reason``) plus diagnostics
    (``n_cells_used``, ``n_cells_missing``, ``n_cells_pruned``,
    ``n_models``). Cells the model does not report are skipped, not
    penalised. Multiply by 100 for the displayed FNI.
    """
    if not coeffs:
        return {"delta_norm": None, "reason": "no_coeffs"}
    if coeffs.get("excluded"):
        return {
            "delta_norm": None,
            "reason": coeffs.get("reason") or "paper_excluded",
            "n_models": int(coeffs.get("n_models", 0) or 0),
        }
    if not isinstance(agent_score, dict):
        return {"delta_norm": None, "reason": "no_score",
                "n_models": int(coeffs.get("n_models", 0) or 0)}
    cells = coeffs.get("cells") or []
    if not cells:
        return {"delta_norm": None, "reason": "no_cells",
                "n_models": int(coeffs.get("n_models", 0) or 0)}

    # Pruned cells: dominant cells dropped for ALL models (recorded in
    # the coefficient file by recompute_paper). Skipped uniformly so
    # every model is scored on the same reduced cell set.
    pruned = {
        (p.get("experiment"), p.get("method"), p.get("metric"))
        for p in (coeffs.get("pruned_cells") or [])
    }

    total = 0.0
    n_used = 0
    n_missing = 0
    n_pruned = 0
    for cell in cells:
        if (cell.get("experiment"), cell.get("method"),
                cell.get("metric")) in pruned:
            n_pruned += 1
            continue
        coef = cell.get("coefficient")
        if not isinstance(coef, (int, float)) or math.isnan(coef):
            continue
        agent_val = _cell_value(
            agent_score,
            cell.get("experiment", ""),
            cell.get("method", ""),
            cell.get("metric", ""),
            known_methods=_known_methods(cells, cell),
        )
        if agent_val is None:
            n_missing += 1
            continue
        higher = bool(cell.get("higher_is_better", True))
        source = float(cell.get("source", 0.0) or 0.0)
        improvement = _raw_improvement(agent_val, source, higher)
        total += coef * improvement
        n_used += 1

    if n_used == 0:
        return {
            "delta_norm": None,
            "reason": "no_cells_used",
            "n_cells_missing": n_missing,
            "n_cells_pruned": n_pruned,
            "n_models": int(coeffs.get("n_models", 0) or 0),
        }
    return {
        "delta_norm": total,
        "n_cells_used": n_used,
        "n_cells_missing": n_missing,
        "n_cells_pruned": n_pruned,
        "n_models": int(coeffs.get("n_models", 0) or 0),
    }


__all__ = [
    "DEFAULT_COHORT_EXCLUDE",
    "MODES",
    "PHASES",
    "recompute_paper",
    "recompute_global_sd",
    "update_for_paper",
    "recompute_all",
    "load_coeffs",
    "load_global_sd",
    "normalize_delta",
]
