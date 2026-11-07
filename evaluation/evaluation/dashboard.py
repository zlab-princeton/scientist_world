"""Flask dashboard + run queue in a single process.

Endpoints:

    GET    /                       HTML page
    GET    /api/runs               all runs (scan of .evaluation.json files)
    GET    /api/run/<run_id>       one run, full record
    GET    /api/verification       grader verdict per run
    GET    /api/leaderboard        per-profile normalized score stats (?mode=)
    GET    /api/leaderboard.png    bar chart of the same (?mode=, ?group=)
    GET    /api/options            papers / profiles / modes for the queue form
    GET    /api/queue              pending and running runs, current caps
    POST   /api/queue              enqueue {papers, models, modes}
    DELETE /api/queue              remove pending runs {papers, models, modes}
    POST   /api/concurrency        change the concurrency caps

There is no authentication and the POST/DELETE endpoints start and drop
runs, so :func:`serve` binds the loopback interface unless ``host`` in
the config says otherwise.

The page polls its own endpoints every 60 s. The orchestrator pool
lives in ``evaluation.orchestrator`` and is reached directly (same
process). Display options (``dashboard.paper_groups``,
``dashboard.model_order``) come from the config.
"""

from __future__ import annotations

import io
import json
import logging
import math
import threading
import time
import os
import signal
import statistics
import sys
from pathlib import Path

from flask import Flask, Response, jsonify, request

from evaluation import config as _config
from evaluation import orchestrator

log = logging.getLogger(__name__)

app = Flask(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _json_safe(obj):
    """Recursively replace Infinity / -Infinity / NaN with None.

    Python's ``json.dumps`` emits bare ``Infinity`` / ``NaN`` tokens by
    default, which are not valid JSON per RFC 8259 and cause
    ``JSON.parse`` to reject the whole payload in the browser. Agents
    legitimately produce these values (e.g. ``psnr_db`` on a perfect
    reconstruction → ``+inf``), so sanitise at the API boundary rather
    than forcing every writer to clamp.
    """
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _extract_failure_detail(env_dir, meta_error: str = "") -> str | None:
    """Surface the actual cause of a failed run, combining the runner's
    one-liner (``meta['error']``) with whatever the last trajectory file
    tells us.

    Four cases:
      1. Last turn emitted a ``result`` with ``is_error=True``: the
         claude CLI's own error message (overloaded, prompt too long, ...)
      2. Last turn emitted a ``result`` with ``is_error=False`` but the
         run is still marked failed: it ended **between** turns (job
         dispatch, next-turn launch, OOM). The return code in
         ``meta_error`` is the clue.
      3. No result record: the process was killed mid-stream.
      4. No trajectory file at all: failed before the first turn
         (scaffold, container build, Nexus project creation).
    """
    env_dir = Path(env_dir)
    traj = env_dir / "trajectory"
    rc_hint = meta_error.strip() if meta_error else ""
    if not traj.exists() or not any(traj.glob("turn_*_raw.jsonl")):
        if rc_hint:
            return (
                f"Failed before first turn finished — no trajectory written. "
                f"Runner: {rc_hint}. Likely cause: scaffold / container build / "
                f"nexus project create."
            )
        return "Failed before first turn — no trajectory."
    turns = sorted(traj.glob("turn_*_raw.jsonl"))
    last_turn = turns[-1]
    turn_num = last_turn.stem.replace("turn_", "").replace("_raw", "")
    last_result = None
    last_text = None
    tool_count = 0
    scaffold_error = None  # type=error or type=turn.failed from codex/claude scaffold
    try:
        with open(last_turn) as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                t = o.get("type")
                if t == "result":
                    last_result = o
                elif t == "error":
                    msg = o.get("message")
                    if isinstance(msg, str) and msg.strip():
                        scaffold_error = msg.strip()
                elif t == "turn.failed":
                    err = o.get("error") or {}
                    msg = err.get("message") if isinstance(err, dict) else None
                    if isinstance(msg, str) and msg.strip():
                        scaffold_error = msg.strip()
                elif t == "assistant":
                    for c in o.get("message", {}).get("content", []) or []:
                        if c.get("type") == "tool_use":
                            tool_count += 1
                        elif c.get("type") == "text":
                            tx = c.get("text", "")
                            if tx:
                                last_text = tx
    except OSError:
        return rc_hint or None
    if last_result and last_result.get("is_error"):
        msg = str(last_result.get("result", "")).strip()
        if msg:
            return f"Turn {turn_num} ({tool_count} tool calls): {msg}"
    if last_result is None:
        # If the trajectory reported its own error (codex/claude scaffold
        # emitted type=error or type=turn.failed), surface that — it's the
        # actual cause. The runner's "OOM / SIGKILL" heuristic is wrong
        # whenever the process exited deliberately on an upstream error.
        if scaffold_error:
            tag = "scaffold error"
            low = scaffold_error.lower()
            if "refresh token" in low or "log out and sign in again" in low:
                tag = "codex auth — refresh-token race; re-login required"
            elif "access token" in low or "unauthorized" in low or "401" in low:
                tag = "auth failure"
            elif "rate" in low and "limit" in low:
                tag = "rate-limited"
            return (
                f"Turn {turn_num} ({tool_count} tool calls): {tag}: "
                f"{scaffold_error[:300]}"
            )
        tail = (last_text or "").strip().splitlines()[-1:] if last_text else []
        last_line = tail[0] if tail else "(no output before kill)"
        extra = f" — {rc_hint}" if rc_hint else ""
        return (
            f"Turn {turn_num} killed mid-stream after {tool_count} tool calls "
            f"— no result record (likely OOM / wall-time / SIGKILL){extra}. "
            f"Last text: {last_line[:200]}"
        )
    # Last turn completed successfully → kill happened between turns.
    if "rc=-9" in rc_hint:
        return (
            f"Turn {turn_num} completed successfully, then the run was killed "
            f"between turns (rc=-9 / SIGKILL — likely OOM under the "
            f"container memory cap, or host-side cleanup). "
            f"Check the harness log near the ended_at time."
        )
    if rc_hint:
        return (
            f"Turn {turn_num} completed successfully; run failed between turns. "
            f"Runner: {rc_hint}."
        )
    return f"Turn {turn_num} completed; cause unclear — check env_dir."


def _compute_comparison(
    agent_score: dict | None,
    source_score: dict | None,
    reference: dict | None,
) -> dict | None:
    """Compare the agent's scores against the pipeline's own paper-reproduction
    scores (``source/scoring/scores_{train,test}.json``), metric-by-metric.

    ``reference.json`` supplies only the per-experiment ``weight`` and the
    per-metric ``coefficient`` / ``higher_is_better`` used for aggregation.
    It is *never* used as a compared value — the anchor is always source.

    Returns ``{cells, n_better, n_worse, n_same, n_missing, n_total, delta}``
    or ``None`` if either score file is unusable.

    ``delta`` is a weighted mean of relative improvements — positive means
    the agent on average beat the source reproduction, negative means worse.
    ``n_missing`` counts cells the source reports that the agent never
    produced (these are surfaced prominently, never silently dropped).
    """
    if not isinstance(agent_score, dict) or not isinstance(source_score, dict):
        return None
    # Two scoring-file layouts exist in the wild:
    #   wrapped: {"experiments": {<exp>: {"results": {<method>: {<metric>: ...}}}}}
    #   flat   : {<exp>: {<method>: {<metric>: ...}}}
    def _flatten(score: dict) -> dict:
        exps = score.get("experiments")
        if isinstance(exps, dict):
            return exps
        # No wrapper — top-level keys are experiments. Filter out any
        # non-dict values just in case (e.g. "_metadata" ints).
        return {k: v for k, v in score.items() if isinstance(v, dict)}
    exps_agent = _flatten(agent_score)
    exps_source = _flatten(source_score)
    exps_ref = (reference or {}).get("experiments") or {}
    if not isinstance(exps_source, dict) or not exps_source:
        return None
    cells: list[dict] = []
    missing: list[dict] = []
    eps = 1e-9
    for exp_name, src_exp in exps_source.items():
        if not isinstance(src_exp, dict):
            continue
        ref_exp = exps_ref.get(exp_name, {}) if isinstance(exps_ref, dict) else {}
        metrics_meta = ref_exp.get("metrics") or {}
        exp_weight = float(ref_exp.get("weight", 1.0) or 1.0)
        # Per-method type from reference.json: "proposed" methods are
        # required (agent must reproduce them; missing = surface as ⚠);
        # "baseline" methods are out of scope (agent isn't expected to
        # reproduce them; skip cleanly). Methods with no type field, or
        # not in reference.json at all, default to required — never
        # silently skip without a positive signal.
        ref_methods = ref_exp.get("results") or {}
        # Both wrapped (results: {...}) and flat (methods at top level) layouts
        # exist; unwrap "results" if present, else use the experiment dict directly.
        src_results = src_exp.get("results") if isinstance(src_exp.get("results"), dict) else src_exp
        agent_exp = exps_agent.get(exp_name) if isinstance(exps_agent, dict) else None
        if isinstance(agent_exp, dict):
            agent_results = agent_exp.get("results") if isinstance(agent_exp.get("results"), dict) else agent_exp
        else:
            agent_results = {}
        for method, src_metrics in (src_results or {}).items():
            if not isinstance(src_metrics, dict):
                continue
            ref_method_entry = ref_methods.get(method) if isinstance(ref_methods, dict) else None
            method_type = ref_method_entry.get("type") if isinstance(ref_method_entry, dict) else None
            if method_type == "baseline":
                # Out of scope — paper baseline, not the agent's responsibility.
                continue
            agent_metrics = agent_results.get(method)
            for metric, src_val in src_metrics.items():
                if not isinstance(src_val, (int, float)):
                    continue
                meta = metrics_meta.get(metric) or {}
                coeff = float(meta.get("coefficient", 1.0) or 0.0)
                higher = bool(meta.get("higher_is_better", True))
                if not isinstance(agent_metrics, dict) or metric not in agent_metrics \
                        or not isinstance(agent_metrics.get(metric), (int, float)) \
                        or not math.isfinite(agent_metrics[metric]):
                    missing.append({
                        "experiment": exp_name, "method": method, "metric": metric,
                        "source": float(src_val),
                    })
                    continue
                agent_val = float(agent_metrics[metric])
                signed = (agent_val - src_val) if higher else (src_val - agent_val)
                # Relative improvement. Clamp magnitude to ±100% (=1.0) per
                # cell so a near-zero source value can't single-handedly
                # dominate the aggregate. Without the clamp,
                # (agent - 0)/eps blows up to ±10^9% which is just math
                # noise, not a real signal.
                scale = max(abs(float(src_val)), eps)
                rel_raw = signed / scale
                rel = max(-1.0, min(1.0, rel_raw))
                w = abs(coeff) * exp_weight
                cells.append({
                    "experiment": exp_name, "method": method, "metric": metric,
                    "source": float(src_val), "agent": agent_val,
                    "signed": signed, "relative": rel, "weight": w,
                    "higher_is_better": higher,
                })
    n_better = sum(1 for c in cells if c["signed"] > eps)
    n_worse = sum(1 for c in cells if c["signed"] < -eps)
    n_same = sum(1 for c in cells if abs(c["signed"]) <= eps)
    total_w = sum(c["weight"] for c in cells) or 0.0
    delta = (sum(c["relative"] * c["weight"] for c in cells) / total_w) if total_w > 0 else None
    return {
        "cells": cells,
        "missing": missing,
        "n_better": n_better,
        "n_worse": n_worse,
        "n_same": n_same,
        "n_missing": len(missing),
        "n_total": len(cells) + len(missing),
        "delta": delta,
    }


# Stale-while-revalidate cache for _scan_runs(). A scan reads several
# files per run workspace, which takes many seconds for hundreds of
# runs on a network filesystem; blocking every poll on it would freeze
# the page.
#
# Strategy: always return whatever cache we have (even if stale),
# instantly. If the cache is older than ``_SCAN_FRESH_FOR_SEC``, kick
# off a background refresh; until that finishes, callers keep getting
# the previous snapshot. The first call (no cache) does block on the
# walk, once per process.
_SCAN_FRESH_FOR_SEC = 30.0
_scan_cache_lock = threading.Lock()
_scan_cache: tuple[float, list[dict]] | None = None
_scan_refresh_in_flight = False


def _scan_refresh() -> None:
    """Run an uncached scan and atomically update the cache."""
    global _scan_cache, _scan_refresh_in_flight
    try:
        runs = _scan_runs_uncached()
        with _scan_cache_lock:
            _scan_cache = (time.monotonic(), runs)
    except Exception:
        log.exception("scan_runs background refresh failed")
    finally:
        with _scan_cache_lock:
            _scan_refresh_in_flight = False


_scan_first_done = threading.Event()


def _scan_runs() -> list[dict]:
    """Derive run list from ``<data_dir>/envs/*/*/.evaluation.json``.

    Stale-while-revalidate: returns whatever the cache has (instantly),
    then triggers a background refresh if the snapshot is older than
    ``_SCAN_FRESH_FOR_SEC``. The first call ever blocks on the walk;
    concurrent callers wait on the same scan via ``_scan_first_done``
    instead of each running their own.
    """
    global _scan_cache, _scan_refresh_in_flight
    with _scan_cache_lock:
        now = time.monotonic()
        cache = _scan_cache
        if cache is not None:
            ts, runs = cache
            stale = (now - ts) >= _SCAN_FRESH_FOR_SEC
            if stale and not _scan_refresh_in_flight:
                _scan_refresh_in_flight = True
                threading.Thread(
                    target=_scan_refresh, name="scan-runs-refresh",
                    daemon=True,
                ).start()
            return runs
        # No cache yet — the first caller computes; everyone else waits.
        if _scan_refresh_in_flight:
            should_compute = False
        else:
            _scan_refresh_in_flight = True
            should_compute = True

    if not should_compute:
        # Wait up to 60 s for the elected caller to finish, then read
        # the cache. If still empty, return [] rather than retrying —
        # the next tick will retry naturally.
        _scan_first_done.wait(timeout=60.0)
        with _scan_cache_lock:
            return _scan_cache[1] if _scan_cache else []

    try:
        runs = _scan_runs_uncached()
    except Exception:
        with _scan_cache_lock:
            _scan_refresh_in_flight = False
        _scan_first_done.set()
        raise
    with _scan_cache_lock:
        _scan_cache = (time.monotonic(), runs)
        _scan_refresh_in_flight = False
    _scan_first_done.set()
    return runs


def _scan_runs_uncached() -> list[dict]:
    envs = _config.envs_dir()
    if not envs.exists():
        return []
    runs = []
    # Only live env dirs (`<profile>-<mode>-eval`) — archived copies
    # (renamed aside as *.failed-* / *.killed-* before a requeue) must
    # not surface as phantom runs.
    for meta_path in envs.glob("*/*-eval/.evaluation.json"):
        meta = _read_json(meta_path) or {}
        env_dir = meta_path.parent
        # The agent writes scores_train.json; the grader writes
        # scores_test.json. scores.json is not part of the protocol and
        # is never read.
        train = _read_json(env_dir / "scoring" / "scores_train.json")
        test = _read_json(env_dir / "scoring" / "scores_test.json")
        meta.setdefault("env_dir", str(env_dir))
        if train is not None and "train_score" not in meta:
            meta["train_score"] = train
        if test is not None and "test_score" not in meta:
            meta["test_score"] = test
        # Compare against the pipeline's own paper-reproduction scores.
        # reference.json supplies only weight + coefficient — never a
        # compared value. The anchor is always source's scores_*.json.
        source_scoring = env_dir.parent / "source" / "scoring"
        reference = _read_json(source_scoring / "reference.json")
        source_train = _read_json(source_scoring / "scores_train.json")
        source_test = _read_json(source_scoring / "scores_test.json")
        meta["train_cmp"] = _compute_comparison(
            meta.get("train_score"), source_train, reference,
        )
        meta["test_cmp"] = _compute_comparison(
            meta.get("test_score"), source_test, reference,
        )
        # Add the normalized paper_score next to the raw delta: read the
        # paper's coefficients (one number per cell) and compute
        # ``Σ coefficient × raw_improvement`` against this run's scoring
        # file (see docs/scoring.md). The cohort's mean absolute score on
        # a paper is 1.0 by construction; other runs are relative to it.
        try:
            from evaluation import normalization as _norm
            paper_id = env_dir.parent.name
            # Per-mode coeffs: a run's normalized score has to be read
            # from the same mode's coefficients (innovate vs improve
            # are not comparable cohorts).
            run_mode = meta.get("mode") or "improve"
            train_coeffs = _norm.load_coeffs(paper_id, "train", run_mode)
            test_coeffs = _norm.load_coeffs(paper_id, "test", run_mode)
            meta["train_norm"] = _norm.normalize_delta(
                train_coeffs, meta.get("train_score"))
            meta["test_norm"] = _norm.normalize_delta(
                test_coeffs, meta.get("test_score"))
        except Exception:
            log.exception("Failed to layer delta_norm for %s", env_dir)
            meta["train_norm"] = {"delta_norm": None, "reason": "exception"}
            meta["test_norm"] = {"delta_norm": None, "reason": "exception"}
        # Completion gate: a run only counts as completed here once it has
        # both score files. A run marked ``completed`` without a held-out
        # score (not graded yet, or grading did not reach a verdict) or
        # without train scores (killed before finalization) is shown as
        # ``aborted`` and listed with the failures, so it does not enter
        # the leaderboard with half a score.
        if meta.get("status") == "completed":
            if meta.get("train_score") is None or meta.get("test_score") is None:
                meta["status"] = "aborted"
                meta["state"] = "failed"
                if not meta.get("error"):
                    missing = []
                    if meta.get("train_score") is None:
                        missing.append("scores_train.json missing")
                    if meta.get("test_score") is None:
                        missing.append("scores_test.json missing")
                    meta["error"] = (
                        "completed without both scores (not graded yet, or "
                        "killed before finalization): "
                        + "; ".join(missing)
                    )
        # For failures, the meta's ``error`` field is a generic one-liner
        # like "transient: agent exited rc=1"; attach the cause read
        # from the last trajectory file so the page can show it.
        if meta.get("state") == "failed" or meta.get("status") in ("failed", "aborted"):
            detail = _extract_failure_detail(env_dir, meta.get("error") or "")
            if detail:
                meta["error_detail"] = detail
        runs.append(meta)
    runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)
    return runs


def _norm_delta(norm: dict | None) -> float | None:
    """Extract delta_norm from a normalize_delta() result dict."""
    if not isinstance(norm, dict):
        return None
    v = norm.get("delta_norm")
    return float(v) if isinstance(v, (int, float)) else None


def _aggregate_leaderboard(runs: list[dict], mode: str | None = None) -> dict:
    """Per-model stats: mean, stddev, min, max, count.

    Only the normalized score (``delta_norm``) contributes. Runs whose
    paper has no coefficients yet are left out rather than mixed in
    with a raw value. Only ``completed`` runs contribute (as decided by
    the completion gate in ``_scan_runs_uncached``): ``incomplete`` and
    ``failed`` runs are excluded since their cell coverage is not
    comparable. A run's ``final`` is the mean of its train and test
    scores, or whichever of the two exists.

    If ``mode`` is given, runs whose ``mode`` field doesn't match are
    skipped (innovate and improve aggregate separately because the
    per-paper coefficients are mode-specific).
    """
    by_model: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        if r.get("status") != "completed":
            continue
        if mode is not None and (r.get("mode") or "improve") != mode:
            continue
        model = r.get("model") or "unknown"
        bucket = by_model.setdefault(model, {"train": [], "test": [], "final": []})
        t = _norm_delta(r.get("train_norm"))
        if t is not None:
            bucket["train"].append(t)
        e = _norm_delta(r.get("test_norm"))
        if e is not None:
            bucket["test"].append(e)
        if t is not None and e is not None:
            bucket["final"].append((t + e) / 2)
        elif t is not None:
            bucket["final"].append(t)
        elif e is not None:
            bucket["final"].append(e)

    def _summary(vs: list[float]) -> dict:
        # statistics.pstdev raises on NaN or inf, so a single non-finite
        # score would fail the whole /api/leaderboard request. Keep
        # finite values only; non-finite ones are not comparable to
        # real scores anyway.
        import math
        finite = [v for v in vs if isinstance(v, (int, float)) and math.isfinite(v)]
        if not finite:
            return {"count": 0}
        return {
            "count": len(finite),
            "mean": statistics.fmean(finite),
            "stddev": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
            "min": min(finite),
            "max": max(finite),
        }

    return {
        model: {phase: _summary(vs) for phase, vs in buckets.items()}
        for model, buckets in by_model.items()
    }


# ── HTML ──────────────────────────────────────────────────────────────────


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Dashboard</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg: #f0fdf4;
    --card: #ffffff;
    --border: #bbf7d0;
    --accent: #047857;
    --accent-light: #059669;
    --success: #16a34a;
    --warning: #d97706;
    --error: #dc2626;
    --blue: #2563eb;
    --amber: #d97706;
    --gray: #64748b;
    --text: #1c1917;
    --text2: #57534e;
    --text3: #9ca3af;
    --radius: 10px;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
}

body {
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.5;
    min-height: 100vh;
}

.header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    background: var(--card);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
}
.header-left { display: flex; align-items: center; gap: 16px; }
.logo {
    font-size: 22px; font-weight: 700; letter-spacing: 3px;
    color: var(--accent); font-family: 'Courier New', monospace;
}
.logo-sub { font-size: 12px; color: var(--text2); font-weight: 400; }
.header-right {
    display: flex; align-items: center; gap: 16px;
    font-size: 13px; color: var(--text2);
}
.refresh-indicator { display: flex; align-items: center; gap: 6px; }
.pulse-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--success); animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}
.pulse-dot.error { background: var(--error); animation: none; }

.error-banner {
    display: none; padding: 10px 24px;
    background: rgba(239, 68, 68, 0.15);
    border-bottom: 1px solid rgba(239, 68, 68, 0.3);
    color: var(--error); font-size: 13px;
}
.error-banner.visible { display: block; }

.main { max-width: 1400px; margin: 0 auto; padding: 24px; }

.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px; margin-bottom: 28px;
}
.card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 18px; box-shadow: var(--shadow);
}
.card-label {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    color: var(--text2); margin-bottom: 6px;
}
.card-value { font-size: 28px; font-weight: 700; font-family: 'Courier New', monospace; }
.card-sub { font-size: 12px; color: var(--text3); margin-top: 4px; }

.section { margin-bottom: 28px; }
.section-header {
    display: flex; align-items: baseline; gap: 8px;
    margin-bottom: 12px;
}
.section-title {
    font-size: 14px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; color: var(--text2);
}
.section-count { font-size: 13px; color: var(--text3); font-family: 'Courier New', monospace; }

.badge {
    display: inline-block; padding: 2px 8px;
    border-radius: 4px; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.badge-running    { background: rgba(37,99,235,0.15); color: var(--blue); }
.badge-completed  { background: rgba(22,163,74,0.15); color: var(--success); }
.badge-failed     { background: rgba(220,38,38,0.15); color: var(--error); }
.badge-incomplete { background: rgba(217,119,6,0.18); color: #b45309; }
.badge-pending   { background: rgba(100,116,139,0.15); color: var(--gray); }
.badge-mode-innovate { background: rgba(4,120,87,0.15); color: var(--accent); }
.badge-mode-improve { background: rgba(79,70,229,0.15); color: #4338ca; }
.badge-mode         { background: rgba(100,116,139,0.15); color: var(--gray); }  /* fallback for unknown modes */
.pub-pushed         { color: var(--success); margin-left: 4px; cursor: help; font-weight: 600; }
.pub-failed         { color: var(--error); margin-left: 4px; cursor: help; font-weight: 700; }
.pub-skipped        { color: var(--gray); margin-left: 4px; cursor: help; }
/* fine-grained lifecycle */
.badge-working     { background: rgba(37,99,235,0.15); color: var(--blue); }
.badge-waiting     { background: rgba(217,119,6,0.15); color: var(--amber); }
.badge-testing     { background: rgba(124,58,237,0.15); color: #7c3aed; }
.badge-scaffolding { background: rgba(100,116,139,0.15); color: var(--gray); }
.badge-done      { background: rgba(22,163,74,0.15); color: var(--success); }

.table-wrap {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow);
}
table { width: 100%; border-collapse: collapse; }
thead th {
    text-align: left; padding: 10px 14px;
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    color: var(--text3); border-bottom: 1px solid var(--border);
    background: rgba(0,0,0,0.02);
}
tbody td {
    padding: 8px 14px; font-size: 12px;
    border-bottom: 1px solid var(--border);
    font-family: 'Courier New', monospace;
}
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: rgba(4,120,87,0.06); }
.done-row { cursor: pointer; }
.done-detail { display: none; }
.done-detail.open { display: table-row; }
.done-detail td { padding: 0; background: var(--bg); }
.done-detail-content {
    padding: 12px 16px; font-size: 12px; color: var(--text2);
    white-space: pre-wrap; word-break: break-word;
    max-height: 200px; overflow-y: auto;
}

.empty-state {
    padding: 24px; text-align: center;
    color: var(--text3); font-size: 13px;
}

.dim { color: var(--text3); }
.mono { font-family: 'Courier New', monospace; }
.score-cell { font-family: 'Courier New', monospace; }

/* Coverage strip — completed/cohort per (model, mode) at a glance. */
.cov-strip {
    display: flex; flex-wrap: wrap; gap: 6px;
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 8px 12px;
    box-shadow: var(--shadow);
}
.cov-chip {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 8px; border-radius: 4px;
    font-size: 11px; font-family: 'Courier New', monospace;
    background: rgba(100,116,139,0.07);
    border: 1px solid var(--border);
}
.cov-chip-mode {
    font-size: 9px; font-weight: 700; opacity: 0.6;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.cov-chip-frac { font-weight: 700; }
.cov-chip-good { color: var(--success); }
.cov-chip-mid  { color: var(--warning); }
.cov-chip-low  { color: var(--error); }
/* Model-grouped layout: keep each model's (imp, inn) chips adjacent
   with tiny gap, and a larger gap between models via the parent flex. */
.cov-group { display: inline-flex; gap: 2px; margin-right: 14px; }

.queue-form {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 16px 18px;
    box-shadow: var(--shadow); margin-bottom: 28px;
    display: grid; grid-template-columns: 1fr 1fr 1fr auto; gap: 12px;
    align-items: end;
}
.queue-form label {
    display: flex; flex-direction: column; gap: 4px;
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; color: var(--text2);
}
.queue-form select {
    padding: 6px 8px; border: 1px solid var(--border);
    border-radius: 6px; background: var(--bg);
    font-family: 'Courier New', monospace; font-size: 13px;
    color: var(--text);
    min-height: 110px;
}
.queue-form select:focus { outline: 2px solid var(--accent-light); outline-offset: -1px; }
.queue-form select option { padding: 4px 6px; background: var(--card); }
.queue-form select option:checked { background: var(--accent); color: white; }
.queue-form .hint { font-size: 10px; color: var(--text3); font-weight: 400; letter-spacing: 0; text-transform: none; }
.queue-form button {
    padding: 9px 18px; background: var(--accent); color: white;
    border: none; border-radius: 6px; font-weight: 600;
    font-size: 13px; cursor: pointer; letter-spacing: 0.5px;
}
.queue-form button:hover { background: var(--accent-light); }

.chart-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 16px;
    box-shadow: var(--shadow);
}
.chart-card img { max-width: 100%; display: block; }
.chart-row {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 12px; align-items: start;
}
.chart-label {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.05em; color: var(--text2); margin-bottom: 8px;
}
.mode-toggle {
    margin-left: auto; padding: 3px 8px; font-size: 12px;
    border: 1px solid var(--border); border-radius: 6px;
    background: var(--card); color: var(--text2); cursor: pointer;
}
.mode-toggle:hover { border-color: var(--accent-light); }

@media (max-width: 900px) {
    .queue-form { grid-template-columns: 1fr; }
}
@media (max-width: 768px) {
    .main { padding: 12px; }
    .cards { grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .header { padding: 12px 16px; flex-wrap: wrap; gap: 8px; }
}
</style>
</head>
<body>

<div class="header">
    <div class="header-left">
        <div class="logo">EVAL <span class="logo-sub">Scientist World evaluation dashboard</span></div>
    </div>
    <div class="header-right">
        <div class="refresh-indicator">
            <div class="pulse-dot" id="pulseDot"></div>
            <span>Auto-refresh (60s) · port {port}</span>
        </div>
        <span id="lastUpdated" class="mono">--</span>
    </div>
</div>

<div class="error-banner" id="errorBanner"></div>

<div class="main">

    <div class="cards">
        <div class="card">
            <div class="card-label">Running</div>
            <div class="card-value" id="cardRunning">--</div>
            <div class="card-sub">agent sessions live</div>
        </div>
        <div class="card">
            <div class="card-label">Completed</div>
            <div class="card-value" id="cardCompleted">--</div>
            <div class="card-sub" id="cardCompletedSub">--</div>
        </div>
        <div class="card">
            <div class="card-label">Failed</div>
            <div class="card-value" id="cardFailed">--</div>
            <div class="card-sub">needs human review</div>
        </div>
        <div class="card">
            <div class="card-label">Models Evaluated</div>
            <div class="card-value" id="cardModels">--</div>
            <div class="card-sub">distinct profiles</div>
        </div>
    </div>

    <form class="queue-form" id="f">
        <label>Papers <span class="hint">(⌘/ctrl-click for multi)</span>
            <select name="papers" id="selPapers" multiple></select>
        </label>
        <label>Models <span class="hint">(⌘/ctrl-click for multi)</span>
            <select name="models" id="selModels" multiple></select>
        </label>
        <label>Modes <span class="hint">(⌘/ctrl-click for multi)</span>
            <select name="modes" id="selModes" multiple></select>
        </label>
        <button type="submit">Enqueue</button>
    </form>

    <form class="queue-form" id="capsForm" style="margin-top:8px">
        <label>Global cap <span class="hint" id="capsActiveHint"></span>
            <input type="number" min="1" id="capsGlobal" name="max_concurrent" style="padding:8px 10px;font:inherit;border:1px solid var(--border);border-radius:4px;background:var(--card)">
        </label>
        <label>Per-profile caps <span class="hint">profile=N, comma-separated</span>
            <input type="text" id="capsPerProfile" placeholder="profile-a=10,profile-b=15" style="padding:8px 10px;font:inherit;border:1px solid var(--border);border-radius:4px;background:var(--card)">
        </label>
        <button type="submit">Apply caps live</button>
    </form>

    <div class="section">
        <div class="section-header">
            <span class="section-title">Running</span>
            <span class="section-count" id="runningCount">(0)</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead><tr>
                    <th>Paper</th><th>Model</th><th>Mode</th>
                    <th>State</th>
                    <th>State Time</th><th>GPU Budget</th><th>Elapsed</th>
                </tr></thead>
                <tbody id="runningBody">
                    <tr><td colspan="7" class="empty-state">No active runs</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <span class="section-title" style="cursor:pointer" onclick="toggleQueuedSection()">
                <span id="queuedToggle">▶</span> Queued
            </span>
            <span class="section-count" id="queuedCount">(0)</span>
            <span class="section-count" id="queuedCap"></span>
        </div>
        <div class="table-wrap" id="queuedWrap" style="display:none">
            <table>
                <thead><tr>
                    <th>Paper</th><th>Model</th><th>Mode</th>
                </tr></thead>
                <tbody id="queuedBody">
                    <tr><td colspan="3" class="empty-state">Queue empty</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <span class="section-title">Completed</span>
            <span class="section-count" id="completedSectionCount">(0)</span>
            <select id="modeCompleted" class="mode-toggle" onchange="onModeChange('Completed')">
                <option value="improve">improve</option>
                <option value="innovate">innovate</option>
            </select>
        </div>
        <div class="table-wrap">
            <table>
                <thead><tr>
                    <th>Paper</th><th>Model</th><th>Mode</th>
                    <th>Status</th><th>Train</th><th>Test</th><th>Final</th>
                    <th>Duration</th><th>Ended</th>
                </tr></thead>
                <tbody id="completedBody">
                    <tr><td colspan="9" class="empty-state">No completed runs yet</td></tr>
                </tbody>
            </table>
            <div id="completedExpandRow" style="display:none;text-align:center;padding:8px;">
                <button id="completedExpandBtn" onclick="toggleCompletedExpand()"
                    style="background:transparent;border:1px solid var(--border);color:var(--text);
                           padding:4px 12px;font-size:12px;cursor:pointer;border-radius:4px;"></button>
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <span class="section-title">Verification</span>
            <span class="section-count" id="verificationCount">(0)</span>
            <select id="verificationFilter" class="mode-toggle" onchange="renderVerification()">
                <option value="all">all</option>
                <option value="unfixable">unfixable</option>
                <option value="exhausted">exhausted</option>
                <option value="verified">verified</option>
                <option value="audit_clean">audit_clean</option>
                <option value="incomplete">incomplete</option>
                <option value="in_progress">in_progress</option>
            </select>
        </div>
        <div class="table-wrap">
            <table>
                <thead><tr>
                    <th>Paper</th><th>Model</th><th>Mode</th><th>Status</th>
                    <th>Source</th><th>Turns</th><th>Finished</th><th>Note</th>
                </tr></thead>
                <tbody id="verificationBody">
                    <tr><td colspan="8" class="empty-state">No verifications yet</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <span class="section-title">Per-Paper × Model</span>
            <span class="section-count" id="matrixCount">(0 papers × 0 models)</span>
            <select id="modeMatrix" class="mode-toggle" onchange="onModeChange('Matrix')">
                <option value="improve">improve</option>
                <option value="innovate">innovate</option>
            </select>
        </div>
        <div class="table-wrap">
            <table id="matrixTable">
                <thead id="matrixHead"></thead>
                <tbody id="matrixBody">
                    <tr><td class="empty-state">No data yet</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <div class="section" style="margin-bottom: 16px;">
        <div class="section-header">
            <span class="section-title">Coverage</span>
            <span class="section-count">completed / papers, per (model, mode)</span>
        </div>
        <div id="coverageStrip" class="cov-strip">
            <span class="dim" style="font-size: 12px;">No runs yet</span>
        </div>
    </div>

    <!-- One coverage strip per configured paper group (dashboard.paper_groups). -->
    <div id="groupCoverage"></div>

    <div class="section">
        <div class="section-header">
            <span class="section-title">Leaderboard</span>
            <span class="section-count" id="leaderboardCount">(0 models)</span>
            <select id="modeLeaderboard" class="mode-toggle" onchange="onModeChange('Leaderboard')">
                <option value="improve">improve</option>
                <option value="innovate">innovate</option>
            </select>
        </div>
        <div class="table-wrap">
            <table>
                <thead><tr>
                    <th>Model</th><th>Train</th><th>Test</th>
                    <th>Final</th><th>n</th>
                </tr></thead>
                <tbody id="leaderboardBody">
                    <tr><td colspan="5" class="empty-state">No completed runs yet</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <span class="section-title">Final Score Distribution</span>
        </div>
        <!-- Both modes shown side-by-side; descending order applied server-side. -->
        <div class="chart-row">
            <div class="chart-card">
                <div class="chart-label">improve</div>
                <img src="/api/leaderboard.png?mode=improve" id="chartImprove" alt="improve">
            </div>
            <div class="chart-card">
                <div class="chart-label">innovate</div>
                <img src="/api/leaderboard.png?mode=innovate" id="chartInnovate" alt="innovate">
            </div>
        </div>
    </div>

    <!-- One pair of charts per configured paper group (dashboard.paper_groups). -->
    <div id="groupCharts"></div>

</div>

<script>
// Display options from the evaluation config (dashboard.paper_groups,
// dashboard.model_order), injected by the server.
const DASHBOARD_CONFIG = __DASHBOARD_CONFIG__;
// [{label, set}] for each named subset of papers; empty when none are configured.
const PAPER_GROUPS = Object.entries(DASHBOARD_CONFIG.paper_groups || {})
    .map(([label, ids]) => ({label, set: new Set(ids || [])}));
const MODEL_COLUMN_ORDER = DASHBOARD_CONFIG.model_order || [];

function escapeHtml(s) {
    if (s == null) return '';
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

function groupChartUrl(mode, label) {
    return '/api/leaderboard.png?mode=' + mode + '&group=' + encodeURIComponent(label);
}

// Build the per-group sections once: a coverage strip and a pair of
// score charts for each configured paper group. Nothing is added when
// no groups are configured.
function buildGroupSections() {
    const cov = document.getElementById('groupCoverage');
    const charts = document.getElementById('groupCharts');
    PAPER_GROUPS.forEach((g, i) => {
        const label = escapeHtml(g.label);
        cov.insertAdjacentHTML('beforeend', `
    <div class="section" style="margin-bottom: 16px;">
        <div class="section-header">
            <span class="section-title">Coverage (${label})</span>
            <span class="section-count">completed / ${g.set.size} papers, per (model, mode)</span>
        </div>
        <div id="coverageStripGroup${i}" class="cov-strip">
            <span class="dim" style="font-size: 12px;">No runs yet</span>
        </div>
    </div>`);
        charts.insertAdjacentHTML('beforeend', `
    <div class="section">
        <div class="section-header">
            <span class="section-title">Final Score Distribution (${label})</span>
        </div>
        <div class="chart-row">
            <div class="chart-card">
                <div class="chart-label">improve</div>
                <img src="${escapeHtml(groupChartUrl('improve', g.label))}" id="chartImproveGroup${i}" alt="improve ${label}">
            </div>
            <div class="chart-card">
                <div class="chart-label">innovate</div>
                <img src="${escapeHtml(groupChartUrl('innovate', g.label))}" id="chartInnovateGroup${i}" alt="innovate ${label}">
            </div>
        </div>
    </div>`);
    });
}

function fmtScore(s) {
    if (s == null) return '<span class="dim">—</span>';
    if (s.mean == null) return '<span class="dim">—</span>';
    const stddev = (s.stddev || 0).toFixed(3);
    return s.mean.toFixed(3) + ' <span class="dim">± ' + stddev + '</span>';
}

// Coverage chips: completed / papers per (model, mode).
// The denominator per mode is the union of paper_ids touched by ANY
// run in that mode (what was actually queued).
// If ``cohortSet`` is provided, restrict to runs whose paper_id is in
// that set AND use the set size (not the run union) as the denominator.
// ``groupByModel`` switches to model-grouped layout (imp+inn for each
// model adjacent, models separated by a small gap), which is easier to
// read per model.
function renderCoverage(runs, cohortSet, groupByModel) {
    const universe = {};        // mode -> Set(paper_id)
    const cov = {};             // "model|mode" -> {model, mode, completed:Set}
    runs.forEach(r => {
        if (!r.paper_id) return;
        if (cohortSet && !cohortSet.has(r.paper_id)) return;
        const m = r.mode || 'improve';
        (universe[m] = universe[m] || new Set()).add(r.paper_id);
        if (!r.model) return;
        const key = r.model + '|' + m;
        const e = cov[key] = (cov[key] || {model: r.model, mode: m, completed: new Set()});
        if (r.status === 'completed') e.completed.add(r.paper_id);
    });
    const entries = Object.values(cov).sort((a, b) => {
        if (groupByModel) {
            if (a.model !== b.model) return a.model < b.model ? -1 : 1;
            if (a.mode !== b.mode) return a.mode === 'improve' ? -1 : 1;
            return 0;
        }
        if (a.mode !== b.mode) return a.mode === 'improve' ? -1 : 1;
        if (a.model !== b.model) return a.model < b.model ? -1 : 1;
        return 0;
    });
    if (entries.length === 0) return '<span class="dim" style="font-size: 12px;">No runs yet</span>';
    const chip = (e) => {
        const total = cohortSet ? cohortSet.size : ((universe[e.mode] || new Set()).size || 0);
        const done = e.completed.size;
        const pct = total > 0 ? done / total : 0;
        const cls = pct >= 0.8 ? 'cov-chip-good' : pct >= 0.4 ? 'cov-chip-mid' : 'cov-chip-low';
        const tag = e.mode === 'improve' ? 'imp' : 'inn';
        return `<span class="cov-chip"><span class="cov-chip-mode">${tag}</span>${escapeHtml(e.model)}<span class="cov-chip-frac ${cls}">${done}/${total}</span></span>`;
    };
    if (groupByModel) {
        // Wrap each model's (imp, inn) pair in a flex group so they stick
        // together and visual gap separates model groups.
        const groups = {};
        entries.forEach(e => { (groups[e.model] = groups[e.model] || []).push(e); });
        return Object.keys(groups).sort().map(model =>
            `<span class="cov-group">${groups[model].map(chip).join('')}</span>`
        ).join('');
    }
    return entries.map(chip).join('');
}

// Render a comparison cell: agent vs source paper reproduction.
// Shows win/loss/tie counts, missing count, and a signed percentage delta.
// Positive delta = agent improved over source; negative = worse.
function fmtCmp(cmp) {
    if (!cmp || typeof cmp !== 'object') return '<span class="dim">—</span>';
    const total = cmp.n_total || 0;
    if (total === 0) return '<span class="dim">—</span>';
    const w = cmp.n_better || 0, l = cmp.n_worse || 0, t = cmp.n_same || 0, m = cmp.n_missing || 0;
    let deltaHtml = '<span class="dim">—</span>';
    if (typeof cmp.delta === 'number') {
        const pct = (cmp.delta * 100);
        const sign = pct >= 0 ? '+' : '';
        const color = pct > 0.5 ? '#2a8f3c' : (pct < -0.5 ? '#c0392b' : 'var(--text3)');
        deltaHtml = `<span style="color:${color}">${sign}${pct.toFixed(1)}%</span>`;
    }
    const missBadge = m > 0
        ? ` <span style="background:#ffd86b;color:#5a4200;padding:1px 5px;border-radius:3px;font-size:10px" title="${m} metric cells in source had no agent value">⚠ ${m} missing</span>`
        : '';
    return `${deltaHtml} <span class="dim">(${w}✓/${l}✗/${t}=  of ${total - m})</span>${missBadge}`;
}

function fmtCmpCompact(cmp) {
    // Compact form for matrix cells: just the colored % delta.
    if (!cmp || typeof cmp.delta !== 'number') return '';
    const pct = cmp.delta * 100;
    const sign = pct >= 0 ? '+' : '';
    const color = pct > 0.5 ? '#2a8f3c' : (pct < -0.5 ? '#c0392b' : 'var(--text3)');
    return `<span style="color:${color}">${sign}${pct.toFixed(0)}%</span>`;
}

function fmtNormCompact(norm) {
    // Normalized paper_score for matrix cells, shown × 100 (the FNI
    // scale). The cohort's mean |paper_score| is 1.0 on every paper by
    // construction, so 100 reads as "typical cohort magnitude", 150 as
    // 50% more, and a negative value as a net regression.
    if (!norm || typeof norm.delta_norm !== 'number') return '';
    const v = norm.delta_norm * 100;
    const sign = v >= 0 ? '+' : '';
    const color = v > 0.5 ? '#2a8f3c' : (v < -0.5 ? '#c0392b' : 'var(--text3)');
    return `<span style="color:${color}">${sign}${v.toFixed(0)}%</span>`;
}

function renderMatrix(runs) {
    // Build {paper_id: {model: run}} map. Pick the freshest run per (paper,model).
    const byKey = {};
    for (const r of runs) {
        const k = `${r.paper_id}|${r.model}`;
        const prev = byKey[k];
        if (!prev || (r.state_updated_at || '') > (prev.state_updated_at || '')) {
            byKey[k] = r;
        }
    }
    // Sort papers by most-recent activity (any model). Use the max
    // ended_at | state_updated_at | started_at across the paper's runs
    // as the recency key. Most recently active paper appears at the top.
    const paperRecency = {};
    for (const r of runs) {
        const ts = r.ended_at || r.state_updated_at || r.started_at || '';
        if (!paperRecency[r.paper_id] || ts > paperRecency[r.paper_id]) {
            paperRecency[r.paper_id] = ts;
        }
    }
    const papers = [...new Set(runs.map(r => r.paper_id))].sort((a, b) => {
        const ta = paperRecency[a] || '';
        const tb = paperRecency[b] || '';
        if (ta === tb) return a < b ? -1 : 1;  // tiebreaker: paper_id asc
        if (!ta) return 1;
        if (!tb) return -1;
        return ta > tb ? -1 : 1;
    });
    // Column order. Profiles listed in dashboard.model_order come first,
    // in that order, and are always shown even before they have runs;
    // every other profile with runs follows alphabetically.
    const present = new Set(runs.map(r => r.model));
    const models = [
        ...MODEL_COLUMN_ORDER,
        ...[...present].filter(m => !MODEL_COLUMN_ORDER.includes(m)).sort(),
    ];
    if (!papers.length) {
        return {head: '', body: '<tr><td class="empty-state">No runs yet</td></tr>', count: `(0 papers × ${models.length} models)`};
    }
    const headCells = ['<th style="position:sticky;left:0;background:var(--bg-card);z-index:2">Paper</th>']
        .concat(models.map(m => `<th>${escapeHtml(m)}</th>`)).join('');
    const head = `<tr>${headCells}</tr>`;
    const rows = papers.map(p => {
        const cells = [`<td style="position:sticky;left:0;background:var(--bg-card);font-weight:500"><a href="#" data-paper="${escapeHtml(p)}">${escapeHtml(p)}</a></td>`];
        for (const m of models) {
            const r = byKey[`${p}|${m}`];
            if (!r) { cells.push('<td class="dim">—</td>'); continue; }
            const status = r.status || 'running';
            const statusIcon = ({
                completed: '✓', failed: '✗', incomplete: '⚠', running: '⋯',
            })[status] || '?';
            const statusColor = ({
                completed: '#2a8f3c', failed: '#c0392b',
                incomplete: '#b45309', running: 'var(--blue)',
            })[status] || 'var(--text3)';
            const trainPct = fmtCmpCompact(r.train_cmp);
            const testPct = fmtCmpCompact(r.test_cmp);
            // The normalized score is the primary display when
            // available; the raw % is shown instead while the paper has
            // no coefficients (cohort too small, excluded, ...).
            const trainNorm = fmtNormCompact(r.train_norm);
            const testNorm = fmtNormCompact(r.test_norm);
            let body = '';
            if (status === 'running') {
                body = `<span style="color:${statusColor}">${statusIcon}</span>`;
            } else if (status === 'failed' || (!trainPct && !testPct)) {
                body = `<span style="color:${statusColor}">${statusIcon}</span>`;
            } else if (trainNorm || testNorm) {
                const primarySep = (trainNorm && testNorm) ? '<span class="dim"> / </span>' : '';
                const secondarySep = (trainPct && testPct) ? '<span class="dim"> / </span>' : '';
                const secondary = (trainPct || testPct)
                    ? `<div style="font-size:10px;opacity:0.6">${trainPct}${secondarySep}${testPct}</div>`
                    : '';
                body = `<span style="color:${statusColor};margin-right:4px">${statusIcon}</span>${trainNorm}${primarySep}${testNorm}${secondary}`;
            } else {
                const sep = (trainPct && testPct) ? '<span class="dim"> / </span>' : '';
                body = `<span style="color:${statusColor};margin-right:4px">${statusIcon}</span>${trainPct}${sep}${testPct}`;
            }
            // Tooltip: normalized score (primary), raw % (secondary), cell counts.
            const tnPct = r.train_cmp ? (100*r.train_cmp.delta).toFixed(1)+'%' : '—';
            const teRaw = r.test_cmp ? (100*r.test_cmp.delta).toFixed(1)+'%' : '—';
            const tnN = (r.train_norm && typeof r.train_norm.delta_norm === 'number')
                ? (100*r.train_norm.delta_norm).toFixed(1)+'% (n='+(r.train_norm.n_models||0)+')'
                : '—';
            const teN = (r.test_norm && typeof r.test_norm.delta_norm === 'number')
                ? (100*r.test_norm.delta_norm).toFixed(1)+'% (n='+(r.test_norm.n_models||0)+')'
                : '—';
            const tooltip = `${status} | norm train: ${tnN} | norm test: ${teN} | raw train: ${tnPct} | raw test: ${teRaw}`;
            cells.push(`<td title="${escapeHtml(tooltip)}">${body}</td>`);
        }
        return `<tr>${cells.join('')}</tr>`;
    }).join('');
    return {
        head,
        body: rows,
        count: `(${papers.length} papers × ${models.length} models)`,
    };
}

function fmtTs(ts) {
    if (!ts) return '<span class="dim">—</span>';
    try { return new Date(ts).toLocaleString([], {month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit'}); }
    catch { return escapeHtml(ts); }
}

function fmtElapsed(ts) {
    if (!ts) return '<span class="dim">—</span>';
    try {
        const diffMs = Date.now() - new Date(ts).getTime();
        if (diffMs < 0) return '<span class="dim">—</span>';
        const s = Math.floor(diffMs / 1000);
        if (s < 60) return s + 's';
        const m = Math.floor(s / 60);
        if (m < 60) return m + 'm ' + (s % 60) + 's';
        const h = Math.floor(m / 60);
        return h + 'h ' + (m % 60) + 'm';
    } catch { return '<span class="dim">—</span>'; }
}

function fmtBudget(used, total) {
    if (used == null && total == null) return '<span class="dim">—</span>';
    const u = used != null ? used.toFixed(1) : '?';
    const t = total != null ? total.toFixed(1) : '?';
    return u + ' / ' + t + ' h';
}

function fmtDuration(start, end) {
    if (!start || !end) return '<span class="dim">—</span>';
    try {
        const ms = new Date(end).getTime() - new Date(start).getTime();
        if (ms < 0) return '<span class="dim">—</span>';
        const s = Math.floor(ms / 1000);
        if (s < 60) return s + 's';
        const m = Math.floor(s / 60);
        if (m < 60) return m + 'm ' + (s % 60) + 's';
        const h = Math.floor(m / 60);
        return h + 'h ' + (m % 60) + 'm';
    } catch { return '<span class="dim">—</span>'; }
}

function renderQueued(qs) {
    if (!qs.length) return '<tr><td colspan="3" class="empty-state">Queue empty</td></tr>';
    return qs.map(q => `<tr>
        <td>${escapeHtml(q.paper_id)}</td>
        <td>${escapeHtml(q.profile_name)}</td>
        <td><span class="badge badge-mode-${escapeHtml(q.mode || 'innovate')}">${escapeHtml(q.mode || 'innovate')}</span></td>
    </tr>`).join('');
}

function renderRunning(rs) {
    if (!rs.length) return '<tr><td colspan="7" class="empty-state">No active runs</td></tr>';
    return rs.map(r => {
        const state = r.state || 'working';
        // The Hive job id is shown inline with the state badge:
        // [queuing] while the submit is in flight, [abc12345] once the
        // id is known.
        let hiveTail = '';
        if (state === 'waiting') {
            if (r.current_hive_id) {
                const full = String(r.current_hive_id);
                const short = full.length > 8 ? full.substring(0, 8) : full;
                hiveTail = ` <span style="font-size:10px;color:var(--text3)" title="${escapeHtml(full)}">[${escapeHtml(short)}]</span>`;
            } else {
                hiveTail = ' <span style="font-size:10px;color:var(--text3)">[queuing]</span>';
            }
        }
        return `<tr>
            <td>${escapeHtml(r.paper_id)}</td>
            <td>${escapeHtml(r.model)}</td>
            <td><span class="badge badge-mode-${escapeHtml(r.mode || 'innovate')}">${escapeHtml(r.mode || 'innovate')}</span></td>
            <td><span class="badge badge-${escapeHtml(state)}">${escapeHtml(state)}</span>${hiveTail}</td>
            <td class="dim">${fmtElapsed(r.state_updated_at)}</td>
            <td>${fmtBudget(r.gpu_hours_used, r.gpu_budget)}</td>
            <td class="dim">${fmtElapsed(r.started_at)}</td>
        </tr>`;
    }).join('');
}

// The rows last rendered in the Completed table. toggleDetail() looks a
// run up here on click, so details are not kept in hidden DOM nodes.
window._completedRuns = [];

function renderPubBadge(p) {
    // Compact publish-status indicator shown next to the run's
    // status badge. ``p`` is r.publish; absent means no publish was
    // attempted (run not completed). Nothing is shown when publishing
    // is disabled, which is the default.
    if (!p || p.status === 'disabled') return '';
    const st = p.status || 'unknown';
    const tip = p.status === 'failed'
        ? 'publish FAILED — ' + (p.error || 'unknown')
        : (p.status === 'pushed'
            ? 'pushed to ' + (p.repo || '') + '@' + (p.branch || '')
            : 'publish skipped: ' + (p.reason || ''));
    const symbol = {pushed: '⬆', failed: '⚠', skipped: '–'}[st] || '?';
    return ` <span class="pub-${escapeHtml(st)}" title="${escapeHtml(tip)}">${symbol}</span>`;
}

// Default cap for the completed section: rendering hundreds of rows on
// every poll makes the page unresponsive, so only the most recent rows
// are shown until the expand button is used.
const _COMPLETED_DEFAULT_LIMIT = 30;

function renderCompleted(rs) {
    if (!rs.length) {
        // Keep the footer hidden when there's nothing to expand.
        const footer = document.getElementById('completedExpandRow');
        if (footer) footer.style.display = 'none';
        return '<tr><td colspan="9" class="empty-state">No completed runs yet</td></tr>';
    }
    // Cap the visible rows. window._completedExpanded is toggled by
    // toggleCompletedExpand() — undefined = collapsed (default).
    const total = rs.length;
    const expanded = !!window._completedExpanded;
    const visible = expanded ? rs : rs.slice(0, _COMPLETED_DEFAULT_LIMIT);
    // Update the footer button + show/hide based on whether there's
    // anything beyond the cap.
    const footer = document.getElementById('completedExpandRow');
    const btn = document.getElementById('completedExpandBtn');
    if (footer && btn) {
        if (total > _COMPLETED_DEFAULT_LIMIT) {
            footer.style.display = '';
            btn.textContent = expanded
                ? '▲ Show fewer'
                : `▼ Show all ${total} (showing ${_COMPLETED_DEFAULT_LIMIT})`;
        } else {
            footer.style.display = 'none';
        }
    }
    // toggleDetail() looks a run up by the index of its rendered row,
    // so _completedRuns must be exactly the rows rendered here.
    window._completedRuns = visible;
    return visible.map((r, i) => {
        const st = r.status || 'pending';
        return `<tr class="done-row" onclick="toggleDetail(${i})">
            <td>${escapeHtml(r.paper_id)}</td>
            <td>${escapeHtml(r.model)}</td>
            <td><span class="badge badge-mode-${escapeHtml(r.mode || 'innovate')}">${escapeHtml(r.mode || 'innovate')}</span></td>
            <td><span class="badge badge-${escapeHtml(st)}">${escapeHtml(st)}</span>${renderPubBadge(r.publish)}</td>
            <td class="score-cell">${fmtCmp(r.train_cmp)}</td>
            <td class="score-cell">${fmtCmp(r.test_cmp)}</td>
            <td class="score-cell">${(r.train_cmp && r.test_cmp && typeof r.train_cmp.delta === 'number' && typeof r.test_cmp.delta === 'number')
                ? fmtCmp({
                    n_total: (r.train_cmp.n_total||0) + (r.test_cmp.n_total||0),
                    n_better: (r.train_cmp.n_better||0) + (r.test_cmp.n_better||0),
                    n_worse: (r.train_cmp.n_worse||0) + (r.test_cmp.n_worse||0),
                    n_missing: (r.train_cmp.n_missing||0) + (r.test_cmp.n_missing||0),
                    delta: (r.train_cmp.delta + r.test_cmp.delta) / 2,
                })
                : '<span class="dim">—</span>'}</td>
            <td class="dim">${fmtDuration(r.started_at, r.ended_at)}</td>
            <td class="dim">${fmtTs(r.ended_at)}</td>
        </tr>`;
    }).join('');
}

// Detail rows are rendered lazily: one is built when its row is
// clicked, inserted directly after it, and removed on the second
// click, instead of keeping a hidden detail row per run in the table.

// Flip the completed-table expand state and force a re-render on
// the next poll by invalidating the html cache. Detail rows are
// closed first since indices into _completedRuns change with the
// expansion state.
function toggleCompletedExpand() {
    window._completedExpanded = !window._completedExpanded;
    document.querySelectorAll('tr.done-detail').forEach(n => n.remove());
    window._detailOpenedAt = 0;
    window._lastCompletedHtml = null;  // bust the dom-write cache
}

function toggleDetail(i) {
    const row = document.querySelector(`tr.done-row[onclick="toggleDetail(${i})"]`);
    if (!row) return;
    const next = row.nextElementSibling;
    if (next && next.classList && next.classList.contains('done-detail')) {
        next.remove();
        if (!document.querySelector('tr.done-detail')) {
            window._detailOpenedAt = 0;
        }
        return;
    }
    const r = (window._completedRuns || [])[i] || {};
    const detailHtml = r.error_detail
        ? '<div><strong>Cause:</strong> ' + escapeHtml(r.error_detail) + '</div>'
          + (r.error ? '<div class="dim" style="margin-top:4px">Runner: ' + escapeHtml(r.error) + '</div>' : '')
        : (escapeHtml(r.error || '') || '<span class="dim">(no error)</span>');
    const envHtml = '<div class="dim" style="margin-top:6px;font-size:11px">' + escapeHtml(r.env_dir || '') + '</div>';
    // Publish outcome — only shown when we actually tried (status=completed).
    let pubHtml = '';
    const p = r.publish;
    if (p) {
        if (p.status === 'pushed') {
            const url = p.repo ? `https://github.com/${p.repo}/tree/${p.branch}` : '';
            const link = url
                ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(p.repo)}@${escapeHtml(p.branch)}</a>`
                : escapeHtml((p.repo || '') + '@' + (p.branch || ''));
            const sz = p.bytes ? ` · ${(p.bytes/1024/1024).toFixed(1)} MB` : '';
            const fc = p.files ? ` · ${p.files} files` : '';
            pubHtml = `<div style="margin-top:8px"><strong>Publish:</strong> <span class="pub-pushed">⬆ pushed</span> → ${link}${fc}${sz}</div>`;
        } else if (p.status === 'failed') {
            pubHtml = `<div style="margin-top:8px"><strong>Publish:</strong> <span class="pub-failed">⚠ FAILED</span> for branch <code>${escapeHtml(p.branch || '')}</code><div class="dim" style="margin-top:4px">${escapeHtml(p.error || '')}</div></div>`;
        } else if (p.status === 'skipped') {
            pubHtml = `<div style="margin-top:8px"><strong>Publish:</strong> <span class="pub-skipped">– skipped</span> <span class="dim">(${escapeHtml(p.reason || '')})</span></div>`;
        }
    }
    const tr = document.createElement('tr');
    tr.className = 'done-detail open';
    tr.innerHTML = `<td colspan="9"><div class="done-detail-content">${detailHtml}${pubHtml}${envHtml}</div></td>`;
    row.parentNode.insertBefore(tr, row.nextSibling);
    window._detailOpenedAt = Date.now();
}

function renderLeaderboard(lb) {
    const models = Object.keys(lb).sort();
    if (!models.length) return '<tr><td colspan="5" class="empty-state">No completed runs yet</td></tr>';
    return models.map(m => {
        const f = lb[m].final || {};
        return `<tr>
            <td>${escapeHtml(m)}</td>
            <td class="score-cell">${fmtScore(lb[m].train)}</td>
            <td class="score-cell">${fmtScore(lb[m].test)}</td>
            <td class="score-cell">${fmtScore(f)}</td>
            <td class="dim">${f.count || 0}</td>
        </tr>`;
    }).join('');
}

async function refresh() {
    const pulse = document.getElementById('pulseDot');
    const banner = document.getElementById('errorBanner');
    try {
        // Leaderboard is fetched per selected leaderboard-mode below.
        const [runsR, queueR] = await Promise.all([
            fetch('/api/runs'),
            fetch('/api/queue'),
        ]);
        if (!runsR.ok || !queueR.ok) throw new Error('HTTP ' + (runsR.status || queueR.status));
        const runs = await runsR.json();
        const queue = await queueR.json();

        // Running = the orchestrator's running set (what the pool is
        // actually working on). For each of those, the run's
        // .evaluation.json (matched by paper, model, mode) supplies the
        // details; if there is none yet, the run is still being
        // scaffolded.
        const diskRuns  = runs.filter(r => r.status === 'running');
        const diskByKey = {};
        diskRuns.forEach(r => {
            diskByKey[`${r.paper_id}|${r.model}|${r.mode}`] = r;
        });
        const orchRunning = queue.running || [];
        const running = orchRunning.map(q => {
            const key = `${q.paper_id}|${q.profile_name}|${q.mode}`;
            const meta = diskByKey[key];
            if (meta) return meta;
            return {
                paper_id: q.paper_id,
                model: q.profile_name,
                mode: q.mode,
                status: 'running',
                state: 'scaffolding',
                started_at: null,
                state_updated_at: null,
            };
        });
        // A run that's currently running OR pending is about to
        // overwrite its own meta on disk — suppress its stale
        // completed/failed row so the Completed section only shows
        // genuinely-finished runs. Match on the full (paper, model,
        // mode) triple, not just paper_id, because different models
        // live in different env_dirs and don't overwrite each other.
        const inFlightKeys = new Set([
            ...(queue.running || []),
            ...(queue.pending || []),
        ].map(q => `${q.paper_id}|${q.profile_name}|${q.mode}`));
        const done      = runs.filter(r => r.status !== 'running'
                                         && !inFlightKeys.has(`${r.paper_id}|${r.model}|${r.mode}`));
        const completed = done.filter(r => r.status === 'completed');
        // ``aborted`` = marked completed but without both score files
        // (not graded yet, or killed before finalization); counted with
        // the failures, not with the completed runs.
        const failed    = done.filter(r => r.status === 'failed' || r.status === 'aborted');

        // cardModels = distinct models that have at least one run on
        // disk (any status, any mode). The mode-specific leaderboard
        // count gets set later when its fetch resolves.
        const distinctModels = new Set(runs.map(r => r.model || 'unknown'));
        document.getElementById('cardRunning').textContent = running.length;
        document.getElementById('cardCompleted').textContent = completed.length;
        document.getElementById('cardCompletedSub').textContent = `${done.length} total`;
        document.getElementById('cardFailed').textContent = failed.length;
        document.getElementById('cardModels').textContent = distinctModels.size;

        document.getElementById('runningCount').textContent = `(${running.length})`;
        document.getElementById('completedSectionCount').textContent = `(${done.length})`;

        // Sort by ended_at descending (most recent on top), falling back
        // to state_updated_at. Runs with neither end up at the bottom.
        done.sort((a, b) => {
            const ka = a.ended_at || a.state_updated_at || '';
            const kb = b.ended_at || b.state_updated_at || '';
            if (ka === kb) return 0;
            if (!ka) return 1;
            if (!kb) return -1;
            return ka > kb ? -1 : 1;
        });

        // Compute new HTML strings, then compare with the last paint:
        // assigning innerHTML triggers a full parse and reflow even
        // when nothing changed, so the assignment is skipped when a
        // section's content is the same.
        //
        // The Completed table is not re-rendered while a detail row is
        // open (replacing innerHTML would close it), unless it has been
        // open for more than 60 s.
        const runningHtml = renderRunning(running);
        if (window._lastRunningHtml !== runningHtml) {
            document.getElementById('runningBody').innerHTML = runningHtml;
            window._lastRunningHtml = runningHtml;
        }

        const openDetails = document.querySelectorAll('.done-detail.open').length;
        const detailOpenedAt = window._detailOpenedAt || 0;
        const detailOpenAge = openDetails ? (Date.now() - detailOpenedAt) : 0;
        const detailFresh = openDetails && detailOpenAge < 60000;
        if (!openDetails) window._detailOpenedAt = 0;

        // Per-section mode filters. Innovate and improve aggregate
        // separately because their per-paper coefficients are separate.
        const modeCompleted   = (document.getElementById('modeCompleted') || {}).value || 'improve';
        const modeMatrix      = (document.getElementById('modeMatrix') || {}).value || 'improve';
        const modeLeaderboard = (document.getElementById('modeLeaderboard') || {}).value || 'improve';
        const filterMode = (xs, m) => xs.filter(r => (r.mode || 'improve') === m);

        const completedHtml = renderCompleted(filterMode(done, modeCompleted));
        if (!detailFresh && window._lastCompletedHtml !== completedHtml) {
            document.getElementById('completedBody').innerHTML = completedHtml;
            window._lastCompletedHtml = completedHtml;
        }

        // Coverage strip: completed / papers per (model, mode).
        // Cheap to recompute every tick; cached by content to skip DOM
        // writes when nothing changed.
        const coverageHtml = renderCoverage(runs);
        if (window._lastCoverageHtml !== coverageHtml) {
            document.getElementById('coverageStrip').innerHTML = coverageHtml;
            window._lastCoverageHtml = coverageHtml;
        }

        // One coverage strip per configured paper group, model-grouped
        // so each model's imp/inn pair sits together.
        window._lastGroupCoverageHtml = window._lastGroupCoverageHtml || {};
        PAPER_GROUPS.forEach((g, i) => {
            const html = renderCoverage(runs, g.set, true);
            if (window._lastGroupCoverageHtml[i] !== html) {
                document.getElementById('coverageStripGroup' + i).innerHTML = html;
                window._lastGroupCoverageHtml[i] = html;
            }
        });

        // Leaderboard fetched per-mode (server-side aggregation); cache
        // by mode so a tick that doesn't change data doesn't re-render.
        const lbResp = await fetch('/api/leaderboard?mode=' + modeLeaderboard).then(r => r.json());
        const lbHtml = renderLeaderboard(lbResp);
        const lbCacheKey = modeLeaderboard + '|' + lbHtml;
        if (window._lastLbHtml !== lbCacheKey) {
            document.getElementById('leaderboardBody').innerHTML = lbHtml;
            document.getElementById('leaderboardCount').textContent = `(${Object.keys(lbResp).length} models)`;
            window._lastLbHtml = lbCacheKey;
            // Reload the chart images (one per mode, plus one pair per
            // paper group) when the leaderboard data changed; the
            // timestamp parameter defeats the browser cache.
            const ts = Date.now();
            document.getElementById('chartImprove').src = '/api/leaderboard.png?mode=improve&t=' + ts;
            document.getElementById('chartInnovate').src = '/api/leaderboard.png?mode=innovate&t=' + ts;
            PAPER_GROUPS.forEach((g, i) => {
                document.getElementById('chartImproveGroup' + i).src = groupChartUrl('improve', g.label) + '&t=' + ts;
                document.getElementById('chartInnovateGroup' + i).src = groupChartUrl('innovate', g.label) + '&t=' + ts;
            });
        }

        // Per-paper × model matrix. Includes all runs (running and
        // terminal) of the selected mode. Not rebuilt while a detail
        // row is open, for the same reason as the Completed table.
        if (!detailFresh) {
            const matrix = renderMatrix(filterMode(runs, modeMatrix));
            const matrixKey = modeMatrix + '|' + matrix.body;
            if (window._lastMatrixBody !== matrixKey) {
                document.getElementById('matrixHead').innerHTML = matrix.head;
                document.getElementById('matrixBody').innerHTML = matrix.body;
                document.getElementById('matrixCount').textContent = matrix.count;
                window._lastMatrixBody = matrixKey;
            }
        }

        // Queue section: pending tasks (submitted to pool but not yet
        // assigned a worker due to max_concurrent cap).
        const pending = queue.pending || [];
        document.getElementById('queuedCount').textContent = `(${pending.length})`;
        document.getElementById('queuedCap').textContent = queue.max_concurrent
            ? `· cap ${queue.max_concurrent}`
            : '';
        const queuedHtml = renderQueued(pending);
        if (window._lastQueuedHtml !== queuedHtml) {
            document.getElementById('queuedBody').innerHTML = queuedHtml;
            window._lastQueuedHtml = queuedHtml;
        }

        document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
        pulse.classList.remove('error');
        banner.classList.remove('visible');
    } catch (err) {
        pulse.classList.add('error');
        banner.textContent = 'Failed to refresh: ' + err.message;
        banner.classList.add('visible');
    }
}

function loadOptions() {
    fetch('/api/options').then(r => r.json()).then(opts => {
        const fill = (id, values, selected) => {
            const sel = document.getElementById(id);
            const prev = new Set([...sel.selectedOptions].map(o => o.value));
            sel.innerHTML = '';
            for (const v of values) {
                const o = document.createElement('option');
                o.value = v; o.textContent = v;
                if (prev.has(v) || (selected && selected.includes(v))) o.selected = true;
                sel.appendChild(o);
            }
        };
        fill('selPapers', opts.papers || []);
        fill('selModels', opts.models || []);
        fill('selModes',  opts.modes  || ['innovate', 'improve'], ['innovate']);
    }).catch(() => {});
}

function selectedValues(id) {
    return [...document.getElementById(id).selectedOptions].map(o => o.value);
}

document.getElementById('f').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const payload = {
        papers: selectedValues('selPapers'),
        models: selectedValues('selModels'),
        modes:  selectedValues('selModes'),
    };
    if (!payload.modes.length) payload.modes = ['innovate'];
    if (!payload.papers.length || !payload.models.length) return;
    await fetch('/api/queue', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
    });
    refresh();
});

async function loadCapsIntoForm() {
    try {
        const q = await (await fetch('/api/queue')).json();
        const g = document.getElementById('capsGlobal');
        if (g && document.activeElement !== g) g.value = q.max_concurrent ?? '';
        const p = document.getElementById('capsPerProfile');
        const pc = q.profile_concurrency || {};
        const formatted = Object.entries(pc).map(([k,v]) => `${k}=${v}`).join(',');
        if (p && document.activeElement !== p && !p.value) p.value = formatted;
        const hint = document.getElementById('capsActiveHint');
        if (hint) hint.textContent = q.override_active ? '(override active)' : '(from yaml)';
    } catch (e) {}
}

document.getElementById('capsForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const gRaw = document.getElementById('capsGlobal').value.trim();
    const pRaw = document.getElementById('capsPerProfile').value.trim();
    const body = {};
    if (gRaw !== '') body.max_concurrent = parseInt(gRaw, 10);
    if (pRaw !== '') {
        const pc = {};
        for (const tok of pRaw.split(',').map(s => s.trim()).filter(Boolean)) {
            const m = tok.match(/^([^=\s]+)\s*=\s*(\d+)$/);
            if (!m) { alert(`Bad token: "${tok}". Expected name=N`); return; }
            pc[m[1]] = parseInt(m[2], 10);
        }
        body.profile_concurrency = pc;
    }
    const r = await fetch('/api/concurrency', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
    });
    if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        alert('Failed: ' + (j.error || r.status));
        return;
    }
    refresh();
});

// Mode-toggle change handler: invalidate the relevant section's
// last-rendered cache so refresh() repaints with the new mode.
function onModeChange(section) {
    if (section === 'Completed') window._lastCompletedHtml = null;
    else if (section === 'Matrix') window._lastMatrixBody = null;
    else if (section === 'Leaderboard') window._lastLbHtml = null;
    refresh();
}

// Queued section is collapsed by default — click the title to expand.
function toggleQueuedSection() {
    const w = document.getElementById('queuedWrap');
    const t = document.getElementById('queuedToggle');
    if (!w) return;
    const showing = w.style.display !== 'none';
    w.style.display = showing ? 'none' : '';
    if (t) t.textContent = showing ? '▶' : '▼';
}

buildGroupSections();
loadOptions();
loadCapsIntoForm();
refresh();
// The page refreshes every 60 s (see the setInterval calls at the end):
// with hundreds of rows, building the HTML is the most expensive thing
// the page does, and runs change state over minutes, not seconds.
const VER_COLORS = {verified:'#4caf7d', audit_clean:'#5b9bd5',
    unfixable:'#e05d5d', exhausted:'#e0a03c', incomplete:'#c9a227',
    in_progress:'#9a9a9a'};
let _verRows = [];
async function refreshVerification() {
    try {
        const r = await fetch('/api/verification');
        if (!r.ok) return;
        const d = await r.json();
        _verRows = d.rows || [];
        const c = d.counts || {};
        const parts = ['verified','audit_clean','unfixable','exhausted','incomplete','in_progress']
            .filter(k => c[k]).map(k => `${k}:${c[k]}`);
        document.getElementById('verificationCount').textContent =
            `(${_verRows.length}` + (parts.length ? ' — ' + parts.join(' · ') : '') + ')';
        renderVerification();
    } catch (e) { /* section is best-effort */ }
}
function renderVerification() {
    const filt = document.getElementById('verificationFilter').value;
    const rows = _verRows.filter(r => filt === 'all' || r.status === filt);
    const body = document.getElementById('verificationBody');
    if (!rows.length) {
        body.innerHTML = '<tr><td colspan="8" class="empty-state">Nothing with this status</td></tr>';
        return;
    }
    // Every field is escaped: the note and status are written by the
    // grader model, and the other fields come from file names.
    body.innerHTML = rows.map(r => {
        const col = VER_COLORS[r.status] || 'var(--text3)';
        const fin = r.finished_at ? String(r.finished_at).replace('T',' ').slice(0,16) : '';
        return `<tr>
            <td>${escapeHtml(r.paper_id)}</td><td>${escapeHtml(r.model)}</td><td>${escapeHtml(r.mode)}</td>
            <td><span style="color:${col};font-weight:600">${escapeHtml(r.status)}</span></td>
            <td>${escapeHtml(r.source || '')}</td><td>${escapeHtml(r.turns ?? '')}</td>
            <td style="white-space:nowrap">${escapeHtml(fin)}</td>
            <td style="max-width:420px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                title="${escapeHtml(r.note || '')}">${escapeHtml(r.note || '')}</td>
        </tr>`;
    }).join('');
}
refreshVerification();
setInterval(refreshVerification, 60000);
setInterval(() => { loadOptions(); loadCapsIntoForm(); refresh(); }, 60000);
</script>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────


def _paper_groups(cfg: dict | None = None) -> dict[str, set[str]]:
    """``dashboard.paper_groups`` from the config as {label: set of paper ids}.

    Malformed entries are skipped, so a bad config cannot break the page.
    """
    cfg = cfg or _config.load()
    raw = (cfg.get("dashboard") or {}).get("paper_groups") or {}
    groups: dict[str, set[str]] = {}
    if isinstance(raw, dict):
        for label, ids in raw.items():
            if isinstance(ids, (list, tuple, set)):
                groups[str(label)] = {str(p) for p in ids}
    return groups


def _model_order(cfg: dict | None = None) -> list[str]:
    """``dashboard.model_order`` from the config (profile names, in order)."""
    cfg = cfg or _config.load()
    raw = (cfg.get("dashboard") or {}).get("model_order") or []
    return [str(m) for m in raw] if isinstance(raw, (list, tuple)) else []


def _render_index() -> str:
    """The dashboard page with the port and display options filled in."""
    cfg = _config.load()
    display = {
        "paper_groups": {k: sorted(v) for k, v in _paper_groups(cfg).items()},
        "model_order": _model_order(cfg),
    }
    # The JSON lands inside a <script> element: escape the characters
    # that could end the element or start an HTML comment.
    blob = (json.dumps(display)
            .replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026"))
    return (_HTML
            .replace("{port}", str(cfg.get("port", 38582)))
            .replace("__DASHBOARD_CONFIG__", blob))


@app.route("/")
def index() -> Response:
    return Response(_render_index(), mimetype="text/html")


def _slim_run_for_list(r: dict) -> dict:
    """Strip per-cell scoring blobs from a run record for the list view.

    A run record carries the raw ``train_score``/``test_score`` JSONs and
    the per-cell arrays inside ``train_cmp``/``test_cmp``, tens of KB
    per run. The list view only renders summary fields (``delta``,
    ``n_better``, ``n_worse``, ``n_same``, ``n_missing``, ``n_total``),
    so the rest is dropped to keep the payload small with hundreds of
    runs. ``/api/run/<id>`` still returns the full record.
    """
    slim = {k: v for k, v in r.items()
            if k not in ("train_score", "test_score")}
    for k in ("train_cmp", "test_cmp"):
        v = slim.get(k)
        if isinstance(v, dict) and "cells" in v:
            slim[k] = {kk: vv for kk, vv in v.items() if kk != "cells"}
    return slim


@app.route("/api/runs")
def api_runs() -> Response:
    return jsonify(_json_safe([_slim_run_for_list(r) for r in _scan_runs()]))


@app.route("/api/verification")
def api_verification() -> Response:
    """Per-run verification state (see evaluation/verifier.py).

    Scans ``envs/*/*/scoring/verification.json``. Statuses: verified /
    unfixable / exhausted / incomplete / in_progress (written by the
    grader phase) and audit_clean (a run inspected by hand and found
    clean, without a grader re-execution).
    """
    rows = []
    counts: dict[str, int] = {}
    envs_root = _config.data_dir() / "envs"
    for vf in sorted(envs_root.glob("*/*/scoring/verification.json")):
        try:
            v = json.loads(vf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        run_dir = vf.parent.parent.name
        mode = "improve" if run_dir.endswith("-improve-eval") else "innovate"
        model = run_dir.removesuffix("-improve-eval").removesuffix("-innovate-eval")
        status = v.get("status") or "unknown"
        counts[status] = counts.get(status, 0) + 1
        note = (v.get("reason") or v.get("progress")
                or v.get("summary") or "")
        rows.append({
            "paper_id": vf.parent.parent.parent.name,
            "model": model,
            "mode": mode,
            "status": status,
            "source": ("audit" if status == "audit_clean" else
                       v.get("grader_profile") or ""),
            "turns": v.get("grader_turns"),
            "finished_at": v.get("finished_at"),
            "note": note[:200],
        })
    order = {"unfixable": 0, "exhausted": 1, "incomplete": 2,
             "in_progress": 2, "verified": 3, "audit_clean": 4}
    rows.sort(key=lambda r: (order.get(r["status"], 9), r["paper_id"], r["model"]))
    return jsonify({"rows": rows, "counts": counts})


@app.route("/api/queue")
def api_queue_state() -> Response:
    """Current orchestrator queue: what's running vs waiting for a slot.

    Pending = submitted to the pool but not yet assigned a worker
    (max_concurrent gate). Running = inside a worker, has started.
    """
    snap = orchestrator.snapshot()
    caps = orchestrator.current_caps()
    return jsonify({
        "max_concurrent": caps["max_concurrent"],
        "profile_concurrency": caps["profile_concurrency"],
        "override_active": caps["override_active"],
        "pool_max_workers": caps["pool_max_workers"],
        "pending": snap["pending"],
        "running": snap["running"],
    })


@app.route("/api/concurrency", methods=["POST"])
def api_concurrency() -> Response:
    """Live-edit the orchestrator caps. Body:
        {"max_concurrent": int, "profile_concurrency": {profile: int, ...}}
    Either field may be omitted to leave that override unchanged.
    Persists in-memory only (lost on dashboard restart).
    """
    body = request.get_json(force=True, silent=True) or {}
    mc = body.get("max_concurrent")
    pc = body.get("profile_concurrency")
    if mc is not None:
        try:
            mc = int(mc)
        except (TypeError, ValueError):
            return jsonify({"error": "max_concurrent must be int"}), 400
        if mc < 1:
            return jsonify({"error": "max_concurrent must be >= 1"}), 400
    if pc is not None:
        if not isinstance(pc, dict):
            return jsonify({"error": "profile_concurrency must be an object"}), 400
        try:
            pc = {str(k): int(v) for k, v in pc.items()}
        except (TypeError, ValueError):
            return jsonify({"error": "profile_concurrency values must be int"}), 400
        if any(v < 0 for v in pc.values()):
            return jsonify({"error": "profile_concurrency values must be >= 0"}), 400
    return jsonify(orchestrator.set_caps(max_concurrent=mc, profile_caps=pc))


@app.route("/api/run/<run_id>")
def api_run(run_id: str) -> Response:
    for r in _scan_runs():
        if r.get("run_id") == run_id:
            return jsonify(_json_safe(r))
    return jsonify({"error": "not found"}), 404


@app.route("/api/leaderboard")
def api_leaderboard() -> Response:
    """Leaderboard for one mode (defaults to ``improve``).

    Pass ``?mode=innovate|improve``. Each mode has its own cohort and
    per-paper coefficients, so they aggregate separately.
    """
    mode = request.args.get("mode") or "improve"
    return jsonify(_json_safe(_aggregate_leaderboard(_scan_runs(), mode=mode)))


def _leaderboard_chart_items(
    runs: list[dict], mode: str, paper_ids: set[str] | None = None,
) -> list[tuple[str, float, float]]:
    """``(model, mean final score, stddev)`` per model, best first.

    ``paper_ids`` restricts the runs to a subset of papers (a group from
    ``dashboard.paper_groups``); ``None`` uses every paper.
    """
    if paper_ids is not None:
        runs = [r for r in runs if r.get("paper_id") in paper_ids]
    lb = _aggregate_leaderboard(runs, mode=mode)
    items = [(m, lb[m]["final"].get("mean", 0.0),
              lb[m]["final"].get("stddev", 0.0))
             for m in lb.keys()
             if lb[m]["final"].get("count", 0) > 0]
    items.sort(key=lambda t: t[1], reverse=True)
    return items


@app.route("/api/leaderboard.png")
def api_leaderboard_png() -> Response:
    """Bar chart of the mean final score per model, × 100 (the FNI scale).

    ``?mode=innovate|improve`` (default ``improve``); ``?group=<label>``
    restricts the chart to one of the configured ``dashboard.paper_groups``.
    """
    mode = request.args.get("mode") or "improve"
    group = request.args.get("group")
    paper_ids: set[str] | None = None
    if group:
        groups = _paper_groups()
        if group not in groups:
            return jsonify({"error": f"unknown paper group {group!r}"}), 404
        paper_ids = groups[group]

    # Lazy import so importing dashboard doesn't force matplotlib at boot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = _leaderboard_chart_items(_scan_runs(), mode, paper_ids)
    # matplotlib's ``barh`` renders the first item at the bottom; reverse
    # so the highest score lands on top of the chart.
    items_for_plot = list(reversed(items))
    models = [t[0] for t in items_for_plot]
    means = [t[1] for t in items_for_plot]
    stds = [t[2] for t in items_for_plot]

    fig, ax = plt.subplots(figsize=(8, 3 + 0.3 * max(1, len(models))))
    if models:
        means_pct = [m * 100.0 for m in means]
        stds_pct = [s * 100.0 for s in stds]
        ax.barh(models, means_pct, xerr=stds_pct, color="#4c78a8", edgecolor="black")
        ax.axvline(0, color="#888", linewidth=0.8)
        title_suffix = f" — {group} ({len(paper_ids)} papers)" if group else ""
        ax.set_title(f"{mode.capitalize()} mode{title_suffix}")
        ax.set_xlabel("paper_score (% mean ± stddev) — positive = better")
    else:
        ax.text(0.5, 0.5, f"no completed {mode} runs yet",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        ax.axis("off")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return Response(buf.getvalue(), mimetype="image/png")


def _load_cohort() -> set[str] | None:
    """The set of papers being evaluated, or ``None`` if none is configured.

    ``cohort_file`` in the config names a JSON list of paper ids (path
    relative to ``data_dir``). When configured, ``/api/options`` offers
    exactly these papers and ``/api/queue`` rejects any other. (This is
    a set of *papers*; the normalization cohort is a set of profiles.)

    Falls back to ``None`` (no filter, all papers) when:
    - the config doesn't set ``cohort_file``
    - the file doesn't exist
    - the file isn't valid JSON or isn't a list

    Failing open is intentional: a misplaced file should not silently
    empty the paper list.
    """
    cfg = _config.load()
    name = cfg.get("cohort_file")
    if not name:
        return None
    path = _config.data_dir() / name
    if not path.exists():
        log.warning("cohort_file %s not found; falling back to no cohort filter", path)
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("cohort_file %s unreadable (%s); no filter", path, exc)
        return None
    if not isinstance(data, list):
        log.warning("cohort_file %s is not a JSON list; no filter", path)
        return None
    return {str(p) for p in data}


@app.route("/api/options")
def api_options() -> Response:
    """Populate the enqueue dropdowns.

    - papers: the active cohort (from ``cohort_file`` config) if set,
      otherwise every scaffolded paper id under ``envs/``. Returning
      the cohort here prevents callers from queueing non-cohort papers
      by accident — the dropdown becomes the source of truth, matching
      what the dashboard's per-paper matrix uses.
    - models: profile names from config
    - modes:  fixed list
    """
    cfg = _config.load()
    models = sorted((cfg.get("profiles") or {}).keys())

    cohort = _load_cohort()
    if cohort is not None:
        papers = sorted(cohort)
    else:
        scaffolded: set[str] = set()
        envs = _config.envs_dir()
        if envs.exists():
            for p in envs.iterdir():
                if p.is_dir():
                    scaffolded.add(p.name)
        papers = sorted(scaffolded)

    return jsonify({
        "papers": papers,
        "models": models,
        "modes": ["innovate", "improve"],
    })


@app.route("/api/queue", methods=["POST"])
def api_queue() -> Response:
    body = request.get_json(force=True, silent=True) or {}
    papers = body.get("papers") or []
    models = body.get("models") or []
    modes = body.get("modes") or ["innovate"]
    if not papers or not models:
        return jsonify({"error": "papers and models required"}), 400
    # Cohort gate. When a cohort_file is configured, reject submissions
    # that include any non-cohort paper — silent acceptance would let
    # callers waste compute on papers that aren't in the active cohort
    # (the per-paper matrix doesn't even render them). When no cohort
    # is configured, fall through unchanged.
    cohort = _load_cohort()
    if cohort is not None:
        bad = [p for p in papers if p not in cohort]
        if bad:
            return jsonify({
                "error": "non-cohort paper(s) rejected",
                "non_cohort_papers": sorted(bad)[:20],
                "non_cohort_count": len(bad),
                "cohort_size": len(cohort),
            }), 400
    orchestrator.bench(papers, models, modes=modes)
    return jsonify({"queued": len(papers) * len(models) * len(modes)})


@app.route("/api/queue", methods=["DELETE"])
def api_queue_remove() -> Response:
    """Remove queued (not-yet-running) cells from the pending queue.

    Body: ``{"models": [...], "papers": [...], "modes": [...]}`` — any field
    omitted (or empty) matches everything on that field. Only pending cells
    are removed; already-running cells are untouched. Returns the count.
    """
    body = request.get_json(force=True, silent=True) or {}
    models = body.get("models") or None
    papers = body.get("papers") or None
    modes = body.get("modes") or None
    removed = orchestrator.remove_pending(models=models, papers=papers, modes=modes)
    return jsonify({"removed": removed})


# ── Entrypoint ────────────────────────────────────────────────────────────


def _kill_agent_subtrees(*_unused) -> None:
    """SIGKILL every agent process group spawned by this dashboard.

    Apptainer double-forks (``starter-suid`` → ``appinit`` → agent CLI),
    so a signal to the dashboard alone would leave the container
    processes running, reparented to PID 1 and holding the bind mounts
    of their workspaces. Each agent is launched with
    ``start_new_session=True`` so its process group id equals its pid;
    this walks the dashboard's direct children, finds the Apptainer
    parents and signals their whole process group.

    Linux only (reads ``/proc``), and it recognises the process name of
    a setuid Apptainer installation (``starter-suid``); with another
    installation type it finds nothing and agents must be stopped by
    hand.
    """
    self_pid = os.getpid()
    killed = 0
    try:
        for tid in os.listdir(f"/proc/{self_pid}/task"):
            try:
                with open(f"/proc/{self_pid}/task/{tid}/children") as f:
                    child_pids = f.read().split()
            except OSError:
                continue
            for cpid in child_pids:
                try:
                    cpid_int = int(cpid)
                    with open(f"/proc/{cpid}/comm") as f:
                        comm = f.read().strip()
                    # Apptainer's parent process is named ``starter-suid``.
                    if comm == "starter-suid":
                        os.killpg(cpid_int, signal.SIGKILL)
                        killed += 1
                except (OSError, ValueError, ProcessLookupError):
                    pass
    except Exception as exc:
        log.warning("kill_agent_subtrees failed: %s", exc)
    if killed:
        log.warning("dashboard shutdown: SIGKILL'd %d agent process group(s)", killed)
    sys.exit(0)


def _sweep_orphan_agents() -> None:
    """On startup, kill agent container trees orphaned by a previous dashboard.

    ``apptainer exec`` double-forks (``starter-suid`` → ``appinit`` →
    agent CLI). When the dashboard process dies without killing the
    Apptainer process group, the container processes are reparented to
    PID 1 and keep running, possibly in a workspace that has since been
    deleted.

    Detection: ``claude`` / ``codex`` processes whose parent is
    ``appinit`` and grandparent ``starter-suid`` (so they run inside an
    Apptainer container), and whose working directory is either deleted
    or inside ``<data_dir>/envs/``. Their whole process group is killed.

    Same limits as :func:`_kill_agent_subtrees` (Linux, setuid
    Apptainer), and only the ``claude`` and ``codex`` CLIs are
    recognised.
    """
    envs_root = str(_config.envs_dir().resolve())
    self_pid = os.getpid()
    killed_pgids: set[int] = set()
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{pid}/comm") as f:
                    comm = f.read().strip()
                if comm not in ("claude", "codex"):
                    continue
                # cwd readlink — must be ours (PIDs running as our UID).
                cwd = os.readlink(f"/proc/{pid}/cwd")
                if "(deleted)" not in cwd and not cwd.startswith(envs_root):
                    continue
                with open(f"/proc/{pid}/status") as f:
                    status = f.read()
                # Require the Apptainer parent and grandparent: this
                # confirms an agent container, not an unrelated claude /
                # codex process of the same user.
                ppid = int(status.split("PPid:")[1].split()[0])
                with open(f"/proc/{ppid}/comm") as f:
                    if f.read().strip() != "appinit":
                        continue
                with open(f"/proc/{ppid}/status") as f:
                    pppid = int(f.read().split("PPid:")[1].split()[0])
                with open(f"/proc/{pppid}/comm") as f:
                    if f.read().strip() != "starter-suid":
                        continue
                # Skip if the starter-suid is still a child of the
                # current dashboard — that's a live agent, not an orphan.
                with open(f"/proc/{pppid}/status") as f:
                    starter_ppid = int(f.read().split("PPid:")[1].split()[0])
                if starter_ppid == self_pid:
                    continue
                # Kill the process group the orphan tree shares (the
                # one created by start_new_session=True at launch).
                pgid = os.getpgid(pid)
                if pgid in killed_pgids:
                    continue
                killed_pgids.add(pgid)
                os.killpg(pgid, signal.SIGKILL)
            except (OSError, ValueError, ProcessLookupError):
                continue
    except Exception as exc:
        log.warning("orphan-agent sweep failed: %s", exc)
    if killed_pgids:
        log.warning(
            "startup sweep: SIGKILL'd %d orphaned agent process group(s): %s",
            len(killed_pgids), sorted(killed_pgids),
        )


def _sweep_orphan_run_artifacts() -> None:
    """On startup, delete leftover overlay.img + container.sif from runs
    that were in flight when a previous dashboard died.

    ``run_one()`` removes these artifacts when it returns. When the
    dashboard process is killed, every in-flight run dies before that
    point and leaves its container image and overlay (tens of GB)
    behind.

    The orchestrator queue lives in memory and starts empty, so right
    after startup any env_dir whose ``.evaluation.json`` still says
    ``status=running`` belongs to a run of a previous process. Only the
    disposable container artifacts are removed; the run's work
    (scoring/, trajectory/, .git, method/, ...) is kept, and the run
    rebuilds the artifacts if it is queued again.

    Assumes one harness process per ``data_dir``: a run started by a
    concurrent ``run`` / ``bench`` command would look the same.
    """
    envs_root = _config.envs_dir()
    if not envs_root.exists():
        return
    deleted = 0
    freed = 0
    for paper_dir in envs_root.iterdir():
        if not paper_dir.is_dir():
            continue
        for run_dir in paper_dir.iterdir():
            if not (run_dir.is_dir() and run_dir.name.endswith("-eval")):
                continue
            meta_file = run_dir / ".evaluation.json"
            if not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if meta.get("status") != "running":
                continue
            for artifact in ("environment/overlay.img",
                             "environment/container.sif"):
                p = run_dir / artifact
                try:
                    if p.exists():
                        sz = p.stat().st_size
                        p.unlink()
                        deleted += 1
                        freed += sz
                except OSError:
                    pass
    if deleted:
        log.warning(
            "startup sweep: reclaimed %d leaked run artifact(s) "
            "(%.1f GB) from %d-run env_dirs",
            deleted, freed / 1e9, deleted // 2,
        )


def serve(port: int | None = None, host: str | None = None) -> None:
    """Run the dashboard. Blocks until killed.

    ``host`` and ``port`` default to the ``host`` / ``port`` config keys
    (``127.0.0.1`` and ``38582``). The dashboard is unauthenticated and
    can start runs, so only bind another interface (``0.0.0.0``) on a
    trusted network.
    """
    cfg = _config.load()
    actual_port = int(port if port is not None else cfg.get("port", 38582))
    actual_host = str(host or cfg.get("host") or "127.0.0.1")
    # Fail now, with a clear message, if the data directory is not set.
    _config.data_dir()

    # Clean up any agent process trees orphaned by a previous dashboard
    # process before starting new runs.
    _sweep_orphan_agents()
    # Reclaim disposable container artifacts (overlay.img, container.sif)
    # from runs that were in flight when the previous dashboard died.
    _sweep_orphan_run_artifacts()

    # On SIGTERM / SIGINT, kill the Apptainer process tree of every
    # running agent so no agent container outlives the dashboard.
    signal.signal(signal.SIGTERM, _kill_agent_subtrees)
    signal.signal(signal.SIGINT, _kill_agent_subtrees)

    log.info("Serving evaluation dashboard on %s:%d", actual_host, actual_port)
    # debug=False so the reloader doesn't spawn a second orchestrator pool.
    app.run(host=actual_host, port=actual_port, debug=False, threaded=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    serve()
