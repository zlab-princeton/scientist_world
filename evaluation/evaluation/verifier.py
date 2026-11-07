"""Independent verification phase — a fresh grader session re-scores a run.

After an agent run completes, a *grader* (a separate model profile, by
convention named ``{model}-{effort}-grader``) is launched in a completely
new session inside the run's workspace. Its job, per
``prompts/task_verify.md``: quarantine agent-produced result files,
re-run the frozen method on train and test, regenerate raw predictions,
grade them with the pristine evaluators, check the run against the
task's constraints, repair evaluation conditions / infrastructure /
mechanical defects (never method logic), and exit by writing
``scoring/verification.json``.

Exit states (the phase always terminates in exactly one):

- ``verified``   — checks passed (possibly after fixes); the grader's
                   recomputed scores overwrite ``scores_{train,test}.json``,
                   are propagated into ``.evaluation.json``, and
                   normalization is re-run for the paper.
- ``unfixable``  — evidence-based conclusion that the run is not a
                   legitimate attempt at the task (never a quality
                   judgment); score invalid pending a fresh rerun.
- ``exhausted``  — the grader session ran and ended without a verdict,
                   or a "verified" claim carried no evidence of executed
                   jobs / recomputed scores (downgraded). NOT a pass.

Invariant: ``verify_run`` always leaves a ``scoring/verification.json``
behind, and downstream consumers should treat only ``status: "verified"``
as trustworthy.

The run's ``.evaluation.json`` is snapshotted before the grader session
and restored afterwards (plus a ``verification`` summary field, plus —
on ``verified`` — the grader's recomputed scores) so the verifier never
clobbers the original run's recorded state or GPU accounting.

A fourth, non-terminal status, ``incomplete``, marks a grader session
that was interrupted (quota, authentication, container build, turn
budget) before reaching a verdict; such a run can be graded again.

The grader profile is ``verifier.grader_profile`` in the config (or
``--grader``); there is no default.

CLI::

    python -m evaluation.verifier <paper_id> <profile_name> <mode>

Exit codes: 0 verified, 2 unfixable, 3 exhausted, 1 anything else
(``incomplete`` or an error).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"verified", "unfixable"})

# "exhausted" means the grader worked and still could not reach a
# verdict. It must never be used for a session that was lost: provider
# quota, expired authentication, a gateway error, a failed container
# build, or a turn budget that ran out are all conditions a later retry
# can clear. Those get "incomplete" instead, which is non-terminal, so
# the run can simply be graded again.
RETRYABLE_STATUS = "incomplete"
_RETRYABLE_REASON_MARKERS = (
    "session error",   # crash / rc=1 / quota / auth
    "setup error",     # container build, setup.sh
    "max_turns",       # ran out of turn budget mid-audit
    "TransientError",
    "quota",
    "rate limit",
    "429",
)

# Mechanical-scan bounds: don't parse multi-GB artifacts (they still get
# a streaming md5 for duplicate detection).
_SCAN_MAX_FILE_BYTES = 64 * 1024 * 1024
_SCAN_MAX_FILES = 200


# ── Mechanical scan (pure Python, no LLM) ────────────────────────────────────


def _git(env_dir: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(env_dir), *args],
        capture_output=True, text=True, timeout=120,
    )
    return (proc.stdout or "").strip()


def _git_safe(env_dir: Path, *args: str) -> str:
    """``_git`` that never raises — for cleanup paths."""
    try:
        return _git(env_dir, *args)
    except Exception as exc:  # noqa: BLE001 — cleanup must not fail
        return f"<git {' '.join(args[:2])} failed: {exc}>"


def _numeric_values(path: Path) -> list[float]:
    """Best-effort extraction of numeric leaves from a json/jsonl/csv file."""
    vals: list[float] = []

    def walk(obj):
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            if math.isfinite(obj):
                vals.append(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    try:
        text = path.read_text(errors="replace")
    except OSError:
        return vals
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                walk(json.loads(line))
            except json.JSONDecodeError:
                continue
    elif path.suffix == ".json":
        try:
            walk(json.loads(text))
        except json.JSONDecodeError:
            pass
    elif path.suffix == ".csv":
        for line in text.splitlines()[1:]:
            for tok in line.split(","):
                try:
                    v = float(tok)
                    if math.isfinite(v):
                        vals.append(v)
                except ValueError:
                    continue
    return vals


def _is_round(v: float) -> bool:
    """Short-mantissa value (≤3 decimal places) — the hand-typed signature."""
    return abs(v - round(v, 3)) < 1e-12


def _is_organic(v: float) -> bool:
    """Long-mantissa value — the signature of a genuinely computed float."""
    if _is_round(v):
        return False
    frac = repr(abs(v) % 1.0)
    return len(frac) >= 10  # "0." + ≥8 fractional digits


def _round_share(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return sum(1 for v in vals if _is_round(v)) / len(vals)


def _scan_file(path: Path, env_dir: Path) -> dict:
    st = path.stat()
    # Workspace-relative path: the scan is committed and published with
    # the run, and the grader sees the workspace at /home/user, so a
    # host path would be both a leak and useless to it.
    entry: dict = {"path": path.relative_to(env_dir).as_posix(),
                   "bytes": st.st_size}
    md5 = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            md5.update(chunk)
    entry["md5"] = md5.hexdigest()
    # Numeric analysis only on reasonably-sized files — a multi-GB
    # artifact gets hashed (for duplicate detection) but not parsed.
    vals = _numeric_values(path) if st.st_size <= _SCAN_MAX_FILE_BYTES else []
    if vals:
        # Repeated-value clusters are computed over non-trivial ROUND
        # values only — integers/0/1 repeat legitimately (labels,
        # counts), and a repeated long-mantissa float is normal in
        # sweep artifacts (the same measured value echoed across rows).
        # A short hand-typed-looking value repeated across rows is the
        # actual fabrication signature.
        nontrivial = [v for v in vals if v not in (0.0, 1.0, -1.0)
                      and not float(v).is_integer()]
        round_nontrivial = [v for v in nontrivial if _is_round(v)]
        top = Counter(round_nontrivial).most_common(1) if round_nontrivial else []
        entry.update({
            "n_numeric_values": len(vals),
            "n_distinct_values": len(set(vals)),
            "round_value_share": round(_round_share(vals), 4),
            "n_organic_values": sum(1 for v in vals if _is_organic(v)),
            "n_round_nontrivial": len(round_nontrivial),
        })
        if top:
            entry["top_value"] = top[0][0]
            entry["top_value_count"] = top[0][1]
    return entry


def mechanical_scan(env_dir: Path) -> dict:
    """Deterministic pre-scan of a run's scored artifacts.

    Produces *leads* for the grader (each finding needs confirmation):
    constant/degenerate outputs, byte-identical files (test copied from
    train), hand-typed values mixed into organic records, and test
    scores whose producing job failed. File paths in the result are
    relative to the workspace.
    """
    findings: list[str] = []
    files: list[dict] = []

    # Candidate files: anything result-shaped — everything under
    # scoring/, plus json/jsonl/csv files whose own name OR any parent
    # directory mentions predictions/records/results/outputs. One
    # bounded walk; bulky non-artifact trees are skipped outright.
    _skip_dirs = {".git", ".claude", ".venv", "data", "downloads",
                  "checkpoints", "wheels", "local-packages", "node_modules",
                  "__pycache__", ".cache", ".uv_cache", "site-packages"}
    _name_hints = ("prediction", "record", "result", "output")
    scoring = env_dir / "scoring"

    def _iter_candidates():
        """Deterministic order, scoring/ strictly first — so the scan's
        truncation cap can never drop the score files themselves."""
        import os
        scoring_files: list[Path] = []
        other_files: list[Path] = []
        for root, dirs, names in os.walk(env_dir):
            dirs[:] = sorted(d for d in dirs if d not in _skip_dirs)
            root_p = Path(root)
            rel = root_p.relative_to(env_dir).as_posix()
            in_scoring = rel == "scoring" or rel.startswith("scoring/")
            dir_hinted = any(h in rel.lower() for h in _name_hints)
            for name in sorted(names):
                p = root_p / name
                if p.suffix not in (".json", ".jsonl", ".csv"):
                    continue
                if in_scoring:
                    scoring_files.append(p)
                elif dir_hinted or any(h in name.lower() for h in _name_hints):
                    other_files.append(p)
        yield from scoring_files
        yield from other_files

    for p in _iter_candidates():
        if len(files) >= _SCAN_MAX_FILES:
            findings.append(f"scan truncated at {_SCAN_MAX_FILES} files")
            break
        try:
            files.append(_scan_file(p, env_dir))
        except OSError:
            continue

    # Cross-file duplicate detection (e.g. test slice = copy of train).
    by_md5: dict[str, list[str]] = {}
    for f in files:
        if "md5" in f and f["bytes"] > 64:
            by_md5.setdefault(f["md5"], []).append(f["path"])
    for md5, paths in by_md5.items():
        if len(paths) > 1:
            findings.append(f"byte-identical files (md5 {md5[:8]}): {paths}")

    for f in files:
        n = f.get("n_numeric_values", 0)
        if n >= 10 and f.get("n_distinct_values", n) <= 2:
            findings.append(
                f"near-constant values in {f['path']} "
                f"({f['n_distinct_values']} distinct of {n})"
            )
        if n >= 10 and f.get("round_value_share", 0.0) >= 0.9:
            findings.append(
                f"values look hand-typed (≥90% round numbers) in {f['path']}"
            )
        # A non-trivial short value repeated many times in a results
        # file (e.g. the same runtime recorded for every test item).
        if f.get("top_value_count", 0) >= 5:
            findings.append(
                f"repeated non-trivial value {f['top_value']} appears "
                f"{f['top_value_count']}x in {f['path']}"
            )
        # Organic floats and short hand-typed values mixed in one file —
        # the signature of a partially hand-edited results file (real
        # rows next to fabricated rows). Requires the hand-typed share
        # to be substantial: a few round config fields inside thousands
        # of measured values is normal.
        n_org = f.get("n_organic_values", 0)
        n_rnd = f.get("n_round_nontrivial", 0)
        if (n_org >= 8 and n_rnd >= 8
                and n_rnd / max(1, n_org + n_rnd) >= 0.15):
            findings.append(
                f"mixed precision in {f['path']}: "
                f"{n_org} organic-looking floats alongside "
                f"{n_rnd} short hand-typed-looking values "
                "— check whether a subset of rows was written by hand"
            )

    # Test-job history from git: a final failed test job + an existing
    # scores_test.json means the test score is agent-authored.
    log_text = _git(env_dir, "log", "--format=%s", "-500")
    rcs = re.findall(r"eval: test job \S+ done \(rc=(-?\d+)\)", log_text)
    if rcs:
        last_rc = int(rcs[0])
        if last_rc != 0 and (scoring / "scores_test.json").exists():
            findings.append(
                f"final test job exited rc={last_rc} but scores_test.json "
                "exists — test score is likely agent-authored, must be "
                "regenerated by a successful pristine job"
            )
    elif (scoring / "scores_test.json").exists():
        findings.append(
            "no 'eval: test job' commits found but scores_test.json exists "
            "— provenance unclear"
        )

    return {
        "findings": findings or ["no mechanical findings — full checks still required"],
        "test_job_rcs_newest_first": rcs[:10],
        "files": files,
    }


# ── Verification phase ───────────────────────────────────────────────────────


def _max_trajectory_turn(env_dir: Path) -> int:
    traj = env_dir / "trajectory"
    turn = 0
    if traj.is_dir():
        for p in traj.iterdir():
            if p.name.startswith("turn_"):
                try:
                    turn = max(turn, int(p.name.split("_")[1].split(".")[0]))
                except (ValueError, IndexError):
                    pass
    return turn


def _restore_pristine_eval(env_dir: Path) -> None:
    """Materialize eval/ and scripts/evaluate_test.sh from source/.

    The grader container cannot see the host-side ``source/`` sibling,
    so the harness puts the pristine evaluators (including the held-out
    one the agent never saw) in place before the session starts.
    """
    source = env_dir.parent / "source"
    src_script = source / "scripts" / "evaluate_test.sh"
    if src_script.exists():
        dst = env_dir / "scripts" / "evaluate_test.sh"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        shutil.copy2(src_script, dst)
        dst.chmod(0o755)
    src_eval = source / "eval"
    if src_eval.exists():
        dst_eval = env_dir / "eval"
        if dst_eval.exists():
            shutil.rmtree(dst_eval)
        shutil.copytree(src_eval, dst_eval, symlinks=False)


def verify_run(
    paper_id: str,
    profile_name: str,
    mode: str,
    *,
    grader_profile_name: str | None = None,
    max_turns: int | None = None,
    gpu_budget_hours: float | None = None,
    publish: bool | None = None,
    force: bool = False,
) -> dict:
    """Run the independent verification phase on one completed run.

    Returns the final ``verification.json`` content. Grader outcomes
    (verified / unfixable / exhausted / incomplete) never raise; only
    pre-flight errors (no grader profile, missing env_dir, existing
    terminal verification without ``force``) do.
    """
    # Lazy imports (avoid import cycles; keep the module importable
    # where only the dashboard is used).
    from evaluation import cache
    from evaluation import config as _config
    from evaluation import container as _container
    from evaluation import git
    from evaluation import loop as _loop
    from evaluation import normalization
    from evaluation.models import get_profile
    from evaluation.nexus import Nexus
    from evaluation.runner import _project_id, _read_json, _utcnow, _write_meta

    cfg = _config.load()
    vcfg = cfg.get("verifier") or {}
    grader_profile_name = grader_profile_name or vcfg.get("grader_profile")
    if not grader_profile_name:
        raise _config.ConfigError(
            "No grader profile: set verifier.grader_profile in "
            f"{_config.CONFIG_PATH} (or pass --grader) to the name of a "
            "profile defined under `profiles`."
        )
    max_turns = int(max_turns or vcfg.get("max_turns") or 12)
    gpu_budget_hours = float(gpu_budget_hours or vcfg.get("gpu_budget_hours") or 8.0)
    max_relaunches = int(vcfg.get("max_relaunches") or 2)
    if publish is None:
        publish = bool(vcfg.get("publish", True))

    grader_profile = get_profile(grader_profile_name)
    data_dir = _config.data_dir()
    env_dir = data_dir / "envs" / paper_id / f"{profile_name}-{mode}-eval"
    if not env_dir.is_dir():
        raise RuntimeError(f"env_dir missing: {env_dir}")

    ver_path = env_dir / "scoring" / "verification.json"
    existing = _read_json(ver_path) or {}
    if existing.get("status") in TERMINAL_STATUSES and not force:
        raise RuntimeError(
            f"verification already terminal ({existing.get('status')}) for "
            f"{env_dir} — pass force=True / --force to redo"
        )

    meta_snapshot = _read_json(env_dir / ".evaluation.json") or {}
    frozen_head = _git(env_dir, "rev-parse", "HEAD")
    ver_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Verify %s/%s-%s: grader=%s frozen=%s",
             paper_id, profile_name, mode, grader_profile_name, frozen_head[:10])

    scan = mechanical_scan(env_dir)
    (env_dir / "scoring" / "verifier_scan.json").write_text(
        json.dumps(scan, indent=2) + "\n"
    )
    ver_path.write_text(json.dumps({
        "status": "in_progress",
        "grader_profile": grader_profile_name,
        "frozen_method_head": frozen_head,
        "started_at": _utcnow(),
    }, indent=2) + "\n")
    git.commit(env_dir, f"verify: mechanical scan + verification start "
                        f"(grader={grader_profile_name})")

    nexus: Nexus | None = None
    project_id = _project_id(paper_id, profile_name, mode, variant="verify")
    loop_reason = "setup"
    loop_turns = 0
    try:
        # ── Setup (inside try: any failure must still produce a
        #    terminal marker and clean up the Nexus project) ──────────
        _restore_pristine_eval(env_dir)
        git.commit(env_dir, "verify: restored pristine eval/ + "
                            "evaluate_test.sh from source")

        extra_binds = cache.apply_to_workspace(paper_id, env_dir)
        container_sif = _container.ensure_container(
            env_dir, container_cache_dir=str(cfg.get("container_cache_dir") or ""),
        )
        overlay_path = _container.ensure_overlay(
            env_dir, container_sif,
            container_cache_dir=str(cfg.get("container_cache_dir") or ""),
            local_tmp_dir=str(cfg.get("local_tmp_dir") or "/tmp"),
            uv_cache_dir=str(cfg.get("uv_cache_dir") or ""),
        )

        nexus = Nexus(cfg["nexus"]["url"])
        nexus.delete_project(project_id)
        import time as _t
        _t.sleep(1)
        nexus.create_project(
            project_id=project_id,
            workspace_dir=str(env_dir),
            sif_path=str(container_sif),
            overlay_path=str(overlay_path) if overlay_path else "",
            gpu_budget=gpu_budget_hours,
            max_gpus=int(cfg["nexus"]["max_gpus"]),
            allowed_resources=_config.nexus_all_resources(cfg),
            checkpoint_interval=int(cfg.get("checkpoint_interval") or 1200),
            checkpoint_timeout=int(cfg.get("checkpoint_timeout") or 1800),
        )

        prompt = _loop._render(
            "verifier",
            paper_id=paper_id,
            run_profile=profile_name,
            mode=mode,
            frozen_head=frozen_head,
            scan_report=json.dumps(
                {"findings": scan["findings"],
                 "test_job_rcs_newest_first": scan["test_job_rcs_newest_first"]},
                indent=2,
            ),
            max_gpus=int(cfg["nexus"]["max_gpus"]),
            gpu_budget=gpu_budget_hours,
        )

        # A claude-code grader must not inherit the original agent's
        # .claude session state: gateway-model thinking blocks carry
        # signatures the Anthropic API rejects on resume ("Invalid
        # `signature` in `thinking` block"). Preserve the agent's state
        # under .claude.agent-original (an audit artifact) and let the
        # grader start a fresh session dir.
        if grader_profile.scaffold in ("claude-code", "claude-tmux"):
            # The parked directory holds the agent's login file: make
            # sure it is ignored before the next commit.
            git.ensure_credentials_ignored(env_dir)
            for name in (".claude", ".claude.json"):
                src = env_dir / name
                dst = env_dir / f"{name}.agent-original"
                if src.exists() and not dst.exists():
                    src.rename(dst)

        # ── Grader session, with bounded relaunches on crash ─────────
        for attempt in range(1, max_relaunches + 1):
            # Offset turn numbering past everything already in
            # trajectory/ so grader artifacts never clobber the
            # agent's (or an earlier attempt's).
            turn_offset = _max_trajectory_turn(env_dir)
            try:
                result = _loop.run(
                    env_dir, grader_profile, nexus, project_id, prompt,
                    max_turns=max_turns,
                    gpu_budget_hours=gpu_budget_hours,
                    extra_binds=extra_binds,
                    turn_offset=turn_offset,
                )
                # loop.run reports the absolute turn number; record only
                # the turns this grader session used.
                loop_reason = result.reason
                loop_turns = result.turns - turn_offset
                break
            except Exception as exc:  # noqa: BLE001 — relaunch or mark
                loop_reason = (f"grader session error (attempt "
                               f"{attempt}/{max_relaunches}): "
                               f"{type(exc).__name__}: {exc}")
                log.exception("Verifier grader session failed for %s "
                              "(attempt %d)", env_dir, attempt)
    except Exception as exc:  # noqa: BLE001 — setup failure
        loop_reason = f"setup error: {type(exc).__name__}: {exc}"
        log.exception("Verifier setup failed for %s", env_dir)
    finally:
        # Every step below is individually guarded: the meta restore
        # and Nexus cleanup must happen no matter what fails above.
        try:
            meta_after = _read_json(env_dir / ".evaluation.json") or {}
        except Exception:  # noqa: BLE001
            meta_after = {}
        verifier_gpu_hours = meta_after.get("gpu_hours_used")

        try:
            ver = _read_json(ver_path) or {}
        except Exception:  # noqa: BLE001
            ver = {}
        if ver.get("status") not in TERMINAL_STATUSES:
            # Distinguish "the grader audited this and could not
            # conclude" from "the session was interrupted". Only the
            # former is a real 'exhausted'; the latter is an
            # infrastructure condition a retry can clear, so it must not
            # be recorded as a terminal status.
            reason = str(loop_reason or "")
            interrupted = (
                not loop_turns
                or any(m in reason for m in _RETRYABLE_REASON_MARKERS)
            )
            if interrupted:
                ver["status"] = RETRYABLE_STATUS
                ver["progress"] = (
                    f"grader session interrupted ({loop_reason}) after "
                    f"{loop_turns} turn(s) — not a verdict; left "
                    f"retryable so the run can be graded again"
                )
            else:
                ver["status"] = "exhausted"
                ver["progress"] = (
                    f"grader session ended ({loop_reason}) after "
                    f"{loop_turns} turn(s) without a terminal "
                    f"verification status"
                )
        # Anti-rubber-stamp: a "verified" with no recomputed scores or
        # no executed jobs carries no evidence — downgrade.
        if ver.get("status") == "verified" and (
                not ver.get("scores_train") or not ver.get("scores_test")
                or not verifier_gpu_hours):
            ver["status"] = "exhausted"
            ver["progress"] = (
                "downgraded from 'verified': missing recomputed scores "
                "and/or zero executed jobs — a verification must carry "
                "evidence of its own re-runs"
            )
        ver.setdefault("grader_profile", grader_profile_name)
        ver.setdefault("frozen_method_head", frozen_head)
        ver["finished_at"] = _utcnow()
        ver["grader_turns"] = loop_turns
        ver["grader_gpu_hours"] = verifier_gpu_hours
        # Audit trail: what (if anything) changed inside method/ during
        # verification. Mechanical fixes are allowed but must be visible.
        ver["method_diff_stat"] = _git_safe(
            env_dir, "diff", "--stat", f"{frozen_head}..HEAD", "--", "method/",
        )
        try:
            ver_path.write_text(json.dumps(ver, indent=2) + "\n")
        except Exception:  # noqa: BLE001
            log.exception("Failed to write verification.json for %s", env_dir)

        # Restore the run's original recorded state, then layer on the
        # verification summary — and, on verified, the grader's
        # recomputed scores (which overwrote scoring/scores_*.json).
        try:
            base = dict(meta_snapshot) if meta_snapshot else dict(meta_after)
            base["verification"] = {
                "status": ver.get("status"),
                "grader_profile": grader_profile_name,
                "finished_at": ver.get("finished_at"),
            }
            if ver.get("status") == "verified":
                ts = _read_json(env_dir / "scoring" / "scores_train.json")
                te = _read_json(env_dir / "scoring" / "scores_test.json")
                if ts:
                    base["train_score"] = ts
                if te:
                    base["test_score"] = te
            _write_meta(env_dir, base)
        except Exception:  # noqa: BLE001
            log.exception("Failed to restore .evaluation.json for %s", env_dir)

        try:
            git.commit(env_dir, f"verify: {ver.get('status')} "
                                f"(grader={grader_profile_name}, turns={loop_turns})")
        except Exception as exc:  # noqa: BLE001
            log.warning("verify final commit failed: %s", exc)
        if nexus is not None:
            try:
                nexus.delete_project(project_id)
            except Exception:  # noqa: BLE001
                pass

    # Verified scores changed the run's effective numbers — recompute
    # the paper's normalization so downstream reads stay consistent.
    if ver.get("status") == "verified":
        try:
            normalization.update_for_paper(paper_id)
        except Exception:  # noqa: BLE001
            log.exception("Normalization update after verification failed "
                          "for %s", paper_id)

    if publish:
        try:
            from evaluation import git_publish
            git_publish.publish(env_dir, paper_id, f"{profile_name}-grader", mode)
        except Exception:  # noqa: BLE001
            log.exception("Verification publish failed for %s/%s — "
                          "verification result unchanged", paper_id, profile_name)

    log.info("Verify %s/%s-%s -> %s", paper_id, profile_name, mode,
             ver.get("status"))
    return ver


_EXIT_CODES = {"verified": 0, "unfixable": 2, "exhausted": 3}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the independent verification phase on one eval run.",
    )
    parser.add_argument("paper_id")
    parser.add_argument("profile_name")
    parser.add_argument("mode", choices=["improve", "innovate"])
    parser.add_argument("--grader", default=None,
                        help="grader profile name (default: config verifier.grader_profile)")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--gpu-budget-hours", type=float, default=None)
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="redo even if a terminal verification exists")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from evaluation.config import ConfigError
    try:
        ver = verify_run(
            args.paper_id, args.profile_name, args.mode,
            grader_profile_name=args.grader,
            max_turns=args.max_turns,
            gpu_budget_hours=args.gpu_budget_hours,
            publish=False if args.no_publish else None,
            force=args.force,
        )
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(ver, indent=2))
    return _EXIT_CODES.get(ver.get("status"), 1)


if __name__ == "__main__":
    sys.exit(main())
