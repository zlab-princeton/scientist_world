"""Per-run orchestration: scaffold the workspace, drive the agent, finalize.

One call to :func:`run_one` is one ``(paper, profile, mode)`` trial:

    1. Resume an interrupted run in place when its workspace has
       completed turns and no held-out score or grader verdict;
       otherwise scaffold a fresh workspace with
       ``sciworld.research(setup_only=True)`` and strip everything the
       agent must not see (held-out evaluator, reference scores).
    2. Build the paper's compute container and dependency overlay,
       install the ``.gitignore``, initialise a local git repository.
    3. Create a per-run Nexus project ("eval-<hash>") with the GPU budget.
    4. Drive the agent loop (:mod:`evaluation.loop`) until ``done``.
    5. Validate ``scoring/scores_train.json`` against the paper's own
       train scores, resuming the agent to fill gaps.
    6. Finalize: status, usage, ``.evaluation.json``, commit, refresh
       the paper's normalization, optionally publish, optionally run the
       independent grader (:mod:`evaluation.verifier`), which produces
       the held-out score.

Retry policy: up to ``max_retries`` further attempts after a
:class:`TransientError`, never once the agent has written anything
under ``scoring/``.
"""

from __future__ import annotations

import ast
import datetime as _dt
import hashlib
import json
import logging
import shutil
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from evaluation import cache, config as _config, container as _container
from evaluation import git
from evaluation import normalization
from evaluation.loop import TransientError, FatalAgentError
from evaluation.loop import run as run_loop, _render as _loop_render
from evaluation.models import get_profile
from evaluation.nexus import Nexus

log = logging.getLogger(__name__)

# Per-paper scaffold lock: when several profiles target the same paper at
# the same time, only one calls ``sciworld.research`` at a time. The
# others wait, then reuse the ``source/`` clone the first one created
# instead of racing concurrent clones and copies of the same tree.
_SCAFFOLD_LOCKS: dict[str, threading.Lock] = {}
_SCAFFOLD_LOCKS_GUARD = threading.Lock()


def _paper_scaffold_lock(paper_id: str) -> threading.Lock:
    with _SCAFFOLD_LOCKS_GUARD:
        lk = _SCAFFOLD_LOCKS.get(paper_id)
        if lk is None:
            lk = threading.Lock()
            _SCAFFOLD_LOCKS[paper_id] = lk
        return lk

# Default .gitignore template for run workspaces: the worker's workspace
# template in this repository. Override with ``workspace_gitignore`` in
# the config when the package is installed outside a full checkout.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GITIGNORE_TEMPLATE = (
    _REPO_ROOT / "infrastructure" / "worker" / "workspace_template" / ".gitignore"
)


def _gitignore_template(cfg: dict) -> Path:
    override = cfg.get("workspace_gitignore")
    return Path(str(override)).expanduser() if override else _GITIGNORE_TEMPLATE


@dataclass
class RunResult:
    run_id: str
    paper_id: str
    model: str
    mode: str
    status: str                # "completed" | "incomplete" | "failed"
    attempt: int
    env_dir: str
    train_score: dict | None
    test_score: dict | None
    error: str | None


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _run_id(paper_id: str, model: str, mode: str, variant: str = "") -> str:
    h = hashlib.sha1(f"{paper_id}|{model}|{mode}{variant}".encode()).hexdigest()[:8]
    return f"{paper_id}-{model}-{mode}{variant}-{h}"


def _project_id(paper_id: str, model: str, mode: str, variant: str = "") -> str:
    # Nexus project IDs are short + stable per (paper, model, mode, variant).
    h = hashlib.sha1(f"{paper_id}|{model}|{mode}{variant}|eval".encode()).hexdigest()[:10]
    return f"eval-{h}"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _is_preexisting_completed(env_dir: Path) -> bool:
    """True if env_dir already contains a completed run with both scores.

    Requires meta ``status == "completed"`` and that both
    ``scoring/scores_train.json`` and ``scoring/scores_test.json``
    validate against the paper's ``source/scoring/scores_{train,test}.json``.
    The test file is written by the grader, so this holds for graded
    runs; a completed run that has not been graded yet is protected by
    :func:`_has_substantial_work` instead.

    Used by ``run_one`` as a safeguard against accidental re-queue: such
    a run is returned as is instead of being wiped and re-scaffolded.
    """
    meta_path = env_dir / ".evaluation.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("status") != "completed":
        return False
    # Lazy import to avoid a cycle at module load time.
    from evaluation import validate as _validate
    train_v = _validate.validate_train(env_dir)
    test_v = _validate.validate_test(env_dir)
    return bool(train_v.ok and test_v.ok)


def _has_substantial_work(env_dir: Path) -> bool:
    """True if env_dir contains substantive in-flight work that should
    not be wiped silently on re-queue.

    Counts as substantial:
      - at least 1 GPU-hour accumulated in meta.gpu_hours_used
      - any file in scoring/ (scores_train.json, scores_test.json, ...)
      - any non-empty turn_NNNN_raw.jsonl trajectory file (0-byte files
        mean the agent died before producing output)

    Covers the case where the harness process died mid-run: the
    workspace keeps status="running" together with the work done so
    far, and a re-queue that only checked for ``status="completed"``
    would wipe it. Paired with ``_is_preexisting_completed``.
    """
    meta_path = env_dir / ".evaluation.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if float(meta.get("gpu_hours_used") or 0) >= 1.0:
                return True
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            pass
    scoring = env_dir / "scoring"
    if scoring.is_dir():
        if any(p.is_file() for p in scoring.iterdir()):
            return True
    trajectory = env_dir / "trajectory"
    if trajectory.is_dir():
        for p in trajectory.iterdir():
            if (
                p.is_file()
                and p.name.startswith("turn_")
                and p.suffix == ".jsonl"
                and p.stat().st_size > 0
            ):
                return True
    return False


def _record_exit(env_dir: Path | None, kind: str, detail: str) -> None:
    """Stamp a structured exit cause into the run's metadata.

    Distinguishes "an API/agent error killed it midway" from "it
    actually finished but grading/finalization broke" without parsing
    free-text error strings. A hard process kill records nothing —
    resume treats an absent record on a non-terminal run as exactly
    that.
    """
    if env_dir is None:
        return
    try:
        meta = _read_json(env_dir / ".evaluation.json") or {}
        meta["last_exit"] = {
            "kind": kind,
            "detail": detail[:300],
            "turn": int(meta.get("current_turn") or 0),
            "at": _utcnow(),
        }
        _write_meta(env_dir, meta)
    except Exception:
        log.debug("could not record exit cause for %s", env_dir)


def _resumable(env_dir: Path | None) -> bool:
    """True when an interrupted main-phase run can be continued in place.

    Requires: a workspace with run metadata, at least one completed
    turn (so there is a session to continue), a status other than
    ``completed``, and neither a held-out score nor a grader verdict.
    """
    if env_dir is None or not (env_dir / ".evaluation.json").exists():
        return False
    meta = _read_json(env_dir / ".evaluation.json") or {}
    if meta.get("status") == "completed":
        return False
    if int(meta.get("current_turn") or 0) < 1:
        return False
    # Only test-side artifacts seal a session. A (possibly partial)
    # scores_train.json must not block resume: the agent writes it
    # incrementally during the main phase, and in-run validation
    # already reopens sessions with train-gap feedback. What may never
    # be reopened is a run whose held-out score or grader verdict
    # exists.
    if (env_dir / "scoring" / "scores_test.json").exists():
        return False
    if _has_grader_verdict(env_dir):
        return False
    return True


# A verification.json only seals a run once the grader actually reached a
# verdict. "in_progress" is a marker a live grader session holds; when
# that session dies (quota, auth, container build) the marker outlives it
# and, read as a verdict, would block both resume and re-grading.
# "incomplete" (verifier.RETRYABLE_STATUS) is retryable by definition.
# Neither is a verdict.
_NON_VERDICT_STATUSES = frozenset({"in_progress", "incomplete"})


def _has_grader_verdict(env_dir: Path) -> bool:
    """True when scoring/verification.json carries a real grader verdict."""
    ver = _read_json(env_dir / "scoring" / "verification.json")
    if not ver:
        return False
    return ver.get("status") not in _NON_VERDICT_STATUSES


def _scoring_touched(env_dir: Path) -> bool:
    """True if the agent already produced anything under ``scoring/``.

    Partial scores mean the agent's state should not be thrown away: the
    run is not retried and is left for review.
    """
    scoring = env_dir / "scoring"
    if not scoring.is_dir():
        return False
    # The grader writes its own files here (verification.json, its scan).
    # Those are not the agent's output and must not block a retry.
    return any(
        p for p in scoring.iterdir()
        if p.is_file() and p.name not in _GRADER_SCORING_FILES
    )


_GRADER_SCORING_FILES = frozenset({"verification.json", "verifier_scan.json"})


def _write_meta(env_dir: Path, meta: dict) -> None:
    (env_dir / ".evaluation.json").write_text(json.dumps(meta, indent=2, default=str))


# Finer-grained state for the dashboard. The coarse ``status`` field
# only moves running → completed | failed | incomplete at finalisation;
# this is the live signal that tells the dashboard what the run is
# doing *now*.
#   - working:    agent turn in flight (thinking, editing, writing)
#   - waiting:    compute job submitted, waiting for terminal state
#   - validating: post-done scoring-file validation, possibly resuming
#                 the agent to fix gaps
#   - testing:    in-run held-out scoring (only set by the disabled
#                 in-run test phase, see _run_test_phase)
#   - done / failed / incomplete: terminal (matches ``status``)
VALID_STATES = ("working", "waiting", "validating", "testing",
                "done", "failed", "incomplete")


def update_state(env_dir: Path, state: str, **extra) -> None:
    """Merge a state update into ``<env_dir>/.evaluation.json``.

    Non-destructive — only the listed fields are changed; everything
    else is preserved.
    """
    if state not in VALID_STATES:
        raise ValueError(f"unknown state {state!r}; valid: {VALID_STATES}")
    meta_path = env_dir / ".evaluation.json"
    meta = _read_json(meta_path) or {}
    meta["state"] = state
    meta["state_updated_at"] = _utcnow()
    meta.update(extra)
    _write_meta(env_dir, meta)


# ── Workspace whitelist ─────────────────────────────────────────────────────
#
# What the agent's workspace contains after scaffolding. sciworld packages
# the workspace; the rules below are enforced on top of it:
#
# - ``checkpoints/`` is never present.
# - ``method/`` always exists as a folder; populated from source only in
#   ``improve`` mode, empty in ``innovate`` mode.
# - Under ``scripts/`` only ``baseline.sh``, ``download.sh`` and
#   ``evaluate_train.sh`` come from the paper repository; ``run.sh`` is
#   replaced by a generic stub.
# - ``eval/test/`` and ``scripts/evaluate_test.sh`` are never present
#   during the agent's run. They live only in ``source/`` until the
#   grader (or the disabled in-run test phase) restores them.
# - ``scoring/reference.json`` is never present: the scoring contract
#   stays in ``source/`` and is read by host-side code only.
# - ``scoring/scores*.json`` are never present: the directory is empty at
#   scaffold time; the agent writes its own scores when its method runs.
#
# Anything in the workspace that violates these rules is removed.

# Files in scripts/ that may stay. Anything else under scripts/ is deleted.
_SCRIPTS_ALLOWED = {
    "baseline.sh",
    "download.sh",
    "evaluate_train.sh",
}

# Subdirectories of eval/ that may stay. eval/test/ must never be kept.
_EVAL_ALLOWED = {"train"}

# Never keepable under eval/, whatever imports claim to need it.
_EVAL_NEVER = {"test"}

# Seconds to wait before each agent retry, indexed by the attempt that
# just failed. Startup failures land in seconds; without a pause the
# retry budget evaporates before the transient condition clears.
_ATTEMPT_BACKOFFS = (60, 180, 420)

# Subpaths anywhere in the workspace that must be removed. Top-level dirs.
_WORKSPACE_FORBIDDEN = (
    "checkpoints",
    "scoring/reference.json",
    "scoring/scores.json",
    "scoring/scores_train.json",
    "scoring/scores_test.json",
)


# Stub ``scripts/run.sh`` placed in env_dir after the whitelist clean.
# Generic: it does not come from the paper repository. It tells the agent
# how a submitted job calls into this script and what it must produce.
# The agent edits or replaces it freely. Agent-facing text.
_EXAMPLE_RUN_SH = """\
#!/bin/bash
# Compute entrypoint for ``action: submit``.
#
# Every time you signal "submit" in action.yaml, this file is what runs
# inside the compute container as ``bash /home/user/scripts/run.sh``.
# Replace its body with whatever your method needs to do.
#
# Workspace layout inside the container (cwd is /home/user):
#   data/         datasets                                         (read-only)
#   baseline/     reference baseline outputs                       (read-only)
#   eval/train/   visible-train evaluator code                     (read-only)
#   READMEs/      task briefing                                    (read-only)
#   method/       your method code                                 (writable)
#   scoring/      where you write scores_train.json                (writable)
#   tmp/          scratch files (created by you)                   (writable)
#
# The harness reads ``scoring/scores_train.json`` after this script
# finishes. Format must match what the train evaluator expects (see
# eval/train/).
#
# A typical skeleton:

set -euo pipefail
cd /home/user

# 1. Train / run your method, write whatever intermediate artifacts you need.
# python -u method/train.py --out tmp/predictions.json

# 2. Score yourself with the visible-train evaluator and write scores_train.json.
# python -u eval/train/evaluate.py --predictions tmp/predictions.json \\
#     --output scoring/scores_train.json

echo "[run.sh] replace this stub with your method's compute command"
exit 1   # ← remove once you've put your real command above
"""


def _rmtree(path: Path) -> None:
    """Best-effort remove. Logs on failure but never raises."""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
    except OSError as exc:
        log.warning("could not remove %s: %s", path, exc)


def _eval_keep_names(eval_dir: Path) -> set[str]:
    """Entries under eval/ that must survive the scaffold whitelist.

    Keeping only ``eval/train/`` is too aggressive. Many environments
    factor their scoring machinery into a module that sits *beside*
    eval/train/ — eval/split_common.py, eval/metrics.py, eval/src/ — and
    eval/train/evaluate.py puts eval/ on sys.path and imports it. Delete
    the sibling and the agent's own train evaluator dies on
    ``ModuleNotFoundError`` before it scores anything. The agent then
    either cannot measure itself at all or rewrites the scorer by hand,
    which is how held-out labels get reinvented.

    So keep eval/train/ plus the transitive closure of the local modules
    it imports. eval/test/ is the leak the whitelist exists to prevent
    and is excluded unconditionally, closure or not.
    """
    keep = {n for n in _EVAL_ALLOWED if (eval_dir / n).exists()}
    entries = {p.name: p for p in eval_dir.iterdir()}

    # eval/__init__.py makes eval/ importable as a package, which the
    # ``from eval.metrics import ...`` spelling needs.
    if "__init__.py" in entries:
        keep.add("__init__.py")

    def _candidates(py: Path) -> set[str]:
        """Top-level eval/ names this file imports."""
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            return set()
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    names.add(parts[0])
                    # ``import eval.metrics`` → eval/metrics
                    if parts[0] == eval_dir.name and len(parts) > 1:
                        names.add(parts[1])
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    # ``from . import x`` inside eval/train/ → eval/x
                    for alias in node.names:
                        names.add(alias.name.split(".")[0])
                    if node.module:
                        names.add(node.module.split(".")[0])
                elif node.module:
                    parts = node.module.split(".")
                    names.add(parts[0])
                    # ``from eval.metrics import ...`` → eval/metrics
                    if parts[0] == eval_dir.name and len(parts) > 1:
                        names.add(parts[1])
                    elif parts[0] == eval_dir.name:
                        # ``from eval import metrics`` → eval/metrics
                        for alias in node.names:
                            names.add(alias.name.split(".")[0])
        return names

    def _py_files(target: Path) -> list[Path]:
        if target.is_dir():
            return sorted(target.rglob("*.py"))
        return [target] if target.suffix == ".py" else []

    queue: list[Path] = []
    for name in sorted(keep):
        queue.extend(_py_files(entries[name]))

    seen: set[Path] = set()
    while queue:
        py = queue.pop()
        if py in seen:
            continue
        seen.add(py)
        for name in _candidates(py):
            if name in _EVAL_NEVER or name in keep:
                continue
            target = entries.get(name) or entries.get(f"{name}.py")
            if target is None:
                continue  # stdlib, third-party, or absent — not ours
            keep.add(target.name)
            queue.extend(_py_files(target))

    return keep


def _enforce_workspace_whitelist(env_dir: Path, mode: str) -> None:
    """Strip any files sciworld leaked into env_dir that the agent must not see.

    Idempotent. Safe to call on a freshly-scaffolded workspace before any
    other Evaluation-side setup runs. After this returns, the workspace
    matches the contract described in the module-level table.
    """
    removed: list[str] = []

    # Forbidden top-level paths.
    for rel in _WORKSPACE_FORBIDDEN:
        p = env_dir / rel
        if p.exists():
            _rmtree(p)
            removed.append(rel)

    # scripts/: keep only the whitelisted entry points.
    scripts_dir = env_dir / "scripts"
    if scripts_dir.is_dir():
        for entry in scripts_dir.iterdir():
            if entry.name in _SCRIPTS_ALLOWED:
                continue
            _rmtree(entry)
            removed.append(f"scripts/{entry.name}")

    # eval/: keep train/ and whatever it imports from beside itself.
    # eval/test/ in particular must never be there at scaffold time —
    # it's materialized only at the test phase.
    eval_dir = env_dir / "eval"
    if eval_dir.is_dir():
        eval_keep = _eval_keep_names(eval_dir)
        for entry in eval_dir.iterdir():
            if entry.name in eval_keep:
                continue
            _rmtree(entry)
            removed.append(f"eval/{entry.name}")

    # scoring/ must always exist (agent writes here). It must be empty
    # at scaffold time — no reference.json, no scores*.json, nothing.
    scoring_dir = env_dir / "scoring"
    scoring_dir.mkdir(exist_ok=True)
    for entry in scoring_dir.iterdir():
        _rmtree(entry)
        removed.append(f"scoring/{entry.name}")

    # method/ must always exist as a folder, and be empty in innovate mode.
    method_dir = env_dir / "method"
    if mode == "innovate":
        if method_dir.is_dir():
            for entry in method_dir.iterdir():
                _rmtree(entry)
                removed.append(f"method/{entry.name}")
        method_dir.mkdir(exist_ok=True)
    else:
        # improve: folder must exist; contents stay.
        method_dir.mkdir(exist_ok=True)

    # paper/ is dropped in both modes. In innovate the agent solves the
    # problem from scratch given only the briefing. In improve the
    # comparison anchor is the existing code in method/, not the paper —
    # showing the paper invites the agent to chase paper-claimed numbers
    # that may not match what the starting code actually produces.
    paper_dir = env_dir / "paper"
    if paper_dir.exists():
        _rmtree(paper_dir)
        removed.append("paper")

    # Drop in our own example ``scripts/run.sh``. The whitelist above
    # already removed any leaked one from the paper repo, so this is a
    # clean replacement. It tells the agent how the submit-job pipeline
    # invokes their compute and gives a skeleton they can edit.
    scripts_dir.mkdir(exist_ok=True)
    run_sh = scripts_dir / "run.sh"
    run_sh.write_text(_EXAMPLE_RUN_SH)
    run_sh.chmod(0o755)

    if removed:
        log.info(
            "Workspace whitelist enforced: removed %d path(s) from %s: %s",
            len(removed), env_dir.name,
            ", ".join(removed[:10]) + ("…" if len(removed) > 10 else ""),
        )


def _seed_train_reference(env_dir: Path) -> None:
    """Copy source's main-phase scores into READMEs/train_reference.json.

    Improve mode only. The agent inherits the starting code in ``method/``
    and gets graded on whether their final ``scores_train.json`` beats
    the numbers the *starting* code produced. This file makes that anchor
    explicit so the agent doesn't have to reverse-engineer it by running
    the unmodified code first.

    Filters to ``type: "proposed"`` methods (or methods absent from
    ``reference.json``, which default to required). Baseline methods are
    out of scope and dropped — including their numbers would just be
    visual noise.

    Best-effort: silently skips if either source file is missing. The
    pipeline still works without the reference file; this is
    informational, not a scoring input.
    """
    source_dir = env_dir.parent / "source" / "scoring"
    src_scores_path = source_dir / "scores_train.json"
    src_ref_path = source_dir / "reference.json"
    if not src_scores_path.exists():
        log.info("No source scores_train.json — skipping train_reference seed")
        return
    try:
        src_scores = json.loads(src_scores_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read source scores_train.json: %s", exc)
        return
    reference: dict | None = None
    if src_ref_path.exists():
        try:
            reference = json.loads(src_ref_path.read_text())
        except (OSError, json.JSONDecodeError):
            reference = None

    def _is_baseline(exp_name: str, method: str) -> bool:
        if not isinstance(reference, dict):
            return False
        exps = reference.get("experiments") or {}
        exp = exps.get(exp_name) if isinstance(exps, dict) else None
        if not isinstance(exp, dict):
            return False
        results = exp.get("results") or {}
        entry = results.get(method) if isinstance(results, dict) else None
        return isinstance(entry, dict) and entry.get("type") == "baseline"

    # Walk the source scores, dropping baseline methods. Preserve the
    # wrapped vs flat layout source uses (don't reformat).
    src_exps = src_scores.get("experiments") if isinstance(src_scores.get("experiments"), dict) else None
    wrapped = src_exps is not None
    iter_src = src_exps if wrapped else src_scores

    out_exps: dict = {}
    for exp_name, exp in iter_src.items():
        if not isinstance(exp, dict):
            continue
        results = exp.get("results") if isinstance(exp.get("results"), dict) else exp
        kept_methods: dict = {}
        for method, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            if _is_baseline(exp_name, method):
                continue
            kept_methods[method] = metrics
        if not kept_methods:
            continue
        # Preserve original shape: wrapped if exp had "results", flat otherwise.
        if isinstance(exp.get("results"), dict):
            out_exps[exp_name] = {"results": kept_methods}
        else:
            out_exps[exp_name] = kept_methods

    if not out_exps:
        log.info("Source scores_train.json had no proposed methods — skipping seed")
        return

    out = {"experiments": out_exps} if wrapped else out_exps
    readmes = env_dir / "READMEs"
    readmes.mkdir(exist_ok=True)
    out_path = readmes / "train_reference.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Seeded %s with %d experiment(s)", out_path, len(out_exps))


def _scaffold(
    paper_id: str, profile_name: str, profile_scaffold: str, mode: str,
    variant: str = "", injected_idea: str | None = None,
    retry_of_own_attempt: bool = False,
) -> tuple[Path, list[tuple[Path, str]], Path, Path | None]:
    """Wipe + re-scaffold via sciworld. Build paper container + overlay.

    Returns ``(env_dir, extra_binds, container_sif, overlay_path)``.
    ``overlay_path`` is ``None`` when the paper has no ``setup.sh``.

    ``variant`` namespaces the env_dir (and, via callers, run/project IDs)
    so parallel variants of the same (paper, profile, mode) — e.g. the
    idea-implementation ablation running base/SFT/GRPO ideas through the
    same innovate env — get separate workspaces. ``injected_idea``, when
    given, is written to ``READMEs/IDEA.md`` as part of the read-only
    briefing so the agent can re-read the idea it was handed.
    """
    import sciworld  # lazy — only required when actually running

    # sciworld's agent registry only knows the base scaffolds. The
    # interactive ``claude-tmux`` scaffold is just claude-code with a
    # different per-turn driver — its workspace scaffolding is identical —
    # so present it to sciworld as ``claude-code``.
    sciworld_agent = "claude-code" if profile_scaffold == "claude-tmux" else profile_scaffold

    cfg = _config.load()
    data_dir = _config.data_dir()
    paper_root = data_dir / "envs" / paper_id
    # env_dir is keyed by profile name (not scaffold) so profiles that
    # share a scaffold get separate workspaces.
    env_dir = paper_root / f"{profile_name}-{mode}{variant}-eval"
    if env_dir.exists():
        # run_one already returns early for a completed run with both
        # scores. If a caller gets here with one anyway, refuse to wipe
        # rather than destroy scoring data.
        if _is_preexisting_completed(env_dir):
            raise RuntimeError(
                f"refusing to wipe completed env_dir {env_dir} "
                "(scores_train.json + scores_test.json present)"
            )
        # Also refuse when there is substantial work on disk even though
        # the status is not "completed" (a run whose harness process
        # died keeps status="running" with GPU hours, scoring files and
        # trajectory turns). The caller should treat this as a signal
        # that the workspace needs review.
        #
        # Exception: run_one's own retry loop. It re-scaffolds after a
        # TransientError, having already refused to retry when scoring/
        # was touched. The work on disk is then its own just-failed
        # attempt (typically a single trajectory file); applying the
        # check there would make any run whose first turn failed
        # permanently unretryable.
        if _has_substantial_work(env_dir) and not retry_of_own_attempt:
            raise RuntimeError(
                f"refusing to wipe env_dir {env_dir} with substantial "
                "in-flight work (gpu_hours, scoring/, or trajectory/ "
                "files on disk). The harness may have been killed mid-"
                "run; review and clean up manually before re-queueing."
            )
        log.info("Wiping existing workspace %s", env_dir)
        shutil.rmtree(env_dir)
    # Reuse the sibling source/ clone across all runs for this paper.
    # sciworld detects an existing source/ and skips the clone, which
    # avoids clone rate limits when many profiles are benchmarked on the
    # same paper. The first scaffold for a paper clones; later ones reuse.
    paper_root.mkdir(parents=True, exist_ok=True)

    # sciworld calls the GitHub REST API to locate the paper repository.
    # Many concurrent scaffolds can hit rate limits, which surface as
    # ``sciworld.research exited with code 1``; retry with backoff.
    #
    # The per-paper lock is held around the whole retry loop so only one
    # run scaffolds a given paper at a time. By the time the next run
    # gets the lock, ``source/`` exists and sciworld skips the clone.
    import time as _time
    _SCI_BACKOFFS = (0, 30, 90, 240)  # 4 attempts total
    result = None
    last_exc: Exception | None = None
    with _paper_scaffold_lock(paper_id):
        for attempt, delay in enumerate(_SCI_BACKOFFS):
            if delay:
                log.warning(
                    "sciworld.research retry %d/%d for %s after %ds (last: %s)",
                    attempt, len(_SCI_BACKOFFS) - 1, paper_id, delay, last_exc,
                )
                _time.sleep(delay)
            try:
                result = sciworld.research(
                    id=paper_id,
                    agent=sciworld_agent,
                    mode=mode,
                    path=str(env_dir),
                    setup_only=True,
                    # Agents and jobs run in containers here: keep the
                    # container paths in the workspace, whatever backend
                    # the CLI itself is configured for.
                    env="container",
                )
                break
            except RuntimeError as exc:
                last_exc = exc
                # Between attempts the partially-scaffolded env_dir may
                # be in a bad state; wipe just env_dir, NOT source/ —
                # source/ is shared across all model runs for this
                # paper, and wiping it would force every concurrent
                # sibling to re-clone from GitHub.
                if env_dir.exists():
                    shutil.rmtree(env_dir, ignore_errors=True)
    if result is None:
        raise RuntimeError(
            f"sciworld.research failed after {len(_SCI_BACKOFFS)} attempts "
            f"for {paper_id}: {last_exc!r}"
        )
    scaffolded = Path(result["env_dir"]).resolve()
    if scaffolded != env_dir.resolve():
        raise RuntimeError(
            f"sciworld scaffolded at {scaffolded}, expected {env_dir}"
        )

    # Enforce the workspace whitelist. sciworld packages the same contract;
    # checking it again here means an evaluated agent can never see the
    # held-out evaluator or the reference scores (eval/test/,
    # scripts/evaluate_test.sh, scoring/reference.json,
    # scoring/scores*.json, ...) whatever version of the CLI is installed.
    _enforce_workspace_whitelist(env_dir, mode)

    # Replace sciworld's flat ``idea.md`` with an ``idea/`` folder so the
    # agent can version its thinking across iterations (idea_v1.md,
    # idea_v2.md, …).
    stock_idea = env_dir / "idea.md"
    idea_dir = env_dir / "idea"
    if stock_idea.exists() and not stock_idea.is_dir():
        stock_idea.unlink()
    idea_dir.mkdir(exist_ok=True)

    # In improve mode, expose source's main-phase scores (filtered to
    # ``type: "proposed"`` methods) at READMEs/train_reference.json. This
    # tells the agent what the starting code in method/ actually produced
    # — the bar to beat. Lives under READMEs/ which is already RO
    # bind-mounted by sciworld's launcher.
    if mode == "improve":
        _seed_train_reference(env_dir)

    # Seed from the paper cache: copy container.sif, overlay.img, and
    # pre-bind downloads. No-op when the cache is empty — the next
    # steps then build container + overlay from scratch and (on the
    # warm path) populate the cache for future runs.
    extra_binds = cache.apply_to_workspace(paper_id, env_dir)

    # Paper's compute container. This is what Nexus registers as
    # sif_path — NOT the agent container. Agent container stays on
    # sciworld's agent.sif and talks to jobs through the workspace bind.
    container_sif = _container.ensure_container(
        env_dir,
        container_cache_dir=str(cfg.get("container_cache_dir") or ""),
    )
    overlay_path = _container.ensure_overlay(
        env_dir, container_sif,
        container_cache_dir=str(cfg.get("container_cache_dir") or ""),
        local_tmp_dir=str(cfg.get("local_tmp_dir") or "/tmp"),
        uv_cache_dir=str(cfg.get("uv_cache_dir") or ""),
    )
    # Idea-implementation ablation: hand the agent a fixed idea as part of
    # the read-only briefing. The prompt also embeds it, but a stable
    # READMEs/IDEA.md lets the agent re-read it across turns.
    if injected_idea:
        readmes = env_dir / "READMEs"
        readmes.mkdir(parents=True, exist_ok=True)
        (readmes / "IDEA.md").write_text(
            "# Assigned Idea\n\n"
            "You must implement the following idea. Do not replace it with a "
            "different approach.\n\n" + injected_idea.strip() + "\n"
        )

    # Snapshot initial hashes so rebuild_if_changed doesn't fire on turn 1.
    _container.snapshot_hashes(env_dir)

    return env_dir, extra_binds, container_sif, overlay_path


def _initial_prompt(
    env_dir: Path, mode: str, *, gpu_budget: float, max_gpus: int,
    injected_idea: str | None = None,
) -> str:
    """Load the per-mode starting prompt from ``prompts/initial_<mode>.md``.

    Every kwarg here is substituted into the rendered prompt via
    :func:`evaluation.loop._render`. Prompt text lives in markdown
    files, never inlined in Python.

    When ``injected_idea`` is given, the idea-implementation prompt
    (``initial_implement.md``) is rendered instead of ``initial_<mode>``,
    with the idea substituted for ``{idea}``. Used by the standalone
    idea-implementation ablation; the main eval passes ``None`` and is
    unaffected.
    """
    from evaluation.loop import _render  # centralised renderer
    if injected_idea:
        return _render(
            "initial_implement", gpu_budget=gpu_budget, max_gpus=max_gpus,
            idea=injected_idea.strip(),
        )
    return _render(
        f"initial_{mode}", gpu_budget=gpu_budget, max_gpus=max_gpus,
    )


# In-run held-out test phase (``_run_test_phase``): fix-turn budget.
# The phase is disabled (see the note on ``_run_test_phase``); these
# constants only matter if it is wired back in. First
# ``_TEST_PHASE_MAX_FIX_TURNS`` turns with ``method/`` locked, then
# ``_TEST_PHASE_UNLOCK_FIX_TURNS`` more with ``method/`` writable so the
# agent can rename / adapt files to the held-out scorer's interface
# (rules in ``prompts/on_test_phase_unlock.md``).
_TEST_PHASE_MAX_FIX_TURNS = 5
_TEST_PHASE_UNLOCK_FIX_TURNS = 3
# Train-phase scoring-file validation retries. After the agent says
# ``done`` the host-side validator (validate.py) checks that its
# ``scores_train.json`` covers every (method, metric) the source reports
# for methods that ``reference.json`` does not tag as ``"baseline"``. On
# gaps the agent is invoked again with ``on_validation_failed.md``, up
# to this many rounds; if gaps remain the run is marked ``incomplete``
# (distinct from completed and failed).
_TRAIN_VALIDATE_MAX_RETRIES = 5
# Turn cap of one validation round. The agent can submit jobs inside a
# round (the full loop is active).
_TRAIN_VALIDATE_MAX_TURNS = 30


def _validate_with_retries(
    env_dir: Path,
    profile,
    nexus: Nexus,
    project_id: str,
    *,
    extra_binds,
    cfg: dict,
    turn_offset: int,
) -> bool:
    """Validate train-phase output, invoking the agent up to N times to fix gaps.

    Returns True if validation passes (with or without retries). Returns
    False if all retries are exhausted and gaps remain, in which case
    the run is finalised as ``incomplete``.
    """
    from evaluation import validate as _validate
    from evaluation.loop import _render

    cumulative_turns = turn_offset
    for retry_n in range(1, _TRAIN_VALIDATE_MAX_RETRIES + 1):
        result = _validate.validate_train(env_dir)
        if result.ok:
            if retry_n > 1:
                log.info(
                    "Train-phase validation passed after %d retry round(s).",
                    retry_n - 1,
                )
            return True
        log.warning(
            "Train-phase validation retry %d/%d: %s",
            retry_n, _TRAIN_VALIDATE_MAX_RETRIES, result.summary(),
        )
        update_state(env_dir, "validating",
                     validation_retry=retry_n,
                     validation_missing=result.n_missing)
        prompt = _render(
            "on_validation_failed",
            phase=result.phase,
            retry_n=retry_n,
            max_retries=_TRAIN_VALIDATE_MAX_RETRIES,
            gaps=result.format_for_prompt(),
        )
        try:
            sub_result = run_loop(
                env_dir, profile, nexus, project_id, prompt,
                max_turns=_TRAIN_VALIDATE_MAX_TURNS,
                gpu_budget_hours=float(cfg["nexus"]["gpu_budget_hours"]),
                extra_binds=extra_binds,
                turn_offset=cumulative_turns,
            )
            # ``turns`` is the last turn number, offset included.
            cumulative_turns = sub_result.turns
        except TransientError as exc:
            log.warning(
                "Train-phase validation retry %d crashed: %s — giving up", retry_n, exc,
            )
            break
    # Final check after the loop in case the last retry just succeeded.
    final = _validate.validate_train(env_dir)
    if final.ok:
        return True
    log.warning(
        "Train-phase validation failed after %d retries: %s",
        _TRAIN_VALIDATE_MAX_RETRIES, final.summary(),
    )
    return False


def _run_test_phase(
    env_dir: Path,
    profile,  # Profile
    nexus: Nexus,
    project_id: str,
    turn_offset: int,
    extra_binds: list[tuple[Path, str]] | None = None,
) -> None:
    """In-run held-out scoring phase with bounded agent-driven recovery.

    Disabled: :func:`run_one` does not call this function and
    :func:`resume_test_phase` only finalizes. The agent must never see
    anything derived from the test split, and the fix turns below feed
    test-job output back to it, so the held-out score is produced by the
    independent grader (:mod:`evaluation.verifier`) after the run
    instead. The implementation is kept because the paper describes this
    phase. When it was wired in, a run counted as
    ``completed`` only if both score files validated and the final test
    job had exited 0 (``last_test_rc`` in the run metadata).

    Submit ``scripts/evaluate_test.sh`` as a Nexus job. After every
    submission, validate the result with :func:`validate.validate_test`.
    The job is "done" only when validation passes — i.e. every
    ``proposed`` (experiment, method, metric) cell present in source's
    ``scores_test.json`` is also present in the agent's. A test job that
    silently dropped cells, or never produced the file at all, both
    surface as fixable failures and trigger an agent fix turn.

    Two failure shapes get different prompts:
      - file missing or unreadable → ``on_test_phase.md`` (infra fix)
      - file present, cells missing → ``on_validation_failed.md`` phase=test

    The agent's ``action: done`` is ignored — we keep retrying until
    validation passes or the fix-turn budget is exhausted. The container
    binds ``method/``, ``eval/test/``, and ``scripts/evaluate_test.sh``
    read-only, so the agent can only fix infrastructure (missing
    packages, broken env) without touching the frozen method or peeking
    at held-out data.

    ``turn_offset`` = number of turns already used in the main phase,
    so trajectory files don't collide.
    """
    from evaluation import agent as _agent  # local to dodge import cycle
    from evaluation import git as _git
    from evaluation import validate as _validate
    from evaluation.loop import _render, _save_job_artifacts
    from evaluation.parsing import parse_turn_action

    test_script = env_dir / "scripts" / "evaluate_test.sh"
    test_scores = env_dir / "scoring" / "scores_test.json"

    # Materialize the held-out test setup from source/. Always overwrite
    # — never trust whatever is in env_dir. The agent must not be able to
    # influence what gets executed at test phase, so the test script and
    # eval/test/ tree always come from the paper's pristine clone, even
    # if env_dir already has a (potentially modified) version.
    source = env_dir.parent / "source"
    src_test_script = source / "scripts" / "evaluate_test.sh"
    src_eval = source / "eval"

    if src_test_script.exists():
        test_script.parent.mkdir(parents=True, exist_ok=True)
        if test_script.exists():
            test_script.unlink()
        shutil.copy2(src_test_script, test_script)
        test_script.chmod(0o755)
        log.info("Restored evaluate_test.sh from source/ (always overwrite)")

    # Restore the full source/eval/ tree, not just eval/test/. The
    # held-out evaluator imports siblings (e.g. ``from eval.metrics
    # import ...``) that live alongside eval/test/ rather than inside
    # it; without restoring those siblings, the import fails at test
    # time. eval/train/ was bind-mounted read-only during the main
    # phase, so overwriting it here is a no-op for content but ensures
    # a known-clean state for the test job.
    dst_eval = env_dir / "eval"
    if src_eval.exists():
        if dst_eval.exists():
            shutil.rmtree(dst_eval)
        shutil.copytree(src_eval, dst_eval, symlinks=False)
        log.info("Restored eval/ from source/ (always overwrite)")

    if not test_script.exists():
        log.info("No evaluate_test.sh in env_dir or source; skipping held-out scoring")
        return

    # Test phase gets a fresh budget — the agent may have legitimately
    # used the full main-phase quota during exploration, but the
    # held-out evaluation is a separate accounting bucket. Reset the
    # project's accumulated GPU-time so the test job's submit isn't
    # rejected by Nexus's soft-budget check.
    try:
        nexus.update_project(project_id, gpu_seconds_used=0.0)
    except Exception as exc:
        log.warning("Test-phase budget reset failed for %s: %s", project_id, exc)

    # Mirror the GPU count of the agent's last action: submit in the
    # main phase. evaluate_test.sh frequently re-uses the same launcher
    # (e.g. torchrun --nproc_per_node=N) the agent settled on; hardcoding
    # 1 GPU here breaks every multi-GPU paper. `test` stays False —
    # held-out scoring always lands on prod partitions.
    last_gpus = 1
    try:
        with open(env_dir / ".evaluation.json") as _f:
            _meta = json.load(_f)
        v = _meta.get("last_submit_gpus")
        if isinstance(v, int) and v > 0:
            last_gpus = v
    except (OSError, json.JSONDecodeError):
        pass
    log.info("Test phase: using last_submit_gpus=%d from main phase", last_gpus)

    total_budget = _TEST_PHASE_MAX_FIX_TURNS + _TEST_PHASE_UNLOCK_FIX_TURNS
    fix_turn = 0
    # Consecutive identical-rc failures while the file validates —
    # signature of a deterministically-broken test script.
    prev_ok_fail_rc: int | None = None
    ok_fail_streak = 0
    while True:
        unlock_method = fix_turn > _TEST_PHASE_MAX_FIX_TURNS
        update_state(
            env_dir, "testing",
            test_fix_turn=fix_turn,
            test_unlock_method=unlock_method,
        )
        log.info(
            "Submitting held-out evaluation job (attempt %d/%d, method=%s)",
            fix_turn + 1, total_budget + 1,
            "unlocked" if unlock_method else "locked",
        )
        try:
            run_id, _ = nexus.submit_job(
                project_id, "bash /home/user/scripts/evaluate_test.sh",
                num_gpus=last_gpus,
                resource=_config.nexus_pick(gpus=last_gpus, test=False),
            )
        except Exception as exc:
            log.warning("Test-phase submit failed: %s", exc)
            return

        # Same checkpoint handling as the main loop: when the job runs
        # longer than Nexus's checkpoint_interval, Nexus pauses it and
        # returns awaiting_response. Without this branch the poll loop
        # would never answer and the job would end at Nexus's
        # checkpoint_timeout, wasting a fix turn.
        ckpt_count = 0
        ckpt_t0 = time.monotonic()
        while True:
            event = nexus.wait_for_job(project_id, run_id, timeout=300)
            ev = event.get("event") or event.get("status")
            if ev in ("completed", "cancelled", "failed", "error", "done"):
                break
            if ev == "awaiting_response":
                ckpt_count += 1
                log.info(
                    "Test phase: job %s checkpoint #%d — agent decision required",
                    run_id, ckpt_count,
                )
                stdout_path, stderr_path, _rc = _save_job_artifacts(
                    env_dir, run_id, nexus,
                )
                elapsed_min = int((time.monotonic() - ckpt_t0) / 60)
                # Test-phase context preamble + standard checkpoint
                # body. The agent's resumed session has no memory of
                # entering test phase (the pipeline submits the scoring
                # job silently, without invoking the agent), so without
                # the preamble it would read on_checkpoint.md's "a job
                # you submitted has been running..." and not recognise
                # the run_id. The preamble explains who submitted the
                # job and why, then defers to the standard prompt's
                # continue/cancel decision criteria.
                ckpt_prompt = _render(
                    "on_test_phase_checkpoint",
                ) + _render(
                    "on_checkpoint",
                    run_id=run_id,
                    elapsed_min=elapsed_min,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                rc_ag, _so, _se = _agent.invoke(
                    env_dir, profile, ckpt_prompt,
                    turn=turn_offset + fix_turn,
                    resume=True,
                    nexus_project_id=project_id,
                    extra_binds=extra_binds,
                )
                decision = parse_turn_action(env_dir)
                decision_name = decision.action if decision else "missing"
                if decision_name == "continue":
                    log.info(
                        "Test phase: agent said continue — resuming job %s",
                        run_id,
                    )
                    nexus.continue_job(run_id)
                    continue
                log.info(
                    "Test phase: agent decision=%s at checkpoint — cancelling job %s",
                    decision_name, run_id,
                )
                nexus.cancel_job(run_id)
                continue

        stdout_path, stderr_path, rc = _save_job_artifacts(env_dir, run_id, nexus)
        _git.commit(env_dir, f"eval: test job {run_id} done (rc={rc})")
        # Persist the rc so finalization (and any later consumer) can
        # tell whether the final test job actually succeeded — the
        # backstop for agent-authored scores_test.json.
        update_state(env_dir, "testing", last_test_rc=rc, test_fix_turn=fix_turn)

        # Validate against source's scores_test.json. The file existing
        # is necessary but not sufficient — a test evaluator that
        # silently dropped half the proposed cells would slip through
        # an existence check.
        result = _validate.validate_test(env_dir)
        if result.ok and rc == 0:
            log.info("Held-out scoring validated ok (fix_turn=%d)", fix_turn)
            return
        if result.ok and rc != 0:
            # Cell-presence validation is necessary but not sufficient: a
            # scores_test.json that exists while the producing job FAILED
            # was written by something other than the pristine test job
            # (in practice: the agent, during a fix turn). Never accept
            # it — keep looping until a successful job produces the file.
            log.warning(
                "scores_test.json validates but the test job exited rc=%d — "
                "rejecting: test scores must come from a successful pristine "
                "job, not agent-authored files", rc,
            )
            # The test script is source-restored and bind-mounted
            # read-only, so a deterministically-failing script fails
            # identically forever — stop spending fix turns after three
            # identical consecutive failures. Finalization's
            # last_test_rc backstop keeps the run from completing.
            if rc == prev_ok_fail_rc:
                ok_fail_streak += 1
            else:
                prev_ok_fail_rc, ok_fail_streak = rc, 1
            if ok_fail_streak >= 3:
                log.warning(
                    "Test job failed with identical rc=%d for %d consecutive "
                    "attempts while scores_test.json validates — the test "
                    "script itself is broken; giving up (run will finalize "
                    "incomplete via the last_test_rc backstop)",
                    rc, ok_fail_streak,
                )
                return
        else:
            prev_ok_fail_rc, ok_fail_streak = None, 0

        if fix_turn >= total_budget:
            log.warning(
                "Test phase exhausted %d fix turns (%d locked + %d unlocked): %s",
                total_budget, _TEST_PHASE_MAX_FIX_TURNS,
                _TEST_PHASE_UNLOCK_FIX_TURNS, result.summary(),
            )
            return

        fix_turn += 1
        next_unlock = fix_turn > _TEST_PHASE_MAX_FIX_TURNS
        # Three distinct failure shapes:
        #   (a) file missing/unreadable → infrastructure fix prompt
        #   (b) file present but every proposed cell scored 0 (all-abstain)
        #       → same fix prompt; it covers the all-zero / keying case
        #   (c) file present but cells missing → validation gap prompt
        # A fourth shape — file validates but the producing job failed
        # (rc != 0) — routes to the infrastructure prompt: the JOB is
        # what needs fixing; the file's contents are not trusted.
        if result.file_missing or result.file_invalid or result.degenerate or result.ok:
            base_prompt = _render(
                "on_test_phase",
                run_id=run_id, returncode=rc,
                stdout_path=stdout_path, stderr_path=stderr_path,
            )
            log.info("Invoking agent for test-phase fix turn %d (%s)",
                     fix_turn,
                     "job failed rc!=0" if result.ok
                     else "all-zero" if result.degenerate
                     else "file invalid" if result.file_invalid else "file missing")
        else:
            base_prompt = _render(
                "on_validation_failed",
                phase="test",
                retry_n=fix_turn,
                max_retries=total_budget,
                gaps=result.format_for_prompt(),
            )
            log.info("Invoking agent for test-phase fix turn %d (%d cell(s) missing)",
                     fix_turn, result.n_missing)
        # On the first unlocked fix turn (and every one after), prepend
        # the strict-honesty preamble so the agent is told the rules
        # before they touch method/.
        if next_unlock:
            prompt = _render("on_test_phase_unlock") + "\n\n" + base_prompt
        else:
            prompt = base_prompt
        rc_agent, _stdout, _stderr = _agent.invoke(
            env_dir, profile, prompt,
            turn=turn_offset + fix_turn,
            resume=True,
            nexus_project_id=project_id,
            test_phase=True,  # bind eval/test/, evaluate_test.sh, scores_train.json RO
            test_phase_unlock_method=next_unlock,
            extra_binds=extra_binds,
        )
        if rc_agent != 0:
            log.warning("Test-phase agent turn exited rc=%d", rc_agent)
            return
        # Don't gate on action.yaml. Whether the agent says "submit",
        # "done", or nothing, we keep iterating until validation passes
        # or the fix-turn budget is exhausted. The agent's "done" used
        # to short-circuit the loop, which is exactly how runs slipped
        # past with incomplete or missing scores_test.json.


def _remove_container_artifacts(env_dir: Path) -> None:
    """Delete the run's built container image and overlay, if present.

    Errors are logged and swallowed so cleanup never breaks finalization.
    """
    for artifact in ("environment/overlay.img", "environment/container.sif"):
        try:
            p = env_dir / artifact
            if p.exists():
                p.unlink()
        except OSError as exc:
            log.warning("cleanup of %s failed: %s", artifact, exc)


def _publish_run(
    env_dir: Path, paper_id: str, profile_name: str, mode: str, cfg: dict,
) -> dict[str, object]:
    """Publish a completed run if publishing is enabled; never raises.

    Returns the record stored under ``publish`` in the run metadata:
    ``status`` is ``disabled``, ``pushed``, ``skipped`` (with ``reason``)
    or ``failed`` (with ``error``).
    """
    from evaluation import git_publish

    record: dict[str, object] = {"at": _utcnow()}
    if not git_publish.publish_enabled(cfg):
        record["status"] = "disabled"
        return record
    try:
        res = git_publish.publish(env_dir, paper_id, profile_name, mode)
        record["branch"] = res.branch
        record["repo"] = res.repo_full_name
        if res.pushed:
            record["status"] = "pushed"
            record["files"] = res.files_committed
            record["bytes"] = res.bytes_committed
            log.info("Published %s/%s → %s@%s",
                     paper_id, profile_name, res.repo_full_name, res.branch)
        else:
            record["status"] = "skipped"
            record["reason"] = res.skipped_reason
            log.info("Publish skipped for %s/%s: %s",
                     paper_id, profile_name, res.skipped_reason)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)[:500]
        log.exception("Publish failed for %s/%s — run status unchanged",
                      paper_id, profile_name)
    return record


def run_one(
    paper_id: str, profile_name: str, mode: str = "innovate",
    *, variant: str = "", injected_idea: str | None = None,
    resume: bool = True,
) -> RunResult:
    """Run one trial: a single paper with a single model profile.

    *mode* is ``innovate`` (design a method from the problem statement)
    or ``improve`` (start from the paper's reference code and improve
    it).

    ``resume`` (default True) continues an interrupted run in place
    (session, turns, and GPU accounting carry over) whenever the
    existing workspace has completed turns, a status other than
    ``completed``, and neither a held-out score nor a grader verdict.
    Otherwise (or for a new run) it falls back to wipe + re-scaffold.
    Move the env_dir aside first when a from-scratch rerun is intended.

    ``variant`` namespaces the env_dir / run_id / project_id so a variant
    of the same (paper, profile, mode) does not collide with the main
    run or with sibling variants. ``injected_idea``, when given, hands
    the agent a fixed idea to implement (written to READMEs/IDEA.md and
    embedded in the ``initial_implement`` prompt); the agent works in the
    normal ``innovate`` workspace but implements that idea rather than
    inventing its own. Variant runs are not normalized, published or
    graded. Both default to inert values.
    """
    cfg = _config.load()
    _config.check_run_config(cfg)
    profile = get_profile(profile_name)
    run_id = _run_id(paper_id, profile_name, mode, variant)
    project_id = _project_id(paper_id, profile_name, mode, variant)
    max_retries = int(cfg.get("max_retries", 1))

    # Preserve-completed safeguard. If a completed run with both scores
    # already exists for this (paper, profile, mode), do not scaffold:
    # ``_scaffold`` would delete env_dir. Returning early makes the
    # caller see the run as completed and adds ``preserved: true`` to
    # its metadata so a kept run can be told from a fresh one.
    data_dir = _config.data_dir()
    env_dir_candidate = (
        data_dir / "envs" / paper_id / f"{profile_name}-{mode}{variant}-eval"
    )
    if _is_preexisting_completed(env_dir_candidate):
        log.warning(
            "Preserving completed run for %s × %s × %s — refusing to "
            "wipe and re-scaffold. Marking preserved.",
            paper_id, profile_name, mode,
        )
        existing_meta = _read_json(env_dir_candidate / ".evaluation.json") or {}
        existing_meta["preserved"] = True
        existing_meta["preserved_at"] = _utcnow()
        _write_meta(env_dir_candidate, existing_meta)
        return RunResult(
            run_id=existing_meta.get("run_id") or run_id,
            paper_id=paper_id,
            model=profile_name,
            mode=mode,
            status="completed",
            attempt=int(existing_meta.get("attempt") or 0),
            env_dir=str(env_dir_candidate),
            train_score=_read_json(
                env_dir_candidate / "scoring" / "scores_train.json"
            ),
            test_score=_read_json(
                env_dir_candidate / "scoring" / "scores_test.json"
            ),
            error=None,
        )

    nexus = Nexus(cfg["nexus"]["url"])
    last_error: str | None = None
    env_dir: Path | None = None
    # Resume-in-place: on the first attempt only when the caller asked
    # for it; on retries whenever the interrupted attempt left real
    # progress behind. Falls back to wipe+re-scaffold when the
    # workspace has nothing to continue.
    resume_next = bool(resume) and _resumable(env_dir_candidate)
    if resume and not resume_next:
        # Not loop-resumable. Distinguish "actually finished but grading
        # or finalization broke" (route to the finish-only path, which
        # never relaunches the agent) from "nothing to continue" (fresh
        # start).
        _m = _read_json(env_dir_candidate / ".evaluation.json") or {}
        _method = env_dir_candidate / "method"
        # The train table must be complete, not merely present.
        # resume_test_phase never relaunches the agent, so it cannot fill
        # a missing cell: routing a partial train table there would
        # yield "incomplete" on every attempt. A gap in
        # scores_train.json is the agent's work to finish.
        from evaluation import validate as _validate_train_gate
        _train_ok = (
            (env_dir_candidate / "scoring" / "scores_train.json").exists()
            and _validate_train_gate.validate_train(env_dir_candidate).ok
        )
        if (not variant
                and _m.get("status") != "completed"
                and _train_ok
                and _method.is_dir() and any(_method.iterdir())):
            log.info(
                "%s finished its main phase (scores_train present, method "
                "built) but is not completed — resuming grading/"
                "finalization only, agent will not be relaunched",
                env_dir_candidate,
            )
            return resume_test_phase(paper_id, profile_name, mode)
        log.info("resume requested but %s has nothing to continue — "
                 "starting fresh", env_dir_candidate)

    for attempt in range(1, max_retries + 2):  # 1 original + max_retries
        try:
            if resume_next:
                env_dir = env_dir_candidate
                _prior = _read_json(env_dir / ".evaluation.json") or {}
                _exit = _prior.get("last_exit") or {}
                log.info(
                    "Resuming interrupted run in place: %s (last exit: %s at "
                    "turn %s — %s)", env_dir,
                    _exit.get("kind") or "hard kill, nothing recorded",
                    _exit.get("turn", "?"),
                    (_exit.get("detail") or "")[:120],
                )
                extra_binds = cache.apply_to_workspace(paper_id, env_dir)
                container_sif = _container.ensure_container(
                    env_dir,
                    container_cache_dir=str(cfg.get("container_cache_dir") or ""),
                )
                overlay_path = _container.ensure_overlay(
                    env_dir, container_sif,
                    container_cache_dir=str(cfg.get("container_cache_dir") or ""),
                    local_tmp_dir=str(cfg.get("local_tmp_dir") or "/tmp"),
                    uv_cache_dir=str(cfg.get("uv_cache_dir") or ""),
                )
                meta = _read_json(env_dir / ".evaluation.json") or {}
                # current_turn can be clobbered by validation-loop-local
                # counters; the trajectory directory is the authority.
                # Resuming below the true max would overwrite the
                # agent's turn files.
                start_turn = int(meta.get("current_turn") or 0)
                traj = env_dir / "trajectory"
                if traj.is_dir():
                    for tp in traj.iterdir():
                        if tp.name.startswith("turn_") and tp.name.endswith(".jsonl"):
                            try:
                                start_turn = max(start_turn, int(tp.name.split("_")[1]))
                            except (ValueError, IndexError):
                                pass
                gpu_hours_start = float(meta.get("gpu_hours_used") or 0.0)
                meta.update({
                    "attempt": attempt,
                    "status": "running",
                    "state": "working",
                    "state_updated_at": _utcnow(),
                })
                meta.setdefault("resumed_at", []).append(_utcnow())
                _write_meta(env_dir, meta)
            else:
                env_dir, extra_binds, container_sif, overlay_path = _scaffold(
                    paper_id, profile_name, profile.scaffold, mode,
                    variant=variant, injected_idea=injected_idea,
                    retry_of_own_attempt=attempt > 1,
                )
                start_turn = 0
                gpu_hours_start = 0.0

                git.install_gitignore(env_dir, _gitignore_template(cfg))
                git.init_repo(env_dir)

                meta = {
                    "run_id": run_id,
                    "paper_id": paper_id,
                    "model": profile_name,
                    "mode": mode,
                    "variant": variant,
                    "nexus_project_id": project_id,
                    "attempt": attempt,
                    "started_at": _utcnow(),
                    "status": "running",
                    "state": "working",
                    "state_updated_at": _utcnow(),
                }
                _write_meta(env_dir, meta)

            # Nexus project IDs are deterministic (sha1 of paper|profile|mode).
            # If a prior run was killed mid-flight, Nexus may still hold
            # the old project and create would be rejected, so delete
            # first, then create.
            _create_kwargs = dict(
                project_id=project_id,
                workspace_dir=str(env_dir),
                sif_path=str(container_sif),
                overlay_path=str(overlay_path) if overlay_path else "",
                gpu_budget=float(cfg["nexus"]["gpu_budget_hours"]),
                max_gpus=int(cfg["nexus"]["max_gpus"]),
                allowed_resources=_config.nexus_all_resources(cfg),
                checkpoint_interval=int(cfg.get("checkpoint_interval") or 1200),
                checkpoint_timeout=int(cfg.get("checkpoint_timeout") or 1800),
            )
            nexus.delete_project(project_id)
            import time as _t
            _t.sleep(1)
            try:
                nexus.create_project(**_create_kwargs)
            except RuntimeError:
                log.warning("Nexus create_project conflict for %s — "
                            "retrying delete+create", project_id)
                nexus.delete_project(project_id)
                _t.sleep(1)
                nexus.create_project(**_create_kwargs)

            # A resumed workspace is not always a resumed conversation.
            # When every earlier turn died before the agent answered, there
            # is nothing to continue: "you were interrupted, carry on" would
            # reach an agent that never received its task. Keep the
            # workspace and start the conversation over with the full task.
            from evaluation.agent import has_conversation
            continue_session = resume_next and has_conversation(env_dir, profile.scaffold)
            if resume_next and continue_session:
                git.commit(env_dir, f"eval: resumed in place at turn {start_turn}")
                prompt = _loop_render("on_resume_interrupted")
            elif resume_next:
                git.commit(env_dir, f"eval: resumed workspace at turn {start_turn}, "
                                    f"no conversation to continue — sending the task afresh")
                log.warning("%s: no agent conversation to continue — sending "
                            "the full task prompt over the existing workspace",
                            env_dir)
                prompt = _initial_prompt(
                    env_dir, mode,
                    gpu_budget=float(cfg["nexus"]["gpu_budget_hours"]),
                    max_gpus=int(cfg["nexus"]["max_gpus"]),
                    injected_idea=injected_idea,
                )
            else:
                git.commit(env_dir, "eval: initial scaffold")
                prompt = _initial_prompt(
                    env_dir, mode,
                    gpu_budget=float(cfg["nexus"]["gpu_budget_hours"]),
                    max_gpus=int(cfg["nexus"]["max_gpus"]),
                    injected_idea=injected_idea,
                )

            try:
                loop_result = run_loop(
                    env_dir, profile, nexus, project_id,
                    prompt,
                    max_turns=int(cfg.get("max_turns", 200)),
                    gpu_budget_hours=float(cfg["nexus"]["gpu_budget_hours"]),
                    extra_binds=extra_binds,
                    turn_offset=start_turn,
                    resume_session=continue_session,
                    gpu_hours_start=gpu_hours_start,
                )
                # Validate the agent's main-phase scores against source's
                # ``scores_train.json`` (baseline methods excluded). Up to
                # _TRAIN_VALIDATE_MAX_RETRIES rounds of "tell the agent
                # what is missing, let it fix it". If still incomplete
                # after that, the run is marked ``incomplete`` (distinct
                # from completed and failed).
                validate_ok = _validate_with_retries(
                    env_dir, profile, nexus, project_id,
                    extra_binds=extra_binds,
                    cfg=cfg,
                    turn_offset=loop_result.turns,
                )
                # No held-out scoring happens inside the agent's run: the
                # in-run test phase (_run_test_phase) is disabled, and
                # the grader produces the held-out score after the run.
                if not validate_ok:
                    log.warning(
                        "Train-phase validation exhausted retries. "
                        "Run will be marked incomplete."
                    )
                break  # success — skip retry
            except FatalAgentError as exc:
                last_error = f"fatal: {exc}"
                _record_exit(env_dir, "provider_safety_filter", str(exc))
                log.warning(
                    "Attempt %d hit non-retryable failure: %s — "
                    "skipping remaining attempts",
                    attempt, exc,
                )
                break
            except TransientError as exc:
                last_error = f"transient: {exc}"
                _record_exit(
                    env_dir,
                    "harness_or_oom_kill" if "SIGKILL" in str(exc)
                    else "api_or_agent_error",
                    str(exc),
                )
                log.warning("Attempt %d failed: %s", attempt, exc)
                if _scoring_touched(env_dir):
                    log.warning(
                        "scoring/ has files — not retrying (human review needed)"
                    )
                    break
                if attempt >= max_retries + 1:
                    break
                # Tear down the project before the next attempt.
                nexus.delete_project(project_id)
                # Continue in place when the dead attempt left real
                # progress; wipe + restart only when there is nothing
                # worth keeping.
                resume_next = _resumable(env_dir)
                if resume_next:
                    log.info("Attempt %d left %d completed turn(s) — "
                             "resuming in place instead of wiping",
                             attempt,
                             int((_read_json(env_dir / ".evaluation.json")
                                  or {}).get("current_turn") or 0))
                # Back off before retrying. A startup failure (a container
                # that dies on launch, an agent CLI that never starts)
                # fails in seconds, so without a pause the whole retry
                # budget would be spent on the same transient condition.
                delay = _ATTEMPT_BACKOFFS[min(attempt, len(_ATTEMPT_BACKOFFS)) - 1]
                log.info("Waiting %ds before attempt %d", delay, attempt + 1)
                time.sleep(delay)
                continue

        except Exception as exc:  # catch-all so we always write meta + commit
            last_error = f"{type(exc).__name__}: {exc}"
            _record_exit(env_dir, "infrastructure_exception", last_error)
            log.error("Run %s failed: %s", run_id, traceback.format_exc())
            break

    # Finalization — runs on both success and failure.
    # The agent writes scores_train.json; scores_test.json is written by
    # the grader (normally absent at this point). scores.json is not
    # part of the protocol and is never read.
    train_score = _read_json(env_dir / "scoring" / "scores_train.json") if env_dir else None
    test_score = _read_json(env_dir / "scoring" / "scores_test.json") if env_dir else None

    # Three terminal statuses, decided on the train artifacts alone
    # (held-out scoring belongs to the grader):
    #   - failed     : scores_train.json is missing or unparseable
    #   - incomplete : scores_train.json exists but validate.py found
    #                  cells the agent owed against the paper's own
    #                  source/scoring/scores_train.json
    #   - completed  : scores_train.json validates cleanly
    if env_dir and env_dir.exists():
        from evaluation import validate as _validate
        train_v = _validate.validate_train(env_dir)
        if train_v.file_missing or train_v.file_invalid:
            status = "failed"
            if not last_error:
                last_error = ("incomplete scoring: scores_train.json "
                              + ("missing" if train_v.file_missing
                                 else "not valid JSON"))
        elif train_v.ok:
            status = "completed"
        else:
            status = "incomplete"
    else:
        status = "failed"
        if not last_error:
            last_error = "scaffold failed: env_dir not present"

    if env_dir and env_dir.exists():
        meta = _read_json(env_dir / ".evaluation.json") or {}
        terminal_state = {
            "completed": "done",
            "incomplete": "incomplete",
            "failed": "failed",
        }[status]
        now = _utcnow()

        # Sum LLM token usage across every turn this run produced, stored
        # under ``meta["usage"]``. ``cost_usd`` is only meaningful for the
        # claude CLI talking to Anthropic directly: when the CLI is routed
        # through a gateway, its ``total_cost_usd`` is an estimate at
        # Anthropic prices, not what the gateway billed. ``cost_source``
        # records where the number came from.
        try:
            from evaluation import usage as _usage
            usage_summary = _usage.extract_usage(env_dir, profile.scaffold)
        except Exception:
            log.exception("usage extraction failed")
            usage_summary = None

        meta.update({
            "status": status,
            "state": terminal_state,
            "state_updated_at": now,
            "ended_at": now,
            "train_score": train_score,
            "test_score": test_score,
            "error": last_error,
        })
        if usage_summary is not None:
            meta["usage"] = usage_summary
        _write_meta(env_dir, meta)
        try:
            git.commit(env_dir, f"eval: final (status={status})")
        except Exception as exc:
            log.warning("final commit failed: %s", exc)

    # Refresh the paper's normalization coefficients and the per-mode
    # summary. Triggered at the end of every run; only runs with a clean
    # grader verdict contribute (see normalization._run_scores), so this
    # is a no-op for the coefficients until the run has been graded.
    # Errors are logged and swallowed: normalization must not break
    # finalization.
    # Variant runs (idea-implementation experiment) are standalone: they
    # do not take part in normalization and are not published (their
    # branch name would collide with the main run's).
    if not variant:
        try:
            normalization.update_for_paper(paper_id)
        except Exception:
            log.exception("Normalization update_for_paper(%s) failed", paper_id)

    # On completion, optionally push the run to branch
    # ``{mode}-{profile}`` of the paper's repository (opt-in, see
    # git_publish.py). Publish errors never change the run's status.
    if not variant and status == "completed" and env_dir and env_dir.exists():
        publish_record = _publish_run(env_dir, paper_id, profile_name, mode, cfg)
        # Persist the publish outcome in the run's metadata so the
        # dashboard can show it. Re-read first so concurrent updates are
        # not overwritten.
        try:
            meta_now = _read_json(env_dir / ".evaluation.json") or {}
            meta_now["publish"] = publish_record
            _write_meta(env_dir, meta_now)
        except Exception:
            log.exception("Failed to record publish outcome to meta")

        # Independent verification phase (off by default). A fresh grader
        # session re-runs the frozen method, regenerates predictions,
        # grades them with the pristine evaluators and writes
        # scoring/verification.json. On "verified" it stores the
        # recomputed scores in the metadata and refreshes normalization.
        if (cfg.get("verifier") or {}).get("enabled"):
            try:
                from evaluation import verifier as _verifier
                _verifier.verify_run(paper_id, profile_name, mode)
            except Exception:
                log.exception("Verification phase failed for %s/%s — "
                              "run status unchanged", paper_id, profile_name)

    # Reclaim the per-run container image and overlay (an overlay can be
    # tens of GB). They are disposable once nothing else will run in
    # this workspace: everything kept for analysis (scores, trajectory,
    # method/, jobs/, .git, .evaluation.json) lives in env_dir itself,
    # and a later resume or grading rebuilds them from the cache. Done
    # last because the verifier above reuses them.
    if env_dir and env_dir.exists():
        _remove_container_artifacts(env_dir)

    nexus.delete_project(project_id)

    return RunResult(
        run_id=run_id,
        paper_id=paper_id,
        model=profile_name,
        mode=mode,
        status=status,
        attempt=attempt,
        env_dir=str(env_dir) if env_dir else "",
        train_score=train_score,
        test_score=test_score,
        error=last_error,
    )


def resume_test_phase(
    paper_id: str, profile_name: str, mode: str = "innovate",
) -> RunResult:
    """Finalize an existing run whose main phase finished; never relaunch the agent.

    For a run that built ``method/`` and produced ``scores_train.json``
    but was not finalized (the harness died between the agent's last
    turn and finalization), this redoes finalization exactly like
    :func:`run_one`: status, metadata, normalization, optional publish
    and optional grading. No re-scaffold and no agent turn.

    The function was written to also re-run the in-run held-out test
    phase (:func:`_run_test_phase`) against the existing method. That
    phase is disabled (see the note there), so the job-execution block
    below is skipped unconditionally.

    Refuses to touch a run that has no built method or no train score,
    or one that is already completed with a test score.
    """
    cfg = _config.load()
    profile = get_profile(profile_name)
    data_dir = _config.data_dir()
    env_dir = data_dir / "envs" / paper_id / f"{profile_name}-{mode}-eval"
    run_id = _run_id(paper_id, profile_name, mode)
    project_id = _project_id(paper_id, profile_name, mode)

    method_dir = env_dir / "method"
    if not env_dir.exists():
        raise RuntimeError(f"env_dir missing: {env_dir}")
    if not (env_dir / "scoring" / "scores_train.json").exists():
        raise RuntimeError(f"no scores_train.json — nothing to resume from: {env_dir}")
    if not method_dir.is_dir() or not any(method_dir.iterdir()):
        raise RuntimeError(f"method/ empty — no built method to score: {env_dir}")

    # Triage on an existing scores_test.json:
    #   - rc-gate had rejected it (last_test_rc != 0): quarantine the
    #     suspect file and re-run the test phase honestly.
    #   - otherwise the run actually finished and only finalization
    #     (status / normalization / publish) died — skip the test phase
    #     and just finalize.
    run_test = True
    test_json = env_dir / "scoring" / "scores_test.json"
    if test_json.exists():
        _prior_meta = _read_json(env_dir / ".evaluation.json") or {}
        if _prior_meta.get("status") == "completed":
            raise RuntimeError(
                f"run already completed — nothing to resume: {env_dir}")
        _rc = _prior_meta.get("last_test_rc")
        if isinstance(_rc, int) and _rc != 0:
            quarantined = test_json.with_name(
                f"scores_test.rejected.{_utcnow().replace(':', '')}.json")
            test_json.rename(quarantined)
            log.warning(
                "existing scores_test.json was produced alongside a "
                "failed test job (rc=%s) — quarantined to %s; re-running "
                "the test phase", _rc, quarantined.name)
        else:
            run_test = False
            log.info(
                "scores_test.json present and its producing job passed — "
                "run actually finished; redoing finalization only")

    log.info("Finalizing %s (method already built, agent not relaunched)", env_dir)

    last_error: str | None = None
    # The in-run held-out test phase is disabled: held-out scoring is the
    # grader's job, so this entry point only finalizes and the block
    # below never runs.
    run_test = False
    if run_test:
        # Rebuild the compute container + overlay (cache-backed) + pre-bind
        # downloads. Does NOT wipe or re-scaffold the workspace.
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
        _create_kwargs = dict(
            project_id=project_id,
            workspace_dir=str(env_dir),
            sif_path=str(container_sif),
            overlay_path=str(overlay_path) if overlay_path else "",
            gpu_budget=float(cfg["nexus"]["gpu_budget_hours"]),
            max_gpus=int(cfg["nexus"]["max_gpus"]),
            allowed_resources=_config.nexus_all_resources(cfg),
            checkpoint_interval=int(cfg.get("checkpoint_interval") or 1200),
            checkpoint_timeout=int(cfg.get("checkpoint_timeout") or 1800),
        )
        nexus.delete_project(project_id)
        import time as _t
        _t.sleep(1)
        nexus.create_project(**_create_kwargs)

        # turn_offset past the highest existing trajectory turn so files don't collide.
        traj = env_dir / "trajectory"
        turn_offset = 0
        if traj.is_dir():
            for p in traj.iterdir():
                if p.name.startswith("turn_") and p.suffix == ".jsonl":
                    try:
                        turn_offset = max(turn_offset, int(p.name.split("_")[1]))
                    except (ValueError, IndexError):
                        pass

        try:
            _run_test_phase(
                env_dir, profile, nexus, project_id,
                turn_offset=turn_offset, extra_binds=extra_binds,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log.error("resume_test_phase %s failed: %s", run_id, traceback.format_exc())

    # ── Finalization (mirrors run_one) ──
    from evaluation import validate as _validate
    train_score = _read_json(env_dir / "scoring" / "scores_train.json")
    test_score = _read_json(env_dir / "scoring" / "scores_test.json")
    train_v = _validate.validate_train(env_dir)
    # Status is decided on the train artifacts alone, as in run_one.
    if train_v.file_missing or train_v.file_invalid:
        status = "failed"
        if not last_error:
            last_error = "incomplete scoring: scores_train.json missing/invalid"
    elif train_v.ok:
        status = "completed"
    elif False:
        # Unreachable while the in-run test phase is disabled. With that
        # phase, a validating scores_test.json whose final producing job
        # failed (last_test_rc != 0) was rejected here.
        _meta_rc = (_read_json(env_dir / ".evaluation.json") or {}).get("last_test_rc")
        if isinstance(_meta_rc, int) and _meta_rc != 0:
            status = "incomplete"
            if not last_error:
                last_error = (
                    f"test phase: final test job exited rc={_meta_rc} — "
                    "scores_test.json rejected (agent-authored suspected)"
                )
        else:
            status = "completed"
    else:
        status = "incomplete"

    meta = _read_json(env_dir / ".evaluation.json") or {}
    now = _utcnow()
    meta.update({
        "status": status,
        "state": {"completed": "done", "incomplete": "incomplete", "failed": "failed"}[status],
        "state_updated_at": now,
        "ended_at": now,
        "train_score": train_score,
        "test_score": test_score,
        "error": last_error,
        "resumed_test_phase": True,
    })
    _write_meta(env_dir, meta)
    try:
        git.commit(env_dir, f"eval: test-phase resume (status={status})")
    except Exception as exc:
        log.warning("resume final commit failed: %s", exc)
    try:
        normalization.update_for_paper(paper_id)
    except Exception:
        log.exception("Normalization update_for_paper(%s) failed", paper_id)
    if status == "completed":
        # Opt-in publish (a no-op unless publishing is enabled).
        try:
            from evaluation import git_publish
            git_publish.publish(env_dir, paper_id, profile_name, mode)
        except Exception:
            log.exception("Publish failed for %s/%s (resume) — status unchanged",
                          paper_id, profile_name)
        # Same verification hook as run_one.
        if (cfg.get("verifier") or {}).get("enabled"):
            try:
                from evaluation import verifier as _verifier
                _verifier.verify_run(paper_id, profile_name, mode)
            except Exception:
                log.exception("Verification phase failed for %s/%s (resume) — "
                              "run status unchanged", paper_id, profile_name)
    # Reclaim the container image and overlay, after the grader (which
    # rebuilds and uses them) is done.
    _remove_container_artifacts(env_dir)
    # nexus only exists when the (disabled) test-job block ran.
    if run_test:
        nexus.delete_project(project_id)

    log.info("resume_test_phase %s -> %s", run_id, status)
    return RunResult(
        run_id=run_id, paper_id=paper_id, model=profile_name, mode=mode,
        status=status, attempt=1, env_dir=str(env_dir),
        train_score=train_score, test_score=test_score, error=last_error,
    )


def warm(paper_id: str, force: bool = False) -> None:
    """Populate ``<data_dir>/cache/<paper_id>/`` for reuse across runs.

    Scaffolds a throwaway workspace, builds the paper's ``container.sif``
    and ``overlay.img`` on the current host (which needs network
    access), runs ``scripts/download.sh`` inside the built container to
    populate ``data/downloads/``, then copies everything into the cache.

    No agent is invoked and no job is submitted; everything runs on the
    host. ``force=True`` wipes any existing cache for this paper first.
    """
    import sciworld  # lazy

    cfg = _config.load()

    if force:
        dest = cache.cache_dir(paper_id)
        if dest.exists():
            log.info("Wiping existing cache %s", dest)
            shutil.rmtree(dest)
    elif cache.is_warm(paper_id):
        log.info("Cache for %s already populated (use force=True to rebuild)",
                 paper_id)
        return

    # One directory per paper: sciworld keeps the paper's environment in
    # the directory ``source`` next to the workspace, so two papers must
    # never share a parent.
    staging_root = _config.data_dir() / "cache-staging" / paper_id
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging = staging_root / "workspace"
    staging.parent.mkdir(parents=True, exist_ok=True)

    log.info("Scaffolding staging workspace at %s", staging)
    sciworld.research(
        id=paper_id, agent="claude-code", mode="innovate",
        path=str(staging), setup_only=True, env="container",
    )

    container_sif = _container.ensure_container(
        staging,
        container_cache_dir=str(cfg.get("container_cache_dir") or ""),
    )
    overlay_path = _container.ensure_overlay(
        staging, container_sif,
        container_cache_dir=str(cfg.get("container_cache_dir") or ""),
        local_tmp_dir=str(cfg.get("local_tmp_dir") or "/tmp"),
        uv_cache_dir=str(cfg.get("uv_cache_dir") or ""),
    )

    download_sh = staging / "scripts" / "download.sh"
    if download_sh.exists():
        log.info("Running download.sh inside paper container")
        _run_in_container(
            container_sif, staging,
            cmd=["bash", "/home/user/scripts/download.sh"],
            log_path=staging / "logs" / "download.log",
            overlay_path=overlay_path,
            timeout=int(cfg.get("warm_download_timeout", 7200)),
        )

    cache.populate_from_workspace(paper_id, staging)
    log.info("Warm complete for %s", paper_id)


def _run_in_container(
    container_sif: Path,
    workspace: Path,
    *,
    cmd: list[str],
    log_path: Path,
    overlay_path: Path | None,
    timeout: int,
) -> None:
    """Run ``cmd`` inside the paper container with workspace bound at ``/home/user``."""
    import subprocess

    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "apptainer", "exec",
        "--containall", "--cleanenv", "--writable-tmpfs",
        "--bind", f"{workspace}:/home/user",
        "--pwd", "/home/user",
    ]
    if overlay_path is not None:
        argv += ["--overlay", f"{overlay_path}:ro"]
    argv += [str(container_sif), *cmd]

    with open(log_path, "a") as fh:
        fh.write(f"\n=== CMD: {' '.join(argv)} ===\n")
        fh.flush()
        result = subprocess.run(
            argv, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
        )
        fh.write(f"=== RETURN CODE: {result.returncode} ===\n")
    if result.returncode != 0:
        raise RuntimeError(f"{cmd!r} in container failed (rc={result.returncode}); see {log_path}")


__all__ = ["RunResult", "run_one", "warm"]
