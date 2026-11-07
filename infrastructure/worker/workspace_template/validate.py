#!/usr/bin/env python3
"""Workspace validator — checks scoring files and structural rules.

Usage:
    python validate.py                          # validate all
    python validate.py --reference-only         # validate reference.json only
    python validate.py --compare                # validate + compare scores against reference
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).parent
SCORING_DIR = WORKSPACE_DIR / "scoring"


# ── Scoring validation ───────────────────────────────────────────────────────


_FORBIDDEN_METRIC_PATTERNS = re.compile(
    r"(?:author|deviation|published|vs_paper|fit_to_published|"
    r"reproduction_error|paper_method|prompt_deviation|"
    r"error_vs_|agreement_with|fidelity_to)",
    re.IGNORECASE,
)


def validate_reference(data: dict) -> list[str]:
    """Validate reference.json structure. Returns list of errors."""
    errors = []

    if "experiments" not in data:
        errors.append("Missing top-level 'experiments' key")
        return errors

    if not isinstance(data["experiments"], dict):
        errors.append("'experiments' must be a dict")
        return errors

    if len(data["experiments"]) == 0:
        errors.append("'experiments' is empty — add at least one experiment")

    # Validate weights sum to ~1.0
    weights = []
    for exp_name, exp in data["experiments"].items():
        if isinstance(exp, dict) and "weight" in exp:
            w = exp["weight"]
            if isinstance(w, (int, float)):
                weights.append(w)
    if weights:
        total = sum(weights)
        if abs(total - 1.0) > 0.01:
            errors.append(f"Experiment weights sum to {total:.3f}, expected 1.0")

    for exp_name, exp in data["experiments"].items():
        prefix = f"experiments.{exp_name}"

        if not isinstance(exp, dict):
            errors.append(f"{prefix}: must be a dict")
            continue

        # Required fields
        if "weight" not in exp:
            errors.append(f"{prefix}: missing 'weight'")
        elif not isinstance(exp["weight"], (int, float)):
            errors.append(f"{prefix}.weight: must be a number")
        elif not (0 < exp["weight"] <= 1):
            errors.append(f"{prefix}.weight: must be between 0 and 1")
        if "primary_metric" not in exp:
            errors.append(f"{prefix}: missing 'primary_metric'")
        if "metrics" not in exp:
            errors.append(f"{prefix}: missing 'metrics'")
        if "results" not in exp:
            errors.append(f"{prefix}: missing 'results'")

        # Validate metrics
        metrics = exp.get("metrics", {})
        if not isinstance(metrics, dict):
            errors.append(f"{prefix}.metrics: must be a dict")
            metrics = {}

        for metric_name, metric_def in metrics.items():
            if not isinstance(metric_def, dict):
                errors.append(f"{prefix}.metrics.{metric_name}: must be a dict")
                continue
            if "higher_is_better" not in metric_def:
                errors.append(f"{prefix}.metrics.{metric_name}: missing 'higher_is_better'")
            elif not isinstance(metric_def["higher_is_better"], bool):
                errors.append(f"{prefix}.metrics.{metric_name}.higher_is_better: must be a bool")
            if "coefficient" not in metric_def:
                errors.append(f"{prefix}.metrics.{metric_name}: missing 'coefficient'")
            elif not isinstance(metric_def["coefficient"], (int, float)):
                errors.append(f"{prefix}.metrics.{metric_name}.coefficient: must be a number")
            if _FORBIDDEN_METRIC_PATTERNS.search(metric_name):
                errors.append(
                    f"{prefix}.metrics.{metric_name}: metric name matches a "
                    f"forbidden paper-agreement pattern. The gym measures "
                    f"the scientific quantity, not agreement with the paper. "
                    f"Either remove this metric or rename it to reflect "
                    f"the underlying quantity it actually measures."
                )

        # primary_metric must be in metrics
        primary = exp.get("primary_metric")
        if primary and metrics and primary not in metrics:
            errors.append(f"{prefix}: primary_metric '{primary}' not found in metrics")

        # Validate results
        results = exp.get("results", {})
        if not isinstance(results, dict):
            errors.append(f"{prefix}.results: must be a dict")
            continue

        if len(results) == 0:
            errors.append(f"{prefix}.results: empty — add at least one method")

        has_proposed = False
        for method_name, method in results.items():
            mp = f"{prefix}.results.{method_name}"
            if not isinstance(method, dict):
                errors.append(f"{mp}: must be a dict")
                continue

            # type is required
            if "type" not in method:
                errors.append(f"{mp}: missing 'type' (must be 'baseline' or 'proposed')")
            elif method["type"] not in ("baseline", "proposed"):
                errors.append(f"{mp}: type must be 'baseline' or 'proposed', got '{method['type']}'")
            elif method["type"] == "proposed":
                has_proposed = True

            # Check that metric values are numbers and use known metric names
            for key, val in method.items():
                if key == "type":
                    continue
                if key not in metrics:
                    errors.append(f"{mp}: unknown metric '{key}' (not in metrics)")
                elif not isinstance(val, (int, float)):
                    errors.append(f"{mp}.{key}: must be a number, got {type(val).__name__}")

            # Check primary metric is present
            if primary and primary not in method and method.get("type") != "baseline":
                errors.append(f"{mp}: missing primary metric '{primary}'")

        if not has_proposed:
            errors.append(f"{prefix}.results: no method with type 'proposed' — at least one is expected")

    return errors


def validate_scores(data: dict, reference: dict | None = None) -> list[str]:
    """Validate scores.json structure. Returns list of errors."""
    errors = []

    if not isinstance(data, dict):
        errors.append("scores.json must be a dict")
        return errors

    # Accept both formats: {experiments: {...}} or flat {exp: {method: {...}}}
    if "experiments" in data and isinstance(data["experiments"], dict):
        experiments = data["experiments"]
    else:
        experiments = data

    if len(experiments) == 0:
        errors.append("scores.json is empty")
        return errors

    ref_exps = {}
    if reference and "experiments" in reference:
        ref_exps = reference["experiments"]

    for exp_name, exp_data in experiments.items():
        if exp_name.startswith("_"):
            continue

        if not isinstance(exp_data, dict):
            errors.append(f"{exp_name}: must be a dict")
            continue

        if exp_name not in ref_exps and ref_exps:
            errors.append(f"{exp_name}: not found in reference.json experiments")
            continue

        ref_metrics = ref_exps.get(exp_name, {}).get("metrics", {}) if ref_exps else {}
        ref_results = ref_exps.get(exp_name, {}).get("results", {}) if ref_exps else {}

        # Accept both: {results: {method: {metrics}}} or flat {method: {metrics}}
        if "results" in exp_data and isinstance(exp_data["results"], dict):
            methods = exp_data["results"]
        else:
            methods = exp_data

        for method_name, method_scores in methods.items():
            if method_name.startswith("_"):
                continue
            mp = f"{exp_name}.{method_name}"

            if not isinstance(method_scores, dict):
                errors.append(f"{mp}: must be a dict of metric_name -> value")
                continue

            if ref_results and method_name not in ref_results:
                errors.append(f"{mp}: method not found in reference.json")

            for metric_name, val in method_scores.items():
                if not isinstance(val, (int, float)):
                    errors.append(f"{mp}.{metric_name}: must be a number, got {type(val).__name__}")
                if ref_metrics and metric_name not in ref_metrics:
                    errors.append(f"{mp}.{metric_name}: not a known metric in reference.json")

    return errors


def _extract_scores_experiments(scores: dict) -> dict:
    """Normalize scores.json into {exp_name: {method_name: {metrics}}}."""
    if "experiments" in scores and isinstance(scores["experiments"], dict):
        raw = scores["experiments"]
    else:
        raw = scores
    out = {}
    for exp_name, exp_data in raw.items():
        if exp_name.startswith("_") or not isinstance(exp_data, dict):
            continue
        if "results" in exp_data and isinstance(exp_data["results"], dict):
            out[exp_name] = exp_data["results"]
        else:
            out[exp_name] = exp_data
    return out


def compare_scores(scores: dict, reference: dict) -> None:
    """Print side-by-side comparison of reproduced scores against reference."""
    exps = reference.get("experiments", {})
    score_exps = _extract_scores_experiments(scores)

    for exp_name, exp_methods in score_exps.items():
        exp_ref = exps.get(exp_name)
        if not exp_ref:
            continue

        weight = exp_ref.get("weight", "?")
        print(f"\n=== {exp_name} (weight: {weight}) ===")
        if exp_ref.get("description"):
            print(f"    {exp_ref['description']}")

        primary = exp_ref.get("primary_metric")
        metrics = exp_ref.get("metrics", {})
        ref_results = exp_ref.get("results", {})

        all_metrics = list(metrics.keys())

        print(f"\n    {'Method':<25} {'Source':<10}", end="")
        for m in all_metrics:
            marker = " *" if m == primary else ""
            print(f"{(m + marker):>15}", end="")
        print()
        print(f"    {'-'*25} {'-'*10}", end="")
        for _ in all_metrics:
            print(f"{'─'*15}", end="")
        print()

        all_methods = list(dict.fromkeys(list(ref_results.keys()) + list(exp_methods.keys())))
        for method_name in all_methods:
            ref_method = ref_results.get(method_name, {})
            score_method = exp_methods.get(method_name)
            tag = f" [{ref_method['type']}]" if ref_method.get("type") else ""

            print(f"    {(method_name + tag):<25} {'paper':<10}", end="")
            for m in all_metrics:
                val = ref_method.get(m)
                print(f"{val:>15.3f}" if val is not None else f"{'—':>15}", end="")
            print()

            if score_method:
                print(f"    {'':<25} {'repro':<10}", end="")
                for m in all_metrics:
                    val = score_method.get(m)
                    print(f"{val:>15.3f}" if val is not None else f"{'—':>15}", end="")
                print()
            print()


# ── Structural checks ────────────────────────────────────────────────────────


_EVAL_REGEN_HINTS = re.compile(
    r"\b(generate_synthetic_data|generate_data|simulate_data|run_simulation|"
    r"np\.random\.(?:default_rng|RandomState|seed)|"
    r"torch\.manual_seed|random\.seed)\b"
)
# A read call (open / pd.read_* / np.load / torch.load / .read_text /
# .read_bytes / json.load(open / joblib.load) whose first string-literal
# argument contains a path fragment that looks like an artifact the agent's
# method writes (predictions, outputs, checkpoints, state_dict, etc.).
_EVAL_OUTPUT_READ = re.compile(
    r"(?:open|pd\.read_csv|pd\.read_json|pd\.read_parquet|"
    r"np\.load(?:txt)?|torch\.load|joblib\.load|"
    r"json\.load\s*\(\s*open|\.read_text|\.read_bytes)"
    r"\s*\(\s*[\"'][^\"']*?"
    r"(?:predictions?|preds?|outputs?|checkpoints?|submission|state_dict|"
    r"\.(?:pt|pth|bin|safetensors|ckpt))",
    re.IGNORECASE,
)


def check_eval_reads_method_outputs() -> list[str]:
    """SOFT WARNING (heuristic): does eval/ read anything the method produced?

    Returns *warnings*, not errors — the validator's caller should print
    these but not fail the run. The heuristic flags eval files that:

    1. contain regen-style calls (``generate_*``, ``np.random.seed``, etc.),
       suggesting eval is rerunning the experiment in-place, AND
    2. don't reference any output-shaped path (``predictions``, ``outputs``,
       ``checkpoints``, ``state_dict``, etc.) — i.e. nothing produced by
       the method.

    A detached eval (regen + no method-output reads) is the failure mode
    where a future agent's edits to ``method/`` produce zero change in
    scores. False positives are possible (an oracle eval that reads only
    the dataset's labels), so this is reported as a warning, not an error.
    """
    warnings: list[str] = []
    eval_dir = WORKSPACE_DIR / "eval"
    if not eval_dir.is_dir():
        return warnings

    for py in sorted(eval_dir.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        try:
            text = py.read_text()
        except OSError:
            continue
        rel = py.relative_to(WORKSPACE_DIR)
        regen_match = _EVAL_REGEN_HINTS.search(text)
        output_match = _EVAL_OUTPUT_READ.search(text)
        if regen_match and not output_match:
            warnings.append(
                f"{rel}: looks detached from the method — found "
                f"'{regen_match.group(0)}' (regenerating the experiment) "
                f"and no reference to any path that looks like a method "
                f"output (predictions, outputs, checkpoints, state_dict, "
                f"etc.). If a future agent edits method/ and these scores "
                f"don't change, eval is broken. Read the method's "
                f"artifacts; do not rerun the science yourself."
            )
    return warnings


def check_import_separation() -> list[str]:
    """Check that eval/ and data/ do not import from method/."""
    errors = []
    pattern = re.compile(r"^\s*(?:from\s+method\b|import\s+method\b)", re.MULTILINE)
    for dirname in ("eval", "data"):
        dirpath = WORKSPACE_DIR / dirname
        if not dirpath.is_dir():
            continue
        for pyfile in dirpath.rglob("*.py"):
            try:
                content = pyfile.read_text()
            except Exception:
                continue
            if pattern.search(content):
                rel = pyfile.relative_to(WORKSPACE_DIR)
                errors.append(f"{rel}: imports from method/")
    return errors


def _resolve_local_import(module: str, from_file: Path | None, level: int) -> Path | None:
    """Resolve a dotted module name to a file inside the workspace, if any.

    Returns None if the import does not resolve to a workspace-local file
    (i.e. it is stdlib, third-party, or unresolvable).
    """
    if level > 0 and from_file is not None:
        anchor = from_file.parent
        for _ in range(level - 1):
            anchor = anchor.parent
        parts = module.split(".") if module else []
        candidate_base = anchor.joinpath(*parts) if parts else anchor
    else:
        if not module:
            return None
        parts = module.split(".")
        candidate_base = WORKSPACE_DIR.joinpath(*parts)

    # Must live inside the workspace
    try:
        candidate_base.resolve().relative_to(WORKSPACE_DIR.resolve())
    except ValueError:
        return None

    py_file = candidate_base.with_suffix(".py")
    if py_file.is_file():
        return py_file
    init_file = candidate_base / "__init__.py"
    if init_file.is_file():
        return init_file
    return None


# Top-level workspace directories that always hold local code / data. An
# import whose first dotted component matches this set is workspace-local by
# convention, even if the directory doesn't exist in the current tree (e.g.
# a fresh clone before download.sh runs).
_WORKSPACE_IMPORT_PREFIXES = frozenset({"data", "method", "eval", "baseline"})


def _expected_import_base(module: str, from_file: Path | None, level: int) -> Path | None:
    """Compute the expected workspace path for an import (existence not required).

    Returns the candidate base path (without .py suffix) if the import looks
    workspace-local, else None. Workspace-local = relative imports anchored
    in from_file OR absolute imports whose first dotted component is a known
    workspace directory.

    Policy: every workspace-local import must resolve to a git-tracked file.
    This includes vendored third-party libraries under ``method/`` — they
    should be committed, not gitignored, so fresh clones work without a
    runtime clone step (up to reasonable size limits).
    """
    if level > 0 and from_file is not None:
        anchor = from_file.parent
        for _ in range(level - 1):
            anchor = anchor.parent
        parts = module.split(".") if module else []
        candidate_base = anchor.joinpath(*parts) if parts else anchor
    else:
        if not module:
            return None
        parts = module.split(".")
        if parts[0] not in _WORKSPACE_IMPORT_PREFIXES:
            return None
        candidate_base = WORKSPACE_DIR.joinpath(*parts)
    try:
        candidate_base.resolve().relative_to(WORKSPACE_DIR.resolve())
    except ValueError:
        return None
    return candidate_base


def check_imports_not_gitignored() -> list[str]:
    """Check that every workspace-local import in eval/, method/, baseline/
    resolves to a file that git *tracks*.

    Two failure modes both break a fresh clone:
      (a) import target exists on disk but is gitignored (won't be committed)
      (b) import target does not exist at all in git (file missing entirely)

    Both fail here — (a) was historically caught by walking the disk and
    cross-checking ``git check-ignore``; (b) was silently skipped because the
    old resolver returned None for missing files. Both now resolve to the
    expected path and are validated against the tracked fileset.
    """
    errors: list[str] = []
    # expected_path -> list of (source_file, lineno) tuples
    import_map: dict[Path, list[tuple[Path, int]]] = {}

    for dirname in ("eval", "method", "baseline"):
        dirpath = WORKSPACE_DIR / dirname
        if not dirpath.is_dir():
            continue
        for pyfile in dirpath.rglob("*.py"):
            try:
                source = pyfile.read_text()
            except Exception:
                continue
            try:
                tree = ast.parse(source, filename=str(pyfile))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        base = _expected_import_base(alias.name, None, 0)
                        if base is not None:
                            import_map.setdefault(base, []).append((pyfile, node.lineno))
                elif isinstance(node, ast.ImportFrom):
                    if node.module is None and node.level == 0:
                        continue
                    base = _expected_import_base(node.module or "", pyfile, node.level)
                    if base is not None:
                        import_map.setdefault(base, []).append((pyfile, node.lineno))

    if not import_map:
        return errors

    # Ask git for the full tracked fileset once, then match each expected path
    # against it. A ".py" file or a "/__init__.py" package either exists in the
    # tree or it doesn't — anything else is a fresh-clone failure.
    try:
        tracked_result = subprocess.run(
            ["git", "-C", str(WORKSPACE_DIR), "ls-files"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return errors
    if tracked_result.returncode != 0:
        # Not a git repository (or git failed): there is no tracked fileset
        # to check against, so every import would be reported as missing.
        return errors
    tracked = {line.strip() for line in tracked_result.stdout.splitlines() if line.strip()}

    import glob as _glob

    for base, callsites in sorted(import_map.items(), key=lambda kv: str(kv[0])):
        rel_base = base.relative_to(WORKSPACE_DIR)
        py_rel = str(rel_base) + ".py"
        pkg_rel = str(rel_base) + "/__init__.py"
        if py_rel in tracked or pkg_rel in tracked:
            continue  # import target is tracked — fresh clone is fine
        # Compiled extensions (.so, .pyd, .cpython-*.so) are built at install
        # time and are never committed to git. Skip if one exists on disk or
        # if the parent directory contains a build system file that would
        # produce one (setup.py, CMakeLists.txt, meson.build, pyproject.toml
        # with a compiled backend, or pybind11 sources).
        module_name = base.name
        parent_dir = base.parent
        ext_globs = [
            str(parent_dir / f"{module_name}.*.so"),
            str(parent_dir / f"{module_name}.*.pyd"),
            str(parent_dir / f"{module_name}.so"),
            str(parent_dir / f"{module_name}.pyd"),
        ]
        if any(_glob.glob(g) for g in ext_globs):
            continue  # compiled extension on disk — built at install time
        build_markers = ("setup.py", "CMakeLists.txt", "meson.build", "pyproject.toml",
                         "Cargo.toml")
        # Walk up from the module's parent to the workspace root looking for
        # a build system file that would produce this compiled extension.
        search = parent_dir
        _found_build = False
        while True:
            if any((search / m).exists() for m in build_markers):
                _found_build = True
                break
            if search == WORKSPACE_DIR or search == search.parent:
                break
            search = search.parent
        if _found_build:
            continue
        for source_file, lineno in callsites:
            rel_source = source_file.relative_to(WORKSPACE_DIR)
            on_disk = (
                (WORKSPACE_DIR / py_rel).is_file()
                or (WORKSPACE_DIR / pkg_rel).is_file()
            )
            reason = (
                f"exists locally but is gitignored"
                if on_disk
                else f"not present in git at all"
            )
            errors.append(
                f"{rel_source}:{lineno}: imports {rel_base} ({py_rel} or "
                f"{pkg_rel}), which is {reason} — fresh clone will hit ImportError"
            )
    return errors


def check_train_test_independent() -> list[str]:
    """Check that eval/train/ and eval/test/ do not import from each other.

    The two slices must be independently runnable. A shortcut agent that reads
    eval/train/ should not be able to learn anything about eval/test/'s shape,
    metrics, or data through a sibling import.
    """
    errors: list[str] = []
    train_dir = WORKSPACE_DIR / "eval" / "train"
    test_dir = WORKSPACE_DIR / "eval" / "test"

    def _scan(src_dir: Path, forbidden_dir: Path, forbidden_label: str) -> None:
        if not src_dir.is_dir() or not forbidden_dir.is_dir():
            return
        for pyfile in src_dir.rglob("*.py"):
            try:
                source = pyfile.read_text()
            except Exception:
                continue
            try:
                tree = ast.parse(source, filename=str(pyfile))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                resolved = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        r = _resolve_local_import(alias.name, None, 0)
                        if r is not None:
                            try:
                                r.resolve().relative_to(forbidden_dir.resolve())
                                resolved = r
                                break
                            except ValueError:
                                pass
                elif isinstance(node, ast.ImportFrom):
                    if node.module is None and node.level == 0:
                        continue
                    r = _resolve_local_import(node.module or "", pyfile, node.level)
                    if r is not None:
                        try:
                            r.resolve().relative_to(forbidden_dir.resolve())
                            resolved = r
                        except ValueError:
                            pass
                if resolved is not None:
                    rel_src = pyfile.relative_to(WORKSPACE_DIR)
                    rel_dst = resolved.relative_to(WORKSPACE_DIR)
                    errors.append(
                        f"{rel_src}:{node.lineno}: imports {rel_dst} "
                        f"(eval/train and eval/{forbidden_label} must be independent)"
                    )

    _scan(train_dir, test_dir, "test")
    _scan(test_dir, train_dir, "train")
    return errors


# Detect Python/bash invocations that point at the wrong eval/ slice. Matches
# any `python …`, `python3 …`, `bash …`, `sh …`, or direct-exec line where the
# interpreter arg contains `eval/<slice>/`. A bare reference inside a `#`
# comment or a string that reads from `scores_train.json` (legitimate for
# method enumeration) is not matched.
_WRONG_SLICE_INVOCATION_RE_TEMPLATE = (
    r"^\s*(?:python\d*|bash|sh|exec|\.)\s+[^\n#]*eval/{slice}/"
)

# Detect `--output …/<wrong>.json`, `> …/<wrong>.json`, and similar write
# operations. Matches the common flag shapes used by argparse-based
# evaluators plus shell redirects.
_WRONG_OUTPUT_RE_TEMPLATE = (
    r"(?:--output[ =]|--out[ =]|-o[ =]|--scores[ =]|--score-file[ =]|"
    r"--output-file[ =]|>\s*)\S*{fname}"
)


def check_train_test_scripts_rewired() -> list[str]:
    """Check that scripts/evaluate_{train,test}.sh point at the right slice.

    A common designer failure mode is copying ``evaluate_train.sh`` to
    ``evaluate_test.sh``, renaming the file, but leaving the body invoking
    ``eval/train/evaluate.py`` and writing to ``scoring/scores.json``. The
    resulting "test" script re-runs the train evaluator against the train
    data and emits train-slice artifacts under a test-slice filename —
    the entire train/test comparison becomes meaningless.

    Static checks (no execution required):
      1. ``evaluate_test.sh`` must not invoke ``eval/train/…`` scripts.
      2. ``evaluate_test.sh`` must not write (via --output / >) to
         ``scores.json`` or ``scores_train.json``.
      3. Symmetrically, ``evaluate_train.sh`` must not invoke
         ``eval/test/…`` or write to ``scores_test.json``.

    Only runs when both shell scripts exist; pre-designer iterations are
    silently skipped.
    """
    errors: list[str] = []
    scripts_dir = WORKSPACE_DIR / "scripts"
    train_script = scripts_dir / "evaluate_train.sh"
    test_script = scripts_dir / "evaluate_test.sh"

    if not train_script.is_file() or not test_script.is_file():
        return errors

    try:
        train_text = train_script.read_text()
        test_text = test_script.read_text()
    except OSError:
        return errors

    def _strip_comments(text: str) -> str:
        # Drop comment-only and trailing-comment portions so a comment like
        # ``# evaluate_train.sh mirrors this shape`` in evaluate_test.sh
        # doesn't false-positive.
        out = []
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Trailing # comment after a code token — only strip when preceded
            # by whitespace so '#' inside quoted strings (rare in sh) is
            # preserved.
            idx = 0
            while idx < len(line):
                if line[idx] == "#" and (idx == 0 or line[idx - 1].isspace()):
                    line = line[:idx]
                    break
                idx += 1
            out.append(line)
        return "\n".join(out)

    test_body = _strip_comments(test_text)
    train_body = _strip_comments(train_text)

    # 1. evaluate_test.sh must not invoke eval/train/
    pat = re.compile(
        _WRONG_SLICE_INVOCATION_RE_TEMPLATE.format(slice="train"),
        re.MULTILINE,
    )
    if pat.search(test_body):
        errors.append(
            "scripts/evaluate_test.sh invokes eval/train/ "
            "(should invoke eval/test/) — looks like copy-paste from "
            "evaluate_train.sh without rewiring the evaluator path"
        )

    # 2. evaluate_test.sh must not write to scores.json or scores_train.json
    for fname, label in (
        (r"scores\.json(?!\w)", "scoring/scores.json"),
        (r"scores_train\.json(?!\w)", "scoring/scores_train.json"),
    ):
        pat = re.compile(_WRONG_OUTPUT_RE_TEMPLATE.format(fname=fname))
        if pat.search(test_body):
            errors.append(
                f"scripts/evaluate_test.sh writes to {label} "
                f"(should write to scoring/scores_test.json) — looks like "
                f"copy-paste from evaluate_train.sh without rewiring the "
                f"output path"
            )

    # 3. Symmetric checks for evaluate_train.sh
    pat = re.compile(
        _WRONG_SLICE_INVOCATION_RE_TEMPLATE.format(slice="test"),
        re.MULTILINE,
    )
    if pat.search(train_body):
        errors.append(
            "scripts/evaluate_train.sh invokes eval/test/ "
            "(should invoke eval/train/)"
        )

    pat = re.compile(
        _WRONG_OUTPUT_RE_TEMPLATE.format(fname=r"scores_test\.json(?!\w)"),
    )
    if pat.search(train_body):
        errors.append(
            "scripts/evaluate_train.sh writes to scoring/scores_test.json "
            "(should write to scoring/scores_train.json)"
        )

    return errors


# Match a bash positional-arg default with a non-empty literal default,
# e.g. ``METHOD_NAME="${3:-gaussmark_200_tokens}"`` or ``method=${2:-foo}``.
# Only flags positional defaults ($1, $2, ...) since those are what callers
# override — an environment-variable default is fine (a user can always
# export a value from the outside).
_HARDCODED_POSITIONAL_DEFAULT_RE = re.compile(
    r'(\w+)\s*=\s*"?\$\{(\d+):-([A-Za-z0-9_][A-Za-z0-9_.\-]*)\}"?'
)

# Variable names we flag as "this is a method identifier, a hardcoded
# default here ships a broken single-method scorer." Matched case-
# insensitively against the LHS captured above. Other positional defaults
# (e.g. ``RESULTS_DIR="${1:-/home/user/checkpoints}"``) are fine.
_METHOD_LIKE_VAR_RE = re.compile(
    r"^(method|method_name|method_id|variant|model|model_name|algo|algorithm)$",
    re.IGNORECASE,
)


def check_no_hardcoded_method_default() -> list[str]:
    """Reject scripts/evaluate_{train,test}.sh with a hardcoded method default.

    A pattern like ``METHOD_NAME="${3:-gaussmark_200_tokens}"`` in either
    script means one invocation without the 3rd positional arg scores ONLY
    that one method. An improve-mode agent who adds a new variant gets
    silently ignored on the test side because the default never runs on
    their variant — every improve run then produces identical "test"
    numbers regardless of what the agent built.

    The fix is to enumerate methods from the outputs directory (or from
    ``scoring/scores.json`` / ``scores_train.json``) and iterate. The
    error message names the exact variable and default so the designer
    can find and fix it.
    """
    errors: list[str] = []
    scripts_dir = WORKSPACE_DIR / "scripts"
    for script_name in ("evaluate_train.sh", "evaluate_test.sh"):
        script_path = scripts_dir / script_name
        if not script_path.is_file():
            continue
        try:
            text = script_path.read_text()
        except OSError:
            continue
        for m in _HARDCODED_POSITIONAL_DEFAULT_RE.finditer(text):
            var_name = m.group(1)
            pos_index = m.group(2)
            default_value = m.group(3)
            if not _METHOD_LIKE_VAR_RE.match(var_name):
                continue
            errors.append(
                f"scripts/{script_name}: ${{{pos_index}:-{default_value}}} "
                f"assigned to '{var_name}' — hardcoded method default. "
                f"When the script runs without a {pos_index}rd positional "
                f"arg (which is always, in automated workflows), it will "
                f"score only '{default_value}' and silently skip every "
                f"other method. Replace the default with enumeration over "
                f"methods found under the outputs directory (or over keys "
                f"in scoring/scores.json), and iterate."
            )
    return errors


def check_method_keys_match_across_train_test() -> list[str]:
    """Reject mismatched method-identifier sets between train and test scores.

    For each experiment that appears in both ``scores_train.json`` and
    ``scores_test.json``, the set of method keys under
    ``experiments.<exp>.results`` must match exactly. If train scored
    methods {A, B, C} but test scored only {A}, the test column is
    incomplete for this paper — either the designer hardcoded a single
    method (see the sibling check) or the script was invoked with a
    narrowing argument.

    Error messages call out the specific missing keys so the designer
    knows which methods the test pass skipped.
    """
    errors: list[str] = []
    train_path = SCORING_DIR / "scores_train.json"
    test_path = SCORING_DIR / "scores_test.json"
    if not train_path.is_file() or not test_path.is_file():
        return errors
    try:
        train = json.loads(train_path.read_text())
        test = json.loads(test_path.read_text())
    except (OSError, json.JSONDecodeError):
        return errors

    train_exps = _extract_scores_experiments(train)
    test_exps = _extract_scores_experiments(test)

    shared = set(train_exps.keys()) & set(test_exps.keys())
    if not shared:
        return errors  # nothing to compare

    for exp in sorted(shared):
        train_methods = {k for k in train_exps[exp].keys() if not k.startswith("_")}
        test_methods = {k for k in test_exps[exp].keys() if not k.startswith("_")}
        if train_methods == test_methods:
            continue
        only_in_train = sorted(train_methods - test_methods)
        only_in_test = sorted(test_methods - train_methods)
        parts = []
        if only_in_train:
            parts.append(
                f"missing from scores_test.json: {only_in_train}"
            )
        if only_in_test:
            parts.append(
                f"missing from scores_train.json: {only_in_test}"
            )
        errors.append(
            f"experiments.{exp}: method-key sets differ between "
            f"scores_train.json and scores_test.json — "
            f"{'; '.join(parts)}. Every method scored on train must also "
            f"be scored on test (and vice versa). Re-run the evaluate_*.sh "
            f"scripts with enumeration over the full method set, not a "
            f"single hardcoded name."
        )
    return errors


# The designer step produces these six artifacts as a bundle, split across
# two phases: the four design artifacts (eval folders + scripts) are written
# during the design iterations; the two score files are produced later by
# running the scripts. Pre-design iterations should have none; during-design
# iterations should have the four design artifacts but NOT the score files
# (stubs this early mislead the validator and the runner). Post-design
# iterations must have all six.
_SPLIT_DESIGN_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("eval/train", "dir"),
    ("eval/test", "dir"),
    ("scripts/evaluate_train.sh", "file"),
    ("scripts/evaluate_test.sh", "file"),
)
_SPLIT_SCORE_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("scoring/scores_train.json", "file"),
    ("scoring/scores_test.json", "file"),
)
_SPLIT_ARTIFACTS: tuple[tuple[str, str], ...] = _SPLIT_DESIGN_ARTIFACTS + _SPLIT_SCORE_ARTIFACTS


def check_split_mode_during_design() -> list[str]:
    """During design iterations: the four design artifacts must exist; the
    two score files must NOT.

    Stage 1 (architect) writes eval/train/, eval/test/, and the two
    evaluate_*.sh scripts. Stage 2 (auditor) may improve those. Neither
    stage should produce scores_train.json or scores_test.json — those
    files are the output of running the scripts, which happens in Stage 3.

    If a design artifact is missing, the design iteration didn't finish —
    fail so the swarm surfaces the gap. If a score file exists, someone
    dropped a stub that will mislead the post-design validator and confuse
    the runner — fail and name the file.
    """
    errors: list[str] = []

    missing_design = [
        (rel, kind) for rel, kind in _SPLIT_DESIGN_ARTIFACTS
        if not ((WORKSPACE_DIR / rel).is_dir() if kind == "dir"
                else (WORKSPACE_DIR / rel).is_file())
    ]
    if missing_design:
        labels = "\n        ".join(
            f"- {rel}/" if kind == "dir" else f"- {rel}"
            for rel, kind in missing_design
        )
        errors.append(
            f"design-phase artifacts missing — a design iteration must "
            f"leave all four of eval/train/, eval/test/, "
            f"scripts/evaluate_train.sh, scripts/evaluate_test.sh in place "
            f"so the runner iteration has scripts to execute. Missing:\n"
            f"        {labels}"
        )

    present_scores = [
        rel for rel, _kind in _SPLIT_SCORE_ARTIFACTS
        if (WORKSPACE_DIR / rel).is_file()
    ]
    if present_scores:
        labels = ", ".join(present_scores)
        errors.append(
            f"score file(s) exist too early: {labels}. Score files are "
            f"produced by running scripts/evaluate_{{train,test}}.sh in the "
            f"runner iteration, not written by hand during design. A stub "
            f"or skeleton here misleads the post-design validator and the "
            f"runner. Delete the file(s) and let the scripts produce them."
        )

    return errors


def check_split_mode_complete() -> list[str]:
    """Require all six split-mode artifacts to exist once the designer ran.

    Trigger (signal that the designer has run): any of ``eval/train/``,
    ``eval/test/``, ``scripts/evaluate_train.sh``, ``scripts/evaluate_test.sh``,
    ``scoring/scores_train.json``, ``scoring/scores_test.json`` exists. All
    six are produced by the designer step and none by earlier reproducer
    iterations, so their presence is a reliable "designer ran" marker.

    Requirement: once the trigger fires, all six must exist. A missing
    artifact means the designer started but didn't finish (timeout, crash,
    incorrect ``action.yaml``) or never produced a complete split. The
    error message names exactly which artifacts are missing so the agent
    sees what to add.

    Pre-designer iterations: none of the six exist → trigger doesn't fire
    → check is silent. The implicit before/after-designer branch is
    signaled by filesystem state.
    """
    def _exists(rel: str, kind: str) -> bool:
        p = WORKSPACE_DIR / rel
        return p.is_dir() if kind == "dir" else p.is_file()

    present = [(rel, kind) for rel, kind in _SPLIT_ARTIFACTS if _exists(rel, kind)]
    missing = [(rel, kind) for rel, kind in _SPLIT_ARTIFACTS if not _exists(rel, kind)]

    if not present or not missing:
        # Either nothing triggered (pre-designer), or everything is present.
        return []

    def _label(rel: str, kind: str) -> str:
        return f"{rel}/" if kind == "dir" else rel

    present_list = ", ".join(_label(r, k) for r, k in present)
    missing_list = "\n        ".join(f"- {_label(r, k)}" for r, k in missing)
    return [
        f"split-mode artifacts incomplete. The designer step produces all "
        f"six together — a half-built split means the designer didn't "
        f"finish and the paper cannot serve as a train/test gym entry.\n"
        f"      Present: {present_list}\n"
        f"      Missing:\n        {missing_list}\n"
        f"      Fix: either re-run the designer iteration to produce the "
        f"missing artifacts, or remove the partial artifacts if this paper "
        f"is intentionally not going through the split workflow."
    ]


def check_at_least_one_train_test_differs() -> list[str]:
    """Reject papers whose train and test scores are byte-identical everywhere.

    When every shared experiment has identical method-score dicts between
    ``scores_train.json`` and ``scores_test.json``, the test slice did not
    actually run against different data — it's either a direct copy from
    train or the script fell through to the train code path. The "test"
    column then provides zero anti-shortcut signal: any agent whose scores
    match the reference on train will match identically on test too.

    Rule: at least ONE shared experiment must have a method-score dict
    that differs between the two files. If ALL shared experiments are
    byte-identical, fail.

    This is intentionally more lenient than the prompt checklist (which
    asks for separation on EVERY experiment). A paper where 1 of 2
    experiments has real separation still ships a meaningful partial
    signal; a paper where 2 of 2 are identical ships nothing. Partial
    passes here, total-failure fails here.
    """
    train_path = SCORING_DIR / "scores_train.json"
    test_path = SCORING_DIR / "scores_test.json"
    if not train_path.is_file() or not test_path.is_file():
        return []
    try:
        train = json.loads(train_path.read_text())
        test = json.loads(test_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    train_exps = _extract_scores_experiments(train)
    test_exps = _extract_scores_experiments(test)

    shared = set(train_exps.keys()) & set(test_exps.keys())
    if not shared:
        return []  # no shared experiments — a different check covers this

    identical_count = 0
    for exp in shared:
        # Compare method-result dicts via sorted-key JSON so dict ordering
        # doesn't cause spurious mismatches, but value precision matters.
        train_methods = {
            k: v for k, v in train_exps[exp].items() if not k.startswith("_")
        }
        test_methods = {
            k: v for k, v in test_exps[exp].items() if not k.startswith("_")
        }
        if json.dumps(train_methods, sort_keys=True) != json.dumps(
            test_methods, sort_keys=True
        ):
            return []  # at least one experiment differs — pass
        identical_count += 1

    return [
        f"scores_train.json and scores_test.json have byte-identical "
        f"method scores across all {identical_count} shared experiment(s) "
        f"— the test slice never ran against different data from train, "
        f"so the test column provides zero anti-shortcut signal. Re-run "
        f"scripts/evaluate_test.sh against the held-out test data, or "
        f"fix the script so it invokes the eval/test/ code path "
        f"(evaluate_test.sh must point at eval/test/ and write to "
        f"scoring/scores_test.json; see the prompt's pre-done checklist)."
    ]


def check_primary_metric_differs_train_test() -> list[str]:
    """Reject papers whose primary metric is identical on train and test.

    The coarser ``check_at_least_one_train_test_differs`` compares entire
    method dicts. A single differing secondary metric passes that check
    even when the primary metric — the one the gym ranks agents by — is
    identical across slices. This happens when the primary metric is
    global (per-run, not per-sample) and the data split only varies
    sample-level outputs.

    Uses ``primary_metric`` from ``reference.json`` (a validated,
    required field) to identify the metric that matters per experiment.
    """
    ref_path = SCORING_DIR / "reference.json"
    train_path = SCORING_DIR / "scores_train.json"
    test_path = SCORING_DIR / "scores_test.json"
    if not all(p.is_file() for p in (ref_path, train_path, test_path)):
        return []
    try:
        ref = json.loads(ref_path.read_text())
        train = json.loads(train_path.read_text())
        test = json.loads(test_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    ref_exps = ref.get("experiments", {})
    train_exps = _extract_scores_experiments(train)
    test_exps = _extract_scores_experiments(test)

    errors: list[str] = []
    for exp_name in sorted(set(train_exps) & set(test_exps) & set(ref_exps)):
        primary = ref_exps[exp_name].get("primary_metric")
        if not primary:
            continue

        train_methods = train_exps[exp_name]
        test_methods = test_exps[exp_name]
        shared_methods = [
            m for m in sorted(set(train_methods) & set(test_methods))
            if not m.startswith("_")
        ]
        if not shared_methods:
            continue

        all_identical = True
        for method in shared_methods:
            train_val = train_methods[method].get(primary) if isinstance(train_methods[method], dict) else None
            test_val = test_methods[method].get(primary) if isinstance(test_methods[method], dict) else None
            if train_val != test_val:
                all_identical = False
                break

        if all_identical:
            sample_method = shared_methods[0]
            sample_val = (
                train_methods[sample_method].get(primary)
                if isinstance(train_methods[sample_method], dict) else None
            )
            errors.append(
                f"experiments.{exp_name}: primary metric '{primary}' is "
                f"identical between scores_train.json and scores_test.json "
                f"for every method (e.g., {sample_method}={sample_val}). "
                f"The train/test split does not differentiate the metric "
                f"the gym ranks agents by — consider a metric split or a "
                f"different evaluation axis instead of a data split."
            )

    return errors


def _classify_deviation(paper_val: float, repro_val: float) -> list[str]:
    """Return reasons ``repro_val`` is wildly off from ``paper_val``, else []."""
    kinds: list[str] = []
    # Sign flip — the strongest signal that the reproduction broke.
    if paper_val != 0 and repro_val != 0 and (paper_val > 0) != (repro_val > 0):
        kinds.append("sign flip")
    # 2x magnitude gap, but only when both sides are non-trivially nonzero
    # (|val| >= 0.01) so tiny-value noise doesn't trip us.
    ap, ar = abs(paper_val), abs(repro_val)
    if ap >= 0.01 and ar >= 0.01:
        ratio = max(ap, ar) / min(ap, ar)
        if ratio >= 2.0:
            kinds.append(f"{ratio:.2f}x off in magnitude")
    # Absolute gap >= 0.3 for metrics that live in the bounded unit-interval
    # range [-1, 1] (R^2, accuracy, F1, correlation, etc.). Picks up the
    # 0.99 -> 0.60 case that ratio alone would miss.
    if -1.5 <= paper_val <= 1.5 and -1.5 <= repro_val <= 1.5:
        if abs(paper_val - repro_val) >= 0.3:
            kinds.append(f"absolute gap {abs(paper_val - repro_val):.2f}")
    seen: set[str] = set()
    out: list[str] = []
    for k in kinds:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def check_reproduction_not_wildly_off_from_paper(
    scores_files: tuple[str, ...],
) -> list[str]:
    """Reject when any of ``scores_files`` deviates wildly from reference.json.

    The caller chooses which scores files to compare. Before designer_stage_3
    only ``scores.json`` exists and is the reproducer's paper-level output,
    so pass ``("scores.json",)``. After designer_stage_3 the runner has
    produced the split files too, so pass
    ``("scores.json", "scores_train.json", "scores_test.json")``.

    For each ``(file, experiment, method)`` triple where the scores file
    and ``reference.json.results`` both carry a numeric value on the
    experiment's ``primary_metric``, fails when the reproduced value is:

    * sign-flipped relative to the paper, or
    * >= 2x off in magnitude (non-trivial values), or
    * >= 0.3 absolute gap when both lie in [-1.5, 1.5] (R^2 / accuracy).

    Error messages name the specific scores file, so the agent knows
    exactly which file to fix.
    """
    ref_path = SCORING_DIR / "reference.json"
    if not ref_path.is_file():
        return []
    try:
        ref = json.loads(ref_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    ref_exps = ref.get("experiments", {}) or {}
    if not ref_exps:
        return []

    errors: list[str] = []
    for scores_filename in scores_files:
        scores_path = SCORING_DIR / scores_filename
        if not scores_path.is_file():
            continue
        try:
            scores = json.loads(scores_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        scores_exps = _extract_scores_experiments(scores)

        for exp_name in sorted(set(ref_exps) & set(scores_exps)):
            primary = ref_exps[exp_name].get("primary_metric")
            if not primary:
                continue
            ref_results = ref_exps[exp_name].get("results") or {}
            score_methods = scores_exps[exp_name]
            if not isinstance(score_methods, dict):
                continue
            for method_name in sorted(set(ref_results) & set(score_methods)):
                if method_name.startswith("_"):
                    continue
                ref_method = ref_results[method_name]
                score_method = score_methods[method_name]
                if not (isinstance(ref_method, dict)
                        and isinstance(score_method, dict)):
                    continue
                paper_val = ref_method.get(primary)
                repro_val = score_method.get(primary)
                if isinstance(paper_val, bool) or isinstance(repro_val, bool):
                    continue
                if not (isinstance(paper_val, (int, float))
                        and isinstance(repro_val, (int, float))):
                    continue
                reasons = _classify_deviation(float(paper_val), float(repro_val))
                if not reasons:
                    continue
                errors.append(
                    f"{scores_filename}: experiments.{exp_name}.{method_name}"
                    f": primary metric '{primary}' — paper reports "
                    f"{paper_val} but {scores_filename} has {repro_val} "
                    f"({', '.join(reasons)}). This gap is too large to "
                    f"accept — one of the two sides is wrong. Either the "
                    f"value in {scores_filename} is wrong (broken training, "
                    f"wrong data split, missing preprocessing, wrong unit, "
                    f"etc.) OR the paper-reported value in reference.json "
                    f"is wrong (reference.json was written by an earlier "
                    f"agent and is not verified). Fix whichever is off — "
                    f"edit either {scores_filename} (re-run the method) "
                    f"or reference.json (correct the transcribed value). "
                    f"Re-read the paper if you are sure your reproduction "
                    f"is good. Do not claim the milestone until this "
                    f"experiment's numbers are consistent."
                )
    return errors


def check_primary_metric_discriminates_on_train() -> list[str]:
    """Reject papers whose primary metric is identical across methods on train.

    The train slice is what a future agent optimizes against during
    development.  If every method on train gets the same primary-metric
    value (the metric saturates — e.g., ``coverage = 1.0`` for all
    methods), the train slice gives that agent no ranking signal: every
    approach they try looks equally good.  The gym cannot measure
    improvement and the anti-shortcut design collapses.

    Skips experiments with fewer than 2 methods that report a numeric
    primary-metric value.
    """
    ref_path = SCORING_DIR / "reference.json"
    train_path = SCORING_DIR / "scores_train.json"
    if not all(p.is_file() for p in (ref_path, train_path)):
        return []
    try:
        ref = json.loads(ref_path.read_text())
        train = json.loads(train_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    ref_exps = ref.get("experiments", {})
    train_exps = _extract_scores_experiments(train)

    errors: list[str] = []
    for exp_name in sorted(set(train_exps) & set(ref_exps)):
        primary = ref_exps[exp_name].get("primary_metric")
        if not primary:
            continue

        methods = train_exps[exp_name]
        if not isinstance(methods, dict):
            continue

        values: list[tuple[str, float]] = []
        for name, data in methods.items():
            if name.startswith("_") or not isinstance(data, dict):
                continue
            v = data.get(primary)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values.append((name, float(v)))

        if len(values) < 2:
            continue
        if len({v for _, v in values}) > 1:
            continue

        # All values identical across all reporting methods.
        sample_val = values[0][1]
        methods_shown = ", ".join(f"{n}={v}" for n, v in values[:4])
        errors.append(
            f"experiments.{exp_name}: primary metric '{primary}' is "
            f"identical ({sample_val}) across every method on the train "
            f"slice ({methods_shown}). The train slice cannot rank methods "
            f"on this metric, so a future agent gets no feedback signal "
            f"while developing against scores_train.json. "
            f"Fix: either (1) change primary_metric on this experiment to "
            f"a metric whose values already differ across methods on "
            f"train, or (2) adjust the experiment so '{primary}' actually "
            f"spreads methods out (harder inputs, a tighter operating "
            f"point, a stricter scoring protocol). Do NOT simply delete "
            f"the experiment — find a metric that discriminates."
        )

    return errors


_SHARED_PATTERN = re.compile(r"(?<![A-Za-z0-9_/])/?shared/(?:datasets|models|hf_cache)\b")


def check_no_shared_references() -> list[str]:
    """Check that committed code does NOT reference shared/ paths.

    `shared/` is a local cache on the swarm machine and does not exist
    on a fresh clone. Any reference in committed code (method/, eval/,
    baseline/, scripts/ including download.sh) breaks reproducibility.
    """
    errors: list[str] = []
    for dirname in ("method", "eval", "baseline", "scripts"):
        dirpath = WORKSPACE_DIR / dirname
        if not dirpath.is_dir():
            continue
        for fpath in dirpath.rglob("*"):
            if not fpath.is_file():
                continue
            if fpath.suffix not in (".py", ".sh"):
                continue
            try:
                lines = fpath.read_text().splitlines()
            except Exception:
                continue
            rel = fpath.relative_to(WORKSPACE_DIR)
            for lineno, line in enumerate(lines, 1):
                if _SHARED_PATTERN.search(line):
                    errors.append(
                        f"{rel}:{lineno}: references shared/ "
                        f"(won't exist on fresh clone — use data/downloads/ instead)"
                    )
    return errors


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Validate workspace")
    parser.add_argument("--reference-only", action="store_true",
                        help="Only validate reference.json")
    parser.add_argument("--compare", action="store_true",
                        help="Compare scores against reference")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--during-design", action="store_true",
                      help="Activate during-design checks (the four design "
                           "artifacts — eval/train/, eval/test/, the two "
                           "evaluate_*.sh scripts — must exist; the two "
                           "score files must NOT exist yet). On for "
                           "designer stage 1 and stage 2.")
    mode.add_argument("--post-design", action="store_true",
                      help="Activate post-design checks (all six split "
                           "artifacts must exist, scores_train/test must be "
                           "populated and differ). On after the final "
                           "designer stage, for the packager, and for the "
                           "verifier.")
    mode.add_argument("--post-reproducer", action="store_true",
                      help="Activate post-reproducer checks: scores.json "
                           "must not deviate wildly from reference.json "
                           "(catches broken reproductions that would poison "
                           "every downstream agent-vs-source comparison). "
                           "On at the end of each reproducer iteration.")
    args = parser.parse_args()

    ok = True

    # 1. Scoring: reference.json
    print("── Scoring ──")
    ref_path = SCORING_DIR / "reference.json"
    if not ref_path.exists():
        print("  ✗ scoring/reference.json not found")
        sys.exit(1)

    ref_data = json.loads(ref_path.read_text())
    ref_errors = validate_reference(ref_data)

    if ref_errors:
        print(f"  reference.json — {len(ref_errors)} error(s):")
        for e in ref_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        n_exp = len(ref_data.get("experiments", {}))
        n_methods = sum(len(e.get("results", {}))
                        for e in ref_data.get("experiments", {}).values())
        print(f"  reference.json — OK ({n_exp} experiments, {n_methods} methods)")

    # 2. Scoring: scores.json (and scores_train.json / scores_test.json if
    # present — those are produced by the designer iteration; earlier
    # iterations don't create them and are not checked for them).
    if not args.reference_only:
        for filename in ("scores.json", "scores_train.json", "scores_test.json"):
            scores_path = SCORING_DIR / filename
            if not scores_path.exists():
                # scores.json absence is surfaced; the designer-produced
                # split files are silently skipped when missing so earlier
                # iterations don't get spurious noise.
                if filename == "scores.json":
                    print(f"  {filename} — not found (skipping)")
                continue

            scores_data = json.loads(scores_path.read_text())
            scores_errors = validate_scores(scores_data, ref_data)

            if scores_errors:
                print(f"  {filename} — {len(scores_errors)} error(s):")
                for e in scores_errors:
                    print(f"    ✗ {e}")
                ok = False
            else:
                n_exp = len(scores_data)
                n_methods = sum(len(m) for m in scores_data.values()
                                if isinstance(m, dict))
                print(f"  {filename} — OK ({n_exp} experiments, {n_methods} methods)")

            if args.compare and not scores_errors and filename == "scores.json":
                compare_scores(scores_data, ref_data)

        # 2b. Split-mode artifacts. Three modes:
        #   - default (pre-design reproducer iterations): ignore all six
        #     artifacts; they aren't expected yet.
        #   - --during-design (stage 1, stage 2): the four design artifacts
        #     must exist; the two score files must NOT exist yet.
        #   - --post-design (stage 3, packager, verifier): all six must
        #     exist, and downstream score-differentiation checks apply.
        if args.during_design:
            split_errors = check_split_mode_during_design()
            if split_errors:
                print(f"  split-mode artifacts — {len(split_errors)} error(s):")
                for e in split_errors:
                    print(f"    ✗ {e}")
                ok = False
            else:
                print("  split-mode artifacts — OK (design artifacts present, no early score files)")
        elif args.post_design:
            split_errors = check_split_mode_complete()
            if split_errors:
                print(f"  split-mode artifacts — {len(split_errors)} error(s):")
                for e in split_errors:
                    print(f"    ✗ {e}")
                ok = False
            elif any(
                (WORKSPACE_DIR / rel).is_dir() if kind == "dir"
                else (WORKSPACE_DIR / rel).is_file()
                for rel, kind in _SPLIT_ARTIFACTS
            ):
                print("  split-mode artifacts — OK (all 6 present)")

        # Scores ↔ reference deviation check. Two branches by design:
        #   --post-reproducer: only scores.json exists and is authoritative,
        #                      so check that one file.
        #   --post-design:     designer_stage_3 has produced the split files,
        #                      so check all three.
        if args.post_reproducer:
            deviation_files: tuple[str, ...] = ("scores.json",)
        elif args.post_design:
            deviation_files = ("scores.json", "scores_train.json", "scores_test.json")
        else:
            deviation_files = ()

        if deviation_files:
            deviation_errors = check_reproduction_not_wildly_off_from_paper(
                deviation_files
            )
            if deviation_errors:
                print(
                    f"  scores ↔ paper consistency — "
                    f"{len(deviation_errors)} error(s):"
                )
                for e in deviation_errors:
                    print(f"    ✗ {e}")
                ok = False
            else:
                print(
                    f"  scores ↔ paper consistency — OK "
                    f"(checked {', '.join(deviation_files)})"
                )

    # 3. Import separation
    print("\n── Structure ──")
    sep_errors = check_import_separation()
    if sep_errors:
        print(f"  import separation — {len(sep_errors)} error(s):")
        for e in sep_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        print("  import separation — OK (eval/ and data/ do not import from method/)")

    # Soft warning: heuristic for "eval/ regenerates the experiment instead
    # of reading method outputs". Does NOT fail validation — only flags so
    # the agent can take a closer look. Prone to false positives.
    detached_warnings = check_eval_reads_method_outputs()
    if detached_warnings:
        print(f"  eval reads method outputs — {len(detached_warnings)} warning(s):")
        for w in detached_warnings:
            print(f"    ⚠ {w}")
    elif (WORKSPACE_DIR / "eval").is_dir():
        print("  eval reads method outputs — OK (no regen-without-read pattern)")

    # 4. Imports don't land on gitignored files (would break on fresh clone)
    ignored_errors = check_imports_not_gitignored()
    if ignored_errors:
        print(f"  gitignored imports — {len(ignored_errors)} error(s):")
        for e in ignored_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        print("  gitignored imports — OK (all eval/method/baseline imports resolve to tracked files)")

    # 5. eval/train and eval/test are independent (no cross-imports)
    indep_errors = check_train_test_independent()
    if indep_errors:
        print(f"  eval/train ↔ eval/test independence — {len(indep_errors)} error(s):")
        for e in indep_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        print("  eval/train ↔ eval/test independence — OK")

    # 6. No shared/ references in committed code (would break on fresh clone)
    shared_errors = check_no_shared_references()
    if shared_errors:
        print(f"  shared/ references — {len(shared_errors)} error(s):")
        for e in shared_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        print("  shared/ references — OK (no shared/ paths in committed code)")

    # 7. scripts/evaluate_{train,test}.sh point at the right slice
    rewire_errors = check_train_test_scripts_rewired()
    if rewire_errors:
        print(f"  evaluate_{{train,test}}.sh rewiring — {len(rewire_errors)} error(s):")
        for e in rewire_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        # Print only if both scripts exist (silent skip otherwise).
        if ((WORKSPACE_DIR / "scripts" / "evaluate_train.sh").is_file()
            and (WORKSPACE_DIR / "scripts" / "evaluate_test.sh").is_file()):
            print("  evaluate_{train,test}.sh rewiring — OK")

    # 8. No hardcoded method-name positional default in evaluate_*.sh
    hardcoded_errors = check_no_hardcoded_method_default()
    if hardcoded_errors:
        print(f"  evaluate_{{train,test}}.sh method defaults — {len(hardcoded_errors)} error(s):")
        for e in hardcoded_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        if ((WORKSPACE_DIR / "scripts" / "evaluate_train.sh").is_file()
            or (WORKSPACE_DIR / "scripts" / "evaluate_test.sh").is_file()):
            print("  evaluate_{train,test}.sh method defaults — OK (no hardcoded method name)")

    # 9. Method-key sets match between scores_train.json and scores_test.json
    method_match_errors = check_method_keys_match_across_train_test()
    if method_match_errors:
        print(f"  scores_train.json ↔ scores_test.json method keys — {len(method_match_errors)} error(s):")
        for e in method_match_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        if ((SCORING_DIR / "scores_train.json").is_file()
            and (SCORING_DIR / "scores_test.json").is_file()):
            print("  scores_train.json ↔ scores_test.json method keys — OK")

    # 10. At least one shared experiment must differ between train and test
    differs_errors = check_at_least_one_train_test_differs()
    if differs_errors:
        print(f"  scores_train.json ↔ scores_test.json value differentiation — {len(differs_errors)} error(s):")
        for e in differs_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        if ((SCORING_DIR / "scores_train.json").is_file()
            and (SCORING_DIR / "scores_test.json").is_file()):
            print("  scores_train.json ↔ scores_test.json value differentiation — OK")

    # 11. Primary metric must differ between train and test
    primary_errors = check_primary_metric_differs_train_test()
    if primary_errors:
        print(f"  primary metric train/test differentiation — {len(primary_errors)} error(s):")
        for e in primary_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        if ((SCORING_DIR / "scores_train.json").is_file()
            and (SCORING_DIR / "scores_test.json").is_file()):
            print("  primary metric train/test differentiation — OK")

    # 12. Primary metric must discriminate between methods on train
    disc_errors = check_primary_metric_discriminates_on_train()
    if disc_errors:
        print(f"  primary metric discriminates methods on train — {len(disc_errors)} error(s):")
        for e in disc_errors:
            print(f"    ✗ {e}")
        ok = False
    else:
        if (SCORING_DIR / "scores_train.json").is_file():
            print("  primary metric discriminates methods on train — OK")

    # Summary
    print()
    if ok:
        print("All checks passed.")
    else:
        print("FAILED — fix the errors above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
