"""Publish a completed run to its per-paper GitHub repository as a branch.

Opt-in: nothing here contacts GitHub unless ``publish.enabled`` is true
in the evaluation config and ``publish.github_org`` names the
organisation that holds the per-paper repositories (as created by
Swarm). A run goes to branch ``<mode>-<profile>`` of
``<org>/<arxiv_id>-<title-slug>``; locked normalization coefficients go
to branch ``normalization`` of the same repository.

Key invariants, tested in tests/test_git_publish_safety.py:

- Never pushes to ``main`` or ``master`` (under any code path).
- Never uses ``--all`` / ``--mirror`` / bare ``--force``.
- Push refspec is always explicit ``<branch>:<branch>``.
- Force-push uses ``--force-with-lease=<branch>:<sha>`` scoped to the
  target branch only, so the lease cannot overwrite ``main``.
- The local staging repo is created with ``git init -b <branch>`` so a
  ``main`` branch *never exists locally*: there is nothing to overwrite
  even if the push refspec were wrong.
- Branch name is asserted ∉ {main, master, HEAD, ""} before any git
  operation.
- ``_safe_push()`` is the only function in the package that invokes
  ``git push``.
- The access token is never placed on a command line: git receives it
  as an HTTP header through its environment.

Sanitization: every text file is rewritten to redact credential
patterns *before* staging. A pre-commit hook in the staging repository
then blocks anything that slipped through the redaction.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

log = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

# Swarm's config, read only as a fallback source of ``github_org`` when
# the evaluation config does not set ``publish.github_org``.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SWARM_CONFIG_PATH = _REPO_ROOT / "infrastructure" / "swarm" / "swarm" / "config" / "config.yaml"

# ``skipped_reason`` of a publish attempt made while publishing is off.
DISABLED_REASON = "publishing disabled (publish.enabled is false)"

# 90 MB — leave 10 MB buffer below GitHub's 100 MB blob limit. Files larger
# than this are split (jsonl) along line boundaries into ``.partNN`` chunks.
_TRAJECTORY_SPLIT_BYTES = 90 * 1024 * 1024

# Hard cap for non-jsonl files: anything bigger gets dropped from the push
# with a logged warning. Catches model-weight checkpoints (.pt, .safetensors,
# .bin) that agents sometimes download into method/ — those have no business
# in git anyway.
_NON_JSONL_FILE_MAX_BYTES = 90 * 1024 * 1024

_GIT_TIMEOUT = 300
_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "scientistworld-eval",
    "GIT_COMMITTER_NAME": "scientistworld-eval",
    "GIT_AUTHOR_EMAIL": "eval@scientistworld",
    "GIT_COMMITTER_EMAIL": "eval@scientistworld",
}

_FORBIDDEN_BRANCH_NAMES = frozenset({"main", "master", "HEAD", "", "head"})

# Folders/files inside an eval env_dir that we publish to the branch.
# Anything not listed (or matched by ``_EXCLUDE_GLOBS``) is dropped.
_INCLUDE_PATHS = (
    "method/",
    "scoring/",
    "trajectory/",
    "idea/",
    "environment/container.def",
    "reflection.md",
    ".evaluation.json",
)

# Hard excludes — even if listed, these never make it through.
_EXCLUDE_GLOBS = (
    "*.sif", "*.img", "*.tar", "*.tar.gz", "*.tar.xz",
    "**/__pycache__/**", "**/.uv_cache/**", "**/.cache/**",
    "**/node_modules/**", "**/site-packages/**", "**/.venv/**",
    "**/wheels/**", "**/local-packages/**",
)

# Credential redaction patterns. Applied to trajectory files (and any
# other text file in the staging tree) before staging. Each tuple is
# (compiled_regex, replacement). We keep the prefix so a reader can
# tell what *kind* of secret was redacted.
_REDACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # GitHub
    (re.compile(r"ghp_[A-Za-z0-9]{30,}"), "ghp_REDACTED"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{60,}"), "github_pat_REDACTED"),
    # HuggingFace
    (re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}"), "hf_REDACTED"),
    # Model-provider and gateway API keys, by their key prefix
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{40,}"), "sk-ant-REDACTED"),
    (re.compile(r"sk-or-v1-[A-Za-z0-9]{40,}"), "sk-or-v1-REDACTED"),
    (re.compile(r"sk-kimi-[A-Za-z0-9]{40,}"), "sk-kimi-REDACTED"),
    (re.compile(r"sk-cp-[A-Za-z0-9_\-]{40,}"), "sk-cp-REDACTED"),
    (re.compile(r"sk-proj-[A-Za-z0-9_\-]{40,}"), "sk-proj-REDACTED"),
    # Anchored like the pre-commit hook's patterns, not with \b: "_" is a
    # word character, so \b misses "x_tp-..." which the hook still rejects.
    (re.compile(r"(?<![A-Za-z0-9])tp-[a-z0-9]{40,}"), "tp-REDACTED"),
    # Dotted key format: 32 hex chars + "." + 16 alphanumerics
    (re.compile(r"[a-f0-9]{32}\.[A-Za-z0-9]{16}"), "<key>.REDACTED"),
    # Bearer / Authorization headers
    (re.compile(r"(?i)Bearer\s+[A-Za-z0-9_\.\-]{20,}"), "Bearer REDACTED"),
    # Basic-auth URL (https://user:token@host)
    (re.compile(r"://[^/\s\"'@:]+:[^/\s\"'@]+@"), "://user:REDACTED@"),
)


# Pre-commit hook installed in the staging repository (same gates as the
# hook Swarm installs in its workspaces, with the credential pattern
# extended to gateway keys). A second line of defence under the
# in-Python redaction.
_PRE_COMMIT_HOOK = r"""#!/bin/bash
# Installed by the Scientist World evaluation publish step. Four gates:
#   1. Credential patterns (tokens, basic-auth URLs) — block
#   2. Per-file size > 95 MB — block (GitHub's hard blob limit is 100 MB)
#   3. Path patterns that are never legitimately committed — block
#   4. Forbidden branch — block any commit that lands on main/master
# Bypass (NOT RECOMMENDED): git commit --no-verify

# ── Gate 4: branch sanity (FIRST — cheap and catastrophic if missed) ─────────
# Use symbolic-ref so this works pre-first-commit (where rev-parse
# --abbrev-ref returns the literal "HEAD"). For detached HEAD,
# symbolic-ref fails and we get an empty string — also blocked.
current_branch=$(git symbolic-ref --short HEAD 2>/dev/null || echo "")
case "$current_branch" in
    main|master|"")
        echo "" >&2
        echo "pre-commit: refusing to commit on branch '$current_branch'." >&2
        echo "The evaluation publish step never writes to main/master/detached." >&2
        exit 1
        ;;
esac

# ── Gate 1: credentials ──────────────────────────────────────────────────────
patterns='ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,}|(^|[^A-Za-z0-9])hf_[A-Za-z0-9]{30,}|sk-ant-[A-Za-z0-9_-]{40,}|sk-or-v1-[A-Za-z0-9]{40,}|sk-kimi-[A-Za-z0-9]{40,}|sk-cp-[A-Za-z0-9_-]{40,}|sk-proj-[A-Za-z0-9_-]{40,}|(^|[^A-Za-z0-9])tp-[a-z0-9]{40,}|[a-f0-9]{32}\.[A-Za-z0-9]{16}|https?://[^/[:space:]\"'"'"'@]+:[^/[:space:]\"'"'"'@]+@'

if hits=$(git diff --cached --no-color -U0 -- 2>/dev/null | grep -E '^\+' | grep -E "$patterns"); then
    echo "" >&2
    echo "pre-commit: staged changes contain credential-like patterns:" >&2
    echo "$hits" | sed -E 's/(ghp_|github_pat_|hf_|sk-ant-|sk-or-v1-|sk-kimi-|sk-cp-|sk-proj-|tp-)[A-Za-z0-9_-]{8}[A-Za-z0-9_-]+/\1XXXXXXXX_REDACTED/g; s/:[^@[:space:]]+@/:XXXXXX@/g' | head -10 >&2
    echo "" >&2
    echo "Remove the secret from the staged changes, then re-stage and commit." >&2
    exit 1
fi

# ── Gate 2: per-file size > 95 MB ────────────────────────────────────────────
MAX_BYTES=$((95 * 1024 * 1024))
oversized=""
while IFS= read -r -d '' f; do
    [ -f "$f" ] || continue
    sz=$(stat -c '%s' "$f" 2>/dev/null || stat -f '%z' "$f" 2>/dev/null || echo 0)
    if [ "$sz" -gt "$MAX_BYTES" ]; then
        mb=$((sz / 1024 / 1024))
        oversized="${oversized}  ${mb} MB  ${f}"$'\n'
    fi
done < <(git diff --cached --name-only --diff-filter=ACMR -z 2>/dev/null)

if [ -n "$oversized" ]; then
    echo "" >&2
    echo "pre-commit: staged file(s) exceed 95 MB (GitHub's blob limit is 100 MB):" >&2
    printf "%s" "$oversized" >&2
    exit 1
fi

# ── Gate 3: never-legitimate path patterns ───────────────────────────────────
bad_paths=""
while IFS= read -r -d '' f; do
    case "$f" in
        */site-packages/*|*/node_modules/*|*/__pycache__/*|*.egg-info/*|*/.venv/*|*/.tox/*) bad_paths="${bad_paths}  ${f}"$'\n' ;;
        *.sif|*.img|*.tar|*.tar.gz|*.tar.xz)                                                bad_paths="${bad_paths}  ${f}"$'\n' ;;
        *_extracted/*|*/*_extracted/*)                                                       bad_paths="${bad_paths}  ${f}"$'\n' ;;
        .pip_cache/*|*/.pip_cache/*|.hf_cache/*|*/.hf_cache/*|.huggingface/*|*/.huggingface/*) bad_paths="${bad_paths}  ${f}"$'\n' ;;
    esac
done < <(git diff --cached --name-only --diff-filter=ACMR -z 2>/dev/null)

if [ -n "$bad_paths" ]; then
    echo "" >&2
    echo "pre-commit: staged path(s) match never-legitimate patterns:" >&2
    printf "%s" "$bad_paths" | head -20 >&2
    exit 1
fi

exit 0
"""


# ── Public API ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PublishResult:
    paper_id: str
    profile_name: str
    mode: str
    branch: str
    repo_full_name: str
    pushed: bool
    skipped_reason: str | None = None
    files_committed: int = 0
    bytes_committed: int = 0
    trajectories_split: int = 0


def publish_enabled(cfg: dict | None = None) -> bool:
    """True when the evaluation config opts in to publishing."""
    if cfg is None:
        from evaluation import config as _config  # lazy: avoid import cycle
        cfg = _config.load()
    return bool((cfg.get("publish") or {}).get("enabled"))


def publish(
    env_dir: Path | str,
    paper_id: str,
    profile_name: str,
    mode: str,
    dry_run: bool = False,
) -> PublishResult:
    """Publish one completed run to its per-paper repository branch.

    Returns ``PublishResult.pushed=False`` (with ``skipped_reason``)
    when publishing is disabled or the run is not eligible, so callers
    do not need to handle exceptions for the common skip paths.
    """
    env_dir = Path(env_dir)
    profile_name = _canonical_profile(profile_name)
    branch = _make_branch_name(mode, profile_name)
    _assert_safe_branch(branch)

    if not publish_enabled():
        return PublishResult(paper_id, profile_name, mode, branch, "",
                             pushed=False, skipped_reason=DISABLED_REASON)

    meta = _read_meta(env_dir)
    if not meta:
        return PublishResult(paper_id, profile_name, mode, branch, "",
                             pushed=False, skipped_reason="no .evaluation.json")
    if meta.get("status") != "completed":
        return PublishResult(paper_id, profile_name, mode, branch, "",
                             pushed=False,
                             skipped_reason=f"status={meta.get('status')!r} (need completed)")

    org = _github_org()
    token = _get_github_token()
    if not token:
        return PublishResult(paper_id, profile_name, mode, branch, "",
                             pushed=False,
                             skipped_reason="no GitHub token in ~/.git-credentials")

    source_dir = env_dir.parent / "source"
    if dry_run:
        # In dry-run, do the cheap title-based guess only and skip the
        # network search. Branch + file count is what matters.
        repo_name = _repo_name_from_paper(paper_id, source_dir)
        full_name = f"{org}/{repo_name}"
        files = sum(1 for _ in _iter_publishable_files(env_dir))
        return PublishResult(paper_id, profile_name, mode, branch, full_name,
                             pushed=False, skipped_reason="dry-run",
                             files_committed=files)

    repo_name = _resolve_repo_name(paper_id, org, token, source_dir)
    if repo_name is None:
        return PublishResult(paper_id, profile_name, mode, branch, "",
                             pushed=False,
                             skipped_reason=f"could not locate per-paper "
                                            f"repo for {paper_id} under {org}/ "
                                            "(neither title-guess nor search matched)")
    full_name = f"{org}/{repo_name}"

    staging = _stage_workspace(env_dir, paper_id, profile_name, mode, meta,
                               branch, repo_name=repo_name)
    try:
        n_files, n_bytes, n_split = _commit_staging(staging, branch)
        _safe_push(staging, branch, full_name, token)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    log.info("Published %s/%s to %s (branch %s)", paper_id, profile_name, full_name, branch)
    return PublishResult(paper_id, profile_name, mode, branch, full_name,
                         pushed=True, files_committed=n_files,
                         bytes_committed=n_bytes, trajectories_split=n_split)


def iter_completed_runs(data_dir: Path | str) -> Iterator[tuple[Path, dict]]:
    """Yield (env_dir, meta_dict) for every completed run under data_dir.

    Eval workspaces live at ``<data_dir>/envs/<paper_id>/<profile>-<mode>-eval/``.
    """
    base = Path(data_dir)
    envs_root = base / "envs" if (base / "envs").is_dir() else base
    if not envs_root.is_dir():
        return
    for paper_dir in sorted(envs_root.iterdir()):
        if not paper_dir.is_dir():
            continue
        for env_dir in sorted(paper_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            meta = _read_meta(env_dir)
            if meta and meta.get("status") == "completed":
                yield env_dir, meta


# ── Internal: branch / org / paper-name plumbing ─────────────────────────────


def _canonical_profile(profile_name: str) -> str:
    """Map an internal serving-variant profile to its canonical external
    identity (``publish_as`` in config). Everything publish() derives —
    branch, author, README, manifest — uses the canonical name; unknown
    profiles pass through unchanged."""
    base, suffix = profile_name, ""
    if base.endswith("-grader"):
        base, suffix = base[: -len("-grader")], "-grader"
    try:
        from . import models
        prof = models.load_profiles().get(base)
        if prof is not None and prof.publish_as:
            return prof.publish_as + suffix
    except Exception:
        pass
    return profile_name


def _make_branch_name(mode: str, profile_name: str) -> str:
    """``{mode}-{profile_name}``, e.g. ``improve-my-profile-high``."""
    return f"{mode}-{profile_name}"


def _assert_safe_branch(branch: str) -> None:
    if branch.lower() in _FORBIDDEN_BRANCH_NAMES:
        raise ValueError(
            f"refusing to publish to forbidden branch {branch!r} — "
            f"eval publish never writes main/master/HEAD"
        )
    if "/" in branch or " " in branch or ":" in branch:
        raise ValueError(f"branch name {branch!r} contains illegal characters")


def _github_org() -> str:
    """GitHub organisation that holds the per-paper repositories.

    ``publish.github_org`` of the evaluation config; if that is empty,
    ``github_org`` of Swarm's config when that file exists.
    """
    from evaluation import config as _config  # lazy: avoid import cycle

    org = str((_config.load().get("publish") or {}).get("github_org") or "").strip()
    if org:
        return org
    if _SWARM_CONFIG_PATH.exists():
        try:
            swarm_cfg = yaml.safe_load(_SWARM_CONFIG_PATH.read_text()) or {}
        except (OSError, yaml.YAMLError):
            swarm_cfg = {}
        org = str(swarm_cfg.get("github_org") or "").strip()
        if org:
            return org
    raise _config.ConfigError(
        "publish.github_org is not set in the evaluation config: name the "
        "GitHub organisation that holds the per-paper repositories."
    )


def _slugify(text: str) -> str:
    """Title slug, identical to Swarm's so both derive the same repo names."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def _repo_name_from_paper(paper_id: str, source_dir: Path | None = None) -> str:
    """Repository name Swarm gives a paper: ``{arxiv_id}-{title-slug}``, ≤100 chars.

    Needs the paper title (see :func:`_paper_title`); without it the
    bare paper id is returned and ``_resolve_repo_name`` falls back to a
    GitHub search.
    """
    title = _paper_title(paper_id, source_dir) or ""
    slug = _slugify(title) if title else ""
    name = f"{paper_id}-{slug}" if slug else paper_id
    if len(name) > 100:
        name = name[:100].rsplit("-", 1)[0]
    return name


# Per-process cache so a batch of publishes queries GitHub at most once
# per paper_id, regardless of how many profiles are being published.
_REPO_NAME_CACHE: dict[str, str | None] = {}


def _resolve_repo_name(
    paper_id: str, org: str, token: str, source_dir: Path | None = None,
) -> str | None:
    """Find the per-paper repository name for this paper.

    Lookup order:
      1. Compute the name from the paper title. If the resulting
         ``<org>/<name>`` exists, return it.
      2. Fall back to a GitHub repository search ``org:<org> <paper_id>``
         and return the unique match named ``<paper_id>`` or starting
         with ``<paper_id>-``.
      3. Return None if there is no such repository or the match is
         ambiguous; the caller then skips the publish.
    """
    if paper_id in _REPO_NAME_CACHE:
        return _REPO_NAME_CACHE[paper_id]

    # 1. Title-based guess
    guess = _repo_name_from_paper(paper_id, source_dir)
    if _repo_exists(f"{org}/{guess}", token):
        _REPO_NAME_CACHE[paper_id] = guess
        return guess

    # 2. Search the org for repos containing the paper id
    code, body = _github_api(
        "GET",
        f"/search/repositories?q=org:{org}+{paper_id}",
        token,
    )
    items = (body or {}).get("items", []) if code == 200 else []
    matches = [
        it["name"] for it in items
        if isinstance(it.get("name"), str)
        and (it["name"] == paper_id or it["name"].startswith(f"{paper_id}-"))
    ]
    if len(matches) == 1:
        log.info("Resolved %s via GitHub search → %s", paper_id, matches[0])
        _REPO_NAME_CACHE[paper_id] = matches[0]
        return matches[0]
    if len(matches) > 1:
        log.warning("Ambiguous repo match for %s: %s — skipping",
                    paper_id, matches)
    _REPO_NAME_CACHE[paper_id] = None
    return None


def _paper_title(paper_id: str, source_dir: Path | None = None) -> str | None:
    """Paper title from the environment's own ``source/manifest.json``.

    ``source_dir`` is the paper's source clone (the sibling of the run
    workspaces). Returns None when it is not given or has no title;
    callers then fall back to the repository name.
    """
    if source_dir is None:
        return None
    try:
        manifest = json.loads((Path(source_dir) / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    paper = manifest.get("paper") if isinstance(manifest, dict) else None
    title = paper.get("title") if isinstance(paper, dict) else None
    return title.strip() if isinstance(title, str) and title.strip() else None


# ── Internal: GitHub API ─────────────────────────────────────────────────────


def _get_github_token() -> str | None:
    """Read the access token of the first github.com entry in
    ``~/.git-credentials`` (the same source Swarm uses)."""
    creds_file = Path.home() / ".git-credentials"
    if not creds_file.exists():
        return None
    try:
        for line in creds_file.read_text().splitlines():
            if "github.com" in line:
                m = re.search(r"://[^:]+:([^@]+)@", line.strip())
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def _github_api(method: str, path: str, token: str,
                body: dict | None = None) -> tuple[int, dict | None]:
    url = f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body else None
    for attempt in range(3):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"token {token}")
        req.add_header("Accept", "application/vnd.github+json")
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            try:
                resp_body = json.loads(e.read().decode() or "{}")
            except Exception:
                resp_body = None
            return e.code, resp_body
        except (urllib.error.URLError, OSError):
            if attempt < 2:
                time.sleep(2 ** attempt)
    return 0, None


def _repo_exists(full_name: str, token: str) -> bool:
    code, _ = _github_api("GET", f"/repos/{full_name}", token)
    return code == 200


# ── Internal: meta + file selection ──────────────────────────────────────────


def _read_meta(env_dir: Path) -> dict | None:
    f = env_dir / ".evaluation.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _matches_exclude(rel_path: str) -> bool:
    from fnmatch import fnmatch
    return any(fnmatch(rel_path, g) for g in _EXCLUDE_GLOBS)


def _iter_publishable_files(env_dir: Path) -> Iterator[Path]:
    """Yield every file under env_dir that should be published (relative paths)."""
    for inc in _INCLUDE_PATHS:
        target = env_dir / inc.rstrip("/")
        if not target.exists():
            continue
        if target.is_file():
            yield target
        else:
            for p in target.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(env_dir).as_posix()
                if _matches_exclude(rel):
                    continue
                yield p


# ── Internal: sanitization ───────────────────────────────────────────────────


def _sanitize_text(text: str) -> str:
    for pat, repl in _REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text


# Public name: the dataset builder applies the same redaction to its rows.
sanitize_text = _sanitize_text


def _is_text_file(path: Path) -> bool:
    """Cheap heuristic: jsonl/json/md/txt/yaml/py/sh — sanitize. Else copy raw."""
    return path.suffix.lower() in {
        ".jsonl", ".json", ".md", ".txt", ".yaml", ".yml",
        ".py", ".sh", ".log", ".cfg", ".ini", ".toml",
    }


# ── Internal: trajectory split ───────────────────────────────────────────────


def _split_jsonl_for_size(src: Path, dest_dir: Path, base_name: str) -> int:
    """Stream ``src`` into ``dest_dir/<base_name>.partNN.jsonl`` chunks ≤ 90MB.

    Splits on line boundaries so each chunk is independently parseable.
    Sanitizes each line as it's written. Writes ``<base_name>.index.json``
    listing the parts in order. Returns the number of parts written.

    If the original is ≤ 90 MB, writes a single file ``<base_name>.jsonl``
    and no index file.
    """
    size = src.stat().st_size
    if size <= _TRAJECTORY_SPLIT_BYTES:
        out = dest_dir / f"{base_name}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as fr, out.open("wb") as fw:
            for line in fr:
                fw.write(_sanitize_text(line.decode("utf-8", "replace")).encode("utf-8"))
        return 1

    parts: list[str] = []
    part_num = 1
    bytes_in_part = 0
    fw = None
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as fr:
            for raw_line in fr:
                line = _sanitize_text(raw_line.decode("utf-8", "replace")).encode("utf-8")
                if fw is None or bytes_in_part + len(line) > _TRAJECTORY_SPLIT_BYTES:
                    if fw is not None:
                        fw.close()
                    part_path = dest_dir / f"{base_name}.part{part_num:02d}.jsonl"
                    parts.append(part_path.name)
                    fw = part_path.open("wb")
                    bytes_in_part = 0
                    part_num += 1
                fw.write(line)
                bytes_in_part += len(line)
    finally:
        if fw is not None:
            fw.close()

    index_path = dest_dir / f"{base_name}.index.json"
    index_path.write_text(json.dumps({
        "original_bytes": size,
        "split_at_bytes": _TRAJECTORY_SPLIT_BYTES,
        "parts": parts,
    }, indent=2) + "\n")
    return len(parts)


# ── Internal: README + manifest ──────────────────────────────────────────────


def _flatten_scores(scores: dict, max_metrics_per_experiment: int = 5) -> list[tuple[str, str, float]]:
    """Flatten nested experiment results into (experiment, metric, value) rows.

    Eval scores have shape::

        {"experiments": {<exp>: {"results": {<sub>: {<metric>: <value>}}}}}

    We headline the first ``max_metrics_per_experiment`` numeric leaves
    per (experiment, sub_experiment) pair. Full JSON is shipped alongside
    in a collapsible ``<details>`` block so nothing is lost.
    """
    rows: list[tuple[str, str, float]] = []
    experiments = (scores or {}).get("experiments") or {}
    for exp_name, exp in experiments.items():
        results = (exp or {}).get("results") or {}
        for sub_name, sub in results.items():
            count = 0
            label = f"{exp_name} / {sub_name}" if sub_name != exp_name else exp_name
            for metric, val in (sub or {}).items():
                if not isinstance(val, (int, float)):
                    continue
                rows.append((label, metric, float(val)))
                count += 1
                if count >= max_metrics_per_experiment:
                    break
    return rows


def _format_score_table(rows: list[tuple[str, str, float]]) -> str:
    if not rows:
        return "_No headline metrics extracted._"
    lines = ["| Experiment | Metric | Value |", "|---|---|---|"]
    for exp, metric, val in rows:
        # 4 sig figs is enough for headline display; full precision in JSON.
        if abs(val) >= 1000 or (val and abs(val) < 0.001):
            v = f"{val:.3e}"
        else:
            v = f"{val:.4g}"
        lines.append(f"| {exp} | {metric} | {v} |")
    return "\n".join(lines)


def _details_json_block(label: str, payload: dict) -> str:
    return (
        f"<details><summary>{label}</summary>\n\n"
        "```json\n"
        f"{json.dumps(payload, indent=2)}\n"
        "```\n"
        "</details>"
    )


def _read_idea_versions(env_dir: Path) -> list[tuple[str, str]]:
    """Return list of (filename, content) for idea_v*.md, sorted by version."""
    idea_dir = env_dir / "idea"
    if not idea_dir.is_dir():
        return []
    pat = re.compile(r"idea_v(\d+)\.md$")
    items: list[tuple[int, str, str]] = []
    for p in idea_dir.iterdir():
        m = pat.match(p.name)
        if m and p.is_file():
            try:
                items.append((int(m.group(1)), p.name, p.read_text(errors="replace")))
            except OSError:
                pass
    items.sort()
    return [(name, body) for _, name, body in items]


def _compose_readme(env_dir: Path, paper_id: str, profile_name: str,
                    mode: str, branch: str, meta: dict,
                    repo_name: str | None = None) -> str:
    # Title from the environment's own manifest; otherwise the
    # repository name (which carries the title slug), then the bare id.
    title = (_paper_title(paper_id, env_dir.parent / "source")
             or repo_name or f"arxiv:{paper_id}")
    ideas = _read_idea_versions(env_dir)
    reflection_path = env_dir / "reflection.md"
    reflection = reflection_path.read_text(errors="replace") if reflection_path.exists() else ""

    parts: list[str] = []
    parts.append(f"# {title} — `{paper_id}`")
    parts.append(f"**Branch:** `{branch}` · **Mode:** {mode} · **Profile:** {profile_name}\n")

    # Idea section(s)
    if len(ideas) == 0:
        pass  # nothing to render
    elif len(ideas) == 1:
        parts.append("## Idea\n")
        parts.append(ideas[0][1].rstrip())
        parts.append("")
    else:
        parts.append("## Initial idea\n")
        parts.append(ideas[0][1].rstrip())
        parts.append("")
        parts.append("## Final idea\n")
        parts.append(ideas[-1][1].rstrip())
        parts.append("")
        if len(ideas) > 2:
            parts.append(
                f"_The agent went through {len(ideas)} idea iterations "
                f"(`{ideas[0][0]}` → `{ideas[-1][0]}`); intermediate "
                f"versions are preserved in [`idea/`](idea/)._\n"
            )

    if reflection.strip():
        parts.append("## Reflection\n")
        parts.append(reflection.rstrip())
        parts.append("")

    train = meta.get("train_score") or {}
    test = meta.get("test_score") or {}

    parts.append("## Scores\n")
    parts.append("### Train phase\n")
    parts.append(_format_score_table(_flatten_scores(train)))
    parts.append("")
    if train:
        parts.append(_details_json_block("Full train scores (JSON)", train))
        parts.append("")

    parts.append("### Test phase\n")
    parts.append(_format_score_table(_flatten_scores(test)))
    parts.append("")
    if test:
        parts.append(_details_json_block("Full test scores (JSON)", test))
        parts.append("")

    parts.append("---")
    parts.append(
        "*Run artifacts: "
        "[`method/`](method/) · "
        "[`scoring/`](scoring/) · "
        "[`trajectory/`](trajectory/) · "
        "[`.evaluation.json`](.evaluation.json) · "
        "[`manifest.json`](manifest.json)*"
    )
    return "\n".join(parts) + "\n"


def _compose_manifest(paper_id: str, profile_name: str, mode: str,
                      branch: str, meta: dict,
                      paper_title: str | None = None) -> dict:
    """Small structured summary of the run for programmatic consumers."""
    return {
        "paper_id": paper_id,
        "paper_title": paper_title,
        "mode": mode,
        "profile_name": profile_name,
        "branch": branch,
        "run_id": meta.get("run_id"),
        "scaffold": (meta.get("usage") or {}).get("scaffold"),
        "turns": meta.get("current_turn"),
        "attempt": meta.get("attempt"),
        "status": meta.get("status"),
        "has_train_score": bool(meta.get("train_score")),
        "has_test_score": bool(meta.get("test_score")),
        # Timestamps and cost are deliberately omitted from this summary.
    }


# ── Internal: staging + commit + push ────────────────────────────────────────


def _iter_source_commits(env_dir: Path) -> list[dict]:
    """Read every commit from the eval workspace's ``.git`` (oldest-first)
    along with the author/date/message metadata needed to faithfully
    recreate it on the publish branch.

    Returns an empty list if the workspace has no .git or no commits
    (caller falls back to the single-commit path).
    """
    if not (env_dir / ".git").exists():
        return []
    sep_field = "\x1f"   # ASCII unit separator
    sep_record = "\x1e"  # ASCII record separator
    fmt = (
        f"%H{sep_field}%an{sep_field}%ae{sep_field}%aI"
        f"{sep_field}%cI{sep_field}%B{sep_record}"
    )
    proc = subprocess.run(
        ["git", "-C", str(env_dir), "log", "--reverse", "--format=" + fmt],
        capture_output=True, text=True, timeout=_GIT_TIMEOUT,
    )
    if proc.returncode != 0:
        return []
    commits: list[dict] = []
    for record in proc.stdout.split(sep_record):
        record = record.strip("\n")
        if not record:
            continue
        parts = record.split(sep_field, 5)
        if len(parts) < 6:
            continue
        sha, an, ae, ad, cd, msg = parts
        commits.append({
            "sha": sha.strip(),
            "author_name": an,
            "author_email": ae,
            "author_date": ad,
            "committer_date": cd,
            "message": msg,
        })
    return commits


def _split_jsonl_blob(blob: bytes, dest_dir: Path, base_name: str) -> int:
    """In-memory analogue of ``_split_jsonl_for_size`` — sanitizes and
    splits a JSONL blob that came from ``git cat-file blob``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if len(blob) <= _TRAJECTORY_SPLIT_BYTES:
        out = dest_dir / f"{base_name}.jsonl"
        sanitized = b"".join(
            _sanitize_text(line.decode("utf-8", "replace")).encode("utf-8")
            for line in blob.splitlines(keepends=True)
        )
        out.write_bytes(sanitized)
        return 1
    parts: list[str] = []
    part_num = 1
    cur = bytearray()
    for raw in blob.splitlines(keepends=True):
        line = _sanitize_text(raw.decode("utf-8", "replace")).encode("utf-8")
        if cur and len(cur) + len(line) > _TRAJECTORY_SPLIT_BYTES:
            p = dest_dir / f"{base_name}.part{part_num:02d}.jsonl"
            parts.append(p.name)
            p.write_bytes(bytes(cur))
            cur = bytearray()
            part_num += 1
        cur.extend(line)
    if cur:
        p = dest_dir / f"{base_name}.part{part_num:02d}.jsonl"
        parts.append(p.name)
        p.write_bytes(bytes(cur))
    (dest_dir / f"{base_name}.index.json").write_text(json.dumps({
        "original_bytes": len(blob),
        "split_at_bytes": _TRAJECTORY_SPLIT_BYTES,
        "parts": parts,
    }, indent=2) + "\n")
    return len(parts)


def _materialize_commit_into_staging(
    env_dir: Path, sha: str, staging: Path,
) -> tuple[int, list[tuple[str, int]]]:
    """Repopulate ``staging`` with the included paths from ``env_dir@sha``.

    Clears prior tracked content (preserves ``.git``), reads each blob
    via ``git cat-file``, applies sanitization + size-skip + the
    reflection.md → idea/ remap, and stream-splits oversized JSONL
    trajectories. Returns (n_split, dropped_oversize_list).
    """
    # Clear staging tree (preserve .git/)
    for entry in list(staging.iterdir()):
        if entry.name == ".git":
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                entry.unlink()
            except OSError:
                pass

    n_split = 0
    dropped: list[tuple[str, int]] = []

    for inc in _INCLUDE_PATHS:
        inc_path = inc.rstrip("/")
        # ``ls-tree -r -l`` includes the blob size, so we can size-check
        # *before* spending I/O on a 1.2 GB cat-file. ``-z`` for safe
        # null-delimited paths.
        tree = subprocess.run(
            ["git", "-C", str(env_dir), "ls-tree", "-r", "-l", "-z",
             sha, "--", inc_path],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT,
        )
        if tree.returncode != 0 or not tree.stdout:
            continue
        for entry in tree.stdout.split("\x00"):
            if not entry:
                continue
            # Format: ``<mode> <type> <object> <size>\t<path>``
            try:
                head, file_path = entry.split("\t", 1)
                _mode, obj_type, blob_sha, size_str = head.split(maxsplit=3)
            except ValueError:
                continue
            if obj_type != "blob":
                continue
            if _matches_exclude(file_path):
                continue
            try:
                size = int(size_str)
            except ValueError:
                size = 0

            # reflection.md → idea/reflection.md remap
            target = "idea/reflection.md" if file_path == "reflection.md" else file_path

            # Trajectory: split-and-sanitize path
            if file_path.startswith("trajectory/") and file_path.endswith(".jsonl"):
                blob = subprocess.run(
                    ["git", "-C", str(env_dir), "cat-file", "blob", blob_sha],
                    capture_output=True, timeout=_GIT_TIMEOUT,
                ).stdout
                base = Path(file_path).stem
                n_split += _split_jsonl_blob(blob, staging / "trajectory", base)
                continue

            # Skip oversized non-trajectory files
            if size > _NON_JSONL_FILE_MAX_BYTES:
                dropped.append((file_path, size))
                continue

            blob = subprocess.run(
                ["git", "-C", str(env_dir), "cat-file", "blob", blob_sha],
                capture_output=True, timeout=_GIT_TIMEOUT,
            ).stdout

            dest = staging / target
            dest.parent.mkdir(parents=True, exist_ok=True)

            ext = Path(file_path).suffix.lower()
            if ext in {".jsonl", ".json", ".md", ".txt", ".yaml", ".yml",
                       ".py", ".sh", ".log", ".cfg", ".ini", ".toml"}:
                try:
                    text = blob.decode("utf-8", "replace")
                    dest.write_text(_sanitize_text(text))
                    continue
                except Exception:
                    pass
            dest.write_bytes(blob)

    return n_split, dropped


def _stage_workspace(env_dir: Path, paper_id: str, profile_name: str,
                     mode: str, meta: dict, branch: str,
                     repo_name: str | None = None) -> Path:
    """Build a staging git repo on ``branch`` that:

    1. Replays every commit from the run workspace's ``.git`` (oldest
       first), preserving the agent's iteration history (commit
       messages and dates), but with each commit's tree filtered to
       the include list, sanitized for credentials, and trajectories
       split where they exceed 90 MB.

    2. Adds one final commit on top with the generated ``README.md`` /
       ``manifest.json`` / ``.gitignore`` / (optional)
       ``EXCLUDED_LARGE_FILES.md``, so the README is at the branch's
       HEAD with the run's history underneath.

    Returns the staging path. Caller is responsible for ``rmtree``.

    The staging directory is a fresh temporary directory, never inside
    env_dir, so the run workspace is not modified.
    """
    _assert_safe_branch(branch)
    import tempfile
    staging = Path(tempfile.mkdtemp(prefix="evalpub_"))
    log.debug("Staging publish under %s", staging)

    # Init empty repo on the publish branch (main/master never exist).
    subprocess.run(
        ["git", "init", "-b", branch],
        cwd=str(staging), capture_output=True, check=True,
        timeout=_GIT_TIMEOUT, env=_GIT_ENV,
    )
    hooks_dir = staging / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook = hooks_dir / "pre-commit"
    hook.write_text(_PRE_COMMIT_HOOK)
    hook.chmod(0o755)

    # ── Replay source commits ─────────────────────────────────────────
    n_split_total = 0
    all_dropped: dict[str, int] = {}  # path → max size seen across commits
    commits_replayed = 0

    # Author identity for replayed commits: the profile name. The
    # workspace's git history attributes everything to one generic
    # local identity, but on the published branch the commits should
    # say which model did the work, as the branch name does.
    author_name = profile_name
    author_email = f"{profile_name}@scientistworld"

    for c in _iter_source_commits(env_dir):
        n_split, dropped = _materialize_commit_into_staging(
            env_dir, c["sha"], staging,
        )
        n_split_total += n_split
        for path, sz in dropped:
            all_dropped[path] = max(all_dropped.get(path, 0), sz)

        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(staging), capture_output=True, check=True,
            timeout=_GIT_TIMEOUT, env=_GIT_ENV,
        )
        # Skip empty commits (this source commit only touched excluded paths).
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(staging), env=_GIT_ENV, timeout=_GIT_TIMEOUT,
        )
        if diff.returncode == 0:
            continue

        # Author = profile (per branch); committer = the publishing
        # identity in _GIT_ENV. Original author/committer dates are
        # preserved so the timeline reflects when the work happened.
        commit_env = {
            **_GIT_ENV,
            "GIT_AUTHOR_NAME": author_name,
            "GIT_AUTHOR_EMAIL": author_email,
            "GIT_AUTHOR_DATE": c["author_date"],
            "GIT_COMMITTER_DATE": c["committer_date"],
        }
        # Use --file=- and pass the message via stdin to handle
        # arbitrarily-long messages with shell-special characters
        # without quoting issues.
        proc = subprocess.run(
            ["git", "commit", "--allow-empty-message", "-F", "-"],
            input=c["message"], cwd=str(staging), capture_output=True,
            text=True, timeout=_GIT_TIMEOUT, env=commit_env,
        )
        if proc.returncode != 0:
            # Pre-commit hook may legitimately reject (e.g. credential
            # leak in an old commit). Log + skip — don't fail the run.
            log.warning(
                "publish: skipping commit %s for %s/%s — git rejected: %s",
                c["sha"][:10], env_dir.name, branch,
                (proc.stderr or "").strip()[:200],
            )
            # Reset index so the next iteration starts clean
            subprocess.run(
                ["git", "reset", "--hard", "HEAD"]
                if commits_replayed > 0 else ["git", "rm", "-rf", "--cached", "."],
                cwd=str(staging), capture_output=True,
                env=_GIT_ENV, timeout=_GIT_TIMEOUT,
            )
            continue
        commits_replayed += 1

    # If no commits got replayed (no .git, empty history, or every
    # commit was skipped), fall back to materializing the env_dir's
    # current files so the branch isn't empty.
    if commits_replayed == 0:
        split_ref = [n_split_total]
        _materialize_current_files_into_staging(
            env_dir, staging, n_split_total_ref=split_ref,
            dropped_ref=all_dropped,
        )
        n_split_total = split_ref[0]
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(staging), capture_output=True, check=True,
            timeout=_GIT_TIMEOUT, env=_GIT_ENV,
        )

    # ── Final META commit ────────────────────────────────────────────
    # The README quotes the agent's idea and reflection files from the
    # live workspace, so it gets the same redaction as replayed files
    # (otherwise a credential-like string in them makes the pre-commit
    # hook reject the commit and the whole publish fails).
    (staging / "README.md").write_text(_sanitize_text(
        _compose_readme(env_dir, paper_id, profile_name, mode, branch, meta,
                        repo_name=repo_name)
    ))
    (staging / "manifest.json").write_text(
        json.dumps(
            _compose_manifest(
                paper_id, profile_name, mode, branch, meta,
                paper_title=_paper_title(paper_id, env_dir.parent / "source"),
            ),
            indent=2,
        )
        + "\n"
    )
    (staging / ".gitignore").write_text(
        "# Auto-generated by evaluation publish.\n"
        "*.sif\n*.img\n*.tar\n*.tar.gz\n*.tar.xz\n"
        "__pycache__/\n.uv_cache/\n.cache/\nnode_modules/\n.venv/\n"
        "site-packages/\nwheels/\nlocal-packages/\n"
    )
    if all_dropped:
        for path, sz in all_dropped.items():
            log.warning("publish: dropped oversized file %s (%.0f MB > %.0f MB cap)",
                        path, sz / 1024 / 1024,
                        _NON_JSONL_FILE_MAX_BYTES / 1024 / 1024)
        (staging / "EXCLUDED_LARGE_FILES.md").write_text(
            "# Files excluded from this branch (too large for GitHub)\n\n"
            "These files exceeded GitHub's 100 MB blob limit and were "
            "skipped during publish (across one or more commits). They "
            "remain in the original run workspace.\n\n"
            + "".join(
                f"- `{p}` — {sz / 1024 / 1024:.0f} MB\n"
                for p, sz in sorted(all_dropped.items())
            )
        )

    subprocess.run(
        ["git", "add", "-A"],
        cwd=str(staging), capture_output=True, check=True,
        timeout=_GIT_TIMEOUT, env=_GIT_ENV,
    )
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=str(staging), env=_GIT_ENV, timeout=_GIT_TIMEOUT,
    )
    if diff.returncode != 0:
        subprocess.run(
            ["git", "commit", "-m", f"eval: publish artifacts ({branch})"],
            cwd=str(staging), capture_output=True, check=True,
            timeout=_GIT_TIMEOUT, env=_GIT_ENV,
        )

    # Stash split count + dropped list as a sidecar for the caller's
    # PublishResult; cleared right before push.
    (staging / ".publish-stats.json").write_text(json.dumps({
        "trajectories_split": n_split_total,
        "commits_replayed": commits_replayed,
        "dropped_count": len(all_dropped),
    }, indent=2) + "\n")
    return staging


def _materialize_current_files_into_staging(
    env_dir: Path, staging: Path,
    n_split_total_ref: list[int],
    dropped_ref: dict[str, int],
) -> None:
    """Fallback for workspaces with no commit history: copy the current
    file tree (the live filesystem state) into staging, applying the
    same filters as the per-commit replay path.

    ``n_split_total_ref`` and ``dropped_ref`` are mutated in place so
    the caller can fold these into its META commit's accounting.
    """
    for src in _iter_publishable_files(env_dir):
        rel = src.relative_to(env_dir)
        rel_str = rel.as_posix()
        if rel_str.startswith("trajectory/") and src.suffix == ".jsonl":
            n_split_total_ref[0] += _split_jsonl_for_size(
                src, staging / "trajectory", src.stem,
            )
            continue
        try:
            sz = src.stat().st_size
        except OSError:
            sz = 0
        if sz > _NON_JSONL_FILE_MAX_BYTES:
            dropped_ref[rel_str] = max(dropped_ref.get(rel_str, 0), sz)
            continue
        if rel_str == "reflection.md":
            rel = Path("idea") / "reflection.md"
        dest = staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if _is_text_file(src):
            try:
                dest.write_text(_sanitize_text(src.read_text(errors="replace")))
                continue
            except OSError:
                pass
        shutil.copy2(src, dest)


def _commit_staging(staging: Path, branch: str) -> tuple[int, int, int]:
    """Last check before the push, and the stats ``publish()`` reports.

    ``_stage_workspace`` has already committed everything (replayed
    history + final commit). This removes the stats sidecar, refuses to
    continue unless HEAD is on ``branch``, and returns
    ``(file count, byte total, trajectories split)``.
    """
    _assert_safe_branch(branch)
    stats_path = staging / ".publish-stats.json"
    n_split = 0
    if stats_path.exists():
        try:
            n_split = int(json.loads(stats_path.read_text()).get("trajectories_split", 0))
        except Exception:
            pass
        stats_path.unlink(missing_ok=True)
        # Amend the final META commit so the deleted sidecar doesn't
        # leave the working tree dirty (we still need the same HEAD).
        subprocess.run(
            ["git", "add", "-A"], cwd=str(staging), capture_output=True,
            env=_GIT_ENV, timeout=_GIT_TIMEOUT,
        )
        # If the deletion is the only diff, fold it into HEAD silently.
        if subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=str(staging),
            env=_GIT_ENV, timeout=_GIT_TIMEOUT,
        ).returncode != 0:
            subprocess.run(
                ["git", "commit", "--amend", "--no-edit"],
                cwd=str(staging), capture_output=True,
                env=_GIT_ENV, timeout=_GIT_TIMEOUT,
            )

    # Sanity-check HEAD is on the requested branch.
    cur = subprocess.run(
        ["git", "symbolic-ref", "--short", "HEAD"],
        cwd=str(staging), capture_output=True, text=True,
        timeout=_GIT_TIMEOUT, env=_GIT_ENV,
    )
    actual = (cur.stdout or "").strip()
    if actual != branch:
        raise RuntimeError(
            f"refuse to push: HEAD is on {actual!r}, expected {branch!r}"
        )

    ls = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=str(staging), capture_output=True, text=True,
        timeout=_GIT_TIMEOUT, env=_GIT_ENV,
    )
    files = [x for x in (ls.stdout or "").splitlines() if x]
    total = 0
    for f in files:
        try:
            total += (staging / f).stat().st_size
        except OSError:
            pass
    return len(files), total, n_split


_GITHUB_URL = "https://github.com/"


def _auth_env(token: str) -> dict[str, str]:
    """Environment that makes git authenticate to GitHub with ``token``.

    The token is passed as an HTTP ``Authorization`` header through
    ``GIT_CONFIG_COUNT`` / ``GIT_CONFIG_KEY_n`` / ``GIT_CONFIG_VALUE_n``
    (git 2.31 or newer), scoped to github.com. Unlike a token embedded
    in the remote URL it never appears on a command line (visible to
    other users in the process list) or in the repository's config.
    """
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = dict(_GIT_ENV)
    try:
        n = int(env.get("GIT_CONFIG_COUNT") or 0)
    except ValueError:
        n = 0
    env[f"GIT_CONFIG_KEY_{n}"] = f"http.{_GITHUB_URL}.extraheader"
    env[f"GIT_CONFIG_VALUE_{n}"] = f"AUTHORIZATION: basic {basic}"
    env["GIT_CONFIG_COUNT"] = str(n + 1)
    # Fail instead of prompting if the token is rejected.
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _safe_push(staging: Path, branch: str, full_name: str, token: str) -> None:
    """The ONLY function in the package that runs ``git push``.

    Multi-layer safety:
      - Branch is asserted ∉ {main,master,HEAD,""}.
      - Refspec is explicit ``<branch>:<branch>``.
      - Force is scoped via ``--force-with-lease=<branch>:<sha>``: the
        lease is on the target branch only and pinned to the exact sha
        just fetched. It cannot affect main and cannot overwrite a
        concurrent push that landed between the fetch and the push.
      - URL is ``https://github.com/<full_name>.git``: explicit, never
        a configured remote.
      - The command is refused if it contains ``:main`` / ``:master`` /
        ``--all`` / ``--mirror``.
      - The token reaches git through the environment (``_auth_env``),
        never through the command line.
    """
    _assert_safe_branch(branch)

    remote_url = f"{_GITHUB_URL}{full_name}.git"
    auth_env = _auth_env(token)

    # Fetch the target branch (if it exists remotely) so the lease can
    # be pinned to its exact sha. If the branch doesn't exist yet,
    # fetch fails — we lease against the empty value so the push only
    # succeeds if no one creates the branch between now and the push.
    fetch = subprocess.run(
        ["git", "fetch", "--quiet", remote_url, branch],
        cwd=str(staging), capture_output=True, text=True,
        timeout=_GIT_TIMEOUT, env=auth_env,
    )
    if fetch.returncode == 0:
        rev = subprocess.run(
            ["git", "rev-parse", "FETCH_HEAD"],
            cwd=str(staging), capture_output=True, text=True,
            timeout=_GIT_TIMEOUT, env=_GIT_ENV,
        )
        remote_sha = (rev.stdout or "").strip()
        # Sanity: a sha is 40 hex chars. Anything else (e.g. branch name
        # from a buggy git) → bail rather than build a malformed lease.
        if not re.fullmatch(r"[0-9a-f]{40}", remote_sha):
            raise RuntimeError(
                f"safe_push: bogus FETCH_HEAD {remote_sha!r} for {branch}"
            )
        lease = f"--force-with-lease={branch}:{remote_sha}"
    else:
        # Branch doesn't exist remotely yet; lease against empty.
        lease = f"--force-with-lease={branch}:"

    refspec = f"{branch}:{branch}"
    cmd = [
        "git", "push",
        lease,
        remote_url,
        refspec,
    ]

    # Last check: scan the command for forbidden tokens. Any of these
    # would indicate a programming error upstream.
    cmd_text = " ".join(cmd)
    forbidden_in_cmd = (":main", ":master", "--all", "--mirror")
    for f in forbidden_in_cmd:
        if f in cmd_text:
            raise RuntimeError(
                f"safe_push aborting: command contains forbidden token {f!r}"
            )
    if token and token in cmd_text:
        raise RuntimeError("safe_push aborting: access token on the command line")

    log.info("Pushing %s → %s (refspec=%s)", staging.name, full_name, refspec)
    proc = subprocess.run(
        cmd, cwd=str(staging), capture_output=True, text=True,
        timeout=_GIT_TIMEOUT * 2, env=auth_env,
    )
    if proc.returncode != 0:
        # Make sure the token cannot end up in the error message, in
        # either its plain or its header-encoded form.
        stderr = proc.stderr or ""
        encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        for secret in (token, encoded):
            if secret:
                stderr = stderr.replace(secret, "REDACTED")
        raise RuntimeError(
            f"git push failed (rc={proc.returncode}) for {full_name} "
            f"branch={branch}: {stderr.strip()}"
        )


_COEFF_BRANCH = "normalization"


def publish_coeffs(
    paper_id: str, branch: str = _COEFF_BRANCH, dry_run: bool = False,
) -> PublishResult:
    """Push a paper's locked normalization coefficients to a dedicated
    standalone branch of its per-paper repository.

    The branch holds only the locked ``coeffs_<mode>_<phase>.json`` files
    (both modes together) plus a README. It is created fresh each push
    and force-updated via ``_safe_push``'s branch-scoped lease, which
    pins to this branch's sha only, so it can never touch ``main``,
    ``master``, or any other existing branch.

    Returns ``PublishResult.pushed=False`` with a ``skipped_reason`` when
    publishing is disabled, there is nothing locked to publish, or the
    repository cannot be located.
    """
    _assert_safe_branch(branch)
    from evaluation import config as _config  # lazy: avoid import cycle

    if not publish_enabled():
        return PublishResult(paper_id, "(coeffs)", "", branch, "",
                             pushed=False, skipped_reason=DISABLED_REASON)

    norm_dir = _config.envs_dir() / paper_id / "normalization"
    if not norm_dir.is_dir():
        return PublishResult(paper_id, "(coeffs)", "", branch, "",
                             pushed=False, skipped_reason="no normalization dir")

    locked: list[Path] = []
    for f in sorted(norm_dir.glob("coeffs_*.json")):
        try:
            if json.loads(f.read_text()).get("locked"):
                locked.append(f)
        except (OSError, json.JSONDecodeError):
            continue
    if not locked:
        return PublishResult(paper_id, "(coeffs)", "", branch, "",
                             pushed=False, skipped_reason="no locked coeffs")

    org = _github_org()
    token = _get_github_token()
    if not token:
        return PublishResult(paper_id, "(coeffs)", "", branch, "",
                             pushed=False, skipped_reason="no GitHub token")

    source_dir = norm_dir.parent / "source"
    if dry_run:
        full = f"{org}/{_repo_name_from_paper(paper_id, source_dir)}"
        return PublishResult(paper_id, "(coeffs)", "", branch, full,
                             pushed=False, skipped_reason="dry-run",
                             files_committed=len(locked))

    repo = _resolve_repo_name(paper_id, org, token, source_dir)
    if repo is None:
        return PublishResult(paper_id, "(coeffs)", "", branch, "",
                             pushed=False,
                             skipped_reason=f"repo not found for {paper_id}")
    full_name = f"{org}/{repo}"

    import tempfile
    staging = Path(tempfile.mkdtemp(prefix="coeffpub_"))
    try:
        subprocess.run(["git", "init", "-b", branch], cwd=str(staging),
                       capture_output=True, check=True, env=_GIT_ENV,
                       timeout=_GIT_TIMEOUT)
        dest = staging / "normalization"
        dest.mkdir(parents=True, exist_ok=True)
        for f in locked:
            shutil.copy2(f, dest / f.name)
        (staging / "README.md").write_text(_coeff_readme(paper_id, locked))
        subprocess.run(["git", "add", "-A"], cwd=str(staging),
                       capture_output=True, check=True, env=_GIT_ENV,
                       timeout=_GIT_TIMEOUT)
        subprocess.run(
            ["git", "commit", "-m",
             f"Locked normalization coefficients ({paper_id})"],
            cwd=str(staging), capture_output=True, check=True,
            env=_GIT_ENV, timeout=_GIT_TIMEOUT)
        _safe_push(staging, branch, full_name, token)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    log.info("Published locked coeffs for %s to %s (branch %s, %d files)",
             paper_id, full_name, branch, len(locked))
    return PublishResult(paper_id, "(coeffs)", "", branch, full_name,
                         pushed=True, files_committed=len(locked))


def _coeff_readme(paper_id: str, locked_files: list[Path]) -> str:
    rows = []
    for f in sorted(locked_files):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            f"- `normalization/{f.name}` — mode=`{d.get('mode')}` "
            f"phase=`{d.get('phase')}` "
            f"locked with **{d.get('locked_n_models', d.get('n_models'))}** "
            f"cohort models at `{d.get('locked_at', '')}`"
        )
    body = "\n".join(rows)
    return (
        f"# Frozen normalization coefficients — {paper_id}\n\n"
        "These are the **locked** per-paper normalization coefficients "
        "used to score runs on this environment. Once locked they are "
        "frozen, so any later run is scored against a stable, "
        "reproducible normalizer (the score no longer drifts as new "
        "models are added to the cohort).\n\n"
        "Each `coeffs_<mode>_<phase>.json` carries `cells` "
        "(per-cell `coefficient`, `source`, `higher_is_better`), "
        "`paper_baseline`, and the cohort it was computed from.\n\n"
        f"{body}\n\n"
        "This is a standalone snapshot branch; it intentionally does not "
        "contain the rest of the paper's code or run history.\n"
    )


__all__ = [
    "publish", "publish_coeffs", "publish_enabled", "iter_completed_runs",
    "PublishResult", "DISABLED_REASON", "sanitize_text",
]
