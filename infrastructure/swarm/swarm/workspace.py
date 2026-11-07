"""Workspace creation and seeding for paper reproduction.

Creates a per-paper workspace by copying the workspace template and
copying paper files into it (not symlinking — copies survive container
isolation where symlink targets may not be bind-mounted).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from swarm.paper import PaperInfo

log = logging.getLogger(__name__)

# Workspace template shipped with the worker package
_WORKER_ROOT = Path(__file__).resolve().parent.parent.parent / "worker"
WORKSPACE_TEMPLATE_DIR = _WORKER_ROOT / "workspace_template"

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "swarm",
    "GIT_COMMITTER_NAME": "swarm",
    "GIT_AUTHOR_EMAIL": "swarm@reproducer",
    "GIT_COMMITTER_EMAIL": "swarm@reproducer",
}

_GITHUB_TOPIC_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,49}$")

_INITIAL_GITHUB_PUSH_RETRY_DELAYS = (1.0, 2.0, 5.0)
_LOCAL_GIT_CONFIG_TIMEOUT = 300
_LOCAL_GIT_CONFIG_RETRY_DELAYS = (1.0, 2.0, 5.0)


def _get_github_token() -> str | None:
    """Read a GitHub PAT from ~/.git-credentials.

    Expects the standard git-credential-store format:
        https://<user>:<token>@github.com

    This file is the only token source Swarm consults. The first line
    that mentions github.com is used.
    """
    creds_file = Path.home() / ".git-credentials"
    if not creds_file.exists():
        return None
    try:
        for line in creds_file.read_text().splitlines():
            line = line.strip()
            if "github.com" in line:
                # https://user:token@github.com
                match = re.search(r"://[^:]+:([^@]+)@", line)
                if match:
                    return match.group(1)
    except OSError:
        pass
    return None


def _github_api(
    method: str,
    path: str,
    token: str,
    body: dict | None = None,
    retries: int = 3,
) -> tuple[int, dict | None]:
    """Make a GitHub REST API call with retries. Returns (status_code, json_body)."""
    url = f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body else None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"token {token}")
        req.add_header("Accept", "application/vnd.github+json")
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_body = json.loads(resp.read().decode())
                return resp.status, resp_body
        except urllib.error.HTTPError as e:
            try:
                resp_body = json.loads(e.read().decode())
            except Exception:
                resp_body = None
            return e.code, resp_body
        except (urllib.error.URLError, OSError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                log.warning("GitHub API transient error (attempt %d/%d): %s %s — %s",
                            attempt + 1, retries, method, path, e)
            else:
                log.warning("GitHub API request failed after %d attempts: %s %s — %s",
                            retries, method, path, e)
                return 0, None
    return 0, None


def _github_repo_exists(full_name: str, token: str) -> bool:
    """Check if a GitHub repo exists."""
    status, _ = _github_api("GET", f"/repos/{full_name}", token)
    return status == 200


def _git_remote_get_url(workspace: Path, remote: str = "origin") -> str | None:
    """Return the configured remote URL, or None if the remote is missing."""
    result = subprocess.run(
        ["git", "remote", "get-url", remote],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=_LOCAL_GIT_CONFIG_TIMEOUT,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _git_set_remote_url(workspace: Path, remote: str, remote_url: str) -> None:
    """Set an existing remote URL, tolerating transient local git stalls."""
    last_error: Exception | None = None
    for attempt, delay_s in enumerate((0.0, *_LOCAL_GIT_CONFIG_RETRY_DELAYS), start=1):
        if delay_s:
            time.sleep(delay_s)
        try:
            result = subprocess.run(
                ["git", "remote", "set-url", remote, remote_url],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=_LOCAL_GIT_CONFIG_TIMEOUT,
            )
            if result.returncode == 0:
                return
            last_error = subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr,
            )
            log.warning(
                "Failed to set git remote %s in %s (attempt %d/%d): %s",
                remote,
                workspace,
                attempt,
                1 + len(_LOCAL_GIT_CONFIG_RETRY_DELAYS),
                (result.stderr or "").strip(),
            )
        except subprocess.TimeoutExpired as e:
            current = _git_remote_get_url(workspace, remote)
            if current == remote_url:
                log.info(
                    "Git remote %s in %s was updated despite timeout",
                    remote,
                    workspace,
                )
                return
            last_error = e
            log.warning(
                "Timed out setting git remote %s in %s (attempt %d/%d)",
                remote,
                workspace,
                attempt,
                1 + len(_LOCAL_GIT_CONFIG_RETRY_DELAYS),
            )

    current = _git_remote_get_url(workspace, remote)
    if current == remote_url:
        return
    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to set git remote {remote} in {workspace}")


def _ensure_git_remote_url(workspace: Path, remote: str, remote_url: str) -> None:
    """Ensure a local git remote exists and points at the expected URL."""
    current = _git_remote_get_url(workspace, remote)
    if current == remote_url:
        return
    if current:
        _git_set_remote_url(workspace, remote, remote_url)
        return

    last_error: Exception | None = None
    for attempt, delay_s in enumerate((0.0, *_LOCAL_GIT_CONFIG_RETRY_DELAYS), start=1):
        if delay_s:
            time.sleep(delay_s)
        try:
            result = subprocess.run(
                ["git", "remote", "add", remote, remote_url],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=_LOCAL_GIT_CONFIG_TIMEOUT,
            )
            if result.returncode == 0:
                return

            stderr = (result.stderr or "").strip()
            current = _git_remote_get_url(workspace, remote)
            if current == remote_url:
                return
            if current and current != remote_url:
                _git_set_remote_url(workspace, remote, remote_url)
                return
            if "already exists" in stderr.lower():
                last_error = RuntimeError(
                    f"git remote {remote} already exists but could not be verified"
                )
            else:
                last_error = subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    output=result.stdout,
                    stderr=result.stderr,
                )
            log.warning(
                "Failed to add git remote %s in %s (attempt %d/%d): %s",
                remote,
                workspace,
                attempt,
                1 + len(_LOCAL_GIT_CONFIG_RETRY_DELAYS),
                stderr,
            )
        except subprocess.TimeoutExpired as e:
            current = _git_remote_get_url(workspace, remote)
            if current == remote_url:
                log.info(
                    "Git remote %s in %s was added despite timeout",
                    remote,
                    workspace,
                )
                return
            if current and current != remote_url:
                _git_set_remote_url(workspace, remote, remote_url)
                return
            last_error = e
            log.warning(
                "Timed out adding git remote %s in %s (attempt %d/%d)",
                remote,
                workspace,
                attempt,
                1 + len(_LOCAL_GIT_CONFIG_RETRY_DELAYS),
            )

    current = _git_remote_get_url(workspace, remote)
    if current == remote_url:
        return
    if current and current != remote_url:
        _git_set_remote_url(workspace, remote, remote_url)
        return
    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to add git remote {remote} in {workspace}")


def _sanitize_github_topics(
    topics: list[str] | None,
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Return (sanitized_topics, rewritten_topics, dropped_topics)."""
    valid: list[str] = []
    rewritten: list[tuple[str, str]] = []
    dropped: list[str] = []
    seen: set[str] = set()

    for topic in topics or []:
        candidate = re.sub(r"[^a-z0-9-]+", "-", topic.strip().lower())
        candidate = re.sub(r"-{2,}", "-", candidate).strip("-")
        candidate = candidate[:50].rstrip("-")

        if not candidate or not candidate[0].isalnum():
            dropped.append(topic)
            continue

        if candidate in seen:
            continue
        seen.add(candidate)

        if candidate != topic:
            rewritten.append((topic, candidate))

        if _GITHUB_TOPIC_RE.fullmatch(candidate):
            valid.append(candidate)
        else:
            dropped.append(topic)
    return valid, rewritten, dropped


def create_workspace(
    paper: PaperInfo,
    base_dir: str,
    template_dir: str | None = None,
    github_org: str = "",
    date: str | None = None,
    topics: list[str] | None = None,
) -> str:
    """Create a workspace for a paper.

    Creates ``base_dir/<arxiv_id>/``, copies template files, and copies
    paper files into ``workspace/paper/``. Initializes a git repository
    and optionally creates a GitHub repo and pushes.

    Args:
        paper: Resolved paper info with file paths.
        base_dir: Parent directory for workspaces.
        template_dir: Override for workspace template directory.
        github_org: GitHub organization name. If set, creates a repo
            and pushes.  If empty, git stays local only.
        date: Paper date (YYYY-MM-DD). If set, added as GitHub topic.
        topics: Extra GitHub topics to add at repo creation.

    Returns:
        Absolute path to the created workspace directory.

    Raises:
        FileNotFoundError: If template directory doesn't exist.
        FileExistsError: If workspace already exists.
    """
    template = Path(template_dir) if template_dir else WORKSPACE_TEMPLATE_DIR
    if not template.is_dir():
        raise FileNotFoundError(f"Workspace template not found: {template}")

    workspace = Path(base_dir) / paper.arxiv_id
    if workspace.exists():
        raise FileExistsError(f"Workspace already exists: {workspace}")

    # Copy template
    shutil.copytree(str(template), str(workspace))
    log.info("Copied workspace template to %s", workspace)

    # Copy paper files into workspace/paper/
    paper_dir = workspace / "paper"
    paper_dir.mkdir(exist_ok=True)

    # Markdown
    md_path = Path(paper.paper_md_path)
    if md_path.exists():
        shutil.copy2(str(md_path), str(paper_dir / md_path.name))

    # Images directory
    images_dir = Path(paper.paper_images_dir)
    if images_dir.is_dir():
        dest_images = paper_dir / "images"
        if not dest_images.exists():
            shutil.copytree(str(images_dir), str(dest_images))

    # PDF
    pdf_path = Path(paper.paper_pdf_path)
    if pdf_path.exists():
        shutil.copy2(str(pdf_path), str(paper_dir / pdf_path.name))

    # Initialize git repo and set up credentials for the agent
    _git_init(workspace)
    _setup_git_credentials(workspace)

    # Create GitHub repo and push
    if github_org:
        _github_create_and_push(workspace, paper, github_org, date=date, topics=topics)

    return str(workspace)


def ensure_github_remote(
    workspace: str,
    paper: PaperInfo,
    github_org: str,
    attempt: int = 1,
) -> None:
    """Ensure the workspace has a GitHub remote, creating the repo if needed.

    Used by the scheduler on continuation to verify the remote is set up
    (the workspace may have been created without a remote, or recreated).

    For re-runs (attempt >= 2), creates and pushes to a new branch.
    """
    ws = Path(workspace)
    if not (ws / ".git").is_dir():
        return

    # Ensure credentials are available
    _setup_git_credentials(ws)

    # Check if remote already exists
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=str(ws), capture_output=True, text=True, timeout=_LOCAL_GIT_CONFIG_TIMEOUT,
    )
    if result.returncode != 0:
        # No remote — set it up and create repo if needed
        _github_create_and_push(ws, paper, github_org)
    elif attempt >= 2:
        # Re-run: create a new branch
        branch = f"attempt-{attempt}"
        subprocess.run(
            ["git", "checkout", "-b", branch],
            cwd=str(ws), capture_output=True, timeout=_LOCAL_GIT_CONFIG_TIMEOUT,
        )
        subprocess.run(
            ["git", "push", "-u", "origin", branch],
            cwd=str(ws), capture_output=True, timeout=_LOCAL_GIT_CONFIG_TIMEOUT, env=_GIT_ENV,
        )
        log.info("Created branch %s for re-run of %s", branch, paper.arxiv_id)


def clone_workspace(
    paper: PaperInfo,
    base_dir: str,
    github_org: str,
) -> str | None:
    """Clone an existing GitHub repo as the workspace.

    Used when continuing a paper whose local workspace is missing but
    the GitHub repo exists.

    Returns:
        Workspace path if cloned successfully, None if repo doesn't exist.
    """
    token = _get_github_token()
    if not token:
        log.warning("No GitHub token found — cannot check for remote repo")
        return None

    repo_name = paper.repo_name()
    full_name = f"{github_org}/{repo_name}"
    workspace = Path(base_dir) / paper.arxiv_id

    if not _github_repo_exists(full_name, token):
        return None

    remote_url = f"https://github.com/{full_name}.git"
    result = subprocess.run(
        ["git", "clone", remote_url, str(workspace)],
        capture_output=True, text=True, timeout=_LOCAL_GIT_CONFIG_TIMEOUT,
    )
    if result.returncode != 0:
        log.warning("Failed to clone %s: %s", full_name, result.stderr)
        return None

    log.info("Cloned %s to %s", full_name, workspace)
    _setup_git_credentials(workspace)
    return str(workspace)


def _git_init(workspace: Path) -> None:
    """Initialize a git repo in the workspace and make an initial commit."""
    try:
        subprocess.run(
            ["git", "init", "-b", "main"],
            cwd=str(workspace), capture_output=True, timeout=_LOCAL_GIT_CONFIG_TIMEOUT,
        )
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(workspace), capture_output=True, timeout=_LOCAL_GIT_CONFIG_TIMEOUT,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial workspace"],
            cwd=str(workspace), capture_output=True, timeout=_LOCAL_GIT_CONFIG_TIMEOUT, env=_GIT_ENV,
        )
        log.debug("Initialized git repo in %s", workspace)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        log.warning("Git init failed in %s: %s", workspace, e)


_PRE_COMMIT_HOOK = r"""#!/bin/bash
# Installed by Scientist World swarm. Four gates:
#   1. Credential patterns (tokens, basic-auth URLs) — block
#   2. Per-file size > 95 MB — block (GitHub's hard blob limit is 100 MB)
#   3. Path patterns that are never legitimately committed — block
#   4. Symlinks that point outside the repository — block
# Bypass (NOT RECOMMENDED): git commit --no-verify

# ── Gate 1: credentials ──────────────────────────────────────────────────────
patterns='ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,}|(^|[^A-Za-z0-9])hf_[A-Za-z0-9]{30,}|https?://[^/[:space:]\"'"'"'@]+:[^/[:space:]\"'"'"'@]+@'

if hits=$(git diff --cached --no-color -U0 -- 2>/dev/null | grep -E '^\+' | grep -E "$patterns"); then
    echo "" >&2
    echo "pre-commit: staged changes contain credential-like patterns:" >&2
    echo "$hits" | sed -E 's/(ghp_|github_pat_|hf_)[A-Za-z0-9_]{8}[A-Za-z0-9_]+/\1XXXXXXXX_REDACTED/g; s/:[^@[:space:]]+@/:XXXXXX@/g' | head -10 >&2
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
    echo "" >&2
    echo "Options: add to .gitignore, move to data/downloads/, or delete before committing." >&2
    exit 1
fi

# ── Gate 3: never-legitimate path patterns ───────────────────────────────────
# These paths are always installed or cached, never authored. Committing them
# bloats the repo. If you genuinely need one of these in the repo, either
# rename it or remove this check deliberately.
bad_paths=""
while IFS= read -r -d '' f; do
    case "$f" in
        site-packages/*|*/site-packages/*|node_modules/*|*/node_modules/*|__pycache__/*|*/__pycache__/*|*.egg-info/*|.venv/*|*/.venv/*|.tox/*|*/.tox/*)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        local-packages/*|*/local-packages/*|local_packages*/*|*/local_packages*/*|local-pkgs/*|*/local-pkgs/*|local_pkgs/*|*/local_pkgs/*)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        pylocal/*|*/pylocal/*)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        nvidia/cu*/*|*/nvidia/cu*/*)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        core.[0-9]*|*/core.[0-9]*)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        *.sif|*.img)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        tmp_eval/*|*/tmp_eval/*)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        oci_layout/*|*/oci_layout/*|oci-image/*|*/oci-image/*|oci-images/*|*/oci-images/*)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        docker_*/*|*/docker_*/*|docker_*.tar|docker_*.tar.gz|docker_*.tar.xz)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        wheels/*|*/wheels/*|python_wheels/*|*/python_wheels/*)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        *_extracted/*|*/*_extracted/*)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        nvidia_libs/*|*/nvidia_libs/*|nvidia-libs/*|*/nvidia-libs/*|nvidia_wheels/*|*/nvidia_wheels/*|nvidia-wheels/*|*/nvidia-wheels/*|all_libs/*|*/all_libs/*)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        .pip_cache/*|*/.pip_cache/*|.pip-cache/*|*/.pip-cache/*|pip_cache/*|*/pip_cache/*|pip-cache/*|*/pip-cache/*)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        .hf_cache/*|*/.hf_cache/*|.hf-cache/*|*/.hf-cache/*|hf_cache/*|*/hf_cache/*|.huggingface/*|*/.huggingface/*|huggingface_hub/*|*/huggingface_hub/*)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        *.bundle|*/*.bundle|*.bundle.lock|*/*.bundle.lock)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
        *.tar.xz|*/*.tar.xz)
            bad_paths="${bad_paths}  ${f}"$'\n'
            ;;
    esac
done < <(git diff --cached --name-only --diff-filter=ACMR -z 2>/dev/null)

if [ -n "$bad_paths" ]; then
    echo "" >&2
    echo "pre-commit: staged path(s) match never-legitimate patterns (installed/cached content):" >&2
    printf "%s" "$bad_paths" | head -20 >&2
    total=$(printf "%s" "$bad_paths" | wc -l)
    if [ "$total" -gt 20 ]; then
        echo "  ... and $((total - 20)) more" >&2
    fi
    echo "" >&2
    echo "Add these paths to .gitignore instead of committing them." >&2
    exit 1
fi

# ── Gate 4: portability-breaking symlinks ────────────────────────────────────
# `shared/` is a local cache on this machine; targets there don't exist on a
# fresh clone. Absolute targets outside the workspace also break on clone.
ws_root="$(git rev-parse --show-toplevel 2>/dev/null)"
bad_links=""
while IFS= read -r -d '' f; do
    [ -L "$f" ] || continue
    target=$(readlink "$f")
    case "$target" in
        *"shared/"*)
            bad_links="${bad_links}  ${f} → ${target}  (points into shared/)"$'\n'
            ;;
        /*)
            # absolute path — must resolve inside the workspace
            case "$target" in
                "$ws_root"/*) : ;;
                *) bad_links="${bad_links}  ${f} → ${target}  (absolute, outside workspace)"$'\n' ;;
            esac
            ;;
    esac
done < <(git diff --cached --name-only --diff-filter=ACMR -z 2>/dev/null)

if [ -n "$bad_links" ]; then
    echo "" >&2
    echo "pre-commit: staged symlink(s) point to locations that won't exist on a fresh clone:" >&2
    printf "%s" "$bad_links" >&2
    echo "" >&2
    echo "Options: move under data/downloads/ (gitignored, symlinks safe there)," >&2
    echo "or add a real fetch to scripts/download.sh and remove the symlink." >&2
    exit 1
fi

exit 0
"""


def _install_pre_commit_hook(workspace: Path) -> None:
    """Install the credential-detection pre-commit hook in this workspace.

    Idempotent: overwrites any existing hook with the current version so
    fixes propagate. The hook lives in .git/hooks/, which is not tracked
    by git — so it must be (re)installed per workspace at setup time.
    """
    hooks_dir = Path(workspace) / ".git" / "hooks"
    if not hooks_dir.parent.is_dir():
        return
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / "pre-commit"
    hook_path.write_text(_PRE_COMMIT_HOOK)
    hook_path.chmod(0o755)


# Directories under a workspace that are known to legitimately contain
# their own .git/ (scheduler/tool-owned). Never strip these.
_NESTED_GIT_SKIP_PREFIXES = (
    ".git",          # the workspace's own repo
    ".cache",        # tool caches
    ".claude",       # claude code session storage
    ".claude-tmp",
    ".codex",        # codex cli internals (plugins/.git etc.)
    "jobs",          # scheduler-owned per-job dirs
    "trajectory",    # scheduler-owned
)


def strip_nested_git_dirs(workspace: Path, max_depth: int = 4) -> list[str]:
    """Remove nested .git/ directories so vendored repos become regular folders.

    Agents frequently `git clone` reference baselines into method/, tmp/, etc.,
    which leaves a .git inside that the parent repo can't track (git treats
    them as gitlinks/submodules, so the cloned files are invisible to
    ``git add -A``). Removing the inner .git makes those files normal
    tracked content.

    Idempotent. Skips known scheduler/tool-owned trees in
    ``_NESTED_GIT_SKIP_PREFIXES``. Bounded walk via ``max_depth`` so we
    don't traverse the entire workspace on every call.

    Returns the list of stripped paths (relative to workspace), for logging.
    """
    ws = Path(workspace)
    if not ws.is_dir():
        return []
    stripped: list[str] = []
    # Walk only top-level entries first; recurse manually with depth bound.
    def _walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = list(path.iterdir())
        except (PermissionError, FileNotFoundError):
            return
        for entry in entries:
            try:
                if not entry.is_dir() or entry.is_symlink():
                    continue
            except OSError:
                continue
            rel = entry.relative_to(ws).as_posix()
            # Skip known protected trees at the top level
            if depth == 1 and entry.name in _NESTED_GIT_SKIP_PREFIXES:
                continue
            if entry.name == ".git" and depth >= 2:
                # This is a nested .git — strip it.
                try:
                    shutil.rmtree(entry)
                    stripped.append(rel)
                except OSError as e:
                    log.warning("Failed to strip nested .git at %s: %s", rel, e)
                continue
            _walk(entry, depth + 1)

    _walk(ws, 1)

    # Also strip a top-level .gitmodules — once nested .git dirs are gone,
    # any submodule references become dangling. Safe because we don't use
    # submodules in workspaces.
    gm = ws / ".gitmodules"
    if gm.exists():
        try:
            gm.unlink()
            stripped.append(".gitmodules")
        except OSError:
            pass

    if stripped:
        log.info("Stripped %d nested .git locations in %s: %s",
                 len(stripped), ws.name, stripped[:5])
    return stripped


def _setup_git_credentials(workspace: Path) -> None:
    """Copy host git credentials into workspace so the agent can push.

    The agent runs with HOME=workspace, so we place .git-credentials
    there and configure the local repo to use credential.helper=store.
    Also installs the credential pre-commit hook.
    """
    host_creds = Path.home() / ".git-credentials"
    if not host_creds.exists():
        return

    ws = Path(workspace)
    dest = ws / ".git-credentials"
    if not dest.exists():
        shutil.copy2(str(host_creds), str(dest))

    # Configure credential helper for this repo
    if (ws / ".git").is_dir():
        subprocess.run(
            ["git", "config", "credential.helper", "store"],
            cwd=str(ws), capture_output=True, timeout=_LOCAL_GIT_CONFIG_TIMEOUT,
        )
        _install_pre_commit_hook(ws)

    # Add .git-credentials to .gitignore so it's never committed
    gitignore = ws / ".gitignore"
    gitignore_content = gitignore.read_text() if gitignore.exists() else ""
    if ".git-credentials" not in gitignore_content:
        with open(str(gitignore), "a") as f:
            if gitignore_content and not gitignore_content.endswith("\n"):
                f.write("\n")
            f.write(".git-credentials\n")


def _github_create_and_push(
    workspace: Path,
    paper: PaperInfo,
    github_org: str,
    date: str | None = None,
    topics: list[str] | None = None,
) -> None:
    """Create a GitHub repo under the org and push the workspace.

    The repository is created private through the organisation endpoint,
    so ``github_org`` must be an organisation, not a user account.
    """
    token = _get_github_token()
    if not token:
        log.warning("No GitHub token found — cannot create repo for %s", paper.arxiv_id)
        return

    repo_name = paper.repo_name()
    full_name = f"{github_org}/{repo_name}"

    # Check if repo already exists
    if not _github_repo_exists(full_name, token):
        # Create the repo
        description = paper.title or f"Reproduction of arxiv:{paper.arxiv_id}"
        status, resp = _github_api("POST", f"/orgs/{github_org}/repos", token, {
            "name": repo_name,
            "description": description[:350],  # GitHub limit
            "private": True,
            "auto_init": False,
        })
        if status not in (200, 201):
            msg = resp.get("message", "") if resp else "unknown error"
            log.warning("Failed to create GitHub repo %s: %s", full_name, msg)
            return
        log.info("Created GitHub repo %s", full_name)

    # Add remote and push
    remote_url = f"https://github.com/{full_name}.git"
    _ensure_git_remote_url(workspace, "origin", remote_url)
    # Always publish local HEAD to remote ``main`` — do not follow the
    # current branch. Prevents silently creating ``master`` branches when
    # a clone picked up the old git default.
    result = None
    for attempt, delay_s in enumerate((0.0, *_INITIAL_GITHUB_PUSH_RETRY_DELAYS), start=1):
        if delay_s:
            time.sleep(delay_s)
        result = subprocess.run(
            ["git", "push", "--force", "origin", "HEAD:main"],
            cwd=str(workspace), capture_output=True, text=True,
            timeout=_LOCAL_GIT_CONFIG_TIMEOUT, env=_GIT_ENV,
        )
        if result.returncode == 0:
            break
        log.warning(
            "Failed to push to %s (attempt %d/%d): %s",
            full_name,
            attempt,
            1 + len(_INITIAL_GITHUB_PUSH_RETRY_DELAYS),
            (result.stderr or "").strip(),
        )
    if result.returncode != 0:
        log.warning("Failed to push to %s after retries: %s", full_name, result.stderr)
    else:
        log.info("Pushed workspace to %s", full_name)

    # Set initial topics (date + custom)
    init_topics = list(topics or [])
    if date:
        init_topics.append(f"date-{date}")
    if init_topics:
        init_topics, rewritten_topics, dropped_topics = _sanitize_github_topics(init_topics)
        if rewritten_topics:
            log.warning(
                "Rewriting GitHub topics for %s: %s",
                full_name,
                rewritten_topics,
            )
        if dropped_topics:
            log.warning(
                "Dropping invalid GitHub topics for %s: %s",
                full_name,
                dropped_topics,
            )
        if not init_topics:
            return
        status, _ = _github_api("PUT", f"/repos/{full_name}/topics", token, {
            "names": init_topics[:20],
        })
        if status == 200:
            log.info("Set topics %s on %s", init_topics, full_name)
        else:
            log.debug("Failed to set topics on %s", full_name)
