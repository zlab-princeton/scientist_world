"""Safety tests for evaluation.git_publish.

Publishing is opt-in, and when it is on it must never write ``main`` /
``master``, never use ``--all`` / ``--mirror`` or a bare ``--force``, and
never put the access token on a command line. The strategy is:

- Drive the public API (``publish()``) with a temp env_dir mocked to
  look like a completed run.
- Mock ``subprocess.run`` to capture every git invocation; assert no
  ``git push`` call is ever made with a forbidden form.
- Test branch-name validation directly with crafted inputs.

Nothing here runs git against a remote or opens a network connection.
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from evaluation import config as _config
from evaluation import git_publish

ORG = "example-org"
PAPER = "1234.56789"
PROFILE = "model-a-xhigh"
BRANCH = f"improve-{PROFILE}"


@pytest.fixture(autouse=True)
def publishing_on(eval_config, monkeypatch):
    """Most tests exercise the enabled path; the disabled path has its own."""
    eval_config(publish={"enabled": True, "github_org": ORG})
    monkeypatch.setattr(git_publish, "_REPO_NAME_CACHE", {})


def _stub_title(title: str | None):
    return lambda paper_id, source_dir=None: title


# ── Branch-name validation ────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["main", "master", "HEAD", "", "Main", "MAIN", "head"])
def test_assert_safe_branch_rejects_forbidden(bad: str) -> None:
    with pytest.raises(ValueError):
        git_publish._assert_safe_branch(bad)


@pytest.mark.parametrize("bad", ["foo/bar", "with space", "has:colon"])
def test_assert_safe_branch_rejects_illegal_chars(bad: str) -> None:
    with pytest.raises(ValueError):
        git_publish._assert_safe_branch(bad)


@pytest.mark.parametrize(
    "good",
    [
        "improve-model-a-xhigh",
        "innovate-model-b-xhigh",
        "improve-model-c-5.2-medium",
        "improve-model_d-high-grader",
    ],
)
def test_assert_safe_branch_accepts_normal(good: str) -> None:
    git_publish._assert_safe_branch(good)


def test_publish_refuses_forbidden_mode_branch(tmp_path: Path) -> None:
    """A mode of 'main' would yield branch 'main-<profile>' which IS safe,
    but the components themselves shouldn't matter — only the joined name.
    Verify a manually-set branch="main" is rejected."""
    with pytest.raises(ValueError):
        git_publish._safe_push(tmp_path, "main", "org/repo", "tok")
    with pytest.raises(ValueError):
        git_publish._safe_push(tmp_path, "master", "org/repo", "tok")


# ── safe_push command-shape tests ─────────────────────────────────────────


def _captured_pushes(captured: list[list[str]]) -> list[list[str]]:
    return [c for c in captured if len(c) >= 2 and c[0] == "git" and c[1] == "push"]


def test_safe_push_uses_explicit_refspec(tmp_path: Path, monkeypatch) -> None:
    """The push must always use ``<branch>:<branch>`` form, never bare,
    never ``--all`` / ``--mirror`` / bare ``--force``."""
    captured: list[list[str]] = []
    fake_sha = "a" * 40  # 40-hex sha

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        m = mock.Mock()
        m.returncode = 0
        # rev-parse FETCH_HEAD must yield a valid 40-hex sha
        if len(cmd) >= 3 and cmd[1] == "rev-parse" and cmd[2] == "FETCH_HEAD":
            m.stdout = fake_sha + "\n"
        else:
            m.stdout = ""
        m.stderr = ""
        return m

    monkeypatch.setattr(git_publish.subprocess, "run", fake_run)
    git_publish._safe_push(tmp_path, BRANCH, "ORG/repo", "TOKEN")

    pushes = _captured_pushes(captured)
    assert len(pushes) == 1, f"expected exactly one push, got {pushes}"
    cmd = pushes[0]

    # Refspec form
    assert f"{BRANCH}:{BRANCH}" in cmd
    # Forbidden flags
    for f in ("--all", "--mirror"):
        assert f not in cmd, f"safe_push should never use {f}"
    # Force only via --force-with-lease, scoped to the branch and pinned
    # to the sha that was just fetched.
    assert "--force" not in cmd
    assert f"--force-with-lease={BRANCH}:{fake_sha}" in cmd
    # No :main / :master refspec sneaks through
    joined = " ".join(cmd)
    assert ":main" not in joined
    assert ":master" not in joined


def test_safe_push_leases_against_nothing_for_a_new_branch(
    tmp_path: Path, monkeypatch
) -> None:
    """When the branch does not exist remotely the fetch fails; the lease
    is then empty, so the push only succeeds if nobody created the branch
    in the meantime."""
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        failed = cmd[1] == "fetch"
        return mock.Mock(returncode=1 if failed else 0, stdout="", stderr="")

    monkeypatch.setattr(git_publish.subprocess, "run", fake_run)
    git_publish._safe_push(tmp_path, BRANCH, "ORG/repo", "TOKEN")

    (cmd,) = _captured_pushes(captured)
    assert f"--force-with-lease={BRANCH}:" in cmd
    assert not any(c[1] == "rev-parse" for c in captured)


def test_safe_push_rejects_a_bogus_fetch_head(tmp_path: Path, monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        return mock.Mock(returncode=0, stdout="not-a-sha\n", stderr="")

    monkeypatch.setattr(git_publish.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="bogus FETCH_HEAD"):
        git_publish._safe_push(tmp_path, BRANCH, "ORG/repo", "TOKEN")


def test_safe_push_aborts_if_cmd_contains_forbidden_token(
    tmp_path: Path, monkeypatch
) -> None:
    """A hostile branch name such as "x:main" is rejected by the branch
    check before any git command runs."""
    captured: list[list[str]] = []

    monkeypatch.setattr(git_publish.subprocess, "run",
                        lambda cmd, **k: captured.append(list(cmd)) or mock.Mock(returncode=0))

    with pytest.raises(ValueError):
        git_publish._safe_push(tmp_path, "x:main", "org/repo", "TOKEN")
    assert len(captured) == 0


# ── The access token never reaches a command line ─────────────────────────


def _header_for(token: str) -> str:
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return f"AUTHORIZATION: basic {basic}"


def test_token_travels_in_the_environment_not_on_argv(
    tmp_path: Path, monkeypatch
) -> None:
    token = "tok_SECRET_value_123"
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        calls.append((list(cmd), dict(kwargs.get("env") or {})))
        m = mock.Mock(returncode=0, stderr="")
        m.stdout = "d" * 40 + "\n" if cmd[1] == "rev-parse" else ""
        return m

    monkeypatch.setattr(git_publish.subprocess, "run", fake_run)
    git_publish._safe_push(tmp_path, BRANCH, f"{ORG}/repo", token)

    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    for cmd, _env in calls:
        joined = " ".join(cmd)
        assert token not in joined, f"token on the command line: {cmd[:2]}"
        assert encoded not in joined
        assert "@" not in joined, "credentials embedded in a URL"

    networked = [(cmd, env) for cmd, env in calls if cmd[1] in ("fetch", "push")]
    assert [cmd[1] for cmd, _ in networked] == ["fetch", "push"]
    for cmd, env in networked:
        assert f"https://github.com/{ORG}/repo.git" in cmd
        n = int(env["GIT_CONFIG_COUNT"])
        assert n >= 1
        assert env[f"GIT_CONFIG_KEY_{n - 1}"] == "http.https://github.com/.extraheader"
        assert env[f"GIT_CONFIG_VALUE_{n - 1}"] == _header_for(token)
        assert env["GIT_TERMINAL_PROMPT"] == "0"

    # Local commands do not get the credential at all.
    (rev_env,) = [env for cmd, env in calls if cmd[1] == "rev-parse"]
    assert not any(token in str(v) or encoded in str(v) for v in rev_env.values())


def test_auth_env_keeps_git_config_already_in_the_environment(monkeypatch) -> None:
    monkeypatch.setitem(git_publish._GIT_ENV, "GIT_CONFIG_COUNT", "2")
    monkeypatch.setitem(git_publish._GIT_ENV, "GIT_CONFIG_KEY_0", "a.b")
    monkeypatch.setitem(git_publish._GIT_ENV, "GIT_CONFIG_VALUE_0", "1")
    monkeypatch.setitem(git_publish._GIT_ENV, "GIT_CONFIG_KEY_1", "c.d")
    monkeypatch.setitem(git_publish._GIT_ENV, "GIT_CONFIG_VALUE_1", "2")

    env = git_publish._auth_env("TOK")
    assert env["GIT_CONFIG_COUNT"] == "3"
    assert (env["GIT_CONFIG_KEY_0"], env["GIT_CONFIG_VALUE_0"]) == ("a.b", "1")
    assert (env["GIT_CONFIG_KEY_1"], env["GIT_CONFIG_VALUE_1"]) == ("c.d", "2")
    assert env["GIT_CONFIG_KEY_2"] == "http.https://github.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_2"] == _header_for("TOK")
    # The shared environment used for local git commands is left alone.
    assert git_publish._GIT_ENV["GIT_CONFIG_COUNT"] == "2"
    assert "GIT_CONFIG_KEY_2" not in git_publish._GIT_ENV


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_itself_picks_the_header_up_from_the_environment(tmp_path: Path) -> None:
    """The real ``git`` (2.31 or newer) resolves the header for github.com
    URLs, and only for those. A local config query: nothing is contacted."""
    env = git_publish._auth_env("TOK")

    def lookup(url: str) -> str:
        proc = subprocess.run(
            ["git", "config", "--get-urlmatch", "http.extraheader", url],
            cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=60)
        return proc.stdout.strip()

    assert lookup(f"https://github.com/{ORG}/repo.git") == _header_for("TOK")
    assert lookup("https://example.com/other/repo.git") == ""


def test_push_failure_message_does_not_leak_the_token(
    tmp_path: Path, monkeypatch
) -> None:
    token = "tok_SECRET_value_123"
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()

    def fake_run(cmd, **kwargs):
        if cmd[1] == "push":
            return mock.Mock(returncode=1, stdout="",
                             stderr=f"fatal: auth failed for {token} / {encoded}")
        return mock.Mock(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(git_publish.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as exc:
        git_publish._safe_push(tmp_path, BRANCH, f"{ORG}/repo", token)
    assert token not in str(exc.value)
    assert encoded not in str(exc.value)
    assert "REDACTED" in str(exc.value)


# ── Whole-publish flow: never-touch-main under any input ──────────────────


def _make_completed_env(tmp: Path, paper_id: str, profile: str, mode: str = "improve") -> Path:
    env = tmp / paper_id / f"{profile}-{mode}-eval"
    env.mkdir(parents=True)
    (env / "method").mkdir()
    (env / "method" / "solution.py").write_text("# stub\n")
    (env / "scoring").mkdir()
    (env / "scoring" / "scores_train.json").write_text('{"experiments": {}}')
    (env / "scoring" / "scores_test.json").write_text('{"experiments": {}}')
    (env / "trajectory").mkdir()
    (env / "trajectory" / "turn_0001_raw.jsonl").write_text(
        '{"event":"init"}\n{"event":"hello"}\n'
    )
    (env / "idea").mkdir()
    (env / "idea" / "idea_v1.md").write_text("# Idea v1\n\nFirst pass.\n")
    (env / "reflection.md").write_text("# Reflection\n\nIt worked.\n")
    (env / ".evaluation.json").write_text(json.dumps({
        "run_id": f"{paper_id}-{profile}-{mode}-stub",
        "paper_id": paper_id,
        "model": profile,
        "mode": mode,
        "status": "completed",
        "state": "done",
        "current_turn": 1,
        "attempt": 1,
        "train_score": {"experiments": {}},
        "test_score": {"experiments": {}},
        "usage": {"scaffold": "claude-code"},
    }))
    return env


def _deny_everything(monkeypatch) -> tuple[list, list]:
    git_calls: list[list[str]] = []
    http_calls: list[str] = []

    def deny_run(cmd, **k):
        git_calls.append(list(cmd))
        raise AssertionError(f"should not invoke a subprocess: {cmd}")

    def deny_urlopen(*a, **k):
        http_calls.append(str(a[0]))
        raise AssertionError(f"should not hit the network: {a}")

    monkeypatch.setattr(git_publish.subprocess, "run", deny_run)
    monkeypatch.setattr(git_publish.urllib.request, "urlopen", deny_urlopen)
    return git_calls, http_calls


def test_publish_dry_run_never_invokes_git(tmp_path: Path, monkeypatch) -> None:
    """Dry-run path must touch zero git commands and zero network."""
    env = _make_completed_env(tmp_path, PAPER, PROFILE)
    git_calls, http_calls = _deny_everything(monkeypatch)
    # The dry run still reports "no token" when there is none, so give it one.
    monkeypatch.setattr(git_publish, "_get_github_token", lambda: "FAKE_TOKEN")
    monkeypatch.setattr(git_publish, "_paper_title", _stub_title("Stub Paper Title"))

    res = git_publish.publish(env, PAPER, PROFILE, "improve", dry_run=True)
    assert res.pushed is False
    assert res.skipped_reason == "dry-run"
    assert res.branch == BRANCH
    assert res.repo_full_name == f"{ORG}/{PAPER}-stub-paper-title"
    assert res.files_committed > 0
    assert git_calls == []
    assert http_calls == []


def test_publish_without_token_is_skipped(tmp_path: Path, monkeypatch) -> None:
    env = _make_completed_env(tmp_path, PAPER, PROFILE)
    git_calls, http_calls = _deny_everything(monkeypatch)
    monkeypatch.setattr(git_publish, "_get_github_token", lambda: None)

    res = git_publish.publish(env, PAPER, PROFILE, "improve")
    assert res.pushed is False
    assert "token" in (res.skipped_reason or "")
    assert git_calls == [] and http_calls == []


def test_publish_skips_non_completed(tmp_path: Path, monkeypatch) -> None:
    env = _make_completed_env(tmp_path, PAPER, PROFILE)
    # Mark as failed
    meta = json.loads((env / ".evaluation.json").read_text())
    meta["status"] = "failed"
    (env / ".evaluation.json").write_text(json.dumps(meta))

    _deny_everything(monkeypatch)
    res = git_publish.publish(env, PAPER, PROFILE, "improve")
    assert res.pushed is False
    assert "completed" in (res.skipped_reason or "")


def test_no_git_push_call_ever_targets_main(tmp_path: Path, monkeypatch) -> None:
    """End-to-end behavioral assertion: across the whole publish flow, no
    invocation of `git push` carries a refspec that writes main/master."""
    env = _make_completed_env(tmp_path, PAPER, PROFILE)
    captured: list[list[str]] = []

    fake_sha = "b" * 40

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        m = mock.Mock()
        m.returncode = 0
        # symbolic-ref must return the branch we just initialized
        if len(cmd) >= 3 and cmd[1] == "symbolic-ref" and cmd[2] == "--short":
            m.stdout = f"{BRANCH}\n"
        elif len(cmd) >= 3 and cmd[1] == "ls-tree":
            m.stdout = "method/solution.py\nREADME.md\nmanifest.json\n"
        elif len(cmd) >= 3 and cmd[1] == "rev-parse" and cmd[2] == "FETCH_HEAD":
            m.stdout = fake_sha + "\n"
        else:
            m.stdout = ""
        m.stderr = ""
        return m

    monkeypatch.setattr(git_publish.subprocess, "run", fake_run)
    monkeypatch.setattr(git_publish, "_paper_title", _stub_title("Stub Paper"))
    monkeypatch.setattr(git_publish, "_get_github_token", lambda: "FAKE_TOKEN")
    monkeypatch.setattr(git_publish, "_repo_exists", lambda *a, **k: True)

    res = git_publish.publish(env, PAPER, PROFILE, "improve")
    assert res.pushed is True
    assert res.repo_full_name == f"{ORG}/{PAPER}-stub-paper"

    pushes = [c for c in captured if len(c) >= 2 and c[0] == "git" and c[1] == "push"]
    assert pushes, "expected at least one git push"
    for cmd in pushes:
        joined = " ".join(cmd)
        assert ":main" not in joined, f"{joined} writes :main"
        assert ":master" not in joined
        assert "--all" not in cmd
        assert "--mirror" not in cmd
        # Bare --force forbidden (only --force-with-lease=... allowed)
        assert "--force" not in cmd
        assert any(arg.startswith("--force-with-lease=") for arg in cmd), \
            f"expected --force-with-lease=…, got {cmd}"
        # Refspec form must be exactly <branch>:<branch>
        refspec = next(
            (a for a in cmd if re.fullmatch(r"[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+", a)),
            None,
        )
        assert refspec is not None, f"no <b>:<b> refspec in {cmd}"
        left, right = refspec.split(":", 1)
        assert left == right, f"asymmetric refspec {refspec}"
        assert left not in {"main", "master", "HEAD"}
    # The token is on no command line of the whole flow.
    assert not any("FAKE_TOKEN" in " ".join(c) for c in captured)


def test_init_uses_explicit_branch_not_main(tmp_path: Path, monkeypatch) -> None:
    """`git init` must always be called with `-b <branch>` so a local
    main branch never exists in the staging repo."""
    env = _make_completed_env(tmp_path, PAPER, PROFILE)
    captured: list[list[str]] = []

    def fake_run(cmd, **k):
        captured.append(list(cmd))
        m = mock.Mock(); m.returncode = 0; m.stderr = ""
        if len(cmd) >= 3 and cmd[1] == "rev-parse" and cmd[2] == "FETCH_HEAD":
            m.stdout = "c" * 40 + "\n"
        else:
            m.stdout = f"{BRANCH}\n"
        return m

    monkeypatch.setattr(git_publish.subprocess, "run", fake_run)
    monkeypatch.setattr(git_publish, "_paper_title", _stub_title("Stub"))
    monkeypatch.setattr(git_publish, "_get_github_token", lambda: "TOK")
    monkeypatch.setattr(git_publish, "_repo_exists", lambda *a, **k: True)

    git_publish.publish(env, PAPER, PROFILE, "improve")

    inits = [c for c in captured if len(c) >= 2 and c[0] == "git" and c[1] == "init"]
    assert inits, "expected git init"
    for cmd in inits:
        # Must specify -b <branch> with a non-main name
        assert "-b" in cmd
        i = cmd.index("-b")
        branch_arg = cmd[i + 1]
        assert branch_arg not in {"main", "master", "HEAD", ""}
        assert branch_arg == BRANCH


# ── Staging with the real git binary (local only, nothing is pushed) ─────

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), env=git_publish._GIT_ENV,
                          capture_output=True, text=True, check=check, timeout=120)


def _workspace_repo(env: Path) -> None:
    """Give a run workspace the local history the harness would have made."""
    _git(env, "init", "-b", "main")
    _git(env, "add", "-A")
    _git(env, "commit", "-m", "eval: initial scaffold")
    (env / "method" / "solution.py").write_text("# stub\nresult = 1\n")
    (env / "jobs" / "j1").mkdir(parents=True)
    (env / "jobs" / "j1" / "stdout.log").write_text("job output\n")
    _git(env, "add", "-A")
    _git(env, "commit", "-m", "eval: job j1 done (rc=0)")


@needs_git
def test_staging_replays_history_onto_the_publish_branch(tmp_path: Path) -> None:
    env = _make_completed_env(tmp_path, PAPER, PROFILE)
    _workspace_repo(env)
    meta = json.loads((env / ".evaluation.json").read_text())

    staging = git_publish._stage_workspace(
        env, PAPER, PROFILE, "improve", meta, BRANCH, repo_name=f"{PAPER}-some-title")
    try:
        n_files, n_bytes, _n_parts = git_publish._commit_staging(staging, BRANCH)
        assert n_files > 0 and n_bytes > 0

        tracked = set(_git(staging, "ls-files").stdout.split())
        # reflection.md is published next to the ideas
        assert {"README.md", "manifest.json", "method/solution.py",
                "scoring/scores_train.json", "idea/idea_v1.md", "idea/reflection.md",
                "trajectory/turn_0001_raw.jsonl", ".evaluation.json"} <= tracked
        # Only the listed parts of a workspace are published.
        assert not any(name.startswith("jobs/") for name in tracked)
        assert ".publish-stats.json" not in tracked

        # The only branch is the publish branch; main never exists.
        assert _git(staging, "branch", "--format=%(refname:short)").stdout.split() == [BRANCH]
        # History: the workspace's commits under the profile's name, then
        # the generated summary commit by the publishing identity.
        log = _git(staging, "log", "--reverse", "--format=%an|%s").stdout.splitlines()
        assert log[0] == f"{PROFILE}|eval: initial scaffold"
        assert log[1] == f"{PROFILE}|eval: job j1 done (rc=0)"
        assert len(log) == 3 and not log[2].startswith(f"{PROFILE}|")
        assert (staging / "README.md").read_text().startswith(
            f"# {PAPER}-some-title — `{PAPER}`")
        # The run workspace itself is untouched: no remote, still on main.
        assert _git(env, "remote").stdout.strip() == ""
        assert _git(env, "log", "--format=%s").stdout.splitlines()[0] == (
            "eval: job j1 done (rc=0)")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@needs_git
def test_staging_a_workspace_without_history(tmp_path: Path, monkeypatch) -> None:
    """No local repository: the current files are staged as one commit,
    and a trajectory above the size limit is split."""
    monkeypatch.setattr(git_publish, "_TRAJECTORY_SPLIT_BYTES", 100)
    env = _make_completed_env(tmp_path, PAPER, PROFILE)
    (env / "trajectory" / "turn_0002_raw.jsonl").write_text(
        "\n".join(json.dumps({"i": i, "pad": "x" * 30}) for i in range(5)) + "\n")
    meta = json.loads((env / ".evaluation.json").read_text())

    staging = git_publish._stage_workspace(env, PAPER, PROFILE, "improve", meta, BRANCH)
    try:
        n_files, _n_bytes, n_parts = git_publish._commit_staging(staging, BRANCH)
        assert n_files > 0
        # one part for the small trajectory file, at least two for the split one
        assert n_parts >= 3, "the part count of the fallback path was lost"
        tracked = set(_git(staging, "ls-files").stdout.split())
        assert "trajectory/turn_0002_raw.index.json" in tracked
        assert "method/solution.py" in tracked
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@needs_git
def test_staging_redacts_credentials_from_published_files(tmp_path: Path) -> None:
    env = _make_completed_env(tmp_path, PAPER, PROFILE)
    secret = "sk-ant-" + "s" * 60
    (env / "trajectory" / "turn_0001_raw.jsonl").write_text(
        json.dumps({"text": f"export ANTHROPIC_API_KEY={secret}"}) + "\n")
    (env / "reflection.md").write_text(f"I used the key {secret}.\n")
    (env / "idea" / "idea_v1.md").write_text(f"# Idea\n\nCall the API with {secret}.\n")
    _workspace_repo(env)
    meta = json.loads((env / ".evaluation.json").read_text())

    staging = git_publish._stage_workspace(env, PAPER, PROFILE, "improve", meta, BRANCH)
    try:
        git_publish._commit_staging(staging, BRANCH)
        everything = _git(staging, "log", "-p", "--all").stdout
        assert secret not in everything
        assert "sk-ant-REDACTED" in everything
        # The generated README quotes the idea and the reflection.
        readme = (staging / "README.md").read_text()
        assert readme.count("sk-ant-REDACTED") == 2 and secret not in readme
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _one_commit_repo(path: Path, branch: str, content: str) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-b", branch)
    (path / "file.txt").write_text(content)
    _git(path, "add", "-A")
    _git(path, "commit", "-m", f"commit on {branch}")
    return _git(path, "rev-parse", "HEAD").stdout.strip()


@needs_git
def test_safe_push_end_to_end_against_a_local_repository(tmp_path: Path, monkeypatch) -> None:
    """The real fetch / lease / push sequence, with a bare repository on
    disk standing in for GitHub (file transport, no network)."""
    remote_root = tmp_path / "remote"
    bare = remote_root / ORG / "repo.git"
    bare.mkdir(parents=True)
    _git(bare, "init", "--bare", "-b", "main")
    seed = tmp_path / "seed"
    main_sha = _one_commit_repo(seed, "main", "the environment itself\n")
    _git(seed, "push", str(bare), "main:main")
    monkeypatch.setattr(git_publish, "_GITHUB_URL", remote_root.as_uri() + "/")

    def remote(ref: str) -> str:
        return _git(bare, "rev-parse", ref).stdout.strip()

    # First publish: the branch does not exist remotely yet.
    first = tmp_path / "staging1"
    first_sha = _one_commit_repo(first, BRANCH, "first publish\n")
    git_publish._safe_push(first, BRANCH, f"{ORG}/repo", "TOKEN")
    assert remote(BRANCH) == first_sha
    assert remote("main") == main_sha

    # Publishing the run again rewrites the branch (fresh staging history):
    # the lease is pinned to the commit that was just fetched.
    second = tmp_path / "staging2"
    second_sha = _one_commit_repo(second, BRANCH, "second publish\n")
    assert second_sha != first_sha
    git_publish._safe_push(second, BRANCH, f"{ORG}/repo", "TOKEN")
    assert remote(BRANCH) == second_sha

    # main is untouched and nothing else was created.
    assert remote("main") == main_sha
    branches = set(_git(bare, "branch", "--format=%(refname:short)").stdout.split())
    assert branches == {"main", BRANCH}
    # No credential ended up in the staging repository's configuration.
    assert "TOKEN" not in (second / ".git" / "config").read_text()
    assert _git(second, "remote").stdout.strip() == ""


@pytest.fixture
def local_github(tmp_path: Path, monkeypatch) -> Path:
    """A bare repository on disk in place of ``github.com/<ORG>/<paper>-a-paper``,
    with a ``main`` branch, plus stubs for the token and the API lookup."""
    remote_root = tmp_path / "remote"
    bare = remote_root / ORG / f"{PAPER}-a-paper.git"
    bare.mkdir(parents=True)
    _git(bare, "init", "--bare", "-b", "main")
    _one_commit_repo(tmp_path / "seed", "main", "the environment itself\n")
    _git(tmp_path / "seed", "push", str(bare), "main:main")
    monkeypatch.setattr(git_publish, "_GITHUB_URL", remote_root.as_uri() + "/")
    monkeypatch.setattr(git_publish, "_get_github_token", lambda: "TOKEN")
    monkeypatch.setattr(git_publish, "_repo_exists", lambda full_name, token: True)

    def no_network(*a, **k):
        raise AssertionError("no HTTP request expected")

    monkeypatch.setattr(git_publish.urllib.request, "urlopen", no_network)
    return bare


def _remote_files(bare: Path, branch: str) -> set[str]:
    return set(_git(bare, "ls-tree", "-r", "--name-only", branch).stdout.split())


@needs_git
def test_publish_end_to_end_against_a_local_repository(
    tmp_path: Path, local_github: Path,
) -> None:
    runs = tmp_path / "data" / "envs"
    env = _make_completed_env(runs, PAPER, PROFILE)
    (env.parent / "source").mkdir()
    (env.parent / "source" / "manifest.json").write_text(
        json.dumps({"paper": {"title": "A Paper"}}))
    _workspace_repo(env)
    main_before = _git(local_github, "rev-parse", "main").stdout.strip()

    res = git_publish.publish(env, PAPER, PROFILE, "improve")

    assert res.pushed is True
    assert res.repo_full_name == f"{ORG}/{PAPER}-a-paper" and res.branch == BRANCH
    files = _remote_files(local_github, BRANCH)
    assert {"README.md", "manifest.json", "method/solution.py",
            "scoring/scores_train.json", "idea/reflection.md"} <= files
    assert res.files_committed == len(files)
    readme = _git(local_github, "show", f"{BRANCH}:README.md").stdout
    assert readme.startswith(f"# A Paper — `{PAPER}`")
    manifest = json.loads(_git(local_github, "show", f"{BRANCH}:manifest.json").stdout)
    assert manifest["paper_title"] == "A Paper" and manifest["branch"] == BRANCH
    assert _git(local_github, "rev-parse", "main").stdout.strip() == main_before
    # Publishing the same run again replaces the branch and still works.
    assert git_publish.publish(env, PAPER, PROFILE, "improve").pushed is True
    # The run workspace was only read.
    assert _git(env, "remote").stdout.strip() == ""
    assert _git(env, "status", "--porcelain").stdout.strip() == ""


@needs_git
def test_publish_coeffs_end_to_end_against_a_local_repository(
    tmp_path: Path, local_github: Path, eval_config,
) -> None:
    data_dir = eval_config()
    paper_dir = data_dir / "envs" / PAPER
    norm = paper_dir / "normalization"
    norm.mkdir(parents=True)
    (norm / "coeffs_improve_test.json").write_text(json.dumps({"locked": True, "cells": []}))
    (norm / "coeffs_improve_train.json").write_text(json.dumps({"locked": True, "cells": []}))
    (norm / "coeffs_innovate_test.json").write_text(json.dumps({"excluded": True}))
    (paper_dir / "source").mkdir()
    (paper_dir / "source" / "manifest.json").write_text(
        json.dumps({"paper": {"title": "A Paper"}}))

    res = git_publish.publish_coeffs(PAPER)
    assert res.pushed is True and res.branch == "normalization"
    assert res.files_committed == 2
    assert _remote_files(local_github, "normalization") == {
        "README.md", "normalization/coeffs_improve_test.json",
        "normalization/coeffs_improve_train.json"}

    # Nothing locked: nothing to publish.
    for f in norm.glob("coeffs_improve_*.json"):
        f.write_text(json.dumps({"locked": False}))
    again = git_publish.publish_coeffs(PAPER)
    assert again.pushed is False and again.skipped_reason == "no locked coeffs"


@needs_git
@pytest.mark.parametrize("branch, filename, content, blocked_by", [
    ("main", "ok.txt", "fine\n", "refusing to commit on branch"),
    ("master", "ok.txt", "fine\n", "refusing to commit on branch"),
    (BRANCH, "leak.txt", "token ghp_" + "t" * 36 + "\n", "credential-like patterns"),
    (BRANCH, "url.txt", "https://user:hunter2@example.com/x.git\n", "credential-like patterns"),
    (BRANCH, "image.sif", "x\n", "never-legitimate patterns"),
    (BRANCH, "a/node_modules/b.js", "x\n", "never-legitimate patterns"),
])
def test_pre_commit_hook_blocks(tmp_path: Path, branch, filename, content, blocked_by) -> None:
    """The hook installed in the staging repository is a second line of
    defence under the Python-side checks."""
    repo = tmp_path / "staging"
    repo.mkdir()
    _git(repo, "init", "-b", branch)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(git_publish._PRE_COMMIT_HOOK)
    hook.chmod(0o755)
    target = repo / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    _git(repo, "add", "-A")

    proc = _git(repo, "commit", "-m", "attempt", check=False)
    assert proc.returncode != 0
    assert blocked_by in proc.stderr
    assert "ghp_" + "t" * 36 not in proc.stderr, "the hook must not echo the secret"


@needs_git
def test_pre_commit_hook_allows_a_normal_commit(tmp_path: Path) -> None:
    repo = tmp_path / "staging"
    repo.mkdir()
    _git(repo, "init", "-b", BRANCH)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(git_publish._PRE_COMMIT_HOOK)
    hook.chmod(0o755)
    (repo / "method.py").write_text("lr = 3e-4  # see https://example.com/paper\n")
    _git(repo, "add", "-A")
    assert _git(repo, "commit", "-m", "normal", check=False).returncode == 0


# ── Publishing is opt-in ─────────────────────────────────────────────────


def test_publishing_is_disabled_by_default(eval_config) -> None:
    assert git_publish.publish_enabled() is True  # this module's fixture
    eval_config(publish={})
    assert git_publish.publish_enabled() is False
    assert git_publish.publish_enabled({"publish": {"enabled": True}}) is True
    assert git_publish.publish_enabled({}) is False


def test_disabled_publish_makes_no_calls(tmp_path: Path, monkeypatch, eval_config) -> None:
    """With publishing off, a completed run is reported as skipped without
    reading credentials, running git or opening a connection."""
    eval_config(publish={"enabled": False, "github_org": ORG})
    env = _make_completed_env(tmp_path, PAPER, PROFILE)
    git_calls, http_calls = _deny_everything(monkeypatch)

    def no_token():
        raise AssertionError("credentials must not be read when publishing is off")

    monkeypatch.setattr(git_publish, "_get_github_token", no_token)

    res = git_publish.publish(env, PAPER, PROFILE, "improve")
    assert res.pushed is False
    assert res.skipped_reason == git_publish.DISABLED_REASON
    assert res.branch == BRANCH
    dry = git_publish.publish(env, PAPER, PROFILE, "improve", dry_run=True)
    assert dry.skipped_reason == git_publish.DISABLED_REASON
    coeffs = git_publish.publish_coeffs(PAPER)
    assert coeffs.pushed is False
    assert coeffs.skipped_reason == git_publish.DISABLED_REASON
    assert git_calls == [] and http_calls == []


def test_disabled_publish_still_rejects_a_forbidden_branch(eval_config) -> None:
    eval_config(publish={"enabled": False})
    with pytest.raises(ValueError):
        git_publish.publish_coeffs(PAPER, branch="main")


# ── GitHub organisation ──────────────────────────────────────────────────


def test_github_org_comes_from_the_evaluation_config(tmp_path: Path, monkeypatch) -> None:
    swarm_cfg = tmp_path / "swarm.yaml"
    swarm_cfg.write_text("github_org: swarm-org\n")
    monkeypatch.setattr(git_publish, "_SWARM_CONFIG_PATH", swarm_cfg)
    assert git_publish._github_org() == ORG


def test_github_org_falls_back_to_swarm_config(tmp_path: Path, monkeypatch, eval_config) -> None:
    eval_config(publish={"enabled": True, "github_org": ""})
    swarm_cfg = tmp_path / "swarm.yaml"
    swarm_cfg.write_text("github_org: swarm-org\n")
    monkeypatch.setattr(git_publish, "_SWARM_CONFIG_PATH", swarm_cfg)
    assert git_publish._github_org() == "swarm-org"


def test_github_org_missing_is_a_config_error(tmp_path: Path, monkeypatch, eval_config) -> None:
    eval_config(publish={"enabled": True, "github_org": ""})
    monkeypatch.setattr(git_publish, "_SWARM_CONFIG_PATH", tmp_path / "absent.yaml")
    with pytest.raises(_config.ConfigError, match="publish.github_org"):
        git_publish._github_org()


# ── Paper title and repository name ──────────────────────────────────────


def test_paper_title_from_the_environment_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    assert git_publish._paper_title(PAPER, source) is None
    assert git_publish._paper_title(PAPER) is None
    (source / "manifest.json").write_text(json.dumps(
        {"paper": {"title": "  A Study of Things: Part 2  "}}))
    assert git_publish._paper_title(PAPER, source) == "A Study of Things: Part 2"
    assert git_publish._repo_name_from_paper(PAPER, source) == (
        f"{PAPER}-a-study-of-things-part-2")
    assert git_publish._repo_name_from_paper(PAPER) == PAPER


def test_repo_name_falls_back_to_a_search_by_paper_id(monkeypatch) -> None:
    """Without a title the bare id does not name the repository; the one
    repository of the organisation that starts with the id is used."""
    monkeypatch.setattr(git_publish, "_repo_exists", lambda *a, **k: False)
    seen: list[str] = []

    def fake_api(method, path, token, *a, **k):
        seen.append(path)
        return 200, {"items": [{"name": f"{PAPER}-some-title"},
                               {"name": f"unrelated-{PAPER}"}]}

    monkeypatch.setattr(git_publish, "_github_api", fake_api)
    assert git_publish._resolve_repo_name(PAPER, ORG, "TOK") == f"{PAPER}-some-title"
    assert seen == [f"/search/repositories?q=org:{ORG}+{PAPER}"]
    # cached: no second query
    assert git_publish._resolve_repo_name(PAPER, ORG, "TOK") == f"{PAPER}-some-title"
    assert len(seen) == 1


def test_repo_name_ambiguous_search_is_skipped(monkeypatch) -> None:
    monkeypatch.setattr(git_publish, "_repo_exists", lambda *a, **k: False)
    monkeypatch.setattr(git_publish, "_github_api", lambda *a, **k: (200, {"items": [
        {"name": f"{PAPER}-one"}, {"name": f"{PAPER}-two"}]}))
    assert git_publish._resolve_repo_name(PAPER, ORG, "TOK") is None


# ── Published identity ───────────────────────────────────────────────────


def test_publish_as_names_the_branch(eval_config) -> None:
    eval_config(profiles={
        "model-a-via-gateway": {"scaffold": "claude-code", "model": "x",
                                "publish_as": "model-a-xhigh"},
        "plain": {"scaffold": "codex", "model": "y"},
    })
    assert git_publish._canonical_profile("model-a-via-gateway") == "model-a-xhigh"
    assert git_publish._canonical_profile("model-a-via-gateway-grader") == (
        "model-a-xhigh-grader")
    assert git_publish._canonical_profile("plain") == "plain"
    assert git_publish._canonical_profile("unknown") == "unknown"


# ── Sanitization tests ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "secret",
    [
        "ghp_" + "a" * 40,
        "github_pat_" + "x" * 80,
        "hf_" + "h" * 34,
        "sk-ant-" + "y" * 80,
        "sk-or-v1-" + "1" * 80,
        "sk-kimi-" + "z" * 80,
        "sk-cp-" + "Q" * 80,
        "sk-proj-" + "p" * 80,
        # dotted key format: 32 hex + "." + 16 alphanumerics (synthetic value)
        "0123456789abcdef0123456789abcdef.0123456789abcdef",
        "Bearer abc123xyz_long_enough_token_string_4567",
    ],
)
def test_sanitize_redacts_known_secret_shapes(secret: str) -> None:
    text = f"prefix {secret} suffix"
    cleaned = git_publish._sanitize_text(text)
    assert secret not in cleaned, f"{secret!r} survived sanitization"
    assert "REDACTED" in cleaned


@pytest.mark.parametrize("text, secret", [
    ("x_tp-" + "a" * 40, "tp-" + "a" * 40),
    ("key0123456789abcdef0123456789abcdef.0123456789abcdef",
     "0123456789abcdef0123456789abcdef.0123456789abcdef"),
    ("0123456789abcdef0123456789abcdef.0123456789abcdefXYZ",
     "0123456789abcdef0123456789abcdef.0123456789abcdef"),
])
def test_sanitize_redacts_what_the_pre_commit_hook_blocks(text: str, secret: str) -> None:
    """The hook's credential patterns have no word-boundary anchors; the
    in-Python redaction must catch the same strings, or the hook rejects
    the publish commit instead."""
    cleaned = git_publish._sanitize_text(f"prefix {text} suffix")
    assert secret not in cleaned
    assert "REDACTED" in cleaned


def test_sanitize_redacts_basic_auth_url() -> None:
    text = 'remote "origin" https://user:hunter2@github.com/foo.git'
    out = git_publish._sanitize_text(text)
    assert "hunter2" not in out
    assert "REDACTED" in out


def test_sanitize_leaves_ordinary_text_alone() -> None:
    text = ("see https://github.com/example-org/repo.git, sha "
            "0123456789abcdef0123456789abcdef01234567, lr=3e-4, hf_hub_download()")
    assert git_publish.sanitize_text(text) == text


# ── Score-table flatten ───────────────────────────────────────────────────


def test_flatten_scores_picks_first_n_metrics() -> None:
    scores = {
        "experiments": {
            "exp_a": {
                "results": {
                    "sub_a": {"m1": 0.5, "m2": 1.5, "m3": "string", "m4": 2.5},
                }
            }
        }
    }
    rows = git_publish._flatten_scores(scores, max_metrics_per_experiment=2)
    metrics = [m for _, m, _ in rows]
    assert metrics == ["m1", "m2"]
    # string-valued metric "m3" must be dropped
    values = [v for _, _, v in rows]
    assert all(isinstance(v, float) for v in values)


def test_flatten_scores_handles_empty() -> None:
    assert git_publish._flatten_scores({}) == []
    assert git_publish._flatten_scores({"experiments": {}}) == []


# ── README composition ───────────────────────────────────────────────────


def test_readme_single_idea_uses_unified_section(tmp_path: Path, monkeypatch) -> None:
    env = _make_completed_env(tmp_path, PAPER, "model-b-xhigh")
    monkeypatch.setattr(git_publish, "_paper_title", _stub_title("T"))
    meta = json.loads((env / ".evaluation.json").read_text())
    md = git_publish._compose_readme(env, PAPER, "model-b-xhigh",
                                     "improve", "improve-model-b-xhigh", meta)
    assert "## Idea\n" in md
    assert "## Initial idea" not in md
    assert "## Final idea" not in md
    assert "## Reflection" in md


def test_readme_multi_idea_splits_initial_final(tmp_path: Path, monkeypatch) -> None:
    env = _make_completed_env(tmp_path, PAPER, "model-b-xhigh")
    (env / "idea" / "idea_v2.md").write_text("# Idea v2\n\nSecond pass.\n")
    (env / "idea" / "idea_v3.md").write_text("# Idea v3\n\nThird pass.\n")
    monkeypatch.setattr(git_publish, "_paper_title", _stub_title("T"))
    meta = json.loads((env / ".evaluation.json").read_text())
    md = git_publish._compose_readme(env, PAPER, "model-b-xhigh",
                                     "improve", "improve-model-b-xhigh", meta)
    assert "## Initial idea" in md
    assert "## Final idea" in md
    assert "First pass." in md  # from v1
    assert "Third pass." in md  # from v3 (the LAST version)
    assert "Second pass." not in md  # v2 should NOT appear in README body


def test_readme_title_sources(tmp_path: Path) -> None:
    """Title: the environment's manifest, else the repository name, else
    the paper id. No external paper index is needed."""
    env = _make_completed_env(tmp_path, PAPER, PROFILE)
    meta = json.loads((env / ".evaluation.json").read_text())

    def title_line(**kw) -> str:
        return git_publish._compose_readme(
            env, PAPER, PROFILE, "improve", BRANCH, meta, **kw).splitlines()[0]

    assert title_line() == f"# arxiv:{PAPER} — `{PAPER}`"
    assert title_line(repo_name=f"{PAPER}-some-title") == (
        f"# {PAPER}-some-title — `{PAPER}`")
    source = env.parent / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"paper": {"title": "Real Title"}}))
    assert title_line(repo_name=f"{PAPER}-some-title") == f"# Real Title — `{PAPER}`"


def test_manifest_omits_timestamps_and_cost(tmp_path: Path, monkeypatch) -> None:
    env = _make_completed_env(tmp_path, PAPER, PROFILE)
    meta = json.loads((env / ".evaluation.json").read_text())
    monkeypatch.setattr(git_publish, "_paper_title", _stub_title("T"))
    m = git_publish._compose_manifest(PAPER, PROFILE, "improve", BRANCH, meta)
    text = json.dumps(m).lower()
    for forbidden in ("started_at", "ended_at", "cost", "timestamp"):
        assert forbidden not in text, f"manifest leaks {forbidden}: {m}"


# ── Trajectory split ─────────────────────────────────────────────────────


def test_split_jsonl_no_split_when_under_threshold(tmp_path: Path) -> None:
    src = tmp_path / "src.jsonl"
    src.write_text('{"a":1}\n{"b":2}\n')
    out_dir = tmp_path / "out"
    n = git_publish._split_jsonl_for_size(src, out_dir, "turn_0001_raw")
    assert n == 1
    assert (out_dir / "turn_0001_raw.jsonl").exists()
    assert not (out_dir / "turn_0001_raw.index.json").exists()


def test_split_jsonl_splits_when_over_threshold(tmp_path: Path, monkeypatch) -> None:
    """Force a tiny threshold so we can test split behavior cheaply."""
    monkeypatch.setattr(git_publish, "_TRAJECTORY_SPLIT_BYTES", 100)
    src = tmp_path / "src.jsonl"
    # 5 lines, ~50 bytes each → 250 bytes total; 100B threshold → 3 parts.
    src.write_text("\n".join([json.dumps({"i": i, "pad": "x" * 30}) for i in range(5)]) + "\n")
    out_dir = tmp_path / "out"
    n = git_publish._split_jsonl_for_size(src, out_dir, "turn_0001_raw")
    assert n >= 2
    parts = sorted(p.name for p in out_dir.iterdir() if p.name.startswith("turn_0001_raw.part"))
    index = json.loads((out_dir / "turn_0001_raw.index.json").read_text())
    assert index["parts"] == parts
    assert index["original_bytes"] == src.stat().st_size

    # Parts re-assemble to exactly the sanitized original (no data loss).
    rejoined = b""
    for name in parts:
        rejoined += (out_dir / name).read_bytes()
    expected = git_publish._sanitize_text(src.read_text()).encode("utf-8")
    assert rejoined == expected
