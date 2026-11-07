"""Tests for ClaudeCodeWorker lifecycle and command building."""

import json
from pathlib import Path

from infrastructure.worker.claude_code import ClaudeCodeWorker, CONFIG_DEFAULTS


# ── initialization ────────────────────────────────────────────────────────


def test_init_defaults(worker):
    assert worker.get_status() == "idle"
    assert worker.get_last_action() is None
    assert worker.scaffold == "claude-code"
    assert worker.model == "claude-opus-4-6"


def test_init_config_merges_with_defaults():
    w = ClaudeCodeWorker("claude-code", "claude-opus-4-6", {"max_turns": 50, "effort": "high"})
    assert w.config["max_turns"] == 50
    assert w.config["effort"] == "high"
    # Defaults preserved for keys not overridden
    assert w.config["allowed_tools"] == CONFIG_DEFAULTS["allowed_tools"]
    assert w.config["container_sif"] is None


def test_init_config_does_not_mutate_defaults():
    ClaudeCodeWorker("claude-code", "claude-opus-4-6", {"max_turns": 999})
    assert CONFIG_DEFAULTS["max_turns"] == 200


def test_rotate_api_key_does_not_mutate_shared_env_vars():
    caller_env = {"FOO": "bar"}
    w = ClaudeCodeWorker("claude-code", "claude-opus-4-6",
                         {"api_keys": ["k0", "k1"], "env_vars": caller_env})
    assert w._try_rotate_api_key() is True
    assert w.config["env_vars"]["ANTHROPIC_AUTH_TOKEN"] == "k1"
    assert caller_env == {"FOO": "bar"}

    w2 = ClaudeCodeWorker("claude-code", "claude-opus-4-6", {"api_keys": ["k0", "k1"]})
    assert w2._try_rotate_api_key() is True
    assert CONFIG_DEFAULTS["env_vars"] == {}


# ── set_paper_and_workspace / set_resources ───────────────────────────────


def test_set_paper_and_workspace(worker, tmp_workspace):
    paper = {"paper_md_path": "/tmp/p.md"}
    worker.set_paper_and_workspace(paper, str(tmp_workspace))
    assert worker._paper == paper
    assert worker._workspace_path == str(tmp_workspace)


def test_set_resources(worker):
    worker.set_resources(gpu_budget=50.0, max_gpus=8, job_time_limit="06:00:00",
                         shared_datasets_path="/a", shared_models_path="/b")
    assert worker._gpu_budget == 50.0
    assert worker._max_gpus == 8
    assert worker._job_time_limit == "06:00:00"
    assert worker._shared_datasets_path == "/a"
    assert worker._shared_models_path == "/b"


# ── terminate ─────────────────────────────────────────────────────────────


def test_terminate(worker):
    worker.terminate()
    assert worker.get_status() == "done"


# ── _build_command (no container) ─────────────────────────────────────────


def test_build_command_no_container(configured_worker):
    cmd = configured_worker._build_command()
    assert cmd[0] != "apptainer"
    assert "--print" in cmd
    assert "--output-format" in cmd
    assert "stream-json" in cmd
    assert "--model" in cmd
    assert "claude-opus-4-6" in cmd


def test_build_command_loads_no_account_level_mcp_servers(configured_worker):
    # Without --strict-mcp-config the CLI loads the connectors of the account
    # the agent authenticates with.
    assert "--strict-mcp-config" in configured_worker._build_command()
    configured_worker.config["container_sif"] = "/path/to/reproducer.sif"
    assert "--strict-mcp-config" in configured_worker._build_command()


def test_build_command_passes_max_turns(configured_worker):
    cmd = configured_worker._build_command()
    assert cmd[cmd.index("--max-turns") + 1] == "200"
    configured_worker.config["max_turns"] = None
    assert "--max-turns" not in configured_worker._build_command()


def test_compact_token_limit_is_an_env_var_not_a_flag(configured_worker):
    configured_worker.config["compact_token_limit"] = 120000
    cmd = configured_worker._build_command()
    env = configured_worker._build_env()

    # "-c" is --continue in the Claude Code CLI, not a config option.
    assert "-c" not in cmd
    assert not any("model_auto_compact_token_limit" in part for part in cmd)
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "120000"


def test_no_compact_window_by_default(configured_worker, monkeypatch):
    # A value inherited from the daemon's own environment is not forwarded.
    monkeypatch.setenv("CLAUDE_CODE_AUTO_COMPACT_WINDOW", "999")
    monkeypatch.setenv("CLAUDECODE", "1")
    env = configured_worker._build_env()
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in env
    assert "CLAUDECODE" not in env
    assert "-c" not in configured_worker._build_command()


def test_build_env_strips_api_keys(configured_worker, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    env = configured_worker._build_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env


def test_build_command_no_resume_on_first_turn(configured_worker):
    cmd = configured_worker._build_command()
    assert "--resume" not in cmd


def test_build_command_resume_with_session(configured_worker):
    configured_worker._session_id = "sess-42"
    cmd = configured_worker._build_command()
    idx = cmd.index("--resume")
    assert cmd[idx + 1] == "sess-42"


# ── _build_command (with container) ───────────────────────────────────────


def test_build_command_with_container(configured_worker):
    configured_worker.config["container_sif"] = "/path/to/reproducer.sif"
    cmd = configured_worker._build_command()
    # The container runs at low scheduling priority.
    assert cmd[:5] == ["nice", "-n", "19", "apptainer", "exec"]
    assert "--contain" in cmd
    assert "--writable-tmpfs" in cmd
    assert cmd[cmd.index("--no-mount") + 1] == "bind-paths"
    assert cmd[cmd.index("--memory") + 1] == "8G"
    assert "OMP_NUM_THREADS=2" in cmd
    sif_idx = cmd.index("/path/to/reproducer.sif")
    claude_idx = cmd.index("claude")
    assert sif_idx < claude_idx


def test_container_limits_come_from_config(configured_worker):
    configured_worker.config.update({
        "container_sif": "/path/to/reproducer.sif",
        "container_nice": 5,
        "container_memory_limit": "16G",
        "container_cpu_threads": 4,
    })
    cmd = configured_worker._build_command()
    assert cmd[:3] == ["nice", "-n", "5"]
    assert cmd[cmd.index("--memory") + 1] == "16G"
    assert "OMP_NUM_THREADS=4" in cmd
    assert "MKL_NUM_THREADS=4" in cmd


def test_container_blocks_git_push_with_a_shim(configured_worker):
    configured_worker.config["container_sif"] = "/path/to/reproducer.sif"
    cmd = configured_worker._build_command()
    ws = Path(configured_worker._workspace_path)

    shim = ws / ".git-shims" / "git"
    assert shim.is_file()
    assert "git push is disabled" in shim.read_text()
    assert f"{ws}/.git-shims:/opt/git-shim:ro" in cmd
    assert configured_worker._build_env()["APPTAINERENV_PREPEND_PATH"] == "/opt/git-shim"
    # The shim directory is scaffolding, not part of the environment repo.
    assert ".git-shims/" in (ws / ".gitignore").read_text().splitlines()


def test_container_bind_mounts(configured_worker):
    configured_worker.config["container_sif"] = "/path/to/reproducer.sif"
    configured_worker._setup_workspace()
    cmd = configured_worker._build_command()
    cmd_str = " ".join(cmd)
    # Shared directories are read-only for the agent.
    assert "/data/datasets:/home/user/shared/datasets:ro" in cmd_str
    assert "/data/models:/home/user/shared/models:ro" in cmd_str
    assert f"{configured_worker._workspace_path}/.claude-tmp:/tmp/claude-code-tmp" in cmd_str


def test_container_home_is_workspace(configured_worker):
    configured_worker.config["container_sif"] = "/path/to/reproducer.sif"
    cmd = configured_worker._build_command()
    ws = configured_worker._workspace_path
    idx = cmd.index("--home")
    assert cmd[idx + 1] == f"{ws}:/home/user"


def test_container_pwd_is_workspace(configured_worker):
    configured_worker.config["container_sif"] = "/path/to/reproducer.sif"
    cmd = configured_worker._build_command()
    idx = cmd.index("--pwd")
    assert cmd[idx + 1] == "/home/user"


def test_container_uses_bare_claude_binary(configured_worker):
    configured_worker.config["container_sif"] = "/path/to/reproducer.sif"
    cmd = configured_worker._build_command()
    sif_idx = cmd.index("/path/to/reproducer.sif")
    assert cmd[sif_idx + 1] == "claude"


def test_container_hides_protocol_md(configured_worker):
    """protocol.md should be hidden via /dev/null bind inside container."""
    configured_worker.config["container_sif"] = "/path/to/reproducer.sif"
    cmd = configured_worker._build_command()
    cmd_str = " ".join(cmd)
    assert "/dev/null:/home/user/protocol.md:ro" in cmd_str


# ── _build_env ────────────────────────────────────────────────────────────


def test_build_env_includes_effort(worker):
    worker.config["effort"] = "high"
    env = worker._build_env()
    assert env["CLAUDE_CODE_EFFORT_LEVEL"] == "high"


def test_build_env_no_effort_by_default(worker):
    env = worker._build_env()
    assert "CLAUDE_CODE_EFFORT_LEVEL" not in env


def test_build_env_uses_private_tmpdir(configured_worker):
    env = configured_worker._build_env()
    assert env["CLAUDE_CODE_TMPDIR"] == str(Path(configured_worker._workspace_path) / ".claude-tmp")
    assert env["TMPDIR"] == str(Path(configured_worker._workspace_path) / ".claude-tmp")


def test_build_env_uses_container_private_tmpdir(configured_worker):
    configured_worker.config["container_sif"] = "/path/to/reproducer.sif"
    env = configured_worker._build_env()
    assert env["CLAUDE_CODE_TMPDIR"] == "/tmp/claude-code-tmp"
    assert env["TMPDIR"] == "/tmp/claude-code-tmp"


def test_build_env_extra_vars(worker):
    worker.config["env_vars"] = {"MY_VAR": "123"}
    env = worker._build_env()
    assert env["MY_VAR"] == "123"


def test_build_env_git_identity(worker):
    env = worker._build_env()
    assert env["GIT_AUTHOR_NAME"] == "claude-code-claude-opus-4-6 worker"
    assert env["GIT_AUTHOR_EMAIL"] == "claude-code@reproducer"


# ── credentials ──────────────────────────────────────────────────────────


def test_credentials_are_copied_only_in_container_mode(configured_worker, isolated_host_env):
    host_claude = isolated_host_env / ".claude"
    host_claude.mkdir()
    (host_claude / ".credentials.json").write_text('{"fake": "credentials"}')
    ws = Path(configured_worker._workspace_path)

    configured_worker._setup_workspace()
    assert not (ws / ".claude" / ".credentials.json").exists()

    configured_worker.config["container_sif"] = "/path/to/reproducer.sif"
    configured_worker._setup_workspace()
    assert (ws / ".claude" / ".credentials.json").read_text() == '{"fake": "credentials"}'
    # The copy stays out of the environment repository.
    assert ".claude/" in (ws / ".gitignore").read_text().splitlines()


# ── _setup_workspace ─────────────────────────────────────────────────────


def test_setup_workspace_writes_settings(configured_worker):
    configured_worker._setup_workspace()
    ws = Path(configured_worker._workspace_path)
    settings_path = ws / ".claude" / "settings.json"
    assert settings_path.exists()
    settings = json.loads(settings_path.read_text())
    assert "permissions" in settings
    allow = settings["permissions"]["allow"]
    assert "Bash(*)" in allow
    assert any("/data/datasets" in rule for rule in allow)
    assert any("/data/models" in rule for rule in allow)
    deny = settings["permissions"]["deny"]
    assert "Read(.claude-tmp/**)" in deny
    assert "Glob(.claude-tmp/**)" in deny
    assert "Grep(.claude-tmp/**)" in deny


def test_setup_workspace_creates_private_tmpdir(configured_worker):
    configured_worker._setup_workspace()
    ws = Path(configured_worker._workspace_path)
    assert (ws / ".claude-tmp").is_dir()


def test_setup_workspace_symlinks_claude_md(configured_worker):
    configured_worker._setup_workspace()
    ws = Path(configured_worker._workspace_path)
    claude_md = ws / "CLAUDE.md"
    assert claude_md.is_symlink()
    assert claude_md.resolve().name == "protocol.md"


def test_setup_workspace_no_duplicate_symlink(configured_worker):
    """Calling _setup_workspace twice doesn't fail."""
    configured_worker._setup_workspace()
    configured_worker._setup_workspace()
    ws = Path(configured_worker._workspace_path)
    assert (ws / "CLAUDE.md").is_symlink()


# ── get_context / save_trajectory ─────────────────────────────────────────


def test_get_context(configured_worker):
    ctx = configured_worker.get_context()
    assert ctx["status"] == "idle"
    assert ctx["session_id"] is None
    assert ctx["turn_count"] == 0


def test_submit_action_keeps_worker_in_working_state(configured_worker):
    ws = Path(configured_worker._workspace_path)
    (ws / "action.yaml").write_text(
        "action: submit\ngpus: 1\ntest: false\nmilestone: method_runs\n"
    )
    configured_worker._invoke = lambda prompt: "ok"

    configured_worker._invoke_and_read_action("submit now")

    assert configured_worker.get_status() == "working"
    assert configured_worker.get_last_action() is not None
    assert configured_worker.get_last_action().action == "submit"


def test_save_trajectory(configured_worker, tmp_path):
    logs_dir = tmp_path / "logs"
    configured_worker.save_trajectory(str(logs_dir))
    traj_path = logs_dir / "trajectory.json"
    assert traj_path.exists()
    data = json.loads(traj_path.read_text())
    assert data["scaffold"] == "claude-code"
    assert data["model"] == "claude-opus-4-6"
    assert data["turn_count"] == 0
    assert isinstance(data["turns"], list)


# ── reproduce requires setup ──────────────────────────────────────────────


def test_reproduce_raises_without_setup(worker):
    import pytest
    with pytest.raises(RuntimeError, match="set_paper_and_workspace"):
        worker.reproduce()
