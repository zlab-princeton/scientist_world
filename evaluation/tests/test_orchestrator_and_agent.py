"""Run queue admission, agent command lines and run pre-flight.

No run is started: the executor, ``run_one`` and the agent container are
stubbed or never reached.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from evaluation import agent, config as _config, orchestrator as orch, runner
from evaluation.models import Profile
from evaluation.orchestrator import QueuedRun
from evaluation.parsing import parse_turn_action


class _FakeExecutor:
    _max_workers = 0

    def __init__(self) -> None:
        self.submitted: list[QueuedRun] = []

    def submit(self, fn, queued: QueuedRun) -> None:
        self.submitted.append(queued)


@pytest.fixture
def queue(monkeypatch) -> _FakeExecutor:
    """An empty queue whose admitted runs are captured, not executed."""
    fake = _FakeExecutor()
    monkeypatch.setattr(orch, "_ensure_executor", lambda: fake)
    monkeypatch.setattr(orch, "_start_dispatcher", lambda: None)
    monkeypatch.setattr(orch, "_executor", None)
    monkeypatch.setattr(orch, "_pending", [])
    monkeypatch.setattr(orch, "_running", {})
    monkeypatch.setattr(orch, "_completed", [])
    monkeypatch.setattr(orch, "_max_concurrent_override", None)
    monkeypatch.setattr(orch, "_profile_caps_override", None)
    return fake


# ── Queue ────────────────────────────────────────────────────────────────


def test_bench_enqueues_the_full_matrix_without_running(queue, eval_config) -> None:
    eval_config()
    orch.bench(["p1", "p2"], ["a", "b"], modes=["innovate", "improve"])
    assert [(q.paper_id, q.profile_name, q.mode) for q in orch._pending] == [
        ("p1", "a", "innovate"), ("p1", "a", "improve"),
        ("p1", "b", "innovate"), ("p1", "b", "improve"),
        ("p2", "a", "innovate"), ("p2", "a", "improve"),
        ("p2", "b", "innovate"), ("p2", "b", "improve"),
    ]
    assert queue.submitted == []
    orch.bench(["p3"], ["a"])
    assert orch._pending[-1].mode == "innovate"


def test_admission_respects_the_global_and_per_profile_caps(queue, eval_config) -> None:
    eval_config(max_concurrent=3, profile_concurrency={"a": 1})
    orch.bench(["p1", "p2"], ["a"])
    orch.bench(["p3", "p4", "p5"], ["b"])

    orch._admit_pass()
    # one run of profile a (its cap), then b fills the remaining slots
    assert [q.paper_id for q in queue.submitted] == ["p1", "p3", "p4"]
    assert [q.paper_id for q in orch._pending] == ["p2", "p5"]
    orch._admit_pass()
    assert len(queue.submitted) == 3  # all three slots taken

    # A slot of profile b frees: p2 is still held back by a's cap.
    del orch._running[orch._runkey(queue.submitted[1])]
    orch._admit_pass()
    assert [q.paper_id for q in queue.submitted] == ["p1", "p3", "p4", "p5"]
    assert [q.paper_id for q in orch._pending] == ["p2"]


def test_live_caps_override_the_config(queue, eval_config) -> None:
    eval_config(max_concurrent=1)
    orch.bench(["p1", "p2", "p3"], ["a"])
    orch._admit_pass()
    assert len(queue.submitted) == 1

    caps = orch.set_caps(max_concurrent=3, profile_caps={"a": 2})
    assert caps["max_concurrent"] == 3 and caps["override_active"] is True
    orch._admit_pass()
    assert len(queue.submitted) == 2  # capped by the per-profile limit now
    # an empty mapping clears the per-profile caps
    assert orch.set_caps(profile_caps={})["profile_concurrency"] == {}
    orch._admit_pass()
    assert len(queue.submitted) == 3


def test_remove_pending_filters_are_anded(queue, eval_config) -> None:
    eval_config()
    orch.bench(["p1", "p2"], ["a", "b"], modes=["improve"])
    assert orch.remove_pending(models=["a"], papers=["p2"]) == 1
    assert [(q.paper_id, q.profile_name) for q in orch._pending] == [
        ("p1", "a"), ("p1", "b"), ("p2", "b")]
    assert orch.remove_pending(modes=["innovate"]) == 0
    assert orch.remove_pending() == 3
    assert orch._pending == []


def test_a_crashing_run_is_recorded_as_failed(queue, monkeypatch) -> None:
    def crash(paper, profile, mode="innovate"):
        raise RuntimeError("scaffold exploded")

    monkeypatch.setattr(orch, "run_one", crash)
    q = QueuedRun("p1", "a", "improve")
    orch._running[orch._runkey(q)] = q
    result = orch._run_and_record(q)
    assert result.status == "failed" and "scaffold exploded" in result.error
    assert orch._running == {}
    assert orch.snapshot()["completed"][-1]["paper_id"] == "p1"


# ── Agent command lines ──────────────────────────────────────────────────


def _profile(scaffold: str, **kw) -> Profile:
    return Profile(name="p", scaffold=scaffold, model="model-x", **kw)


def test_claude_code_command_line() -> None:
    argv = agent._build_agent_argv(_profile("claude-code", effort="high"), resume=False)
    assert argv[0] == "claude" and "--print" in argv
    assert argv[argv.index("--model") + 1] == "model-x"
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    # never load the operator account's connectors into an evaluated agent
    assert "--strict-mcp-config" in argv
    assert "--continue" not in argv
    assert "--continue" in agent._build_agent_argv(_profile("claude-code"), resume=True)


def test_claude_tmux_command_line_is_interactive() -> None:
    argv = agent._build_agent_argv(_profile("claude-tmux"), resume=True)
    assert argv[:2] == ["claude", "--continue"]
    assert "--print" not in argv and "--output-format" not in argv
    assert "--strict-mcp-config" in argv


def test_codex_command_line() -> None:
    argv = agent._build_agent_argv(
        _profile("codex", effort="medium", compact_token_limit=1000), resume=False)
    assert argv[:2] == ["codex", "exec"] and argv[-1] == "-"
    assert "reasoning_effort=medium" in argv
    assert "model_auto_compact_token_limit=1000" in argv
    resumed = agent._build_agent_argv(_profile("codex"), resume=True)
    assert resumed[:4] == ["codex", "exec", "resume", "--last"]


def test_unknown_scaffold_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown scaffold"):
        agent._build_agent_argv(_profile("something-else"), resume=False)


def test_agent_environment(eval_config, monkeypatch) -> None:
    """Profile settings reach the container; the operator's own API keys
    and CLI session variables do not."""
    eval_config(nexus={"url": "http://nexus.example:39000"},
                claude_code_max_retries=7, api_timeout_ms=1234)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "operator-key")
    monkeypatch.setenv("OPENAI_API_KEY", "operator-key")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    profile = _profile("claude-code", effort="high", compact_token_limit=5000,
                       env_vars={"ANTHROPIC_BASE_URL": "https://gateway.example.com"})
    env = agent._build_env(profile, "project-1", cpu_threads=3)

    assert "ANTHROPIC_API_KEY" not in env and "OPENAI_API_KEY" not in env
    assert "CLAUDE_CODE_ENTRYPOINT" not in env
    assert env["APPTAINERENV_ANTHROPIC_BASE_URL"] == "https://gateway.example.com"
    assert env["APPTAINERENV_CLAUDE_CODE_EFFORT_LEVEL"] == "high"
    assert env["APPTAINERENV_CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "5000"
    assert env["APPTAINERENV_OMP_NUM_THREADS"] == "3"
    assert env["APPTAINERENV_NEXUS_URL"] == "http://nexus.example:39000"
    assert env["APPTAINERENV_NEXUS_PROJECT_ID"] == "project-1"
    assert env["APPTAINERENV_CLAUDE_CODE_MAX_RETRIES"] == "7"
    assert env["APPTAINERENV_API_TIMEOUT_MS"] == "1234"


def test_redact_secrets_in_agent_output() -> None:
    text = ("token ghp_" + "a" * 36 + " and hf_" + "b" * 34
            + " at https://user:pass@example.com/x")
    out = agent.redact_secrets(text)
    assert "ghp_" + "a" * 36 not in out and "hf_" + "b" * 34 not in out
    assert "user:pass@" not in out
    assert "https://[REDACTED]@example.com/x" in out


# ── Pre-flight ───────────────────────────────────────────────────────────


def test_invoke_without_an_agent_image_is_a_config_error(eval_config, tmp_path: Path) -> None:
    eval_config()
    with pytest.raises(_config.ConfigError, match="agent_sif"):
        agent.invoke(tmp_path, _profile("codex"), "prompt", turn=1, resume=False,
                     nexus_project_id="project-1")


def test_run_one_checks_the_configuration_first(eval_config) -> None:
    data_dir = eval_config()
    with pytest.raises(_config.ConfigError) as exc:
        runner.run_one("2501.00001", "my-profile")
    assert "agent_sif" in str(exc.value) and "nexus.resources.gpu" in str(exc.value)
    assert not (data_dir / "envs").exists(), "nothing may be created before the check"


def test_run_one_needs_a_configured_profile(eval_config) -> None:
    eval_config(agent_sif="/x/agent.sif",
                nexus={"resources": {"gpu": ["g"], "gpu_test": ["g"]}})
    with pytest.raises(KeyError, match="Unknown profile"):
        runner.run_one("2501.00001", "my-profile")


# ── action.yaml ──────────────────────────────────────────────────────────


def test_parse_turn_action(tmp_path: Path) -> None:
    assert parse_turn_action(tmp_path) is None
    action = tmp_path / "action.yaml"

    action.write_text("action: submit\ngpus: 2\ntest: true\nnotes: smoke\n")
    parsed = parse_turn_action(tmp_path)
    assert (parsed.action, parsed.gpus, parsed.test, parsed.notes) == (
        "submit", 2, True, "smoke")

    action.write_text("action: done\n")
    assert parse_turn_action(tmp_path).action == "done"
    for decision in ("continue", "cancel"):
        action.write_text(f"action: {decision}\n")
        assert parse_turn_action(tmp_path).action == decision

    for bad in ("action: launch\n", "- a list\n", "action: [unclosed\n", ""):
        action.write_text(bad)
        assert parse_turn_action(tmp_path) is None


@pytest.mark.parametrize("decision", ["continue", "cancel"])
def test_checkpoint_only_action_outside_a_checkpoint_submits_nothing(
        tmp_path: Path, monkeypatch, decision) -> None:
    """``continue`` / ``cancel`` only answer a checkpoint; as a turn's
    action they must not dispatch a GPU job."""
    from evaluation import loop

    submitted: list = []

    class _Nexus:
        def submit_job(self, *args, **kwargs):
            submitted.append((args, kwargs))
            return "run-1", "hive-1"

        def get_job(self, run_id):
            return {}

        def wait_for_job(self, project_id, run_id, timeout=300):
            return {"event": "completed"}

    def invoke(env_dir, profile, prompt, **kwargs):
        (env_dir / "action.yaml").write_text(f"action: {decision}\n")
        return 0, "", ""

    monkeypatch.setattr(loop.agent, "invoke", invoke)
    monkeypatch.setattr(loop._config, "load", lambda: {})
    monkeypatch.setattr(loop._config, "nexus_pick", lambda **kwargs: "gpu")
    monkeypatch.setattr(loop._container, "rebuild_if_changed",
                        lambda env_dir, **kwargs: (None, None, False))
    monkeypatch.setattr(loop.git, "commit", lambda *args, **kwargs: None)

    result = loop.run(
        tmp_path, Profile(name="p", scaffold="codex", model="m"), _Nexus(),
        "proj", "start", max_turns=2, gpu_budget_hours=10.0,
    )
    assert submitted == []
    assert result.reason == "max_turns"


def test_every_job_requests_at_least_one_gpu(tmp_path: Path) -> None:
    """Current behaviour (README, Limitations): ``gpus: 0`` is raised to 1."""
    (tmp_path / "action.yaml").write_text("action: submit\ngpus: 0\n")
    assert parse_turn_action(tmp_path).gpus == 1
    (tmp_path / "action.yaml").write_text("action: submit\ngpus: many\n")
    assert parse_turn_action(tmp_path).gpus == 1
