"""Command-line entry point, config seeding and token-usage extraction."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evaluation import cli, config as _config, seed_config, usage


# ── CLI ──────────────────────────────────────────────────────────────────


def test_cli_requires_a_command(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
    assert "sw-eval" in capsys.readouterr().err


def test_cli_reports_configuration_errors_without_a_traceback(capsys) -> None:
    assert cli.main(["renormalize"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("configuration error: data_dir is not set")


def test_cli_publish_refuses_when_publishing_is_disabled(eval_config, capsys) -> None:
    eval_config()
    with pytest.raises(SystemExit) as exc:
        cli.main(["publish", "--dry-run"])
    assert exc.value.code == 2
    assert "publishing is disabled" in capsys.readouterr().err


def test_cli_publish_dry_run_lists_completed_runs(eval_config, capsys, monkeypatch) -> None:
    from evaluation import git_publish

    data_dir = eval_config(publish={"enabled": True, "github_org": "example-org"})
    env = data_dir / "envs" / "2501.00001" / "my-profile-improve-eval"
    (env / "method").mkdir(parents=True)
    (env / "method" / "solution.py").write_text("# stub\n")
    (env / ".evaluation.json").write_text(json.dumps({
        "paper_id": "2501.00001", "model": "my-profile", "mode": "improve",
        "status": "completed"}))
    monkeypatch.setattr(git_publish, "_get_github_token", lambda: "TOK")

    assert cli.main(["publish", "--dry-run", "--mode", "improve"]) == 0
    out = capsys.readouterr().out
    assert "skipped  2501.00001" in out and "dry-run" in out
    assert "0 pushed, 1 skipped, 0 failed" in out
    # the mode filter applies
    assert cli.main(["publish", "--dry-run", "--mode", "innovate"]) == 0
    assert "0 pushed, 0 skipped, 0 failed" in capsys.readouterr().out


def test_cli_serve_passes_host_and_port(monkeypatch) -> None:
    from evaluation import dashboard

    seen: list[tuple] = []
    monkeypatch.setattr(dashboard, "serve",
                        lambda port=None, host=None: seen.append((host, port)))
    assert cli.main(["serve"]) == 0
    assert cli.main(["serve", "--host", "0.0.0.0", "--port", "9"]) == 0
    assert seen == [(None, None), ("0.0.0.0", 9)]


def test_cli_run_exit_code_follows_the_run_status(monkeypatch) -> None:
    from evaluation import runner

    def fake_run_one(paper, profile, mode="innovate"):
        status = "completed" if paper == "ok" else "failed"
        return runner.RunResult(
            run_id="r", paper_id=paper, model=profile, mode=mode, status=status,
            attempt=1, env_dir="", train_score=None, test_score=None, error=None)

    monkeypatch.setattr(runner, "run_one", fake_run_one)
    assert cli.main(["run", "ok", "p", "--mode", "improve"]) == 0
    assert cli.main(["run", "bad", "p"]) == 1


# ── seed ─────────────────────────────────────────────────────────────────


SWARM_CONFIG = {
    "github_org": "swarm-org",
    "profiles": {
        "swarm-profile": {"scaffold": "codex", "model": "m1"},
        "shared": {"scaffold": "codex", "model": "from-swarm"},
    },
    "nexus": {
        "url": "http://localhost:39000",
        "gpu_budget_hours": 50.0,
        "resources": {"gpu": ["gpu"], "gpu_test": ["gpu-test"]},
    },
}


@pytest.fixture
def swarm_config(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "swarm-config.yaml"
    path.write_text(yaml.safe_dump(SWARM_CONFIG))
    # No sciworld settings unless a test provides them.
    monkeypatch.setenv("SCIWORLD_HOME", str(tmp_path / "no-sciworld"))
    return path


def test_seed_writes_defaults_plus_swarm_profiles_and_nexus(swarm_config: Path) -> None:
    written = seed_config.seed(swarm_config=swarm_config)
    assert written == _config.CONFIG_PATH
    on_disk = yaml.safe_load(written.read_text())
    assert set(on_disk) == set(_config.DEFAULT_CONFIG)
    assert set(on_disk["profiles"]) == {"swarm-profile", "shared"}

    cfg = _config.load()
    assert cfg["nexus"]["gpu_budget_hours"] == 50.0
    assert cfg["nexus"]["max_gpus"] == 2  # default kept: Swarm's block omits it
    assert cfg["nexus"]["resources"] == {
        "gpu": ["gpu"], "gpu_test": ["gpu-test"], "cpu": [], "cpu_test": []}
    # Not seeded: these stay at their safe defaults.
    assert cfg["publish"] == {"enabled": False, "github_org": ""}
    assert cfg["data_dir"] == "" and cfg["agent_sif"] == ""


def test_seed_keeps_what_the_existing_config_states(swarm_config: Path, eval_config) -> None:
    data_dir = eval_config(
        max_turns=7,
        profiles={"shared": {"scaffold": "claude-code", "model": "mine"},
                  "local-only": {"scaffold": "codex", "model": "x"}},
    )
    seed_config.seed(swarm_config=swarm_config)
    cfg = _config.load()
    assert cfg["data_dir"] == str(data_dir) and cfg["max_turns"] == 7
    assert set(cfg["profiles"]) == {"swarm-profile", "shared", "local-only"}
    assert cfg["profiles"]["shared"]["model"] == "mine"
    # The existing file had no nexus block, so Swarm's is taken.
    assert cfg["nexus"]["gpu_budget_hours"] == 50.0


def test_seed_does_not_replace_an_existing_nexus_block(swarm_config: Path, eval_config) -> None:
    eval_config(nexus={"gpu_budget_hours": 5.0})
    seed_config.seed(swarm_config=swarm_config)
    cfg = _config.load()
    assert cfg["nexus"]["gpu_budget_hours"] == 5.0
    assert cfg["nexus"]["resources"]["gpu"] == []


def test_seed_overwrite_replaces_existing_keys(swarm_config: Path, eval_config) -> None:
    eval_config(max_turns=7, nexus={"gpu_budget_hours": 5.0},
                profiles={"shared": {"scaffold": "claude-code", "model": "mine"}})
    seed_config.seed(overwrite=True, swarm_config=swarm_config)
    cfg = _config.load()
    assert cfg["max_turns"] == _config.DEFAULT_CONFIG["max_turns"]
    assert cfg["profiles"]["shared"]["model"] == "from-swarm"
    assert cfg["nexus"]["gpu_budget_hours"] == 50.0


def test_seed_takes_paths_from_the_sciworld_settings(
    swarm_config: Path, tmp_path: Path, monkeypatch,
) -> None:
    sw_home = tmp_path / "sciworld-home"
    sw_data = tmp_path / "sciworld-data"
    (sw_data / "containers").mkdir(parents=True)
    (sw_data / "containers" / "agent.sif").write_bytes(b"sif")
    sw_home.mkdir()
    (sw_home / "config.json").write_text(json.dumps({"data_dir": str(sw_data)}))
    monkeypatch.setenv("SCIWORLD_HOME", str(sw_home))

    seed_config.seed(swarm_config=swarm_config)
    cfg = _config.load()
    assert cfg["data_dir"] == str(sw_data / "evaluation")
    assert cfg["agent_sif"] == str(sw_data / "containers" / "agent.sif")


def test_seed_without_a_swarm_config_is_a_config_error(tmp_path: Path, capsys) -> None:
    with pytest.raises(_config.ConfigError, match="--swarm-config"):
        seed_config.seed(swarm_config=tmp_path / "absent.yaml")
    assert cli.main(["seed", "--swarm-config", str(tmp_path / "absent.yaml")]) == 2
    assert "configuration error" in capsys.readouterr().err
    assert not _config.CONFIG_PATH.exists()


def test_cli_seed(swarm_config: Path, capsys) -> None:
    assert cli.main(["seed", "--swarm-config", str(swarm_config)]) == 0
    assert str(_config.CONFIG_PATH) in capsys.readouterr().out
    assert "swarm-profile" in _config.load()["profiles"]


# ── Token usage ──────────────────────────────────────────────────────────


def _turn(env_dir: Path, n: int, *events: dict) -> None:
    traj = env_dir / "trajectory"
    traj.mkdir(exist_ok=True)
    (traj / f"turn_{n:04d}_raw.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events) + "not json\n\n")


def test_usage_claude_code(tmp_path: Path) -> None:
    _turn(tmp_path, 1,
          {"type": "assistant", "message": {}},
          {"type": "result", "total_cost_usd": 0.5, "usage": {
              "input_tokens": 10, "cache_read_input_tokens": 100,
              "cache_creation_input_tokens": 5, "output_tokens": 20}})
    _turn(tmp_path, 2,
          {"type": "result", "total_cost_usd": 0.25, "usage": {
              "input_tokens": 1, "output_tokens": 2}})
    out = usage.extract_usage(tmp_path, "claude-code")
    assert out == {
        "input_tokens_new": 11, "input_tokens_cached_read": 100,
        "input_tokens_cached_create": 5, "output_tokens": 22,
        "reasoning_tokens": 0, "cost_usd": 0.75,
        "cost_source": "claude_total_cost_usd", "turn_count": 2,
        "scaffold": "claude-code",
    }


def test_usage_codex_subtracts_cached_input(tmp_path: Path) -> None:
    _turn(tmp_path, 1, {"type": "turn.completed", "usage": {
        "input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 7,
        "reasoning_output_tokens": 3}})
    out = usage.extract_usage(tmp_path, "codex")
    assert (out["input_tokens_new"], out["input_tokens_cached_read"]) == (40, 60)
    assert (out["output_tokens"], out["reasoning_tokens"]) == (7, 3)
    assert out["cost_usd"] == 0.0 and out["cost_source"] == "none"
    assert out["turn_count"] == 1


def test_usage_gemini(tmp_path: Path) -> None:
    _turn(tmp_path, 1, {"type": "result", "status": "success", "stats": {
        "input": 9, "cached": 4, "output_tokens": 2}})
    out = usage.extract_usage(tmp_path, "gemini")
    assert (out["input_tokens_new"], out["input_tokens_cached_read"],
            out["output_tokens"], out["turn_count"]) == (9, 4, 2, 1)


def test_usage_skips_non_object_json_lines(tmp_path: Path) -> None:
    traj = tmp_path / "trajectory"
    traj.mkdir()
    (traj / "turn_0001_raw.jsonl").write_text(
        '"progress"\n42\n[1, 2]\nnull\n'
        + json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 5, "output_tokens": 3}}) + "\n")
    out = usage.extract_usage(tmp_path, "codex")
    assert (out["input_tokens_new"], out["output_tokens"], out["turn_count"]) == (5, 3, 1)


@pytest.mark.parametrize("scaffold", ["kimi", "claude-tmux", "something-else"])
def test_usage_is_zero_for_scaffolds_without_a_handler(tmp_path: Path, scaffold) -> None:
    _turn(tmp_path, 1, {"type": "result", "usage": {"input_tokens": 10}})
    out = usage.extract_usage(tmp_path, scaffold)
    assert out["scaffold"] == scaffold
    assert out["turn_count"] == 0 and out["input_tokens_new"] == 0


def test_usage_without_a_trajectory(tmp_path: Path) -> None:
    out = usage.extract_usage(tmp_path, "codex")
    assert out["turn_count"] == 0 and out["scaffold"] == "codex"
