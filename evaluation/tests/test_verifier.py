"""Unit tests for the independent verification phase (evaluation/verifier.py).

No cluster needed: covers the pure-Python mechanical scan, the verifier
prompt template and the grader-profile requirement. The scan tests use
the local ``git`` binary on a throwaway repository.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from evaluation import config as _config
from evaluation import verifier
from evaluation.verifier import TERMINAL_STATUSES, mechanical_scan


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True,
        env={**os.environ,
             "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
             "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull},
    )


def _make_env(tmp_path: Path) -> Path:
    env = tmp_path / "run-eval"
    (env / "scoring").mkdir(parents=True)
    _git(env, "init", "-b", "work")
    return env


class TestMechanicalScan:
    def test_clean_run_has_no_findings_beyond_placeholder(self, tmp_path):
        env = _make_env(tmp_path)
        # Organic-looking floats, all distinct.
        (env / "scoring" / "scores_train.json").write_text(json.dumps(
            {"experiments": {"e": {"results": {"m": {
                f"metric{i}": 0.123456789 + i * 0.0371 for i in range(12)
            }}}}}
        ))
        (env / "x.txt").write_text("x")
        _git(env, "add", "-A")
        _git(env, "commit", "-m", "eval: test job abc done (rc=0)")
        scan = mechanical_scan(env)
        assert scan["findings"] == [
            "no mechanical findings — full checks still required"
        ]
        assert scan["test_job_rcs_newest_first"] == ["0"]

    def test_detects_constant_and_hand_typed_values(self, tmp_path):
        env = _make_env(tmp_path)
        (env / "scoring" / "records.json").write_text(json.dumps(
            {"rows": [{"t": 112.5} for _ in range(20)]}
        ))
        scan = mechanical_scan(env)
        text = "\n".join(scan["findings"])
        assert "near-constant values" in text
        assert "hand-typed" in text

    def test_detects_byte_identical_files(self, tmp_path):
        env = _make_env(tmp_path)
        payload = json.dumps({"v": [0.1234567 + i * 0.017 for i in range(30)]})
        (env / "scoring" / "train_records.json").write_text(payload)
        (env / "scoring" / "test_records.json").write_text(payload)
        scan = mechanical_scan(env)
        assert any("byte-identical" in f for f in scan["findings"])

    def test_flags_failed_final_test_job_with_scores_present(self, tmp_path):
        env = _make_env(tmp_path)
        (env / "scoring" / "scores_test.json").write_text(
            json.dumps({"experiments": {}})
        )
        (env / "x.txt").write_text("x")
        _git(env, "add", "-A")
        _git(env, "commit", "-m", "eval: test job abc done (rc=1)")
        scan = mechanical_scan(env)
        assert any("agent-authored" in f for f in scan["findings"])

    def test_successful_rerun_after_failure_not_flagged(self, tmp_path):
        env = _make_env(tmp_path)
        (env / "scoring" / "scores_test.json").write_text(
            json.dumps({"experiments": {}})
        )
        (env / "a.txt").write_text("a")
        _git(env, "add", "-A")
        _git(env, "commit", "-m", "eval: test job abc done (rc=1)")
        (env / "b.txt").write_text("b")
        _git(env, "add", "-A")
        _git(env, "commit", "-m", "eval: test job def done (rc=0)")
        scan = mechanical_scan(env)
        # Newest-first: rc=0 is the final job → no agent-authored flag.
        assert scan["test_job_rcs_newest_first"][0] == "0"
        assert not any("agent-authored" in f for f in scan["findings"])

    def test_paths_are_relative_to_the_workspace(self, tmp_path):
        """The scan is committed with the run and may be published: it must
        not carry the host location of the workspace."""
        env = _make_env(tmp_path)
        payload = json.dumps({"v": [0.1234567 + i * 0.017 for i in range(30)]})
        (env / "scoring" / "scores_train.json").write_text(payload)
        (env / "outputs" / "run1").mkdir(parents=True)
        (env / "outputs" / "run1" / "predictions.json").write_text(payload)
        scan = mechanical_scan(env)
        assert [f["path"] for f in scan["files"]] == [
            "scoring/scores_train.json", "outputs/run1/predictions.json"]
        assert str(tmp_path) not in json.dumps(scan)
        assert any("byte-identical" in f for f in scan["findings"])


class TestGraderProfile:
    def test_no_default_grader(self):
        assert _config.load()["verifier"]["grader_profile"] == ""

    def test_verify_run_needs_a_grader_profile(self, eval_config):
        eval_config()
        with pytest.raises(_config.ConfigError, match="grader_profile"):
            verifier.verify_run("2501.00001", "some-model-xhigh", "improve")

    def test_cli_reports_the_missing_grader(self, eval_config, capsys):
        eval_config()
        assert verifier.main(["2501.00001", "some-model-xhigh", "improve"]) == 1
        assert "grader_profile" in capsys.readouterr().err

    def test_unknown_grader_profile(self, eval_config):
        eval_config(verifier={"grader_profile": "not-configured"})
        with pytest.raises(KeyError, match="not-configured"):
            verifier.verify_run("2501.00001", "some-model-xhigh", "improve")


class TestVerifierPrompt:
    def test_template_renders_without_key_errors(self):
        from evaluation.loop import _render
        text = _render(
            "verifier",
            paper_id="0000.00000",
            run_profile="some-model-xhigh",
            mode="improve",
            frozen_head="deadbeef",
            scan_report='{"findings": []}',
            max_gpus=4,
            gpu_budget=8.0,
        )
        assert "verify" in text.lower()
        assert "0000.00000" in text
        assert "deadbeef" in text
        # Both terminal statuses must be spelled out for the grader.
        for status in TERMINAL_STATUSES:
            assert status in text


class TestGraderTurns:
    def test_grader_turns_exclude_the_agent_trajectory(self, eval_config,
                                                       monkeypatch):
        """loop.run returns the absolute turn number (turn_offset + k);
        the verifier must record only the k turns the grader used."""
        from evaluation import cache, container, loop, nexus

        data_dir = eval_config(
            profiles={"grader": {"scaffold": "codex", "model": "g"}},
            verifier={"grader_profile": "grader", "max_turns": 12},
            nexus={"url": "http://nexus.invalid", "max_gpus": 1},
        )
        env = data_dir / "envs" / "2501.00001" / "m-improve-eval"
        (env / "trajectory").mkdir(parents=True)
        # The agent already used 40 turns.
        (env / "trajectory" / "turn_40.json").write_text("{}")
        _git(env, "init", "-b", "work")
        _git(env, "add", "-A")
        _git(env, "commit", "-m", "agent run")

        class _Nexus:
            def __init__(self, url):
                pass

            def delete_project(self, project_id):
                pass

            def create_project(self, **kwargs):
                pass

        seen = {}

        def fake_run(env_dir, profile, nx, project_id, prompt, *,
                     max_turns, turn_offset, **kwargs):
            seen["turn_offset"] = turn_offset
            return loop.LoopResult(turns=turn_offset + 3,
                                   last_action="done", reason="done")

        monkeypatch.setattr(cache, "apply_to_workspace", lambda *a, **k: [])
        monkeypatch.setattr(container, "ensure_container",
                            lambda *a, **k: Path("/x.sif"))
        monkeypatch.setattr(container, "ensure_overlay",
                            lambda *a, **k: None)
        monkeypatch.setattr(nexus, "Nexus", _Nexus)
        monkeypatch.setattr(loop, "run", fake_run)
        monkeypatch.setattr("time.sleep", lambda s: None)

        ver = verifier.verify_run("2501.00001", "m", "improve", publish=False)

        assert seen["turn_offset"] == 40
        assert ver["grader_turns"] == 3
        assert "after 3 turn(s)" in ver["progress"]
