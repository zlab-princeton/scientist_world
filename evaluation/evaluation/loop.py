"""Within-run agent loop.

One turn of the cycle:

    agent.invoke(...)                # one CLI call, writes turn_NNNN_raw.jsonl
    action = parse_turn_action(ws)   # reads action.yaml
    if action.action == "done": break
    if action.action == "submit":
        rebuild container / overlay if the agent edited their sources
        run_id = nexus.submit_job(...)
        event = nexus.wait_for_job(...)   # checkpoint events go back to
                                          # the agent: continue or cancel
        save jobs/<run_id>/{stdout,stderr,meta}
        prompt = on_job_completion template

All agent-facing text is rendered from ``prompts/*.md`` by :func:`_render`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from evaluation import agent, config as _config, container as _container, git
from evaluation.agent import redact_secrets
from evaluation.models import Profile
from evaluation.nexus import Nexus
from evaluation.parsing import parse_turn_action

log = logging.getLogger(__name__)

CONTAINER_CMD = "bash /home/user/scripts/run.sh"  # agent's entrypoint in compute jobs


class TransientError(RuntimeError):
    """Raised on malformed or missing action.yaml / agent CLI crash.

    The runner may retry the whole run on these (up to ``max_retries``).
    """


class FatalAgentError(RuntimeError):
    """Raised when an agent CLI failure is provably non-retryable.

    The only case today is a provider safety filter rejecting the
    conversation context. The agent has already loaded the paper's data
    into context and the filter tripped on its contents, so every retry
    would load the same data and trip the same filter. The runner
    catches this and skips the remaining attempts.
    """


@dataclass
class LoopResult:
    turns: int
    last_action: str | None
    reason: str  # "done" | "max_turns" | "error"


import re as _re

_PROMPTS_DIR = Path(__file__).parent / "prompts"
# Only match plain ``{name}`` (no format spec, no whitespace). Kwargs
# with format specs like ``{gpu_hours_used:.1f}`` go through the second
# pass below via str.format.
_INCLUDE_RE = _re.compile(r"\{([a-z_][a-z_0-9]*)\}")


def _render(name: str, **fmt) -> str:
    """Load ``prompts/<name>.md`` and substitute placeholders.

    Two passes:

    1. ``{name}`` where ``name`` matches another file in ``prompts/``
       (e.g. ``{protocol}`` → ``prompts/protocol.md``) is inlined.
       This lets prompts share snippets without duplication.
    2. The result goes through ``str.format(**fmt)`` so kwargs resolve,
       including Python format specs (e.g. ``{gpu_hours_used:.1f}``).

    All agent-facing prompt text lives in ``evaluation/prompts/*.md`` —
    never inline prompt strings into Python.
    """
    tmpl = (_PROMPTS_DIR / f"{name}.md").read_text()

    def resolve_include(m):
        key = m.group(1)
        # Only substitute include files here; leave kwargs for pass 2.
        if key in fmt:
            return m.group(0)
        include_path = _PROMPTS_DIR / f"{key}.md"
        if include_path.exists():
            return include_path.read_text().strip()
        return m.group(0)

    tmpl = _INCLUDE_RE.sub(resolve_include, tmpl)
    return tmpl.format(**fmt) if fmt else tmpl


def _save_job_artifacts(
    env_dir: Path, run_id: str, nexus: Nexus,
) -> tuple[Path, Path, int]:
    """Capture logs + metadata for a completed Nexus job.

    Returns ``(stdout_path, stderr_path, returncode)``.
    """
    jobs_dir = env_dir / "jobs" / run_id
    jobs_dir.mkdir(parents=True, exist_ok=True)

    info = nexus.get_job(run_id)
    (jobs_dir / "meta.json").write_text(json.dumps(info, indent=2, default=str))

    # Nexus exposes log paths in the job record; copy them to jobs/<run_id>/.
    # Fall back to placeholder files so downstream code has deterministic paths.
    stdout_path = jobs_dir / "stdout.log"
    stderr_path = jobs_dir / "stderr.log"

    for key, dst in (("stdout_path", stdout_path), ("stderr_path", stderr_path)):
        src = info.get(key)
        if src and Path(src).exists():
            try:
                # Read as text + redact secrets before writing to per-run
                # workspace. Job logs are committed to the run's git repo
                # so any token in setup.sh / env dump / git remote output
                # would otherwise live in history forever.
                raw = Path(src).read_text(errors="replace")
                dst.write_text(redact_secrets(raw))
                continue
            except OSError as exc:
                log.warning("could not copy %s: %s", src, exc)
        if not dst.exists():
            dst.write_text("")

    rc = info.get("returncode")
    if rc is None:
        rc = info.get("rc", -1)
    try:
        rc = int(rc)
    except (TypeError, ValueError):
        rc = -1
    return stdout_path, stderr_path, rc


def run(
    env_dir: Path,
    profile: Profile,
    nexus: Nexus,
    nexus_project_id: str,
    initial_prompt: str,
    *,
    max_turns: int,
    gpu_budget_hours: float,
    extra_binds: list[tuple[Path, str]] | None = None,
    turn_offset: int = 0,
    resume_session: bool = False,
    gpu_hours_start: float = 0.0,
) -> LoopResult:
    """Drive the agent until it says done or we hit a ceiling.

    ``turn_offset`` shifts turn numbering (and therefore trajectory
    file names) past an earlier session's turns — used by the
    verification phase so grader artifacts never clobber the agent's.
    ``resume_session`` makes the first invoke continue the agent's
    existing session instead of starting a new one — used when
    resuming an interrupted run in place. ``gpu_hours_start`` seeds the
    GPU accounting with what the interrupted portion already spent, so
    a resumed run cannot exceed its original budget.
    """
    prompt = initial_prompt
    resume = resume_session
    turn = turn_offset
    gpu_hours_used = gpu_hours_start

    # Lazy import to avoid a circular module-load (runner imports loop).
    from evaluation.runner import update_state

    while turn < turn_offset + max_turns:
        turn += 1

        update_state(
            env_dir, "working",
            current_turn=turn,
            gpu_hours_used=gpu_hours_used,
            gpu_budget=gpu_budget_hours,
        )
        rc, stdout, stderr = agent.invoke(
            env_dir,
            profile,
            prompt,
            turn=turn,
            resume=resume,
            nexus_project_id=nexus_project_id,
            extra_binds=extra_binds,
        )
        resume = True

        if rc != 0:
            log.warning("Turn %d: agent rc=%d stderr=%s", turn, rc, stderr[:300])
            if rc == -9:
                # SIGKILL: the memory cgroup (OOM) or the inactivity
                # guard in agent.py — the agent log's last lines say
                # which.
                raise TransientError(
                    f"agent SIGKILL'd (rc=-9) — memory cgroup OOM or "
                    f"inactivity kill (check the agent log tail)"
                )
            # Provider safety-filter rejection is non-retryable: the
            # offending bytes live in the conversation context (data
            # files the agent loaded), and every retry re-loads the
            # same data and re-trips the same filter. Detect the
            # standard rejection strings and raise FatalAgentError so
            # the runner skips remaining attempts.
            combined = (stdout or "") + "\n" + (stderr or "")
            fatal_markers = (
                "Invalid prompt: we've limited access",
                "limited access to this content for safety reasons",
                "content_policy_violation",
            )
            matched = next((m for m in fatal_markers if m in combined), None)
            if matched is not None:
                raise FatalAgentError(
                    f"provider safety filter rejected conversation context "
                    f"(matched marker: {matched!r}); retries would re-trip"
                )
            raise TransientError(f"agent exited rc={rc}")

        action = parse_turn_action(env_dir)
        if action is None:
            # Don't immediately bail — give the agent one more turn to
            # correct malformed YAML with a nudge prompt.
            log.warning("Turn %d: action.yaml missing/invalid", turn)
            prompt = _render("on_invalid_action")
            continue

        if action.action == "done":
            log.info("Turn %d: agent signalled done", turn)
            return LoopResult(turns=turn, last_action="done", reason="done")

        # action == "submit"
        if gpu_hours_used >= gpu_budget_hours:
            prompt = _render(
                "on_budget_exhausted",
                gpu_hours_used=gpu_hours_used,
                gpu_budget=gpu_budget_hours,
            )
            continue

        # Agent-driven environment iteration: if the agent edited
        # container.def or setup.sh during this turn, rebuild the
        # affected layer(s) before dispatching the job.
        try:
            cfg = _config.load()
            update_state(
                env_dir, "working",
                current_turn=turn,
                gpu_hours_used=gpu_hours_used,
                gpu_budget=gpu_budget_hours,
                building_env=True,
            )
            new_sif, new_overlay, did_rebuild = _container.rebuild_if_changed(
                env_dir,
                container_cache_dir=str(cfg.get("container_cache_dir") or ""),
                local_tmp_dir=str(cfg.get("local_tmp_dir") or "/tmp"),
                uv_cache_dir=str(cfg.get("uv_cache_dir") or ""),
            )
            if did_rebuild:
                log.info("Turn %d: environment rebuilt; updating Nexus project", turn)
                nexus.update_project(
                    nexus_project_id,
                    sif_path=str(new_sif),
                    overlay_path=str(new_overlay) if new_overlay else "",
                )
                git.commit(env_dir, f"eval: env rebuilt at turn {turn}")
        except Exception as exc:
            log.warning("Turn %d: environment rebuild failed: %s", turn, exc)
            prompt = _render(
                "on_env_rebuild_failed",
                error=str(exc),
            )
            continue

        log.info("Turn %d: submitting job (gpus=%d, test=%s)",
                 turn, action.gpus, action.test)
        # Per-job resource pick: the agent sets `test: true` in
        # action.yaml for short smoke tests, which selects the
        # ``gpu_test`` resource group; the default `test: false` selects
        # ``gpu``.
        run_id, hive_job_id = nexus.submit_job(
            nexus_project_id, CONTAINER_CMD, num_gpus=action.gpus,
            resource=_config.nexus_pick(gpus=action.gpus, test=action.test),
        )
        # Nexus often returns an empty hive_job_id on submit (Hive
        # hasn't assigned one yet). Poll get_job once more to fetch it.
        if not hive_job_id:
            try:
                info = nexus.get_job(run_id) or {}
                hive_job_id = info.get("hive_job_id") or ""
            except Exception:
                pass
        update_state(
            env_dir, "waiting",
            current_job=run_id,
            current_hive_id=hive_job_id or "",
            current_turn=turn,
            gpu_hours_used=gpu_hours_used,
            gpu_budget=gpu_budget_hours,
            # Remember the last submit's GPU count. The (disabled) in-run
            # test phase uses it to dispatch evaluate_test.sh with the
            # GPU count the agent's run code expects.
            last_submit_gpus=action.gpus,
        )

        # Poll until terminal state. Periodic checkpoint events are
        # routed through the agent: render on_checkpoint.md, parse its
        # continue/cancel decision, tell Nexus.
        terminal = None
        t0 = time.monotonic()
        checkpoint_count = 0
        while True:
            event = nexus.wait_for_job(nexus_project_id, run_id, timeout=300)
            # Once Hive assigns a job id, surface it in state so the
            # dashboard can show it (the submit response is often too
            # early to include it).
            if not hive_job_id:
                try:
                    info = nexus.get_job(run_id) or {}
                    hj = info.get("hive_job_id") or ""
                    if hj:
                        hive_job_id = hj
                        update_state(
                            env_dir, "waiting",
                            current_job=run_id,
                            current_hive_id=hive_job_id,
                            current_turn=turn,
                            gpu_hours_used=gpu_hours_used,
                            gpu_budget=gpu_budget_hours,
                        )
                except Exception:
                    pass
            ev = event.get("event") or event.get("status")
            if ev in ("completed", "cancelled", "failed", "error", "done"):
                terminal = event
                break
            if ev == "awaiting_response":
                checkpoint_count += 1
                log.info(
                    "Turn %d: job %s checkpoint #%d — agent decision required",
                    turn, run_id, checkpoint_count,
                )
                # Save fresh stdout/stderr so the agent has current logs.
                stdout_path, stderr_path, _rc = _save_job_artifacts(
                    env_dir, run_id, nexus,
                )
                elapsed_min = int((time.monotonic() - t0) / 60)
                update_state(
                    env_dir, "working",
                    current_turn=turn,
                    current_job=run_id,
                    current_hive_id=event.get("hive_job_id") or "",
                    gpu_hours_used=gpu_hours_used,
                    gpu_budget=gpu_budget_hours,
                    checkpoint_count=checkpoint_count,
                )
                ckpt_prompt = _render(
                    "on_checkpoint",
                    run_id=run_id,
                    elapsed_min=elapsed_min,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                rc_ag, _so, _se = agent.invoke(
                    env_dir, profile, ckpt_prompt,
                    turn=turn,
                    resume=True,
                    nexus_project_id=nexus_project_id,
                    extra_binds=extra_binds,
                )
                decision = parse_turn_action(env_dir)
                decision_name = decision.action if decision else "missing"
                if rc_ag != 0:
                    # A crashed agent may have left a ``continue`` from an
                    # earlier checkpoint in action.yaml; don't trust it.
                    decision_name = f"agent crash (rc={rc_ag})"
                if decision_name == "continue":
                    log.info("Turn %d: agent said continue — resuming job %s",
                             turn, run_id)
                    nexus.continue_job(run_id)
                    update_state(
                        env_dir, "waiting",
                        current_turn=turn,
                        current_job=run_id,
                        current_hive_id=event.get("hive_job_id") or "",
                        gpu_hours_used=gpu_hours_used,
                        gpu_budget=gpu_budget_hours,
                        checkpoint_count=checkpoint_count,
                    )
                    continue
                # Any other response (cancel / done / submit / missing /
                # agent crash): cancel rather than keep a GPU job nobody
                # is watching.
                log.info(
                    "Turn %d: agent decision=%s at checkpoint — cancelling job %s",
                    turn, decision_name, run_id,
                )
                nexus.cancel_job(run_id)
                # Don't break yet; let the next wait_for_job see the
                # cancelled event and fall through with a real terminal.
                continue
            if ev is None and event.get("error"):
                log.warning("wait_for_job error: %s — retrying", event["error"])
                time.sleep(10)
                continue
        elapsed_hr = (time.monotonic() - t0) / 3600.0
        gpu_hours_used += elapsed_hr * max(1, action.gpus)

        stdout_path, stderr_path, rc = _save_job_artifacts(env_dir, run_id, nexus)
        git.commit(env_dir, f"eval: job {run_id} done (rc={rc})")

        budget_kwargs = dict(
            gpu_hours_used=gpu_hours_used,
            gpu_budget=gpu_budget_hours,
        )
        if (terminal or {}).get("event") == "cancelled":
            prompt = _render(
                "on_job_cancelled",
                run_id=run_id,
                stdout_path=stdout_path, stderr_path=stderr_path,
                **budget_kwargs,
            )
        else:
            prompt = _render(
                "on_job_completion",
                run_id=run_id, returncode=rc,
                stdout_path=stdout_path, stderr_path=stderr_path,
                **budget_kwargs,
            )

    log.warning("Hit max_turns=%d without done", max_turns)
    return LoopResult(turns=turn, last_action="submit", reason="max_turns")
