"""Per-run token-usage extractor.

Walks ``<env_dir>/trajectory/turn_*_raw.jsonl``, sums tokens across every
turn the agent ran, and returns a normalized dict so we can record per-run
LLM spend independent of the underlying CLI scaffold.

The stream-json layouts differ per scaffold:

- **claude-code** (Claude models, and any model served through an
  Anthropic-compatible gateway):
  one ``{"type":"result","is_error":...,"usage":{input_tokens,
  cache_creation_input_tokens, cache_read_input_tokens, output_tokens},
  "total_cost_usd":F}`` event per turn. ``input_tokens`` is the *new* (un-cached)
  prompt portion; cache_read is the rest of the context.
- **codex**: one ``{"type":"turn.completed","usage":{input_tokens,
  cached_input_tokens, output_tokens, reasoning_output_tokens}}`` event per
  turn. ``input_tokens`` is the *total* context (cached + new); subtract
  ``cached_input_tokens`` for the "new" portion.
- **gemini** (official gemini-cli): one ``{"type":"result","status":...,
  "stats":{input, cached, output_tokens, total_tokens, ...}}`` event per
  turn. ``input`` is the new portion; ``cached`` is cache-read.

The ``kimi`` and ``claude-tmux`` scaffolds have no handler, so their
runs record zeros.

Normalized output shape (easy to sum across runs / per-model leaderboards):
    {
        "input_tokens_new":          int,    # billed-as-new input
        "input_tokens_cached_read":  int,    # context delivered via cache
        "input_tokens_cached_create": int,   # context being cached this turn (claude only)
        "output_tokens":             int,    # generated tokens (incl. reasoning)
        "reasoning_tokens":          int,    # subset of output (codex/gemini only)
        "cost_usd":                  float,  # claude-code only: the CLI's own
                                             # estimate at list prices, which is
                                             # not what a gateway model costs;
                                             # other scaffolds report 0.0 here
        "cost_source":               str,    # "claude_total_cost_usd" | "none"
        "turn_count":                int,    # number of usage events seen
        "scaffold":                  str,    # "claude-code" | "codex" | "gemini"
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

_EMPTY = {
    "input_tokens_new": 0,
    "input_tokens_cached_read": 0,
    "input_tokens_cached_create": 0,
    "output_tokens": 0,
    "reasoning_tokens": 0,
    "cost_usd": 0.0,
    "cost_source": "none",   # "claude_total_cost_usd" | "none"
    "turn_count": 0,
    "scaffold": "",
}


def _iter_events(path: Path) -> Iterable[dict]:
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(ev, dict):   # skip bare scalars/lists some streams emit
                    yield ev
    except OSError:
        return


def _add_claude_code(out: dict, ev: dict) -> bool:
    """Returns True if this event was a usage-bearing terminal event."""
    if ev.get("type") != "result":
        return False
    u = ev.get("usage") or {}
    out["input_tokens_new"] += int(u.get("input_tokens") or 0)
    out["input_tokens_cached_read"] += int(u.get("cache_read_input_tokens") or 0)
    out["input_tokens_cached_create"] += int(u.get("cache_creation_input_tokens") or 0)
    out["output_tokens"] += int(u.get("output_tokens") or 0)
    cost = ev.get("total_cost_usd")
    if cost is not None:
        try:
            out["cost_usd"] += float(cost)
            out["cost_source"] = "claude_total_cost_usd"
        except (TypeError, ValueError):
            pass
    return True


def _add_codex(out: dict, ev: dict) -> bool:
    if ev.get("type") != "turn.completed":
        return False
    u = ev.get("usage") or {}
    total_in = int(u.get("input_tokens") or 0)
    cached = int(u.get("cached_input_tokens") or 0)
    out["input_tokens_new"] += max(0, total_in - cached)
    out["input_tokens_cached_read"] += cached
    out["output_tokens"] += int(u.get("output_tokens") or 0)
    out["reasoning_tokens"] += int(u.get("reasoning_output_tokens") or 0)
    return True


def _add_gemini(out: dict, ev: dict) -> bool:
    if ev.get("type") != "result":
        return False
    s = ev.get("stats") or {}
    out["input_tokens_new"] += int(s.get("input") or 0)
    out["input_tokens_cached_read"] += int(s.get("cached") or 0)
    out["output_tokens"] += int(s.get("output_tokens") or 0)
    return True


_HANDLER = {
    "claude-code": _add_claude_code,
    "codex": _add_codex,
    "gemini": _add_gemini,
}


def extract_usage(env_dir: Path, scaffold: str) -> dict:
    """Sum token usage across every ``trajectory/turn_*_raw.jsonl`` file.

    Caller passes the run's ``profile.scaffold`` so we know which event
    schema to look for. Returns the normalized dict (see module docstring).

    Idempotent: safe to call multiple times. Missing trajectory dir →
    zeros (returns ``_EMPTY`` shape with the requested scaffold).
    """
    out = dict(_EMPTY)
    out["scaffold"] = scaffold
    handler = _HANDLER.get(scaffold)
    if handler is None:
        log.warning("extract_usage: no usage handler for scaffold %r — returning zeros", scaffold)
        return out
    traj = Path(env_dir) / "trajectory"
    if not traj.is_dir():
        return out
    for turn_file in sorted(traj.glob("turn_*_raw.jsonl")):
        for ev in _iter_events(turn_file):
            if handler(out, ev):
                out["turn_count"] += 1
    return out


# ── CLI for ad-hoc inspection ──────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Extract token usage from a run's env_dir.")
    p.add_argument("env_dir")
    p.add_argument("--scaffold", required=True,
                   choices=sorted(_HANDLER.keys()))
    args = p.parse_args()
    result = extract_usage(Path(args.env_dir), args.scaffold)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
