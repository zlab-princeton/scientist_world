#!/usr/bin/env python3
"""Structured analysis: one coding-agent run per paper.

For each paper in an ID list (or a daily listing), launch an agent CLI that
reads the paper and fills in a structured catalog entry (result.json)
following prompts/analyst.md. The entry is validated against
prompts/example.json; on a schema error the agent session is resumed with
the error list.

The agent is the Claude Code CLI (`claude`), which can also be pointed at any
Anthropic-compatible endpoint (--api-url, or ANTHROPIC_BASE_URL through --env).

Needs the agent CLI with its login or API credentials in the environment,
and network access for the model API and for the agent's web search.

The agent is not sandboxed: it runs with file read/write and web tools on
text derived from untrusted PDFs. Run it under an account, container or
machine that holds nothing the agent must not read.

Usage:
    python analyze_papers.py --ids-file ids.json                 # explicit ID list
    python analyze_papers.py --date 2025-06-09 --category cs.CV  # one daily listing
    python analyze_papers.py --ids-file ids.json --limit 10      # at most 10
    python analyze_papers.py --ids-file ids.json --model MODEL_ID --effort medium
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────


def resolve_data_dir(cli_value) -> Path:
    """Return the data directory: --data-dir, else $SCOUT_DATA_DIR, else ./data."""
    return Path(cli_value or os.environ.get("SCOUT_DATA_DIR") or "data").resolve()


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = resolve_data_dir(None)
LISTING_DIR = DATA_DIR / "arxiv_listings"
PDF_DIR = DATA_DIR / "arxiv_pdfs"
MD_DIR = DATA_DIR / "arxiv_mds"
ANALYSIS_DIR = DATA_DIR / "arxiv_analysis"
PROMPT_FILE = SCRIPT_DIR / "prompts" / "analyst.md"
EXAMPLE_FILE = SCRIPT_DIR / "prompts" / "example.json"
REF_DIR = SCRIPT_DIR / "reference_sheets"

RUNTIME_DIR = SCRIPT_DIR / "analyst_runtime"
DONE_MARKER = ".done"
AGENT_OUTPUT = "analysis.txt"
REF_SHEETS = ["model_sheet.jsonl", "api_price_sheet.jsonl",
              "benchmark_dataset_registry.jsonl",
              "domain_registry.jsonl", "area_registry.jsonl"]


# ── Agent result ─────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    text: str
    returncode: int
    stderr: str = ""
    session_id: str | None = None
    stats: dict = field(default_factory=dict)
    endpoint: str | None = None


# ── Provider runners ─────────────────────────────────────────────────────────
# Each provider function takes (prompt, result_folder, model_cfg, session_id)
# and returns an AgentResult (with stats parsed from output).
# To add a new provider: write a runner function and add it to PROVIDERS.

def _parse_claude_output(stdout: str) -> tuple[str, str | None, dict, list[str]]:
    """Parse Claude Code --output-format stream-json output.

    Returns (text, session_id, stats, errors) where stats includes cost, tokens,
    tool use counts, etc., and errors is a list of error messages.
    """
    text = ""
    session_id = None
    errors = []
    stats = {
        "cost_usd": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "num_turns": 0,
        "duration_api_ms": 0,
        "tool_uses": {},
    }

    for line in stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = data.get("type")

        if msg_type == "error":
            # Error events: {"type": "error", "error": {"message": "..."}}
            # or {"type": "error", "message": "..."}
            err = data.get("error", {})
            msg = err.get("message") if isinstance(err, dict) else None
            msg = msg or data.get("message") or str(err or data)
            errors.append(msg)

        elif msg_type == "result":
            text = data.get("result", "")
            session_id = data.get("session_id")
            if data.get("is_error"):
                errors.append(text)
            for key in ("cost_usd", "num_turns", "duration_api_ms"):
                if data.get(key) is not None:
                    stats[key] = data[key]
            # Current Claude Code releases report the cost as total_cost_usd.
            if data.get("total_cost_usd") is not None:
                stats["cost_usd"] = data["total_cost_usd"]

        elif msg_type == "assistant":
            message = data.get("message", data)
            # Detect error field (e.g., rate_limit, auth errors)
            if data.get("error"):
                for block in message.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        errors.append(block["text"])
            # Token usage
            usage = message.get("usage", {})
            stats["input_tokens"] += usage.get("input_tokens", 0)
            stats["output_tokens"] += usage.get("output_tokens", 0)
            # Tool uses from content blocks
            for block in message.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "unknown")
                    stats["tool_uses"][name] = stats["tool_uses"].get(name, 0) + 1

    # Fallback: try entire stdout as single JSON (plain --output-format json)
    if not text and not session_id:
        try:
            data = json.loads(stdout)
            text = data.get("result", stdout)
            session_id = data.get("session_id")
            for key in ("cost_usd", "num_turns", "duration_api_ms"):
                if data.get(key) is not None:
                    stats[key] = data[key]
            if data.get("total_cost_usd") is not None:
                stats["cost_usd"] = data["total_cost_usd"]
        except json.JSONDecodeError:
            text = stdout

    return text, session_id, stats, errors


def _apply_env_overrides(env: dict, model_cfg: dict) -> dict:
    """Apply env_set and env_unset from model config to env dict."""
    for k in model_cfg.get("env_unset", []):
        env.pop(k, None)
    env.update(model_cfg.get("env_set", {}))
    return env


def _normalize_api_url_for_provider(url: str, provider: str) -> str:
    """Normalize a user-facing API URL to the provider's expected base URL."""
    url = url.rstrip("/")
    if provider == "claude" and url.endswith("/v1"):
        return url[:-3]
    return url


def _select_endpoint(model_cfg: dict) -> tuple[dict, str | None]:
    """Return the next endpoint-specific env overrides and a label."""
    endpoint_env_sets = model_cfg.get("endpoint_env_sets") or []
    endpoint_labels = model_cfg.get("endpoint_labels") or []
    if not endpoint_env_sets:
        return {}, None

    with model_cfg["_endpoint_lock"]:
        idx = model_cfg["_endpoint_idx"]
        model_cfg["_endpoint_idx"] = (idx + 1) % len(endpoint_env_sets)

    label = endpoint_labels[idx] if idx < len(endpoint_labels) else None
    return endpoint_env_sets[idx], label


def _run_claude(prompt: str, result_folder: Path, model_cfg: dict,
                session_id: str | None = None) -> AgentResult:
    """Run analysis via Claude Code CLI."""
    env = os.environ.copy()
    if model_cfg.get("effort"):
        env["CLAUDE_CODE_EFFORT_LEVEL"] = model_cfg["effort"]
    _apply_env_overrides(env, model_cfg)
    endpoint_env, endpoint_label = _select_endpoint(model_cfg)
    env.update(endpoint_env)

    cmd = ["claude"]
    if session_id:
        cmd.extend(["--resume", session_id])
    cmd.extend(["-p", prompt])
    if model_cfg.get("id"):
        cmd.extend(["--model", model_cfg["id"]])
    cmd.extend([
        "--max-turns", str(model_cfg.get("max_turns", 100)),
        "--allowedTools", "Read,Write,Edit,WebSearch,WebFetch,Glob,Grep",
        "--output-format", "stream-json",
        "--verbose",
        "--setting-sources", "",
    ])

    # Run from an otherwise empty directory, with no setting sources, so the
    # agent picks up no project or user configuration.
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        env=env, cwd=str(RUNTIME_DIR),
    )

    # Save raw output for debugging
    (result_folder / "raw_output.jsonl").write_text(result.stdout)

    text, new_session_id, stats, errors = _parse_claude_output(result.stdout)
    stderr = result.stderr
    if errors:
        stderr = (stderr + "\n" if stderr else "") + "\n".join(errors)
    return AgentResult(
        text=text, returncode=result.returncode,
        stderr=stderr, session_id=new_session_id, stats=stats,
        endpoint=endpoint_label,
    )


PROVIDERS = {
    "claude": _run_claude,
}
DEFAULT_PROVIDER = "claude"


# ── Stats helpers ────────────────────────────────────────────────────────────

def _empty_stats() -> dict:
    """Return a fresh cumulative stats dict."""
    return {
        "cost_usd": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "num_turns": 0,
        "duration_api_ms": 0,
        "tool_uses": {},
        "last_endpoint": None,
    }


def _accumulate_stats(cumulative: dict, new: dict, endpoint: str | None = None):
    """Merge stats from one agent run into cumulative totals."""
    if not new:
        return
    for key in ("cost_usd", "num_turns", "duration_api_ms",
                "input_tokens", "output_tokens"):
        cumulative[key] = cumulative.get(key, 0) + new.get(key, 0)
    for tool, count in new.get("tool_uses", {}).items():
        cumulative["tool_uses"][tool] = cumulative["tool_uses"].get(tool, 0) + count
    if endpoint:
        cumulative["last_endpoint"] = endpoint


def _snapshot_ref_sheets(ref_dir: Path = REF_DIR) -> dict[str, int]:
    """Return {filename: line_count} for each reference sheet."""
    counts = {}
    for name in REF_SHEETS:
        path = ref_dir / name
        if path.exists():
            counts[name] = len(path.read_text().strip().split("\n"))
        else:
            counts[name] = 0
    return counts


# Lock for merging ref sheet updates back into canonical files.
_ref_lock = threading.Lock()


def _copy_ref_sheets(dest_dir: Path) -> None:
    """Copy canonical reference sheets to dest_dir for isolated agent access."""
    with _ref_lock:
        for name in REF_SHEETS:
            src = REF_DIR / name
            if src.exists():
                shutil.copy2(src, dest_dir / name)


# Fields that identify a row of a reference sheet. Rows are keyed by "name",
# except in the price sheet, whose rows have no name and are identified by
# provider and model.
REF_SHEET_KEYS = {"api_price_sheet.jsonl": ("provider", "model")}


def _ref_row_key(sheet_name: str, line: str):
    """Return the identity of one reference-sheet row, or None if it has none."""
    try:
        row = json.loads(line)
        key = tuple(row[k] for k in REF_SHEET_KEYS.get(sheet_name, ("name",)))
        hash(key)
        return key
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _merge_ref_sheets(agent_dir: Path) -> None:
    """Merge new entries from an agent's local ref sheets into canonical files.

    Deduplicates by the identity fields of each sheet (see REF_SHEET_KEYS).
    Only appends rows that are not in the canonical file; changes an agent
    made to existing rows are not merged.
    """
    with _ref_lock:
        for sheet_name in REF_SHEETS:
            canonical_path = REF_DIR / sheet_name
            agent_file = agent_dir / sheet_name
            if not agent_file.exists():
                continue

            # Collect canonical row identities
            seen_keys = set()
            if canonical_path.exists():
                for line in canonical_path.read_text().strip().split("\n"):
                    if not line.strip():
                        continue
                    key = _ref_row_key(sheet_name, line)
                    if key is not None:
                        seen_keys.add(key)

            # Find new entries from agent's copy
            new_lines = []
            for line in agent_file.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                key = _ref_row_key(sheet_name, line)
                if key is None:
                    continue
                if key not in seen_keys:
                    seen_keys.add(key)
                    new_lines.append(line)

            if new_lines:
                with open(canonical_path, "a") as f:
                    for line in new_lines:
                        f.write(line + "\n")


def _insert_metadata(result_folder: Path, model_cfg: dict,
                     session_id: str | None, duration: float,
                     retries: int, stats: dict, ref_changes: dict):
    """Insert a metadata dict at the top of result.json."""
    result_path = result_folder / "result.json"
    result_data = json.loads(result_path.read_text())

    metadata = {
        "model": model_cfg["id"],
        "provider": model_cfg["provider"],
        "endpoint": stats.get("last_endpoint"),
        "session_id": session_id,
        "duration_seconds": round(duration, 1),
        "cost_usd": round(stats.get("cost_usd", 0), 4),
        "num_turns": stats.get("num_turns", 0),
        "input_tokens": stats.get("input_tokens", 0),
        "output_tokens": stats.get("output_tokens", 0),
        "retries": retries,
        "tool_uses": stats.get("tool_uses", {}),
        "reference_sheet_updates": ref_changes,
    }

    result_data = {"metadata": metadata, **result_data}
    result_path.write_text(json.dumps(result_data, indent=4))


# ── Schema validation ───────────────────────────────────────────────────────

def validate_schema(result, example, path="root") -> list[str]:
    """Compare structure of result against example JSON. Returns list of errors."""
    errors = []

    if type(result) is not type(example):
        errors.append(f"{path}: expected {type(example).__name__}, got {type(result).__name__}")
        return errors

    if isinstance(example, dict):
        missing = set(example.keys()) - set(result.keys())
        extra = set(result.keys()) - set(example.keys())
        for k in sorted(missing):
            errors.append(f"{path}.{k}: missing key")
        for k in sorted(extra):
            errors.append(f"{path}.{k}: unexpected extra key")
        for k in sorted(set(example.keys()) & set(result.keys())):
            errors.extend(validate_schema(result[k], example[k], f"{path}.{k}"))

    elif isinstance(example, list):
        if example and result:
            # Check structure of first element against example's first element
            errors.extend(validate_schema(result[0], example[0], f"{path}[0]"))

    return errors


def validate_score_ranges(result_data: dict) -> list[str]:
    """Enforce numeric score ranges beyond pure schema shape checks."""
    errors = []
    score_paths = [
        ("reproduction_scores.coding_difficulty.score", 0, 10),
        ("reproduction_scores.compute_difficulty.score", 0, 10),
        ("reproduction_scores.speed_difficulty.score", 0, 10),
        ("reproduction_scores.data_model_difficulty.score", 0, 10),
        ("reproduction_scores.overall.score", 0, 10),
        ("paper_quality.score", 0, 10),
        ("problem_novelty.score", 0, 10),
    ]

    for dotted_path, lo, hi in score_paths:
        cur = result_data
        missing = False
        for part in dotted_path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                missing = True
                break
            cur = cur[part]
        if missing:
            continue
        if not isinstance(cur, int):
            errors.append(f"root.{dotted_path}: expected int, got {type(cur).__name__}")
            continue
        if not (lo <= cur <= hi):
            errors.append(f"root.{dotted_path}: expected {lo}-{hi}, got {cur}")

    return errors


def check_result(result_folder: Path) -> list[str]:
    """Load result.json and validate against example.json schema.

    Ignores the 'metadata' key which is added by the script, not the agent.
    Also detects if result.json is unchanged from the template.
    """
    result_path = result_folder / "result.json"
    try:
        result_data = json.loads(result_path.read_text())
    except FileNotFoundError:
        return ["result.json not found"]
    except json.JSONDecodeError as e:
        return [f"result.json is not valid JSON: {e}"]

    # Strip metadata (added by script, not part of the agent's schema)
    result_data.pop("metadata", None)

    example_data = json.loads(EXAMPLE_FILE.read_text())

    # Detect unchanged template — agent didn't do any work
    if result_data == example_data:
        return ["result.json is unchanged from template — agent produced no output"]

    errors = validate_schema(result_data, example_data)
    errors.extend(validate_score_ranges(result_data))
    return errors


# ── Helpers ──────────────────────────────────────────────────────────────────

def default_date() -> str:
    """Most recent weekday before today (matches fetch_daily_arxiv.py)."""
    d = datetime.now() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def is_done(arxiv_id: str) -> bool:
    """Check if a paper has already been analyzed."""
    return (ANALYSIS_DIR / arxiv_id / DONE_MARKER).exists()


def count_done(arxiv_ids: list[str]) -> int:
    """Count completed papers from the provided ID list."""
    return sum(1 for aid in arxiv_ids if is_done(aid))


def cleanup_incomplete_folders(arxiv_ids: list[str]) -> int:
    """Remove all non-done analysis folders for the provided ID list."""
    removed = 0
    for aid in arxiv_ids:
        folder = ANALYSIS_DIR / aid
        if folder.exists() and not (folder / DONE_MARKER).exists():
            shutil.rmtree(folder)
            removed += 1
    return removed


def setup_folder(arxiv_id: str) -> Path:
    """Create or reset the analysis folder for a paper.

    - New folder: create and copy example JSON template.
    - Incomplete folder (no .done marker): clean up and restart.
    - Done folder: should not reach here (filtered earlier).
    """
    folder = ANALYSIS_DIR / arxiv_id
    if folder.exists():
        if (folder / DONE_MARKER).exists():
            return folder
        # Exists but not done — clean up and restart
        print(f"  [{arxiv_id}] Cleaning up incomplete analysis...")
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(EXAMPLE_FILE, folder / "result.json")
    return folder


TOOLS_CLAUDE = """\
### Tools & Permissions
You have access to: **Read**, **Write**, **Edit**, **Glob**, **Grep**, **WebSearch**, **WebFetch**.
You do NOT have access to Bash or any other tools.
You may only read and write files within the paths listed above. Do not access any other files on the system."""

TOOLS_BY_PROVIDER = {
    "claude": TOOLS_CLAUDE,
}


def build_prompt(arxiv_id: str, result_folder: Path, provider: str,
                 ref_dir: Path = REF_DIR) -> str:
    """Construct the full prompt: analyst.md + runtime context."""
    analyst_prompt = PROMPT_FILE.read_text()

    paper_md_dir = MD_DIR / arxiv_id
    paper_pdf = PDF_DIR / f"{arxiv_id}.pdf"
    tools_section = TOOLS_BY_PROVIDER.get(provider, TOOLS_CLAUDE)

    runtime = f"""
---

## Runtime Context

- **arXiv ID**: `{arxiv_id}`
- **Paper markdown**: `{paper_md_dir}` (read this first — contains the paper in markdown format with images)
- **Paper PDF**: `{paper_pdf}` — **Only read the PDF if the markdown is converted poorly or does not contain enough information.** Reading the PDF consumes significant tokens.
- **Result folder**: `{result_folder}`
- **Result file**: `{result_folder}/result.json` — modify this file with your analysis
- **Reference sheets**:
  - `{ref_dir}/model_sheet.jsonl`
  - `{ref_dir}/api_price_sheet.jsonl`
  - `{ref_dir}/benchmark_dataset_registry.jsonl`
  - `{ref_dir}/domain_registry.jsonl`
  - `{ref_dir}/area_registry.jsonl`

Read the paper thoroughly, then fill in `result.json` following the schema already in the file. You may also update the reference sheet JSONL files if you find missing or incorrect entries.

{tools_section}"""
    return analyst_prompt + "\n" + runtime


# ── Agent runner ─────────────────────────────────────────────────────────────

def run_agent(arxiv_id: str, result_folder: Path,
              model_cfg: dict, max_retries: int = 3) -> bool:
    """Launch an LLM agent to analyze one paper. Returns True on success.

    Each agent gets its own copy of reference sheets in result_folder to
    avoid conflicts when running in parallel. After the agent finishes,
    new ref sheet entries are merged back into canonical files under a lock.

    Validates result.json against the example schema. If validation fails,
    resumes the agent session with error details up to max_retries times.
    On success, inserts metadata into result.json.
    """
    provider_fn = PROVIDERS[model_cfg["provider"]]
    cumulative = _empty_stats()
    retries_used = 0

    # Give agent its own copy of reference sheets (safe for parallel runs).
    # The lock inside _copy_ref_sheets ensures we get a consistent snapshot
    # that includes all previously merged updates.
    _copy_ref_sheets(result_folder)
    prompt = build_prompt(arxiv_id, result_folder, model_cfg["provider"],
                          ref_dir=result_folder)

    # Snapshot local reference sheets before agent runs
    ref_before = _snapshot_ref_sheets(result_folder)
    start_time = time.time()

    print(f"  [{arxiv_id}] Starting agent "
          f"({model_cfg['id'] or model_cfg['provider'] + ' default model'})...")
    result = provider_fn(prompt, result_folder, model_cfg)
    _accumulate_stats(cumulative, result.stats, result.endpoint)

    # Save agent's text output
    (result_folder / AGENT_OUTPUT).write_text(result.text)

    if result.returncode != 0:
        print(f"  [{arxiv_id}] Failed (exit code {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[:5]:
                print(f"    {line}")
        return False

    # Validate and retry loop
    for attempt in range(max_retries):
        errors = check_result(result_folder)
        if not errors:
            break

        retries_used = attempt + 1
        print(f"  [{arxiv_id}] Schema errors (retry {retries_used}/{max_retries}):")
        for e in errors[:10]:
            print(f"    {e}")

        if not result.session_id:
            print(f"  [{arxiv_id}] Cannot retry — no session to resume")
            break

        fix_prompt = (
            "The result.json you wrote has schema errors — it does not match the "
            "expected structure. Fix result.json so it matches exactly.\n\n"
            "Errors:\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\n\nHere is the expected JSON structure:\n```json\n"
            + EXAMPLE_FILE.read_text()
            + "\n```\n"
        )
        result = provider_fn(fix_prompt, result_folder, model_cfg,
                             session_id=result.session_id)
        _accumulate_stats(cumulative, result.stats, result.endpoint)

        # Append retry output
        with open(result_folder / AGENT_OUTPUT, "a") as f:
            f.write(f"\n\n=== Retry {retries_used} ===\n{result.text}")

        if result.returncode != 0:
            print(f"  [{arxiv_id}] Retry failed (exit code {result.returncode})")
            break

    # Final check
    errors = check_result(result_folder)
    if errors:
        print(f"  [{arxiv_id}] Schema validation failed after {max_retries} retries — not marked done")
        return False

    # Success — merge ref sheets, insert metadata, mark done
    duration = time.time() - start_time
    ref_after = _snapshot_ref_sheets(result_folder)
    ref_changes = {
        name: ref_after[name] - ref_before[name]
        for name in REF_SHEETS
    }
    _merge_ref_sheets(result_folder)
    _insert_metadata(result_folder, model_cfg, result.session_id,
                     duration, retries_used, cumulative, ref_changes)
    (result_folder / DONE_MARKER).touch()
    print(f"  [{arxiv_id}] Done ({round(duration)}s, ${cumulative['cost_usd']:.3f})")
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze arXiv papers using LLM agents.")
    parser.add_argument("--date", default=None,
                        help="YYYY-MM-DD (default: last weekday)")
    parser.add_argument("--category", nargs="+", default=["cs.LG"],
                        help="arXiv categories (default: cs.LG)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process (default: all)")
    parser.add_argument("--ids-file", default=None,
                        help="JSON file containing an explicit list of arXiv IDs "
                             "to process. Overrides --date/--category listings.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER,
                        choices=list(PROVIDERS.keys()),
                        help="Agent CLI that runs the analysis "
                             f"(default: {DEFAULT_PROVIDER})")
    parser.add_argument("--model", "--model-id", dest="model_id", default=None,
                        help="Model ID passed to the agent CLI "
                             "(default: the CLI's own default model)")
    parser.add_argument("--env", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Extra environment variable for the agent subprocess, "
                             "e.g. ANTHROPIC_BASE_URL for an Anthropic-compatible "
                             "endpoint. Write --env=-KEY to unset KEY. Repeatable. "
                             "The subprocess inherits the calling environment, so "
                             "export secrets there instead of passing them here: "
                             "command-line arguments are visible in the process list.")
    parser.add_argument("--api-url", nargs="+", default=None,
                        help="One or more backend URLs for load balancing. "
                             "A trailing /v1 is stripped and the URL is used "
                             "as ANTHROPIC_BASE_URL.")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max schema validation retries per paper (default: 3)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    parser.add_argument("--pass-retries", type=int, default=5,
                        help="Number of full-pass retries for unfinished papers "
                             "after the initial pass (default: 5)")
    parser.add_argument("--retry-sleep-sec", type=int, default=600,
                        help="Base sleep between full-pass retries in seconds "
                             "(default: 600)")
    parser.add_argument("--retry-sleep-increment-sec", type=int, default=1800,
                        help="Extra sleep added after each no-progress retry "
                             "in seconds (default: 1800)")
    parser.add_argument("--effort", default=None,
                        choices=["low", "medium", "high"],
                        help="Reasoning effort passed to the agent CLI "
                             "(default: the CLI's own default)")
    parser.add_argument("--tag", default=None,
                        help="Analysis dir suffix (e.g. --tag v2 writes to arxiv_analysis_v2)")
    parser.add_argument("--data-dir", default=None,
                        help="Data directory (default: $SCOUT_DATA_DIR, else ./data)")
    args = parser.parse_args()

    # Override data paths if --data-dir is provided
    global DATA_DIR, LISTING_DIR, PDF_DIR, MD_DIR, ANALYSIS_DIR
    if args.data_dir:
        DATA_DIR = resolve_data_dir(args.data_dir)
        LISTING_DIR = DATA_DIR / "arxiv_listings"
        PDF_DIR = DATA_DIR / "arxiv_pdfs"
        MD_DIR = DATA_DIR / "arxiv_mds"
        ANALYSIS_DIR = DATA_DIR / "arxiv_analysis"

    if args.tag:
        ANALYSIS_DIR = DATA_DIR / f"arxiv_analysis_{args.tag}"

    date_str = args.date or default_date()

    # Build model config
    model_cfg = {"provider": args.provider, "id": args.model_id}

    if args.effort:
        model_cfg["effort"] = args.effort

    # Parse --env overrides into model_cfg
    if args.env:
        env_set, env_unset = {}, []
        for e in args.env:
            if e.startswith("-"):
                env_unset.append(e[1:])
            elif "=" in e:
                k, v = e.split("=", 1)
                env_set[k] = v
            else:
                parser.error(f"Invalid --env format: {e}. Use KEY=VALUE or -KEY")
        model_cfg["env_set"] = env_set
        model_cfg["env_unset"] = env_unset

    if args.api_url:
        endpoint_env_sets = []
        endpoint_labels = []
        env_set = model_cfg.get("env_set", {})
        auth_present = (
            "ANTHROPIC_AUTH_TOKEN" in env_set
            or "ANTHROPIC_API_KEY" in env_set
            or "ANTHROPIC_AUTH_TOKEN" in os.environ
            or "ANTHROPIC_API_KEY" in os.environ
        )
        for raw_url in args.api_url:
            url = _normalize_api_url_for_provider(raw_url, model_cfg["provider"])
            if model_cfg["provider"] == "claude":
                endpoint_env = {"ANTHROPIC_BASE_URL": url}
                if not auth_present:
                    endpoint_env["ANTHROPIC_AUTH_TOKEN"] = "dummy"
            else:
                parser.error(f"--api-url is not supported for provider {model_cfg['provider']}")

            endpoint_env_sets.append(endpoint_env)
            endpoint_labels.append(url)

        model_cfg["endpoint_env_sets"] = endpoint_env_sets
        model_cfg["endpoint_labels"] = endpoint_labels
        model_cfg["_endpoint_idx"] = 0
        model_cfg["_endpoint_lock"] = threading.Lock()

    # Load IDs either from an explicit ids file or from listings.
    all_ids = []
    if args.ids_file:
        ids_path = Path(args.ids_file)
        if not ids_path.exists():
            print(f"IDs file not found: {ids_path}")
            return
        all_ids = json.loads(ids_path.read_text())
        if not isinstance(all_ids, list):
            print(f"IDs file must contain a JSON list: {ids_path}")
            return
        all_ids = [str(aid) for aid in all_ids]
        print(f"IDs file: {ids_path} — {len(all_ids)} papers")
    else:
        for category in args.category:
            listing_file = LISTING_DIR / date_str / f"{category}.json"
            if not listing_file.exists():
                print(f"Listing not found: {date_str}/{category} — skipping.")
                continue
            ids = json.loads(listing_file.read_text())
            print(f"Listing: {date_str}/{category} — {len(ids)} papers")
            all_ids.extend(ids)
        all_ids = list(dict.fromkeys(all_ids))  # deduplicate, preserve order

    if not all_ids:
        print("No listings found.")
        return

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    def process_one(arxiv_id: str) -> bool:
        folder = setup_folder(arxiv_id)
        return run_agent(arxiv_id, folder, model_cfg, args.max_retries)

    def run_pass(ready: list[str]) -> tuple[int, int]:
        success, failed = 0, 0
        if args.workers <= 1:
            # Sequential
            for i, arxiv_id in enumerate(ready, 1):
                print(f"\n[{i}/{len(ready)}] {arxiv_id}")
                if process_one(arxiv_id):
                    success += 1
                else:
                    failed += 1
        else:
            # Parallel
            print(f"\nRunning with {args.workers} parallel workers")
            done_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(process_one, aid): aid
                    for aid in ready
                }
                for future in concurrent.futures.as_completed(futures):
                    arxiv_id = futures[future]
                    done_count += 1
                    try:
                        if future.result():
                            success += 1
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"  [{arxiv_id}] Exception: {e}")
                        failed += 1
                    print(f"  Progress: {done_count}/{len(ready)}")
        return success, failed

    if args.limit is not None:
        # Fix the selection once; limiting again on each retry pass would move
        # on to the next papers after any paper of this selection finished.
        pending = [aid for aid in all_ids if not is_done(aid)]
        print(f"  Already done: {len(all_ids) - len(pending)}  |  Remaining: {len(pending)}")
        all_ids = pending[:args.limit]
        print(f"  Processing: {len(all_ids)} (limited)")

    total_retries_used = 0
    no_progress_retries = 0
    pass_num = 0
    done_before = count_done(all_ids)

    while True:
        todo = [aid for aid in all_ids if not is_done(aid)]
        print(f"  Already done: {len(all_ids) - len(todo)}  |  Remaining: {len(todo)}")

        if not todo:
            print("Nothing to process.")
            break

        # Silently skip papers missing PDF or markdown (may be unavailable on arXiv)
        ready = [aid for aid in todo
                 if (MD_DIR / aid).exists() and (PDF_DIR / f"{aid}.pdf").exists()]
        skipped = len(todo) - len(ready)

        if not ready:
            print("No papers with complete files to process.")
            break

        if skipped:
            print(f"  Ready: {len(ready)}  |  Skipped (missing files): {skipped}")

        pass_num += 1
        print(f"\n=== Pass {pass_num} ===")
        success, failed = run_pass(ready)
        print(f"\nPass {pass_num} complete. Success: {success}  |  Failed: {failed}  |  Total: {len(ready)}")

        done_after = count_done(all_ids)
        gained = done_after - done_before
        done_before = done_after

        remaining = len([aid for aid in all_ids if not is_done(aid)])
        if remaining == 0:
            print("\nAll papers complete.")
            break

        if no_progress_retries >= args.pass_retries:
            print(f"\nStopping with {remaining} papers unfinished after {pass_num} pass(es); "
                  f"consecutive no-progress retry budget exhausted ({args.pass_retries}).")
            break

        cleaned = cleanup_incomplete_folders(all_ids)
        if gained == 0:
            no_progress_retries += 1
        else:
            no_progress_retries = 0
        total_retries_used += 1

        sleep_sec = args.retry_sleep_sec + (
            no_progress_retries * args.retry_sleep_increment_sec
        )
        print(f"\nRetry pass {total_retries_used}: cleaned {cleaned} unfinished folders, "
              f"gained {gained} new done papers, {remaining} still unfinished.")
        print(f"Consecutive no-progress retries: {no_progress_retries}/{args.pass_retries}")
        print(f"Sleeping {sleep_sec}s before the next pass.")
        time.sleep(sleep_sec)


if __name__ == "__main__":
    main()
