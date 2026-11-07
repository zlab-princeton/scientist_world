"""Field extractors for the traces dataset.

Three concerns:
  1. INPUT files (problem/direction/constraint/target/...) with leaky
     sub-sections stripped.
  2. THINKING TRACE — three different trajectory schemas, one per scaffold
     (codex / claude-code / kimi). Returns plain reasoning text in turn
     order. The gemini scaffold has no parser: its rows have no trace.
  3. IDEA EVOLUTION — the ordered list of idea_vN.md (cumulative deltas) plus
     the final reflection.md (self-contained summary).
"""
from __future__ import annotations

import json
import re
from pathlib import Path


# ── input files ───────────────────────────────────────────────────────

def read_text(p: Path) -> str | None:
    try:
        return p.read_text(errors="replace") if p.exists() else None
    except OSError:
        return None


# Markdown section headers whose bodies leak target numbers / results into
# what would otherwise be a clean task-input column. Dropped from the
# `overview` and `evaluation` inputs (the rest of those files is spec).
_LEAKY_HEADERS = re.compile(
    r"^#{1,6}\s*(key results|target numbers?|reported results?|"
    r"results? from the paper|paper results?|critical finding)\b",
    re.IGNORECASE,
)


def strip_leaky_sections(md: str | None) -> str | None:
    """Drop any markdown section whose header matches a leaky pattern, up to
    the next header of the same-or-higher level (or EOF)."""
    if not md:
        return md
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _LEAKY_HEADERS.match(lines[i].strip())
        if m:
            level = len(lines[i]) - len(lines[i].lstrip("#"))
            i += 1
            fence = None   # a ``#`` line inside a code block is not a header
            while i < len(lines):
                nxt = lines[i].lstrip()
                if fence is None and nxt.startswith(("```", "~~~")):
                    fence = nxt[:3]
                elif fence is not None:
                    if nxt.startswith(fence):
                        fence = None
                elif nxt.startswith("#"):
                    nlevel = len(lines[i]) - len(lines[i].lstrip("#"))
                    if nlevel <= max(level, 1):
                        break
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out).strip() or None


# Map dataset input column -> (subdir, filename, strip_leaky?)
INPUT_FILES = {
    "problem":     ("briefing", "problem.md", False),
    "direction":   ("scoring", "DIRECTION.md", False),
    "constraint":  ("scoring", "CONSTRAINTS.md", False),
    "target":      ("scoring", "TARGETS.md", True),
    "experiments": ("scoring", "EXPERIMENTS.md", False),
    "overview":    ("briefing", "overview.md", True),
    "evaluation":  ("briefing", "evaluation.md", True),
}


def extract_inputs(source_dir: Path) -> dict:
    out = {}
    for col, (sub, fname, strip) in INPUT_FILES.items():
        txt = read_text(source_dir / sub / fname)
        out[col] = strip_leaky_sections(txt) if strip else txt
    return out


def extract_paper_manifest(source_dir: Path) -> dict:
    """Per-PAPER metadata from ``source/manifest.json`` (shared across every
    run of the paper): the paper title and taxonomy topics.

    Returns keys ``paper_title`` and ``paper_topics``, or Nones when absent.
    """
    empty = {"paper_title": None, "paper_topics": None}
    try:
        man = json.loads((source_dir / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(man, dict):
        return empty
    paper = man.get("paper") or {}
    return {
        "paper_title": paper.get("title") or None,
        "paper_topics": paper.get("topics") or None,
    }


# ── idea evolution + final output ─────────────────────────────────────

_IDEA_RE = re.compile(r"idea_v(\d+)\.md$")


def extract_ideas(run_dir: Path) -> tuple[list[str], str | None]:
    """Return (ordered idea_vN texts, final idea text). Ordered by version."""
    idea_dir = run_dir / "idea"
    if not idea_dir.is_dir():
        return [], None
    versioned = []
    for f in idea_dir.glob("idea_v*.md"):
        m = _IDEA_RE.search(f.name)
        if m:
            versioned.append((int(m.group(1)), f))
    versioned.sort()
    texts = [read_text(f) or "" for _, f in versioned]
    final = texts[-1] if texts else None
    return texts, final


def extract_reflection(run_dir: Path) -> str | None:
    """Self-contained final summary. Returns None if it's still the unfilled
    template placeholder."""
    txt = read_text(run_dir / "reflection.md")
    if not txt:
        return None
    t = txt.strip()
    # Unfilled placeholder runs leave an HTML-comment stub.
    if t.startswith("<!--") and t.endswith("-->"):
        return None
    if len(t) < 20:
        return None
    return txt


# ── thinking trace (per-scaffold trajectory parsers) ──────────────────

def _turn_files(run_dir: Path) -> list[Path]:
    tdir = run_dir / "trajectory"
    if not tdir.is_dir():
        return []
    files = [f for f in tdir.glob("turn_*.jsonl") if f.stat().st_size > 0]
    files.sort(key=lambda f: f.name)
    return files


def _iter_jsonl(path: Path):
    try:
        for ln in path.read_text(errors="replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):   # skip bare scalars/lists some streams emit
                yield obj
    except OSError:
        return


def _trace_codex(files: list[Path]) -> list[str]:
    """Codex stream: reasoning is only narrated in agent_message items
    (no hidden CoT logged)."""
    out = []
    for f in files:
        for o in _iter_jsonl(f):
            if o.get("type") == "item.completed":
                item = o.get("item") or {}
                if item.get("type") == "agent_message":
                    txt = (item.get("text") or "").strip()
                    if txt:
                        out.append(txt)
    return out


def _trace_claude_code(files: list[Path]) -> list[str]:
    """Claude Agent SDK stream-json: extended-thinking blocks on assistant
    messages (genuine chain-of-thought)."""
    out = []
    for f in files:
        for o in _iter_jsonl(f):
            if o.get("type") != "assistant":
                continue
            content = ((o.get("message") or {}).get("content")) or []
            if not isinstance(content, list):
                continue
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "thinking":
                    txt = (blk.get("thinking") or "").strip()
                    if txt:
                        out.append(txt)
    return out


def _trace_kimi(files: list[Path]) -> list[str]:
    """Kimi-cli stream: 'think' content blocks (genuine chain-of-thought).
    Records have no top-level type; each line is {role, content:[...]}.."""
    out = []
    for f in files:
        for o in _iter_jsonl(f):
            content = o.get("content")
            if not isinstance(content, list):
                continue
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "think":
                    txt = (blk.get("think") or "").strip()
                    if txt:
                        out.append(txt)
    return out


_TRACE_PARSERS = {
    "codex": _trace_codex,
    "claude-code": _trace_claude_code,
    # The tmux driver saves claude's own session records per turn
    # (turn_NNNN_session.jsonl); they use the same message layout.
    "claude-tmux": _trace_claude_code,
    "kimi": _trace_kimi,
}


def extract_thinking_trace(run_dir: Path, scaffold: str | None) -> tuple[list[str], bool]:
    """Return (list of reasoning segments in turn order, has_real_cot).

    has_real_cot is False for codex (only narration is logged) and when no
    trajectory exists, True for claude-code/kimi when thinking blocks present.
    """
    files = _turn_files(run_dir)
    if not files or not scaffold:
        return [], False
    parser = _TRACE_PARSERS.get(scaffold)
    if parser is None:
        return [], False
    segs = parser(files)
    has_cot = bool(segs) and scaffold in ("claude-code", "claude-tmux", "kimi")
    return segs, has_cot
