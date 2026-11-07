"""Rephrase the N best-scoring SFT examples, to augment the SFT set.

The examples are those of the held-out-excluded SFT set (the ten held-out
papers never enter), ranked by test_score. Each gets a faithful rewording of
its whole completion (reasoning + idea), which sft.py --exclude-heldout adds
as an extra example. Codex only, no Tinker. Resumable: re-running skips the
run_ids already reworded and retries the failed ones.

    python rephrase_batch.py [--n 75] [--workers 6] [--data PARQUET]
        [--model gpt-5.5] [--effort medium]

Writes out/rephrased.jsonl: {run_id, task_id, rephrased_completion}. Needs
out/reasoning.jsonl (clean_reasoning_batch.py) and the `codex` CLI on PATH,
logged in.
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from data_prep import HELDOUT_PAPERS, _finite, add_data_argument, sft_examples
from rephrase import DEFAULT_EFFORT, DEFAULT_MODEL, rephrase

_OUT = Path(__file__).resolve().parent / "out"
_CKPT = _OUT / "rephrased.jsonl"


def _load_done() -> set:
    done = set()
    if _CKPT.exists():
        for ln in _CKPT.read_text().splitlines():
            try:
                d = json.loads(ln)
                if d.get("rephrased_completion"):
                    done.add(d["run_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=75, help="how many best to rephrase")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model passed to `codex exec` (published: gpt-5.5; "
                         "default also settable with SW_CODEX_MODEL)")
    ap.add_argument("--effort", default=DEFAULT_EFFORT,
                    help="reasoning effort passed to `codex exec` (published: "
                         "medium; default also settable with SW_CODEX_EFFORT)")
    add_data_argument(ap)
    args = ap.parse_args()
    _OUT.mkdir(exist_ok=True)

    sft = sft_examples(source=args.data, exclude_papers=HELDOUT_PAPERS,
                       n_single_backfill=None)
    scored = [e for e in sft if _finite(e["test_score"])]
    scored.sort(key=lambda e: e["test_score"], reverse=True)
    top = scored[: args.n]
    done = _load_done()
    todo = [e for e in top if e["run_id"] not in done]
    print(f"SFT examples (held-out papers excluded): {len(sft)} | "
          f"rephrasing top {len(top)} by score | "
          f"already done: {len(done)} | to do: {len(todo)} | "
          f"codex model: {args.model} (effort {args.effort})", flush=True)
    if todo and shutil.which("codex") is None:
        raise SystemExit("the `codex` CLI was not found on PATH; install it "
                         "and log in before running this pass")

    def _one(e):
        comp = e["conversation"][2]["content"]
        rw = rephrase(comp, model=args.model, effort=args.effort)
        rec = {"run_id": e["run_id"], "task_id": e["task_id"]}
        if rw:
            rec["rephrased_completion"] = rw
        else:
            rec["error"] = "rephrase_failed"
        return rec

    n_ok = n_fail = 0
    with _CKPT.open("a") as ck, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, e): e["run_id"] for e in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as ex_:
                rec = {"run_id": futs[fut], "error": str(ex_)[:120]}
            ck.write(json.dumps(rec) + "\n")
            ck.flush()
            n_fail += 1 if rec.get("error") else 0
            n_ok += 0 if rec.get("error") else 1
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ok={n_ok} fail={n_fail}", flush=True)
    print(f"DONE: ok={n_ok} fail={n_fail} | checkpoint={_CKPT}", flush=True)


if __name__ == "__main__":
    main()
