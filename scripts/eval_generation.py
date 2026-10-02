"""Generation eval with per-question JSONL caching (resumable).

Usage:
  .venv\\Scripts\\python scripts\\eval_generation.py --split test --config rag --model qwen2.5:7b
  configs: closed_book | oracle | rag     (rag uses the dev-chosen retrieval in config.yaml)
Output: results/gen/{split}__{config}__{model}.jsonl  (one line per question; re-running skips done ids)
Answers are also traced to data/traces.db with source='eval'.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import path, questions_for  # noqa: E402
from src.evaluation import append_jsonl, read_jsonl  # noqa: E402
from src.generation import Pipeline  # noqa: E402
from src.llm import check_ollama  # noqa: E402
from src.llm_lock import llm_lock  # noqa: E402


def out_path(split: str, config: str, model: str, variant: str = "") -> Path:
    tag = f"{config}-{variant}" if variant else config
    return path("results") / "gen" / f"{split}__{tag}__{model.replace(':', '-')}.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--config", choices=["closed_book", "oracle", "rag"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated financebench ids (dev sanity checks)")
    ap.add_argument("--chunk-size", type=int, default=None, help="override config (dev experiments only)")
    ap.add_argument("--top-k", type=int, default=None, help="override config (dev experiments only)")
    a = ap.parse_args()
    variant = ""
    if a.chunk_size or a.top_k:
        if a.split != "dev":
            sys.exit("Retrieval overrides are for dev experiments only; test uses config.yaml.")
        variant = f"cs{a.chunk_size or 'cfg'}k{a.top_k or 'cfg'}"

    check_ollama(a.model)
    qs = questions_for(a.split)[: a.limit]
    if a.ids:
        want = set(a.ids.split(","))
        qs = [q for q in qs if q["financebench_id"] in want]
    op = out_path(a.split, a.config, a.model, variant)
    done = {r["id"] for r in read_jsonl(op)}
    todo = [q for q in qs if q["financebench_id"] not in done]
    print(f"{op.name}: {len(done)} cached, {len(todo)} to run", flush=True)
    if not todo:
        return
    pipe = Pipeline(chunk_size=a.chunk_size, top_k=a.top_k, trace_source="eval")
    with llm_lock(f"eval_generation {op.name}"):
        run(a, pipe, todo, op)


def run(a, pipe, todo, op) -> None:
    t_all = time.time()
    for n, q in enumerate(todo, 1):
        ans = pipe.answer(q["question"], a.model, a.config, evidence=q["evidence"])
        append_jsonl(op, {
            "id": q["financebench_id"], "split": a.split, "config": a.config, "model": a.model,
            "mode": ans.mode, "chunk_size": pipe.chunk_size if a.config == "rag" else None,
            "top_k": pipe.rc["top_k"] if a.config == "rag" else None,
            "question": q["question"], "answer": ans.answer,
            "context": [[b.chunk_id, b.doc_name, b.page, round(b.score, 5)] for b in ans.blocks],
            "route_candidates": ans.route_candidates,
            "citations": [list(c) for c in ans.citations],
            "invalid_citations": [list(c) for c in ans.invalid_citations],
            "timings": {k: round(v, 3) for k, v in ans.timings.items()},
            "prompt_tokens": ans.prompt_tokens, "completion_tokens": ans.completion_tokens,
            "num_ctx": ans.num_ctx, "trace_id": ans.trace_id,
        })
        el = time.time() - t_all
        print(f"[{n}/{len(todo)}] {q['financebench_id']} {ans.timings['total_s']:.0f}s "
              f"(ptok {ans.prompt_tokens}, ctok {ans.completion_tokens}) eta {el / n * (len(todo) - n) / 60:.0f} min",
              flush=True)


if __name__ == "__main__":
    main()
