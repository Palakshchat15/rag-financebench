"""Evaluation helpers: retrieval metrics, routing accuracy, generation aggregates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KS = (1, 3, 5, 10)


def gold_pages(q: dict) -> set[tuple[str, int]]:
    return {(e["doc_name"], int(e["evidence_page_num"])) for e in q["evidence"]}


def gold_docs(q: dict) -> set[str]:
    return {e["doc_name"] for e in q["evidence"]} | {q["doc_name"]}


def retrieval_metrics(hits: list[tuple[str, int]], q: dict) -> dict:
    """hits: ranked (doc_name, page) of retrieved chunks."""
    gp, gd = gold_pages(q), gold_docs(q)
    out = {}
    first = next((r for r, h in enumerate(hits, 1) if h in gp), None)
    for k in KS:
        out[f"page_hit@{k}"] = float(any(h in gp for h in hits[:k]))
        out[f"doc_hit@{k}"] = float(any(h[0] in gd for h in hits[:k]))
    out["mrr@10"] = 1.0 / first if first and first <= 10 else 0.0
    return out


def mean_metrics(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k.startswith(("page_hit", "doc_hit", "mrr"))]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def routing_accuracy(router, qs: list[dict], level: str = "company_year") -> dict:
    n = len(qs)
    in_cands = exact = fallback = 0
    n_cands = []
    for q in qs:
        r = router.route(q["question"], level=level)
        fallback += r.fallback_shared
        in_cands += q["doc_name"] in r.candidates
        exact += r.candidates == [q["doc_name"]]
        n_cands.append(len(r.candidates))
    return {"n": n, "gold_doc_in_candidates": in_cands / n, "exact_single_doc": exact / n,
            "fallback_to_shared": fallback / n, "mean_candidates": float(np.mean(n_cands))}


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(p: Path, row: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
