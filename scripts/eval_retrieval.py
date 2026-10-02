"""Retrieval evaluation.

  --split dev   : run the full ablation grid on dev (sizes x method x rerank x mode, plus
                  hybrid fusion-weight and rerank-depth variants) and pick the config
                  (page_hit@5, then MRR@10). Writes results/retrieval_dev.csv and
                  results/retrieval_choice.json.
  --split test  : score the dev-chosen config and its same-chunk-size ablation rows on test,
                  once. Writes results/retrieval_test.csv and per-question hits
                  results/retrieval_test_hits.jsonl. Refuses to run without a dev choice.

Also writes routing accuracy to results/routing.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, load_doc_info, path, questions_for  # noqa: E402
from src.evaluation import mean_metrics, retrieval_metrics, routing_accuracy  # noqa: E402
from src.indexing import Index  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.routing import Router  # noqa: E402


MODES = ["shared", "routed:company", "routed:company_year"]


def run_config(ret: Retriever, router: Router, qs, method, rerank, mode, rrf_k=60,
               fusion_depth=100, rerank_depth=30, weights=(1.0, 1.0), keep_hits=False):
    rows, t0 = [], time.perf_counter()
    for q in qs:
        docs = None
        if mode.startswith("routed"):
            level = mode.split(":", 1)[1] if ":" in mode else "company_year"
            r = router.route(q["question"], level=level)
            docs = r.candidates or None
        res = ret.search(q["question"], method, docs, rerank, top_k=10, fusion_depth=fusion_depth,
                         rerank_depth=rerank_depth, rrf_k=rrf_k, weights=weights)
        hits = [(h.doc_name, h.page) for h in res.hits]
        m = retrieval_metrics(hits, q)
        m["id"] = q["financebench_id"]
        if keep_hits:
            m["hits"] = [[h.chunk_id, h.doc_name, h.page, round(h.score, 5)] for h in res.hits]
        rows.append(m)
    agg = mean_metrics(rows)
    agg["sec_per_query"] = (time.perf_counter() - t0) / len(qs)
    return agg, rows


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    a = ap.parse_args()
    qs = questions_for(a.split)
    info = load_doc_info()
    res_dir = path("results")
    res_dir.mkdir(exist_ok=True)

    sizes = cfg["chunking"]["sizes"]
    indexes = {}

    def get_ret(size):
        if size not in indexes:
            indexes[size] = Retriever(Index(path("index") / f"cs{size}"))
        return indexes[size]

    ret0 = get_ret(sizes[0])
    router = Router(info, sorted(set(ret0.index.doc_names)))

    # Routing accuracy for this split
    rp = res_dir / "routing.json"
    routing = json.loads(rp.read_text()) if rp.exists() else {}
    routing[a.split] = {lvl: routing_accuracy(router, qs, lvl) for lvl in ("company", "company_year")}
    rp.write_text(json.dumps(routing, indent=1))
    print("routing", a.split, routing[a.split], flush=True)

    base = dict(rrf_k=60, fusion_depth=100, rerank_depth=30, weights=(1.0, 1.0))
    if a.split == "dev":
        grid = []
        for size, method, rerank, mode in itertools.product(sizes, ["bm25", "dense", "hybrid"],
                                                            [False, True], MODES):
            grid.append(dict(size=size, method=method, rerank=rerank, mode=mode, **base))
        # Hybrid tuning variants (dev only): fusion weights (bm25, dense), rerank depth.
        for size, w, mo in itertools.product(sizes, [(1.0, 2.0), (2.0, 1.0)], MODES[1:]):
            grid.append(dict(size=size, method="hybrid", rerank=False, mode=mo,
                             **{**base, "weights": w}))
            grid.append(dict(size=size, method="hybrid", rerank=True, mode=mo,
                             **{**base, "weights": w}))
        for size, rd, mo in itertools.product(sizes, [10, 20], MODES[1:]):
            grid.append(dict(size=size, method="hybrid", rerank=True, mode=mo,
                             **{**base, "rerank_depth": rd}))
    else:
        cp = res_dir / "retrieval_choice.json"
        if not cp.exists():
            sys.exit("No dev choice found. Run --split dev first.")
        choice = json.loads(cp.read_text())
        c = choice["chosen"]
        s = c["size"]
        grid = [dict(size=s, method=m, rerank=r, mode=mo, **base)
                for m, r, mo in itertools.product(["bm25", "dense", "hybrid"], [False, True],
                                                  MODES)]
        chosen_cfg = dict(size=s, method=c["method"], rerank=c["rerank"], mode=c["mode"],
                          rrf_k=c["rrf_k"], fusion_depth=c["fusion_depth"],
                          rerank_depth=c["rerank_depth"], weights=tuple(c["weights"]))
        if chosen_cfg not in grid:
            grid.append(chosen_cfg)
        # Chunk-size ablation of the chosen method on test, for completeness.
        for s2 in sizes:
            if s2 != s:
                grid.append({**chosen_cfg, "size": s2})

    out_rows = []
    for g in grid:
        ret = get_ret(g["size"])
        is_chosen = a.split == "test" and g == chosen_cfg
        agg, rows = run_config(ret, router, qs, g["method"], g["rerank"], g["mode"], g["rrf_k"],
                               g["fusion_depth"], g["rerank_depth"], g["weights"], keep_hits=is_chosen)
        row = {**g, "weights": list(g["weights"]), **agg, "chosen": is_chosen}
        out_rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        if is_chosen:
            with open(res_dir / "retrieval_test_hits.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

    df = pd.DataFrame(out_rows)
    df.to_csv(res_dir / f"retrieval_{a.split}.csv", index=False)

    if a.split == "dev":
        best = df.sort_values(["page_hit@5", "mrr@10", "sec_per_query"],
                              ascending=[False, False, True]).iloc[0]
        chosen = {k: (best[k].item() if hasattr(best[k], "item") else best[k])
                  for k in ["size", "method", "rerank", "mode", "rrf_k", "fusion_depth",
                            "rerank_depth", "weights"]}
        choice = {"criterion": "max dev page_hit@5, then MRR@10, then speed",
                  "chosen": chosen, "dev_metrics": {k: float(best[k]) for k in df.columns
                                                     if k.startswith(("page_hit", "doc_hit", "mrr"))}}
        (res_dir / "retrieval_choice.json").write_text(json.dumps(choice, indent=1))
        print("CHOSEN", choice)


if __name__ == "__main__":
    main()
