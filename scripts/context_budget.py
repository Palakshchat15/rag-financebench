"""Context budget (DEV only): evidence-page recall vs. prompt size.

For each chunk size, run the dev-chosen retrieval (hybrid RRF, weights from config.yaml,
routed company_year, no rerank) and record, for k = 1..12:
  page_hit@k    share of dev questions whose gold evidence page is among the top-k chunks
  ctx_tokens@k  mean tokens (cl100k, as stored in the index) of the top-k chunks put in the prompt
Writes results/context_budget_dev.csv. No LLM calls.

Usage: .venv\\Scripts\\python scripts\\context_budget.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, load_doc_info, path, questions_for  # noqa: E402
from src.evaluation import gold_pages  # noqa: E402
from src.indexing import Index  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.routing import Router  # noqa: E402

KMAX = 12


def main() -> None:
    cfg = load_config()
    rc = cfg["retrieval"]
    qs = questions_for("dev")
    rows = []
    router = None
    for size in cfg["chunking"]["sizes"]:
        ret = Retriever(Index(path("index") / f"cs{size}"))
        ntok = dict(zip(ret.index.chunks.chunk_id, ret.index.chunks.n_tokens))
        if router is None:
            router = Router(load_doc_info(), sorted(set(ret.index.doc_names)))
        hit = np.zeros((len(qs), KMAX))
        tok = np.zeros((len(qs), KMAX))
        for qi, q in enumerate(qs):
            cands = router.route(q["question"], level=rc["route_level"]).candidates or None
            res = ret.search(q["question"], rc["method"], cands, False, KMAX, rc["fusion_depth"],
                             rc["rerank_depth"], rc["rrf_k"], tuple(rc["weights"]))
            gp = gold_pages(q)
            found, total = False, 0
            for k in range(KMAX):
                if k < len(res.hits):
                    h = res.hits[k]
                    found = found or (h.doc_name, h.page) in gp
                    total += int(ntok[h.chunk_id])
                hit[qi, k], tok[qi, k] = found, total
        for k in range(KMAX):
            rows.append({"chunk_size": size, "k": k + 1, "page_hit": hit[:, k].mean(),
                         "ctx_tokens_mean": tok[:, k].mean(), "ctx_tokens_p95": np.percentile(tok[:, k], 95)})
        print(f"cs{size} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path("results") / "context_budget_dev.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
