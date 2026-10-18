"""Compare the dense-retrieval backends (numpy | chroma | faiss_flat | faiss_hnsw) on the dev and
test retrieval questions. Read-only with respect to every existing result: it uses the
config.yaml retrieval settings (chunk size, RRF weights, fusion depth, route level), no reranker,
and writes only results/vector_backends_*.csv.

For each split x backend x method (dense, hybrid) x mode (routed company_year, shared):
  page_hit@1/3/5/10, doc_hit@k, MRR@10           (src/evaluation.py, as eval_retrieval.py)
  overlap@10        |top-10 chunks  n  NumPy top-10| / |NumPy top-10|   (agreement with exact)
  dense_overlap@D   same for the D = fusion_depth dense candidates before fusion
  dense_ms p50/p95  vector search only (query embedding precomputed once and shared)
  retrieve_ms p50/p95  whole Retriever.search (BM25 + dense + RRF), embedding excluded
Plus results/vector_backends_build.csv: build time (from the build-info JSON), size on disk and
load time per backend.

Usage:  .venv\\Scripts\\python scripts\\bench_vector_backends.py [--splits dev test]
            [--backends numpy chroma faiss_flat faiss_hnsw] [--size 256]
Needs:  scripts\\build_vector_index.py run first for chroma / faiss. Backends whose package or
        index is missing are skipped with a message; numpy (the reference) always runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, load_doc_info, path, questions_for  # noqa: E402
from src.evaluation import mean_metrics, retrieval_metrics  # noqa: E402
from src.indexing import Index, embed_query  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.routing import Router  # noqa: E402
from src.vector_backends import (DENSE_BACKENDS, FAISS_KINDS, StaleIndexError,  # noqa: E402
                                 backend_dirs, dir_size_bytes, load_backend)

METHODS = ["dense", "hybrid"]


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def build_row(name: str, index: Index, size_dir: str, load_s: float) -> dict:
    dirs = backend_dirs(size_dir)
    row = {"backend": name, "chunk_size": int(size_dir[2:]), "n_vectors": len(index),
           "dim": int(index.emb.shape[1]), "load_s": round(load_s, 3)}
    if name == "numpy":
        row.update(build_s=None, disk_mb=dir_size_bytes(index.dir / "emb.npy") / 2**20,
                   params="", note="emb.npy from build_index.py (embedding time not counted)")
        return row
    if name == "chroma":
        info_p, files = dirs["chroma"] / "build_info.json", [dirs["chroma"]]
    else:
        kind = FAISS_KINDS[name]
        info_p = dirs["faiss"] / f"{kind}.json"
        files = [dirs["faiss"] / f"{kind}.faiss", dirs["faiss"] / "ids.json"]
    info = json.loads(info_p.read_text()) if info_p.exists() else {}
    row.update(build_s=info.get("build_s"), disk_mb=sum(dir_size_bytes(f) for f in files) / 2**20,
               params=json.dumps(info.get("params", {})), note=f"built {info.get('built_at', '?')}")
    return row


def main() -> None:
    cfg = load_config()
    rc = cfg["retrieval"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["dev", "test"], choices=["dev", "test"])
    ap.add_argument("--backends", nargs="*", default=list(DENSE_BACKENDS), choices=list(DENSE_BACKENDS))
    ap.add_argument("--size", type=int, default=rc["chunk_size"])
    a = ap.parse_args()
    backends = ["numpy"] + [b for b in a.backends if b != "numpy"]   # reference first
    size_dir = f"cs{a.size}"
    level = rc.get("route_level", "company_year")
    modes = [f"routed:{level}", "shared"]
    depth, weights = rc["fusion_depth"], tuple(rc["weights"])
    res_dir = path("results")
    res_dir.mkdir(exist_ok=True)

    index = Index(path("index") / size_dir)
    router = Router(load_doc_info(), sorted(set(index.doc_names)))
    print(f"{size_dir}: {len(index)} chunks; settings: weights {weights}, fusion_depth {depth}, "
          f"rrf_k {rc['rrf_k']}, route {level}", flush=True)

    # Questions, routes and query embeddings once, shared by every backend.
    qs = {s: questions_for(s) for s in a.splits}
    docs = {s: {m: [(router.route(q["question"], level=level).candidates or None) if m != "shared" else None
                    for q in qs[s]] for m in modes} for s in a.splits}
    qvec, embed_ms = {}, []
    embed_query("warm up the embedder")
    for s in a.splits:
        qvec[s] = []
        for q in qs[s]:
            t0 = time.perf_counter()
            qvec[s].append(embed_query(q["question"]))
            embed_ms.append((time.perf_counter() - t0) * 1000)
    print(f"query embedding p50 {pct(embed_ms, 50):.1f} ms", flush=True)

    ref: dict = {}          # (split, method, mode) -> [top-10 chunk ids]; (split, mode) -> dense rows
    rows = {s: [] for s in a.splits}
    build_rows = []
    for name in backends:
        try:
            t0 = time.perf_counter()
            backend = load_backend(name, index)
            ret = Retriever(index, "numpy" if name == "numpy" else backend)
            load_s = time.perf_counter() - t0
        except (ImportError, FileNotFoundError, StaleIndexError) as e:
            print(f"SKIP {name}: {e}", flush=True)
            continue
        build_rows.append(build_row(name, index, size_dir, load_s))
        backend.search(qvec[a.splits[0]][0], depth)                  # warm-up, not timed
        for s in a.splits:
            for mode in modes:
                dense_ms, dense_ov = [], []
                dense_rows = []
                for qi in range(len(qs[s])):
                    t0 = time.perf_counter()
                    cand = backend.search(qvec[s][qi], depth, docs[s][mode][qi])
                    dense_ms.append((time.perf_counter() - t0) * 1000)
                    dense_rows.append([r for r, _ in cand])
                if name == "numpy":
                    ref[(s, mode)] = dense_rows
                for mine, exact in zip(dense_rows, ref[(s, mode)]):
                    dense_ov.append(len(set(mine) & set(exact)) / max(1, len(exact)))
                for method in METHODS:
                    per_q, ret_ms, top10s, ov = [], [], [], []
                    for qi, q in enumerate(qs[s]):
                        t0 = time.perf_counter()
                        res = ret.search(q["question"], method, docs[s][mode][qi], False, top_k=10,
                                         fusion_depth=depth, rerank_depth=rc["rerank_depth"],
                                         rrf_k=rc["rrf_k"], weights=weights, qvec=qvec[s][qi])
                        ret_ms.append((time.perf_counter() - t0) * 1000)
                        per_q.append(retrieval_metrics([(h.doc_name, h.page) for h in res.hits], q))
                        top10s.append([h.chunk_id for h in res.hits])
                    if name == "numpy":
                        ref[(s, method, mode)] = top10s
                    for mine, exact in zip(top10s, ref[(s, method, mode)]):
                        ov.append(len(set(mine) & set(exact)) / max(1, len(exact)))
                    row = {"split": s, "chunk_size": a.size, "backend": name, "method": method,
                           "mode": mode, "n": len(qs[s]), **mean_metrics(per_q),
                           "overlap@10": float(np.mean(ov)),
                           f"dense_overlap@{depth}": float(np.mean(dense_ov)),
                           "dense_ms_p50": pct(dense_ms, 50), "dense_ms_p95": pct(dense_ms, 95),
                           "retrieve_ms_p50": pct(ret_ms, 50), "retrieve_ms_p95": pct(ret_ms, 95),
                           "embed_ms_p50": pct(embed_ms, 50)}
                    rows[s].append(row)
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
                          flush=True)
        del ret, backend

    for s in a.splits:
        pd.DataFrame(rows[s]).to_csv(res_dir / f"vector_backends_{s}.csv", index=False)
    pd.DataFrame(build_rows).to_csv(res_dir / "vector_backends_build.csv", index=False)
    print("wrote", ", ".join(f"results/vector_backends_{s}.csv" for s in a.splits),
          "and results/vector_backends_build.csv")


if __name__ == "__main__":
    main()
