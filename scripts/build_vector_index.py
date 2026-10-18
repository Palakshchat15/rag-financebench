"""Build the optional vector-DB dense backends from the EXISTING chunk embeddings.

Never re-embeds: reads data/index/cs{size}/chunks.parquet + emb.npy written by build_index.py.
Writes data/index/chroma/cs{size}/ and data/index/faiss/cs{size}/ (gitignored), each with a
build-info JSON (build time, parameters, embedding fingerprint). Skips backends that are already
built unless --force.

Usage:  .venv\\Scripts\\python scripts\\build_vector_index.py [--sizes 256] \\
            [--backends chroma faiss_flat faiss_hnsw] [--force]
Needs:  .venv\\Scripts\\python -m pip install -r requirements-vectordb.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, load_doc_info, path  # noqa: E402
from src.vector_backends import DENSE_BACKENDS, build_backend  # noqa: E402


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=[cfg["retrieval"]["chunk_size"]])
    ap.add_argument("--backends", nargs="*", default=list(DENSE_BACKENDS[1:]),
                    choices=list(DENSE_BACKENDS[1:]))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    info = load_doc_info()

    for size in a.sizes:
        src = path("index") / f"cs{size}"
        if not (src / "emb.npy").exists():
            sys.exit(f"{src / 'emb.npy'} not found: run scripts\\build_index.py first (this script never embeds)")
        chunks = pd.read_parquet(src / "chunks.parquet", columns=["chunk_id", "doc_name", "page"])
        emb = np.load(src / "emb.npy")
        if len(chunks) != len(emb):
            sys.exit(f"cs{size}: {len(chunks)} chunks vs {len(emb)} embeddings")
        print(f"cs{size}: {len(emb)} vectors, dim {emb.shape[1]}", flush=True)
        for b in a.backends:
            bi = build_backend(b, f"cs{size}", emb, chunks, info, force=a.force)
            print(f"cs{size} {b}: {bi['n_vectors']} vectors, build {bi['build_s']:.1f}s "
                  f"(built {bi['built_at']})", flush=True)


if __name__ == "__main__":
    main()
