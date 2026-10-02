"""Parse PDFs -> data/processed/pages.parquet, then chunk + index each chunk size.

Usage:  .venv\\Scripts\\python scripts\\build_index.py [--sizes 256 512 1024] [--force]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.chunking import TiktokenTokenizer, chunk_pages  # noqa: E402
from src.config import load_config, path  # noqa: E402
from src.indexing import build_index  # noqa: E402
from src.ingest import parse_all  # noqa: E402


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=cfg["chunking"]["sizes"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    pages_path = path("processed") / "pages.parquet"
    if pages_path.exists() and not a.force:
        pages = pd.read_parquet(pages_path)
    else:
        t0 = time.time()
        pages = parse_all(path("pdfs"), pages_path)
        print(f"parsed {len(pages)} pages from {pages.doc_name.nunique()} PDFs in {time.time()-t0:.0f}s")
    empty = (pages.text.str.strip().str.len() == 0).sum()
    print(f"pages: {len(pages)}  (empty text: {empty})")

    tok = TiktokenTokenizer(cfg["chunking"]["tokenizer"])
    for size in a.sizes:
        out = path("index") / f"cs{size}"
        if (out / "emb.npy").exists() and not a.force:
            print(f"cs{size}: exists, skipping")
            continue
        overlap = int(size * cfg["chunking"]["overlap_frac"])
        t0 = time.time()
        chunks = chunk_pages(pages.to_dict("records"), size, overlap, tok)
        df = pd.DataFrame([c.__dict__ for c in chunks])
        print(f"cs{size}: {len(df)} chunks (overlap {overlap}) in {time.time()-t0:.0f}s", flush=True)
        build_index(df, out, cfg["embedding"]["batch_size"])
        print(f"cs{size}: done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
