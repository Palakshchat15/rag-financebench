"""Build and load file-based indexes: BM25 (bm25s) + dense (fastembed ONNX, numpy matrix).

Layout per chunk size:  data/index/cs{size}/
    chunks.parquet   chunk_id, doc_name, page, idx, text, n_tokens
    emb.npy          float32 [n_chunks, dim], L2-normalised
    bm25/            bm25s index
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, load_doc_info

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def doc_header(doc_name: str, page: int, info: dict | None) -> str:
    """Short metadata header prepended to the *indexed* text (not the shown text), so a chunk
    of a table page still carries its company / year / filing type."""
    if info:
        return f"{info['company']} {info['doc_type']} {info['doc_period']} | {doc_name} page {page}\n"
    return f"{doc_name} page {page}\n"


def bm25_tokenize(texts: list[str], show_progress: bool = False):
    import bm25s
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    return bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=show_progress)


_EMBEDDERS: dict = {}


def get_embedder(model: str | None = None):
    from fastembed import TextEmbedding

    cfg = load_config()
    model = model or cfg["embedding"]["model"]
    if model not in _EMBEDDERS:
        _EMBEDDERS[model] = TextEmbedding(model_name=model)
    return _EMBEDDERS[model]


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    emb = get_embedder()
    vecs = np.asarray(list(emb.embed(texts, batch_size=batch_size)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs


def embed_query(q: str) -> np.ndarray:
    prefix = BGE_QUERY_PREFIX if "bge" in load_config()["embedding"]["model"].lower() else ""
    return embed_texts([prefix + q])[0]


def build_index(chunks_df: pd.DataFrame, out_dir: Path, batch_size: int = 64) -> None:
    import bm25s

    out_dir.mkdir(parents=True, exist_ok=True)
    info = load_doc_info()
    index_texts = [doc_header(d, int(p), info.get(d)) + t
                   for d, p, t in zip(chunks_df.doc_name, chunks_df.page, chunks_df.text)]
    chunks_df.to_parquet(out_dir / "chunks.parquet", index=False)

    t0 = time.time()
    retriever = bm25s.BM25()
    retriever.index(bm25_tokenize(index_texts, show_progress=False), show_progress=False)
    retriever.save(str(out_dir / "bm25"))
    print(f"  bm25 built in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    parts = []
    step = 2048
    for s in range(0, len(index_texts), step):
        parts.append(embed_texts(index_texts[s:s + step], batch_size))
        print(f"  embedded {min(s + step, len(index_texts))}/{len(index_texts)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    np.save(out_dir / "emb.npy", np.concatenate(parts))


class Index:
    """In-memory view of one chunk-size index."""

    def __init__(self, index_dir: Path):
        import bm25s

        self.dir = index_dir
        self.chunks = pd.read_parquet(index_dir / "chunks.parquet")
        self.emb = np.load(index_dir / "emb.npy")
        self.bm25 = bm25s.BM25.load(str(index_dir / "bm25"))
        self.chunk_ids = self.chunks.chunk_id.to_numpy()
        self.doc_names = self.chunks.doc_name.to_numpy()
        self.pages = self.chunks.page.to_numpy()

    def __len__(self) -> int:
        return len(self.chunks)

    def bm25_scores(self, query: str) -> np.ndarray:
        import bm25s
        import Stemmer

        words = bm25s.tokenize([query], stopwords="en", stemmer=Stemmer.Stemmer("english"),
                               return_ids=False, show_progress=False)[0]
        words = [w for w in words if w in self.bm25.vocab_dict]
        if not words:
            return np.zeros(len(self), dtype=np.float32)
        return np.asarray(self.bm25.get_scores(words), dtype=np.float32)

    def dense_scores(self, query: str) -> np.ndarray:
        return self.emb @ embed_query(query)
