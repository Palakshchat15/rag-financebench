"""BM25, dense and hybrid (RRF) retrieval with optional cross-encoder reranking."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import load_config
from .indexing import Index


def rrf(rankings: list[list], k: int = 60, weights: list[float] | None = None) -> list[tuple]:
    """Reciprocal rank fusion. rankings: lists of ids, best first. Returns [(id, score)] sorted
    by fused score desc; ties broken by first appearance order for determinism."""
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    scores: dict = {}
    order: dict = {}
    for w, ranking in zip(weights, rankings):
        for r, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + r)
            order.setdefault(doc, len(order))
    return sorted(scores.items(), key=lambda x: (-x[1], order[x[0]]))


@dataclass
class Hit:
    chunk_id: str
    doc_name: str
    page: int
    text: str
    score: float
    rank: int


@dataclass
class RetrievalResult:
    hits: list[Hit]
    timings: dict = field(default_factory=dict)


_RERANKERS: dict = {}


def get_reranker(model: str | None = None):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    model = model or load_config()["reranker"]["model"]
    if model not in _RERANKERS:
        _RERANKERS[model] = TextCrossEncoder(model_name=model)
    return _RERANKERS[model]


def _topn(scores: np.ndarray, mask: np.ndarray | None, n: int) -> np.ndarray:
    s = scores if mask is None else np.where(mask, scores, -np.inf)
    n = min(n, int(np.isfinite(s).sum()))
    if n <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-s, n - 1)[:n]
    return idx[np.argsort(-s[idx], kind="stable")]


class Retriever:
    """dense_backend: "numpy" (default; exact, every reported result) | "chroma" | "faiss_flat" |
    "faiss_hnsw", or a ready DenseBackend object (tests). None = config.yaml
    retrieval.dense_backend. Only the dense ranking changes; BM25 and RRF are shared."""

    def __init__(self, index: Index, dense_backend=None):
        self.index = index
        self._rr_cache: dict = {}
        if dense_backend is None:
            dense_backend = load_config()["retrieval"].get("dense_backend", "numpy")
        self._backend = None
        if isinstance(dense_backend, str):
            self.dense_backend = dense_backend
            if dense_backend != "numpy":
                from .vector_backends import load_backend

                self._backend = load_backend(dense_backend, index)
        else:
            self._backend = dense_backend
            self.dense_backend = getattr(dense_backend, "name", type(dense_backend).__name__)

    def _dense_rank(self, query: str, qvec, mask, docs, n: int) -> list[tuple[int, float]]:
        """Top-n dense candidates [(row, score)]. qvec = precomputed query embedding (optional)."""
        if self._backend is None:      # original NumPy path, unchanged
            s = self.index.dense_scores(query) if qvec is None else self.index.emb @ qvec
            idx = _topn(s, mask, n)
            return [(int(i), float(s[i])) for i in idx]
        if qvec is None:
            from .indexing import embed_query

            qvec = embed_query(query)
        return self._backend.search(qvec, n, docs)

    def doc_mask(self, docs: list[str] | None) -> np.ndarray | None:
        if not docs:
            return None
        return np.isin(self.index.doc_names, list(docs))

    def search(self, query: str, method: str = "hybrid", docs: list[str] | None = None,
               rerank: bool = False, top_k: int = 10, fusion_depth: int = 100,
               rerank_depth: int = 30, rrf_k: int = 60,
               weights: tuple[float, float] = (1.0, 1.0), qvec=None) -> RetrievalResult:
        t: dict = {}
        mask = self.doc_mask(docs)
        t0 = time.perf_counter()
        depth = max(top_k, rerank_depth if rerank else top_k, fusion_depth if method == "hybrid" else 0)
        if method == "bm25":
            s = self.index.bm25_scores(query)
            idx = _topn(s, mask, depth)
            cand = [(int(i), float(s[i])) for i in idx]
        elif method == "dense":
            cand = self._dense_rank(query, qvec, mask, docs, depth)
        elif method == "hybrid":
            sb = self.index.bm25_scores(query)
            rb = list(_topn(sb, mask, fusion_depth))
            rd = [i for i, _ in self._dense_rank(query, qvec, mask, docs, fusion_depth)]
            fused = rrf([[int(i) for i in rb], [int(i) for i in rd]], k=rrf_k, weights=list(weights))
            cand = [(i, s_) for i, s_ in fused[:depth]]
        else:
            raise ValueError(f"unknown method {method}")
        t["retrieve_s"] = time.perf_counter() - t0

        if rerank and cand:
            t0 = time.perf_counter()
            pool = cand[:rerank_depth]
            # Cross-encoder scores are deterministic per (query, chunk): cache them so the eval
            # grid (and repeated app queries) never rescore the same pair.
            todo = [i for i, _ in pool if (query, i) not in self._rr_cache]
            if todo:
                texts = [self.index.chunks.text.iat[i] for i in todo]
                for i, s_ in zip(todo, get_reranker().rerank(query, texts)):
                    self._rr_cache[(query, i)] = float(s_)
            scores = [self._rr_cache[(query, i)] for i, _ in pool]
            order = sorted(range(len(pool)), key=lambda j: (-scores[j], j))
            cand = [(pool[j][0], float(scores[j])) for j in order]
            t["rerank_s"] = time.perf_counter() - t0

        ch = self.index.chunks
        hits = [Hit(ch.chunk_id.iat[i], ch.doc_name.iat[i], int(ch.page.iat[i]), ch.text.iat[i], sc, r)
                for r, (i, sc) in enumerate(cand[:top_k], start=1)]
        return RetrievalResult(hits, t)
