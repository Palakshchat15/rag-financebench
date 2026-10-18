"""Dense backends on tiny synthetic vectors. The chromadb / faiss tests skip cleanly when the
optional packages (requirements-vectordb.txt) are not installed."""
import numpy as np
import pandas as pd
import pytest

from src.retrieval import Retriever, _topn
from src.vector_backends import (FaissBackend, NumpyBackend, StaleIndexError, chroma_where,
                                 chunk_metadata, emb_fingerprint)

N, D = 240, 16
DOCS = ["A_2020_10K", "A_2021_10K", "B_2021_10K", "C_2022_10Q"]
INFO = {"A_2020_10K": {"company": "A", "doc_type": "10k", "doc_period": 2020},
        "A_2021_10K": {"company": "A", "doc_type": "10k", "doc_period": 2021},
        "B_2021_10K": {"company": "B", "doc_type": "10k", "doc_period": 2021},
        "C_2022_10Q": {"company": "C", "doc_type": "10q", "doc_period": 2022}}


def _data(seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(N, D)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    doc_names = np.array([DOCS[i % len(DOCS)] for i in range(N)])
    pages = np.arange(N) // 8
    chunk_ids = np.array([f"{d}|p{p}|c{i}" for i, (d, p) in enumerate(zip(doc_names, pages))])
    q = rng.normal(size=D).astype(np.float32)
    q /= np.linalg.norm(q)
    return emb, chunk_ids, doc_names, pages, q


def _exact(emb, doc_names, q, k, docs=None):
    s = emb @ q
    mask = None if not docs else np.isin(doc_names, docs)
    return [int(i) for i in _topn(s, mask, k)]


class FakeIndex:
    """Just enough of src.indexing.Index for Retriever (no BM25 / embedder files)."""

    def __init__(self, emb, chunk_ids, doc_names, pages, seed=1):
        self.emb, self.chunk_ids, self.doc_names, self.pages = emb, chunk_ids, doc_names, pages
        self.chunks = pd.DataFrame({"chunk_id": chunk_ids, "doc_name": doc_names, "page": pages,
                                    "text": [f"t{i}" for i in range(len(chunk_ids))]})
        self._bm25 = np.random.default_rng(seed).random(len(chunk_ids)).astype(np.float32)

    def __len__(self):
        return len(self.chunk_ids)

    def bm25_scores(self, query):
        return self._bm25


# ---------------------------------------------------------------- no optional packages needed
def test_numpy_backend_matches_exact_and_filters():
    emb, ids, docs, _, q = _data()
    nb = NumpyBackend(emb, ids, docs)
    got = nb.search(q, 10)
    assert [r for r, _ in got] == _exact(emb, docs, q, 10)
    assert got[0][1] == pytest.approx(float(emb[got[0][0]] @ q))
    sub = nb.search(q, 500, ["B_2021_10K"])          # k capped at the 60 B chunks
    assert len(sub) == N // 4 and {docs[r] for r, _ in sub} == {"B_2021_10K"}
    assert nb.search(q, 5, ["NOT_A_DOC"]) == []


def test_allowed_rows_and_counts():
    emb, ids, docs, _, _ = _data()
    nb = NumpyBackend(emb, ids, docs)
    assert nb.allowed_rows(None) is None and nb.n_allowed(None) == N
    rows = nb.allowed_rows(["A_2020_10K", "C_2022_10Q", "A_2020_10K"])
    assert len(rows) == N // 2 and set(docs[rows]) == {"A_2020_10K", "C_2022_10Q"}


def test_chroma_where_filters():
    assert chroma_where() is None
    assert chroma_where(docs=["X", "X"]) == {"doc_name": {"$in": ["X"]}}
    w = chroma_where(docs=["X"], companies=["3M"], years=[2018, 2022])
    assert w == {"$and": [{"doc_name": {"$in": ["X"]}}, {"company": {"$in": ["3M"]}},
                          {"year": {"$in": [2018, 2022]}}]}


def test_chunk_metadata_has_no_none():
    _, ids, docs, pages, _ = _data()
    meta = chunk_metadata(ids, docs, pages, {k: v for k, v in INFO.items() if k != "C_2022_10Q"})
    assert meta[0] == {"row": 0, "doc_name": "A_2020_10K", "page": 0, "company": "A", "year": 2020,
                       "doc_type": "10k"}
    assert meta[3]["company"] == "" and meta[3]["year"] == -1        # unknown doc -> defaults
    assert all(v is not None for m in meta for v in m.values())


def test_fingerprint_changes_with_content():
    emb, *_ = _data()
    e2 = emb.copy()
    e2[0, 0] += 1e-3
    assert emb_fingerprint(emb) == emb_fingerprint(emb.copy()) != emb_fingerprint(e2)


@pytest.mark.parametrize("method", ["dense", "hybrid"])
@pytest.mark.parametrize("docs", [None, ["A_2021_10K", "B_2021_10K"]])
def test_retriever_backend_object_equals_default_numpy_path(method, docs):
    emb, ids, dn, pages, q = _data()
    idx = FakeIndex(emb, ids, dn, pages)
    base = Retriever(idx, "numpy").search("q", method, docs, top_k=10, fusion_depth=50,
                                          weights=(1.0, 2.0), qvec=q)
    alt = Retriever(idx, NumpyBackend(emb, ids, dn)).search("q", method, docs, top_k=10,
                                                            fusion_depth=50, weights=(1.0, 2.0), qvec=q)
    assert [h.chunk_id for h in base.hits] == [h.chunk_id for h in alt.hits]
    assert [h.score for h in base.hits] == pytest.approx([h.score for h in alt.hits])
    if docs:
        assert {h.doc_name for h in base.hits} <= set(docs)


def test_unknown_backend_name():
    from src.vector_backends import load_backend

    emb, ids, dn, pages, _ = _data()
    with pytest.raises(ValueError):
        load_backend("pinecone", FakeIndex(emb, ids, dn, pages))


# ---------------------------------------------------------------- FAISS (optional)
def test_faiss_flat_is_exact_with_and_without_filter(tmp_path):
    pytest.importorskip("faiss")
    emb, ids, dn, _, q = _data()
    FaissBackend.build(tmp_path, "flat", emb, ids)
    fb = FaissBackend.load(tmp_path, "flat", ids, dn, fingerprint=emb_fingerprint(emb))
    assert [r for r, _ in fb.search(q, 10)] == _exact(emb, dn, q, 10)
    sub = ["A_2020_10K", "C_2022_10Q"]
    got = fb.search(q, 15, sub)
    assert [r for r, _ in got] == _exact(emb, dn, q, 15, sub)
    assert got[0][1] == pytest.approx(float(emb[got[0][0]] @ q), abs=1e-5)
    assert len(fb.search(q, 1000, ["B_2021_10K"])) == N // 4
    assert fb.search(q, 5, ["NOT_A_DOC"]) == []


def test_faiss_hnsw_recall_and_filter(tmp_path):
    pytest.importorskip("faiss")
    emb, ids, dn, _, q = _data()
    FaissBackend.build(tmp_path, "hnsw", emb, ids, {"M": 16, "ef_construction": 100})
    fb = FaissBackend.load(tmp_path, "hnsw", ids, dn, ef_search=64)
    got = [r for r, _ in fb.search(q, 10)]
    assert len(set(got) & set(_exact(emb, dn, q, 10))) >= 9
    sub = ["B_2021_10K"]
    got = fb.search(q, 10, sub)
    assert got and {dn[r] for r, _ in got} == set(sub)


def test_faiss_stale_ids_and_fingerprint(tmp_path):
    pytest.importorskip("faiss")
    emb, ids, dn, _, _ = _data()
    FaissBackend.build(tmp_path, "flat", emb, ids)
    with pytest.raises(StaleIndexError):
        FaissBackend.load(tmp_path, "flat", ids[::-1], dn[::-1])
    with pytest.raises(StaleIndexError):
        FaissBackend.load(tmp_path, "flat", ids, dn, fingerprint="other")


def test_faiss_rejects_unnormalised(tmp_path):
    pytest.importorskip("faiss")
    emb, ids, *_ = _data()
    with pytest.raises(ValueError):
        FaissBackend.build(tmp_path, "flat", emb * 3, ids)


def test_retriever_with_faiss_flat_matches_numpy(tmp_path):
    pytest.importorskip("faiss")
    emb, ids, dn, pages, q = _data()
    FaissBackend.build(tmp_path, "flat", emb, ids)
    fb = FaissBackend.load(tmp_path, "flat", ids, dn)
    idx = FakeIndex(emb, ids, dn, pages)
    for docs in (None, ["A_2021_10K"]):
        a = Retriever(idx, "numpy").search("q", "hybrid", docs, fusion_depth=40, qvec=q)
        b = Retriever(idx, fb).search("q", "hybrid", docs, fusion_depth=40, qvec=q)
        assert [h.chunk_id for h in a.hits] == [h.chunk_id for h in b.hits]


# ---------------------------------------------------------------- Chroma (optional)
@pytest.fixture
def chroma_backend(tmp_path):
    pytest.importorskip("chromadb")
    from src.vector_backends import ChromaBackend

    emb, ids, dn, pages, q = _data()
    meta = chunk_metadata(ids, dn, pages, INFO)
    info = ChromaBackend.build(tmp_path, "test_cs", emb, list(ids), meta,
                               {"ef_construction": 200, "ef_search": 200, "batch_size": 100})
    assert info["n_vectors"] == N
    cb = ChromaBackend.load(tmp_path, "test_cs", ids, dn, fingerprint=emb_fingerprint(emb))
    return cb, emb, ids, dn, q, tmp_path


def test_chroma_matches_exact(chroma_backend):
    cb, emb, ids, dn, q, _ = chroma_backend
    got = cb.search(q, 10)
    assert len(set(r for r, _ in got) & set(_exact(emb, dn, q, 10))) >= 9
    assert got[0][1] == pytest.approx(float(emb[got[0][0]] @ q), abs=1e-4)


def test_chroma_where_doc_filter_like_router(chroma_backend):
    cb, emb, ids, dn, q, _ = chroma_backend
    sub = ["A_2021_10K", "C_2022_10Q"]
    got = cb.search(q, 10, sub)
    assert got and {dn[r] for r, _ in got} <= set(sub)
    assert len(set(r for r, _ in got) & set(_exact(emb, dn, q, 10, sub))) >= 9
    assert len(cb.search(q, 1000, ["B_2021_10K"])) == N // 4      # k capped at filter size
    assert cb.search(q, 5, ["NOT_A_DOC"]) == []


def test_chroma_company_year_metadata(chroma_backend):
    cb, emb, ids, dn, q, _ = chroma_backend
    got = cb.search_where(q, 20, chroma_where(companies=["A"], years=[2021]))
    assert got and {dn[r] for r, _ in got} == {"A_2021_10K"}


def test_chroma_persists_and_detects_stale(chroma_backend):
    from src.vector_backends import ChromaBackend

    cb, emb, ids, dn, q, path = chroma_backend
    first = cb.search(q, 5)
    reopened = ChromaBackend.load(path, "test_cs", ids, dn)
    assert [r for r, _ in reopened.search(q, 5)] == [r for r, _ in first]
    with pytest.raises(StaleIndexError):
        ChromaBackend.load(path, "test_cs", ids, dn, fingerprint="other")
