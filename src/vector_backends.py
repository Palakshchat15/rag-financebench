"""Optional dense-retrieval backends: the default NumPy matrix product, a ChromaDB collection
and FAISS indexes (exact IndexFlatIP and approximate IndexHNSWFlat).

All backends are built from the *existing* chunk embeddings (data/index/cs{size}/emb.npy, written
by scripts/build_index.py). Nothing is re-embedded. The NumPy backend stays the default
(config.yaml -> retrieval.dense_backend), so every reported number was produced by it.

Layout (all under data/index/, gitignored):
    chroma/cs{size}/            Chroma PersistentClient dir, collection "financebench_cs{size}"
    chroma/cs{size}/build_info.json
    faiss/cs{size}/flat.faiss   IndexFlatIP over the L2-normalised vectors (exact cosine)
    faiss/cs{size}/hnsw.faiss   IndexHNSWFlat, inner-product metric (approximate)
    faiss/cs{size}/ids.json     FAISS id (= row of chunks.parquet) -> chunk_id
    faiss/cs{size}/{flat,hnsw}.json   build info

Every backend answers search(qvec, k, docs) -> [(row, score)], best first, where `row` is the
position in chunks.parquet (the same id space the NumPy path and the BM25 index use) and `docs`
is the router's candidate filings (None = shared store). Chroma applies the restriction as a
metadata `where` filter, FAISS as an IDSelector, NumPy as a boolean mask.

chromadb and faiss-cpu are optional (requirements-vectordb.txt) and only imported when used.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

DENSE_BACKENDS = ("numpy", "chroma", "faiss_flat", "faiss_hnsw")
FAISS_KINDS = {"faiss_flat": "flat", "faiss_hnsw": "hnsw"}

DEFAULT_PARAMS = {
    "chroma": {"space": "cosine", "M": 16, "ef_construction": 200, "ef_search": 200,
               "batch_size": 4096},
    "faiss_hnsw": {"M": 32, "ef_construction": 200, "ef_search": 128},
}


def emb_fingerprint(emb: np.ndarray) -> str:
    """Content hash of the embedding matrix; a built backend is refused if emb.npy changed."""
    h = hashlib.sha1(str(emb.shape).encode())
    h.update(np.ascontiguousarray(emb, dtype=np.float32).tobytes())
    return h.hexdigest()


def collection_name(size_dir: str) -> str:
    return f"financebench_{size_dir}"


def dir_size_bytes(p: Path) -> int:
    p = Path(p)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


class StaleIndexError(RuntimeError):
    """The built backend does not match the current chunks / embeddings."""


# --------------------------------------------------------------------------------------------
class DenseBackend:
    """Common part: row bookkeeping and the docs -> allowed-rows restriction."""

    name = "base"

    def __init__(self, chunk_ids, doc_names):
        self.chunk_ids = np.asarray(chunk_ids)
        self.doc_names = np.asarray(doc_names)
        self._rows_by_doc: dict[str, np.ndarray] = {}
        for d in np.unique(self.doc_names):
            self._rows_by_doc[str(d)] = np.flatnonzero(self.doc_names == d).astype(np.int64)

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def allowed_rows(self, docs) -> np.ndarray | None:
        if not docs:
            return None
        parts = [self._rows_by_doc[d] for d in dict.fromkeys(docs) if d in self._rows_by_doc]
        return np.concatenate(parts) if parts else np.array([], dtype=np.int64)

    def n_allowed(self, docs) -> int:
        rows = self.allowed_rows(docs)
        return len(self) if rows is None else len(rows)

    def search(self, qvec: np.ndarray, k: int, docs=None) -> list[tuple[int, float]]:
        raise NotImplementedError


class NumpyBackend(DenseBackend):
    """Exact brute force, identical to the original Index.dense_scores + _topn path."""

    name = "numpy"

    def __init__(self, emb: np.ndarray, chunk_ids, doc_names):
        super().__init__(chunk_ids, doc_names)
        self.emb = emb

    def search(self, qvec, k, docs=None):
        from .retrieval import _topn

        s = self.emb @ qvec
        mask = None if not docs else np.isin(self.doc_names, list(docs))
        idx = _topn(s, mask, k)
        return [(int(i), float(s[i])) for i in idx]


# --------------------------------------------------------------------------------------------
def _import_faiss():
    try:
        import faiss
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("faiss-cpu is not installed: pip install -r requirements-vectordb.txt") from e
    return faiss


def _id_selector(faiss, rows: np.ndarray):
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    try:
        return faiss.IDSelectorBatch(rows)                         # newer python wrapper
    except TypeError:
        return faiss.IDSelectorBatch(len(rows), faiss.swig_ptr(rows))


class FaissBackend(DenseBackend):
    """FAISS index whose ids are the chunk rows (0..n-1, added in chunks.parquet order);
    ids.json keeps the row -> chunk_id mapping and is checked on load."""

    def __init__(self, index, kind: str, chunk_ids, doc_names, ef_search: int = 128):
        super().__init__(chunk_ids, doc_names)
        if index.ntotal != len(self.chunk_ids):
            raise StaleIndexError(f"faiss {kind}: {index.ntotal} vectors vs {len(self.chunk_ids)} chunks")
        self.index = index
        self.kind = kind
        self.name = f"faiss_{kind}"
        self.ef_search = int(ef_search)

    # -- build / persist ------------------------------------------------------------------
    @staticmethod
    def build_index(emb: np.ndarray, kind: str, M: int = 32, ef_construction: int = 200):
        faiss = _import_faiss()
        x = np.ascontiguousarray(emb, dtype=np.float32)
        norms = np.linalg.norm(x, axis=1)
        if not np.allclose(norms[norms > 0], 1.0, atol=1e-3):
            raise ValueError("embeddings must be L2-normalised (inner product = cosine)")
        d = x.shape[1]
        if kind == "flat":
            index = faiss.IndexFlatIP(d)
        elif kind == "hnsw":
            index = faiss.IndexHNSWFlat(d, int(M), faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = int(ef_construction)
        else:
            raise ValueError(f"unknown faiss kind {kind}")
        index.add(x)          # ids = 0..n-1 = rows of chunks.parquet
        return index

    @classmethod
    def build(cls, out_dir: Path, kind: str, emb: np.ndarray, chunk_ids, params: dict | None = None,
              force: bool = False) -> dict:
        faiss = _import_faiss()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        params = params or {}
        f_index, f_info = out_dir / f"{kind}.faiss", out_dir / f"{kind}.json"
        if f_index.exists() and f_info.exists() and not force:
            return json.loads(f_info.read_text())
        t0 = time.perf_counter()
        index = cls.build_index(emb, kind, params.get("M", 32), params.get("ef_construction", 200))
        build_s = time.perf_counter() - t0
        faiss.write_index(index, str(f_index))
        (out_dir / "ids.json").write_text(json.dumps([str(c) for c in chunk_ids]))
        info = {"backend": f"faiss_{kind}", "n_vectors": int(index.ntotal), "dim": int(emb.shape[1]),
                "build_s": round(build_s, 3), "params": params if kind == "hnsw" else {},
                "emb_fingerprint": emb_fingerprint(emb),
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        f_info.write_text(json.dumps(info, indent=1))
        return info

    @classmethod
    def load(cls, in_dir: Path, kind: str, chunk_ids, doc_names, fingerprint: str | None = None,
             ef_search: int = 128) -> "FaissBackend":
        faiss = _import_faiss()
        in_dir = Path(in_dir)
        f_index = in_dir / f"{kind}.faiss"
        if not f_index.exists():
            raise FileNotFoundError(f"{f_index} missing: run scripts/build_vector_index.py first")
        ids = json.loads((in_dir / "ids.json").read_text())
        if ids != [str(c) for c in chunk_ids]:
            raise StaleIndexError(f"{in_dir}/ids.json does not match chunks.parquet: rebuild with --force")
        info = json.loads((in_dir / f"{kind}.json").read_text())
        if fingerprint and info.get("emb_fingerprint") != fingerprint:
            raise StaleIndexError(f"{f_index} was built from different embeddings: rebuild with --force")
        return cls(faiss.read_index(str(f_index)), kind, chunk_ids, doc_names, ef_search)

    # -- search ---------------------------------------------------------------------------
    def search(self, qvec, k, docs=None):
        faiss = _import_faiss()
        rows = self.allowed_rows(docs)
        k = min(int(k), self.index.ntotal if rows is None else len(rows))
        if k <= 0:
            return []
        q = np.ascontiguousarray(np.asarray(qvec, dtype=np.float32).reshape(1, -1))
        sel = _id_selector(faiss, rows) if rows is not None else None
        if self.kind == "hnsw":
            params = faiss.SearchParametersHNSW()
            params.efSearch = max(self.ef_search, k)
        else:
            params = faiss.SearchParameters() if sel is not None else None
        if sel is not None:
            params.sel = sel
        D, I = self.index.search(q, k, params=params)
        return [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i >= 0]


# --------------------------------------------------------------------------------------------
def _import_chromadb():
    try:
        import chromadb
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("chromadb is not installed: pip install -r requirements-vectordb.txt") from e
    return chromadb


def chroma_client(persist_dir: Path):
    chromadb = _import_chromadb()
    from chromadb.config import Settings

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir),
                                     settings=Settings(anonymized_telemetry=False))


def chroma_where(docs=None, companies=None, years=None) -> dict | None:
    """Chroma metadata filter for a routed query. docs = the router's candidate filings
    (exactly what the NumPy mask uses); companies / years allow coarser filters."""
    conds = []
    if docs:
        conds.append({"doc_name": {"$in": [str(d) for d in dict.fromkeys(docs)]}})
    if companies:
        conds.append({"company": {"$in": [str(c) for c in dict.fromkeys(companies)]}})
    if years:
        conds.append({"year": {"$in": [int(y) for y in dict.fromkeys(years)]}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def chunk_metadata(chunk_ids, doc_names, pages, doc_info: dict | None) -> list[dict]:
    """Per-chunk Chroma metadata (no None values: Chroma rejects them)."""
    doc_info = doc_info or {}
    out = []
    for row, (d, p) in enumerate(zip(doc_names, pages)):
        info = doc_info.get(str(d)) or {}
        out.append({"row": row, "doc_name": str(d), "page": int(p),
                    "company": str(info.get("company", "")),
                    "year": int(info.get("doc_period", -1) or -1),
                    "doc_type": str(info.get("doc_type", ""))})
    return out


class ChromaBackend(DenseBackend):
    name = "chroma"

    def __init__(self, collection, chunk_ids, doc_names, space: str = "cosine"):
        super().__init__(chunk_ids, doc_names)
        self.collection = collection
        self.space = space
        self._row_of = {str(c): i for i, c in enumerate(self.chunk_ids)}

    @staticmethod
    def _create(client, name: str, params: dict, info: dict):
        hnsw = {"space": params.get("space", "cosine"),
                "ef_construction": int(params.get("ef_construction", 200)),
                "ef_search": int(params.get("ef_search", 200)),
                "max_neighbors": int(params.get("M", 16))}
        try:
            return client.create_collection(name, configuration={"hnsw": hnsw}, metadata=info,
                                            embedding_function=None)
        except (TypeError, ValueError):
            # Older clients: legacy "hnsw:*" collection metadata.
            legacy = {"hnsw:space": hnsw["space"], "hnsw:construction_ef": hnsw["ef_construction"],
                      "hnsw:search_ef": hnsw["ef_search"], "hnsw:M": hnsw["max_neighbors"]}
            return client.create_collection(name, metadata={**info, **legacy}, embedding_function=None)

    @classmethod
    def build(cls, persist_dir: Path, name: str, emb: np.ndarray, chunk_ids, metadatas: list[dict],
              params: dict | None = None, force: bool = False) -> dict:
        """Add the existing embeddings (no re-embedding; documents are not stored, the text
        lives in chunks.parquet)."""
        params = {**DEFAULT_PARAMS["chroma"], **(params or {})}
        persist_dir = Path(persist_dir)
        f_info = persist_dir / "build_info.json"
        client = chroma_client(persist_dir)
        exists = name in [getattr(c, "name", c) for c in client.list_collections()]
        if exists and f_info.exists() and not force:
            return json.loads(f_info.read_text())
        if exists:
            client.delete_collection(name)
        fp = emb_fingerprint(emb)
        t0 = time.perf_counter()
        col = cls._create(client, name, params, {"n_chunks": int(len(chunk_ids)), "emb_fingerprint": fp})
        try:
            max_batch = int(client.get_max_batch_size())
        except Exception:  # noqa: BLE001 - older clients
            max_batch = 5000
        step = max(1, min(int(params["batch_size"]), max_batch))
        ids = [str(c) for c in chunk_ids]
        for s in range(0, len(ids), step):
            col.add(ids=ids[s:s + step], embeddings=np.ascontiguousarray(emb[s:s + step], dtype=np.float32),
                    metadatas=metadatas[s:s + step])
            print(f"  chroma: added {min(s + step, len(ids))}/{len(ids)} ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
        build_s = time.perf_counter() - t0
        info = {"backend": "chroma", "collection": name, "n_vectors": int(col.count()),
                "dim": int(emb.shape[1]), "build_s": round(build_s, 3), "params": params,
                "emb_fingerprint": fp, "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        f_info.write_text(json.dumps(info, indent=1))
        return info

    @classmethod
    def load(cls, persist_dir: Path, name: str, chunk_ids, doc_names,
             fingerprint: str | None = None) -> "ChromaBackend":
        persist_dir = Path(persist_dir)
        if not (persist_dir / "build_info.json").exists():
            raise FileNotFoundError(f"{persist_dir} has no Chroma build: run scripts/build_vector_index.py first")
        client = chroma_client(persist_dir)
        col = client.get_collection(name, embedding_function=None)
        meta = col.metadata or {}
        if col.count() != len(chunk_ids):
            raise StaleIndexError(f"chroma {name}: {col.count()} vectors vs {len(chunk_ids)} chunks")
        if fingerprint and meta.get("emb_fingerprint") != fingerprint:
            raise StaleIndexError(f"chroma {name} was built from different embeddings: rebuild with --force")
        space = meta.get("hnsw:space", "cosine")
        try:
            space = col.configuration["hnsw"]["space"]          # chromadb >= 1.0
        except Exception:  # noqa: BLE001
            pass
        return cls(col, chunk_ids, doc_names, space=str(space))

    def _score(self, dist: float) -> float:
        # cosine / ip distance = 1 - dot for normalised vectors; l2 = 2 - 2 dot
        return 1.0 - dist / 2.0 if self.space == "l2" else 1.0 - dist

    def search_where(self, qvec, k: int, where: dict | None) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        res = self.collection.query(query_embeddings=[np.asarray(qvec, dtype=np.float32).tolist()],
                                    n_results=int(k), where=where, include=["distances"])
        ids, dists = res["ids"][0], res["distances"][0]
        return [(self._row_of[i], self._score(float(d))) for i, d in zip(ids, dists)]

    def search(self, qvec, k, docs=None):
        # Cap k at the number of filtered chunks (as NumPy's _topn does); some Chroma versions
        # raise instead of returning fewer results.
        k = min(int(k), self.n_allowed(docs))
        return self.search_where(qvec, k, chroma_where(docs=docs))


# --------------------------------------------------------------------------------------------
def backend_dirs(size_dir: str) -> dict[str, Path]:
    from .config import path

    return {"chroma": path("chroma") / size_dir, "faiss": path("faiss") / size_dir}


def backend_params(name: str) -> dict:
    from .config import load_config

    cfg = (load_config().get("vector_db") or {}).get(name) or {}
    return {**DEFAULT_PARAMS.get(name, {}), **cfg}


def build_backend(name: str, size_dir: str, emb: np.ndarray, chunks, doc_info: dict,
                  force: bool = False) -> dict:
    """Build one backend for one chunk size from existing embeddings; returns its build info."""
    dirs = backend_dirs(size_dir)
    if name == "chroma":
        meta = chunk_metadata(chunks.chunk_id, chunks.doc_name, chunks.page, doc_info)
        return ChromaBackend.build(dirs["chroma"], collection_name(size_dir), emb,
                                   chunks.chunk_id.tolist(), meta, backend_params("chroma"), force)
    if name in FAISS_KINDS:
        return FaissBackend.build(dirs["faiss"], FAISS_KINDS[name], emb, chunks.chunk_id.tolist(),
                                  backend_params(name), force)
    raise ValueError(f"cannot build backend {name!r} (choose from {DENSE_BACKENDS[1:]})")


def load_backend(name: str, index) -> DenseBackend:
    """Open a backend for an Index (src.indexing.Index). Checks it matches the index."""
    if name not in DENSE_BACKENDS:
        raise ValueError(f"unknown dense backend {name!r} (choose from {DENSE_BACKENDS})")
    if name == "numpy":
        return NumpyBackend(index.emb, index.chunk_ids, index.doc_names)
    size_dir = Path(index.dir).name
    dirs = backend_dirs(size_dir)
    fp = emb_fingerprint(index.emb)
    if name == "chroma":
        return ChromaBackend.load(dirs["chroma"], collection_name(size_dir), index.chunk_ids,
                                  index.doc_names, fingerprint=fp)
    kind = FAISS_KINDS[name]
    ef = backend_params("faiss_hnsw").get("ef_search", 128)
    return FaissBackend.load(dirs["faiss"], kind, index.chunk_ids, index.doc_names, fingerprint=fp,
                             ef_search=ef)
