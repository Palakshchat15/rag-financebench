# Upgrade notes: vector-database dense backends (code 2026-10-17)

**Run on 2026-10-18: done.** pytest 104 passed; chroma, faiss_flat and faiss_hnsw built for cs256 and benchmarked
on dev and test. The numpy hybrid routed test row reproduces the reported result (page hit@10 0.66, MRR 0.370);
chroma and faiss_flat match it exactly (overlap@10 1.0), faiss_hnsw overlap@10 0.988 with the same hit@k. Results:
README section "Vector database backends" and `results/vector_backends_*.csv`.

## What was added
- Configurable dense-retrieval backend for bm25 / dense / hybrid (RRF) retrieval:
  `config.yaml` -> `retrieval.dense_backend: numpy | chroma | faiss_flat | faiss_hnsw`.
  **Default `numpy`**: the original code path (exact `emb @ q` + mask), so every existing result
  and the app behave exactly as before.
- **ChromaDB**: persistent collection `financebench_cs{size}` in `data/index/chroma/cs{size}/`
  (gitignored), HNSW with cosine space, built from the existing `emb.npy` (no re-embedding, no
  documents stored). Metadata per chunk: `row`, `doc_name`, `page`, `company`, `year`, `doc_type`.
  The router's candidate filings become a Chroma `where` filter (`{"doc_name": {"$in": [...]}}`)
  instead of a NumPy mask; `chroma_where(companies=..., years=...)` and
  `ChromaBackend.search_where()` allow company/year filters too.
- **FAISS**: `IndexFlatIP` (exact cosine on the normalised vectors) and `IndexHNSWFlat`
  (inner product, approximate) in `data/index/faiss/cs{size}/`. FAISS ids are the chunk rows;
  `ids.json` maps them to `chunk_id` and is checked on load. Router restriction = `IDSelectorBatch`.
- Every built backend stores a SHA-1 fingerprint of `emb.npy`; loading refuses a stale build.
- Benchmark of the backends on dev and test: page hit@k, MRR, overlap@10 with the NumPy exact
  ranking (also dense-candidate overlap@100), query latency p50/p95 (vector search alone and the
  whole retrieval call), build time, size on disk, load time.

## Files
- New: `src/vector_backends.py`, `scripts/build_vector_index.py`, `scripts/bench_vector_backends.py`,
  `tests/test_vector_backends.py`, `requirements-vectordb.txt` (chromadb==1.5.9, faiss-cpu==1.15.1),
  `UPGRADE_NOTES.md`.
- Changed: `src/retrieval.py` (`Retriever(index, dense_backend=None)`, `_dense_rank()`, optional
  `qvec=` in `search()`; the numpy branch computes exactly what it did before),
  `config.yaml` (`paths.chroma`, `paths.faiss`, `retrieval.dense_backend`, `vector_db:` params),
  `.gitignore` (explicit chroma/faiss dirs; `data/index/` already covered them), `README.md`
  (section "Vector database backends"), `PROGRESS.md` (item 11).
- Not changed: any file in `results/`, the indexes, BM25, routing, generation, the app.

## How to run (PowerShell, project folder, in this order)
```powershell
# 1. optional packages, keeping the existing pins
.venv\Scripts\python -m pip install -r requirements-vectordb.txt -c requirements.txt
# 2. unit tests (the 9 chroma/faiss tests now run instead of skipping)
.venv\Scripts\python -m pytest -q
# 3. build the backends for the used chunk size (cs256) from the existing emb.npy
.venv\Scripts\python scripts\build_vector_index.py
#    (other sizes: --sizes 256 512 1024; one backend: --backends faiss_hnsw; rebuild: --force)
# 4. benchmark on dev + test -> results\vector_backends_dev.csv, _test.csv, _build.csv
.venv\Scripts\python scripts\bench_vector_backends.py
# 5. (optional) try a backend in the app / evals: set retrieval.dense_backend in config.yaml,
#    then set it back to numpy. Do not re-run the reported evals with another backend.
```
Steps 3-4 need `data/index/cs256/` (built earlier by `build_index.py`); nothing is re-embedded.
Only the query embeddings (150 questions) are computed, once, by the benchmark.

## What to verify
- Step 2: 95 + 9 tests pass, none skipped (before installing: 95 passed, 9 skipped).
- `vector_backends_test.csv`, backend `numpy`, method `hybrid`, mode `routed:company_year`:
  page_hit@10 must equal the existing `retrieval_test.csv` row (cs256, hybrid, weights [1, 2],
  no rerank, routed:company_year; README 66%). Same check on dev against `retrieval_dev.csv`.
  If it differs, the benchmark is wrong, not the reported result.
- `faiss_flat`: overlap@10 and dense_overlap@100 about 1.00 (exact search; tiny float ties only).
- `faiss_hnsw` and `chroma`: overlap close to 1; note any drop in the shared store vs routed mode
  (filtered HNSW search can lose recall with very selective filters).
- `vector_backends_build.csv`: build_s, disk_mb, load_s filled for each backend.
- Then fill the README table from the CSVs (test, hybrid, routed) and tick item 11 in PROGRESS.md.

## Expected run time (i5-1235U, 8 GB)
- pip install: a few minutes (chromadb pulls ~30 dependencies, roughly 150-250 MB download).
- Build, cs256 (46,816 x 384): faiss_flat seconds; faiss_hnsw about 1-2 min; chroma about 1-4 min.
- Benchmark: about 5-10 min (index + BM25 load, 150 query embeddings, 4 backends x 2 methods x
  2 modes x 150 questions). RAM well under 2 GB. No LLM, so `.llm_lock` is not needed, but don't
  run it alongside another CPU-heavy job if the latencies go into the README.

## Risks
- Dependency drift: chromadb brings opentelemetry, grpcio, kubernetes, pydantic, etc. Installing with
  `-c requirements.txt` keeps the existing pins; if pip reports a conflict, install into a separate
  venv rather than loosening `requirements.txt`. The venv is Python 3.14; both packages have
  Windows wheels for it (chromadb abi3, faiss-cpu cp314), checked on PyPI, not installed.
- Chroma API drift: collection HNSW settings use the 1.x `configuration={"hnsw": ...}` form, with a
  fallback to the legacy `hnsw:*` metadata. Chroma telemetry is turned off in the client settings.
- Approximate search with metadata filters: HNSW (Chroma, faiss_hnsw) may return lower recall or
  fewer than k results when the router leaves only one filing; the benchmark measures exactly this.
- Score scale: Chroma returns cosine distance, converted to similarity (1 - d). Scores differ in the
  last float digits between backends; RRF uses ranks only, so fusion is unaffected.
- A changed `emb.npy` or `chunks.parquet` makes the built backends stale; they refuse to load
  until rebuilt with `--force`.
