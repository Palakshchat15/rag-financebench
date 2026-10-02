"""Download the FinanceBench open-source sample (questions, doc info, referenced PDFs),
write data/raw/MANIFEST.csv with sha256 checksums, and make the dev/test split.

Usage:  .venv\\Scripts\\python scripts\\download_data.py
Re-running skips files that already exist (checksums are recomputed).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, path  # noqa: E402

# Generic UA; no personal identifiers.
HEADERS = {"User-Agent": "rag-financebench-portfolio/1.0"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers=HEADERS, timeout=300)
    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    print(f"downloaded {dest.name} ({len(r.content)/1e6:.1f} MB)")


def make_split(questions: list[dict], dev_size: int, seed: int) -> dict:
    """Stratified by question_type: dev gets dev_size/len share of each type."""
    by_type = defaultdict(list)
    for q in questions:
        by_type[q["question_type"]].append(q["financebench_id"])
    rng = random.Random(seed)
    dev, test = [], []
    n = len(questions)
    types = sorted(by_type)
    # Largest-remainder allocation so dev sums exactly to dev_size.
    quotas = {t: dev_size * len(by_type[t]) / n for t in types}
    alloc = {t: int(quotas[t]) for t in types}
    rest = dev_size - sum(alloc.values())
    for t in sorted(types, key=lambda t: quotas[t] - alloc[t], reverse=True)[:rest]:
        alloc[t] += 1
    for t in types:
        ids = sorted(by_type[t])
        rng.shuffle(ids)
        dev += ids[: alloc[t]]
        test += ids[alloc[t]:]
    return {"seed": seed, "stratify_by": "question_type", "dev": sorted(dev), "test": sorted(test)}


def main() -> None:
    cfg = load_config()
    repo, commit = cfg["data"]["repo"], cfg["data"]["repo_commit"]
    base = f"https://raw.githubusercontent.com/{repo}/{commit}"
    raw = path("raw")
    files = ["financebench_open_source.jsonl", "financebench_document_information.jsonl"]
    for fn in files:
        fetch(f"{base}/data/{fn}", raw / fn)

    with open(raw / files[0], encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    docs = sorted({q["doc_name"] for q in questions} | {e["doc_name"] for q in questions for e in q["evidence"]})
    print(f"{len(questions)} questions reference {len(docs)} documents")
    for d in docs:
        fetch(f"{base}/pdfs/{d}.pdf", path("pdfs") / f"{d}.pdf")

    rows = []
    for fn in files:
        p = raw / fn
        rows.append((f"{fn}", p.stat().st_size, sha256(p), f"{base}/data/{fn}"))
    for d in docs:
        p = path("pdfs") / f"{d}.pdf"
        rows.append((f"pdfs/{d}.pdf", p.stat().st_size, sha256(p), f"{base}/pdfs/{d}.pdf"))
    with open(raw / "MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "bytes", "sha256", "source_url"])
        w.writerows(rows)
    print(f"wrote MANIFEST.csv ({len(rows)} files)")

    split_path = path("splits")
    if split_path.exists():
        print("splits.json exists; not overwriting (the split is fixed).")
    else:
        split = make_split(questions, cfg["data"]["dev_size"], cfg["data"]["split_seed"])
        split_path.write_text(json.dumps(split, indent=1), encoding="utf-8")
        print(f"wrote splits.json: dev={len(split['dev'])} test={len(split['test'])}")


if __name__ == "__main__":
    main()
