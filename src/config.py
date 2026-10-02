"""Load config.yaml and resolve paths relative to the project root."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Keep tokenizer / model caches inside the project so the demo laptop is self-contained.
    cache = ROOT / cfg["paths"]["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache / "tiktoken"))
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(cache / "fastembed"))
    return cfg


def path(key: str) -> Path:
    return ROOT / load_config()["paths"][key]


def load_questions() -> list[dict]:
    with open(path("raw") / "financebench_open_source.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_doc_info() -> dict[str, dict]:
    with open(path("raw") / "financebench_document_information.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["doc_name"]: r for r in rows}


def load_splits() -> dict[str, list[str]]:
    with open(path("splits"), encoding="utf-8") as f:
        return json.load(f)


def questions_for(split: str) -> list[dict]:
    ids = set(load_splits()[split])
    return [q for q in load_questions() if q["financebench_id"] in ids]
