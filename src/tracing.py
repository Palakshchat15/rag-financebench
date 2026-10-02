"""SQLite query tracing (data/traces.db). One row per answered query."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    source TEXT,              -- app | eval
    question TEXT,
    config TEXT,              -- rag | closed_book | oracle
    mode TEXT,                -- routed | shared | -
    model TEXT,
    route_candidates TEXT,    -- JSON list
    retrieved TEXT,           -- JSON [[chunk_id, score], ...]
    t_route REAL, t_retrieve REAL, t_rerank REAL, t_generate REAL, t_total REAL,
    prompt_tokens INTEGER, completion_tokens INTEGER,
    answer TEXT,
    citations TEXT,           -- JSON [[doc, page], ...]
    invalid_citations TEXT,   -- JSON [[doc, page], ...]
    citations_valid INTEGER,  -- 1 if every citation is in the retrieved context
    error TEXT
);
"""


class Tracer:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def log(self, **kw) -> int:
        row = {"ts": time.time(), **kw}
        for k in ("route_candidates", "retrieved", "citations", "invalid_citations"):
            if k in row and not isinstance(row[k], str):
                row[k] = json.dumps(row[k])
        cols = ", ".join(row)
        q = f"INSERT INTO traces ({cols}) VALUES ({', '.join('?' * len(row))})"
        with self._conn() as c:
            cur = c.execute(q, list(row.values()))
            return int(cur.lastrowid)

    def read(self, limit: int | None = None):
        import pandas as pd

        q = "SELECT * FROM traces ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
        with self._conn() as c:
            return pd.read_sql_query(q, c)
