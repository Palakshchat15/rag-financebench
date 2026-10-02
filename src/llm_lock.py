"""Shared Ollama lock across the AI Engineer portfolio projects.

Before an Ollama-heavy job (generation eval, judge scoring) we create
AI_Engineer_Portfolio/.llm_lock holding the project name, PID and start time, and delete it
when the job ends (also on failure). If the file exists and belongs to another project, we wait
and poll. If it already belongs to this project (e.g. held by scripts/run_eval_queue.py around
a whole queue), the job proceeds and leaves the file for the holder to delete.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

PROJECT = "rag_financebench"
LOCK = Path(__file__).resolve().parents[2] / ".llm_lock"


def _owner() -> str:
    try:
        return LOCK.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return ""


@contextmanager
def llm_lock(job: str, poll_s: int = 30):
    waited = False
    while True:
        if not LOCK.exists():
            try:
                with open(LOCK, "x", encoding="utf-8") as f:
                    f.write(f"{PROJECT}\nstarted {datetime.now().isoformat(timespec='seconds')}\n"
                            f"pid {os.getpid()}\njob: {job}\n")
                created = True
                break
            except FileExistsError:
                continue
        if _owner() == PROJECT:
            created = False
            break
        if not waited:
            print(f"LLM lock held by '{_owner()}' ({LOCK}); waiting ...", flush=True)
            waited = True
        time.sleep(poll_s)
    try:
        yield
    finally:
        if created:
            try:
                LOCK.unlink()
            except FileNotFoundError:
                pass
