"""Run the test-set generation evals strictly one after another (never two LLM jobs at once),
then score them. Each job resumes from its per-question JSONL cache, so re-running this script
after an interruption continues where it stopped.

Usage:  .venv\\Scripts\\python scripts\\run_eval_queue.py [--jobs closed_book:qwen2.5:7b,oracle:qwen2.5:7b] [--no-score]
Logs:   results/logs/gen_test__{config}__{model}.log, results/logs/score_test.log
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# Required runs first (cheapest first, so the most finishes if interrupted), optional ones last.
DEFAULT_JOBS = ["closed_book:qwen2.5:7b", "oracle:qwen2.5:7b", "rag:qwen2.5:3b", "rag:qwen2.5:7b",
                "closed_book:qwen2.5:3b", "oracle:qwen2.5:3b"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=",".join(DEFAULT_JOBS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args()
    logs = ROOT / "results" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from src.llm_lock import llm_lock

    # Hold the shared lock for the whole queue (the child jobs see our own lock and proceed).
    with llm_lock(f"run_eval_queue {a.jobs}"):
        run_jobs(a, logs)


def run_jobs(a, logs: Path) -> None:
    for job in a.jobs.split(","):
        config, model = job.split(":", 1)
        log = logs / f"gen_{a.split}__{config}__{model.replace(':', '-')}.log"
        print(f"== {config} / {model} -> {log.name}", flush=True)
        with open(log, "a", encoding="utf-8") as f:
            rc = subprocess.call([PY, "-u", str(ROOT / "scripts" / "eval_generation.py"), "--split", a.split,
                                  "--config", config, "--model", model], stdout=f, stderr=subprocess.STDOUT,
                                 cwd=ROOT)
        print(f"   exit {rc}", flush=True)
        if rc != 0:
            sys.exit(rc)
        # Score after every job (judge calls are cached), so finished runs are scored even if a
        # later job is interrupted. Still sequential: the judge never runs alongside generation.
        if not a.no_score:
            with open(logs / f"score_{a.split}.log", "a", encoding="utf-8") as f:
                rc = subprocess.call([PY, "-u", str(ROOT / "scripts" / "score_generation.py"), "--split",
                                      a.split], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
            print(f"   scoring exit {rc}", flush=True)


if __name__ == "__main__":
    main()
