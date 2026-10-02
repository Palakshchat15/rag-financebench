"""Measure prompt-eval and generation tokens/sec for each model on this machine.

Usage:  .venv\\Scripts\\python scripts\\bench_models.py
Writes results/model_speed.json
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, path  # noqa: E402
from src.llm import chat, check_ollama  # noqa: E402

SHORT = "In two sentences, explain what free cash flow is."
# ~1,000-token prompt, closer to a RAG prompt, to measure prompt-eval speed.
LONG = ("Context:\n" + ("Revenue grew 5% to $10.2 billion while operating expenses rose 3%. " * 60)
        + "\nQuestion: What was revenue? Answer in one sentence.")


def main() -> None:
    cfg = load_config()
    out = {"machine": platform.processor(), "results": []}
    for model in cfg["llm"]["models_evaluated"]:
        check_ollama(model)
        chat([{"role": "user", "content": "hi"}], model, 4096, 8)  # warm-up / load
        for name, prompt in (("short", SHORT), ("long_1k", LONG)):
            r = chat([{"role": "user", "content": prompt}], model, 4096, 128)
            row = {
                "model": model, "prompt": name,
                "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                "prompt_tok_per_s": round(r["prompt_tokens"] / max(r["prompt_eval_s"], 1e-9), 1),
                "gen_tok_per_s": round(r["completion_tokens"] / max(r["eval_s"], 1e-9), 1),
                "wall_s": round(r["wall_s"], 1),
            }
            print(row, flush=True)
            out["results"].append(row)
    p = path("results") / "model_speed.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
