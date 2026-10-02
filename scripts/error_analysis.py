"""Bucket every non-correct answer of a test run (default: full RAG with the main model).

Buckets, first match wins:
  routing error        routed mode and the gold filing was not among the routed candidates
  retrieval miss       gold evidence page not in the context given to the model
  refusal              model said "I don't know" although the gold page was in its context
  judge disagreement   the developer's hand label says 'correct' (from human_labels.jsonl or
                       results/error_review.jsonl), so the automatic label was wrong
  reading/calculation  gold page was in context but the answer is wrong / partial

Writes results/error_analysis.json with counts and up to 3 examples per bucket.
Usage: .venv\\Scripts\\python scripts\\error_analysis.py [--run test__rag__qwen2.5-7b.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, path, questions_for  # noqa: E402
from src.evaluation import read_jsonl  # noqa: E402

BUCKETS = ["routing error", "retrieval miss", "refusal", "judge disagreement", "reading/calculation error"]


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=f"test__rag__{cfg['llm']['model'].replace(':', '-')}.jsonl")
    ap.add_argument("--out", default="error_analysis.json", help="file name under results/")
    a = ap.parse_args()
    res = path("results")
    qs = {q["financebench_id"]: q for q in questions_for("test")}
    scored = read_jsonl(res / "scored" / a.run)
    gen = {r["id"]: r for r in read_jsonl(res / "gen" / a.run)}
    human = {(h["run"], h["id"]): h for h in read_jsonl(res / "human_labels.jsonl")}
    human.update({(h["run"], h["id"]): h for h in read_jsonl(res / "error_review.jsonl")})

    buckets = {b: [] for b in BUCKETS}
    for s in scored:
        if s["label"] == "correct":
            continue
        q, g = qs[s["id"]], gen[s["id"]]
        h = human.get((a.run, s["id"]))
        if g["mode"] == "routed" and g["route_candidates"] and q["doc_name"] not in g["route_candidates"]:
            b, note = "routing error", f"routed to {g['route_candidates']}, gold filing {q['doc_name']}"
        elif not s["gold_in_context"]:
            b = "retrieval miss"
            note = (f"gold page(s) {[(e['doc_name'], e['evidence_page_num']) for e in q['evidence']]} not in "
                    f"context {[(c[1], c[2]) for c in g['context']]}")
        elif s["refused"]:
            b, note = "refusal", "gold page was in the context, model said I don't know"
        elif h and h["label"] == "correct":
            b, note = "judge disagreement", f"auto label {s['label']} ({s['method']}); hand label correct: {h.get('note', '')}"
        else:
            b = "reading/calculation error"
            note = f"auto label {s['label']} ({s['method']}). " + (h.get("note", "") if h else s.get("reason", ""))
        buckets[b].append({"id": s["id"], "question": q["question"], "gold": q["answer"],
                           "answer": g["answer"], "label": s["label"], "note": note})
    out = {"run": a.run, "n_questions": len(scored),
           "n_not_correct": sum(len(v) for v in buckets.values()),
           "buckets": {k: {"count": len(v), "ids": [x["id"] for x in v], "examples": v[:3]}
                       for k, v in buckets.items()}}
    cap = cfg["llm"]["num_predict"]
    capped = [s for s in scored if (s.get("completion_tokens") or 0) >= cap]
    out["hit_token_cap"] = {"cap": cap, "count": len(capped),
                            "ids_labels": [[s["id"], s["label"]] for s in capped]}
    (res / a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print({k: v["count"] for k, v in out["buckets"].items()}, "of", out["n_not_correct"], "not correct;", len(capped), f"answers hit the {cap}-token cap")


if __name__ == "__main__":
    main()
