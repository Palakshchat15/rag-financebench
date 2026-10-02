"""Score cached generations: numeric scorer for numeric gold answers, LLM judge otherwise.
Judge calls are cached per (file, id) in results/judge/. Writes:
  results/scored/{split}__{config}__{model}.jsonl   per-question labels + metrics
  results/generation_{split}.csv                    one row per (config, model)

Usage:  .venv\\Scripts\\python scripts\\score_generation.py --split test [--no-judge]
"""
from __future__ import annotations

import argparse
import hashlib
from contextlib import nullcontext
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, path, questions_for  # noqa: E402
from src.evaluation import append_jsonl, gold_docs, gold_pages, read_jsonl  # noqa: E402
from src.judge import judge  # noqa: E402
from src.llm import check_ollama  # noqa: E402
from src.llm_lock import llm_lock  # noqa: E402
from src.scoring import is_refusal, numeric_gold, score_numeric  # noqa: E402


def pct(xs, q):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, q)) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--no-judge", action="store_true", help="skip rows that need the LLM judge")
    a = ap.parse_args()
    cfg = load_config()
    qs = {q["financebench_id"]: q for q in questions_for(a.split)}
    gen_dir = path("results") / "gen"
    judge_dir = path("results") / "judge"
    scored_dir = path("results") / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    if not a.no_judge:
        check_ollama(cfg["llm"]["judge_model"])
    with (nullcontext() if a.no_judge else llm_lock(f"score_generation --split {a.split}")):
        score_all(a, qs, gen_dir, judge_dir, scored_dir)


def score_all(a, qs, gen_dir, judge_dir, scored_dir) -> None:
    summary = []
    for gp in sorted(gen_dir.glob(f"{a.split}__*.jsonl")):
        rows = read_jsonl(gp)
        jp = judge_dir / gp.name
        jcache = {(r["id"], r["answer_sha"]): r for r in read_jsonl(jp)}
        out = []
        for r in rows:
            q = qs[r["id"]]
            ng = numeric_gold(q["answer"])
            sha = hashlib.sha256(r["answer"].encode()).hexdigest()[:16]
            if is_refusal(r["answer"]):
                label, method, reason = "refused", "rule", "exact refusal"
            elif ng is not None:
                label, method, reason = score_numeric(r["answer"], ng), "numeric", ""
            else:
                key = (r["id"], sha)
                if key not in jcache:
                    if a.no_judge:
                        continue
                    j = judge(q["question"], q["answer"], r["answer"])
                    jrow = {"id": r["id"], "answer_sha": sha, **j}
                    append_jsonl(jp, jrow)
                    jcache[key] = jrow
                    print(f"  judged {gp.stem} {r['id']}: {j['label']}", flush=True)
                label, method, reason = jcache[key]["label"], "judge", jcache[key]["reason"]
                if label == "unparsed":
                    label = "incorrect"
            gpages = gold_pages(q)
            ctx_pages = {(c[1], int(c[2])) for c in r["context"]}
            cits = [tuple(c) for c in r["citations"]]
            gl = {(d.lower(), p) for d, p in gpages}
            out.append({
                "id": r["id"], "question_type": q["question_type"], "config": r["config"],
                "model": r["model"], "label": label, "method": method, "reason": reason,
                "refused": label == "refused",
                "gold_in_context": bool(ctx_pages & gpages) if r["config"] != "closed_book" else None,
                "gold_doc_routed": (q["doc_name"] in r["route_candidates"]) if r["route_candidates"] else None,
                "n_citations": len(cits), "n_invalid_citations": len(r["invalid_citations"]),
                "cites_gold_page": any((d.lower(), p) in gl for d, p in cits),
                "cites_gold_doc": any(d.lower() in {x.lower() for x in gold_docs(q)} for d, _ in cits),
                "timings": r["timings"], "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
            })
        with open(scored_dir / gp.name, "w", encoding="utf-8") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
        if not out:
            continue
        df = pd.DataFrame(out)
        n = len(df)
        cited = df[df.n_citations > 0]
        tot_c = int(df.n_citations.sum())
        refused = df[df.refused]
        row = {
            "run": gp.stem, "config": df.config.iat[0], "model": df.model.iat[0], "n": n,
            "complete": n == len(qs),
            "correct": (df.label == "correct").mean(),
            "partially_correct": (df.label == "partially_correct").mean(),
            "correct_or_partial": df.label.isin(["correct", "partially_correct"]).mean(),
            "incorrect": (df.label == "incorrect").mean(),
            "refusal_rate": df.refused.mean(),
            "n_numeric_scored": int((df.method == "numeric").sum()),
            "n_judge_scored": int((df.method == "judge").sum()),
            "correct_numeric": (df[df.method == "numeric"].label == "correct").mean() if (df.method == "numeric").any() else None,
        }
        for t in ("metrics-generated", "domain-relevant", "novel-generated"):
            s = df[df.question_type == t]
            row[f"correct_{t}"] = (s.label == "correct").mean() if len(s) else None
        if df.config.iat[0] != "closed_book":
            # A refusal is 'correct' only when the gold evidence page was NOT in the context.
            row["refusal_gold_absent"] = int((refused.gold_in_context == False).sum())  # noqa: E712
            row["refusal_gold_present"] = int((refused.gold_in_context == True).sum())  # noqa: E712
            row["answers_with_citation"] = (df.n_citations > 0).mean()
            row["citation_validity"] = 1 - df.n_invalid_citations.sum() / tot_c if tot_c else None
            row["answers_all_citations_valid"] = (cited.n_invalid_citations == 0).mean() if len(cited) else None
            row["cites_gold_page"] = df.cites_gold_page.mean()
            row["gold_page_in_context"] = df.gold_in_context.mean()
        for st in ("route_s", "retrieve_s", "rerank_s", "generate_s", "total_s"):
            vals = [t.get(st) for t in df.timings]
            row[f"p50_{st}"], row[f"p95_{st}"] = pct(vals, 50), pct(vals, 95)
        row["mean_prompt_tokens"] = df.prompt_tokens.mean()
        row["mean_completion_tokens"] = df.completion_tokens.mean()
        summary.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    if summary:
        pd.DataFrame(summary).to_csv(path("results") / f"generation_{a.split}.csv", index=False)


if __name__ == "__main__":
    main()
