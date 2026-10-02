"""Judge validation.

  --make   : draw a fixed random sample (seed 7) of judge-scored TEST answers pooled over all
             runs, and write results/judge_sample_for_labeling.jsonl WITHOUT the judge labels
             (so the human labels are made blind to the judge).
  --score  : compare results/human_labels.jsonl (id, run, label, note) with the judge labels;
             writes results/judge_agreement.json (agreement, Cohen's kappa, confusion).

Human labels are made by the developer reading the question, gold answer, evidence and model
answer. They are not independent annotations.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import path, questions_for  # noqa: E402
from src.evaluation import read_jsonl  # noqa: E402
from src.judge import LABELS  # noqa: E402


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    res = path("results")
    qs = {q["financebench_id"]: q for q in questions_for("test")}

    if a.make:
        pool = []
        for sp in sorted((res / "scored").glob("test__*.jsonl")):
            gen = {r["id"]: r for r in read_jsonl(res / "gen" / sp.name)}
            for r in read_jsonl(sp):
                if r["method"] == "judge":
                    pool.append((sp.name, r["id"], gen[r["id"]]["answer"]))
        rng = random.Random(7)
        sample = rng.sample(pool, min(a.n, len(pool)))
        with open(res / "judge_sample_for_labeling.jsonl", "w", encoding="utf-8") as f:
            for run, qid, ans in sample:
                q = qs[qid]
                f.write(json.dumps({
                    "run": run, "id": qid, "question": q["question"], "gold": q["answer"],
                    "justification": q.get("justification"),
                    "evidence": [e["evidence_text"][:1500] for e in q["evidence"]],
                    "model_answer": ans}, ensure_ascii=False) + "\n")
        print(f"wrote {len(sample)} items (pool {len(pool)})")

    if a.score:
        human = read_jsonl(res / "human_labels.jsonl")
        pairs = []
        for h in human:
            s = {r["id"]: r for r in read_jsonl(res / "scored" / h["run"])}[h["id"]]
            pairs.append((h["label"], s["label"], h))
        hl, jl = [p[0] for p in pairs], [p[1] for p in pairs]
        agree = sum(x == y for x, y in zip(hl, jl)) / len(pairs)
        # Binary view: correct vs not correct (what the headline accuracy uses)
        hb = ["correct" if x == "correct" else "not" for x in hl]
        jb = ["correct" if x == "correct" else "not" for x in jl]
        conf = {h: {j: sum(1 for x, y in zip(hl, jl) if x == h and y == j) for j in LABELS} for h in LABELS}
        out = {
            "n": len(pairs), "agreement": agree, "kappa": cohen_kappa(hl, jl),
            "binary_agreement": sum(x == y for x, y in zip(hb, jb)) / len(pairs),
            "binary_kappa": cohen_kappa(hb, jb),
            "confusion_human_rows_judge_cols": conf,
            "judge_more_lenient": sum(1 for x, y in zip(hl, jl) if LABELS.index(y) < LABELS.index(x) and y != "refused"),
            "disagreements": [{"run": h["run"], "id": h["id"], "human": x, "judge": y, "note": h.get("note", "")}
                              for x, y, h in pairs if x != y],
            "note": "Human labels by the developer (not independent annotators), made blind to judge labels.",
        }
        (res / "judge_agreement.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=1))


if __name__ == "__main__":
    main()
