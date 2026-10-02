"""LLM-as-judge for non-numeric gold answers (qwen2.5:7b, temperature 0, JSON output).

Labels: correct | partially_correct | incorrect | refused. The rubric was written and checked
on dev answers only; its agreement with hand labels is measured on a test sample
(scripts/label_judge_sample.py) and reported, not tuned against.
"""
from __future__ import annotations

import json
import re

from .config import load_config
from .llm import chat, num_ctx_for

LABELS = ("correct", "partially_correct", "incorrect", "refused")

JUDGE_SYSTEM = """You grade answers to financial questions about company filings. You are given the \
question, the gold (reference) answer written by a financial analyst, and a model's answer. \
Compare the model's answer to the gold answer only; do not use your own knowledge of the company. \
The model's answer may contain a 'Reasoning:' part and a final 'Answer:' part: grade the final \
answer, and use the reasoning only to understand what the final answer means. \
Do not penalise extra facts or numbers that the gold answer does not mention: you cannot verify \
them. Judge only whether the gold's key facts and conclusion are present and not contradicted.

Labels:
- correct: the model's final answer conveys the same key facts and conclusion as the gold answer. \
Numbers must match the gold within rounding (about 1%). Different wording, extra correct detail, \
or a different but equivalent unit is fine. For yes/no questions the yes/no must match and the \
main reason must be consistent with the gold.
- partially_correct: some of the gold's key content is present and correct, but a key part is \
missing or wrong (e.g. right conclusion with a wrong supporting number, or only one of several \
items the gold lists).
- incorrect: the answer contradicts the gold, has the wrong number or conclusion, or answers a \
different question.
- refused: the model says it does not know or cannot answer, without giving a substantive answer.

Return JSON only: {"label": "<one of correct, partially_correct, incorrect, refused>", \
"reason": "<one sentence>"}"""


def judge_messages(question: str, gold: str, answer: str) -> list[dict]:
    user = (f"Question:\n{question}\n\nGold answer:\n{gold}\n\n"
            f"Model answer:\n{answer}\n\nGrade the model answer.")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_label(text: str) -> tuple[str, str]:
    try:
        d = json.loads(text)
        label = str(d.get("label", "")).strip().lower().replace(" ", "_").replace("-", "_")
        reason = str(d.get("reason", ""))
    except Exception:  # noqa: BLE001
        m = re.search(r"(partially_correct|incorrect|correct|refused)", text.lower())
        label, reason = (m.group(1) if m else "unparsed"), text[:200]
    if label == "partially":
        label = "partially_correct"
    return (label if label in LABELS else "unparsed"), reason


def judge(question: str, gold: str, answer: str, model: str | None = None) -> dict:
    model = model or load_config()["llm"]["judge_model"]
    msgs = judge_messages(question, gold, answer)
    est = int(sum(len(m["content"]) for m in msgs) / 3.2)
    out = chat(msgs, model, num_ctx_for(est, 200), num_predict=200, json_format=True)
    label, reason = parse_label(out["content"])
    return {"label": label, "reason": reason, "judge_model": model, "raw": out["content"],
            "judge_s": out["wall_s"]}
