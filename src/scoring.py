"""Answer scoring: deterministic numeric matching, refusal detection, and the final label.

Numeric gold answers (a single number, optionally with $, %, a scale word) are scored by
parsing numbers from the model's final-answer line and comparing with a 1% relative tolerance
(or half a unit in the gold's last decimal place, to allow correct rounding). Units / scale
("$1.2 billion" vs "1,200 million"), percent vs ratio, and signs are handled below.
Everything else goes to the LLM judge (src/judge.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .validation import strip_citations

SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "millions": 1e6, "mn": 1e6, "mm": 1e6,
          "m": 1e6, "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12, "t": 1e12,
          "thousands": 1e3}

_NUM = re.compile(
    r"(?<![A-Za-z0-9_.])"                       # not glued to a word (e.g. FY2019, Q2)
    r"(?P<neg>[-−–(]\s?)?"            # minus / en dash / opening parenthesis
    r"(?:US)?\$?\s?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<close>\))?"
    r"\s?(?P<pct>%|percent\b|pct\b)?"
    r"(?:\s?(?P<scale>thousands?|millions?|billions?|trillion|mn|mm|bn|[kmbt])\b)?",
    re.I,
)
_YEARLIKE = re.compile(r"^(19[5-9]\d|20[0-4]\d)$")
_NEG_WORDS = re.compile(r"\b(decrease[sd]?|declin\w*|drop\w*|fell|fall\w*|lower|down|negative|loss|"
                        r"contract\w*|shr[ua]nk|reduc\w*|deficit)\b", re.I)
_REFUSAL = re.compile(r"\bI\s+do(?:n['’]?t| not)\s+know\b", re.I)


@dataclass
class Num:
    value: float        # signed, with scale applied
    raw: float          # unsigned mantissa as written
    decimals: int
    pct: bool
    scale: float        # 1.0 if no scale word
    has_scale: bool
    negative: bool
    text: str


def parse_numbers(text: str, drop_years: bool = True) -> list[Num]:
    text = strip_citations(text).replace(" ", " ")
    out = []
    for m in _NUM.finditer(text):
        num = m.group("num")
        raw = float(num.replace(",", ""))
        neg_tok = (m.group("neg") or "").strip()
        negative = neg_tok in ("-", "−", "–") or (neg_tok == "(" and bool(m.group("close")))
        scale_w = (m.group("scale") or "").lower()
        # single letters only count as a scale when directly attached to a $ amount or number
        if scale_w in ("k", "m", "b", "t") and not re.search(r"\d\s?[kmbtKMBT]\b", m.group(0)):
            scale_w = ""
        scale = SCALES.get(scale_w, 1.0)
        pct = bool(m.group("pct"))
        dec = len(num.split(".")[1]) if "." in num else 0
        is_money = "$" in m.group(0)
        if drop_years and _YEARLIKE.match(num) and not (pct or scale_w or is_money):
            continue
        val = raw * scale * (-1 if negative else 1)
        out.append(Num(val, raw, dec, pct, scale, bool(scale_w), negative, m.group(0).strip()))
    return out


def final_answer_segment(answer: str) -> str:
    """The model is asked for 'Reasoning: ...' then a final 'Answer: ...' line. Use the LAST
    non-empty 'Answer:' line. If there is none and the reply starts with 'Reasoning:' it was cut
    off before the answer, so there is no final answer (""). Otherwise use the first line."""
    if not answer:
        return ""
    ms = [m.group(1).strip() for m in
          re.finditer(r"(?im)^\s*[*#]*\s*(?:final\s+)?answer\s*[*]*\s*:\s*[*]*(.*)$", answer)]
    ms = [x for x in ms if x]
    if ms:
        return ms[-1]
    # Small models sometimes put 'Answer:' at the end of the reasoning paragraph instead of on
    # its own line ("... so the total is X. Answer: I don't know"). Use the last such inline
    # 'Answer:' (capitalised, after a sentence end / bracket) before treating the reply as cut off.
    inline = [m.group(1).strip() for m in re.finditer(r"(?:^|[.!?\])\s])\s*\**Answer\**\s*:\s*\**(.*)$", answer, re.M)]
    inline = [x for x in inline if x]
    if inline:
        return inline[-1]
    if re.match(r"(?i)^\s*[*#]*\s*reasoning\s*[*]*\s*:", answer):
        return ""
    for line in answer.splitlines():
        if line.strip():
            return line.strip()
    return ""


def is_refusal(answer: str) -> bool:
    seg = final_answer_segment(answer)
    return bool(_REFUSAL.search(seg)) or bool(_REFUSAL.match((answer or "").strip()))


_GOLD_ALLOWED = re.compile(
    r"^(?:usd|us|\$|%|percent|x|times|thousands?|millions?|billions?|trillion|mn|bn|mm|in|of|"
    r"per|share|approximately|about|roughly|[\s,.;:()\-])*$", re.I)


def numeric_gold(gold: str) -> Num | None:
    """Return the single number if the gold answer is essentially one number, else None."""
    g = (gold or "").strip()
    if len(g) > 40:
        return None
    nums = parse_numbers(g, drop_years=False)
    if len(nums) != 1:
        return None
    rest = _NUM.sub(" ", strip_citations(g))
    return nums[0] if _GOLD_ALLOWED.match(rest) else None


def _close(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    d = abs(a - b)
    return d <= rel_tol * abs(b) + 1e-12 or d <= abs_tol + 1e-12


def number_matches(p: Num, g: Num, rel_tol: float = 0.01, seg: str = "") -> bool:
    """Compare magnitudes under the allowed unit interpretations, then check the sign."""
    half_unit = 0.5 * 10 ** (-g.decimals)
    pa, ga = abs(p.value), abs(g.value)
    trials: list[tuple[float, float, float]] = []   # (pred, gold, abs_tol in gold units)
    trials.append((pa, ga, half_unit * g.scale))           # fully scaled
    trials.append((p.raw, g.raw, half_unit))                # same written units
    # Gold unit implied by the question (e.g. "in USD millions" -> gold "$1577.00"): allow a
    # power-of-1000 rescale, but only if the prediction states its own scale or is written out
    # in full (>= 1000), so a bare "660" never matches a gold ratio of 0.66.
    if not g.has_scale and (p.has_scale or p.raw >= 1000):
        for j in range(-4, 5):
            f = 10.0 ** (3 * j)
            trials.append((pa, g.raw * f, half_unit * f))
    if g.pct and not p.pct:
        trials.append((p.raw * 100, g.raw, half_unit))      # 0.019 vs 1.9%
    if p.pct and not g.pct:
        trials.append((p.raw / 100, g.raw, half_unit))      # 66% vs 0.66
    if not any(_close(a, b, rel_tol, t) for a, b, t in trials):
        return False
    if g.value == 0 or p.negative == g.negative:
        return True
    # Magnitude matches but sign differs: accept "a decrease of 5%" for gold "-5%".
    return g.negative and not p.negative and bool(_NEG_WORDS.search(seg))


def score_numeric(answer: str, gold: Num, rel_tol: float = 0.01) -> str:
    if is_refusal(answer):
        return "refused"
    seg = final_answer_segment(answer)
    for p in parse_numbers(seg, drop_years=not _YEARLIKE.match(str(int(gold.raw)) if gold.raw.is_integer() else "")):
        if number_matches(p, gold, rel_tol, seg):
            return "correct"
    return "incorrect"
