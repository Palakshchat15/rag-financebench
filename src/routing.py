"""Deterministic router: match a question to candidate filings by company, year and doc type.

No learning, no LLM. Rules were written from the document list (metadata) and checked on
dev only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_SUFFIXES = r"\b(corporation|corp|company|co|inc|incorporated|plc|ltd|group|holdings)\b\.?"
# Company names that are also common English words: require case-sensitive match.
_AMBIGUOUS = {"block"}
_QUARTER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4}

# Common abbreviations / tickers for the companies in the corpus (general knowledge, written
# for all 32 companies, not just ones seen failing). Matched case-sensitively.
EXTRA_ALIASES = {
    "3M": ["MMM"], "Activision Blizzard": ["ATVI", "Activision"], "Adobe": ["ADBE"],
    "AES Corporation": ["AES"], "AMD": ["Advanced Micro Devices"], "Amazon": ["AMZN"],
    "Amcor": ["AMCR", "AMCOR"], "American Express": ["AMEX", "Amex", "AXP"],
    "American Water Works": ["AWK", "American Water"], "Best Buy": ["BBY", "BestBuy"],
    "Block": ["Square", "SQ"], "Boeing": ["BA"], "CVS Health": ["CVS"], "Coca-Cola": ["KO", "Coke"],
    "Corning": ["GLW"], "Costco": ["COST"], "Foot Locker": ["FL", "Footlocker"],
    "General Mills": ["GIS"], "JPMorgan": ["JPM", "JP Morgan", "JPMorgan Chase"],
    "Johnson & Johnson": ["JNJ", "JnJ", "J&J"], "Kraft Heinz": ["KHC"], "Lockheed Martin": ["LMT"],
    "MGM Resorts": ["MGM"], "Microsoft": ["MSFT"], "Netflix": ["NFLX"], "Nike": ["NKE"],
    "Paypal": ["PYPL", "PayPal"], "PepsiCo": ["PEP", "Pepsi"], "Pfizer": ["PFE"],
    "Ulta Beauty": ["ULTA", "Ulta"], "Verizon": ["VZ"], "Walmart": ["WMT", "Wal-Mart"],
}


def _norm(s: str) -> str:
    s = s.lower().replace("&", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def company_aliases(company: str, doc_names: list[str]) -> set[str]:
    """Normalised spellings of a company name."""
    base = _norm(company)
    stripped = _norm(re.sub(_SUFFIXES, " ", company, flags=re.I))
    al = {base, stripped}
    for d in doc_names:
        pref = d.rsplit("_", 1)[0] if re.search(r"_\d{4}", d) is None else re.split(r"_\d{4}", d)[0]
        al.add(_norm(pref.replace("_", " ")))
    # compact forms ("coca cola" -> "cocacola") for multi-word names
    for a in list(al):
        if " " in a:
            al.add(a.replace(" ", ""))
    return {a for a in al if a}


def extract_years(q: str) -> list[int]:
    years = set()
    for m in re.finditer(r"(?<!\d)(?:FY\s?'?)?((?:19|20)\d{2})(?!\d)", q, flags=re.I):
        years.add(int(m.group(1)))
    for m in re.finditer(r"\bFY\s?'?(\d{2})\b", q, flags=re.I):
        years.add(2000 + int(m.group(1)))
    return sorted(years)


def extract_quarter(q: str) -> int | None:
    m = re.search(r"(?<![A-Za-z])Q([1-4])\b", q, flags=re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(first|second|third|fourth)\s+(fiscal\s+)?quarter\b", q, flags=re.I)
    if m:
        return _QUARTER_WORDS[m.group(1).lower()]
    if re.search(r"\bH1\b|\bfirst half\b", q, flags=re.I):
        return 2
    return None


def doc_type_hint(q: str) -> str | None:
    ql = q.lower()
    if re.search(r"\b8-?k\b", ql):
        return "8k"
    if re.search(r"\b10-?q\b", ql) or "quarterly report" in ql:
        return "10q"
    if "earnings" in ql and ("release" in ql or "call" in ql or "report" in ql):
        return "earnings"
    if re.search(r"\b10-?k\b", ql) or "annual report" in ql:
        return "10k"
    return None


def _doc_quarter(doc_name: str) -> int | None:
    m = re.search(r"\d{4}Q([1-4])", doc_name)
    return int(m.group(1)) if m else None


@dataclass
class RouteResult:
    companies: list[str]
    years: list[int]
    quarter: int | None
    doc_type: str | None
    candidates: list[str] = field(default_factory=list)   # doc_names, best first
    company_docs: list[str] = field(default_factory=list)
    fallback_shared: bool = False


class Router:
    def __init__(self, doc_info: dict[str, dict], indexed_docs: list[str]):
        self.docs = {d: doc_info[d] for d in indexed_docs}
        by_company: dict[str, list[str]] = {}
        for d, info in self.docs.items():
            by_company.setdefault(info["company"], []).append(d)
        self.by_company = by_company
        self.aliases = {c: company_aliases(c, ds) for c, ds in by_company.items()}
        self.cs_aliases = {c: EXTRA_ALIASES.get(c, []) for c in by_company}

    def match_companies(self, q: str) -> list[str]:
        qn = f" {_norm(q)} "
        qc = qn.replace(" ", "")
        hits = []
        for c, als in self.aliases.items():
            for a in als:
                if a in _AMBIGUOUS:
                    if re.search(rf"\b{re.escape(a.capitalize())}\b", q):
                        hits.append((c, len(a)))
                        break
                    continue
                if f" {a} " in qn or (" " not in a and len(a) >= 8 and a in qc):
                    hits.append((c, len(a)))
                    break
            else:
                for a in self.cs_aliases[c]:
                    if re.search(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])", q):
                        hits.append((c, len(a)))
                        break
        # Drop companies whose match is a substring of a longer matched name
        hits.sort(key=lambda x: -x[1])
        return [c for c, _ in hits]

    def route(self, q: str, level: str = "company_year") -> RouteResult:
        """level="company": all filings of the matched company.
        level="company_year": those whose fiscal year best matches the years in the question
        (and the filing type, if the question names one explicitly). The type/quarter cues only
        order the candidates; they never drop a filing unless the type is named explicitly."""
        companies = self.match_companies(q)
        years = extract_years(q)
        quarter = extract_quarter(q)
        dtype = doc_type_hint(q)
        res = RouteResult(companies, years, quarter, dtype)
        cands = [d for c in companies for d in self.by_company[c]]
        if not cands:
            res.fallback_shared = True
            return res

        def ttype(d: str) -> str:
            t = self.docs[d]["doc_type"].lower()
            return "10k" if t.startswith("10k") else t

        def year_score(d: str) -> float:
            if not years:
                return 0.0
            p = self.docs[d]["doc_period"]
            return (2.0 if p in years else 0.0) + (1.0 if p == max(years) else 0.0)

        def type_score(d: str) -> float:
            t = ttype(d)
            if dtype:
                return 2.0 if t == dtype else 0.0
            if quarter is not None:
                return (1.0 if t in ("10q", "earnings") else 0.0) + (1.0 if _doc_quarter(d) == quarter else 0.0)
            return 0.5 if t == "10k" else 0.0

        ordered = sorted(cands, key=lambda d: (-(year_score(d) + type_score(d)), d))
        res.company_docs = ordered
        if level == "company":
            res.candidates = ordered
            return res
        best = max(year_score(d) for d in ordered)
        keep = [d for d in ordered if year_score(d) == best]
        if dtype and any(ttype(d) == dtype for d in keep):
            keep = [d for d in keep if ttype(d) == dtype]
        res.candidates = keep
        return res
