"""Citation parsing and validation. Citations look like [DOC_NAME p.N] (N is the 0-indexed
PDF page, as in FinanceBench). A bracket may hold several: [DOC p.3; DOC p.4] or [DOC p.3, p.4]."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_PART = re.compile(r"^\s*(?:(?P<doc>[A-Za-z0-9][A-Za-z0-9_\-\.]*?)\s*,?\s+)?(?:p\.?|pp\.?|page)\s*(?P<page>\d+)\s*$",
                   re.I)


def parse_citations(text: str) -> list[tuple[str, int]]:
    """Return citations in order of appearance, de-duplicated."""
    out: list[tuple[str, int]] = []
    for m in _BRACKET.finditer(text or ""):
        last_doc = None
        for part in re.split(r"[;,]", m.group(1)):
            pm = _PART.match(part)
            if not pm:
                continue
            doc = pm.group("doc") or last_doc
            if not doc:
                continue
            last_doc = doc
            c = (doc, int(pm.group("page")))
            if c not in out:
                out.append(c)
    return out


def strip_citations(text: str) -> str:
    return _BRACKET.sub(" ", text or "")


@dataclass
class CitationReport:
    citations: list[tuple[str, int]]
    valid: list[tuple[str, int]] = field(default_factory=list)
    invalid: list[tuple[str, int]] = field(default_factory=list)   # hallucinated citations

    @property
    def n(self) -> int:
        return len(self.citations)

    @property
    def all_valid(self) -> bool:
        return not self.invalid

    def to_dict(self) -> dict:
        return {"citations": [list(c) for c in self.citations],
                "valid": [list(c) for c in self.valid],
                "invalid": [list(c) for c in self.invalid]}


def validate_citations(answer: str, allowed: set[tuple[str, int]]) -> CitationReport:
    """allowed: the (doc_name, page) pairs actually given to the model as context.
    Doc-name matching is case-insensitive (the model sometimes changes case)."""
    cits = parse_citations(answer)
    allowed_l = {(d.lower(), p) for d, p in allowed}
    rep = CitationReport(cits)
    for d, p in cits:
        (rep.valid if (d.lower(), p) in allowed_l else rep.invalid).append((d, p))
    return rep
