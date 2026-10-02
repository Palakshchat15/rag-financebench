"""Page-aware, token-based chunking. A chunk never crosses a page, so every chunk has
exactly one citable (doc_name, page)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class TiktokenTokenizer:
    def __init__(self, name: str = "cl100k_base"):
        import tiktoken

        self._enc = tiktoken.get_encoding(name)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, disallowed_special=())

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_name: str
    page: int
    idx: int          # position of the chunk within its page
    text: str
    n_tokens: int


def chunk_page(doc_name: str, page: int, text: str, size: int, overlap: int,
               tok: Tokenizer) -> list[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    if not text or not text.strip():
        return []
    ids = tok.encode(text)
    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    while True:
        window = ids[start:start + size]
        piece = tok.decode(window).strip()
        if piece:
            i = len(chunks)
            chunks.append(Chunk(f"{doc_name}|p{page}|c{i}", doc_name, page, i, piece, len(window)))
        if start + size >= len(ids):
            break
        start += step
    return chunks


def chunk_pages(pages, size: int, overlap: int, tok: Tokenizer) -> list[Chunk]:
    """pages: iterable of dicts/rows with doc_name, page, text."""
    out: list[Chunk] = []
    for r in pages:
        out.extend(chunk_page(r["doc_name"], int(r["page"]), r["text"], size, overlap, tok))
    return out
