import pytest

from src.chunking import chunk_page, chunk_pages
from src.retrieval import rrf


class WordTok:
    """Whitespace tokenizer for tests (1 word = 1 token)."""

    def __init__(self):
        self.vocab, self.inv = {}, {}

    def encode(self, text):
        out = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
                self.inv[self.vocab[w]] = w
            out.append(self.vocab[w])
        return out

    def decode(self, ids):
        return " ".join(self.inv[i] for i in ids)


def words(prefix, n):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_chunks_never_cross_pages():
    tok = WordTok()
    pages = [{"doc_name": "D", "page": p, "text": words(f"p{p}w", 23)} for p in range(3)]
    chunks = chunk_pages(pages, size=10, overlap=2, tok=tok)
    for c in chunks:
        # every word in a chunk carries its page prefix -> no chunk mixes pages
        assert all(w.startswith(f"p{c.page}w") for w in c.text.split())
        assert c.doc_name == "D"
    assert {c.page for c in chunks} == {0, 1, 2}


def test_chunk_sizes_overlap_and_coverage():
    tok = WordTok()
    text = words("w", 23)
    cs = chunk_page("D", 5, text, size=10, overlap=2, tok=tok)
    assert [c.n_tokens for c in cs] == [10, 10, 7]
    assert cs[0].text.split()[-2:] == cs[1].text.split()[:2]          # overlap of 2
    covered = set(w for c in cs for w in c.text.split())
    assert covered == set(text.split())                                # nothing dropped
    assert [c.chunk_id for c in cs] == ["D|p5|c0", "D|p5|c1", "D|p5|c2"]
    assert all(c.page == 5 for c in cs)


def test_short_and_empty_pages():
    tok = WordTok()
    assert chunk_page("D", 0, "", 10, 2, tok) == []
    assert chunk_page("D", 0, "   \n ", 10, 2, tok) == []
    cs = chunk_page("D", 0, "only three words", 10, 2, tok)
    assert len(cs) == 1 and cs[0].n_tokens == 3


def test_exact_multiple_has_no_empty_tail():
    tok = WordTok()
    cs = chunk_page("D", 0, words("w", 18), size=10, overlap=2, tok=tok)
    assert [c.n_tokens for c in cs] == [10, 10]


def test_bad_params():
    with pytest.raises(ValueError):
        chunk_page("D", 0, "a b", 10, 10, WordTok())


def test_rrf_basic():
    fused = rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == "b"                     # 1/62 + 1/61 beats 1/61
    assert set(ids) == {"a", "b", "c", "d"}
    d = dict(fused)
    assert d["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert d["d"] == pytest.approx(1 / 63)


def test_rrf_weights_and_ties():
    fused = rrf([["a"], ["b"]], k=60, weights=[1.0, 2.0])
    assert fused[0][0] == "b"
    tie = rrf([["x"], ["y"]], k=60)
    assert [i for i, _ in tie] == ["x", "y"]          # deterministic tie-break
    with pytest.raises(ValueError):
        rrf([["a"]], weights=[1, 2])


def test_rrf_single_list_preserves_order():
    assert [i for i, _ in rrf([["c", "a", "b"]])] == ["c", "a", "b"]
