from src.validation import parse_citations, strip_citations, validate_citations


def test_single_and_multiple_citations():
    t = "Capex was $1,577m [3M_2018_10K p.59]. Revenue [3M_2018_10K p.60; 3M_2018_10K p.61]."
    assert parse_citations(t) == [("3M_2018_10K", 59), ("3M_2018_10K", 60), ("3M_2018_10K", 61)]


def test_page_inherits_doc_and_dedup():
    t = "[AMCOR_2023_10K p.3, p.4] and again [AMCOR_2023_10K p.3]"
    assert parse_citations(t) == [("AMCOR_2023_10K", 3), ("AMCOR_2023_10K", 4)]


def test_doc_names_with_dashes():
    t = "[FOOTLOCKER_2022_8K_dated-2022-05-20 p.2]"
    assert parse_citations(t) == [("FOOTLOCKER_2022_8K_dated-2022-05-20", 2)]


def test_variants_and_non_citations():
    assert parse_citations("[NIKE_2023_10K page 12]") == [("NIKE_2023_10K", 12)]
    assert parse_citations("[NIKE_2023_10K p 12]") == [("NIKE_2023_10K", 12)]
    assert parse_citations("see [note 3] and [1]") == []
    assert parse_citations("") == []


def test_validate_flags_hallucinated():
    allowed = {("3M_2018_10K", 59), ("3M_2018_10K", 60)}
    rep = validate_citations("a [3M_2018_10K p.59] b [3M_2018_10K p.99] c [3m_2018_10k p.60]", allowed)
    assert rep.valid == [("3M_2018_10K", 59), ("3m_2018_10k", 60)]
    assert rep.invalid == [("3M_2018_10K", 99)]
    assert not rep.all_valid and rep.n == 3


def test_validate_no_citations():
    rep = validate_citations("I don't know", {("X", 1)})
    assert rep.n == 0 and rep.all_valid


def test_strip_citations():
    assert "p.59" not in strip_citations("x [3M_2018_10K p.59] y")
