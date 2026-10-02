import pytest

from src.scoring import (final_answer_segment, is_refusal, numeric_gold, parse_numbers,
                         score_numeric)


def S(pred: str, gold: str) -> str:
    g = numeric_gold(gold)
    assert g is not None, f"gold {gold!r} should be numeric"
    return score_numeric(pred, g)


@pytest.mark.parametrize("gold,expected", [
    ("$1577.00", 1577.0), ("1.9%", 1.9), ("0.66", 0.66), ("-0.02", -0.02),
    ("$59268.00", 59268.0), ("24.26", 24.26), ("$1.2 billion", 1.2e9), ("(370)", -370.0),
    ("$8.70", 8.70), ("12,345", 12345.0), ("-4.5%", -4.5),
])
def test_numeric_gold_detected(gold, expected):
    g = numeric_gold(gold)
    assert g is not None and g.value == pytest.approx(expected)


@pytest.mark.parametrize("gold", [
    "No, the company is managing its CAPEX well", "AES has converted inventory 9.5 times in FY 2022.",
    "Data Center", "$400,000,000 increase.", "Yes, one customer accounted for 16% of revenue",
    "0.67 times to 0.69 times", "",
])
def test_non_numeric_gold(gold):
    assert numeric_gold(gold) is None


@pytest.mark.parametrize("pred,gold,label", [
    # exact / formatting
    ("Answer: $1,577 million [3M_2018_10K p.59]", "$1577.00", "correct"),
    ("Answer: 1577", "$1577.00", "correct"),
    ("Answer: $1.577 billion", "$1577.00", "correct"),
    ("Answer: $1,577,000,000", "$1577.00", "correct"),
    ("Answer: 1,200 million", "$1.2 billion", "correct"),
    ("Answer: $1.2B", "$1.2 billion", "correct"),
    ("Answer: $8.7 per share", "$8.70", "correct"),
    # tolerance and rounding
    ("Answer: 24.3", "24.26", "correct"),                # within 1%
    ("Answer: 25.0", "24.26", "incorrect"),              # 3% off
    ("Answer: 1.87%", "1.9%", "correct"),
    ("Answer: 2.0%", "1.9%", "incorrect"),
    ("Answer: 0.2%", "0.2%", "correct"),
    ("Answer: 0.24%", "0.2%", "correct"),                # rounds to gold's 1 decimal
    ("Answer: 0.26%", "0.2%", "incorrect"),
    # percent vs ratio
    ("Answer: 0.019", "1.9%", "correct"),
    ("Answer: 66%", "0.66", "correct"),
    ("Answer: 1.9", "1.9%", "correct"),
    # scale guard: a bare 660 must not match 0.66
    ("Answer: 660", "0.66", "incorrect"),
    # signs
    ("Answer: -0.02", "-0.02", "correct"),
    ("Answer: (0.02)", "-0.02", "correct"),
    ("Answer: 0.02", "-0.02", "incorrect"),
    ("Answer: a decrease of 4.5%", "-4.5%", "correct"),
    ("Answer: -4.5%", "4.5%", "incorrect"),
    ("Answer: −4.5%", "-4.5%", "correct"),          # unicode minus
    # citations and years must not be read as the answer
    ("Answer: In FY2018 capex was not stated [3M_2018_10K p.1577]", "$1577.00", "incorrect"),
    ("Answer: FY2019 ratio is 63.86", "63.86", "correct"),
    ("Answer: 2019", "63.86", "incorrect"),
    # the final-answer line is what counts, not the working
    ("Answer: 30.8%\nCalculation: 1,234 / 4,000 = 0.3085", "30.8%", "correct"),
    ("Answer: 12%\nWorking: we first got 30.8% but ...", "30.8%", "incorrect"),
    ("**Answer:** $9,068 million", "$9068.00", "correct"),
    # refusals
    ("Answer: I don't know", "$1577.00", "refused"),
    ("I do not know.", "$1577.00", "refused"),
    ("Answer: I don’t know.", "1.9%", "refused"),
    ("Answer: none", "1.9%", "incorrect"),
])
def test_score_numeric(pred, gold, label):
    assert S(pred, gold) == label


def test_final_answer_segment_fallback_first_line():
    assert final_answer_segment("$5 million\nmore text") == "$5 million"
    assert final_answer_segment("") == ""


def test_parse_numbers_scale_letters_only_when_attached():
    nums = parse_numbers("Revenue was $2.5B and plan B had 3 items")
    assert nums[0].value == pytest.approx(2.5e9)
    assert nums[1].value == 3.0


def test_final_answer_segment_reasoning_first():
    txt = "Reasoning: capex 2017 was $155 million and revenue $7,017 million.\n\nAnswer: 1.9% [X p.1]"
    assert final_answer_segment(txt) == "1.9% [X p.1]"
    # the last Answer line wins
    assert final_answer_segment("Answer: draft 5\nReasoning: redo\n**Answer:** 7") == "7"
    # cut off before the final answer: no answer, so reasoning numbers are not scored
    assert final_answer_segment("Reasoning: revenue was $7,017 million and") == ""


def test_final_answer_segment_inline_answer():
    # 3B often appends 'Answer:' to the reasoning paragraph instead of a new line (test audit)
    txt = "Reasoning: [X p.84] says VaR decreased by $7 million. Answer: I don't know"
    assert final_answer_segment(txt) == "I don't know"
    assert is_refusal(txt)
    assert final_answer_segment("Reasoning: 1,244.5 / 2,707.3 = 0.46 [X p.2] Answer: 0.46") == "0.46"
    # lower-case 'answer:' inside the reasoning is not a final answer
    assert final_answer_segment("Reasoning: to answer: we need revenue $7,017 million and") == ""


def test_score_numeric_ignores_reasoning_numbers():
    g = numeric_gold("$155.00")
    assert score_numeric("Reasoning: capex was $155 million in 2017.\nAnswer: $131 million", g) == "incorrect"
    assert score_numeric("Reasoning: capex was $155 million in 2017 and", g) == "incorrect"


def test_is_refusal():
    assert is_refusal("Answer: I don't know")
    assert is_refusal("Reasoning: the documents lack cost of sales.\nAnswer: I don't know")
    assert not is_refusal("Reasoning: I don't know the 2016 value, but 2017 is given.\nAnswer: $5 million")
    assert not is_refusal("Answer: $5 million. We know this from the cash flow statement.")
