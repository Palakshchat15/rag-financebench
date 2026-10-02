from src.routing import Router, extract_quarter, extract_years

INFO = {
    "3M_2018_10K": {"company": "3M", "doc_type": "10k", "doc_period": 2018},
    "3M_2022_10K": {"company": "3M", "doc_type": "10k", "doc_period": 2022},
    "3M_2023Q2_10Q": {"company": "3M", "doc_type": "10q", "doc_period": 2023},
    "COCACOLA_2021_10K": {"company": "Coca-Cola", "doc_type": "10k", "doc_period": 2021},
    "COCACOLA_2022_10K": {"company": "Coca-Cola", "doc_type": "10k", "doc_period": 2022},
    "JOHNSON_JOHNSON_2022_10K": {"company": "Johnson & Johnson", "doc_type": "10k", "doc_period": 2022},
    "JOHNSON_JOHNSON_2023Q2_EARNINGS": {"company": "Johnson & Johnson", "doc_type": "Earnings", "doc_period": 2023},
    "AES_2022_10K": {"company": "AES Corporation", "doc_type": "10k", "doc_period": 2022},
    "BLOCK_2020_10K": {"company": "Block", "doc_type": "10k", "doc_period": 2020},
    "ADOBE_2015_10K": {"company": "Adobe", "doc_type": "10k", "doc_period": 2015},
    "ADOBE_2016_10K": {"company": "Adobe", "doc_type": "10k", "doc_period": 2016},
    "PEPSICO_2023_8K_dated-2023-05-05": {"company": "PepsiCo", "doc_type": "8k", "doc_period": 2023},
    "PEPSICO_2023Q1_EARNINGS": {"company": "PepsiCo", "doc_type": "Earnings", "doc_period": 2023},
    "PEPSICO_2022_10K": {"company": "PepsiCo", "doc_type": "10k", "doc_period": 2022},
}
R = Router(INFO, list(INFO))


def test_years_and_quarters():
    assert extract_years("FY2018 capex") == [2018]
    assert extract_years("from FY2015 to FY2016") == [2015, 2016]
    assert extract_years("FY22 margin") == [2022]
    assert extract_years("in fiscal 2021") == [2021]
    assert extract_years("no year here, 10,000 units") == []
    assert extract_quarter("Q2 of FY2023") == 2
    assert extract_quarter("the second quarter of 2023") == 2
    assert extract_quarter("annual") is None


def test_company_and_year():
    r = R.route("What is the FY2018 capital expenditure amount for 3M?")
    assert r.companies == ["3M"] and r.candidates == ["3M_2018_10K"]


def test_quarter_prefers_10q():
    r = R.route("Does 3M have a healthy quick ratio for Q2 of FY2023?")
    assert r.candidates == ["3M_2023Q2_10Q"]


def test_aliases():
    assert R.route("What was Coca Cola's FY2022 revenue?").candidates == ["COCACOLA_2022_10K"]
    assert R.route("CocaCola FY2021 revenue").candidates == ["COCACOLA_2021_10K"]
    assert R.route("Johnson & Johnson FY2022 inventory turnover").candidates == ["JOHNSON_JOHNSON_2022_10K"]
    assert R.route("Did AES grow in FY 2022?").candidates == ["AES_2022_10K"]


def test_ambiguous_word_block():
    assert R.route("Which block of shares was sold in 2020?").fallback_shared
    assert R.route("What was Block's 2020 revenue?").candidates == ["BLOCK_2020_10K"]


def test_multi_year_prefers_latest():
    r = R.route("Adobe change in operating income from FY2015 to FY2016")
    assert r.candidates == ["ADOBE_2016_10K"]


def test_doc_type_hint():
    assert R.route("In PepsiCo's 8-K from May 2023, what was announced?").candidates == [
        "PEPSICO_2023_8K_dated-2023-05-05"]


def test_no_year_returns_all_company_docs_10k_first():
    r = R.route("Does 3M maintain a stable trend of dividend distribution?")
    assert set(r.candidates) == {"3M_2018_10K", "3M_2022_10K", "3M_2023Q2_10Q"}
    assert r.candidates[-1] == "3M_2023Q2_10Q"


def test_company_level_keeps_all_years():
    r = R.route("What is the FY2018 capital expenditure amount for 3M?", level="company")
    assert set(r.candidates) == {"3M_2018_10K", "3M_2022_10K", "3M_2023Q2_10Q"}
    assert r.candidates[0] == "3M_2018_10K"


def test_tickers_case_sensitive():
    assert R.route("Are JnJ's FY2022 financials strong?").candidates == ["JOHNSON_JOHNSON_2022_10K"]
    assert R.route("Is JNJ growing in FY2022?").candidates == ["JOHNSON_JOHNSON_2022_10K"]
    assert R.route("the pep talk in 2022").fallback_shared           # lower-case 'pep' is a word
    assert R.route("PEP revenue 2022").candidates == ["PEPSICO_2022_10K"]


def test_compact_quarter_and_half_year():
    assert extract_quarter("As of FY2023Q1, why") == 1
    assert extract_quarter("in H1 FY2023") == 2
    assert extract_years("As of FY2023Q1") == [2023]
    # same year: both 2023 PepsiCo filings stay, Q1 earnings ranked first
    r = R.route("As of FY2023Q1, why did Pepsico raise guidance?")
    assert r.candidates[0] == "PEPSICO_2023Q1_EARNINGS"
    assert set(r.candidates) == {"PEPSICO_2023Q1_EARNINGS", "PEPSICO_2023_8K_dated-2023-05-05"}


def test_no_company_falls_back_to_shared():
    r = R.route("What is the average capex of the S&P 500?")
    assert r.fallback_shared and r.candidates == []
