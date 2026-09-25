"""
Fixed Regression Test Suite for Business Entity Resolution Pipeline.
Tests end-to-end normalization, parsing, and candidate blocking behavior on 10 hand-picked tricky edge cases.
"""
import pytest
import pandas as pd

from src.name_normalizer import normalize_business_name
from src.address_parser import normalize_business_address
from src.country_normalizer import normalize_country
from src.blocker import CandidateBlocker


@pytest.fixture
def regression_blocker():
    return CandidateBlocker(fuzzy_threshold=0.28, max_fuzzy_candidates_per_s1=50, max_total_candidates_per_s1=80)


def test_case_1_severe_typo_near_miss(regression_blocker):
    """Severe typo in both name and address where no exact keys coincide."""
    s1 = pd.DataFrame([{
        "entity_id": "S1-TYPO-1",
        "business_name": "Apex Digital Solutions Inc",
        "business_address": "120 Market Street, San Jose, CA",
        "country": "US"
    }])
    other = pd.DataFrame([{
        "entity_id": "S2-TYPO-1",
        "business_name": "Apex Digitl Solutns",
        "business_address": "120 Market St, San Jose, California",
        "country": "US"
    }])
    cands = regression_blocker.generate_candidate_pairs(s1, other, verbose=False)
    assert "S2-TYPO-1" in cands["S1-TYPO-1"]


def test_case_2_trade_name_drift_fuzzy_bridge(regression_blocker):
    """Name drifted completely (trade name drift), but address matches."""
    s1 = pd.DataFrame([{
        "entity_id": "S1-DRIFT-1",
        "business_name": "Maure Williams Colombier Inc",
        "business_address": "85 Wayne Avenue, Ticonderoga, NY",
        "country": "US"
    }])
    other = pd.DataFrame([{
        "entity_id": "S2-DRIFT-1",
        "business_name": "Dréxkor",
        "business_address": "85 Wanye Avenue, Ticonderoga Townshiip, New York",
        "country": "US"
    }])
    cands = regression_blocker.generate_candidate_pairs(s1, other, verbose=False)
    assert "S2-DRIFT-1" in cands["S1-DRIFT-1"]


def test_case_3_ct_vs_connecticut_state_collision():
    """Ensure CT abbreviation at end of US address resolves to Connecticut, not Court."""
    parsed = normalize_business_address("33 Sleepy Hollow Drive, Danbury, CT", country="US")
    assert parsed["state"] == "connecticut"
    assert "court" not in parsed["clean_address"].split()[-1]


def test_case_4_indian_address_without_commas():
    """No commas, anchor-based parser must extract Punjab as state and Hoshiarpur as city."""
    parsed = normalize_business_address("H NO 204 C ROAD HOSHIARPUR PUNJAB Punjab", country="India")
    assert parsed["state"] == "punjab"
    assert "hoshiarpur" in parsed["city"]


def test_case_5_france_open_set_retrieval(regression_blocker):
    """Unseen test country (France) with word order variation and abbreviation."""
    s1 = pd.DataFrame([{
        "entity_id": "S1-FRA-1",
        "business_name": "Boulangerie Patisserie Parisienne",
        "business_address": "15 Rue de Rivoli, Paris",
        "country": "France"
    }])
    other = pd.DataFrame([{
        "entity_id": "S2-FRA-1",
        "business_name": "Boulangerie Parisienne Patisserie",
        "business_address": "15 Rue de Rivoli, Paris",
        "country": "FRA"
    }])
    cands = regression_blocker.generate_candidate_pairs(s1, other, verbose=False)
    assert "S2-FRA-1" in cands["S1-FRA-1"]


def test_case_6_s2_s3_near_duplicate_collapse():
    """Verify genuine S2 duplicate listing normalizes to identical core name and sorted tokens."""
    pn1 = normalize_business_name("Empire School of Medicine")
    pn2 = normalize_business_name("Empire Schóol of Medicine")
    pa1 = normalize_business_address("0477 MELISSA COURT, GAHANNA, OH", country="US")
    pa2 = normalize_business_address("0477 MELISSA CT, GAHANNA, OH", country="US")

    assert pn1["core_name"] == pn2["core_name"] == "empire school of medicine"
    assert pa1["sorted_address_tokens"] == pa2["sorted_address_tokens"]


def test_case_7_parenthetical_noise_filtering():
    """(India) and (Regd) parentheticals should be stripped from core_name."""
    pn = normalize_business_name("Techno (India) Infocom Private Limited")
    assert pn["core_name"] == "techno infocom"
    assert "india" not in pn["core_name"].split()


def test_case_8_asymmetric_nan_address_retrieval(regression_blocker):
    """When address is missing (nan), exact core name channel still catches the match without error."""
    s1 = pd.DataFrame([{
        "entity_id": "S1-NAN-1",
        "business_name": "Interstate Israel",
        "business_address": "2023 11C, Lawrence, NY",
        "country": "US"
    }])
    other = pd.DataFrame([{
        "entity_id": "S3-NAN-1",
        "business_name": "INTERSTATE ISRAEL",
        "business_address": "nan",
        "country": "US"
    }])
    cands = regression_blocker.generate_candidate_pairs(s1, other, verbose=False)
    assert "S3-NAN-1" in cands["S1-NAN-1"]


def test_case_9_leading_zero_street_number():
    """Leading zeros on street numbers (05034 vs 5034) should normalize identically."""
    pa1 = normalize_business_address("05034 19TH AVENUE, SEATTLE, WA", country="US")
    pa2 = normalize_business_address("5034 19th Ave, Seattle, Washington", country="US")
    # Street number 05034 stripped to 5034, street matches
    assert pa1["street"] == pa2["street"] == "5034 19th avenue"
    assert pa1["city"] == pa2["city"] == "seattle"
    assert pa1["state"] == pa2["state"] == "washington"


def test_case_10_prefix_legal_suffix_repositioning():
    """Legal suffix at beginning ('LLC Cornerstone Cloud Labs') should match suffix at end."""
    pn1 = normalize_business_name("Cornerstone Cloud Labs LLC")
    pn2 = normalize_business_name("LLC Cornerstone Cloud Labs")
    assert pn1["core_name"] == pn2["core_name"] == "cornerstone cloud labs"
