"""
Unit tests for CandidateBlocker.
"""
import os
import tempfile
import pandas as pd
import pytest
from src.blocker import CandidateBlocker


@pytest.fixture
def sample_data():
    s1 = pd.DataFrame([
        {
            "entity_id": "S1-001",
            "business_name": "Apex Digital Solutions Inc",
            "business_address": "120 Market Street, San Jose, CA",
            "country": "US"
        },
        {
            "entity_id": "S1-002",
            "business_name": "Shree Ganesh Enterprises",
            "business_address": "Shop 4, MG Road, Pune, Maharashtra",
            "country": "India"
        },
        {
            "entity_id": "S1-003",
            "business_name": "Boulangerie Patisserie Parisienne",
            "business_address": "15 Rue de Rivoli, Paris",
            "country": "France"
        }
    ])

    other = pd.DataFrame([
        # Near-miss on S1-001: typo in name and address (would fail exact-only matching)
        {
            "entity_id": "S2-101",
            "business_name": "Apex Digitl Solutns",
            "business_address": "120 Market St, San Jose, California",
            "country": "US"
        },
        # Exact match on S1-002
        {
            "entity_id": "S3-201",
            "business_name": "Shree Ganesh Enterprises Pvt Ltd",
            "business_address": "MG Rd, Pune, MH",
            "country": "India"
        },
        # Match for French entity S1-003
        {
            "entity_id": "S2-301",
            "business_name": "Boulangerie Parisienne Patisserie",
            "business_address": "15 Rue de Rivoli, Paris",
            "country": "France"
        },
        # Irrelevant entity
        {
            "entity_id": "S3-999",
            "business_name": "Completely Different Bookstore",
            "business_address": "99 Broadway, New York, NY",
            "country": "US"
        }
    ])

    return s1, other


def test_candidate_blocker_retrieval(sample_data):
    s1, other = sample_data
    blocker = CandidateBlocker(fuzzy_threshold=0.30)
    candidates = blocker.generate_candidate_pairs(s1, other, verbose=False)

    assert "S1-001" in candidates
    assert "S1-002" in candidates
    assert "S1-003" in candidates

    # Check S1-001 retrieved near-miss S2-101 via fuzzy channel
    assert "S2-101" in candidates["S1-001"], f"Expected S2-101 in candidates for S1-001, got {candidates['S1-001']}"

    # Check S1-002 retrieved S3-201
    assert "S3-201" in candidates["S1-002"], f"Expected S3-201 in candidates for S1-002, got {candidates['S1-002']}"

    # Check S1-003 retrieved S2-301
    assert "S2-301" in candidates["S1-003"], f"Expected S2-301 in candidates for S1-003, got {candidates['S1-003']}"

    # Check irrelevant entity S3-999 is NOT matched to S1-002 or S1-003
    assert "S3-999" not in candidates["S1-002"]
    assert "S3-999" not in candidates["S1-003"]


def test_candidate_pairs_tsv_format(sample_data):
    s1, other = sample_data
    blocker = CandidateBlocker(fuzzy_threshold=0.30)
    candidates = blocker.generate_candidate_pairs(s1, other, verbose=False)

    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        blocker.save_candidate_pairs(candidates, tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert lines[0].strip() == "source1_entity_id\tcandidate_entity_ids"
        assert len(lines) == 4  # Header + 3 entities
        for line in lines[1:]:
            parts = line.strip().split("\t")
            assert len(parts) >= 1
            assert parts[0].startswith("S1-")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_mixed_raw_country_spellings_blocking():
    """
    Verify that records with different raw country spellings ('United States' vs 'USA',
    'Bharat' vs 'IND', 'fr' vs 'FRA') normalize and match in the same partition.
    """
    s1 = pd.DataFrame([
        {
            "entity_id": "S1-USA-TEST",
            "business_name": "Liberty Mutual Insurance",
            "business_address": "175 Berkeley St, Boston, MA",
            "country": "United States"  # Raw spelled out
        },
        {
            "entity_id": "S1-IND-TEST",
            "business_name": "Tata Consultancy Services",
            "business_address": "TCS House, Raveline Street, Mumbai, MH",
            "country": "Bharat"  # Raw alternate name
        },
        {
            "entity_id": "S1-FRA-TEST",
            "business_name": "Carrefour Hypermarche",
            "business_address": "Massy, Ile-de-France",
            "country": "fr"  # Lowercase 2-letter
        }
    ])

    other = pd.DataFrame([
        {
            "entity_id": "S2-USA-MATCH",
            "business_name": "Liberty Mutual",
            "business_address": "175 Berkeley Street, Boston, Massachusetts",
            "country": "USA"  # 3-letter abbreviation
        },
        {
            "entity_id": "S3-IND-MATCH",
            "business_name": "Tata Consultancy Services Ltd",
            "business_address": "TCS House, Raveline St, Mumbai, Maharashtra",
            "country": "IND"  # 3-letter abbreviation
        },
        {
            "entity_id": "S2-FRA-MATCH",
            "business_name": "Carrefour",
            "business_address": "Massy, France",
            "country": "FRA"  # 3-letter uppercase
        }
    ])

    blocker = CandidateBlocker(fuzzy_threshold=0.30)
    candidates = blocker.generate_candidate_pairs(s1, other, verbose=False)

    assert "S2-USA-MATCH" in candidates["S1-USA-TEST"], f"Failed to match USA pair across 'United States' and 'USA'. Got: {candidates['S1-USA-TEST']}"
    assert "S3-IND-MATCH" in candidates["S1-IND-TEST"], f"Failed to match India pair across 'Bharat' and 'IND'. Got: {candidates['S1-IND-TEST']}"
    assert "S2-FRA-MATCH" in candidates["S1-FRA-TEST"], f"Failed to match France pair across 'fr' and 'FRA'. Got: {candidates['S1-FRA-TEST']}"

