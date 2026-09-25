"""
Unit tests for CountryNormalizer.
"""
import pytest
from src.country_normalizer import CountryNormalizer, normalize_country


def test_standard_us():
    cases = ["US", "us", "USA", "usa", "U.S.", "United States", "united states of america"]
    for c in cases:
        assert normalize_country(c) == "US"


def test_standard_india():
    cases = ["India", "india", "IN", "in", "IND", "ind", "bharat"]
    for c in cases:
        assert normalize_country(c) == "India"


def test_standard_france():
    cases = ["France", "FRANCE", "fr", "fra", "Fra"]
    for c in cases:
        assert normalize_country(c) == "France"


def test_open_set_countries():
    assert normalize_country("DE") == "DE"
    assert normalize_country("germany") == "Germany"
    assert normalize_country("japan") == "Japan"
    assert normalize_country("GBR") == "GBR"


def test_missing_and_empty():
    assert normalize_country(None) == "UNKNOWN"
    assert normalize_country("") == "UNKNOWN"
    assert normalize_country("   ") == "UNKNOWN"
