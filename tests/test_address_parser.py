#!/usr/bin/env python3
"""
Unit tests for Address Cleaning & Structuring Pipeline (Section 5).
"""

from src.address_parser import (
    extract_landmark,
    extract_postal_code,
    expand_street_abbreviations,
    normalize_business_address,
    get_sorted_address_tokens,
    get_address_char_ngrams
)


def run_tests():
    print("Running Section 5 Address Cleaning & Structuring Tests...")

    # 1. Landmark extraction BEFORE abbreviation expansion
    addr1 = "Mulund Goreagon Link Road, Near Fortis Hospital, Bhandup West, Mumbai, Maharashtra"
    clean1, landmark1 = extract_landmark(addr1)
    assert "near fortis hospital" in landmark1.lower()
    assert "fortis hospital" not in clean1.lower()
    print("  [x] Landmark extraction passed.")

    # 2. Behind / Opp landmark variants
    addr2 = "303, Sakar 5 B/H Natraj Cinema Ashram Road, Ahmedabad, Gujarat"
    clean2, landmark2 = extract_landmark(addr2)
    assert "b/h natraj cinema" in landmark2.lower()
    print("  [x] Alternate landmark trigger (B/H) passed.")

    # 3. Postal code extraction BEFORE abbreviation expansion
    addr_us_zip = "1795 Westchester Dr, High Point, NC 27262"
    clean_zip, zip_code = extract_postal_code(addr_us_zip, country_code="US")
    assert zip_code == "27262"
    assert "27262" not in clean_zip
    print("  [x] US ZIP extraction passed.")

    addr_in_pin = "Bhandup West, Mumbai, Maharashtra 400078"
    clean_pin, pin_code = extract_postal_code(addr_in_pin, country_code="India")
    assert pin_code == "400078"
    assert "400078" not in clean_pin
    print("  [x] India PIN extraction passed.")

    # 4. Street abbreviation expansion
    expanded = expand_street_abbreviations("105 Elm St, 282 Saxony Dr, 914 Pierpont Ave, 12 Sunset Blvd")
    assert "elm street" in expanded.lower()
    assert "saxony drive" in expanded.lower()
    assert "pierpont avenue" in expanded.lower()
    assert "sunset boulevard" in expanded.lower()
    print("  [x] Street abbreviation expansion passed.")

    # 5. Full US Address Parsing (Standard: Street, City, State)
    us_res1 = normalize_business_address("1795 Westchester Dr, High Point, NC", country="US")
    assert "westchester drive" in us_res1["clean_address"]
    assert us_res1["state"] == "north carolina"
    assert us_res1["city"] == "high point"
    assert us_res1["has_state"] is True
    assert us_res1["has_city"] is True
    assert us_res1["has_street"] is True
    print("  [x] US standard address parsing passed.")

    # 6. US Address Component Inversion (State, City, Street)
    us_res2 = normalize_business_address("OH, Columbus, 5559 Orville Avenue", country="US")
    assert us_res2["state"] == "ohio"
    assert us_res2["city"] == "columbus"
    assert "5559 orville avenue" in us_res2["street"]
    print("  [x] US inverted address parsing (State, City, Street) passed.")

    # 7. Indian Address Parsing (Locality, City, State + Landmark)
    in_res = normalize_business_address(
        "2505, Tower 1, Oakwood, Mulund Link Rd, Near Fortis Hospital, Mumbai, Maharashtra",
        country="India"
    )
    assert in_res["state"] == "maharashtra"
    assert in_res["city"] == "mumbai"
    assert "near fortis hospital" in in_res["landmark_text"]
    assert "fortis hospital" not in in_res["clean_address"]
    assert "mulund link road" in in_res["clean_address"]
    print("  [x] India address parsing with landmark passed.")

    # 8. Component Reordering Invariance (sorted_address_tokens)
    inv1 = normalize_business_address("OH, Columbus, 5559 Orville Avenue", country="US")
    inv2 = normalize_business_address("5559 Orville Avenue, Columbus, OH", country="US")
    assert inv1["sorted_address_tokens"] == inv2["sorted_address_tokens"]
    print("  [x] Component reordering invariance passed.")

    # 9. Generic Fallback Parser for Open Set (France)
    fr_res = normalize_business_address(
        "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine",
        country="France"
    )
    assert fr_res["state"] == "nouvelle-aquitaine"
    assert fr_res["city"] == "bordeaux"
    assert "boulevard" in fr_res["clean_address"]
    print("  [x] Generic fallback parser (France open set) passed.")

    # 10. Explicit Missingness (no placeholder 'unknown')
    empty_res = normalize_business_address(None, country="US")
    assert empty_res["street"] == ""
    assert empty_res["city"] == ""
    assert empty_res["state"] == ""
    assert empty_res["postal_code"] == ""
    assert empty_res["has_postal_code"] is False
    assert empty_res["has_city"] is False
    assert empty_res["has_state"] is False
    assert "unknown" not in str(empty_res).lower()
    # 11. On-Demand Character N-Grams
    ngrams = get_address_char_ngrams("1795 westchester drive", n=3)
    assert "wes" in ngrams and "est" in ngrams
    print("  [x] On-demand character n-grams passed.")

    # 12. Anchor-based parsing without commas (unpunctuated Indian address)
    unp_res = normalize_business_address("H NO 204 C ROAD HOSHIARPUR PUNJAB Punjab", country="India")
    assert unp_res["state"] == "punjab"
    assert "hoshiarpur" in unp_res["city"]
    assert "h no 204 c road" in unp_res["street"]
    print("  [x] Anchor-based parsing without commas passed.")

    # 13. Leading zero stripping on street/house numbers
    zero_res1 = normalize_business_address("02814 Cherry St, Tuscaloosa, AL", country="US")
    assert "2814 cherry street" in zero_res1["clean_address"]
    assert zero_res1["state"] == "alabama"

    zero_res2 = normalize_business_address("House No. 01689, Sector 14, Gurgaon, Haryana", country="India")
    assert "house no 1689" in zero_res2["clean_address"]
    print("  [x] Leading zero stripping on street numbers passed.")

    # 14. Middle state anchor (City, State, Street)
    mid_res = normalize_business_address("GREENSBORO, NC, 19 1/2 STARDUST TRAIL", country="US")
    assert mid_res["state"] == "north carolina"
    assert mid_res["city"] == "greensboro"
    assert "19 1/2 stardust trail" in mid_res["street"]
    print("  [x] Middle state anchor (City, State, Street) passed.")

    print("\nALL SECTION 5 ADDRESS PIPELINE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()

