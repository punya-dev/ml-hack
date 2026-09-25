#!/usr/bin/env python3
"""
Unit tests for Business Name Cleaning Pipeline (Section 4).
"""

from src.name_normalizer import normalize_business_name


def run_tests():
    print("Running Section 4 Business Name Cleaning Tests...")

    # 1. Legal suffix expansion & core_name stripping
    res1 = normalize_business_name("Seven Solutions Pvt Ltd")
    assert res1["clean_name"] == "seven solutions private limited"
    assert res1["core_name"] == "seven solutions"
    print("  [x] Legal suffix expansion and core_name stripping passed.")

    # 2. Suffix repositioning (suffix at beginning)
    res2 = normalize_business_name("LLC North Marshall")
    assert res2["core_name"] == "north marshall"
    assert res2["clean_name"] == "north marshall limited liability company"
    print("  [x] Suffix repositioning passed.")

    # 3. Guard against empty/too-short core_name
    res3 = normalize_business_name("LLC")
    assert len(res3["core_name"]) >= 3
    print("  [x] Short core_name guard passed.")

    # 4. Token-order normalization
    res_ord1 = normalize_business_name("Star Electronics")
    res_ord2 = normalize_business_name("Electronics Star")
    assert res_ord1["sorted_tokens"] == res_ord2["sorted_tokens"] == "electronics star"
    print("  [x] Token order normalization passed.")

    # 5. Acronym generation
    res_acronym = normalize_business_name("International Business Machines")
    assert res_acronym["acronym_key"] == "ibm"
    print("  [x] Acronym generation passed.")

    # 6. Prefix key
    res_prefix = normalize_business_name("North Marshall")
    assert res_prefix["prefix_key"] == "north"
    print("  [x] Prefix key passed.")

    # 7. Phonetic code & typo tolerance
    res_p1 = normalize_business_name("North Marshall")
    res_p2 = normalize_business_name("North Marshel")
    assert res_p1["phonetic_key"] == res_p2["phonetic_key"]
    print("  [x] Phonetic Soundex typo tolerance passed.")

    # 8. DBA / Trade name splitting & (India) qualifier filter
    res_dba = normalize_business_name("Miranex formerly known as Sunrise Services Private Limited")
    assert res_dba["core_name"] == "miranex"
    assert any("sunrise services" in alt for alt in res_dba["alt_names"])

    # Verify (India) is filtered and NOT emitted as a trade name
    res_geo = normalize_business_name("Toyota Kirloskar Motor (India) Pvt Ltd")
    assert "india" not in res_geo["alt_names"]

    # Verify real trade name in parentheses IS captured
    res_paren = normalize_business_name("Acme Corp (Rocket Supplies)")
    assert any("rocket supplies" in alt for alt in res_paren["alt_names"])
    print("  [x] DBA splitting and (India) filtering passed.")

    # 9. Indic business name through full pipeline
    res_indic = normalize_business_name("राम मार्केटिंग प्राइवेट लिमिटेड")
    assert "ram marketing" in res_indic["core_name"]
    print("  [x] Indic business name pipeline passed.")

    # 10. Sorted prefix key & char n-grams
    from src.name_normalizer import get_char_ngrams
    res_sort_pfx = normalize_business_name("Electronics Star")
    assert res_sort_pfx["sorted_prefix_key"] == "elect"
    ngrams = get_char_ngrams("acme corp", n=3)
    assert "acm" in ngrams and "cme" in ngrams and "orp" in ngrams
    print("  [x] Sorted prefix key and matching-stage char n-grams passed.")

    # 11. Comma before suffix handling
    res_comma = normalize_business_name("Acme Oncology, Inc.")
    assert res_comma["clean_name"] == "acme oncology incorporated"
    assert res_comma["core_name"] == "acme oncology"
    print("  [x] Comma before legal suffix handling passed.")

    print("\nALL SECTION 4 BUSINESS NAME TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()

