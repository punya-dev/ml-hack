#!/usr/bin/env python3
"""
Unit tests for shared text normalization primitives (Section 3).
"""

from src.text_utils import (
    normalize_unicode,
    strip_decorative_noise,
    transliterate_to_latin,
    strip_domain_suffix,
    strip_abbreviation_periods,
    standardize_symbols,
    normalize_digits_and_numbers,
    deconcatenate_blob,
    collapse_whitespace,
    normalize_primitive
)


def run_tests():
    print("Running Section 3 Shared Text-Normalization Primitive Tests...")

    # 1. Unicode NFKC
    assert normalize_unicode("Ｆｕｌｌｗｉｄｔｈ") == "Fullwidth"
    assert normalize_unicode("Café") == "Café"
    print("  [x] Unicode NFKC passed.")

    # 2. Decorative Junk Stripping
    assert strip_decorative_noise(">> Lumyx Studios Llc") == "Lumyx Studios Llc"
    assert strip_decorative_noise("##233 Keyser Road") == "233 Keyser Road"
    assert strip_decorative_noise("-- Holloway Peak Inc Seafood") == "Holloway Peak Inc Seafood"
    assert strip_decorative_noise("Regional Sterling [Yhn]") == "Regional Sterling Yhn"
    assert strip_decorative_noise("*Star Electronics*") == "Star Electronics"
    print("  [x] Decorative junk stripping passed.")

    # 3. Transliteration (Hindi, Bengali, French accents via Sanscript + Schwa Deletion)
    t_hindi = transliterate_to_latin("सेवन सॉल्यूशंस प्रा. लि.")
    assert "sevn" in t_hindi.lower() and "solyushans" in t_hindi.lower()
    
    t_bengali = transliterate_to_latin("সানরাইজ সার্ভিসেস প্রাইভেট লিমিটেড")
    assert "sanraij" in t_bengali.lower() and "limited" in t_bengali.lower()
    
    t_french = transliterate_to_latin("Léarning Centre Thénard")
    assert t_french == "Learning Centre Thenard"
    
    t_state = transliterate_to_latin("महाराष्ट्र")
    assert "maharashtra" in t_state.lower()
    print("  [x] Script-detected Indic and French transliteration passed.")

    # 4. Domain / TLD suffix stripping
    assert strip_domain_suffix("northmarshall.com") == "northmarshall"
    assert strip_domain_suffix("wilfordhancock.com") == "wilfordhancock"
    assert strip_domain_suffix("regionalsterlingyhncom") == "regionalsterlingyhn"
    assert strip_domain_suffix("Summit Inc") == "Summit Inc"
    print("  [x] Domain / TLD stripping passed.")

    # 5. Abbreviation period stripping
    assert strip_abbreviation_periods("Pvt. Ltd.") == "Pvt Ltd"
    assert strip_abbreviation_periods("L.L.C.") == "LLC"
    assert strip_abbreviation_periods("P.C.") == "PC"
    assert strip_abbreviation_periods("H.No. 16") == "H No 16"
    assert strip_abbreviation_periods("Plot. 19") == "Plot 19"
    print("  [x] Abbreviation period stripping passed.")

    # 6. Symbol standardization & slash preservation
    assert "and" in standardize_symbols("B+ Retail")
    assert "and" in standardize_symbols("Grain & Fils")
    assert standardize_symbols("‘curly’ “quotes”") == "'curly' \"quotes\""
    
    # Slash preservation for t/a, f/k/a, b/h, a/k/a
    slash_text = "Sunrise f/k/a Miranex t/a Brand b/h Natraj"
    assert standardize_symbols(slash_text, preserve_slashes=True) == slash_text
    print("  [x] Symbol standardization & slash preservation passed.")

    # 7. Digit & ordinal normalization
    assert normalize_digits_and_numbers("Plot No. 53").lower() == "plot no 53"
    assert normalize_digits_and_numbers("Plot No.53").lower() == "plot no 53"
    assert normalize_digits_and_numbers("H.No 16").lower() == "h no 16"
    assert normalize_digits_and_numbers("KH NO. -570").lower() == "kh no 570"
    assert normalize_digits_and_numbers("2Nd Floor").lower() == "2 floor"
    print("  [x] Digit and ordinal normalization passed.")

    # 8. De-concatenation
    blob_res = deconcatenate_blob("regionalsterlingyhn")
    assert "regional" in blob_res and "sterling" in blob_res
    assert deconcatenate_blob("northmarshall") == "north marshall"
    assert deconcatenate_blob("microsoft") == "microsoft"
    print("  [x] De-concatenation passed.")

    # 9. End-to-end normalize_primitive pipeline
    raw_sample1 = "  >> NORTH MARSHALL L.L.C.  "
    norm1 = normalize_primitive(raw_sample1, is_name=True)
    assert norm1 == "north marshall llc"

    raw_sample2 = "Regional Sterling [Yhn] Inc."
    norm2 = normalize_primitive(raw_sample2, is_name=True)
    assert norm2 == "regional sterling yhn inc"

    raw_sample3 = "Plot No. 16, Near Lokmat Building, Wardha Road, NAGPUR, महाराष्ट्र"
    norm3 = normalize_primitive(raw_sample3, is_name=False)
    assert "plot no 16" in norm3 and "wardha road" in norm3 and "nagpur" in norm3 and "maharashtra" in norm3

    raw_sample4 = "northmarshall.com"
    norm4 = normalize_primitive(raw_sample4, is_name=True, apply_deconcat=True)
    assert norm4 == "north marshall"

    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
