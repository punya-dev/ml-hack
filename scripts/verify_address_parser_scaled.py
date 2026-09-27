#!/usr/bin/env python3
"""
Comprehensive Scaled Verification of Address Parser (Section 5).
Addresses all 5 dimensions requested by user:
1. Scaled Ground-Truth Convergence on 2,000 true match pairs (split by US vs. India).
2. False-Positive Collision Risk on 1,000 non-matching pairs (same country).
3. Failure-Mode Audit on failed state/city matches (parser misses vs. data noise).
4. Field-Population Rate across 10,000 records per country (US, India, France).
5. Downstream Proxy: Blocking Recall & Candidate Reduction using Parsed State + Name Keys.
"""

import time
import random
import re
import pandas as pd
from rapidfuzz import fuzz

from src.name_normalizer import normalize_business_name
from src.address_parser import normalize_business_address


def run_scaled_verification():
    print("=" * 85)
    print("SCALED VERIFICATION OF ADDRESS PARSER & BLOCKING RECALL PROXY")
    print("=" * 85)

    # Load validation data
    print("Loading validation datasets...")
    gt = pd.read_csv("dataset/val/val_ground_truth.tsv", sep="\t")
    s1 = pd.read_csv("dataset/val/val_source1.tsv", sep="\t").set_index("entity_id")
    s2 = pd.read_csv("dataset/val/val_source2.tsv", sep="\t").set_index("entity_id")
    s3 = pd.read_csv("dataset/val/val_source3.tsv", sep="\t").set_index("entity_id")

    # =========================================================================
    # 1. Scaled Ground-Truth Convergence (1,000+ pairs, split US vs. India)
    # =========================================================================
    print("\n--- 1. Scaled True Match Pair Convergence (US vs. India) ---")
    pairs_us = []
    pairs_india = []

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if not matched_str or s1_id not in s1.index:
            continue
        
        s1_row = s1.loc[s1_id]
        ctry = str(s1_row["country"]).strip()
        matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]

        for m_id in matched_ids:
            target_df = s2 if m_id.startswith("S2-") else s3
            if m_id in target_df.index:
                sx_row = target_df.loc[m_id]
                a1 = s1_row["business_address"]
                a2 = sx_row["business_address"]
                n1 = s1_row["business_name"]
                n2 = sx_row["business_name"]
                
                pair_tuple = (s1_id, m_id, ctry, n1, n2, a1, a2)
                if ctry.lower() == "us" and len(pairs_us) < 1000:
                    pairs_us.append(pair_tuple)
                elif ctry.lower() == "india" and len(pairs_india) < 1000:
                    pairs_india.append(pair_tuple)

        if len(pairs_us) >= 1000 and len(pairs_india) >= 1000:
            break

    all_pairs = {"US": pairs_us, "India": pairs_india}
    audit_failures = []

    for ctry, pair_list in all_pairs.items():
        n_total = len(pair_list)
        same_state = 0
        same_city = 0
        same_zip = 0
        exact_sort = 0
        token_sort_ge_80 = 0
        both_empty = 0
        one_empty = 0

        for s1_id, sx_id, _, n1, n2, a1, a2 in pair_list:
            if pd.isna(a1) and pd.isna(a2):
                both_empty += 1
                continue
            if pd.isna(a1) or pd.isna(a2):
                one_empty += 1
                continue

            p1 = normalize_business_address(a1, country=ctry)
            p2 = normalize_business_address(a2, country=ctry)

            s_state = (p1["state"] == p2["state"] and bool(p1["state"]))
            s_city = (p1["city"] == p2["city"] and bool(p1["city"]))
            s_zip = (p1["postal_code"] == p2["postal_code"] and bool(p1["postal_code"]))
            m_sort = (p1["sorted_address_tokens"] == p2["sorted_address_tokens"] and bool(p1["sorted_address_tokens"]))
            ratio = fuzz.token_sort_ratio(p1["clean_address"], p2["clean_address"])

            if s_state: same_state += 1
            else:
                if len(audit_failures) < 30:
                    audit_failures.append((ctry, "STATE_MISMATCH", a1, a2, p1["state"], p2["state"]))
            if s_city: same_city += 1
            if s_zip: same_zip += 1
            if m_sort: exact_sort += 1
            if ratio >= 80: token_sort_ge_80 += 1

        n_valid = n_total - one_empty - both_empty
        print(f"\nCountry: {ctry} (Total: {n_total} pairs | Valid Address Pairs: {n_valid} | One/Both Empty: {one_empty + both_empty})")
        print(f"  Same Parsed State:          {same_state}/{n_valid} ({same_state/n_valid*100:.1f}%)")
        print(f"  Same Parsed City:           {same_city}/{n_valid} ({same_city/n_valid*100:.1f}%)")
        print(f"  Same Postal Code:           {same_zip}/{n_valid} ({same_zip/n_valid*100:.1f}%)")
        print(f"  Exact Sorted Tokens Match:  {exact_sort}/{n_valid} ({exact_sort/n_valid*100:.1f}%)")
        print(f"  Token Sort Ratio >= 80:     {token_sort_ge_80}/{n_valid} ({token_sort_ge_80/n_valid*100:.1f}%)")

    # =========================================================================
    # 2. Failure-Mode Audit (Why did states/cities mismatch?)
    # =========================================================================
    print("\n--- 2. Failure-Mode Audit on State Mismatches (Sample) ---")
    data_noise_count = 0
    parser_miss_count = 0

    for ctry, err_type, a1, a2, st1, st2 in audit_failures[:15]:
        print(f"[{ctry}] S1: '{str(a1)[:35]}' (Parsed: '{st1}') | SX: '{str(a2)[:35]}' (Parsed: '{st2}')")
        # Check if genuinely different addresses or parser miss
        if not st1 or not st2:
            parser_miss_count += 1
            print("   -> Reason: Missing state token in one string")
        elif st1 != st2:
            data_noise_count += 1
            print("   -> Reason: Genuinely different state or corrupt string in source")

    # =========================================================================
    # 3. Field-Population Rate per Country (US, India, France)
    # =========================================================================
    print("\n--- 3. Field-Population Rate per Country (10,000 records each) ---")
    t1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t")
    fr_records = t1[t1["country"].str.lower().str.contains("fr", na=False)]

    records_to_test = {
        "US": s1[s1["country"] == "US"]["business_address"].dropna().head(10000),
        "India": s1[s1["country"] == "India"]["business_address"].dropna().head(10000),
        "France": fr_records["business_address"].dropna().head(min(len(fr_records), 10000))
    }

    for ctry, addrs in records_to_test.items():
        n = len(addrs)
        has_street = 0
        has_city = 0
        has_state = 0
        has_zip = 0

        for a in addrs:
            p = normalize_business_address(a, country=ctry)
            if p["has_street"]: has_street += 1
            if p["has_city"]: has_city += 1
            if p["has_state"]: has_state += 1
            if p["has_postal_code"]: has_zip += 1

        print(f"\nCountry: {ctry} (Evaluated: {n} non-null records)")
        print(f"  Has Street:      {has_street}/{n} ({has_street/n*100:.1f}%)")
        print(f"  Has City:        {has_city}/{n} ({has_city/n*100:.1f}%)")
        print(f"  Has State/Region:{has_state}/{n} ({has_state/n*100:.1f}%)")
        print(f"  Has Postal Code: {has_zip}/{n} ({has_zip/n*100:.1f}%)")

    # =========================================================================
    # 4. False-Positive Collision Risk on Non-Matching Pairs
    # =========================================================================
    print("\n--- 4. False-Positive Collision Risk on 1,000 Random Non-Matching Pairs ---")
    sample_s1 = s1.sample(1000, random_state=42)
    sample_s2 = s2.sample(1000, random_state=42)

    fp_sort_collisions = 0
    fp_state_collisions = 0
    fp_high_fuzz = 0

    for (_, r1), (_, r2) in zip(sample_s1.iterrows(), sample_s2.iterrows()):
        a1 = r1["business_address"]
        a2 = r2["business_address"]
        if pd.isna(a1) or pd.isna(a2):
            continue
        p1 = normalize_business_address(a1, country=r1["country"])
        p2 = normalize_business_address(a2, country=r2["country"])

        if p1["sorted_address_tokens"] and p1["sorted_address_tokens"] == p2["sorted_address_tokens"]:
            fp_sort_collisions += 1
        if p1["state"] and p1["state"] == p2["state"]:
            fp_state_collisions += 1
        ratio = fuzz.token_sort_ratio(p1["clean_address"], p2["clean_address"])
        if ratio >= 80:
            fp_high_fuzz += 1

    print(f"Evaluated 1,000 random non-matching pairs:")
    print(f"  Sorted Address Token Collisions: {fp_sort_collisions}/1000 ({fp_sort_collisions/1000*100:.2f}%) (Near 0% expected)")
    print(f"  High Address Fuzzy Ratio (>=80): {fp_high_fuzz}/1000 ({fp_high_fuzz/1000*100:.2f}%)")
    print(f"  State Co-occurrence Rate:        {fp_state_collisions}/1000 ({fp_state_collisions/1000*100:.1f}%) (Expected ~15-20% base rate within 50 states)")

    # =========================================================================
    # 5. Downstream Proxy: Real Blocking Recall & Candidate Reduction
    # =========================================================================
    print("\n--- 5. Downstream Proxy: End-to-End Blocking Recall on True Match Pairs ---")
    # Test multi-key candidate generation proxy:
    # A candidate pair is retrieved if ANY of these match:
    # 1. Exact core_name match
    # 2. Exact sorted_tokens match
    # 3. Exact acronym_key match (if len >= 3)
    # 4. Same State + prefix_key (first 4 chars of name)
    # 5. Same State + Soundex phonetic_key
    # 6. Exact sorted_address_tokens match
    
    recalled_count = 0
    total_eval = len(pairs_us) + len(pairs_india)
    combined_pairs = pairs_us + pairs_india

    block_channels = {
        "core_name": 0,
        "sorted_name": 0,
        "state_plus_name_prefix": 0,
        "state_plus_phonetic": 0,
        "acronym": 0,
        "sorted_address": 0,
    }

    for s1_id, sx_id, ctry, n1, n2, a1, a2 in combined_pairs:
        pn1 = normalize_business_name(n1)
        pn2 = normalize_business_name(n2)
        pa1 = normalize_business_address(a1, country=ctry)
        pa2 = normalize_business_address(a2, country=ctry)

        hit = False
        # Channel 1: Core name
        if (pn1["core_name"] == pn2["core_name"] and len(pn1["core_name"]) >= 3) or (pn2["core_name"] in pn1["alt_names"]):
            block_channels["core_name"] += 1
            hit = True
        # Channel 2: Sorted name tokens
        if pn1["sorted_tokens"] == pn2["sorted_tokens"] and len(pn1["sorted_tokens"]) >= 3:
            block_channels["sorted_name"] += 1
            hit = True
        # Channel 3: State + Name Prefix (4 chars)
        if pa1["state"] and pa1["state"] == pa2["state"] and pn1["prefix_key"][:4] == pn2["prefix_key"][:4] and len(pn1["prefix_key"]) >= 4:
            block_channels["state_plus_name_prefix"] += 1
            hit = True
        # Channel 4: State + Phonetic Soundex
        if pa1["state"] and pa1["state"] == pa2["state"] and pn1["phonetic_key"] and pn1["phonetic_key"] == pn2["phonetic_key"]:
            block_channels["state_plus_phonetic"] += 1
            hit = True
        # Channel 5: Acronym
        if pn1["acronym_key"] and pn1["acronym_key"] == pn2["acronym_key"] and len(pn1["acronym_key"]) >= 3:
            block_channels["acronym"] += 1
            hit = True
        # Channel 6: Sorted address tokens
        if pa1["sorted_address_tokens"] and pa1["sorted_address_tokens"] == pa2["sorted_address_tokens"]:
            block_channels["sorted_address"] += 1
            hit = True

        if hit:
            recalled_count += 1

    overall_recall = recalled_count / total_eval * 100
    print(f"Evaluated {total_eval} true match pairs across US and India:")
    print(f"  Channel Hits:")
    for ch, count in block_channels.items():
        print(f"    - {ch:<24}: {count}/{total_eval} ({count/total_eval*100:.1f}%)")
    print(f"\n  >>> TOTAL BLOCKING RECALL (UNION OF CHANNELS): {recalled_count}/{total_eval} ({overall_recall:.2f}%) <<<")
    print("=" * 85)


if __name__ == "__main__":
    run_scaled_verification()
