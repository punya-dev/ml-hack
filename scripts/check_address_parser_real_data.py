#!/usr/bin/env python3
"""
Real-Data Quality & Convergence Check for Section 5 Address Cleaning & Structuring.
1. 30 real records across US, India, France eyeballed side-by-side.
2. 50 true match pairs: checking address convergence (clean_address, city, state, postal_code, sorted_tokens).
3. Throughput benchmark on 10,000 real records.
"""

import time
import pandas as pd
from rapidfuzz import fuzz

from src.address_parser import normalize_business_address


def check_1_sample_eyeball(n=30):
    print("=" * 80)
    print(f"CHECK 1: Eyeball Sample of Real Addresses across US, India, France")
    print("=" * 80)

    s1 = pd.read_csv("dataset/val/val_source1.tsv", sep="\t")
    s2 = pd.read_csv("dataset/val/val_source2.tsv", sep="\t")
    t1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", nrows=5000)
    france_sub = t1[t1["country"].str.lower().str.contains("fr", na=False)]

    samples = []
    # Sample US and India from S1/S2
    for df, s_label in [(s1, "S1"), (s2, "S2")]:
        for c in ["US", "India"]:
            sub = df[df["country"] == c].dropna(subset=["business_address"])
            take = sub.sample(min(len(sub), 6), random_state=42)
            for _, row in take.iterrows():
                samples.append((s_label, c, row["business_address"]))

    # Sample France
    for _, row in france_sub.head(6).iterrows():
        samples.append(("Test_S1", "France", row["business_address"]))

    rows = []
    for s_label, ctry, raw in samples:
        norm = normalize_business_address(raw, country=ctry)
        rows.append({
            "Src": s_label,
            "Ctry": ctry,
            "Raw": str(raw)[:35],
            "Clean": norm["clean_address"][:30],
            "City": norm["city"],
            "State": norm["state"],
            "Zip": norm["postal_code"],
            "Landmark": norm["landmark_text"][:15],
            "SortedTok": norm["sorted_address_tokens"][:25]
        })

    df_res = pd.DataFrame(rows)
    print(df_res.to_string(index=False))
    print()


def check_2_ground_truth_convergence(n_pairs=50):
    print("=" * 80)
    print(f"CHECK 2: Ground-Truth Address Convergence on {n_pairs} True Match Pairs")
    print("=" * 80)

    gt = pd.read_csv("dataset/val/val_ground_truth.tsv", sep="\t")
    s1 = pd.read_csv("dataset/val/val_source1.tsv", sep="\t").set_index("entity_id")
    s2 = pd.read_csv("dataset/val/val_source2.tsv", sep="\t").set_index("entity_id")
    s3 = pd.read_csv("dataset/val/val_source3.tsv", sep="\t").set_index("entity_id")

    pairs = []
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if not matched_str:
            continue
        matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]

        if s1_id in s1.index:
            s1_row = s1.loc[s1_id]
            for m_id in matched_ids:
                if m_id.startswith("S2-") and m_id in s2.index:
                    s2_row = s2.loc[m_id]
                    if pd.notna(s1_row["business_address"]) and pd.notna(s2_row["business_address"]):
                        pairs.append((s1_id, m_id, s1_row["country"], s1_row["business_address"], s2_row["business_address"]))
                elif m_id.startswith("S3-") and m_id in s3.index:
                    s3_row = s3.loc[m_id]
                    if pd.notna(s1_row["business_address"]) and pd.notna(s3_row["business_address"]):
                        pairs.append((s1_id, m_id, s1_row["country"], s1_row["business_address"], s3_row["business_address"]))

        if len(pairs) >= n_pairs * 2:
            break

    test_pairs = pairs[:n_pairs]

    exact_clean = 0
    exact_sort = 0
    same_state = 0
    same_city = 0
    same_zip = 0
    lev_ge_80 = 0

    print(f"{'S1 Address':<32} | {'Matched S2/S3 Address':<32} | {'Sort Match?':<11} | {'Same State':<10} | {'Fuzz'}")
    print("-" * 105)

    for s1_id, sx_id, ctry, a1, a2 in test_pairs:
        p1 = normalize_business_address(a1, country=ctry)
        p2 = normalize_business_address(a2, country=ctry)

        m_clean = (p1["clean_address"] == p2["clean_address"])
        m_sort = (p1["sorted_address_tokens"] == p2["sorted_address_tokens"])
        s_state = (p1["state"] == p2["state"] and bool(p1["state"]))
        s_city = (p1["city"] == p2["city"] and bool(p1["city"]))
        s_zip = (p1["postal_code"] == p2["postal_code"] and bool(p1["postal_code"]))
        ratio = fuzz.token_sort_ratio(p1["clean_address"], p2["clean_address"])

        if m_clean: exact_clean += 1
        if m_sort: exact_sort += 1
        if s_state: same_state += 1
        if s_city: same_city += 1
        if s_zip: same_zip += 1
        if ratio >= 80: lev_ge_80 += 1

        print(f"{str(a1)[:30]:<32} | {str(a2)[:30]:<32} | {str(m_sort):<11} | {str(s_state):<10} | {ratio:.1f}")

    print("-" * 105)
    print("CONVERGENCE SUMMARY ON TRUE MATCH ADDRESSES:")
    print(f"  Exact Clean Address Match:   {exact_clean}/{len(test_pairs)} ({exact_clean/len(test_pairs)*100:.1f}%)")
    print(f"  Exact Sorted Tokens Match:   {exact_sort}/{len(test_pairs)} ({exact_sort/len(test_pairs)*100:.1f}%)")
    print(f"  Same Parsed State:           {same_state}/{len(test_pairs)} ({same_state/len(test_pairs)*100:.1f}%)")
    print(f"  Same Parsed City:            {same_city}/{len(test_pairs)} ({same_city/len(test_pairs)*100:.1f}%)")
    print(f"  Same Extracted Postal Code:  {same_zip}/{len(test_pairs)} ({same_zip/len(test_pairs)*100:.1f}%)")
    print(f"  Token Sort Ratio >= 80:      {lev_ge_80}/{len(test_pairs)} ({lev_ge_80/len(test_pairs)*100:.1f}%)")
    print()


def check_3_throughput(n=10000):
    print("=" * 80)
    print(f"CHECK 3: Throughput Benchmark on {n} Real Records")
    print("=" * 80)

    s1 = pd.read_csv("dataset/val/val_source1.tsv", sep="\t", nrows=n)
    start = time.time()
    res = [normalize_business_address(row["business_address"], country=row["country"]) for _, row in s1.iterrows()]
    elapsed = time.time() - start
    print(f"Processed {len(res)} addresses in {elapsed:.2f}s ({len(res)/elapsed:.0f} addresses/sec)")
    print()


if __name__ == "__main__":
    check_1_sample_eyeball(30)
    check_2_ground_truth_convergence(50)
    check_3_throughput(10000)
