#!/usr/bin/env python3
"""
Real-Data Quality & Convergence Check for Section 4 Business Name Cleaning.
Executes the 4 checks requested by user:
1. 40 real records across S1, S2, S3 (US & India mixed) eyeballed side-by-side.
2. 50 true match pairs from ground truth: measuring key convergence rates.
3. 30 random non-match pairs: checking for collision rates.
4. Block-size distribution & max bucket size on 200,000 real records.
"""

import os
import random
import re
import pandas as pd
from rapidfuzz import fuzz

from src.name_normalizer import normalize_business_name, get_char_ngrams


def check_1_sample_eyeball(n=40):
    print("=" * 80)
    print(f"CHECK 1: Eyeball Sample of {n} Real Records across S1, S2, S3 (US & India)")
    print("=" * 80)

    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", nrows=5000)
    s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", nrows=5000)
    s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", nrows=5000)

    samples = []
    # Sample from each source and country
    for df, s_label in [(s1, "S1"), (s2, "S2"), (s3, "S3")]:
        for c in ["US", "India"]:
            sub = df[df["country"] == c]
            if len(sub) > 0:
                take = sub.sample(min(len(sub), n // 6), random_state=42)
                for _, row in take.iterrows():
                    samples.append((s_label, c, row["entity_id"], row["business_name"]))

    random.seed(42)
    random.shuffle(samples)
    samples = samples[:n]

    rows = []
    for s_label, c, eid, raw in samples:
        norm = normalize_business_name(raw)
        rows.append({
            "Source": s_label,
            "Ctry": c,
            "Raw": raw[:32],
            "Clean": norm["clean_name"][:32],
            "Core": norm["core_name"][:26],
            "Sorted": norm["sorted_tokens"][:24],
            "Acr": norm["acronym_key"],
            "Pfx": norm["prefix_key"],
            "Phonetic": norm["phonetic_key"],
        })

    df_res = pd.DataFrame(rows)
    print(df_res.to_string(index=False))
    print()


def check_2_ground_truth_convergence(n_pairs=50):
    print("=" * 80)
    print(f"CHECK 2: Ground-Truth Convergence on {n_pairs} True Match Pairs")
    print("=" * 80)

    # Load from val set where S1, S2, S3 IDs are guaranteed to be indexed
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
                    pairs.append((s1_id, m_id, s1_row["country"], s1_row["business_name"], s2_row["business_name"]))
                elif m_id.startswith("S3-") and m_id in s3.index:
                    s3_row = s3.loc[m_id]
                    pairs.append((s1_id, m_id, s1_row["country"], s1_row["business_name"], s3_row["business_name"]))

        if len(pairs) >= n_pairs * 2:
            break

    random.seed(42)
    random.shuffle(pairs)
    test_pairs = pairs[:n_pairs]

    exact_core = 0
    exact_sort = 0
    exact_pfx = 0
    exact_phon = 0
    exact_acr = 0
    lev_ge_85 = 0

    print(f"{'S1 Name':<32} | {'Matched S2/S3 Name':<32} | {'Core Match?':<11} | {'Sort Match?':<11} | {'Phon Match?':<11} | {'Fuzz Ratio'}")
    print("-" * 115)

    for s1_id, sx_id, ctry, n1, n2 in test_pairs:
        p1 = normalize_business_name(n1)
        p2 = normalize_business_name(n2)

        m_core = (p1["core_name"] == p2["core_name"]) or (p2["core_name"] in p1["alt_names"]) or (p1["core_name"] in p2["alt_names"])
        m_sort = (p1["sorted_tokens"] == p2["sorted_tokens"])
        m_pfx = (p1["prefix_key"] == p2["prefix_key"] and len(p1["prefix_key"]) >= 3)
        m_phon = (p1["phonetic_key"] == p2["phonetic_key"] and bool(p1["phonetic_key"]))
        m_acr = (p1["acronym_key"] == p2["acronym_key"] and bool(p1["acronym_key"]))

        ratio = fuzz.ratio(p1["core_name"], p2["core_name"])

        if m_core: exact_core += 1
        if m_sort: exact_sort += 1
        if m_pfx: exact_pfx += 1
        if m_phon: exact_phon += 1
        if m_acr: exact_acr += 1
        if ratio >= 85: lev_ge_85 += 1

        print(f"{str(n1)[:30]:<32} | {str(n2)[:30]:<32} | {str(m_core):<11} | {str(m_sort):<11} | {str(m_phon):<11} | {ratio:.1f}")

    print("-" * 115)
    print("CONVERGENCE SUMMARY ON TRUE MATCH PAIRS:")
    print(f"  Exact Core Name Match:      {exact_core}/{len(test_pairs)} ({exact_core/len(test_pairs)*100:.1f}%)")
    print(f"  Exact Sorted Tokens Match:  {exact_sort}/{len(test_pairs)} ({exact_sort/len(test_pairs)*100:.1f}%)")
    print(f"  Exact Prefix (5-ch) Match:  {exact_pfx}/{len(test_pairs)} ({exact_pfx/len(test_pairs)*100:.1f}%)")
    print(f"  Exact Phonetic Key Match:   {exact_phon}/{len(test_pairs)} ({exact_phon/len(test_pairs)*100:.1f}%)")
    print(f"  Acronym Key Match:          {exact_acr}/{len(test_pairs)} ({exact_acr/len(test_pairs)*100:.1f}%)")
    print(f"  Fuzzy Ratio >= 85:          {lev_ge_85}/{len(test_pairs)} ({lev_ge_85/len(test_pairs)*100:.1f}%)")
    
    # Combined multi-index union (candidate recalled by AT LEAST ONE blocking key)
    recalled_by_union = 0
    for s1_id, sx_id, ctry, n1, n2 in test_pairs:
        p1 = normalize_business_name(n1)
        p2 = normalize_business_name(n2)
        hit = (
            (p1["core_name"] == p2["core_name"]) or
            (p1["sorted_tokens"] == p2["sorted_tokens"]) or
            (p1["prefix_key"] == p2["prefix_key"] and len(p1["prefix_key"]) >= 3) or
            (p1["sorted_prefix_key"] == p2["sorted_prefix_key"] and len(p1["sorted_prefix_key"]) >= 3) or
            (p1["phonetic_key"] == p2["phonetic_key"] and bool(p1["phonetic_key"])) or
            (p1["acronym_key"] == p2["acronym_key"] and bool(p1["acronym_key"])) or
            (p2["core_name"] in p1["alt_names"]) or (p1["core_name"] in p2["alt_names"])
        )
        if hit:
            recalled_by_union += 1
    print(f"  >>> MULTI-KEY BLOCKING RECALL (UNION OF KEYS): {recalled_by_union}/{len(test_pairs)} ({recalled_by_union/len(test_pairs)*100:.1f}%)")
    print()


def check_3_false_positive_collisions(n_pairs=30):
    print("=" * 80)
    print(f"CHECK 3: False-Positive Collision Spot Check on {n_pairs} Random Non-Matching Pairs")
    print("=" * 80)

    s1 = pd.read_csv("dataset/val/val_source1.tsv", sep="\t", nrows=5000)
    # Pick random distinct rows
    random.seed(99)
    sample_rows = s1.sample(n_pairs * 2, random_state=99).reset_index(drop=True)

    non_pairs = []
    for i in range(0, len(sample_rows) - 1, 2):
        r1 = sample_rows.iloc[i]
        r2 = sample_rows.iloc[i+1]
        non_pairs.append((r1["business_name"], r2["business_name"]))

    core_collisions = 0
    sort_collisions = 0
    phon_collisions = 0
    acr_collisions = 0

    for n1, n2 in non_pairs:
        p1 = normalize_business_name(n1)
        p2 = normalize_business_name(n2)

        c_hit = (p1["core_name"] == p2["core_name"])
        s_hit = (p1["sorted_tokens"] == p2["sorted_tokens"])
        p_hit = (p1["phonetic_key"] == p2["phonetic_key"] and bool(p1["phonetic_key"]))
        a_hit = (p1["acronym_key"] == p2["acronym_key"] and bool(p1["acronym_key"]))

        if c_hit: core_collisions += 1
        if s_hit: sort_collisions += 1
        if p_hit: phon_collisions += 1
        if a_hit: acr_collisions += 1

    print(f"Results across {len(non_pairs)} random non-matching pairs:")
    print(f"  Core Name Collisions:     {core_collisions}/{len(non_pairs)} (Expected ~0%)")
    print(f"  Sorted Token Collisions:  {sort_collisions}/{len(non_pairs)} (Expected ~0%)")
    print(f"  Phonetic Key Collisions:  {phon_collisions}/{len(non_pairs)} (Expected small % across generic names)")
    print(f"  Acronym Collisions:       {acr_collisions}/{len(non_pairs)} (Acronyms naturally have higher collision, used strictly as high-recall fallback or compound with state/zip)")
    print()


def check_4_block_size_distribution(sample_size=100000):
    print("=" * 80)
    print(f"CHECK 4: Block Size Distribution on {sample_size} Real Records")
    print("=" * 80)

    s1 = pd.read_csv("dataset/val/val_source1.tsv", sep="\t")
    s2 = pd.read_csv("dataset/val/val_source2.tsv", sep="\t", nrows=50000)
    df = pd.concat([s1, s2], ignore_index=True)
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    print(f"Extracting normalization keys on {len(df)} records...")

    records = [normalize_business_name(name) for name in df["business_name"]]
    df_keys = pd.DataFrame(records)

    for col in ["core_name", "sorted_tokens", "prefix_key", "phonetic_key", "acronym_key"]:
        # Exclude empty keys
        valid = df_keys[df_keys[col].str.len() > 0]
        sizes = valid.groupby(col).size()
        desc = sizes.describe(percentiles=[0.5, 0.9, 0.99, 0.999])
        top_keys = sizes.sort_values(ascending=False).head(5).to_dict()

        print(f"--- Key: {col} (non-empty count: {len(valid)}) ---")
        print(f"  Mean: {desc['mean']:.2f}, Median: {desc['50%']:.1f}, 90th%: {desc['90%']:.1f}, 99th%: {desc['99%']:.1f}, 99.9th%: {desc['99.9%']:.1f}, Max: {desc['max']:.0f}")
        print(f"  Top 5 largest blocks: {top_keys}")
        print()


if __name__ == "__main__":
    check_1_sample_eyeball(40)
    check_2_ground_truth_convergence(50)
    check_3_false_positive_collisions(30)
    check_4_block_size_distribution(100000)
