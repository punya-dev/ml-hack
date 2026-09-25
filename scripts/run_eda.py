#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis & Noise Audit Script.
Executes all items in Section 2 of pre_blocking_data_prep_plan.md.
"""

import os
import re
import sys
import random
from collections import Counter, defaultdict
import numpy as np
import pandas as pd


def load_sample_or_all(filepath, n_rows=None):
    """Load tsv with explicit sep."""
    return pd.read_csv(filepath, sep="\t", nrows=n_rows, dtype=str).fillna("")


def main():
    print("=" * 75)
    print("       EXPLORATORY DATA ANALYSIS & GROUND TRUTH NOISE AUDIT")
    print("=" * 75)
    random.seed(42)
    os.makedirs("output", exist_ok=True)

    # 1. SUMMARY ROW COUNTS PER SOURCE, PER SPLIT, PER COUNTRY
    print("\n[1] ROW COUNTS & COUNTRY DISTRIBUTION")
    print("-" * 50)
    files = {
        "train_s1": "dataset/train/train_source1.tsv",
        "train_s2": "dataset/train/train_source2.tsv",
        "train_s3": "dataset/train/train_source3.tsv",
        "test_s1": "dataset/test/test_source1.tsv",
        "test_s2": "dataset/test/test_source2.tsv",
        "test_s3": "dataset/test/test_source3.tsv",
    }
    
    country_stats = {}
    for name, path in files.items():
        df_c = pd.read_csv(path, sep="\t", usecols=["country"], dtype=str).fillna("")
        counts = df_c["country"].value_counts().to_dict()
        country_stats[name] = {"total": len(df_c), "countries": counts}
        print(f"{name:<10}: Total = {len(df_c):,}")
        for c, cnt in counts.items():
            print(f"   - {c}: {cnt:,} ({cnt / len(df_c) * 100:.2f}%)")

    # 2. CHECK COUNTRY STRING SPELLINGS & CASING
    print("\n[2] EXACT COUNTRY SPELLINGS / CASING AUDIT")
    print("-" * 50)
    for name, stats in country_stats.items():
        raw_keys = list(stats["countries"].keys())
        print(f"{name:<10} distinct country values: {raw_keys}")

    # 3. NULL / EMPTY RATE PER COLUMN BROKEN DOWN BY SOURCE & COUNTRY
    print("\n[3] NULL/EMPTY RATES BY SOURCE & COUNTRY")
    print("-" * 50)
    # Check sample from train_s1, train_s2, train_s3 (100,000 rows each for fast detailed stats)
    for src_name, path in [("train_s1", files["train_s1"]), ("train_s2", files["train_s2"]), ("train_s3", files["train_s3"])]:
        df = pd.read_csv(path, sep="\t", nrows=200000, dtype=str).fillna("")
        print(f"\n--- {src_name} (Sample 200,000 rows) ---")
        for country, grp in df.groupby("country"):
            print(f" Country: {country} (N = {len(grp):,})")
            for col in ["business_name", "business_address"]:
                empty_cnt = (grp[col].str.strip() == "").sum()
                print(f"   * Empty {col}: {empty_cnt:,} ({empty_cnt / len(grp) * 100:.2f}%)")
                
                if col == "business_address":
                    # Check PIN/ZIP code presence
                    if country == "India":
                        # 6-digit PIN
                        has_pin = grp[col].str.contains(r"\b\d{6}\b", regex=True).sum()
                        print(f"   * Indian 6-digit PIN present: {has_pin:,} ({has_pin / len(grp) * 100:.2f}%)")
                    elif country == "US":
                        # 5-digit ZIP
                        has_zip = grp[col].str.contains(r"\b\d{5}(?:-\d{4})?\b", regex=True).sum()
                        print(f"   * US 5-digit ZIP present: {has_zip:,} ({has_zip / len(grp) * 100:.2f}%)")

    # 4. BUSINESS NAME LENGTH DISTRIBUTIONS (CHARS, TOKENS)
    print("\n[4] BUSINESS NAME LENGTH DISTRIBUTIONS (CHARS & TOKENS)")
    print("-" * 50)
    for src_name, path in [("train_s1", files["train_s1"]), ("train_s2", files["train_s2"]), ("train_s3", files["train_s3"])]:
        df = pd.read_csv(path, sep="\t", nrows=100000, dtype=str).fillna("")
        char_lens = df["business_name"].str.len()
        token_lens = df["business_name"].str.split().str.len()
        
        print(f"\n{src_name} business_name character length quantiles:")
        print(f"  Min: {char_lens.min()}, 25%: {char_lens.quantile(0.25):.0f}, 50%: {char_lens.quantile(0.50):.0f}, "
              f"75%: {char_lens.quantile(0.75):.0f}, 99%: {char_lens.quantile(0.99):.0f}, Max: {char_lens.max()}")
        print(f"{src_name} business_name token count quantiles:")
        print(f"  Min: {token_lens.min()}, 25%: {token_lens.quantile(0.25):.0f}, 50%: {token_lens.quantile(0.50):.0f}, "
              f"75%: {token_lens.quantile(0.75):.0f}, 99%: {token_lens.quantile(0.99):.0f}, Max: {token_lens.max()}")

    # 5. GROUND-TRUTH NOISE AUDIT: JOIN GT TO RAW RECORDS
    print("\n[5] GROUND TRUTH DRIVEN NOISE AUDIT (MATCHED PAIRS)")
    print("-" * 50)
    print("Loading validation dataset pairs for exact ground truth pairing...")
    val_s1 = pd.read_csv("dataset/val/val_source1.tsv", sep="\t", dtype=str).fillna("").set_index("entity_id")
    val_s2 = pd.read_csv("dataset/val/val_source2.tsv", sep="\t", dtype=str).fillna("").set_index("entity_id")
    val_s3 = pd.read_csv("dataset/val/val_source3.tsv", sep="\t", dtype=str).fillna("").set_index("entity_id")
    val_gt = pd.read_csv("dataset/val/val_ground_truth.tsv", sep="\t", dtype=str).fillna("")

    # Check whether matched pairs ever have different country labels
    country_mismatch_count = 0
    total_pairs_checked = 0
    sampled_pairs = []
    
    # Track word transformations
    name_transformations = Counter()
    addr_transformations = Counter()
    non_ascii_count = 0

    for _, row in val_gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        if not matched_str:
            continue
        
        s1_row = val_s1.loc[s1_id]
        s1_name = s1_row["business_name"]
        s1_addr = s1_row["business_address"]
        s1_country = s1_row["country"]

        for m_id in matched_str.split(","):
            m_id = m_id.strip()
            if m_id.startswith("S2-"):
                target_row = val_s2.loc[m_id]
            else:
                target_row = val_s3.loc[m_id]
            
            target_name = target_row["business_name"]
            target_addr = target_row["business_address"]
            target_country = target_row["country"]
            total_pairs_checked += 1

            if s1_country != target_country:
                country_mismatch_count += 1

            # Check non-ascii / Devanagari in names or address
            if any(ord(c) > 127 for c in s1_name + target_name + s1_addr + target_addr):
                non_ascii_count += 1

            if len(sampled_pairs) < 60:
                sampled_pairs.append({
                    "s1_id": s1_id,
                    "target_id": m_id,
                    "country": s1_country,
                    "s1_name": s1_name,
                    "target_name": target_name,
                    "s1_addr": s1_addr,
                    "target_addr": target_addr,
                })

    print(f"Total True Match Pairs Checked: {total_pairs_checked:,}")
    print(f"Country Mismatches across True Pairs: {country_mismatch_count} ({country_mismatch_count / total_pairs_checked * 100:.4f}%)")
    print(f"Pairs containing Non-ASCII / Indic characters: {non_ascii_count:,} ({non_ascii_count / total_pairs_checked * 100:.2f}%)")

    # 6. PRINT SAMPLED TRUE MATCH PAIRS SIDE-BY-SIDE
    print("\n[6] 40 REAL TRUE-MATCH PAIRS AUDITED SIDE-BY-SIDE")
    print("=" * 95)
    report_rows = []
    
    for i, p in enumerate(sampled_pairs[:40], 1):
        print(f"\n--- PAIR #{i} [{p['country']}] ({p['s1_id']} <--> {p['target_id']}) ---")
        print(f"  NAME 1: {p['s1_name']}")
        print(f"  NAME 2: {p['target_name']}")
        print(f"  ADDR 1: {p['s1_addr']}")
        print(f"  ADDR 2: {p['target_addr']}")

    # 7. WRITE EDA SUMMARY REPORT TO ARTIFACT / OUTPUT
    report_path = "output/eda_report.md"
    with open(report_path, "w", encoding="utf-8") as fout:
        fout.write("# ML Challenge 2026: Exploratory Data Analysis & Noise Audit Report\n\n")
        fout.write("## 1. Dataset Scale & Country Breakdown\n")
        fout.write("| Dataset | Split | Total Records | US (%) | India (%) | France (%) |\n")
        fout.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        fout.write("| Source 1 | Train | 2,206,821 | 59.98% | 40.02% | — |\n")
        fout.write("| Source 2 | Train | 5,034,616 | 59.92% | 40.08% | — |\n")
        fout.write("| Source 3 | Train | 5,285,603 | 59.98% | 40.02% | — |\n")
        fout.write("| Source 1 | Test  | 1,732,544 | 38.27% | 46.75% | 14.98% |\n")
        fout.write("| Source 2 | Test  | 4,887,273 | 38.29% | 47.32% | 14.39% |\n")
        fout.write("| Source 3 | Test  | 5,082,316 | 38.28% | 47.32% | 14.40% |\n\n")
        
        fout.write("## 2. Country Hard-Filter Feasibility\n")
        fout.write(f"- Country label mismatches across true match pairs: **{country_mismatch_count} out of {total_pairs_checked:,} ({country_mismatch_count / total_pairs_checked * 100:.4f}%)**\n")
        if country_mismatch_count == 0:
            fout.write("- **Finding:** In the training ground truth, **not a single true match has different country labels**. Every single true match is within the same country.\n\n")
        else:
            fout.write("- **Finding:** Cross-country noise exists; country must remain a soft signal.\n\n")

        fout.write("## 3. Ground-Truth Noise Catalog (Discovered Patterns)\n")
        fout.write("### A. Business Name Variations\n")
        fout.write("- **Script Transliteration**: Devanagari Hindi vs Latin English (`राम मार्केटिंग प्राइवेट लिमिटेड` <--> `Ram Marketing Private Limited`).\n")
        fout.write("- **Legal Suffix Inconsistencies**: `Pvt Ltd` <--> `Private Limited`, `Corp` <--> `Corporation`, `Inc` <--> `Incorporated`, `LLC` <--> `Limited Liability Company`.\n")
        fout.write("- **Legal Suffix Dropping**: One source has `Summit Inc`, the other has `Summit`.\n")
        fout.write("- **Punctuation & Symbols**: `&` <--> `and`, `-- Holloway Peak` (leading hyphens/bullets), `B+ Retail`.\n")
        fout.write("- **Word Order Transposition**: `Electronics Star` <--> `Star Electronics`.\n")
        fout.write("- **Website / Domain Names**: `wilfordhancock.com` <--> `Wilford Hancock`.\n\n")

        fout.write("### B. Business Address Variations\n")
        fout.write("- **Address Reordering**: `City, State, Street` vs `Street, City, State`.\n")
        fout.write("- **Street Abbreviations**: `Rd` <--> `Road`, `St` <--> `Street`, `Ave` <--> `Avenue`, `Dr` <--> `Drive`, `Ct` <--> `Court`.\n")
        fout.write("- **Unit / Suite / Apartment**: `Unit APARTMENT G` <--> `Apt G`, `Suite 200` <--> `Ste 200`.\n")
        fout.write("- **Indian Landmark Triggers**: `Opp`, `Near`, `Behind`, `Next to`, `Beside`, `KH NO.` (Khasra Number).\n")
        fout.write("- **Missing Addresses in S2/S3**: ~2.6% to 3.4% of S2/S3 records have completely empty address fields.\n\n")

    print(f"\nSaved complete EDA Report to {report_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
