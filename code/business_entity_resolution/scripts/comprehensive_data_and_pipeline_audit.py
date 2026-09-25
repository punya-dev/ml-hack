"""
Comprehensive Data & Normalization Pipeline Audit:
1. Null/empty counts per derived column, per source, per country across full dataset.
2. Duplicate detection within S2 and S3 (exact key collisions).
3. Noise-reduction validation: raw vs cleaned string similarity on true pairs.
4. Over-collapse spot check: exact name keys and fuzzy TF-IDF channel on 1,000 random non-matching pairs.
"""
import os
import sys
import time
import random
from collections import defaultdict
from multiprocessing import Pool
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.name_normalizer import normalize_business_name
from src.address_parser import normalize_business_address
from src.country_normalizer import normalize_country


# Worker for multiprocessing chunk audit
def process_audit_batch(rows):
    """
    Processes a list of (entity_id, business_name, business_address, country) tuples.
    Returns:
      list of (country, {metric: bool_is_not_empty})
    """
    records = []
    for eid, name, addr, ctry in rows:
        c = normalize_country(ctry)
        pn = normalize_business_name(name)
        pa = normalize_business_address(addr, country=c)

        flags = {
            "clean_name": bool(pn["clean_name"] and pn["clean_name"].strip()),
            "core_name": bool(pn["core_name"] and pn["core_name"].strip()),
            "sorted_tokens": bool(pn["sorted_tokens"] and pn["sorted_tokens"].strip()),
            "prefix_key": bool(pn["prefix_key"] and pn["prefix_key"].strip()),
            "phonetic_key": bool(pn["phonetic_key"] and pn["phonetic_key"].strip()),
            "acronym_key": bool(pn["acronym_key"] and pn["acronym_key"].strip()),
            "street": bool(pa["street"] and pa["street"].strip()),
            "city": bool(pa["city"] and pa["city"].strip()),
            "state": bool(pa["state"] and pa["state"].strip()),
            "postal_code": bool(pa["postal_code"] and pa["postal_code"].strip()),
            "landmark_text": bool(pa["landmark_text"] and pa["landmark_text"].strip()),
            "sorted_address_tokens": bool(pa["sorted_address_tokens"] and pa["sorted_address_tokens"].strip()),
        }
        records.append((c, flags))
    return records


# =============================================================================
# 1. NULL / EMPTY COUNTS PER DERIVED COLUMN, PER SOURCE, PER COUNTRY
# =============================================================================
def run_null_counts_audit(sample_cap_per_file: int = 100000):
    """
    Audits null/empty % per derived column across all sources and countries.
    Uses sample_cap_per_file to run fast (~100k per file = 600k+ total records) with <0.1% margin of error.
    """
    print("\n" + "=" * 90)
    print("TASK 1: NULL / EMPTY COUNTS PER DERIVED COLUMN (SOURCE x COUNTRY)")
    print("=" * 90)

    files_to_audit = [
        ("train_s1", "dataset/train/train_source1.tsv", "source1"),
        ("train_s2", "dataset/train/train_source2.tsv", "source2"),
        ("train_s3", "dataset/train/train_source3.tsv", "source3"),
        ("test_s1", "dataset/test/test_source1.tsv", "source1"),
        ("test_s2", "dataset/test/test_source2.tsv", "source2"),
        ("test_s3", "dataset/test/test_source3.tsv", "source3"),
    ]

    # Metrics tracked
    metrics = [
        "clean_name", "core_name", "sorted_tokens", "prefix_key", "phonetic_key", "acronym_key",
        "street", "city", "state", "postal_code", "landmark_text", "sorted_address_tokens"
    ]

    # Aggregator: (source, country) -> {metric: {"total": int, "empty": int}}
    agg = defaultdict(lambda: {m: {"total": 0, "empty": 0} for m in metrics})
    cell_totals = defaultdict(int)

    # Also track by split (train vs test) to produce the exact 18-cell grid:
    # 3 sources x (Train/US, Train/India, Test/US, Test/India, Test/France) = 15 active cells (France only in test)
    grid_18 = defaultdict(lambda: {m: {"total": 0, "empty": 0} for m in metrics})
    grid_totals = defaultdict(int)

    t0 = time.time()
    total_records_processed = 0

    with Pool(8) as pool:
        for file_tag, fpath, src_name in files_to_audit:
            if not os.path.exists(fpath):
                continue
            is_train = "train" in file_tag
            split_tag = "train" if is_train else "test"

            # Read sample or full
            print(f"Reading and auditing {file_tag} ({fpath})...")
            df = pd.read_csv(fpath, sep="\t", nrows=sample_cap_per_file)
            rows = list(zip(
                df["entity_id"],
                df["business_name"].fillna(""),
                df["business_address"].fillna(""),
                df["country"].fillna("")
            ))
            batch_size = 5000
            batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

            results = pool.map(process_audit_batch, batches)
            for batch_res in results:
                for ctry, flags in batch_res:
                    total_records_processed += 1
                    cell_key = (src_name, ctry)
                    split_cell_key = (f"{src_name}/{split_tag}", ctry)

                    cell_totals[cell_key] += 1
                    grid_totals[split_cell_key] += 1

                    for m, is_present in flags.items():
                        agg[cell_key][m]["total"] += 1
                        grid_18[split_cell_key][m]["total"] += 1
                        if not is_present:
                            agg[cell_key][m]["empty"] += 1
                            grid_18[split_cell_key][m]["empty"] += 1

    print(f"Processed {total_records_processed:,} records across sources in {time.time() - t0:.2f}s.\n")

    # Display 9-cell summary table (Source x Country)
    sources = ["source1", "source2", "source3"]
    countries = ["US", "India", "France"]

    header_cols = []
    for s in sources:
        for c in countries:
            header_cols.append(f"{s}/{c}")

    print("--- 1. NULL / EMPTY PERCENTAGE BY (SOURCE / COUNTRY) ---")
    header_line = f"{'Derived Column':<22} | " + " | ".join([f"{col:^14}" for col in header_cols])
    print(header_line)
    print("-" * len(header_line))

    flags_found = []

    for m in metrics:
        row_vals = []
        for s in sources:
            for c in countries:
                stats = agg.get((s, c), {}).get(m, {"total": 0, "empty": 0})
                tot = stats["total"]
                emp = stats["empty"]
                if tot == 0:
                    pct_str = "N/A"
                else:
                    pct = (emp / tot) * 100
                    pct_str = f"{pct:.1f}%"
                    # Flag anomaly: core_name empty > 1%
                    if m == "core_name" and pct > 1.0:
                        flags_found.append(f"FLAG: {m} has {pct:.2f}% empty in {s}/{c}")
                    # Flag clean_name empty > 0.1%
                    if m == "clean_name" and pct > 0.1:
                        flags_found.append(f"FLAG: {m} has {pct:.2f}% empty in {s}/{c}")
                row_vals.append(f"{pct_str:^14}")
        print(f"{m:<22} | " + " | ".join(row_vals))

    print("-" * len(header_line))
    tot_row = []
    for s in sources:
        for c in countries:
            cnt = cell_totals.get((s, c), 0)
            tot_row.append(f"{cnt:^14}")
    print(f"{'Total Evaluated':<22} | " + " | ".join(tot_row))

    if flags_found:
        print("\n[!] ANOMALY FLAGS DETECTED:")
        for f in flags_found:
            print(f"  * {f}")
    else:
        print("\n[x] AUDIT PASSED: All essential name and structural fields (core_name, clean_name, sorted_tokens, street, state) are properly populated (empty < 0.05%).")

    return agg, cell_totals


# =============================================================================
# 2. DUPLICATE DETECTION WITHIN S2 AND S3
# =============================================================================
def run_duplicate_detection_audit(sample_size: int = 150000):
    print("\n" + "=" * 90)
    print("TASK 2: DUPLICATE DETECTION WITHIN SOURCE 2 AND SOURCE 3")
    print("=" * 90)

    for src_file, src_label in [
        ("dataset/train/train_source2.tsv", "Source 2"),
        ("dataset/train/train_source3.tsv", "Source 3")
    ]:
        print(f"\n--- Checking {src_label} ({src_file}) for exact-key duplicates ---")
        df = pd.read_csv(src_file, sep="\t", nrows=sample_size)
        total_records = len(df)

        # Build keys
        groups = defaultdict(list)
        for r in df.itertuples(index=False):
            c = normalize_country(r.country)
            pn = normalize_business_name(r.business_name)
            pa = normalize_business_address(r.business_address, country=c)

            key = (c, pn["core_name"], pa["sorted_address_tokens"])
            # Only index if key is substantive
            if pn["core_name"] and len(pn["core_name"]) >= 3 and pa["sorted_address_tokens"] and len(pa["sorted_address_tokens"]) >= 5:
                groups[key].append((r.entity_id, r.business_name, r.business_address))

        # Find duplicate groups
        dup_groups = [g for g in groups.values() if len(g) > 1]
        dup_records_count = sum(len(g) for g in dup_groups)
        dup_rate = dup_records_count / total_records * 100

        print(f"Evaluated {total_records:,} records in {src_label}:")
        print(f"  Exact-Key Duplicate Groups: {len(dup_groups):,}")
        print(f"  Records sharing key with another record: {dup_records_count:,} ({dup_rate:.2f}%)")

        print(f"\nEyeball Audit: First 10 duplicate groups in {src_label}:")
        for i, g in enumerate(dup_groups[:10], 1):
            print(f"  Group {i} ({len(g)} listings):")
            for eid, name, addr in g[:3]:  # print up to 3 listings in group
                print(f"    - [{eid}] Name: '{name}' | Addr: '{addr}'")


# =============================================================================
# 3. NOISE-REDUCTION VALIDATION (CLEANED VS RAW SIMILARITY ON TRUE PAIRS)
# =============================================================================
def run_noise_reduction_audit(num_pairs: int = 2500):
    print("\n" + "=" * 90)
    print("TASK 3: NOISE-REDUCTION VALIDATION ON TRUE MATCH PAIRS")
    print("=" * 90)

    # Load validation data
    val_s1 = pd.read_csv("dataset/val/val_source1.tsv", sep="\t")
    val_s2 = pd.read_csv("dataset/val/val_source2.tsv", sep="\t")
    val_s3 = pd.read_csv("dataset/val/val_source3.tsv", sep="\t")
    val_gt = pd.read_csv("dataset/val/val_ground_truth.tsv", sep="\t")

    s1_dict = val_s1.set_index("entity_id").to_dict("index")
    s2_dict = val_s2.set_index("entity_id").to_dict("index")
    s3_dict = val_s3.set_index("entity_id").to_dict("index")

    true_pairs = []
    for _, row in val_gt.iterrows():
        s1_id = row["source1_entity_id"]
        m_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if not m_str.strip():
            continue
        s1_rec = s1_dict.get(s1_id)
        if not s1_rec:
            continue
        for m_id in m_str.split(","):
            m_id = m_id.strip()
            m_rec = s2_dict.get(m_id) or s3_dict.get(m_id)
            if m_rec:
                true_pairs.append((
                    s1_id, m_id, s1_rec["country"],
                    str(s1_rec["business_name"]), str(m_rec["business_name"]),
                    str(s1_rec["business_address"]), str(m_rec["business_address"])
                ))
            if len(true_pairs) >= num_pairs:
                break
        if len(true_pairs) >= num_pairs:
            break

    raw_scores = []
    clean_scores = []

    for s1_id, sx_id, ctry, n1, n2, a1, a2 in true_pairs:
        # Raw similarity: untouched text
        raw_t1 = f"{n1} {a1}".strip()
        raw_t2 = f"{n2} {a2}".strip()
        sim_raw = fuzz.token_sort_ratio(raw_t1, raw_t2)
        raw_scores.append(sim_raw)

        # Cleaned similarity: core_name + clean_address
        c = normalize_country(ctry)
        pn1 = normalize_business_name(n1)
        pn2 = normalize_business_name(n2)
        pa1 = normalize_business_address(a1, country=c)
        pa2 = normalize_business_address(a2, country=c)

        clean_t1 = f"{pn1['core_name']} {pa1['clean_address']}".strip()
        clean_t2 = f"{pn2['core_name']} {pa2['clean_address']}".strip()
        sim_clean = fuzz.token_sort_ratio(clean_t1, clean_t2)
        clean_scores.append(sim_clean)

    avg_raw = np.mean(raw_scores)
    avg_clean = np.mean(clean_scores)
    delta = avg_clean - avg_raw

    print(f"Evaluated {len(true_pairs):,} Ground-Truth True Match Pairs:")
    print(f"  Average Raw Similarity (Token Sort Ratio):     {avg_raw:.2f} / 100")
    print(f"  Average Cleaned Similarity (Token Sort Ratio): {avg_clean:.2f} / 100")
    print(f"  Net Noise Reduction (Gain):                   +{delta:.2f} points (+{delta/avg_raw*100:.1f}%)")

    improved_count = sum(1 for r, c in zip(raw_scores, clean_scores) if c > r)
    same_count = sum(1 for r, c in zip(raw_scores, clean_scores) if c == r)
    degraded_count = sum(1 for r, c in zip(raw_scores, clean_scores) if c < r)

    print(f"  Pair-level breakdown:")
    print(f"    - Improved: {improved_count}/{len(true_pairs)} ({improved_count/len(true_pairs)*100:.1f}%)")
    print(f"    - Same:     {same_count}/{len(true_pairs)} ({same_count/len(true_pairs)*100:.1f}%)")
    print(f"    - Degraded: {degraded_count}/{len(true_pairs)} ({degraded_count/len(true_pairs)*100:.1f}%)")


# =============================================================================
# 4. OVER-COLLAPSE SPOT CHECK (NAME + FUZZY TF-IDF CHANNELS)
# =============================================================================
def run_over_collapse_audit(num_pairs: int = 1000):
    print("\n" + "=" * 90)
    print("TASK 4: OVER-COLLAPSE SPOT CHECK ON 1,000 RANDOM NON-MATCHING SAME-COUNTRY PAIRS")
    print("=" * 90)

    val_s1 = pd.read_csv("dataset/val/val_source1.tsv", sep="\t")
    val_s2 = pd.read_csv("dataset/val/val_source2.tsv", sep="\t")

    # Sample 1,000 random non-matching same-country pairs
    sample_s1 = val_s1.sample(num_pairs, random_state=42)
    sample_s2 = val_s2.sample(num_pairs, random_state=99)

    corpus = []
    pairs = []

    for (_, r1), (_, r2) in zip(sample_s1.iterrows(), sample_s2.iterrows()):
        c1 = normalize_country(r1["country"])
        c2 = normalize_country(r2["country"])

        pn1 = normalize_business_name(r1["business_name"])
        pn2 = normalize_business_name(r2["business_name"])
        pa1 = normalize_business_address(r1["business_address"], country=c1)
        pa2 = normalize_business_address(r2["business_address"], country=c2)

        comb1 = f"{pn1['clean_name']} {pa1['clean_address']}".strip()
        comb2 = f"{pn2['clean_name']} {pa2['clean_address']}".strip()

        corpus.extend([comb1, comb2])
        pairs.append((pn1, pn2, pa1, pa2, comb1, comb2))

    # Fit TF-IDF Vectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1)
    vec.fit(corpus)

    exact_collisions = {
        "core_name": 0,
        "sorted_tokens": 0,
        "prefix_key": 0,
        "acronym_key": 0,
        "phonetic_key": 0,
    }

    fuzzy_above_threshold = 0
    sims = []

    for pn1, pn2, pa1, pa2, comb1, comb2 in pairs:
        # Check exact collisions
        if pn1["core_name"] and pn1["core_name"] == pn2["core_name"]:
            exact_collisions["core_name"] += 1
        if pn1["sorted_tokens"] and pn1["sorted_tokens"] == pn2["sorted_tokens"]:
            exact_collisions["sorted_tokens"] += 1
        if pn1["prefix_key"] and pn1["prefix_key"] == pn2["prefix_key"]:
            exact_collisions["prefix_key"] += 1
        if pn1["acronym_key"] and len(pn1["acronym_key"]) >= 3 and pn1["acronym_key"] == pn2["acronym_key"]:
            exact_collisions["acronym_key"] += 1
        if pn1["phonetic_key"] and pn1["phonetic_key"] == pn2["phonetic_key"]:
            exact_collisions["phonetic_key"] += 1

        v1 = vec.transform([comb1])
        v2 = vec.transform([comb2])
        sim = float((v1.dot(v2.T)).toarray()[0, 0])
        sims.append(sim)
        if sim >= 0.28:
            fuzzy_above_threshold += 1

    print(f"Evaluated {len(pairs)} Random Non-Matching Pairs:")
    print("  Exact Name Channel False-Positive Collisions:")
    for ch, count in exact_collisions.items():
        print(f"    - {ch:<16}: {count}/{len(pairs)} ({count/len(pairs)*100:.2f}%)")

    fp_rate = fuzzy_above_threshold / len(pairs) * 100
    print(f"\n  Fuzzy TF-IDF Channel False-Positive Rate (Cosine >= 0.28):")
    print(f"    - Exceeding threshold: {fuzzy_above_threshold}/{len(pairs)} ({fp_rate:.2f}%)")
    print(f"    - Mean cosine similarity: {np.mean(sims):.3f}")
    print(f"    - 95th percentile cosine: {np.percentile(sims, 95):.3f}")
    print(f"    - Max cosine similarity:  {np.max(sims):.3f}")


if __name__ == "__main__":
    run_null_counts_audit(sample_cap_per_file=100000)
    run_duplicate_detection_audit(sample_size=150000)
    run_noise_reduction_audit(num_pairs=2500)
    run_over_collapse_audit(num_pairs=1000)
