"""
Analyze Exact Channels vs. Fuzzy Channel Recall on Ground Truth Pairs.
Measures:
1. Exact channel recall
2. Near-miss analysis of the failed pairs
3. TF-IDF / character n-gram similarity distribution on true match pairs
4. Combined recall with generous fuzzy threshold
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.name_normalizer import normalize_business_name
from src.address_parser import normalize_business_address
from src.country_normalizer import normalize_country


def evaluate_blocking_recall():
    print("=" * 80)
    print("EVALUATING EXACT CHANNELS + FUZZY CHANNEL RECALL ON GROUND TRUTH PAIRS")
    print("=" * 80)

    # Load training data
    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
    s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
    s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
    gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

    s1_dict = s1.set_index("entity_id").to_dict("index")
    s2_dict = s2.set_index("entity_id").to_dict("index")
    s3_dict = s3.set_index("entity_id").to_dict("index")

    # Sample 2,000 true match pairs (1,000 US, 1,000 India)
    us_pairs = []
    india_pairs = []

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if not matched_str.strip():
            continue
        s1_rec = s1_dict.get(s1_id)
        if not s1_rec:
            continue
        ctry = normalize_country(s1_rec["country"])
        matches = [m.strip() for m in matched_str.split(",") if m.strip()]

        for m_id in matches:
            m_rec = s2_dict.get(m_id) or s3_dict.get(m_id)
            if not m_rec:
                continue
            if ctry == "US" and len(us_pairs) < 1000:
                us_pairs.append((s1_id, m_id, ctry, s1_rec["business_name"], m_rec["business_name"], s1_rec["business_address"], m_rec["business_address"]))
            elif ctry == "India" and len(india_pairs) < 1000:
                india_pairs.append((s1_id, m_id, ctry, s1_rec["business_name"], m_rec["business_name"], s1_rec["business_address"], m_rec["business_address"]))

        if len(us_pairs) >= 1000 and len(india_pairs) >= 1000:
            break

    pairs = us_pairs + india_pairs
    print(f"Loaded {len(pairs)} true match pairs ({len(us_pairs)} US, {len(india_pairs)} India).")

    # Build corpus of combined text for vectorizer
    all_texts = []
    pair_data = []

    for s1_id, sx_id, ctry, n1, n2, a1, a2 in pairs:
        pn1 = normalize_business_name(n1)
        pn2 = normalize_business_name(n2)
        pa1 = normalize_business_address(a1, country=ctry)
        pa2 = normalize_business_address(a2, country=ctry)

        comb1 = f"{pn1['clean_name']} {pa1['clean_address']}".strip()
        comb2 = f"{pn2['clean_name']} {pa2['clean_address']}".strip()

        all_texts.extend([comb1, comb2])
        pair_data.append({
            "s1_id": s1_id,
            "sx_id": sx_id,
            "ctry": ctry,
            "pn1": pn1,
            "pn2": pn2,
            "pa1": pa1,
            "pa2": pa2,
            "comb1": comb1,
            "comb2": comb2,
            "raw_n1": n1,
            "raw_n2": n2,
            "raw_a1": a1,
            "raw_a2": a2
        })

    # Fit char-wb 3-gram TF-IDF vectorizer
    print("Fitting TF-IDF Vectorizer on combined name+address char-ngrams...")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1)
    vectorizer.fit(all_texts)

    # Evaluate exact channels and fuzzy channel
    exact_hit_count = 0
    fuzzy_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    fuzzy_hits = {th: 0 for th in fuzzy_thresholds}
    combined_hits = {th: 0 for th in fuzzy_thresholds}

    missed_by_exact = []

    for item in pair_data:
        pn1, pn2 = item["pn1"], item["pn2"]
        pa1, pa2 = item["pa1"], item["pa2"]

        # Exact channels
        ch_core = (pn1["core_name"] == pn2["core_name"] and len(pn1["core_name"]) >= 3) or (pn2["core_name"] in pn1["alt_names"])
        ch_sort_name = (pn1["sorted_tokens"] == pn2["sorted_tokens"] and len(pn1["sorted_tokens"]) >= 3)
        ch_state_pfx = (pa1["state"] and pa1["state"] == pa2["state"] and pn1["prefix_key"][:4] == pn2["prefix_key"][:4] and len(pn1["prefix_key"]) >= 4)
        ch_state_phon = (pa1["state"] and pa1["state"] == pa2["state"] and pn1["phonetic_key"] and pn1["phonetic_key"] == pn2["phonetic_key"])
        ch_acronym = (pn1["acronym_key"] and pn1["acronym_key"] == pn2["acronym_key"] and len(pn1["acronym_key"]) >= 3)
        ch_sort_addr = (pa1["sorted_address_tokens"] and pa1["sorted_address_tokens"] == pa2["sorted_address_tokens"])

        exact_hit = ch_core or ch_sort_name or ch_state_pfx or ch_state_phon or ch_acronym or ch_sort_addr
        if exact_hit:
            exact_hit_count += 1
        else:
            missed_by_exact.append(item)

        # Compute fuzzy cosine similarity
        v1 = vectorizer.transform([item["comb1"]])
        v2 = vectorizer.transform([item["comb2"]])
        cosine_sim = float((v1.dot(v2.T)).toarray()[0, 0])
        item["cosine_sim"] = cosine_sim

        name_fuzz = fuzz.token_sort_ratio(pn1["clean_name"], pn2["clean_name"]) / 100.0
        item["name_fuzz"] = name_fuzz

        for th in fuzzy_thresholds:
            # Fuzzy hit if cosine similarity on combined >= th OR high name fuzz
            f_hit = (cosine_sim >= th)
            if f_hit:
                fuzzy_hits[th] += 1
            if exact_hit or f_hit:
                combined_hits[th] += 1

    total = len(pair_data)
    print("\n" + "=" * 80)
    print(f"RECALL RESULTS ON {total} TRUE MATCH PAIRS:")
    print(f"  Exact Channels Only Recall: {exact_hit_count}/{total} ({exact_hit_count/total*100:.2f}%)")
    print("-" * 80)
    print("  Adding Fuzzy TF-IDF Char-Ngram Channel (clean_name + clean_address):")
    for th in fuzzy_thresholds:
        comb_rec = combined_hits[th] / total * 100
        f_rec = fuzzy_hits[th] / total * 100
        recovered = combined_hits[th] - exact_hit_count
        print(f"    Threshold >= {th:.2f} -> Fuzzy Channel Alone: {f_rec:.2f}% | Combined Recall: {comb_rec:.2f}% (Recovered {recovered} missed pairs)")
    print("=" * 80)

    # Inspect some pairs missed by exact channels and recovered by fuzzy
    print(f"\nAudit of {len(missed_by_exact)} pairs missed by exact channels (showing first 5):")
    for idx, item in enumerate(missed_by_exact[:5], 1):
        print(f"\nMissed Pair {idx}:")
        print(f"  S1:   Name: '{item['raw_n1']}' -> Clean: '{item['pn1']['clean_name']}'")
        print(f"        Addr: '{item['raw_a1']}' -> State: '{item['pa1']['state']}'")
        print(f"  S2/3: Name: '{item['raw_n2']}' -> Clean: '{item['pn2']['clean_name']}'")
        print(f"        Addr: '{item['raw_a2']}' -> State: '{item['pa2']['state']}'")
        print(f"  Metrics: TF-IDF Cosine: {item['cosine_sim']:.3f} | Name Fuzz: {item['name_fuzz']:.2f}")


if __name__ == "__main__":
    evaluate_blocking_recall()
