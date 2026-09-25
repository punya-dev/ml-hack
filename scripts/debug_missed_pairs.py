import os
import sys
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.name_normalizer import normalize_business_name
from src.address_parser import normalize_business_address

df_cand = pq.read_table("output/val_candidates.parquet").to_pandas()
valid_pairs = df_cand[df_cand["candidate_entity_id"].notna()]
cand_pairs = set(zip(valid_pairs["source1_entity_id"], valid_pairs["candidate_entity_id"]))

gt_df = pd.read_csv("dataset/val/val_ground_truth.tsv", sep="\t")
gt_pairs = []
for _, row in gt_df.iterrows():
    if pd.isna(row["matched_entity_ids"]):
        continue
    s1 = row["source1_entity_id"]
    for m in str(row["matched_entity_ids"]).split(","):
        m = m.strip()
        if m and m != "nan":
            gt_pairs.append((s1, m))

missed = set(gt_pairs) - cand_pairs
print(f"Total GT pairs: {len(gt_pairs):,}")
print(f"Retrieved: {len(gt_pairs) - len(missed):,} ({(len(gt_pairs)-len(missed))/len(gt_pairs)*100:.2f}%)")
print(f"Missed: {len(missed):,}")

s1_df = pd.read_csv("dataset/val/val_source1.tsv", sep="\t").set_index("entity_id")
s2_df = pd.read_csv("dataset/val/val_source2.tsv", sep="\t").set_index("entity_id")
s3_df = pd.read_csv("dataset/val/val_source3.tsv", sep="\t").set_index("entity_id")

cand_counts = valid_pairs.groupby("source1_entity_id").size().to_dict()

print("\nSample 10 missed pairs analysis:")
for idx, (s1_id, m_id) in enumerate(list(missed)[:10], 1):
    s1_row = s1_df.loc[s1_id]
    m_row = s2_df.loc[m_id] if m_id in s2_df.index else s3_df.loc[m_id]
    n1, a1, c1 = s1_row["business_name"], s1_row["business_address"], s1_row["country"]
    n2, a2, c2 = m_row["business_name"], m_row["business_address"], m_row["country"]
    
    pn1 = normalize_business_name(n1)
    pn2 = normalize_business_name(n2)
    pa1 = normalize_business_address(a1, country=c1)
    pa2 = normalize_business_address(a2, country=c2)
    
    c_count = cand_counts.get(s1_id, 0)
    print(f"\n[{idx}] S1: {s1_id} (total_cands={c_count}) <-> Match: {m_id}")
    print(f"  S1:   Name: '{n1}' | Clean: '{pn1['clean_name']}' | Core: '{pn1['core_name']}' | Pfx: '{pn1['prefix_key'][:4]}' | State: '{pa1['state']}'")
    print(f"  Cand: Name: '{n2}' | Clean: '{pn2['clean_name']}' | Core: '{pn2['core_name']}' | Pfx: '{pn2['prefix_key'][:4]}' | State: '{pa2['state']}'")
