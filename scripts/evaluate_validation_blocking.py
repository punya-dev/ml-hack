"""
Evaluate Multi-Channel Candidate Blocker on Validation Set.
Validates:
1. Exact channels recall
2. Fuzzy TF-IDF char-ngram channel recall
3. Combined recall (aiming for 97-98%+)
4. Average candidate count per S1 record
5. Reduction ratio
"""
import os
import sys
import time
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.blocker import CandidateBlocker
from src.country_normalizer import normalize_country


def evaluate_val_blocking(
    sample_s1_size: int = 2000,
    fuzzy_threshold: float = 0.25,
    max_fuzzy_cands: int = 50,
    max_total_cands: int = 80,
):
    print("=" * 85)
    print(f"EVALUATING MULTI-CHANNEL BLOCKER ON VALIDATION SPLIT (S1 Sample: {sample_s1_size})")
    print("=" * 85)

    # 1. Load validation data
    t0 = time.time()
    val_s1 = pd.read_csv("dataset/val/val_source1.tsv", sep="\t")
    val_s2 = pd.read_csv("dataset/val/val_source2.tsv", sep="\t")
    val_s3 = pd.read_csv("dataset/val/val_source3.tsv", sep="\t")
    val_gt = pd.read_csv("dataset/val/val_ground_truth.tsv", sep="\t")

    print(f"Loaded validation files in {time.time() - t0:.2f}s:")
    print(f"  val_s1: {len(val_s1)} records")
    print(f"  val_s2: {len(val_s2)} records")
    print(f"  val_s3: {len(val_s3)} records")
    print(f"  val_gt: {len(val_gt)} records")

    # Combine S2 and S3 as candidate pool
    other_df = pd.concat([val_s2, val_s3], ignore_index=True)

    # Parse ground truth to find true match pairs
    gt_map = {}
    total_true_pairs_in_gt = 0
    for _, row in val_gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if matched_str.strip():
            matches = set(m.strip() for m in matched_str.split(",") if m.strip())
            gt_map[s1_id] = matches
            total_true_pairs_in_gt += len(matches)
        else:
            gt_map[s1_id] = set()

    matched_s1_ids = [s1_id for s1_id, matches in gt_map.items() if len(matches) > 0]
    unmatched_s1_ids = [s1_id for s1_id, matches in gt_map.items() if len(matches) == 0]

    if sample_s1_size is None or sample_s1_size >= len(val_s1):
        sample_s1_df = val_s1.copy()
        sample_true_pairs = total_true_pairs_in_gt
        sample_matched_count = len(matched_s1_ids)
        sample_unmatched_count = len(unmatched_s1_ids)
    else:
        # Sample balanced set of matched and unmatched S1 records
        sample_matched_count = min(len(matched_s1_ids), int(sample_s1_size * 0.8))
        sample_unmatched_count = min(len(unmatched_s1_ids), sample_s1_size - sample_matched_count)

        sampled_s1_ids = set(matched_s1_ids[:sample_matched_count] + unmatched_s1_ids[:sample_unmatched_count])
        sample_s1_df = val_s1[val_s1["entity_id"].isin(sampled_s1_ids)].copy()

        # Count how many true pairs exist in this sample
        sample_true_pairs = 0
        for s1_id in sampled_s1_ids:
            sample_true_pairs += len(gt_map.get(s1_id, set()))

    print(f"\nEvaluation Sample:")
    print(f"  S1 Entities: {len(sample_s1_df)} ({sample_matched_count} matched, {sample_unmatched_count} singletons)")
    print(f"  True Match Pairs to Recall: {sample_true_pairs}")
    print(f"  Candidate Pool (S2 + S3): {len(other_df)} records")

    # 2. Run Candidate Blocker
    print("\n--- Running Candidate Blocker ---")
    blocker = CandidateBlocker(
        fuzzy_threshold=fuzzy_threshold,
        max_fuzzy_candidates_per_s1=max_fuzzy_cands,
        max_total_candidates_per_s1=max_total_cands,
    )

    t_block_start = time.time()
    candidates = blocker.generate_candidate_pairs(sample_s1_df, other_df, verbose=True)
    block_time = time.time() - t_block_start

    # Use exact-only candidates from the blocker run
    exact_candidates = getattr(blocker, "last_exact_candidates", {})

    # 3. Calculate Metrics
    exact_recalled = 0
    total_recalled = 0
    total_candidates_generated = 0
    distribution = []

    for s1_id in sample_s1_df["entity_id"]:
        cand_set = set(candidates.get(s1_id, []))
        exact_set = exact_candidates.get(s1_id, set())
        true_matches = gt_map.get(s1_id, set())

        exact_recalled += len(exact_set.intersection(true_matches))
        total_recalled += len(cand_set.intersection(true_matches))

        num_cands = len(cand_set)
        total_candidates_generated += num_cands
        distribution.append(num_cands)

    exact_recall = exact_recalled / sample_true_pairs * 100 if sample_true_pairs > 0 else 0
    total_recall = total_recalled / sample_true_pairs * 100 if sample_true_pairs > 0 else 0
    recovered_by_fuzzy = total_recalled - exact_recalled
    avg_cands = total_candidates_generated / len(sample_s1_df)

    # Reduction ratio: 1 - (candidates / total_cartesian_product)
    total_possible_comparisons = len(sample_s1_df) * len(other_df)
    reduction_ratio = (1.0 - (total_candidates_generated / total_possible_comparisons)) * 100

    print("\n" + "=" * 85)
    print("VALIDATION BLOCKING PERFORMANCE SUMMARY:")
    print(f"  S1 Records Processed:         {len(sample_s1_df)}")
    print(f"  Total Candidate Pool:         {len(other_df)} (S2 + S3)")
    print(f"  Time Elapsed:                 {block_time:.2f}s ({len(sample_s1_df)/block_time:.1f} S1/sec)")
    print(f"  Fuzzy Similarity Threshold:   {fuzzy_threshold}")
    print(f"  Max Fuzzy Candidates per S1:  {max_fuzzy_cands}")
    print(f"  Max Total Candidates per S1:  {max_total_cands}")
    print("-" * 85)
    print(f"  Exact-Match Channels Recall:  {exact_recalled}/{sample_true_pairs} ({exact_recall:.2f}%)")
    print(f"  Pairs Recovered by Fuzzy:     +{recovered_by_fuzzy} (+{recovered_by_fuzzy/sample_true_pairs*100:.2f}%)")
    print(f"  >>> COMBINED BLOCKING RECALL:  {total_recalled}/{sample_true_pairs} ({total_recall:.2f}%) <<<")
    print("-" * 85)
    print(f"  Average Candidates per S1:    {avg_cands:.1f}")
    print(f"  Max Candidates per S1:        {max(distribution)}")
    print(f"  Median Candidates per S1:     {pd.Series(distribution).median():.1f}")
    print(f"  Reduction Ratio:              {reduction_ratio:.6f}%")
    print("=" * 85)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--max-fuzzy", type=int, default=50)
    parser.add_argument("--max-total", type=int, default=80)
    args = parser.parse_args()

    evaluate_val_blocking(
        sample_s1_size=args.sample_size,
        fuzzy_threshold=args.threshold,
        max_fuzzy_cands=args.max_fuzzy,
        max_total_cands=args.max_total,
    )
