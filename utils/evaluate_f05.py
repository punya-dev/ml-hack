#!/usr/bin/env python3
"""
Macro-averaged F_0.5 evaluation script for Business Entity Resolution Challenge.

Formula:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Computed as a macro-average: F_0.5 is calculated per Source 1 entity,
then averaged across ALL Source 1 entities in the evaluation set.

Singletons (entities with 0 matches) are included:
- True singleton, predicted empty -> 1.0
- True singleton, predicted non-empty -> 0.0
- Non-singleton, predicted empty -> 0.0
- Non-singleton, predicted non-empty with no overlap -> 0.0
- Non-singleton, predicted non-empty with overlap -> F_0.5
"""

import argparse
import sys
from typing import Dict, Set, Tuple
import pandas as pd


def compute_entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> Tuple[float, float, float]:
    """Compute (precision, recall, f05) for a single Source 1 entity."""
    is_true_singleton = len(true_ids) == 0
    is_pred_singleton = len(pred_ids) == 0

    if is_true_singleton:
        if is_pred_singleton:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    # True entity is not a singleton
    if is_pred_singleton:
        return 0.0, 0.0, 0.0

    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return precision, recall, f05


def evaluate(ground_truth_path: str, predictions_path: str) -> Dict[str, float]:
    """Evaluate predictions against ground truth and return metric summary."""
    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str).fillna("")
    pred_df = pd.read_csv(predictions_path, sep="\t", dtype=str).fillna("")

    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        matched = set(x.strip() for x in row["matched_entity_ids"].split(",") if x.strip())
        gt_dict[s1_id] = matched

    pred_dict = {}
    for _, row in pred_df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        matched = set(x.strip() for x in row["matched_entity_ids"].split(",") if x.strip())
        pred_dict[s1_id] = matched

    missing_s1 = set(gt_dict.keys()) - set(pred_dict.keys())
    if missing_s1:
        print(f"WARNING: {len(missing_s1)} Source 1 entities from ground truth are missing in predictions!")
        for s1_id in missing_s1:
            pred_dict[s1_id] = set()

    total_f05 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    singleton_count = 0
    singleton_correct = 0

    for s1_id, true_ids in gt_dict.items():
        pred_ids = pred_dict.get(s1_id, set())
        p, r, f = compute_entity_f05(true_ids, pred_ids)
        total_precision += p
        total_recall += r
        total_f05 += f

        if len(true_ids) == 0:
            singleton_count += 1
            if len(pred_ids) == 0:
                singleton_correct += 1

    n = len(gt_dict)
    macro_f05 = total_f05 / n if n > 0 else 0.0
    macro_precision = total_precision / n if n > 0 else 0.0
    macro_recall = total_recall / n if n > 0 else 0.0
    singleton_acc = singleton_correct / singleton_count if singleton_count > 0 else 0.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "singleton_count": singleton_count,
        "singleton_accuracy": singleton_acc,
        "total_evaluated": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Entity Resolution F_0.5 Macro Score")
    parser.add_argument("--gt", required=True, help="Path to ground truth TSV")
    parser.add_argument("--pred", required=True, help="Path to predictions TSV")
    args = parser.parse_args()

    metrics = evaluate(args.gt, args.pred)
    print("\n" + "=" * 45)
    print("        ENTITY RESOLUTION EVALUATION         ")
    print("=" * 45)
    print(f"Macro F_0.5 Score:    {metrics['macro_f05']:.5f}")
    print(f"Macro Precision:      {metrics['macro_precision']:.5f}")
    print(f"Macro Recall:         {metrics['macro_recall']:.5f}")
    print(f"Singleton Accuracy:   {metrics['singleton_accuracy']:.2%} ({metrics['singleton_count']} singletons)")
    print(f"Total Evaluated S1:   {metrics['total_evaluated']:,}")
    print("=" * 45)


if __name__ == "__main__":
    main()
