#!/usr/bin/env python3
"""
run_blocking.py — End-to-End Candidate Blocking Script
=======================================================
Runs Step 2 (Train Blocking) and Step 3 (Test Blocking) back-to-back,
or independently via --mode flag.

Usage:
    # Both steps (default):
    python run_blocking.py

    # Train only (Step 2):
    python run_blocking.py --mode train

    # Test only (Step 3):
    python run_blocking.py --mode test

    # Custom paths / tuning:
    python run_blocking.py --mode train --chunk-size 50000 --threshold 0.24

Outputs:
    output/train_candidates.parquet          — labelled candidate pairs for training
    output/test_candidates.parquet           — candidate pairs for inference
    output/candidate_pairs.tsv              — submission-ready TSV (test mode only)
    output/train_blocking_summary.txt        — timing + recall audit (train mode)
    output/test_blocking_summary.txt         — timing summary (test mode)
"""

import os
import sys
import time
import argparse
import gc

# ---------------------------------------------------------------------------
# Resolve project root so src/ and configs/ are always importable regardless
# of where the script is launched from (local, Colab, GPU server).
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # project root first

# Also support running from code/business_entity_resolution layout
LEGACY_SRC = os.path.join(SCRIPT_DIR, "code", "business_entity_resolution")
if os.path.isdir(LEGACY_SRC):
    sys.path.insert(1, LEGACY_SRC)

# ---------------------------------------------------------------------------
# Hardcoded paths — update these if your layout changes.
# Colab layout:
#   /content/
#     ml-hack/          ← this script lives here
#       src/, configs/, output/, ...
#     student_resource/
#       dataset/
#         train/  test/
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR   = "/content/student_resource/dataset"
DEFAULT_OUTPUT_DIR = "/content/ml-hack/output"

# ---------------------------------------------------------------------------
# Dependency check + optional auto-install
# ---------------------------------------------------------------------------
REQUIRED = {
    "numpy": "numpy>=2.0.0",
    "pandas": "pandas>=2.2.0",
    "sklearn": "scikit-learn>=1.5.0",
    "scipy": "scipy>=1.14.0",
    "pyarrow": "pyarrow>=14.0.0",
    "rapidfuzz": "rapidfuzz>=3.9.0",
    "unidecode": "unidecode>=1.3.8",
    "anyascii": "anyascii>=0.3.2",
    "wordninja": "wordninja>=2.0.0",
    "tqdm": "tqdm>=4.66.0",
}


def _auto_install():
    import subprocess
    pkgs = list(REQUIRED.values())
    print(f"[setup] Auto-installing {len(pkgs)} packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + pkgs)


def _check_deps():
    missing = []
    for mod in REQUIRED:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[setup] Missing packages: {missing}. Attempting auto-install...")
        _auto_install()


_check_deps()

# Now safe to import heavy deps
import pandas as pd
import numpy as np
from src.blocker import CandidateBlocker


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def _resolve_data_paths(mode: str, data_dir: str):
    """Return (s1, s2, s3, gt) absolute paths for the given split."""
    split = "train" if mode == "train" else "test"
    s1 = os.path.join(data_dir, split, f"{split}_source1.tsv")
    s2 = os.path.join(data_dir, split, f"{split}_source2.tsv")
    s3 = os.path.join(data_dir, split, f"{split}_source3.tsv")
    gt = os.path.join(data_dir, split, f"{split}_ground_truth.tsv") if mode == "train" else None
    return s1, s2, s3, gt


def _validate_paths(*paths):
    for p in paths:
        if p and not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_step(
    mode: str,
    data_dir: str,
    output_dir: str,
    chunk_size: int,
    threshold: float,
    max_fuzzy: int,
    max_total: int,
):
    """
    Execute one blocking step (train or test).

    train -> output/train_candidates.parquet  + recall audit vs ground truth
    test  -> output/test_candidates.parquet   + output/candidate_pairs.tsv
    """
    assert mode in ("train", "test"), f"mode must be 'train' or 'test', got '{mode}'"

    s1_path, s2_path, s3_path, gt_path = _resolve_data_paths(mode, data_dir)

    print(f"\n{'='*80}")
    print(f"  STEP {'2' if mode == 'train' else '3'} --- {mode.upper()} BLOCKING")
    print(f"{'='*80}")
    print(f"  S1 : {s1_path}")
    print(f"  S2 : {s2_path}")
    print(f"  S3 : {s3_path}")
    if gt_path:
        print(f"  GT : {gt_path}")
    print(f"  chunk_size  : {chunk_size:,}")
    print(f"  threshold   : {threshold}")
    print(f"  max_fuzzy   : {max_fuzzy}")
    print(f"  max_total   : {max_total}")
    print(f"{'='*80}\n")

    # Validate inputs
    required = [s1_path, s2_path, s3_path]
    if gt_path:
        required.append(gt_path)
    _validate_paths(*required)

    os.makedirs(output_dir, exist_ok=True)

    # Output paths
    out_parquet = os.path.join(output_dir, f"{mode}_candidates.parquet")
    out_tsv = os.path.join(output_dir, "candidate_pairs.tsv") if mode == "test" else None
    summary_path = os.path.join(output_dir, f"{mode}_blocking_summary.txt")

    blocker = CandidateBlocker(
        fuzzy_threshold=threshold,
        max_fuzzy_candidates_per_s1=max_fuzzy,
        max_total_candidates_per_s1=max_total,
    )

    t_start = time.time()
    blocker.run_batched_blocking(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        output_parquet=out_parquet,
        output_tsv=out_tsv,
        chunk_size=chunk_size,
        ground_truth_path=gt_path,
        verbose=True,
    )
    elapsed = time.time() - t_start

    # Write summary
    with open(summary_path, "w") as f:
        f.write(f"mode            : {mode}\n")
        f.write(f"s1_path         : {s1_path}\n")
        f.write(f"s2_path         : {s2_path}\n")
        f.write(f"s3_path         : {s3_path}\n")
        f.write(f"gt_path         : {gt_path}\n")
        f.write(f"out_parquet     : {out_parquet}\n")
        if out_tsv:
            f.write(f"out_tsv         : {out_tsv}\n")
        f.write(f"chunk_size      : {chunk_size}\n")
        f.write(f"threshold       : {threshold}\n")
        f.write(f"max_fuzzy       : {max_fuzzy}\n")
        f.write(f"max_total       : {max_total}\n")
        f.write(f"elapsed_seconds : {elapsed:.2f}\n")
        f.write(f"elapsed_minutes : {elapsed / 60:.2f}\n")
    print(f"\n[done] Summary written to: {summary_path}")

    # Validate output parquet is non-empty
    if not os.path.exists(out_parquet) or os.path.getsize(out_parquet) < 100:
        raise RuntimeError(f"Output parquet missing or empty: {out_parquet}")

    parquet_mb = os.path.getsize(out_parquet) / (1024 * 1024)
    print(f"[done] {mode}_candidates.parquet  -> {parquet_mb:.1f} MB")
    if out_tsv and os.path.exists(out_tsv):
        tsv_mb = os.path.getsize(out_tsv) / (1024 * 1024)
        print(f"[done] candidate_pairs.tsv        -> {tsv_mb:.1f} MB")

    print(f"[done] Step {'2' if mode == 'train' else '3'} completed in {elapsed/60:.2f} min.\n")
    gc.collect()
    return elapsed


# ---------------------------------------------------------------------------
# Recall audit helper (standalone, reads output parquet)
# ---------------------------------------------------------------------------

def audit_recall(parquet_path: str, gt_path: str, label: str = "train"):
    """
    Post-hoc recall ceiling audit: reads output parquet and compares
    against ground truth to compute blocking recall ceiling.
    Only meaningful for train mode (gt available).
    """
    if not os.path.exists(gt_path):
        print(f"[audit] Ground truth not found at {gt_path}, skipping audit.")
        return

    import pyarrow.parquet as pq

    print(f"\n{'='*70}")
    print(f"  POST-HOC RECALL AUDIT --- {label.upper()}")
    print(f"{'='*70}")

    # Load ground truth
    gt_df = pd.read_csv(gt_path, sep="\t")
    match_col = "matched_entity_ids" if "matched_entity_ids" in gt_df.columns else "matched_entity_id"
    gt_map = {}
    total_gt_pairs = 0
    for _, row in gt_df.iterrows():
        if pd.isna(row[match_col]):
            continue
        s1_id = row["source1_entity_id"]
        matches = {m.strip() for m in str(row[match_col]).split(",") if m.strip() and m.strip() != "nan"}
        if matches:
            gt_map[s1_id] = matches
            total_gt_pairs += len(matches)
    print(f"  Ground truth: {len(gt_map):,} S1 with matches -> {total_gt_pairs:,} total true pairs")

    # Stream parquet in batches to compute recall
    pf = pq.ParquetFile(parquet_path)
    retrieved = 0
    cand_counts = {}

    BATCH = 500_000
    for batch in pf.iter_batches(batch_size=BATCH):
        df = batch.to_pandas()
        valid = df[df["candidate_entity_id"].notna()]
        for s1_id, grp in valid.groupby("source1_entity_id"):
            cand_set = set(grp["candidate_entity_id"].tolist())
            true_set = gt_map.get(s1_id, set())
            retrieved += len(true_set & cand_set)
            cand_counts[s1_id] = cand_counts.get(s1_id, 0) + len(cand_set)

    recall = retrieved / total_gt_pairs * 100 if total_gt_pairs > 0 else 0.0
    avg_cands = np.mean(list(cand_counts.values())) if cand_counts else 0.0
    max_cands = max(cand_counts.values()) if cand_counts else 0

    print(f"  True pairs retrieved : {retrieved:,} / {total_gt_pairs:,}")
    print(f"  Missed               : {total_gt_pairs - retrieved:,}")
    print(f"  RECALL CEILING       : {recall:.4f}%")
    print(f"  Avg candidates/S1    : {avg_cands:.2f}")
    print(f"  Max candidates/S1    : {max_cands}")
    if recall >= 99.0:
        print(f"  PASS -- Recall ceiling {recall:.2f}% exceeds 99.0% target.")
    elif recall >= 97.0:
        print(f"  WARN -- Recall ceiling {recall:.2f}% is below 99.0% target.")
    else:
        print(f"  FAIL -- Recall ceiling {recall:.2f}% is well below target.")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End Candidate Blocking: Train (Step 2) + Test (Step 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        default="both",
        help="Which step(s) to run: 'train' (Step 2), 'test' (Step 3), or 'both'.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Root directory containing train/ and test/ sub-folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write output parquet/tsv/summary files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="S1 batch chunk size (50k ~= 2 min/batch locally; increase to 100k on GPU server).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.24,
        help="Fuzzy TF-IDF cosine similarity threshold (lower -> higher recall, more candidates).",
    )
    parser.add_argument(
        "--max-fuzzy",
        type=int,
        default=50,
        help="Max fuzzy candidates per S1 entity from TF-IDF channel.",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=80,
        help="Max total candidates per S1 entity (exact + fuzzy combined).",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        default=False,
        help="Skip post-hoc recall audit after train blocking.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("  RUN_BLOCKING.PY -- Entity Resolution Candidate Blocking Pipeline")
    print("=" * 80)
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  mode      : {args.mode}")
    print(f"  data_dir  : {args.data_dir}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  chunk_size: {args.chunk_size:,}")
    print(f"  threshold : {args.threshold}")
    print(f"  max_fuzzy : {args.max_fuzzy}")
    print(f"  max_total : {args.max_total}")
    print("=" * 80)

    wall_start = time.time()
    modes_to_run = ["train", "test"] if args.mode == "both" else [args.mode]

    for m in modes_to_run:
        run_step(
            mode=m,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            threshold=args.threshold,
            max_fuzzy=args.max_fuzzy,
            max_total=args.max_total,
        )

        # Post-hoc recall audit for train
        if m == "train" and not args.skip_audit:
            _, _, _, gt_path = _resolve_data_paths("train", args.data_dir)
            parquet_path = os.path.join(args.output_dir, "train_candidates.parquet")
            audit_recall(parquet_path, gt_path, label="train")

    wall_elapsed = time.time() - wall_start
    print("\n" + "=" * 80)
    print(f"  ALL STEPS COMPLETE -- Total wall time: {wall_elapsed/60:.2f} min")
    print("=" * 80)
    print("\n  Outputs:")
    for m in modes_to_run:
        pq_path = os.path.join(args.output_dir, f"{m}_candidates.parquet")
        if os.path.exists(pq_path):
            mb = os.path.getsize(pq_path) / (1024 * 1024)
            print(f"    {pq_path}  ({mb:.1f} MB)")
        if m == "test":
            tsv_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
            if os.path.exists(tsv_path):
                mb = os.path.getsize(tsv_path) / (1024 * 1024)
                print(f"    {tsv_path}  ({mb:.1f} MB)")
    print()


if __name__ == "__main__":
    main()
