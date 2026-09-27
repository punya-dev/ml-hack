#!/usr/bin/env python3
"""
=============================================================================
PIPELINE: RAW TRAIN DATA TO CANDIDATE BLOCKING
=============================================================================
This turnkey script automates the complete pipeline that takes raw train (and
optional test) TSVs and converts them into normalized, indexed candidate pools
and blocking outputs.

Pipeline Stages:
  Stage 0: Pre-flight Verification (Directory paths, raw TSVs, JSON configs)
  Stage 1 (Optional): Leak-Free Stratified Validation Split Creation
  Stage 2: Multi-Channel Candidate Blocking (Streaming Parquet & TSV)
           - Field Normalization (Country, Name, Address)
           - Inverted Index Ingestion (6 exact channels)
           - Country-Partitioned TF-IDF Char-Ngram Cosine Similarity
           - Deduplication & Candidate Capping
  Stage 3: Post-Hoc Recall Ceiling & Quality Audit (vs Ground Truth)
  Stage 4: Submission Format Verification (via utils/validate_submission.py)

Usage:
  # 1. Run full train & test blocking (default, 100,000 batch size):
  python pipeline_raw_to_blocking.py

  # 2. Run train blocking only (with recall ceiling audit):
  python pipeline_raw_to_blocking.py --mode train

  # 3. Run validation pipeline (creates val split then blocks & evaluates):
  python pipeline_raw_to_blocking.py --mode val --create-val

  # 4. Run test blocking only (creates candidate_pairs.tsv for inference):
  python pipeline_raw_to_blocking.py --mode test

  # 5. Run all (train, val, test):
  python pipeline_raw_to_blocking.py --mode all --create-val
=============================================================================
"""

import os
import sys
import time
import argparse
import subprocess
from typing import Dict, List, Set, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def _find_project_root(dirs: List[str]) -> str:
    """Walk up from dirs to find the repository/project root."""
    for d in dirs:
        curr = os.path.abspath(d)
        for _ in range(5):
            if (
                os.path.isdir(os.path.join(curr, "dataset"))
                or os.path.isdir(os.path.join(curr, "code", "business_entity_resolution"))
                or os.path.exists(os.path.join(curr, ".git"))
            ):
                return curr
            parent = os.path.dirname(curr)
            if parent == curr:
                break
            curr = parent
    return os.path.abspath(os.getcwd())


PROJECT_ROOT = _find_project_root([SCRIPT_DIR, REAL_SCRIPT_DIR, os.getcwd()])
CODE_DIR = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution")
SRC_DIR = os.path.join(CODE_DIR, "src")
ROOT_SRC = os.path.join(PROJECT_ROOT, "src")

# Prepend all candidate source directories to sys.path
for p in [SRC_DIR, ROOT_SRC, SCRIPT_DIR, REAL_SCRIPT_DIR, CODE_DIR, PROJECT_ROOT]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from code.business_entity_resolution.src.blocker import CandidateBlocker
except ImportError:
    try:
        from src.blocker import CandidateBlocker
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Dynamic Path Resolution
# ---------------------------------------------------------------------------
def detect_data_dir() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "dataset"),
        os.path.join(PROJECT_ROOT, "student_resource", "dataset"),
        os.path.join(os.getcwd(), "dataset"),
        os.path.join(os.getcwd(), "student_resource", "dataset"),
        os.path.join(SCRIPT_DIR, "dataset"),
        os.path.join(SCRIPT_DIR, "student_resource", "dataset"),
        "/content/ml-hack/student_resource/dataset",
        "/content/ml-hack/dataset",
        "/content/dataset",
    ]
    for c in candidates:
        if os.path.isdir(c) and (
            os.path.isdir(os.path.join(c, "train"))
            or os.path.isdir(os.path.join(c, "test"))
            or os.path.isdir(os.path.join(c, "val"))
        ):
            return os.path.abspath(c)
    return os.path.join(PROJECT_ROOT, "dataset")


def detect_output_dir() -> str:
    if os.path.exists("/content"):
        return "/content/ml-hack/output"
    out = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(out, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Pre-flight Checks
# ---------------------------------------------------------------------------
def preflight_check(data_dir: str, output_dir: str, mode: str, create_val: bool):
    print("=" * 80)
    print("STAGE 0: PRE-FLIGHT VERIFICATION")
    print("=" * 80)
    print(f"  Working Directory : {os.getcwd()}")
    print(f"  Project Root      : {SCRIPT_DIR}")
    print(f"  Data Directory    : {data_dir}")
    print(f"  Output Directory  : {output_dir}")
    print(f"  Selected Mode     : {mode}")

    # Check configs
    config_dir = os.path.join(SCRIPT_DIR, "configs")
    if not os.path.exists(config_dir):
        alt_cfg = os.path.join(SCRIPT_DIR, "code", "business_entity_resolution", "configs")
        if os.path.exists(alt_cfg):
            config_dir = alt_cfg
    print(f"  Configs Directory : {config_dir} (exists: {os.path.exists(config_dir)})")

    # Check raw train files
    train_dir = os.path.join(data_dir, "train")
    raw_train_files = [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
        "train_ground_truth.tsv",
    ]
    print("\n  Raw Train Files:")
    train_missing = []
    for f in raw_train_files:
        fpath = os.path.join(train_dir, f)
        exists = os.path.exists(fpath)
        size_mb = os.path.getsize(fpath) / (1024 * 1024) if exists else 0.0
        status = f"EXISTS ({size_mb:.1f} MB)" if exists else "MISSING"
        print(f"    - {f:<26}: {status}")
        if not exists:
            train_missing.append(fpath)

    if (mode in ("train", "val", "all", "both") or create_val) and train_missing:
        raise FileNotFoundError(f"Missing required raw train files in {train_dir}: {train_missing}")

    # Check test files if test mode
    if mode in ("test", "all", "both"):
        test_dir = os.path.join(data_dir, "test")
        raw_test_files = [
            "test_source1.tsv",
            "test_source2.tsv",
            "test_source3.tsv",
        ]
        print("\n  Raw Test Files:")
        test_missing = []
        for f in raw_test_files:
            fpath = os.path.join(test_dir, f)
            exists = os.path.exists(fpath)
            size_mb = os.path.getsize(fpath) / (1024 * 1024) if exists else 0.0
            status = f"EXISTS ({size_mb:.1f} MB)" if exists else "MISSING"
            print(f"    - {f:<26}: {status}")
            if not exists:
                test_missing.append(fpath)
        if test_missing:
            raise FileNotFoundError(f"Missing required raw test files in {test_dir}: {test_missing}")

    os.makedirs(output_dir, exist_ok=True)
    print("\n  [Pre-flight] All input paths validated successfully.\n")


# ---------------------------------------------------------------------------
# Stage 1: Create Validation Split
# ---------------------------------------------------------------------------
def run_stage_create_val_split(data_dir: str, val_size: int, seed: int):
    print("=" * 80)
    print("STAGE 1: GENERATING STRATIFIED LEAK-FREE VALIDATION SPLIT")
    print("=" * 80)
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    val_gt = os.path.join(val_dir, "val_ground_truth.tsv")

    if os.path.exists(val_gt) and os.path.getsize(val_gt) > 1000:
        print(f"  Existing validation split detected at {val_dir}. Re-verifying...")

    script_path = os.path.join(SCRIPT_DIR, "scripts", "create_validation_split.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Could not find validation script at {script_path}")

    cmd = [
        sys.executable,
        script_path,
        "--train-dir", train_dir,
        "--output-dir", val_dir,
        "--sample-size", str(val_size),
        "--seed", str(seed),
    ]
    print(f"  Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print(f"  [Stage 1 Complete] Validation set ready at: {val_dir}\n")


# ---------------------------------------------------------------------------
# Stage 2: Candidate Blocking
# ---------------------------------------------------------------------------
def run_stage_blocking(
    mode: str,
    data_dir: str,
    output_dir: str,
    chunk_size: int,
    threshold: float,
    max_fuzzy: int,
    max_total: int,
    n_jobs: Optional[int] = None,
    skip_audit: bool = False,
):
    detected_cores = os.cpu_count() or 4
    eff_n_jobs = n_jobs if n_jobs is not None else detected_cores

    print("=" * 80)
    print(f"STAGE 2: MULTI-CHANNEL CANDIDATE BLOCKING [{mode.upper()}]")
    print(f"  CPU Hardware  : {detected_cores} logical CPU cores detected | {eff_n_jobs} active worker cores")
    print(f"  Parallelization: Multi-processing (loky) + Multi-threaded BLAS/OpenMP")
    print("=" * 80)

    try:
        from code.business_entity_resolution.src.blocker import CandidateBlocker
    except ImportError:
        try:
            from src.blocker import CandidateBlocker
        except ImportError:
            for p in [SCRIPT_DIR, SRC_DIR, CODE_DIR]:
                if p not in sys.path:
                    sys.path.insert(0, p)
            from code.business_entity_resolution.src.blocker import CandidateBlocker

    # Resolve paths for split
    if mode == "val":
        s1 = os.path.join(data_dir, "val", "val_source1.tsv")
        s2 = os.path.join(data_dir, "val", "val_source2.tsv")
        s3 = os.path.join(data_dir, "val", "val_source3.tsv")
        gt = os.path.join(data_dir, "val", "val_ground_truth.tsv")
    elif mode == "train":
        s1 = os.path.join(data_dir, "train", "train_source1.tsv")
        s2 = os.path.join(data_dir, "train", "train_source2.tsv")
        s3 = os.path.join(data_dir, "train", "train_source3.tsv")
        gt = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    else:  # test
        s1 = os.path.join(data_dir, "test", "test_source1.tsv")
        s2 = os.path.join(data_dir, "test", "test_source2.tsv")
        s3 = os.path.join(data_dir, "test", "test_source3.tsv")
        gt = None

    for p in [s1, s2, s3]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Input file not found: {p}")

    out_parquet = os.path.join(output_dir, f"{mode}_candidates.parquet")
    out_tsv = os.path.join(output_dir, "candidate_pairs.tsv") if mode == "test" else os.path.join(output_dir, f"{mode}_candidate_pairs.tsv")
    summary_path = os.path.join(output_dir, f"{mode}_blocking_summary.txt")

    print(f"  S1 Input      : {s1}")
    print(f"  S2 Input      : {s2}")
    print(f"  S3 Input      : {s3}")
    if gt:
        print(f"  Ground Truth  : {gt}")
    print(f"  Parquet Output: {out_parquet}")
    print(f"  TSV Output    : {out_tsv}")
    print(f"  Chunk Size    : {chunk_size:,}")
    print(f"  Cosine Thresh : {threshold}")
    print(f"  Max Fuzzy     : {max_fuzzy}")
    print(f"  Max Total     : {max_total}")
    print(f"  Worker Cores  : {eff_n_jobs}\n")

    blocker = CandidateBlocker(
        fuzzy_threshold=threshold,
        max_fuzzy_candidates_per_s1=max_fuzzy,
        max_total_candidates_per_s1=max_total,
        n_jobs=eff_n_jobs,
    )

    t0 = time.time()
    blocker.run_batched_blocking(
        s1_path=s1,
        s2_path=s2,
        s3_path=s3,
        output_parquet=out_parquet,
        output_tsv=out_tsv,
        chunk_size=chunk_size,
        ground_truth_path=gt,
        verbose=True,
    )
    elapsed = time.time() - t0

    # Write summary
    with open(summary_path, "w") as f:
        f.write(f"mode            : {mode}\n")
        f.write(f"s1_path         : {s1}\n")
        f.write(f"s2_path         : {s2}\n")
        f.write(f"s3_path         : {s3}\n")
        f.write(f"gt_path         : {gt}\n")
        f.write(f"out_parquet     : {out_parquet}\n")
        f.write(f"out_tsv         : {out_tsv}\n")
        f.write(f"chunk_size      : {chunk_size}\n")
        f.write(f"threshold       : {threshold}\n")
        f.write(f"max_fuzzy       : {max_fuzzy}\n")
        f.write(f"max_total       : {max_total}\n")
        f.write(f"elapsed_seconds : {elapsed:.2f}\n")
        f.write(f"elapsed_minutes : {elapsed / 60:.2f}\n")

    print(f"\n  [Stage 2 Complete] {mode.upper()} blocking finished in {elapsed/60:.2f} min.")
    print(f"  Summary saved: {summary_path}")

    # Stage 3: Post-hoc audit if ground truth available
    if gt and os.path.exists(gt) and not skip_audit:
        run_stage_audit(out_parquet, gt, mode)

    # Stage 4: Validate format if candidate TSV exists
    if mode == "test" and os.path.exists(out_tsv):
        run_stage_validation(out_tsv, os.path.join(data_dir, "test"))


# ---------------------------------------------------------------------------
# Stage 3: Post-Hoc Recall Ceiling Audit
# ---------------------------------------------------------------------------
def run_stage_audit(parquet_path: str, gt_path: str, mode: str):
    import pandas as pd
    import numpy as np
    import pyarrow.parquet as pq

    print("\n" + "=" * 80)
    print(f"STAGE 3: POST-HOC RECALL CEILING & COVERAGE AUDIT [{mode.upper()}]")
    print("=" * 80)

    gt_df = pd.read_csv(gt_path, sep="\t")
    match_col = "matched_entity_ids" if "matched_entity_ids" in gt_df.columns else "matched_entity_id"
    gt_map = {}
    total_gt_pairs = 0
    for _, row in gt_df.iterrows():
        val = row[match_col]
        if pd.isna(val) or not str(val).strip():
            continue
        matches = {m.strip() for m in str(val).split(",") if m.strip() and m.strip() != "nan"}
        if matches:
            gt_map[row["source1_entity_id"]] = matches
            total_gt_pairs += len(matches)

    print(f"  Ground Truth: {len(gt_map):,} S1 entities with matches -> {total_gt_pairs:,} true pairs.")

    pf = pq.ParquetFile(parquet_path)
    retrieved = 0
    cand_counts = {}

    for batch in pf.iter_batches(batch_size=200_000):
        df = batch.to_pandas()
        valid = df[df["candidate_entity_id"].notna()]
        for s1_id, grp in valid.groupby("source1_entity_id"):
            cands = set(grp["candidate_entity_id"].tolist())
            true_matches = gt_map.get(s1_id, set())
            retrieved += len(true_matches & cands)
            cand_counts[s1_id] = cand_counts.get(s1_id, 0) + len(cands)

    recall = (retrieved / total_gt_pairs * 100.0) if total_gt_pairs > 0 else 0.0
    avg_cands = np.mean(list(cand_counts.values())) if cand_counts else 0.0
    max_cands = max(cand_counts.values()) if cand_counts else 0

    print("-" * 80)
    print(f"  True Pairs Retrieved : {retrieved:,} / {total_gt_pairs:,}")
    print(f"  Missed Pairs         : {total_gt_pairs - retrieved:,}")
    print(f"  >>> RECALL CEILING   : {recall:.4f}% <<<")
    print(f"  Avg Candidates / S1  : {avg_cands:.2f}")
    print(f"  Max Candidates / S1  : {max_cands}")
    if recall >= 99.0:
        print(f"  Result               : [PASS] Exceeds 99.0% recall requirement.")
    elif recall >= 97.0:
        print(f"  Result               : [ACCEPTABLE] Above 97.0% competitive baseline.")
    else:
        print(f"  Result               : [WARN] Recall is below 97.0%.")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Stage 4: Submission Validator Check
# ---------------------------------------------------------------------------
def run_stage_validation(candidate_tsv: str, test_dir: str):
    validator_path = os.path.join(SCRIPT_DIR, "utils", "validate_submission.py")
    if not os.path.exists(validator_path):
        return

    print("=" * 80)
    print("STAGE 4: SUBMISSION CANDIDATE FORMAT VALIDATION")
    print("=" * 80)
    cmd = [
        sys.executable,
        validator_path,
        "--candidate", candidate_tsv,
        "--test-dir", test_dir,
    ]
    try:
        subprocess.check_call(cmd)
        print("  [Validation] candidate_pairs.tsv successfully validated!\n")
    except subprocess.CalledProcessError as e:
        print(f"  [Validation Warning] Submission validator returned exit code {e.returncode}\n")


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End Pipeline: Raw Train Data -> Normalized Candidate Blocking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "val", "test", "both", "all"],
        default="both",
        help="Which split to block: 'both' (train + test), 'train', 'val', 'test', or 'all'.",
    )
    parser.add_argument(
        "--data-dir",
        default=detect_data_dir(),
        help="Root directory containing train/, val/, test/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=detect_output_dir(),
        help="Output directory to save parquets, TSVs, and summaries.",
    )
    parser.add_argument(
        "--create-val",
        action="store_true",
        default=False,
        help="Whether to generate the stratified validation split before blocking.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=50000,
        help="Number of S1 entities to include in the validation split.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Batch chunk size for S1 streaming (100,000 for fast server execution).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.24,
        help="Cosine similarity threshold for TF-IDF char-ngram channel.",
    )
    parser.add_argument(
        "--max-fuzzy",
        type=int,
        default=50,
        help="Maximum fuzzy candidates retrieved per S1 record.",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=80,
        help="Maximum total candidates (exact + fuzzy) kept per S1 record.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of CPU cores/workers to use for multiprocessing and parallel threads (default: all cores).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for validation sampling.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        default=False,
        help="Skip post-hoc recall audit against ground truth.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def main():
    t_pipeline_start = time.time()
    args = parse_args()
    detected_cores = os.cpu_count() or 4

    print("=" * 80)
    print("PIPELINE EXECUTION: RAW DATA TO CANDIDATE BLOCKING")
    print(f"  CPU Cores Available: {detected_cores} logical CPU cores detected")
    print(f"  Worker Pool Active : {args.n_jobs} cores allocated")
    print(f"  Execution Mode     : {args.mode.upper()}")
    print("=" * 80 + "\n")

    # Pre-flight
    preflight_check(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        create_val=args.create_val,
    )

    # Optional Stage 1: Validation Split
    if args.create_val or (args.mode in ("val", "all") and not os.path.exists(os.path.join(args.data_dir, "val", "val_ground_truth.tsv"))):
        run_stage_create_val_split(
            data_dir=args.data_dir,
            val_size=args.val_size,
            seed=args.seed,
        )

    # Stage 2 & 3: Run Candidate Blocking
    if args.mode == "both":
        modes = ["train", "test"]
    elif args.mode == "train":
        modes = ["train"]
    elif args.mode == "val":
        modes = ["val"]
    elif args.mode == "test":
        modes = ["test"]
    else:  # all
        modes = ["train", "val", "test"]

    for m in modes:
        run_stage_blocking(
            mode=m,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            threshold=args.threshold,
            max_fuzzy=args.max_fuzzy,
            max_total=args.max_total,
            n_jobs=args.n_jobs,
            skip_audit=args.skip_audit,
        )

    total_time = time.time() - t_pipeline_start
    print("=" * 80)
    print(f"PIPELINE RUN COMPLETE — Total Wall Time: {total_time/60:.2f} minutes")
    print("=" * 80)
    print("Generated Artifacts:")
    for f in os.listdir(args.output_dir):
        fp = os.path.join(args.output_dir, f)
        if os.path.isfile(fp):
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"  - {f:<30} ({size_mb:.2f} MB)")
    print()


if __name__ == "__main__":
    main()
