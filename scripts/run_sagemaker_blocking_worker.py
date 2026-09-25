"""
SageMaker Processing Job Worker Script for Candidate Blocking.
Executes Step 2 (Train Blocking) or Step 3 (Test Blocking) inside the SageMaker container.
"""
import os
import sys
import time
import subprocess
import argparse

# Force unbuffered output so CloudWatch receives real-time progress
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def install_dependencies():
    print("[Worker] Checking and installing required dependencies...")
    t0 = time.time()
    reqs = ["pyarrow", "rapidfuzz", "anyascii", "unidecode", "wordninja", "tqdm"]
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + reqs
    subprocess.check_call(cmd)
    print(f"[Worker] Dependencies installed successfully in {time.time() - t0:.2f}s.")

def main():
    parser = argparse.ArgumentParser(description="SageMaker Candidate Blocker Worker")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Mode: train or test")
    parser.add_argument("--data-dir", default="/opt/ml/processing/input/data", help="Data directory")
    parser.add_argument("--code-dir", default="/opt/ml/processing/input/code", help="Code directory")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output", help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=50000, help="S1 batch chunk size")
    parser.add_argument("--threshold", type=float, default=0.24, help="Fuzzy TF-IDF cosine threshold")
    parser.add_argument("--max-fuzzy", type=int, default=50, help="Max fuzzy candidates per S1 entity")
    parser.add_argument("--max-total", type=int, default=80, help="Max total candidates per S1 entity")
    args = parser.parse_args()

    print("=" * 80)
    print(f"SAGEMAKER CANDIDATE BLOCKING WORKER: MODE={args.mode.upper()}")
    print("=" * 80)
    print(f"Python: {sys.version}")
    print(f"Code Dir: {args.code_dir}")
    print(f"Data Dir: {args.data_dir}")
    print(f"Output Dir: {args.output_dir}")

    # Install packages first
    install_dependencies()

    # Set working directory to code_dir and add to sys.path
    os.chdir(args.code_dir)
    sys.path.insert(0, args.code_dir)

    from src.blocker import CandidateBlocker

    os.makedirs(args.output_dir, exist_ok=True)

    blocker = CandidateBlocker(
        fuzzy_threshold=args.threshold,
        max_fuzzy_candidates_per_s1=args.max_fuzzy,
        max_total_candidates_per_s1=args.max_total,
    )

    if args.mode == "train":
        s1_path = os.path.join(args.data_dir, "train_source1.tsv")
        s2_path = os.path.join(args.data_dir, "train_source2.tsv")
        s3_path = os.path.join(args.data_dir, "train_source3.tsv")
        gt_path = os.path.join(args.data_dir, "train_ground_truth.tsv")
        out_parquet = os.path.join(args.output_dir, "train_candidates.parquet")
        out_tsv = None
    else:
        s1_path = os.path.join(args.data_dir, "test_source1.tsv")
        s2_path = os.path.join(args.data_dir, "test_source2.tsv")
        s3_path = os.path.join(args.data_dir, "test_source3.tsv")
        gt_path = None
        out_parquet = os.path.join(args.output_dir, "test_candidates.parquet")
        out_tsv = os.path.join(args.output_dir, "candidate_pairs.tsv")

    print(f"\n[Worker] Launching {args.mode} blocking execution...")
    t_start = time.time()
    blocker.run_batched_blocking(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        output_parquet=out_parquet,
        output_tsv=out_tsv,
        chunk_size=args.chunk_size,
        ground_truth_path=gt_path,
        verbose=True,
    )
    total_time = time.time() - t_start

    # Summary report
    summary_path = os.path.join(args.output_dir, f"{args.mode}_blocking_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Total time seconds: {total_time:.2f}\n")
        f.write(f"Parquet output: {out_parquet}\n")
        if out_tsv:
            f.write(f"TSV output: {out_tsv}\n")

    print("\n" + "=" * 80)
    print(f"SAGEMAKER WORKER COMPLETED SUCCESSFULLY IN {total_time:.2f}s ({total_time / 60:.2f} mins)")
    print("=" * 80)


if __name__ == "__main__":
    main()
