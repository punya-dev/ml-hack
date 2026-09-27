#!/usr/bin/env python3
"""
=============================================================================
PHASE 2 — CANDIDATE LABELING
=============================================================================
Joins Phase-1 candidate pairs with ground-truth matches to produce a binary
`label` column.  Preserves true singletons (S1 entities with 0 GT matches)
so the downstream feature + model code can treat them correctly.

Usage:
    python phase2_label_candidates.py --split val
    python phase2_label_candidates.py --split train

    # Custom paths override (e.g. SageMaker / Colab):
    python phase2_label_candidates.py --split train \
        --candidates /data/candidates/train_candidates.parquet \
        --ground-truth /data/train_ground_truth.tsv \
        --output /data/labeled/train_labeled.parquet

Outputs:
    data/labeled/{split}_labeled.parquet   — labeled candidate pairs
    data/labeled/{split}_singletons.parquet — true-singleton S1 manifest
    data/labeled/{split}_label_audit.txt   — label-balance + channel audit
=============================================================================
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-root resolution (works from any CWD)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Channel column names — must match Phase-1 parquet schema exactly
EXACT_CHANNELS = [
    "exact_ch1_core_name",
    "exact_ch2_sorted_tokens",
    "exact_ch3_state_prefix",
    "exact_ch4_state_phonetic",
    "exact_ch5_acronym",
    "exact_ch6_address_tokens",
]

CHANNEL_LABELS = {
    "exact_ch1_core_name":     "Ch1  core_name",
    "exact_ch2_sorted_tokens": "Ch2  sorted_tokens",
    "exact_ch3_state_prefix":  "Ch3  state+prefix",
    "exact_ch4_state_phonetic":"Ch4  state+phonetic",
    "exact_ch5_acronym":       "Ch5  acronym",
    "exact_ch6_address_tokens":"Ch6  address_tokens",
    "tfidf_only":              "Ch7  tfidf_only (no exact hit)",
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _default_candidates(split: str) -> Path:
    """Return the most likely location of the candidates parquet."""
    # Priority 1: plan-standardised layout data/candidates/
    p = SCRIPT_DIR / "data" / "candidates" / f"{split}_candidates.parquet"
    if p.exists():
        return p
    # Priority 2: flat output/ layout (current actual layout)
    p = SCRIPT_DIR / "output" / f"{split}_candidates.parquet"
    if p.exists():
        return p
    return SCRIPT_DIR / "output" / f"{split}_candidates.parquet"


def _default_ground_truth(split: str) -> Path:
    """Return the most likely location of the ground-truth TSV."""
    return SCRIPT_DIR / "dataset" / split / f"{split}_ground_truth.tsv"


def _default_output(split: str) -> Path:
    """Standardised output under data/labeled/."""
    out = SCRIPT_DIR / "data" / "labeled"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{split}_labeled.parquet"


def _default_audit(split: str) -> Path:
    out = SCRIPT_DIR / "data" / "labeled"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{split}_label_audit.txt"


# ---------------------------------------------------------------------------
# Core: explode ground truth -> {s1_id: set(matched_ids)}
# ---------------------------------------------------------------------------
def load_ground_truth(gt_path: Path) -> Dict[str, set]:
    """
    Read ground-truth TSV and return a dict mapping every S1 entity to the
    *set* of matched S2/S3 IDs.  Singletons map to an empty set.

    TSV schema: source1_entity_id  matched_entity_ids (comma-separated)
    """
    log.info(f"Loading ground truth from {gt_path}")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")

    required_cols = {"source1_entity_id", "matched_entity_ids"}
    missing = required_cols - set(gt_df.columns)
    if missing:
        raise ValueError(f"Ground-truth TSV missing columns: {missing}")

    gt_map: Dict[str, set] = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        raw = row["matched_entity_ids"].strip()
        if raw:
            matches = {m.strip() for m in raw.split(",") if m.strip()}
        else:
            matches = set()  # true singleton
        gt_map[s1_id] = matches

    n_singletons = sum(1 for v in gt_map.values() if len(v) == 0)
    n_pairs = sum(len(v) for v in gt_map.values())
    log.info(
        f"  Ground truth: {len(gt_map):,} S1 entities | "
        f"{n_pairs:,} match pairs | {n_singletons:,} true singletons"
    )
    return gt_map


# ---------------------------------------------------------------------------
# Core: label candidate pairs
# ---------------------------------------------------------------------------
def label_candidates(
    cand_df: pd.DataFrame,
    gt_map: Dict[str, set],
) -> pd.DataFrame:
    """
    Assign binary `label` to every candidate pair in `cand_df`.

    label = 1  iff  candidate_entity_id in gt_map[source1_entity_id]
    label = 0  otherwise (hard negative)

    Uses a vectorised merge: explode GT into a flat (s1, candidate) table,
    left-join against candidates, fill NaN -> 0.
    """
    log.info(f"Labeling {len(cand_df):,} candidate pairs ...")

    t0 = time.time()

    # Explode GT into flat (s1_id, candidate_id) table
    gt_pairs: List[tuple] = []
    for s1_id, matches in gt_map.items():
        for m in matches:
            gt_pairs.append((s1_id, m))

    if gt_pairs:
        gt_flat = pd.DataFrame(gt_pairs, columns=["source1_entity_id", "candidate_entity_id"])
        gt_flat["label"] = pa.array([1] * len(gt_flat), type=pa.int8())
    else:
        gt_flat = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "label"]
        )
        gt_flat["label"] = gt_flat["label"].astype("int8")

    # Left-merge: candidates keep all rows; GT hits get label=1, rest NaN -> 0
    labeled = cand_df.merge(
        gt_flat,
        on=["source1_entity_id", "candidate_entity_id"],
        how="left",
    )
    labeled["label"] = labeled["label"].fillna(0).astype("int8")

    elapsed = time.time() - t0
    log.info(f"  Merge complete in {elapsed:.2f}s")
    return labeled


# ---------------------------------------------------------------------------
# Singleton manifest builder
# ---------------------------------------------------------------------------
def build_singleton_rows(gt_map: Dict[str, set], cand_s1_ids: set) -> pd.DataFrame:
    """
    Build a manifest of true-singleton S1 entities (GT has 0 matches).
    Records whether the blocker correctly returned 0 candidates for them.

    Does NOT add synthetic rows to the labeled set — just a metadata table.
    """
    singletons = [s1 for s1, m in gt_map.items() if len(m) == 0]
    df = pd.DataFrame({"source1_entity_id": singletons})
    df["is_true_singleton"] = True
    df["blocker_returned_empty"] = ~df["source1_entity_id"].isin(cand_s1_ids)

    correctly_empty = int(df["blocker_returned_empty"].sum())
    log.info(
        f"  True singletons: {len(singletons):,} | "
        f"Blocker correctly empty for {correctly_empty:,}"
    )
    return df


# ---------------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------------
def build_audit(
    labeled: pd.DataFrame,
    singleton_df: pd.DataFrame,
    gt_map: Dict[str, set],
    split: str,
    audit_path: Path,
) -> str:
    """Compute, print, and write the label-balance + channel audit report."""

    n_total = len(labeled)
    n_pos   = int((labeled["label"] == 1).sum())
    n_neg   = n_total - n_pos
    pos_rate = n_pos / n_total if n_total > 0 else 0.0
    ratio    = n_neg / n_pos if n_pos > 0 else float("inf")

    total_gt_pairs   = sum(len(v) for v in gt_map.values())
    recall_in_cands  = n_pos / total_gt_pairs if total_gt_pairs > 0 else 0.0
    missed           = total_gt_pairs - n_pos

    # Per-channel stats
    channel_stats: List[tuple] = []
    for ch in EXACT_CHANNELS:
        if ch in labeled.columns:
            ch_pos = int(labeled.loc[labeled["label"] == 1, ch].sum())
            ch_tot = int(labeled[ch].sum())
            channel_stats.append((CHANNEL_LABELS[ch], ch_pos, ch_tot))

    # tfidf-only positives: label=1 AND no exact channel fired
    if all(c in labeled.columns for c in EXACT_CHANNELS):
        any_exact = labeled[EXACT_CHANNELS].max(axis=1)
        tf_pos = int(((labeled["label"] == 1) & (any_exact == 0)).sum())
        tf_tot = int((any_exact == 0).sum())
        channel_stats.append((CHANNEL_LABELS["tfidf_only"], tf_pos, tf_tot))

    lines = [
        "=" * 72,
        f"  PHASE 2 LABEL AUDIT  |  split = {split.upper()}",
        "=" * 72,
        "",
        "[Candidate Pool]",
        f"  Total candidate pairs        : {n_total:>12,}",
        f"  Positive (label=1) pairs     : {n_pos:>12,}  ({pos_rate:.4%})",
        f"  Negative (label=0) pairs     : {n_neg:>12,}  ({1-pos_rate:.4%})",
        f"  Neg : Pos ratio              : {ratio:>11.2f} : 1",
        "",
        "[Recall Ceiling Check]",
        f"  Total GT match pairs         : {total_gt_pairs:>12,}",
        f"  GT pairs found in candidates : {n_pos:>12,}  ({recall_in_cands:.4%})",
        f"  GT pairs missed by blocker   : {missed:>12,}",
        "",
        "[Singletons]",
        f"  True singleton S1 entities   : {len(singleton_df):>12,}",
        f"  Blocker correctly returned 0 : {int(singleton_df['blocker_returned_empty'].sum()):>12,}",
        f"  Blocker returned >=1 cand.   : {int((~singleton_df['blocker_returned_empty']).sum()):>12,}",
        "",
        "[Per-Channel Positive Contribution]",
        f"  {'Channel':<30}  {'Pos via ch':>10}  {'% of all pos':>13}  {'Total via ch':>13}",
        "  " + "-" * 72,
    ]
    for ch_label, ch_pos, ch_tot in channel_stats:
        pct = ch_pos / n_pos * 100 if n_pos > 0 else 0.0
        lines.append(
            f"  {ch_label:<30}  {ch_pos:>10,}  {pct:>12.2f}%  {ch_tot:>13,}"
        )

    # Country breakdown
    if "country" in labeled.columns:
        lines += [
            "",
            "[Country Breakdown]",
            f"  {'Country':<12}  {'Total':>10}  {'Pos':>10}  {'Pos%':>8}  {'Neg:Pos':>9}",
        ]
        for country, grp in labeled.groupby("country", observed=True):
            g_tot = len(grp)
            g_pos = int((grp["label"] == 1).sum())
            g_neg = g_tot - g_pos
            g_rate = g_pos / g_tot if g_tot > 0 else 0.0
            g_ratio = g_neg / g_pos if g_pos > 0 else float("inf")
            lines.append(
                f"  {str(country):<12}  {g_tot:>10,}  {g_pos:>10,}  "
                f"{g_rate:>8.4%}  {g_ratio:>8.2f}:1"
            )

    lines += ["", "=" * 72, ""]
    report = "\n".join(lines)

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(report, encoding="utf-8")
    log.info(f"  Audit written -> {audit_path}")
    return report


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_labeled_parquet(labeled: pd.DataFrame, output_path: Path):
    if "tfidf_cosine_score" in labeled.columns:
        labeled["tfidf_cosine_score"] = labeled["tfidf_cosine_score"].astype("float32")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(labeled, preserve_index=False)
    pq.write_table(table, output_path, compression="snappy")
    size_mb = output_path.stat().st_size / 1_048_576
    log.info(f"  Labeled parquet -> {output_path}  ({size_mb:.1f} MB, {len(labeled):,} rows)")


def write_singleton_manifest(singleton_df: pd.DataFrame, labeled_path: Path):
    if singleton_df.empty:
        log.info("  No true singletons — skipping singleton manifest.")
        return
    out = labeled_path.parent / (labeled_path.stem.replace("_labeled", "_singletons") + ".parquet")
    table = pa.Table.from_pandas(singleton_df, preserve_index=False)
    pq.write_table(table, out, compression="snappy")
    log.info(f"  Singleton manifest -> {out}  ({len(singleton_df):,} rows)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2: Label candidate pairs against ground truth."
    )
    p.add_argument(
        "--split", required=True, choices=["val", "train"],
        help="Which split to label.",
    )
    p.add_argument(
        "--candidates", type=Path, default=None,
        help="Path to Phase-1 candidates parquet.  Auto-detected if omitted.",
    )
    p.add_argument(
        "--ground-truth", type=Path, default=None,
        help="Path to ground-truth TSV.  Auto-detected if omitted.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output path for labeled parquet.  Default: data/labeled/{split}_labeled.parquet",
    )
    p.add_argument(
        "--audit", type=Path, default=None,
        help="Output path for the audit text report.",
    )
    p.add_argument(
        "--no-write", action="store_true",
        help="Run all audits but skip writing output parquet (dry-run).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    split = args.split

    cand_path  = args.candidates   or _default_candidates(split)
    gt_path    = args.ground_truth or _default_ground_truth(split)
    out_path   = args.output       or _default_output(split)
    audit_path = args.audit        or _default_audit(split)

    log.info("=" * 60)
    log.info(f"  PHASE 2 — CANDIDATE LABELING  (split={split})")
    log.info("=" * 60)
    log.info(f"  Candidates : {cand_path}")
    log.info(f"  GT         : {gt_path}")
    log.info(f"  Output     : {out_path}")
    log.info(f"  Audit      : {audit_path}")

    # --- Validate inputs ---
    if not cand_path.exists():
        log.error(f"Candidates parquet not found: {cand_path}")
        log.error("Hint: run blocking first, or pass --candidates <path>.")
        sys.exit(1)
    if not gt_path.exists():
        log.error(f"Ground-truth TSV not found: {gt_path}")
        sys.exit(1)

    t_start = time.time()

    # --- Load candidates ---
    log.info("Loading candidate pairs ...")
    cand_df = pq.read_table(cand_path).to_pandas()
    log.info(f"  Candidate rows: {len(cand_df):,}")

    missing_cols = {"source1_entity_id", "candidate_entity_id"} - set(cand_df.columns)
    if missing_cols:
        log.error(f"Candidate parquet missing required columns: {missing_cols}")
        sys.exit(1)

    missing_chs = [c for c in EXACT_CHANNELS if c not in cand_df.columns]
    if missing_chs:
        log.warning(f"Missing channel columns (won't appear in audit): {missing_chs}")

    # --- Load ground truth ---
    gt_map = load_ground_truth(gt_path)

    # --- Coverage check ---
    cand_s1 = set(cand_df["source1_entity_id"].unique())
    gt_s1   = set(gt_map.keys())
    missing_s1 = gt_s1 - cand_s1
    if missing_s1:
        log.warning(
            f"  {len(missing_s1):,} S1 entities in GT but absent from candidates "
            f"(singletons or blocker gaps)."
        )

    # --- Label ---
    labeled = label_candidates(cand_df, gt_map)

    # --- Singleton manifest ---
    singleton_df = build_singleton_rows(gt_map, cand_s1)

    # --- Quick summary ---
    n_pos = int((labeled["label"] == 1).sum())
    total_gt = sum(len(v) for v in gt_map.values())
    recall   = n_pos / total_gt if total_gt > 0 else 0.0
    log.info(
        f"  Label balance: {n_pos:,} pos / {len(labeled)-n_pos:,} neg  "
        f"(recall = {recall:.4%})"
    )
    if recall < 0.95:
        log.warning(f"  Recall {recall:.4%} < 95% — check blocker or GT path.")

    # --- Audit ---
    report = build_audit(labeled, singleton_df, gt_map, split, audit_path)
    print("\n" + report)

    # --- Write ---
    if not args.no_write:
        write_labeled_parquet(labeled, out_path)
        write_singleton_manifest(singleton_df, out_path)
    else:
        log.info("  --no-write flag set, skipping output.")

    log.info(f"  Phase 2 complete in {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
