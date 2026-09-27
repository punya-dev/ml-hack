#!/usr/bin/env python3
"""
=============================================================================
PHASE 4 + 5 — DATASET ASSEMBLY & MODEL TRAINING
=============================================================================
Loads Phase-3 features, assembles the training matrix, trains a LightGBM
classifier, evaluates with macro-averaged F₀.₅ (the competition metric),
and runs a threshold/top-k sweep to find the best decision policy.

Usage:
    # Train on val, evaluate on val (local iteration loop):
    python phase45_train_model.py --split val

    # Train on train split (server/SageMaker):
    python phase45_train_model.py --split train \
        --eval-split val

    # Quick smoke-test (subsample 10k rows):
    python phase45_train_model.py --split val --sample 10000

Outputs:
    data/models/lgbm_val.pkl             — trained LightGBM model
    data/models/training_report_val.txt  — full metrics + feature importance
=============================================================================
"""

import os
import sys
import json
import time
import logging
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Feature columns used for training
# Excludes zero-variance (country_match always=1), near-useless landmark_jaccard
# flagged during Phase-3 audit, and ID/meta columns.
# Edit this list to ablate features during iteration.
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # Name
    "token_sort_ratio",
    "token_set_ratio",
    "partial_ratio",
    "jaro_winkler",
    "char_3gram_jaccard",
    "len_ratio",
    "best_alt_name_score",
    "core_name_exact",
    # Address
    "street_number_exact",
    "street_name_fuzzy",
    "city_fuzzy",
    "state_equal",
    "postal_exact",
    "both_postal_present",
    "landmark_jaccard",       # kept — LGB will learn its low importance
    "sorted_address_tokens_exact",
    # Channel provenance
    "exact_ch1_core_name",
    "exact_ch2_sorted_tokens",
    "exact_ch3_state_prefix",
    "exact_ch4_state_phonetic",
    "exact_ch5_acronym",
    "exact_ch6_address_tokens",
    "tfidf_cosine_score",
    "channel_hit_count",
    # Missingness
    "s1_missing_street",
    "s1_missing_city",
    "s1_missing_state",
    "s1_missing_postal",
    "cand_missing_street",
    "cand_missing_city",
    "cand_missing_state",
    "cand_missing_postal",
    # NOTE: country_match omitted (zero variance = always 1)
]

ID_COLS = ["source1_entity_id", "candidate_entity_id"]

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _features_path(split: str) -> Path:
    return SCRIPT_DIR / "data" / "features" / f"{split}_features.parquet"

def _model_path(split: str) -> Path:
    d = SCRIPT_DIR / "data" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"lgbm_{split}.pkl"

def _report_path(split: str) -> Path:
    d = SCRIPT_DIR / "data" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"training_report_{split}.txt"

def _singletons_path(split: str) -> Path:
    return SCRIPT_DIR / "data" / "labeled" / f"{split}_singletons.parquet"

def _gt_path(split: str) -> Path:
    return SCRIPT_DIR / "dataset" / split / f"{split}_ground_truth.tsv"


# ---------------------------------------------------------------------------
# F₀.₅ scorer (per-entity macro-average — the competition metric)
# ---------------------------------------------------------------------------
def compute_f05(true_ids: Set[str], pred_ids: Set[str]) -> float:
    is_true_singleton = len(true_ids) == 0
    is_pred_singleton = len(pred_ids) == 0
    if is_true_singleton:
        return 1.0 if is_pred_singleton else 0.0
    if is_pred_singleton:
        return 0.0
    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(
    df: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    pred_col: str = "pred_label",
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
) -> Tuple[float, Dict]:
    """
    Compute macro-averaged F₀.₅ across ALL S1 entities in gt_map.
    Singletons (0 GT matches) are included: correct empty prediction → 1.0.
    """
    # Build predicted sets per S1 from rows where pred_label=1
    pred_sets: Dict[str, Set[str]] = {}
    for _, row in df[df[pred_col] == 1].iterrows():
        s1 = row[s1_col]
        pred_sets.setdefault(s1, set()).add(row[cand_col])

    scores = []
    for s1_id, true_ids in gt_map.items():
        pred_ids = pred_sets.get(s1_id, set())
        scores.append(compute_f05(true_ids, pred_ids))

    macro = float(np.mean(scores)) if scores else 0.0
    breakdown = {
        "macro_f05": macro,
        "n_entities": len(scores),
        "n_predicted_match": sum(1 for s in pred_sets.values() if len(s) > 0),
    }
    return macro, breakdown


def load_gt_map(split: str) -> Dict[str, Set[str]]:
    gt_path = _gt_path(split)
    if not gt_path.exists():
        log.warning(f"GT not found for split={split}: {gt_path}")
        return {}
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    gt_map: Dict[str, Set[str]] = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"].strip()
        raw = row["matched_entity_ids"].strip()
        gt_map[s1_id] = {m.strip() for m in raw.split(",") if m.strip()} if raw else set()
    return gt_map


# ---------------------------------------------------------------------------
# Threshold sweep — finds per-entity decision policy maximising F₀.₅
# ---------------------------------------------------------------------------
def threshold_sweep(
    df: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    score_col: str = "score",
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, Dict]:
    """
    Sweep a global score threshold and return (best_threshold, best_metrics).
    Also evaluates top-k and margin policies for comparison.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37).tolist()  # 0.05 step

    log.info(f"  Threshold sweep over {len(thresholds)} values ...")
    best_thr, best_f05, best_metrics = 0.5, 0.0, {}

    results = []
    for thr in thresholds:
        df_copy = df.copy()
        df_copy["pred_label"] = (df_copy[score_col] >= thr).astype(int)
        f05, meta = macro_f05(df_copy, gt_map)
        results.append((thr, f05))
        if f05 > best_f05:
            best_f05, best_thr = f05, thr
            best_metrics = {**meta, "threshold": thr}

    log.info(f"  Best threshold: {best_thr:.3f}  →  macro-F₀.₅ = {best_f05:.6f}")

    # ── Top-k sweep (per entity, keep top-k by score) ──────────────────
    log.info("  Top-k sweep ...")
    best_k, best_k_f05 = 1, 0.0
    for k in [1, 2, 3, 5, 8, 10, 15, 20]:
        df_topk = (
            df.sort_values(score_col, ascending=False)
            .groupby(s1_col, group_keys=False)
            .head(k)
            .copy()
        )
        df_topk["pred_label"] = 1
        # Entities with 0 predicted rows → empty set = singleton prediction
        f05_k, _ = macro_f05(df_topk, gt_map)
        if f05_k > best_k_f05:
            best_k_f05, best_k = f05_k, k

    log.info(f"  Best top-k: k={best_k}  →  macro-F₀.₅ = {best_k_f05:.6f}")

    # Return the better of the two policies
    if best_k_f05 > best_f05:
        best_metrics = {"top_k": best_k, "macro_f05": best_k_f05, "n_entities": len(gt_map)}
        return best_k_f05, best_metrics
    return best_f05, best_metrics


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------
def get_lgbm_params(scale_pos_weight: float) -> Dict:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
    }


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    scale_pos_weight: float,
) -> lgb.LGBMClassifier:
    params = get_lgbm_params(scale_pos_weight)
    model = lgb.LGBMClassifier(**params)

    log.info(f"  Training LightGBM  (scale_pos_weight={scale_pos_weight:.2f}) ...")
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    elapsed = time.time() - t0
    log.info(
        f"  Trained {model.best_iteration_} trees in {elapsed:.1f}s  "
        f"(best val logloss = {model.best_score_['valid_0']['binary_logloss']:.6f})"
    )
    return model


# ---------------------------------------------------------------------------
# Feature importance report
# ---------------------------------------------------------------------------
def importance_table(model: lgb.LGBMClassifier, feature_names: List[str]) -> str:
    imp = model.feature_importances_
    pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
    lines = [
        "",
        "[Feature Importances — split gain]",
        f"  {'Feature':<32}  {'Importance':>12}  {'% of total':>10}",
        "  " + "-" * 58,
    ]
    total = sum(imp)
    for name, val in pairs:
        pct = val / total * 100 if total > 0 else 0
        lines.append(f"  {name:<32}  {val:>12.0f}  {pct:>10.2f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Phase 4+5: Train LightGBM entity matcher.")
    p.add_argument("--split", required=True, choices=["val", "train"],
                   help="Features split to train on.")
    p.add_argument("--eval-split", default=None,
                   help="Separate split to evaluate on after training (default: same as --split).")
    p.add_argument("--features", type=Path, default=None,
                   help="Path to features parquet. Auto-detected if omitted.")
    p.add_argument("--sample", type=int, default=None,
                   help="Subsample N rows for smoke-testing.")
    p.add_argument("--cv-folds", type=int, default=0,
                   help="Number of stratified CV folds (0 = no CV, train on full split).")
    p.add_argument("--no-write", action="store_true",
                   help="Skip writing model to disk.")
    return p.parse_args()


def main():
    args = parse_args()
    split = args.split
    eval_split = args.eval_split or split
    feat_path = args.features or _features_path(split)

    log.info("=" * 62)
    log.info(f"  PHASE 4+5 — MODEL TRAINING  (split={split})")
    log.info("=" * 62)
    log.info(f"  Features : {feat_path}")

    if not feat_path.exists():
        log.error(f"Features parquet not found: {feat_path}")
        log.error("Run phase3_feature_engineering.py --split {split} first.")
        sys.exit(1)

    t_global = time.time()

    # ── Load features ─────────────────────────────────────────────────────
    log.info("Loading features ...")
    df = pq.read_table(feat_path).to_pandas()

    if args.sample:
        log.info(f"  Subsampling to {args.sample:,} rows ...")
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)

    log.info(f"  Shape: {df.shape}  |  Positives: {(df['label']==1).sum():,}")

    # ── Validate feature columns ──────────────────────────────────────────
    missing_feats = [f for f in FEATURE_COLS if f not in df.columns]
    if missing_feats:
        log.warning(f"Missing features (will skip): {missing_feats}")
    use_feats = [f for f in FEATURE_COLS if f in df.columns]
    log.info(f"  Using {len(use_feats)} features")

    X = df[use_feats].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    log.info(f"  scale_pos_weight = {n_neg}/{n_pos} = {scale_pos_weight:.2f}")

    # ── Load ground truth for F₀.₅ evaluation ────────────────────────────
    gt_map = load_gt_map(eval_split)
    if not gt_map:
        log.warning("No ground truth loaded — F₀.₅ evaluation will be skipped.")

    # ── Optional CV for variance estimate ─────────────────────────────────
    cv_f05_scores: List[float] = []
    if args.cv_folds > 1:
        log.info(f"Running {args.cv_folds}-fold stratified CV ...")
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]
            model_cv = train_lgbm(X_tr, y_tr, X_va, y_va, use_feats, scale_pos_weight)

            if gt_map:
                df_va = df.iloc[va_idx].copy()
                df_va["score"] = model_cv.predict_proba(X_va)[:, 1]
                best_f05, meta = threshold_sweep(df_va, gt_map)
                cv_f05_scores.append(best_f05)
                log.info(f"  Fold {fold}: macro-F₀.₅ = {best_f05:.6f}")

        if cv_f05_scores:
            log.info(
                f"  CV macro-F₀.₅: {np.mean(cv_f05_scores):.6f} "
                f"± {np.std(cv_f05_scores):.6f}"
            )

    # ── Full training split for final model ───────────────────────────────
    # Use 90/10 stratified split for early-stopping monitor
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va, idx_tr, idx_va = train_test_split(
        X, y, np.arange(len(df)),
        test_size=0.10, stratify=y, random_state=42,
    )
    log.info(f"  Train/val split: {len(X_tr):,} / {len(X_va):,}")

    model = train_lgbm(X_tr, y_tr, X_va, y_va, use_feats, scale_pos_weight)

    # ── Predict on full split ─────────────────────────────────────────────
    log.info("Predicting on full split ...")
    df["score"] = model.predict_proba(X)[:, 1]

    # ── F₀.₅ threshold sweep ──────────────────────────────────────────────
    report_lines = [
        "=" * 70,
        f"  PHASE 4+5 TRAINING REPORT  |  split = {split.upper()}",
        "=" * 70,
        f"  Total rows         : {len(df):,}",
        f"  Positives          : {n_pos:,}  ({n_pos/len(df):.4%})",
        f"  Negatives          : {n_neg:,}",
        f"  scale_pos_weight   : {scale_pos_weight:.2f}",
        f"  Features used      : {len(use_feats)}",
        f"  Best LGB iteration : {model.best_iteration_}",
        "",
    ]

    if cv_f05_scores:
        report_lines += [
            f"[{args.cv_folds}-Fold CV]",
            f"  macro-F₀.₅ = {np.mean(cv_f05_scores):.6f} ± {np.std(cv_f05_scores):.6f}",
            "",
        ]

    if gt_map:
        best_f05, best_meta = threshold_sweep(df, gt_map)
        policy = best_meta.get("threshold") or best_meta.get("top_k")
        policy_type = "threshold" if "threshold" in best_meta else "top_k"
        report_lines += [
            f"[Decision Policy Sweep — macro-F₀.₅]",
            f"  Best {policy_type:<12}: {policy}",
            f"  Best macro-F₀.₅   : {best_f05:.6f}",
            f"  Entities evaluated : {best_meta.get('n_entities', 'n/a'):,}",
            "",
        ]
        # Fast per-country F0.5 breakdown at best threshold
        if "threshold" in best_meta and "country" in df.columns:
            thr = best_meta["threshold"]
            df["pred_label"] = (df["score"] >= thr).astype(int)
            report_lines.append(f"[Country Breakdown at threshold={thr:.3f}]")

            # Build per-S1 predicted sets in one vectorised pass
            pred_sets_all: Dict[str, Set[str]] = {}
            for row in df[df["pred_label"] == 1][["source1_entity_id", "candidate_entity_id"]].itertuples(index=False):
                pred_sets_all.setdefault(row.source1_entity_id, set()).add(row.candidate_entity_id)

            # Map S1 -> country
            s1_country = (
                df[["source1_entity_id", "country"]]
                .drop_duplicates("source1_entity_id")
                .set_index("source1_entity_id")["country"]
            )

            for country in sorted(df["country"].dropna().unique()):
                c_scores = [
                    compute_f05(true_ids, pred_sets_all.get(s1_id, set()))
                    for s1_id, true_ids in gt_map.items()
                    if s1_country.get(s1_id) == country
                ]
                if c_scores:
                    report_lines.append(
                        f"  {str(country):<10}: macro-F0.5 = {np.mean(c_scores):.6f} "
                        f"(n={len(c_scores):,})"
                    )
            report_lines.append("")

    report_lines.append(importance_table(model, use_feats))
    report_lines += ["", "=" * 70, ""]
    report = "\n".join(report_lines)
    print("\n" + report)

    # ── Write model ───────────────────────────────────────────────────────
    if not args.no_write:
        model_path = _model_path(split)
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "feature_cols": use_feats}, f)
        log.info(f"  Model saved -> {model_path}")

        rpt_path = _report_path(split)
        rpt_path.write_text(report, encoding="utf-8")
        log.info(f"  Report saved -> {rpt_path}")

    log.info(f"  Phase 4+5 complete in {time.time()-t_global:.1f}s")


if __name__ == "__main__":
    main()
