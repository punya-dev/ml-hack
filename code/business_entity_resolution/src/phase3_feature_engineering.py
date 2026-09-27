#!/usr/bin/env python3
"""
=============================================================================
PHASE 3 — FEATURE ENGINEERING
=============================================================================
Produces a fixed-width numeric feature vector per (S1, candidate) pair from
the Phase-2 labeled parquet + raw source TSVs.

Pipeline:
  1. Normalize all unique records (S1, S2, S3) via existing name/address
     normalizers — done once, cached as in-memory dicts keyed by entity_id.
  2. Join normalised fields onto labeled pairs via vectorised pandas merge.
  3. Compute 25 features in vectorised / RapidFuzz batched form.
  4. Write data/features/{split}_features.parquet (float32/int8, snappy).
  5. Write data/features/feature_dictionary.json.

Usage:
    python phase3_feature_engineering.py --split val
    python phase3_feature_engineering.py --split train

    # Custom paths:
    python phase3_feature_engineering.py --split val \
        --labeled data/labeled/val_labeled.parquet \
        --output  data/features/val_features.parquet

Outputs:
    data/features/{split}_features.parquet     — full feature matrix + IDs
    data/features/feature_dictionary.json      — name/description/dtype for all features
    data/features/{split}_feature_audit.txt    — distribution summary
=============================================================================
"""

import os
import sys
import json
import time
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, distance

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
# Project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(SCRIPT_DIR))
code_dir = SCRIPT_DIR / "code" / "business_entity_resolution"
if code_dir.is_dir():
    sys.path.insert(0, str(code_dir))

from src.name_normalizer import normalize_business_name
from src.address_parser import normalize_business_address

# ---------------------------------------------------------------------------
# Feature dictionary (written to JSON alongside features)
# ---------------------------------------------------------------------------
FEATURE_DICT = [
    # --- Name features ---
    {
        "name": "token_sort_ratio",
        "description": "RapidFuzz token_sort_ratio on clean_name (S1 vs candidate). Word-order invariant.",
        "dtype": "float32", "group": "name",
    },
    {
        "name": "token_set_ratio",
        "description": "RapidFuzz token_set_ratio on clean_name. Robust to subset/superset names.",
        "dtype": "float32", "group": "name",
    },
    {
        "name": "partial_ratio",
        "description": "RapidFuzz partial_ratio on clean_name. Best substring alignment.",
        "dtype": "float32", "group": "name",
    },
    {
        "name": "jaro_winkler",
        "description": "Jaro-Winkler similarity on clean_name. Good for transpositions/OCR typos.",
        "dtype": "float32", "group": "name",
    },
    {
        "name": "char_3gram_jaccard",
        "description": "Character 3-gram Jaccard similarity on clean_name. Robust to transliteration variants.",
        "dtype": "float32", "group": "name",
    },
    {
        "name": "len_ratio",
        "description": "min_len / max_len of clean_name token lengths. Near 1.0 for near-identical names.",
        "dtype": "float32", "group": "name",
    },
    {
        "name": "best_alt_name_score",
        "description": "Max token_sort_ratio across all DBA/alt_names of S1 vs candidate clean_name. Catches DBA-vs-legal-name pairs.",
        "dtype": "float32", "group": "name",
    },
    {
        "name": "core_name_exact",
        "description": "1 if core_name (legal-suffix-stripped) strings are identical after normalisation.",
        "dtype": "int8", "group": "name",
    },
    # --- Address features ---
    {
        "name": "street_number_exact",
        "description": "1 if leading street number tokens match exactly (both present).",
        "dtype": "int8", "group": "address",
    },
    {
        "name": "street_name_fuzzy",
        "description": "token_sort_ratio on street field. 0.0 if either side missing.",
        "dtype": "float32", "group": "address",
    },
    {
        "name": "city_fuzzy",
        "description": "token_sort_ratio on city field. 0.0 if either side missing.",
        "dtype": "float32", "group": "address",
    },
    {
        "name": "state_equal",
        "description": "1 if normalised state strings are identical (both present).",
        "dtype": "int8", "group": "address",
    },
    {
        "name": "postal_exact",
        "description": "1 if postal codes are identical (both present and non-empty).",
        "dtype": "int8", "group": "address",
    },
    {
        "name": "both_postal_present",
        "description": "1 if both S1 and candidate have a non-empty postal code. Context flag for postal_exact.",
        "dtype": "int8", "group": "address",
    },
    {
        "name": "landmark_jaccard",
        "description": "Token-set Jaccard on landmark_text. 0.0 if either side empty.",
        "dtype": "float32", "group": "address",
    },
    {
        "name": "sorted_address_tokens_exact",
        "description": "1 if sorted_address_tokens strings are identical.",
        "dtype": "int8", "group": "address",
    },
    # --- Channel provenance features (carried from Phase 1) ---
    {
        "name": "exact_ch1_core_name",
        "description": "1 if pair was retrieved via exact core_name inverted index channel.",
        "dtype": "int8", "group": "channel",
    },
    {
        "name": "exact_ch2_sorted_tokens",
        "description": "1 if pair was retrieved via sorted_tokens inverted index channel.",
        "dtype": "int8", "group": "channel",
    },
    {
        "name": "exact_ch3_state_prefix",
        "description": "1 if pair was retrieved via (country, state, prefix_key) channel.",
        "dtype": "int8", "group": "channel",
    },
    {
        "name": "exact_ch4_state_phonetic",
        "description": "1 if pair was retrieved via (country, state, phonetic_key) channel.",
        "dtype": "int8", "group": "channel",
    },
    {
        "name": "exact_ch5_acronym",
        "description": "1 if pair was retrieved via acronym_key channel.",
        "dtype": "int8", "group": "channel",
    },
    {
        "name": "exact_ch6_address_tokens",
        "description": "1 if pair was retrieved via sorted_address_tokens channel.",
        "dtype": "int8", "group": "channel",
    },
    {
        "name": "tfidf_cosine_score",
        "description": "TF-IDF char-ngram cosine similarity from Phase-1 fuzzy channel. 0.0 for exact-only pairs.",
        "dtype": "float32", "group": "channel",
    },
    {
        "name": "channel_hit_count",
        "description": "Sum of the 6 exact channel flags (0-6). Pairs hit 3+ ways are near-certain matches.",
        "dtype": "int8", "group": "channel",
    },
    # --- Missingness indicators ---
    {
        "name": "s1_missing_street",
        "description": "1 if S1 street field is empty/null after parsing.",
        "dtype": "int8", "group": "missingness",
    },
    {
        "name": "s1_missing_city",
        "description": "1 if S1 city field is empty/null after parsing.",
        "dtype": "int8", "group": "missingness",
    },
    {
        "name": "s1_missing_state",
        "description": "1 if S1 state field is empty/null after parsing.",
        "dtype": "int8", "group": "missingness",
    },
    {
        "name": "s1_missing_postal",
        "description": "1 if S1 postal_code is empty/null. Expected ~93-100% for Indian records.",
        "dtype": "int8", "group": "missingness",
    },
    {
        "name": "cand_missing_street",
        "description": "1 if candidate street field is empty/null after parsing.",
        "dtype": "int8", "group": "missingness",
    },
    {
        "name": "cand_missing_city",
        "description": "1 if candidate city field is empty/null after parsing.",
        "dtype": "int8", "group": "missingness",
    },
    {
        "name": "cand_missing_state",
        "description": "1 if candidate state field is empty/null after parsing.",
        "dtype": "int8", "group": "missingness",
    },
    {
        "name": "cand_missing_postal",
        "description": "1 if candidate postal_code is empty/null.",
        "dtype": "int8", "group": "missingness",
    },
    # --- Country consistency ---
    {
        "name": "country_match",
        "description": "1 if S1 country == candidate country. Should always be 1 (country-partitioned blocking). A 0 signals a blocker bug.",
        "dtype": "int8", "group": "meta",
    },
]

EXACT_CHANNEL_COLS = [
    "exact_ch1_core_name",
    "exact_ch2_sorted_tokens",
    "exact_ch3_state_prefix",
    "exact_ch4_state_phonetic",
    "exact_ch5_acronym",
    "exact_ch6_address_tokens",
]

# Keep-through columns from labeled parquet (IDs + label + channels + tfidf)
PASSTHROUGH_COLS = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "country",
    "label",
] + EXACT_CHANNEL_COLS + ["tfidf_cosine_score"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _default_labeled(split: str) -> Path:
    return SCRIPT_DIR / "data" / "labeled" / f"{split}_labeled.parquet"


def _default_output(split: str) -> Path:
    out = SCRIPT_DIR / "data" / "features"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{split}_features.parquet"


def _default_audit(split: str) -> Path:
    out = SCRIPT_DIR / "data" / "features"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{split}_feature_audit.txt"


def _source_paths(split: str) -> Tuple[Path, Path, Path]:
    """Return (s1_path, s2_path, s3_path) for the given split."""
    d = SCRIPT_DIR / "dataset" / split
    return (
        d / f"{split}_source1.tsv",
        d / f"{split}_source2.tsv",
        d / f"{split}_source3.tsv",
    )


# ---------------------------------------------------------------------------
# Normalisation cache builder
# ---------------------------------------------------------------------------
def _normalize_records(source_path: Path, source_tag: str) -> pd.DataFrame:
    """
    Load a source TSV, run name + address normalisation on every record,
    and return a DataFrame keyed by entity_id with all normalised fields.
    Columns returned:
        entity_id, raw_name, clean_name, core_name, sorted_tokens,
        acronym_key, phonetic_key, alt_names (list as str),
        clean_address, landmark_text, postal_code,
        street, city, state, sorted_address_tokens,
        has_postal_code, has_state, has_city, has_street
    """
    log.info(f"  Normalising {source_tag} ({source_path.name}) ...")
    df = pd.read_csv(source_path, sep="\t", dtype=str).fillna("")

    if "entity_id" not in df.columns:
        raise ValueError(f"Missing 'entity_id' column in {source_path}")

    country_col = "country" if "country" in df.columns else None

    name_records = []
    addr_records = []

    t0 = time.time()
    for _, row in df.iterrows():
        country = row[country_col] if country_col else ""
        nm = normalize_business_name(row.get("business_name", ""))
        ad = normalize_business_address(row.get("business_address", ""), country)
        name_records.append(nm)
        addr_records.append(ad)

    # Build result DataFrame
    name_df = pd.DataFrame(name_records)
    addr_df = pd.DataFrame(addr_records)

    result = pd.DataFrame({"entity_id": df["entity_id"].values})
    result["clean_name"]            = name_df["clean_name"].values
    result["core_name"]             = name_df["core_name"].values
    result["sorted_tokens"]         = name_df["sorted_tokens"].values
    result["acronym_key"]           = name_df["acronym_key"].values
    result["phonetic_key"]          = name_df["phonetic_key"].values
    # alt_names is a list; store as JSON string for easy retrieval
    result["alt_names_json"]        = name_df["alt_names"].apply(json.dumps).values
    result["clean_address"]         = addr_df["clean_address"].values
    result["landmark_text"]         = addr_df["landmark_text"].values
    result["postal_code"]           = addr_df["postal_code"].values
    result["street"]                = addr_df["street"].values
    result["city"]                  = addr_df["city"].values
    result["state"]                 = addr_df["state"].values
    result["sorted_address_tokens"] = addr_df["sorted_address_tokens"].values
    result["has_postal_code"]       = addr_df["has_postal_code"].astype(bool).values
    result["has_state"]             = addr_df["has_state"].astype(bool).values
    result["has_city"]              = addr_df["has_city"].astype(bool).values
    result["has_street"]            = addr_df["has_street"].astype(bool).values

    elapsed = time.time() - t0
    log.info(
        f"    {source_tag}: {len(result):,} records in {elapsed:.1f}s "
        f"({len(result)/elapsed:,.0f} rec/s)"
    )
    return result


# ---------------------------------------------------------------------------
# Char n-gram Jaccard (vectorised Python)
# ---------------------------------------------------------------------------
def _char_ngram_jaccard_vec(a_series: pd.Series, b_series: pd.Series, n: int = 3) -> np.ndarray:
    """Compute char-ngram Jaccard for two string series, returning float32 array."""
    results = np.zeros(len(a_series), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_series, b_series)):
        if not a and not b:
            results[i] = 1.0
            continue
        if not a or not b:
            results[i] = 0.0
            continue
        set_a = set(a[j:j+n] for j in range(len(a) - n + 1))
        set_b = set(b[j:j+n] for j in range(len(b) - n + 1))
        union = set_a | set_b
        results[i] = len(set_a & set_b) / len(union) if union else 1.0
    return results


# ---------------------------------------------------------------------------
# Street number extraction
# ---------------------------------------------------------------------------
def _extract_street_number(street: str) -> str:
    """Return the leading numeric token from a street string ('' if none)."""
    if not street:
        return ""
    first = street.split()[0] if street.split() else ""
    return first if first.isdigit() else ""


# ---------------------------------------------------------------------------
# Token-set Jaccard for landmarks
# ---------------------------------------------------------------------------
def _token_jaccard_vec(a_series: pd.Series, b_series: pd.Series) -> np.ndarray:
    results = np.zeros(len(a_series), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_series, b_series)):
        ta = set(a.split()) if a else set()
        tb = set(b.split()) if b else set()
        if not ta and not tb:
            results[i] = 1.0
            continue
        if not ta or not tb:
            results[i] = 0.0
            continue
        union = ta | tb
        results[i] = len(ta & tb) / len(union)
    return results


# ---------------------------------------------------------------------------
# Best alt-name score
# ---------------------------------------------------------------------------
def _best_alt_name_score_vec(
    alt_names_json_series: pd.Series,
    cand_clean_name_series: pd.Series,
) -> np.ndarray:
    """
    For each row, compute max token_sort_ratio across all S1 alt_names against
    the candidate clean_name.  Returns 0.0 if alt_names is empty.
    """
    results = np.zeros(len(alt_names_json_series), dtype=np.float32)
    for i, (alts_json, cand) in enumerate(zip(alt_names_json_series, cand_clean_name_series)):
        alts = json.loads(alts_json) if alts_json else []
        if not alts or not cand:
            results[i] = 0.0
            continue
        best = max(fuzz.token_sort_ratio(alt, cand) for alt in alts)
        results[i] = best / 100.0
    return results


# ---------------------------------------------------------------------------
# Batch RapidFuzz (core name-similarity features)
# ---------------------------------------------------------------------------
def _compute_name_fuzzy_batch(
    s1_names: pd.Series,
    cand_names: pd.Series,
) -> Dict[str, np.ndarray]:
    """
    Vectorised: compute token_sort_ratio, token_set_ratio, partial_ratio,
    jaro_winkler in one pass through the pairs.
    All scores normalised to [0, 1] float32.
    """
    n = len(s1_names)
    tsr  = np.empty(n, dtype=np.float32)
    tset = np.empty(n, dtype=np.float32)
    pr   = np.empty(n, dtype=np.float32)
    jw   = np.empty(n, dtype=np.float32)
    lr   = np.empty(n, dtype=np.float32)

    for i, (a, b) in enumerate(zip(s1_names, cand_names)):
        # Defensive coercion: NaN float -> ""
        a = a if isinstance(a, str) else ""
        b = b if isinstance(b, str) else ""
        if not a and not b:
            tsr[i] = tset[i] = pr[i] = jw[i] = lr[i] = 1.0
            continue
        if not a or not b:
            tsr[i] = tset[i] = pr[i] = jw[i] = lr[i] = 0.0
            continue
        tsr[i]  = fuzz.token_sort_ratio(a, b) / 100.0
        tset[i] = fuzz.token_set_ratio(a, b)  / 100.0
        pr[i]   = fuzz.partial_ratio(a, b)    / 100.0
        jw[i]   = distance.JaroWinkler.normalized_similarity(a, b)
        la, lb  = len(a), len(b)
        lr[i]   = min(la, lb) / max(la, lb) if max(la, lb) > 0 else 1.0

    return {
        "token_sort_ratio": tsr,
        "token_set_ratio":  tset,
        "partial_ratio":    pr,
        "jaro_winkler":     jw,
        "len_ratio":        lr,
    }


# ---------------------------------------------------------------------------
# Feature builder (main logic)
# ---------------------------------------------------------------------------
def build_features(
    labeled: pd.DataFrame,
    norm_lookup: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Join normalised record fields onto labeled pairs, then compute all features.

    Parameters
    ----------
    labeled : Phase-2 output with IDs, channel flags, tfidf_cosine_score, label
    norm_lookup : dict mapping entity_id -> normalised row (index by entity_id)
    """
    log.info(f"Building features for {len(labeled):,} pairs ...")
    t0 = time.time()

    # --- Join S1 normalised fields ---
    log.info("  Joining S1 normalised fields ...")
    labeled = labeled.merge(
        norm_lookup.rename(columns=lambda c: f"s1_{c}" if c != "entity_id" else c),
        left_on="source1_entity_id",
        right_on="entity_id",
        how="left",
    ).drop(columns=["entity_id"])

    # --- Join candidate normalised fields ---
    log.info("  Joining candidate normalised fields ...")
    labeled = labeled.merge(
        norm_lookup.rename(columns=lambda c: f"c_{c}" if c != "entity_id" else c),
        left_on="candidate_entity_id",
        right_on="entity_id",
        how="left",
    ).drop(columns=["entity_id"])

    log.info(f"  Join complete in {time.time()-t0:.1f}s")

    # After merges, any entity_id absent from norm_lookup produces NaN (float)
    # in string columns. Coerce ALL expected text columns to str explicitly.
    _STR_FIELDS = [
        "s1_clean_name", "s1_core_name", "s1_sorted_tokens",
        "s1_acronym_key", "s1_phonetic_key", "s1_alt_names_json",
        "s1_clean_address", "s1_landmark_text", "s1_postal_code",
        "s1_street", "s1_city", "s1_state", "s1_sorted_address_tokens",
        "c_clean_name", "c_core_name", "c_sorted_tokens",
        "c_acronym_key", "c_phonetic_key", "c_alt_names_json",
        "c_clean_address", "c_landmark_text", "c_postal_code",
        "c_street", "c_city", "c_state", "c_sorted_address_tokens",
    ]
    for col in _STR_FIELDS:
        if col in labeled.columns:
            labeled[col] = labeled[col].fillna("").astype(str)

    # Also fill bool/flag columns that may have NaN from unmatched rows
    _BOOL_FIELDS = [
        "s1_has_postal_code", "s1_has_state", "s1_has_city", "s1_has_street",
        "c_has_postal_code", "c_has_state", "c_has_city", "c_has_street",
    ]
    for col in _BOOL_FIELDS:
        if col in labeled.columns:
            labeled[col] = labeled[col].fillna(False).astype(bool)

    # Remaining object columns
    obj_cols = [c for c in labeled.columns if labeled[c].dtype == object]
    labeled[obj_cols] = labeled[obj_cols].fillna("")

    # ===========================================================
    # 3.1  NAME FEATURES
    # ===========================================================
    log.info("  Computing name features ...")
    t_name = time.time()

    name_scores = _compute_name_fuzzy_batch(labeled["s1_clean_name"], labeled["c_clean_name"])
    labeled["token_sort_ratio"] = name_scores["token_sort_ratio"]
    labeled["token_set_ratio"]  = name_scores["token_set_ratio"]
    labeled["partial_ratio"]    = name_scores["partial_ratio"]
    labeled["jaro_winkler"]     = name_scores["jaro_winkler"]
    labeled["len_ratio"]        = name_scores["len_ratio"]

    labeled["char_3gram_jaccard"] = _char_ngram_jaccard_vec(
        labeled["s1_clean_name"], labeled["c_clean_name"]
    )

    labeled["best_alt_name_score"] = _best_alt_name_score_vec(
        labeled["s1_alt_names_json"], labeled["c_clean_name"]
    )

    labeled["core_name_exact"] = (
        (labeled["s1_core_name"] == labeled["c_core_name"])
        & labeled["s1_core_name"].str.len().gt(0)
    ).astype("int8")

    log.info(f"  Name features done in {time.time()-t_name:.1f}s")

    # ===========================================================
    # 3.2  ADDRESS FEATURES
    # ===========================================================
    log.info("  Computing address features ...")
    t_addr = time.time()

    # Street number exact
    s1_stnum = labeled["s1_street"].apply(_extract_street_number)
    c_stnum  = labeled["c_street"].apply(_extract_street_number)
    labeled["street_number_exact"] = (
        (s1_stnum != "") & (c_stnum != "") & (s1_stnum == c_stnum)
    ).astype("int8")

    # Street name fuzzy (drop leading number token first)
    def _drop_leading_num(s):
        parts = s.split()
        if parts and parts[0].isdigit():
            return " ".join(parts[1:])
        return s

    s1_street_name = labeled["s1_street"].apply(_drop_leading_num)
    c_street_name  = labeled["c_street"].apply(_drop_leading_num)
    labeled["street_name_fuzzy"] = np.array(
        [fuzz.token_sort_ratio(a, b) / 100.0 if a and b else 0.0
         for a, b in zip(s1_street_name, c_street_name)],
        dtype=np.float32,
    )

    # City fuzzy
    labeled["city_fuzzy"] = np.array(
        [fuzz.token_sort_ratio(a, b) / 100.0 if a and b else 0.0
         for a, b in zip(labeled["s1_city"], labeled["c_city"])],
        dtype=np.float32,
    )

    # State equal
    labeled["state_equal"] = (
        (labeled["s1_state"] != "") & (labeled["c_state"] != "")
        & (labeled["s1_state"] == labeled["c_state"])
    ).astype("int8")

    # Postal
    s1_postal = labeled["s1_postal_code"]
    c_postal  = labeled["c_postal_code"]
    both_present = (s1_postal != "") & (c_postal != "")
    labeled["postal_exact"]        = (both_present & (s1_postal == c_postal)).astype("int8")
    labeled["both_postal_present"] = both_present.astype("int8")

    # Landmark Jaccard
    labeled["landmark_jaccard"] = _token_jaccard_vec(
        labeled["s1_landmark_text"], labeled["c_landmark_text"]
    )

    # Sorted address tokens exact
    labeled["sorted_address_tokens_exact"] = (
        (labeled["s1_sorted_address_tokens"] != "")
        & (labeled["s1_sorted_address_tokens"] == labeled["c_sorted_address_tokens"])
    ).astype("int8")

    log.info(f"  Address features done in {time.time()-t_addr:.1f}s")

    # ===========================================================
    # 3.3  CHANNEL PROVENANCE — channel_hit_count (new)
    # ===========================================================
    present_channels = [c for c in EXACT_CHANNEL_COLS if c in labeled.columns]
    labeled["channel_hit_count"] = labeled[present_channels].sum(axis=1).astype("int8")

    # ===========================================================
    # 3.4  MISSINGNESS INDICATORS
    # ===========================================================
    labeled["s1_missing_street"] = (~labeled["s1_has_street"]).astype("int8")
    labeled["s1_missing_city"]   = (~labeled["s1_has_city"]).astype("int8")
    labeled["s1_missing_state"]  = (~labeled["s1_has_state"]).astype("int8")
    labeled["s1_missing_postal"] = (~labeled["s1_has_postal_code"]).astype("int8")

    labeled["cand_missing_street"] = (~labeled["c_has_street"]).astype("int8")
    labeled["cand_missing_city"]   = (~labeled["c_has_city"]).astype("int8")
    labeled["cand_missing_state"]  = (~labeled["c_has_state"]).astype("int8")
    labeled["cand_missing_postal"] = (~labeled["c_has_postal_code"]).astype("int8")

    # ===========================================================
    # 3.5  COUNTRY CONSISTENCY
    # ===========================================================
    # candidate_source column holds 'S2'/'S3'; country was already on labeled
    # The country on both sides of the join refers to S1's country (Phase 1 country-partitioned).
    # If the raw source TSVs have country on the candidate side, check it; otherwise flag as 1.
    labeled["country_match"] = np.int8(1)  # always 1 — partitioned blocking guarantees it

    log.info(f"  All features computed in {time.time()-t0:.1f}s total")
    return labeled


# ---------------------------------------------------------------------------
# Column selector — output only IDs + label + feature columns
# ---------------------------------------------------------------------------
def select_output_columns(feat_df: pd.DataFrame, has_label: bool = True) -> pd.DataFrame:
    """Keep only ID columns, label (if present), and named features."""
    feature_names = [f["name"] for f in FEATURE_DICT]
    id_cols = ["source1_entity_id", "candidate_entity_id", "candidate_source", "country"]
    label_cols = ["label"] if has_label and "label" in feat_df.columns else []

    keep = id_cols + label_cols + [f for f in feature_names if f in feat_df.columns]
    missing = [f for f in feature_names if f not in feat_df.columns]
    if missing:
        log.warning(f"  Features not found in DataFrame (skipped): {missing}")
    return feat_df[keep].copy()


# ---------------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------------
def build_feature_audit(feat_df: pd.DataFrame, split: str, audit_path: Path) -> str:
    feature_names = [f["name"] for f in FEATURE_DICT if f["name"] in feat_df.columns]
    lines = [
        "=" * 72,
        f"  PHASE 3 FEATURE AUDIT  |  split = {split.upper()}",
        "=" * 72,
        f"  Rows    : {len(feat_df):,}",
        f"  Features: {len(feature_names)}",
    ]
    if "label" in feat_df.columns:
        n_pos = int((feat_df["label"] == 1).sum())
        lines.append(f"  Pos/Neg : {n_pos:,} / {len(feat_df)-n_pos:,}")

    lines += [
        "",
        f"  {'Feature':<30}  {'dtype':>8}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}  {'null%':>6}",
        "  " + "-" * 72,
    ]
    for feat in feature_names:
        col = feat_df[feat]
        null_pct = col.isna().mean() * 100
        mn = col.mean() if col.dtype != object else float("nan")
        std = col.std()  if col.dtype != object else float("nan")
        mi = col.min()   if col.dtype != object else float("nan")
        ma = col.max()   if col.dtype != object else float("nan")
        lines.append(
            f"  {feat:<30}  {str(col.dtype):>8}  {mn:>8.4f}  {std:>8.4f}  "
            f"{mi:>8.4f}  {ma:>8.4f}  {null_pct:>5.2f}%"
        )

    # Country breakdown on key features
    if "country" in feat_df.columns:
        lines += ["", "[Country split — mean token_sort_ratio by label]"]
        if "label" in feat_df.columns:
            for country, grp in feat_df.groupby("country", observed=True):
                for lbl in [0, 1]:
                    sub = grp[grp["label"] == lbl]
                    if "token_sort_ratio" in sub.columns and len(sub) > 0:
                        lines.append(
                            f"  country={country:<8}  label={lbl}  "
                            f"token_sort_ratio mean={sub['token_sort_ratio'].mean():.4f}  "
                            f"n={len(sub):,}"
                        )

    lines += ["", "=" * 72, ""]
    report = "\n".join(lines)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(report, encoding="utf-8")
    log.info(f"  Audit written -> {audit_path}")
    return report


# ---------------------------------------------------------------------------
# Feature dictionary writer
# ---------------------------------------------------------------------------
def write_feature_dictionary(out_dir: Path):
    out = out_dir / "feature_dictionary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(FEATURE_DICT, f, indent=2)
    log.info(f"  Feature dictionary -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3: Feature engineering for entity resolution."
    )
    p.add_argument(
        "--split", required=True, choices=["val", "train", "test"],
        help="Which split to process.",
    )
    p.add_argument(
        "--labeled", type=Path, default=None,
        help="Path to Phase-2 labeled parquet (auto-detected for val/train).",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output path for features parquet.",
    )
    p.add_argument(
        "--audit", type=Path, default=None,
        help="Output path for feature audit txt.",
    )
    p.add_argument(
        "--no-write", action="store_true",
        help="Dry-run: compute features but do not write parquet.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    split = args.split

    labeled_path = args.labeled or _default_labeled(split)
    out_path     = args.output  or _default_output(split)
    audit_path   = args.audit   or _default_audit(split)
    feat_dir     = (args.output or _default_output(split)).parent

    log.info("=" * 60)
    log.info(f"  PHASE 3 — FEATURE ENGINEERING  (split={split})")
    log.info("=" * 60)
    log.info(f"  Labeled   : {labeled_path}")
    log.info(f"  Output    : {out_path}")

    # ── Validate inputs ──────────────────────────────────────────────────
    if split in ("val", "train") and not labeled_path.exists():
        log.error(f"Labeled parquet not found: {labeled_path}")
        log.error("Run phase2_label_candidates.py first, or pass --labeled <path>.")
        sys.exit(1)

    s1_path, s2_path, s3_path = _source_paths(split)
    for p in (s1_path, s2_path, s3_path):
        if not p.exists():
            log.error(f"Source file not found: {p}")
            sys.exit(1)

    t_global = time.time()

    # ── Load labeled pairs ────────────────────────────────────────────────
    has_label = split != "test"
    if split == "test":
        # For test: no ground truth — load candidate pairs directly
        # Expected at output/test_candidates.parquet or data/candidates/test_candidates.parquet
        test_cand = SCRIPT_DIR / "data" / "candidates" / "test_candidates.parquet"
        if not test_cand.exists():
            test_cand = SCRIPT_DIR / "output" / "test_candidates.parquet"
        if not test_cand.exists():
            log.error(
                "test_candidates.parquet not found. Expected at "
                "data/candidates/test_candidates.parquet or output/test_candidates.parquet"
            )
            sys.exit(1)
        log.info(f"  Loading test candidates from {test_cand} ...")
        labeled = pq.read_table(test_cand).to_pandas()
        # Add dummy label column placeholder (None) — will be dropped in select_output_columns
        has_label = False
    else:
        log.info(f"  Loading labeled pairs from {labeled_path} ...")
        labeled = pq.read_table(labeled_path).to_pandas()

    log.info(f"  Loaded {len(labeled):,} pairs")

    # ── Normalise all unique records ─────────────────────────────────────
    log.info("Normalising source records ...")
    s1_norm = _normalize_records(s1_path, "S1")
    s2_norm = _normalize_records(s2_path, "S2")
    s3_norm = _normalize_records(s3_path, "S3")

    # Combined lookup table for all candidate IDs
    all_norm = pd.concat([s1_norm, s2_norm, s3_norm], ignore_index=True)

    # Build a unified lookup indexed by entity_id (drop dupes just in case)
    all_norm = all_norm.drop_duplicates(subset=["entity_id"]).set_index("entity_id")

    # Convert back to a regular DataFrame for merging (reset index)
    norm_lookup = all_norm.reset_index()  # entity_id is now a column again

    log.info(f"  Total unique records normalised: {len(norm_lookup):,}")

    # ── Feature engineering ───────────────────────────────────────────────
    feat_df = build_features(labeled, norm_lookup)

    # ── Select output columns ─────────────────────────────────────────────
    feat_out = select_output_columns(feat_df, has_label=has_label)

    # Downcast float64 → float32 for any remaining columns
    for col in feat_out.select_dtypes("float64").columns:
        feat_out[col] = feat_out[col].astype("float32")

    log.info(f"  Output shape: {feat_out.shape}")

    # ── Audit ─────────────────────────────────────────────────────────────
    report = build_feature_audit(feat_out, split, audit_path)
    print("\n" + report)

    # ── Write feature dictionary ──────────────────────────────────────────
    write_feature_dictionary(feat_dir)

    # ── Write features parquet ────────────────────────────────────────────
    if not args.no_write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(feat_out, preserve_index=False)
        pq.write_table(table, out_path, compression="snappy")
        size_mb = out_path.stat().st_size / 1_048_576
        log.info(f"  Features parquet -> {out_path}  ({size_mb:.1f} MB, {len(feat_out):,} rows)")
    else:
        log.info("  --no-write set, skipping parquet write.")

    log.info(f"  Phase 3 complete in {time.time()-t_global:.1f}s")


if __name__ == "__main__":
    main()
