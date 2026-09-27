"""
Multi-Channel Candidate Blocker with Exact Keys and Fuzzy TF-IDF Char-Ngram Channel.
Generates candidate pairs across Source 1 and Source 2/3 records.

Captures all 6 exact channel flags and TF-IDF cosine score during blocking:
- source1_entity_id (str)
- candidate_entity_id (str, None if 0 candidates)
- candidate_source (str: 'S2' or 'S3')
- country (str)
- exact_ch1_core_name (int8)
- exact_ch2_sorted_tokens (int8)
- exact_ch3_state_prefix (int8)
- exact_ch4_state_phonetic (int8)
- exact_ch5_acronym (int8)
- exact_ch6_address_tokens (int8)
- tfidf_cosine_score (float32)
"""
import os
import sys
import gc
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.name_normalizer import normalize_business_name
from src.address_parser import normalize_business_address
from src.country_normalizer import normalize_country


CANDIDATE_PA_SCHEMA = pa.schema([
    ("source1_entity_id", pa.string()),
    ("candidate_entity_id", pa.string()),
    ("candidate_source", pa.string()),
    ("country", pa.string()),
    ("exact_ch1_core_name", pa.int8()),
    ("exact_ch2_sorted_tokens", pa.int8()),
    ("exact_ch3_state_prefix", pa.int8()),
    ("exact_ch4_state_phonetic", pa.int8()),
    ("exact_ch5_acronym", pa.int8()),
    ("exact_ch6_address_tokens", pa.int8()),
    ("tfidf_cosine_score", pa.float32()),
])


class CandidateBlocker:
    """
    Multi-channel candidate blocker combining:
    1. Exact Name Channels (core_name, alt_names, sorted_tokens, acronym)
    2. Exact Geolocation + Name Channels (state + prefix_key, state + phonetic)
    3. Exact Address Channel (sorted_address_tokens)
    4. Fuzzy Channel: TF-IDF char-ngram cosine similarity on clean_name + clean_address
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.24,
        max_fuzzy_candidates_per_s1: int = 50,
        max_total_candidates_per_s1: int = 80,
        ngram_range: Tuple[int, int] = (3, 3),
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.max_fuzzy_candidates_per_s1 = max_fuzzy_candidates_per_s1
        self.max_total_candidates_per_s1 = max_total_candidates_per_s1
        self.ngram_range = ngram_range

    def preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize name, address, and country fields for all records in df.
        Uses itertuples for high-speed namedtuple streaming without dictionary allocation.
        """
        eids = []
        norm_names = []
        core_names = []
        sorted_name_tokens = []
        prefix_keys = []
        phonetic_keys = []
        acronym_keys = []
        alt_names_list = []

        norm_addrs = []
        states = []
        cities = []
        sorted_addr_tokens = []
        countries = []
        search_texts = []

        for r in df.itertuples(index=False):
            eid = getattr(r, "entity_id", "")
            eids.append(eid)

            raw_c = getattr(r, "country", None)
            c = normalize_country(str(raw_c)) if pd.notna(raw_c) and raw_c != "" else ""
            countries.append(c)

            raw_name = getattr(r, "business_name", None)
            raw_name_str = "" if pd.isna(raw_name) else str(raw_name)
            pn = normalize_business_name(raw_name_str)
            clean_name = pn["clean_name"]
            norm_names.append(clean_name)
            core_names.append(pn["core_name"])
            sorted_name_tokens.append(pn["sorted_tokens"])
            prefix_keys.append(pn["prefix_key"])
            phonetic_keys.append(pn["phonetic_key"])
            acronym_keys.append(pn["acronym_key"])
            alt_names_list.append(pn["alt_names"])

            raw_addr = getattr(r, "business_address", None)
            raw_addr_str = "" if pd.isna(raw_addr) else str(raw_addr)
            pa_dict = normalize_business_address(raw_addr_str, country=c)
            clean_addr = pa_dict["clean_address"]
            norm_addrs.append(clean_addr)
            states.append(pa_dict["state"])
            cities.append(pa_dict["city"])
            sorted_addr_tokens.append(pa_dict["sorted_address_tokens"])

            combined = f"{clean_name} {clean_addr}".strip()
            search_texts.append(combined)

        res = pd.DataFrame({
            "entity_id": eids,
            "country": countries,
            "clean_name": norm_names,
            "core_name": core_names,
            "sorted_tokens": sorted_name_tokens,
            "prefix_key": prefix_keys,
            "phonetic_key": phonetic_keys,
            "acronym_key": acronym_keys,
            "alt_names": alt_names_list,
            "clean_address": norm_addrs,
            "state": states,
            "city": cities,
            "sorted_address_tokens": sorted_addr_tokens,
            "search_text": search_texts,
        })
        return res

    def index_candidate_chunk(self, other_prep: pd.DataFrame, indexes: Dict[str, dict]):
        """
        Incrementally index candidate pool records into the 6 inverted index maps.
        """
        idx_core_name = indexes["core_name"]
        idx_sorted_tokens = indexes["sorted_tokens"]
        idx_state_pfx = indexes["state_pfx"]
        idx_state_phon = indexes["state_phon"]
        idx_acronym = indexes["acronym"]
        idx_sorted_addr = indexes["sorted_addr"]

        for r in other_prep.itertuples(index=False):
            eid = r.entity_id
            ctry = r.country

            # 1. Core name & alt names
            if r.core_name and len(r.core_name) >= 3:
                idx_core_name[(ctry, r.core_name)].append(eid)
            for alt in r.alt_names:
                if alt and len(alt) >= 3:
                    idx_core_name[(ctry, alt)].append(eid)

            # 2. Sorted tokens
            if r.sorted_tokens and len(r.sorted_tokens) >= 3:
                idx_sorted_tokens[(ctry, r.sorted_tokens)].append(eid)

            # 3. State + Prefix key (4 chars) with missing state fallback
            if r.prefix_key and len(r.prefix_key) >= 4:
                pfx4 = r.prefix_key[:4]
                if r.state:
                    idx_state_pfx[(ctry, r.state, pfx4)].append(eid)
                else:
                    idx_state_pfx[(ctry, "", pfx4)].append(eid)

            # 4. State + Phonetic key with missing state fallback
            if r.phonetic_key:
                if r.state:
                    idx_state_phon[(ctry, r.state, r.phonetic_key)].append(eid)
                else:
                    idx_state_phon[(ctry, "", r.phonetic_key)].append(eid)

            # 5. Acronym key
            if r.acronym_key and len(r.acronym_key) >= 3:
                idx_acronym[(ctry, r.acronym_key)].append(eid)

            # 6. Sorted address tokens
            if r.sorted_address_tokens and len(r.sorted_address_tokens) >= 5:
                idx_sorted_addr[(ctry, r.sorted_address_tokens)].append(eid)

    def retrieve_exact_candidates(
        self, s1_df: pd.DataFrame, indexes: Dict[str, dict]
    ) -> Dict[str, Dict[str, int]]:
        """
        Query inverted indexes for each S1 record.
        Returns dict: s1_id -> {cand_eid: channel_bitmask}
        """
        exact_candidates = defaultdict(lambda: defaultdict(int))
        idx_core_name = indexes["core_name"]
        idx_sorted_tokens = indexes["sorted_tokens"]
        idx_state_pfx = indexes["state_pfx"]
        idx_state_phon = indexes["state_phon"]
        idx_acronym = indexes["acronym"]
        idx_sorted_addr = indexes["sorted_addr"]

        for r in s1_df.itertuples(index=False):
            s1_id = r.entity_id
            ctry = r.country
            s1_map = exact_candidates[s1_id]

            # 1. Core name & alt names
            if r.core_name and len(r.core_name) >= 3:
                matches = idx_core_name.get((ctry, r.core_name))
                if matches:
                    for eid in matches:
                        s1_map[eid] |= 1
            for alt in r.alt_names:
                if alt and len(alt) >= 3:
                    matches = idx_core_name.get((ctry, alt))
                    if matches:
                        for eid in matches:
                            s1_map[eid] |= 1

            # 2. Sorted tokens
            if r.sorted_tokens and len(r.sorted_tokens) >= 3:
                matches = idx_sorted_tokens.get((ctry, r.sorted_tokens))
                if matches:
                    for eid in matches:
                        s1_map[eid] |= 2

            # 3. State + Prefix key
            if r.prefix_key and len(r.prefix_key) >= 4:
                pfx4 = r.prefix_key[:4]
                if r.state:
                    matches = idx_state_pfx.get((ctry, r.state, pfx4))
                    if matches:
                        for eid in matches:
                            s1_map[eid] |= 4
                    missing_pfx = idx_state_pfx.get((ctry, "", pfx4))
                    if missing_pfx and len(missing_pfx) <= 25:
                        for eid in missing_pfx:
                            s1_map[eid] |= 4
                else:
                    matches = idx_state_pfx.get((ctry, "", pfx4))
                    if matches and len(matches) <= 25:
                        for eid in matches:
                            s1_map[eid] |= 4

            # 4. State + Phonetic key
            if r.phonetic_key:
                if r.state:
                    matches = idx_state_phon.get((ctry, r.state, r.phonetic_key))
                    if matches:
                        for eid in matches:
                            s1_map[eid] |= 8
                    missing_phon = idx_state_phon.get((ctry, "", r.phonetic_key))
                    if missing_phon and len(missing_phon) <= 25:
                        for eid in missing_phon:
                            s1_map[eid] |= 8
                else:
                    matches = idx_state_phon.get((ctry, "", r.phonetic_key))
                    if matches and len(matches) <= 25:
                        for eid in matches:
                            s1_map[eid] |= 8

            # 5. Acronym key
            if r.acronym_key and len(r.acronym_key) >= 3:
                matches = idx_acronym.get((ctry, r.acronym_key))
                if matches:
                    for eid in matches:
                        s1_map[eid] |= 16

            # 6. Sorted address tokens
            if r.sorted_address_tokens and len(r.sorted_address_tokens) >= 5:
                matches = idx_sorted_addr.get((ctry, r.sorted_address_tokens))
                if matches:
                    for eid in matches:
                        s1_map[eid] |= 32

        return exact_candidates

    def retrieve_fuzzy_candidates_for_batch(
        self,
        s1_batch_df: pd.DataFrame,
        country_models: Dict[str, Tuple[TfidfVectorizer, sp.csr_matrix, np.ndarray]],
        chunk_size: int = 1000,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Query country-partitioned TF-IDF matrices for a batch of S1 records.
        Returns dict: s1_id -> list of (cand_eid, cosine_score)
        """
        fuzzy_candidates = defaultdict(list)

        for ctry, grp in s1_batch_df.groupby("country"):
            if ctry not in country_models:
                continue
            vectorizer, X_other, other_ids = country_models[ctry]
            s1_ids = grp["entity_id"].tolist()
            s1_texts = grp["search_text"].tolist()

            X_s1 = vectorizer.transform(s1_texts)
            n_s1 = len(s1_ids)

            for start_idx in range(0, n_s1, chunk_size):
                end_idx = min(start_idx + chunk_size, n_s1)
                batch_s1 = X_s1[start_idx:end_idx]
                sim_chunk = batch_s1.dot(X_other.T)

                # Filter below threshold in-place
                sim_chunk.data[sim_chunk.data < self.fuzzy_threshold] = 0
                sim_chunk.eliminate_zeros()

                for local_i in range(end_idx - start_idx):
                    global_i = start_idx + local_i
                    s1_id = s1_ids[global_i]

                    row_start = sim_chunk.indptr[local_i]
                    row_end = sim_chunk.indptr[local_i + 1]
                    if row_start == row_end:
                        continue

                    col_indices = sim_chunk.indices[row_start:row_end]
                    scores = sim_chunk.data[row_start:row_end]

                    if len(col_indices) > self.max_fuzzy_candidates_per_s1:
                        top_k_idx = np.argpartition(scores, -self.max_fuzzy_candidates_per_s1)[-self.max_fuzzy_candidates_per_s1:]
                        sorted_order = np.argsort(-scores[top_k_idx])
                        selected_cols = col_indices[top_k_idx[sorted_order]]
                        selected_scores = scores[top_k_idx[sorted_order]]
                    else:
                        sorted_order = np.argsort(-scores)
                        selected_cols = col_indices[sorted_order]
                        selected_scores = scores[sorted_order]

                    for c_idx, sc in zip(selected_cols, selected_scores):
                        fuzzy_candidates[s1_id].append((other_ids[c_idx], float(sc)))

        return fuzzy_candidates

    def assemble_candidate_rows(
        self,
        s1_prep: pd.DataFrame,
        exact_candidates: Dict[str, Dict[str, int]],
        fuzzy_candidates: Dict[str, List[Tuple[str, float]]],
    ) -> List[dict]:
        """
        Assemble candidate pair rows with all exact channel indicators and cosine score.
        Preserves S1 entities with 0 candidates using candidate_entity_id = None.
        """
        rows = []

        for r in s1_prep.itertuples(index=False):
            s1_id = r.entity_id
            ctry = r.country

            s1_exact = exact_candidates.get(s1_id, {})
            s1_fuzzy = fuzzy_candidates.get(s1_id, [])

            fuzzy_score_map = {cand_id: score for cand_id, score in s1_fuzzy}

            # 1. Exact candidates (always prioritized)
            seen_cands = set()
            selected_cands = []

            for cand_id, mask in s1_exact.items():
                seen_cands.add(cand_id)
                f_score = fuzzy_score_map.get(cand_id, 0.0)
                selected_cands.append((
                    cand_id,
                    mask,
                    f_score
                ))

            # 2. Fill remaining slots with top fuzzy matches
            if s1_fuzzy and len(selected_cands) < self.max_total_candidates_per_s1:
                sorted_fuzzy = sorted(s1_fuzzy, key=lambda x: x[1], reverse=True)
                for cand_id, score in sorted_fuzzy:
                    if cand_id not in seen_cands:
                        seen_cands.add(cand_id)
                        selected_cands.append((
                            cand_id,
                            0,
                            score
                        ))
                        if len(selected_cands) >= self.max_total_candidates_per_s1:
                            break

            # 3. Sentinel row if 0 candidates
            if not selected_cands:
                rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": None,
                    "candidate_source": None,
                    "country": ctry,
                    "exact_ch1_core_name": 0,
                    "exact_ch2_sorted_tokens": 0,
                    "exact_ch3_state_prefix": 0,
                    "exact_ch4_state_phonetic": 0,
                    "exact_ch5_acronym": 0,
                    "exact_ch6_address_tokens": 0,
                    "tfidf_cosine_score": 0.0,
                })
            else:
                for cand_id, mask, score in selected_cands:
                    cand_src = "S2" if cand_id.startswith("S2") else "S3"
                    rows.append({
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": cand_id,
                        "candidate_source": cand_src,
                        "country": ctry,
                        "exact_ch1_core_name": 1 if (mask & 1) else 0,
                        "exact_ch2_sorted_tokens": 1 if (mask & 2) else 0,
                        "exact_ch3_state_prefix": 1 if (mask & 4) else 0,
                        "exact_ch4_state_phonetic": 1 if (mask & 8) else 0,
                        "exact_ch5_acronym": 1 if (mask & 16) else 0,
                        "exact_ch6_address_tokens": 1 if (mask & 32) else 0,
                        "tfidf_cosine_score": float(score),
                    })

        return rows

    def generate_candidate_pairs(
        self, s1_df: pd.DataFrame, other_df: pd.DataFrame, verbose: bool = False
    ) -> Dict[str, List[str]]:
        """
        In-memory candidate generation for testing or small DataFrames.
        Returns dict: s1_id -> list of candidate entity_ids.
        """
        indexes = {
            "core_name": defaultdict(list),
            "sorted_tokens": defaultdict(list),
            "state_pfx": defaultdict(list),
            "state_phon": defaultdict(list),
            "acronym": defaultdict(list),
            "sorted_addr": defaultdict(list),
        }
        country_texts = defaultdict(list)
        country_eids = defaultdict(list)

        other_prep = self.preprocess_df(other_df)
        self.index_candidate_chunk(other_prep, indexes)
        for ctry, grp in other_prep.groupby("country"):
            country_eids[ctry].extend(grp["entity_id"].tolist())
            country_texts[ctry].extend(grp["search_text"].tolist())

        country_models = {}
        for ctry in country_texts:
            texts = country_texts[ctry]
            eids_arr = np.array(country_eids[ctry])
            min_df_val = 1 if len(texts) < 10 else 2
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.ngram_range,
                min_df=min_df_val,
                sublinear_tf=True,
                dtype=np.float32,
            )
            X_other = vectorizer.fit_transform(texts)
            country_models[ctry] = (vectorizer, X_other, eids_arr)

        s1_prep = self.preprocess_df(s1_df)
        exact_cands = self.retrieve_exact_candidates(s1_prep, indexes)
        self.last_exact_candidates = {s1_id: set(cands.keys()) for s1_id, cands in exact_cands.items()}
        fuzzy_cands = self.retrieve_fuzzy_candidates_for_batch(s1_prep, country_models)

        rows = self.assemble_candidate_rows(s1_prep, exact_cands, fuzzy_cands)
        cand_grouped = {r.entity_id: [] for r in s1_prep.itertuples(index=False)}
        for row in rows:
            if row["candidate_entity_id"]:
                cand_grouped[row["source1_entity_id"]].append(row["candidate_entity_id"])
        return cand_grouped

    @staticmethod
    def save_candidate_pairs(candidates: Dict[str, List[str]], output_path: str):
        """Save candidates dictionary to submission-ready TSV."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id, cands in candidates.items():
                f.write(f"{s1_id}\t{','.join(cands)}\n")

    def run_batched_blocking(
        self,
        s1_path: str,
        s2_path: str,
        s3_path: str,
        output_parquet: str,
        output_tsv: Optional[str] = None,
        chunk_size: int = 100000,
        ground_truth_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Run full-scale candidate blocking with streaming ParquetWriter and TSV export.
        Streams S1 in batches to disk, maintaining low memory usage.
        """
        t_start = time.time()
        print("=" * 80)
        print("STARTING FULL-SCALE CANDIDATE BLOCKING (PHASE 1)")
        print(f"  S1: {s1_path}")
        print(f"  S2: {s2_path}")
        print(f"  S3: {s3_path}")
        print(f"  Output Parquet: {output_parquet}")
        if output_tsv:
            print(f"  Output TSV:     {output_tsv}")
        print(f"  Chunk Size:     {chunk_size:,} S1 entities/batch")
        print("=" * 80)

        # 1. Initialize exact inverted indexes
        indexes = {
            "core_name": defaultdict(list),
            "sorted_tokens": defaultdict(list),
            "state_pfx": defaultdict(list),
            "state_phon": defaultdict(list),
            "acronym": defaultdict(list),
            "sorted_addr": defaultdict(list),
        }

        # Structures for fitting TF-IDF per country
        country_texts = defaultdict(list)
        country_eids = defaultdict(list)

        # Ingestion chunk size for candidate pool (500k at a time to keep RAM minimal)
        INGEST_CHUNK = 500000

        print("\n[Step 1/5] Ingesting and indexing Source 2 candidate pool in streaming chunks...")
        t0 = time.time()
        total_s2 = 0
        for s2_chunk in pd.read_csv(s2_path, sep="\t", chunksize=INGEST_CHUNK):
            total_s2 += len(s2_chunk)
            s2_prep = self.preprocess_df(s2_chunk)
            self.index_candidate_chunk(s2_prep, indexes)
            for ctry, grp in s2_prep.groupby("country"):
                country_eids[ctry].extend(grp["entity_id"].tolist())
                country_texts[ctry].extend(grp["search_text"].tolist())
            del s2_chunk, s2_prep
            gc.collect()
        print(f"  Ingested {total_s2:,} S2 records in {time.time() - t0:.2f}s.")

        print("\n[Step 2/5] Ingesting and indexing Source 3 candidate pool in streaming chunks...")
        t0 = time.time()
        total_s3 = 0
        for s3_chunk in pd.read_csv(s3_path, sep="\t", chunksize=INGEST_CHUNK):
            total_s3 += len(s3_chunk)
            s3_prep = self.preprocess_df(s3_chunk)
            self.index_candidate_chunk(s3_prep, indexes)
            for ctry, grp in s3_prep.groupby("country"):
                country_eids[ctry].extend(grp["entity_id"].tolist())
                country_texts[ctry].extend(grp["search_text"].tolist())
            del s3_chunk, s3_prep
            gc.collect()
        print(f"  Ingested {total_s3:,} S3 records in {time.time() - t0:.2f}s.")
        print(f"  Total candidate pool: {total_s2 + total_s3:,} entities indexed.")

        print("\n[Step 3/5] Fitting country-partitioned TF-IDF matrices over candidate pool...")
        t0 = time.time()
        country_models = {}
        for ctry in list(country_texts.keys()):
            texts = country_texts[ctry]
            eids_arr = np.array(country_eids[ctry])
            # Free raw id list
            del country_eids[ctry]

            min_df_val = 1 if len(texts) < 10 else 2
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.ngram_range,
                min_df=min_df_val,
                sublinear_tf=True,
                dtype=np.float32,
            )
            X_other = vectorizer.fit_transform(texts)
            del texts, country_texts[ctry]
            gc.collect()

            country_models[ctry] = (vectorizer, X_other, eids_arr)
            print(f"  Country '{ctry}': {X_other.shape[0]:,} records, {X_other.shape[1]:,} features, {X_other.nnz:,} non-zeros.")

        print(f"  All TF-IDF matrices built in {time.time() - t0:.2f}s.")
        del country_texts, country_eids
        gc.collect()

        # Load ground truth for streaming recall tracking if provided
        gt_map = {}
        total_gt_pairs = 0
        if ground_truth_path and os.path.exists(ground_truth_path):
            print(f"\n[Step 4/5] Loading ground truth pairs for real-time recall ceiling tracking...")
            t0 = time.time()
            gt_df = pd.read_csv(ground_truth_path, sep="\t")
            match_col = "matched_entity_ids" if "matched_entity_ids" in gt_df.columns else "matched_entity_id"
            for _, row in gt_df.iterrows():
                if pd.isna(row[match_col]):
                    continue
                s1_id = row["source1_entity_id"]
                matches = {m.strip() for m in str(row[match_col]).split(",") if m.strip() and m.strip() != "nan"}
                if matches:
                    gt_map[s1_id] = matches
                    total_gt_pairs += len(matches)
            del gt_df
            gc.collect()
            print(f"  Loaded {len(gt_map):,} true S1 matches ({total_gt_pairs:,} total true pairs) in {time.time() - t0:.2f}s.")

        # Prepare streaming output writers
        os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
        parquet_writer = pq.ParquetWriter(output_parquet, schema=CANDIDATE_PA_SCHEMA, compression="snappy")

        tsv_f = None
        if output_tsv:
            os.makedirs(os.path.dirname(os.path.abspath(output_tsv)), exist_ok=True)
            tsv_f = open(output_tsv, "w", encoding="utf-8")
            tsv_f.write("source1_entity_id\tcandidate_entity_ids\n")

        # 2. Stream S1 in batches
        print(f"\n[Step 5/5] Processing Source 1 in chunks of {chunk_size:,}...")
        batch_idx = 0
        total_s1_records = 0
        total_candidate_pairs = 0
        total_zero_candidates = 0
        running_true_retrieved = 0
        cand_counts_all = []

        country_stats = defaultdict(lambda: {"total": 0, "pairs": 0, "zeros": 0})

        for s1_chunk in pd.read_csv(s1_path, sep="\t", chunksize=chunk_size):
            t_batch = time.time()
            batch_idx += 1
            n_batch = len(s1_chunk)
            total_s1_records += n_batch

            s1_prep = self.preprocess_df(s1_chunk)

            # Query exact indexes
            exact_candidates = self.retrieve_exact_candidates(s1_prep, indexes)

            # Query fuzzy matrices
            fuzzy_candidates = self.retrieve_fuzzy_candidates_for_batch(s1_prep, country_models)

            # Assemble candidate rows
            rows = self.assemble_candidate_rows(s1_prep, exact_candidates, fuzzy_candidates)
            df_batch = pd.DataFrame(rows)

            # Type casting
            df_batch["exact_ch1_core_name"] = df_batch["exact_ch1_core_name"].astype("int8")
            df_batch["exact_ch2_sorted_tokens"] = df_batch["exact_ch2_sorted_tokens"].astype("int8")
            df_batch["exact_ch3_state_prefix"] = df_batch["exact_ch3_state_prefix"].astype("int8")
            df_batch["exact_ch4_state_phonetic"] = df_batch["exact_ch4_state_phonetic"].astype("int8")
            df_batch["exact_ch5_acronym"] = df_batch["exact_ch5_acronym"].astype("int8")
            df_batch["exact_ch6_address_tokens"] = df_batch["exact_ch6_address_tokens"].astype("int8")
            df_batch["tfidf_cosine_score"] = df_batch["tfidf_cosine_score"].astype("float32")

            n_pairs = len(df_batch[df_batch["candidate_entity_id"].notna()])
            total_candidate_pairs += n_pairs

            # Stream directly to Parquet file
            table = pa.Table.from_pandas(df_batch, schema=CANDIDATE_PA_SCHEMA, preserve_index=False)
            parquet_writer.write_table(table)

            # Stream TSV if requested
            if tsv_f is not None:
                # Group candidates for this batch
                valid_b = df_batch[df_batch["candidate_entity_id"].notna()]
                cand_grouped = valid_b.groupby("source1_entity_id")["candidate_entity_id"].apply(lambda x: ",".join(x)).to_dict()
                for s1_id in s1_chunk["entity_id"]:
                    cand_str = cand_grouped.get(s1_id, "")
                    tsv_f.write(f"{s1_id}\t{cand_str}\n")

            # Track ground truth recall if available
            if gt_map:
                valid_b = df_batch[df_batch["candidate_entity_id"].notna()]
                b_cands_by_s1 = valid_b.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()
                for s1_id, true_set in gt_map.items():
                    if s1_id in b_cands_by_s1:
                        running_true_retrieved += len(true_set.intersection(b_cands_by_s1[s1_id]))

            # Batch stats
            batch_zeros = len(df_batch[df_batch["candidate_entity_id"].isna()])
            total_zero_candidates += batch_zeros

            # Country stats update
            for ctry, grp in s1_prep.groupby("country"):
                country_stats[ctry]["total"] += len(grp)
                b_valid_ctry = df_batch[(df_batch["country"] == ctry) & (df_batch["candidate_entity_id"].notna())]
                country_stats[ctry]["pairs"] += len(b_valid_ctry)
                b_zero_ctry = df_batch[(df_batch["country"] == ctry) & (df_batch["candidate_entity_id"].isna())]
                country_stats[ctry]["zeros"] += len(b_zero_ctry)

            recall_str = ""
            if total_gt_pairs > 0:
                cur_recall = (running_true_retrieved / total_gt_pairs) * 100
                recall_str = f" | Recall Ceiling: {cur_recall:.3f}%"

            print(f"  Batch {batch_idx:03d}: {n_batch:,} S1 | Pairs: {n_pairs:,} (avg {n_pairs/n_batch:.1f}/S1) | 0-cands: {batch_zeros} | Elapsed: {time.time() - t_batch:.1f}s{recall_str}")

            del s1_prep, exact_candidates, fuzzy_candidates, rows, df_batch, table
            gc.collect()

        # Close streams
        parquet_writer.close()
        if tsv_f is not None:
            tsv_f.close()
            print(f"TSV deliverable written to {output_tsv}.")

        del indexes, country_models
        gc.collect()

        # Final validation report
        print("\n" + "=" * 80)
        print("FULL BLOCKING RUN COMPLETE — SUMMARY AUDIT")
        print("=" * 80)
        print(f"Total S1 entities processed: {total_s1_records:,}")
        print(f"Total valid candidate pairs: {total_candidate_pairs:,}")
        print(f"Average candidates per S1:   {total_candidate_pairs / total_s1_records:.2f}")
        print(f"Entities with 0 candidates:  {total_zero_candidates:,} ({total_zero_candidates / total_s1_records * 100:.4f}%)")

        print("\nPer-Country Distribution:")
        for ctry, stats in country_stats.items():
            tot = stats["total"]
            pairs = stats["pairs"]
            zeros = stats["zeros"]
            avg_p = pairs / tot if tot > 0 else 0
            print(f"  Country '{ctry}': {tot:,} S1 | {pairs:,} pairs (avg {avg_p:.1f}/S1) | {zeros:,} zero-cands ({zeros/tot*100:.3f}%)")

        if total_gt_pairs > 0:
            final_recall = (running_true_retrieved / total_gt_pairs) * 100
            print(f"\nGROUND TRUTH RECALL CEILING:")
            print(f"  Total True Pairs:    {total_gt_pairs:,}")
            print(f"  Retrieved by Blocker:{running_true_retrieved:,}")
            print(f"  Missed by Blocker:   {total_gt_pairs - running_true_retrieved:,}")
            print(f"  RECALL CEILING:      {final_recall:.3f}%")
            if final_recall >= 99.0:
                print(f"  ✅ PASS: Recall ceiling is {final_recall:.2f}% (exceeds 99.0% target).")
            else:
                print(f"  ⚠️ WARNING: Recall ceiling is {final_recall:.2f}%.")

        print(f"\nParquet file saved to: {output_parquet} (size: {os.path.getsize(output_parquet) / (1024*1024):.1f} MB)")
        print(f"TOTAL EXECUTION TIME: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Channel Candidate Blocker with Parquet and TSV output.")
    parser.add_argument("--s1", required=True, help="Path to Source 1 TSV file")
    parser.add_argument("--s2", required=True, help="Path to Source 2 TSV file")
    parser.add_argument("--s3", required=True, help="Path to Source 3 TSV file")
    parser.add_argument("--output-parquet", required=True, help="Path to output candidate_pairs.parquet")
    parser.add_argument("--output-tsv", default=None, help="Path to output candidate_pairs.tsv (deliverable)")
    parser.add_argument("--ground-truth", default=None, help="Path to ground truth TSV for recall ceiling audit")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Batch chunk size for S1 records")
    parser.add_argument("--threshold", type=float, default=0.24, help="Fuzzy TF-IDF cosine threshold")
    parser.add_argument("--max-fuzzy", type=int, default=50, help="Max fuzzy candidates per S1 entity")
    parser.add_argument("--max-total", type=int, default=80, help="Max total candidates per S1 entity")
    args = parser.parse_args()

    blocker = CandidateBlocker(
        fuzzy_threshold=args.threshold,
        max_fuzzy_candidates_per_s1=args.max_fuzzy,
        max_total_candidates_per_s1=args.max_total
    )

    blocker.run_batched_blocking(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        output_parquet=args.output_parquet,
        output_tsv=args.output_tsv,
        chunk_size=args.chunk_size,
        ground_truth_path=args.ground_truth,
        verbose=True
    )
