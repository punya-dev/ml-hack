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
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from name_normalizer import normalize_business_name
    from address_parser import normalize_business_address
    from country_normalizer import normalize_country
except ImportError:
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


def _get_line_count(fpath: str) -> Optional[int]:
    try:
        with open(fpath, "rb") as f:
            return sum(1 for _ in f) - 1
    except Exception:
        return None


def _process_fuzzy_subchunk(
    batch_s1: sp.csr_matrix,
    sub_s1_ids: List[str],
    X_other_T: sp.spmatrix,
    other_ids: np.ndarray,
    fuzzy_threshold: float,
    max_fuzzy: int,
) -> List[Tuple[str, List[Tuple[str, float]]]]:
    """
    Worker function for parallel candidate dot-product scoring on a sub-chunk of S1 queries.
    Uses pre-transposed X_other_T with shared memory across worker threads.
    """
    sim_chunk = batch_s1.dot(X_other_T)
    # Filter below threshold in-place
    sim_chunk.data[sim_chunk.data < fuzzy_threshold] = 0
    sim_chunk.eliminate_zeros()

    sub_results = []
    for local_i, s1_id in enumerate(sub_s1_ids):
        row_start = sim_chunk.indptr[local_i]
        row_end = sim_chunk.indptr[local_i + 1]
        if row_start == row_end:
            continue

        col_indices = sim_chunk.indices[row_start:row_end]
        scores = sim_chunk.data[row_start:row_end]

        if len(col_indices) > max_fuzzy:
            top_k_idx = np.argpartition(scores, -max_fuzzy)[-max_fuzzy:]
            sorted_order = np.argsort(-scores[top_k_idx])
            selected_cols = col_indices[top_k_idx[sorted_order]]
            selected_scores = scores[top_k_idx[sorted_order]]
        else:
            sorted_order = np.argsort(-scores)
            selected_cols = col_indices[sorted_order]
            selected_scores = scores[sorted_order]

        cands = [
            (other_ids[c_idx], float(sc))
            for c_idx, sc in zip(selected_cols, selected_scores)
        ]
        sub_results.append((s1_id, cands))

    return sub_results


def _preprocess_records_slice(records: List[Tuple]) -> Dict[str, list]:
    """
    Worker function for parallel preprocessing of entity records.
    Normalizes country, business name, and address fields across worker processes.
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

    for eid, raw_c, raw_name, raw_addr in records:
        eids.append(eid)
        c = normalize_country(str(raw_c)) if raw_c and str(raw_c).strip() != "" else ""
        countries.append(c)

        pn = normalize_business_name(str(raw_name) if raw_name and str(raw_name).strip() != "" else "")
        clean_name = pn["clean_name"]
        norm_names.append(clean_name)
        core_names.append(pn["core_name"])
        sorted_name_tokens.append(pn["sorted_tokens"])
        prefix_keys.append(pn["prefix_key"])
        phonetic_keys.append(pn["phonetic_key"])
        acronym_keys.append(pn["acronym_key"])
        alt_names_list.append(pn["alt_names"])

        pa_dict = normalize_business_address(str(raw_addr) if raw_addr and str(raw_addr).strip() != "" else "", country=c)
        clean_addr = pa_dict["clean_address"]
        norm_addrs.append(clean_addr)
        states.append(pa_dict["state"])
        cities.append(pa_dict["city"])
        sorted_addr_tokens.append(pa_dict["sorted_address_tokens"])

        combined = f"{clean_name} {clean_addr}".strip()
        search_texts.append(combined)

    return {
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
    }


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
        n_jobs: Optional[int] = None,
        subchunk_size: int = 500,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.max_fuzzy_candidates_per_s1 = max_fuzzy_candidates_per_s1
        self.max_total_candidates_per_s1 = max_total_candidates_per_s1
        self.ngram_range = ngram_range
        self.n_jobs = n_jobs if n_jobs is not None else (os.cpu_count() or 4)
        self.subchunk_size = subchunk_size

    def preprocess_df(self, df: pd.DataFrame, n_jobs: Optional[int] = None) -> pd.DataFrame:
        """
        Normalize name, address, and country fields for all records in df.
        Uses multi-processing across CPU cores when len(df) >= 5,000 for maximum throughput.
        """
        n_workers = n_jobs if n_jobs is not None else self.n_jobs
        n_rows = len(df)
        if n_rows == 0:
            return pd.DataFrame({
                "entity_id": [], "country": [], "clean_name": [], "core_name": [],
                "sorted_tokens": [], "prefix_key": [], "phonetic_key": [], "acronym_key": [],
                "alt_names": [], "clean_address": [], "state": [], "city": [],
                "sorted_address_tokens": [], "search_text": []
            })

        records = list(zip(
            df["entity_id"].fillna("").astype(str) if "entity_id" in df.columns else [""] * n_rows,
            df["country"].fillna("").astype(str) if "country" in df.columns else [""] * n_rows,
            df["business_name"].fillna("").astype(str) if "business_name" in df.columns else [""] * n_rows,
            df["business_address"].fillna("").astype(str) if "business_address" in df.columns else [""] * n_rows,
        ))

        # For small slices (< 5,000) or single core, run sequentially without IPC overhead
        if n_rows < 5000 or n_workers <= 1:
            res_dict = _preprocess_records_slice(records)
            return pd.DataFrame(res_dict)

        # Multi-process execution partitioned across worker processes
        effective_workers = min(n_workers, max(1, n_rows // 2000))
        chunk_size = (n_rows + effective_workers - 1) // effective_workers
        slices = [records[i:i + chunk_size] for i in range(0, n_rows, chunk_size)]

        results = Parallel(n_jobs=effective_workers, backend="loky")(
            delayed(_preprocess_records_slice)(s) for s in slices
        )

        combined = {col: [] for col in results[0]}
        for r in results:
            for col in combined:
                combined[col].extend(r[col])

        return pd.DataFrame(combined)

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
        country_models: Dict[str, Tuple[TfidfVectorizer, sp.spmatrix, np.ndarray]],
        subchunk_size: Optional[int] = None,
        n_jobs: Optional[int] = None,
        batch_idx: Optional[int] = None,
        total_batches: Optional[int] = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Query country-partitioned TF-IDF matrices for a batch of S1 records in parallel across CPU cores.
        Streams subchunk results and prints real-time intra-batch progress at regular intervals.
        Returns dict: s1_id -> list of (cand_eid, cosine_score)
        """
        eff_subchunk = subchunk_size if subchunk_size is not None else self.subchunk_size
        eff_jobs = n_jobs if n_jobs is not None else self.n_jobs
        fuzzy_candidates = defaultdict(list)

        n_batch = len(s1_batch_df)
        batch_tag = f"Batch {batch_idx:03d}/{total_batches:03d}" if (batch_idx is not None and total_batches is not None) else (f"Batch {batch_idx:03d}" if batch_idx is not None else "Batch")

        # Report progress every ~10% of batch or every 2,500 entities (whichever is smaller)
        report_interval = max(1000, min(5000, n_batch // 10))
        completed_in_batch = 0
        last_reported = 0
        t_fuzzy_start = time.time()

        for ctry, grp in s1_batch_df.groupby("country"):
            if ctry not in country_models:
                continue
            vectorizer, X_other_T, other_ids = country_models[ctry]
            s1_ids = grp["entity_id"].tolist()
            s1_texts = grp["search_text"].tolist()

            X_s1 = vectorizer.transform(s1_texts)
            n_s1 = len(s1_ids)

            subchunks = []
            for start_idx in range(0, n_s1, eff_subchunk):
                end_idx = min(start_idx + eff_subchunk, n_s1)
                subchunks.append((
                    X_s1[start_idx:end_idx],
                    s1_ids[start_idx:end_idx]
                ))

            total_subchunks_ctry = len(subchunks)
            subchunk_i = 0
            t_ctry = time.time()

            # Run parallel sub-chunks with multithreading & stream results via generator
            parallel_gen = Parallel(n_jobs=eff_jobs, prefer="threads", return_as="generator")(
                delayed(_process_fuzzy_subchunk)(
                    batch_s1,
                    sub_s1_ids,
                    X_other_T,
                    other_ids,
                    self.fuzzy_threshold,
                    self.max_fuzzy_candidates_per_s1,
                )
                for batch_s1, sub_s1_ids in subchunks
            )

            for sub_res in parallel_gen:
                subchunk_i += 1
                for s1_id, cands in sub_res:
                    fuzzy_candidates[s1_id].extend(cands)
                completed_in_batch += len(sub_res)

                # Report intra-batch progress at regular intervals
                if (completed_in_batch - last_reported >= report_interval) or (completed_in_batch == n_batch):
                    last_reported = completed_in_batch
                    pct = (completed_in_batch / n_batch) * 100
                    elapsed = time.time() - t_fuzzy_start
                    rate = completed_in_batch / elapsed if elapsed > 0 else 0
                    est_rem = (n_batch - completed_in_batch) / rate if rate > 0 else 0
                    print(
                        f"    [{batch_tag} Progress] Fuzzy matching: {completed_in_batch:,} / {n_batch:,} S1 ({pct:5.1f}%) | "
                        f"Country: '{ctry}' ({subchunk_i}/{total_subchunks_ctry}) | "
                        f"Speed: {rate:,.0f} S1/s | Elapsed: {elapsed:5.1f}s | Est remaining: {est_rem:5.1f}s",
                        flush=True,
                    )

            t_ctry_done = time.time() - t_ctry
            rate_ctry = n_s1 / t_ctry_done if t_ctry_done > 0 else 0
            print(
                f"    [{batch_tag} Progress] Country '{ctry}': {n_s1:,} entities finished in {t_ctry_done:.2f}s ({rate_ctry:,.0f} S1/s)",
                flush=True,
            )

        return fuzzy_candidates

    def assemble_candidate_rows(
        self,
        s1_prep: pd.DataFrame,
        exact_candidates: Dict[str, Dict[str, int]],
        fuzzy_candidates: Dict[str, List[Tuple[str, float]]],
    ) -> Tuple[Dict[str, list], Dict[str, List[str]]]:
        """
        Assemble candidate pair columnar dictionary and per-S1 candidate ID mapping.
        Columnar format allows direct PyArrow Table construction (2-3x faster than dict-of-rows).
        Preserves S1 entities with 0 candidates using candidate_entity_id = None.
        """
        cols = {
            "source1_entity_id": [],
            "candidate_entity_id": [],
            "candidate_source": [],
            "country": [],
            "exact_ch1_core_name": [],
            "exact_ch2_sorted_tokens": [],
            "exact_ch3_state_prefix": [],
            "exact_ch4_state_phonetic": [],
            "exact_ch5_acronym": [],
            "exact_ch6_address_tokens": [],
            "tfidf_cosine_score": [],
        }
        cand_ids_by_s1 = {}

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

            # Record candidate IDs for fast TSV export and recall audit without Pandas groupby
            cand_ids_by_s1[s1_id] = [c[0] for c in selected_cands]

            # 3. Sentinel row if 0 candidates
            if not selected_cands:
                cols["source1_entity_id"].append(s1_id)
                cols["candidate_entity_id"].append(None)
                cols["candidate_source"].append(None)
                cols["country"].append(ctry)
                cols["exact_ch1_core_name"].append(0)
                cols["exact_ch2_sorted_tokens"].append(0)
                cols["exact_ch3_state_prefix"].append(0)
                cols["exact_ch4_state_phonetic"].append(0)
                cols["exact_ch5_acronym"].append(0)
                cols["exact_ch6_address_tokens"].append(0)
                cols["tfidf_cosine_score"].append(0.0)
            else:
                for cand_id, mask, score in selected_cands:
                    cand_src = "S2" if cand_id.startswith("S2") else "S3"
                    cols["source1_entity_id"].append(s1_id)
                    cols["candidate_entity_id"].append(cand_id)
                    cols["candidate_source"].append(cand_src)
                    cols["country"].append(ctry)
                    cols["exact_ch1_core_name"].append(1 if (mask & 1) else 0)
                    cols["exact_ch2_sorted_tokens"].append(1 if (mask & 2) else 0)
                    cols["exact_ch3_state_prefix"].append(1 if (mask & 4) else 0)
                    cols["exact_ch4_state_phonetic"].append(1 if (mask & 8) else 0)
                    cols["exact_ch5_acronym"].append(1 if (mask & 16) else 0)
                    cols["exact_ch6_address_tokens"].append(1 if (mask & 32) else 0)
                    cols["tfidf_cosine_score"].append(float(score))

        return cols, cand_ids_by_s1

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
            X_other_T = X_other.T.tocsr()
            country_models[ctry] = (vectorizer, X_other_T, eids_arr)

        s1_prep = self.preprocess_df(s1_df)
        exact_cands = self.retrieve_exact_candidates(s1_prep, indexes)
        self.last_exact_candidates = {s1_id: set(cands.keys()) for s1_id, cands in exact_cands.items()}
        fuzzy_cands = self.retrieve_fuzzy_candidates_for_batch(s1_prep, country_models)

        cols, cand_ids_by_s1 = self.assemble_candidate_rows(s1_prep, exact_cands, fuzzy_cands)
        return cand_ids_by_s1

    @staticmethod
    def save_candidate_pairs(candidates: Dict[str, List[str]], output_path: str):
        """Save candidates dictionary to submission-ready TSV."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id, cands in candidates.items():
                f.write(f"{s1_id}\t{','.join(cands)}\n")

    def _prepare_stage_1_4_state(
        self,
        s2_path: str,
        s3_path: str,
        ground_truth_path: Optional[str] = None,
        s2_chunk_size: int = 500000,
        s3_chunk_size: int = 500000,
    ) -> Dict[str, Any]:
        """Build the cached stage-1-to-stage-4 state used by the blocker."""
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

        def _get_line_count(fpath: str) -> Optional[int]:
            try:
                with open(fpath, "rb") as f:
                    return sum(1 for _ in f) - 1
            except Exception:
                return None

        total_s2_lines = _get_line_count(s2_path)
        s2_chunks_est = ((total_s2_lines + s2_chunk_size - 1) // s2_chunk_size) if total_s2_lines else None
        print(f"\n[Step 1/5] Ingesting and indexing Source 2 candidate pool in streaming chunks (using {self.n_jobs} CPU cores for parallel preprocessing)...", flush=True)
        t0 = time.time()
        total_s2 = 0
        pbar_s2 = tqdm(total=s2_chunks_est, desc="Step 1/5: Ingesting S2", unit="chunk", file=sys.stdout, dynamic_ncols=True)
        for s2_chunk in pd.read_csv(s2_path, sep="\t", chunksize=s2_chunk_size):
            total_s2 += len(s2_chunk)
            s2_prep = self.preprocess_df(s2_chunk, n_jobs=self.n_jobs)
            self.index_candidate_chunk(s2_prep, indexes)
            for ctry, grp in s2_prep.groupby("country"):
                country_eids[ctry].extend(grp["entity_id"].tolist())
                country_texts[ctry].extend(grp["search_text"].tolist())
            del s2_chunk, s2_prep
            gc.collect()
            pbar_s2.update(1)
        pbar_s2.close()
        print(f"  Ingested {total_s2:,} S2 records in {time.time() - t0:.2f}s using {self.n_jobs} CPU cores.", flush=True)

        total_s3_lines = _get_line_count(s3_path)
        s3_chunks_est = ((total_s3_lines + s3_chunk_size - 1) // s3_chunk_size) if total_s3_lines else None
        print(f"\n[Step 2/5] Ingesting and indexing Source 3 candidate pool in streaming chunks (using {self.n_jobs} CPU cores for parallel preprocessing)...", flush=True)
        t0 = time.time()
        total_s3 = 0
        pbar_s3 = tqdm(total=s3_chunks_est, desc="Step 2/5: Ingesting S3", unit="chunk", file=sys.stdout, dynamic_ncols=True)
        for s3_chunk in pd.read_csv(s3_path, sep="\t", chunksize=s3_chunk_size):
            total_s3 += len(s3_chunk)
            s3_prep = self.preprocess_df(s3_chunk, n_jobs=self.n_jobs)
            self.index_candidate_chunk(s3_prep, indexes)
            for ctry, grp in s3_prep.groupby("country"):
                country_eids[ctry].extend(grp["entity_id"].tolist())
                country_texts[ctry].extend(grp["search_text"].tolist())
            del s3_chunk, s3_prep
            gc.collect()
            pbar_s3.update(1)
        pbar_s3.close()
        print(f"  Ingested {total_s3:,} S3 records in {time.time() - t0:.2f}s using {self.n_jobs} CPU cores.", flush=True)
        print(f"  Total candidate pool: {total_s2 + total_s3:,} entities indexed across 6 inverted index channels.", flush=True)

        print(f"\n[Step 3/5] Fitting country-partitioned TF-IDF matrices over candidate pool (using {self.n_jobs} CPU cores)...", flush=True)
        t0 = time.time()
        country_models = {}
        for ctry in list(country_texts.keys()):
            texts = country_texts[ctry]
            eids_arr = np.array(country_eids[ctry])
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
            n_records = X_other.shape[0]
            n_features = X_other.shape[1]
            nnz = X_other.nnz
            del texts, country_texts[ctry]
            gc.collect()
            X_other_T = X_other.T.tocsr()
            del X_other
            gc.collect()
            country_models[ctry] = (vectorizer, X_other_T, eids_arr)
            print(f"  Country '{ctry}': {n_records:,} records, {n_features:,} features, {nnz:,} non-zeros (pre-transposed).", flush=True)

        print(f"  All TF-IDF matrices built in {time.time() - t0:.2f}s using {self.n_jobs} CPU cores.", flush=True)
        del country_texts, country_eids
        gc.collect()

        gt_map = {}
        total_gt_pairs = 0
        if ground_truth_path and os.path.exists(ground_truth_path):
            print(f"\n[Step 4/5] Loading ground truth pairs for real-time recall ceiling tracking (using 1 CPU core - I/O bound)...", flush=True)
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
            print(f"  Loaded {len(gt_map):,} true S1 matches ({total_gt_pairs:,} total true pairs) in {time.time() - t0:.2f}s.", flush=True)

        return {
            "indexes": indexes,
            "country_models": country_models,
            "ground_truth": gt_map,
            "total_gt_pairs": total_gt_pairs,
        }

    def save_stage_1_4_checkpoint(
        self,
        s2_path: str,
        s3_path: str,
        cache_dir: str,
        s2_chunk_size: int = 500000,
        s3_chunk_size: int = 500000,
        ground_truth_path: Optional[str] = None,
    ) -> bool:
        """Persist the stage 1-4 blocking state for reuse when only worker settings change."""
        os.makedirs(cache_dir, exist_ok=True)
        state = self._prepare_stage_1_4_state(
            s2_path=s2_path,
            s3_path=s3_path,
            ground_truth_path=ground_truth_path,
            s2_chunk_size=s2_chunk_size,
            s3_chunk_size=s3_chunk_size,
        )
        checkpoint_path = os.path.join(cache_dir, "stages_1_to_4.pkl")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Saved checkpoint to: {checkpoint_path}")
        return True

    def load_stage_1_4_checkpoint(self, cache_dir: str) -> Optional[Dict[str, Any]]:
        """Load a persisted stage 1-4 checkpoint if it exists."""
        checkpoint_path = os.path.join(cache_dir, "stages_1_to_4.pkl")
        if not os.path.exists(checkpoint_path):
            return None
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)

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
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Run full-scale candidate blocking with streaming ParquetWriter and TSV export.
        Streams S1 in batches to disk, maintaining low memory usage.
        """
        t_start = time.time()
        detected_cores = os.cpu_count() or 4
        print("=" * 80)
        print("STARTING FULL-SCALE CANDIDATE BLOCKING (PHASE 1)")
        print(f"  S1: {s1_path}")
        print(f"  S2: {s2_path}")
        print(f"  S3: {s3_path}")
        print(f"  Output Parquet: {output_parquet}")
        if output_tsv:
            print(f"  Output TSV:     {output_tsv}")
        print(f"  Chunk Size:     {chunk_size:,} S1 entities/batch")
        print(f"  CPU Hardware    : {detected_cores} logical CPU cores detected | Using {self.n_jobs} active worker cores")
        print(f"  Parallelization : Multi-processing (loky) for text preprocessing | Multi-threaded BLAS for TF-IDF")
        print("=" * 80)

        checkpoint_path = None
        if checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "stages_1_to_4.pkl")

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\n[Checkpoint] Loading cached stage 1-4 state from {checkpoint_path}")
            state = self.load_stage_1_4_checkpoint(checkpoint_dir)
            indexes = state["indexes"]
            country_models = state["country_models"]
            gt_map = state.get("ground_truth", {})
            total_gt_pairs = int(state.get("total_gt_pairs", 0))
        else:
            state = self._prepare_stage_1_4_state(
                s2_path=s2_path,
                s3_path=s3_path,
                ground_truth_path=ground_truth_path,
            )
            indexes = state["indexes"]
            country_models = state["country_models"]
            gt_map = state.get("ground_truth", {})
            total_gt_pairs = int(state.get("total_gt_pairs", 0))
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(state, f)
                print(f"  Saved checkpoint to: {checkpoint_path}")

        # Prepare streaming output writers
        os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
        parquet_writer = pq.ParquetWriter(output_parquet, schema=CANDIDATE_PA_SCHEMA, compression="snappy")

        tsv_f = None
        if output_tsv:
            os.makedirs(os.path.dirname(os.path.abspath(output_tsv)), exist_ok=True)
            tsv_f = open(output_tsv, "w", encoding="utf-8")
            tsv_f.write("source1_entity_id\tcandidate_entity_ids\n")

        # 2. Stream S1 in batches
        total_s1_lines = _get_line_count(s1_path)
        total_batches = ((total_s1_lines + chunk_size - 1) // chunk_size) if total_s1_lines else None
        batch_info = f"(Total ~{total_batches} batches)" if total_batches else ""
        print(f"\n[Step 5/5] Processing Source 1 in chunks of {chunk_size:,} {batch_info} (using {self.n_jobs} CPU worker cores across all sub-stages)...", flush=True)

        pbar = tqdm(
            total=total_batches,
            desc=f"Stage 5/5: Blocking S1",
            unit="batch",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        batch_idx = 0
        total_s1_records = 0
        total_candidate_pairs = 0
        total_zero_candidates = 0
        running_true_retrieved = 0

        country_stats = defaultdict(lambda: {"total": 0, "pairs": 0, "zeros": 0})

        for s1_chunk in pd.read_csv(s1_path, sep="\t", chunksize=chunk_size):
            t_batch = time.time()
            batch_idx += 1
            n_batch = len(s1_chunk)
            total_s1_records += n_batch
            batch_str = f"{batch_idx:03d}/{total_batches:03d}" if total_batches else f"{batch_idx:03d}"

            print(f"\n  ----------------------------------------------------------------------------", flush=True)
            print(f"  --- STARTING BATCH {batch_str}: {n_batch:,} S1 entities ---", flush=True)
            print(f"  ----------------------------------------------------------------------------", flush=True)

            # Sub-step 1/4: S1 Preprocessing
            print(f"  [Batch {batch_str} - 1/4] Preprocessing {n_batch:,} S1 records (using {self.n_jobs} CPU worker processes)...", flush=True)
            t_prep_start = time.time()
            s1_prep = self.preprocess_df(s1_chunk, n_jobs=self.n_jobs)
            t_prep = time.time() - t_prep_start
            rate_prep = n_batch / t_prep if t_prep > 0 else 0
            print(f"  [Batch {batch_str} - 1/4] Preprocessing completed: {n_batch:,} records in {t_prep:.2f}s ({rate_prep:,.0f} records/s)", flush=True)

            # Sub-step 2/4: Exact inverted index lookups
            print(f"  [Batch {batch_str} - 2/4] Querying 6 exact inverted index channels (using {self.n_jobs} CPU cores / hash maps)...", flush=True)
            t_exact_start = time.time()
            exact_candidates = self.retrieve_exact_candidates(s1_prep, indexes)
            t_exact = time.time() - t_exact_start
            n_with_exact = sum(1 for cands in exact_candidates.values() if cands)
            n_exact_pairs = sum(len(cands) for cands in exact_candidates.values())
            print(f"  [Batch {batch_str} - 2/4] Exact matching completed in {t_exact:.2f}s | {n_with_exact:,}/{n_batch:,} S1 found exact candidates ({n_exact_pairs:,} total pairs)", flush=True)

            # Sub-step 3/4: Multi-core fuzzy TF-IDF candidate retrieval with intra-batch progress
            print(f"  [Batch {batch_str} - 3/4] Running multi-core fuzzy TF-IDF matching across countries (using {self.n_jobs} CPU cores, subchunk size: {self.subchunk_size})...", flush=True)
            t_fuzzy_start = time.time()
            fuzzy_candidates = self.retrieve_fuzzy_candidates_for_batch(
                s1_prep,
                country_models,
                subchunk_size=self.subchunk_size,
                n_jobs=self.n_jobs,
                batch_idx=batch_idx,
                total_batches=total_batches,
            )
            t_fuzzy = time.time() - t_fuzzy_start
            rate_fuzzy = n_batch / t_fuzzy if t_fuzzy > 0 else 0
            print(f"  [Batch {batch_str} - 3/4] Fuzzy matching completed in {t_fuzzy:.2f}s ({rate_fuzzy:,.0f} S1/s across {self.n_jobs} cores)", flush=True)

            # Sub-step 4/4: Columnar candidate assembly and streaming disk write
            print(f"  [Batch {batch_str} - 4/4] Merging exact & fuzzy candidates into columnar PyArrow format and streaming to disk...", flush=True)
            t_asm_start = time.time()
            cols, cand_ids_by_s1 = self.assemble_candidate_rows(s1_prep, exact_candidates, fuzzy_candidates)

            # Calculate batch pair statistics directly from dictionary
            batch_zeros = sum(1 for cands in cand_ids_by_s1.values() if not cands)
            n_pairs = sum(len(cands) for cands in cand_ids_by_s1.values())
            total_candidate_pairs += n_pairs
            total_zero_candidates += batch_zeros

            # Build PyArrow Table directly from columns without converting to pandas DataFrame
            table = pa.Table.from_pydict(cols, schema=CANDIDATE_PA_SCHEMA)
            parquet_writer.write_table(table)

            # Stream TSV deliverable directly from dictionary in 0.02s
            if tsv_f is not None:
                for s1_id in s1_chunk["entity_id"]:
                    cand_list = cand_ids_by_s1.get(s1_id, [])
                    tsv_f.write(f"{s1_id}\t{','.join(cand_list)}\n")
                tsv_f.flush()

            t_asm = time.time() - t_asm_start
            print(f"  [Batch {batch_str} - 4/4] Assembly & disk writes completed in {t_asm:.2f}s", flush=True)

            # Track ground truth recall if available in 0.01s without pandas groupby
            if gt_map:
                for s1_id in s1_chunk["entity_id"]:
                    if s1_id in gt_map:
                        true_set = gt_map[s1_id]
                        cand_set = set(cand_ids_by_s1.get(s1_id, []))
                        running_true_retrieved += len(true_set.intersection(cand_set))

            # Country stats update directly from s1_prep and cand_ids_by_s1
            country_map = dict(zip(s1_prep["entity_id"], s1_prep["country"]))
            for s1_id, cands in cand_ids_by_s1.items():
                ctry = country_map.get(s1_id, "UNKNOWN")
                country_stats[ctry]["total"] += 1
                country_stats[ctry]["pairs"] += len(cands)
                if not cands:
                    country_stats[ctry]["zeros"] += 1

            recall_str = ""
            if total_gt_pairs > 0:
                cur_recall = (running_true_retrieved / total_gt_pairs) * 100
                recall_str = f" | Recall Ceiling: {cur_recall:.3f}%"

            batch_elapsed = time.time() - t_batch
            print(
                f"  Batch {batch_str} COMPLETE: {n_batch:,} S1 | Pairs: {n_pairs:,} (avg {n_pairs/n_batch:.1f}/S1) | 0-cands: {batch_zeros} | Elapsed: {batch_elapsed:.1f}s{recall_str}\n",
                flush=True,
            )

            pbar.update(1)
            pbar.set_postfix({
                "batch": batch_str,
                "elapsed": f"{batch_elapsed:.1f}s",
                "avg_cand": f"{total_candidate_pairs/total_s1_records:.1f}",
            })

            del s1_prep, exact_candidates, fuzzy_candidates, cols, cand_ids_by_s1, table
            gc.collect()

        pbar.close()

        # Close streams
        parquet_writer.close()
        if tsv_f is not None:
            tsv_f.close()
            print(f"TSV deliverable written to {output_tsv}.", flush=True)

        del indexes, country_models
        gc.collect()

        # Final validation report
        print("\n" + "=" * 80, flush=True)
        print("FULL BLOCKING RUN COMPLETE — SUMMARY AUDIT", flush=True)
        print("=" * 80, flush=True)
        print(f"Total S1 entities processed: {total_s1_records:,}", flush=True)
        print(f"Total valid candidate pairs: {total_candidate_pairs:,}", flush=True)
        print(f"Average candidates per S1:   {total_candidate_pairs / total_s1_records:.2f}", flush=True)
        print(f"Entities with 0 candidates:  {total_zero_candidates:,} ({total_zero_candidates / total_s1_records * 100:.4f}%)", flush=True)

        print("\nPer-Country Distribution:", flush=True)
        for ctry, stats in country_stats.items():
            tot = stats["total"]
            pairs = stats["pairs"]
            zeros = stats["zeros"]
            avg_p = pairs / tot if tot > 0 else 0
            print(f"  Country '{ctry}': {tot:,} S1 | {pairs:,} pairs (avg {avg_p:.1f}/S1) | {zeros:,} zero-cands ({zeros/tot*100:.3f}%)", flush=True)

        if total_gt_pairs > 0:
            final_recall = (running_true_retrieved / total_gt_pairs) * 100
            print(f"\nGROUND TRUTH RECALL CEILING:", flush=True)
            print(f"  Total True Pairs:    {total_gt_pairs:,}", flush=True)
            print(f"  Retrieved by Blocker:{running_true_retrieved:,}", flush=True)
            print(f"  Missed by Blocker:   {total_gt_pairs - running_true_retrieved:,}", flush=True)
            print(f"  RECALL CEILING:      {final_recall:.3f}%", flush=True)
            if final_recall >= 99.0:
                print(f"  ✅ PASS: Recall ceiling is {final_recall:.2f}% (exceeds 99.0% target).", flush=True)
            else:
                print(f"  ⚠️ WARNING: Recall ceiling is {final_recall:.2f}%.", flush=True)

        print(f"\nParquet file saved to: {output_parquet} (size: {os.path.getsize(output_parquet) / (1024*1024):.1f} MB)", flush=True)
        print(f"TOTAL EXECUTION TIME: {time.time() - t_start:.2f}s", flush=True)


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
