# Comprehensive Pipeline Implementation Summary & Progress Report

**Project:** Business Entity Resolution ML Challenge (2026)  
**Evaluation Metric:** Macro-averaged $F_{0.5}$ (Precision weighted 2x over Recall: $\beta = 0.5$)  
**Status:** Pre-Blocking Data Preparation, Feature Engineering, Multi-Channel Blocker, and Comprehensive Quality Audits Complete. Ready for Downstream Matcher (LightGBM).

---

## 1. Executive Summary & Key Milestones

| Stage | Component | Key Achievements & Metrics | Status |
| :--- | :--- | :--- | :---: |
| **§0 & §1** | **Architecture & Validation** | Leak-free stratified 50k validation split; 1:1 dual mirror (`src/` $\leftrightarrow$ `code/business_entity_resolution/src/`) | **COMPLETE** |
| **§2** | **EDA & Noise Audit** | Audited 26.4M records; verified 0.0000% country mismatch in true pairs; documented OCR/Indic/DBA noise | **COMPLETE** |
| **§3** | **Text Primitives** | Unicode NFKC, script-aware Indic transliteration (Sanscript ITRANS + schwa-deletion), URL/digit normalization | **COMPLETE** |
| **§4** | **Name Pipeline** | Bidirectional legal suffix expansion, suffix repositioning, DBA/trade-name splitting, token sorting, phonetic Soundex | **COMPLETE** |
| **§5** | **Address Pipeline** | Pre-expansion landmark/PIN extraction, comma-independent state anchor parser, lookahead protection, street standardizer | **COMPLETE** |
| **§6** | **Country Normalization** | Alias mapping (`USA`/`Bharat`/`FRA` $\to$ canonical) with open-set pass-through for unseen test countries | **COMPLETE** |
| **§7** | **Multi-Channel Blocker** | 6 exact inverted index channels + Country-partitioned fuzzy TF-IDF char-ngram channel; **99.13% blocking recall** | **COMPLETE** |
| **Audit** | **Quality & Safety Checks** | 600k row null check (0.0% name drop), S2/S3 duplicate check (0% over-collapse), +32.6% noise reduction, 10-fixture regression suite | **COMPLETE** |

---

## 2. Architecture & Foundation (§0 – §2)

### 2.1. Strict Submission Parity
Every configuration, module, and utility is maintained in a 1:1 synchronized structure between the active development environment and the competition package mirror:
- Active development root: `src/`, `configs/`, `tests/`, `scripts/`
- Submission mirror: `code/business_entity_resolution/src/`, `code/business_entity_resolution/configs/`

### 2.2. Leak-Free Validation Split
- Generated via `scripts/create_validation_split.py`.
- Evaluated on 50,000 reference entities from `train_source1.tsv` and all their corresponding matches in `train_source2.tsv` and `train_source3.tsv`.
- Hash-partitioned on `source1_entity_id`, ensuring zero entity overlap between training and validation sets while preserving source distributions and country proportions.

### 2.3. Key Insights from 26.4M Row Noise Audit
1. **Country Integrity:** Out of 172,839 ground-truth true match pairs in training, **0 pairs cross country boundaries (0.0000% error rate)**. Hard-partitioning candidate blocking by country is mathematically sound and cuts candidate generation search complexity by ~60%.
2. **Open-Set Mandate:** Training data covers `US` and `India`; test data introduces `France` and potential unseen countries. Normalizers must never filter or crash on unseen countries.
3. **Address Granularity Discrepancy:** Indian records exhibit heavy suburb-vs-metro mismatches (e.g. *Powai* vs *Mumbai*), and postal codes are present in <1% of Indian records and <7% of US records. Address blocking must rely on State anchors and sorted tokens rather than rigid city equality or postal codes.

---

## 3. Text-Normalization Primitives (§3)

Implemented in [`src/text_utils.py`](file:///Users/punyapratap/Downloads/ML_hackathon/src/text_utils.py):
- **Unicode NFKC Normalization:** Decomposes and recomposes ligatures, non-breaking spaces (`\u00a0`), smart quotes, em-dashes, and decorative bullets.
- **Script-Detected Transliteration:**
  - Standard `anyascii` systematically drops Indic vowel matras (e.g. बेंगलुरु $\to$ *bgluru*).
  - Implemented regex routing: Devanagari (`[\u0900-\u097F]`) and Bengali (`[\u0980-\u09FF]`) route to `indic_transliteration.sanscript` with ITRANS scheme, followed by custom programmatic schwa-deletion (dropping terminal 'a' and medial cluster 'a's).
  - Latin and French diacritics pass cleanly through ASCII folding.
- **Domain & URL Stripping:** Strips common domain extensions (`.com`, `.in`, `.org`, `.net`, `.co`, `.io`, `.gov`) frequently attached to company names.
- **Symbol & Number Standardization:**
  - Standardizes `&` $\to$ `and`, `@` $\to$ `at`.
  - Preserves slashes `/` in addresses for suite, plot, and floor designations (e.g., `219 1/2`, `Flat No. 101/A`).
  - Leading zero stripping on street numbers (`re.sub(r'\b0+([1-9]\d*)\b', r'\1', text)`), resolving `02814` $\to$ `2814`.
- **CamelCase De-concatenation:** Splits accidental concatenated words and number-word boundaries (`ApexSolutions` $\to$ `Apex Solutions`).

---

## 4. Business Name Cleaning Pipeline (§4)

Implemented in [`src/name_normalizer.py`](file:///Users/punyapratap/Downloads/ML_hackathon/src/name_normalizer.py) and [`configs/name_abbreviations.json`](file:///Users/punyapratap/Downloads/ML_hackathon/configs/name_abbreviations.json):

1. **Bidirectional Legal Suffix Expansion:**
   Expands legal abbreviations to canonical long forms:
   - `pvt ltd` $\to$ `private limited`
   - `inc` $\to$ `incorporated`
   - `llc` $\to$ `limited liability company`
   - `llp` $\to$ `limited liability partnership`
   - `corp` $\to$ `corporation`
2. **Suffix Repositioning:** Detects inverted legal suffixes at the start of names (`LLC Cornerstone Cloud Labs` $\to$ `Cornerstone Cloud Labs limited liability company`).
3. **Core Name Stripping:** Produces `core_name` by stripping legal suffixes, guarded with a minimum length constraint ($\ge 3$ characters) to prevent names like `LLC` or `Inc` from disappearing.
4. **DBA / Trade-Name Splitting (`split_dba_names`):**
   Detects DBA triggers (`dba`, `doing business as`, `t/a`, `trading as`, `formerly`, `f/k/a`, pipe `|`) and parenthetical aliases. Filters out parenthetical region or registration markers like `(India)` or `(Regd)`. Emits legal name and alternative trade names into `alt_names`.
5. **Multi-Representation Key Generation:**
   - `sorted_tokens`: Word-order invariant token string (`"star electronics"` $\leftrightarrow$ `"electronics star"`).
   - `prefix_key`: 5-character prefix of primary name tokens.
   - `acronym_key`: First letter of significant tokens (filtered for length $\ge 3$).
   - `phonetic_key`: Soundex phonetic key filtered of 50 high-frequency generic corporate tokens ([`configs/generic_name_tokens.json`](file:///Users/punyapratap/Downloads/ML_hackathon/configs/generic_name_tokens.json)) to avoid candidate explosion.
- **Throughput:** 84,400+ names/sec.

---

## 5. Address Cleaning & Structuring Pipeline (§5)

Implemented in [`src/address_parser.py`](file:///Users/punyapratap/Downloads/ML_hackathon/src/address_parser.py) and [`configs/address_abbreviations.json`](file:///Users/punyapratap/Downloads/ML_hackathon/configs/address_abbreviations.json):

1. **Pre-Expansion Extraction:**
   - Extracts postal codes (US 5-digit ZIP, Indian 6-digit PIN) and landmarks (`near`, `opp`, `behind`, `next to`, `b/h`) **before** expanding street abbreviations.
   - Stored in a separate `landmark_text` field; prevents trigger words from being corrupted by abbreviation expanders.
2. **Street Abbreviation Standardizer:**
   - Normalizes street types across US, India, and France (`rd` $\to$ `road`, `st` $\to$ `street`, `ave` $\to$ `avenue`, `blvd` $\to$ `boulevard`, `hwy` $\to$ `highway`).
   - Boundary protection: Prevents `CT` (Connecticut) at line starts or state boundaries from expanding to `court`.
3. **Anchor-First Structural Parsing (`split_by_state_anchor`):**
   - Independent of comma delimiters. Locates known state names, 2-letter state codes (`CA`, `NY`, `MH`, `DL`, `KA`, `WB`, etc.), or French regions anywhere in the string.
   - Negative lookahead protection (`STREET_OR_CITY_LOOKAHEAD`): Rejects state matches that are actually street names (e.g. `Washington Street`, `Indiana Ave`) or cities (`Kansas City`).
   - Handles inverted addresses (`State, City, Street`) and middle state occurrences (`City, State, Street`).
4. **Generic Fallback Parser:**
   - Open-set handler for France or any unseen country in test data.
   - Emits structured components (`street`, `city`, `state`, `postal_code`, `sorted_address_tokens`) without crashing.
- **Throughput:** 23,800+ addresses/sec.

---

## 6. Country Field Standardization (§6)

Implemented in [`src/country_normalizer.py`](file:///Users/punyapratap/Downloads/ML_hackathon/src/country_normalizer.py) and [`configs/country_aliases.json`](file:///Users/punyapratap/Downloads/ML_hackathon/configs/country_aliases.json):
- Normalizes country aliases into canonical labels:
  - `USA`, `United States`, `U.S.`, `U.S.A.` $\to$ `US`
  - `IND`, `IN`, `Bharat` $\to$ `India`
  - `FR`, `FRA` $\to$ `France`
- Open-set pass-through: Any unseen country label in test data is sanitized, whitespace-trimmed, and passed through in title-case (or uppercase if 2–3 letter code) without record drops.

---

## 7. Multi-Channel Candidate Blocker (§7)

Implemented in [`src/blocker.py`](file:///Users/punyapratap/Downloads/ML_hackathon/src/blocker.py):

### 7.1. Channel Architecture
Combines 6 exact-key inverted index channels with a robust fuzzy cosine similarity channel:

```
[Reference Entity S1]
       │
       ├── Exact Channel 1: (country, core_name) + DBA alt_names
       ├── Exact Channel 2: (country, sorted_tokens)
       ├── Exact Channel 3: (country, state, prefix_key[:4])
       ├── Exact Channel 4: (country, state, phonetic_key)
       ├── Exact Channel 5: (country, acronym_key) [len >= 3]
       ├── Exact Channel 6: (country, sorted_address_tokens)
       │
       └── Fuzzy Channel: Country-Partitioned Sublinear TF-IDF Char-3grams
                          (clean_name + " " + clean_address)
                          Chunked Sparse Dot-Product (cosine >= 0.28)
       │
[Priority Assembly]
       ├── 1. Exact-match candidates inserted first (Guaranteed 0% eviction)
       └── 2. Top-scoring fuzzy matches append up to cap (max_total = 75)
       │
[Candidate Pool for Matcher]
```

### 7.2. Empirical Validation Results
Evaluated on **2,000 S1 records** representing **5,881 true match pairs** against the entire validation pool of **233,825 candidate records** (`val_source2` + `val_source3`):

| Evaluation Metric | Exact Channels Only | Combined (+ Fuzzy TF-IDF Channel) | Impact |
| :--- | :---: | :---: | :---: |
| **True Match Recall** | **86.29%** (5,075 / 5,881) | **99.13%** (5,830 / 5,881) | **+12.84%** (+755 pairs recovered) |
| **Average Candidates / S1** | ~8.4 | **51.7** (Median: 47.0) | High efficiency |
| **Search Space Reduction** | 99.996% | **99.978%** | 99.98% cartesian product pruned |
| **Throughput** | 120 S1/sec | **91.9 S1/sec** | ~21.7s for 2,000 S1 entities |

---

## 8. Quality & Safety Audits

### 8.1. Null / Empty Counts per Derived Field (600,000 Records Audited)

Audited across `source1`, `source2`, and `source3` in all 3 countries (`US`, `India`, `France`):

```
Derived Column         |   source1/US   | source1/India  | source1/France |   source2/US   | source2/India  | source2/France |   source3/US   | source3/India  | source3/France
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
clean_name             |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%     
core_name              |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%     
sorted_tokens          |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%     
prefix_key             |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%      |      0.0%     
phonetic_key           |      0.0%      |      0.0%      |      0.0%      |      0.7%      |      0.4%      |      0.3%      |      0.9%      |      0.5%      |      0.4%     
acronym_key            |      6.0%      |      0.5%      |      0.5%      |      8.1%      |      4.0%      |      4.4%      |     10.9%      |      7.0%      |      7.0%     
street                 |      0.0%      |      0.0%      |      0.0%      |      3.5%      |      2.7%      |      3.3%      |      3.4%      |      2.9%      |      3.2%     
city                   |      0.0%      |      0.1%      |      0.0%      |      3.4%      |      2.7%      |      3.4%      |      3.3%      |      2.8%      |      3.2%     
state                  |      0.0%      |      0.0%      |      0.0%      |      3.4%      |      2.6%      |     34.2%      |      3.3%      |      2.7%      |     32.9%     
postal_code            |     93.0%      |     100.0%     |     99.6%      |     93.7%      |     100.0%     |     99.7%      |     93.5%      |     100.0%     |     99.7%     
landmark_text          |     100.0%     |     88.8%      |     100.0%     |     100.0%     |     90.4%      |     100.0%     |     100.0%     |     92.8%      |     100.0%    
sorted_address_tokens  |      0.0%      |      0.0%      |      0.0%      |      3.4%      |      2.6%      |      3.3%      |      3.3%      |      2.7%      |      3.2%     
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Total Evaluated        |     98309      |     86708      |     14983      |     98105      |     87392      |     14503      |     97995      |     87778      |     14227     
```
- **Zero Names Disappeared:** `clean_name`, `core_name`, and `sorted_tokens` are 100.0% populated across all sources and countries.

### 8.2. Duplicate Detection within S2 and S3 (150,000 Records Each)
- **Source 2 Duplicate Rate:** **0.50%** (753 records across 375 multi-listing groups).
- **Source 3 Duplicate Rate:** **0.36%** (534 records across 266 multi-listing groups).
- **Eyeball Verification:** All inspected groups were genuine duplicate listings of the same businesses (differing only by diacritics, domain suffixes, or suffix order). **Zero over-collapse of distinct businesses.**

### 8.3. Noise Reduction Validation (2,500 True Pairs)
- **Raw Untouched String Similarity:** **63.66 / 100**
- **Cleaned Pipeline Similarity:** **84.39 / 100**
- **Net Gain from Normalization:** **+20.73 points (+32.6% improvement)**

### 8.4. Over-Collapse Spot Check (1,000 Random Non-Matching Pairs)
- **Exact Name Channel Collisions:** `core_name` (0.00%), `sorted_tokens` (0.00%), `acronym_key` (0.00%), `phonetic_key` (0.00%), `prefix_key` (0.10%).
- **Fuzzy TF-IDF Channel False-Positive Rate:** Only **0.50%** (5 / 1,000) exceeded `threshold = 0.28`. Mean cosine similarity on non-matching pairs was **0.037** (95th percentile: **0.109**).

### 8.5. Fixed Regression Test Suite ([`tests/test_pipeline_regression.py`](file:///Users/punyapratap/Downloads/ML_hackathon/tests/test_pipeline_regression.py))
10 end-to-end regression fixtures covering:
1. Severe typo pair (*Apex Digital Solutions Inc* $\leftrightarrow$ *Apex Digitl Solutns*)
2. DBA trade name drift (*Maure Williams Colombier Inc* $\leftrightarrow$ *Dréxkor*)
3. CT vs Connecticut state/court disambiguation
4. Comma-free Indian address state/city extraction
5. France open-set word-order invariant matching
6. S2/S3 near-duplicate listing convergence
7. Parenthetical region filtering (`(India)`)
8. Asymmetric missing address (`nan`) stability
9. Standalone leading zero street number normalization (`05034` $\leftrightarrow$ `5034`)
10. Legal suffix repositioning (`LLC Cornerstone Cloud Labs` $\leftrightarrow$ `Cornerstone Cloud Labs LLC`)

All 10 regression fixtures pass:
```bash
PYTHONPATH=. ./.venv/bin/pytest tests/test_pipeline_regression.py
======================= 10 passed, 88 warnings in 0.68s ========================
```

---

## 9. Immediate Next Steps: Downstream Matcher

With candidate blocking recall verified at **99.13%** and average candidate volume at **51.7 candidates per S1 entity**, we are set to implement the matching model:
1. **Pairwise Feature Engineering:**
   - Name string metrics: RapidFuzz `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, `jaro_winkler`, character 3-gram Jaccard, length ratio.
   - Address string metrics: Street number exact match, street name fuzzy ratio, city fuzzy ratio, state equality flag, landmark token overlap.
   - Candidate channel indicators: Exact core-name hit flag, exact address hit flag, TF-IDF cosine score.
2. **Model Training & Tuning:**
   - Train LightGBM binary classifier on candidate pairs generated from `dataset/train/`.
   - Optimize decision threshold strictly for macro-$F_{0.5}$ on `dataset/val/`.
3. **End-to-End Test Set Inference & Submission Validation:**
   - Generate `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
   - Validate with `utils/validate_submission.py`.
