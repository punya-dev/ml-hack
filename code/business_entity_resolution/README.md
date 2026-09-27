# Business Entity Resolution Pipeline

This directory contains the self-contained source code, configuration files, and pinned dependencies for the Business Entity Resolution solution.

## Structure

```
code/business_entity_resolution/
├── src/
│   ├── blocker.py            # Multi-channel inverted index + TF-IDF candidate blocker
│   ├── name_normalizer.py    # Business name cleaning, legal suffix normalization & phonetic keys
│   ├── address_parser.py     # Address parsing, landmark & postal extraction, street expansion
│   ├── country_normalizer.py # Country alias mapping & ISO normalization
│   ├── text_utils.py         # Unicode NFKC normalization, noise strip, script routing
│   └── __init__.py
├── configs/
│   ├── name_abbreviations.json     # Legal suffix mappings and rules
│   ├── address_abbreviations.json  # Street suffix and state abbreviation lookup
│   ├── generic_name_tokens.json    # High-frequency stop tokens
│   └── country_aliases.json        # Country variant aliases
├── README.md                 # Pipeline reproduction instructions
└── requirements.txt          # Pinned runtime dependencies
```

## Reproduction & Pipeline Execution

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Candidate Blocking (Step 1)**:
   From project root:
   ```bash
   python pipeline_raw_to_blocking.py --mode train
   python pipeline_raw_to_blocking.py --mode test
   ```
   Outputs:
   - `output/test_candidates.parquet`
   - `output/candidate_pairs.tsv`

3. **Feature Engineering & Model Matching (Step 2)**:
   ```bash
   python phase2_label_candidates.py --split train
   python phase3_feature_engineering.py --split train
   python phase45_train_model.py
   ```
   Outputs:
   - `output/matching_results.tsv`
