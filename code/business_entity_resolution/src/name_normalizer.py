#!/usr/bin/env python3
"""
Business Name Cleaning & Normalization Pipeline.
Section 4 of pre_blocking_data_prep_plan.md.

Produces clean, comparable, information-preserving name representations for blocking & matching:
1. Shared text primitives (Unicode NFKC, noise strip, script-routed transliteration, domain strip).
2. DBA / trade-name splitting (dba, t/a, f/k/a, parenthetical trade names, filtering out (India)/(Regd) noise).
3. Legal-suffix expansion (pvt ltd -> private limited, inc -> incorporated, etc.).
4. Core-name extraction (legal suffix stripped, guarded against empty/too-short outputs).
5. Token-order normalization (sorted_tokens key).
6. Acronym generation (first-letter-of-each-token).
7. Prefix key generation (first 4-5 characters of core_name).
8. Phonetic key generation (token Soundex).
"""

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from src.text_utils import normalize_primitive


def _find_config(filename: str) -> str:
    """Find config file across root, src parent, or submission code directory."""
    candidates = [
        os.path.join(os.getcwd(), "configs", filename),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", filename)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", filename)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


def _load_json_config(filename: str) -> dict:
    path = _find_config(filename)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


NAME_CONFIG = _load_json_config("name_abbreviations.json")
LEGAL_EXPANSION = NAME_CONFIG.get("legal_suffixes_expansion", {})
LEGAL_STRIP_LIST = NAME_CONFIG.get("legal_suffixes_to_strip", [])

GENERIC_CONFIG = _load_json_config("generic_name_tokens.json")
GENERIC_TOKENS: Set[str] = {item["token"] for item in GENERIC_CONFIG} if isinstance(GENERIC_CONFIG, list) else set()

# Pre-compile DBA trigger patterns
RE_DBA_TRIGGERS = re.compile(
    r'\b(?:doing\s+business\s+as|formerly\s+known\s+as|trading\s+as|d/b/a|dba|t/a|f/k/a|a/k/a|formerly|c/o)\b',
    re.IGNORECASE
)

# Parentheses pattern
RE_PARENS = re.compile(r'\(([^)]+)\)')

# Geographic and administrative qualifiers inside parentheses that must NOT be treated as DBA trade names
PAREN_STOPWORDS = {
    "india", "i", "us", "usa", "regd", "registered", "urban", "rural",
    "east", "west", "north", "south", "pvt ltd", "ltd", "inc", "corp",
    "llc", "llp", "mumbai", "bombay", "delhi", "calcutta", "kolkata",
    "chennai", "madras", "pune", "gujarat", "karnataka", "maharashtra",
    "partix", "ind", "mohali", "indore", "goa", "thrissur", "global"
}

# Compile sorted legal suffix patterns for regex substitution (longest first)
SORTED_LEGAL_STRIP = sorted(LEGAL_STRIP_LIST, key=len, reverse=True)
RE_LEGAL_STRIP_SUFFIX = re.compile(
    r'(?:\s+|,|\.)+(?:' + '|'.join(re.escape(s) for s in SORTED_LEGAL_STRIP) + r')\s*$',
    re.IGNORECASE
)
RE_LEGAL_STRIP_PREFIX = re.compile(
    r'^\s*(?:' + '|'.join(re.escape(s) for s in SORTED_LEGAL_STRIP) + r')(?:\s+|,|\.)+',
    re.IGNORECASE
)

# Soundex mapping table
SOUNDEX_MAPPING = {
    "B": "1", "F": "1", "P": "1", "V": "1",
    "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
    "D": "3", "T": "3", "L": "4", "M": "5", "N": "5", "R": "6"
}


def soundex(token: str) -> str:
    """Compute standard American Soundex phonetic code for a single token."""
    if not token or not token[0].isalpha():
        return ""
    token = token.upper()
    first = token[0]
    res = [first]
    prev = SOUNDEX_MAPPING.get(first, "")
    for ch in token[1:]:
        code = SOUNDEX_MAPPING.get(ch, "")
        if code and code != prev:
            res.append(code)
            if len(res) == 4:
                break
        prev = code
    return ("".join(res) + "000")[:4]


def split_dba_names(raw_name: str) -> Tuple[str, List[str]]:
    """
    Detect DBA / trade names (e.g. 'X Corp DBA Y', 'X t/a Y', 'X formerly known as Y', 'X (Y)').
    Filters out geographic parenthetical qualifiers like '(India)' or '(Regd.)'.
    Returns:
        (primary_name, [list_of_alternate_trade_names])
    """
    if not raw_name:
        return "", []

    alt_names = []
    primary = raw_name

    # Check pipe separator (e.g. 'NL LLC Pgim | www.nlllcpgi.com' or 'Brand | Trade')
    if "|" in primary:
        pipe_parts = primary.split("|")
        primary = pipe_parts[0].strip()
        for p in pipe_parts[1:]:
            cleaned_p = p.strip()
            if not re.match(r'^(?:https?://|www\.)\S+$', cleaned_p, re.I) and len(cleaned_p) >= 3:
                alt_names.append(cleaned_p)

    # Check explicit DBA triggers: 'dba', 'doing business as', 't/a', 'formerly', 'f/k/a'
    parts = RE_DBA_TRIGGERS.split(primary)
    if len(parts) > 1:
        primary = parts[0].strip()
        for p in parts[1:]:
            cleaned_p = p.strip()
            if len(cleaned_p) >= 3:
                alt_names.append(cleaned_p)

    # Check parentheses
    paren_matches = RE_PARENS.findall(primary)
    if paren_matches:
        for p in paren_matches:
            p_clean = p.strip().lower()
            # Only treat as DBA if not a geographic/administrative qualifier
            if p_clean not in PAREN_STOPWORDS and len(p_clean) >= 3 and not p_clean.isdigit():
                alt_names.append(p.strip())
        # Remove parenthetical content from primary name
        primary = RE_PARENS.sub(" ", primary).strip()

    return primary, alt_names


def expand_legal_suffixes(text: str) -> str:
    """Expand legal abbreviations to canonical long forms (pvt ltd -> private limited, etc.)."""
    if not text:
        return ""
    tokens = text.split()
    if not tokens:
        return ""

    # Check trailing two-token suffix
    if len(tokens) >= 2:
        last_two = f"{tokens[-2]} {tokens[-1]}"
        if last_two in LEGAL_EXPANSION:
            return " ".join(tokens[:-2]) + " " + LEGAL_EXPANSION[last_two]

    # Check trailing single-token suffix
    if tokens[-1] in LEGAL_EXPANSION:
        return " ".join(tokens[:-1]) + " " + LEGAL_EXPANSION[tokens[-1]]

    # Check leading single-token suffix (suffix repositioning: e.g. 'LLC North Marshall')
    if tokens[0] in LEGAL_EXPANSION:
        return " ".join(tokens[1:]) + " " + LEGAL_EXPANSION[tokens[0]]

    return text


def strip_legal_suffixes(text: str) -> str:
    """
    Remove legal suffixes completely to produce core_name.
    Guarded against empty or too-short names (<3 chars).
    """
    if not text:
        return ""

    # Strip trailing legal suffix
    core = RE_LEGAL_STRIP_SUFFIX.sub("", text).strip()
    # Strip leading legal suffix if transposed
    core = RE_LEGAL_STRIP_PREFIX.sub("", core).strip()

    # Guard: if stripped name became empty or too short, revert to original text
    if len(core) < 3:
        return text.strip()
    return core


def get_sorted_tokens_key(text: str) -> str:
    """Return lowercase, alphabetically sorted tokens joined by single space."""
    if not text:
        return ""
    tokens = sorted(text.split())
    return " ".join(tokens)


def get_acronym_key(text: str) -> str:
    """
    Generate acronym key from first letter of each significant token.
    Only generated if name has at least 2 tokens.
    Example: 'international business machines' -> 'ibm'
    """
    if not text:
        return ""
    tokens = [t for t in text.split() if t.isalpha()]
    if len(tokens) >= 2:
        return "".join(t[0] for t in tokens)
    return ""


def get_char_ngrams(text: str, n: int = 3) -> List[str]:
    """
    Generate character n-grams on-demand for matching-stage candidate pairs.
    (Not precomputed across the 5M dataset to save memory and blocking time).
    """
    if not text:
        return []
    compact = re.sub(r'\s+', ' ', text.strip().lower())
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[i:i+n] for i in range(len(compact) - n + 1)]


def get_prefix_key(text: str, length: int = 5) -> str:
    """Return first N alphanumeric characters of the name as a prefix blocking key."""
    if not text:
        return ""
    clean = re.sub(r'[^a-z0-9]', '', text.lower())
    return clean[:length]


def get_phonetic_key(text: str, max_tokens: int = 2) -> str:
    """
    Compute compound Soundex phonetic key for the most significant non-generic tokens.
    Example: 'north marshall' -> 'N630-M624'
    """
    if not text:
        return ""
    tokens = [t for t in text.split() if t.isalpha()]
    if not tokens:
        return ""

    # Prioritize non-generic tokens if available
    sig_tokens = [t for t in tokens if t not in GENERIC_TOKENS]
    chosen = sig_tokens if sig_tokens else tokens

    codes = [soundex(t) for t in chosen[:max_tokens] if soundex(t)]
    return "-".join(codes)


def normalize_business_name(raw_name: str) -> Dict[str, any]:
    """
    Full Business Name Cleaning Pipeline (Section 4).
    
    Returns:
        dict with:
        - raw_name: original unaltered input
        - clean_name: primitive cleaned + legal suffix expanded
        - core_name: legal suffix stripped, length-guarded
        - sorted_tokens: alphabetically sorted core tokens
        - acronym_key: acronym from core tokens (if >= 2 words)
        - prefix_key: first 4-5 chars of core name
        - sorted_prefix_key: first 4-5 chars of sorted_tokens version
        - phonetic_key: compound Soundex of top significant tokens
    """
    if raw_name is None or (isinstance(raw_name, float) and (raw_name != raw_name)) or not str(raw_name).strip():
        return {
            "raw_name": "" if raw_name is None or (isinstance(raw_name, float) and raw_name != raw_name) else str(raw_name),
            "clean_name": "",
            "core_name": "",
            "sorted_tokens": "",
            "acronym_key": "",
            "prefix_key": "",
            "sorted_prefix_key": "",
            "phonetic_key": "",
            "alt_names": []
        }

    # Step 1: Detect and split DBA / parenthetical trade names
    primary_raw, alt_raw_list = split_dba_names(raw_name)

    # Step 2: Apply shared primitives (NFKC, noise strip, script-routed transliteration, domain strip, lowercase)
    clean_base = normalize_primitive(primary_raw, is_name=True, apply_deconcat=True)
    # Strip any trailing/internal commas, semicolons, and hyphens separating entity from suffix
    # e.g. 'One Systems-L.L.P.' -> 'one systems llp', 'Lightcap and-Velasquez' -> 'lightcap and velasquez'
    clean_base = re.sub(r'[,;]+', ' ', clean_base)
    clean_base = re.sub(r'[-–—_]+', ' ', clean_base)
    clean_base = re.sub(r'\s+', ' ', clean_base).strip()

    # Step 3: Expand legal suffixes to canonical long form
    clean_name = expand_legal_suffixes(clean_base)

    # Step 4: Extract core_name (strip legal suffixes, guarded against <3 chars)
    core_name = strip_legal_suffixes(clean_base)
    if len(core_name) < 3:
        core_name = clean_name if len(clean_name) >= 3 else clean_base

    # Step 5: Generate derived blocking keys
    sorted_tokens = get_sorted_tokens_key(core_name)
    acronym = get_acronym_key(core_name)
    prefix_key = get_prefix_key(core_name, length=5)
    sorted_prefix_key = get_prefix_key(sorted_tokens, length=5)
    phonetic_code = get_phonetic_key(core_name)

    # Step 6: Process alternate DBA names with same primitive pipeline
    cleaned_alts = []
    for alt in alt_raw_list:
        alt_norm = normalize_primitive(alt, is_name=True, apply_deconcat=True)
        alt_norm = re.sub(r'[,;]+', ' ', alt_norm)
        alt_norm = re.sub(r'[-–—_]+', ' ', alt_norm)
        alt_norm = re.sub(r'\s+', ' ', alt_norm).strip()
        alt_core = strip_legal_suffixes(alt_norm)
        if alt_core and alt_core != core_name:
            cleaned_alts.append(alt_core)

    return {
        "raw_name": raw_name,
        "clean_name": clean_name,
        "core_name": core_name,
        "sorted_tokens": sorted_tokens,
        "acronym_key": acronym,
        "prefix_key": prefix_key,
        "sorted_prefix_key": sorted_prefix_key,
        "phonetic_key": phonetic_code,
        "alt_names": cleaned_alts
    }
