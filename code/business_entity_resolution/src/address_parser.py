#!/usr/bin/env python3
"""
Address Cleaning & Structuring Pipeline.
Section 5 of pre_blocking_data_prep_plan.md.

Implements country-aware address cleaning and structuring:
1. Shared text primitives (Unicode NFKC, noise strip, script-routed transliteration).
2. Postal code & landmark extraction (done BEFORE abbreviation expansion).
3. Street-type abbreviation expansion (Rd -> road, St -> street, etc.).
4. Country-aware structural parsing (US, India, Generic Fallback for open set / France).
5. Explicit missingness tracking (boolean flags, no 'unknown' imputation).
6. Component reordering tolerance (sorted_address_tokens key).
7. On-demand character n-grams for matching stage.
"""

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from src.text_utils import (
    normalize_primitive,
    collapse_whitespace
)


def _find_config(filename: str) -> str:
    """Find config file across root, src parent, or submission code directory."""
    candidates = [
        os.path.join(os.getcwd(), "configs", filename),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", filename)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", filename)),
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


ADDR_CONFIG = _load_json_config("address_abbreviations.json")
STREET_ABBREV = ADDR_CONFIG.get("street_abbreviations", {})
LANDMARK_TRIGGERS = ADDR_CONFIG.get("landmark_triggers", [
    "near", "opp", "oppo", "opposite", "behind", "b/h", "bh", "next to", "close to", "adjacent to"
])
US_STATES = ADDR_CONFIG.get("us_states", {})
INDIAN_STATES = set(ADDR_CONFIG.get("indian_states", []))
INDIAN_STATE_CODES = ADDR_CONFIG.get("indian_state_codes", {})
FRENCH_REGIONS = set(ADDR_CONFIG.get("french_regions", []))

# Compile landmark extraction regex
# Matches landmark trigger followed by phrase up to comma, semicolon, or end of string
RE_LANDMARK = re.compile(
    r'(?:^|[\s,;])\b(' + '|'.join(re.escape(t) for t in sorted(LANDMARK_TRIGGERS, key=len, reverse=True)) +
    r')\b[:\.\s]+([^,;]+)',
    re.IGNORECASE
)

# Postal code patterns (protect 5-digit house numbers followed by street words)
RE_US_ZIP = re.compile(
    r'\b(\d{5}(?:-\d{4})?)(?!\s+(?:st\b|street|rd\b|road|ave\b|avenue|blvd|dr\b|drive|ln\b|lane|way|trail|[a-z]+\s+(?:street|road|drive|avenue|dr|rd|ave|blvd)))\b',
    re.IGNORECASE
)
RE_INDIA_PIN = re.compile(r'\b([1-9]\d{5})\b')
RE_GENERIC_POSTAL = re.compile(r'\b([1-9]\d{4,5})\b')

# Exclude state names followed by street words ('Washington Street') or city ('Kansas City')
STREET_OR_CITY_LOOKAHEAD = r'(?!\s+(?:street|st\b|road|rd\b|avenue|ave\b|blvd|dr\b|drive|lane|ln\b|way|court|ct\b|city\b))'

# Pre-compiled state anchor patterns (sorted longest first)
RE_US_STATE_NAMES = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in sorted(US_STATES.values(), key=len, reverse=True)) + r')\b' + STREET_OR_CITY_LOOKAHEAD,
    re.IGNORECASE
)
RE_US_STATE_CODES = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(US_STATES.keys(), key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)
RE_INDIA_STATE_NAMES = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in sorted(INDIAN_STATES, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)
RE_INDIA_STATE_CODES = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(INDIAN_STATE_CODES.keys(), key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)
RE_FRENCH_REGION_NAMES = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in sorted(FRENCH_REGIONS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)


def extract_landmark(text: str) -> Tuple[str, str]:
    """
    Extract landmark phrase (e.g. 'Near Fortis Hospital', 'Opp. RTA Office', 'B/H Natraj Cinema')
    BEFORE abbreviation expansion.
    Returns:
        (cleaned_text_without_landmark, landmark_text)
    """
    if not text:
        return "", ""

    match = RE_LANDMARK.search(text)
    if not match:
        return text, ""

    trigger = match.group(1).strip()
    phrase = match.group(2).strip()
    full_landmark = f"{trigger} {phrase}".strip()

    # Remove the landmark from the text
    cleaned = text[:match.start()] + " " + text[match.end():]
    cleaned = re.sub(r'[,;]\s*[,;]', ',', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned, full_landmark


def extract_postal_code(text: str, country_code: str = "") -> Tuple[str, str]:
    """
    Extract postal code (ZIP / PIN) BEFORE abbreviation expansion.
    Returns:
        (cleaned_text_without_postal_code, postal_code)
    """
    if not text:
        return "", ""

    code = ""
    c_lower = country_code.lower() if country_code else ""

    if c_lower in ("us", "usa", "united states"):
        m = RE_US_ZIP.search(text)
        if m:
            code = m.group(1)
            text = text[:m.start()] + " " + text[m.end():]
    elif c_lower in ("india", "in", "ind"):
        m = RE_INDIA_PIN.search(text)
        if m:
            code = m.group(1)
            text = text[:m.start()] + " " + text[m.end():]
    else:
        # Generic fallback
        m = RE_GENERIC_POSTAL.search(text)
        if m:
            code = m.group(1)
            text = text[:m.start()] + " " + text[m.end():]

    text = re.sub(r'[,;]\s*[,;]', ',', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text, code


def expand_street_abbreviations(text: str) -> str:
    """Expand street type abbreviations (st -> street, rd -> road, etc.)."""
    if not text:
        return ""

    tokens = text.split()
    new_tokens = []
    n_tokens = len(tokens)
    for idx, t in enumerate(tokens):
        # Strip trailing punctuation for dictionary check
        clean_t = t.rstrip(".,;:")
        punct = t[len(clean_t):]
        lower_t = clean_t.lower()

        # Protect 'ct' (Connecticut) from becoming 'court' if at beginning (CT,), at end, or preceded by comma
        if lower_t == "ct" and (idx == 0 or idx == n_tokens - 1 or (idx > 0 and tokens[idx-1].endswith(","))):
            new_tokens.append(t)
            continue

        if lower_t in STREET_ABBREV:
            new_tokens.append(STREET_ABBREV[lower_t] + punct)
        else:
            new_tokens.append(t)
    return " ".join(new_tokens)


def parse_us_address(clean_text: str) -> Dict[str, str]:
    """
    Country-specific structural parser for US addresses.
    Handles standard (Street, City, State) and inverted (State, City, Street / City, State, Street) formats.
    """
    res = {"street": "", "city": "", "state": ""}
    if not clean_text:
        return res

def split_by_state_anchor(clean_text: str, start_idx: int, end_idx: int, state_val: str) -> Dict[str, str]:
    """
    Split address by an identified state anchor.
    Takes everything before anchor as street+city, anchor as state, and handles any trailing tokens.
    Operates independently of commas.
    """
    prefix = clean_text[:start_idx].strip(",; ")
    suffix = clean_text[end_idx:].strip(",; ")

    # Strip any duplicate/echoed state tokens from prefix or suffix
    if state_val:
        prefix = re.sub(r'\b' + re.escape(state_val) + r'\b', '', prefix, flags=re.I).strip(",; ")
        suffix = re.sub(r'\b' + re.escape(state_val) + r'\b', '', suffix, flags=re.I).strip(",; ")

    # Case 1: Anchor is at or near end (Standard format: Street + City, State)
    if not suffix:
        if "," in prefix:
            segs = [s.strip() for s in prefix.split(",") if s.strip()]
            city = segs[-1]
            street = ", ".join(segs[:-1])
        else:
            tokens = prefix.split()
            if len(tokens) >= 2:
                city = tokens[-1]
                street = " ".join(tokens[:-1])
            else:
                city = ""
                street = prefix
        return {"street": street, "city": city, "state": state_val}

    # Case 2: Anchor is at beginning (Inverted format: State, City, Street)
    if not prefix:
        if "," in suffix:
            segs = [s.strip() for s in suffix.split(",") if s.strip()]
            city = segs[0]
            street = ", ".join(segs[1:])
        else:
            tokens = suffix.split()
            if len(tokens) >= 2:
                city = tokens[0]
                street = " ".join(tokens[1:])
            else:
                city = ""
                street = suffix
        return {"street": street, "city": city, "state": state_val}

    # Case 3: Anchor is in the middle (e.g. "GREENSBORO, NC, 19 1/2 STARDUST TRAIL")
    street_pattern = re.compile(
        r'\d|\b(?:street|road|avenue|drive|lane|way|trail|blvd|st|rd|dr|ave|no|plot|flat|h|house|tower|floor|gut)\b',
        re.I
    )
    pref_is_street = bool(street_pattern.search(prefix))
    suff_is_street = bool(street_pattern.search(suffix))

    if pref_is_street and not suff_is_street:
        street, city = prefix, suffix
    elif suff_is_street and not pref_is_street:
        street, city = suffix, prefix
    else:
        # If ambiguous, prefix is usually city if shorter, suffix is street
        street, city = suffix, prefix

    return {"street": street, "city": city, "state": state_val}


def parse_us_address(clean_text: str) -> Dict[str, str]:
    """
    Anchor-first structural parser for US addresses.
    Searches for state anchor anywhere in string (robust to missing commas and reordered components).
    """
    if not clean_text:
        return {"street": "", "city": "", "state": ""}

    # 1. Search full state name anchor (e.g. "north carolina", "alabama")
    m = list(RE_US_STATE_NAMES.finditer(clean_text))
    if m:
        last_m = m[-1]
        state_val = last_m.group(1).lower()
        return split_by_state_anchor(clean_text, last_m.start(), last_m.end(), state_val)

    # 2. Search 2-letter state code anchor (e.g. "nc", "oh", "tx")
    m = list(RE_US_STATE_CODES.finditer(clean_text))
    if m:
        for match in reversed(m):
            code = match.group(1).lower()
            if code in US_STATES:
                state_val = US_STATES[code]
                return split_by_state_anchor(clean_text, match.start(), match.end(), state_val)

    # 3. Fallback to comma-separated splitting if no state anchor found
    segments = [s.strip() for s in clean_text.split(",") if s.strip()]
    if len(segments) >= 2:
        return {"street": segments[0], "city": segments[-1], "state": ""}
    return {"street": clean_text, "city": "", "state": ""}


def parse_india_address(clean_text: str) -> Dict[str, str]:
    """
    Anchor-first structural parser for Indian addresses.
    Searches for state name or 2-letter code anchor anywhere in string (robust to unpunctuated runs).
    """
    if not clean_text:
        return {"street": "", "city": "", "state": ""}

    # 1. Search full Indian state name anchor (e.g. "punjab", "maharashtra", "west bengal")
    m = list(RE_INDIA_STATE_NAMES.finditer(clean_text))
    if m:
        last_m = m[-1]
        state_val = last_m.group(1).lower()
        return split_by_state_anchor(clean_text, last_m.start(), last_m.end(), state_val)

    # 2. Search 2-letter state code or alias anchor (e.g. "mh", "dl", "wb", "pashchimvang")
    m = list(RE_INDIA_STATE_CODES.finditer(clean_text))
    if m:
        last_m = m[-1]
        code = last_m.group(1).lower()
        state_val = INDIAN_STATE_CODES.get(code, code)
        return split_by_state_anchor(clean_text, last_m.start(), last_m.end(), state_val)

    # 3. Fallback to comma-separated splitting if no state anchor found
    segments = [s.strip() for s in clean_text.split(",") if s.strip()]
    if len(segments) >= 3:
        return {"street": ", ".join(segments[:-2]), "city": segments[-2], "state": segments[-1]}
    elif len(segments) == 2:
        return {"street": segments[0], "city": segments[1], "state": ""}
    return {"street": clean_text, "city": "", "state": ""}


def parse_generic_fallback_address(clean_text: str) -> Dict[str, str]:
    """
    Generic fallback parser for open-set countries (France, unseen countries).
    Extracts state/region anchor if recognizable, tokenizes components cleanly without hardcoded country assumptions.
    """
    if not clean_text:
        return {"street": "", "city": "", "state": ""}

    # 1. Search French region anchor if applicable
    m = list(RE_FRENCH_REGION_NAMES.finditer(clean_text))
    if m:
        last_m = m[-1]
        state_val = last_m.group(1).lower()
        return split_by_state_anchor(clean_text, last_m.start(), last_m.end(), state_val)

    # 2. Positional comma fallback
    segments = [s.strip() for s in clean_text.split(",") if s.strip()]
    if len(segments) >= 3:
        return {"street": segments[0], "city": segments[1], "state": segments[-1]}
    elif len(segments) == 2:
        return {"street": segments[0], "city": segments[1], "state": ""}
    return {"street": clean_text, "city": "", "state": ""}


# Parser Registry (open set routing)
PARSER_REGISTRY = {
    "us": parse_us_address,
    "usa": parse_us_address,
    "united states": parse_us_address,
    "india": parse_india_address,
    "in": parse_india_address,
    "ind": parse_india_address,
}


def get_sorted_address_tokens(text: str) -> str:
    """Return lowercase, alphabetically sorted alphanumeric tokens joined by space."""
    if not text:
        return ""
    # Strip punctuation and get clean tokens
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return " ".join(sorted(tokens))


def get_address_char_ngrams(text: str, n: int = 3) -> List[str]:
    """Generate character n-grams on-demand for matching-stage candidate pairs."""
    if not text:
        return []
    compact = re.sub(r'\s+', ' ', text.strip().lower())
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[i:i+n] for i in range(len(compact) - n + 1)]


def normalize_business_address(raw_address: Optional[str], country: str = "") -> Dict[str, any]:
    """
    Full Address Cleaning & Structuring Pipeline (Section 5).
    
    Returns:
        dict with:
        - raw_address: original input
        - clean_address: normalized, landmark and postal code removed, abbreviations expanded
        - landmark_text: extracted landmark (or "")
        - postal_code: extracted postal code (or "")
        - street: parsed street / building / locality
        - city: parsed city
        - state: parsed state / region
        - has_postal_code: bool
        - has_state: bool
        - has_city: bool
        - has_street: bool
        - sorted_address_tokens: component-reorder tolerant blocking key
    """
    if not raw_address or not str(raw_address).strip() or str(raw_address).strip().lower() in ("nan", "none", "null"):
        return {
            "raw_address": raw_address or "",
            "clean_address": "",
            "landmark_text": "",
            "postal_code": "",
            "street": "",
            "city": "",
            "state": "",
            "has_postal_code": False,
            "has_state": False,
            "has_city": False,
            "has_street": False,
            "sorted_address_tokens": ""
        }

    raw = str(raw_address).strip()

    # Step 1: Apply shared text primitives (NFKC, noise strip, script-routed transliteration)
    clean_base = normalize_primitive(raw, is_name=False, preserve_slashes=True, apply_deconcat=False)

    # Step 2: Extract landmark and postal code BEFORE abbreviation expansion
    clean_no_landmark, landmark_text = extract_landmark(clean_base)
    clean_no_postal, postal_code = extract_postal_code(clean_no_landmark, country_code=country)

    # Step 3: Expand street abbreviations in clean address text
    clean_expanded = expand_street_abbreviations(clean_no_postal)
    # Remove lingering punctuation artifacts
    clean_expanded = re.sub(r'[,;]\s*[,;]', ',', clean_expanded)
    clean_address = collapse_whitespace(clean_expanded)

    # Step 4: Country-aware structural parsing (route by registered parser or generic fallback)
    c_key = country.strip().lower() if country else ""
    parser_func = PARSER_REGISTRY.get(c_key, parse_generic_fallback_address)
    parsed_fields = parser_func(clean_address)

    street = parsed_fields.get("street", "").strip()
    city = parsed_fields.get("city", "").strip()
    state = parsed_fields.get("state", "").strip()

    # Step 5: Explicit missingness flags
    has_postal = bool(postal_code and len(postal_code) >= 3)
    has_state = bool(state and len(state) >= 2)
    has_city = bool(city and len(city) >= 2)
    has_street = bool(street and len(street) >= 2)

    # Step 6: Component reordering tolerance
    sorted_tokens = get_sorted_address_tokens(clean_address)

    return {
        "raw_address": raw,
        "clean_address": clean_address,
        "landmark_text": landmark_text,
        "postal_code": postal_code,
        "street": street if has_street else "",
        "city": city if has_city else "",
        "state": state if has_state else "",
        "has_postal_code": has_postal,
        "has_state": has_state,
        "has_city": has_city,
        "has_street": has_street,
        "sorted_address_tokens": sorted_tokens
    }
