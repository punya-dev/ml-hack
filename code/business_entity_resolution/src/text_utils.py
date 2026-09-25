#!/usr/bin/env python3
"""
Shared Text-Normalization Primitives for Business Entity Resolution.
Section 3 of pre_blocking_data_prep_plan.md.

Pure, stateless functions for low-level text cleaning.
Pipeline Sequence:
1. Unicode NFKC normalization
2. Decorative junk character stripping (#, >, <, [, ], ~, *, leading/trailing bullets)
3. Script-routed Transliteration:
   - Indic scripts (Devanagari, Bengali, Gujarati, Tamil, Kannada, Telugu, Malayalam, Gurmukhi, Oriya):
     Sanscript multi-script transliteration + programmatic linguistic schwa-deletion
     (terminal schwa deletion & medial VC_1 a C_2 V syncope respecting phonotactic constraints).
   - Latin / French / European text: anyascii (retaining clean accents -> ASCII).
4. Domain / TLD suffix stripping (.com, .org, .net, .in, etc.)
5. Punctuation & abbreviation period standardization (Pvt. -> pvt, H.No. -> h no)
   * Preserves '/' in abbreviation contexts (t/a, f/k/a, b/h, a/k/a, c/o)
6. Symbol standardization (& -> and, curly quotes, en/em dashes)
7. Digit / number & ordinal normalization (No.12 -> no 12, KH NO. -> kh no)
8. Case folding (lowercase)
9. Optional de-concatenation for long unspaced blobs (wordninja)
10. Final whitespace and hyphen collapse
"""

import re
import unicodedata
from typing import Optional

try:
    from indic_transliteration import sanscript
except ImportError:
    sanscript = None

try:
    import anyascii
except ImportError:
    anyascii = None

try:
    import unidecode
except ImportError:
    unidecode = None

try:
    import wordninja
except ImportError:
    wordninja = None


# Script detection regexes covering all major Indian scripts
RE_INDIC = re.compile(r'[\u0900-\u0D7F]')

INDIC_SCRIPT_MAP = [
    (re.compile(r'[\u0900-\u097F]'), 'devanagari'),
    (re.compile(r'[\u0980-\u09FF]'), 'bengali'),
    (re.compile(r'[\u0A80-\u0AFF]'), 'gujarati'),
    (re.compile(r'[\u0B80-\u0BFF]'), 'tamil'),
    (re.compile(r'[\u0C80-\u0CFF]'), 'kannada'),
    (re.compile(r'[\u0C00-\u0C7F]'), 'telugu'),
    (re.compile(r'[\u0D00-\u0D7F]'), 'malayalam'),
    (re.compile(r'[\u0A00-\u0A7F]'), 'gurmukhi'),
    (re.compile(r'[\u0B00-\u0B7F]'), 'oriya'),
]

# Pre-compiled regular expressions for high throughput
RE_DECORATIVE_CHARS = re.compile(r'^[#><\[\]~*_\-=:\s]+|[#><\[\]~*_\-=:\s]+$')
RE_INLINE_BRACKETS = re.compile(r'[\[\]\(\)\{\}]')
RE_MULTI_DASH = re.compile(r'[-—–]{2,}')
RE_MULTI_DOT = re.compile(r'\.{2,}')
RE_MULTI_SPACE = re.compile(r'\s+')

# Common domain suffixes (TLDs)
RE_DOMAIN_SUFFIX = re.compile(
    r'\.(?:com|org|net|co\.in|co|in|fr|info|biz|io|gov|edu)(?:\s|$|/|\?)',
    re.IGNORECASE
)
RE_BARE_DOMAIN_SUFFIX = re.compile(
    r'(?<=[a-zA-Z0-9]{3})(?:com|org|net)(?:\s|$)',
    re.IGNORECASE
)

# Abbreviation periods: e.g. L.L.C. -> LLC, P.C. -> PC, U.S. -> US, H.No -> H No
RE_INITIALS = re.compile(r'\b([A-Za-z])\.(?=[A-Za-z]\.|\s|$|,|;)')
RE_INTERIOR_ABBREV_DOT = re.compile(r'\b([A-Za-z]{1,4})\.([A-Za-z]{2,})')
RE_ABBREV_TRAILING_DOT = re.compile(r'\b([A-Za-z]{1,8})\.(?=\s|$|,|;)')

# Digit and Number prefix normalization
RE_NO_PATTERN = re.compile(
    r'\b(kh|plot|h|house|flat|bldg|shop|shp|unit|ste|suite|apt|apartment|fl|floor|no)\s*[-.:#]?\s*no\.?\s*[-.:#]?\s*(\d+)',
    re.IGNORECASE
)
RE_GENERIC_NO = re.compile(r'\bno\.?\s*[-.:#]?\s*(\d+)', re.IGNORECASE)
RE_ORDINAL_FLOOR = re.compile(r'\b(\d+)\s*(?:st|nd|rd|th)\s+(?:fl|floor)\b', re.IGNORECASE)

# Preserved abbreviations with slash: t/a, f/k/a, b/h, a/k/a, c/o, d/b/a
RE_PRESERVED_SLASH = re.compile(r'\b(t/a|f/k/a|b/h|a/k/a|c/o|d/b/a)\b', re.IGNORECASE)


def normalize_unicode(text: str) -> str:
    """Normalize text using Unicode NFKC form."""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def strip_decorative_noise(text: str) -> str:
    """Strip leading/trailing decorative noise symbols (#, >, <, [, ], ~, *, dashes)."""
    if not text:
        return ""
    cleaned = RE_INLINE_BRACKETS.sub(" ", text)
    cleaned = RE_DECORATIVE_CHARS.sub("", cleaned)
    cleaned = RE_MULTI_DASH.sub("-", cleaned)
    cleaned = RE_MULTI_DOT.sub(".", cleaned)
    cleaned = RE_MULTI_SPACE.sub(" ", cleaned)
    return cleaned.strip()


def apply_schwa_deletion(s: str) -> str:
    """
    Apply standard linguistic Hindi/Indic Schwa-Deletion (syncope) rules:
    Rule 1 (Medial Schwa Syncope): Drop medial short /a/ in VC_1 a C_2 V sequences,
           avoiding aspirate clusters with 'h' (e.g. nagapura -> nagpur, karanala -> karnal).
    Rule 2 (Terminal Schwa Deletion): Drop word-final short /a/ after a consonant,
           except for standard tatsama conjuncts (e.g. -tra, -ndra) which retain /a/ in English.
    """
    if not s or len(s) < 3:
        return s
    # Rule 1: Medial Schwa Deletion (avoid creating illegal clusters with h)
    s = re.sub(r'([aeiou][b-df-gj-np-tv-z])a([b-df-hj-np-tv-z][aeiouy])', r'\1\2', s)
    # Rule 2: Terminal Schwa Deletion
    if not re.search(r'(?:tr|ndr|shr|jny)a\b', s):
        if len(s) >= 3:
            s = re.sub(r'([b-df-hj-np-tv-z])a\b', r'\1', s)
    return s


def transliterate_indic_token(token: str) -> str:
    """
    Transliterate a single Indic word using Sanscript with automatic script detection,
    ITRANS phonetic mapping, and linguistic schwa deletion.
    """
    if not sanscript or not RE_INDIC.search(token):
        return token

    # Detect script scheme
    target_scheme = None
    for pattern, scheme_name in INDIC_SCRIPT_MAP:
        if pattern.search(token):
            target_scheme = getattr(sanscript, scheme_name.upper(), None)
            break

    if not target_scheme:
        return token

    try:
        t = sanscript.transliterate(token, target_scheme, sanscript.ITRANS)
        # Normalize ITRANS phonetic tokens to English equivalents
        t = re.sub(r"~[Nn]|\.[Nn]|M", "n", t)
        t = re.sub(r"Sh|sh|z", "sh", t)
        t = re.sub(r"Dh|dh", "dh", t)
        t = re.sub(r"Th|th", "th", t)
        t = re.sub(r"Bh|bh", "bh", t)
        t = re.sub(r"Kh|kh", "kh", t)
        t = re.sub(r"Gh|gh", "gh", t)
        t = re.sub(r"Ch|ch", "ch", t)
        t = re.sub(r"Jh|jh", "jh", t)
        t = re.sub(r"A|aa", "a", t)
        t = re.sub(r"I|ii", "i", t)
        t = re.sub(r"U|uu", "u", t)
        t = re.sub(r"E|ee", "e", t)
        t = re.sub(r"O|oo", "o", t)
        t = re.sub(r"aॉ|ॉ", "o", t)
        t = t.lower()
        # Apply programmatic linguistic schwa deletion
        t = apply_schwa_deletion(t)
        return t
    except Exception:
        if anyascii:
            return anyascii.anyascii(token)
        return token


def transliterate_to_latin(text: str) -> str:
    """
    General-purpose transliteration to Latin script:
    - If Indic characters detected: splits into tokens, detects script, transliterates via Sanscript,
      and applies programmatic schwa deletion.
    - If Latin / French / European text: applies anyascii (fast, clean accent removal).
    """
    if not text:
        return ""
    if text.isascii():
        return text

    # Route 1: Indic Script (Devanagari, Bengali, Gujarati, Tamil, Kannada, Telugu, etc.)
    if RE_INDIC.search(text):
        tokens = text.split()
        return " ".join(transliterate_indic_token(tok) for tok in tokens)

    # Route 2: French / European / Latin-accented text
    if anyascii:
        return anyascii.anyascii(text)
    if unidecode:
        return unidecode.unidecode(text)
    return text


def strip_domain_suffix(text: str) -> str:
    """Strip website / domain extensions like .com, .org, .net, .in, .fr from business names."""
    if not text:
        return ""
    cleaned = RE_DOMAIN_SUFFIX.sub("", text)
    if " " not in cleaned.strip() and len(cleaned.strip()) > 8:
        cleaned = RE_BARE_DOMAIN_SUFFIX.sub("", cleaned)
    return cleaned.strip()


def strip_abbreviation_periods(text: str) -> str:
    """Strip periods inside or immediately following abbreviations."""
    if not text or "." not in text:
        return text
    s = RE_INITIALS.sub(r"\1", text)
    s = RE_INTERIOR_ABBREV_DOT.sub(r"\1 \2", s)
    s = RE_ABBREV_TRAILING_DOT.sub(r"\1", s)
    s = s.rstrip(".")
    return s.strip()


def standardize_symbols(text: str, preserve_slashes: bool = True) -> str:
    """Standardize punctuation and symbols, preserving slashes for DBA/landmark processing."""
    if not text:
        return ""
    s = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r'\s*&\s*', ' and ', s)
    s = re.sub(r'(?<=[a-zA-Z0-9])\s*\+\s*(?=[a-zA-Z0-9])', ' and ', s)

    if not preserve_slashes:
        preserved = RE_PRESERVED_SLASH.findall(s)
        if not preserved:
            s = s.replace("/", " ")
        else:
            parts = s.split()
            new_parts = []
            for p in parts:
                if RE_PRESERVED_SLASH.match(p):
                    new_parts.append(p)
                else:
                    new_parts.append(p.replace("/", " "))
            s = " ".join(new_parts)

    return s


def normalize_digits_and_numbers(text: str) -> str:
    """Normalize number and ordinal patterns in addresses and names."""
    if not text:
        return ""
    # Strip leading zeros off standalone numeric tokens (e.g. 02814 -> 2814, 01689 -> 1689)
    s = re.sub(r'\b0+([1-9]\d*)\b', r'\1', text)
    s = RE_NO_PATTERN.sub(r"\1 no \2", s)
    s = RE_GENERIC_NO.sub(r"no \1", s)
    s = RE_ORDINAL_FLOOR.sub(r"\1 floor", s)
    return s


def deconcatenate_blob(text: str, min_len: int = 12) -> str:
    """Split single unspaced blobs into words using wordninja."""
    if not text or not wordninja:
        return text
    tokens = text.split()
    if len(tokens) == 1 and len(tokens[0]) >= min_len and tokens[0].isalpha():
        split_words = wordninja.split(tokens[0])
        if len(split_words) > 1:
            return " ".join(split_words)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse consecutive whitespace and trim."""
    if not text:
        return ""
    return RE_MULTI_SPACE.sub(" ", text).strip()


def normalize_primitive(
    text: Optional[str],
    is_name: bool = False,
    preserve_slashes: bool = True,
    apply_deconcat: bool = False
) -> str:
    """
    Full Shared Primitive Normalization Pipeline.
    """
    if not text:
        return ""
    
    # 1. Unicode NFKC
    s = normalize_unicode(text)
    
    # 2. Strip decorative junk characters
    s = strip_decorative_noise(s)
    
    # 3. Script-routed Transliteration (Indic -> Sanscript + Schwa Deletion, French -> anyascii)
    s = transliterate_to_latin(s)
    
    # 4. Strip domain suffixes (for business names)
    if is_name:
        s = strip_domain_suffix(s)
        
    # 5. Strip abbreviation periods
    s = strip_abbreviation_periods(s)
    
    # 6. Standardize symbols (preserving / for DBA/landmark processing)
    s = standardize_symbols(s, preserve_slashes=preserve_slashes)
    
    # 7. Normalize digits and numbers
    s = normalize_digits_and_numbers(s)
    
    # 8. Case fold
    s = s.lower()
    
    # 9. Optional de-concatenation
    if apply_deconcat:
        s = deconcatenate_blob(s)
        
    # 10. Final whitespace collapse
    s = collapse_whitespace(s)
    
    return s
