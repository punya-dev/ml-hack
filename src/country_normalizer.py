"""
Country Normalization Module.
Normalizes country field labels across sources while supporting open-set unseen countries (e.g., France or future countries).
"""
import json
import os
import re
from typing import Dict, Optional

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

DEFAULT_ALIASES = {
    "us": "US",
    "usa": "US",
    "u.s.": "US",
    "u.s.a.": "US",
    "united states": "US",
    "united states of america": "US",
    "in": "India",
    "ind": "India",
    "india": "India",
    "bharat": "India",
    "fr": "France",
    "fra": "France",
    "france": "France"
}


class CountryNormalizer:
    """
    Normalizes country strings using alias mapping and open-set capitalization.
    """
    def __init__(self, config_path: Optional[str] = None):
        self.aliases: Dict[str, str] = dict(DEFAULT_ALIASES)
        if config_path is None:
            config_path = os.path.join(CONFIG_DIR, "country_aliases.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "aliases" in data:
                        self.aliases.update({k.lower().strip(): v for k, v in data["aliases"].items()})
            except Exception as e:
                # Fallback to default aliases
                pass

    def normalize_country(self, raw_country: Optional[str]) -> str:
        """
        Normalize a country string to its canonical form.
        Supports open-set unseen countries gracefully without dropping or corrupting records.
        """
        if raw_country is None:
            return "UNKNOWN"
        
        s = str(raw_country).strip()
        if not s:
            return "UNKNOWN"
        
        lower_s = s.lower()
        # Direct lookup in alias dictionary
        if lower_s in self.aliases:
            return self.aliases[lower_s]
        
        # Clean punctuation and check again
        cleaned = re.sub(r'[^a-zA-Z\s]', '', lower_s).strip()
        if cleaned in self.aliases:
            return self.aliases[cleaned]
        
        # Open-set fallback: Return cleaned title-cased string
        # If it's a 2 or 3 letter uppercase code, keep uppercase (e.g., "DE", "GB")
        if len(cleaned) in (2, 3) and cleaned.isalpha():
            return cleaned.upper()
        
        # Otherwise title case (e.g. "Germany", "United Kingdom")
        return cleaned.title() if cleaned else s.strip()


# Module-level singleton
_default_normalizer = CountryNormalizer()


def normalize_country(raw_country: Optional[str]) -> str:
    """Convenience functional interface."""
    return _default_normalizer.normalize_country(raw_country)
