import re
from typing import Optional

LEGAL_SUFFIXES = [
    r"\bsas\b",
    r"\bsarl\b",
    r"\bsa\b",
    r"\beurl\b",
    r"\binc\b",
    r"\bltd\b",
    r"\bllc\b",
]


def normalize_company_name(name: str) -> str:
    """Normalizes a company name by lowercasing, removing legal suffixes, and trimming punctuation."""
    cleaned = name.lower()

    # Remove legal suffixes
    for suffix in LEGAL_SUFFIXES:
        cleaned = re.sub(suffix, "", cleaned)

    # Remove punctuation and multiple spaces
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_same_company(
    name_a: str,
    siren_a: Optional[str] = None,
    name_b: str = "",
    siren_b: Optional[str] = None,
) -> bool:
    """
    Determines whether two company mentions point to the same entity.
    Priority 1: SIREN match (hard identifier).
    Priority 2: Normalized name match.
    """
    # 1. SIREN match
    if siren_a and siren_b:
        clean_siren_a = re.sub(r"\D", "", siren_a)
        clean_siren_b = re.sub(r"\D", "", siren_b)
        if len(clean_siren_a) == 9 and clean_siren_a == clean_siren_b:
            return True

    # 2. Normalized name match
    norm_a = normalize_company_name(name_a)
    norm_b = normalize_company_name(name_b)

    if norm_a and norm_b and norm_a == norm_b:
        return True

    # 3. Substring inclusion if names are long enough (>= 6 chars)
    if len(norm_a) >= 6 and len(norm_b) >= 6:
        if norm_a in norm_b or norm_b in norm_a:
            return True

    return False
