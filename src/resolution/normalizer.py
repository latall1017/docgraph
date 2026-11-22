# Entity resolution and normalization for company mentions using exact and fuzzy matching.

import re
from typing import Optional
from rapidfuzz import fuzz

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


GENERIC_WORDS = {
    "services",
    "logistics",
    "consulting",
    "solutions",
    "technologies",
    "cloud",
    "group",
    "groupe",
    "france",
    "international",
    "holding",
    "transport",
    "conseil",
    "systemes",
    "systems",
}


def is_same_company(
    name_a: str,
    siren_a: Optional[str] = None,
    name_b: str = "",
    siren_b: Optional[str] = None,
) -> bool:
    """
    Determines whether two company mentions point to the same entity.
    Priority 1: SIREN match (hard identifier).
    Priority 2: Normalized name exact match.
    Priority 3: Fuzzy matching with RapidFuzz for minor typos.
    Priority 4: Safe token set matching (rejecting generic-word false positives).
    """
    # 1. SIREN match (strict 9 digits)
    if siren_a and siren_b:
        clean_siren_a = re.sub(r"\D", "", siren_a)
        clean_siren_b = re.sub(r"\D", "", siren_b)
        if len(clean_siren_a) == 9 and clean_siren_a == clean_siren_b:
            return True

    # 2. Normalized exact match
    norm_a = normalize_company_name(name_a)
    norm_b = normalize_company_name(name_b)

    if norm_a and norm_b and norm_a == norm_b:
        return True

    # 3. Fuzzy matching with RapidFuzz for typos and minor variations
    if norm_a and norm_b:
        similarity = fuzz.token_sort_ratio(norm_a, norm_b)
        if similarity >= 85.0:
            return True

    # 4. Token set matching (safe alternative to raw substring inclusion)
    if norm_a and norm_b:
        tokens_a = set(norm_a.split())
        tokens_b = set(norm_b.split())
        shared_tokens = tokens_a & tokens_b
        # Ensure shared tokens are not solely composed of generic business words
        distinctive_shared = shared_tokens - GENERIC_WORDS
        if distinctive_shared:
            set_ratio = fuzz.token_set_ratio(norm_a, norm_b)
            if set_ratio >= 90.0:
                return True

    return False
