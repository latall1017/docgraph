from src.resolution.normalizer import normalize_company_name, is_same_company


def test_normalize_company_name():
    assert normalize_company_name("Alpha Services SAS") == "alpha services"
    assert normalize_company_name("BETA LOGISTICS SARL") == "beta logistics"
    assert normalize_company_name("Gamma Cloud, Inc.") == "gamma cloud"


def test_is_same_company_via_siren():
    # Different spelling but same SIREN -> True
    assert is_same_company(
        name_a="Alpha France",
        siren_a="111222333",
        name_b="Alpha Services SAS",
        siren_b="111222333",
    )


def test_is_same_company_via_normalized_name():
    # No SIREN but identical after stripping legal forms -> True
    assert is_same_company(
        name_a="Beta Logistics SAS",
        name_b="Beta Logistics SARL",
    )


def test_is_same_company_via_fuzzy_matching():
    # Minor typos without SIREN -> RapidFuzz matches >= 85%
    assert is_same_company(
        name_a="Alpha Servises SAS",
        name_b="Alpha Services SAS",
    )


def test_different_companies_do_not_match():
    # Completely different companies -> False
    assert not is_same_company(
        name_a="Alpha Services SAS",
        siren_a="111222333",
        name_b="Beta Logistics SARL",
        siren_b="444555666",
    )
