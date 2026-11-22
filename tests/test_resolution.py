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


def test_generic_word_collision_rejected():
    # Companies sharing only generic business words must NOT match
    assert not is_same_company("Alpha Logistics SAS", name_b="Beta Logistics SARL")
    assert not is_same_company("Global Consulting", name_b="European Consulting")
    assert not is_same_company("Total Energies Services", name_b="Beta Services")


def test_knowledge_graph_company_resolution():
    from src.graph.graph_builder import KnowledgeGraph
    from src.extraction.schema import ExtractedEntity

    kg = KnowledgeGraph()
    kg.add_entity(ExtractedEntity(
        id="SUP-001",
        entity_type="Company",
        name="Alpha Services SAS",
        properties={"siren": "111222333"},
        evidence="Alpha Services SAS",
        page_number=1,
    ))

    # Match by same SIREN even with variant name
    match1 = kg.find_company_entity("Alpha France", siren="111222333")
    assert match1 == "SUP-001"

    # Match by normalized name without legal suffix
    match2 = kg.find_company_entity("Alpha Services")
    assert match2 == "SUP-001"

    # Do not match different company
    match3 = kg.find_company_entity("Beta Logistics", siren="444555666")
    assert match3 is None

