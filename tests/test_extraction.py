from src.ingestion.pdf_parser import parse_pdf
from src.extraction.rules import extract_from_parsed_document


def test_extract_invoice_with_po_relation():
    pdf = "data/synthetic/documents/SUP-001_Alpha_Services_SAS/invoice_FAC-2026-001.pdf"
    doc = parse_pdf(pdf)
    res = extract_from_parsed_document(doc)

    assert len(res.entities) == 1
    inv = res.entities[0]
    assert inv.id == "FAC-2026-001"
    assert inv.properties["amount_ht"] == 10000.0
    assert inv.properties["amount_ttc"] == 12000.0

    # Cross-document relation
    assert len(res.relations) == 1
    rel = res.relations[0]
    assert rel.source_id == "FAC-2026-001"
    assert rel.relation_type == "REFERS_TO"
    assert rel.target_id == "BC-2026-001"


def test_extract_orphan_invoice_has_no_po_relation():
    pdf = "data/synthetic/documents/SUP-004_Delta_Consulting/invoice_FAC-2026-004.pdf"
    doc = parse_pdf(pdf)
    res = extract_from_parsed_document(doc)

    assert len(res.entities) == 1
    assert res.entities[0].id == "FAC-2026-004"
    assert len(res.relations) == 0  # Missing PO relation


def test_extract_amendment_modifies_contract():
    pdf = "data/synthetic/documents/SUP-003_Gamma_Cloud_SAS/amendment_AVN-2026-001.pdf"
    doc = parse_pdf(pdf)
    res = extract_from_parsed_document(doc)

    assert len(res.entities) == 1
    amend = res.entities[0]
    assert amend.id == "AVN-2026-001"
    assert amend.properties["new_maximum_amount"] == 40000.0

    assert len(res.relations) == 1
    rel = res.relations[0]
    assert rel.relation_type == "MODIFIES"
    assert rel.target_id == "CON-003"


def test_parse_currency_formats():
    from src.extraction.rules import parse_currency
    assert parse_currency("1 250,50 €") == 1250.5
    assert parse_currency("10 000,00 €") == 10000.0
    assert parse_currency("10 000.00 €") == 10000.0
    assert parse_currency("1,250.50 €") == 1250.5
    assert parse_currency("1.250,50 €") == 1250.5
    assert parse_currency("7500 €") == 7500.0
    assert parse_currency("10\xa0000,50 €") == 10000.5

