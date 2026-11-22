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

    # Traceability checks
    assert inv.chunk_id is not None
    assert "invoice_FAC-2026-001.pdf_p1_" in inv.chunk_id
    assert inv.page_number == 1
    assert rel.chunk_id is not None


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


def test_extract_case_insensitivity_and_english():
    from src.ingestion.models import ParsedDocument, DocumentElement
    doc = ParsedDocument(
        file_name="invoice_en.pdf",
        total_pages=1,
        elements=[
            DocumentElement(
                chunk_id="invoice_en.pdf_p1_e0",
                element_type="paragraph",
                content="Invoice INV-2026-999\nPurchase Order: PO-2026-888\nSubtotal: 5,400.50 €\nTotal: 6,480.60 €",
                page_number=1,
            )
        ],
        raw_text="Invoice INV-2026-999\nPurchase Order: PO-2026-888\nSubtotal: 5,400.50 €\nTotal: 6,480.60 €",
    )
    res = extract_from_parsed_document(doc)
    assert len(res.entities) == 1
    assert res.entities[0].id == "INV-2026-999"
    assert res.entities[0].properties["amount_ht"] == 5400.5
    assert len(res.relations) == 1
    assert res.relations[0].target_id == "PO-2026-888"


def test_extract_multi_entity_in_single_document():
    from src.ingestion.models import ParsedDocument, DocumentElement
    text = (
        "CONTRAT DE PRESTATION\nRéférence : CTR-BUNDLE-01\nplafond global de 100 000 €\n\n"
        "AVENANT N° AVN-BUNDLE-01\nContrat concerné : CTR-BUNDLE-01\nnouveau montant de 120 000 €"
    )
    doc = ParsedDocument(
        file_name="bundle.pdf",
        total_pages=2,
        elements=[
            DocumentElement(
                chunk_id="bundle.pdf_p1_e0",
                element_type="paragraph",
                content="CONTRAT DE PRESTATION\nRéférence : CTR-BUNDLE-01\nplafond global de 100 000 €",
                page_number=1,
            ),
            DocumentElement(
                chunk_id="bundle.pdf_p2_e0",
                element_type="paragraph",
                content="AVENANT N° AVN-BUNDLE-01\nContrat concerné : CTR-BUNDLE-01\nnouveau montant de 120 000 €",
                page_number=2,
            ),
        ],
        raw_text=text,
    )
    res = extract_from_parsed_document(doc)
    # Both Contract AND Amendment must be extracted (no elif drop)
    entity_ids = [e.id for e in res.entities]
    assert "CTR-BUNDLE-01" in entity_ids
    assert "AVN-BUNDLE-01" in entity_ids
    assert len(res.entities) == 2
    assert len(res.relations) == 1
    assert res.relations[0].relation_type == "MODIFIES"
    assert res.relations[0].source_id == "AVN-BUNDLE-01"
    assert res.relations[0].target_id == "CTR-BUNDLE-01"


def test_extract_multiple_pos_in_invoice():
    from src.ingestion.models import ParsedDocument, DocumentElement
    doc = ParsedDocument(
        file_name="multi_po_inv.pdf",
        total_pages=1,
        elements=[
            DocumentElement(
                chunk_id="multi_po_inv.pdf_p1_e0",
                element_type="paragraph",
                content="FACTURE FAC-MULTI\nTotal HT : 8 000 €\nBons de commande associés : BC-2026-001, BC-2026-002",
                page_number=1,
            )
        ],
        raw_text="FACTURE FAC-MULTI\nTotal HT : 8 000 €\nBons de commande associés : BC-2026-001, BC-2026-002",
    )
    res = extract_from_parsed_document(doc)
    target_pos = [r.target_id for r in res.relations if r.relation_type == "REFERS_TO"]
    assert "BC-2026-001" in target_pos
    assert "BC-2026-002" in target_pos
    assert len(target_pos) == 2


def test_extract_page_number_provenance():
    from src.ingestion.models import ParsedDocument, DocumentElement
    doc = ParsedDocument(
        file_name="contract_multipage.pdf",
        total_pages=5,
        elements=[
            DocumentElement(
                chunk_id="contract_multipage.pdf_p1_e0",
                element_type="paragraph",
                content="Page 1: Préambule et définitions générales.",
                page_number=1,
            ),
            DocumentElement(
                chunk_id="contract_multipage.pdf_p4_e1",
                element_type="paragraph",
                content="CONTRAT DE PRESTATION DE SERVICES\nRéférence : CTR-PAGE4\nplafond global de 75 000 €",
                page_number=4,
            ),
        ],
        raw_text="Page 1: Préambule et définitions générales.\n\nCONTRAT DE PRESTATION DE SERVICES\nRéférence : CTR-PAGE4\nplafond global de 75 000 €",
    )
    res = extract_from_parsed_document(doc)
    assert len(res.entities) == 1
    # Page number must be 4, NOT hardcoded 1!
    assert res.entities[0].page_number == 4
    assert res.entities[0].chunk_id == "contract_multipage.pdf_p4_e1"


