import pytest
from src.ingestion.models import ParsedDocument, DocumentElement
from src.extraction.cascade_extractor import CascadedExtractor, extract_document_cascaded
from src.extraction.llm_extractor import LLMExtractor


def test_cascaded_extractor_tier1_fast_path():
    """Documents matching standard rules should be resolved by Tier 1 immediately."""
    doc = ParsedDocument(
        file_name="invoice_FAC-2026-001.pdf",
        total_pages=1,
        raw_text="FACTURE N° FAC-2026-001\nTotal HT : 10 000,00 €\nTotal TTC : 12 000,00 €\nBon de commande associé : BC-2026-001",
        elements=[
            DocumentElement(
                element_type="text",
                content="FACTURE N° FAC-2026-001\nTotal HT : 10 000,00 €",
                page_number=1,
                chunk_id="chunk_1",
            ),
            DocumentElement(
                element_type="text",
                content="Bon de commande associé : BC-2026-001",
                page_number=1,
                chunk_id="chunk_2",
            ),
        ],
    )

    extractor = CascadedExtractor()
    res = extractor.extract(doc)

    assert len(res.entities) >= 1
    inv = next(e for e in res.entities if e.id == "FAC-2026-001")
    assert inv.entity_type == "Invoice"
    assert inv.properties["amount_ht"] == 10000.0
    assert len(res.relations) >= 1
    assert res.relations[0].target_id == "BC-2026-001"


def test_cascaded_extractor_tier2_english_invoice_fallback():
    """English invoices not matching French regex should be successfully extracted by Tier 2."""
    doc = ParsedDocument(
        file_name="invoice_international_INV-999.pdf",
        total_pages=1,
        raw_text="INVOICE NUMBER: INV-2026-GLOBAL-999\nSubtotal: 4,500.00 USD\nPurchase Order Ref: PO-INTL-888\nGoverning Agreement: CTR-GLOBAL-01",
        elements=[
            DocumentElement(
                element_type="text",
                content="INVOICE NUMBER: INV-2026-GLOBAL-999\nSubtotal: 4,500.00 USD\nPurchase Order Ref: PO-INTL-888",
                page_number=1,
                chunk_id="chunk_intl_1",
            ),
        ],
    )

    # Directly test Tier 2 LLM extractor with heuristic fallback
    extractor = CascadedExtractor(llm_extractor=LLMExtractor(api_key=None))
    res = extractor.extract(doc)

    assert len(res.entities) >= 1
    inv = next(e for e in res.entities if e.entity_type == "Invoice")
    assert "INV-2026-GLOBAL-999" in inv.id
    assert inv.properties["amount_ht"] == 4500.0

    # Verify cross-document relations extracted
    assert any(r.relation_type == "REFERS_TO" and "PO-INTL-888" in r.target_id for r in res.relations)
    assert any(r.relation_type == "GOVERNED_BY" and "CTR-GLOBAL-01" in r.target_id for r in res.relations)


def test_cascaded_extractor_tier2_english_agreement_fallback():
    """English agreements with ceilings should be extracted by Tier 2."""
    doc = ParsedDocument(
        file_name="agreement_msa_01.pdf",
        total_pages=1,
        raw_text="MASTER SERVICES AGREEMENT ref: MSA-2026-CORP\nMaximum budget ceiling: 150,000.00 USD\nParties: Global Tech Solutions",
        elements=[
            DocumentElement(
                element_type="text",
                content="MASTER SERVICES AGREEMENT ref: MSA-2026-CORP\nMaximum budget ceiling: 150,000.00 USD",
                page_number=1,
                chunk_id="chunk_msa_1",
            ),
        ],
    )

    extractor = CascadedExtractor(llm_extractor=LLMExtractor(api_key=None))
    res = extractor.extract(doc)

    assert len(res.entities) >= 1
    con = next(e for e in res.entities if e.entity_type == "Contract")
    assert "MSA-2026-CORP" in con.id
    assert con.properties["maximum_amount"] == 150000.0


def test_cascaded_extractor_tier2_amendment_fallback():
    """English amendments with modified ceilings should link to target contract."""
    doc = ParsedDocument(
        file_name="addendum_01.pdf",
        total_pages=1,
        raw_text="ADDENDUM No. AMD-2026-001\nAmends agreement: MSA-2026-CORP\nNew maximum ceiling: 200,000.00 USD",
        elements=[
            DocumentElement(
                element_type="text",
                content="ADDENDUM No. AMD-2026-001\nAmends agreement: MSA-2026-CORP\nNew maximum ceiling: 200,000.00 USD",
                page_number=1,
                chunk_id="chunk_amd_1",
            ),
        ],
    )

    extractor = CascadedExtractor(llm_extractor=LLMExtractor(api_key=None))
    res = extractor.extract(doc)

    assert len(res.entities) >= 1
    amd = next(e for e in res.entities if e.entity_type == "Amendment")
    assert "AMD-2026-001" in amd.id
    assert amd.properties["new_maximum_amount"] == 200000.0

    # Verify relation MODIFIES target contract
    mod_rel = next(r for r in res.relations if r.relation_type == "MODIFIES")
    assert "MSA-2026-CORP" in mod_rel.target_id
