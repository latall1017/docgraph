import os
from src.ingestion.pdf_parser import parse_pdf


def test_parse_invoice_pdf():
    pdf_path = "data/synthetic/documents/SUP-001_Alpha_Services_SAS/invoice_FAC-2026-001.pdf"
    assert os.path.exists(pdf_path)

    parsed = parse_pdf(pdf_path)
    assert parsed.file_name == "invoice_FAC-2026-001.pdf"
    assert parsed.total_pages == 1
    assert "Alpha Services SAS" in parsed.raw_text
    assert "BC-2026-001" in parsed.raw_text
    assert "Total TTC" in parsed.raw_text

    # Check that elements were extracted
    tables = [e for e in parsed.elements if e.element_type == "table"]
    assert len(tables) >= 1
    header_row = tables[0].table_data[0]
    assert "Description" in header_row[0]


def test_parse_contract_pdf():
    pdf_path = "data/synthetic/documents/SUP-001_Alpha_Services_SAS/contract_CTR-2026-ALPHA.pdf"
    assert os.path.exists(pdf_path)

    parsed = parse_pdf(pdf_path)
    assert parsed.file_name == "contract_CTR-2026-ALPHA.pdf"
    assert "CTR-2026-ALPHA" in parsed.raw_text
    assert "50,000.00" in parsed.raw_text
