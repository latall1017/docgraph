# Parses PDF documents into paragraphs and structured table elements using PyMuPDF.

from pathlib import Path
import pymupdf
from src.ingestion.models import DocumentElement, ParsedDocument


def parse_pdf(file_path: str) -> ParsedDocument:
    """Extracts text paragraphs and structured tables from a PDF."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = pymupdf.open(file_path)
    elements = []
    full_text_parts = []

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]

        # 1. Extract tables
        tables = page.find_tables()
        for tab in tables:
            data = tab.extract()
            if data:
                elements.append(
                    DocumentElement(
                        chunk_id=f"{path.name}_p{page_num}_e{len(elements)}",
                        element_type="table",
                        content="\n".join([" | ".join([str(c or "") for c in row]) for row in data]),
                        page_number=page_num,
                        table_data=data,
                    )
                )

        # 2. Extract text blocks (paragraphs)
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if text:
                elements.append(
                    DocumentElement(
                        chunk_id=f"{path.name}_p{page_num}_e{len(elements)}",
                        element_type="paragraph",
                        content=text,
                        page_number=page_num,
                    )
                )
                full_text_parts.append(text)

    return ParsedDocument(
        file_name=path.name,
        total_pages=len(doc),
        elements=elements,
        raw_text="\n\n".join(full_text_parts),
    )
