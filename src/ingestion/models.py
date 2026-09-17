from typing import List, Optional
from pydantic import BaseModel


class DocumentElement(BaseModel):
    element_type: str  # "paragraph", "table", or "header"
    content: str
    page_number: int
    table_data: Optional[List[List[str]]] = None


class ParsedDocument(BaseModel):
    file_name: str
    total_pages: int
    elements: List[DocumentElement] = []
    raw_text: str = ""
