from typing import Dict, List, Any,Optional
from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    chunk_id: Optional[str] = None
    id: str
    entity_type: str
    name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    evidence: str
    page_number: int = 1
    confidence: float = 1.0


class ExtractedRelation(BaseModel):
    chunk_id: Optional[str] = None
    source_id: str
    relation_type: str
    target_id: str
    evidence: str
    confidence: float = 1.0


class ExtractionResult(BaseModel):
    document_name: str
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relations: List[ExtractedRelation] = Field(default_factory=list)
