# Orchestrates document extraction in cascade: Tier 1 (deterministic rules) -> Tier 2 (LLM / flexible NER).

from typing import Optional
from src.ingestion.models import ParsedDocument
from src.extraction.schema import ExtractionResult
from src.extraction.rules import extract_from_parsed_document
from src.extraction.llm_extractor import LLMExtractor


class CascadedExtractor:
    """Cascaded document extractor:
    - Tier 1: Fast deterministic rule extraction (0 token, 0 ms) for standard templates.
    - Tier 2: LLM structured extraction (or flexible heuristic fallback) when Tier 1 detects no entity.
    """

    def __init__(self, llm_extractor: Optional[LLMExtractor] = None):
        self.llm_extractor = llm_extractor or LLMExtractor()

    def extract(self, doc: ParsedDocument) -> ExtractionResult:
        """Extracts entities and relations, trying Tier 1 first, falling back to Tier 2 if needed."""
        # =====================================================================
        # Tier 1: Fast deterministic rules
        # =====================================================================
        tier1_result = extract_from_parsed_document(doc)
        if len(tier1_result.entities) > 0:
            return tier1_result

        # =====================================================================
        # Tier 2: LLM structured fallback (or flexible heuristic)
        # =====================================================================
        tier2_result = self.llm_extractor.extract(doc)
        return tier2_result


def extract_document_cascaded(doc: ParsedDocument) -> ExtractionResult:
    """Convenience helper for cascaded extraction."""
    extractor = CascadedExtractor()
    return extractor.extract(doc)
