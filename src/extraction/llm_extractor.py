# Extracts structured entities and relations from parsed documents using LLM with heuristic fallback.

import os
import re
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from src.extraction.schema import ExtractedEntity, ExtractedRelation, ExtractionResult
from src.ingestion.models import ParsedDocument
from src.extraction.rules import find_element, parse_currency


class LLMDocumentExtraction(BaseModel):
    """Structured extraction format for LLM document analysis."""
    doc_type: Optional[str] = Field(None, description="Invoice, PurchaseOrder, Contract, Amendment, or Other")
    identifier: Optional[str] = Field(None, description="Document reference ID (e.g. INV-999, PO-888, CTR-777, AVN-123)")
    counterparty_name: Optional[str] = Field(None, description="Name of company or supplier issuing/participating")
    amount_ht: Optional[float] = Field(None, description="Pre-tax or subtotal amount")
    amount_ttc: Optional[float] = Field(None, description="Total amount including tax")
    maximum_amount: Optional[float] = Field(None, description="Contract initial ceiling budget")
    new_maximum_amount: Optional[float] = Field(None, description="Amended contract ceiling budget")
    referenced_pos: List[str] = Field(default_factory=list, description="Referenced Purchase Order identifiers")
    referenced_contracts: List[str] = Field(default_factory=list, description="Referenced Contract identifiers")
    target_contract_modified: Optional[str] = Field(None, description="Target contract modified by this amendment")
    key_evidence: Optional[str] = Field(None, description="Exact text excerpt supporting the extraction")


EXTRACTION_PROMPT = """You are an expert document parser for a legal and financial audit Knowledge Graph.
Extract key procurement entities and relationships from the provided document text.

Document Text:
---
{text}
---

Rules:
1. Identify the primary document type (Invoice, PurchaseOrder, Contract, Amendment, or Other).
2. Extract the exact document identifier (reference number).
3. Extract any monetary amounts (HT, TTC, maximum ceiling, new ceiling).
4. Extract any referenced related documents (PO references, governing contracts).
5. If the document is an amendment, extract the target contract being modified.
6. Provide an exact short text excerpt as key_evidence.
"""


class LLMExtractor:
    """Tier 2 extractor: Uses Gemini Flash with structured schema, falling back to flexible heuristic NER."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.client = None
        if self.api_key and genai is not None:
            try:
                self.client = genai.Client(
                    api_key=self.api_key,
                    http_options={"api_version": "v1beta"},
                )
            except Exception:
                self.client = None

    def extract(self, doc: ParsedDocument) -> ExtractionResult:
        """Extracts entities and relations from a parsed document using LLM or heuristic fallback."""
        result = ExtractionResult(document_name=doc.file_name)
        text = doc.raw_text.strip()
        if not text:
            return result

        extracted_data: Optional[LLMDocumentExtraction] = None

        # 1. Try LLM extraction if client is available
        if self.client:
            extracted_data = self._extract_with_llm(text)

        # 2. Heuristic fallback if LLM failed, unconfigured, or returned empty
        if not extracted_data or not extracted_data.identifier:
            extracted_data = self._extract_with_heuristic(text)

        if not extracted_data or not extracted_data.identifier:
            return result

        # Convert LLMDocumentExtraction into Knowledge Graph ExtractedEntity and ExtractedRelation
        evidence_text = extracted_data.key_evidence or extracted_data.identifier
        el = find_element(doc, evidence_text)
        page_num = el.page_number if el else 1
        chunk_id = el.chunk_id if el else None

        doc_id = extracted_data.identifier.upper()
        doc_type = extracted_data.doc_type or "Unknown"

        properties: Dict[str, Any] = {}
        if extracted_data.amount_ht is not None:
            properties["amount_ht"] = extracted_data.amount_ht
        if extracted_data.amount_ttc is not None:
            properties["amount_ttc"] = extracted_data.amount_ttc
        if extracted_data.maximum_amount is not None:
            properties["maximum_amount"] = extracted_data.maximum_amount
        if extracted_data.new_maximum_amount is not None:
            properties["new_maximum_amount"] = extracted_data.new_maximum_amount

        result.entities.append(
            ExtractedEntity(
                id=doc_id,
                entity_type=doc_type,
                name=f"{doc_type} {doc_id}",
                properties=properties,
                evidence=evidence_text,
                page_number=page_num,
                chunk_id=chunk_id,
            )
        )

        # Counterparty / Company entity
        if extracted_data.counterparty_name:
            comp_id = f"COMP-{doc_id}"
            result.entities.append(
                ExtractedEntity(
                    id=comp_id,
                    entity_type="Company",
                    name=extracted_data.counterparty_name,
                    properties={},
                    evidence=evidence_text,
                    page_number=page_num,
                    chunk_id=chunk_id,
                )
            )
            result.relations.append(
                ExtractedRelation(
                    source_id=doc_id,
                    relation_type="ISSUED_BY",
                    target_id=comp_id,
                    evidence=evidence_text,
                    chunk_id=chunk_id,
                )
            )

        # Referenced Purchase Orders (Invoice -> PO)
        for po_ref in extracted_data.referenced_pos:
            clean_po = po_ref.upper().strip()
            if clean_po:
                result.relations.append(
                    ExtractedRelation(
                        source_id=doc_id,
                        relation_type="REFERS_TO",
                        target_id=clean_po,
                        evidence=evidence_text,
                        chunk_id=chunk_id,
                    )
                )

        # Referenced Contracts (Invoice / PO -> Contract)
        for con_ref in extracted_data.referenced_contracts:
            clean_con = con_ref.upper().strip()
            if clean_con:
                result.relations.append(
                    ExtractedRelation(
                        source_id=doc_id,
                        relation_type="GOVERNED_BY",
                        target_id=clean_con,
                        evidence=evidence_text,
                        chunk_id=chunk_id,
                    )
                )

        # Amendment modifying target contract
        if extracted_data.target_contract_modified:
            clean_target = extracted_data.target_contract_modified.upper().strip()
            if clean_target:
                result.relations.append(
                    ExtractedRelation(
                        source_id=doc_id,
                        relation_type="MODIFIES",
                        target_id=clean_target,
                        evidence=evidence_text,
                        chunk_id=chunk_id,
                    )
                )

        return result

    def _extract_with_llm(self, text: str) -> Optional[LLMDocumentExtraction]:
        """Calls Gemini Flash with structured schema to parse document text."""
        # Truncate text to avoid excessive tokens while capturing headers and tables
        sample_text = text[:4000]
        prompt = EXTRACTION_PROMPT.format(text=sample_text)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=LLMDocumentExtraction,
                    temperature=0.0,
                ),
            )
            raw = response.text.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            return LLMDocumentExtraction.model_validate_json(raw.strip())
        except Exception:
            return None

    def _extract_with_heuristic(self, text: str) -> LLMDocumentExtraction:
        """Flexible regex and token-based fallback for non-standard or foreign document formats."""
        # 1. Invoice
        inv_match = re.search(
            r"\b(?:invoice|bill|statement|facture|note\s+d'honoraires)(?:\s+(?:no\.?|#|number|n°|numéro|num|réf\.?|ref\.?))?\s*[:#]?\s*([A-Za-z0-9_-]+)",
            text,
            re.IGNORECASE,
        )
        if inv_match:
            inv_id = inv_match.group(1).upper()
            amt_match = re.search(
                r"(?:subtotal|total\s+net|total\s+excl\.?\s*vat|total\s+ht|montant\s+ht|amount)\s*[:]?\s*([\d\s,.]+)\s*(?:€|EUR|\$|USD)?",
                text,
                re.IGNORECASE,
            )
            po_match = re.search(
                r"(?:purchase\s+order(?:\s+ref\.?)?|po\s*ref\.?|order\s+ref\.?|bon\s+de\s+commande(?:\s+associé)?)\s*[:]?\s*([A-Za-z0-9_-]+)",
                text,
                re.IGNORECASE,
            )
            con_match = re.search(
                r"(?:governing\s+agreement|governing\s+contract|contract|agreement|convention)\s*[:]?\s*([A-Za-z0-9_-]+)",
                text,
                re.IGNORECASE,
            )
            return LLMDocumentExtraction(
                doc_type="Invoice",
                identifier=inv_id,
                amount_ht=parse_currency(amt_match.group(1)) if amt_match else None,
                referenced_pos=[po_match.group(1)] if po_match else [],
                referenced_contracts=[con_match.group(1)] if con_match else [],
                key_evidence=inv_match.group(0),
            )

        # 2. Purchase Order
        po_match = re.search(
            r"\b(?:purchase\s+order|order\s+confirmation|order\s+form|bon\s+de\s+commande)(?:\s+(?:no\.?|#|n°|number|ref\.?))?\s*[:#]?\s*([A-Za-z0-9_-]+)",
            text,
            re.IGNORECASE,
        )
        if po_match:
            po_id = po_match.group(1).upper()
            amt_match = re.search(
                r"(?:total|amount|montant\s+total|total\s+ht)\s*[:]?\s*([\d\s,.]+)\s*(?:€|EUR|\$|USD)?",
                text,
                re.IGNORECASE,
            )
            con_match = re.search(
                r"(?:under\s+contract|governing\s+contract|contrat\s+cadre)\s*[:]?\s*([A-Za-z0-9_-]+)",
                text,
                re.IGNORECASE,
            )
            return LLMDocumentExtraction(
                doc_type="PurchaseOrder",
                identifier=po_id,
                amount_ht=parse_currency(amt_match.group(1)) if amt_match else None,
                referenced_contracts=[con_match.group(1)] if con_match else [],
                key_evidence=po_match.group(0),
            )

        # 3. Contract
        con_match = re.search(
            r"\b(?:master\s+services\s+agreement|services\s+agreement|framework\s+contract|contract|contrat)(?:\s+(?:no\.?|#|ref\.?|n°|number))?\s*[:#]?\s*([A-Za-z0-9_-]+)",
            text,
            re.IGNORECASE,
        )
        if con_match:
            con_id = con_match.group(1).upper()
            max_match = re.search(
                r"(?:maximum\s+(?:budget\s+)?(?:ceiling|amount)|budget\s+ceiling|ceiling|budget|plafond\s+maximum)\s*[:]?\s*([\d\s,.]+)\s*(?:€|EUR|\$|USD)?",
                text,
                re.IGNORECASE,
            )
            return LLMDocumentExtraction(
                doc_type="Contract",
                identifier=con_id,
                maximum_amount=parse_currency(max_match.group(1)) if max_match else None,
                key_evidence=con_match.group(0),
            )

        # 4. Amendment
        amend_match = re.search(
            r"\b(?:amendment|addendum|avenant)(?:\s+(?:no\.?|#|n°|number|ref\.?))?\s*[:#]?\s*([A-Za-z0-9_-]+)",
            text,
            re.IGNORECASE,
        )
        if amend_match:
            amend_id = amend_match.group(1).upper()
            new_max_match = re.search(
                r"(?:new\s+(?:maximum\s+)?ceiling|nouveau\s+plafond)\s*[:]?\s*([\d\s,.]+)\s*(?:€|EUR|\$|USD)?",
                text,
                re.IGNORECASE,
            )
            target_match = re.search(
                r"(?:modifies\s+contract|amends\s+agreement|modifie\s+le\s+contrat)\s*[:]?\s*([A-Za-z0-9_-]+)",
                text,
                re.IGNORECASE,
            )
            return LLMDocumentExtraction(
                doc_type="Amendment",
                identifier=amend_id,
                new_maximum_amount=parse_currency(new_max_match.group(1)) if new_max_match else None,
                target_contract_modified=target_match.group(1) if target_match else None,
                key_evidence=amend_match.group(0),
            )

        return LLMDocumentExtraction()
