# Generates structured and grounded audit answers using Gemini Flash.

import os
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


class Finding(BaseModel):
    type: str
    invoice_amount: Optional[float] = None
    order_amount: Optional[float] = None
    difference: Optional[float] = None
    currency: str = "EUR"


class Evidence(BaseModel):
    document_id: str
    page: int = 1
    claim: str


class GroundedAnswer(BaseModel):
    answer: str
    findings: List[Finding] = Field(default_factory=list, description="Findings related to the query")
    evidence: List[Evidence] = Field(default_factory=list, description="Evidence supporting the findings")
    limitations: List[str] = Field(default_factory=list, description="Limitations of the findings")


SYSTEM_PROMPT = """You are an assistant specialized in business document analysis and invoice auditing.
You must answer exclusively based on the provided context.

Strict rules:
1. Do not invent any information.
2. If an information is missing, explicitly state it in the limitations field.
3. Distinguish verified facts from assumptions.
4. Precisely cite used documents and pages.
5. Do not treat an anomaly as fraud without sufficient evidence.
"""


class AnswerGenerator:
    """Generates grounded, auditable answers using Gemini Flash."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.client = None
        if self.api_key and genai is not None:
            self.client = genai.Client(
                api_key=self.api_key,
                http_options={"api_version": "v1beta"}
            )

    def build_prompt(self, query: str, retrieval_result: Dict[str, Any]) -> str:
        """Formats the prompt combining query, vector passages, and graph relations."""
        # 1. Format vector passages
        doc_passages = []
        for ev in retrieval_result.get("ranked_evidence", []):
            doc_passages.append(f"- [{ev.get('file_name', 'document')}]: {ev.get('content')}")
        passages_text = "\n".join(doc_passages) if doc_passages else "No specific passages found."

        # 2. Format graph audit context
        audit = retrieval_result.get("graph_audit")
        if audit:
            entity_type = audit.get("entity_type")
            if entity_type == "PurchaseOrder":
                graph_lines = [
                    f"- Audited Purchase Order: {audit.get('po_id')} (Amount: {audit.get('po_amount_ht')} EUR)",
                    f"- Governing Contract: {audit.get('contract_id')} (Effective Ceiling: {audit.get('effective_ceiling')} EUR)",
                    f"- Invoices Referencing PO: {audit.get('invoices')} (Total Invoiced: {audit.get('total_invoiced_ht')} EUR)",
                    f"- Exceeds PO: {audit.get('exceeds_po')} (Difference: {audit.get('amount_diff')} EUR)",
                    f"- Exceeds Contract Ceiling: {audit.get('exceeds_contract')}",
                ]
            elif entity_type == "Contract":
                graph_lines = [
                    f"- Audited Contract: {audit.get('contract_id')} (Initial Ceiling: {audit.get('initial_ceiling')} EUR, Effective: {audit.get('effective_ceiling')} EUR)",
                    f"- Applied Amendments: {audit.get('applied_amendments')}",
                    f"- Purchase Orders: {audit.get('purchase_orders')} (Total Committed: {audit.get('total_po_committed')} EUR)",
                    f"- Linked Invoices: {audit.get('invoices')} (Total Invoiced: {audit.get('total_invoiced')} EUR)",
                    f"- Remaining Budget: {audit.get('remaining_budget')} EUR (Exceeds Ceiling: {audit.get('exceeds_ceiling')})",
                ]
            else:
                graph_lines = [
                    f"- Audited Invoice: {audit.get('invoice_id')} (Amount: {audit.get('invoice_amount_ht')} EUR)",
                    f"- Associated Purchase Order: {audit.get('po_id')} (Amount: {audit.get('po_amount_ht')} EUR)",
                    f"- Amount Discrepancy: {audit.get('amount_diff')} EUR (Exceeds PO: {audit.get('exceeds_po')})",
                    f"- Framework Contract: {audit.get('contract_id')} (Initial Ceiling: {audit.get('contract_ceiling')} EUR)",
                    f"- Modifying Amendment: {audit.get('amendment_id')} (Effective Ceiling: {audit.get('effective_ceiling')} EUR)",
                    f"- Cumulative Contract Spend: {audit.get('cumulative_contract_amount')} EUR (Exceeds Cumulative Ceiling: {audit.get('exceeds_contract_cumulative')})",
                    f"- Justified by Amendment: {audit.get('justified_by_amendment')}",
                ]
            graph_text = "\n".join(graph_lines)
        else:
            graph_text = "No graph relations found."

        return f"""{SYSTEM_PROMPT}

Question:
{query}

Documentary Context (extracted passages):
{passages_text}

Knowledge Graph Relations (verified data):
{graph_text}

Produce a structured response strictly following the expected JSON schema.
"""

    def generate(self, query: str, retrieval_result: Dict[str, Any]) -> GroundedAnswer:
        """Generates a structured GroundedAnswer via Gemini Flash."""
        prompt = self.build_prompt(query, retrieval_result)

        if self.client:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GroundedAnswer,
                    temperature=0.0,
                ),
            )
            return GroundedAnswer.model_validate_json(response.text)

        # Fallback if no API key is available
        audit = retrieval_result.get("graph_audit") or {}
        return GroundedAnswer(
            answer="Audit generated without external LLM call (local mode).",
            findings=[Finding(
                type="amount_difference" if audit.get("exceeds_po") else "compliant",
                invoice_amount=audit.get("invoice_amount_ht"),
                order_amount=audit.get("po_amount_ht"),
                difference=audit.get("amount_diff"),
            )],
            evidence=[],
            limitations=["Gemini API key not configured."]
        )
