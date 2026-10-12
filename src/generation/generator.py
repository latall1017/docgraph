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
    findings: List[Finding] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'analyse de documents commerciaux et l'audit de facturation.
Tu dois répondre exclusivement à partir des informations fournies dans le contexte.

Règles strictes :
1. N'invente aucune information.
2. Si une information est absente, indique-le explicitement dans le champ limitations.
3. Distingue les faits vérifiés des hypothèses.
4. Cite précisément les documents et les pages utilisés.
5. Ne considère pas une anomalie comme une fraude sans preuve suffisante.
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
            doc_passages.append(f"- [{ev.get('file_name', 'document')}] : {ev.get('content')}")
        passages_text = "\n".join(doc_passages) if doc_passages else "Aucun passage spécifique."

        # 2. Format graph audit context
        audit = retrieval_result.get("graph_audit")
        if audit:
            graph_lines = [
                f"- Facture auditée : {audit.get('invoice_id')} (Montant : {audit.get('invoice_amount_ht')} € HT)",
                f"- Bon de commande associé : {audit.get('po_id')} (Montant : {audit.get('po_amount_ht')} € HT)",
                f"- Écart constaté : {audit.get('amount_diff')} € HT (Dépassement : {audit.get('exceeds_po')})",
                f"- Contrat cadre : {audit.get('contract_id')} (Plafond initial : {audit.get('contract_ceiling')} € HT)",
                f"- Avenant modificateur : {audit.get('amendment_id')} (Plafond effectif : {audit.get('effective_ceiling')} € HT)",
                f"- Justifié par avenant : {audit.get('justified_by_amendment')}",
            ]
            graph_text = "\n".join(graph_lines)
        else:
            graph_text = "Aucune relation de graphe retrouvée."

        return f"""{SYSTEM_PROMPT}

Question :
{query}

Contexte documentaire (passages extraits) :
{passages_text}

Relations du Knowledge Graph (données vérifiées) :
{graph_text}

Produis une réponse structurée en respectant scrupuleusement le schéma JSON attendu.
"""

    def generate(self, query: str, retrieval_result: Dict[str, Any]) -> GroundedAnswer:
        """Generates a structured GroundedAnswer via Gemini 2.0 Flash (with deterministic fallback)."""
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
            answer="Audit généré sans appel LLM externe (mode local).",
            findings=[Finding(
                type="amount_difference" if audit.get("exceeds_po") else "compliant",
                invoice_amount=audit.get("invoice_amount_ht"),
                order_amount=audit.get("po_amount_ht"),
                difference=audit.get("amount_diff"),
            )],
            evidence=[],
            limitations=["Clé API Gemini non configurée."]
        )
