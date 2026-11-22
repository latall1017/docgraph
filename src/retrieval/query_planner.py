# Plans and routes queries to deterministic graph audit, dense vector search, or hybrid retrieval.

import os
import re
import json
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from src.resolution.normalizer import is_same_company, normalize_company_name


class RouteType(str, Enum):
    GRAPH_AUDIT = "GRAPH_AUDIT"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    HYBRID = "HYBRID"


class QueryPlan(BaseModel):
    """Execution plan determined for a user query."""
    route: RouteType
    intent: str = "general_query"
    entity_type: Optional[str] = None  # "Invoice", "PurchaseOrder", "Contract", "Amendment", "Company"
    entity_id: Optional[str] = None
    target_company: Optional[str] = None
    explanation: str = ""
    tier_used: str = "tier1_regex"  # "tier1_regex", "tier2_llm", "tier2_heuristic"


# Fast regex patterns for Tier 1 detection
INVOICE_PATTERN = re.compile(r"\b(FAC-[A-Z0-9_-]+|INV-[A-Z0-9_-]+)\b", re.IGNORECASE)
PO_PATTERN = re.compile(r"\b(BC-[A-Z0-9_-]+|PO-[A-Z0-9_-]+)\b", re.IGNORECASE)
CONTRACT_PATTERN = re.compile(r"\b(CON-[A-Z0-9_-]+|CTR-[A-Z0-9_-]+)\b", re.IGNORECASE)
AMENDMENT_PATTERN = re.compile(r"\b(AVN-[A-Z0-9_-]+|AMD-[A-Z0-9_-]+)\b", re.IGNORECASE)

QUALITATIVE_PATTERN = re.compile(
    r"\b(pourquoi|why|clause|litige|dispute|justifi|expli|comment|raisons?|contexte|détails?)\b",
    re.IGNORECASE,
)

AUDIT_KEYWORDS = [
    "facture", "invoice", "payer", "paiement", "dépassement", "dépasse",
    "conforme", "audit", "montant", "règlement", "bon de commande", "contrat",
    "budget", "plafond", "coût", "prix", "honoraires", "pénalités", "avenant",
]


class QueryPlanner:
    """Cascaded Query Planner: Tier 1 (fast deterministic regex) -> Tier 2 (LLM / heuristic routing)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.client = None
        if self.api_key and genai is not None:
            try:
                self.client = genai.Client(
                    api_key=self.api_key,
                    http_options={"api_version": "v1beta"}
                )
            except Exception:
                self.client = None

    def plan(
        self,
        query: str,
        kg: Optional[Any] = None,
        vector_retriever: Optional[Any] = None,
    ) -> QueryPlan:
        """Determines the retrieval route and resolves target entities."""
        clean_query = query.strip()

        # =====================================================================
        # Tier 1: Fast Regex on Document Identifiers (0 token, 0 ms)
        # =====================================================================
        inv_match = INVOICE_PATTERN.search(clean_query)
        po_match = PO_PATTERN.search(clean_query)
        con_match = CONTRACT_PATTERN.search(clean_query)
        amend_match = AMENDMENT_PATTERN.search(clean_query)

        if inv_match or po_match or con_match or amend_match:
            is_qualitative = bool(QUALITATIVE_PATTERN.search(clean_query))
            route = RouteType.HYBRID if is_qualitative else RouteType.GRAPH_AUDIT

            if inv_match:
                return QueryPlan(
                    route=route,
                    intent="audit_invoice",
                    entity_type="Invoice",
                    entity_id=inv_match.group(0).upper(),
                    explanation="Detected explicit invoice reference in query via Tier 1 regex.",
                    tier_used="tier1_regex",
                )
            if po_match:
                return QueryPlan(
                    route=route,
                    intent="audit_po",
                    entity_type="PurchaseOrder",
                    entity_id=po_match.group(0).upper(),
                    explanation="Detected explicit purchase order reference in query via Tier 1 regex.",
                    tier_used="tier1_regex",
                )
            if con_match:
                return QueryPlan(
                    route=route,
                    intent="contract_summary",
                    entity_type="Contract",
                    entity_id=con_match.group(0).upper(),
                    explanation="Detected explicit contract reference in query via Tier 1 regex.",
                    tier_used="tier1_regex",
                )
            if amend_match:
                return QueryPlan(
                    route=route,
                    intent="amendment_review",
                    entity_type="Amendment",
                    entity_id=amend_match.group(0).upper(),
                    explanation="Detected explicit amendment reference in query via Tier 1 regex.",
                    tier_used="tier1_regex",
                )

        # =====================================================================
        # Tier 2: Intent Classification & Entity Resolution
        # =====================================================================
        has_audit_keywords = any(kw in clean_query.lower() for kw in AUDIT_KEYWORDS)

        # If query is purely qualitative or informational without financial audit keywords
        if not has_audit_keywords:
            return QueryPlan(
                route=RouteType.SEMANTIC_SEARCH,
                intent="semantic_clause",
                explanation="General qualitative inquiry without audit keywords.",
                tier_used="tier2_heuristic",
            )

        # Query has financial / audit intent but no explicit ID (e.g. "Audit les factures d'Acme")
        # Try LLM routing if client is available
        llm_plan = self._plan_with_llm(clean_query) if self.client else None
        if llm_plan:
            resolved_plan = self._resolve_entities_in_graph(llm_plan, clean_query, kg, vector_retriever)
            return resolved_plan

        # Heuristic Tier 2 Fallback (Offline / No API Key)
        return self._plan_with_heuristic(clean_query, kg, vector_retriever)

    def _plan_with_llm(self, query: str) -> Optional[QueryPlan]:
        """Calls Gemini Flash to classify query intent and extract target company/concept."""
        prompt = f"""You are a query classifier for a legal and financial audit Knowledge Graph.
Classify the following user query:
"{query}"

Output ONLY a JSON object with this schema:
{{
  "route": "GRAPH_AUDIT" or "SEMANTIC_SEARCH" or "HYBRID",
  "intent": "audit_invoice" or "audit_po" or "contract_summary" or "semantic_clause",
  "entity_type": "Invoice" or "PurchaseOrder" or "Contract" or "Amendment" or null,
  "target_company": "company name if mentioned, otherwise null",
  "explanation": "brief justification"
}}
"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            raw_text = response.text.strip()
            data = json.loads(raw_text)
            return QueryPlan(
                route=RouteType(data.get("route", "GRAPH_AUDIT")),
                intent=data.get("intent", "general_query"),
                entity_type=data.get("entity_type"),
                target_company=data.get("target_company"),
                explanation=data.get("explanation", "Classified via Gemini Flash"),
                tier_used="tier2_llm",
            )
        except Exception:
            return None

    def _plan_with_heuristic(
        self,
        query: str,
        kg: Optional[Any],
        vector_retriever: Optional[Any],
    ) -> QueryPlan:
        """Determines plan using heuristic intent detection and graph/vector entity resolution."""
        lower_q = query.lower()
        is_qualitative = bool(QUALITATIVE_PATTERN.search(query))

        # Detect primary entity type intent
        if any(w in lower_q for w in ["facture", "invoice", "payer", "règlement"]):
            entity_type = "Invoice"
            intent = "audit_invoice"
        elif any(w in lower_q for w in ["bon de commande", "commande", "purchase order", "po"]):
            entity_type = "PurchaseOrder"
            intent = "audit_po"
        elif any(w in lower_q for w in ["avenant", "amendment"]):
            entity_type = "Amendment"
            intent = "amendment_review"
        else:
            entity_type = "Contract"
            intent = "contract_summary"

        route = RouteType.HYBRID if is_qualitative else RouteType.GRAPH_AUDIT

        base_plan = QueryPlan(
            route=route,
            intent=intent,
            entity_type=entity_type,
            explanation="Detected audit intent via Tier 2 heuristic.",
            tier_used="tier2_heuristic",
        )

        return self._resolve_entities_in_graph(base_plan, query, kg, vector_retriever)

    def _resolve_entities_in_graph(
        self,
        plan: QueryPlan,
        query: str,
        kg: Optional[Any],
        vector_retriever: Optional[Any],
    ) -> QueryPlan:
        """Resolves target company or related document ID in the Knowledge Graph or Vector chunks."""
        # 1. If KG is available, look for company or document nodes matching keywords
        if kg and hasattr(kg, "graph"):
            # Check company match
            for node_id, data in kg.graph.nodes(data=True):
                etype = data.get("entity_type")
                node_name = data.get("name") or str(node_id)
                # If node is a company matching query or target_company
                if etype in ("Company", "Supplier"):
                    norm_name = normalize_company_name(node_name)
                    is_match = False
                    if plan.target_company and is_same_company(plan.target_company, None, node_name, None):
                        is_match = True
                    elif norm_name and norm_name in query.lower():
                        is_match = True
                    elif node_name.lower() in query.lower():
                        is_match = True

                    if is_match:
                        plan.target_company = node_name
                        # Find connected document matching target entity_type
                        for u, v, edata in kg.graph.edges(data=True):
                            if u == node_id or v == node_id:
                                other = v if u == node_id else u
                                other_data = kg.get_node(other) or {}
                                if other_data.get("entity_type") == plan.entity_type:
                                    plan.entity_id = other
                                    return plan

        # 2. Vector search chunk fallback to resolve entity ID from context
        if vector_retriever:
            chunks = []
            if hasattr(vector_retriever, "search"):
                try:
                    chunks = vector_retriever.search(query, top_k=3)
                except Exception:
                    chunks = []
            # Fallback to direct chunk inspection if index is not built or search yielded nothing
            if not chunks and hasattr(vector_retriever, "chunks"):
                chunks = vector_retriever.chunks[:5]

            for chunk in chunks:
                content = chunk.get("content", "")
                if plan.entity_type == "Invoice":
                    m = INVOICE_PATTERN.search(content)
                    if m:
                        plan.entity_id = m.group(0).upper()
                        return plan
                elif plan.entity_type == "PurchaseOrder":
                    m = PO_PATTERN.search(content)
                    if m:
                        plan.entity_id = m.group(0).upper()
                        return plan
                elif plan.entity_type in ("Contract", "Amendment"):
                    m = CONTRACT_PATTERN.search(content)
                    if m:
                        plan.entity_id = m.group(0).upper()
                        return plan

        # If no entity could be resolved for audit, downgrade to semantic search
        if not plan.entity_id:
            plan.route = RouteType.SEMANTIC_SEARCH
            plan.explanation += " (No specific graph entity resolved; downgraded to semantic search)."

        return plan
