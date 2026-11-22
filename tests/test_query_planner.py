import pytest
from src.retrieval.query_planner import QueryPlanner, RouteType, QueryPlan
from src.graph.graph_builder import KnowledgeGraph
from src.extraction.schema import ExtractedEntity, ExtractedRelation
from src.retrieval.vector_retriever import VectorRetriever


def test_tier1_invoice_regex_detection():
    planner = QueryPlanner()
    plan = planner.plan("Vérifier la facture FAC-2026-001 s'il vous plaît")
    assert plan.tier_used == "tier1_regex"
    assert plan.route == RouteType.GRAPH_AUDIT
    assert plan.entity_type == "Invoice"
    assert plan.entity_id == "FAC-2026-001"
    assert plan.intent == "audit_invoice"


def test_tier1_qualitative_inquiry_becomes_hybrid():
    planner = QueryPlanner()
    plan = planner.plan("Pourquoi la facture FAC-2026-002 présente un litige sur le montant ?")
    assert plan.tier_used == "tier1_regex"
    assert plan.route == RouteType.HYBRID
    assert plan.entity_type == "Invoice"
    assert plan.entity_id == "FAC-2026-002"


def test_tier1_po_and_contract_detection():
    planner = QueryPlanner()
    plan_po = planner.plan("Audit du bon de commande BC-2026-002")
    assert plan_po.tier_used == "tier1_regex"
    assert plan_po.entity_type == "PurchaseOrder"
    assert plan_po.entity_id == "BC-2026-002"

    plan_con = planner.plan("Quel est le plafond du contrat CON-2026-ALPHA ?")
    assert plan_con.tier_used == "tier1_regex"
    assert plan_con.entity_type == "Contract"
    assert plan_con.entity_id == "CON-2026-ALPHA"


def test_tier2_pure_semantic_question_routes_to_vector():
    planner = QueryPlanner()
    plan = planner.plan("Quelles sont les clauses de confidentialité et de préavis applicables ?")
    assert plan.route == RouteType.SEMANTIC_SEARCH
    assert plan.tier_used == "tier2_heuristic"
    assert plan.entity_id is None
    assert plan.intent == "semantic_clause"


def test_tier2_natural_language_company_resolution_in_graph():
    kg = KnowledgeGraph()
    # Add company node
    kg.add_entity(ExtractedEntity(
        id="SUP-001",
        entity_type="Company",
        name="Alpha Services SAS",
        properties={},
        evidence="Alpha Services SAS",
        page_number=1,
    ))
    # Add invoice node
    kg.add_entity(ExtractedEntity(
        id="FAC-2026-001",
        entity_type="Invoice",
        name="Invoice 001",
        properties={"amount_ht": 10000.0},
        evidence="FACTURE FAC-2026-001",
        page_number=1,
    ))
    # Link company and invoice
    kg.add_relation(ExtractedRelation(
        source_id="FAC-2026-001",
        relation_type="ISSUED_BY",
        target_id="SUP-001",
        evidence="Émise par Alpha Services",
    ))

    planner = QueryPlanner()
    # Query mentions Alpha Services and invoice without explicit FAC- ID
    plan = planner.plan(
        "Vérifie la facture émise par Alpha Services pour le dernier mois",
        kg=kg,
    )
    assert plan.route == RouteType.GRAPH_AUDIT
    assert plan.entity_type == "Invoice"
    assert plan.entity_id == "FAC-2026-001"
    assert plan.target_company == "Alpha Services SAS"


def test_tier2_vector_chunk_fallback_resolution():
    vr = VectorRetriever()
    vr.chunks = [{
        "chunk_id": "c1",
        "file_name": "invoice_beta.pdf",
        "page_number": 1,
        "content": "Société Beta Logistics SARL - Facture N° FAC-2026-002 - Total HT : 7500 EUR",
    }]
    vr.build_index()
    planner = QueryPlanner()
    plan = planner.plan(
        "Vérifie s'il y a un dépassement sur la facture de transport",
        vector_retriever=vr,
    )
    assert plan.entity_type == "Invoice"
    assert plan.entity_id == "FAC-2026-002"
    assert plan.route in (RouteType.GRAPH_AUDIT, RouteType.HYBRID)


def test_unresolvable_audit_query_gracefully_downgrades():
    planner = QueryPlanner()
    # Query has audit keyword "dépassement", but no entity in graph or vectors
    plan = planner.plan("Y a-t-il eu un dépassement budgétaire non documenté ?")
    assert plan.route == RouteType.SEMANTIC_SEARCH
    assert plan.entity_id is None
