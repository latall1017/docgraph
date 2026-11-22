from src.ingestion.pdf_parser import parse_pdf
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever, compute_rrf_score
from src.graph.graph_builder import KnowledgeGraph
from src.extraction.schema import ExtractedEntity, ExtractedRelation


def test_compute_rrf_score():
    score_1 = compute_rrf_score(1)
    score_2 = compute_rrf_score(2)
    assert score_1 > score_2
    assert score_1 == 1.0 / 61.0


def test_hybrid_retriever_query():
    # Setup Vector Retriever
    pdf = "data/synthetic/documents/SUP-002_Beta_Logistics_SARL/invoice_FAC-2026-002.pdf"
    doc = parse_pdf(pdf)
    v_retriever = VectorRetriever()
    v_retriever.add_document(doc)
    v_retriever.build_index()

    # Setup Graph with Beta Logistics (exceeding PO)
    kg = KnowledgeGraph()
    kg.add_entity(ExtractedEntity(
        id="FAC-2026-002",
        entity_type="Invoice",
        name="Invoice Beta",
        properties={"amount_ht": 7500.0},
        evidence="FACTURE FAC-2026-002",
        page_number=1,
    ))
    kg.add_entity(ExtractedEntity(
        id="BC-2026-002",
        entity_type="PurchaseOrder",
        name="PO Beta",
        properties={"amount_ht": 5000.0},
        evidence="BON DE COMMANDE BC-2026-002",
        page_number=1,
    ))
    kg.add_relation(ExtractedRelation(
        source_id="FAC-2026-002",
        relation_type="REFERS_TO",
        target_id="BC-2026-002",
        evidence="Bon de commande associé : BC-2026-002",
    ))
    g_retriever = GraphRetriever(kg)

    # Hybrid Retrieval
    hybrid = HybridRetriever(v_retriever, g_retriever)
    result = hybrid.retrieve("Vérifier la facture FAC-2026-002 et les frais de transport", top_k=3)

    assert result["detected_invoice"] == "FAC-2026-002"
    assert result["graph_audit"] is not None
    assert result["graph_audit"]["exceeds_po"] is True
    assert result["graph_audit"]["amount_diff"] == 2500.0
    assert len(result["ranked_evidence"]) > 0


def test_hybrid_retriever_po_query():
    kg = KnowledgeGraph()
    kg.add_entity(ExtractedEntity(
        id="BC-2026-002",
        entity_type="PurchaseOrder",
        name="PO Beta",
        properties={"amount_ht": 5000.0},
        evidence="BON DE COMMANDE BC-2026-002",
        page_number=1,
    ))
    kg.add_entity(ExtractedEntity(
        id="FAC-2026-002",
        entity_type="Invoice",
        name="Invoice Beta",
        properties={"amount_ht": 7500.0},
        evidence="FACTURE FAC-2026-002",
        page_number=1,
    ))
    kg.add_relation(ExtractedRelation(
        source_id="FAC-2026-002",
        relation_type="REFERS_TO",
        target_id="BC-2026-002",
        evidence="Bon de commande associé : BC-2026-002",
    ))
    g_retriever = GraphRetriever(kg)
    v_retriever = VectorRetriever()

    hybrid = HybridRetriever(v_retriever, g_retriever)
    result = hybrid.retrieve("Quel est le statut du bon de commande BC-2026-002 ?", top_k=3)

    assert result["detected_entity"] == "BC-2026-002"
    assert result["detected_entity_type"] == "PurchaseOrder"
    assert result["graph_audit"] is not None
    assert result["graph_audit"]["entity_type"] == "PurchaseOrder"
    assert result["graph_audit"]["exceeds_po"] is True
    assert result["graph_audit"]["total_invoiced_ht"] == 7500.0


def test_hybrid_retriever_contract_query():
    kg = KnowledgeGraph()
    kg.add_entity(ExtractedEntity(
        id="CON-2026-ALPHA",
        entity_type="Contract",
        name="Contract Alpha",
        properties={"maximum_amount": 50000.0},
        evidence="Contrat CON-2026-ALPHA",
        page_number=1,
    ))
    kg.add_entity(ExtractedEntity(
        id="AVN-001",
        entity_type="Amendment",
        name="Avenant 1",
        properties={"new_maximum_amount": 75000.0},
        evidence="Avenant AVN-001",
        page_number=1,
    ))
    kg.add_relation(ExtractedRelation(
        source_id="AVN-001",
        relation_type="MODIFIES",
        target_id="CON-2026-ALPHA",
        evidence="Modifie CON-2026-ALPHA",
    ))
    g_retriever = GraphRetriever(kg)
    v_retriever = VectorRetriever()

    hybrid = HybridRetriever(v_retriever, g_retriever)
    result = hybrid.retrieve("Quel est le plafond du contrat CON-2026-ALPHA ?", top_k=3)

    assert result["detected_entity"] == "CON-2026-ALPHA"
    assert result["detected_entity_type"] == "Contract"
    assert result["graph_audit"] is not None
    assert result["graph_audit"]["effective_ceiling"] == 75000.0
    assert "AVN-001" in result["graph_audit"]["applied_amendments"]


def test_prevent_query_hijacking_on_general_questions():
    """Verify that a general question does not get hijacked by an invoice audit."""
    g_retriever = GraphRetriever(KnowledgeGraph())
    v_retriever = VectorRetriever()
    # Add dummy chunk containing a FAC mention
    v_retriever.chunks = [{
        "chunk_id": "test_1",
        "file_name": "contract.pdf",
        "page_number": 1,
        "content": "Clause de résiliation. Note: exemple de facturation FAC-2026-999.",
    }]

    hybrid = HybridRetriever(v_retriever, g_retriever)
    result = hybrid.retrieve("Quelle est la clause de préavis et résiliation du contrat ?", top_k=1)

    # Since the query does not express an audit/payment intent, it must NOT hijack FAC-2026-999
    assert result["detected_invoice"] is None
    assert result["detected_entity"] is None
    assert result["graph_audit"] is None

