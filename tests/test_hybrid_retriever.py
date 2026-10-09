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
