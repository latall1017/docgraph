import pytest
from src.evaluation.benchmark import get_benchmark_dataset, BenchmarkRunner
from src.graph.graph_builder import KnowledgeGraph
from src.extraction.schema import ExtractedEntity, ExtractedRelation
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.ingestion.pdf_parser import parse_pdf
import os
import glob


@pytest.fixture
def populated_bench_system():
    # 1. Setup Knowledge Graph with full multi-hop scenarios
    kg = KnowledgeGraph()

    # Scenario Alpha
    kg.add_entity(ExtractedEntity(id="FAC-2026-001", entity_type="Invoice", name="Invoice Alpha", properties={"amount_ht": 10000.0}, evidence="FAC-2026-001", page_number=1))
    kg.add_entity(ExtractedEntity(id="BC-2026-001", entity_type="PurchaseOrder", name="PO Alpha", properties={"amount_ht": 10000.0}, evidence="BC-2026-001", page_number=1))
    kg.add_entity(ExtractedEntity(id="CTR-2026-ALPHA", entity_type="Contract", name="Contract Alpha", properties={"maximum_amount": 50000.0}, evidence="CTR-2026-ALPHA", page_number=1))
    kg.add_relation(ExtractedRelation(source_id="FAC-2026-001", relation_type="REFERS_TO", target_id="BC-2026-001", evidence="Ref BC-2026-001"))
    kg.add_relation(ExtractedRelation(source_id="BC-2026-001", relation_type="GOVERNED_BY", target_id="CTR-2026-ALPHA", evidence="Governed by CTR-2026-ALPHA"))

    # Scenario Beta (Discrepancy)
    kg.add_entity(ExtractedEntity(id="FAC-2026-002", entity_type="Invoice", name="Invoice Beta", properties={"amount_ht": 7500.0}, evidence="FAC-2026-002", page_number=1))
    kg.add_entity(ExtractedEntity(id="BC-2026-002", entity_type="PurchaseOrder", name="PO Beta", properties={"amount_ht": 5000.0}, evidence="BC-2026-002", page_number=1))
    kg.add_entity(ExtractedEntity(id="CTR-2026-BETA", entity_type="Contract", name="Contract Beta", properties={"maximum_amount": 30000.0}, evidence="CTR-2026-BETA", page_number=1))
    kg.add_relation(ExtractedRelation(source_id="FAC-2026-002", relation_type="REFERS_TO", target_id="BC-2026-002", evidence="Ref BC-2026-002"))
    kg.add_relation(ExtractedRelation(source_id="BC-2026-002", relation_type="GOVERNED_BY", target_id="CTR-2026-BETA", evidence="Governed by CTR-2026-BETA"))

    # Scenario Gamma (Justified by Amendment)
    kg.add_entity(ExtractedEntity(id="FAC-2026-003", entity_type="Invoice", name="Invoice Gamma", properties={"amount_ht": 28000.0}, evidence="FAC-2026-003", page_number=1))
    kg.add_entity(ExtractedEntity(id="BC-2026-003", entity_type="PurchaseOrder", name="PO Gamma", properties={"amount_ht": 28000.0}, evidence="BC-2026-003", page_number=1))
    kg.add_entity(ExtractedEntity(id="CTR-2026-GAMMA", entity_type="Contract", name="Contract Gamma", properties={"maximum_amount": 20000.0}, evidence="CTR-2026-GAMMA", page_number=1))
    kg.add_entity(ExtractedEntity(id="AVN-2026-001", entity_type="Amendment", name="Amendment Gamma", properties={"new_maximum_amount": 40000.0}, evidence="AVN-2026-001", page_number=1))
    kg.add_relation(ExtractedRelation(source_id="FAC-2026-003", relation_type="REFERS_TO", target_id="BC-2026-003", evidence="Ref BC-2026-003"))
    kg.add_relation(ExtractedRelation(source_id="BC-2026-003", relation_type="GOVERNED_BY", target_id="CTR-2026-GAMMA", evidence="Governed by CTR-2026-GAMMA"))
    kg.add_relation(ExtractedRelation(source_id="AVN-2026-001", relation_type="MODIFIES", target_id="CTR-2026-GAMMA", evidence="Modifies CTR-2026-GAMMA"))

    # Scenario Delta (Missing PO)
    kg.add_entity(ExtractedEntity(id="FAC-2026-004", entity_type="Invoice", name="Invoice Delta", properties={"amount_ht": 3000.0}, evidence="FAC-2026-004", page_number=1))

    # 2. Setup Vector Retriever with synthetic PDF documents
    v_retriever = VectorRetriever()
    pdf_files = glob.glob("data/synthetic/documents/**/*.pdf", recursive=True)
    for pdf_path in pdf_files[:6]:  # Index sample PDFs
        if os.path.exists(pdf_path):
            doc = parse_pdf(pdf_path)
            v_retriever.add_document(doc)
    v_retriever.build_index()

    # 3. Setup Graph and Hybrid Retrievers
    g_retriever = GraphRetriever(kg)
    hybrid_retriever = HybridRetriever(v_retriever, g_retriever)

    return BenchmarkRunner(v_retriever, g_retriever, hybrid_retriever)


def test_benchmark_dataset():
    dataset = get_benchmark_dataset()
    assert len(dataset) >= 12
    categories = {q.category for q in dataset}
    assert categories == {"simple", "relational", "multihop", "anomaly", "abstention"}


def test_benchmark_runner_evaluation(populated_bench_system):
    runner = populated_bench_system
    metrics = runner.evaluate_retrieval()

    # Validate structure
    assert "naive_vector" in metrics
    assert "graph_only" in metrics
    assert "hybrid_graph_rag" in metrics

    # Verify that Hybrid Graph RAG outperforms Naive Vector on multi-hop questions
    assert metrics["hybrid_graph_rag"]["multihop_success"] > metrics["naive_vector"]["multihop_success"]

    # Verify high overall recall and perfect abstention on Hybrid Graph RAG
    assert metrics["hybrid_graph_rag"]["entity_recall"] >= 0.70
    assert metrics["hybrid_graph_rag"]["abstention_accuracy"] == 1.0
