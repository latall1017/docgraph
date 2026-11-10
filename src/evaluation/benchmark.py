# Benchmark suite evaluating Naive Vector RAG, Graph Only, and Hybrid Graph RAG.

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


@dataclass
class BenchmarkQuery:
    id: str
    category: str  # "simple", "relational", "multihop", "anomaly", "abstention"
    question: str
    target_invoice: Optional[str] = None
    expected_entities: List[str] = field(default_factory=list)
    expected_anomaly: Optional[str] = None
    is_answerable: bool = True


def get_benchmark_dataset() -> List[BenchmarkQuery]:
    """Returns a curated evaluation dataset covering the 5 benchmark categories."""
    return [
        # 1. Simple lookup queries
        BenchmarkQuery(
            id="SMP-01",
            category="simple",
            question="What is the pre-tax amount of invoice FAC-2026-001?",
            target_invoice="FAC-2026-001",
            expected_entities=["FAC-2026-001", "10000"],
        ),
        BenchmarkQuery(
            id="SMP-02",
            category="simple",
            question="What is the initial ceiling of contract CTR-2026-BETA?",
            expected_entities=["CTR-2026-BETA", "30000"],
        ),
        BenchmarkQuery(
            id="SMP-03",
            category="simple",
            question="What is the date of invoice FAC-2026-003?",
            target_invoice="FAC-2026-003",
            expected_entities=["FAC-2026-003"],
        ),

        # 2. Relational queries (2 documents)
        BenchmarkQuery(
            id="REL-01",
            category="relational",
            question="Which purchase order is referenced by invoice FAC-2026-001?",
            target_invoice="FAC-2026-001",
            expected_entities=["FAC-2026-001", "BC-2026-001"],
        ),
        BenchmarkQuery(
            id="REL-02",
            category="relational",
            question="Which contract governs purchase order BC-2026-002?",
            expected_entities=["BC-2026-002", "CTR-2026-BETA"],
        ),
        BenchmarkQuery(
            id="REL-03",
            category="relational",
            question="Which purchase order corresponds to invoice FAC-2026-002?",
            target_invoice="FAC-2026-002",
            expected_entities=["FAC-2026-002", "BC-2026-002"],
        ),

        # 3. Multi-hop reasoning queries (3-4 documents: Invoice -> PO -> Contract -> Amendment)
        BenchmarkQuery(
            id="HOP-01",
            category="multihop",
            question="Does invoice FAC-2026-003 have an amendment modifying its governing contract CTR-2026-GAMMA?",
            target_invoice="FAC-2026-003",
            expected_entities=["FAC-2026-003", "BC-2026-003", "CTR-2026-GAMMA", "AVN-2026-001"],
        ),
        BenchmarkQuery(
            id="HOP-02",
            category="multihop",
            question="Trace the full procurement chain for invoice FAC-2026-001 from order to framework contract.",
            target_invoice="FAC-2026-001",
            expected_entities=["FAC-2026-001", "BC-2026-001", "CTR-2026-ALPHA"],
        ),
        BenchmarkQuery(
            id="HOP-03",
            category="multihop",
            question="What is the effective contractual ceiling applying to invoice FAC-2026-003 after amendments?",
            target_invoice="FAC-2026-003",
            expected_entities=["CTR-2026-GAMMA", "AVN-2026-001", "40000"],
        ),

        # 4. Anomaly and discrepancy detection
        BenchmarkQuery(
            id="ANO-01",
            category="anomaly",
            question="Does invoice FAC-2026-002 exceed the amount of its associated purchase order?",
            target_invoice="FAC-2026-002",
            expected_entities=["FAC-2026-002", "BC-2026-002"],
            expected_anomaly="invoice_exceeds_po",
        ),
        BenchmarkQuery(
            id="ANO-02",
            category="anomaly",
            question="Is invoice FAC-2026-004 compliant and linked to an approved purchase order?",
            target_invoice="FAC-2026-004",
            expected_entities=["FAC-2026-004"],
            expected_anomaly="missing_po",
        ),
        BenchmarkQuery(
            id="ANO-03",
            category="anomaly",
            question="Is invoice FAC-2026-001 fully compliant with its order and contract limits?",
            target_invoice="FAC-2026-001",
            expected_entities=["FAC-2026-001", "BC-2026-001", "CTR-2026-ALPHA"],
            expected_anomaly=None,
        ),

        # 5. Abstention and Out-of-corpus queries
        BenchmarkQuery(
            id="ABS-01",
            category="abstention",
            question="What was the total invoice amount paid to Zeta Solutions in 2024?",
            is_answerable=False,
        ),
        BenchmarkQuery(
            id="ABS-02",
            category="abstention",
            question="What is the bank account IBAN number for supplier Epsilon SARL?",
            is_answerable=False,
        ),
    ]


class BenchmarkRunner:
    """Evaluates and compares Naive Vector RAG, Graph RAG, and Hybrid Graph RAG."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
        hybrid_retriever: HybridRetriever,
    ):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.hybrid_retriever = hybrid_retriever

    def evaluate_retrieval(self, queries: Optional[List[BenchmarkQuery]] = None) -> Dict[str, Dict[str, float]]:
        """Runs offline retrieval evaluation across the three architectures."""
        if queries is None:
            queries = get_benchmark_dataset()

        metrics = {
            "naive_vector": {"entity_recall": 0.0, "multihop_success": 0.0, "abstention_accuracy": 0.0},
            "graph_only": {"entity_recall": 0.0, "multihop_success": 0.0, "abstention_accuracy": 0.0},
            "hybrid_graph_rag": {"entity_recall": 0.0, "multihop_success": 0.0, "abstention_accuracy": 0.0},
        }

        multihop_queries = [q for q in queries if q.category == "multihop"]
        abstention_queries = [q for q in queries if q.category == "abstention"]
        entity_queries = [q for q in queries if q.expected_entities]

        # 1. Evaluate Naive Vector
        v_entity_hits = 0
        v_multihop_hits = 0
        v_abstention_correct = 0

        for q in queries:
            v_res = self.vector_retriever.search(q.question, top_k=3)
            retrieved_text = " ".join([c["content"] for c in v_res])

            # Check entity recall
            if q.expected_entities:
                hits = sum(1 for ent in q.expected_entities if ent in retrieved_text)
                v_entity_hits += hits / len(q.expected_entities)

            # Check multihop completeness (must retrieve ALL expected entities)
            if q.category == "multihop":
                if all(ent in retrieved_text for ent in q.expected_entities):
                    v_multihop_hits += 1

            # Check abstention (vector search always returns top-k chunks even if irrelevant)
            if q.category == "abstention":
                # If top score is below a strict threshold (e.g. 0.75)
                max_score = v_res[0]["score"] if v_res else 0.0
                if max_score < 0.75:
                    v_abstention_correct += 1

        # 2. Evaluate Graph Only
        g_entity_hits = 0
        g_multihop_hits = 0
        g_abstention_correct = 0

        for q in queries:
            g_audit = self.graph_retriever.audit_invoice(q.target_invoice) if q.target_invoice else None
            g_text = str(g_audit) if g_audit else ""

            if q.expected_entities:
                hits = sum(1 for ent in q.expected_entities if ent in g_text)
                g_entity_hits += hits / len(q.expected_entities)

            if q.category == "multihop":
                if q.expected_entities and all(ent in g_text for ent in q.expected_entities):
                    g_multihop_hits += 1

            if q.category == "abstention":
                # Graph correctly finds no matching invoice node
                if not g_audit or g_audit.get("status") == "not_found":
                    g_abstention_correct += 1

        # 3. Evaluate Hybrid Graph RAG
        h_entity_hits = 0
        h_multihop_hits = 0
        h_abstention_correct = 0

        for q in queries:
            h_res = self.hybrid_retriever.retrieve(q.question, top_k=3)
            h_evidence = " ".join([e.get("content", "") for e in h_res.get("ranked_evidence", [])])
            h_audit_text = str(h_res.get("graph_audit") or "")
            combined_context = f"{h_evidence} {h_audit_text}"

            if q.expected_entities:
                hits = sum(1 for ent in q.expected_entities if ent in combined_context)
                h_entity_hits += hits / len(q.expected_entities)

            if q.category == "multihop":
                if all(ent in combined_context for ent in q.expected_entities):
                    h_multihop_hits += 1

            if q.category == "abstention":
                # Abstention succeeds when the system retrieves zero false-positive evidence for out-of-corpus entities
                unanswerable_terms = ["zeta", "epsilon", "2024", "iban"]
                has_false_evidence = any(term in combined_context.lower() for term in unanswerable_terms)
                if not has_false_evidence:
                    h_abstention_correct += 1

        # Aggregate percentages
        metrics["naive_vector"]["entity_recall"] = round(v_entity_hits / len(entity_queries), 3)
        metrics["naive_vector"]["multihop_success"] = round(v_multihop_hits / len(multihop_queries), 3)
        metrics["naive_vector"]["abstention_accuracy"] = round(v_abstention_correct / len(abstention_queries), 3)

        metrics["graph_only"]["entity_recall"] = round(g_entity_hits / len(entity_queries), 3)
        metrics["graph_only"]["multihop_success"] = round(g_multihop_hits / len(multihop_queries), 3)
        metrics["graph_only"]["abstention_accuracy"] = round(g_abstention_correct / len(abstention_queries), 3)

        metrics["hybrid_graph_rag"]["entity_recall"] = round(h_entity_hits / len(entity_queries), 3)
        metrics["hybrid_graph_rag"]["multihop_success"] = round(h_multihop_hits / len(multihop_queries), 3)
        metrics["hybrid_graph_rag"]["abstention_accuracy"] = round(h_abstention_correct / len(abstention_queries), 3)

        return metrics
