import re
from typing import List, Dict, Any, Optional
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.graph_retriever import GraphRetriever


def compute_rrf_score(rank: int, k: int = 60) -> float:
    """Computes Reciprocal Rank Fusion score for an item at a given 1-based rank."""
    return 1.0 / (k + rank)


class HybridRetriever:
    """Combines semantic vector search with graph traversal using Reciprocal Rank Fusion."""

    def __init__(self, vector_retriever: VectorRetriever, graph_retriever: GraphRetriever):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Performs hybrid retrieval: vector passages + graph reconciliation."""
        # 1. Dense vector search
        vector_results = self.vector_retriever.search(query, top_k=top_k)

        # 2. Entity reference detection in query or top results
        detected_invoice = None
        inv_match = re.search(r"FAC-[A-Z0-9_-]+", query)
        if inv_match:
            detected_invoice = inv_match.group(0)
        elif vector_results:
            # Fallback: check if the top vector chunk mentions an invoice
            for chunk in vector_results:
                chunk_match = re.search(r"FAC-[A-Z0-9_-]+", chunk["content"])
                if chunk_match:
                    detected_invoice = chunk_match.group(0)
                    break

        # 3. Graph retrieval
        graph_audit = None
        if detected_invoice:
            graph_audit = self.graph_retriever.audit_invoice(detected_invoice)

        # 4. RRF scoring and deduplication
        fused_items: Dict[str, Dict[str, Any]] = {}

        # Add vector chunks
        for rank, chunk in enumerate(vector_results, start=1):
            key = f"{chunk['file_name']}_p{chunk['page_number']}_{chunk['content'][:30]}"
            score = compute_rrf_score(rank)
            fused_items[key] = {
                "source": "vector",
                "file_name": chunk["file_name"],
                "content": chunk["content"],
                "rrf_score": score,
            }

        # Add graph evidence if available
        if graph_audit and graph_audit.get("evidence"):
            for rank, ev in enumerate(graph_audit["evidence"], start=1):
                key = f"graph_ev_{ev[:30]}"
                score = compute_rrf_score(rank)
                if key in fused_items:
                    fused_items[key]["rrf_score"] += score
                else:
                    fused_items[key] = {
                        "source": "graph",
                        "file_name": graph_audit.get("invoice_id", "graph"),
                        "content": ev,
                        "rrf_score": score,
                    }

        ranked_evidence = sorted(fused_items.values(), key=lambda x: x["rrf_score"], reverse=True)

        return {
            "query": query,
            "detected_invoice": detected_invoice,
            "graph_audit": graph_audit,
            "ranked_evidence": ranked_evidence[:top_k],
        }
