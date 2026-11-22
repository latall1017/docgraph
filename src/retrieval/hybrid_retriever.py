# Combines dense vector search and deterministic graph traversal using Reciprocal Rank Fusion.

import re
from typing import List, Dict, Any, Optional
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.query_planner import QueryPlanner, RouteType, QueryPlan


def compute_rrf_score(rank: int, k: int = 60) -> float:
    """Computes Reciprocal Rank Fusion score for an item at a given 1-based rank."""
    return 1.0 / (k + rank)


class HybridRetriever:
    """Combines semantic vector search with graph traversal using Reciprocal Rank Fusion and Query Planning."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
        planner: Optional[QueryPlanner] = None,
    ):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.planner = planner or QueryPlanner()

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Performs hybrid retrieval: vector passages and graph reconciliation using cascaded query planning."""
        # 1. Determine execution plan via QueryPlanner (Tier 1 regex -> Tier 2 heuristic / LLM)
        plan = self.planner.plan(
            query=query,
            kg=self.graph_retriever.kg,
            vector_retriever=self.vector_retriever,
        )

        detected_entity = plan.entity_id
        detected_entity_type = plan.entity_type
        detected_invoice = plan.entity_id if plan.entity_type == "Invoice" else None

        # 2. Dense vector search
        vector_results = self.vector_retriever.search(query, top_k=top_k)

        # 3. Graph retrieval dispatch according to plan
        graph_audit = None
        if plan.route in (RouteType.GRAPH_AUDIT, RouteType.HYBRID) and plan.entity_id:
            if plan.entity_type == "Invoice":
                graph_audit = self.graph_retriever.audit_invoice(plan.entity_id)
            elif plan.entity_type == "PurchaseOrder":
                graph_audit = self.graph_retriever.audit_po(plan.entity_id)
            elif plan.entity_type == "Contract":
                graph_audit = self.graph_retriever.get_contract_summary(plan.entity_id)
            elif plan.entity_type == "Amendment":
                # For amendment, inspect modifying relation to target contract
                target_con = None
                if plan.entity_id in self.graph_retriever.kg.graph:
                    for _, tgt, edata in self.graph_retriever.kg.graph.out_edges(plan.entity_id, data=True):
                        if edata.get("relation_type") == "MODIFIES":
                            target_con = tgt
                            break
                if target_con:
                    graph_audit = self.graph_retriever.get_contract_summary(target_con)
                else:
                    amend_node = self.graph_retriever.kg.get_node(plan.entity_id)
                    graph_audit = {
                        "entity_type": "Amendment",
                        "amendment_id": plan.entity_id,
                        "properties": amend_node.get("properties") if amend_node else {},
                        "evidence": [amend_node.get("evidence", "")] if amend_node else [],
                    }
        else:
            # If route is purely SEMANTIC_SEARCH or no entity resolved, clear entity labels
            if plan.route == RouteType.SEMANTIC_SEARCH:
                detected_entity = None
                detected_entity_type = None
                detected_invoice = None

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
            entity_label = detected_entity or graph_audit.get("invoice_id") or "graph"
            for rank, ev in enumerate(graph_audit["evidence"], start=1):
                key = f"graph_ev_{ev[:30]}"
                score = compute_rrf_score(rank)
                if key in fused_items:
                    fused_items[key]["rrf_score"] += score
                else:
                    fused_items[key] = {
                        "source": "graph",
                        "file_name": entity_label,
                        "content": ev,
                        "rrf_score": score,
                    }

        ranked_evidence = sorted(fused_items.values(), key=lambda x: x["rrf_score"], reverse=True)

        return {
            "query": query,
            "detected_entity": detected_entity,
            "detected_entity_type": detected_entity_type,
            "detected_invoice": detected_invoice,
            "graph_audit": graph_audit,
            "ranked_evidence": ranked_evidence[:top_k],
            "plan": plan.model_dump(),
        }
