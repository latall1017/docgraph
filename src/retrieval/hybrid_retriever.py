# Combines dense vector search and deterministic graph traversal using Reciprocal Rank Fusion.

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
        """Performs hybrid retrieval: vector passages and graph reconciliation."""
        # 1. Dense vector search
        vector_results = self.vector_retriever.search(query, top_k=top_k)

        # 2. Entity reference detection in query or top results (Entity Dispatcher)
        detected_entity = None
        detected_entity_type = None
        detected_invoice = None

        inv_match = re.search(r"\b(FAC-[A-Z0-9_-]+|INV-[A-Z0-9_-]+)\b", query, re.IGNORECASE)
        po_match = re.search(r"\b(BC-[A-Z0-9_-]+|PO-[A-Z0-9_-]+)\b", query, re.IGNORECASE)
        con_match = re.search(r"\b(CON-[A-Z0-9_-]+|CTR-[A-Z0-9_-]+)\b", query, re.IGNORECASE)
        amend_match = re.search(r"\b(AVN-[A-Z0-9_-]+|AMD-[A-Z0-9_-]+)\b", query, re.IGNORECASE)

        if inv_match:
            detected_entity = inv_match.group(0).upper()
            detected_entity_type = "Invoice"
            detected_invoice = detected_entity
        elif po_match:
            detected_entity = po_match.group(0).upper()
            detected_entity_type = "PurchaseOrder"
        elif con_match:
            detected_entity = con_match.group(0).upper()
            detected_entity_type = "Contract"
        elif amend_match:
            detected_entity = amend_match.group(0).upper()
            detected_entity_type = "Amendment"
        elif vector_results:
            # Only fallback to vector chunks if query expresses an explicit financial/audit intent
            # Prevents query hijacking on general clauses (e.g., termination, notice, signatures)
            audit_intent = any(kw in query.lower() for kw in [
                "facture", "invoice", "payer", "paiement", "dépassement",
                "conforme", "audit", "montant", "règlement", "bon de commande", "contrat", "budget"
            ])
            if audit_intent:
                for chunk in vector_results:
                    c_inv = re.search(r"\b(FAC-[A-Z0-9_-]+|INV-[A-Z0-9_-]+)\b", chunk["content"])
                    c_po = re.search(r"\b(BC-[A-Z0-9_-]+|PO-[A-Z0-9_-]+)\b", chunk["content"])
                    c_con = re.search(r"\b(CON-[A-Z0-9_-]+|CTR-[A-Z0-9_-]+)\b", chunk["content"])
                    if c_inv:
                        detected_entity = c_inv.group(0).upper()
                        detected_entity_type = "Invoice"
                        detected_invoice = detected_entity
                        break
                    elif c_po:
                        detected_entity = c_po.group(0).upper()
                        detected_entity_type = "PurchaseOrder"
                        break
                    elif c_con:
                        detected_entity = c_con.group(0).upper()
                        detected_entity_type = "Contract"
                        break

        # 3. Graph retrieval dispatch
        graph_audit = None
        if detected_entity_type == "Invoice":
            graph_audit = self.graph_retriever.audit_invoice(detected_entity)
        elif detected_entity_type == "PurchaseOrder":
            graph_audit = self.graph_retriever.audit_po(detected_entity)
        elif detected_entity_type == "Contract":
            graph_audit = self.graph_retriever.get_contract_summary(detected_entity)
        elif detected_entity_type == "Amendment":
            # For amendment, inspect modifying relation to target contract
            target_con = None
            if detected_entity in self.graph_retriever.kg.graph:
                for _, tgt, edata in self.graph_retriever.kg.graph.out_edges(detected_entity, data=True):
                    if edata.get("relation_type") == "MODIFIES":
                        target_con = tgt
                        break
            if target_con:
                graph_audit = self.graph_retriever.get_contract_summary(target_con)
            else:
                amend_node = self.graph_retriever.kg.get_node(detected_entity)
                graph_audit = {
                    "entity_type": "Amendment",
                    "amendment_id": detected_entity,
                    "properties": amend_node.get("properties") if amend_node else {},
                    "evidence": [amend_node.get("evidence", "")] if amend_node else [],
                }

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
        }
