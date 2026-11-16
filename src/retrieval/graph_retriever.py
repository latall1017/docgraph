# Traverses the knowledge graph to perform multi-hop invoice and contract audits.

from typing import Dict, Any, Optional
from src.graph.graph_builder import KnowledgeGraph


class GraphRetriever:
    """Retrieves interconnected procurement context and performs deterministic reconciliation."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def audit_invoice(self, invoice_id: str) -> Dict[str, Any]:
        """Audits an invoice against its associated purchase order, contract, and amendments."""
        inv_node = self.kg.get_node(invoice_id)
        if not inv_node:
            return {"status": "not_found", "message": f"Invoice {invoice_id} not found in graph."}

        inv_amount = inv_node.get("properties", {}).get("amount_ht", 0.0)
        report: Dict[str, Any] = {
            "invoice_id": invoice_id,
            "invoice_amount_ht": inv_amount,
            "has_po": False,
            "po_id": None,
            "po_amount_ht": None,
            "amount_diff": 0.0,
            "exceeds_po": False,
            "contract_id": None,
            "contract_ceiling": None,
            "amendment_id": None,
            "effective_ceiling": None,
            "exceeds_contract": False,
            "justified_by_amendment": False,
            "evidence": [inv_node.get("evidence", "")],
        }

        # 1. Hop 1: Find associated Purchase Order (REFERS_TO)
        po_id = None
        for rel in self.kg.get_outgoing_relations(invoice_id):
            if rel["relation_type"] == "REFERS_TO":
                po_id = rel["target_id"]
                report["evidence"].append(rel.get("evidence", ""))
                break

        if not po_id:
            return report

        po_node = self.kg.get_node(po_id)
        report["has_po"] = True
        report["po_id"] = po_id

        if po_node:
            po_amount = po_node.get("properties", {}).get("amount_ht", 0.0)
            report["po_amount_ht"] = po_amount
            report["amount_diff"] = round(inv_amount - po_amount, 2)
            report["exceeds_po"] = report["amount_diff"] > 0

        # 2. Hop 2: Find governing Contract (GOVERNED_BY)
        con_id = None
        for rel in self.kg.get_outgoing_relations(po_id):
            if rel["relation_type"] == "GOVERNED_BY":
                con_id = rel["target_id"]
                report["evidence"].append(rel.get("evidence", ""))
                break

        if not con_id:
            return report

        con_node = self.kg.get_node(con_id)
        report["contract_id"] = con_id
        initial_ceiling = con_node.get("properties", {}).get("maximum_amount") if con_node else None
        report["contract_ceiling"] = initial_ceiling
        effective_ceiling = initial_ceiling

        # 3. Hop 3: Check for Amendments modifying this contract (MODIFIES)
        # Search all incoming MODIFIES edges to this contract
        for src_node, _, edge_data in self.kg.graph.in_edges(con_id, data=True):
            if edge_data.get("relation_type") == "MODIFIES":
                amend_node = self.kg.get_node(src_node)
                if amend_node:
                    new_ceiling = amend_node.get("properties", {}).get("new_maximum_amount")
                    if new_ceiling:
                        effective_ceiling = new_ceiling
                        report["amendment_id"] = src_node
                        report["evidence"].append(edge_data.get("evidence", ""))
                        break

        report["effective_ceiling"] = effective_ceiling
        if effective_ceiling is not None:
            report["exceeds_contract"] = inv_amount > effective_ceiling
            if initial_ceiling is not None:
                report["justified_by_amendment"] = (
                    inv_amount > initial_ceiling and inv_amount <= effective_ceiling
                )

        return report
