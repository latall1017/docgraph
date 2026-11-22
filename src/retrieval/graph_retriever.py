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
            "cumulative_po_amount": inv_amount,
            "exceeds_po_cumulative": False,
            "po_invoices": [invoice_id],
            "contract_id": None,
            "contract_ceiling": None,
            "amendment_id": None,
            "applied_amendments": [],
            "effective_ceiling": None,
            "cumulative_contract_amount": inv_amount,
            "contract_amount_diff": 0.0,
            "exceeds_contract": False,
            "exceeds_contract_cumulative": False,
            "contract_invoices": [invoice_id],
            "justified_by_amendment": False,
            "evidence": [inv_node.get("evidence", "")] if inv_node.get("evidence") else [],
        }

        # 1. Hop 1: Find associated Purchase Order (REFERS_TO) or direct Contract
        po_id = None
        con_id = None

        for rel in self.kg.get_outgoing_relations(invoice_id):
            target_node = self.kg.get_node(rel["target_id"])
            rel_type = rel.get("relation_type")
            target_type = target_node.get("entity_type") if target_node else None

            if rel_type == "REFERS_TO" and (target_type == "PurchaseOrder" or rel["target_id"].startswith("BC-") or not target_type):
                po_id = rel["target_id"]
                if rel.get("evidence"):
                    report["evidence"].append(rel["evidence"])
                break
            elif rel_type in ("GOVERNED_BY", "REFERS_TO") and (target_type == "Contract" or rel["target_id"].startswith("CON-") or rel["target_id"].startswith("CTR-")):
                con_id = rel["target_id"]
                if rel.get("evidence"):
                    report["evidence"].append(rel["evidence"])

        # If no PO was found and no direct contract link
        if not po_id and not con_id:
            return report

        # Process PO if available
        if po_id:
            po_node = self.kg.get_node(po_id)
            report["has_po"] = True
            report["po_id"] = po_id

            if po_node:
                po_amount = po_node.get("properties", {}).get("amount_ht", 0.0)
                report["po_amount_ht"] = po_amount
                report["amount_diff"] = round(inv_amount - po_amount, 2)
                report["exceeds_po"] = report["amount_diff"] > 0

                # Compute cumulative PO spend across all invoices pointing to this PO
                po_invoices = set()
                total_billed_po = 0.0
                for u, _, edge_data in self.kg.graph.in_edges(po_id, data=True):
                    if edge_data.get("relation_type") == "REFERS_TO":
                        po_invoices.add(u)
                        u_node = self.kg.get_node(u)
                        if u_node:
                            total_billed_po += u_node.get("properties", {}).get("amount_ht", 0.0)

                po_invoices.add(invoice_id)
                if inv_amount > 0 and total_billed_po == 0.0:
                    total_billed_po = inv_amount

                report["cumulative_po_amount"] = round(total_billed_po, 2)
                report["po_invoices"] = sorted(list(po_invoices))
                if po_amount > 0:
                    report["exceeds_po_cumulative"] = round(total_billed_po - po_amount, 2) > 0

            # Hop 2: Find governing Contract (GOVERNED_BY) from PO
            if not con_id:
                for rel in self.kg.get_outgoing_relations(po_id):
                    if rel["relation_type"] == "GOVERNED_BY":
                        con_id = rel["target_id"]
                        if rel.get("evidence"):
                            report["evidence"].append(rel["evidence"])
                        break

        if not con_id:
            return report

        con_node = self.kg.get_node(con_id)
        report["contract_id"] = con_id
        initial_ceiling = con_node.get("properties", {}).get("maximum_amount") if con_node else None
        report["contract_ceiling"] = initial_ceiling
        effective_ceiling = initial_ceiling

        # 3. Hop 3: Check for Amendments modifying this contract (MODIFIES)
        # Collect and sort all amendments deterministically (by date if present, then by ID)
        amendments = []
        for src_node, _, edge_data in self.kg.graph.in_edges(con_id, data=True):
            if edge_data.get("relation_type") == "MODIFIES":
                amend_node = self.kg.get_node(src_node)
                if amend_node:
                    props = amend_node.get("properties", {})
                    new_ceiling = props.get("new_maximum_amount")
                    amend_date = str(props.get("date") or "")
                    amendments.append({
                        "id": src_node,
                        "date": amend_date,
                        "new_ceiling": new_ceiling,
                        "evidence": edge_data.get("evidence", ""),
                    })

        if amendments:
            # Sort deterministically: chronological date or lexicographical ID
            amendments.sort(key=lambda a: (a["date"], a["id"]))
            report["applied_amendments"] = [a["id"] for a in amendments]

            for a in amendments:
                if a["new_ceiling"] is not None:
                    effective_ceiling = a["new_ceiling"]
                    report["amendment_id"] = a["id"]
                if a["evidence"] and a["evidence"] not in report["evidence"]:
                    report["evidence"].append(a["evidence"])

        report["effective_ceiling"] = effective_ceiling

        # 4. Hop 4: Compute cumulative spend on this contract across all POs & invoices
        contract_invoices = set()
        for u, _, edge_data in self.kg.graph.in_edges(con_id, data=True):
            rel_type = edge_data.get("relation_type")
            if rel_type == "GOVERNED_BY":
                u_node = self.kg.get_node(u)
                if u_node and u_node.get("entity_type") == "PurchaseOrder":
                    for inv, _, inv_edge in self.kg.graph.in_edges(u, data=True):
                        if inv_edge.get("relation_type") == "REFERS_TO":
                            contract_invoices.add(inv)
                elif u_node and u_node.get("entity_type") == "Invoice":
                    contract_invoices.add(u)
            elif rel_type == "REFERS_TO":
                contract_invoices.add(u)

        contract_invoices.add(invoice_id)
        total_billed_contract = 0.0
        for inv in contract_invoices:
            inv_entity = self.kg.get_node(inv)
            if inv_entity:
                total_billed_contract += inv_entity.get("properties", {}).get("amount_ht", 0.0)

        report["cumulative_contract_amount"] = round(total_billed_contract, 2)
        report["contract_invoices"] = sorted(list(contract_invoices))

        if effective_ceiling is not None:
            report["contract_amount_diff"] = round(total_billed_contract - effective_ceiling, 2)
            report["exceeds_contract"] = (inv_amount > effective_ceiling) or (total_billed_contract > effective_ceiling)
            report["exceeds_contract_cumulative"] = total_billed_contract > effective_ceiling

            if initial_ceiling is not None:
                report["justified_by_amendment"] = (
                    (inv_amount > initial_ceiling and inv_amount <= effective_ceiling) or
                    (total_billed_contract > initial_ceiling and total_billed_contract <= effective_ceiling)
                )

        return report

    def audit_po(self, po_id: str) -> Dict[str, Any]:
        """Audits a purchase order against its governing contract and referencing invoices."""
        po_node = self.kg.get_node(po_id)
        if not po_node:
            return {"status": "not_found", "message": f"Purchase Order {po_id} not found in graph."}

        po_amount = po_node.get("properties", {}).get("amount_ht", 0.0)
        report: Dict[str, Any] = {
            "entity_type": "PurchaseOrder",
            "po_id": po_id,
            "po_amount_ht": po_amount,
            "contract_id": None,
            "contract_ceiling": None,
            "effective_ceiling": None,
            "applied_amendments": [],
            "exceeds_contract": False,
            "invoices": [],
            "total_invoiced_ht": 0.0,
            "amount_diff": 0.0,
            "exceeds_po": False,
            "evidence": [po_node.get("evidence", "")] if po_node.get("evidence") else [],
        }

        # 1. Hop 1: Governing Contract (GOVERNED_BY)
        con_id = None
        for rel in self.kg.get_outgoing_relations(po_id):
            if rel.get("relation_type") == "GOVERNED_BY":
                con_id = rel["target_id"]
                if rel.get("evidence"):
                    report["evidence"].append(rel["evidence"])
                break

        if con_id:
            con_node = self.kg.get_node(con_id)
            report["contract_id"] = con_id
            initial_ceiling = con_node.get("properties", {}).get("maximum_amount") if con_node else None
            report["contract_ceiling"] = initial_ceiling
            effective_ceiling = initial_ceiling

            # Check amendments modifying this contract
            amendments = []
            for src_node, _, edge_data in self.kg.graph.in_edges(con_id, data=True):
                if edge_data.get("relation_type") == "MODIFIES":
                    amend_node = self.kg.get_node(src_node)
                    if amend_node:
                        props = amend_node.get("properties", {})
                        new_ceiling = props.get("new_maximum_amount")
                        amend_date = str(props.get("date") or "")
                        amendments.append({
                            "id": src_node,
                            "date": amend_date,
                            "new_ceiling": new_ceiling,
                            "evidence": edge_data.get("evidence", ""),
                        })

            if amendments:
                amendments.sort(key=lambda a: (a["date"], a["id"]))
                report["applied_amendments"] = [a["id"] for a in amendments]
                for a in amendments:
                    if a["new_ceiling"] is not None:
                        effective_ceiling = a["new_ceiling"]
                    if a["evidence"] and a["evidence"] not in report["evidence"]:
                        report["evidence"].append(a["evidence"])

            report["effective_ceiling"] = effective_ceiling
            if effective_ceiling is not None:
                report["exceeds_contract"] = po_amount > effective_ceiling

        # 2. Invoices referencing this PO (incoming REFERS_TO)
        invoices = []
        total_invoiced = 0.0
        for u, _, edge_data in self.kg.graph.in_edges(po_id, data=True):
            if edge_data.get("relation_type") == "REFERS_TO":
                invoices.append(u)
                u_node = self.kg.get_node(u)
                if u_node:
                    total_invoiced += u_node.get("properties", {}).get("amount_ht", 0.0)
                if edge_data.get("evidence") and edge_data["evidence"] not in report["evidence"]:
                    report["evidence"].append(edge_data["evidence"])

        report["invoices"] = sorted(invoices)
        report["total_invoiced_ht"] = round(total_invoiced, 2)
        report["amount_diff"] = round(total_invoiced - po_amount, 2)
        report["exceeds_po"] = report["amount_diff"] > 0

        return report

    def get_contract_summary(self, contract_id: str) -> Dict[str, Any]:
        """Summarizes a contract, its amendments, associated purchase orders, and total spend."""
        con_node = self.kg.get_node(contract_id)
        if not con_node:
            return {"status": "not_found", "message": f"Contract {contract_id} not found in graph."}

        initial_ceiling = con_node.get("properties", {}).get("maximum_amount", 0.0)
        report: Dict[str, Any] = {
            "entity_type": "Contract",
            "contract_id": contract_id,
            "initial_ceiling": initial_ceiling,
            "effective_ceiling": initial_ceiling,
            "applied_amendments": [],
            "purchase_orders": [],
            "total_po_committed": 0.0,
            "invoices": [],
            "total_invoiced": 0.0,
            "remaining_budget": initial_ceiling,
            "exceeds_ceiling": False,
            "evidence": [con_node.get("evidence", "")] if con_node.get("evidence") else [],
        }

        # 1. Amendments modifying this contract (MODIFIES)
        amendments = []
        effective_ceiling = initial_ceiling
        for src_node, _, edge_data in self.kg.graph.in_edges(contract_id, data=True):
            if edge_data.get("relation_type") == "MODIFIES":
                amend_node = self.kg.get_node(src_node)
                if amend_node:
                    props = amend_node.get("properties", {})
                    new_ceiling = props.get("new_maximum_amount")
                    amend_date = str(props.get("date") or "")
                    amendments.append({
                        "id": src_node,
                        "date": amend_date,
                        "new_ceiling": new_ceiling,
                        "evidence": edge_data.get("evidence", ""),
                    })

        if amendments:
            amendments.sort(key=lambda a: (a["date"], a["id"]))
            report["applied_amendments"] = [a["id"] for a in amendments]
            for a in amendments:
                if a["new_ceiling"] is not None:
                    effective_ceiling = a["new_ceiling"]
                if a["evidence"] and a["evidence"] not in report["evidence"]:
                    report["evidence"].append(a["evidence"])

        report["effective_ceiling"] = effective_ceiling

        # 2. Purchase orders governed by this contract (GOVERNED_BY)
        po_ids = set()
        total_po = 0.0
        contract_invoices = set()

        for u, _, edge_data in self.kg.graph.in_edges(contract_id, data=True):
            rel_type = edge_data.get("relation_type")
            if rel_type == "GOVERNED_BY":
                u_node = self.kg.get_node(u)
                if u_node and u_node.get("entity_type") == "PurchaseOrder":
                    po_ids.add(u)
                    total_po += u_node.get("properties", {}).get("amount_ht", 0.0)
                    if edge_data.get("evidence") and edge_data["evidence"] not in report["evidence"]:
                        report["evidence"].append(edge_data["evidence"])

                    # Also collect invoices on this PO
                    for inv, _, inv_edge in self.kg.graph.in_edges(u, data=True):
                        if inv_edge.get("relation_type") == "REFERS_TO":
                            contract_invoices.add(inv)
                elif u_node and u_node.get("entity_type") == "Invoice":
                    contract_invoices.add(u)
            elif rel_type == "REFERS_TO":
                contract_invoices.add(u)

        report["purchase_orders"] = sorted(list(po_ids))
        report["total_po_committed"] = round(total_po, 2)

        # 3. Sum total invoiced
        total_invoiced = 0.0
        for inv in contract_invoices:
            inv_node = self.kg.get_node(inv)
            if inv_node:
                total_invoiced += inv_node.get("properties", {}).get("amount_ht", 0.0)

        report["invoices"] = sorted(list(contract_invoices))
        report["total_invoiced"] = round(total_invoiced, 2)

        if effective_ceiling is not None:
            report["remaining_budget"] = round(effective_ceiling - total_invoiced, 2)
            report["exceeds_ceiling"] = total_invoiced > effective_ceiling

        return report
