# Builds and manages the directed knowledge graph of documents and relations.

from typing import Dict, List, Optional, Any
import networkx as nx
from src.extraction.schema import ExtractedEntity, ExtractedRelation


class KnowledgeGraph:
    """Directed knowledge graph storing document entities and cross-document relationships."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity: ExtractedEntity):
        """Adds a document entity as a graph node."""
        self.graph.add_node(
            entity.id,
            entity_type=entity.entity_type,
            name=entity.name,
            properties=entity.properties,
            evidence=entity.evidence,
            page_number=entity.page_number,
        )

    def add_relation(self, relation: ExtractedRelation):
        """Adds a directed relationship between two entities."""
        # Ensure endpoints exist in graph
        if relation.source_id not in self.graph:
            self.graph.add_node(relation.source_id, entity_type="Unknown", properties={})
        if relation.target_id not in self.graph:
            self.graph.add_node(relation.target_id, entity_type="Unknown", properties={})

        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            relation_type=relation.relation_type,
            evidence=relation.evidence,
            confidence=relation.confidence,
        )

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves entity node data by identifier."""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None

    def get_outgoing_relations(self, node_id: str) -> List[Dict[str, Any]]:
        """Returns all outgoing relationships from a node."""
        if node_id not in self.graph:
            return []
        results = []
        for _, target_id, edge_data in self.graph.out_edges(node_id, data=True):
            results.append({
                "target_id": target_id,
                "relation_type": edge_data.get("relation_type"),
                "evidence": edge_data.get("evidence"),
            })
        return results

    def trace_invoice_to_contract(self, invoice_id: str) -> Optional[str]:
        """Multi-hop traversal: Invoice -> REFERS_TO -> PurchaseOrder -> GOVERNED_BY -> Contract."""
        if invoice_id not in self.graph:
            return None

        # 1. Hop 1: Invoice -> PO
        po_id = None
        for _, target_id, edge_data in self.graph.out_edges(invoice_id, data=True):
            if edge_data.get("relation_type") == "REFERS_TO":
                po_id = target_id
                break

        if not po_id or po_id not in self.graph:
            return None

        # 2. Hop 2: PO -> Contract
        for _, target_id, edge_data in self.graph.out_edges(po_id, data=True):
            if edge_data.get("relation_type") == "GOVERNED_BY":
                return target_id

        return None
