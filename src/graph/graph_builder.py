# Builds and manages the directed knowledge graph of documents and relations.

import os
import json
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path
import networkx as nx
from src.extraction.schema import ExtractedEntity, ExtractedRelation
from src.resolution.normalizer import is_same_company


class KnowledgeGraph:
    """Directed knowledge graph storing document entities and cross-document relationships."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._lock = threading.RLock()

    def add_entity(self, entity: ExtractedEntity):
        """Adds a document entity as a graph node."""
        with self._lock:
            self.graph.add_node(
                entity.id,
                entity_type=entity.entity_type,
                name=entity.name,
                properties=entity.properties,
                evidence=entity.evidence,
                page_number=entity.page_number,
                chunk_id=entity.chunk_id,
            )

    def add_relation(self, relation: ExtractedRelation):
        """Adds a directed relationship between two entities."""
        with self._lock:
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
                chunk_id=relation.chunk_id,
            )

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves entity node data by identifier."""
        with self._lock:
            if node_id in self.graph:
                return dict(self.graph.nodes[node_id])
            return None

    def get_outgoing_relations(self, node_id: str) -> List[Dict[str, Any]]:
        """Returns all outgoing relationships from a node."""
        with self._lock:
            if node_id not in self.graph:
                return []
            results = []
            for _, target_id, edge_data in self.graph.out_edges(node_id, data=True):
                results.append({
                    "target_id": target_id,
                    "relation_type": edge_data.get("relation_type"),
                    "evidence": edge_data.get("evidence"),
                    "chunk_id": edge_data.get("chunk_id"),
                })
            return results

    def find_company_entity(self, name: str, siren: Optional[str] = None) -> Optional[str]:
        """Finds an existing company node matching name or SIREN via entity normalization."""
        with self._lock:
            for node_id, data in self.graph.nodes(data=True):
                if data.get("entity_type") in ("Company", "Supplier"):
                    node_name = data.get("name") or str(node_id)
                    node_siren = data.get("properties", {}).get("siren")
                    if is_same_company(name, siren, node_name, node_siren):
                        return node_id
            return None

    def trace_invoice_to_contract(self, invoice_id: str) -> Optional[str]:
        """Multi-hop traversal: Invoice -> PO -> Contract OR direct Invoice -> Contract."""
        with self._lock:
            if invoice_id not in self.graph:
                return None

            # 1. Direct link: Invoice -> Contract
            for _, target_id, edge_data in self.graph.out_edges(invoice_id, data=True):
                rel_type = edge_data.get("relation_type")
                target_node = self.get_node(target_id)
                target_type = target_node.get("entity_type") if target_node else ""
                if rel_type in ("GOVERNED_BY", "REFERS_TO") and (target_type == "Contract" or target_id.startswith("CTR-") or target_id.startswith("CON-")):
                    return target_id

            # 2. Hop 1: Invoice -> PO
            po_id = None
            for _, target_id, edge_data in self.graph.out_edges(invoice_id, data=True):
                if edge_data.get("relation_type") == "REFERS_TO":
                    po_id = target_id
                    break

            if not po_id or po_id not in self.graph:
                return None

            # 3. Hop 2: PO -> Contract
            for _, target_id, edge_data in self.graph.out_edges(po_id, data=True):
                if edge_data.get("relation_type") == "GOVERNED_BY":
                    return target_id

            return None

    def save_to_disk(self, file_path: str = "data/knowledge_graph.json"):
        """Serializes the graph topology and node properties to JSON."""
        with self._lock:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = nx.node_link_data(self.graph)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_disk(self, file_path: str = "data/knowledge_graph.json") -> bool:
        """Loads a previously serialized graph from disk. Returns True if successful."""
        with self._lock:
            path = Path(file_path)
            if not path.exists():
                return False
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data, directed=True)
                return True
            except Exception:
                return False
