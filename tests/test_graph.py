from src.graph.graph_builder import KnowledgeGraph
from src.extraction.schema import ExtractedEntity, ExtractedRelation


def test_build_and_traverse_graph():
    kg = KnowledgeGraph()

    # 1. Add entities
    inv = ExtractedEntity(
        id="FAC-2026-001",
        entity_type="Invoice",
        name="Invoice FAC-2026-001",
        properties={"amount_ht": 10000.0},
        evidence="FACTURE FAC-2026-001",
        page_number=1,
    )
    po = ExtractedEntity(
        id="BC-2026-001",
        entity_type="PurchaseOrder",
        name="Purchase Order BC-2026-001",
        properties={"amount_ht": 10000.0},
        evidence="BON DE COMMANDE BC-2026-001",
        page_number=1,
    )
    con = ExtractedEntity(
        id="CTR-2026-ALPHA",
        entity_type="Contract",
        name="Contract CTR-2026-ALPHA",
        properties={"maximum_amount": 50000.0},
        evidence="Référence CTR-2026-ALPHA",
        page_number=1,
    )

    kg.add_entity(inv)
    kg.add_entity(po)
    kg.add_entity(con)

    # 2. Add relationships
    kg.add_relation(ExtractedRelation(
        source_id="FAC-2026-001",
        relation_type="REFERS_TO",
        target_id="BC-2026-001",
        evidence="Bon de commande associé : BC-2026-001",
    ))
    kg.add_relation(ExtractedRelation(
        source_id="BC-2026-001",
        relation_type="GOVERNED_BY",
        target_id="CTR-2026-ALPHA",
        evidence="Contrat cadre : CTR-2026-ALPHA",
    ))

    # 3. Test node properties
    node = kg.get_node("FAC-2026-001")
    assert node is not None
    assert node["properties"]["amount_ht"] == 10000.0

    # 4. Multi-hop traversal test: Invoice -> Contract
    contract_id = kg.trace_invoice_to_contract("FAC-2026-001")
    assert contract_id == "CTR-2026-ALPHA"

    # 5. Non-existent or orphan invoice
    assert kg.trace_invoice_to_contract("NON_EXISTENT") is None


def test_trace_invoice_direct_to_contract_without_po():
    """Verify that an invoice directly linked to a contract without PO can be traced."""
    kg = KnowledgeGraph()
    kg.add_entity(ExtractedEntity(
        id="FAC-DIRECT-01",
        entity_type="Invoice",
        name="Invoice Direct",
        properties={"amount_ht": 5000.0},
        evidence="FAC-DIRECT-01",
        page_number=1,
    ))
    kg.add_entity(ExtractedEntity(
        id="CTR-DIRECT-01",
        entity_type="Contract",
        name="Contract Direct",
        properties={"maximum_amount": 20000.0},
        evidence="CTR-DIRECT-01",
        page_number=1,
    ))
    kg.add_relation(ExtractedRelation(
        source_id="FAC-DIRECT-01",
        relation_type="GOVERNED_BY",
        target_id="CTR-DIRECT-01",
        evidence="Direct link to contract",
    ))

    traced_contract = kg.trace_invoice_to_contract("FAC-DIRECT-01")
    assert traced_contract == "CTR-DIRECT-01"


def test_knowledge_graph_persistence(tmp_path):
    """Verify that KnowledgeGraph can be saved to and loaded from disk."""
    kg = KnowledgeGraph()
    kg.add_entity(ExtractedEntity(
        id="CTR-PERSIST",
        entity_type="Contract",
        name="Contract Persist",
        properties={"maximum_amount": 10000.0},
        evidence="Evidence 1",
        page_number=1,
    ))
    save_file = str(tmp_path / "kg_test.json")
    kg.save_to_disk(save_file)

    new_kg = KnowledgeGraph()
    loaded = new_kg.load_from_disk(save_file)
    assert loaded is True
    assert len(new_kg.graph.nodes) == 1
    assert new_kg.get_node("CTR-PERSIST")["properties"]["maximum_amount"] == 10000.0

