from src.graph.graph_builder import KnowledgeGraph
from src.retrieval.graph_retriever import GraphRetriever
from src.extraction.schema import ExtractedEntity, ExtractedRelation


def test_audit_invoice_scenarios():
    kg = KnowledgeGraph()

    # Scenario 2: Invoice exceeds PO (Beta Logistics)
    kg.add_entity(ExtractedEntity(
        id="FAC-2026-002",
        entity_type="Invoice",
        name="Invoice Beta",
        properties={"amount_ht": 7500.0},
        evidence="FACTURE FAC-2026-002",
        page_number=1,
    ))
    kg.add_entity(ExtractedEntity(
        id="BC-2026-002",
        entity_type="PurchaseOrder",
        name="PO Beta",
        properties={"amount_ht": 5000.0},
        evidence="BON DE COMMANDE BC-2026-002",
        page_number=1,
    ))
    kg.add_relation(ExtractedRelation(
        source_id="FAC-2026-002",
        relation_type="REFERS_TO",
        target_id="BC-2026-002",
        evidence="Réf BC-2026-002",
    ))

    retriever = GraphRetriever(kg)
    report2 = retriever.audit_invoice("FAC-2026-002")

    assert report2["has_po"] is True
    assert report2["exceeds_po"] is True
    assert report2["amount_diff"] == 2500.0

    # Scenario 3: Spend justified by an amendment (Gamma Cloud)
    kg.add_entity(ExtractedEntity(
        id="FAC-2026-003",
        entity_type="Invoice",
        name="Invoice Gamma",
        properties={"amount_ht": 28000.0},
        evidence="FACTURE FAC-2026-003",
        page_number=1,
    ))
    kg.add_entity(ExtractedEntity(
        id="BC-2026-003",
        entity_type="PurchaseOrder",
        name="PO Gamma",
        properties={"amount_ht": 28000.0},
        evidence="BON DE COMMANDE BC-2026-003",
        page_number=1,
    ))
    kg.add_entity(ExtractedEntity(
        id="CTR-2026-GAMMA",
        entity_type="Contract",
        name="Contract Gamma",
        properties={"maximum_amount": 20000.0},
        evidence="Contrat CTR-2026-GAMMA",
        page_number=1,
    ))
    kg.add_entity(ExtractedEntity(
        id="AVN-2026-001",
        entity_type="Amendment",
        name="Amendment Gamma",
        properties={"new_maximum_amount": 40000.0},
        evidence="Avenant AVN-2026-001",
        page_number=1,
    ))

    kg.add_relation(ExtractedRelation(
        source_id="FAC-2026-003", relation_type="REFERS_TO", target_id="BC-2026-003", evidence="Réf BC-2026-003"
    ))
    kg.add_relation(ExtractedRelation(
        source_id="BC-2026-003", relation_type="GOVERNED_BY", target_id="CTR-2026-GAMMA", evidence="Réf CTR-2026-GAMMA"
    ))
    kg.add_relation(ExtractedRelation(
        source_id="AVN-2026-001", relation_type="MODIFIES", target_id="CTR-2026-GAMMA", evidence="Modifie CTR-2026-GAMMA"
    ))

    report3 = retriever.audit_invoice("FAC-2026-003")
    assert report3["exceeds_po"] is False
    assert report3["contract_ceiling"] == 20000.0
    assert report3["effective_ceiling"] == 40000.0
    assert report3["justified_by_amendment"] is True
    assert report3["exceeds_contract"] is False


def test_audit_orphan_invoice():
    kg = KnowledgeGraph()
    kg.add_entity(ExtractedEntity(
        id="FAC-2026-004",
        entity_type="Invoice",
        name="Invoice Delta",
        properties={"amount_ht": 3000.0},
        evidence="FACTURE FAC-2026-004",
        page_number=1,
    ))
    retriever = GraphRetriever(kg)
    report4 = retriever.audit_invoice("FAC-2026-004")

    assert report4["has_po"] is False
    assert report4["po_id"] is None
