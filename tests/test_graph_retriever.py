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


def test_audit_multiple_amendments_deterministic_order():
    """Verify that multiple amendments are resolved deterministically by date/ID."""
    kg = KnowledgeGraph()
    kg.add_entity(ExtractedEntity(
        id="FAC-MULTI-01",
        entity_type="Invoice",
        name="Invoice Multi",
        properties={"amount_ht": 45000.0},
        evidence="FACTURE FAC-MULTI-01",
        page_number=1,
    ))
    kg.add_entity(ExtractedEntity(
        id="BC-MULTI-01",
        entity_type="PurchaseOrder",
        name="PO Multi",
        properties={"amount_ht": 45000.0},
        evidence="BON DE COMMANDE BC-MULTI-01",
        page_number=1,
    ))
    kg.add_entity(ExtractedEntity(
        id="CON-MULTI-01",
        entity_type="Contract",
        name="Contract Multi",
        properties={"maximum_amount": 20000.0},
        evidence="Contrat CON-MULTI-01",
        page_number=1,
    ))
    # Amendment 1 sets 30k
    kg.add_entity(ExtractedEntity(
        id="AVN-2026-001",
        entity_type="Amendment",
        name="Amendment 1",
        properties={"new_maximum_amount": 30000.0, "date": "2026-03-01"},
        evidence="Avenant 1",
        page_number=1,
    ))
    # Amendment 2 sets 50k
    kg.add_entity(ExtractedEntity(
        id="AVN-2026-002",
        entity_type="Amendment",
        name="Amendment 2",
        properties={"new_maximum_amount": 50000.0, "date": "2026-06-01"},
        evidence="Avenant 2",
        page_number=1,
    ))

    kg.add_relation(ExtractedRelation(
        source_id="FAC-MULTI-01", relation_type="REFERS_TO", target_id="BC-MULTI-01", evidence="Ref BC"
    ))
    kg.add_relation(ExtractedRelation(
        source_id="BC-MULTI-01", relation_type="GOVERNED_BY", target_id="CON-MULTI-01", evidence="Ref CON"
    ))
    # Add in reverse order to ensure graph ordering does not affect outcome
    kg.add_relation(ExtractedRelation(
        source_id="AVN-2026-002", relation_type="MODIFIES", target_id="CON-MULTI-01", evidence="Modifies CON"
    ))
    kg.add_relation(ExtractedRelation(
        source_id="AVN-2026-001", relation_type="MODIFIES", target_id="CON-MULTI-01", evidence="Modifies CON"
    ))

    retriever = GraphRetriever(kg)
    report = retriever.audit_invoice("FAC-MULTI-01")

    assert report["contract_ceiling"] == 20000.0
    assert report["effective_ceiling"] == 50000.0
    assert report["amendment_id"] == "AVN-2026-002"
    assert report["applied_amendments"] == ["AVN-2026-001", "AVN-2026-002"]
    assert report["exceeds_contract"] is False
    assert report["justified_by_amendment"] is True


def test_audit_cumulative_contract_budget_burn():
    """Verify that multiple invoices on the same contract trigger a cumulative overrun."""
    kg = KnowledgeGraph()
    # Contract ceiling: 30 000 €
    kg.add_entity(ExtractedEntity(
        id="CON-CUMUL",
        entity_type="Contract",
        name="Contract Cumul",
        properties={"maximum_amount": 30000.0},
        evidence="Contrat CON-CUMUL",
        page_number=1,
    ))
    kg.add_entity(ExtractedEntity(
        id="BC-CUMUL-01",
        entity_type="PurchaseOrder",
        name="PO 1",
        properties={"amount_ht": 20000.0},
        evidence="BC 1",
        page_number=1,
    ))
    kg.add_entity(ExtractedEntity(
        id="BC-CUMUL-02",
        entity_type="PurchaseOrder",
        name="PO 2",
        properties={"amount_ht": 20000.0},
        evidence="BC 2",
        page_number=1,
    ))
    kg.add_relation(ExtractedRelation(source_id="BC-CUMUL-01", relation_type="GOVERNED_BY", target_id="CON-CUMUL", evidence="Ref CON-CUMUL"))
    kg.add_relation(ExtractedRelation(source_id="BC-CUMUL-02", relation_type="GOVERNED_BY", target_id="CON-CUMUL", evidence="Ref CON-CUMUL"))

    # Invoice 1: 18 000 € (alone < 30 000 €)
    kg.add_entity(ExtractedEntity(
        id="FAC-01",
        entity_type="Invoice",
        name="Invoice 1",
        properties={"amount_ht": 18000.0},
        evidence="FAC-01",
        page_number=1,
    ))
    kg.add_relation(ExtractedRelation(source_id="FAC-01", relation_type="REFERS_TO", target_id="BC-CUMUL-01", evidence="Ref BC-CUMUL-01"))

    # Invoice 2: 15 000 € (alone < 30 000 €, but 18k + 15k = 33k > 30k)
    kg.add_entity(ExtractedEntity(
        id="FAC-02",
        entity_type="Invoice",
        name="Invoice 2",
        properties={"amount_ht": 15000.0},
        evidence="FAC-02",
        page_number=1,
    ))
    kg.add_relation(ExtractedRelation(source_id="FAC-02", relation_type="REFERS_TO", target_id="BC-CUMUL-02", evidence="Ref BC-CUMUL-02"))

    retriever = GraphRetriever(kg)
    report1 = retriever.audit_invoice("FAC-01")
    report2 = retriever.audit_invoice("FAC-02")

    # Cumulative spend on CON-CUMUL should be 33 000 €
    assert report1["cumulative_contract_amount"] == 33000.0
    assert report2["cumulative_contract_amount"] == 33000.0
    assert report2["exceeds_contract"] is True
    assert report2["exceeds_contract_cumulative"] is True
    assert report2["contract_amount_diff"] == 3000.0


def test_audit_direct_invoice_to_contract_without_po():
    """Verify that an invoice linked directly to a contract without PO is correctly audited."""
    kg = KnowledgeGraph()
    kg.add_entity(ExtractedEntity(
        id="CON-DIRECT",
        entity_type="Contract",
        name="Contract Direct",
        properties={"maximum_amount": 10000.0},
        evidence="Contrat CON-DIRECT",
        page_number=1,
    ))
    kg.add_entity(ExtractedEntity(
        id="FAC-DIRECT",
        entity_type="Invoice",
        name="Invoice Direct",
        properties={"amount_ht": 12000.0},
        evidence="FAC-DIRECT",
        page_number=1,
    ))
    kg.add_relation(ExtractedRelation(
        source_id="FAC-DIRECT",
        relation_type="GOVERNED_BY",
        target_id="CON-DIRECT",
        evidence="Contrat de rattachement CON-DIRECT",
    ))

    retriever = GraphRetriever(kg)
    report = retriever.audit_invoice("FAC-DIRECT")

    assert report["has_po"] is False
    assert report["contract_id"] == "CON-DIRECT"
    assert report["contract_ceiling"] == 10000.0
    assert report["cumulative_contract_amount"] == 12000.0
    assert report["exceeds_contract"] is True

