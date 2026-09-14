from datetime import date
from typing import List, Optional
from pydantic import BaseModel
from src.schemas import Company, Contract, Amendment, PurchaseOrder, Invoice, LineItem


class ProcurementPackage(BaseModel):
    """A complete document package for one supplier with known ground truth."""
    supplier: Company
    contract: Optional[Contract] = None
    amendments: List[Amendment] = []
    purchase_orders: List[PurchaseOrder] = []
    invoices: List[Invoice] = []
    expected_anomalies: List[str] = []


def build_sample_packages() -> List[ProcurementPackage]:
    """
    Builds a small set of controlled packages representing key business scenarios:
    1. Normal compliant flow
    2. Invoice exceeding purchase order
    3. Spend exceeding initial contract, justified by an amendment
    4. Invoice without associated purchase order
    """
    packages = []

    # Scenario 1: Normal compliant package
    sup1 = Company(
        id="SUP-001",
        name="Alpha Services SAS",
        siren="111222333",
        address="10 Rue de Paris, 75001 Paris"
    )
    con1 = Contract(
        id="CON-001",
        contract_ref="CTR-2026-ALPHA",
        supplier_id=sup1.id,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
        maximum_amount=50000.0
    )
    po1 = PurchaseOrder(
        id="PO-001",
        po_ref="BC-2026-001",
        supplier_id=sup1.id,
        contract_id=con1.id,
        date=date(2026, 2, 10),
        items=[LineItem(description="Software maintenance", quantity=1.0, unit_price=10000.0, total_price=10000.0)],
        total_amount_ht=10000.0
    )
    inv1 = Invoice(
        id="INV-001",
        invoice_ref="FAC-2026-001",
        supplier_id=sup1.id,
        po_ref=po1.po_ref,
        date=date(2026, 3, 1),
        items=[LineItem(description="Software maintenance", quantity=1.0, unit_price=10000.0, total_price=10000.0)],
        total_amount_ht=10000.0,
        vat_rate=0.20,
        total_amount_ttc=12000.0
    )
    packages.append(ProcurementPackage(
        supplier=sup1,
        contract=con1,
        purchase_orders=[po1],
        invoices=[inv1],
        expected_anomalies=[]
    ))

    # Scenario 2: Invoice exceeds purchase order
    sup2 = Company(
        id="SUP-002",
        name="Beta Logistics SARL",
        siren="444555666",
        address="25 Avenue des Fleurs, 69002 Lyon"
    )
    con2 = Contract(
        id="CON-002",
        contract_ref="CTR-2026-BETA",
        supplier_id=sup2.id,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
        maximum_amount=30000.0
    )
    po2 = PurchaseOrder(
        id="PO-002",
        po_ref="BC-2026-002",
        supplier_id=sup2.id,
        contract_id=con2.id,
        date=date(2026, 2, 15),
        items=[LineItem(description="Warehouse transport", quantity=5.0, unit_price=1000.0, total_price=5000.0)],
        total_amount_ht=5000.0
    )
    inv2 = Invoice(
        id="INV-002",
        invoice_ref="FAC-2026-002",
        supplier_id=sup2.id,
        po_ref=po2.po_ref,
        date=date(2026, 3, 5),
        items=[LineItem(description="Warehouse transport + extra fees", quantity=5.0, unit_price=1500.0, total_price=7500.0)],
        total_amount_ht=7500.0,
        vat_rate=0.20,
        total_amount_ttc=9000.0
    )
    packages.append(ProcurementPackage(
        supplier=sup2,
        contract=con2,
        purchase_orders=[po2],
        invoices=[inv2],
        expected_anomalies=["invoice_exceeds_po"]
    ))

    # Scenario 3: Spend justified by an amendment
    sup3 = Company(
        id="SUP-003",
        name="Gamma Cloud SAS",
        siren="777888999",
        address="5 Boulevard Tech, 35000 Rennes"
    )
    con3 = Contract(
        id="CON-003",
        contract_ref="CTR-2026-GAMMA",
        supplier_id=sup3.id,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
        maximum_amount=20000.0
    )
    amend3 = Amendment(
        id="AMEND-001",
        amendment_ref="AVN-2026-001",
        contract_id=con3.id,
        date=date(2026, 5, 1),
        new_maximum_amount=40000.0,
        description="Extension of GPU computing capacity"
    )
    po3 = PurchaseOrder(
        id="PO-003",
        po_ref="BC-2026-003",
        supplier_id=sup3.id,
        contract_id=con3.id,
        date=date(2026, 6, 1),
        items=[LineItem(description="GPU cluster allocation", quantity=1.0, unit_price=28000.0, total_price=28000.0)],
        total_amount_ht=28000.0
    )
    inv3 = Invoice(
        id="INV-003",
        invoice_ref="FAC-2026-003",
        supplier_id=sup3.id,
        po_ref=po3.po_ref,
        date=date(2026, 6, 15),
        items=[LineItem(description="GPU cluster allocation", quantity=1.0, unit_price=28000.0, total_price=28000.0)],
        total_amount_ht=28000.0,
        vat_rate=0.20,
        total_amount_ttc=33600.0
    )
    packages.append(ProcurementPackage(
        supplier=sup3,
        contract=con3,
        amendments=[amend3],
        purchase_orders=[po3],
        invoices=[inv3],
        expected_anomalies=[]  # Compliant because 28k > 20k initial but <= 40k amended
    ))

    # Scenario 4: Invoice without purchase order
    sup4 = Company(
        id="SUP-004",
        name="Delta Consulting",
        siren="999000111",
        address="12 Rue du Conseil, 44000 Nantes"
    )
    inv4 = Invoice(
        id="INV-004",
        invoice_ref="FAC-2026-004",
        supplier_id=sup4.id,
        po_ref=None,  # Missing reference
        date=date(2026, 4, 10),
        items=[LineItem(description="Strategy consulting session", quantity=1.0, unit_price=3000.0, total_price=3000.0)],
        total_amount_ht=3000.0,
        vat_rate=0.20,
        total_amount_ttc=3600.0
    )
    packages.append(ProcurementPackage(
        supplier=sup4,
        contract=None,
        purchase_orders=[],
        invoices=[inv4],
        expected_anomalies=["missing_po"]
    ))

    return packages
