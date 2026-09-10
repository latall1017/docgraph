from datetime import date
from src.schemas import Company, Contract, Amendment, PurchaseOrder, Invoice, LineItem


def test_create_procurement_models():
    # 1. Company
    supplier = Company(
        id="SUP-001",
        name="ACME SAS",
        siren="123456789",
        address="10 Rue de la Paix, 75001 Paris"
    )
    assert supplier.name == "ACME SAS"

    # 2. Contract
    contract = Contract(
        id="CON-2026-001",
        contract_ref="CTR-2026-01",
        supplier_id=supplier.id,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
        maximum_amount=50000.0
    )
    assert contract.maximum_amount == 50000.0

    # 3. Amendment
    amendment = Amendment(
        id="AMEND-2026-001",
        amendment_ref="AVN-2026-01",
        contract_id=contract.id,
        date=date(2026, 6, 1),
        new_maximum_amount=75000.0,
        description="Budget increase for project scope extension"
    )
    assert amendment.new_maximum_amount > contract.maximum_amount

    # 4. Purchase Order
    item = LineItem(description="Cloud GPU Hosting", quantity=10.0, unit_price=500.0, total_price=5000.0)
    po = PurchaseOrder(
        id="PO-2026-001",
        po_ref="BC-2026-001",
        supplier_id=supplier.id,
        contract_id=contract.id,
        date=date(2026, 3, 1),
        items=[item],
        total_amount_ht=5000.0
    )
    assert len(po.items) == 1
    assert po.total_amount_ht == 5000.0

    # 5. Invoice
    inv = Invoice(
        id="INV-2026-001",
        invoice_ref="FAC-2026-001",
        supplier_id=supplier.id,
        po_ref=po.po_ref,
        date=date(2026, 4, 15),
        items=[item],
        total_amount_ht=5000.0,
        vat_rate=0.20,
        total_amount_ttc=6000.0
    )
    assert inv.po_ref == "BC-2026-001"
    assert inv.total_amount_ttc == 6000.0
