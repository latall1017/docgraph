# Business domain schemas for companies, contracts, amendments, purchase orders, and invoices.

from datetime import date
from typing import List, Optional
from pydantic import BaseModel


class Company(BaseModel):
    id: str
    name: str
    siren: str
    address: str


class Contract(BaseModel):
    id: str
    contract_ref: str
    supplier_id: str
    start_date: date
    end_date: date
    maximum_amount: float


class Amendment(BaseModel):
    id: str
    amendment_ref: str
    contract_id: str
    date: date
    new_maximum_amount: float
    description: str


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    total_price: float


class PurchaseOrder(BaseModel):
    id: str
    po_ref: str
    supplier_id: str
    contract_id: Optional[str] = None
    date: date
    items: List[LineItem] = []
    total_amount_ht: float


class Invoice(BaseModel):
    id: str
    invoice_ref: str
    supplier_id: str
    po_ref: Optional[str] = None
    date: date
    items: List[LineItem] = []
    total_amount_ht: float
    vat_rate: float = 0.20
    total_amount_ttc: float
