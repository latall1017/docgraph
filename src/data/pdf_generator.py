# Generates synthetic commercial PDF documents with structured tables using ReportLab.

import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from src.schemas import Company, Contract, Amendment, PurchaseOrder, Invoice
from src.data.generator import ProcurementPackage


styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Heading1"],
    fontSize=18,
    leading=22,
    textColor=colors.HexColor("#1A202C"),
    spaceAfter=12,
)
body_style = ParagraphStyle(
    "DocBody",
    parent=styles["Normal"],
    fontSize=10,
    leading=14,
    textColor=colors.HexColor("#2D3748"),
)


def render_invoice_pdf(invoice: Invoice, supplier: Company, output_path: str):
    """Renders a formatted invoice PDF with itemized table."""
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    elements = []

    # Title and Header
    elements.append(Paragraph(f"FACTURE {invoice.invoice_ref}", title_style))
    elements.append(Paragraph(f"<b>Émetteur :</b> {supplier.name}<br/>SIREN : {supplier.siren}<br/>{supplier.address}", body_style))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"<b>Date :</b> {invoice.date.strftime('%d/%m/%Y')}", body_style))
    if invoice.po_ref:
        elements.append(Paragraph(f"<b>Bon de commande associé :</b> {invoice.po_ref}", body_style))
    else:
        elements.append(Paragraph("<b>Bon de commande associé :</b> Non spécifié", body_style))
    elements.append(Spacer(1, 15))

    # Table of items
    table_data = [["Description", "Quantité", "Prix unitaire HT", "Total HT"]]
    for item in invoice.items:
        table_data.append([item.description, f"{item.quantity:.0f}", f"{item.unit_price:,.2f} €", f"{item.total_price:,.2f} €"])
    table_data.append(["", "", "Total HT :", f"{invoice.total_amount_ht:,.2f} €"])
    table_data.append(["", "", f"TVA ({int(invoice.vat_rate * 100)}%) :", f"{(invoice.total_amount_ht * invoice.vat_rate):,.2f} €"])
    table_data.append(["", "", "Total TTC :", f"{invoice.total_amount_ttc:,.2f} €"])

    table = Table(table_data, colWidths=[240, 60, 100, 100])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A202C")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)
    doc.build(elements)


def render_po_pdf(po: PurchaseOrder, supplier: Company, output_path: str):
    """Renders a formatted purchase order PDF with items and governing contract reference."""
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    elements = []

    elements.append(Paragraph(f"BON DE COMMANDE {po.po_ref}", title_style))
    elements.append(Paragraph(f"<b>Fournisseur :</b> {supplier.name}<br/>SIREN : {supplier.siren}", body_style))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"<b>Date :</b> {po.date.strftime('%d/%m/%Y')}", body_style))
    if po.contract_id:
        elements.append(Paragraph(f"<b>Contrat cadre :</b> {po.contract_id}", body_style))
    elements.append(Spacer(1, 15))

    table_data = [["Article / Prestation", "Quantité", "Prix unitaire HT", "Total HT"]]
    for item in po.items:
        table_data.append([item.description, f"{item.quantity:.0f}", f"{item.unit_price:,.2f} €", f"{item.total_price:,.2f} €"])
    table_data.append(["", "", "Total HT :", f"{po.total_amount_ht:,.2f} €"])

    table = Table(table_data, colWidths=[240, 60, 100, 100])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EBF8FF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BEE3F8")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)
    doc.build(elements)


def render_contract_pdf(contract: Contract, supplier: Company, output_path: str):
    """Renders a formatted framework contract PDF with validity dates and maximum financial ceiling."""
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    elements = []

    elements.append(Paragraph(f"CONTRAT DE PRESTATION DE SERVICES", title_style))
    elements.append(Paragraph(f"<b>Référence :</b> {contract.contract_ref}", body_style))
    elements.append(Spacer(1, 15))

    p1 = f"Entre la société <b>Client Principal</b> d'une part, et la société <b>{supplier.name}</b> (SIREN {supplier.siren}), sise {supplier.address}, d'autre part."
    elements.append(Paragraph(p1, body_style))
    elements.append(Spacer(1, 10))

    p2 = f"<b>Article 1 - Période de validité :</b> Le présent contrat entre en vigueur le {contract.start_date.strftime('%d/%m/%Y')} et expire le {contract.end_date.strftime('%d/%m/%Y')}."
    elements.append(Paragraph(p2, body_style))
    elements.append(Spacer(1, 10))

    p3 = f"<b>Article 2 - Plafond financier :</b> Le montant cumulé des prestations commandées sous le présent contrat ne pourra dépasser un plafond global de <b>{contract.maximum_amount:,.2f} € HT</b> sans accord préalable."
    elements.append(Paragraph(p3, body_style))
    elements.append(Spacer(1, 25))

    elements.append(Paragraph("Fait en deux exemplaires originaux.<br/>Pour le Client &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Pour le Prestataire", body_style))
    doc.build(elements)


def render_amendment_pdf(amendment: Amendment, supplier: Company, output_path: str):
    """Renders a formatted contract amendment PDF with modified ceiling."""
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    elements = []

    elements.append(Paragraph(f"AVENANT N° {amendment.amendment_ref}", title_style))
    elements.append(Paragraph(f"<b>Contrat concerné :</b> {amendment.contract_id}", body_style))
    elements.append(Paragraph(f"<b>Prestataire :</b> {supplier.name}", body_style))
    elements.append(Paragraph(f"<b>Date de signature :</b> {amendment.date.strftime('%d/%m/%Y')}", body_style))
    elements.append(Spacer(1, 15))

    p1 = f"<b>Objet de l'avenant :</b> {amendment.description}"
    elements.append(Paragraph(p1, body_style))
    elements.append(Spacer(1, 10))

    p2 = f"<b>Modification financière :</b> À compter de la date de signature, le plafond contractuel autorisé est rehaussé à un nouveau montant de <b>{amendment.new_maximum_amount:,.2f} € HT</b>."
    elements.append(Paragraph(p2, body_style))
    elements.append(Spacer(1, 25))

    elements.append(Paragraph("Pour le Client &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Pour le Prestataire", body_style))
    doc.build(elements)


def render_package_to_disk(package: ProcurementPackage, base_dir: str):
    """Generates all PDFs of a procurement package into a structured supplier directory."""
    supplier_dir = Path(base_dir) / f"{package.supplier.id}_{package.supplier.name.replace(' ', '_')}"
    supplier_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    # 1. Contract
    if package.contract:
        p = str(supplier_dir / f"contract_{package.contract.contract_ref}.pdf")
        render_contract_pdf(package.contract, package.supplier, p)
        created_files.append(p)

    # 2. Amendments
    for amend in package.amendments:
        p = str(supplier_dir / f"amendment_{amend.amendment_ref}.pdf")
        render_amendment_pdf(amend, package.supplier, p)
        created_files.append(p)

    # 3. Purchase Orders
    for po in package.purchase_orders:
        p = str(supplier_dir / f"po_{po.po_ref}.pdf")
        render_po_pdf(po, package.supplier, p)
        created_files.append(p)

    # 4. Invoices
    for inv in package.invoices:
        p = str(supplier_dir / f"invoice_{inv.invoice_ref}.pdf")
        render_invoice_pdf(inv, package.supplier, p)
        created_files.append(p)

    return created_files


generate_package_documents = render_package_to_disk
