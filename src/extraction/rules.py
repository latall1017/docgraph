# Deterministic rule-based information and relation extraction from parsed documents.

import re
from typing import Optional
from src.ingestion.models import ParsedDocument
from src.extraction.schema import ExtractedEntity, ExtractedRelation, ExtractionResult


def parse_currency(amount_str: str) -> float:
    """Converts a formatted currency string to float."""
    clean = amount_str.replace(" ", "").replace(",", "").replace("€", "").strip()
    return float(clean)


def extract_from_parsed_document(doc: ParsedDocument) -> ExtractionResult:
    """Extracts entities and relationships from a parsed document using deterministic rules."""
    text = doc.raw_text
    result = ExtractionResult(document_name=doc.file_name)

    # 1. Invoice
    if "FACTURE" in text:
        ref_match = re.search(r"FACTURE\s+([A-Z0-9_-]+)", text)
        if ref_match:
            inv_ref = ref_match.group(1)
            ht_match = re.search(r"Total HT\s*:\s*([\d\s,.]+)\s*€", text)
            ttc_match = re.search(r"Total TTC\s*:\s*([\d\s,.]+)\s*€", text)

            properties = {}
            if ht_match:
                properties["amount_ht"] = parse_currency(ht_match.group(1))
            if ttc_match:
                properties["amount_ttc"] = parse_currency(ttc_match.group(1))

            result.entities.append(
                ExtractedEntity(
                    id=inv_ref,
                    entity_type="Invoice",
                    name=f"Invoice {inv_ref}",
                    properties=properties,
                    evidence=ref_match.group(0),
                    page_number=1,
                )
            )

            # Check for referenced purchase order
            po_match = re.search(r"Bon de commande associé\s*:\s*(BC-[A-Z0-9_-]+)", text)
            if po_match:
                po_ref = po_match.group(1)
                result.relations.append(
                    ExtractedRelation(
                        source_id=inv_ref,
                        relation_type="REFERS_TO",
                        target_id=po_ref,
                        evidence=po_match.group(0),
                    )
                )

    # 2. Purchase Order
    elif "BON DE COMMANDE" in text:
        ref_match = re.search(r"BON DE COMMANDE\s+([A-Z0-9_-]+)", text)
        if ref_match:
            po_ref = ref_match.group(1)
            ht_match = re.search(r"Total HT\s*:\s*([\d\s,.]+)\s*€", text)
            properties = {}
            if ht_match:
                properties["amount_ht"] = parse_currency(ht_match.group(1))

            result.entities.append(
                ExtractedEntity(
                    id=po_ref,
                    entity_type="PurchaseOrder",
                    name=f"Purchase Order {po_ref}",
                    properties=properties,
                    evidence=ref_match.group(0),
                    page_number=1,
                )
            )

            # Check for governing contract
            con_match = re.search(r"Contrat cadre\s*:\s*([A-Z0-9_-]+)", text)
            if con_match:
                con_ref = con_match.group(1)
                result.relations.append(
                    ExtractedRelation(
                        source_id=po_ref,
                        relation_type="GOVERNED_BY",
                        target_id=con_ref,
                        evidence=con_match.group(0),
                    )
                )

    # 3. Contract
    elif "CONTRAT DE PRESTATION" in text:
        ref_match = re.search(r"Référence\s*:\s*([A-Z0-9_-]+)", text)
        if ref_match:
            con_ref = ref_match.group(1)
            max_match = re.search(r"plafond global de\s*([\d\s,.]+)\s*€", text)
            properties = {}
            if max_match:
                properties["maximum_amount"] = parse_currency(max_match.group(1))

            result.entities.append(
                ExtractedEntity(
                    id=con_ref,
                    entity_type="Contract",
                    name=f"Contract {con_ref}",
                    properties=properties,
                    evidence=ref_match.group(0),
                    page_number=1,
                )
            )

    # 4. Amendment
    elif "AVENANT N°" in text:
        ref_match = re.search(r"AVENANT N°\s*([A-Z0-9_-]+)", text)
        if ref_match:
            amend_ref = ref_match.group(1)
            new_max_match = re.search(r"nouveau montant de\s*([\d\s,.]+)\s*€", text)
            properties = {}
            if new_max_match:
                properties["new_maximum_amount"] = parse_currency(new_max_match.group(1))

            result.entities.append(
                ExtractedEntity(
                    id=amend_ref,
                    entity_type="Amendment",
                    name=f"Amendment {amend_ref}",
                    properties=properties,
                    evidence=ref_match.group(0),
                    page_number=1,
                )
            )

            # Check target contract being modified
            con_match = re.search(r"Contrat concerné\s*:\s*([A-Z0-9_-]+)", text)
            if con_match:
                target_con = con_match.group(1)
                result.relations.append(
                    ExtractedRelation(
                        source_id=amend_ref,
                        relation_type="MODIFIES",
                        target_id=target_con,
                        evidence=con_match.group(0),
                    )
                )

    return result
