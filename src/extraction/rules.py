# Deterministic rule-based information and relation extraction from parsed documents.

import re
from typing import Optional
from src.extraction.schema import ExtractedEntity, ExtractedRelation, ExtractionResult
from src.ingestion.models import DocumentElement, ParsedDocument

def parse_currency(amount_str: str) -> float:
    """Converts a formatted currency string to float handling French and international formats."""
    clean = re.sub(r"[€$£\s\xa0\u202f]", "", amount_str).strip()
    if not clean:
        return 0.0

    try:
        # If both . and , are present
        if "." in clean and "," in clean:
            if clean.rfind(",") > clean.rfind("."):
                # e.g. 1.250,50 -> 1250.50 (French/European format with thousand dot)
                clean = clean.replace(".", "").replace(",", ".")
            else:
                # e.g. 1,250.50 -> 1250.50 (US/UK format with thousand comma)
                clean = clean.replace(",", "")
        elif "," in clean:
            # e.g. 1250,50 or 10 000,00 -> comma is decimal separator
            clean = clean.replace(",", ".")

        return float(clean)
    except ValueError:
        return 0.0

def find_element(doc : ParsedDocument , snippet : str) -> Optional[DocumentElement] : 
    """Find the DocumentElement which contains the snippet."""
    for el in doc.elements : 
        if snippet in el.content : 
            return el 
    return None


def extract_from_parsed_document(doc: ParsedDocument) -> ExtractionResult:
    """Extracts entities and relationships from a parsed document using deterministic rules."""
    text = doc.raw_text
    result = ExtractionResult(document_name=doc.file_name)

    # 1. Invoice
    if "FACTURE" in text:
        ref_match = re.search(r"FACTURE\s+([A-Z0-9_-]+)", text)
        if ref_match:

            
            inv_ref = ref_match.group(1)
            evidence_text = ref_match.group(0)


            el = find_element(doc, evidence_text)
            page_num = el.page_number if el else 1
            chunk_id = el.chunk_id if el else None
            ht_match = re.search(r"Total HT\s*:\s*([\d\s,.]+)\s*€", text)
            ttc_match = re.search(r"Total TTC\s*:\s*([\d\s,.]+)\s*€", text)

            properties = {}
            if ht_match:
                properties["amount_ht"] = parse_currency(ht_match.group(1))
            if ttc_match:
                properties["amount_ttc"] = parse_currency(ttc_match.group(1))

            # Now we check witwhich element we are with the given infos.


            result.entities.append(
                ExtractedEntity(
                    id=inv_ref,
                    entity_type="Invoice",
                    name=f"Invoice {inv_ref}",
                    properties=properties,
                    evidence=evidence_text,
                    chunk_id= chunk_id,
                    page_number= page_num,
                )
            )

            # Check for referenced purchase order
            po_match = re.search(r"Bon de commande associé\s*:\s*(BC-[A-Z0-9_-]+)", text)
            if po_match:
                po_ref = po_match.group(1)
                po_evidence = po_match.group(0)
                po_el = find_element(doc,po_evidence)
                result.relations.append(
                    ExtractedRelation(
                        source_id=inv_ref,
                        relation_type="REFERS_TO",
                        target_id=po_ref,
                        evidence=po_evidence,
                        chunk_id = po_el.chunk_id if po_el else None, 
                    )
                )

    # 2. Purchase Order
    elif "BON DE COMMANDE" in text:
        ref_match = re.search(r"BON DE COMMANDE\s+([A-Z0-9_-]+)", text)
        if ref_match:
            po_ref = ref_match.group(1)
            po_evidence = ref_match.group(0)
            po_el = find_element(doc, po_evidence)
            page_num = po_el.page_number if po_el else 1
            chunk_id = po_el.chunk_id if po_el else None

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
                    evidence=po_evidence,
                    chunk_id=chunk_id,
                    page_number=page_num,
                )
            )

            # Check for governing contract
            con_match = re.search(r"Contrat cadre\s*:\s*([A-Z0-9_-]+)", text)
            if con_match:
                con_ref = con_match.group(1)
                con_evidence = con_match.group(0)
                con_el = find_element(doc, con_evidence)
                result.relations.append(
                    ExtractedRelation(
                        source_id=po_ref,
                        relation_type="GOVERNED_BY",
                        target_id=con_ref,
                        evidence=con_evidence,
                        chunk_id=con_el.chunk_id if con_el else None,
                    )
                )

    # 3. Contract
    elif "CONTRAT DE PRESTATION" in text:
        ref_match = re.search(r"Référence\s*:\s*([A-Z0-9_-]+)", text)
        if ref_match:
            con_ref = ref_match.group(1)
            con_evidence = ref_match.group(0)
            con_el = find_element(doc, con_evidence)
            page_num = con_el.page_number if con_el else 1
            chunk_id = con_el.chunk_id if con_el else None

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
                    evidence=con_evidence,
                    chunk_id=chunk_id,
                    page_number=page_num,
                )
            )

    # 4. Amendment
    elif "AVENANT N°" in text:
        ref_match = re.search(r"AVENANT N°\s*([A-Z0-9_-]+)", text)
        if ref_match:
            amend_ref = ref_match.group(1)
            amend_evidence = ref_match.group(0)
            amend_el = find_element(doc, amend_evidence)
            page_num = amend_el.page_number if amend_el else 1
            chunk_id = amend_el.chunk_id if amend_el else None

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
                    evidence=amend_evidence,
                    chunk_id=chunk_id,
                    page_number=page_num,
                )
            )

            # Check target contract being modified
            con_match = re.search(r"Contrat concerné\s*:\s*([A-Z0-9_-]+)", text)
            if con_match:
                target_con = con_match.group(1)
                con_evidence = con_match.group(0)
                con_el = find_element(doc, con_evidence)
                result.relations.append(
                    ExtractedRelation(
                        source_id=amend_ref,
                        relation_type="MODIFIES",
                        target_id=target_con,
                        evidence=con_evidence,
                        chunk_id=con_el.chunk_id if con_el else None,
                    )
                )

    return result
