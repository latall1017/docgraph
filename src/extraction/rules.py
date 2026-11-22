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

def find_element(doc: ParsedDocument, snippet: str) -> Optional[DocumentElement]:
    """Find the DocumentElement which contains the snippet (exact then case-insensitive)."""
    clean_snippet = snippet.strip()
    if not clean_snippet:
        return None
    for el in doc.elements:
        if clean_snippet in el.content:
            return el
    lower_snippet = clean_snippet.lower()
    for el in doc.elements:
        if lower_snippet in el.content.lower():
            return el
    return None


def extract_from_parsed_document(doc: ParsedDocument) -> ExtractionResult:
    """Extracts entities and relationships from a parsed document using robust deterministic rules."""
    text = doc.raw_text
    result = ExtractionResult(document_name=doc.file_name)
    referenced_pos_set = set()
    referenced_cons_set = set()

    # 1. Invoice (FACTURE / INVOICE / NOTE D'HONORAIRES)
    inv_match = re.search(
        r"(?:FACTURE|INVOICE|NOTE\s+D'HONORAIRES)(?:\s+(?:N°|#|NO\.?|NUMBER|NUMÉRO|NUM))?\s*[:#]?\s*([A-Z0-9_-]+)",
        text,
        re.IGNORECASE,
    )
    if inv_match and inv_match.group(1).upper() not in ("NUMBER", "NO", "REF", "NUMERO", "NUM", "ASSOCIE"):
        inv_ref = inv_match.group(1).upper()
        evidence_text = inv_match.group(0)

        el = find_element(doc, evidence_text)
        page_num = el.page_number if el else 1
        chunk_id = el.chunk_id if el else None

        ht_match = re.search(
            r"(?:Total\s+HT|Montant\s+HT|Subtotal|Total\s+Excl\.?\s*VAT)\s*:\s*([\d\s,.]+)\s*(?:€|EUR|\$)?",
            text,
            re.IGNORECASE,
        )
        ttc_match = re.search(
            r"(?:Total\s+TTC|Montant\s+TTC|Total\s+Incl\.?\s*VAT)\s*:\s*([\d\s,.]+)\s*(?:€|EUR|\$)?",
            text,
            re.IGNORECASE,
        )

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
                evidence=evidence_text,
                chunk_id=chunk_id,
                page_number=page_num,
            )
        )

        # Check for referenced purchase order(s) - support multiple occurrences & comma lists
        po_matches = list(re.finditer(
            r"(?:bon(?:s)?\s+de\s+commande(?:\s+associé[es]*)?|purchase\s+order(?:s)?|réf(?:érence)?\s+bc)\s*[:]?\s*([^\n\r]+)",
            text,
            re.IGNORECASE,
        ))
        seen_pos = set()
        for pm in po_matches:
            line = pm.group(1)
            found_pos = re.findall(r"\b(BC-[A-Z0-9_-]+|PO-[A-Z0-9_-]+)\b", line, re.IGNORECASE)

            for po_code in found_pos:
                code_up = po_code.upper()
                if code_up not in seen_pos and code_up not in ("NON", "AUCUN", "NULL", "NONE"):
                    seen_pos.add(code_up)
                    referenced_pos_set.add(code_up)
                    po_evidence = pm.group(0)
                    po_el = find_element(doc, po_evidence)
                    result.relations.append(
                        ExtractedRelation(
                            source_id=inv_ref,
                            relation_type="REFERS_TO",
                            target_id=code_up,
                            evidence=po_evidence,
                            chunk_id=po_el.chunk_id if po_el else None,
                        )
                    )

        # Check for directly referenced contract(s) on the invoice
        con_matches = list(re.finditer(
            r"(?:contrat(?:\s+cadre|\s+associé)?|governing\s+agreement|governing\s+contract|contract(?:\s+ref\.?)?)\s*[:]?\s*([^\n\r]+)",
            text,
            re.IGNORECASE,
        ))
        seen_cons = set()
        for cm in con_matches:
            line = cm.group(1)
            found_cons = re.findall(r"\b(CON-[A-Z0-9_-]+|CTR-[A-Z0-9_-]+)\b", line, re.IGNORECASE)
            for con_code in found_cons:
                code_up = con_code.upper()
                if code_up not in seen_cons and code_up not in ("NON", "AUCUN", "NULL", "NONE"):
                    seen_cons.add(code_up)
                    referenced_cons_set.add(code_up)
                    con_evidence = cm.group(0)
                    con_el = find_element(doc, con_evidence)
                    result.relations.append(
                        ExtractedRelation(
                            source_id=inv_ref,
                            relation_type="GOVERNED_BY",
                            target_id=code_up,
                            evidence=con_evidence,
                            chunk_id=con_el.chunk_id if con_el else None,
                        )
                    )

    # 2. Purchase Order (BON DE COMMANDE / PURCHASE ORDER / ORDRE D'ACHAT)
    po_match = re.search(
        r"(?:BON\s+DE\s+COMMANDE|PURCHASE\s+ORDER|ORDRE\s+D'ACHAT)(?!\s+associ[ée]s?)(?:\s+(?:N°|#|NO\.?|NUMBER|NUMÉRO|NUM))?\s*[:#]?\s*([A-Z0-9_-]+)",
        text,
        re.IGNORECASE,
    )
    if (
        po_match
        and po_match.group(1).upper() not in referenced_pos_set
        and po_match.group(1).upper() not in ("NUMBER", "NO", "REF", "NUMERO", "NUM", "ASSOCIE", "ASSOCIEE", "ASSOCIES", "ASSOCIEES", "NON", "AUCUN")
    ):
        po_ref = po_match.group(1).upper()
        evidence_text = po_match.group(0)

        el = find_element(doc, evidence_text)
        page_num = el.page_number if el else 1
        chunk_id = el.chunk_id if el else None

        ht_match = re.search(
            r"(?:Total\s+HT|Montant\s+HT|Subtotal|Total\s+Excl\.?\s*VAT)\s*:\s*([\d\s,.]+)\s*(?:€|EUR|\$)?",
            text,
            re.IGNORECASE,
        )
        properties = {}
        if ht_match:
            properties["amount_ht"] = parse_currency(ht_match.group(1))

        result.entities.append(
            ExtractedEntity(
                id=po_ref,
                entity_type="PurchaseOrder",
                name=f"Purchase Order {po_ref}",
                properties=properties,
                evidence=evidence_text,
                chunk_id=chunk_id,
                page_number=page_num,
            )
        )

        # Check for governing contract(s) - support multiple occurrences
        con_matches = list(re.finditer(
            r"(?:contrat(?: cadre)?|framework contract|governing contract|réf(?:érence)? contrat)\s*[:]?\s*([^\n\r]+)",
            text,
            re.IGNORECASE,
        ))
        seen_cons = set()
        for cm in con_matches:
            line = cm.group(1)
            found_cons = re.findall(r"\b(CON-[A-Z0-9_-]+|CTR-[A-Z0-9_-]+)\b", line, re.IGNORECASE)
            if not found_cons:
                token = line.strip().split()[0].strip(",;.")
                if re.match(r"^[A-Z0-9_-]+$", token, re.IGNORECASE):
                    found_cons = [token]

            for con_code in found_cons:
                code_up = con_code.upper()
                if code_up not in seen_cons:
                    seen_cons.add(code_up)
                    con_evidence = cm.group(0)
                    con_el = find_element(doc, con_evidence)
                    result.relations.append(
                        ExtractedRelation(
                            source_id=po_ref,
                            relation_type="GOVERNED_BY",
                            target_id=code_up,
                            evidence=con_evidence,
                            chunk_id=con_el.chunk_id if con_el else None,
                        )
                    )

    # 3. Contract (CONTRAT DE PRESTATION / CONTRAT CADRE / FRAMEWORK CONTRACT)
    is_contract = re.search(r"\b(?:CONTRAT\s+DE\s+PRESTATION|CONTRAT\s+CADRE|FRAMEWORK\s+CONTRACT|CONVENTION\s+CADRE|SERVICE\s+AGREEMENT)\b", text, re.IGNORECASE)
    if is_contract:
        ref_match = re.search(r"(?:Référence|Reference|Réf)\s*[:]?\s*([A-Z0-9_-]+)", text, re.IGNORECASE)
        if not ref_match:
            ref_match = re.search(r"\b(CON-[A-Z0-9_-]+|CTR-[A-Z0-9_-]+)\b", text, re.IGNORECASE)

        if ref_match and ref_match.group(1).upper() not in referenced_cons_set:
            con_ref = ref_match.group(1).upper()
            evidence_text = ref_match.group(0)

            el = find_element(doc, evidence_text)
            page_num = el.page_number if el else 1
            chunk_id = el.chunk_id if el else None

            max_match = re.search(
                r"(?:plafond\s+(?:global\s+)?de|montant\s+maximum\s+(?:de\s+)?|maximum\s+ceiling\s+(?:of\s+)?|plafond\s*[:])\s*([\d\s,.]+)\s*(?:€|EUR|\$)?",
                text,
                re.IGNORECASE,
            )
            properties = {}
            if max_match:
                properties["maximum_amount"] = parse_currency(max_match.group(1))

            result.entities.append(
                ExtractedEntity(
                    id=con_ref,
                    entity_type="Contract",
                    name=f"Contract {con_ref}",
                    properties=properties,
                    evidence=evidence_text,
                    chunk_id=chunk_id,
                    page_number=page_num,
                )
            )

    # 4. Amendment (AVENANT / AMENDMENT)
    amend_match = re.search(
        r"(?:AVENANT|AMENDMENT)(?:\s+(?:N°|#|NO\.?|NUMBER|NUMÉRO|NUM))?\s*[:#]?\s*([A-Z0-9_-]+)",
        text,
        re.IGNORECASE,
    )
    if amend_match and amend_match.group(1).upper() not in ("NUMBER", "NO", "REF", "NUMERO", "NUM"):
        amend_ref = amend_match.group(1).upper()
        evidence_text = amend_match.group(0)

        el = find_element(doc, evidence_text)
        page_num = el.page_number if el else 1
        chunk_id = el.chunk_id if el else None

        new_max_match = re.search(
            r"(?:nouveau\s+(?:montant|plafond)\s+de|porte\s+le\s+(?:montant|plafond)\s+à|new\s+maximum\s+(?:amount\s+)?(?:of\s+)?|nouveau\s+plafond\s*[:])\s*([\d\s,.]+)\s*(?:€|EUR|\$)?",
            text,
            re.IGNORECASE,
        )
        properties = {}
        if new_max_match:
            properties["new_maximum_amount"] = parse_currency(new_max_match.group(1))

        result.entities.append(
            ExtractedEntity(
                id=amend_ref,
                entity_type="Amendment",
                name=f"Amendment {amend_ref}",
                properties=properties,
                evidence=evidence_text,
                chunk_id=chunk_id,
                page_number=page_num,
            )
        )

        # Check target contract being modified
        con_matches = list(re.finditer(
            r"(?:contrat concerné|target contract|modifie le contrat|contrat cadre)\s*[:]?\s*([^\n\r]+)",
            text,
            re.IGNORECASE,
        ))
        seen_amend_cons = set()
        for cm in con_matches:
            line = cm.group(1)
            found_cons = re.findall(r"\b(CON-[A-Z0-9_-]+|CTR-[A-Z0-9_-]+)\b", line, re.IGNORECASE)
            if not found_cons:
                token = line.strip().split()[0].strip(",;.")
                if re.match(r"^[A-Z0-9_-]+$", token, re.IGNORECASE):
                    found_cons = [token]

            for con_code in found_cons:
                code_up = con_code.upper()
                if code_up not in seen_amend_cons:
                    seen_amend_cons.add(code_up)
                    con_evidence = cm.group(0)
                    con_el = find_element(doc, con_evidence)
                    result.relations.append(
                        ExtractedRelation(
                            source_id=amend_ref,
                            relation_type="MODIFIES",
                            target_id=code_up,
                            evidence=con_evidence,
                            chunk_id=con_el.chunk_id if con_el else None,
                        )
                    )

    return result
