import os
from src.generation.generator import AnswerGenerator, GroundedAnswer


def test_build_prompt():
    gen = AnswerGenerator()
    query = "Vérifier la facture FAC-2026-002"
    retrieval_result = {
        "ranked_evidence": [{"file_name": "invoice_FAC-2026-002.pdf", "content": "Total HT : 7 500,00 €"}],
        "graph_audit": {
            "invoice_id": "FAC-2026-002",
            "invoice_amount_ht": 7500.0,
            "po_id": "BC-2026-002",
            "po_amount_ht": 5000.0,
            "amount_diff": 2500.0,
            "exceeds_po": True,
            "contract_id": "CTR-2026-BETA",
            "contract_ceiling": 30000.0,
            "amendment_id": None,
            "effective_ceiling": 30000.0,
            "justified_by_amendment": False,
        }
    }

    prompt = gen.build_prompt(query, retrieval_result)
    assert "FAC-2026-002" in prompt
    assert "7500.0" in prompt
    assert "2500.0" in prompt
    assert "N'invente aucune information" in prompt


def test_gemini_generation_live():
    # Only runs if GEMINI_API_KEY is available
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return

    gen = AnswerGenerator(api_key=api_key)
    query = "Quelles sont les anomalies sur la facture FAC-2026-002 et les documents justificatifs ?"
    retrieval_result = {
        "ranked_evidence": [
            {"file_name": "invoice_FAC-2026-002.pdf", "content": "Total HT : 7 500,00 €"},
            {"file_name": "po_BC-2026-002.pdf", "content": "Total HT : 5 000,00 €"},
        ],
        "graph_audit": {
            "invoice_id": "FAC-2026-002",
            "invoice_amount_ht": 7500.0,
            "po_id": "BC-2026-002",
            "po_amount_ht": 5000.0,
            "amount_diff": 2500.0,
            "exceeds_po": True,
            "contract_id": "CTR-2026-BETA",
            "contract_ceiling": 30000.0,
            "amendment_id": None,
            "effective_ceiling": 30000.0,
            "justified_by_amendment": False,
        }
    }

    result = gen.generate(query, retrieval_result)
    assert isinstance(result, GroundedAnswer)
    assert len(result.answer) > 20
    assert len(result.findings) >= 1
    # Check that Gemini detected the 2500 EUR difference
    assert any(f.difference == 2500.0 or f.invoice_amount == 7500.0 for f in result.findings)
