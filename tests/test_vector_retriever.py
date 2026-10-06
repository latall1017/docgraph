from src.ingestion.pdf_parser import parse_pdf
from src.retrieval.vector_retriever import VectorRetriever


def test_vector_retriever_semantic_search():
    pdf1 = "data/synthetic/documents/SUP-001_Alpha_Services_SAS/invoice_FAC-2026-001.pdf"
    pdf2 = "data/synthetic/documents/SUP-002_Beta_Logistics_SARL/invoice_FAC-2026-002.pdf"

    doc1 = parse_pdf(pdf1)
    doc2 = parse_pdf(pdf2)

    retriever = VectorRetriever()
    retriever.add_document(doc1)
    retriever.add_document(doc2)
    retriever.build_index()

    assert len(retriever.chunks) > 0
    assert retriever.embeddings is not None

    # 1. Query related to logistics/transport should rank Beta Logistics top
    results_transport = retriever.search("frais de transport et entrepôt", top_k=2)
    assert len(results_transport) > 0
    assert "invoice_FAC-2026-002.pdf" in results_transport[0]["file_name"]
    assert results_transport[0]["score"] > 0.6

    # 2. Query related to software maintenance should rank Alpha Services top
    results_software = retriever.search("maintenance informatique de logiciels", top_k=2)
    assert len(results_software) > 0
    assert "invoice_FAC-2026-001.pdf" in results_software[0]["file_name"]
    assert results_software[0]["score"] > 0.6
