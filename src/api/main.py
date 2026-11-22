# FastAPI REST backend exposing health, graph inspection, document ingestion, and audit endpoints.

import os
import glob
import threading
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

from src.graph.graph_builder import KnowledgeGraph
from src.ingestion.pdf_parser import parse_pdf
from src.extraction.cascade_extractor import extract_document_cascaded
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.generator import AnswerGenerator, GroundedAnswer


app = FastAPI(
    title="DocGraph API",
    description="Document Intelligence and Knowledge Graph RAG for Business Document Auditing",
    version="1.0.0",
)

# Global system state with thread-safe ingestion lock
ingest_lock = threading.Lock()
kg = KnowledgeGraph()
vector_retriever = VectorRetriever()
graph_retriever = GraphRetriever(kg)
hybrid_retriever = HybridRetriever(vector_retriever, graph_retriever)
generator = AnswerGenerator()


def load_synthetic_documents(corpus_dir: str = "data/synthetic/documents", cache_dir: str = "data"):
    """Ingests synthetic documents into Knowledge Graph and Vector index with cache support."""
    with ingest_lock:
        kg_cache = os.path.join(cache_dir, "knowledge_graph.json")
        vec_cache = os.path.join(cache_dir, "vector_cache")

        # Try loading from persistent disk cache
        kg_loaded = kg.load_from_disk(kg_cache)
        vec_loaded = vector_retriever.load_from_disk(vec_cache)

        if kg_loaded and vec_loaded and len(kg.graph.nodes) > 0 and len(vector_retriever.chunks) > 0:
            return

        pdf_files = glob.glob(os.path.join(corpus_dir, "**/*.pdf"), recursive=True)
        for pdf_path in pdf_files:
            try:
                doc = parse_pdf(pdf_path)
                extraction = extract_document_cascaded(doc)
                for entity in extraction.entities:
                    kg.add_entity(entity)
                for relation in extraction.relations:
                    kg.add_relation(relation)
                vector_retriever.add_document(doc)
            except Exception:
                pass
        if vector_retriever.chunks:
            vector_retriever.build_index()

        # Persist to disk for fast subsequent startups
        try:
            kg.save_to_disk(kg_cache)
            vector_retriever.save_to_disk(vec_cache)
        except Exception:
            pass


# Bootstrap with existing corpus if available
load_synthetic_documents()


class AuditRequest(BaseModel):
    query: str = Field(..., description="Audit question (e.g. 'Verify invoice FAC-2026-002')")
    top_k: int = Field(default=3, description="Number of evidence chunks to retrieve")


class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Absolute or relative path to the PDF file")


@app.get("/")
def root():
    """Returns basic service identification and online status."""
    return {
        "service": "DocGraph API",
        "status": "online",
        "description": "Multi-hop Graph RAG for procurement document audits",
    }


@app.get("/health")
def health_check():
    """Returns service health metrics including graph node count and vector chunks."""
    return {
        "status": "healthy",
        "nodes": len(kg.graph.nodes),
        "edges": len(kg.graph.edges),
        "indexed_chunks": len(vector_retriever.chunks),
    }


@app.get("/api/graph/stats")
def graph_stats():
    """Returns the breakdown of entities and relations currently stored in the graph."""
    node_types: Dict[str, int] = {}
    for _, data in kg.graph.nodes(data=True):
        t = data.get("entity_type", "Unknown")
        node_types[t] = node_types.get(t, 0) + 1

    relation_types: Dict[str, int] = {}
    for _, _, data in kg.graph.edges(data=True):
        r = data.get("relation_type", "Unknown")
        relation_types[r] = relation_types.get(r, 0) + 1

    return {
        "total_nodes": len(kg.graph.nodes),
        "total_edges": len(kg.graph.edges),
        "node_types": node_types,
        "relation_types": relation_types,
    }


@app.post("/api/documents/ingest")
def ingest_document(req: IngestRequest):
    """Parses a new PDF document, extracts entities and relations, updates the graph, and persists."""
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    with ingest_lock:
        try:
            doc = parse_pdf(req.file_path)
            extraction = extract_document_cascaded(doc)

            for entity in extraction.entities:
                kg.add_entity(entity)
            for relation in extraction.relations:
                kg.add_relation(relation)

            vector_retriever.add_document(doc)
            vector_retriever.build_index()

            # Persist updated state to disk
            try:
                kg.save_to_disk("data/knowledge_graph.json")
                vector_retriever.save_to_disk("data/vector_cache")
            except Exception:
                pass

            return {
                "status": "success",
                "file": req.file_path,
                "entities_extracted": len(extraction.entities),
                "relations_extracted": len(extraction.relations),
                "total_graph_nodes": len(kg.graph.nodes),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/audit")
def audit_query(req: AuditRequest):
    """Runs hybrid retrieval and grounded LLM generation to audit procurement documents."""
    retrieval_result = hybrid_retriever.retrieve(req.query, top_k=req.top_k)
    answer: GroundedAnswer = generator.generate(req.query, retrieval_result)

    return {
        "query": req.query,
        "detected_invoice": retrieval_result.get("detected_invoice"),
        "answer": answer.answer,
        "findings": [f.model_dump() for f in answer.findings],
        "evidence": [e.model_dump() for e in answer.evidence],
        "limitations": answer.limitations,
        "graph_audit": retrieval_result.get("graph_audit"),
    }
