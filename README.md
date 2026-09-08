# DocGraph: Hybrid Graph RAG for Complex Document Packages

DocGraph is an end-to-end Document Intelligence and Knowledge Graph system designed to ingest, connect, and reason over multi-document business packages (Invoices, Purchase Orders, Contracts, and Amendments).

While traditional naive RAG struggles to correlate cross-document relationships (e.g. matching an invoice against its original purchase order and verifying contractual ceiling adjustments via amendments), DocGraph builds a deterministic knowledge graph with full provenance and combines it with hybrid retrieval (vector, lexical, and graph traversal).

## Key Components
- **Document Processing & Provenance**: Structure extraction preserving page coordinates and evidence.
- **Entity Resolution**: Normalized entity linking across variations (e.g. legal company name variants).
- **Knowledge Graph**: Property graph schema powered by Neo4j and Cypher queries.
- **Hybrid Retrieval**: Combining dense vector embeddings (E5), full-text search, and multi-hop graph traversal.
- **Evaluation Benchmark**: Rigorous empirical comparison between Naive RAG and Hybrid Graph RAG on 150+ annotated questions.

## Architecture
```
Documents (PDFs) ──> Parsing & Ingestion ──> Information Extraction
                                                    │
                                         Entity Resolution
                                                    │
                                            Knowledge Graph (Neo4j)
                                                    │
User Query ──> Hybrid Retrieval (Vector + Full-text + Graph) ──> Grounded Answer + Citations
```
