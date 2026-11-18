# DocGraph: Multi-Hop Knowledge Graph and Grounded Audit Assistant

DocGraph is a document intelligence and Graph RAG system designed to audit complex business document packages: Invoices, Purchase Orders, Framework Contracts, and Modifying Amendments.

While standard naive RAG struggles to correlate cross-document relationships, DocGraph constructs a deterministic knowledge graph with documentary provenance, combines it with dense semantic search through Reciprocal Rank Fusion (RRF), and generates structured, grounded answers using Gemini Flash with strict JSON schemas.

---

## The Problem: Why Naive RAG Fails

In procurement and contract auditing, critical business logic is distributed across multiple separate documents:
* An Invoice (`FAC-2026-003`) bills 28,000 EUR and references a Purchase Order (`BC-2026-003`).
* The Purchase Order references a Framework Contract (`CON-003`) with an initial ceiling of 20,000 EUR.
* A separate Contract Amendment (`AVN-2026-001`) raised that ceiling to 40,000 EUR.

### Failure of Naive Vector RAG
When asking: *"Does invoice FAC-2026-003 comply with contract limits?"*:
1. Vector similarity retrieves the invoice and purchase order chunks.
2. The amendment chunk is missed because it only references the contract, not the invoice number.
3. The model reports an unjustified overbilling of +8,000 EUR, creating a false alarm.

### The DocGraph Solution
DocGraph traverses the multi-hop dependency path:
$$\text{Invoice} \xrightarrow{\text{REFERS\_TO}} \text{Purchase Order} \xrightarrow{\text{GOVERNED\_BY}} \text{Contract} \xleftarrow{\text{MODIFIES}} \text{Amendment}$$

It verifies that the invoice amount (28,000 EUR) is within the effective amended ceiling (40,000 EUR).

---

## Empirical Benchmark and Ablation Study

Evaluation results across 5 query categories (Simple, Relational, Multi-Hop, Anomaly Detection, and Abstention) on our ground truth corpus:

| Architecture | Multi-hop Success | Entity Recall | Abstention Accuracy |
|---|:---:|:---:|:---:|
| Naive Vector RAG (`multilingual-e5-small`) | 0.0 % | 37.5 % | 0.0 % |
| Graph Only (`NetworkX traversal`) | 100.0 % | 83.3 % | 100.0 % |
| Hybrid Graph RAG (`Vector + Graph + RRF`) | 100.0 % | 91.7 % | 100.0 % |

### Summary of Findings:
1. **Multi-hop limitation**: Vector search on isolated chunks fails to correlate separate documents (0 % success on multi-hop questions).
2. **Deterministic traversal**: Knowledge graph traversal resolves cross-document chains with 100 % accuracy.
3. **Hybrid combination**: Combining dense retrieval and graph traversal yields the highest recall (91.7 %).
4. **Abstention**: If a queried entity is absent from the corpus, the system correctly abstains rather than hallucinating.

---

## Architecture and Pipeline

```
Documents (PDFs) ──> PyMuPDF Parser ──> Rule-Based Extraction (with Provenance)
                                                     │
                                            Entity Normalizer (RapidFuzz)
                                                     │
                                            Knowledge Graph (NetworkX)
                                                     │
User Query ──> Hybrid Retriever (E5-small + Graph Audit + RRF)
                     │
                     ▼
         Answer Generator (Gemini Flash + Strict Pydantic Schema)
                     │
                     ▼
             Grounded Answer (Answer, Findings, Evidence, Limitations)
```

1. **PDF Parsing and Table Ingestion**: PyMuPDF extracts text and tabular structures from digital PDFs without OCR overhead.
2. **Extraction with Provenance**: Extracted entities and relationships preserve document ID, page number, and evidence text.
3. **Entity Resolution**: RapidFuzz normalizes supplier and company names across legal variations (SAS, SARL, etc.).
4. **Hybrid Retrieval**: Combines semantic cosine similarity (`multilingual-e5-small`) and deterministic multi-hop graph traversal using Reciprocal Rank Fusion.
5. **Grounded Answer Synthesis**: Gemini Flash generates structured output validated against the `GroundedAnswer` schema.

---

## Quickstart

### 1. Installation
Requires Python 3.11+.

```bash
git clone https://github.com/your-username/docgraph.git
cd docgraph

uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_google_ai_studio_api_key_here
```

### 3. Generate Synthetic Ground-Truth Corpus
Generates multi-page invoices, purchase orders, contracts, and amendments:
```bash
python scripts/generate_dataset.py
```

### 4. Run Automated Tests
```bash
PYTHONPATH=. pytest tests/ -v
```

---

## Running the Application

### Streamlit Interface
```bash
streamlit run app.py
```
Provides:
* **Document Audit Assistant**: Interactive compliance queries and preset test cases.
* **Knowledge Graph Explorer**: Inspection of nodes, properties, and relationships.
* **Benchmark Dashboard**: Ablation study metrics and comparative analysis.

### FastAPI REST Backend
```bash
uvicorn src.api.main:app --reload --port 8000
```
Key endpoints:
* `GET /health`: Service status and graph statistics.
* `GET /api/graph/stats`: Node and relation counts by category.
* `POST /api/audit`: End-to-end hybrid audit endpoint.
* `POST /api/documents/ingest`: Dynamic document parsing and graph ingestion.

---

## Engineering Choices

| Component | Choice | Rationale |
|---|---|---|
| PDF Extraction | PyMuPDF | Fast native text and table extraction without OCR overhead. |
| Knowledge Graph | NetworkX | In-memory graph engine with sub-millisecond traversal and low memory footprint. |
| Entity Resolution | RapidFuzz | Fast string matching to normalize legal entity names. |
| Embeddings | multilingual-e5-small | 118M parameters, efficient on CPU, strong multilingual semantic retrieval. |
| Generation | Gemini Flash | Sub-second latency, zero local RAM usage, strict JSON schema validation. |

---

## Project Organization and Where to Look

If you are exploring the codebase, here is a simple guide on where to start:

* **Entry Points**:
  * `app.py`: Streamlit interface (Audit chat, Graph explorer, and Benchmark views).
  * `src/api/main.py`: FastAPI service exposing `/api/audit` and `/api/graph/stats`.

* **Core Pipeline Modules (`src/`)**:
  * `src/ingestion/`: PDF text and table extraction using PyMuPDF (`pdf_parser.py`).
  * `src/extraction/`: Rule-based entity and relation extractor with provenance (`rules.py`).
  * `src/resolution/`: Supplier name normalization with RapidFuzz (`normalizer.py`).
  * `src/graph/`: Knowledge graph construction and multi-hop traversal (`graph_builder.py`).
  * `src/retrieval/`: Vector (`e5-small`), graph, and hybrid RRF retrieval (`hybrid_retriever.py`).
  * `src/generation/`: Structured synthesis with Gemini Flash (`generator.py`).
  * `src/evaluation/`: Benchmark runner and 5-category test suite (`benchmark.py`).

* **Tests and Data**:
  * `tests/`: 29 automated Pytest unit and integration tests covering every component.
  * `data/synthetic/`: Multi-document PDF packages and ground truth scenarios.

---

## License
MIT License.
