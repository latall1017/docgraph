# DocGraph: Multi-Hop Knowledge Graph & Grounded Audit Assistant

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Pytest-24%20passed-brightgreen.svg)](https://pytest.org/)

**DocGraph** is an enterprise-grade Document Intelligence and Graph RAG system designed to audit complex business document packages (Invoices, Purchase Orders, Framework Contracts, and Modifying Amendments).

While traditional Naive RAG fails to correlate cross-document dependencies, DocGraph builds a deterministic knowledge graph with strict documentary provenance, combines it with dense semantic search via **Reciprocal Rank Fusion (RRF)**, and produces hallucination-free, auditable answers using **Gemini Flash** with strict JSON schemas.

---

## 🎯 The Core Problem: Why Naive RAG Fails

In procurement and B2B contract auditing, critical business logic is distributed across multiple separate documents:
* An **Invoice** (`FAC-2026-003`) bills **28 000 €** and references a **Purchase Order** (`BC-2026-003`).
* The **Purchase Order** references a **Framework Contract** (`CTR-2026-GAMMA`) with a maximum ceiling of **20 000 €**.
* A separate **Contract Amendment** (`AVN-2026-001`) raised that ceiling to **40 000 €**.

### The Failure of Naive Vector RAG
When an auditor asks: *"Does invoice FAC-2026-003 comply with contract limits?"*:
1. Vector similarity matches the invoice and purchase order chunks.
2. **The amendment chunk is completely missed** because it only cites the contract reference (`CTR-2026-GAMMA`), not the invoice number (`FAC-2026-003`).
3. The LLM hallucinates an unauthorized overbilling violation of +8 000 €, triggering a false alarm.

### The DocGraph Solution
DocGraph traverses the multi-hop dependency path:
$$\text{Invoice} \xrightarrow{\text{REFERS\_TO}} \text{Purchase Order} \xrightarrow{\text{GOVERNED\_BY}} \text{Contract} \xleftarrow{\text{MODIFIES}} \text{Amendment}$$
It mathematically computes the effective ceiling ($40\,000\text{ \texteuro}$) and verifies compliance deterministically.

---

## 📊 Empirical Benchmark & Ablation Study

We evaluated three architectures across 5 query categories (*Simple*, *2-Doc Relational*, *Multi-Hop*, *Discrepancy Detection*, and *Abstention / Out-of-Corpus*) on an annotated ground-truth test suite:

| Architecture | Multi-hop Success | Entity Recall | Abstention Accuracy |
|---|:---:|:---:|:---:|
| **Naive Vector RAG** (`multilingual-e5-small`) | **0.0 %** | 37.5 % | 0.0 % |
| **Graph Only** (`NetworkX traversal`) | **100.0 %** | 83.3 % | 100.0 % |
| **Hybrid Graph RAG** (`Vector + Graph + RRF`) | **100.0 %** | **91.7 %** | **100.0 %** |

### Benchmark Takeaways:
1. **0.0 % Multi-hop Success for Naive Vector RAG**: Standard chunk-based semantic search cannot connect disconnected documents across multi-stage procurement chains.
2. **Deterministic Multi-hop (100 %)**: Graph traversal guarantees that contract amendments are always found if they exist.
3. **Superior Recall with Hybrid (91.7 %)**: Combining dense embeddings with graph edges yields the highest coverage while eliminating false positives.
4. **Perfect Abstention (100 %)**: If an entity or fact is outside the knowledge base, the system refrains from hallucinating.

---

## 🏗️ Architecture & Pipeline

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
         Answer Generator (Gemini Flash + Strict Pydantic JSON Schema)
                     │
                     ▼
             Grounded Answer { answer, findings, evidence, limitations }
```

1. **PDF Parsing & Table Ingestion**: PyMuPDF extracts raw text and tabular structures with zero OCR hallucination.
2. **Provenance-Aware Extraction**: Every extracted entity and relationship retains its `source_document_id`, `page_number`, and literal `evidence_text`.
3. **Entity Resolution**: RapidFuzz normalizes company names across legal suffixes (`SAS`, `SARL`, `France`, etc.) to prevent duplicate nodes.
4. **Hybrid Retrieval**: Combines semantic cosine similarity (`intfloat/multilingual-e5-small`) with multi-hop graph reconciliation using **Reciprocal Rank Fusion (RRF)**.
5. **Grounded Answer Synthesis**: Gemini 2.5 Flash validates context and generates structured JSON matching the `GroundedAnswer` schema.

---

## ⚡ Quickstart

### 1. Prerequisites & Installation
Requires Python 3.11+.

```bash
git clone https://github.com/your-username/docgraph.git
cd docgraph

# Create virtual environment and install dependencies
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
Generates realistic multi-page invoices, purchase orders, contracts, and amendments with intentional business discrepancies:
```bash
python scripts/generate_dataset.py
```

### 4. Run Automated Test Suite
```bash
PYTHONPATH=. pytest tests/ -v
```
*(All 24 unit and integration tests run in ~20 seconds).*

---

## 🚀 Running the Services

### Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** to access:
* **Document Audit Assistant**: Interactive compliance chat with 1-click test scenarios.
* **Knowledge Graph Explorer**: Real-time table of nodes and directional relations.
* **Benchmark Dashboard**: Live ablation metrics and architectural analysis.

### FastAPI REST Backend
```bash
uvicorn src.api.main:app --reload --port 8000
```
* **Swagger Interactive Docs**: `http://localhost:8000/docs`
* **Health Check**: `GET http://localhost:8000/health`
* **Graph Topology**: `GET http://localhost:8000/api/graph/stats`
* **Audit Endpoint**: `POST http://localhost:8000/api/audit`

Example request:
```bash
curl -X POST http://localhost:8000/api/audit \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What anomalies exist on invoice FAC-2026-002 and is there any justification?",
    "top_k": 3
  }'
```

---

## 🛠️ Key Engineering Trade-offs

| Component | Choice | Rationale |
|---|---|---|
| **PDF Extraction** | `PyMuPDF` | Fast, lightweight, exact native text and table extraction without GPU or heavy OCR overhead. |
| **Knowledge Graph** | `NetworkX` | In-memory graph engine with sub-millisecond traversals; avoids running heavy 2 GB JVM Docker containers on local developer machines while keeping full Cypher exportability. |
| **Entity Resolution** | `RapidFuzz` | C++ token sort ratio matching company names and legal forms with minimal latency. |
| **Embeddings** | `intfloat/multilingual-e5-small` | 118M parameters; runs fast on CPU with strong multilingual retrieval quality. |
| **Generation** | `Gemini Flash (v1beta)` | Zero local RAM usage, sub-second response times, and native JSON schema compliance via Pydantic. |

---

## 📁 Repository Structure

```
docgraph/
├── app.py                      # Streamlit interactive UI
├── data/
│   └── synthetic/              # Generated PDF documents & ground truth
├── requirements.txt            # Project dependencies
├── scripts/
│   └── generate_dataset.py     # Corpus generator CLI
├── src/
│   ├── api/                    # FastAPI backend endpoints
│   ├── data/                   # Data generators & PDF builders
│   ├── evaluation/             # Quantitative benchmark runner
│   ├── extraction/             # Rule-based extractors with provenance
│   ├── generation/             # Gemini Flash grounded synthesis
│   ├── graph/                  # NetworkX knowledge graph builder
│   ├── ingestion/              # PyMuPDF parser and document models
│   ├── resolution/             # RapidFuzz entity normalizer
│   ├── retrieval/              # Vector, Graph & Hybrid RRF retrievers
│   └── schemas/                # Domain models (Invoice, PO, Contract)
└── tests/                      # 24 automated Pytest unit and integration tests
```

---

## 📜 License
MIT License.
