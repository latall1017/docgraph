# DocGraph: Knowledge Graph & Grounded Document Audit Assistant

DocGraph is an intelligent document audit and Graph RAG system designed to check and reconcile business documents: **Invoices**, **Purchase Orders**, **Contracts**, and **Amendments**.

Standard search often fails when rules or numbers are spread across multiple documents. DocGraph connects related documents in a clear knowledge graph, combines it with semantic text search, and uses Gemini Flash to generate grounded, fact-checked answers.

---

## The Problem: Why Simple Search Fails

In business contracts and finance, key information is rarely in a single file:
* An **Invoice** (`FAC-2026-003`) bills **28,000 EUR** and mentions a Purchase Order (`BC-2026-003`).
* That **Purchase Order** links to a **Contract** (`CON-003`) with a maximum budget of **20,000 EUR**.
* A separate **Amendment** (`AVN-2026-001`) increased that budget to **40,000 EUR**.

### What happens with basic vector search?
When asking: *"Does invoice FAC-2026-003 comply with contract limits?"*:
1. Simple search only finds the invoice and purchase order.
2. The amendment is missed because it only mentions the contract, not the invoice.
3. The AI wrongly flags an overbilling of +8,000 EUR, creating a false alarm.

### How DocGraph solves this:
DocGraph follows the real relationship chain:

```
[Invoice] ──REFERS_TO──> [Purchase Order] ──GOVERNED_BY──> [Contract] <──MODIFIES── [Amendment]
    │                                                          ▲
    └─────────────────────── DIRECT LINK ──────────────────────┘
```

It checks the full picture and confirms that 28,000 EUR is well within the updated 40,000 EUR limit.

---

## Key Features

1. **Cascaded Query Planner (Smart Routing):**
   * **Direct ID Queries** (*"Audit invoice FAC-2026-001"*): Instant check (0 AI token, 0 ms latency).
   * **Plain Language Questions** (*"Check the invoices for Alpha Services"*): Automatically finds the company and audits the right documents.
   * **General Questions** (*"What are the termination clauses?"*): Routes directly to text search without unnecessary calculations.

2. **Cascaded Document Extraction:**
   * **Fast Rules:** Extracts standard French and English documents instantly at 0 cost.
   * **Smart Fallback:** If a document has an unusual layout or foreign labels, the system automatically uses structured AI extraction to avoid missing any files.

3. **Multi-Hop Traversal:**
   * Supports both standard paths (*Invoice $\rightarrow$ Purchase Order $\rightarrow$ Contract $\rightarrow$ Amendments*) and direct paths (*Invoice $\rightarrow$ Contract*).

4. **Reliable Storage & Persistence:**
   * Graph relationships and vector embeddings are saved to disk, so you do not lose your indexed data when restarting the server.

---

## Architecture

```
PDF Documents
     │
     ▼
Cascaded Extractor (Fast Rules ──> Smart Fallback)
     │
     ▼
Knowledge Graph (NetworkX) + Vector Index (E5-small)
     │
     ▼
User Query ──> Query Planner (Chooses: Graph, Text Search, or Both)
                     │
                     ▼
             Answer Generator (Gemini Flash)
                     │
                     ▼
           Fact-Checked Structured Answer
```

---

## Benchmark Results

Testing across standard test cases on our benchmark dataset:

| Architecture | Multi-hop Accuracy | Entity Recall | Abstention on Missing Data |
|---|:---:|:---:|:---:|
| Simple Vector Search (`multilingual-e5-small`) | 0.0 % | 37.5 % | 0.0 % |
| Graph Only (`NetworkX traversal`) | 100.0 % | 83.3 % | 100.0 % |
| **DocGraph Hybrid (`Graph + Vector + Planner`)** | **100.0 %** | **91.7 %** | **100.0 %** |

---

## Quickstart

### 1. Installation
Requires Python 3.11+.

```bash
git clone https://github.com/latall1017/docgraph.git
cd docgraph

uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Set your API Key
Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Generate Sample Documents
```bash
python scripts/generate_dataset.py
```

### 4. Run Tests
```bash
PYTHONPATH=. pytest tests/ -v
```

---

## Running the Application

### 1. Interactive Web Interface (Streamlit)
```bash
streamlit run app.py
```
* **Document Audit:** Ask questions and audit invoices with live feedback.
* **Graph Explorer:** View nodes, connections, and contract amounts.
* **Benchmark:** Run evaluation tests directly in the UI.

### 2. REST API (FastAPI)
```bash
uvicorn src.api.main:app --reload --port 8000
```
Key endpoints:
* `POST /api/audit`: Ask any question in plain language or by ID.
* `POST /api/documents/ingest`: Add and parse a new document.
* `GET /api/graph/stats`: View graph counts and document types.
* `GET /health`: Check service health.

---

## Repository Structure

```
docgraph/
├── app.py                      # Interactive Streamlit application
├── data/                       # Local document storage and cache
├── src/
│   ├── api/                    # FastAPI backend endpoints
│   ├── extraction/             # Cascaded extractor (Rules + Smart Fallback)
│   ├── generation/             # Gemini Flash grounded generator
│   ├── graph/                  # Knowledge graph and multi-hop traversal
│   ├── ingestion/              # PDF parsing
│   ├── resolution/             # Company name matching (RapidFuzz)
│   └── retrieval/              # Query planner, vector search, and hybrid fusion
└── tests/                      # Automated unit and integration test suite
```

---

## License
MIT License.
