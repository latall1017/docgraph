import os
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_root_and_health(client):
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "online"

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"
    assert res_health.json()["nodes"] >= 0


def test_graph_stats(client):
    res = client.get("/api/graph/stats")
    assert res.status_code == 200
    data = res.json()
    assert "total_nodes" in data
    assert "total_edges" in data
    assert "node_types" in data
    assert "relation_types" in data


def test_ingest_document(client):
    pdf_path = "data/synthetic/documents/SUP-001_Alpha_Services_SAS/invoice_FAC-2026-001.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"Test PDF not found: {pdf_path}")

    res = client.post("/api/documents/ingest", json={"file_path": pdf_path})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["entities_extracted"] >= 1


def test_ingest_nonexistent_document(client):
    res = client.post("/api/documents/ingest", json={"file_path": "nonexistent.pdf"})
    assert res.status_code == 404


def test_audit_endpoint(client):
    res = client.post("/api/audit", json={"query": "Verify invoice FAC-2026-002", "top_k": 3})
    assert res.status_code == 200
    data = res.json()
    assert "query" in data
    assert "answer" in data
    assert "findings" in data
    assert isinstance(data["findings"], list)
