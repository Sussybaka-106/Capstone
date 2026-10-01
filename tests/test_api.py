"""
Integration tests for FastAPI endpoints.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_kpis_endpoint():
    response = client.get("/kpis")
    assert response.status_code == 200
    data = response.json()
    assert "total_tenders_analyzed" in data
    assert data["total_tenders_analyzed"] > 0


def test_cases_endpoint():
    response = client.get("/cases?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 10
    if len(data) > 0:
        first_case = data[0]
        assert "Tender_ID" in first_case
        assert "ml_anomaly_prob" in first_case
        assert "priority_rank" in first_case


def test_network_graph_endpoint():
    response = client.get("/graph/network?min_cobids=2")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
