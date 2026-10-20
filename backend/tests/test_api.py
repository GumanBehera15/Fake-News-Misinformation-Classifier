from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_is_available():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "model_available" in response.json()

def test_empty_prediction_is_rejected():
    response = client.post("/api/predict", json={"text": "   "})
    assert response.status_code == 422

def test_missing_model_is_explicit():
    response = client.post("/api/predict", json={"text": "A sufficiently long statement for prediction."})
    assert response.status_code in (200, 503)
    if response.status_code == 503:
        assert "Model" in response.json()["detail"]

