"""
Tests for HeartGuard Flask API.
"""

import json
import pytest

from app import app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestFlaskAPI:
    """Test suite for the Flask endpoints."""

    def test_index_returns_200(self, client):
        """GET / should return the form page."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"HeartGuard" in response.data

    def test_predict_valid_input(self, client):
        """POST /predict with valid input should return a prediction."""
        valid_input = {
            "age": "55",
            "sex": "1",
            "cp": "0",
            "trestbps": "130",
            "chol": "250",
            "fbs": "0",
            "restecg": "0",
            "thalach": "150",
            "exang": "0",
            "oldpeak": "1.0",
            "slope": "1",
            "ca": "0",
            "thal": "2",
        }
        response = client.post(
            "/predict",
            data=json.dumps(valid_input),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "prediction" in data
        assert "probability" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1

    def test_predict_missing_fields(self, client):
        """POST /predict with missing fields should return 400."""
        incomplete_input = {
            "age": "55",
            "sex": "1",
            # Missing many required fields
        }
        response = client.post(
            "/predict",
            data=json.dumps(incomplete_input),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_predict_invalid_values(self, client):
        """POST /predict with non-numeric values should return 400."""
        invalid_input = {
            "age": "abc",
            "sex": "1",
            "cp": "0",
            "trestbps": "130",
            "chol": "250",
            "fbs": "0",
            "restecg": "0",
            "thalach": "150",
            "exang": "0",
            "oldpeak": "1.0",
            "slope": "1",
            "ca": "0",
            "thal": "2",
        }
        response = client.post(
            "/predict",
            data=json.dumps(invalid_input),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_predict_form_submission(self, client):
        """POST /predict via form should return HTML result page."""
        valid_input = {
            "age": "55",
            "sex": "1",
            "cp": "0",
            "trestbps": "130",
            "chol": "250",
            "fbs": "0",
            "restecg": "0",
            "thalach": "150",
            "exang": "0",
            "oldpeak": "1.0",
            "slope": "1",
            "ca": "0",
            "thal": "2",
        }
        response = client.post("/predict", data=valid_input)
        assert response.status_code == 200
        # Should return HTML, not JSON
        assert b"Predicted Risk" in response.data or b"prediction" in response.data.lower()

    def test_predict_returns_label(self, client):
        """JSON prediction should include human-readable label."""
        valid_input = {
            "age": "55", "sex": "1", "cp": "0", "trestbps": "130",
            "chol": "250", "fbs": "0", "restecg": "0", "thalach": "150",
            "exang": "0", "oldpeak": "1.0", "slope": "1", "ca": "0", "thal": "2",
        }
        response = client.post(
            "/predict",
            data=json.dumps(valid_input),
            content_type="application/json",
        )
        data = json.loads(response.data)
        assert "label" in data
        assert data["label"] in ["Higher Predicted Risk", "Lower Predicted Risk"]
