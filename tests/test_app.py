from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_root_returns_welcome_message():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ML API"}


def test_health_returns_ok_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_class_name():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], str)
    assert data["prediction"] in {"setosa", "versicolor", "virginica"}
