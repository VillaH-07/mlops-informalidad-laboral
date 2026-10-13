import json

import pytest
from fastapi.testclient import TestClient

from informalidad.api import app as app_module
from informalidad.api.model_loader import ModelService


@pytest.fixture
def client(model_dir, tmp_path, monkeypatch):
    service = ModelService(model_dir=model_dir, prediction_log=tmp_path / "pred.jsonl")
    monkeypatch.setattr(app_module, "service", service)
    with TestClient(app_module.app) as c:
        c.log_path = tmp_path / "pred.jsonl"
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["modelo_cargado"] is True


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["threshold"] == 0.5
    assert "categorical" in body["features"]


def test_predict_valid_worker(client, worker):
    r = client.post("/predict", json=worker)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probabilidad_informal"] <= 1.0
    assert body["etiqueta"] in {"Informal", "Formal"}
    assert body["prediccion"] == int(body["probabilidad_informal"] >= body["umbral"])


@pytest.mark.parametrize(
    "field,value",
    [("SEXO", 3), ("EDAD", 5), ("INGRESO_LABORAL", -1), ("POSICION_OCUPACIONAL", 9)],
)
def test_predict_rejects_invalid_input(client, worker, field, value):
    r = client.post("/predict", json={**worker, field: value})
    assert r.status_code == 422


def test_predict_rejects_missing_field(client, worker):
    worker.pop("EDAD")
    assert client.post("/predict", json=worker).status_code == 422


def test_batch_prediction(client, worker):
    r = client.post("/predict/batch", json={"trabajadores": [worker, worker, worker]})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3
    assert 0.0 <= body["tasa_informalidad_predicha"] <= 1.0


def test_batch_limits(client, worker):
    assert client.post("/predict/batch", json={"trabajadores": []}).status_code == 422
    too_many = {"trabajadores": [worker] * 1001}
    assert client.post("/predict/batch", json=too_many).status_code == 422


def test_predictions_are_logged_for_monitoring(client, worker):
    client.post("/predict", json=worker)
    lines = client.log_path.read_text().strip().splitlines()
    record = json.loads(lines[-1])
    assert record["EDAD"] == worker["EDAD"]
    assert "probabilidad_informal" in record and "timestamp" in record


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "informal" in r.text.lower()
