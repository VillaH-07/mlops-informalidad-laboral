"""Prueba de integración: entrenamiento + MLflow + Model Registry, sobre la muestra.

Usa una base MLflow temporal, así no toca el mlflow.db del proyecto.
"""

import json

import joblib
import mlflow
import pytest

from informalidad.config import get_config
from informalidad.data import temporal_split, validate_dataset
from informalidad.models.candidates import candidate_models
from informalidad.models.optimize import optimize_xgboost
from informalidad.models.registry import export_champion, register_and_promote
from informalidad.models.train import run_candidate, train_final_model

pytestmark = pytest.mark.integration


@pytest.fixture
def mlflow_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # mlruns/ se crea dentro de tmp_path
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test-informalidad")
    yield tmp_path
    mlflow.set_tracking_uri(None)


def test_end_to_end_training_and_promotion(mlflow_tmp, sample_df):
    df, _ = validate_dataset(sample_df)
    splits = temporal_split(df)
    tags = {"data_version": "test"}

    result = run_candidate("logistic_regression", candidate_models()["logistic_regression"],
                           splits, tags)  # fmt: skip
    assert 0.5 < result["test_roc_auc"] <= 1.0

    best = optimize_xgboost(splits, n_trials=2, tags=tags)
    assert 0.0 < best["threshold"] < 1.0

    final = train_final_model(splits, best["best_params"], best["threshold"], tags)
    first = register_and_promote(final["run_id"], final["model_uri"], final["metrics"], splits.test)
    assert first["promoted"] is True
    assert first["champion_score_same_data"] is None

    # Un modelo claramente peor NO debe reemplazar al champion
    worse = train_final_model(splits, {"n_estimators": 2, "max_depth": 1}, 0.5, tags)
    second = register_and_promote(
        worse["run_id"], worse["model_uri"], worse["metrics"], splits.test
    )
    assert second["promoted"] is False

    name = get_config()["mlflow"]["registered_model_name"]
    client = mlflow.tracking.MlflowClient()
    assert str(client.get_model_version_by_alias(name, "champion").version) == first["version"]
    assert str(client.get_model_version_by_alias(name, "candidate").version) == second["version"]

    out = export_champion(mlflow_tmp / "export")
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["version"] == first["version"]
    model = joblib.load(out / "model.joblib")
    assert model.predict_proba(splits.test.head(5))[:, 1].shape == (5,)
    assert (out / "reference_profile.json").exists()
