"""Fixtures compartidas. Las pruebas usan la muestra versionada (data/sample), nunca el dataset
completo, para que corran en segundos tanto en un laptop como en GitHub Actions."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest

from informalidad.config import feature_columns, get_config, raw_feature_columns, resolve_path
from informalidad.features import build_model_pipeline
from informalidad.models.candidates import xgb_estimator
from informalidad.monitoring.drift import build_reference_profile


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return pd.read_csv(resolve_path(get_config()["data"]["sample_path"]))


@pytest.fixture
def worker() -> dict:
    return {
        "EDAD": 34,
        "HORAS_SEMANA": 48,
        "INGRESO_LABORAL": 1300000,
        "SEXO": 1,
        "NIVEL_EDUCATIVO": 5,
        "ESTADO_CIVIL": 2,
        "AFILIADO_SALUD": 1,
        "POSICION_OCUPACIONAL": 4,
        "TIENE_CONTRATO": 2,
        "RAMA_ACTIVIDAD": 47,
    }


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, sample_df) -> str:
    """Entrena un modelo pequeño y lo guarda con el mismo formato que export_champion."""
    out = tmp_path_factory.mktemp("champion")
    X, y = sample_df[raw_feature_columns()], sample_df["INFORMAL"]
    pipeline = build_model_pipeline(xgb_estimator(n_estimators=30, max_depth=3)).fit(X, y)
    joblib.dump(pipeline, out / "model.joblib")
    numeric, categorical = feature_columns()
    metadata = {
        "model_name": "test-model",
        "version": "0",
        "alias": "champion",
        "run_id": "test",
        "threshold": 0.5,
        "metrics": {"test_roc_auc": 0.95, "test_f1": 0.9},
        "features": {"numeric": numeric, "categorical": categorical},
        "trained_at": "2026-01-01T00:00:00+00:00",
        "library_versions": {},
    }
    (out / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    profile = build_reference_profile(X, numeric, categorical, pipeline.predict_proba(X)[:, 1], y)
    (out / "reference_profile.json").write_text(json.dumps(profile), encoding="utf-8")
    return str(out)
