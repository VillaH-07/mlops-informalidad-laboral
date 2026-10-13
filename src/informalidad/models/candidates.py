"""Algoritmos candidatos que se comparan en el experimento."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from informalidad.config import get_config
from informalidad.features import build_model_pipeline


def xgb_estimator(**params) -> XGBClassifier:
    seed = get_config()["project"]["random_state"]
    defaults = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": seed,
    }
    defaults.update(params)
    return XGBClassifier(**defaults)


def candidate_models() -> dict:
    """Pipelines listos para entrenar. El Dummy es el piso mínimo (baseline)."""
    seed = get_config()["project"]["random_state"]
    return {
        "baseline_dummy": build_model_pipeline(DummyClassifier(strategy="prior")),
        "logistic_regression": build_model_pipeline(
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        ),
        "random_forest": build_model_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=14,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=seed,
            )
        ),
        "xgboost": build_model_pipeline(xgb_estimator()),
    }
