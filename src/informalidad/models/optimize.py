"""Optimización de hiperparámetros de XGBoost con Optuna + MLflow (nested runs).

Cada trial queda como un run hijo dentro de un run padre, igual que en el módulo 02 del curso.
El preprocesamiento se ajusta una sola vez (no en cada trial) para que la búsqueda sea rápida.
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from informalidad.config import get_config
from informalidad.data.split import DataSplits
from informalidad.features import build_preprocessor
from informalidad.models.candidates import xgb_estimator
from informalidad.models.evaluate import best_f1_threshold

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def optimize_xgboost(splits: DataSplits, n_trials: int, tags: dict[str, str]) -> dict[str, Any]:
    cfg = get_config()
    target = cfg["data"]["target"]
    seed = cfg["project"]["random_state"]

    prep = build_preprocessor()
    X_train = prep.fit_transform(splits.train)
    X_valid = prep.transform(splits.valid)
    y_train, y_valid = splits.train[target], splits.valid[target]

    with mlflow.start_run(run_name="xgboost_optuna") as parent:
        mlflow.set_tags({**tags, "stage": "optimizacion", "model_family": "xgboost"})
        mlflow.log_param("n_trials", n_trials)

        def objective(trial: optuna.Trial) -> float:
            params = _suggest_params(trial)
            with mlflow.start_run(run_name=f"trial_{trial.number:02d}", nested=True):
                model = xgb_estimator(**params).fit(X_train, y_train)
                proba = model.predict_proba(X_valid)[:, 1]
                auc = roc_auc_score(y_valid, proba)
                mlflow.log_params(params)
                mlflow.log_metric("valid_roc_auc", auc)
            trial.set_user_attr("threshold", best_f1_threshold(y_valid, proba))
            return auc

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=n_trials)

        best = study.best_trial
        threshold = float(best.user_attrs["threshold"])
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metrics({"best_valid_roc_auc": best.value, "best_threshold": threshold})

        history = pd.DataFrame(
            [{"trial": t.number, "valid_roc_auc": t.value, **t.params} for t in study.trials]
        )
        mlflow.log_table(history, artifact_file="optuna_trials.json")

    logger.info("Mejor trial #%s: valid_auc=%.4f", best.number, best.value)
    return {
        "best_params": best.params,
        "best_valid_roc_auc": float(best.value),
        "threshold": threshold,
        "parent_run_id": parent.info.run_id,
    }
