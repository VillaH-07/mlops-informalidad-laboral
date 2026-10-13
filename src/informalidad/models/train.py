"""Entrenamiento y tracking de experimentos con MLflow."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score

from informalidad.config import (
    feature_columns,
    get_config,
    mlflow_tracking_uri,
    raw_feature_columns,
)
from informalidad.data.split import DataSplits
from informalidad.features import build_model_pipeline
from informalidad.models.candidates import xgb_estimator
from informalidad.models.evaluate import compute_metrics
from informalidad.monitoring.drift import build_reference_profile

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parents[1]

# MLflow 3 serializa los modelos sklearn con skops (más seguro que pickle): solo carga los
# tipos que se declaren como confiables. Estos son los tipos propios del Pipeline.
SKOPS_TRUSTED_TYPES = [
    "informalidad.features.engineering.add_features",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


def setup_mlflow() -> str:
    cfg = get_config()["mlflow"]
    mlflow.set_tracking_uri(mlflow_tracking_uri())
    experiment = mlflow.set_experiment(cfg["experiment_name"])
    return experiment.experiment_id


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target = get_config()["data"]["target"]
    return df[raw_feature_columns()], df[target]


def _simple_params(estimator) -> dict[str, Any]:
    """Solo parámetros escalares (MLflow no acepta objetos complejos como params)."""
    return {
        k: v
        for k, v in estimator.get_params().items()
        if isinstance(v, int | float | str | bool) or v is None
    }


def cross_validate(pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """ROC-AUC con k-fold estratificado sobre una muestra (para que sea rápido en un laptop)."""
    cfg = get_config()
    seed = cfg["project"]["random_state"]
    n = min(cfg["training"]["cv_sample_size"], len(X))
    if n < len(X):
        X = X.sample(n=n, random_state=seed)
        y = y.loc[X.index]
    folds = StratifiedKFold(n_splits=cfg["training"]["cv_folds"], shuffle=True, random_state=seed)
    scores = cross_val_score(clone(pipeline), X, y, cv=folds, scoring="roc_auc")
    return {"cv_roc_auc_mean": float(scores.mean()), "cv_roc_auc_std": float(scores.std())}


def run_candidate(name: str, pipeline, splits: DataSplits, tags: dict[str, str]) -> dict[str, Any]:
    """Entrena un algoritmo candidato y registra params/métricas en un run de MLflow."""
    X_train, y_train = xy(splits.train)
    X_valid, y_valid = xy(splits.valid)
    X_test, y_test = xy(splits.test)

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({**tags, "stage": "comparacion", "model_family": name})
        mlflow.log_params(_simple_params(pipeline.named_steps["model"]))
        mlflow.log_params(splits.sizes())

        cv = cross_validate(pipeline, X_train, y_train)
        pipeline.fit(X_train, y_train)
        metrics = {
            **cv,
            **compute_metrics(y_valid, pipeline.predict_proba(X_valid)[:, 1], prefix="valid_"),
            **compute_metrics(y_test, pipeline.predict_proba(X_test)[:, 1], prefix="test_"),
        }
        mlflow.log_metrics(metrics)

    logger.info(
        "%-20s cv_auc=%.4f valid_auc=%.4f test_auc=%.4f",
        name,
        metrics["cv_roc_auc_mean"],
        metrics["valid_roc_auc"],
        metrics["test_roc_auc"],
    )
    return {"model": name, "run_id": run.info.run_id, **metrics}


def train_final_model(
    splits: DataSplits,
    params: dict[str, Any],
    threshold: float,
    tags: dict[str, str],
    extra_artifacts: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Reentrena con train+valid usando los mejores hiperparámetros y evalúa en test (out-of-time).

    Registra en MLflow: params, métricas, gráficos, importancia de variables, perfil de referencia
    para monitoreo y el modelo (Pipeline completo) con firma e input example.
    """
    from informalidad.models import plots  # import local: matplotlib solo se necesita aquí

    history = pd.concat([splits.train, splits.valid])
    X_hist, y_hist = xy(history)
    X_test, y_test = xy(splits.test)

    pipeline = build_model_pipeline(xgb_estimator(**params))
    pipeline.fit(X_hist, y_hist)
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, proba_test, threshold=threshold, prefix="test_")

    numeric, categorical = feature_columns()
    reference = build_reference_profile(
        X_hist,
        numeric,
        categorical,
        scores=pipeline.predict_proba(X_hist)[:, 1],
        target=y_hist,
    )

    with mlflow.start_run(run_name="xgboost_final") as run:
        mlflow.set_tags({**tags, "stage": "final", "model_family": "xgboost"})
        mlflow.log_params(
            {**params, "threshold": round(threshold, 6), "n_train_final": len(X_hist)}
        )
        mlflow.log_metrics(test_metrics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plots.plot_confusion_matrix(
                y_test, proba_test, threshold, tmp_path / "confusion_matrix_test.png"
            )
            plots.plot_roc_curve(y_test, proba_test, tmp_path / "roc_curve_test.png", "XGBoost")
            importance = plots.feature_importance_frame(pipeline)
            importance.to_csv(tmp_path / "feature_importance.csv", index=False)
            plots.plot_feature_importance(importance, tmp_path / "feature_importance.png")
            report = classification_report(
                y_test,
                (proba_test >= threshold).astype(int),
                target_names=["Formal", "Informal"],
                digits=4,
            )
            (tmp_path / "classification_report_test.txt").write_text(report, encoding="utf-8")
            (tmp_path / "reference_profile.json").write_text(json.dumps(reference) + "\n", "utf-8")
            for name, content in (extra_artifacts or {}).items():
                (tmp_path / name).write_text(json.dumps(content, indent=2), encoding="utf-8")
            mlflow.log_artifacts(tmp, artifact_path="evaluation")

        sample = X_test.head(50)
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=infer_signature(sample, pipeline.predict(sample)),
            input_example=sample.head(3),
            code_paths=[str(PACKAGE_DIR)],
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    logger.info(
        "Modelo final: test_auc=%.4f test_f1=%.4f (umbral %.3f)",
        test_metrics["test_roc_auc"],
        test_metrics["test_f1"],
        threshold,
    )
    return {
        "run_id": run.info.run_id,
        "model_uri": model_info.model_uri,
        "threshold": threshold,
        "metrics": test_metrics,
    }
