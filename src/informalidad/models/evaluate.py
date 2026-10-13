"""Métricas de clasificación y selección del umbral de decisión."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_proba, threshold: float = 0.5, prefix: str = "") -> dict[str, float]:
    """Métricas estándar. La clase positiva (1) es INFORMAL."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_proba),
    }
    return {f"{prefix}{k}": float(v) for k, v in metrics.items()}


def best_f1_threshold(y_true, y_proba) -> float:
    """Umbral que maximiza el F1 de la clase informal (se calcula en validación, nunca en test)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # precision_recall_curve devuelve un punto más de precision/recall que de thresholds
    best = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best])
