"""Gráficos que se guardan como artifacts en MLflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: funciona en servidores, Docker y CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402


def plot_confusion_matrix(y_true, y_proba, threshold: float, path: Path) -> Path:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Formal", "Informal"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Matriz de confusión (umbral={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc_curve(y_true, y_proba, path: Path, label: str = "modelo") -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, name=label, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title("Curva ROC - test fuera de tiempo")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def feature_importance_frame(pipeline) -> pd.DataFrame:
    """Importancia de variables del estimador final (si la expone)."""
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["prep"].named_steps["encode"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    frame = pd.DataFrame({"feature": names, "importance": values})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(importance: pd.DataFrame, path: Path, top: int = 20) -> Path:
    top_df = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top_df["feature"], top_df["importance"], color="#2a6fbb")
    ax.set_title(f"Top {top} variables más importantes")
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
