"""Detección de data drift con PSI (Population Stability Index).

Idea: al entrenar se guarda un "perfil de referencia" (cómo se distribuye cada variable en los
datos de entrenamiento). En producción se compara la distribución actual contra ese perfil.

Interpretación estándar del PSI (usada en banca y riesgo de crédito):
    PSI < 0.1        -> sin cambio relevante
    0.1 <= PSI < 0.2 -> cambio moderado (vigilar)
    PSI >= 0.2       -> cambio significativo (alerta: evaluar reentrenamiento)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI entre dos vectores de proporciones (cada uno suma 1)."""
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_profile(values: pd.Series, n_bins: int) -> dict[str, Any]:
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(values.dropna(), quantiles))
    counts = np.bincount(np.digitize(values.dropna(), edges), minlength=len(edges) + 1)
    return {
        "type": "numeric",
        "edges": edges.tolist(),
        "proportions": (counts / counts.sum()).tolist(),
        "mean": float(values.mean()),
    }


def _as_category(values: pd.Series) -> pd.Series:
    """Normaliza códigos: 2, 2.0 y "2" deben contar como la misma categoría."""
    if pd.api.types.is_numeric_dtype(values):
        return values.round().astype("Int64").astype(str)
    return values.astype(str)


def _categorical_profile(values: pd.Series) -> dict[str, Any]:
    props = _as_category(values).value_counts(normalize=True)
    return {"type": "categorical", "proportions": {k: float(v) for k, v in props.items()}}


def build_reference_profile(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    scores: np.ndarray | None = None,
    target: pd.Series | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Resume la distribución de referencia (datos de entrenamiento)."""
    profile: dict[str, Any] = {"n_rows": len(df), "features": {}}
    for col in numeric:
        profile["features"][col] = _numeric_profile(df[col], n_bins)
    for col in categorical:
        profile["features"][col] = _categorical_profile(df[col])
    if scores is not None:
        profile["prediction"] = _numeric_profile(pd.Series(scores), n_bins)
    if target is not None:
        profile["target_rate"] = float(np.mean(target))
    return profile


def _current_proportions(values: pd.Series, ref: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if ref["type"] == "numeric":
        edges = np.asarray(ref["edges"])
        counts = np.bincount(np.digitize(values.dropna(), edges), minlength=len(edges) + 1)
        return np.asarray(ref["proportions"]), counts / max(counts.sum(), 1)
    ref_props = ref["proportions"]
    cur = _as_category(values).value_counts(normalize=True)
    categories = sorted(set(ref_props) | set(cur.index))
    expected = np.array([ref_props.get(c, 0.0) for c in categories])
    actual = np.array([cur.get(c, 0.0) for c in categories])
    return expected, actual


def drift_level(value: float, warning: float = 0.1, alert: float = 0.2) -> str:
    if value >= alert:
        return "alerta"
    if value >= warning:
        return "moderado"
    return "estable"


def compute_drift(
    current: pd.DataFrame,
    reference_profile: dict[str, Any],
    scores: np.ndarray | None = None,
    warning: float = 0.1,
    alert: float = 0.2,
) -> pd.DataFrame:
    """Tabla con el PSI de cada variable (y de la predicción) frente a la referencia."""
    rows = []
    for col, ref in reference_profile["features"].items():
        if col not in current.columns:
            continue
        expected, actual = _current_proportions(current[col], ref)
        value = psi(expected, actual)
        rows.append({"variable": col, "psi": value, "nivel": drift_level(value, warning, alert)})
    if scores is not None and "prediction" in reference_profile:
        expected, actual = _current_proportions(pd.Series(scores), reference_profile["prediction"])
        value = psi(expected, actual)
        rows.append(
            {"variable": "PREDICCION", "psi": value, "nivel": drift_level(value, warning, alert)}
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
