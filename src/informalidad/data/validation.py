"""Validación de calidad de datos (contrato de datos de la GEIH procesada).

Dos niveles:
- Errores críticos (columnas faltantes, target inválido, muy pocas filas) -> se lanza excepción
  y el pipeline se detiene: es mejor no entrenar que entrenar con datos rotos.
- Problemas por fila (valores fuera de rango, códigos desconocidos, nulos) -> se eliminan esas
  filas y se reporta cuántas fueron, para que quede trazabilidad en logs y en MLflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from informalidad.config import get_config, raw_feature_columns

logger = logging.getLogger(__name__)

# Rangos válidos para variables numéricas
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "EDAD": (10, 110),
    "HORAS_SEMANA": (0, 168),  # una semana tiene 168 horas
    "INGRESO_LABORAL": (0, 500_000_000),
}

# Códigos válidos según el diccionario de la GEIH (los observados en los datos)
CATEGORICAL_DOMAINS: dict[str, set[int]] = {
    "SEXO": {1, 2},
    "NIVEL_EDUCATIVO": set(range(1, 14)) | {99},  # P3042: 1 Ninguno ... 13 Doctorado, 99 NS/NR
    "ESTADO_CIVIL": {1, 2, 3, 4, 5, 6},
    "AFILIADO_SALUD": {1, 2, 9},
    "POSICION_OCUPACIONAL": {1, 2, 3, 4, 5, 6, 7, 8},
    "TIENE_CONTRATO": {1, 2},
    "RAMA_ACTIVIDAD": set(range(0, 100)),
}


class DataValidationError(ValueError):
    """Error crítico de calidad de datos."""


@dataclass
class ValidationReport:
    n_rows_in: int
    n_rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def dropped_fraction(self) -> float:
        return 1 - self.n_rows_out / self.n_rows_in if self.n_rows_in else 0.0

    def as_dict(self) -> dict:
        return {
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "dropped_fraction": round(self.dropped_fraction, 6),
            **{f"dropped_{k}": v for k, v in self.dropped.items()},
        }


def validate_dataset(
    df: pd.DataFrame, require_target: bool = True, max_drop_fraction: float = 0.05
) -> tuple[pd.DataFrame, ValidationReport]:
    """Valida el dataset y devuelve (df_limpio, reporte)."""
    cfg = get_config()["data"]
    target = cfg["target"]
    required = raw_feature_columns() + ([target] if require_target else [])

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Faltan columnas obligatorias: {missing}")

    report = ValidationReport(n_rows_in=len(df))
    mask = pd.Series(True, index=df.index)

    nulls = df[required].isna().any(axis=1)
    report.dropped["nulls"] = int(nulls.sum())
    mask &= ~nulls

    for col, (low, high) in NUMERIC_RANGES.items():
        bad = ~df[col].between(low, high) & ~nulls
        report.dropped[f"range_{col}"] = int(bad.sum())
        mask &= ~bad

    for col, domain in CATEGORICAL_DOMAINS.items():
        bad = ~df[col].isin(domain) & ~nulls
        report.dropped[f"domain_{col}"] = int(bad.sum())
        mask &= ~bad

    if require_target:
        bad_target = ~df[target].isin([0, 1])
        if bad_target.any():
            n_bad = int(bad_target.sum())
            raise DataValidationError(
                f"La variable objetivo {target} tiene {n_bad} valores fuera de {{0,1}}"
            )

    clean = df.loc[mask].copy()
    # Contrato de tipos: los códigos categóricos son enteros (ESTADO_CIVIL viene como float)
    for col in CATEGORICAL_DOMAINS:
        clean[col] = clean[col].astype("int64")
    report.n_rows_out = len(clean)

    if report.n_rows_out < cfg["min_rows"]:
        raise DataValidationError(
            f"Quedaron {report.n_rows_out} filas válidas (< mínimo {cfg['min_rows']})"
        )
    if report.dropped_fraction > max_drop_fraction:
        raise DataValidationError(
            f"Se descartó el {report.dropped_fraction:.1%} de las filas "
            f"(máximo permitido {max_drop_fraction:.0%}). Revisar la fuente de datos."
        )

    if require_target:
        rate = clean[target].mean()
        if not 0.05 < rate < 0.95:
            report.warnings.append(f"Tasa de informalidad atípica: {rate:.1%}")

    for w in report.warnings:
        logger.warning(w)
    logger.info(
        "Validación OK: %s -> %s filas (%.2f%% descartadas)",
        report.n_rows_in,
        report.n_rows_out,
        100 * report.dropped_fraction,
    )
    return clean, report
