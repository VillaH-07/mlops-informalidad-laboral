"""Reporte de monitoreo: drift de datos, drift de predicción, desempeño y equidad.

Lógica de decisión (data drift no siempre implica que el modelo empeoró):
- ADVERTENCIA: una variable de entrada cambia de distribución (PSI >= 0.2). Se investiga.
- ALERTA (reentrenar): cae el desempeño real (AUC) o cambia la distribución de la PREDICCIÓN.

Funciona en dos modos:
- Con etiquetas (p. ej. un mes nuevo de la GEIH): drift + desempeño real (AUC, F1) + equidad.
- Sin etiquetas (log de predicciones de la API): solo drift de entradas y de predicciones,
  que es lo único medible en tiempo real porque la "verdad" llega con retraso.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from informalidad.config import get_config, raw_feature_columns
from informalidad.models.evaluate import compute_metrics
from informalidad.monitoring.drift import compute_drift

SEXO_LABELS = {1: "Hombre", 2: "Mujer"}


@dataclass
class MonitoringResult:
    source: str
    n_rows: int
    drift: pd.DataFrame
    performance: dict[str, float] | None = None
    reference_performance: dict[str, float] | None = None
    fairness: pd.DataFrame | None = None
    alerts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())

    @property
    def retraining_recommended(self) -> bool:
        return bool(self.alerts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "source": self.source,
            "n_rows": self.n_rows,
            "alerts": self.alerts,
            "warnings": self.warnings,
            "retraining_recommended": self.retraining_recommended,
            "performance": self.performance,
            "reference_performance": self.reference_performance,
            "drift": self.drift.round(4).to_dict(orient="records"),
            "fairness": None if self.fairness is None else self.fairness.to_dict(orient="records"),
        }

    def to_markdown(self) -> str:
        if self.alerts:
            status = "🔴 Reentrenamiento recomendado"
        elif self.warnings:
            status = "🟡 Estable, con advertencias de drift"
        else:
            status = "🟢 Modelo estable"
        lines = [
            "# Reporte de monitoreo del modelo",
            "",
            f"- Generado: {self.generated_at}",
            f"- Fuente: `{self.source}` ({self.n_rows:,} registros)",
            f"- Estado: **{status}**",
            "",
        ]
        if self.alerts:
            lines += ["## Alertas", *[f"- {a}" for a in self.alerts], ""]
        if self.warnings:
            lines += ["## Advertencias", *[f"- {w}" for w in self.warnings], ""]
        if self.performance:
            ref = self.reference_performance or {}
            lines += ["## Desempeño", "| Métrica | Actual | Referencia (test) |", "|---|---|---|"]
            for k in ["roc_auc", "f1", "precision", "recall"]:
                ref_v = ref.get(f"test_{k}")
                ref_txt = f"{ref_v:.4f}" if ref_v is not None else "-"
                lines.append(f"| {k} | {self.performance[k]:.4f} | {ref_txt} |")
            lines.append("")
        lines += ["## Data drift (PSI)", self.drift.round(4).to_markdown(index=False), ""]
        if self.fairness is not None:
            lines += ["## Equidad por sexo", self.fairness.round(4).to_markdown(index=False), ""]
        return "\n".join(lines)

    def save(self, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
        md_path = out_dir / f"monitoring_{stamp}.md"
        json_path = out_dir / f"monitoring_{stamp}.json"
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
        )
        return md_path, json_path


def fairness_by_group(y_true, y_proba, threshold: float, groups: pd.Series) -> pd.DataFrame:
    rows = []
    for value, idx in groups.groupby(groups).groups.items():
        m = compute_metrics(np.asarray(y_true)[idx], np.asarray(y_proba)[idx], threshold)
        rows.append(
            {
                "grupo": SEXO_LABELS.get(value, value),
                "n": len(idx),
                "tasa_real": float(np.asarray(y_true)[idx].mean()),
                "tasa_predicha": float((np.asarray(y_proba)[idx] >= threshold).mean()),
                "recall": m["recall"],
                "precision": m["precision"],
            }
        )
    return pd.DataFrame(rows)


def run_monitoring(
    current: pd.DataFrame,
    model,
    metadata: dict[str, Any],
    reference_profile: dict[str, Any],
    source: str,
) -> MonitoringResult:
    cfg = get_config()
    mon = cfg["monitoring"]
    target = cfg["data"]["target"]
    threshold = float(metadata.get("threshold", 0.5))

    current = current.reset_index(drop=True)
    X = current[raw_feature_columns()]
    scores = model.predict_proba(X)[:, 1]
    drift = compute_drift(X, reference_profile, scores, mon["psi_warning"], mon["psi_alert"])
    result = MonitoringResult(source=source, n_rows=len(current), drift=drift)
    result.reference_performance = metadata.get("metrics")

    if len(current) < mon["min_rows_psi"]:
        result.warnings.append(
            f"Solo {len(current)} registros: el PSI no es confiable con menos de "
            f"{mon['min_rows_psi']}. No se generan alertas de drift."
        )
    else:
        for row in drift.itertuples():
            if row.nivel != "alerta":
                continue
            msg = f"Drift en {row.variable} (PSI={row.psi:.3f})"
            if row.variable == "PREDICCION":
                result.alerts.append(msg + ": cambió la distribución de las predicciones")
            else:
                result.warnings.append(msg)

    if target in current.columns and current[target].nunique() == 2:
        perf = compute_metrics(current[target], scores, threshold)
        result.performance = perf
        ref_auc = (metadata.get("metrics") or {}).get("test_roc_auc")
        if ref_auc is not None and ref_auc - perf["roc_auc"] > mon["max_auc_drop"]:
            result.alerts.append(
                f"Caída de desempeño: AUC {perf['roc_auc']:.4f} vs {ref_auc:.4f} en referencia"
            )
        result.fairness = fairness_by_group(current[target], scores, threshold, current["SEXO"])
    return result
