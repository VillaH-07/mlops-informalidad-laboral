"""Carga del modelo champion exportado (models/champion) y registro de predicciones."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from informalidad.config import get_config, raw_feature_columns, resolve_path

logger = logging.getLogger(__name__)


class ModelService:
    """Envuelve el modelo: carga una sola vez, predice y deja log de cada predicción."""

    def __init__(
        self, model_dir: str | Path | None = None, prediction_log: str | Path | None = None
    ):
        serving = get_config()["serving"]
        self.model_dir = Path(model_dir) if model_dir else resolve_path(serving["model_dir"])
        log_path = prediction_log or serving["prediction_log"]
        self.prediction_log = (
            Path(log_path) if Path(log_path).is_absolute() else resolve_path(log_path)
        )
        self.model = None
        self.metadata: dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "desconocida"))

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    def load(self) -> None:
        model_path = self.model_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No existe {model_path}. Corre el pipeline de entrenamiento o "
                "`uv run python scripts/export_champion.py` antes de levantar la API."
            )
        self.model = joblib.load(model_path)
        self.metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        logger.info("Modelo %s v%s cargado", self.metadata.get("model_name"), self.version)

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.is_loaded:
            raise RuntimeError("El modelo no está cargado")
        frame = pd.DataFrame.from_records(records, columns=raw_feature_columns())
        proba = self.model.predict_proba(frame)[:, 1]
        preds = (proba >= self.threshold).astype(int)
        return [
            {
                "probabilidad_informal": round(float(p), 4),
                "prediccion": int(y),
                "etiqueta": "Informal" if y == 1 else "Formal",
                "umbral": round(self.threshold, 4),
                "version_modelo": self.version,
            }
            for p, y in zip(np.asarray(proba), preds, strict=True)
        ]

    def log_predictions(self, records: list[dict], results: list[dict], latency_ms: float) -> None:
        """Agrega cada predicción a un archivo JSONL: es la materia prima del monitoreo."""
        timestamp = datetime.now(tz=UTC).isoformat()
        lines = [
            json.dumps(
                {
                    "timestamp": timestamp,
                    "model_version": self.version,
                    "latency_ms": round(latency_ms, 2),
                    **rec,
                    "probabilidad_informal": res["probabilidad_informal"],
                    "prediccion": res["prediccion"],
                }
            )
            for rec, res in zip(records, results, strict=True)
        ]
        try:
            self.prediction_log.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, open(self.prediction_log, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError as exc:  # el log nunca debe tumbar una predicción
            logger.warning("No se pudo escribir el log de predicciones: %s", exc)
