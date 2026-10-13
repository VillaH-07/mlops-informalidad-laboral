"""API REST (FastAPI) para predecir informalidad laboral.

Endpoints:
    GET  /               interfaz web sencilla para probar el modelo
    GET  /health         health check (lo usa Docker)
    GET  /model/info     versión, métricas y umbral del modelo en servicio
    POST /predict        predicción para un trabajador
    POST /predict/batch  predicción para hasta 1000 trabajadores
    GET  /docs           documentación interactiva (Swagger)

Ejecutar en local:
    uv run uvicorn informalidad.api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from informalidad.api.model_loader import ModelService
from informalidad.api.schemas import (
    InfoModelo,
    LoteTrabajadores,
    Prediccion,
    PrediccionLote,
    Salud,
    Trabajador,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("informalidad.api")

LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "true").lower() == "true"
STATIC_DIR = Path(__file__).parent / "static"

service = ModelService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # El modelo se carga UNA vez al arrancar, no en cada request
    service.load()
    yield


app = FastAPI(
    title="API Informalidad Laboral - GEIH",
    description=(
        "Predice la probabilidad de que un trabajador ocupado en Colombia sea informal "
        "(no cotiza a pensión), con un modelo XGBoost entrenado sobre la GEIH del DANE."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _run_prediction(records: list[dict]) -> list[dict]:
    start = time.perf_counter()
    try:
        results = service.predict(records)
    except Exception as exc:
        logger.exception("Error al predecir")
        raise HTTPException(status_code=500, detail=f"Error al predecir: {exc}") from exc
    latency_ms = (time.perf_counter() - start) * 1000
    if LOG_PREDICTIONS:
        service.log_predictions(records, results, latency_ms)
    return results


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health", response_model=Salud)
def health() -> Salud:
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return Salud(status="ok", modelo_cargado=True, version_modelo=service.version)


@app.get("/model/info", response_model=InfoModelo)
def model_info() -> InfoModelo:
    return InfoModelo(**service.metadata)


@app.post("/predict", response_model=Prediccion)
def predict(trabajador: Trabajador) -> Prediccion:
    return Prediccion(**_run_prediction([trabajador.model_dump()])[0])


@app.post("/predict/batch", response_model=PrediccionLote)
def predict_batch(lote: LoteTrabajadores) -> PrediccionLote:
    results = _run_prediction([t.model_dump() for t in lote.trabajadores])
    rate = sum(r["prediccion"] for r in results) / len(results)
    return PrediccionLote(
        predicciones=[Prediccion(**r) for r in results],
        total=len(results),
        tasa_informalidad_predicha=round(rate, 4),
    )
