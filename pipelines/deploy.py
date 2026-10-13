"""Deployments programados de Prefect (igual que en el módulo 03 del curso, con `serve`).

- Reentrenamiento mensual: la GEIH se publica una vez al mes, así que reentrenar con más
  frecuencia no aporta. Día 15 de cada mes, 6:00 a.m. (hora Colombia).
- Monitoreo semanal: lunes 7:00 a.m., revisa drift y desempeño del champion.

Uso (deja el proceso corriendo; Ctrl+C para detener):
    uv run prefect server start                 # terminal 1 (UI en http://127.0.0.1:4200)
    uv run python pipelines/deploy.py           # terminal 2
    uv run prefect deployment run "informalidad-training-pipeline/reentrenamiento-mensual"
"""

from __future__ import annotations

from monitoring_flow import monitoring_flow
from prefect import serve
from prefect.schedules import Cron
from training_flow import training_flow

TIMEZONE = "America/Bogota"

if __name__ == "__main__":
    training = training_flow.to_deployment(
        name="reentrenamiento-mensual",
        schedule=Cron("0 6 15 * *", timezone=TIMEZONE),
        parameters={"n_trials": 20},
        tags=["informalidad", "training"],
        description="Reentrena, registra en MLflow y promueve a champion solo si mejora.",
    )
    monitoring = monitoring_flow.to_deployment(
        name="monitoreo-semanal",
        schedule=Cron("0 7 * * 1", timezone=TIMEZONE),
        parameters={"source": "data", "auto_retrain": False},
        tags=["informalidad", "monitoring"],
        description="Calcula drift (PSI), desempeño y equidad del champion.",
    )
    serve(training, monitoring)
