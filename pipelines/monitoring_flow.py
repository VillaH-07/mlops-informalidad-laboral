"""Flujo de monitoreo del modelo en producción (Prefect).

Compara los datos actuales contra el perfil de referencia guardado al entrenar:
    - data drift (PSI por variable) y drift de la predicción
    - desempeño real (si hay etiquetas) y equidad por sexo
    - si hay alertas: lo deja en el reporte y, opcionalmente, dispara el reentrenamiento

Uso:
    uv run python pipelines/monitoring_flow.py                              # último mes de la GEIH
    uv run python pipelines/monitoring_flow.py --periods "Enero 2026"       # un mes específico
    uv run python pipelines/monitoring_flow.py --source api-log             # log de la API
    uv run python pipelines/monitoring_flow.py --simulate-drift             # demo de una alerta
"""

from __future__ import annotations

import argparse
import json
import logging
import os

os.environ.setdefault("PREFECT_LOGGING_EXTRA_LOGGERS", "informalidad")

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from informalidad.config import get_config, resolve_path, use_project_root  # noqa: E402
from informalidad.data import load_dataset, validate_dataset  # noqa: E402
from informalidad.monitoring.report import run_monitoring  # noqa: E402

logging.getLogger("informalidad").setLevel(logging.INFO)


@task(name="cargar_champion")
def load_champion_task() -> tuple[object, dict, dict]:
    model_dir = resolve_path(get_config()["serving"]["model_dir"])
    model = joblib.load(model_dir / "model.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    profile = json.loads((model_dir / "reference_profile.json").read_text(encoding="utf-8"))
    return model, metadata, profile


@task(name="cargar_datos_actuales", cache_policy=NO_CACHE)
def load_current_task(source: str, periods: list[str] | None) -> tuple[pd.DataFrame, str]:
    logger = get_run_logger()
    if source == "api-log":
        path = resolve_path(get_config()["serving"]["prediction_log"])
        if not path.exists():
            raise FileNotFoundError(f"No hay log de predicciones en {path}. Usa la API primero.")
        df = pd.read_json(path, lines=True)
        logger.info("Log de la API: %s predicciones", len(df))
        return df, f"api-log ({path.name})"

    df, _ = validate_dataset(load_dataset())
    period_col = get_config()["data"]["period_column"]
    if not periods:
        # GEIH: "Febrero 2026" -> ordenar por fecha real, no alfabéticamente
        meses = {"Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5, "Junio": 6,
                 "Julio": 7, "Agosto": 8, "Septiembre": 9, "Octubre": 10, "Noviembre": 11,
                 "Diciembre": 12}  # fmt: skip
        key = {p: (int(p.split()[1]), meses[p.split()[0]]) for p in df[period_col].unique()}
        periods = [max(key, key=key.get)]
    current = df[df[period_col].isin(periods)]
    logger.info("Periodo(s) monitoreado(s): %s (%s filas)", periods, len(current))
    return current, f"GEIH {', '.join(periods)}"


def _simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Escenario hipotético: caen los ingresos y aumenta el trabajo por cuenta propia."""
    out = df.copy()
    out["INGRESO_LABORAL"] = out["INGRESO_LABORAL"] * 0.55
    idx = out.sample(frac=0.25, random_state=0).index
    out.loc[idx, "POSICION_OCUPACIONAL"] = 4
    out.loc[idx, "TIENE_CONTRATO"] = 2
    return out


@flow(name="informalidad-monitoring", log_prints=True)
def monitoring_flow(
    source: str = "data",
    periods: list[str] | None = None,
    simulate_drift: bool = False,
    auto_retrain: bool = False,
) -> dict:
    use_project_root()
    logger = get_run_logger()
    model, metadata, profile = load_champion_task()
    current, label = load_current_task(source, periods)
    if simulate_drift:
        current, label = _simulate_drift(current), f"{label} + drift simulado"

    result = run_monitoring(current, model, metadata, profile, source=label)
    md_path, json_path = result.save(resolve_path(get_config()["monitoring"]["reports_dir"]))
    create_markdown_artifact(
        key="reporte-monitoreo", markdown=result.to_markdown(), description="Monitoreo del modelo"
    )

    for alert in result.alerts:
        logger.warning("ALERTA: %s", alert)
    for warning in result.warnings:
        logger.warning("Advertencia: %s", warning)
    logger.info("Reporte guardado en %s", md_path)

    if result.retraining_recommended and auto_retrain:
        from training_flow import training_flow  # mismo directorio pipelines/

        logger.warning("Se dispara el reentrenamiento automático por las alertas detectadas")
        training_flow()

    return {
        "report": str(md_path),
        "alerts": result.alerts,
        "warnings": result.warnings,
        "json": str(json_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitoreo del modelo (drift + desempeño)")
    parser.add_argument("--source", choices=["data", "api-log"], default="data")
    parser.add_argument("--periods", nargs="*", default=None, help='Ej: "Febrero 2026"')
    parser.add_argument("--simulate-drift", action="store_true")
    parser.add_argument("--auto-retrain", action="store_true")
    args = parser.parse_args()
    out = monitoring_flow(
        source=args.source,
        periods=args.periods,
        simulate_drift=args.simulate_drift,
        auto_retrain=args.auto_retrain,
    )
    print(f"\nReporte: {out['report']}")
    print("Alertas:", out["alerts"] or "ninguna")
    print("Advertencias:", out["warnings"] or "ninguna")
