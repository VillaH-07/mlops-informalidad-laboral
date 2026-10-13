"""Pipeline de entrenamiento orquestado con Prefect.

    carga -> validación -> split temporal -> comparación de algoritmos (CV) ->
    optimización Optuna -> modelo final -> Model Registry (candidate/champion) -> export

Uso:
    uv run python pipelines/training_flow.py                 # dataset completo (o muestra)
    uv run python pipelines/training_flow.py --n-trials 5    # más rápido
    uv run python pipelines/training_flow.py --data-path data/sample/geih_informalidad_sample.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

# Hace que Prefect muestre también los logs de nuestro paquete (no solo los suyos)
os.environ.setdefault("PREFECT_LOGGING_EXTRA_LOGGERS", "informalidad")

import pandas as pd  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact, create_table_artifact  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from informalidad.config import get_config, use_project_root  # noqa: E402
from informalidad.data import (  # noqa: E402
    DataSplits,
    default_data_path,
    file_checksum,
    load_dataset,
    temporal_split,
    validate_dataset,
)

logging.getLogger("informalidad").setLevel(logging.INFO)

# NO_CACHE: las tasks reciben DataFrames grandes; no queremos que Prefect los "hashee" para cache.


@task(name="cargar_datos", retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
def load_data_task(data_path: str | None) -> tuple[pd.DataFrame, dict]:
    path = Path(data_path) if data_path else default_data_path()
    df = load_dataset(path)
    info = {"data_file": path.name, "data_version": file_checksum(path)[:12], "n_rows": len(df)}
    get_run_logger().info("Datos: %s (versión %s)", info["data_file"], info["data_version"])
    return df, info


@task(name="validar_datos", cache_policy=NO_CACHE)
def validate_task(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    clean, report = validate_dataset(df)
    return clean, report.as_dict()


@task(name="split_temporal", cache_policy=NO_CACHE)
def split_task(df: pd.DataFrame) -> DataSplits:
    splits = temporal_split(df)
    get_run_logger().info("Tamaños: %s", splits.sizes())
    return splits


@task(name="comparar_algoritmos", cache_policy=NO_CACHE)
def compare_task(splits: DataSplits, tags: dict) -> pd.DataFrame:
    from informalidad.models.candidates import candidate_models
    from informalidad.models.train import run_candidate

    results = [run_candidate(n, p, splits, tags) for n, p in candidate_models().items()]
    table = pd.DataFrame(results).drop(columns="run_id")
    cols = [
        "model",
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
        "valid_roc_auc",
        "test_roc_auc",
        "test_f1",
    ]
    create_table_artifact(
        key="comparacion-algoritmos",
        table=table[cols].round(4).to_dict(orient="records"),
        description="Comparación de algoritmos (CV + validación + test fuera de tiempo)",
    )
    return table


@task(name="optimizar_xgboost", cache_policy=NO_CACHE)
def optimize_task(splits: DataSplits, n_trials: int, tags: dict) -> dict:
    from informalidad.models.optimize import optimize_xgboost

    return optimize_xgboost(splits, n_trials=n_trials, tags=tags)


@task(name="entrenar_modelo_final", cache_policy=NO_CACHE)
def train_final_task(splits: DataSplits, best: dict, tags: dict, validation: dict) -> dict:
    from informalidad.models.train import train_final_model

    return train_final_model(
        splits,
        params=best["best_params"],
        threshold=best["threshold"],
        tags=tags,
        extra_artifacts={"data_validation_report.json": validation},
    )


@task(name="registrar_modelo", cache_policy=NO_CACHE)
def register_task(final: dict, test_df: pd.DataFrame) -> dict:
    from informalidad.models.registry import register_and_promote

    return register_and_promote(final["run_id"], final["model_uri"], final["metrics"], test_df)


@task(name="exportar_champion")
def export_task() -> str:
    from informalidad.models.registry import export_champion

    return str(export_champion())


@flow(name="informalidad-training-pipeline", log_prints=True)
def training_flow(
    data_path: str | None = None,
    n_trials: int | None = None,
    compare_algorithms: bool = True,
    export_if_promoted: bool = True,
) -> dict:
    use_project_root()
    from informalidad.models.train import setup_mlflow

    logger = get_run_logger()
    cfg = get_config()
    n_trials = n_trials or cfg["training"]["n_trials"]
    setup_mlflow()

    df, data_info = load_data_task(data_path)
    df, validation = validate_task(df)
    splits = split_task(df)

    tags = {
        "data_version": data_info["data_version"],
        "data_file": data_info["data_file"],
        "test_periods": ", ".join(cfg["data"]["test_periods"]),
    }

    comparison = compare_task(splits, tags) if compare_algorithms else None
    best = optimize_task(splits, n_trials, tags)
    final = train_final_task(splits, best, tags, validation)
    registry = register_task(final, splits.test)

    exported = None
    if registry["promoted"] and export_if_promoted:
        exported = export_task()
        logger.info("Champion exportado a %s (listo para la API / Docker)", exported)

    m = final["metrics"]
    comparison_md = (
        comparison[["model", "cv_roc_auc_mean", "valid_roc_auc", "test_roc_auc", "test_f1"]]
        .round(4)
        .to_markdown(index=False)
        if comparison is not None
        else "_(omitida)_"
    )
    create_markdown_artifact(
        key="resumen-entrenamiento",
        markdown=f"""# Resumen del pipeline de entrenamiento

## Datos
- Archivo: `{data_info["data_file"]}` (versión `{data_info["data_version"]}`)
- Filas: {data_info["n_rows"]:,} -> válidas {validation["n_rows_out"]:,}
- Split: {splits.sizes()} | test fuera de tiempo: {tags["test_periods"]}

## Comparación de algoritmos
{comparison_md}

## Modelo final (XGBoost + Optuna, {n_trials} trials)
| Métrica (test) | Valor |
|---|---|
| ROC-AUC | {m["test_roc_auc"]:.4f} |
| F1 | {m["test_f1"]:.4f} |
| Precision | {m["test_precision"]:.4f} |
| Recall | {m["test_recall"]:.4f} |
| Umbral | {final["threshold"]:.3f} |

## Model Registry
- {registry["model_name"]} **v{registry["version"]}** -> {registry["message"]}
- Exportado a: `{exported or "no exportado"}`
""",
        description="Resumen de la corrida de entrenamiento",
    )
    return {"run_id": final["run_id"], "metrics": m, "registry": registry, "exported": exported}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento (Prefect + MLflow)")
    parser.add_argument("--data-path", default=None, help="CSV de entrada (por defecto config)")
    parser.add_argument("--n-trials", type=int, default=None, help="Trials de Optuna")
    parser.add_argument(
        "--skip-comparison", action="store_true", help="No correr la comparación de algoritmos"
    )
    args = parser.parse_args()
    result = training_flow(
        data_path=args.data_path,
        n_trials=args.n_trials,
        compare_algorithms=not args.skip_comparison,
    )
    print("\n=== Pipeline terminado ===")
    print(f"Run final: {result['run_id']}")
    print(f"Registry:  {result['registry']['message']}")
    print("MLflow UI: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db")
