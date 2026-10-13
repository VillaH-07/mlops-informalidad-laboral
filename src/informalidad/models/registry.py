"""Model Registry: registro con aliases (candidate / champion) y exportación del champion.

Regla de promoción (champion/challenger): toda versión nueva queda como "candidate". Solo pasa a
"champion" si supera al champion actual evaluados AMBOS sobre el mismo test fuera de tiempo.
Así un reentrenamiento automático nunca reemplaza un buen modelo por uno peor.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
import sklearn
import xgboost
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

from informalidad.config import feature_columns, get_config, raw_feature_columns, resolve_path

logger = logging.getLogger(__name__)


def _registry_cfg() -> dict[str, str]:
    return get_config()["mlflow"]


def champion_score_on(test_df: pd.DataFrame) -> float | None:
    """ROC-AUC del champion actual sobre el mismo test del candidato (None si no hay champion)."""
    cfg = _registry_cfg()
    target = get_config()["data"]["target"]
    uri = f"models:/{cfg['registered_model_name']}@{cfg['champion_alias']}"
    try:
        champion = mlflow.sklearn.load_model(uri)
    except (MlflowException, OSError) as exc:
        logger.info("No hay champion todavía (%s)", type(exc).__name__)
        return None
    try:
        proba = champion.predict_proba(test_df[raw_feature_columns()])[:, 1]
    except Exception as exc:  # p. ej. el champion usa otras features
        logger.warning("No se pudo evaluar el champion en el test actual: %s", exc)
        return None
    return float(roc_auc_score(test_df[target], proba))


def register_and_promote(
    run_id: str, model_uri: str, metrics: dict[str, float], test_df: pd.DataFrame
) -> dict[str, Any]:
    cfg = _registry_cfg()
    name = cfg["registered_model_name"]
    metric_name = cfg["promotion_metric"]
    client = MlflowClient()

    champion_score = champion_score_on(test_df)  # se evalúa ANTES de registrar el nuevo
    version = mlflow.register_model(model_uri=model_uri, name=name).version
    candidate_score = metrics[metric_name]

    client.update_model_version(
        name=name,
        version=version,
        description=(
            f"XGBoost (Pipeline sklearn) para predecir informalidad laboral (GEIH). "
            f"{metric_name}={candidate_score:.4f} en test fuera de tiempo."
        ),
    )
    for key, value in {
        metric_name: f"{candidate_score:.4f}",
        "model_type": "xgboost",
        "orchestrator": "prefect",
        "hpo": "optuna",
        "run_id": run_id,
    }.items():
        client.set_model_version_tag(name, version, key, value)

    client.set_registered_model_alias(name, cfg["candidate_alias"], version)
    promoted = champion_score is None or candidate_score > champion_score
    if promoted:
        client.set_registered_model_alias(name, cfg["champion_alias"], version)
    client.set_model_version_tag(name, version, "promoted_to_champion", str(promoted))

    if champion_score is None:
        msg = f"v{version} es el primer modelo: se promueve a champion"
    elif promoted:
        msg = f"v{version} ({candidate_score:.4f}) supera al champion ({champion_score:.4f})"
    else:
        msg = f"v{version} ({candidate_score:.4f}) NO supera al champion ({champion_score:.4f})"
    logger.info(msg)
    return {
        "model_name": name,
        "version": str(version),
        "promoted": promoted,
        "candidate_score": candidate_score,
        "champion_score_same_data": champion_score,
        "message": msg,
    }


def export_champion(out_dir: str | Path | None = None) -> Path:
    """Copia el champion del registry a una carpeta local (la que usa la API y Docker).

    Se exporta con joblib para que la imagen Docker NO necesite MLflow (imagen más liviana).
    Junto al modelo van metadata.json (versión, métricas, umbral) y el perfil de referencia
    para monitoreo. Nota: joblib/pickle solo debe cargarse desde fuentes confiables.
    """
    cfg = _registry_cfg()
    name, alias = cfg["registered_model_name"], cfg["champion_alias"]
    out = Path(out_dir) if out_dir else resolve_path(get_config()["serving"]["model_dir"])
    out.mkdir(parents=True, exist_ok=True)

    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    run = client.get_run(mv.run_id)
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    joblib.dump(model, out / "model.joblib", compress=3)

    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path="evaluation/reference_profile.json", dst_path=tmp
        )
        shutil.copy(local, out / "reference_profile.json")

    numeric, categorical = feature_columns()
    metadata = {
        "model_name": name,
        "version": str(mv.version),
        "alias": alias,
        "run_id": mv.run_id,
        "threshold": float(run.data.params["threshold"]),
        "metrics": {k: round(v, 4) for k, v in run.data.metrics.items() if k.startswith("test_")},
        "features": {"numeric": numeric, "categorical": categorical},
        "data_version": run.data.tags.get("data_version"),
        "test_periods": run.data.tags.get("test_periods"),
        "trained_at": datetime.fromtimestamp(run.info.start_time / 1000, tz=UTC).isoformat(),
        "exported_at": datetime.now(tz=UTC).isoformat(),
        "library_versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("Champion v%s exportado a %s", mv.version, out)
    return out
