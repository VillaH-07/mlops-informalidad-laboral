"""Carga de la configuración central (configs/config.yaml).

Todas las rutas se resuelven respecto a la raíz del proyecto, así los scripts
funcionan igual sin importar desde qué carpeta se ejecuten (Windows, Mac o Linux).
Algunas claves se pueden sobreescribir con variables de entorno, útil en Docker y CI.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_PATH = Path(os.getenv("CONFIG_PATH", PROJECT_ROOT / "configs" / "config.yaml"))

# variable de entorno -> (sección, clave)
_ENV_OVERRIDES = {
    "DATA_PATH": ("data", "raw_path"),
    "MLFLOW_TRACKING_URI": ("mlflow", "tracking_uri"),
    "MODEL_DIR": ("serving", "model_dir"),
    "PREDICTION_LOG": ("serving", "prediction_log"),
    "N_TRIALS": ("training", "n_trials"),
}


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Lee el YAML una sola vez y aplica los overrides de entorno."""
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    for env_var, (section, key) in _ENV_OVERRIDES.items():
        value = os.getenv(env_var)
        if value:
            current = cfg[section].get(key)
            cfg[section][key] = int(value) if isinstance(current, int) else value
    return cfg


def resolve_path(relative: str | Path) -> Path:
    """Convierte una ruta relativa del config en ruta absoluta desde la raíz del proyecto."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


def feature_columns() -> tuple[list[str], list[str]]:
    """Devuelve (numéricas, categóricas) tal como vienen en los datos crudos."""
    feats = get_config()["features"]
    return list(feats["numeric"]), list(feats["categorical"])


def raw_feature_columns() -> list[str]:
    numeric, categorical = feature_columns()
    return numeric + categorical


def mlflow_tracking_uri() -> str:
    """URI de MLflow. Por defecto `sqlite:///mlflow.db` (relativa a la raíz del proyecto).

    Se deja relativa a propósito: en Windows las rutas absolutas con espacios o tildes
    (p. ej. C:/Users/Juan José/...) pueden romper la URI de SQLAlchemy. Por eso los
    puntos de entrada llaman a `use_project_root()` antes de tocar MLflow.
    """
    return get_config()["mlflow"]["tracking_uri"]


def use_project_root() -> None:
    """Cambia el directorio de trabajo a la raíz del proyecto (rutas relativas consistentes)."""
    os.chdir(PROJECT_ROOT)
