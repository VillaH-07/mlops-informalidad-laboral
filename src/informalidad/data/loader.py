"""Carga de datos y versionado ligero (checksum) del dataset."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from informalidad.config import get_config, resolve_path

logger = logging.getLogger(__name__)


def default_data_path() -> Path:
    """Usa el dataset completo si existe; si no, la muestra versionada en git."""
    cfg = get_config()["data"]
    full = resolve_path(cfg["raw_path"])
    if full.exists():
        return full
    sample = resolve_path(cfg["sample_path"])
    logger.warning("No se encontró %s; se usará la muestra %s", full, sample)
    return sample


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Lee el CSV procesado de la GEIH (una fila = un trabajador ocupado)."""
    path = Path(path) if path else default_data_path()
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de datos: {path}")
    df = pd.read_csv(path)
    logger.info("Datos cargados: %s filas x %s columnas desde %s", *df.shape, path.name)
    return df


def file_checksum(path: str | Path, algorithm: str = "md5") -> str:
    """Huella del archivo de datos: si cambia el archivo, cambia la versión de datos."""
    digest = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
