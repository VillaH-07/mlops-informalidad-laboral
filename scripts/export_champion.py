"""Exporta el modelo con alias "champion" del MLflow Model Registry a models/champion/.

El pipeline ya lo hace automáticamente cuando promueve un modelo; este script sirve para
hacerlo a mano (por ejemplo, si cambiaste el alias champion desde la UI de MLflow).

Uso:
    uv run python scripts/export_champion.py
"""

from __future__ import annotations

from informalidad.config import use_project_root
from informalidad.models.registry import export_champion
from informalidad.models.train import setup_mlflow

if __name__ == "__main__":
    use_project_root()
    setup_mlflow()
    path = export_champion()
    print(f"Champion exportado en: {path}")
    print("Siguiente paso: docker compose up --build")
