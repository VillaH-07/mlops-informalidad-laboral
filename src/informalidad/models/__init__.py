"""Modelos: candidatos, evaluación, entrenamiento, optimización y registry.

Los submódulos que dependen de MLflow/Optuna (train, optimize, registry) se importan
explícitamente para que la API (que no instala MLflow) pueda usar el paquete.
"""

from informalidad.models.evaluate import best_f1_threshold, compute_metrics

__all__ = ["best_f1_threshold", "compute_metrics"]
