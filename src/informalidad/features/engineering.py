"""Feature engineering reproducible.

Todo el preprocesamiento vive DENTRO del sklearn Pipeline del modelo. Así, el mismo objeto
que se entrena es el que se despliega en la API: imposible que entrenamiento y producción
transformen los datos de forma distinta (training-serving skew).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from informalidad.config import feature_columns

SEMANAS_POR_MES = 4.33

ENGINEERED_NUMERIC = ["LOG_INGRESO_LABORAL", "LOG_INGRESO_HORA"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas a partir de las columnas crudas de la GEIH.

    - LOG_INGRESO_LABORAL: el ingreso tiene asimetría muy alta (12.1); el log la reduce.
    - LOG_INGRESO_HORA: ingreso por hora trabajada; separa "gana poco porque trabaja pocas
      horas" de "gana poco por hora", que es más informativo sobre la calidad del empleo.
    """
    out = df.copy()
    ingreso = out["INGRESO_LABORAL"].clip(lower=0).astype(float)
    horas_mes = (out["HORAS_SEMANA"].astype(float) * SEMANAS_POR_MES).clip(lower=1)
    out["LOG_INGRESO_LABORAL"] = np.log1p(ingreso)
    out["LOG_INGRESO_HORA"] = np.log1p(ingreso / horas_mes)
    return out


def model_numeric_columns() -> list[str]:
    numeric, _ = feature_columns()
    # INGRESO_LABORAL crudo se reemplaza por sus versiones en log
    return [c for c in numeric if c != "INGRESO_LABORAL"] + ENGINEERED_NUMERIC


def build_preprocessor() -> Pipeline:
    """Feature engineering + encoding. Se reutiliza para todos los algoritmos."""
    _, categorical = feature_columns()
    encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), model_numeric_columns()),
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(add_features, validate=False)),
            ("encode", encoder),
        ]
    )


def build_model_pipeline(estimator) -> Pipeline:
    """Pipeline completo: preprocesamiento + estimador."""
    return Pipeline(steps=[("prep", build_preprocessor()), ("model", estimator)])
