"""División de datos: validación fuera de tiempo (out-of-time).

- test: los meses más recientes (configurable). El modelo nunca los ve al entrenar.
- train / valid: el resto, dividido de forma estratificada (valid se usa para Optuna).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from informalidad.config import get_config


@dataclass
class DataSplits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"n_train": len(self.train), "n_valid": len(self.valid), "n_test": len(self.test)}


def temporal_split(df: pd.DataFrame, test_periods: list[str] | None = None) -> DataSplits:
    cfg = get_config()
    data_cfg = cfg["data"]
    period_col, target = data_cfg["period_column"], data_cfg["target"]
    test_periods = test_periods or data_cfg["test_periods"]

    is_test = df[period_col].isin(test_periods)
    if not is_test.any():
        raise ValueError(f"Ningún registro pertenece a los periodos de test {test_periods}")

    history, test = df.loc[~is_test], df.loc[is_test]
    train, valid = train_test_split(
        history,
        test_size=data_cfg["validation_size"],
        stratify=history[target],
        random_state=cfg["project"]["random_state"],
    )
    return DataSplits(train=train, valid=valid, test=test)
