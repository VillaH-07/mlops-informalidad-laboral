import numpy as np
import pandas as pd

from informalidad.config import raw_feature_columns
from informalidad.features import add_features, build_preprocessor, model_numeric_columns
from informalidad.models.evaluate import best_f1_threshold, compute_metrics


def test_add_features_creates_log_columns(worker):
    out = add_features(pd.DataFrame([worker]))
    assert np.isclose(out.loc[0, "LOG_INGRESO_LABORAL"], np.log1p(1_300_000))
    expected_hourly = np.log1p(1_300_000 / (48 * 4.33))
    assert np.isclose(out.loc[0, "LOG_INGRESO_HORA"], expected_hourly)


def test_add_features_handles_zero_hours_and_income(worker):
    row = {**worker, "HORAS_SEMANA": 0, "INGRESO_LABORAL": 0}
    out = add_features(pd.DataFrame([row]))
    assert np.isfinite(out[["LOG_INGRESO_LABORAL", "LOG_INGRESO_HORA"]].to_numpy()).all()


def test_add_features_does_not_mutate_input(worker):
    df = pd.DataFrame([worker])
    add_features(df)
    assert "LOG_INGRESO_LABORAL" not in df.columns


def test_preprocessor_output_and_unknown_category(sample_df, worker):
    prep = build_preprocessor().fit(sample_df[raw_feature_columns()])
    unseen = pd.DataFrame([{**worker, "RAMA_ACTIVIDAD": 4}])  # código poco frecuente/no visto
    out = prep.transform(unseen)
    assert out.shape[0] == 1
    names = prep.named_steps["encode"].get_feature_names_out()
    assert set(model_numeric_columns()).issubset(set(names))


def test_compute_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), prefix="test_")
    assert m["test_roc_auc"] == 1.0
    assert m["test_f1"] == 1.0
    assert set(m) >= {"test_precision", "test_recall", "test_accuracy", "test_brier"}


def test_best_threshold_is_a_probability():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 500), 0, 1)
    t = best_f1_threshold(y, proba)
    assert 0.0 < t < 1.0
