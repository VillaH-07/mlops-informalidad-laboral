import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from informalidad.monitoring.drift import build_reference_profile, compute_drift, psi
from informalidad.monitoring.report import run_monitoring


def test_psi_is_zero_for_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert psi(p, p) == 0.0


def test_psi_detects_large_shift():
    assert psi(np.array([0.5, 0.5]), np.array([0.9, 0.1])) > 0.2


def test_numeric_shift_is_flagged():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    cur = pd.DataFrame({"x": rng.normal(1.5, 1, 5000)})
    profile = build_reference_profile(ref, ["x"], [])
    table = compute_drift(cur, profile)
    assert table.loc[0, "nivel"] == "alerta"


def test_same_distribution_is_stable():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    cur = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    table = compute_drift(cur, build_reference_profile(ref, ["x"], []))
    assert table.loc[0, "nivel"] == "estable"


def test_categorical_codes_int_and_float_are_equivalent():
    ref = pd.DataFrame({"c": [1.0, 2.0, 2.0, 3.0] * 100})
    cur = pd.DataFrame({"c": [1, 2, 2, 3] * 100})
    table = compute_drift(cur, build_reference_profile(ref, [], ["c"]))
    assert table.loc[0, "psi"] < 1e-6


def _load(model_dir):
    d = Path(model_dir)
    return (
        joblib.load(d / "model.joblib"),
        json.loads((d / "metadata.json").read_text()),
        json.loads((d / "reference_profile.json").read_text()),
    )


def test_run_monitoring_with_labels(model_dir, sample_df):
    model, meta, profile = _load(model_dir)
    result = run_monitoring(sample_df, model, meta, profile, source="test")
    assert result.performance is not None
    assert {"Hombre", "Mujer"} == set(result.fairness["grupo"])
    assert "Reporte de monitoreo" in result.to_markdown()


def test_small_samples_do_not_trigger_drift_alerts(model_dir, sample_df):
    model, meta, profile = _load(model_dir)
    tiny = sample_df.drop(columns=["INFORMAL"]).head(20)
    result = run_monitoring(tiny, model, meta, profile, source="test")
    assert not result.alerts
    assert any("no es confiable" in w for w in result.warnings)


def test_simulated_income_drop_triggers_warning(model_dir, sample_df):
    model, meta, profile = _load(model_dir)
    shifted = sample_df.copy()
    shifted["INGRESO_LABORAL"] = shifted["INGRESO_LABORAL"] * 0.3
    result = run_monitoring(shifted, model, meta, profile, source="test")
    assert any("INGRESO_LABORAL" in w for w in result.warnings)
