import pandas as pd
import pytest

from informalidad.data import DataValidationError, temporal_split, validate_dataset


def test_sample_passes_validation(sample_df):
    clean, report = validate_dataset(sample_df)
    assert report.n_rows_out == len(clean) > 0
    assert report.dropped_fraction < 0.05


def test_categorical_codes_are_cast_to_int(sample_df):
    clean, _ = validate_dataset(sample_df)
    assert clean["ESTADO_CIVIL"].dtype == "int64"


def test_missing_column_raises(sample_df):
    with pytest.raises(DataValidationError, match="Faltan columnas"):
        validate_dataset(sample_df.drop(columns=["EDAD"]))


def test_out_of_range_rows_are_dropped(sample_df):
    df = sample_df.copy()
    df.loc[df.index[:5], "EDAD"] = 500  # edad imposible
    clean, report = validate_dataset(df)
    assert report.dropped["range_EDAD"] == 5
    assert len(clean) == len(df) - 5


def test_unknown_category_rows_are_dropped(sample_df):
    df = sample_df.copy()
    df.loc[df.index[:3], "SEXO"] = 7
    _, report = validate_dataset(df)
    assert report.dropped["domain_SEXO"] == 3


def test_invalid_target_raises(sample_df):
    df = sample_df.copy()
    df.loc[df.index[0], "INFORMAL"] = 2
    with pytest.raises(DataValidationError, match="fuera de"):
        validate_dataset(df)


def test_too_many_bad_rows_raises(sample_df):
    df = sample_df.copy()
    df.loc[df.index[: len(df) // 5], "HORAS_SEMANA"] = 999
    with pytest.raises(DataValidationError, match="descartó"):
        validate_dataset(df)


def test_temporal_split_keeps_test_periods_out_of_training(sample_df):
    splits = temporal_split(sample_df)
    test_periods = {"Enero 2026", "Febrero 2026"}
    assert set(splits.test["MES_PERIODO"]) == test_periods
    assert not set(splits.train["MES_PERIODO"]) & test_periods
    assert not set(splits.valid["MES_PERIODO"]) & test_periods
    total = len(splits.train) + len(splits.valid) + len(splits.test)
    assert total == len(sample_df)
    assert splits.train.index.intersection(splits.valid.index).empty


def test_temporal_split_without_test_rows_raises(sample_df):
    with pytest.raises(ValueError):
        temporal_split(sample_df, test_periods=["Diciembre 2030"])


def test_validation_without_target(sample_df):
    clean, _ = validate_dataset(sample_df.drop(columns=["INFORMAL"]), require_target=False)
    assert isinstance(clean, pd.DataFrame)
