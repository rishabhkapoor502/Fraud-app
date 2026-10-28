import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.preprocessing import (
    class_weights_from_labels,
    clean_dataset,
    fit_scaler,
    split_summary,
    stratified_splits,
    transform_features,
)
from src.validation import DataValidationError


def _toy_dataset(n_legit: int = 80, n_fraud: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n_legit):
        row = {col: float(i) for col in FEATURE_COLUMNS}
        row[TARGET_COLUMN] = 0
        rows.append(row)
    for i in range(n_fraud):
        row = {col: float(1000 + i) for col in FEATURE_COLUMNS}
        row[TARGET_COLUMN] = 1
        rows.append(row)
    return pd.DataFrame(rows)


def test_clean_dataset_drops_duplicates_and_nulls():
    df = _toy_dataset(4, 2)
    df.loc[0, "Amount"] = None
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)
    cleaned = clean_dataset(df)
    assert cleaned["Amount"].notna().all()
    assert cleaned.duplicated().sum() == 0
    assert set(cleaned[TARGET_COLUMN].unique()) <= {0, 1}


def test_invalid_class_values_rejected():
    df = _toy_dataset(4, 2)
    df.loc[0, TARGET_COLUMN] = 2
    with pytest.raises(DataValidationError):
        clean_dataset(df)


def test_stratified_splits_preserve_both_classes_and_exclude_target_from_x():
    df = clean_dataset(_toy_dataset())
    splits = stratified_splits(df)
    for part in ("train", "val", "test"):
        x = splits[f"x_{part}"]
        y = splits[f"y_{part}"]
        assert list(x.columns) == FEATURE_COLUMNS
        assert TARGET_COLUMN not in x.columns
        assert set(y.unique()) == {0, 1}
    total = len(splits["x_train"]) + len(splits["x_val"]) + len(splits["x_test"])
    assert total == len(df)


def test_scaler_fits_on_train_only_shape():
    df = clean_dataset(_toy_dataset())
    splits = stratified_splits(df)
    scaler = fit_scaler(splits["x_train"])
    transformed = transform_features(scaler, splits["x_val"])
    assert transformed.shape == (len(splits["x_val"]), len(FEATURE_COLUMNS))


def test_class_weights_penalize_majority_less():
    labels = pd.Series([0] * 90 + [1] * 10)
    weights = class_weights_from_labels(labels)
    assert weights[1] > weights[0]


def test_split_summary_counts():
    df = clean_dataset(_toy_dataset())
    summary = split_summary(stratified_splits(df))
    assert summary["train"]["rows"] + summary["val"]["rows"] + summary["test"]["rows"] == len(df)
