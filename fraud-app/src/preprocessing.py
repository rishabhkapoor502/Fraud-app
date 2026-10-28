"""Dataset loading, cleaning, stratified splits, and scaling."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    SCALER_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from src.validation import DataValidationError


class DatasetNotFoundError(FileNotFoundError):
    """Raised when creditcard.csv is not in the expected location."""


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path else DATASET_PATH
    if not dataset_path.exists():
        raise DatasetNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Download the ULB Credit Card Fraud Detection CSV from Kaggle "
            "and save it as data/creditcard.csv."
        )

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise DataValidationError(f"{dataset_path.name} is empty.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(
            "Dataset is missing required columns: " + ", ".join(missing) + "."
        )
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    working = df[REQUIRED_COLUMNS].copy()
    working = working.dropna()
    working = working.drop_duplicates()
    working[TARGET_COLUMN] = working[TARGET_COLUMN].astype(int)

    invalid_labels = set(working[TARGET_COLUMN].unique()) - {0, 1}
    if invalid_labels:
        raise DataValidationError(
            "Class column must contain only 0 (legitimate) and 1 (fraud)."
        )
    if working.empty:
        raise DataValidationError("No usable rows remain after cleaning.")
    return working.reset_index(drop=True)


def stratified_splits(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Create stratified train, validation, and test splits.

    Test is held out first. Validation is then split from the remaining data.
    Neither split is used for fitting the scaler or training the network.
    """
    features = df[FEATURE_COLUMNS]
    labels = df[TARGET_COLUMN]

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    relative_val = validation_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val,
        stratify=y_train_val,
        random_state=random_state,
    )

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def fit_scaler(x_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(x_train[FEATURE_COLUMNS])
    return scaler


def transform_features(scaler: StandardScaler, features: pd.DataFrame) -> np.ndarray:
    return scaler.transform(features[FEATURE_COLUMNS])


def save_scaler(scaler: StandardScaler, path: Path | None = None) -> Path:
    output = path or SCALER_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output)
    return output


def load_scaler(path: Path | None = None) -> StandardScaler:
    scaler_path = path or SCALER_PATH
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Train the model first with train_model.py."
        )
    return joblib.load(scaler_path)


def class_weights_from_labels(y_train: pd.Series) -> dict[int, float]:
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.to_numpy(),
    )
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def split_summary(splits: dict[str, pd.DataFrame | pd.Series]) -> dict:
    summary = {}
    for name in ("train", "val", "test"):
        labels = splits[f"y_{name}"]
        summary[name] = {
            "rows": int(len(labels)),
            "legitimate": int((labels == 0).sum()),
            "fraud": int((labels == 1).sum()),
            "fraud_rate": float((labels == 1).mean()),
        }
    return summary
