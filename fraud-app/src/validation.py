"""Input validation for single rows and uploaded CSV files."""

from __future__ import annotations

import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN


class DataValidationError(ValueError):
    """Raised when transaction data cannot be used for prediction."""


def _is_empty_upload(df: pd.DataFrame) -> bool:
    return df is None or df.empty or df.shape[1] == 0


def validate_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned feature frame with only model inputs.

    Class is never used as an input. If present, it is dropped.
    """
    if _is_empty_upload(df):
        raise DataValidationError("The uploaded file is empty or has no columns.")

    working = df.copy()
    working.columns = [str(col).strip() for col in working.columns]

    if working.empty:
        raise DataValidationError("The uploaded file contains no rows.")

    missing = [col for col in FEATURE_COLUMNS if col not in working.columns]
    if missing:
        raise DataValidationError(
            "Missing required columns: " + ", ".join(missing) + ". "
            "The model expects Time, V1–V28, and Amount."
        )

    dropped_target = TARGET_COLUMN in working.columns
    features = working[FEATURE_COLUMNS].copy()

    if bool(features.isna().to_numpy().all()):
        raise DataValidationError("Every value in the feature columns is missing.")

    numeric = features.apply(pd.to_numeric, errors="coerce")
    invalid_mask = numeric.isna() & features.notna()
    if bool(invalid_mask.to_numpy().any()):
        bad_cols = [col for col in FEATURE_COLUMNS if invalid_mask[col].any()]
        raise DataValidationError(
            "Non-numeric values found in: " + ", ".join(bad_cols) + "."
        )

    if bool(numeric.isna().to_numpy().any()):
        missing_counts = numeric.isnull().sum()
        missing_cols = [col for col, count in missing_counts.items() if count]
        raise DataValidationError(
            "Missing values found in: " + ", ".join(missing_cols) + "."
        )

    numeric.attrs["dropped_class_column"] = dropped_target
    numeric.attrs["extra_columns"] = [
        col for col in working.columns if col not in FEATURE_COLUMNS and col != TARGET_COLUMN
    ]
    return numeric


def extract_binary_class_labels(series: pd.Series) -> pd.Series | None:
    """Return 0/1 labels if every value is a valid Class entry, otherwise None."""
    numeric = pd.to_numeric(series, errors="coerce")
    if bool(numeric.isna().to_numpy().any()):
        return None
    as_int = numeric.astype(int)
    if set(as_int.unique()) - {0, 1}:
        return None
    return as_int


def sample_template_csv(include_class_header: bool = False) -> str:
    """Return a header-only CSV. This is a format example, not sample transactions."""
    columns = list(FEATURE_COLUMNS)
    if include_class_header:
        columns.append(TARGET_COLUMN)
    return ",".join(columns) + "\n"


def validate_single_transaction(values: dict) -> pd.DataFrame:
    """Validate a mapping of feature name -> value."""
    missing = [col for col in FEATURE_COLUMNS if col not in values]
    if missing:
        raise DataValidationError("Missing fields: " + ", ".join(missing) + ".")

    row = {col: values[col] for col in FEATURE_COLUMNS}
    return validate_prediction_frame(pd.DataFrame([row]))
