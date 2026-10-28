import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.validation import (
    DataValidationError,
    extract_binary_class_labels,
    sample_template_csv,
    validate_prediction_frame,
    validate_single_transaction,
)


def _valid_row(**overrides):
    row = {col: 0.0 for col in FEATURE_COLUMNS}
    row.update(overrides)
    return row


def test_valid_frame_keeps_feature_order():
    frame = pd.DataFrame([_valid_row(Amount=12.5, Time=10)])
    result = validate_prediction_frame(frame)
    assert list(result.columns) == FEATURE_COLUMNS
    assert result.iloc[0]["Amount"] == 12.5


def test_class_column_is_dropped_and_never_required():
    row = _valid_row()
    row["Class"] = 1
    result = validate_prediction_frame(pd.DataFrame([row]))
    assert "Class" not in result.columns
    assert result.attrs["dropped_class_column"] is True


def test_missing_columns_raise():
    frame = pd.DataFrame([{"Time": 1, "Amount": 2}])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_prediction_frame(frame)


def test_empty_file_raises():
    with pytest.raises(DataValidationError, match="empty"):
        validate_prediction_frame(pd.DataFrame())


def test_non_numeric_values_raise():
    frame = pd.DataFrame([_valid_row(Amount="not-a-number")])
    with pytest.raises(DataValidationError, match="Non-numeric"):
        validate_prediction_frame(frame)


def test_missing_values_raise():
    frame = pd.DataFrame([_valid_row(V1=None)])
    with pytest.raises(DataValidationError, match="Missing values"):
        validate_prediction_frame(frame)


def test_single_transaction_validation():
    result = validate_single_transaction(_valid_row(Amount=9.99))
    assert len(result) == 1
    assert result.iloc[0]["Amount"] == 9.99


def test_extract_binary_class_labels_accepts_zero_and_one():
    labels = extract_binary_class_labels(pd.Series([0, 1, 0, 1.0]))
    assert labels is not None
    assert list(labels) == [0, 1, 0, 1]


def test_extract_binary_class_labels_rejects_invalid_values():
    assert extract_binary_class_labels(pd.Series([0, 2])) is None
    assert extract_binary_class_labels(pd.Series([0, "fraud"])) is None


def test_sample_template_csv_has_required_headers_only():
    text = sample_template_csv()
    assert text.endswith("\n")
    headers = text.strip().split(",")
    assert headers == FEATURE_COLUMNS
    assert "Class" not in headers
