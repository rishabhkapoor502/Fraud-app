"""Load saved artifacts and score transactions without retraining."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_PATH, FEATURE_COLUMNS, MODEL_PATH
from src.preprocessing import load_scaler, transform_features
from src.validation import validate_prediction_frame, validate_single_transaction


class ModelNotTrainedError(FileNotFoundError):
    """Raised when the app is started before train_model.py has been run."""


@dataclass
class PredictionResult:
    probability: float
    predicted_class: int
    label: str
    threshold: float
    risk_score: float
    amount: float
    time_value: float
    features: dict


def model_is_ready() -> bool:
    return MODEL_PATH.exists() and ARTIFACTS_PATH.exists()


def load_artifacts() -> dict:
    if not ARTIFACTS_PATH.exists():
        raise ModelNotTrainedError(
            "Training artifacts were not found. Run python train_model.py after placing "
            "creditcard.csv in the data/ folder."
        )
    return json.loads(ARTIFACTS_PATH.read_text())


@lru_cache(maxsize=1)
def load_trained_model():
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            "Saved model was not found. Run python train_model.py before starting the app."
        )
    from tensorflow.keras.models import load_model

    return load_model(MODEL_PATH)


def clear_model_cache() -> None:
    load_trained_model.cache_clear()


def _label(predicted_class: int) -> str:
    if predicted_class == 1:
        return "Potentially Fraudulent"
    return "Likely Genuine"


def predict_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = validate_prediction_frame(df)
    artifacts = load_artifacts()
    threshold = float(artifacts["threshold"])
    scaler = load_scaler()
    model = load_trained_model()

    scaled = transform_features(scaler, features)
    probabilities = model.predict(scaled, verbose=0).ravel()
    predicted = (probabilities >= threshold).astype(int)

    output = features.copy()
    output["fraud_probability"] = probabilities
    output["risk_score"] = probabilities * 100.0
    output["predicted_class"] = predicted
    output["prediction_label"] = [
        _label(int(value)) for value in predicted
    ]
    output["threshold_used"] = threshold
    output.attrs["dropped_class_column"] = features.attrs.get("dropped_class_column", False)
    output.attrs["extra_columns"] = features.attrs.get("extra_columns", [])
    return output


def predict_single(values: dict) -> PredictionResult:
    frame = validate_single_transaction(values)
    scored = predict_frame(frame).iloc[0]
    return PredictionResult(
        probability=float(scored["fraud_probability"]),
        predicted_class=int(scored["predicted_class"]),
        label=str(scored["prediction_label"]),
        threshold=float(scored["threshold_used"]),
        risk_score=float(scored["risk_score"]),
        amount=float(scored["Amount"]),
        time_value=float(scored["Time"]),
        features={col: float(scored[col]) for col in FEATURE_COLUMNS},
    )


def predict_proba_array(x_scaled: np.ndarray) -> np.ndarray:
    model = load_trained_model()
    return model.predict(x_scaled, verbose=0).ravel()
