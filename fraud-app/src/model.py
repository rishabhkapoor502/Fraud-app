"""Feed-forward ANN architecture for binary fraud classification."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from src.config import DEFAULT_LEARNING_RATE, FEATURE_COLUMNS


def build_fraud_ann(
    input_dim: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> keras.Model:
    n_features = input_dim or len(FEATURE_COLUMNS)

    model = keras.Sequential(
        [
            layers.Input(shape=(n_features,)),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.30),
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.20),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.10),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="credit_card_fraud_ann",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
