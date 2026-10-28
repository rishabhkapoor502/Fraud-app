"""Train the ANN, select a validation threshold, and persist artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.config import (
    ARTIFACTS_PATH,
    CURVES_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    FEATURE_COLUMNS,
    HISTORY_PATH,
    MODEL_PATH,
    REDUCE_LR_PATIENCE,
)
from src.evaluation import curve_payload, evaluate_predictions, select_threshold
from src.model import build_fraud_ann
from src.preprocessing import (
    class_weights_from_labels,
    clean_dataset,
    fit_scaler,
    load_raw_dataset,
    save_scaler,
    split_summary,
    stratified_splits,
    transform_features,
)


def _json_ready(history: dict) -> dict:
    return {key: [float(v) for v in values] for key, values in history.items()}


def train_pipeline(
    dataset_path: Path | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> dict:
    raw = load_raw_dataset(dataset_path)
    cleaned = clean_dataset(raw)
    splits = stratified_splits(cleaned)
    scaler = fit_scaler(splits["x_train"])

    x_train = transform_features(scaler, splits["x_train"])
    x_val = transform_features(scaler, splits["x_val"])
    x_test = transform_features(scaler, splits["x_test"])
    y_train = splits["y_train"].to_numpy().astype(int)
    y_val = splits["y_val"].to_numpy().astype(int)
    y_test = splits["y_test"].to_numpy().astype(int)

    model = build_fraud_ann(input_dim=x_train.shape[1], learning_rate=learning_rate)
    callbacks = [
        EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights_from_labels(splits["y_train"]),
        callbacks=callbacks,
        verbose=1,
    )

    val_prob = model.predict(x_val, batch_size=batch_size, verbose=0).ravel()
    threshold_info = select_threshold(y_val, val_prob)

    test_prob = model.predict(x_test, batch_size=batch_size, verbose=0).ravel()
    test_metrics = evaluate_predictions(y_test, test_prob, threshold_info["threshold"])
    validation_metrics = evaluate_predictions(y_val, val_prob, threshold_info["threshold"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    save_scaler(scaler)

    artifacts = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold_info["threshold"],
        "threshold_selection": threshold_info,
        "split_summary": split_summary(splits),
        "class_weights": class_weights_from_labels(splits["y_train"]),
        "training_config": {
            "epochs_requested": epochs,
            "epochs_run": len(history.history.get("loss", [])),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "architecture": "dense-64-32-16-1",
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "notes": [
            "Threshold was selected on the validation set only.",
            "Test metrics were computed once after threshold selection.",
            "Class was never used as an input feature.",
        ],
    }
    ARTIFACTS_PATH.write_text(json.dumps(artifacts, indent=2))
    CURVES_PATH.write_text(json.dumps(curve_payload(y_test, test_prob), indent=2))
    HISTORY_PATH.write_text(json.dumps(_json_ready(history.history), indent=2))

    return artifacts
