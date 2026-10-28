"""Threshold selection and hold-out evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Choose a classification threshold using validation predictions only.

    The threshold that maximises F1-score is selected because accuracy is
    misleading when fraudulent transactions are rare.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float).ravel()

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return {"threshold": 0.5, "validation_f1": 0.0}

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = int(np.nanargmax(f1_scores))
    best_threshold = float(thresholds[best_index])
    best_threshold = min(max(best_threshold, 1e-6), 1.0 - 1e-6)

    predicted = (y_prob >= best_threshold).astype(int)
    return {
        "threshold": best_threshold,
        "validation_f1": float(f1_score(y_true, predicted, zero_division=0)),
        "validation_precision": float(precision_score(y_true, predicted, zero_division=0)),
        "validation_recall": float(recall_score(y_true, predicted, zero_division=0)),
    }


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float((tn + tp) / max(tn + fp + fn + tp, 1)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "support": {
            "legitimate": int((y_true == 0).sum()),
            "fraud": int((y_true == 1).sum()),
        },
    }


def curve_payload(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    # Downsample long curves so the Streamlit app can load them quickly.
    roc = _downsample_curve(fpr, tpr, 400)
    pr = _downsample_curve(recall, precision, 400)
    return {
        "roc": {"fpr": roc[0], "tpr": roc[1]},
        "pr": {"recall": pr[0], "precision": pr[1]},
    }


def _downsample_curve(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[list, list]:
    if len(x) <= max_points:
        return x.astype(float).tolist(), y.astype(float).tolist()
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return x[idx].astype(float).tolist(), y[idx].astype(float).tolist()
