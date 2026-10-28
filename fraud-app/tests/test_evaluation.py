import numpy as np

from src.evaluation import evaluate_predictions, select_threshold


def test_threshold_uses_labels_and_is_between_zero_and_one():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 0.95])
    result = select_threshold(y_true, y_prob)
    assert 0 < result["threshold"] < 1
    assert result["validation_f1"] > 0.9


def test_evaluate_predictions_confusion_matrix_and_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)
    assert metrics["confusion_matrix"]["true_negative"] == 2
    assert metrics["confusion_matrix"]["true_positive"] == 2
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert 0 < metrics["roc_auc"] <= 1
    assert 0 < metrics["pr_auc"] <= 1
