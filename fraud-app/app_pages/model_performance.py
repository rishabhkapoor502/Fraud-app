"""Hold-out evaluation display."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import CURVES_PATH
from src.ui import page_header, require_model

ARCHITECTURE_HTML = """
<div class="ann-stack" role="img" aria-label="Feed-forward ANN layers from 30 inputs to one sigmoid output">
  <div class="ann-layer ann-input">
    <div class="ann-title">Input · 30 features</div>
    <div class="ann-meta">Time, V1–V28, Amount</div>
  </div>
  <div class="ann-arrow">↓</div>
  <div class="ann-layer ann-dense">
    <div class="ann-title">Dense 64 · ReLU</div>
    <div class="ann-meta">1,984 parameters</div>
  </div>
  <div class="ann-arrow">↓</div>
  <div class="ann-layer ann-aux">
    <div class="ann-title">BatchNorm 64 · Dropout 0.30</div>
  </div>
  <div class="ann-arrow">↓</div>
  <div class="ann-layer ann-dense">
    <div class="ann-title">Dense 32 · ReLU</div>
    <div class="ann-meta">2,080 parameters</div>
  </div>
  <div class="ann-arrow">↓</div>
  <div class="ann-layer ann-aux">
    <div class="ann-title">BatchNorm 32 · Dropout 0.20</div>
  </div>
  <div class="ann-arrow">↓</div>
  <div class="ann-layer ann-dense">
    <div class="ann-title">Dense 16 · ReLU</div>
    <div class="ann-meta">528 parameters</div>
  </div>
  <div class="ann-arrow">↓</div>
  <div class="ann-layer ann-aux">
    <div class="ann-title">Dropout 0.10</div>
  </div>
  <div class="ann-arrow">↓</div>
  <div class="ann-layer ann-out">
    <div class="ann-title">Dense 1 · Sigmoid</div>
    <div class="ann-meta">Fraud probability · 17 parameters</div>
  </div>
</div>
"""


def _curve_figure(x: list, y: list, title: str, x_title: str, y_title: str, diagonal: bool = False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line={"color": "#1f4e79", "width": 2}, name="Model"))
    if diagonal:
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line={"color": "#8a95a1", "dash": "dash"}, name="Random guess",
            )
        )
    fig.update_layout(
        template="simple_white",
        title=title,
        title_font_size=14,
        xaxis_title=x_title,
        yaxis_title=y_title,
        margin={"t": 40, "b": 40, "l": 40, "r": 20},
        height=340,
        showlegend=diagonal,
    )
    return fig


def _render_curves() -> None:
    if not CURVES_PATH.exists():
        return
    try:
        curves = json.loads(CURVES_PATH.read_text())
        roc, pr = curves["roc"], curves["pr"]
    except (json.JSONDecodeError, KeyError, OSError):
        return
    with st.expander("ROC and Precision-Recall curves (test set)"):
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            _curve_figure(roc["fpr"], roc["tpr"], "ROC curve", "False positive rate", "True positive rate", True),
            use_container_width=True,
        )
        c2.plotly_chart(
            _curve_figure(pr["recall"], pr["precision"], "Precision-Recall curve", "Recall", "Precision"),
            use_container_width=True,
        )


def render() -> None:
    page_header(
        "Model Performance",
        "Trained network architecture next to held-out test results.",
    )
    artifacts = require_model()
    if artifacts is None:
        return

    metrics = artifacts.get("test_metrics", {})
    if not metrics:
        st.error("No saved evaluation results were found. Train the model first with `python train_model.py`.")
        return

    split = artifacts.get("split_summary", {})
    test_split = split.get("test", {})
    config = artifacts.get("training_config", {})
    matrix = metrics["confusion_matrix"]

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Architecture")
        st.caption("credit_card_fraud_ann · 4,801 trainable parameters")
        st.markdown(ARCHITECTURE_HTML, unsafe_allow_html=True)

    with right:
        st.subheader("Trained model")
        st.write(
            f"Adam optimizer, learning rate **{config.get('learning_rate', 0.001)}**, "
            "binary cross-entropy loss. "
            f"Ran **{config.get('epochs_run', '?')}** of {config.get('epochs_requested', '?')} epochs "
            f"with batch size {config.get('batch_size', '?')}."
        )
        st.metric("Decision threshold", f"{artifacts.get('threshold', 0):.4f}")

        if test_split:
            st.caption(
                f"Test set: {test_split.get('rows', 'unknown')} transactions "
                f"({test_split.get('legitimate', '?')} genuine, {test_split.get('fraud', '?')} fraud). "
                "Scores were calculated once after the threshold was chosen on validation data."
            )

        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        m2.metric("Precision", f"{metrics['precision']:.4f}")
        m3, m4 = st.columns(2)
        m3.metric("Recall", f"{metrics['recall']:.4f}")
        m4.metric("F1-score", f"{metrics['f1_score']:.4f}")
        m5, m6 = st.columns(2)
        m5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        m6.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")

        st.subheader("Confusion matrix")
        st.table(
            pd.DataFrame(
                {
                    "Predicted genuine": [matrix["true_negative"], matrix["false_negative"]],
                    "Predicted fraud": [matrix["false_positive"], matrix["true_positive"]],
                },
                index=["Actual genuine", "Actual fraud"],
            )
        )
        st.caption(
            "True negative = correctly genuine. False positive = genuine flagged as fraud. "
            "False negative = fraud missed. True positive = fraud found."
        )

    st.subheader("What these numbers mean")
    st.markdown(
        """
        - **Accuracy**: share of all test transactions that were classified correctly. This can look high even if fraud is missed, because most transactions are genuine.
        - **Precision**: of the transactions flagged as fraud, how many really were fraud.
        - **Recall**: of the actual fraud cases, how many the model found.
        - **F1-score**: a balance of precision and recall.
        """
    )

    _render_curves()
