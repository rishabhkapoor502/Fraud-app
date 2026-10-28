"""Shared Streamlit helpers and simple college-project styling."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import ARTIFACTS_PATH, CURVES_PATH, DATASET_PATH, HISTORY_PATH, MODEL_PATH
from src.prediction import ModelNotTrainedError, load_artifacts, model_is_ready

DISCLAIMER = (
    "These results are model estimates for a college project. "
    "They are not confirmed fraud and should not be used to accuse a person or block a real payment."
)

CSS = """
<style>
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    [data-testid="stSidebar"] {
        background-color: #f3f5f7;
        border-right: 1px solid #d9dee5;
    }
    h1 { font-size: 1.6rem !important; font-weight: 650 !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.05rem !important; }
    .stButton>button, .stDownloadButton>button {
        border-radius: 4px;
        border: 1px solid #1f4e79;
        background-color: #1f4e79;
        color: #ffffff;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #163a5c;
        border-color: #163a5c;
        color: #ffffff;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d9dee5;
        padding: 0.7rem 0.85rem;
        border-radius: 4px;
    }
    .result-box {
        border: 1px solid #d9dee5;
        padding: 0.9rem 1rem;
        margin: 0.6rem 0 1rem 0;
        background: #f8fafc;
    }
    .result-genuine { border-left: 4px solid #2f6f4e; }
    .result-fraud { border-left: 4px solid #a33b3b; }
    .ann-stack { display: flex; flex-direction: column; gap: 0.35rem; }
    .ann-layer {
        border: 1px solid #d9dee5;
        background: #f8fafc;
        padding: 0.55rem 0.75rem;
        border-radius: 4px;
    }
    .ann-layer.ann-dense { border-left: 4px solid #1f4e79; background: #eef3f8; }
    .ann-layer.ann-input { border-left: 4px solid #5a6a7a; }
    .ann-layer.ann-out { border-left: 4px solid #2f6f4e; background: #eef6f1; }
    .ann-layer.ann-aux { color: #4a5560; }
    .ann-title { font-weight: 650; font-size: 0.92rem; }
    .ann-meta { font-size: 0.8rem; color: #4a5560; margin-top: 0.1rem; }
    .ann-arrow { text-align: center; color: #8a95a1; font-size: 0.75rem; line-height: 1; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def render_disclaimer() -> None:
    st.info(DISCLAIMER)


def model_status() -> dict:
    ready = model_is_ready()
    dataset_present = DATASET_PATH.exists()
    artifacts = None
    error = None
    if ready:
        try:
            artifacts = load_artifacts()
        except (ModelNotTrainedError, json.JSONDecodeError, OSError) as exc:
            error = str(exc)
            ready = False
    return {
        "ready": ready,
        "dataset_present": dataset_present,
        "artifacts": artifacts,
        "error": error,
        "model_path": str(MODEL_PATH),
        "curves_present": CURVES_PATH.exists(),
        "history_present": HISTORY_PATH.exists(),
    }


def require_model() -> dict | None:
    status = model_status()
    if status["ready"]:
        return status["artifacts"]
    st.error(
        "The trained model was not found. Put `creditcard.csv` in the `data` folder "
        "and run `python train_model.py` once. This app does not train the model when it starts."
    )
    if not status["dataset_present"]:
        st.warning(
            "The dataset file is also missing. Download the ULB Credit Card Fraud Detection CSV "
            "from Kaggle and save it as `data/creditcard.csv`."
        )
    if status["error"]:
        st.caption(status["error"])
    return None


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def simple_bar_chart(labels: list[str], values: list[int], title: str):
    frame = pd.DataFrame({"Category": labels, "Count": values})
    fig = px.bar(frame, x="Category", y="Count", title=title, color_discrete_sequence=["#1f4e79"])
    fig.update_layout(
        template="simple_white",
        title_font_size=14,
        margin={"t": 40, "b": 40, "l": 40, "r": 20},
        yaxis_title="Count",
        xaxis_title="",
    )
    return fig


def session_prediction_counts() -> dict[str, int]:
    return st.session_state.setdefault(
        "session_predictions",
        {"genuine": 0, "flagged": 0},
    )


def add_session_predictions(genuine: int, flagged: int) -> None:
    counts = session_prediction_counts()
    counts["genuine"] += int(genuine)
    counts["flagged"] += int(flagged)


def show_prediction_result(label: str, probability: float | None, threshold: float | None) -> None:
    is_fraud = "fraud" in label.lower()
    css_class = "result-fraud" if is_fraud else "result-genuine"
    icon = "⚠️" if is_fraud else "✅"
    st.markdown(
        f'<div class="result-box {css_class}"><strong>Result:</strong> {icon} {label}</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    if probability is not None:
        cols[0].metric("Estimated fraud probability", f"{probability:.4f}")
    if threshold is not None:
        cols[1].metric("Decision threshold", f"{threshold:.4f}")
    st.caption(
        "A transaction is marked potentially fraudulent when the probability is at or above the threshold."
    )
