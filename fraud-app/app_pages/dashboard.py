"""Dashboard page."""

from __future__ import annotations

import streamlit as st

from src.config import DATASET_PATH, TARGET_COLUMN
from src.preprocessing import DatasetNotFoundError, clean_dataset, load_raw_dataset
from src.ui import model_status, page_header, session_prediction_counts, simple_bar_chart


@st.cache_data(show_spinner=False)
def _dataset_label_counts() -> dict | None:
    if not DATASET_PATH.exists():
        return None
    try:
        raw = load_raw_dataset()
        cleaned = clean_dataset(raw)
    except (DatasetNotFoundError, ValueError, OSError):
        return None
    return {
        "rows": int(len(cleaned)),
        "genuine": int((cleaned[TARGET_COLUMN] == 0).sum()),
        "fraud": int((cleaned[TARGET_COLUMN] == 1).sum()),
    }


def render() -> None:
    page_header(
        "Dashboard",
        "Overview of the project dataset labels and any predictions made in this session.",
    )

    status = model_status()
    if status["ready"]:
        st.success("A trained model is loaded. You can run predictions from the other pages.")
    else:
        st.warning("No trained model found yet. Train it with `python train_model.py` before predicting.")

    st.subheader("Actual dataset labels")
    st.caption(
        "These numbers come from the Class column in `data/creditcard.csv`. "
        "They are known labels, not model predictions."
    )
    overview = _dataset_label_counts()
    if overview is None:
        st.info(
            "No dataset is loaded. Download the ULB Credit Card Fraud Detection CSV "
            "and save it as `data/creditcard.csv`."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total labelled transactions", f"{overview['rows']:,}")
        c2.metric("Actual genuine (Class = 0)", f"{overview['genuine']:,}")
        c3.metric("Actual fraud (Class = 1)", f"{overview['fraud']:,}")
        fraud_share = overview["fraud"] / overview["rows"] * 100 if overview["rows"] else 0
        st.caption(
            f"Fraud is only {fraud_share:.3f}% of these transactions. This heavy imbalance is why the "
            "project judges the model with precision, recall, and PR-AUC rather than accuracy alone."
        )
        st.plotly_chart(
            simple_bar_chart(
                ["Actual genuine", "Actual fraud"],
                [overview["genuine"], overview["fraud"]],
                "Actual labels in the dataset",
            ),
            use_container_width=True,
        )

    st.subheader("Model predictions in this session")
    st.caption(
        "These numbers are created only after you analyse transactions on the "
        "Batch Analysis or Single Transaction Prediction pages."
    )
    counts = session_prediction_counts()
    total = counts["genuine"] + counts["flagged"]
    if total == 0:
        st.info("No transactions have been analysed in this session yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total transactions analysed", f"{total:,}")
        c2.metric("Predicted as genuine", f"{counts['genuine']:,}")
        c3.metric("Flagged as potentially fraudulent", f"{counts['flagged']:,}")
        st.plotly_chart(
            simple_bar_chart(
                ["Predicted genuine", "Potentially fraudulent"],
                [counts["genuine"], counts["flagged"]],
                "Model predictions in this session",
            ),
            use_container_width=True,
        )
