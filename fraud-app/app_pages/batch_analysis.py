"""Batch CSV scoring page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN
from src.database import log_prediction_frame
from src.evaluation import evaluate_predictions
from src.prediction import predict_frame
from src.ui import add_session_predictions, page_header, render_disclaimer, require_model
from src.validation import DataValidationError, extract_binary_class_labels, sample_template_csv


def _sample_download() -> tuple[bytes, str, str]:
    if DATASET_PATH.exists():
        frame = pd.read_csv(DATASET_PATH, nrows=8)
        cols = [col for col in FEATURE_COLUMNS if col in frame.columns]
        if TARGET_COLUMN in frame.columns:
            cols.append(TARGET_COLUMN)
        data = frame[cols].to_csv(index=False).encode("utf-8")
        return data, "sample_transactions.csv", "First 8 rows from the project dataset (real values)."
    template = sample_template_csv(include_class_header=True)
    return (
        template.encode("utf-8"),
        "required_columns_template.csv",
        "Header-only template. Fill it with real transaction rows; it is not example data.",
    )


def render() -> None:
    page_header(
        "Batch Analysis",
        "Upload a CSV of transactions and get a prediction for each row.",
    )
    artifacts = require_model()
    if artifacts is None:
        return
    render_disclaimer()

    st.subheader("Accepted CSV format")
    st.write(
        "The file must include these columns: **Time**, **V1 to V28**, and **Amount**. "
        "`Class` is optional. If it is present, it is not used as an input. "
        "It is only used afterwards to compare predictions with known labels."
    )
    st.code(", ".join(FEATURE_COLUMNS), language=None)

    sample_bytes, sample_name, sample_note = _sample_download()
    st.download_button(
        "Download sample CSV",
        data=sample_bytes,
        file_name=sample_name,
        mime="text/csv",
    )
    st.caption(sample_note)

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    if uploaded is None:
        return

    if uploaded.size == 0:
        st.error("The uploaded file is empty. Please choose a CSV that contains rows.")
        return

    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001 - show parse errors to the user
        st.error(f"The file could not be read as CSV. Details: {exc}")
        return

    if frame.empty:
        st.error("The uploaded file has column names but no transaction rows.")
        return

    batch_key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("batch_key") != batch_key:
        try:
            results = predict_frame(frame)
        except DataValidationError as exc:
            st.error(str(exc))
            return
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        st.session_state["batch_key"] = batch_key
        st.session_state["batch_results"] = results
        st.session_state["last_uploaded_frame"] = frame
        flagged = int((results["predicted_class"] == 1).sum())
        genuine = int(len(results) - flagged)
        add_session_predictions(genuine=genuine, flagged=flagged)
        if len(results) <= 300:
            log_prediction_frame(results, source="batch", max_rows=300)
    else:
        results = st.session_state["batch_results"]
        flagged = int((results["predicted_class"] == 1).sum())
        genuine = int(len(results) - flagged)

    st.subheader("Prediction summary")
    st.caption("These figures are model predictions, not confirmed fraud.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total transactions analysed", f"{len(results):,}")
    c2.metric("Predicted genuine", f"{genuine:,}")
    c3.metric("Predicted potentially fraudulent", f"{flagged:,}")

    display_cols = [
        "Time",
        "Amount",
        "predicted_class",
        "prediction_label",
        "fraud_probability",
    ]
    only_flagged = st.checkbox("Show only transactions flagged as potentially fraudulent")
    table = results[display_cols]
    if only_flagged:
        table = table[table["predicted_class"] == 1]
        if table.empty:
            st.info("No transactions were flagged in this file.")
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "predicted_class": st.column_config.NumberColumn("Predicted class", format="%d"),
            "prediction_label": "Prediction",
            "fraud_probability": st.column_config.NumberColumn("Fraud probability", format="%.4f"),
        },
    )

    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results CSV",
        data=csv_bytes,
        file_name="fraud_batch_results.csv",
        mime="text/csv",
    )

    if TARGET_COLUMN in frame.columns:
        labels = extract_binary_class_labels(frame[TARGET_COLUMN])
        if labels is None:
            st.warning(
                "A Class column was found, but it could not be used for comparison. "
                "Values must be 0 (genuine) or 1 (fraud). Predictions were still made without using Class."
            )
        elif len(labels) != len(results):
            st.warning("The Class column could not be aligned with the predictions, so no comparison is shown.")
        else:
            st.subheader("Comparison with actual labels")
            st.write(
                "This uploaded file included known Class labels, so the table below compares "
                "predictions with those labels. This is **not** the official test-set result "
                "shown on the Model Performance page."
            )
            metrics = evaluate_predictions(
                labels.to_numpy(),
                results["fraud_probability"].to_numpy(),
                float(results["threshold_used"].iloc[0]),
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            m2.metric("Precision", f"{metrics['precision']:.4f}")
            m3.metric("Recall", f"{metrics['recall']:.4f}")
            m4.metric("F1-score", f"{metrics['f1_score']:.4f}")
            matrix = metrics["confusion_matrix"]
            st.table(
                pd.DataFrame(
                    {
                        "Predicted genuine": [matrix["true_negative"], matrix["false_negative"]],
                        "Predicted fraud": [matrix["false_positive"], matrix["true_positive"]],
                    },
                    index=["Actual genuine", "Actual fraud"],
                )
            )
    else:
        st.caption("No Class column was found, so only predictions are shown. That is expected for unlabelled files.")
