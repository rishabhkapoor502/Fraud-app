"""Single transaction scoring page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import DATASET_PATH, FEATURE_COLUMNS, PCA_FEATURES, TARGET_COLUMN
from src.database import log_prediction
from src.prediction import predict_single
from src.ui import add_session_predictions, page_header, render_disclaimer, require_model, show_prediction_result
from src.validation import DataValidationError


@st.cache_data(show_spinner="Loading dataset...")
def _load_dataset(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [col for col in FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError("Dataset is missing: " + ", ".join(missing))
    keep = list(FEATURE_COLUMNS)
    if TARGET_COLUMN in frame.columns:
        keep.append(TARGET_COLUMN)
    return frame[keep]


def _readable_summary(row: pd.Series) -> pd.DataFrame:
    items = [
        ("Time", row["Time"]),
        ("Amount", row["Amount"]),
    ]
    if TARGET_COLUMN in row.index and pd.notna(row[TARGET_COLUMN]):
        actual = int(row[TARGET_COLUMN])
        items.append(
            (
                "Actual label in dataset (not used for prediction)",
                "Fraud (1)" if actual == 1 else "Genuine (0)",
            )
        )
    return pd.DataFrame(items, columns=["Field", "Value"])


def _row_to_features(row: pd.Series) -> dict:
    return {col: float(row[col]) for col in FEATURE_COLUMNS}


def _run_prediction(values: dict) -> None:
    try:
        result = predict_single(values)
    except DataValidationError as exc:
        st.error(str(exc))
        return
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    log_prediction(
        source="single",
        predicted_class=result.predicted_class,
        prediction_label=result.label,
        probability=result.probability,
        risk_score=result.risk_score,
        threshold=result.threshold,
        amount=result.amount,
        time_value=result.time_value,
        features=result.features,
    )
    add_session_predictions(
        genuine=0 if result.predicted_class == 1 else 1,
        flagged=1 if result.predicted_class == 1 else 0,
    )
    show_prediction_result(result.label, result.probability, result.threshold)


def render() -> None:
    page_header(
        "Single Transaction Prediction",
        "Choose an existing transaction and run the trained model on it.",
    )
    artifacts = require_model()
    if artifacts is None:
        return
    render_disclaimer()

    st.info(
        "V1 to V28 are anonymized numerical features from the original credit card transaction dataset. "
        "They are not ordinary details like card number or cardholder name."
    )

    source = st.radio(
        "Where should the transaction come from?",
        [
            "Project dataset (data/creditcard.csv)",
            "Upload a CSV and pick a row",
        ],
        index=0 if DATASET_PATH.exists() else 1,
    )

    selected_row = None

    if source.startswith("Project dataset"):
        if not DATASET_PATH.exists():
            st.warning(
                "The project dataset was not found. Save the ULB file as `data/creditcard.csv`, "
                "or upload a CSV below."
            )
        else:
            try:
                dataset = _load_dataset(str(DATASET_PATH))
            except ValueError as exc:
                st.error(str(exc))
                dataset = None
            if dataset is not None:
                st.caption(f"Loaded {len(dataset):,} transactions from the project dataset.")
                row_number = st.number_input(
                    "Row number to predict",
                    min_value=0,
                    max_value=max(len(dataset) - 1, 0),
                    value=0,
                    step=1,
                )
                selected_row = dataset.iloc[int(row_number)]
    else:
        uploaded = st.file_uploader("Upload a CSV with Time, V1–V28, and Amount", type=["csv"])
        uploaded_frame = None
        if uploaded is not None:
            if uploaded.size == 0:
                st.error("The uploaded file is empty.")
            else:
                try:
                    uploaded_frame = pd.read_csv(uploaded)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"The file could not be read as CSV. Details: {exc}")
        elif "last_uploaded_frame" in st.session_state:
            uploaded_frame = st.session_state["last_uploaded_frame"]
            st.caption("Using the CSV last uploaded on the Batch Analysis page.")

        if uploaded_frame is not None:
            missing = [col for col in FEATURE_COLUMNS if col not in uploaded_frame.columns]
            if missing:
                st.error(
                    "This CSV cannot be used yet. Missing columns: "
                    + ", ".join(missing)
                    + ". The model needs Time, V1–V28, and Amount."
                )
            elif uploaded_frame.empty:
                st.error("The uploaded file has no transaction rows.")
            else:
                row_number = st.number_input(
                    "Row number to predict",
                    min_value=0,
                    max_value=max(len(uploaded_frame) - 1, 0),
                    value=0,
                    step=1,
                    key="upload_row_number",
                )
                selected_row = uploaded_frame.iloc[int(row_number)]

    if selected_row is not None:
        st.subheader("Selected transaction")
        st.table(_readable_summary(selected_row))
        with st.expander("Show V1 to V28 for this row"):
            st.dataframe(
                pd.DataFrame(
                    [{"Feature": name, "Value": selected_row[name]} for name in PCA_FEATURES]
                ),
                hide_index=True,
                use_container_width=True,
            )
        if st.button("Predict Transaction"):
            _run_prediction(_row_to_features(selected_row))
    else:
        st.info(
            "Select a transaction from the project dataset or upload a CSV that contains "
            "Time, V1–V28, and Amount. The app will not invent a transaction for you."
        )

    with st.expander("Advanced: Enter Feature Values Manually"):
        st.write(
            "Use this only if you already have valid numerical values from a real transaction row. "
            "Do not type random numbers. Random values are not a realistic credit card transaction."
        )
        with st.form("manual_feature_form"):
            c1, c2 = st.columns(2)
            with c1:
                time_value = st.number_input("Time", value=None, placeholder="from dataset", format="%.6f")
            with c2:
                amount = st.number_input("Amount", value=None, placeholder="from dataset", format="%.6f")
            pca_values = {}
            columns = st.columns(4)
            for index, name in enumerate(PCA_FEATURES):
                with columns[index % 4]:
                    pca_values[name] = st.number_input(
                        name,
                        value=None,
                        placeholder="numeric",
                        format="%.6f",
                        key=f"manual_{name}",
                    )
            submitted = st.form_submit_button("Predict manually entered values")

        if submitted:
            payload = {"Time": time_value, "Amount": amount, **pca_values}
            if any(value is None for value in payload.values()):
                st.error("Please fill in Time, Amount, and all V1–V28 values from a valid transaction row.")
            else:
                _run_prediction(payload)
