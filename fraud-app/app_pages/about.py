"""About page."""

from __future__ import annotations

import streamlit as st

from src.config import KAGGLE_DATASET_URL
from src.ui import page_header, render_disclaimer


def render() -> None:
    page_header(
        "About Project",
        "Credit Card Fraud Detection Using Deep Learning",
    )
    render_disclaimer()

    st.subheader("Purpose")
    st.write(
        "This college project trains a neural network to estimate whether a credit card "
        "transaction looks genuine or potentially fraudulent, then shows those estimates in a simple web app."
    )

    st.subheader("Dataset")
    st.write(
        "It uses the public ULB Credit Card Fraud Detection dataset. "
        "Each row has Time, Amount, anonymized features V1 to V28, and a Class label "
        "(0 = genuine, 1 = fraud). The Class label is used only for training and checking results."
    )
    st.markdown(f"[Dataset on Kaggle]({KAGGLE_DATASET_URL})")

    st.subheader("Deep learning approach")
    st.write(
        "A feed-forward artificial neural network is trained on scaled features. "
        "The data is split into training, validation, and test sets. "
        "A decision threshold is chosen on the validation set, and final scores are reported on the test set."
    )

    st.subheader("Technologies")
    st.write("Python, TensorFlow/Keras, pandas, scikit-learn, Streamlit, Plotly, SQLite, and pytest.")
