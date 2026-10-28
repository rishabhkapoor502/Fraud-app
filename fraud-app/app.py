#!/usr/bin/env python3
"""Streamlit web application for credit card fraud detection."""

from __future__ import annotations

import streamlit as st

from app_pages import about, batch_analysis, dashboard, history, model_performance, single_prediction
from src.database import initialize_database
from src.ui import inject_css, model_status

PAGES = {
    "Dashboard": dashboard.render,
    "Batch Analysis": batch_analysis.render,
    "Single Transaction Prediction": single_prediction.render,
    "Model Performance": model_performance.render,
    "Transaction History": history.render,
    "About Project": about.render,
}


def main() -> None:
    st.set_page_config(
        page_title="Credit Card Fraud Detection",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    initialize_database()

    status = model_status()
    with st.sidebar:
        st.header("Credit Card Fraud Detection")
        st.caption("College project using a deep learning model")
        st.divider()
        page = st.radio("Go to", list(PAGES.keys()), index=0)
        st.divider()
        st.markdown(
            ("✅ **Model:** ready" if status["ready"] else "⚠️ **Model:** not trained yet")
        )
        st.markdown(
            ("✅ **Dataset:** found in data/" if status["dataset_present"] else "➖ **Dataset:** not found")
        )
        st.caption("The app loads a saved model. It does not retrain when it starts.")

    PAGES[page]()


if __name__ == "__main__":
    main()
