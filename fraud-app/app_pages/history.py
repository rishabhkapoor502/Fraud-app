"""Prediction history page."""

from __future__ import annotations

import streamlit as st

from src.database import clear_predictions, fetch_predictions, prediction_stats
from src.ui import page_header, render_disclaimer


def render() -> None:
    page_header(
        "Transaction history",
        "Predictions from the single-transaction and batch tools are stored locally in SQLite.",
    )
    render_disclaimer()

    if st.session_state.pop("history_cleared", False):
        st.success("History cleared.")

    stats = prediction_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Stored rows", stats["total"])
    c2.metric("Flagged", stats["flagged"])
    c3.metric("Latest entry", stats["latest"] or "None")

    rows = fetch_predictions(limit=500)
    if rows.empty:
        st.info("No predictions have been stored yet.")
        return

    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.download_button(
        "Download history CSV",
        data=rows.to_csv(index=False).encode("utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv",
    )

    with st.expander("Clear local history"):
        st.caption("This deletes rows from the local SQLite file only.")
        if st.button("Delete all saved predictions"):
            clear_predictions()
            st.session_state["history_cleared"] = True
            st.rerun()
