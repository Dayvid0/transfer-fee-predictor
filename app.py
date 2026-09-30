"""Streamlit dashboard for the transfer-fee pipeline.

    .venv/bin/streamlit run app.py --server.headless true

The sidebar routes to one page per pipeline question. All numbers come from
src/ui.py, which reuses the training and scoring code unchanged.
"""
from __future__ import annotations

import streamlit as st

from src import ui
from src import views_data as VD
from src import views_models as VM
from src import views_try as VT

st.set_page_config(
    page_title="Transfer fee predictor",
    page_icon="€",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Overview": VD.overview,
    "Model comparison": VM.model_comparison,
    "Predicted vs actual": VM.predicted_vs_actual,
    "Feature importance": VD.feature_importance,
    "Try it yourself": VT.try_it,
    "Worst errors": VD.worst_errors,
}


def main() -> None:
    st.sidebar.title("Transfer fees")
    st.sidebar.caption(f"Synthetic market, seed {ui.C.SEED}")

    choice = st.sidebar.radio("Pages", list(PAGES), key="page")
    with st.sidebar.expander("Model in use"):
        st.caption("LightGBM (C) is used for charts unless noted.")
        st.caption("Scored on seasons 2023-2025 only.")

    with st.spinner("Generating data and fitting models..."):
        pipeline = ui.load_pipeline()

    PAGES[choice](pipeline)
    st.caption(
        f"Data generated locally and cached in memory. "
        f"Price index estimated at {pipeline.index:.4f}/yr from training seasons."
    )


if __name__ == "__main__":
    main()
else:
    main()