"""Streamlit dashboard for the transfer-fee pipeline.

    .venv/bin/streamlit run app.py --server.headless true

The sidebar has two sections. The synthetic market reuses ``src/features``,
``src/train`` and ``src/evaluate`` unchanged. The real section is a separate
stack: football-data.org via ``requests`` for crests, winners and tables, a
public snapshot of real valuations, and scikit-learn fitted on it.
"""
from __future__ import annotations

import streamlit as st

from src import football_data as FD
from src import ui
from src import views_data as VD
from src import views_lookup as VL
from src import views_models as VM
from src import views_real as VR
from src import views_try as VT
from src import views_value as VV

st.set_page_config(
    page_title="Transfer fee predictor",
    page_icon="€",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYNTHETIC = {
    "Overview": VD.overview,
    "Model comparison": VM.model_comparison,
    "Predicted vs actual": VM.predicted_vs_actual,
    "Feature importance": VD.feature_importance,
    "Try it yourself": VT.try_it,
    "Worst errors": VD.worst_errors,
    "Player lookup": VL.player_lookup,
}

REAL = {
    "Trophy cabinet": VR.trophy_cabinet,
    "League table": VR.league_table_page,
    "Real player values": VR.player_list,
    "Value predictor": VV.value_predictor,
}


def main() -> None:
    ui.inject_css()
    st.sidebar.title("Transfer fees")
    st.sidebar.caption(f"Synthetic market, seed {ui.C.SEED}")

    section = st.sidebar.radio(
        "Section", ["Synthetic market", "Real players"], key="section"
    )

    if section == "Real players":
        page = st.sidebar.radio("Pages", list(REAL), key="real_page")
        with st.sidebar.expander("Where the data comes from"):
            st.caption("Valuations: public snapshot of 2,570 players, June 2024.")
            st.caption(
                "football-data.org: live via requests."
                if FD.available()
                else "football-data.org: no token in .env, so offline."
            )
        with st.spinner("Loading real player data and fitting models..."):
            data = _load_real()
        st.session_state["crest_map"] = _crest_map()
        REAL[page](data)
        st.caption(
            "Real valuations are a June 2024 snapshot: there is no time axis, so "
            "the honest test holds out whole clubs."
        )
        return

    page = st.sidebar.radio("Pages", list(SYNTHETIC), key="page")
    with st.sidebar.expander("Model in use"):
        st.caption("LightGBM (C) is used for charts unless noted.")
        st.caption("Scored on seasons 2023-2025 only.")

    with st.spinner("Generating data and fitting models..."):
        pipeline = ui.load_pipeline()

    SYNTHETIC[page](pipeline)
    st.caption(
        "Data generated locally and cached in memory. "
        f"Price index estimated at {pipeline.index:.4f}/yr from training seasons."
    )


@st.cache_resource(show_spinner=False)
def _load_real():
    from src import real_ui

    return real_ui.load_real()


@st.cache_resource(show_spinner=False)
def _crest_map():
    if not FD.available():
        return {}
    try:
        return FD.club_crests()
    except FD.ApiUnavailable:
        return {}


if __name__ == "__main__":
    main()
else:
    main()
