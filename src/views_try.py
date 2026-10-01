"""Page 5: try-it-yourself scoring with sliders."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src import ui
from src import visuals as V
from src.views_data import _style

AGE_RANGE = np.arange(16, 39)


def season_median(paid: pd.DataFrame, season: int) -> float:
    """Median paid fee in one season: the gauge's reference point.

    A descriptive statistic of the market, not a model output, so it is read
    from every paid transfer in that season rather than from the test block.
    """
    fees = paid.loc[(paid["season"] == season) & (paid["fee_eur"] > 0), "fee_eur"]
    return float(fees.median()) if len(fees) else 0.0


def _profile_inputs() -> dict:
    d = ui.DEFAULT_PROFILE
    c1, c2, c3 = st.columns(3)
    with c1:
        position = st.selectbox("Position", list(C.POSITIONS), index=list(C.POSITIONS).index(d["position"]))
        league = st.selectbox("Destination league", list(C.LEAGUES), index=list(C.LEAGUES).index(d["league"]))
        age = st.slider("Age", 16, 38, int(d["age"]))
    with c2:
        goals = st.slider("Goals per 90 (last season)", 0.0, 0.70, float(d["goals_p90"]), 0.01)
        xg = st.slider("Expected goals per 90", 0.0, 0.70, float(d["xg_p90"]), 0.01)
        minutes = st.slider("Minutes last season", 200, 3960, int(d["minutes"]), 20)
    with c3:
        contract = st.slider("Contract years left", 0.0, 5.0, float(d["contract_years_left"]), 0.25)
        market_m = st.slider("Club market value (EUR millions)", 0.1, 200.0, d["market_value_eur"] / 1e6, 0.5)
        season = st.slider("Transfer season", int(C.FIRST_SEASON), int(C.LAST_SEASON), int(d["season"]))
    return {
        "age": age,
        "position": position,
        "league": league,
        "goals_p90": goals,
        "xg_p90": xg,
        "minutes": minutes,
        "contract_years_left": contract,
        "market_value_eur": market_m * 1e6,
        "season": season,
    }


def try_it(pipeline: ui.Pipeline) -> None:
    st.header("Try it yourself")
    st.write(
        "Build a player profile and see what both models would charge. The "
        "slider values describe a transfer **before** it happens, which is the "
        "only information the models are allowed to use."
    )

    profile = _profile_inputs()

    st.caption(
        "Assists per 90 are not a control: they are inferred from the chosen "
        "position's assists-per-goal ratio in the training seasons, so the "
        "comparison stays apples-to-apples across profiles."
    )

    preds = ui.predict_fee(pipeline, profile)
    pitch_col, gauge_col = st.columns([1, 1])

    with pitch_col:
        st.markdown(
            V.position_pitch_svg(profile["position"]), unsafe_allow_html=True
        )
        st.caption(
            f"**{profile['position']}** on a half-pitch. Striker, attacking "
            "midfield and wide men sit highest; the keeper owns the six-yard box."
        )

    with gauge_col:
        median = season_median(pipeline.paid, int(profile["season"]))
        st.plotly_chart(
            V.fee_gauge(
                preds["lightgbm"], median,
                title=f"LightGBM fee vs {int(profile['season'])} median",
            ),
            width="stretch",
        )
        st.caption(
            f"The season median is {ui.eur_m(median)}; the gauge shows how far "
            "above or below it this profile is priced."
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Linear regression", ui.eur(preds["linear"]))
    c2.metric("LightGBM", ui.eur(preds["lightgbm"]))
    c3.metric("Club market value (input)", ui.eur(preds["market_value"]))

    gap = (preds["lightgbm"] / preds["market_value"] - 1) * 100
    st.caption(
        f"LightGBM prices this player {abs(gap):.0f}% "
        f"{'above' if gap >= 0 else 'below'} the club's own valuation - "
        "typically a sign that the valuation is out of line with the rest of "
        "the profile."
    )

    curve = ui.age_curve(pipeline, profile, AGE_RANGE)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=AGE_RANGE,
            y=curve["lightgbm"],
            name="LightGBM",
            line=dict(color=ui.PALETTE["accent"], width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=AGE_RANGE,
            y=curve["linear"],
            name="Linear regression",
            line=dict(color=ui.PALETTE["warn"], width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[AGE_RANGE[np.argmax(curve["lightgbm"])]],
            y=[float(np.max(curve["lightgbm"]))],
            mode="markers",
            name="Peak",
            marker=dict(color=ui.PALETTE["accent"], size=11, symbol="star"),
        )
    )
    fig = _style(
        fig,
        "Predicted fee by age for this profile",
        "Age",
        "Predicted fee (EUR)",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Everything except age is held fixed. Value peaks in the mid-twenties "
        "and decays either side of it, matching the age curve baked into the "
        "synthetic market."
    )

    table = pd.DataFrame(
        {"age": AGE_RANGE, "linear_eur": curve["linear"], "lightgbm_eur": curve["lightgbm"]}
    )
    for col in ("linear_eur", "lightgbm_eur"):
        table[col] = table[col].map(lambda v: f"{v:,.0f}")
    with st.expander("Predicted fee by age (table)"):
        st.dataframe(table, width="stretch", hide_index=True)