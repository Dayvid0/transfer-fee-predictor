"""Pages: overview, feature importance, worst errors."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ui

CHART_TEMPLATE = "plotly_white"


def _style(fig: go.Figure, title: str, x: str, y: str) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        template=CHART_TEMPLATE,
        margin=dict(l=10, r=10, t=60, b=10),
        height=420,
    )
    return fig


def overview(pipeline: ui.Pipeline) -> None:
    st.header("Overview")
    st.write(
        "This project predicts what a football player will cost when he moves. "
        "It runs on a synthetic transfer market that the pipeline itself "
        "generates: 11 seasons of transfers across seven European leagues, with "
        "a hidden pricing formula behind every fee. Three models are then "
        "trained on the early seasons and scored on the later ones, so the "
        "scores reflect genuine forecasting rather than memorisation."
    )

    raw, paid = pipeline.raw, pipeline.paid
    removed = len(raw) - len(paid)
    cols = st.columns(4)
    cols[0].metric("Transfers", f"{len(raw):,}")
    cols[1].metric("Seasons", f"{int(raw.season.min())}-{int(raw.season.max())}")
    cols[2].metric("Zero-fee rows removed", f"{removed:,} ({removed / len(raw):.1%})")
    cols[3].metric("Median paid fee", ui.eur_m(paid.fee_eur.median()))

    st.caption(
        f"{removed:,} loans and free transfers (fee = EUR 0) carry no clearing "
        "price, so they are dropped before modelling and never scored."
    )

    per_season = (
        paid.groupby("season")
        .agg(median_fee=("fee_eur", "median"), transfers=("fee_eur", "size"))
        .reset_index()
    )
    fig = px.bar(
        per_season,
        x="season",
        y="median_fee",
        hover_data={"transfers": ":,", "median_fee": ":,.0f"},
        color_discrete_sequence=["#2E6F9E"],
    )
    fig.update_traces(
        hovertemplate="Season %{x}<br>Median fee EUR %{y:,.0f}<br>"
        "%{customdata[0]:,} transfers<extra></extra>",
        customdata=per_season[["transfers"]].to_numpy(),
    )
    _style(fig, "Median transfer fee by season", "Season", "Median fee (EUR)")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The synthetic market inflates about 10% a year, so the median fee "
        "climbs from roughly EUR 2.3M in 2015 to EUR 5.8M in 2025."
    )

    st.subheader("How the split works")
    split = pipeline.models.split.summary
    st.write(
        f"Fitting uses seasons {split['train_seasons']} "
        f"({split['train_rows']:,} rows), season {split['valid_season']} is held "
        f"back only for early stopping ({split['valid_rows']:,} rows), and "
        f"seasons {split['test_seasons']} ({split['test_rows']:,} rows) are "
        "scored exactly once. Nothing from the test years touches a fit, a "
        "scaler, or a tuning decision."
    )


def feature_importance(pipeline: ui.Pipeline) -> None:
    st.header("Feature importance")
    st.write(
        "LightGBM's gain share shows how much each pre-transfer input actually "
        "moved the model's predictions. Market value dominates, which is the "
        "honest headline: this is a market-value correction problem, not a "
        "discovery problem."
    )

    imp = pipeline.importance.iloc[::-1]
    fig = px.bar(
        imp,
        x="share_pct",
        y="feature",
        orientation="h",
        color_discrete_sequence=["#2E6F9E"],
    )
    _style(
        fig,
        "LightGBM gain share by feature",
        "Share of total gain (%)",
        "Feature",
    )
    fig.update_layout(height=max(420, 26 * len(imp)))
    st.plotly_chart(fig, width="stretch")
    top = pipeline.importance.iloc[0]
    st.caption(
        f"`{top['feature']}` alone accounts for {top['share_pct']:.0f}% of the "
        "gain; the age, output and contract features supply the rest."
    )

    table = pipeline.importance.copy()
    table["gain"] = table["gain"].map(lambda v: f"{v:,.0f}")
    table["share_pct"] = table["share_pct"].map(lambda v: f"{v:.1f}%")
    st.dataframe(table, width="stretch", hide_index=True)


def worst_errors(pipeline: ui.Pipeline) -> None:
    st.header("Worst errors")
    st.write(
        "The ten biggest LightGBM misses on the test block, in euros. These are "
        "mostly negotiation shocks: the generator adds a random error to every "
        "fee that no feature can reveal, so a handful of deals will always land "
        "far from any prediction."
    )

    table = pipeline.worst.copy()
    for col in ("actual_fee_eur", "predicted_fee_eur", "error_eur", "abs_error_eur"):
        table[col] = table[col].map(lambda v: f"{v:,.0f}")
    table["error_pct"] = table["error_pct"].map(lambda v: f"{v:+.1f}%")
    table = table.rename(
        columns={
            "player_id": "Player",
            "season": "Season",
            "age": "Age",
            "position": "Pos",
            "league": "League",
            "actual_fee_eur": "Actual (EUR)",
            "predicted_fee_eur": "Predicted (EUR)",
            "error_eur": "Error (EUR)",
            "abs_error_eur": "Abs error (EUR)",
            "error_pct": "Error %",
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "Error = predicted minus actual, in euros. Player IDs refer to the "
        "synthetic dataset, not real footballers."
    )