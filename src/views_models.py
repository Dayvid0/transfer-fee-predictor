"""Pages: model comparison, predicted vs actual, try it yourself."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.views_data import _style

MODEL_LABELS = {
    "A_market_value": "A. Market value (baseline)",
    "B_linear": "B. Linear regression",
    "C_lightgbm": "C. LightGBM",
    "oracle_noise_floor": "Oracle noise floor (not a model)",
}
SCATTER_MODEL = "C_lightgbm"


def _display_table(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["mae_log"] = out["mae_log"].map(lambda v: f"{v:.4f}")
    for col in ("mae_eur", "rmse_eur", "bias_eur"):
        out[col] = out[col].map(
            lambda v: "-" if pd.isna(v) else f"{v:,.0f}"
        )
    out["mape_pct"] = out["mape_pct"].map(
        lambda v: "-" if pd.isna(v) else f"{v:.1f}%"
    )
    out["r2_log"] = out["r2_log"].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")
    out["model"] = out["model"].map(lambda v: MODEL_LABELS.get(v, v))
    return out.rename(
        columns={
            "model": "Model",
            "mae_log": "MAE (log)",
            "mae_eur": "MAE (EUR)",
            "rmse_eur": "RMSE (EUR)",
            "mape_pct": "MAPE",
            "bias_eur": "Bias (EUR)",
            "r2_log": "R2 (log)",
        }
    )


def model_comparison(pipeline: ui.Pipeline) -> None:
    st.header("Model comparison")
    st.write(
        "All three models are scored on the same held-out seasons, 2023 to "
        "2025, and on the same euro scale. Predictions are converted back from "
        "base-year euros before scoring, so these are fees as actually paid."
    )

    st.dataframe(
        _display_table(pipeline.metrics), width="stretch", hide_index=True
    )

    scale = st.radio(
        "Bar chart scale",
        ["MAE (EUR)", "MAE (log scale)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    column = "mae_eur" if "EUR" in scale else "mae_log"
    plot = pipeline.metrics.copy()
    plot["label"] = plot["model"].map(lambda v: MODEL_LABELS.get(v, v))
    plot["value"] = plot[column]
    plot["colour"] = [
        ui.PALETTE["muted"] if "oracle" in m else ui.PALETTE["accent"] for m in plot.model
    ]

    fig = go.Figure(
        go.Bar(
            x=plot["value"],
            y=plot["label"],
            orientation="h",
            marker_color=plot["colour"],
            text=[f"{v:,.0f}" if "EUR" in scale else f"{v:.4f}" for v in plot.value],
            textposition="outside",
        )
    )
    fig = _style(
        fig,
        f"Test-block error by model ({scale})",
        "Mean absolute error (EUR)" if "EUR" in scale else "Mean absolute error (log1p fee)",
        "",
    )
    fig.update_layout(height=300, showlegend=False, margin=dict(l=10, r=70, t=60, b=10))
    fig.update_xaxes(range=[0, float(plot["value"].max()) * 1.22])
    st.plotly_chart(fig, width="stretch")

    floor = pipeline.metrics.set_index("model").loc["oracle_noise_floor"]
    best = pipeline.metrics[pipeline.metrics.model != "oracle_noise_floor"].sort_values(
        "mae_log"
    ).iloc[0]
    st.info(
        f"The oracle noise floor ({floor['mae_log']:.4f} log-MAE, "
        f"{ui.eur(floor['mae_eur'])}) is not a model: it is the score an oracle "
        "would get by predicting the generator's noise-free fee exactly. The "
        "market adds a random negotiation shock to every deal that no feature "
        "can reveal, so this is the best any model could possibly do. "
        f"{MODEL_LABELS[best['model']]} comes closest at {best['mae_log']:.4f} "
        f"({100 * (best['mae_log'] / floor['mae_log'] - 1):.1f}% above the floor). "
        "Any model scoring below this line would be reading the answer key, "
        "not learning the market."
    )


def predicted_vs_actual(pipeline: ui.Pipeline) -> None:
    st.header("Predicted vs actual")
    st.write(
        "Every point is one held-out transfer from 2023 to 2025. The dashed "
        "line is a perfect prediction: points above it were overpriced by the "
        "model, points below it were underpriced."
    )

    model = st.selectbox(
        "Model",
        ["B_linear", "C_lightgbm"],
        format_func=lambda v: MODEL_LABELS[v],
    )
    colour_by = st.radio(
        "Colour points by",
        ["season", "league", "position"],
        horizontal=True,
    )

    test = pipeline.test
    plot = pd.DataFrame(
        {
            "actual": pipeline.test_actual_eur(),
            "predicted": pipeline.predictions_eur(model),
            colour_by: test[colour_by].to_numpy(),
        }
    )
    plot["residual_log"] = np.log(plot["predicted"]) - np.log(plot["actual"])

    fig = px.scatter(
        plot,
        x="actual",
        y="predicted",
        color=colour_by,
        color_discrete_sequence=ui.SERIES,
        log_x=True,
        log_y=True,
        opacity=0.6,
        hover_data={"actual": ":,.0f", "predicted": ":,.0f"},
    )
    lo = float(min(plot.actual.min(), plot.predicted.min())) * 0.7
    hi = float(max(plot.actual.max(), plot.predicted.max())) * 1.4
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="perfect prediction",
            line=dict(dash="dash", color=ui.PALETTE["muted"], width=1.5),
        )
    )
    fig = _style(
        fig,
        f"Predicted vs actual fee, coloured by {colour_by} (log-log)",
        "Actual fee (EUR, log)",
        f"Predicted fee (EUR, log) - {MODEL_LABELS[model]}",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"{len(plot):,} transfers; the cloud tightens around the diagonal as "
        "fees grow, because a fixed euro error matters less on a large deal."
    )

    fig = px.histogram(
        plot,
        x="residual_log",
        nbins=60,
        color_discrete_sequence=[ui.PALETTE["secondary"]],
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color=ui.PALETTE["muted"])
    fig = _style(
        fig,
        "Residuals: log predicted minus log actual",
        "Residual (log ratio, predicted / actual)",
        "Transfers",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "A well-calibrated model centres on zero with a symmetric spread. The "
        "negative tail is the estimated price index sitting just under the "
        "generator's true 10% inflation."
    )