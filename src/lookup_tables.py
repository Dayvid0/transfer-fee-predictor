"""Pure table and lookup helpers for the player lookup page.

Everything here is a plain function over dataframes: no Streamlit calls, no
rendering. Splitting them out keeps the page module about layout and lets the
shapes be tested directly.

The join key matters. ``(player_id, season)`` is unique in the raw table, so
joining a player's history to its held-out predictions on that pair can never
attach a prediction to the wrong transfer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import ui

TEST_SEASONS = "2023-2025"
MODEL_KEYS = (
    ("pred_a_eur", "A. Market value"),
    ("pred_b_eur", "B. Linear regression"),
    ("pred_c_eur", "C. LightGBM"),
)
PROFILE_ROWS = [
    ("Age", "age", "{:.0f}"), ("Position", "position", "{}"),
    ("Destination league", "league", "{}"), ("Origin league", "origin_league", "{}"),
    ("Goals per 90", "goals_p90", "{:.3f}"), ("Assists per 90", "assists_p90", "{:.3f}"),
    ("Expected goals per 90", "xg_p90", "{:.3f}"), ("Minutes", "minutes", "{:,.0f}"),
    ("Contract years left", "contract_years_left", "{:.2f}"),
]
SUMMARY_COLUMNS = {
    "season": "Season", "age": "Age", "position": "Position", "league": "League",
    "goals_p90": "Goals/90", "minutes": "Minutes",
    "contract_years_left": "Contract yrs",
    "market_value_eur": "Market value (EUR)", "fee_eur": "Fee paid (EUR)",
}
SUMMARY_FORMATS = {
    "Market value (EUR)": "{:,.0f}", "Fee paid (EUR)": "{:,.0f}",
    "Goals/90": "{:.3f}", "Contract yrs": "{:.2f}",
}


def scored_transfers(pipeline: ui.Pipeline) -> pd.DataFrame:
    """Predictions in nominal EUR for every held-out transfer.

    Rows exist only for the test block, so a left join leaves NaN everywhere
    else - which is how training-window moves end up unscored.
    """
    test = pipeline.test
    return pd.DataFrame(
        {
            "player_id": test["player_id"].to_numpy(),
            "season": test["season"].to_numpy(),
            "pred_a_eur": pipeline.predictions_eur("A_market_value"),
            "pred_b_eur": pipeline.predictions_eur("B_linear"),
            "pred_c_eur": pipeline.predictions_eur("C_lightgbm"),
            "signal_eur": np.exp(test["log_fee_signal"].to_numpy()),
        }
    )


def player_transfers(pipeline: ui.Pipeline, player_id: int) -> pd.DataFrame:
    """One row per transfer for this player, newest season first."""
    raw = pipeline.raw.loc[pipeline.raw["player_id"] == player_id]
    merged = raw.merge(scored_transfers(pipeline), on=["player_id", "season"], how="left")
    return merged.sort_values("season", ascending=False).reset_index(drop=True)


def player_choices(raw: pd.DataFrame) -> dict[int, str]:
    """A searchable label per player ID, ordered by ID."""
    moves = raw.groupby("player_id").agg(
        n=("season", "size"), first=("season", "min"), last=("season", "max")
    )
    top_fee = raw.loc[raw["fee_eur"] > 0].groupby("player_id")["fee_eur"].max()
    labels: dict[int, str] = {}
    for pid, r in moves.iterrows():
        fee = top_fee.get(pid)
        tag = f"top {ui.eur_m(fee)}" if pd.notna(fee) else "loans / free only"
        plural = "s" if r["n"] > 1 else ""
        labels[int(pid)] = (
            f"ID {pid} · {r['n']} move{plural} · {r['first']}-{r['last']} · {tag}"
        )
    return labels


def summary_table(transfers: pd.DataFrame) -> pd.DataFrame:
    """One formatted row per transfer."""
    out = transfers[list(SUMMARY_COLUMNS)].rename(columns=SUMMARY_COLUMNS).copy()
    for col, fmt in SUMMARY_FORMATS.items():
        out[col] = out[col].map(lambda v, f=fmt: f.format(v))
    return out


def error_table(row: pd.Series) -> pd.DataFrame:
    """Signed and percentage error for each model against the fee paid."""
    actual = float(row["fee_eur"])
    return pd.DataFrame(
        [
            {
                "Model": label,
                "Predicted (EUR)": f"{float(row[col]):,.0f}",
                "Actual (EUR)": f"{actual:,.0f}",
                "Error (EUR)": f"{float(row[col]) - actual:,.0f}",
                "Error %": f"{100 * (float(row[col]) - actual) / actual:+.1f}%",
            }
            for col, label in MODEL_KEYS
        ]
    )


def profile_table(row: pd.Series) -> pd.DataFrame:
    """Every pre-transfer input the models were given for this move."""
    return pd.DataFrame(
        [{"Field": name, "Value": fmt.format(row[col])} for name, col, fmt in PROFILE_ROWS]
    )


def cards(transfers: pd.DataFrame) -> list[dict]:
    """Card rows for the five largest fees paid, richest first.

    Players are synthetic and unnamed, so the label is the generated ID.
    """
    top = transfers.loc[transfers["fee_eur"] > 0].nlargest(5, "fee_eur")
    return [
        {
            "name": f"Player {int(r['player_id'])}",
            "position": str(r["position"]),
            "league": str(r["league"]),
            "age": int(r["age"]),
            "fee_eur": float(r["fee_eur"]),
        }
        for _, r in top.iterrows()
    ]
