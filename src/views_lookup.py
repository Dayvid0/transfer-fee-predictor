"""Page: look up one synthetic player and see how the models priced the move.

The dataset has no player names, only ``player_id``s, so "search" means picking
an ID and reading that player's transfer history. Predictions exist only for the
held-out seasons 2023-2025: training-window transfers show their actual fee and
an explicit "not scored" note rather than a number the models cannot honestly
produce.

This module is layout only. The data shaping lives in ``src.lookup_tables``.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import lookup_tables as LT
from src import ui
from src import visuals as V

UNSCORED_NOTE = (
    "The club valuation is still worth reading: it is the strongest single "
    "feature, so the gap to the fee paid shows how far the two disagreed."
)


def _detail(transfers: pd.DataFrame, pipeline: ui.Pipeline) -> None:
    """Pick one transfer and show its predictions, errors and profile."""
    by_season = transfers.set_index("season")
    season = st.selectbox(
        "Transfer to inspect",
        transfers["season"].tolist(),
        format_func=lambda s: f"{s} · {ui.eur(by_season.loc[s, 'fee_eur'])}",
        key="lookup_season",
    )
    row = by_season.loc[season]
    actual = float(row["fee_eur"])
    scored = not pd.isna(row["pred_c_eur"])

    cols = st.columns(4)
    cols[0].metric("Fee actually paid", ui.eur(actual))
    cols[1].metric("Club valuation", ui.eur(float(row["market_value_eur"])))
    for col, name, pred in (
        (cols[2], "Linear prediction", row["pred_b_eur"]),
        (cols[3], "LightGBM prediction", row["pred_c_eur"]),
    ):
        col.metric(name, ui.eur(float(pred)) if scored else "not scored")

    if not scored:
        st.info(
            f"Season {season} falls inside the fitting window, so the models saw "
            "this transfer while training. Any prediction here would be "
            "memorisation, so only the fee paid is shown. The held-out seasons "
            f"are {LT.TEST_SEASONS}."
        )
        st.caption(UNSCORED_NOTE)
    else:
        st.dataframe(LT.error_table(row), width="stretch", hide_index=True)
        shock = float(row["signal_eur"]) - actual
        st.caption(
            f"The generator's noise-free fee for this profile was "
            f"{ui.eur(float(row['signal_eur']))}, so {ui.eur(abs(shock))} of the "
            "gap between prediction and outcome is negotiation shock no feature "
            "could reveal - the same noise behind the oracle floor on the model "
            "comparison page."
        )

    st.markdown("**Pre-transfer profile** (the models' entire input)")
    st.dataframe(LT.profile_table(row), width="stretch", hide_index=True)
    st.caption(
        f"Modelled in base-year euros, then re-inflated at "
        f"{pipeline.index:.4f} per year. Player {int(row['player_id'])} is "
        "synthetic, not a real footballer."
    )


def _biggest_moves(transfers: pd.DataFrame) -> None:
    """The player's five largest fees paid, drawn as a row of player cards."""
    st.markdown("**Biggest moves on record**")
    st.markdown(V.top_cards(LT.cards(transfers)), unsafe_allow_html=True)
    st.caption("Largest fees paid for this player, not model predictions.")


def player_lookup(pipeline: ui.Pipeline) -> None:
    st.header("Player lookup")
    st.write(
        "Look up a synthetic player and read what the models predicted for each "
        "of their moves. The dataset has no real names, so players carry the ID "
        "assigned during generation; type into the box to filter."
    )

    labels = LT.player_choices(pipeline.raw)
    player_id = st.selectbox(
        "Player", sorted(labels), format_func=lambda p: labels[p], key="lookup_player"
    )
    transfers = LT.player_transfers(pipeline, int(player_id))
    if transfers.empty:
        st.warning(f"Player {player_id} has no transfers in this dataset.")
        return

    st.caption(f"{len(transfers)} transfer(s) on record; pick one to see it in full.")
    st.dataframe(LT.summary_table(transfers), width="stretch", hide_index=True)
    _detail(transfers, pipeline)
    _biggest_moves(transfers)
