"""Page: type a player's name, see the model's valuation against the real one.

The model was fitted on the same 2,570 players the answer is drawn from, so this
page is an illustration of what the model does rather than a clean forecast.
The held-out numbers live on the model comparison page.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import plots
from src import real_model as RM
from src import real_ui as RU

SPLIT_LABELS = {
    "clubs": "Leave clubs out (the honest split)",
    "random": "Random 80/20 (the flattering one)",
}

MATCH_LIMIT = 40


def _fitted(data: RU.RealData, split: str) -> RM.Fitted:
    return data.fitted_clubs if split == "clubs" else data.fitted_random


def _player_card(data: RU.RealData, row: pd.Series) -> None:
    """Crest, identity block and the attribute radar for one player."""
    crest = st.session_state.get("crest_map", {}).get(str(row["club"]))
    left, right = st.columns([1, 3])
    if crest:
        left.image(crest, width=74)
    with right:
        st.subheader(f"{row['name']}")
        st.caption(
            f"{row['position']} · age {row['age']:.0f} · {row['club']} · "
            f"rating {row['overall']:.0f} (potential {row['potential']:.0f})"
        )
    left.pyplot(plots.radar(row))


def _predictions(fitted: RM.Fitted, row: pd.Series) -> dict:
    return RM.predict(fitted, pd.DataFrame([row]))


def value_predictor(data: RU.RealData) -> None:
    st.header("Value predictor")
    st.write(
        "Type a real player's name. The model reads their statistics, predicts a "
        "market value, and shows it next to the value their club actually "
        "assigned - so you can see where the two disagree and why."
    )

    split = st.radio(
        "Which fitted model scores this player",
        list(SPLIT_LABELS),
        format_func=lambda k: SPLIT_LABELS[k],
        horizontal=True,
    )
    fitted = _fitted(data, split)

    names = RU.options(data)
    query = st.text_input(
        "Player",
        value=names[0] if names else "",
        key="value_query",
        placeholder="Type a name, e.g. Haaland or Mbappe (accents optional)",
    )
    matches = data.find(query)
    if matches.empty:
        st.warning(f"No player matching {query!r}. Try part of a surname.")
        return

    # Each row is its own option, so a player listed at two clubs is offered
    # twice with the club spelled out rather than being silently collapsed.
    shown = matches.head(MATCH_LIMIT)
    choice = st.selectbox(
        f"{len(matches):,} match(es)", shown.index.tolist(),
        format_func=lambda i: RU.player_label(shown.loc[i]),
    )
    row = shown.loc[choice]

    _player_card(data, row)
    predicted = _predictions(fitted, row)
    actual = float(row["value_eur"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Real market value", RU.eur(actual))
    c2.metric("Model prediction", RU.eur(float(predicted["booster"][0])), delta=None)
    c3.metric("Linear regression", RU.eur(float(predicted["linear"][0])), delta=None)
    c4.metric("Scout rating only", RU.eur(float(predicted["scout"][0])), delta=None)

    gap = (float(predicted["booster"][0]) / actual - 1) * 100
    direction = "above" if gap >= 0 else "below"
    st.info(
        f"The boosted model prices this player {abs(gap):.0f}% {direction} the "
        f"club's own figure ({RU.eur(float(predicted['booster'][0]))} against "
        f"{RU.eur(actual)}). Where the two disagree, it is usually because the "
        "rating already embeds the vendor's view of the same player - the "
        "attributes and the value come from one source, so a large gap means "
        "the model is re-deriving a number it was partly handed."
    )

    st.markdown("**What the model saw**")
    facts = pd.DataFrame(
        [
            {"Input": "Age", "Value": f"{row['age']:.0f}"},
            {"Input": "Position", "Value": str(row["position"])},
            {"Input": "Club", "Value": str(row["club"])},
            {"Input": "Height / weight", "Value": f"{row['height_cm']:.0f}cm / {row['weight_kg']:.0f}kg"},
            {
                "Input": "Contract years left",
                "Value": "unknown (loan)" if pd.isna(row["contract_years_left"])
                else f"{row['contract_years_left']:.0f}",
            },
            {"Input": "Rating / potential", "Value": f"{row['overall']:.0f} / {row['potential']:.0f}"},
            {"Input": "Wage (excluded)", "Value": f"{RU.eur(float(row['wage_eur']))} - not a feature"},
            {"Input": "Release clause (excluded)", "Value": f"{RU.eur(float(row['release_clause_eur']))} - not a feature"},
        ]
    )
    st.dataframe(facts, width="stretch", hide_index=True)

    st.caption(
        "Wage and release clause are excluded on purpose: the source derives both "
        "from the same valuation, so using them would smuggle the answer back in."
    )

    with st.expander("How this model does on held-out players"):
        st.dataframe(
            RU.metrics_frame(data).loc[
                lambda t: t["Model"].isin(["B. Linear regression", "C. Gradient boosting"])
            ],
            width="stretch",
            hide_index=True,
        )
        st.pyplot(plots.value_scatter(
            fitted.split.test,
            fitted.split.test[RM.TARGET].to_numpy(),
            RM.predict(fitted, fitted.split.test)["booster"],
        ))
