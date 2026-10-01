"""Real-data pages: trophy cabinet, live league table, real player list.

Everything here sits on top of real sources rather than the synthetic market:
club crests, league winners and league tables come from football-data.org,
player valuations from the Kaggle snapshot.
"""
from __future__ import annotations

import streamlit as st

from src import football_data as FD
from src import plots
from src import real_ui as RU

TROPHY_LIMIT = 24


def _competition_picker(key: str) -> str:
    return st.selectbox(
        "Competition",
        list(RU.COMPETITION_LABELS),
        format_func=lambda c: RU.COMPETITION_LABELS[c],
        key=key,
    )


def _offline_notice() -> None:
    st.error(
        "No football-data.org token found. Add one to `.env` as "
        "`FOOTBALL_DATA_KEY=...` and restart. Everything else on this page "
        "still works; only the live API pages need it."
    )


def trophy_cabinet(data: RU.RealData) -> None:
    st.header("Trophy cabinet")
    st.write(
        "Winners as recorded by football-data.org, with each club's own crest. "
        "Coverage is uneven, and the page states it rather than implying a "
        "complete list: the Premier League backfills 1990-2022, while Serie A "
        "carries only 2017-2020 and the Champions League has gaps even inside "
        "the years it does cover."
    )
    if not RU.api_ready():
        _offline_notice()
        return

    code = _competition_picker("trophy_comp")
    try:
        cabinet = RU.load_cabinet(code)
    except FD.ApiUnavailable as exc:
        st.error(str(exc))
        return

    if cabinet.empty:
        st.warning(f"{RU.COMPETITION_LABELS[code]} has no recorded winners in the API.")
        return

    emblem = cabinet["emblem"].iloc[0]
    head, body = st.columns([1, 5])
    if emblem:
        head.image(emblem, width=88)
    body.subheader(f"{RU.COMPETITION_LABELS[code]}: {len(cabinet)} recorded winners")
    body.caption(
        f"Most recent: {cabinet.iloc[0]['winner']} in {int(cabinet.iloc[0]['season'])}. "
        f"Oldest on record: {int(cabinet.iloc[-1]['season'])}."
    )
    st.info(
        "These are the seasons football-data.org actually populates, not the full "
        "history of the competition. Treat the cabinet as a sample."
    )

    st.subheader("Recent winners")
    show = cabinet.head(TROPHY_LIMIT)
    columns = st.columns(4)
    for position, (_, row) in enumerate(show.iterrows()):
        with columns[position % 4]:
            if row["crest"]:
                st.image(row["crest"], width=54)
            st.markdown(f"**{int(row['season'])}**")
            st.caption(row["winner"])

    with st.expander("All recorded winners as a table"):
        table = show.rename(
            columns={"season": "Season", "winner": "Winner", "competition": "Competition"}
        )
        st.dataframe(
            table[["Season", "Winner"]], width="stretch", hide_index=True
        )


def league_table_page(data: RU.RealData) -> None:
    st.header("League table")
    st.write(
        "The live table straight from football-data.org, with each club's crest. "
        "These are the real standings, not generated ones."
    )
    if not RU.api_ready():
        _offline_notice()
        return

    code = _competition_picker("table_comp")
    try:
        table = RU.load_table(code)
    except FD.ApiUnavailable as exc:
        st.error(str(exc))
        return

    if table.empty:
        st.warning(f"No standings returned for {RU.COMPETITION_LABELS[code]}.")
        return

    shown = table.copy()
    shown["record"] = (
        shown["won"].astype(str) + "-" + shown["draw"].astype(str) + "-"
        + shown["lost"].astype(str)
    )
    st.dataframe(
        shown[["position", "team", "playedGames", "record", "goalDifference", "points"]],
        width="stretch",
        hide_index=True,
        column_config={
            "position": "Pos",
            "team": "Club",
            "playedGames": "P",
            "record": "W-D-L",
            "goalDifference": "GD",
            "points": "Pts",
        },
    )
    st.caption(
        f"{len(table)} clubs. Crests for these clubs are available on the trophy "
        "page; the valuation dataset spells club names differently, so crests "
        "are only shown on players where the names match exactly."
    )


def player_list(data: RU.RealData) -> None:
    st.header("Real player values")
    st.write(
        "2,570 real footballers with the market value their club assigned them "
        "in June 2024. Search by name, or filter to a position. Values are an "
        "asking price, not a fee that changed hands."
    )

    head = st.columns(4)
    head[0].metric("Players", f"{len(data.players):,}")
    head[1].metric("Clubs", f"{data.players['club'].nunique():,}")
    head[2].metric("Median value", RU.eur_m(data.players["value_eur"].median()))
    head[3].metric(
        "Most valuable", RU.eur_m(data.players["value_eur"].max())
    )

    st.pyplot(plots.value_distribution(data.players))
    st.pyplot(plots.model_bars(data.metrics, "clubs"))
    st.caption(
        "Bars are test error for the hardest split, where whole clubs are held "
        "out. The model comparison page explains why that split is the honest one."
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        query = st.text_input("Search by name", key="list_query", placeholder="e.g. Haaland")
    with c2:
        position = st.selectbox("Position", ["All"] + sorted(data.players["position"].unique()))

    rows = data.find(query) if query else data.players
    if position != "All":
        rows = rows.loc[rows["position"] == position]

    st.caption(f"{len(rows):,} player(s) match.")
    table = rows.nlargest(300, "value_eur")[
        ["name", "age", "position", "club", "overall", "potential", "value_eur"]
    ].rename(
        columns={
            "name": "Player", "age": "Age", "position": "Pos", "club": "Club",
            "overall": "Rating", "potential": "Potential", "value_eur": "Value (EUR)",
        }
    )
    table["Age"] = table["Age"].astype("Int64")
    table["Value (EUR)"] = table["Value (EUR)"].map(lambda v: f"{v:,.0f}")
    st.dataframe(table, width="stretch", hide_index=True)
    if len(rows) > 300:
        st.caption("Showing the 300 highest-valued of the matches.")
