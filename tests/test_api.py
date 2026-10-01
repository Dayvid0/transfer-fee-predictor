"""Tests that touch the network or render pages.

Both need something the CI box will not have, so both skip rather than fail: a
football-data.org token for the API, and the downloaded snapshot for the pages.
The page tests drive the real sidebar rather than calling the view functions, so
a broken import or a renamed radio key is caught here too.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src import football_data as FD
from src.real_data import CLEAN_CACHE

APP = str(Path(__file__).resolve().parents[1] / "app.py")
REAL_PAGES = ["Trophy cabinet", "League table", "Real player values", "Value predictor"]


def needs_token() -> None:
    if not FD.available():
        pytest.skip("no FOOTBALL_DATA_KEY configured")


def needs_snapshot() -> None:
    if not CLEAN_CACHE.exists():
        pytest.skip("real_players.csv not downloaded; run python -m src.real_data")


# --- API ---------------------------------------------------------------------

def test_trophy_cabinet_returns_winners_with_crests():
    needs_token()
    cabinet = FD.trophy_cabinet("PL")
    assert not cabinet.empty
    assert {"season", "winner", "crest"} <= set(cabinet.columns)
    assert cabinet["winner"].notna().all()
    assert cabinet["crest"].str.startswith("http").all()
    assert cabinet["season"].is_monotonic_decreasing


def test_league_table_has_a_full_set_of_clubs_with_crests():
    needs_token()
    table = FD.league_table("PL")
    assert len(table) >= 18
    assert {"team", "points", "won", "crest"} <= set(table.columns)
    assert table["crest"].notna().all()


# --- pages -------------------------------------------------------------------

@pytest.mark.parametrize("page", REAL_PAGES)
def test_every_real_page_renders(page: str) -> None:
    needs_snapshot()
    needs_token()
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300).run()
    at.radio(key="section").set_value("Real players").run()
    at.radio(key="real_page").set_value(page).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.header[0].value == page
