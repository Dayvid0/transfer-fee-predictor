"""Tests for parsing and the real-data model.

Parsing tests are hermetic and use a hand-built frame, so they catch the source's
quirks without touching the network. The modelling tests need the downloaded
snapshot and skip when it is absent, so the suite still runs offline. The live
API and page tests live in ``test_api.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import real_data as RD
from src import real_model as RM
from src.real_data import CLEAN_CACHE


def needs_snapshot() -> None:
    if not CLEAN_CACHE.exists():
        pytest.skip("real_players.csv not downloaded; run python -m src.real_data")


@pytest.fixture(scope="module")
def players() -> pd.DataFrame:
    """The frame the app trains on: priced, and each player counted once."""
    needs_snapshot()
    return RM.usable(RM.load())


@pytest.fixture(scope="module")
def fitted(players: pd.DataFrame) -> RM.Fitted:
    return RM.fit_split(players, RM.clubs_split(players))


# --- parsing -----------------------------------------------------------------

def test_parse_eur_handles_every_suffix():
    assert RD.parse_eur("€28.5M") == 28_500_000
    assert RD.parse_eur("€12K") == 12_000
    assert RD.parse_eur("€1.1B") == 1_100_000_000
    assert RD.parse_eur("€500") == 500
    assert np.isnan(RD.parse_eur("n/a"))
    assert np.isnan(RD.parse_eur(""))


def test_parse_rating_applies_in_form_modifier():
    """The source writes '76+2' and '85-1'; a naive to_numeric loses these."""
    assert RD.parse_rating("76+2") == 78
    assert RD.parse_rating("85-1") == 84
    assert RD.parse_rating("70") == 70
    assert np.isnan(RD.parse_rating("n/a"))


def test_clean_name_strips_position_codes():
    assert RD.clean_name("B. Šeško ST") == "B. Šeško"
    assert RD.clean_name("L. Rodríguez RW LW ST") == "L. Rodríguez"
    # A name that merely contains a code keeps every token.
    assert RD.clean_name("Van Dijk") == "Van Dijk"


def test_fold_makes_search_accent_insensitive():
    assert RD.fold("K. Mbappé") == RD.fold("K. Mbappe")
    assert "mbappe" in RD.fold("K. Mbappé")


def test_build_yields_numeric_features_and_a_parsed_value():
    """Every non-text column must come out numeric, or sklearn will refuse it."""
    raw = pd.DataFrame(
        {
            "name": ["A. Tester ST", "B. Second CM"],
            "Age": [24, 30],
            "Overall rating": ["80+2", "70"],
            "Potential": ["85+1", "70"],
            "Team & Contract": ["Test FC\n2022 ~ 2026", "Other FC\nOn loan"],
            "ID": [1, 2],
            "Height": ["180cm / 5'11\"", "175cm / 5'9\""],
            "Weight": ["75kg / 165lbs", "70kg / 154lbs"],
            "foot": ["Right", "Left"],
            "Best overall": [80, 70],
            "Best position": ["ST", "CM"],
            "Growth": [3, 0],
            "Value": ["€10M", "€2.5M"],
            "Wage": ["€50K", "€10K"],
            "Release clause": ["€20M", "-"],
            "Finishing": ["77+1", "55"],
            "Unnamed: 64": [np.nan, np.nan],
        }
    )
    out = RD.build(raw)

    assert "Unnamed: 64" not in out.columns
    assert out["value_eur"].tolist() == [10_000_000.0, 2_500_000.0]
    assert out["height_cm"].tolist() == [180.0, 175.0]
    assert out["contract_years_left"].iloc[0] == 2.0
    assert np.isnan(out["contract_years_left"].iloc[1]), "a loan has no expiry to count"
    assert out["on_loan"].tolist() == [False, True]
    assert out["Finishing"].tolist() == [78.0, 55.0]
    assert out["name"].tolist() == ["A. Tester", "B. Second"]
    assert not out["age"].isna().any(), "derived columns must survive the coercion"


# --- model inputs ------------------------------------------------------------

def test_usable_drops_zero_value_and_duplicate_rows():
    """The source repeats a player per competition, so raw row counts overstate
    the sample. Player_id is not unique in the raw file and must not be either."""
    needs_snapshot()
    raw = RM.load()

    frame = RM.usable(raw)
    assert (frame[RM.TARGET] > 0).all()
    assert frame["player_id"].is_unique, "a repeated player is counted once"
    assert len(frame) < len(raw)
    assert len(frame) == 2570, "distinct priced players in the snapshot"

    columns = [c for c in raw.columns if c != "player_id"]
    assert not frame.duplicated(subset=columns).any()


def test_features_exclude_the_label_and_its_close_allies(players):
    """Wage and release clause come from the same vendor valuation, so using
    them would leak the answer back into the inputs."""
    numeric, categorical = RM.feature_columns(players)
    combined = set(numeric) | set(categorical)

    for banned in ("value_eur", "wage_eur", "release_clause_eur", "club", "name"):
        assert banned not in combined, banned
    assert "overall" in numeric
    assert categorical == ["position", "foot"]


def test_clubs_split_shares_no_club_or_player(players):
    split = RM.clubs_split(players)

    assert split.name == "clubs"
    assert not set(split.train["club"]) & set(split.test["club"]), "a club spans the split"
    assert not set(split.train["player_id"]) & set(split.test["player_id"]), (
        "a player spans the split"
    )
    assert len(split.test) > 0
    assert len(split.train) + len(split.test) + split.dropped_players == len(players)


def test_random_split_is_row_disjoint(players):
    split = RM.random_split(players)
    assert split.name == "random"
    assert set(split.train.index) & set(split.test.index) == set()
    assert len(split.train) + len(split.test) == len(players)


def test_hard_split_is_not_easier_than_the_random_one(players):
    """The whole reason for both splits: the random one is optimistic."""
    table = pd.concat(
        [
            RM.fit_split(players, RM.random_split(players)).metrics,
            RM.fit_split(players, RM.clubs_split(players)).metrics,
        ],
        ignore_index=True,
    )
    wide = table.pivot(index="model", columns="split", values="mae_eur")
    assert (wide["clubs"] > wide["random"]).all(), wide


def test_predict_returns_positive_finite_euros(fitted, players):
    rows = players.head(25)
    out = RM.predict(fitted, rows)
    for key in ("scout", "linear", "booster"):
        assert len(out[key]) == len(rows)
        assert np.isfinite(out[key]).all(), key
        assert (out[key] > 0).all(), key


def test_predictions_are_the_same_whether_scored_whole_or_row_by_row(fitted, players):
    """The dashboard scores one player; the page must not disagree with the batch."""
    rows = players.head(8)
    batch = RM.predict(fitted, rows)["booster"]
    for position in range(len(rows)):
        single = RM.predict(fitted, rows.iloc[[position]])["booster"][0]
        assert np.isclose(single, batch[position]), position


def test_metrics_table_covers_three_models_and_two_splits(players):
    table = pd.concat(
        [
            RM.fit_split(players, RM.random_split(players)).metrics,
            RM.fit_split(players, RM.clubs_split(players)).metrics,
        ],
        ignore_index=True,
    )
    assert len(table) == 6
    assert set(table["split"]) == {"random", "clubs"}
    assert (table["mae_eur"] > 0).all()
    assert (table["mape_pct"] < 200).all()
