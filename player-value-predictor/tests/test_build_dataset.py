from datetime import date

import pandas as pd
import pytest

import build_dataset
import config
from fixture_db import SNAPSHOT_UPDATE

EXPECTED_KEYS = {(1, "2015-01-01"), (2, "2016-01-01"), (2, "2018-01-01"), (5, "2020-07-01")}


def build(con, **kwargs):
    df, steps = build_dataset.build_training_table(con, **kwargs)
    return df, steps


def row(df, player_id, day):
    match = df[(df.player_id == player_id) & (df.valuation_date.dt.strftime("%Y-%m-%d") == day)]
    assert len(match) == 1, f"expected one row for player {player_id} on {day}"
    return match.iloc[0]


def test_only_the_expected_rows_survive(con):
    df, _ = build(con)
    keys = {(int(r.player_id), r.valuation_date.strftime("%Y-%m-%d")) for r in df.itertuples()}
    assert keys == EXPECTED_KEYS


def test_filter_waterfall_counts(con):
    _, steps = build(con)
    assert [n for _, n in steps] == [8, 6, 5, 4]


def test_window_includes_365_days_back_and_nothing_later(con):
    df, _ = build(con)
    r = row(df, 1, "2015-01-01")
    assert r.minutes_365 == 450       # five 90-minute games; the three "poison" games are out
    assert r.appearances_365 == 5
    assert r.goals_365 == 4           # 1 + 2 + 1, across league, Champions League and cup
    assert r.assists_365 == 1
    assert r.yellow_365 == 1
    assert r.red_365 == 0


def test_minutes_threshold_is_inclusive(con):
    df, _ = build(con)                              # exactly 450 minutes is kept ...
    assert (1, "2015-01-01") in {(int(r.player_id), r.valuation_date.strftime("%Y-%m-%d")) for r in df.itertuples()}
    df, _ = build(con, min_minutes=451)             # ... and 451 removes it
    assert set(df.player_id) == {5}


def test_age_is_measured_at_the_valuation_date(con):
    df, _ = build(con)
    expected = (date(2016, 1, 1) - date(1999, 3, 1)).days / 365.25
    assert row(df, 2, "2016-01-01").age_years == pytest.approx(expected, abs=0.01)
    expected = (date(2018, 1, 1) - date(1999, 3, 1)).days / 365.25
    assert row(df, 2, "2018-01-01").age_years == pytest.approx(expected, abs=0.01)


def test_league_comes_from_the_valuation_row(con):
    df, _ = build(con)
    assert row(df, 2, "2016-01-01").league == "ES1"
    assert row(df, 2, "2018-01-01").league == "IT1"


def test_position_groups(con):
    df, _ = build(con)
    assert row(df, 1, "2015-01-01").position_group == "ATT"
    assert row(df, 2, "2016-01-01").position_group == "MID"
    assert row(df, 5, "2020-07-01").position_group == "GK"


def test_missing_height_stays_missing(con):
    df, _ = build(con)
    assert pd.isna(row(df, 2, "2016-01-01").height_cm)


def test_columns_are_exactly_the_agreed_ones(con):
    df, _ = build(con)
    assert df.columns.tolist() == build_dataset.COLUMNS
    forbidden = {"market_value_in_eur", "highest_market_value_in_eur", "contract_expiration_date",
                 "current_club_id", "current_club_name", "current_club_domestic_competition_id",
                 "last_season", "international_caps", "international_goals", "current_national_team_id"}
    assert forbidden.isdisjoint(df.columns)


def test_changing_every_snapshot_column_changes_nothing(con):
    """If a 'today' column leaked into a feature, this test would fail."""
    before, _ = build(con)
    con.execute(SNAPSHOT_UPDATE)
    after, _ = build(con)
    pd.testing.assert_frame_equal(before, after)


def test_no_duplicate_player_dates(con):
    df, _ = build(con)
    assert not df.duplicated(["player_id", "valuation_date"]).any()


def test_split_dates_are_in_order():
    assert config.FIRST_VALUATION_DATE < config.TRAIN_END < config.VALID_END < config.TEST_END
