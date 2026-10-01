"""Dashboard helper tests: the app must reuse the real pipeline, and the
prediction helper must return sane EUR numbers for a user-supplied profile."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import lookup_tables as LT
from src import ui
from src.views_data import _style
from src.views_models import MODEL_LABELS, _display_table
from src.views_try import AGE_RANGE, season_median

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = [
    "Overview",
    "Model comparison",
    "Predicted vs actual",
    "Feature importance",
    "Try it yourself",
    "Worst errors",
    "Player lookup",
]


@pytest.fixture(scope="module")
def dashboard(csv_path: str) -> ui.Pipeline:
    """The same call the app makes, reading a CSV rather than the default path."""
    return ui.build_pipeline(csv_path)


def test_predict_fee_returns_positive_fee_for_a_sample_input(dashboard):
    prediction = ui.predict_fee(dashboard)

    for key in ("linear", "lightgbm", "market_value"):
        value = prediction[key]
        assert isinstance(value, float), key
        assert value > 0, f"{key} returned {value}"
        assert np.isfinite(value)

    # A sanity band: the profile is elite, so the fee must be well above the
    # dataset median but nowhere near absurd.
    assert 1e6 < prediction["lightgbm"] < 200e6
    assert 1e6 < prediction["linear"] < 200e6


def test_predict_fee_handles_every_position_and_league(dashboard):
    for position in ui.C.POSITIONS:
        for league in ui.C.LEAGUES[:3]:
            out = ui.predict_fee(dashboard, dict(position=position, league=league))
            assert out["lightgbm"] > 0, (position, league)


def test_profile_frame_matches_the_trained_design_matrix(dashboard):
    """The profile row must carry exactly the columns the models were fit on."""
    row = ui.profile_frame(
        pd.DataFrame([ui.DEFAULT_PROFILE]), dashboard.index, dashboard.assist_ratio
    )
    matrix = row.reindex(columns=dashboard.cols, fill_value=0.0)
    assert list(matrix.columns) == dashboard.cols
    assert not matrix.isna().any().any(), "profile row has NaNs"
    assert np.isfinite(matrix.to_numpy()).all()


def test_age_curve_peaks_in_the_mid_twenties(dashboard):
    curve = ui.age_curve(dashboard, dict(ui.DEFAULT_PROFILE), AGE_RANGE)
    assert len(curve["lightgbm"]) == len(AGE_RANGE)
    assert (curve["lightgbm"] > 0).all()
    peak = int(AGE_RANGE[int(np.argmax(curve["lightgbm"]))])
    assert 23 <= peak <= 28, f"age curve peaked at {peak}"


def test_dashboard_tables_are_formatted(dashboard):
    table = _display_table(dashboard.metrics)
    assert len(table) == 4
    assert "MAE (EUR)" in table.columns
    assert table["MAE (EUR)"].str.contains(",").any(), "EUR must use separators"
    # The oracle row has undefined secondary metrics.
    oracle = table[table["Model"] == MODEL_LABELS["oracle_noise_floor"]].iloc[0]
    assert oracle["RMSE (EUR)"] == "-"


def test_chart_helper_sets_title_and_axis_labels():
    fig = _style(_blank_figure(), "A title", "X label", "Y label")
    assert fig.layout.title.text == "A title"
    assert fig.layout.xaxis.title.text == "X label"
    assert fig.layout.yaxis.title.text == "Y label"


def _blank_figure():
    import plotly.graph_objects as go

    return go.Figure()


def test_lookup_cards_cover_the_largest_fees(dashboard):
    """Cards must be ordered richest first and never show a zero-fee loan."""
    rows = LT.player_transfers(dashboard, _scored_player(dashboard))
    cards = LT.cards(rows)
    assert len(cards) == min(5, int((rows["fee_eur"] > 0).sum()))
    fees = [c["fee_eur"] for c in cards]
    assert fees == sorted(fees, reverse=True)
    assert all(f > 0 for f in fees)
    assert all(c["name"].startswith("Player ") for c in cards)


def test_cached_loader_returns_the_same_pipeline():
    """st.cache_resource must not refit per call, and must work outside a
    Streamlit runtime as well as inside one."""
    first = ui.load_pipeline()
    second = ui.load_pipeline()
    assert first is second or first.metrics.equals(second.metrics)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_error(page: str) -> None:
    """End-to-end: Streamlit actually executes the script and each page runs
    its real widgets, so this catches plotting and data-shape mistakes that a
    unit test on the helper functions would miss."""
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio(key="page").set_value(page).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.header[0].value == page


def _scored_player(dashboard: ui.Pipeline) -> int:
    """A player with at least one held-out transfer to look up."""
    test = dashboard.test
    return int(test.iloc[0]["player_id"])


def test_player_choices_covers_every_player_in_the_raw_table(dashboard):
    labels = LT.player_choices(dashboard.raw)
    assert len(labels) == dashboard.raw["player_id"].nunique()
    assert set(labels) == set(dashboard.raw["player_id"].unique())
    # The label carries the ID so a filtered list stays readable.
    assert all(str(pid) in text for pid, text in labels.items())


def test_player_lookup_returns_only_that_players_transfers(dashboard):
    player_id = _scored_player(dashboard)
    rows = LT.player_transfers(dashboard, player_id)

    assert not rows.empty
    assert set(rows["player_id"]) == {player_id}
    # Newest season first, one row per (player, season).
    assert rows["season"].is_monotonic_decreasing
    assert not rows.duplicated(["player_id", "season"]).any()
    expected = dashboard.raw.loc[dashboard.raw["player_id"] == player_id]
    assert len(rows) == len(expected)


def test_lookup_joins_predictions_to_the_right_transfer(dashboard):
    """The join is on (player_id, season): a prediction must never be attached
    to a different season of the same player, or a different player's row."""
    player_id = _scored_player(dashboard)
    rows = LT.player_transfers(dashboard, player_id)

    test = dashboard.test.copy()
    test["_pred_c"] = dashboard.predictions_eur("C_lightgbm")
    test = test.set_index(["player_id", "season"])
    for _, row in rows.iterrows():
        key = (int(row["player_id"]), int(row["season"]))
        if key not in test.index:
            assert pd.isna(row["pred_c_eur"]), key
            assert pd.isna(row["signal_eur"]), key
            continue
        assert np.isclose(row["pred_c_eur"], test.loc[key, "_pred_c"]), key
        assert np.isclose(row["signal_eur"], np.exp(test.loc[key, "log_fee_signal"]))


def test_unscored_seasons_leave_predictions_empty(dashboard):
    """Training-window transfers must not be given a prediction: the models saw
    them, so any number would be memorisation rather than forecasting."""
    train_rows = dashboard.raw.loc[dashboard.raw["season"] <= ui.C.TRAIN_LAST_SEASON]
    assert not train_rows.empty

    checked = 0
    for player_id in train_rows["player_id"].unique()[:25]:
        rows = LT.player_transfers(dashboard, int(player_id))
        early = rows.loc[rows["season"] <= ui.C.TRAIN_LAST_SEASON]
        assert early["pred_b_eur"].isna().all()
        assert early["pred_c_eur"].isna().all()
        checked += len(early)
    assert checked > 0


def test_lookup_tables_format_money_with_separators(dashboard):
    rows = LT.player_transfers(dashboard, _scored_player(dashboard))
    summary = LT.summary_table(rows)
    assert len(summary) == len(rows)
    assert summary["Fee paid (EUR)"].str.contains(",").any()
    assert not summary["Fee paid (EUR)"].str.contains("nan").any()

    scored = rows.loc[rows["pred_c_eur"].notna()]
    if not scored.empty:
        errors = LT.error_table(scored.iloc[0])
        assert len(errors) == 3
        assert errors["Error %"].str.endswith("%").all()
        assert errors["Predicted (EUR)"].str.contains(",").all()

    profile = LT.profile_table(rows.iloc[0])
    assert set(profile.columns) == {"Field", "Value"}
    assert not profile["Value"].str.contains("nan").any()


def test_season_median_is_a_positive_reference_for_the_gauge(dashboard):
    """The gauge needs a real reference: the paid-fee median for one season."""
    median = season_median(dashboard.paid, ui.C.LAST_SEASON)
    paid_season = dashboard.paid.loc[dashboard.paid["season"] == ui.C.LAST_SEASON]
    assert median == pytest.approx(paid_season["fee_eur"].median())
    assert median > 0
    # A season with no paid rows must not blow up the gauge.
    assert season_median(dashboard.paid, 1900) == 0.0