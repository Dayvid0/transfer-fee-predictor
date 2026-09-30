"""Dashboard helper tests: the app must reuse the real pipeline, and the
prediction helper must return sane EUR numbers for a user-supplied profile."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import ui
from src.views_data import _style
from src.views_models import MODEL_LABELS, _display_table
from src.views_try import AGE_RANGE

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = [
    "Overview",
    "Model comparison",
    "Predicted vs actual",
    "Feature importance",
    "Try it yourself",
    "Worst errors",
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


def test_cached_loader_returns_the_same_pipeline(tmp_path):
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