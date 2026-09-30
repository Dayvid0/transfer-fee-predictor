"""Time-split and model-quality invariants: no test-season row reaches
training, and no model beats the oracle noise floor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import evaluate as E
from src.train import time_split

TEST_SEASONS = range(C.TEST_FIRST_SEASON, C.LAST_SEASON + 1)


# --- time leakage -----------------------------------------------------------


def test_no_test_season_row_is_ever_used_for_training(models):
    """Fit + early-stopping windows must sit entirely before the test block."""
    split = models.split

    assert split.train.season.max() <= C.TRAIN_LAST_SEASON - 1
    assert split.valid.season.max() <= C.TRAIN_LAST_SEASON
    assert set(split.train.season.unique()) & set(TEST_SEASONS) == set()
    assert set(split.valid.season.unique()) & set(TEST_SEASONS) == set()

    def keys(frame: pd.DataFrame) -> set[tuple[int, int]]:
        return set(zip(frame.player_id.astype(int), frame.season.astype(int)))

    fitted = keys(split.train) | keys(split.valid)
    assert fitted & keys(split.test) == set()


def test_test_block_is_strictly_future_and_non_empty(models):
    split = models.split
    assert len(split.test) > 0
    assert split.test.season.min() == C.TEST_FIRST_SEASON
    assert split.test.season.max() == C.LAST_SEASON
    assert split.test.index.equals(
        pd.RangeIndex(len(split.test))
    ), "test rows must be re-indexed so predictions align by position"


def test_time_split_assertions_catch_a_leaky_split(keyed):
    """Sanity check on the guard rails themselves."""
    with pytest.raises(AssertionError):
        time_split(keyed, train_last=C.LAST_SEASON, test_first=C.TEST_FIRST_SEASON)
    with pytest.raises(AssertionError):
        time_split(keyed, train_last=C.TRAIN_LAST_SEASON, test_first=C.FIRST_SEASON)


# --- model quality ----------------------------------------------------------


def test_no_model_beats_the_oracle_noise_floor(models, preds):
    """The negotiation shock is independent noise, so nothing can beat it.

    This is the test that would have caught the market-value leak: a model
    scoring under the floor is reading the answer, not learning the market.
    """
    table = E.metrics_table(models, preds).set_index("model")
    floor = table.loc["oracle_noise_floor", "mae_log"]
    assert floor == pytest.approx(
        C.FEE_NOISE_SIGMA * np.sqrt(2 / np.pi), abs=0.02
    ), "floor should match the analytic value for the injected noise"

    for name in ("A_market_value", "B_linear", "C_lightgbm"):
        assert (
            table.loc[name, "mae_log"] >= floor
        ), f"{name} scored {table.loc[name, 'mae_log']:.4f} below floor {floor:.4f}"


def test_both_models_beat_the_market_value_baseline(models, preds):
    table = E.metrics_table(models, preds).set_index("model")
    for name in ("B_linear", "C_lightgbm"):
        assert table.loc[name, "mae_log"] < table.loc["A_market_value", "mae_log"], name
        assert table.loc[name, "mae_eur"] < table.loc["A_market_value", "mae_eur"], name


def test_predictions_are_scored_on_the_nominal_scale(models, preds):
    """Predictions must be re-inflated before being compared to paid fees."""
    test = models.split.test
    assert float(preds["B_linear"].mean()) < float(test["target_nominal"].mean()), (
        "predictions look like they were never re-inflated"
    )


def test_reports_cover_three_models_and_ten_worst_errors(models, preds):
    assert list(preds.columns) == ["A_market_value", "B_linear", "C_lightgbm"]

    table = E.metrics_table(models, preds)
    assert len(table) == 4  # three models plus the oracle floor row
    assert {"mae_log", "mae_eur", "rmse_eur", "bias_eur"} <= set(table.columns)

    worst = E.worst_errors(models, preds, "C_lightgbm", n=10)
    assert len(worst) == 10
    assert worst.abs_error_eur.is_monotonic_decreasing
    assert (worst.season >= C.TEST_FIRST_SEASON).all()

    imp = E.feature_importance(models)
    assert imp.shape[0] > 0
    assert imp.gain.sum() > 0
    assert set(imp.feature) <= set(models.split.cols)


def test_main_writes_a_report_end_to_end(tmp_path, monkeypatch, csv_path):
    """Full run: generate -> features -> train -> report, in a temp directory."""
    from src import config as C
    import main as pipeline

    monkeypatch.setattr(C, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(C, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(C, "RAW_CSV", tmp_path / "data" / "transfers.csv")
    monkeypatch.setattr(pipeline, "C", C)

    out = pipeline.run(regenerate=True)
    report = (tmp_path / "reports" / "results.md").read_text()
    assert out.endswith("results.md")
    assert (tmp_path / "data" / "transfers.csv").exists()
    for heading in (
        "## Time split (never random)",
        "## Results on the 2023-2025 test block",
        "### Noise floor",
        "## LightGBM feature importance",
        "## 10 worst",
        "## Assumptions",
    ):
        assert heading in report, heading
    for model in ("A_market_value", "B_linear", "C_lightgbm", "oracle_noise_floor"):
        assert model in report
    assert "No model beats the noise floor" in report