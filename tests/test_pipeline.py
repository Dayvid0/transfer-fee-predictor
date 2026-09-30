"""The pipeline invariants that matter: no time leakage, no post-transfer
features, and models that actually beat the naive baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import evaluate as E
from src import features as F
from src import make_data as MD
from src.train import time_split

TEST_SEASONS = range(C.TEST_FIRST_SEASON, C.LAST_SEASON + 1)


# --- time leakage -----------------------------------------------------------


def test_no_test_season_row_is_ever_used_for_training(keyed, models):
    """Fit + early-stopping windows must sit entirely before the test block."""
    split = models.split

    assert split.train.season.max() <= C.TRAIN_LAST_SEASON - 1
    assert split.valid.season.max() <= C.TRAIN_LAST_SEASON
    assert set(split.train.season.unique()) & set(TEST_SEASONS) == set()
    assert set(split.valid.season.unique()) & set(TEST_SEASONS) == set()

    # No (player, season) event may appear in both a training block and test.
    def keys(frame: pd.DataFrame) -> set[tuple[int, int]]:
        return set(zip(frame.player_id.astype(int), frame.season.astype(int)))

    fitted = keys(split.train) | keys(split.valid)
    assert fitted & keys(split.test) == set()
    assert fitted != keys(split.test)


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


# --- features and target ----------------------------------------------------


def test_target_and_filtering_are_correct(raw):
    paid = F.drop_unpaid(raw)
    assert (paid.fee_eur > 0).all()
    assert len(paid) < len(raw)

    model = F.build_features(raw)
    np.testing.assert_allclose(model["target"].to_numpy(), np.log1p(paid.fee_eur))
    assert len(model) == len(paid)


def test_features_contain_no_post_transfer_information(keyed):
    cols = F.feature_columns(keyed)
    for forbidden in ("fee_eur", "player_id", "season", "is_free_or_loan",
                      "origin_league", "target"):
        assert forbidden not in cols, f"{forbidden} must not be a feature"
    assert "age_sq" in cols
    assert "season_index" in cols
    assert np.isfinite(keyed[cols].to_numpy()).all()


# --- data generation --------------------------------------------------------


def test_generation_is_deterministic_and_shaped(raw):
    again = MD.make_transfers()
    pd.testing.assert_frame_equal(raw, again)
    assert len(raw) == C.N_TRANSFERS
    assert set(raw.season) == set(range(C.FIRST_SEASON, C.LAST_SEASON + 1))
    assert (raw.fee_eur == 0).sum() > 0, "loans/free transfers must be present"
    assert raw[raw.fee_eur > 0].fee_eur.min() > 0


# --- model quality ----------------------------------------------------------


def test_both_models_beat_the_market_value_baseline(models, preds):
    table = E.metrics_table(models, preds).set_index("model")
    for name in ("B_linear", "C_lightgbm"):
        assert table.loc[name, "mae_log"] < table.loc["A_market_value", "mae_log"], name
        assert table.loc[name, "mae_eur"] < table.loc["A_market_value", "mae_eur"], name


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
        "## LightGBM feature importance",
        "## 10 worst",
        "## Assumptions",
    ):
        assert heading in report, heading
    for model in ("A_market_value", "B_linear", "C_lightgbm"):
        assert model in report


def test_reports_cover_three_models_and_ten_worst_errors(models, preds):
    assert list(E.predictions(models).columns) == [
        "A_market_value",
        "B_linear",
        "C_lightgbm",
    ]
    table = E.metrics_table(models, preds)
    assert len(table) == 3
    assert {"mae_log", "mae_eur", "rmse_eur", "bias_eur"} <= set(table.columns)
    assert table.notna().all().all()

    worst = E.worst_errors(models, preds, "C_lightgbm", n=10)
    assert len(worst) == 10
    assert worst.abs_error_eur.is_monotonic_decreasing
    assert (worst.season >= C.TEST_FIRST_SEASON).all()

    imp = E.feature_importance(models)
    assert imp.shape[0] > 0
    assert imp.gain.sum() > 0
    assert set(imp.feature) <= set(models.split.cols)