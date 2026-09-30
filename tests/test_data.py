"""Generator and feature invariants: independent noise terms, correct target,
no post-transfer columns, and deflation that removes the inflation trend."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import features as F
from src import make_data as MD


def test_fee_and_market_value_noise_are_independent(raw):
    """The valuation must not carry the fee's own noise.

    ``log_fee_signal`` is the generator's answer key. Market value is the signal
    plus an independent error, so the two residuals must be uncorrelated.
    Deriving the valuation from the noisy fee instead would put the correlation
    at ~0.7 and let models "beat" the noise floor.
    """
    paid = raw.loc[raw["fee_eur"] > 0]
    fee_resid = np.log1p(paid["fee_eur"]) - paid["log_fee_signal"]
    mv_resid = np.log(paid["market_value_eur"]) - paid["log_fee_signal"]

    corr = float(np.corrcoef(fee_resid, mv_resid)[0, 1])
    # 4/sqrt(n) is a ~4-sigma band: wide enough for sampling noise at this n and
    # far tighter than the ~0.7 a real leak would produce.
    band = 4 / np.sqrt(len(paid))
    assert abs(corr) < band, f"fee/valuation noise correlated: {corr:.4f} (band {band:.4f})"

    assert fee_resid.std() == pytest.approx(C.FEE_NOISE_SIGMA, rel=0.10)
    assert mv_resid.std() == pytest.approx(C.MARKET_VALUE_NOISE_SIGMA, rel=0.10)


def test_generation_is_deterministic_and_shaped(raw):
    pd.testing.assert_frame_equal(raw, MD.make_transfers())
    assert len(raw) == C.N_TRANSFERS
    assert set(raw.season) == set(range(C.FIRST_SEASON, C.LAST_SEASON + 1))
    assert (raw.fee_eur == 0).sum() > 0, "loans/free transfers must be present"
    assert raw[raw.fee_eur > 0].fee_eur.min() > 0


def test_target_and_filtering_are_correct(raw):
    paid = F.drop_unpaid(raw)
    assert (paid.fee_eur > 0).all()
    assert len(paid) < len(raw)

    model = F.build_features(raw)
    np.testing.assert_allclose(
        model["target_nominal"].to_numpy(), np.log1p(paid.fee_eur)
    )
    assert len(model) == len(paid)


def test_features_contain_no_post_transfer_information(keyed):
    cols = F.feature_columns(keyed)
    for forbidden in ("fee_eur", "player_id", "season", "is_free_or_loan",
                      "origin_league", "target", "target_nominal", "log_fee_index",
                      "log_fee_signal", "market_value_eur"):
        assert forbidden not in cols, f"{forbidden} must not be a feature"
    assert "age_sq" in cols
    assert "season_index" in cols
    assert np.isfinite(keyed[cols].to_numpy()).all()


def test_deflation_removes_the_inflation_trend(keyed, price_index):
    assert 1.0 < price_index < 1.2, "index should recover ~10%/yr inflation"
    assert price_index == pytest.approx(C.YEARLY_INFLATION, rel=0.05)

    # Nominal fees must trend upward across seasons, real-terms fees must not.
    recent = keyed.loc[keyed["season"] >= C.TEST_FIRST_SEASON, "target_nominal"].median()
    early = keyed.loc[keyed["season"] == C.FIRST_SEASON, "target_nominal"].median()
    assert recent > early

    real_recent = np.expm1(keyed.loc[keyed["season"] >= C.TEST_FIRST_SEASON, "target"]).median()
    real_early = np.expm1(keyed.loc[keyed["season"] == C.FIRST_SEASON, "target"]).median()
    assert real_recent == pytest.approx(real_early, rel=0.15)


def test_index_is_estimated_without_test_seasons(raw):
    """Adding future seasons must not move the estimated index."""
    full = F.build_features(raw)
    index_now = F.estimate_price_index(full, C.TRAIN_LAST_SEASON)
    train_only = F.build_features(raw.loc[raw["season"] <= C.TRAIN_LAST_SEASON])
    assert index_now == pytest.approx(
        F.estimate_price_index(train_only, C.TRAIN_LAST_SEASON), rel=1e-9
    )


def test_inflation_round_trips(keyed):
    rows = keyed.iloc[[0, 100, 2500]]
    nominal = F.inflate_log(rows["target"].to_numpy(), rows["log_fee_index"].to_numpy())
    np.testing.assert_allclose(nominal, rows["target_nominal"].to_numpy(), rtol=1e-9)