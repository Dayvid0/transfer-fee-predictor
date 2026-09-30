"""Model training on a strict chronological split.

Train seasons are <= 2022 and test seasons are >= 2023. Nothing from a test
season ever reaches a fit, a validation set or a preprocessing statistic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src import features as F

N_JOBS = 2


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    cols: list[str]

    @property
    def summary(self) -> dict:
        return {
            "train_rows": len(self.train),
            "valid_rows": len(self.valid),
            "test_rows": len(self.test),
            "train_seasons": f"{int(self.train.season.min())}-{int(self.train.season.max())}",
            "valid_season": int(self.valid.season.iloc[0]) if len(self.valid) else None,
            "test_seasons": f"{int(self.test.season.min())}-{int(self.test.season.max())}",
        }


@dataclass
class TrainedModels:
    split: Split
    baseline_log: np.ndarray
    linear: Pipeline
    booster: lgb.LGBMRegressor
    valid_years: int = field(default=0)


def time_split(
    frame: pd.DataFrame,
    train_last: int = C.TRAIN_LAST_SEASON,
    test_first: int = C.TEST_FIRST_SEASON,
) -> Split:
    """Cut the frame by season only. The newest training season is held out
    for early stopping so no test season is ever used for tuning."""
    cols = F.feature_columns(frame)
    assert "season" not in cols, "season must drive the split, not the fit"

    train_pool = frame.loc[frame["season"] < test_first]
    train = train_pool.loc[train_pool["season"] <= train_last - 1]
    valid = train_pool.loc[train_pool["season"] == train_last]
    test = frame.loc[frame["season"] >= test_first].reset_index(drop=True)

    assert train.season.max() <= train_last - 1, "training set reached into test years"
    assert valid.season.max() <= train_last, "validation set reached into test years"
    assert test.season.min() >= test_first, "test set predates the test window"

    return Split(train=train, valid=valid, test=test, cols=cols)


def fit_models(frame: pd.DataFrame) -> TrainedModels:
    """Fit models A, B and C.

    Note the deliberate asymmetry with the usual convention: the *fit* window
    ends at 2021 and the 2022 season is used only for early stopping, so the
    2023-2025 test block stays completely untouched.
    """
    split = time_split(frame)

    x_tr = split.train[split.cols]
    y_tr = split.train["target"]
    x_va = split.valid[split.cols]
    y_va = split.valid["target"]

    # StandardScaler matters: raw EUR columns are ~1e8-scale, which makes
    # uncentred OLS extrapolate wildly onto unseen seasons.
    linear = Pipeline(
        [("scale", StandardScaler()), ("model", LinearRegression())]
    ).fit(x_tr, y_tr)

    # Shallow trees with linear leaves, chosen on the 2022 validation season
    # only. linear_tree=True is the important bit: plain regression trees
    # cannot extrapolate the fee-inflation trend past the last training season,
    # whereas linear leaves can.
    booster = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=8000,
        learning_rate=0.02,
        num_leaves=8,
        min_child_samples=60,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=10.0,
        linear_tree=True,
        n_jobs=N_JOBS,
        random_state=C.SEED,
        verbose=-1,
    )
    booster.fit(
        x_tr,
        y_tr,
        eval_X=x_va,
        eval_y=y_va,
        eval_names=["valid"],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )

    # Model A needs no fitting: the club's own pre-transfer valuation, mapped
    # onto the same log1p scale as the target.
    baseline_log = np.log1p(split.test["market_value_eur"].to_numpy())

    return TrainedModels(
        split=split,
        baseline_log=baseline_log,
        linear=linear,
        booster=booster,
        valid_years=int(booster.best_iteration_ or booster.n_estimators),
    )