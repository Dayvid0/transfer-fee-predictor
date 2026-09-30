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
    """All three models, fitted on real-terms targets."""

    split: Split
    price_index: float
    baseline_log_real: np.ndarray
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


def fit_models(frame: pd.DataFrame, index: float) -> TrainedModels:
    """Fit models A, B and C on the real-terms target.

    Note the deliberate asymmetry with the usual convention: the *fit* window
    ends at 2021 and the 2022 season is used only for early stopping, so the
    2023-2025 test block stays completely untouched. ``index`` is the annual
    fee-inflation estimate, also derived from training seasons only.
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

    # Ordinary regression trees are now viable because the target is deflated to
    # base-year euros, which removes the inflation trend they could not
    # extrapolate. linear_tree=True still buys a small but reproducible gain on
    # the validation season (0.2741 vs 0.2772 log-MAE), so it stays. All
    # hyperparameters were chosen on the 2022 validation season only.
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

    # Model A needs no fitting: the club's own pre-transfer valuation, on the
    # same real-terms log1p scale the other two models predict.
    baseline_log_real = np.log1p(split.test["market_value_real"].to_numpy())

    return TrainedModels(
        split=split,
        price_index=index,
        baseline_log_real=baseline_log_real,
        linear=linear,
        booster=booster,
        valid_years=int(booster.best_iteration_ or booster.n_estimators),
    )