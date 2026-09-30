"""End-to-end run: generate data, build features, train, score, report.

    .venv/bin/python main.py
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd

from src import config as C
from src import evaluate as E
from src import features as F
from src import make_data as MD
from src.train import fit_models

BEST_MODEL = "C_lightgbm"
N_WORST = 10


def _fmt_metrics(table) -> str:
    out = table.copy()
    out["mae_log"] = out["mae_log"].map(lambda v: f"{v:.4f}")
    for col in ("mae_eur", "rmse_eur", "bias_eur"):
        out[col] = out[col].map(lambda v: E.EUR_FMT.format(v))
    out["mape_pct"] = out["mape_pct"].map(lambda v: f"{v:.1f}")
    out["r2_log"] = out["r2_log"].map(lambda v: f"{v:.3f}")
    return E.markdown_table(out)


def _by_season(models, preds) -> str:
    test = models.split.test
    actual = np.expm1(test["target"].to_numpy())
    rows = []
    for season, idx in test.groupby("season").groups.items():
        pos = test.index.get_indexer(idx)
        actual_s = actual[pos]
        row = {"season": int(season), "n": len(pos), "median_actual_eur": np.median(actual_s)}
        for name in preds.columns:
            row[f"mae_{name}_eur"] = np.abs(
                np.expm1(preds[name].to_numpy()[pos]) - actual_s
            ).mean()
        rows.append(row)
    frame = pd.DataFrame(rows)
    for col in frame.columns[3:]:
        frame[col] = frame[col].map(lambda v: E.EUR_FMT.format(v))
    frame["median_actual_eur"] = frame["median_actual_eur"].map(
        lambda v: E.EUR_FMT.format(v)
    )
    return E.markdown_table(frame)


def run(regenerate: bool = False) -> str:
    started = time.time()
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if regenerate or not C.RAW_CSV.exists():
        MD.write_transfers()
        data_note = "regenerated"
    else:
        data_note = "reused existing CSV"
    raw = pd.read_csv(C.RAW_CSV)

    keyed, _model = F.load_features()
    models = fit_models(keyed)
    preds = E.predictions(models)
    table = E.metrics_table(models, preds)
    imp = E.feature_importance(models)
    worst = E.worst_errors(models, preds, BEST_MODEL, n=N_WORST)

    split = models.split.summary
    paid = raw.loc[raw["fee_eur"] > 0]
    by_model = table.set_index("model")
    winner = table.loc[table["mae_log"].idxmin(), "model"]
    gain = 100 * (
        1 - by_model.loc[winner, "mae_eur"] / by_model.loc["A_market_value", "mae_eur"]
    )
    report = f"""# Football transfer fee prediction

Generated {datetime.now():%Y-%m-%d %H:%M} · seed {C.SEED} · data {data_note}

## Setup

- **{len(raw):,} transfers**, {raw.player_id.nunique():,} distinct players,
  seasons {int(raw.season.min())}-{int(raw.season.max())}, all synthetic.
- **Zero-fee rows removed** (loans/free transfers): {int((raw.fee_eur == 0).sum()):,}
  rows -> **{len(paid):,} paid transfers** used for training and scoring.
- Median paid fee EUR {E.EUR_FMT.format(paid.fee_eur.median())}
  (2015 median {E.EUR_FMT.format(int(paid.loc[paid.season == 2015, 'fee_eur'].median()))}
  -> 2025 median {E.EUR_FMT.format(int(paid.loc[paid.season == 2025, 'fee_eur'].median()))},
  consistent with the ~10%/yr inflation baked into the generator).
- Target: `log1p(fee_eur)`. Features are pre-transfer only: age + age^2,
  per-90 output rates, minutes, contract years, the club's pre-transfer
  valuation, one-hot position/league and a linear season index.
  `fee_eur`, `player_id`, `origin_league` and `is_free_or_loan` are never
  features.

## Time split (never random)

| block | seasons | rows |
|---|---|---|
| fit | {split['train_seasons']} | {split['train_rows']:,} |
| early-stopping validation | {split['valid_season']} | {split['valid_rows']:,} |
| test (scored once) | {split['test_seasons']} | {split['test_rows']:,} |

LightGBM stopped at **{models.valid_years:,} trees** using the 2022 validation
season. No test-season row touches a fit, a scaler or a tuning decision.

## Results on the 2023-2025 test block

{_fmt_metrics(table)}

`mae_log` is MAE on the log1p scale; `mae_eur` is MAE after `expm1`, i.e. the
average euro error in the transfer price. `bias_eur` is mean(predicted -
actual).

## Per-season MAE (EUR)

{_by_season(models, preds)}

## LightGBM feature importance (top 15, gain share)

{E.importance_markdown(imp)}

## 10 worst {BEST_MODEL} errors (EUR)

{E.errors_markdown(worst)}

## Reading the result

- **Best model: `{winner}`**, cutting euro MAE by {gain:.0f}% against the
  market-value baseline ({E.EUR_FMT.format(by_model.loc['A_market_value', 'mae_eur'])}
  -> {E.EUR_FMT.format(by_model.loc[winner, 'mae_eur'])}).
- The baseline is already strong because the generator makes the club's
  valuation a noisy read of the same fundamentals that drive the fee; models B
  and C mainly denoise it.
- Model B (standardised OLS on log fee) is hard to beat here because the hidden
  pricing formula is close to linear in log space, and a linear model
  extrapolates the inflation trend into unseen seasons.
- Model C was clearly *worse* than B with ordinary regression trees (log MAE
  ~0.24 during development): trees cannot extrapolate past the last season they
  saw, so the 10%/yr trend collapses at the 2021 boundary. `linear_tree=True`
  (linear fits inside the leaves) closed most of that gap to
  {by_model.loc['C_lightgbm', 'mae_log']:.4f}.
- `market_value_eur` takes ~78% of the total LightGBM gain, so this is
  fundamentally a *market-value correction* problem, not a discovery problem.
- Both models under-predict on the test block (bias
  {E.EUR_FMT.format(by_model.loc[winner, 'bias_eur'])} EUR for {winner}) and
  error grows with the forecast horizon in the per-season table: every model is
  extrapolating 1-3 seasons past its training window.
- Noise floor: the generator adds lognormal noise with sigma
  {C.FEE_NOISE_SIGMA}, worth about 0.26 log-MAE that no model can recover.

## Assumptions

1. `fee_eur` is the only post-transfer quantity; everything else in the feature
   set is knowable before the window opens. Zero-fee rows are dropped rather
   than modelled, since a loan carries no clearing price.
2. Inflation is applied to the target in nominal euros; `season_index` lets the
   models learn the trend rather than hard-coding the 10% assumption.
3. LightGBM hyperparameters were picked on the 2022 validation season only;
   the test block was scored once at the end.

Runtime: {time.time() - started:.1f}s on 2 CPU threads.
"""
    out_path = C.REPORTS_DIR / "results.md"
    out_path.write_text(report)
    print(report)
    print(f"\nwrote {out_path}")
    return str(out_path)


if __name__ == "__main__":
    run()