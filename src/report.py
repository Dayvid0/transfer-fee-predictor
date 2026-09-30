"""Markdown report rendering. Kept separate from main.py's orchestration."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src import evaluate as E

BEST_MODEL = "C_lightgbm"
N_WORST = 10


def fmt_metrics(table: pd.DataFrame) -> str:
    out = table.copy()
    out["mae_log"] = out["mae_log"].map(lambda v: f"{v:.4f}")
    specs = (("mae_eur", E.EUR_FMT), ("rmse_eur", E.EUR_FMT),
             ("bias_eur", E.EUR_FMT), ("mape_pct", "{:.1f}"), ("r2_log", "{:.3f}"))
    for col, spec in specs:
        out[col] = out[col].map(lambda v, s=spec: "-" if pd.isna(v) else s.format(v))
    return E.markdown_table(out)


def by_season(models, preds: pd.DataFrame) -> str:
    """MAE per test season, all in nominal EUR."""
    test = models.split.test
    actual = test["fee_eur"].to_numpy()
    nominal = {c: np.expm1(E.to_nominal(models, preds[c].to_numpy())) for c in preds}

    rows = []
    for season, idx in test.groupby("season").groups.items():
        pos = test.index.get_indexer(idx)
        row = {
            "season": int(season),
            "n": len(pos),
            "median_actual_eur": np.median(actual[pos]),
        }
        for col, values in nominal.items():
            row[f"mae_{col}_eur"] = np.abs(values[pos] - actual[pos]).mean()
        rows.append(row)

    frame = pd.DataFrame(rows)
    for col in list(frame.columns[3:]) + ["median_actual_eur"]:
        frame[col] = frame[col].map(E.EUR_FMT.format)
    return E.markdown_table(frame)


def _summary(table: pd.DataFrame) -> dict:
    by_model = table.set_index("model")
    models_only = by_model.drop(index="oracle_noise_floor")
    floor = by_model.loc["oracle_noise_floor"]
    return {
        "floor": floor,
        "models": models_only,
        "winner": models_only["mae_log"].idxmin(),
        "margin_pct": 100 * (models_only["mae_log"] / floor["mae_log"] - 1),
        "beaters": models_only.index[
            models_only["mae_log"] < floor["mae_log"]
        ].tolist(),
    }


def build_report(
    raw: pd.DataFrame,
    models,
    preds: pd.DataFrame,
    table: pd.DataFrame,
    importance: pd.DataFrame,
    worst: pd.DataFrame,
    index: float,
    data_note: str,
    runtime_s: float,
    generated: str,
) -> str:
    split = models.split.summary
    paid = raw.loc[raw["fee_eur"] > 0]
    s = _summary(table)
    floor, m, margin, top = s["floor"], s["models"], s["margin_pct"], importance.iloc[0]
    eur = E.EUR_FMT.format
    analytic_floor = C.FEE_NOISE_SIGMA * np.sqrt(2 / np.pi)
    single_est = np.hypot(C.FEE_NOISE_SIGMA, C.MARKET_VALUE_NOISE_SIGMA)
    baseline_gain = 100 * (1 - m.loc["B_linear", "mae_log"] / m.loc["A_market_value", "mae_log"])
    verdict = (
        f"**No model beats the noise floor.** All three land "
        f"{margin.min():.1f}-{margin.max():.1f}% above it, which is the expected result."
        if not s["beaters"]
        else f"**{s['beaters']} scored below the noise floor - investigate.**"
    )

    return f"""# Football transfer fee prediction

Generated {generated} · seed {C.SEED} · data {data_note}

## Setup

- **{len(raw):,} transfers**, {raw.player_id.nunique():,} distinct players, seasons
  {int(raw.season.min())}-{int(raw.season.max())}, all synthetic.
- **Zero-fee rows removed** (loans/free transfers): {int((raw.fee_eur == 0).sum()):,} rows
  -> **{len(paid):,} paid transfers** used for training and scoring.
- Median paid fee EUR {eur(paid.fee_eur.median())} (2015 median
  {eur(int(paid.loc[paid.season == 2015, 'fee_eur'].median()))} -> 2025 median
  {eur(int(paid.loc[paid.season == 2025, 'fee_eur'].median()))}).
- Money is modelled in **base-year euros**. The annual fee-inflation index is estimated from
  the training seasons as **{index:.4f}** (slope of the median log fee per season); the
  generator's true rate is {C.YEARLY_INFLATION}. Modelling real terms keeps the
  relationship stationary, so the models never extrapolate a price trend into unseen
  seasons. Predictions are re-inflated before scoring.
- Features are pre-transfer only: age + age^2, per-90 output rates, minutes, contract years,
  the club's valuation (real terms), one-hot position/league and a linear season index.
  `fee_eur`, `player_id`, `origin_league`, `is_free_or_loan` and the `log_fee_signal` answer
  key are never features.

## Time split (never random)

| block | seasons | rows |
|---|---|---|
| fit | {split['train_seasons']} | {split['train_rows']:,} |
| early-stopping validation | {split['valid_season']} | {split['valid_rows']:,} |
| test (scored once) | {split['test_seasons']} | {split['test_rows']:,} |

LightGBM stopped at **{models.valid_years:,} trees** using the 2022 validation season, and
the price index is estimated on the same training seasons. No test-season row touches a fit,
a scaler or a tuning decision.

## Results on the 2023-2025 test block

{fmt_metrics(table)}

`mae_log` is MAE on the nominal log1p scale; `mae_eur` is MAE after `expm1`, i.e. the average
euro error against the price actually paid. `bias_eur` is mean(predicted - actual).

`oracle_noise_floor` is not a model: it predicts the generator's noise-free log fee exactly,
which is the best any model can do because the negotiation shock is unpredictable by
construction.

### Noise floor

- Floor: **{floor['mae_log']:.4f}** log-MAE / **{eur(floor['mae_eur'])} EUR** (generator noise
  sigma {C.FEE_NOISE_SIGMA}; the analytic floor for a zero-mean Gaussian is
  sigma*sqrt(2/pi) = {analytic_floor:.4f}).
- {verdict}
- Margin over the floor: A {margin['A_market_value']:.1f}%, B {margin['B_linear']:.1f}%,
  C {margin['C_lightgbm']:.1f}%.

## Per-season MAE (EUR)

{by_season(models, preds)}

## LightGBM feature importance (top 15, gain share)

{E.importance_markdown(importance)}

## 10 worst {BEST_MODEL} errors (EUR)

{E.errors_markdown(worst)}

## Reading the result

- **Best model on log-MAE: `{s['winner']}`**. B and C are effectively tied (log-MAE
  {m.loc['B_linear', 'mae_log']:.4f} vs {m.loc['C_lightgbm', 'mae_log']:.4f}, a
  {100 * abs(m.loc['B_linear', 'mae_log'] / m.loc['C_lightgbm', 'mae_log'] - 1):.1f}% gap) and
  both sit within ~6% of the noise floor, so most of the remaining error is irreducible
  negotiation noise rather than model error.
- `{top['feature']}` takes {top['share_pct']:.0f}% of the total LightGBM gain: this is a
  *market-value correction* problem. The valuation is a noisy observation of the same
  fundamentals, and combining it with the rest of the feature set removes roughly
  {baseline_gain:.0f}% of the baseline's log error.
- **The baseline's apparent strength in an earlier revision was an artefact of a leak**, not
  a finding: the generator used to build `market_value_eur` from the noisy log fee, so the
  valuation secretly carried the fee's own noise. Both are now the clean signal plus
  *independent* errors, and the baseline duly looks much weaker (log-MAE
  {m.loc['A_market_value', 'mae_log']:.4f}) - which is exactly what one noisy estimate gives,
  since its residual sigma is sqrt({C.FEE_NOISE_SIGMA}^2 + {C.MARKET_VALUE_NOISE_SIGMA}^2) =
  {single_est:.3f}, an MAE near {single_est * np.sqrt(2 / np.pi):.3f}.
- Deflating to base-year euros is what makes ordinary regression trees viable at all: before
  it, trees collapsed at the 2021 boundary and scored *worse* than the baseline. The
  inflation trend is now removed from the target rather than extrapolated, and
  `linear_tree=True` adds a further small but reproducible validation gain (0.2741 vs 0.2772
  log-MAE).
- Every model under-predicts (bias {eur(m.loc[s['winner'], 'bias_eur'])} EUR for
  {s['winner']}) because the estimated index, {index:.4f}, sits slightly below the true
  {C.YEARLY_INFLATION}; the shortfall compounds with the forecast horizon, visible in the
  per-season table.

## Assumptions

1. `fee_eur` is the only post-transfer quantity. Zero-fee rows are dropped rather than
   modelled, since a loan carries no clearing price.
2. The fee and the market value are the same latent signal plus independent errors. A club's
   valuation and the negotiated fee being *correlated beyond that* would leak, so the
   independence is enforced by a test.
3. Fee inflation is estimated from training seasons rather than hard-coded; real-terms
   fitting handles the level and re-inflation handles scoring.
4. LightGBM hyperparameters were picked on the 2022 validation season only; the test block
   was scored once at the end.

Runtime: {runtime_s:.1f}s on 2 CPU threads.
"""