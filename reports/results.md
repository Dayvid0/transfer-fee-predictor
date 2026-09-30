# Football transfer fee prediction

Generated 2026-09-30 15:13 · seed 20240517 · data regenerated

## Setup

- **5,000 transfers**, 2,266 distinct players,
  seasons 2015-2025, all synthetic.
- **Zero-fee rows removed** (loans/free transfers): 200
  rows -> **4,800 paid transfers** used for training and scoring.
- Median paid fee EUR 3,617,500
  (2015 median 2,345,000
  -> 2025 median 5,765,000,
  consistent with the ~10%/yr inflation baked into the generator).
- Target: `log1p(fee_eur)`. Features are pre-transfer only: age + age^2,
  per-90 output rates, minutes, contract years, the club's pre-transfer
  valuation, one-hot position/league and a linear season index.
  `fee_eur`, `player_id`, `origin_league` and `is_free_or_loan` are never
  features.

## Time split (never random)

| block | seasons | rows |
|---|---|---|
| fit | 2015-2021 | 3,015 |
| early-stopping validation | 2022 | 480 |
| test (scored once) | 2023-2025 | 1,305 |

LightGBM stopped at **698 trees** using the 2022 validation
season. No test-season row touches a fit, a scaler or a tuning decision.

## Results on the 2023-2025 test block

| model          | mae_log | mae_eur   | rmse_eur  | mape_pct | bias_eur | r2_log |
| -------------- | ------- | --------- | --------- | -------- | -------- | ------ |
| A_market_value | 0.2742  | 1,855,958 | 3,201,131 | 28.8     | 419,176  | 0.766  |
| B_linear       | 0.1952  | 1,238,668 | 2,078,406 | 19.1     | -391,508 | 0.878  |
| C_lightgbm     | 0.2069  | 1,359,455 | 2,433,196 | 20.4     | -486,832 | 0.865  |

`mae_log` is MAE on the log1p scale; `mae_eur` is MAE after `expm1`, i.e. the
average euro error in the transfer price. `bias_eur` is mean(predicted -
actual).

## Per-season MAE (EUR)

| season | n   | median_actual_eur | mae_A_market_value_eur | mae_B_linear_eur | mae_C_lightgbm_eur |
| ------ | --- | ----------------- | ---------------------- | ---------------- | ------------------ |
| 2023   | 475 | 4,815,000         | 1,701,926              | 1,022,896        | 1,096,660          |
| 2024   | 432 | 5,377,500         | 1,839,259              | 1,291,267        | 1,395,873          |
| 2025   | 398 | 5,765,000         | 2,057,915              | 1,439,091        | 1,633,562          |

## LightGBM feature importance (top 15, gain share)

| feature             | gain   | share_pct |
| ------------------- | ------ | --------- |
| market_value_eur    | 20,029 | 77.9      |
| log_market_value    | 2,089  | 8.1       |
| contract_years_left | 992    | 3.9       |
| age                 | 451    | 1.8       |
| goal_contrib_p90    | 340    | 1.3       |
| season_index        | 260    | 1.0       |
| output_x_minutes    | 234    | 0.9       |
| goals_p90           | 202    | 0.8       |
| xg_p90              | 163    | 0.6       |
| contract_x_age      | 153    | 0.6       |
| pos_ST              | 133    | 0.5       |
| minutes             | 98     | 0.4       |
| xg_over_shots       | 63     | 0.2       |
| lg_Premier League   | 61     | 0.2       |
| assists_p90         | 59     | 0.2       |

## 10 worst C_lightgbm errors (EUR)

| player_id | season | age  | position | league         | actual_fee_eur | predicted_fee_eur | error_eur   | abs_error_eur | error_pct |
| --------- | ------ | ---- | -------- | -------------- | -------------- | ----------------- | ----------- | ------------- | --------- |
| 115       | 2025   | 27.0 | ST       | La Liga        | 37,650,000     | 65,675,892        | 28,025,892  | 28,025,892    | +74.4%    |
| 1786      | 2025   | 28.0 | RW       | Premier League | 52,985,000     | 26,662,917        | -26,322,083 | 26,322,083    | -49.7%    |
| 1517      | 2024   | 26.0 | RW       | La Liga        | 45,605,000     | 25,876,440        | -19,728,560 | 19,728,560    | -43.3%    |
| 126       | 2025   | 25.0 | RB       | Premier League | 21,635,000     | 39,167,762        | 17,532,762  | 17,532,762    | +81.0%    |
| 650       | 2025   | 30.0 | AM       | Bundesliga     | 18,465,000     | 35,130,931        | 16,665,931  | 16,665,931    | +90.3%    |
| 215       | 2025   | 23.0 | LW       | Premier League | 25,895,000     | 13,906,884        | -11,988,116 | 11,988,116    | -46.3%    |
| 381       | 2024   | 24.0 | CM       | Bundesliga     | 27,025,000     | 15,422,574        | -11,602,426 | 11,602,426    | -42.9%    |
| 96        | 2024   | 23.0 | LW       | Bundesliga     | 35,135,000     | 46,729,777        | 11,594,777  | 11,594,777    | +33.0%    |
| 1455      | 2025   | 27.0 | RW       | Premier League | 26,695,000     | 15,293,335        | -11,401,665 | 11,401,665    | -42.7%    |
| 335       | 2024   | 25.0 | LW       | Eredivisie     | 14,885,000     | 26,179,692        | 11,294,692  | 11,294,692    | +75.9%    |

## Reading the result

- **Best model: `B_linear`**, cutting euro MAE by 33% against the
  market-value baseline (1,855,958
  -> 1,238,668).
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
  0.2069.
- `market_value_eur` takes ~78% of the total LightGBM gain, so this is
  fundamentally a *market-value correction* problem, not a discovery problem.
- Both models under-predict on the test block (bias
  -391,508 EUR for B_linear) and
  error grows with the forecast horizon in the per-season table: every model is
  extrapolating 1-3 seasons past its training window.
- Noise floor: the generator adds lognormal noise with sigma
  0.33, worth about 0.26 log-MAE that no model can recover.

## Assumptions

1. `fee_eur` is the only post-transfer quantity; everything else in the feature
   set is knowable before the window opens. Zero-fee rows are dropped rather
   than modelled, since a loan carries no clearing price.
2. Inflation is applied to the target in nominal euros; `season_index` lets the
   models learn the trend rather than hard-coding the 10% assumption.
3. LightGBM hyperparameters were picked on the 2022 validation season only;
   the test block was scored once at the end.

Runtime: 1.4s on 2 CPU threads.
