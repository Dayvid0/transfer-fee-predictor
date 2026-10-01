# Football transfer fee prediction

Generated 2026-10-01 13:08 · seed 20240517 · data reused existing CSV

## Setup

- **5,000 transfers**, 2,266 distinct players, seasons
  2015-2025, all synthetic.
- **Zero-fee rows removed** (loans/free transfers): 200 rows
  -> **4,800 paid transfers** used for training and scoring.
- Median paid fee EUR 3,617,500 (2015 median
  2,345,000 -> 2025 median
  5,765,000).
- Money is modelled in **base-year euros**. The annual fee-inflation index is estimated from
  the training seasons as **1.0928** (slope of the median log fee per season); the
  generator's true rate is 1.1. Modelling real terms keeps the
  relationship stationary, so the models never extrapolate a price trend into unseen
  seasons. Predictions are re-inflated before scoring.
- Features are pre-transfer only: age + age^2, per-90 output rates, minutes, contract years,
  the club's valuation (real terms), one-hot position/league and a linear season index.
  `fee_eur`, `player_id`, `origin_league`, `is_free_or_loan` and the `log_fee_signal` answer
  key are never features.

## Time split (never random)

| block | seasons | rows |
|---|---|---|
| fit | 2015-2021 | 3,015 |
| early-stopping validation | 2022 | 480 |
| test (scored once) | 2023-2025 | 1,305 |

LightGBM stopped at **496 trees** using the 2022 validation season, and
the price index is estimated on the same training seasons. No test-season row touches a fit,
a scaler or a tuning decision.

## Results on the 2023-2025 test block

| model              | mae_log | mae_eur   | rmse_eur  | mape_pct | bias_eur | r2_log |
| ------------------ | ------- | --------- | --------- | -------- | -------- | ------ |
| A_market_value     | 0.3816  | 2,441,858 | 3,743,735 | 42.1     | 137,086  | 0.545  |
| B_linear           | 0.2706  | 1,669,460 | 2,791,261 | 26.0     | -654,846 | 0.761  |
| C_lightgbm         | 0.2695  | 1,724,281 | 2,937,707 | 27.6     | -601,701 | 0.763  |
| oracle_noise_floor | 0.2544  | 1,626,890 | -         | -        | -        | -      |

`mae_log` is MAE on the nominal log1p scale; `mae_eur` is MAE after `expm1`, i.e. the average
euro error against the price actually paid. `bias_eur` is mean(predicted - actual).

`oracle_noise_floor` is not a model: it predicts the generator's noise-free log fee exactly,
which is the best any model can do because the negotiation shock is unpredictable by
construction.

### Noise floor

- Floor: **0.2544** log-MAE / **1,626,890 EUR** (generator noise
  sigma 0.33; the analytic floor for a zero-mean Gaussian is
  sigma*sqrt(2/pi) = 0.2633).
- **No model beats the noise floor.** All three land 5.9-50.0% above it, which is the expected result.
- Margin over the floor: A 50.0%, B 6.4%,
  C 5.9%.

## Per-season MAE (EUR)

| season | n   | median_actual_eur | mae_A_market_value_eur | mae_B_linear_eur | mae_C_lightgbm_eur |
| ------ | --- | ----------------- | ---------------------- | ---------------- | ------------------ |
| 2023   | 475 | 4,815,000         | 2,110,432              | 1,360,430        | 1,417,085          |
| 2024   | 432 | 5,377,500         | 2,519,306              | 1,782,543        | 1,848,367          |
| 2025   | 398 | 5,765,000         | 2,753,342              | 1,915,533        | 1,956,223          |

## LightGBM feature importance (top 15, gain share)

| feature               | gain   | share_pct |
| --------------------- | ------ | --------- |
| market_value_real     | 10,259 | 50.6      |
| contract_years_left   | 2,406  | 11.9      |
| log_market_value_real | 1,647  | 8.1       |
| age                   | 974    | 4.8       |
| goal_contrib_p90      | 838    | 4.1       |
| goals_p90             | 610    | 3.0       |
| output_x_minutes      | 602    | 3.0       |
| contract_x_age        | 538    | 2.7       |
| pos_ST                | 479    | 2.4       |
| xg_p90                | 429    | 2.1       |
| lg_Premier League     | 162    | 0.8       |
| pos_CB                | 146    | 0.7       |
| minutes               | 136    | 0.7       |
| pos_LW                | 109    | 0.5       |
| pos_LB                | 105    | 0.5       |

## 10 worst C_lightgbm errors (EUR)

| player_id | season | age  | position | league         | actual_fee_eur | predicted_fee_eur | error_eur   | abs_error_eur | error_pct |
| --------- | ------ | ---- | -------- | -------------- | -------------- | ----------------- | ----------- | ------------- | --------- |
| 1786      | 2025   | 28.0 | RW       | Premier League | 52,985,000     | 21,565,970        | -31,419,030 | 31,419,030    | -59.3%    |
| 1517      | 2024   | 26.0 | RW       | La Liga        | 45,605,000     | 16,329,853        | -29,275,147 | 29,275,147    | -64.2%    |
| 2176      | 2025   | 30.0 | ST       | Premier League | 37,910,000     | 16,571,768        | -21,338,232 | 21,338,232    | -56.3%    |
| 96        | 2024   | 23.0 | LW       | Bundesliga     | 35,135,000     | 15,030,810        | -20,104,190 | 20,104,190    | -57.2%    |
| 215       | 2025   | 23.0 | LW       | Premier League | 25,895,000     | 8,718,708         | -17,176,292 | 17,176,292    | -66.3%    |
| 115       | 2025   | 27.0 | ST       | La Liga        | 37,650,000     | 20,546,407        | -17,103,593 | 17,103,593    | -45.4%    |
| 835       | 2025   | 26.0 | LW       | Eredivisie     | 32,915,000     | 17,600,220        | -15,314,780 | 15,314,780    | -46.5%    |
| 288       | 2024   | 26.0 | DM       | Premier League | 28,185,000     | 14,292,520        | -13,892,480 | 13,892,480    | -49.3%    |
| 740       | 2024   | 25.0 | ST       | Premier League | 38,610,000     | 24,740,155        | -13,869,845 | 13,869,845    | -35.9%    |
| 126       | 2025   | 25.0 | RB       | Premier League | 21,635,000     | 8,707,312         | -12,927,688 | 12,927,688    | -59.8%    |

## Reading the result

- **Best model on log-MAE: `C_lightgbm`**. B and C are effectively tied (log-MAE
  0.2706 vs 0.2695, a
  0.4% gap) and
  both sit within ~6% of the noise floor, so most of the remaining error is irreducible
  negotiation noise rather than model error.
- `market_value_real` takes 51% of the total LightGBM gain: this is a
  *market-value correction* problem. The valuation is a noisy observation of the same
  fundamentals, and combining it with the rest of the feature set removes roughly
  29% of the baseline's log error.
- **The baseline's apparent strength in an earlier revision was an artefact of a leak**, not
  a finding: the generator used to build `market_value_eur` from the noisy log fee, so the
  valuation secretly carried the fee's own noise. Both are now the clean signal plus
  *independent* errors, and the baseline duly looks much weaker (log-MAE
  0.3816) - which is exactly what one noisy estimate gives,
  since its residual sigma is sqrt(0.33^2 + 0.34^2) =
  0.474, an MAE near 0.378.
- Deflating to base-year euros is what makes ordinary regression trees viable at all: before
  it, trees collapsed at the 2021 boundary and scored *worse* than the baseline. The
  inflation trend is now removed from the target rather than extrapolated, and
  `linear_tree=True` adds a further small but reproducible validation gain (0.2741 vs 0.2772
  log-MAE).
- Every model under-predicts (bias -601,701 EUR for
  C_lightgbm) because the estimated index, 1.0928, sits slightly below the true
  1.1; the shortfall compounds with the forecast horizon, visible in the
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

Runtime: 1.1s on 2 CPU threads.
