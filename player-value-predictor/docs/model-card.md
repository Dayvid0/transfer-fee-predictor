# Player value model — Phase 2b

## Intended use

Estimate a player's Transfermarkt **listed market value**, in EUR, from their
profile and preceding 365 days of match statistics. These are historical listed
values, not transfer fees, future forecasts or a player's true worth. Scope is
players in the top five leagues with at least 450 minutes in the prior year.

## Development protocol

- Candidate training: 2013-07-01 through 2024-06-30.
- Model selection: 2024-07-01 through 2024-12-31. Choose lowest MAE, then RMSLE,
  then candidate name for deterministic ties.
- Drift handling: compare all history with history from 2020-07-01 onward.
  The selected window is used again for the final fit.
- Candidates: Ridge with alpha 1, 10 or 100; HistGradientBoosting with 15 leaves
  and 200 iterations or 31 leaves and 300 iterations; each under both history
  policies. Both predict log1p(EUR). No random early-stopping holdout is used.
- Refit the chosen configuration through 2024-12-31. Do not refit after range
  calibration, because that would change the residual distribution.
- Calibrate ranges on 2025-01-01 through 2025-06-30, held out of fitting and
  model selection. An 80% nominal interval uses the finite-sample order
  statistic of absolute log1p residuals. Bounds are transformed to EUR and
  clipped at zero at the lower end.
- Freeze the model, radius, configuration, source hashes, development-data hash,
  dependency versions and supporting files before the final test is loaded.
- Final test: 2025-07-01 through 2026-06-12. No model selection, bias correction,
  range calibration, or refitting follows test evaluation.

Phase 2a baseline results covered the full validation year. Phase 2b's candidate
ranking uses only its first half; those are different comparison samples.
Calibration coverage describes the data used to construct the range and must
not be described as independent validation coverage.

## Features and preprocessing

The ten ordered inputs after transformation are age, height, minutes, appearances,
goals/assists/yellow/red per 90, league and position group. Height is imputed with
a training-fitted median plus a missing-value indicator. Categories are one-hot
encoded with unseen categories ignored; numeric features are standardized for
Ridge. All fitted preprocessing stays inside the model pipeline. Dates, player
IDs, current or previous listed values and club snapshots are not model inputs.
The previous-value baseline has historical target information the model does
not receive, so it is a demanding reference rather than a feature-matched model.

## Evaluation

Every valuation row has equal weight. Metrics are MAE and RMSE in EUR, natural
log1p RMSLE, median absolute percentage error, and mean signed error
(prediction minus listing). Report overall scores, calendar quarters, new/known
players, leagues and positions. Candidate-selection cohorts share the full training-player roster through June
2024 so candidate slices are directly comparable. Calibration and test cohorts
use the selected model's actual fit roster: its selected history window through
December 2024. Baselines use all pre-calibration history, frozen
at December 2024; they do not absorb calibration or test labels.

The saved development and final-test metrics are the authoritative numeric
results. Test range coverage and width are reported overall and by slice.
Small slices should not be treated as stable estimates.

## Limitations

Repeated valuations of the same player are correlated. The chronological split
measures performance on later valuations, not just unseen players. Temporal
shift and player dependence invalidate a blanket exchangeability assumption,
so the nominal 80% range is not a guaranteed coverage level. Undercoverage and
wide ranges must be reported rather than silently adjusted after testing.

The player lookup contains the latest eligible **historical** development row
per player, with its valuation date, age, and optional name. It is not current
squad data. No date of birth is included. Names are for display/search only and
must never be included in LLM prompts. Phase 5 recorded the upstream CC0 licence at the dataset revision in
`docs/data-findings.md`; the checksummed release preserves these frozen artifacts.

## Artifacts and reproducibility

`artifacts/phase2b/` holds `model.joblib`, `players.parquet`,
`baseline_history.parquet`, `schema.json`, `development_metrics.json`,
`selection.json` and, after final evaluation, `test_metrics.json`. Load only
locally trusted joblib files. Integrity hashes detect accidental changes; they
do not authenticate an untrusted model. Use the pinned training environment.
The manifest is written last. Both fitting and test evaluation refuse to
overwrite prior output; final evaluation never modifies the frozen model.

## Recorded results — 7 November 2026

Selected: HistGradientBoosting, 31 leaves, 300 iterations, recent history from
July 2020. Candidate selection MAE was EUR 6.906 million on 2,507 early-validation
rows, versus EUR 7.968 million for the same model using all history and
EUR 3.360 million for the previous-value baseline. The final model was refitted
on 25,840 rows through December 2024. Its radius was calibrated on 2,681 rows.
The calibration set's new-player coverage was 66.9%; this limitation was
recorded before test evaluation, and the range was not changed in response.

Final test evaluation covers **5,687 valuations of 2,385 players** within
2025-07-01 through 2026-06-12. No eligible test rows occur in 2025 Q3, so that
quarter has no reported score. The model and range were frozen before loading
these rows.

| Method | MAE (EUR m) | RMSE (EUR m) | RMSLE | Median APE | Signed error (EUR m) |
|---|---:|---:|---:|---:|---:|
| Overall median | 12.628 | 22.810 | 1.320 | 72.2% | -10.962 |
| League/position median | 11.601 | 21.435 | 1.198 | 66.7% | -9.792 |
| Previous value | 7.108 | 11.845 | 0.786 | 44.4% | -1.502 |
| Selected model | 6.627 | 12.362 | 0.679 | 39.4% | -4.092 |

The model's test MAE is 6.8% lower than the previous-value baseline,
but its RMSE is higher. It underestimates listings by EUR 4.09 million on average.
This is mixed performance, not evidence that every kind of error improved.
The previous-value reference is frozen at December 2024 for this evaluation;
it is not refreshed with calibration/test labels. Its historical values become
staler over time, which helps explain why it can win early validation yet
lose on later test valuations.

The nominal 80% prediction range covered **81.3%** of test listings. Mean width
was EUR 22.93 million and median width EUR 13.62 million: coverage comes with
broad ranges, not precise valuations.

| Model test slice | Rows | MAE (EUR m) | Signed error (EUR m) | Range coverage |
|---|---:|---:|---:|---:|
| known | 4,314 | 7.149 | -4.738 | 81.8% |
| new | 1,373 | 4.986 | -2.063 | 79.6% |
| 2025Q4 | 2,838 | 6.432 | -3.856 | 81.2% |
| 2026Q1 | 790 | 7.583 | -5.092 | 81.6% |
| 2026Q2 | 2,059 | 6.529 | -4.033 | 81.3% |

Known/new in this table means present/absent in the recent-history final fit,
not necessarily absent from every older raw record. Lower absolute error for
new players does not establish that they are easier: their value distribution
also differs. No correction or refit was made after the final test.

The saved model is approximately 465 KiB, and the historical lookup contains
6,725 named players. A fresh-process load reproduced finite predictions and
ordered ranges; all frozen file hashes still match after evaluation. The full
67-test suite passed, including development/test isolation, training-only
preprocessing, range calculations, serialization, and artifact integrity.
