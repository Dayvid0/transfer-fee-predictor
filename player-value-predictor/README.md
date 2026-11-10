# Player Value Predictor

Estimates a top-five-league player's Transfermarkt listed market value from their profile and recent match stats, and explains the estimate in plain English.

Status: Phase 5 (Docker packaging, CI and Render setup). See PLAN.md for the approved blueprint.

## Phase 2a

From the project root, using the existing environment and Phase 1 training table:

```sh
.venv/bin/python -m pytest
.venv/bin/python src/baseline_report.py
```

An alternate parquet table can be passed with `--table PATH`. The report selects
only training and validation dates in DuckDB before returning rows to Python.
The test period stays closed; there is no test-scoring option. No model is fitted
and no artifacts are saved by this command.

- `src/features.py`: stateless features shared by future training and serving.
  Exact order: age, height, minutes, appearances, goals/assists/yellow/red per 90,
  league, position group. Rates are count × 90 / minutes. Missing height stays
  missing; learned imputation and categorical encoding belong to Phase 2b.
  Targets, player IDs, dates and snapshot columns are excluded.
- `src/evaluate.py`: inclusive train end of 2024-06-30, validation from 2024-07-01
  through 2025-06-30. Metrics take original euro values: MAE, RMSE, RMSLE
  (natural log1p), median absolute percentage error, and mean signed error
  (prediction minus listing; negative means underestimation). Each valuation
  row has equal weight. Invalid or empty metric inputs fail explicitly.
- `src/baselines.py`: overall median, league-and-position median, and previous
  value. Group medians fall back to overall median; previous value falls back
  to group then overall median. All references must be strictly earlier than
  the query date, including median fallbacks; same-day values cannot leak.
- `src/baseline_report.py`: trains the baselines on training history only and
  freezes that history for the full validation period. No validation labels
  update later predictions. Previous value means the latest eligible valuation
  in the filtered training table, including a prior league, not the latest raw
  Transfermarkt listing. It is a historical benchmark, not a model input.
  The report includes overall scores, calendar-quarter scores, and new versus
  known players. Known means present in training, fixed at the cutoff. Empty
  cohorts have count zero and unavailable metrics.


## Phase 2b

```sh
.venv/bin/python src/train.py fit --db /mnt/data/datasets/transfermarkt/transfermarkt-datasets.duckdb
.venv/bin/python src/train.py evaluate-test
```

Run `fit` first. It only loads development rows: candidates fit through June
2024, selection uses July–December 2024, the winner is refitted through December
2024, and January–June 2025 calibrates an 80% nominal prediction range. The model
is chosen by selection MAE (RMSLE breaks ties), comparing Ridge and
HistGradientBoosting with all-history and recent-history training. No model
refit or bias adjustment follows calibration. `--db` is optional and is used
only for display names in the historical player lookup.

`evaluate-test` verifies frozen artifacts, code, versions and development data
before loading July 2025–June 2026. It writes the final evaluation once; no test
results affect model selection or ranges. The two commands refuse to overwrite
existing results. Read saved results instead of rerunning them for tuning.
Both commands accept `--table PATH` and `--output DIRECTORY`.

Artifacts live in `artifacts/phase2b/`: the trained pipeline, historical player
lookup, frozen baseline history, feature schema, configuration/integrity
manifest, development metrics and final test metrics. The original artifacts
remain ignored by Git. Phase 5 includes a checksummed serving copy under
`artifacts/release/` after checking the upstream CC0 licence. Use only trusted model
files and the pinned environment. `predict_values` and `prediction_range` in
`src/train.py` are the serving entry points for the next phase.

See [the model card](docs/model-card.md) for split boundaries, scoring rules,
range limitations, and baseline comparability. The FastAPI service is described below.

## Phase 3

Start the local prediction service from the project root:

```sh
.venv/bin/uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000 --workers 1 --no-proxy-headers
```

Visit <http://127.0.0.1:8000/docs> for interactive requests. Endpoints are
`GET /health`, `POST /predict`, `GET /players/search`, and
`GET /players/{player_id}/estimate`. The server loads the frozen model and
historical lookup once at startup, and returns 503 if they are unavailable.
Set `PVP_ARTIFACT_DIR` to use another trusted artifact directory.

[API documentation](docs/api.md) covers exact inputs, dates, response fields,
search pagination, examples, and errors. Inputs require at least 450 minutes in
the prior year. Responses show the estimate, nominal prediction range, profile
date and model development dates. Lookup profiles are explicitly historical;
they are not live player data or independent backtests. No model training, test
evaluation or raw database access happens during serving. Optional Groq wording
receives only server-generated numerical facts.

Tests run with `.venv/bin/python -m pytest -q`. Phase 5 adds Docker, CI and deployment.

## Phase 4

Prediction responses now include grounded explanations and up to three local
what-if comparisons by default. Groq is optional; the complete deterministic
fallback works without an API key. Add `?include_explanation=false` for only
numbers. Player estimates also explain the historical listing difference.

[Explanation documentation](docs/explanations.md) describes the facts contract,
non-causal sensitivities, strict wording/number validation, configuration,
request limits and persistent daily call budget. `GROQ_MODEL` is read from the
environment. No names or user text reach the prompt. All provider failures
return fallback text while preserving the numerical prediction.

This phase was verified with mocked provider responses and local fallback, as
requested. A live Groq faithfulness/latency check remains deferred until real
credentials are configured. This feature does not change model accuracy.

## Phase 5

The API is packaged for Docker, tested by GitHub Actions, and has a free Render
Blueprint. See [deployment instructions](docs/deployment.md) for the nested
GitHub layout, local Docker commands, release verification, logs, CI, free-tier
limits and the steps to create the service. The existing application in
`Dayvid0/transfer-fee-predictor` remains separate.

```sh
docker build -t player-value-predictor:local .
docker run --rm -p 8000:8000 -e PVP_LLM_ENABLED=false player-value-predictor:local
```

The image includes the frozen serving bundle and runs as a non-root user.
No raw dataset, test scoring, retraining or external provider call is needed.
[Dataset provenance](docs/data-findings.md) records the CC0 licence at the
source snapshot revision. Render setup keeps local fallback explanations on
the free plan and waits for passing CI before subsequent automatic deployments.
