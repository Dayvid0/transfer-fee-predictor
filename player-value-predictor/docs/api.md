# Prediction API — Phases 3 and 4

Run commands from `/mnt/data/workspace/player-value-predictor`.

```sh
.venv/bin/uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000 --workers 1 --no-proxy-headers
```

Open <http://127.0.0.1:8000/docs> for interactive documentation, or
<http://127.0.0.1:8000/openapi.json> for the schema. This starts a local service.
No raw dataset, Groq key or training run is needed. Complete local explanations
work offline. A configured Groq key enables optional constrained wording;
deployment is Phase 5.

The default artifacts are in `artifacts/phase2b/`. To select another trusted
bundle, set `PVP_ARTIFACT_DIR` to its absolute path before starting the service.
The application reads environment variables directly; it does not automatically
load `.env`. Frozen dependency versions and source/artifact integrity checks must
pass. Restart the server after fixing an unavailable bundle; requests never
reload or refit it. Use the same Python environment as training.

## Endpoints

| Method and path | Purpose | Success |
|---|---|---|
| `GET /health` | Model and lookup readiness, dates, player count | 200 |
| `POST /predict` | Estimate from a supplied profile | 200 |
| `GET /players/search?q=...` | Search historical player names | 200 |
| `GET /players/{player_id}/estimate` | Estimate using a saved historical profile | 200 |

`GET /health` returns 503 when startup loading, integrity validation, or its
warm-up prediction fails. Prediction and search endpoints then return a safe
503 error, while documentation remains available. Diagnostics stay in server
logs rather than exposing local paths in responses.

## Predict a supplied profile

```sh
curl -sS http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "as_of_date": "2025-06-15",
    "age_years": 24,
    "height_cm": 180,
    "minutes_365": 900,
    "appearances_365": 12,
    "goals_365": 10,
    "assists_365": 5,
    "yellow_365": 2,
    "red_365": 1,
    "league": "GB1",
    "position_group": "ATT"
  }'
```

`as_of_date` is a strict `YYYY-MM-DD` date from 2013-07-01 through today in UTC.
Age is measured on that date. Match totals cover the preceding 365 days,
including the start of that window and excluding the profile date. The date is
response context, not an extra model input. Supplying a future date is rejected;
this service estimates listed values and does not forecast future changes.

| Field | Accepted values |
|---|---|
| `age_years` | Number from 15 through 46 |
| `height_cm` | Number from 140 through 220, null, or omitted |
| `minutes_365` | Integer from 450 through 9,000 |
| `appearances_365` | Integer from 1 through 100 |
| `goals_365`, `assists_365` | Integers from 0 through 100 |
| `yellow_365` | Integer from 0 through 200, at most twice appearances |
| `red_365` | Integer from 0 through 100, at most appearances |
| `league` | `GB1`, `ES1`, `IT1`, `L1`, `FR1` |
| `position_group` | `GK`, `DEF`, `MID`, `ATT` |

Minutes must also be no more than 130 per appearance, allowing extra time.
Bounds are input sanity and eligibility rules, not claims that every accepted
profile is well represented in training. Only height may be missing. Numeric
strings, booleans, fractional counts, NaN/Infinity and unknown fields are
rejected. Names, dates of birth, player IDs and listed values are not accepted
by the prediction endpoint.

The response includes `estimate_eur`, `currency`, `target`, `prediction_range`,
`as_of_date`, `profile_source`, `model`, `notes`, and `explanation`. The model context contains
an artifact-version identifier, `trained_through` and
`range_calibrated_through`. Prices are rounded to whole euros; range endpoints
round outward. The range includes `nominal_coverage` (currently 0.8), not an
individual player's probability or a guarantee. See the model card for its
measured coverage and substantial width.

## Find and estimate a player

```sh
curl -sS 'http://127.0.0.1:8000/players/search?q=messi&limit=5&offset=0'
```

Search is a literal, case- and accent-insensitive substring match. Names are
sorted by normalized name and then player ID, so pagination is stable. `q` must
be nonblank and at most 100 characters. `limit` defaults to 20 and accepts 1–100;
`offset` defaults to zero and accepts 0–100,000. The response contains the total
match count, page settings, lookup cutoff, and matching summaries. No matches
is a successful response with an empty list.

Use a returned ID in `GET /players/{player_id}/estimate`. The response includes
the player's name, exact historical profile, model prediction, historical
`listed_value_eur`, and `listing_minus_estimate_eur`. A positive difference means
the listing exceeded the estimate; it does not prove a player was overpriced.

The lookup has 6,725 eligible historical player profiles, each with its own
`as_of_date`, and a maximum allowed cutoff of 2025-06-30. Profiles may be much
older. League, position, age and totals describe that historical row. These are
not current squads or current listed values. No birth dates are returned.

Profiles dated at or before the model/range development cutoff carry a note
that this is a retrospective estimate, not an independent backtest. Later
profiles carry a note that the model does not adjust for subsequent market
changes. Historical API requests are demonstrations of the fitted model; they
are not new evaluation runs.

## Errors

All ordinary errors use one JSON shape:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Request validation failed.",
    "details": [
      {"field": "body.minutes_365", "message": "...", "type": "greater_than_equal"}
    ]
  }
}
```

- 422: malformed JSON, invalid fields, query settings or ID format.
- 429: per-IP rolling-minute request limit exceeded; retry after 60 seconds.
- 404: positive player ID absent from the historical lookup.
- 503: unavailable artifacts or a prediction failure. No invented estimate.
- 500: an unexpected server error with a generic public message.

The `/health` 503 response follows the health schema instead, with
`status: unavailable` and loading flags set to false.

## Verification

```sh
.venv/bin/python -m pytest -q
```

API tests use synthetic frozen artifacts and cover strict validation, search,
missing players, readiness failures, model-loading isolation, output parity,
OpenAPI, and concurrent predictions. They never reopen real test-set scoring.
The current Starlette version still supports the pinned HTTPX test client but
emits a deprecation notice recommending HTTPX2; the runtime server does not use
that testing dependency.

A running-service smoke check additionally exercises all four endpoints against
the actual frozen model/lookup and verifies their artifact hashes afterward.

## Explanations

Both prediction endpoints include `explanation` by default. For a player lookup,
it is nested under `prediction`. The object contains text, source (`fallback`
or `groq`), a fallback reason when relevant, and up to three model sensitivity
checks. These compare small changes to stated inputs; they are not causal
claims or forecasts. The estimate and prediction range remain unchanged.

Use `POST /predict?include_explanation=false` or
`GET /players/{player_id}/estimate?include_explanation=false` to skip all
what-if and language-service work. Health and search never call Groq. Missing
credentials, rejected provider text, provider errors, a busy service or an
exhausted daily provider cap all return HTTP 200 with a complete local fallback
for otherwise valid predictions. A per-IP request limit instead returns 429.

See [grounded explanations](explanations.md) for the exact privacy, grounding,
usage-limit and configuration contracts. Live Groq verification is deferred;
all automated tests and the current example review use mocks and local fallback.
