# Grounded explanations — Phase 4

Prediction endpoints now include an `explanation` object by default. Existing
estimate, range, profile and listing values are unchanged. Add
`?include_explanation=false` to either prediction endpoint for the previous
numbers-only behavior (`explanation: null`). Readiness and search never request
an explanation or consume Groq calls.

## Response

- `text`: plain-English estimate, range, applicable listing comparison, local
  model sensitivities and limitations.
- `source`: `fallback` for deterministic local text or `groq` for accepted
  provider-selected wording.
- `fallback_reason`: why local text was used; null for accepted Groq output.
- `factors`: up to three local sensitivity comparisons. Each has `factor`,
  `changes` with explicit before/after values, alternate `estimate_eur`,
  `difference_eur` (alternate minus original), and `direction`.

The fallback is a complete explanation, not an error. Missing credentials,
provider failures, rate/cost limits and invalid wording never replace the
numerical estimate with an invented value or block a valid prediction. A
what-if calculation failure returns a basic summary without factors.

## What the comparisons mean

Four declared local probes are evaluated in one batch using the same frozen
model and feature transform:

1. One year older (one year younger if the older profile would exceed age 46).
2. One additional goal (one fewer at the upper input bound).
3. One additional assist (one fewer at the upper input bound).
4. 90 additional minutes and one additional appearance, only if both stay within
   the API bounds. All goal, assist and card totals remain fixed, so per-90 rates
   change when minutes change.

Return the three largest absolute changes, with deterministic ties. These are
local model sensitivities with stated inputs varied. They are not causes,
forecasts, advice about improving a player's worth, additive feature importance,
or a complete explanation of why the model produced the estimate. No training
medians or test labels are introduced for these comparisons. The target remains
Transfermarkt's listing, not true worth or a transfer fee.

The explanation retains the range caveat and historical/retrospective or frozen
model caveat. Per-player explanations compare the original historical listing
with the estimate; a difference does not establish mispricing.

## Provider grounding and privacy

The facts object is built from typed numeric predictions and internally selected
scenario labels. It contains pre-formatted euro amounts, coverage, comparisons
and caveats. Player names, IDs, birth dates, profile dates, search queries,
headers and user-written text never enter the prompt. The model/version metadata
and user-facing notes are not copied into the prompt.

Groq selects one of two approved sentence phrasings for each fact. It must
return every fact exactly once in JSON. Validation checks numbers per fact and
exact approved wording, then assembles sentences in canonical order. This
rejects changed numbers, reversed relations, omissions, added injury/contract
claims, causal assertions, duplicate keys and unknown facts. Unconstrained
paraphrases are intentionally rejected: a number check alone cannot prove
faithfulness. The provider is a constrained wording layer, not an analyst.

The fixed HTTPS endpoint is Groq's `/openai/v1/chat/completions`. No redirects,
automatic retries, environment proxies or prompt/credential logging are used.
Requests use temperature zero, JSON output and 4,096 max output tokens. HTTP
operation timeout is six seconds (connect timeout two seconds), response size
is capped, and at most two provider calls run concurrently per worker. Busy
requests use fallback immediately. Provider availability and response format
still require a live check with real credentials.

## Configuration

The app reads environment variables; it does not automatically load `.env`.
Without a key, the default is complete local fallback explanations. To force
local-only operation even if a key exists, set `PVP_LLM_ENABLED=false`.

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | unset | Optional key; never put it in code or a request |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Provider model, independent of the value-prediction model |
| `PVP_LLM_ENABLED` | `true` | `false` forces local fallback |
| `PVP_LLM_DAILY_CAP` | `100` | Attempted provider calls per UTC day, maximum 10,000 |
| `PVP_LLM_USAGE_DB` | `data/llm_usage.sqlite3` | Local persistent daily reservations; no player/request data |
| `PVP_RATE_LIMIT_PER_MINUTE` | `60` | Requests per peer IP per rolling minute, maximum 1,000 |

Every provider attempt, including errors, consumes a reservation before the
outbound call. SQLite transactions make the daily cap atomic across workers
sharing this file and preserve it across restarts. If the file is unwritable,
use fallback rather than bypass the cap. Separate machines do not share a cap;
a persistent volume is needed on an ephemeral host. This is a call cap, not a
currency budget or provider pricing guarantee.

The request limiter covers prediction and search, including invalid requests.
It is bounded in memory, per worker, and returns 429 with `Retry-After: 60`.
Health and documentation remain accessible. Use **one worker** with
`--no-proxy-headers` locally; the app uses the actual peer IP and does not inspect
forwarded headers. Trusted proxy configuration and shared multi-worker limits
belong to deployment, not arbitrary client headers. Request-limit state resets
on restart; the daily provider cap does not.

## Verification and remaining live check

Unit/API tests mock all Groq responses, even if the machine has a real key.
They cover exact wording and number acceptance, invented facts and reversed
comparisons, missing/duplicate facts, unavailable provider, quota exhaustion,
concurrent budget reservations, restart/day rollover, prediction parity, safe
fallbacks and IP rate limiting.

Run the reproducible mocked review with:

```sh
.venv/bin/python scripts/check_explanations.py --output /tmp/explanation-review.json
```

It uses six synthetic profiles with the saved value model, checks both approved
wordings, rejects four deliberately unfaithful responses per profile, and saves
examples for review. It does not load training/test rows or call Groq.

A live Groq faithfulness review was explicitly deferred by the user. The
integration is implemented and mocked; live model availability, output acceptance
rate, response latency and a human review of real Groq responses remain untested.
When credentials are configured later, review a small set of examples, verify
`source: groq`, and record fallback rates before relying on live wording.
This explanation layer does not improve or recalibrate numerical accuracy.
