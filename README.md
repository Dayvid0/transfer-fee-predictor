# Transfer fee predictor

Predicting what a football player costs when he moves, on a synthetic transfer
market, with a Streamlit dashboard over the results.

The interesting part isn't the leaderboard — it's the reasoning about what the
models are allowed to see. The strongest finding is negative: an early revision
of this project had a baseline that looked competitive, and that turned out to be
a leak rather than a finding.

## Why synthetic data

There is no public dataset of transfer fees that is both properly licensed and
free to download, so the primary pipeline generates its own. `src/make_data.py`
builds 5,000 transfers across 11 seasons (2015–2025) and seven European leagues
from a hidden pricing formula: an age curve peaking in the mid-twenties, output
and contract length, roughly 10% annual inflation, a league-strength multiplier,
and lognormal negotiation noise. About 4% of rows are loans or free transfers
with `fee_eur = 0` and get filtered out, because a loan carries no clearing price.

Everything is a deterministic function of one seed, so the CSV regenerates
byte-identically and any result below is reproducible.

That gives a specific advantage: the true formula is known, so the report can
state exactly how well any model *could* do. See "The noise floor" below.

A second stack runs on real valuations. It's kept deliberately separate, and the
reason why is also the most common modelling mistake in this domain — see
"Real player values" below.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py                              # writes reports/results.md
.venv/bin/streamlit run app.py --server.headless true
```

`main.py` takes about 1.1s on 2 CPU threads. No GPU, no deep learning.

## The split is chronological, never random

| block | seasons | rows | purpose |
|---|---|---|---|
| fit | 2015–2021 | 3,015 | training |
| early stopping | 2022 | 480 | tuning only |
| test | 2023–2025 | 1,305 | scored exactly once |

Random splits leak here in a way that's easy to miss. `market_value_eur` and
`fee_eur` are both functions of the same latent signal, and the season a transfer
happened in shifts both. A random split puts near-duplicates on both sides and
flatters every model.

`src/train.py` asserts the boundaries rather than trusting them, and
`tests/test_pipeline.py` checks that no `(player_id, season)` pair appears in
both the fitted rows and the test block.

## Results on the 2023–2025 test block

| model | MAE (log) | MAE (EUR) | MAPE | R² (log) |
|---|---|---|---|---|
| A. Market value (baseline) | 0.3816 | 2,441,858 | 42.1% | 0.545 |
| B. Linear regression | 0.2706 | 1,669,460 | 26.0% | 0.761 |
| C. LightGBM | **0.2695** | 1,724,281 | 27.6% | **0.763** |
| *oracle noise floor* | *0.2544* | *1,626,890* | — | — |

B and C are effectively tied (0.4% apart) and both land within ~6% of the floor.

## The noise floor

`oracle_noise_floor` is not a fourth model. It is the score an oracle would get
by predicting the generator's noise-free fee exactly — the best any model could
possibly do, because the negotiation shock is independent noise that no feature
can reveal.

A test asserts that **no model scores below it**. That's the test that would have
caught the leak, and it's the one I'd point a reviewer at first: a model under
the floor is reading the answer key, not learning the market.

The floor also says something useful about the task. At 0.2544 log-MAE, roughly
six points of error are irreducible. Any claim about squeezing the last few
points out of a transfer-fee model deserves suspicion.

## What the models actually learned

`market_value_real` takes 51% of LightGBM's total gain. This is a
**market-value correction problem**, not a discovery problem: the club's
valuation is a noisy observation of the same fundamentals, and combining it with
age, output and contract data removes ~29% of the baseline's log error.

An earlier revision reported a much stronger baseline. That was an artefact —
the generator built `market_value_eur` from the noisy log fee, so the valuation
quietly carried the fee's own noise. Both are now the clean signal plus
*independent* errors, and the baseline duly looks weaker, which is what one noisy
estimate should do. `tests/test_data.py` now enforces the independence.

Two other things the report is explicit about:

- **Deflation is load-bearing.** Fees are modelled in base-year euros, with the
  inflation index estimated from training seasons (1.0928 recovered against a
  true 1.1). Before this, ordinary regression trees collapsed at the 2021 season
  boundary and scored *worse* than the baseline.
- **Every model under-predicts** (bias −601,701 EUR for LightGBM) because the
  estimated index sits slightly under the true rate, and the shortfall compounds
  over the forecast horizon. Visible per-season in the report.

## Real player values

The second stack scores real valuations from a public Kaggle snapshot
(`dhruvpjariwala/football-players-transfer-value-prediction`, June 2024). It
carries a club's *asking price*, not a fee actually paid, and it is a single
snapshot — so **there is no time axis and a chronological split is not available**.

That's the point worth internalising. Given no date, the tempting move is a
random split, and a random split on this data is flattered by the same leak as
above: players from the same club, and players in the same team, share most of
their feature profile. The obvious fix — holding out whole clubs — still leaks,
because a player's value is largely set by the league they play in, and dropping
a club leaves its league in the training set. Both splits are therefore reported
side by side, and the gap between them is the finding.

2,570 players are actually modelled. The source has 3,001 rows, but 791 are
exact duplicate players repeated once per competition appearance, and club-less
players (a national team standing in for the club) carry a value of 0, which
makes percentage error undefined. Both are removed before fitting.

`football-data.org` supplies club crests, league tables and trophy cabinets. It
needs a token in `.env` (gitignored) and the app degrades to offline mode
without one. On the free plan's 10 requests/minute, the client reads the API's own
budget headers rather than sleeping blind.

## Dashboard

Seven pages over the synthetic market, four over real players.

| page | what it shows |
|---|---|
| Overview | pitch hero, scale of the market, how the split works |
| Model comparison | all three models against the oracle floor |
| Predicted vs actual | log-log scatter, residuals, coloured by season/league/position |
| Feature importance | LightGBM gain share |
| Try it yourself | sliders for a player profile, position diagram, fee gauge |
| Worst errors | the ten largest misses |
| Player lookup | one synthetic player's history and per-transfer predictions |

All artwork is generated locally: `src/visuals.py` builds pitch and
position-pitch SVG, initials avatars, player cards and a Plotly gauge as pure
functions. Nothing is fetched, no font is downloaded, no randomness, so output is
deterministic and testable.

Training-window transfers on the lookup page show their actual fee and an
explicit "not scored" note, rather than a number the models can't honestly
produce.

## Tests

```
.venv/bin/python -m pytest -q     # 78 passed
```

The suite is built around the failure modes rather than line coverage:

- no test-season row reaches a fit, a scaler or a tuning decision
- no model beats the oracle noise floor
- both models beat the market-value baseline
- market value and fee errors stay independent (the leak guard)
- time-split assertions actually catch a leaky split
- the dashboard reuses the real pipeline and every page renders
- every SVG is well-formed, deterministic, and draws colours only from `PALETTE`

## Assumptions

1. `fee_eur` is the only post-transfer quantity. Zero-fee rows are dropped rather
   than modelled.
2. Fee and market value are the same latent signal plus independent errors;
   correlated beyond that would leak. Enforced by a test.
3. Fee inflation is estimated from training seasons, not hard-coded.
4. LightGBM hyperparameters were chosen on the 2022 validation season only.
5. Real valuations are asking prices, not paid fees.

Full detail, including per-season MAE and the complete importance table, is in
[`reports/results.md`](reports/results.md).


## Historical player-value API

The separate [player-value-predictor](player-value-predictor/README.md) project
estimates Transfermarkt listed market values from historical player profiles
and match statistics. It includes a frozen model, prediction API, grounded
explanations, Docker packaging, CI, and a free Render Blueprint. Its chronological
evaluation and data licence are documented separately; its results do not alter
the synthetic transfer-fee experiment above.
