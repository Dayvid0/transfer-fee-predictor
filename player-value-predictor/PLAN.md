# Player Value Predictor: approved blueprint (v1)

Rules of work: implement one phase at a time, summarise before each phase, report failures honestly, and ask before any major architecture change.

## Blueprint
1. **Problem.** Estimate a top-five-league player's Transfermarkt *listed* market value from their profile and stats from the 365 days before the valuation date, show where the listing differs from the estimate, and explain the estimate in plain English. Portfolio and learning project, $0 budget. We predict the listing, not a player's true worth.
2. **Users.** The author, a mentor and employers reviewing a portfolio; secondarily football fans. Anonymous, no accounts.
3. **MVP.** F1 to F9: data pipeline, training table, time-split model with baselines, prediction API (predict, search, per-player estimate), grounded explanations, input validation and safety, evaluations, Docker + CI + deployment. Not in the MVP: frontend, forecasting future changes, chat agent, text RAG, scraping, per-position models.
4. **Architecture.** Offline pipeline (DuckDB query, training table, model and small artifacts), then a FastAPI container that predicts, computes what-if factors, and asks Groq to word the result. No user text reaches the prompt.
5. **Stack.** DuckDB + pandas, scikit-learn (Ridge and HistGradientBoosting), FastAPI + pydantic, Groq (model name in config), Docker, Render free tier, GitHub Actions, pytest.
6. **Data.** Open Transfermarkt dataset, frozen at July 2026 (valuations end 2026-06-12). Features from the 365 days before each valuation. Split by valuation date: train to mid-2024, validate 2024/25, test 2025/26. Snapshot columns of `players` (current value, current club, contract expiry) are excluded as features. Artifacts: model, small lookup table, metrics, schema.
7. **API.** GET /health, POST /predict, GET /players/search, GET /players/{id}/estimate. If the LLM fails, the numbers still return.
8. **Structure.** Flat src/ (config, build_dataset, features, train, evaluate, explain, llm, schemas, api), tests/, artifacts/, scripts/, docs/. features.py is shared by training and serving.
9. **Phases (36 h).** 0 data check (3h); 1 training table (6h); 2 baselines and model (7h); 3 API (5h); 4 explanations (6h); 5 ship (5h); 6 buffer and review (4h).
10. **Testing.** Leakage fixtures; unit tests; API tests; LLM mocked in CI; Docker image smoke test; range calibration; manual live-LLM faithfulness eval.
11. **Security.** Secrets in env only; per-IP rate limit and a daily LLM-call cap; strict validation; player names and user text never in prompts; show age, not date of birth; check the dataset license before publishing any data.
12. **AI.** The LLM only words the result: a facts object with pre-formatted numbers, a number check, a deterministic fallback text, generous max_tokens. No RAG in the MVP.
13. **Deployment.** Same pinned versions in dev and Docker; slim image with src/ and artifacts/ only; Render free tier; JSON logs to stdout.
14. **CI/CD.** Lint, tests, image build, in-image smoke test; deploys wait for green checks.
15. **Docs.** README, PLAN.md, data findings note, model card, decision log.
16. **Risks.** License, data fields, weak signal, time overrun, leakage, free-tier limits, train/serve version mismatch.

## Cut order if time runs short
1. rate limit, 2. per-position breakdown, 3. size of the faithfulness eval, 4. search endpoint.

## Decision log
- The earlier prototype (match-predictor / transfer-model-test) stays separate: its synthetic results and game-rating data do not support claims about real player values.
- DuckDB is used for the 12-month window query because the dataset ships as a DuckDB file.
- No text RAG in the MVP: the facts are structured numbers.
- Dataset license: to be recorded in docs/data-findings.md during Phase 0.
