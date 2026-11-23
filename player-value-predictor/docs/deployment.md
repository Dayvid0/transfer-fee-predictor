# Packaging and deployment — Phase 5

## Layout and repository

The local development project is `/mnt/data/workspace/player-value-predictor`.
The GitHub destination is `Dayvid0/transfer-fee-predictor`, which already contains
an earlier synthetic/Streamlit application. This project is added there under
`player-value-predictor/`; the earlier application stays intact. The workflow is
placed at the GitHub repository root under `.github/workflows/` so Actions can
run it. The nested API keeps its own requirements, tests, artifacts and README.

## Run locally with Docker

From the local project root (or the nested project folder after cloning):

```sh
docker build -t player-value-predictor:local .
docker run --rm --name player-value-api -p 8000:8000 \
  --read-only --tmpfs /tmp:rw,nosuid,size=64m \
  --memory=512m --cpus=1 \
  -e PVP_LLM_ENABLED=false player-value-predictor:local
```

Open <http://127.0.0.1:8000/docs>. From another terminal, run the in-image check:

```sh
docker exec player-value-api python src/smoke.py
```

The image uses a slim Python 3.12 runtime, the exact pinned runtime requirements,
`libgomp1`, non-root UID 10001, and only source, release artifacts, dependency
manifest and data licence. `.dockerignore` excludes `.env`, Git metadata, raw
DuckDB data, training tables, tests and local environments. The health check
requests `/health`. `PORT` defaults to 8000 and is honored when the host overrides
it. Runtime code uses one worker and two native math threads at most.

Startup/runtime application logs are JSON on stdout. Request logs include only
method, route template, status and elapsed time. They do not log request bodies,
query strings, player names, IDs, IPs, prompts or credentials. Uvicorn access logs
are off. Internal exception traces stay in server logs, not HTTP responses.

## Reproducible release and CI

`artifacts/release/` is a copy of the original frozen Phase 2b bundle, with its
existing metrics and added provenance/checksums. No refitting or new test scoring
occurs in packaging or CI. Run `python scripts/verify_release.py` to verify file
checksums, original model/source/version checks and readiness. The checked-in
bundle is small (about 1 MB). Raw data remains ignored by Git.

`Player value CI` runs on pull requests and pushes to `main` touching the API or
its workflow. It installs pinned dependencies, checks dependency consistency,
runs correctness lint, runs unit/API tests with all provider calls mocked,
verifies release files, builds the image, and runs the API smoke test **inside**
the image under a 512 MB memory limit and one CPU. Failed checks fail the job;
logs are printed and the temporary container is removed even on failure.

Correctness lint targets syntax errors, undefined names and invalid control
flow. Existing frozen training files are not rewritten for stylistic rules.
Changing those files intentionally requires a separately reviewed model-release
process; never alter frozen hashes merely to bypass compatibility checks.

## Create the new free Render service

After the review branch is merged and its checks pass on `main`:

1. Sign in to <https://dashboard.render.com/> and choose **New → Blueprint**.
2. Connect `Dayvid0/transfer-fee-predictor` and select the `main` branch.
3. Set **Blueprint Path** to `player-value-predictor/render.yaml`.
4. Review `player-value-predictor-api`, runtime Docker, **Free** plan, and create
   the service. No Groq key or paid disk is required.
5. Wait for the build and `/health` readiness check; open the supplied service
   URL at `/docs` and try player `449151`.

The Blueprint sets `autoDeployTrigger: checksPass` so subsequent automatic
builds wait for the linked branch's CI checks. It uses the nested Dockerfile and
build context. Do not deploy a review branch before its checks pass. Initial
service creation, account linking, and the actual public health check require
Render account access; they are not completed by writing this file.

## Free-service limits and explanations

`PVP_LLM_ENABLED=false` is intentional. Render Free has an ephemeral filesystem
and does not support persistent disks, so a SQLite daily Groq-call counter would
reset on restart/spindown. The free deployment serves the tested, grounded local
explanations with no provider cost or key. Optional live Groq remains deferred
as requested. To enable it later, use a persistent usage store and a reviewed
budget, then complete the live-provider faithfulness check.

Render Free may sleep after inactivity and cold-start on the next request.
Shared free-instance hours and other provider limits still apply. See
[Render Free](https://render.com/docs/free) for current limits.

The server does not blindly trust forwarded IP headers. With an unconfigured
reverse proxy, requests from its peer share a conservative 60-per-minute bucket.
That avoids spoofing but may rate-limit multiple users together. Configure
verified trusted-proxy ranges/shared rate limiting before heavier public usage;
never set unrestricted proxy trust just to bypass the limiter. Health/docs are
exempt. No authentication/accounts are part of this portfolio MVP.

## Rollback

Keep the last successful commit and image. If a release fails, restore that
commit through a reviewed revert, wait for its checks, and redeploy it. Do not
retrain or change prediction ranges during rollback. Verify `/health`, the model
version, and a saved request against the release before sharing its URL.

References checked 7 November 2026:
[Blueprint specification](https://render.com/docs/blueprint-spec),
[Free instances](https://render.com/docs/free),
[dataset provenance](data-findings.md).
