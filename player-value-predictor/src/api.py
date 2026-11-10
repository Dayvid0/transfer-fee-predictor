"""Phase 4 FastAPI service for the frozen Phase 2b model.

Start from the project root: .venv/bin/uvicorn api:app --app-dir src
Only frozen artifacts are loaded. Optional Groq calls receive approved numerical
facts only. No training, test evaluation or raw database access takes place.
"""
from contextlib import asynccontextmanager
from datetime import date
import json
import logging
import math
import os
import time
from pathlib import Path
from threading import Lock
from typing import Annotated
import unicodedata

import duckdb
from fastapi import Depends, FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import pandas as pd
from starlette.exceptions import HTTPException as StarletteHTTPException

from explain import basic_fallback, compute_factors, make_facts
from llm import ExplanationEngine, RateLimiter, positive_env
from features import FEATURE_COLUMNS, INPUT_COLUMNS
from schemas import (
    ErrorResponse, HealthResponse, ModelContext, PlayerEstimateResponse,
    PlayerSummary, PredictionRequest, PredictionResponse, SearchResponse,
)
from train import ARTIFACT_DIR, digest, load_bundle, predict_values, prediction_range

LOGGER = logging.getLogger('player_value.api')


def normalize_name(text):
    decomposed = unicodedata.normalize('NFKD', text.casefold())
    return ' '.join(''.join(c for c in decomposed if not unicodedata.combining(c)).split())


class PredictionService:
    """Read-only lookup and one loaded pipeline per worker; serialized inference."""

    def __init__(self, bundle, metadata, schema, lookup, model_version):
        self.bundle = bundle
        self.model_context = ModelContext(
            version=model_version, trained_through=metadata['fit_end'],
            range_calibrated_through=metadata['calibration_end'],
        )
        self.lookup_as_of = date.fromisoformat(schema['lookup_as_of'])
        self._predict_lock = Lock()
        if schema['feature_columns'] != FEATURE_COLUMNS or schema['input_columns'] != INPUT_COLUMNS:
            raise ValueError('Artifact feature schema does not match serving features')
        if schema['range']['nominal_coverage'] != bundle['nominal_coverage'] or schema['range']['log_radius'] != bundle['radius']:
            raise ValueError('Artifact range schema does not match model')
        if self.lookup_as_of != self.model_context.range_calibrated_through:
            raise ValueError('Lookup date does not match development cutoff')
        required = {'player_id', 'player_name', 'valuation_date', 'value_eur', *INPUT_COLUMNS}
        if lookup.empty or not required.issubset(lookup.columns):
            raise ValueError('Player lookup is missing required data')
        if lookup.player_id.isna().any() or lookup.player_id.duplicated().any():
            raise ValueError('Player lookup requires unique player IDs')
        self._players = {}
        for row in lookup.to_dict(orient='records'):
            day = pd.Timestamp(row['valuation_date'])
            if pd.isna(day) or day != day.normalize() or day.date() > self.lookup_as_of:
                raise ValueError('Player lookup has an invalid historical date')
            raw_profile = {key: row[key] for key in INPUT_COLUMNS}
            if pd.isna(raw_profile['height_cm']):
                raw_profile['height_cm'] = None
            profile = PredictionRequest(as_of_date=day.date(), **raw_profile)
            name = None if pd.isna(row['player_name']) else str(row['player_name'])
            summary = PlayerSummary(
                player_id=row['player_id'], name=name, league=profile.league,
                position_group=profile.position_group, age_years=profile.age_years, as_of_date=profile.as_of_date,
            )
            listed = float(row['value_eur'])
            if not math.isfinite(listed) or listed <= 0 or not listed.is_integer():
                raise ValueError('Historical listed values must be positive whole euros')
            self._players[summary.player_id] = (summary, profile, int(listed))
        self._search = sorted(
            ((normalize_name(summary.name or ''), pid) for pid, (summary, _, _) in self._players.items()),
            key=lambda item: (item[0], item[1]),
        )
        # Readiness includes an actual prediction, not just successful deserialization.
        first = next(iter(self._players.values()))[1]
        self.predict(first, 'historical_lookup')

    @classmethod
    def load(cls, artifact_dir):
        artifact_dir = Path(artifact_dir)
        bundle, metadata = load_bundle(artifact_dir)
        schema = json.loads((artifact_dir / 'schema.json').read_text())
        with duckdb.connect(':memory:') as con:
            lookup = con.execute('SELECT * FROM read_parquet(?)', [str(artifact_dir / 'players.parquet')]).df()
        return cls(bundle, metadata, schema, lookup, digest(artifact_dir / 'selection.json')[:12])

    @property
    def player_count(self):
        return len(self._players)

    def predict(self, profile, source='submitted_profile'):
        rows = pd.DataFrame([profile.model_dump(exclude={'as_of_date'})])
        # threadpool_limits in the shared predictor affects the process, so two
        # worker threads must not enter/restore that context concurrently.
        with self._predict_lock:
            predicted = predict_values(self.bundle['model'], rows)
            lower, upper = prediction_range(predicted, self.bundle['radius'])
        notes = [
            'The range targets coverage across historical valuations; it is not a guarantee for this player.',
        ]
        if source == 'historical_lookup':
            notes.append('This is a saved historical profile, not current player data.')
        if profile.as_of_date <= self.model_context.range_calibrated_through:
            notes.append('This retrospective estimate uses a model or range developed with data at or after this profile date; it is not a backtest.')
        else:
            notes.append('The model and range are frozen at the dates shown and do not adjust for later market changes.')
        return PredictionResponse(
            estimate_eur=round(float(predicted[0])),
            prediction_range={
                'lower_eur': math.floor(float(lower[0])), 'upper_eur': math.ceil(float(upper[0])),
                'nominal_coverage': self.bundle['nominal_coverage'],
            },
            as_of_date=profile.as_of_date, profile_source=source,
            model=self.model_context, notes=notes,
        )

    def explain(self, profile, prediction, engine, listed_value=None):
        try:
            def predictor(rows):
                with self._predict_lock:
                    return predict_values(self.bundle['model'], rows)
            factors = compute_factors(profile, prediction.estimate_eur, predictor)
            facts = make_facts(prediction, factors, listed_value)
            explanation = engine.explain(facts, factors)
        except Exception:
            LOGGER.warning('Explanation unavailable; returning a numerical summary')
            explanation = basic_fallback(prediction)
        return prediction.model_copy(update={'explanation': explanation})

    def search(self, query, limit, offset):
        normalized = normalize_name(query)
        if not normalized:
            raise HTTPException(422, {'code': 'invalid_query', 'message': 'Search query must contain a visible character.'})
        matches = [pid for name, pid in self._search if normalized in name]
        return SearchResponse(
            query=query.strip(), total=len(matches), limit=limit, offset=offset,
            lookup_as_of=self.lookup_as_of,
            players=[self._players[pid][0] for pid in matches[offset:offset + limit]],
        )

    def estimate_player(self, player_id):
        if player_id not in self._players:
            raise HTTPException(404, {'code': 'player_not_found', 'message': 'Player not found in the historical lookup.'})
        summary, profile, listed = self._players[player_id]
        prediction = self.predict(profile, 'historical_lookup')
        return PlayerEstimateResponse(
            player=summary, profile=profile, prediction=prediction,
            listed_value_eur=listed, listing_minus_estimate_eur=listed - prediction.estimate_eur,
        )


def error_response(status, code, message, details=None, headers=None):
    body = ErrorResponse(error={'code': code, 'message': message, 'details': details or []})
    return JSONResponse(status_code=status, content=body.model_dump(mode='json'), headers=headers)


def create_app(artifact_dir=None, service_loader=None, explanation_engine=None, rate_limiter=None):
    artifact_dir = Path(artifact_dir or os.environ.get('PVP_ARTIFACT_DIR', ARTIFACT_DIR))
    loader = service_loader or PredictionService.load
    limiter = rate_limiter or RateLimiter(positive_env('PVP_RATE_LIMIT_PER_MINUTE', 60, 1000))

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = None
        try:
            app.state.service = loader(artifact_dir)
        except Exception:
            # Log diagnostics server-side; never send local paths or traces to clients.
            LOGGER.exception('Prediction artifacts could not be loaded')
        try:
            app.state.explanation_engine = explanation_engine or ExplanationEngine.from_environment()
        except Exception:
            LOGGER.warning('Explanation configuration unavailable; using local fallback')
            app.state.explanation_engine = ExplanationEngine(unavailable_reason='explanation_unavailable')
        try:
            yield
        finally:
            app.state.explanation_engine.close()
            app.state.service = None

    app = FastAPI(
        title='Player Value Predictor', version='0.4.0', lifespan=lifespan,
        description='Estimate Transfermarkt listed market values from profiles and prior-year match statistics. Player lookup data is historical.',
    )

    @app.middleware('http')
    async def enforce_rate_limit(request, call_next):
        protected = request.url.path == '/predict' or request.url.path.startswith('/players/')
        if protected:
            # Do not read forwarded headers here. Run Uvicorn with
            # --no-proxy-headers unless a trusted proxy is explicitly configured.
            address = request.client.host if request.client else 'unknown'
            if not limiter.allow(address):
                return error_response(429, 'rate_limited', 'Too many requests; try again in a minute.',
                                      headers={'Retry-After': '60'})
        started = time.monotonic()
        response = await call_next(request)
        route = request.scope.get('route')
        LOGGER.info('request_completed', extra={
            'method': request.method, 'route': getattr(route, 'path', 'unmatched'),
            'status': response.status_code,
            'duration_ms': round((time.monotonic() - started) * 1000, 2),
        })
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        details = [{'field': '.'.join(str(part) for part in e['loc']), 'message': e['msg'], 'type': e['type']}
                   for e in exc.errors()]
        return error_response(422, 'invalid_request', 'Request validation failed.', details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        detail = exc.detail if isinstance(exc.detail, dict) else {'code': 'http_error', 'message': str(exc.detail)}
        return error_response(exc.status_code, detail['code'], detail['message'], headers=exc.headers)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        LOGGER.exception('Unexpected API failure', exc_info=exc)
        return error_response(500, 'internal_error', 'The request could not be completed.')

    def ready_service(request: Request):
        service = getattr(request.app.state, 'service', None)
        if service is None:
            raise HTTPException(503, {'code': 'service_unavailable', 'message': 'Prediction service is unavailable. Check readiness and try again later.'})
        return service

    Service = Annotated[PredictionService, Depends(ready_service)]
    errors = {429: {'model': ErrorResponse}, 422: {'model': ErrorResponse}, 503: {'model': ErrorResponse}, 500: {'model': ErrorResponse}}

    @app.get('/health', response_model=HealthResponse, responses={503: {'model': HealthResponse}})
    def health(request: Request):
        service = getattr(request.app.state, 'service', None)
        if service is None:
            result = HealthResponse(status='unavailable', model_loaded=False, lookup_loaded=False, players=0)
            return JSONResponse(status_code=503, content=result.model_dump(mode='json'))
        return HealthResponse(status='ok', model_loaded=True, lookup_loaded=True, players=service.player_count,
                              model=service.model_context, lookup_as_of=service.lookup_as_of)

    @app.post('/predict', response_model=PredictionResponse, responses=errors)
    def predict(profile: PredictionRequest, service: Service, request: Request,
                include_explanation: Annotated[bool, Query(description='Include grounded text and model sensitivity checks.')] = True):
        try:
            prediction = service.predict(profile)
            if include_explanation:
                prediction = service.explain(profile, prediction, request.app.state.explanation_engine)
            return prediction
        except Exception as exc:
            LOGGER.exception('Prediction failed')
            raise HTTPException(503, {'code': 'prediction_unavailable', 'message': 'An estimate is temporarily unavailable.'}) from exc

    # Register static search before the player-ID route.
    @app.get('/players/search', response_model=SearchResponse, responses=errors)
    def search_players(
        service: Service,
        q: Annotated[str, Query(min_length=1, max_length=100)],
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        offset: Annotated[int, Query(ge=0, le=100000)] = 0,
    ):
        return service.search(q, limit, offset)

    @app.get('/players/{player_id}/estimate', response_model=PlayerEstimateResponse,
             responses={**errors, 404: {'model': ErrorResponse}})
    def player_estimate(service: Service, request: Request,
                        player_id: Annotated[int, PathParam(gt=0, le=2_147_483_647)],
                        include_explanation: Annotated[bool, Query(description='Include grounded text and model sensitivity checks.')] = True):
        try:
            result = service.estimate_player(player_id)
            if include_explanation:
                prediction = service.explain(result.profile, result.prediction,
                    request.app.state.explanation_engine, result.listed_value_eur)
                result = result.model_copy(update={'prediction': prediction})
            return result
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception('Historical player estimate failed')
            raise HTTPException(503, {'code': 'prediction_unavailable', 'message': 'An estimate is temporarily unavailable.'}) from exc

    return app


app = create_app()
