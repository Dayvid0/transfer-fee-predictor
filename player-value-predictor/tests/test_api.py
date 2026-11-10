"""HTTP contracts use synthetic frozen artifacts; no real test-set access."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
import math

from fastapi.testclient import TestClient
from llm import ExplanationEngine
import numpy as np
import pandas as pd
import pytest

import api
import train
from schemas import PredictionRequest
from test_train import RIDGE, sample_table


def make_test_app(*args, **kwargs):
    kwargs.setdefault('explanation_engine', ExplanationEngine(unavailable_reason='disabled'))
    return api.create_app(*args, **kwargs)


def request_body(**changes):
    body = {
        'as_of_date': '2025-06-15', 'age_years': 24, 'height_cm': 180,
        'minutes_365': 900, 'appearances_365': 12, 'goals_365': 10,
        'assists_365': 5, 'yellow_365': 2, 'red_365': 1,
        'league': 'GB1', 'position_group': 'ATT',
    }
    body.update(changes)
    return body


@pytest.fixture(scope='module')
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp('api-artifacts')
    table = root/'table.parquet'
    train.write_parquet(sample_table(), table)
    output = root/'artifacts'
    train.freeze_artifacts(table, output, specs=[RIDGE], progress=None)
    return output


@pytest.fixture(scope='module')
def service(frozen):
    bundle, metadata = train.load_bundle(frozen)
    with api.duckdb.connect(':memory:') as con:
        lookup = con.execute('SELECT * FROM read_parquet(?)', [str(frozen/'players.parquet')]).df()
    lookup['player_name'] = [f'Player {i}' for i in lookup.player_id]
    lookup.loc[lookup.player_id == 1, 'player_name'] = 'Ángel Test'
    lookup.loc[lookup.player_id == 2, 'player_name'] = 'Angel Test'
    lookup.loc[lookup.player_id == 3, 'player_name'] = 'Literal.* Name'
    return api.PredictionService(bundle, metadata, json.loads((frozen/'schema.json').read_text()), lookup, 'fixture-v1')


@pytest.fixture
def client(service):
    with TestClient(make_test_app(service_loader=lambda path: service)) as client:
        yield client


def test_health_loads_only_once_and_reports_metadata(service):
    calls = []
    def loader(path):
        calls.append(path)
        return service
    app = make_test_app('/example/artifacts', loader)
    with TestClient(app) as client:
        for _ in range(2):
            response = client.get('/health')
            assert response.status_code == 200
            assert response.json() == {
                'status': 'ok', 'model_loaded': True, 'lookup_loaded': True, 'players': 12,
                'model': {'version': 'fixture-v1', 'trained_through': '2024-12-31', 'range_calibrated_through': '2025-06-30'},
                'lookup_as_of': '2025-06-30',
            }
        assert client.post('/predict', json=request_body()).status_code == 200
    assert len(calls) == 1
    assert app.state.service is None


def test_prediction_matches_direct_frozen_model(client, service):
    body = request_body()
    response = client.post('/predict', json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    direct = train.predict_values(service.bundle['model'], pd.DataFrame([{k:v for k,v in body.items() if k!='as_of_date'}]))
    low, high = train.prediction_range(direct, service.bundle['radius'])
    assert result['estimate_eur'] == round(float(direct[0]))
    assert result['prediction_range'] == {'lower_eur': math.floor(low[0]), 'upper_eur': math.ceil(high[0]), 'nominal_coverage': .8}
    assert result['currency'] == 'EUR'
    assert result['profile_source'] == 'submitted_profile'
    assert result['as_of_date'] == body['as_of_date']
    assert any('not a backtest' in note for note in result['notes'])
    assert 'name' not in result and 'listed_value_eur' not in result


def test_missing_height_and_inclusive_minimum_minutes(client):
    body = request_body(minutes_365=450, appearances_365=4, goals_365=0, assists_365=0, red_365=0)
    body.pop('height_cm')
    missing = client.post('/predict', json=body)
    explicit_null = client.post('/predict', json={**body, 'height_cm': None})
    assert missing.status_code == explicit_null.status_code == 200
    assert missing.json() == explicit_null.json()


@pytest.mark.parametrize('field,value', [
    ('minutes_365', 449), ('minutes_365', 9001), ('minutes_365', 2000),
    ('minutes_365', '900'), ('minutes_365', True), ('appearances_365', 0),
    ('appearances_365', 101), ('appearances_365', 12.5), ('goals_365', -1),
    ('goals_365', 101), ('assists_365', None), ('red_365', 13), ('yellow_365', 25),
    ('age_years', 14), ('age_years', 47), ('age_years', True), ('height_cm', 1000),
    ('league', 'ZZ9'), ('position_group', 'striker'), ('as_of_date', '2099-01-01'),
    ('as_of_date', '2013-06-30'), ('as_of_date', '2025-02-30'),
    ('as_of_date', '2025-06-15T00:00:00'), ('as_of_date', 1750000000),
    ('name', 'must not enter predictor'), ('value_eur', 10), ('current_club_id', 1),
])
def test_invalid_profiles_return_safe_422(client, field, value):
    response = client.post('/predict', json=request_body(**{field:value}))
    assert response.status_code == 422, response.text
    error = response.json()['error']
    assert error['code'] == 'invalid_request'
    assert error['details']
    assert 'input' not in error['details'][0]  # no full request echo


@pytest.mark.parametrize('raw', ['{"age_years": NaN}', '{"age_years": Infinity}', '{broken', '[]', 'null'])
def test_nonfinite_and_malformed_json_is_safe(client, raw):
    response = client.post('/predict', content=raw, headers={'content-type': 'application/json'})
    assert response.status_code == 422, response.text
    assert response.json()['error']['code'] == 'invalid_request'


def test_case_accent_normalization_literal_matching_and_pagination(client):
    response = client.get('/players/search', params={'q':'  ANGEL  ', 'limit':1})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['total'] == 2 and len(body['players']) == 1
    assert body['players'][0]['player_id'] == 1
    assert body['profile_source'] == 'historical_lookup'
    assert body['lookup_as_of'] == '2025-06-30'
    next_page = client.get('/players/search', params={'q':'ángel', 'limit':1, 'offset':1}).json()
    assert next_page['players'][0]['player_id'] == 2
    literal = client.get('/players/search', params={'q':'.*'}).json()
    assert literal['total'] == 1 and literal['players'][0]['player_id'] == 3
    empty = client.get('/players/search', params={'q':'not found'}).json()
    assert empty['players'] == [] and empty['total'] == 0
    after_end = client.get('/players/search', params={'q':'angel', 'offset':99}).json()
    assert after_end['players'] == [] and after_end['total'] == 2


@pytest.mark.parametrize('params', [{}, {'q':''}, {'q':'   '}, {'q':'x'*101}, {'q':'p','limit':0},
                                    {'q':'p','limit':101}, {'q':'p','offset':-1}, {'q':'p','offset':100001}])
def test_invalid_search_requests(client, params):
    response = client.get('/players/search', params=params)
    assert response.status_code == 422
    assert response.json()['error']['code'] in ['invalid_request', 'invalid_query']


def test_player_estimate_is_historical_and_uses_same_post_prediction(client):
    response = client.get('/players/1/estimate')
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['player']['name'] == 'Ángel Test'
    assert body['profile']['height_cm'] is None
    assert body['prediction']['profile_source'] == 'historical_lookup'
    assert body['prediction']['as_of_date'] == body['player']['as_of_date'] == body['profile']['as_of_date']
    post = client.post('/predict', json=body['profile']).json()
    assert post['estimate_eur'] == body['prediction']['estimate_eur']
    assert post['prediction_range'] == body['prediction']['prediction_range']
    assert body['listing_minus_estimate_eur'] == body['listed_value_eur'] - post['estimate_eur']
    assert any('historical' in note for note in body['prediction']['notes'])
    assert 'date_of_birth' not in response.text and 'current_club' not in response.text


def test_unknown_and_invalid_player_ids(client):
    missing = client.get('/players/999/estimate')
    assert missing.status_code == 404
    assert missing.json()['error']['code'] == 'player_not_found'
    for player_id in ['0', '-1', 'abc', '2147483648']:
        response = client.get(f'/players/{player_id}/estimate')
        assert response.status_code == 422


def test_missing_or_corrupt_artifacts_yield_unavailable_without_path_leak(tmp_path):
    with TestClient(make_test_app(tmp_path/'missing')) as client:
        health = client.get('/health')
        assert health.status_code == 503
        assert health.json()['model_loaded'] is False
        for response in [client.post('/predict', json=request_body()),
                         client.get('/players/search?q=player'), client.get('/players/1/estimate')]:
            assert response.status_code == 503
            assert response.json()['error']['code'] == 'service_unavailable'
            assert str(tmp_path) not in response.text


def test_failed_predictions_have_safe_errors(service, monkeypatch):
    def broken(*args):
        raise RuntimeError('/private/file and details')
    monkeypatch.setattr(service, 'predict', broken)
    with TestClient(make_test_app(service_loader=lambda path: service)) as client:
        for response in [client.post('/predict', json=request_body()), client.get('/players/1/estimate')]:
            assert response.status_code == 503
            assert response.json()['error']['code'] == 'prediction_unavailable'
            assert '/private' not in response.text


def test_openapi_documents_routes_and_errors(client):
    doc = client.get('/openapi.json').json()
    assert set(doc['paths']) == {'/health','/predict','/players/search','/players/{player_id}/estimate'}
    assert doc['paths']['/predict']['post']['responses']['422']['content']['application/json']['schema']['$ref'].endswith('/ErrorResponse')
    assert client.get('/docs').status_code == 200
    assert client.get('/unknown').json()['error']['code'] == 'http_error'


def test_loading_real_format_does_not_read_training_data_or_fit(frozen, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Serving must never read training/test rows, fit, or evaluate')
    monkeypatch.setattr(train, 'fit_model', forbidden)
    monkeypatch.setattr(train, 'evaluate_test', forbidden)
    monkeypatch.setattr(train, 'load_development_table', forbidden)
    service = api.PredictionService.load(frozen)
    assert service.player_count == 12
    assert service.predict(PredictionRequest(**request_body())).estimate_eur >= 0


def test_predictions_remain_stable_under_concurrent_calls(service):
    profile = PredictionRequest(**request_body())
    expected = service.predict(profile).model_dump()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: service.predict(profile).model_dump(), range(12)))
    assert all(result == expected for result in results)
