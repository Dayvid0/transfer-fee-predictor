"""Grounding, sensitivity math and safe API fallback contracts."""
import copy
import json

from fastapi.testclient import TestClient
import pandas as pd
import pytest

import api
import explain
from llm import DailyBudget, ExplanationEngine, RateLimiter
from schemas import PredictionRequest
from test_api import frozen, service, request_body


def valid_wording(facts, variant=1):
    return json.dumps({'sentences': [{'fact_id': fact['id'], 'text': fact['variants'][variant]}
                                   for fact in facts['facts']]}, ensure_ascii=False)


def test_what_if_math_and_bounded_profile_changes():
    profile = PredictionRequest(**request_body())
    original = profile.model_dump()
    def predictor(rows):
        return rows.age_years * 1000 + rows.goals_365 * 100 + rows.assists_365 * 10 + rows.minutes_365
    baseline = int(predictor(pd.DataFrame([original])).iloc[0])
    factors = explain.compute_factors(profile, baseline, predictor)
    assert [(x.factor, x.difference_eur) for x in factors] == [('age',1000),('goals',100),('playing_time',90)]
    assert all(f.direction == 'higher' for f in factors)
    assert all(f.estimate_eur == baseline + f.difference_eur for f in factors)
    assert profile.model_dump() == original
    boundary = PredictionRequest(**request_body(age_years=46, goals_365=100, assists_365=100,
                                                 appearances_365=100, minutes_365=9000))
    cases = explain.scenarios(boundary)
    assert [name for name, _, _ in cases] == ['age','goals','assists']
    assert cases[0][1].age_years == 45
    assert cases[1][1].goals_365 == 99
    assert cases[2][1].assists_365 == 99
    assert all(c.model_dump() != boundary.model_dump() for _, c, _ in cases)


def test_flat_and_negative_sensitivities_have_correct_directions():
    profile = PredictionRequest(**request_body())
    flat = explain.compute_factors(profile, 100, lambda rows: [100] * len(rows))
    assert all(f.direction == 'unchanged' and f.difference_eur == 0 for f in flat)
    decreasing = explain.compute_factors(profile, 100, lambda rows: [50] * len(rows))
    assert all(f.direction == 'lower' and f.difference_eur == -50 for f in decreasing)


def test_facts_include_exact_numbers_listing_relation_and_caveats(service):
    profile = PredictionRequest(**request_body())
    prediction = service.predict(profile, 'historical_lookup')
    factors = explain.compute_factors(profile, prediction.estimate_eur, lambda rows: [prediction.estimate_eur + 100] * len(rows))
    for delta, word in [(100,'above'),(-100,'below'),(0,'equal')]:
        facts = explain.make_facts(prediction, factors, prediction.estimate_eur + delta)
        text = explain.validate_wording(valid_wording(facts), facts)
        assert f'€{prediction.estimate_eur:,}' in text
        assert word in text
        assert 'not an independent backtest' in text
        assert 'historical' in text
        assert 'not causation' in text
        assert '80%' in text
        assert 'not a guarantee' in explain.fallback_explanation(facts, factors, 'not_configured').text
        for forbidden in ['Ángel', 'player_id', 'as_of_date', 'date_of_birth', '2025-06-15']:
            assert forbidden not in json.dumps(facts, ensure_ascii=False)


@pytest.mark.parametrize('mutation', ['number','reverse','claim','omitted','duplicate','unknown','schema','duplicate_key','invalid_json'])
def test_rejects_invented_numbers_unsupported_claims_and_missing_facts(service, mutation):
    profile = PredictionRequest(**request_body())
    prediction = service.predict(profile)
    facts = explain.make_facts(prediction, [], prediction.estimate_eur + 100)
    data = json.loads(valid_wording(facts))
    if mutation == 'number': data['sentences'][0]['text'] = data['sentences'][0]['text'].replace(f'€{prediction.estimate_eur:,}', '€999,999,999')
    elif mutation == 'reverse':
        sentence = next(x for x in data['sentences'] if x['fact_id']=='listing')
        sentence['text'] = sentence['text'].replace('above', 'below')
    elif mutation == 'claim': data['sentences'][0]['text'] += ' This proves the player is undervalued because of an injury.'
    elif mutation == 'omitted': data['sentences'].pop()
    elif mutation == 'duplicate': data['sentences'].append(data['sentences'][0])
    elif mutation == 'unknown': data['sentences'][0]['fact_id'] = 'injury'
    elif mutation == 'schema': data['comment'] = 'unexpected'
    content = json.dumps(data)
    if mutation == 'duplicate_key': content = '{"sentences": [], "sentences": []}'
    if mutation == 'invalid_json': content = 'not json'
    with pytest.raises(ValueError):
        explain.validate_wording(content, facts)


def test_api_default_fallback_and_optout_preserve_prediction(service):
    engine = ExplanationEngine(unavailable_reason='not_configured')
    with TestClient(api.create_app(service_loader=lambda _: service, explanation_engine=engine)) as client:
        result = client.get('/players/1/estimate').json()
        explanation = result['prediction']['explanation']
        assert explanation['source'] == 'fallback'
        assert explanation['fallback_reason'] == 'not_configured'
        assert len(explanation['factors']) == 3
        assert f"€{result['listed_value_eur']:,}" in explanation['text']
        assert 'Ángel' not in explanation['text']
        without = client.get('/players/1/estimate?include_explanation=false').json()
        assert without['prediction']['explanation'] is None
        result['prediction']['explanation'] = None
        assert result == without
        posted = client.post('/predict', json=request_body()).json()
        assert posted['explanation']['source'] == 'fallback'


def test_valid_mocked_groq_output_integrates_without_name_or_id(service, tmp_path):
    calls = []
    class Provider:
        def generate(self, facts):
            calls.append(facts)
            return valid_wording(facts)
        def close(self): pass
    engine = ExplanationEngine(Provider(), DailyBudget(tmp_path/'budget.sqlite3', 2))
    with TestClient(api.create_app(service_loader=lambda _: service, explanation_engine=engine)) as client:
        assert calls == []  # warm-up and readiness never spend Groq calls
        assert client.get('/health').status_code == 200
        assert client.get('/players/search?q=angel').status_code == 200
        response = client.get('/players/1/estimate').json()
        assert response['prediction']['explanation']['source'] == 'groq'
        assert response['prediction']['explanation']['fallback_reason'] is None
        assert len(calls) == 1
        encoded = json.dumps(calls[0], ensure_ascii=False)
        assert 'Ángel' not in encoded and 'player_id' not in encoded and '2025-06' not in encoded
        client.post('/predict?include_explanation=false', json=request_body())
        assert len(calls) == 1


@pytest.mark.parametrize('mode,reason', [('invalid','invalid_output'),('timeout','provider_error'),('cap','daily_limit')])
def test_provider_problem_keeps_numbers_available(service, tmp_path, mode, reason):
    class Provider:
        def generate(self, facts):
            if mode == 'timeout': raise TimeoutError('secret provider detail')
            return '{"sentences":[]}'
        def close(self): pass
    budget=DailyBudget(tmp_path/'budget.sqlite3',1)
    if mode=='cap': assert budget.reserve()
    engine=ExplanationEngine(Provider(),budget)
    with TestClient(api.create_app(service_loader=lambda _: service, explanation_engine=engine)) as client:
        response = client.post('/predict', json=request_body())
        assert response.status_code == 200
        body=response.json()
        assert body['explanation']['fallback_reason'] == reason
        assert body['estimate_eur'] == service.predict(PredictionRequest(**request_body())).estimate_eur
        assert 'secret' not in response.text


def test_what_if_failure_preserves_core_prediction(service, monkeypatch):
    def fail(*args, **kwargs): raise RuntimeError('private path')
    monkeypatch.setattr(api,'compute_factors',fail)
    with TestClient(api.create_app(service_loader=lambda _: service,
                                   explanation_engine=ExplanationEngine())) as client:
        response=client.post('/predict',json=request_body())
        assert response.status_code==200
        explanation=response.json()['explanation']
        assert explanation['fallback_reason']=='explanation_unavailable'
        assert explanation['factors']==[]
        assert 'private' not in response.text


def test_http_rate_limit_ignores_forwarded_headers_and_keeps_health_open(service):
    limiter=RateLimiter(limit=2)
    with TestClient(api.create_app(service_loader=lambda _:service, explanation_engine=ExplanationEngine(),
                                   rate_limiter=limiter)) as client:
        assert client.get('/players/search?q=angel').status_code==200
        assert client.post('/predict?include_explanation=false',json=request_body()).status_code==200
        denied=client.get('/players/1/estimate',headers={'X-Forwarded-For':'198.51.100.8'})
        assert denied.status_code==429
        assert denied.headers['retry-after']=='60'
        assert denied.json()['error']['code']=='rate_limited'
        assert client.get('/health').status_code==200
        assert client.get('/docs').status_code==200


def test_punctuation_after_amount_is_not_part_of_number():
    facts={'facts':[{'id':'listing','variants':[
        'The listing is €1,500,000, €222,828 above the estimate.',
        'The recorded listing of €1,500,000 is €222,828 above the estimate.',
    ]}]}
    for variant in (0,1):
        assert explain.validate_wording(valid_wording(facts,variant),facts)==facts['facts'][0]['variants'][variant]
