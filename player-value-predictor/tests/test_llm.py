"""Mocked Groq contract, persistent call cap and bounded rate-limit checks."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import json
from threading import Event

import httpx
import pytest

from llm import DailyBudget, ExplanationEngine, GroqClient, RateLimiter

FACTS = {'facts':[{'id':'estimate','variants':['The estimate is €1,000.', 'The model estimates €1,000.']}]}
VALID = json.dumps({'sentences':[{'fact_id':'estimate','text':'The model estimates €1,000.'}]})


def test_groq_request_contract_and_success(tmp_path):
    calls=[]
    def handler(request):
        calls.append(request)
        assert str(request.url)=='https://api.groq.com/openai/v1/chat/completions'
        assert request.headers['authorization']=='Bearer local-test-key'
        body=json.loads(request.content)
        assert body['model']=='test-model'
        assert body['temperature']==0 and body['max_tokens']==4096
        assert body['response_format']=={'type':'json_object'}
        assert json.loads(body['messages'][1]['content'])==FACTS
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':VALID}}]})
    provider=GroqClient('local-test-key','test-model',httpx.MockTransport(handler))
    engine=ExplanationEngine(provider,DailyBudget(tmp_path/'calls.sqlite3',1))
    assert engine.explain(FACTS,[]).source=='groq'
    assert engine.explain(FACTS,[]).fallback_reason=='daily_limit'
    assert len(calls)==1
    engine.close()


@pytest.mark.parametrize('case',['429','500','redirect','timeout','truncated','bad_json','oversized'])
def test_provider_failures_are_safe_without_retry(tmp_path,case):
    calls=[]
    def handler(request):
        calls.append(request)
        if case=='timeout': raise httpx.ReadTimeout('provider timed out',request=request)
        if case in ['429','500']: return httpx.Response(int(case),text='private details')
        if case=='redirect': return httpx.Response(302,headers={'location':'https://example.invalid/steal'})
        if case=='bad_json': return httpx.Response(200,text='not JSON')
        if case=='oversized': return httpx.Response(200,content=b'x'*128001)
        return httpx.Response(200,json={'choices':[{'finish_reason':'length','message':{'content':VALID}}]})
    provider=GroqClient('test-key',transport=httpx.MockTransport(handler))
    budget=DailyBudget(tmp_path/'calls.sqlite3',1)
    result=ExplanationEngine(provider,budget).explain(FACTS,[])
    assert result.source=='fallback' and result.fallback_reason=='provider_error'
    assert result.text=='The estimate is €1,000.'
    assert len(calls)==1
    assert not budget.reserve()  # failed provider attempts still count
    provider.close()


def test_daily_cap_survives_recreation_and_resets_on_utc_day(tmp_path):
    clock=[datetime(2026,11,7,23,59,tzinfo=timezone.utc)]
    path=tmp_path/'nested/calls.sqlite3'
    budget=DailyBudget(path,2,lambda:clock[0])
    assert budget.reserve()
    assert DailyBudget(path,2,lambda:clock[0]).reserve()
    assert not budget.reserve()
    clock[0]+=timedelta(minutes=2)
    assert budget.reserve()


def test_daily_cap_is_atomic_for_concurrent_workers(tmp_path):
    path=tmp_path/'calls.sqlite3'
    # Independent instances represent workers sharing the same local database.
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda _:DailyBudget(path,5).reserve(),range(20)))
    assert sum(results)==5


def test_budget_failure_does_not_call_provider(tmp_path):
    class Budget:
        def reserve(self): raise OSError('unwritable')
    class Provider:
        def generate(self,facts): pytest.fail('must not call without reservation')
    result=ExplanationEngine(Provider(),Budget()).explain(FACTS,[])
    assert result.fallback_reason=='budget_unavailable'


def test_concurrency_bound_returns_fallback_without_reserving_extra_budget(tmp_path):
    started, release=Event(),Event()
    class Provider:
        def generate(self,facts):
            started.set()
            assert release.wait(5)
            return VALID
    engine=ExplanationEngine(Provider(),DailyBudget(tmp_path/'calls.sqlite3',1),max_concurrent=1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(engine.explain,FACTS,[])
        assert started.wait(5)
        assert engine.explain(FACTS,[]).fallback_reason=='service_busy'
        release.set()
        assert future.result().source=='groq'


def test_missing_key_and_explicit_disable_never_create_budget(monkeypatch,tmp_path):
    monkeypatch.delenv('GROQ_API_KEY',raising=False)
    monkeypatch.setenv('PVP_LLM_ENABLED','true')
    monkeypatch.setenv('PVP_LLM_USAGE_DB',str(tmp_path/'calls.sqlite3'))
    assert ExplanationEngine.from_environment().explain(FACTS,[]).fallback_reason=='not_configured'
    monkeypatch.setenv('GROQ_API_KEY','not-a-real-key')
    monkeypatch.setenv('PVP_LLM_ENABLED','false')
    assert ExplanationEngine.from_environment().explain(FACTS,[]).fallback_reason=='disabled'
    assert not (tmp_path/'calls.sqlite3').exists()


def test_rate_limit_window_ip_isolation_and_memory_bound():
    clock=[0.0]
    rate=RateLimiter(limit=2,clock=lambda:clock[0],max_clients=2)
    assert rate.allow('a') and rate.allow('a')
    assert not rate.allow('a')
    assert rate.allow('b')
    assert not rate.allow('c')
    clock[0]=60
    assert rate.allow('a') and rate.allow('c')
    assert len(rate.entries)<=2
