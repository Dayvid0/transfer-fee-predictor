"""Optional Groq wording, bounded requests, persistent daily budget and fallback.

Provider input contains only server-generated numerical facts and approved
wordings. No user text, name, player ID, date of birth or request headers enter it.
"""
from collections import OrderedDict, deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from threading import BoundedSemaphore, Lock
import time

import httpx

import config
from explain import fallback_explanation, validate_wording
from schemas import Explanation

GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'
DEFAULT_MODEL = 'openai/gpt-oss-120b'
SYSTEM_PROMPT = (
    'You are a constrained editor of a football valuation explanation. '
    'For each supplied fact select exactly one complete sentence from its variants. '
    'Copy it exactly, preserving every number, comparison and caveat. '
    'Do not add, omit, merge or rewrite facts. Return JSON only: '
    '{"sentences":[{"fact_id":"the supplied id","text":"one exact supplied variant"}]}. '
    'Include every fact exactly once. Prefer natural, concise wording. '
    'These are model estimates and sensitivities, never true worth or causal claims.'
)


def positive_env(name, default, maximum):
    value = int(os.environ.get(name, default))
    if not 1 <= value <= maximum:
        raise ValueError(f'{name} is outside its allowed range')
    return value


class DailyBudget:
    """SQLite reservations survive restarts and coordinate workers on one host.

    Every attempted provider call consumes a reservation, including failures.
    A transaction commits before any outbound request. Database failure denies
    the attempt rather than bypassing the cap. No request or player data stored.
    """
    def __init__(self, path, limit=100, utc_now=None):
        self.path = Path(path)
        self.limit = limit
        self.utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        if limit < 1:
            raise ValueError('Daily limit must be positive')

    def reserve(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        day = self.utc_now().astimezone(timezone.utc).date().isoformat()
        with sqlite3.connect(self.path, timeout=1) as con:
            con.execute('CREATE TABLE IF NOT EXISTS daily_calls (day TEXT PRIMARY KEY, calls INTEGER NOT NULL)')
            con.execute('BEGIN IMMEDIATE')
            row = con.execute('SELECT calls FROM daily_calls WHERE day = ?', (day,)).fetchone()
            if row is not None and row[0] >= self.limit:
                return False
            con.execute('INSERT INTO daily_calls VALUES (?, 1) ON CONFLICT(day) DO UPDATE SET calls = calls + 1', (day,))
        return True


class RateLimiter:
    """Bounded, per-peer-IP rolling-minute limit. Use one API worker by default."""
    def __init__(self, limit=60, clock=time.monotonic, max_clients=10000):
        self.limit, self.clock, self.max_clients = limit, clock, max_clients
        self.entries = OrderedDict()
        self.lock = Lock()

    def allow(self, address):
        now = self.clock()
        with self.lock:
            # Entries are ordered by most recent accepted access; expired IPs
            # are pruned incrementally so memory cannot grow without bound.
            while self.entries:
                key, times = next(iter(self.entries.items()))
                if times[-1] > now - 60:
                    break
                self.entries.pop(key)
            times = self.entries.get(address)
            if times is None:
                if len(self.entries) >= self.max_clients:
                    return False
                times = self.entries[address] = deque()
            while times and times[0] <= now - 60:
                times.popleft()
            if len(times) >= self.limit:
                return False
            times.append(now)
            self.entries.move_to_end(address)
            return True


class GroqClient:
    """Fixed HTTPS endpoint; no redirects, retries, environment proxy or prompt logs."""
    def __init__(self, api_key, model=DEFAULT_MODEL, transport=None):
        self.model = model
        self.client = httpx.Client(
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=httpx.Timeout(6.0, connect=2.0), follow_redirects=False, trust_env=False,
            transport=transport,
        )

    def generate(self, facts):
        body = {
            'model': self.model, 'temperature': 0, 'max_tokens': 4096,
            'response_format': {'type': 'json_object'},
            'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                         {'role': 'user', 'content': json.dumps(facts, ensure_ascii=False)}],
        }
        # Timeout is per HTTP operation. Limit response size as well as tokens.
        with self.client.stream('POST', GROQ_URL, json=body) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_bytes():
                content.extend(chunk)
                if len(content) > 128000:
                    raise ValueError('Provider response is too large')
        data = json.loads(content)
        choice = data['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('Incomplete provider response')
        return choice['message']['content']

    def close(self):
        self.client.close()


class ExplanationEngine:
    def __init__(self, provider=None, budget=None, unavailable_reason='not_configured', max_concurrent=2):
        self.provider = provider
        self.budget = budget
        self.unavailable_reason = unavailable_reason
        self.capacity = BoundedSemaphore(max_concurrent)

    @classmethod
    def from_environment(cls):
        enabled = os.environ.get('PVP_LLM_ENABLED', 'true').lower()
        if enabled not in ('true', 'false'):
            raise ValueError('PVP_LLM_ENABLED must be true or false')
        if enabled == 'false':
            return cls(unavailable_reason='disabled')
        key = os.environ.get('GROQ_API_KEY', '').strip()
        if not key:
            return cls()
        model = os.environ.get('GROQ_MODEL', DEFAULT_MODEL).strip()
        if not model:
            raise ValueError('GROQ_MODEL must not be empty')
        cap = positive_env('PVP_LLM_DAILY_CAP', 100, 10000)
        budget = DailyBudget(os.environ.get('PVP_LLM_USAGE_DB', config.ROOT / 'data' / 'llm_usage.sqlite3'), cap)
        return cls(GroqClient(key, model), budget)

    def explain(self, facts, factors):
        def fallback(reason):
            return fallback_explanation(facts, factors, reason)
        if self.provider is None:
            return fallback(self.unavailable_reason)
        if not self.capacity.acquire(blocking=False):
            return fallback('service_busy')
        try:
            try:
                allowed = self.budget is not None and self.budget.reserve()
            except Exception:
                return fallback('budget_unavailable')
            if not allowed:
                return fallback('daily_limit')
            try:
                response = self.provider.generate(facts)
            except Exception:
                return fallback('provider_error')
            try:
                text = validate_wording(response, facts)
            except (ValueError, TypeError, KeyError):
                return fallback('invalid_output')
            return Explanation(text=text, source='groq', factors=factors)
        finally:
            self.capacity.release()

    def close(self):
        if self.provider is not None:
            self.provider.close()
