"""Deployment behavior without contacting Render, GitHub or Groq."""
import json
import logging
from pathlib import Path
import sys

import pytest
import yaml

from server import JsonFormatter, server_options

ROOT = Path(__file__).resolve().parents[1]


def test_platform_port_and_safe_worker_defaults(monkeypatch):
    monkeypatch.setenv('PORT', '10000')
    options = server_options()
    assert options['host'] == '0.0.0.0' and options['port'] == 10000
    assert options['workers'] == 1 and options['proxy_headers'] is False
    assert options['access_log'] is False
    monkeypatch.setenv('PORT', '70000')
    with pytest.raises(ValueError):
        server_options()


def test_json_logs_include_only_explicit_request_fields():
    record = logging.LogRecord('player_value.api', logging.INFO, 'api.py', 1, 'request_completed', (), None)
    record.method = 'GET'
    record.route = '/players/{player_id}/estimate'
    record.status = 200
    record.duration_ms = 12.5
    result = json.loads(JsonFormatter().format(record))
    assert result['route'] == '/players/{player_id}/estimate'
    assert result['status'] == 200 and result['duration_ms'] == 12.5
    assert set(result) == {'timestamp', 'level', 'logger', 'message', 'method', 'route', 'status', 'duration_ms'}
    try:
        raise ValueError('example')
    except ValueError:
        record.exc_info = sys.exc_info()
    assert 'ValueError: example' in json.loads(JsonFormatter().format(record))['exception']


def test_render_and_ci_contract():
    blueprint = yaml.safe_load((ROOT / 'render.yaml').read_text())
    service = blueprint['services'][0]
    assert service['runtime'] == 'docker' and service['plan'] == 'free'
    assert service['autoDeployTrigger'] == 'checksPass'
    assert service['healthCheckPath'] == '/health'
    env = {entry['key']: entry['value'] for entry in service['envVars']}
    assert env['PVP_LLM_ENABLED'] == 'false'
    assert 'GROQ_API_KEY' not in env
    # In the published monorepo the workflow is installed at the repository root.
    local = ROOT / '.github/workflows/player-value-ci.yml'
    workflow = local if local.exists() else ROOT.parent / '.github/workflows/player-value-ci.yml'
    ci = yaml.safe_load(workflow.read_text())
    assert ci['permissions'] == {'contents': 'read'}
    assert ci['jobs']['verify']['env']['PVP_LLM_ENABLED'] == 'false'
    steps = ci['jobs']['verify']['steps']
    commands = '\n'.join(step.get('run', '') for step in steps)
    assert 'python src/smoke.py' in commands
    assert '--memory=512m' in commands
    assert 'evaluate-test' not in commands
