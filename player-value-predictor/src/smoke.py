"""In-image smoke test. Uses packaged artifacts, never raw data or test scoring."""
import argparse
import json
import os
from pathlib import Path
import time

import duckdb
import httpx


def check(base_url, artifact_dir, wait_seconds=45):
    deadline = time.monotonic() + wait_seconds
    with httpx.Client(base_url=base_url, timeout=10, trust_env=False) as client:
        while True:
            try:
                health = client.get('/health')
                if health.status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError('Service failed readiness within smoke-test deadline')
            time.sleep(1)
        ready = health.json()
        assert ready['model_loaded'] and ready['lookup_loaded'] and ready['players'] > 0
        with duckdb.connect(':memory:') as con:
            player_id, name = con.execute(
                'SELECT player_id, player_name FROM read_parquet(?) WHERE player_name IS NOT NULL ORDER BY player_id LIMIT 1',
                [str(Path(artifact_dir) / 'players.parquet')],
            ).fetchone()
        search = client.get('/players/search', params={'q': name, 'limit': 100})
        assert search.status_code == 200
        assert any(p['player_id'] == player_id for p in search.json()['players'])
        historical = client.get(f'/players/{player_id}/estimate')
        assert historical.status_code == 200, historical.text
        result = historical.json()
        prediction = result['prediction']
        assert prediction['explanation']['source'] == 'fallback', 'Smoke tests require PVP_LLM_ENABLED=false'
        supplied = client.post('/predict', json=result['profile'])
        assert supplied.status_code == 200, supplied.text
        assert supplied.json()['estimate_eur'] == prediction['estimate_eur']
        assert supplied.json()['prediction_range'] == prediction['prediction_range']
        bounds = prediction['prediction_range']
        assert 0 <= bounds['lower_eur'] <= prediction['estimate_eur'] <= bounds['upper_eur']
        invalid = client.post('/predict', json={**result['profile'], 'minutes_365': 0})
        assert invalid.status_code == 422 and invalid.json()['error']['code'] == 'invalid_request'
        assert client.get('/players/2147483647/estimate').status_code == 404
        schema = client.get('/openapi.json')
        assert schema.status_code == 200 and len(schema.json()['paths']) == 4
        assert client.get('/docs').status_code == 200
        return {'status': 'passed', 'players': ready['players'], 'model_version': ready['model']['version'],
                'checks': ['readiness', 'search', 'historical_prediction', 'submitted_prediction',
                           'prediction_parity', 'fallback', 'validation', 'missing_player', 'docs']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=f"http://127.0.0.1:{os.environ.get('PORT', '8000')}")
    parser.add_argument('--artifacts', default=os.environ.get('PVP_ARTIFACT_DIR', '/app/artifacts/release'))
    args = parser.parse_args()
    print(json.dumps(check(args.url, args.artifacts)))
