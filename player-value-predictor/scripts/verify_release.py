"""Check the committed release bundle without training or reading holdout rows."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from api import PredictionService
from train import digest


def verify(root):
    root = Path(root)
    manifest = json.loads((root / 'release.json').read_text())
    expected = set(manifest['files']) | {'release.json'}
    actual = {p.name for p in root.iterdir() if p.is_file()}
    if actual != expected:
        raise ValueError('Release bundle files differ from the manifest')
    for name, expected_hash in manifest['files'].items():
        if digest(root / name) != expected_hash:
            raise ValueError(f'Release checksum mismatch: {name}')
    if manifest['dataset_revision'] != 'e44f186d6f06dd8452aaf54c7921ba66c961f637':
        raise ValueError('Unexpected dataset provenance')
    service = PredictionService.load(root)
    return {'status': 'verified', 'players': service.player_count,
            'model_version': service.model_context.version, 'files': len(actual)}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1] / 'artifacts/release'
    print(json.dumps(verify(root)))
