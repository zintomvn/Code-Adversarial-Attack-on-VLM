"""Saved-PNG bounds and 300-step first-sample evidence verification."""
import hashlib
import json
import math


def verify(root):
    import numpy as np
    from PIL import Image
    samples = []
    for directory in sorted((root / 'results/m_attack_v2/samples').glob('*')):
        if not (directory / 'complete.json').is_file():
            continue
        source = np.array(Image.open(directory / 'source.png').convert('RGB')).astype('int16')
        adv = np.array(Image.open(directory / 'adversarial.png').convert('RGB')).astype('int16')
        if source.shape != (224, 224, 3) or adv.shape != source.shape:
            raise RuntimeError('Invalid saved image shape')
        delta = np.abs(adv - source)
        if delta.max() > 16:
            raise RuntimeError('Saved PNG epsilon exceeds 16')
        samples.append({'pair': directory.name, 'epsilon_pixel': int(delta.max()), 'mean_abs_perturbation_pixel': float(delta.mean()), 'output_sha256': hashlib.sha256((directory / 'adversarial.png').read_bytes()).hexdigest()})
    gate = json.loads((root / 'audit/first_step_gate.json').read_text())
    metrics = [json.loads(line) for line in (root / 'audit/metrics.jsonl').read_text().splitlines()]
    if len(samples) != 2 or gate['technical_checks'] != 'PASS' or [m['epoch'] for m in metrics] != list(range(300)):
        raise RuntimeError('Incomplete samples, first-step gate, or 300-step metrics')
    if not all((math.isfinite(m['similarity_per_image']) for m in metrics)):
        raise RuntimeError('Nonfinite similarity metrics')
    return {'samples': samples, 'first_step_gate': gate, 'steps': len(metrics), 'first_metrics': metrics[0], 'last_metrics': metrics[-1], 'captions': 'PENDING', 'asr': None, 'peak_vram': 'not_measured'}
