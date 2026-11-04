"""Saved-PNG and evaluator checks, independent of attack models."""
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image


def pixel_hash(path):
    with Image.open(path) as image:
        return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()


def verify_png(folder, epsilon=16):
    folder = Path(folder)
    images = []
    hashes = {}
    for role in ('source', 'adversarial'):
        path = folder / f'{role}.png'
        with Image.open(path) as handle:
            if handle.format != 'PNG' or handle.mode != 'RGB':
                raise ValueError('Expected saved RGB PNG')
            images.append(np.asarray(handle).astype(np.int16))
        hashes[role] = hashlib.sha256(path.read_bytes()).hexdigest()
    if images[0].shape != images[1].shape:
        raise ValueError('Saved image shapes differ')
    difference = int(np.abs(images[1] - images[0]).max())
    if difference > epsilon:
        raise ValueError(f'Saved PNG epsilon violated: {difference}')
    return {'status': 'PASS', 'linf_pixels': difference, 'shape': list(images[0].shape), 'sha256': hashes}


def validate_caption(text, status='completed', truncated=False):
    if status != 'completed' or truncated or not isinstance(text, str) or not text.strip():
        raise ValueError('Empty, truncated or incomplete caption')
    return text.strip()


def caption_cache_hit(row, key):
    if row.get('cache_key') != key:
        return False
    status = row.get('status')
    validate_caption(row.get('caption'), status='completed' if status == 'ok' else status,
                     truncated=row.get('truncated', False))
    return True


def summarize_scores(scores, planned=2):
    valid = [float(value) for value in scores if value is not None]
    if len(valid) > planned or any(not math.isfinite(value) or not 0 <= value <= 1 for value in valid):
        raise ValueError('Invalid judge scores')
    complete = len(valid) == planned
    primary = sum(value >= 0.5 for value in valid) / len(valid) if valid else None
    legacy = sum(value >= 0.3 for value in valid) / len(valid) if valid else None
    return {'planned': planned, 'judged': len(valid), 'asr_primary': primary if complete else None,
            'asr_legacy': legacy if complete else None, 'observed_asr_primary': primary,
            'avg_similarity': sum(valid) / len(valid) if valid else None, 'status': 'complete' if complete else 'pending'}
