"""Shared helpers in the same location as MPCAttack/utils.py."""
import base64
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import time


def sample_seed(seed, image_path):
    """Stable image-ID seed: IDs0/1 retain V3's2026/2027 development seeds."""
    stem = Path(image_path).stem
    offset = int(stem) if stem.isdigit() else int(hashlib.sha256(stem.encode()).hexdigest()[:8], 16)
    return (seed + offset) % (2 ** 32)


@contextmanager
def execution_deadline(seconds):
    """Bound the complete CLI, including model loading/generation, on Linux/Molab."""
    import signal
    import threading
    if seconds <= 0:
        raise ValueError('Execution budget must be positive')
    supported = hasattr(signal, 'SIGALRM') and threading.current_thread() is threading.main_thread()
    if not supported:
        # Windows/programmatic callers retain the explicit per-step/call checks.
        yield
        return
    def expired(signum, frame):
        raise TimeoutError('Reviewed execution budget exhausted')
    previous_handler = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.time()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(.001, previous_timer[0] - (time.time() - started)), previous_timer[1])


VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def ensure_dir(path):
    Path(path or '.').mkdir(parents=True, exist_ok=True)


def atomic_write_json(path, value):
    path = Path(path)
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    temporary.replace(path)


def load_api_keys(path=None):
    candidates = [Path(path)] if path else [Path(f'api_keys.{extension}') for extension in ('yaml', 'yml', 'json')]
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open(encoding='utf-8') as handle:
                if candidate.suffix == '.json':
                    return json.load(handle)
                import yaml
                return yaml.safe_load(handle) or {}
    return {}


def get_api_key(model_name, path=None):
    keys = load_api_keys(path)
    key = keys.get(model_name) or os.environ.get(f'{model_name.upper().replace("-", "_")}_API_KEY')
    if not key and (model_name.startswith('gpt') or model_name.startswith('o')):
        key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError(f'Configure an API key for {model_name} in the environment or api_keys.yaml')
    return key


def encode_image(image_path):
    return base64.b64encode(Path(image_path).read_bytes()).decode('utf-8')


def hash_training_config(cfg):
    value = asdict(cfg) if is_dataclass(cfg) else vars(cfg)
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def get_output_paths(cfg, config_hash):
    return {'output_dir': str(Path(cfg.data.output) / 'img' / config_hash),
            'desc_output_dir': str(Path(cfg.data.output) / 'description' / config_hash)}


def read_descriptions(file_path):
    path = Path(file_path)
    if not path.exists():
        return []
    descriptions, seen = [], set()
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        if ':' not in line:
            raise ValueError(f'Invalid description row in {path}')
        filename, caption = [value.strip() for value in line.split(':', 1)]
        if not filename or not caption or filename in seen:
            raise ValueError(f'Empty or duplicate description ID in {path}: {filename}')
        seen.add(filename)
        descriptions.append((filename, caption))
    return descriptions


def save_descriptions(descriptions, file_path):
    path = Path(file_path)
    ensure_dir(path.parent)
    rows = list(descriptions)
    if len({name for name, _ in rows}) != len(rows):
        raise ValueError('Duplicate description IDs')
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(f'{name}: {" ".join(caption.split())}\n' for name, caption in rows), encoding='utf-8')
    temporary.replace(path)


def find_image(root, stem):
    candidates = [path for path in Path(root).rglob('*') if path.is_file()
                  and path.suffix.lower() in VALID_IMAGE_EXTENSIONS and path.stem == stem]
    if len(candidates) != 1:
        raise ValueError(f'Expected one image with ID {stem} under {root}; found {len(candidates)}')
    return candidates[0]


def require_experiment_approval(path):
    """CLI boundary: the supplied reviewed plan must explicitly record approval.

    This records scope provenance; a filename or credential does not itself
    grant approval. The operator must use a plan actually approved by the user.
    """
    if not path or not Path(path).is_file():
        raise ValueError('Supply --approved_plan pointing to the user-reviewed execution plan')
    text = Path(path).read_text(encoding='utf-8')
    if 'APPROVED' not in text or 'PENDING EXPLICIT USER REVIEW' in text or 'NOT APPROVED' in text:
        raise ValueError('The execution plan does not record explicit user approval')
    return {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(text.encode()).hexdigest()}


def wait_for_review(review_dir, key, evidence_path, deadline, complete=False):
    """Small file review gate shared by the MPC-shaped scripts, not an attack framework."""
    expected = hashlib.sha256(Path(evidence_path).read_bytes()).hexdigest()
    review_path = Path(review_dir) / f'{key}.json'
    print(f'AWAITING_REQUIRED_REVIEW {key}: {review_path}; evidence_sha256={expected}', flush=True)
    while not review_path.exists():
        if time.time() >= deadline:
            raise TimeoutError('Reviewed execution time limit reached while waiting for inspection')
        time.sleep(2)
    review = json.loads(review_path.read_text(encoding='utf-8'))
    if (review.get('status') != 'PASS' or review.get('evidence_sha256') != expected
            or not review.get('inspector') or not review.get('notes')):
        raise ValueError(f'Required review failed or is stale: {key}')
    if complete:
        for name in ('augmented_data', 'concepts', 'negatives', 'loss', 'crops', 'attack_outcome', 'adversarial_captions'):
            item = review.get('checklist', {}).get(name, {})
            if item.get('status') not in ('PASS', 'N/A') or not item.get('artifacts'):
                raise ValueError(f'Required first-sample check is incomplete: {name}')
            if item['status'] == 'N/A' and not item.get('justification'):
                raise ValueError(f'Unjustified N/A: {name}')
    return review


def info_nce_loss(feat_adv, feat_tgt, feat_src, temperature=0.2, mode='img', omega=2.0):
    """MPC negative CE; feat_src accepts either its original anchor or K negatives.

    Maximize the returned score, as in MPC's MI-FGSM loop. For image features,
    query/positive are [B,D], negatives are [K,D]. No negatives mix data pairs.
    """
    import torch
    import torch.nn.functional as F
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if mode != 'img':
        raise ValueError('My-Attack-V3 uses the original MPC image-query feature space')
    if feat_adv.ndim != 2 or feat_tgt.shape != feat_adv.shape:
        raise ValueError('Expected matching query/positive [B,D] tensors')
    if feat_src.ndim != 2 or not len(feat_src) or feat_src.shape[-1] != feat_adv.shape[-1]:
        raise ValueError('Expected negative anchors [K,D]')
    query = F.normalize(feat_adv, dim=-1)
    positive = F.normalize(feat_tgt.detach(), dim=-1)
    negatives = F.normalize(feat_src.detach(), dim=-1)
    sim_pos = (query * positive).sum(dim=-1, keepdim=True) * omega / temperature
    sim_neg = (query @ negatives.T) / temperature
    logits = torch.cat([sim_pos, sim_neg], dim=1)
    labels = torch.zeros(feat_adv.shape[0], dtype=torch.long, device=feat_adv.device)
    return -F.cross_entropy(logits, labels)


def save_image(tensor, path):
    from PIL import Image
    pixels = tensor.detach().cpu()[0].permute(1, 2, 0).clamp(0, 255).round().byte().numpy()
    ensure_dir(Path(path).parent)
    Image.fromarray(pixels).save(path)


def verify_perturbation(image_path, source, epsilon):
    import numpy as np
    import torch
    from PIL import Image
    pixels = torch.from_numpy(np.array(Image.open(image_path).convert('RGB'), copy=True)).permute(2, 0, 1)[None]
    linf = (pixels.float() - source.detach().cpu()).abs().max().item()
    if linf > epsilon:
        raise ValueError(f'Saved PNG bound {linf} exceeds epsilon {epsilon}')
    return {'saved_png_linf': linf, 'image_sha256': hashlib.sha256(Path(image_path).read_bytes()).hexdigest()}
