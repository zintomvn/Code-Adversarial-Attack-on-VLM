"""Attack -> PNG -> three local captions -> local paid evaluator -> durable bundle.

The supervisor waits for evidence-bearing audit files at each approved barrier.
It never restarts an already attempted attack.
"""
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

from .config import RUN_ID, MODEL_REVISIONS
from .state import StageStore, atomic_json, fingerprint, require_gate
from .verification import pixel_hash, verify_png, summarize_scores


def create_bundle(root, name='results.zip'):
    root = Path(root)
    destination = root / name
    with zipfile.ZipFile(destination.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix in ('.tmp', '.lock') or (path.suffix == '.zip' and path.name != 'source.zip'):
                continue
            if any(part in {'venv311', '__pycache__', '.cache', 'model_cache'} for part in path.parts):
                continue
            archive.write(path, path.relative_to(root))
    destination.with_suffix('.tmp').replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    atomic_json(root / (name + '.sha256.json'), {'file': name, 'sha256': digest, 'bytes': destination.stat().st_size})
    return destination


class Supervisor:
    def __init__(self, root, package):
        self.root, self.package = Path(root), Path(package)
        self.root.mkdir(parents=True, exist_ok=True)
        approval = json.loads((self.package / 'approval.json').read_text(encoding='utf-8'))
        if approval.get('run') != RUN_ID or approval.get('status') != 'APPROVED':
            raise RuntimeError('This full run scope has not been approved')
        source_hashes = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted((Path(package) / 'src/cara_v641').glob('*.py'))}
        self.identity = {'run': RUN_ID, 'model_revisions': MODEL_REVISIONS, 'source_sha256': source_hashes,
                         'seed': 2026, 'n': 2, 'steps': 300, 'epsilon': 16, 'cached_banks': True,
                         'bank_sha256': hashlib.sha256((self.package / 'inputs/comparison_banks.json').read_bytes()).hexdigest(),
                         'approval_sha256': hashlib.sha256((self.package / 'approval.json').read_bytes()).hexdigest(),
                         'plan_sha256': hashlib.sha256((self.package / 'approved_plan.md').read_bytes()).hexdigest()}
        self.store = StageStore(self.root, self.identity)
        timing = self.store.get('timing')
        if timing is None:
            timing = {'started': time.time(), 'gpu_started': None}
            self.store.save('timing', timing)
        self.timing = timing
        self.deadline = timing['started'] + 3 * 3600
        self.identity_hash = self.store.identity

    def limits(self):
        if time.time() >= self.deadline:
            raise TimeoutError('Approved 3-hour wall-clock budget exhausted')
        gpu_start = self.timing.get('gpu_started')
        if gpu_start and time.time() - gpu_start >= 2 * 3600:
            raise TimeoutError('Approved 2 GPU-hour budget exhausted')
        stat = os.statvfs(self.root) if hasattr(os, 'statvfs') else None
        if stat and stat.f_bavail * stat.f_frsize < 2 * 2**30:
            raise RuntimeError('Less than 2GiB disk space remaining')

    def status(self, **fields):
        self.store.save('status', {'run': RUN_ID, 'updated': time.time(), **fields})

    def barrier(self, name):
        self.status(stage='audit_pending', gate=name)
        create_bundle(self.root, 'audit.zip')
        path = self.root / f'{name}.json'
        while not path.exists():
            candidate = self.root.parent / path.name
            if candidate.exists():
                import shutil
                shutil.copy2(candidate, path)
                break
            self.limits()
            time.sleep(2)
        require_gate(path, self.identity_hash)

    def prepare_data(self):
        import torch
        import kagglehub
        from .data import ImagePairDataset
        from .common import image_to_tensor
        dataset_root = kagglehub.dataset_download('zintom/blackbox-adversarial-attack-dataset/versions/1')
        dataset = ImagePairDataset(dataset_root)
        pairs = dataset.get_pairs(0, 2)
        banks = json.loads((self.package / 'inputs/comparison_banks.json').read_text(encoding='utf-8'))
        manifest = []
        for index, (source_path, target_path) in enumerate(pairs):
            folder = self.root / f'{index:05d}'
            folder.mkdir(exist_ok=True)
            entry = {'sample': index, 'seed': 2026 + index, 'dataset_version': 1, 'original_paths': {}}
            for role, path in [('source', source_path), ('target', target_path)]:
                image = dataset.load_image(path)
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                if digest != banks[str(index)]['pixels_sha256'][role]:
                    raise ValueError(f'Input pixel hash mismatch for {index}/{role}')
                image.save(folder / f'{role}.png')
                entry['original_paths'][role] = str(path)
                entry[role + '_pixels_sha256'] = digest
            atomic_json(folder / 'bank.json', banks[str(index)]['bank'])
            atomic_json(folder / 'bank_provenance.json', {
                'source': 'molab/cara_v641_pilot/comparison_banks.json',
                'note': 'Exact cached v6.3/v6.4/v6.4.1 bank. No new generation or selection.',
                'historical_generation_revision': 'not locked in old bank; unknown, do not fabricate',
                'input_hashes_verified': True})
            manifest.append(entry)
        self.store.save('data', {'manifest': manifest, 'transform': 'EXIF transpose/RGB/Resize224 bicubic/CenterCrop224',
                                'tensor_range': [0, 255], 'model_normalization': 'See models.py CLIP per-channel mean/std'})
        return dataset, pairs, banks

    def attack_one(self, index, dataset, paths, bank):
        import torch
        from . import models
        from .models import CLIPEnsemble, QwenVL
        from .attention import QwenAttentionXAI
        from .attack import CARAAttack
        from .common import image_to_tensor, set_seed, release_memory
        folder = self.root / f'{index:05d}'
        if (folder / 'attack_attempt.json').exists():
            if (folder / 'summary.json').exists() and (folder / 'adversarial.png').exists():
                verify_png(folder)
                return
            raise RuntimeError('Attack already attempted; preserve outputs, no automatic rerun')
        self.limits()
        atomic_json(folder / 'attack_attempt.json', {'started': time.time(), 'run_identity': self.identity_hash})
        models.CLIP_DEVICES = {name: 'cuda:0' for name in ('G14', 'B16', 'B32')}
        models.QWEN_MAX_MEMORY_GIB = {0: 80}
        encoder = CLIPEnsemble()
        guide = QwenVL('Qwen/Qwen3-VL-8B-Instruct', attention=True)
        explainer = QwenAttentionXAI(guide, 29)
        attack = CARAAttack(encoder, explainer, self.root)
        attack.check_limits = self.limits
        attack.first_step_gate = lambda sample_id: self.barrier(f'gate_step1_{sample_id:05d}')
        set_seed(2026 + index)
        source, target = (image_to_tensor(dataset.load_image(path)) for path in paths)
        start = time.time()
        self.status(stage='attack', sample=index)
        attack.attack(source, target, bank, index)
        atomic_json(folder / 'attack_complete.json', {'status': 'complete', 'seconds': time.time() - start,
                    'peak_vram_bytes': torch.cuda.max_memory_allocated(0), 'updates': 300})
        del encoder, guide, explainer, attack, source, target
        gc.collect()
        release_memory()

    def run(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Approved MoLab GPU required')
        if self.timing['gpu_started'] is None:
            self.timing['gpu_started'] = time.time()
            self.store.save('timing', self.timing)
        torch.set_num_threads(4)
        self.store.save('runtime', {'python': sys.version, 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(0), 'total_vram': torch.cuda.get_device_properties(0).total_memory,
            'environment': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()})
        dataset, pairs, banks = self.prepare_data()
        from .evaluation import local_captions
        for index in (0, 1):
            folder = self.root / f'{index:05d}'
            if (folder / 'sample_complete.json').exists():
                continue
            if index:
                require_gate(self.root / 'gate_complete_00000.json', self.identity_hash)
            self.barrier(f'gate_data_{index:05d}')
            self.attack_one(index, dataset, pairs[index], banks[str(index)]['bank'])
            atomic_json(folder / 'png_verification.json', verify_png(folder))
            self.status(stage='local_captions', sample=index)
            local_captions(folder, min(self.deadline, self.timing['gpu_started'] + 7200))
            self.status(stage='paid_evaluation_pending', sample=index)
            create_bundle(self.root, f'evaluation_ready_{index:05d}.zip')
            # The local collector calls GPT and returns its exact evaluation.json in this archive.
            transfer = self.root / f'paid_{index:05d}.zip'
            while not transfer.exists():
                candidate = self.root.parent / transfer.name
                if candidate.exists():
                    import shutil
                    shutil.copy2(candidate, transfer)
                    break
                self.limits()
                time.sleep(2)
            with zipfile.ZipFile(transfer) as archive:
                data = json.loads(archive.read('evaluation.json'))
            if data.get('run_identity') != self.identity_hash or data.get('sample') != index:
                raise ValueError('Paid-evaluation return identity mismatch')
            atomic_json(folder / 'evaluation.json', data)
            expected = {'LLaVA-1.5-7B', 'InternVL3-8B', 'Qwen3-VL-8B', 'GPT-5 mini'}
            if set(data.get('judgments', {})) != expected or any(v.get('score') is None for v in data['judgments'].values()):
                raise RuntimeError('Incomplete evaluation; sample gate cannot pass')
            for row in data['judgments'].values():
                summarize_scores([row['score']], planned=1)
            self.barrier(f'gate_complete_{index:05d}')
            atomic_json(folder / 'sample_complete.json', {'status': 'complete', 'finished': time.time()})
            create_bundle(self.root)
        self.status(stage='complete', completed=2)
        create_bundle(self.root)


def run_remote(root, package):
    supervisor = Supervisor(root, package)
    try:
        supervisor.run()
    except BaseException as error:
        supervisor.status(stage='failed', error_type=type(error).__name__, error=str(error)[:700],
                          traceback=traceback.format_exc())
        create_bundle(root)
        raise
