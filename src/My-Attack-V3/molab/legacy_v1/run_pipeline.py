"""Approved N2 pilot: independent stage caches, manual evidence gates, no retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
import signal

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from attack import make_crop_schedule, run_attack, save_png, verify_png
from model_lock import MODEL_REVISIONS
from storage import RunStorage, write_json
from runtime import AttemptLedger, validate_review, pending_report

PROJECT = Path(__file__).resolve().parent
CONFIG = json.loads((PROJECT / 'config.json').read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Pilot:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = CONFIG
        clock = self.root / 'clock.json'
        if not clock.exists():
            write_json(clock, {'started': time.time()})
        self.started = json.loads(clock.read_text())['started']
        self.deadline = self.started + min(CONFIG['max_wall_seconds'], CONFIG['max_gpu_seconds'])
        self.store = RunStorage(self.root, CONFIG['drive_folder_id'], CONFIG['run_id'])

    def status(self, state, **fields):
        write_json(self.root / 'status.json', {'run_id': CONFIG['run_id'], 'state': state,
                   'elapsed_seconds': time.time() - self.started, 'official_asr': 'PENDING', **fields})
        pending_report(self.root, CONFIG['run_id'], state, fields)
        print(json.dumps({'state': state, **fields}, default=str), flush=True)

    def commit(self, stage):
        return self.store.commit(stage)

    def check_time(self):
        if time.time() >= self.deadline:
            raise TimeoutError('Reviewed cumulative compute budget reached')

    def gate(self, key, evidence):
        """Only a recorded inspection bound to current evidence can release a gate."""
        evidence_path = self.root / 'gates' / f'{key}_evidence.json'
        write_json(evidence_path, evidence)
        review = self.root / 'reviews' / f'{key}.json'
        self.status('AWAITING_REQUIRED_REVIEW', gate=key, evidence_sha256=sha(evidence_path))
        self.commit(f'gate_{key}')
        while not review.exists():
            self.check_time()
            time.sleep(2)
        decision = json.loads(review.read_text())
        validate_review(key, decision, sha(evidence_path))
        self.commit(f'review_{key}')

    def prepare(self):
        import kagglehub
        self.store.sentinel()
        write_json(self.root / 'config.json', CONFIG)
        write_json(self.root / 'model_revisions.json', MODEL_REVISIONS)
        files = {str(path.relative_to(PROJECT)): sha(path) for path in PROJECT.rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
        write_json(self.root / 'source_manifest.json', files)
        write_json(self.root / 'runtime.json', {'python': sys.version, 'platform': platform.platform(),
                   'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   'packages': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()})
        data = Path(kagglehub.dataset_download('zintom/blackbox-adversarial-attack-dataset/versions/1'))
        preprocess = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                          transforms.CenterCrop(224), transforms.Lambda(lambda image: image.convert('RGB'))])
        manifest = []
        for index, sample_id in enumerate(CONFIG['sample_ids']):
            folder = self.root / sample_id
            folder.mkdir(exist_ok=True)
            record = {'sample_id': sample_id, 'seed': CONFIG['seeds'][index], 'dataset_version': 1,
                      'labels': 'targeted source-target pair; no classification labels used',
                      'transform': 'MPC Resize224 bicubic / CenterCrop224 / RGB; pixel space [0,255]'}
            for role, relative in [('source', f'images/bigscale/nips17/{index}.png'),
                                   ('target', f'images/target_images/1/{index}.jpg')]:
                path = data / relative
                image = preprocess(Image.open(path))
                image.save(folder / f'{role}.png')
                record[role] = {'original_path': str(path), 'raw_sha256': sha(path),
                                'png_sha256': sha(folder / f'{role}.png'),
                                'pixels_sha256': hashlib.sha256(np.array(image).tobytes()).hexdigest()}
            write_json(folder / 'data.json', record)
            manifest.append(record)
        write_json(self.root / 'data_manifest.json', manifest)
        self.status('STORAGE_AND_DATA_PREPARED', drive_folder_id=self.store.folder_id)
        self.commit('prepared')

    def tensor(self, path):
        image = np.array(Image.open(path).convert('RGB'), dtype=np.uint8, copy=True)
        return torch.from_numpy(image).permute(2, 0, 1).float()[None].cuda()

    def make_bank(self, sample_id, seed):
        from models import Captioner
        from negatives import NegativeBank
        folder = self.root / sample_id
        if (folder / 'bank.json').exists():
            return json.loads((folder / 'bank.json').read_text())
        if (folder / 'bank_started.json').exists():
            raise RuntimeError('Bank generation already attempted; no automatic regeneration')
        write_json(folder / 'bank_started.json', {'seed': seed, 'max_calls': 190})
        seed_all(seed)
        self.commit(f'{sample_id}_bank_attempt_identity')
        generator = None
        bank_builder = None
        try:
            generator = Captioner(CONFIG['bank_id'], folder / 'bank_generations', 190,
                                  persist=lambda stage: self.commit(f'{sample_id}_bank_{stage}'))
            bank_builder = NegativeBank(generator, seed=seed)
            bank = bank_builder.create(Image.open(folder / 'source.png').convert('RGB'))
            bank['model'] = CONFIG['bank_id']
            bank['revision'] = MODEL_REVISIONS[CONFIG['bank_id']]
            bank['input_png_sha256'] = sha(folder / 'source.png')
            write_json(folder / 'bank.json', bank)
            return bank
        finally:
            write_json(folder / 'bank_provenance_partial.json', bank_builder.provenance if bank_builder else {})
            if generator:
                generator.close()
            self.commit(f'{sample_id}_bank')

    def attack_pair(self, sample_id, seed, bank):
        from models import MPCEncoder
        folder = self.root / sample_id
        if all((folder / arm / 'attack_complete.json').exists() for arm in CONFIG['arms']):
            return
        encoder = MPCEncoder()
        try:
            source, target = self.tensor(folder / 'source.png'), self.tensor(folder / 'target.png')
            descriptions_file = folder / 'mpc_descriptions.json'
            descriptions = json.loads(descriptions_file.read_text()) if descriptions_file.exists() else {}
            description_ledger = AttemptLedger(folder / 'mpc_caption_attempts', 2)
            for role, image in [('source', source), ('target', target)]:
                if role not in descriptions:
                    cached = description_ledger.completed(role)
                    if cached is None:
                        attempt = description_ledger.start(role, {'model': 'OpenGVLab/InternVL3-1B-hf'})
                        self.commit(f'{sample_id}_mpc_caption_attempt_{role}')
                        cached = {'text': encoder.internvl.generate_description(image)}
                        description_ledger.finish(attempt, cached)
                        self.commit(f'{sample_id}_mpc_caption_complete_{role}')
                    descriptions[role] = cached['text']
                    write_json(descriptions_file, descriptions)
                write_json(folder / 'mpc_caption_protocol.json', {
                    'model': 'OpenGVLab/InternVL3-1B-hf',
                    'revision': MODEL_REVISIONS['OpenGVLab/InternVL3-1B-hf'],
                    'prompt': 'Describe this image in one concise sentence, no longer than 20 words.',
                    'max_new_tokens': 50, 'decoding': 'original upstream model generation defaults'})
            with torch.no_grad():
                positive, baseline, variant = encoder.anchors(source, target, descriptions['source'], descriptions['target'], bank)
            torch.save({'positive': positive.cpu(), 'baseline': baseline.cpu(), 'negative10': variant.cpu()}, folder / 'anchors.pt')
            seed_all(seed)
            initial = 16 / 255 * torch.randn_like(source.cpu()).to(source.device)
            crops = make_crop_schedule(224, CONFIG['steps'], seed + 10000)
            write_json(folder / 'crop_schedule.json', crops)
            torch.save({'delta': initial.cpu(), 'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}, folder / 'initial_state.pt')
            for arm, negatives in [('mpc_original', baseline), ('mpc_negative10', variant)]:
                arm_dir = folder / arm
                if (arm_dir / 'attack_complete.json').exists():
                    continue
                self.status('ATTACK_RUNNING', sample_id=sample_id, arm=arm)

                def checkpoint(step, image, record, delta, momentum):
                    path = arm_dir / 'steps' / f'{step:03d}.png'
                    save_png(image, path)
                    write_json(path.with_suffix('.json'), record)
                    torch.save({'delta': delta.detach().cpu(), 'momentum': momentum.detach().cpu(),
                                'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                                'step': step, 'config': CONFIG}, path.with_suffix('.pt'))
                    self.commit(f'{sample_id}_{arm}_step{step:03d}')
                    print(f'{sample_id} {arm} checkpoint {step}', flush=True)

                history, verification = run_attack(
                    encoder, source, positive, negatives, initial, crops, arm_dir, checkpoint,
                    lambda record: self.gate(f'{sample_id}_{arm}_update1', record), self.deadline)
                write_json(arm_dir / 'history.json', history)
                write_json(arm_dir / 'attack_complete.json', {**verification, 'steps': 300,
                           'adversarial_sha256': sha(arm_dir / 'adversarial.png'), 'official_asr': 'PENDING'})
                self.commit(f'{sample_id}_{arm}_attack_complete')
        finally:
            encoder.close()

    def captions(self, sample_id):
        from models import Captioner
        folder = self.root / sample_id
        records_path = folder / 'captions.json'
        records = json.loads(records_path.read_text()) if records_path.exists() else {}
        captioner = Captioner(CONFIG['evaluation_id'], folder / 'evaluation_generations', 4, max_tokens=64,
                              persist=lambda stage: self.commit(f'{sample_id}_eval_{stage}'))
        try:
            images = [('source', folder / 'source.png'), ('target', folder / 'target.png')]
            images += [(arm, folder / arm / 'adversarial.png') for arm in CONFIG['arms']]
            for name, path in images:
                if name in records:
                    continue
                if name in CONFIG['arms']:
                    verify_png(path, self.tensor(folder / 'source.png'))
                records[name] = {'caption': captioner.caption(Image.open(path).convert('RGB'), key=name),
                                 'png_sha256': sha(path), 'model': CONFIG['evaluation_id'],
                                 'revision': MODEL_REVISIONS[CONFIG['evaluation_id']]}
                write_json(records_path, records)
                self.commit(f'{sample_id}_caption_{name}')
        finally:
            captioner.close()
        self.gate(f'{sample_id}_final_checklist', {'captions': records,
                  'required': ['augmented_data', 'concepts', 'negatives', 'loss', 'crops', 'attack_outcome', 'adversarial_captions'],
                  'official_asr': 'PENDING: pilot diagnostic only'})

    def run(self):
        self.store.sentinel()
        self.commit('process_entry_storage_verified')
        if not (self.root / 'data_manifest.json').exists():
            self.prepare()
        for sample_id, seed in zip(CONFIG['sample_ids'], CONFIG['seeds']):
            if (self.root / sample_id / 'sample_complete.json').exists():
                continue
            self.check_time()
            bank = self.make_bank(sample_id, seed)
            self.gate(f'{sample_id}_data_bank', {
                'data': json.loads((self.root / sample_id / 'data.json').read_text()),
                'bank_sha256': sha(self.root / sample_id / 'bank.json'), 'bank': bank})
            self.attack_pair(sample_id, seed, bank)
            self.captions(sample_id)
            write_json(self.root / sample_id / 'sample_complete.json', {'state': 'DIAGNOSTIC_COMPLETE', 'official_asr': 'PENDING'})
            self.commit(f'{sample_id}_complete')
        self.status('PILOT_DIAGNOSTIC_COMPLETE', completed_pairs=2, full_batch_enabled=False)
        self.commit('pilot_complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/marimo/MY-ATTACK-V3-20261102-N2-v1'))
    args = parser.parse_args()
    pilot = None
    try:
        def terminate(signum, frame):
            raise InterruptedError('Supervisor requested cooperative stop and durable partial export')
        signal.signal(signal.SIGTERM, terminate)
        pilot = Pilot(args.root)
        pilot.run()
    except BaseException as error:
        args.root.mkdir(parents=True, exist_ok=True)
        write_json(args.root / 'error.json', {'type': type(error).__name__, 'message': str(error),
                   'traceback': traceback.format_exc(), 'time': time.time(), 'official_asr': 'PENDING'})
        if pilot is not None:
            pilot.status('STOPPED_PARTIAL', error=f'{type(error).__name__}: {error}')
            try:
                pilot.commit('stopped_partial')
            except Exception as sync_error:
                write_json(args.root / 'drive_sync_pending.json', {'error': str(sync_error)})
        raise


if __name__ == '__main__':
    main()
