"""One approved recovery: retain bank prefix, exact replay, then original N2 pipeline."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
import traceback

from PIL import Image
import torch

from models import Captioner
from negatives import NegativeBank, CAPTION_ATTEMPTS
from recovery_state import recover_caption_prefix, verify_replay, verify_source_identity
from run_pipeline import Pilot, CONFIG, PROJECT, seed_all, sha
from runtime import validate_review
from storage import RunStorage, write_json


class RecordedCaptioner(Captioner):
    """Add post-call RNG evidence without changing the original generation function."""
    def _generate(self, *args, **kwargs):
        text = super()._generate(*args, **kwargs)
        write_json(self.record_dir / f'rng_after_{self.calls:04d}.json', {
            'cpu': torch.get_rng_state().tolist(), 'cuda': torch.cuda.get_rng_state().cpu().tolist()})
        self.persist(f'rng_after_{self.calls:04d}')
        return text


class ContinuedNegativeBank(NegativeBank):
    def __init__(self, captioner, historical, seed=2026):
        super().__init__(captioner, seed)
        self.historical = historical

    def generate_captions(self, image):
        captions, attempts = recover_caption_prefix(self.historical)
        self.provenance['caption_attempts'] = attempts
        self.provenance['historical_physical_calls'] = len(self.historical)
        seen = {' '.join(text.casefold().split()) for text in captions}
        for attempt in range(len(attempts), CAPTION_ATTEMPTS):
            if len(captions) == self.k:
                return captions
            raw = self.captioner.caption(image, sample=True)
            caption = raw.strip()
            key = ' '.join(caption.casefold().split())
            accepted = bool(key) and key not in seen
            self.provenance['caption_attempts'].append({'attempt': attempt + 1, 'raw_output': raw, 'accepted': accepted})
            if accepted:
                captions.append(caption)
                seen.add(key)
        if len(captions) != self.k:
            raise RuntimeError(f'Only {len(captions)} distinct captions after approved cumulative80 attempts')
        return captions


class RecoveryPilot(Pilot):
    def __init__(self, root):
        self.root = Path(root)
        self.config = CONFIG
        clock = json.loads((self.root / 'recovery1/clock.json').read_text())
        self.started = clock['started']
        self.deadline = self.started + 75 * 60
        verify_source_identity(PROJECT, json.loads((self.root / 'source_manifest.json').read_text()))
        if json.loads((self.root / 'config.json').read_text()) != CONFIG:
            raise RuntimeError('Recovered scientific configuration differs')
        if json.loads((self.root / 'model_revisions.json').read_text()) != json.loads((PROJECT / 'model_revisions.json').read_text()):
            raise RuntimeError('Recovered model identity differs')
        runtime = json.loads((self.root / 'runtime.json').read_text())
        if (torch.__version__ != runtime['torch'] or torch.version.cuda != runtime['cuda']
                or torch.cuda.get_device_name(0) != runtime['gpu']):
            raise RuntimeError('Recovery torch/CUDA/GPU identity differs from v1')
        self.store = RunStorage(self.root, CONFIG['drive_folder_id'], CONFIG['run_id'])
        self.store.sentinel()
        pointer = json.loads(self.store.transport.get('status/latest.json'))
        if pointer['sequence'] != 59 or self.store.sequence != 59:
            raise RuntimeError('Remote run advanced or recovery already attempted; refusing a second launch')
        self.reviews_folder_id = self.store.transport._folder(self.store.transport.prefix + '/reviews', create=True)
        write_json(self.root / 'recovery1/source_manifest.json', {
            str(path.relative_to(PROJECT)): sha(path) for path in PROJECT.rglob('*')
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'})
        self.status('RECOVERY1_STARTED', reviews_folder_id=self.reviews_folder_id)
        self.commit('recovery1_identity')

    def status(self, state, **fields):
        super().status(state, continuation_id='recovery1', **fields)

    def gate(self, key, evidence):
        path = self.root / 'gates' / f'{key}_evidence.json'
        write_json(path, evidence)
        self.status('AWAITING_REQUIRED_REVIEW', gate=key, evidence_sha256=sha(path),
                    reviews_folder_id=self.reviews_folder_id)
        self.commit(f'gate_{key}')
        remote = f'reviews/{key}.json'
        while not self.store.transport.exists(remote):
            self.check_time()
            time.sleep(5)
        decision = json.loads(self.store.transport.get(remote))
        validate_review(key, decision, sha(path))
        write_json(self.root / 'reviews' / f'{key}.json', decision)
        self.commit(f'review_{key}')

    def make_bank(self, sample_id, seed):
        folder = self.root / sample_id
        if (folder / 'bank.json').exists():
            return json.loads((folder / 'bank.json').read_text())
        history = []
        if sample_id == '00000':
            history = [json.loads(path.read_text()) for path in sorted((folder / 'bank_generations').glob('attempt_*.json'))]
            if len(history) != 28:
                raise RuntimeError('Recovery1 requires the exact28-call prefix')
        elif (folder / 'bank_started.json').exists():
            raise RuntimeError('Sample1 already attempted; no automatic regeneration')
        seed_all(seed)
        marker = folder / 'recovery1_bank_started.json'
        if marker.exists():
            raise RuntimeError('Recovery bank already attempted; no second launch')
        write_json(marker, {'seed': seed, 'physical_call_cap': 191 if history else 190})
        self.commit(f'recovery1_{sample_id}_bank_attempt')
        generator = builder = None
        try:
            generator = RecordedCaptioner(CONFIG['bank_id'], folder / 'bank_generations', 191 if history else 190,
                                           persist=lambda stage: self.commit(f'{sample_id}_{stage}'))
            image = Image.open(folder / 'source.png').convert('RGB')
            if history:
                expected = history[-1]['result']
                torch.set_rng_state(torch.tensor(expected['cpu_rng_before'], dtype=torch.uint8))
                torch.cuda.set_rng_state(torch.tensor(expected['cuda_rng_before'], dtype=torch.uint8))
                key = 'recovery1_replay_call0028'
                generator._generate(expected['prompt'], image=image, sample=True, max_new_tokens=50, key=key)
                actual = generator.ledger.completed(key)
                write_json(self.root / 'recovery1/replay_check.json', {
                    'match': actual['token_ids'] == expected['token_ids'] and actual['text'] == expected['text'],
                    'expected_text': expected['text'], 'actual_text': actual['text'],
                    'expected_token_ids': expected['token_ids'], 'actual_token_ids': actual['token_ids'],
                    'physical_calls_after_replay': generator.calls})
                self.commit('recovery1_exact_replay_check')
                verify_replay(expected, actual)
                builder = ContinuedNegativeBank(generator, history, seed=seed)
            else:
                builder = NegativeBank(generator, seed=seed)
            bank = builder.create(image)
            bank.update(model=CONFIG['bank_id'], revision=generator.revision,
                        input_png_sha256=sha(folder / 'source.png'), continuation_id='recovery1')
            write_json(folder / 'bank.json', bank)
            return bank
        finally:
            write_json(folder / 'recovery1_bank_provenance_partial.json', builder.provenance if builder else {})
            if generator:
                generator.close()
            self.commit(f'recovery1_{sample_id}_bank_final')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    pilot = None
    try:
        def terminate(signum, frame):
            raise InterruptedError('Recovery watchdog requested durable shutdown')
        signal.signal(signal.SIGTERM, terminate)
        pilot = RecoveryPilot(args.root)
        pilot.run()
    except BaseException as error:
        write_json(args.root / 'recovery1/error.json', {'type': type(error).__name__, 'error': str(error),
                    'traceback': traceback.format_exc(), 'official_asr': 'PENDING'})
        if pilot:
            pilot.status('RECOVERY1_STOPPED_PARTIAL', error=f'{type(error).__name__}: {error}')
            try:
                pilot.commit('recovery1_stopped_partial')
            except Exception as sync_error:
                write_json(args.root / 'recovery1/drive_sync_pending.json', {'error': str(sync_error)})
        raise


if __name__ == '__main__':
    main()
