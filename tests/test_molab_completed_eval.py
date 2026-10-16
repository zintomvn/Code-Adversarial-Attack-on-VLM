"""CPU-only integrity tests for selective restoration and evaluation resume."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import threading
import time
from collections import deque
from types import SimpleNamespace
import unittest
import uuid
from zipfile import ZipFile

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('runner', REPO / 'notebooks/molab_eval_completed.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class LocalDrive:
    def __init__(self, base):
        self.base = base

    def exists(self, path):
        return (self.base / path).exists()

    def get_file(self, remote, local):
        shutil.copyfile(self.base / remote, local)

    def put_file(self, local, remote):
        shutil.copyfile(local, self.base / remote)

    def makedirs(self, remote, exist_ok=True):
        (self.base / remote).mkdir(parents=True, exist_ok=exist_ok)

    def open(self, path, mode):
        return (self.base / path).open(mode)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.root = REPO / 'tmp' / ('eval_qa_' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root)

    def cohort(self, base, count=200):
        records = []
        for i in range(count):
            sid = f'{i:05d}'
            directory = base / 'samples' / sid
            directory.mkdir(parents=True)
            record = dict(index=i, sample_id=sid, run_fingerprint='locked',
                          saved_linf_255=16, epsilon_255=16, image_sha256={})
            for role in ('source', 'target', 'adversarial'):
                p = directory / (role + '.png')
                p.write_bytes((sid + role).encode())
                record[role] = p.relative_to(base).as_posix()
                record['image_sha256'][role] = runner.sha(p)
            runner.write_json(directory / 'complete.json', record)
            records.append(record)
        (base / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        return records

    def test_full_cohort_and_corruption(self):
        self.cohort(self.root / 'input')
        self.assertEqual(len(runner.validate_manifest(self.root / 'input')), 200)
        (self.root / 'input/samples/00002/adversarial.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Missing/changed'):
            runner.validate_manifest(self.root / 'input')

    def test_partial_cohort_rejected(self):
        self.cohort(self.root / 'input', 13)
        with self.assertRaisesRegex(ValueError, 'full ordered cohort'):
            runner.validate_manifest(self.root / 'input')

    def test_completion_marker_must_match(self):
        self.cohort(self.root / 'input')
        p = self.root / 'input/samples/00000/complete.json'
        marker = runner.read_json(p)
        marker['run_fingerprint'] = 'other'
        runner.write_json(p, marker)
        with self.assertRaisesRegex(ValueError, 'Completion marker'):
            runner.validate_manifest(self.root / 'input')

    def test_path_traversal_rejected(self):
        for bad in ('../outside', '/absolute', 'samples/../../escape', 'C:/private', '..\\escape'):
            with self.assertRaises(ValueError):
                runner.contained(self.root, bad)
        archive = self.root / 'bad.zip'
        with ZipFile(archive, 'w') as z:
            z.writestr('../escape', 'malicious')
        with self.assertRaises(ValueError):
            runner.unpack_metadata(archive, self.root / 'restore')

    def test_selective_sample_restore(self):
        item = self.cohort(self.root / 'original', 1)[0]
        remote = self.root / 'remote'
        remote.mkdir()
        with ZipFile(remote / 'sample.zip', 'w') as z:
            for p in (self.root / 'original').rglob('*'):
                if p.is_file():
                    z.write(p, 'foa_reference/' + p.relative_to(self.root / 'original').as_posix())
            z.writestr('foa_reference/samples/00000/optimizer.pt', b'expensive-unused')
        runner.restore_sample(LocalDrive(remote), 'sample.zip', self.root / 'download.zip',
                              self.root / 'restored', item, 'foa_reference')
        self.assertFalse((self.root / 'restored/samples/00000/optimizer.pt').exists())
        self.assertEqual(runner.sha(self.root / 'restored' / item['target']), item['image_sha256']['target'])

    def test_reuse_only_identical_caption_protocol(self):
        run = self.root / 'runs/first'
        run.mkdir(parents=True)
        signature = dict(model='victim', caption={'revision': 'immutable', 'prompt': 'same'}, roles=['target', 'adversarial'], seed=2023)
        runner.write_json(run / 'run_config.json', {'caption_config': signature})
        (run / 'caption_items.jsonl').write_bytes(
            json.dumps({'sha256': 'image1', 'status': 'ok', 'caption': 'valid'}).encode() + b'\n' + b'{"partial":')
        state = {'tasks': {'one': str(run)}}
        self.assertEqual(len(runner.cached_items(self.root, state, signature, {'image1'})), 1)
        changed = {**signature, 'caption': {'revision': 'new', 'prompt': 'same'}}
        self.assertFalse(runner.cached_items(self.root, state, changed, {'image1'}))

    def test_immutable_backup_pointer_and_readback(self):
        fs = LocalDrive(self.root / 'drive')
        run = self.root / 'runs/one'
        run.mkdir(parents=True)
        (run / 'caption_items.jsonl').write_text('first\n', encoding='utf-8')
        state = {'tasks': {'one': str(run)}, 'model_revisions': {'model': 'revision'}, 'backup_prefix': 'backups'}
        runner.backup_task(self.root, state, 'one', fs)
        first = runner.read_json(self.root / 'drive/backups/one/pointer.json')
        (run / 'caption_items.jsonl').write_text('second\n', encoding='utf-8')
        runner.backup_task(self.root, state, 'one', fs)
        second = runner.read_json(self.root / 'drive/backups/one/pointer.json')
        self.assertNotEqual(first['archive'], second['archive'])
        self.assertTrue((self.root / 'drive/backups/one' / first['archive']).exists())
        self.assertEqual(runner.sha(self.root / 'drive/backups/one' / second['archive']), second['archive_sha256'])

    def test_wrong_remote_readback_fails(self):
        class CorruptDrive(LocalDrive):
            def open(self, path, mode):
                return io.BytesIO(b'corrupt')
        p = self.root / 'payload'
        p.write_bytes(b'original')
        with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
            runner.verified_upload(CorruptDrive(self.root / 'drive'), p, 'prefix/file')

    def guarded_limiter(self, deadline):
        limiter = SimpleNamespace(lock=threading.Lock(), events=deque(), rpm=12, tpm=8000,
                                  cooldown_until=0., next_start=0., requests_reserved=0)
        def cooldown(duration, **kwargs):
            limiter.cooldown_until = time.monotonic() + duration
        limiter.cooldown = cooldown
        ns = {'JUDGE_LIMITER': limiter, 'judge_fatal': lambda error: False}
        runner.install_quota_guard(ns, deadline, self.root / 'quota.json')
        return ns, limiter

    def test_long_quota_cooldown_pauses_without_new_requests(self):
        ns, limiter = self.guarded_limiter(time.monotonic() + 3600)
        limiter.acquire(500)
        limiter.cooldown(6912.5, count_429=True)
        with self.assertRaises(runner.QuotaPauseError):
            limiter.acquire(500)
        self.assertEqual(limiter.requests_reserved, 1)
        self.assertTrue(ns['judge_fatal'](runner.QuotaPauseError()))
        self.assertEqual(runner.read_json(self.root / 'quota.json')['retry_after_seconds'], 6912.5)

    def test_waiting_worker_is_woken_by_quota_pause(self):
        _, limiter = self.guarded_limiter(time.monotonic() + 3600)
        limiter.next_start = time.monotonic() + 60
        errors = []
        def acquire():
            try:
                limiter.acquire(500)
            except runner.QuotaPauseError:
                errors.append('paused')
        thread = threading.Thread(target=acquire)
        thread.start()
        limiter.cooldown(200, count_429=True)
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ['paused'])

    def test_deadline_blocks_new_judge_requests(self):
        _, limiter = self.guarded_limiter(time.monotonic() - 1)
        with self.assertRaises(runner.QuotaPauseError):
            limiter.acquire(500)
        self.assertEqual(limiter.requests_reserved, 0)


if __name__ == '__main__':
    unittest.main()
