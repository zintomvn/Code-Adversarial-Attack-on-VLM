import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import storage


class FakeTransport:
    folder_id = 'actual-child-id'

    def __init__(self):
        self.files = {}
        self.calls = []
        self.corrupt = None
        self.fail = False

    def exists(self, name):
        self.calls.append(('exists', name))
        return name in self.files

    def put(self, name, data):
        self.calls.append(('put', name))
        if self.fail:
            raise OSError('synthetic upload failure')
        self.files[name] = data

    def get(self, name):
        self.calls.append(('get', name))
        return b'corrupted' if self.corrupt and self.corrupt in name else self.files[name]


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.transport = FakeTransport()
        self.store = storage.RunStorage(self.root, 'parent', 'run', transport=self.transport)

    def test_commit_verified_snapshot_and_reviews(self):
        (self.root / 'output.txt').write_text('result')
        (self.root / 'review.json').write_text('{}')
        receipt = self.store.commit('attack / done')
        self.assertEqual(receipt['folder_id'], 'actual-child-id')
        name = receipt['snapshot']
        self.assertEqual(name, 'snapshots/0001_attack_done.zip')
        payload = self.transport.files[name]
        self.assertEqual(receipt['sha256'], hashlib.sha256(payload).hexdigest())
        self.assertEqual(receipt['bytes'], len(payload))
        self.assertEqual(receipt['stage'], 'attack / done')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertEqual(set(archive.namelist()), {'output.txt', 'review.json'})
        self.assertEqual(json.loads(self.transport.files['status/latest.json']), receipt)
        self.assertEqual(json.loads((self.root / 'drive_receipts.json').read_text()), [receipt])
        second = self.store.commit('caption')
        self.assertEqual(second['snapshot'], 'snapshots/0002_caption.zip')
        self.assertEqual(self.transport.files[name], payload)

    def test_secrets_and_temporary_files_excluded(self):
        for name in ['.env', '.env.local', 'tokens/cache.json', 'secrets/key.txt',
                     '__pycache__/cached.pyc', 'credentials.json', 'oauth_token.json',
                     'api_keys.json', 'unfinished.tmp', '.git/config']:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('SENSITIVE_MARKER')
        (self.root / 'review.md').write_text('review evidence')
        receipt = self.store.commit('safe')
        with zipfile.ZipFile(io.BytesIO(self.transport.files[receipt['snapshot']])) as archive:
            self.assertEqual(archive.namelist(), ['review.md'])
            self.assertNotIn(b'SENSITIVE_MARKER', b''.join(archive.read(n) for n in archive.namelist()))

    def test_corruption_fails_closed_for_snapshot_and_pointer(self):
        for corrupt in ['snapshots/', 'status/latest.json']:
            with self.subTest(corrupt=corrupt):
                fake = FakeTransport()
                fake.corrupt = corrupt
                store = storage.RunStorage(self.root, 'parent', 'run', transport=fake)
                with self.assertRaises(storage.DriveStorageError):
                    store.commit('attack')
                self.assertFalse(store.healthy)
                calls = len(fake.calls)
                with self.assertRaises(storage.DriveStorageError):
                    store.sentinel()
                with self.assertRaises(storage.DriveStorageError):
                    store.commit('retry')
                self.assertEqual(len(fake.calls), calls)

    def test_upload_failure_no_retry(self):
        self.transport.fail = True
        with self.assertRaises(storage.DriveStorageError):
            self.store.commit('attack')
        self.assertEqual(sum(c[0] == 'put' for c in self.transport.calls), 1)
        self.assertFalse(self.store.healthy)
        self.assertNotIn('status/latest.json', self.transport.files)

    def test_existing_snapshot_never_overwritten(self):
        self.transport.files['snapshots/0001_attack.zip'] = b'original'
        with self.assertRaises(storage.DriveStorageError):
            self.store.commit('attack')
        self.assertEqual(self.transport.files['snapshots/0001_attack.zip'], b'original')

    def test_sentinel_readback_and_failure(self):
        receipt = self.store.sentinel()
        self.assertEqual(receipt['folder_id'], self.transport.folder_id)
        self.assertEqual(receipt['stage'], 'sentinel')
        self.assertFalse((self.root / 'drive_receipts.json').exists())
        self.transport.corrupt = 'sentinel'
        with self.assertRaises(storage.DriveStorageError):
            self.store.sentinel()
        self.assertFalse(self.store.healthy)

    def test_write_json_atomic_and_default_string(self):
        target = self.root / 'nested/result.json'
        storage.write_json(target, {'path': Path('example')})
        self.assertEqual(json.loads(target.read_text()), {'path': 'example'})
        storage.write_json(target, {'new': True})
        self.assertEqual(json.loads(target.read_text()), {'new': True})
        self.assertFalse(target.with_suffix('.json.tmp').exists())


if __name__ == '__main__':
    unittest.main()
