"""Verify queued snapshots preserve bytes and failures stop an attack task."""
import importlib.util
import json
from pathlib import Path
import shutil
import threading
import types
import unittest
import uuid
from unittest.mock import patch
from zipfile import ZipFile

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tasks', REPO / 'notebooks/attacks/molab_remaining_tasks.py')
tasks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tasks)


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.root = REPO / 'tmp' / ('attack_tasks_qa_' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root)

    def fake_module(self, fail=False):
        stored = []
        run = self.root / 'work/primary_ablations/run'
        run.mkdir(parents=True)
        (run / 'status.json').write_text('{"version":1}', encoding='utf-8')
        state = {'attack_runs': {'primary_ablations': str(run)}, 'evaluations': {}}
        (self.root / 'campaign_state.json').write_text(json.dumps(state), encoding='utf-8')
        (self.root / 'molab_hypotheses_n200.json').write_text(json.dumps({'attack_jobs': [{'name': 'primary_ablations'}]}), encoding='utf-8')
        from hashlib import sha256

        class BackupFailure(BaseException):
            pass

        class Backup:
            def __init__(self, *args):
                pass

            def put(self, local, remote):
                if fail:
                    raise BackupFailure('simulated network failure')
                stored.append((remote, Path(local).read_bytes()))

        def read(path):
            return json.loads(Path(path).read_text(encoding='utf-8'))

        def write(path, value):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(value), encoding='utf-8')

        def attack_worker(root, plan, job, state_path, prefix, hours):
            backup = module.DriveBackup(prefix, job['name'])
            backup.snapshot(run)
            (run / 'status.json').write_text('{"version":2}', encoding='utf-8')
            backup.snapshot(run)
            return 75

        module = types.SimpleNamespace(DriveBackup=Backup, BackupFailure=BackupFailure,
                 read_json=read, write_json=write, digest=lambda p: sha256(str(p).encode()).hexdigest(),
                 attack_worker=attack_worker)
        return module, stored

    def test_immutable_queue_and_pause_flush(self):
        module, stored = self.fake_module()
        with patch.object(tasks, 'load_driver', return_value=module):
            self.assertEqual(tasks.task_worker(self.root, 'primary_ablations', 'backup', 1), 75)
        import io
        versions = []
        for name, data in stored:
            if name == 'root_latest.zip':
                with ZipFile(io.BytesIO(data)) as z:
                    versions.append(json.loads(z.read('status.json'))['version'])
        self.assertEqual(versions[:2], [1, 2])
        self.assertEqual(versions[-1], 2)
        self.assertFalse(list((self.root / 'parallel_tasks/primary_ablations/immutable_snapshots').glob('*.zip')))

    def test_network_failure_propagates(self):
        module, _ = self.fake_module(fail=True)
        with patch.object(tasks, 'load_driver', return_value=module):
            with self.assertRaises(module.BackupFailure):
                tasks.task_worker(self.root, 'primary_ablations', 'backup', 1)

    def test_conflicting_worker_states_rejected(self):
        module, _ = self.fake_module()
        module.write_json(self.root / 'parallel_tasks/other/campaign_state.json',
                          {'attack_runs': {'primary_ablations': 'different_run'}})
        with self.assertRaisesRegex(ValueError, 'Conflicting task state'):
            tasks.merged_state(self.root, module)


if __name__ == '__main__':
    unittest.main()
