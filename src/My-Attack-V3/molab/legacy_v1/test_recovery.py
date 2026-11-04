"""Synthetic recovery contracts; no model loading or account calls."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from recovery_state import recover_caption_prefix, verify_replay, restore_snapshot, verify_source_identity


class RecoveryTests(unittest.TestCase):
    def record(self, text):
        return {'state': 'COMPLETE', 'result': {'text': text, 'token_ids': [1, 2],
                'cpu_rng_before': [1], 'cuda_rng_before': [2]}}

    def test_prefix_preserves_attempt_counter_and_rejects_duplicates(self):
        captions, attempts = recover_caption_prefix([self.record('Panda'), self.record(' Panda '), self.record('Panda face')])
        self.assertEqual(captions, ['Panda', 'Panda face'])
        self.assertEqual(len(attempts), 3)
        self.assertFalse(attempts[1]['accepted'])

    def test_incomplete_prefix_is_not_silently_repeated(self):
        with self.assertRaises(ValueError):
            recover_caption_prefix([{'state': 'STARTED'}])

    def test_replay_requires_exact_tokens_and_text(self):
        expected = {'text': 'panda', 'token_ids': [1, 2]}
        verify_replay(expected, dict(expected))
        with self.assertRaises(RuntimeError):
            verify_replay(expected, {'text': 'panda', 'token_ids': [1, 3]})
        with self.assertRaises(RuntimeError):
            verify_replay(expected, {'text': 'bear', 'token_ids': [1, 2]})

    def test_restore_reconciles_sequence_without_overwriting_snapshot(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('drive_receipts.json', json.dumps([{'sequence': 58}]))
            archive.writestr('config.json', '{}')
        data = buffer.getvalue()
        pointer = {'sequence': 59, 'sha256': hashlib.sha256(data).hexdigest(), 'snapshot': 'snapshots/0059.zip'}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / 'run'
            restore_snapshot(data, pointer, root)
            self.assertEqual(json.loads((root / 'drive_receipts.json').read_text())[-1]['sequence'], 59)
            with self.assertRaises(RuntimeError):
                restore_snapshot(data, pointer, root)

    def test_restore_rejects_corruption_and_path_escape(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / 'run'
            with self.assertRaises(ValueError):
                restore_snapshot(b'bad', {'sha256': 'wrong'}, root)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as archive:
                archive.writestr('../outside', 'bad')
            data = buffer.getvalue()
            with self.assertRaises(ValueError):
                restore_snapshot(data, {'sha256': hashlib.sha256(data).hexdigest()}, root)

    def test_original_scientific_source_cannot_change(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'attack.py').write_text('original')
            lock = {'attack.py': hashlib.sha256(b'original').hexdigest()}
            verify_source_identity(root, lock)
            (root / 'attack.py').write_text('changed')
            with self.assertRaises(RuntimeError):
                verify_source_identity(root, lock)

    def test_continued_bank_uses_prefix_and_only_missing_caption_calls(self):
        from recover_pipeline import ContinuedNegativeBank
        class FakeCaptioner:
            calls = 0
            def caption(self, image, sample):
                self.calls += 1
                return f'New caption {self.calls}'
        historical = [self.record(f'Caption {i % 8}') for i in range(28)]
        captioner = FakeCaptioner()
        bank = ContinuedNegativeBank(captioner, historical)
        bank.provenance = {}
        captions = bank.generate_captions(None)
        self.assertEqual(len(captions), 10)
        self.assertEqual(captioner.calls, 2)
        self.assertEqual(len(bank.provenance['caption_attempts']), 30)

    def test_cumulative80_caption_limit_does_not_reset_on_restore(self):
        from recover_pipeline import ContinuedNegativeBank
        class ForbiddenCaptioner:
            def caption(self, image, sample):
                raise AssertionError('Attempt81 was not approved')
        bank = ContinuedNegativeBank(ForbiddenCaptioner(), [self.record('Same') for _ in range(80)])
        bank.provenance = {}
        with self.assertRaises(RuntimeError):
            bank.generate_captions(None)

    def test_setup_uses_uv_for_freeze_and_requires_source_archive(self):
        from setup_recovery import freeze_command, require_source_archive, remaining_seconds
        python = Path('/env/python')
        self.assertEqual(freeze_command('uv', python), ['uv', 'pip', 'freeze', '--python', str(python)])
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(RuntimeError):
                require_source_archive(Path(name) / 'missing.zip')
        with self.assertRaises(TimeoutError):
            remaining_seconds(1, now=2)


if __name__ == '__main__':
    unittest.main()
