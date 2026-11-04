"""Synthetic coverage of recovery budgets and mandatory reporting/gates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runtime import AttemptLedger, validate_review, pending_report


class RuntimeTests(unittest.TestCase):
    def test_generation_attempts_survive_restart_without_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            ledger = AttemptLedger(folder, 2)
            first = ledger.start('source', {'prompt': 'caption'})
            ledger.finish(first, {'text': 'one'})
            resumed = AttemptLedger(folder, 2)
            self.assertEqual(resumed.completed('source')['text'], 'one')
            second = resumed.start('target', {})
            self.assertNotEqual(first, second)
            with self.assertRaises(RuntimeError):
                AttemptLedger(folder, 2).start('third', {})

    def test_interrupted_attempt_cannot_silently_repeat(self):
        with tempfile.TemporaryDirectory() as name:
            AttemptLedger(Path(name), 3).start('source', {})
            with self.assertRaises(RuntimeError):
                AttemptLedger(Path(name), 3).start('source', {})

    def test_final_gate_requires_seven_results_and_diagnostic(self):
        decision = {'status': 'PASS', 'inspector': 'reviewer', 'notes': 'inspected', 'evidence_sha256': 'abc'}
        with self.assertRaises(ValueError):
            validate_review('00000_final_checklist', decision, 'abc')
        names = ['augmented_data', 'concepts', 'negatives', 'loss', 'crops', 'attack_outcome', 'adversarial_captions']
        decision['checklist'] = {key: {'status': 'PASS', 'artifacts': ['evidence.json']} for key in names}
        decision['diagnostic'] = {'mpc_original': {'target_alignment': 'partial', 'source_persistence': 'yes'},
                                  'mpc_negative10': {'target_alignment': 'partial', 'source_persistence': 'yes'}}
        validate_review('00000_final_checklist', decision, 'abc')
        decision['checklist']['loss']['status'] = 'PENDING'
        with self.assertRaises(ValueError):
            validate_review('00000_final_checklist', decision, 'abc')

    def test_reports_leave_asr_pending_and_capture_partial_stage(self):
        with tempfile.TemporaryDirectory() as name:
            pending_report(Path(name), 'RUN', 'STOPPED_PARTIAL', {'error': 'test'})
            value = json.loads((Path(name) / 'sheet_update_pending.json').read_text())
            self.assertEqual(value['official_asr'], 'PENDING')
            self.assertEqual(value['state'], 'STOPPED_PARTIAL')
            self.assertTrue((Path(name) / 'run_results_pending.md').exists())

    def test_caption_only_recovery_does_not_load_attack_models(self):
        from run_pipeline import Pilot, CONFIG
        with tempfile.TemporaryDirectory() as name:
            pilot = Pilot.__new__(Pilot)
            pilot.root = Path(name)
            for arm in CONFIG['arms']:
                folder = pilot.root / '00000' / arm
                folder.mkdir(parents=True)
                (folder / 'attack_complete.json').write_text('{}')
            with patch('models.MPCEncoder', side_effect=AssertionError('Unnecessary ensemble load')):
                pilot.attack_pair('00000', 2026, {})


if __name__ == '__main__':
    unittest.main()
