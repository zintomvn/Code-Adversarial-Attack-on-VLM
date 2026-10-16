import importlib.util
import json
from pathlib import Path
import shutil
import unittest
import uuid

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('overnight', REPO / 'notebooks/attacks/molab_overnight.py')
night = importlib.util.module_from_spec(spec)
spec.loader.exec_module(night)


class OvernightTests(unittest.TestCase):
    def test_h4_waits_for_the_full_factual_caption_cache(self):
        root = REPO / 'tmp' / ('overnight_qa_' + uuid.uuid4().hex)
        root.mkdir(parents=True)
        try:
            state = {'attack_runs': {'primary_ablations': str(root)}}
            self.assertFalse(night.factual_captions_ready(state, 200))
            texts = {str(i): {'source': 'source', 'target': 'target'} for i in range(95)}
            (root / 'text_records.json').write_text(json.dumps(texts), encoding='utf-8')
            self.assertFalse(night.factual_captions_ready(state, 200))
            texts.update({str(i): {'source': 'source', 'target': 'target'} for i in range(95, 200)})
            (root / 'text_records.json').write_text(json.dumps(texts), encoding='utf-8')
            self.assertTrue(night.factual_captions_ready(state, 200))
            texts['0']['caption_error'] = 'failed'
            (root / 'text_records.json').write_text(json.dumps(texts), encoding='utf-8')
            self.assertFalse(night.factual_captions_ready(state, 200))
        finally:
            shutil.rmtree(root)

    def test_pending_jobs_keep_the_original_variant_order(self):
        configs = [{'id': 'primary__a', 'job': 'primary', 'variant': 'a'},
                   {'id': 'primary__b', 'job': 'primary', 'variant': 'b'},
                   {'id': 'h4__a', 'job': 'h4', 'variant': 'a'}]
        state = {'attack_runs': {}}
        self.assertEqual([c['id'] for c in night.pending_attacks(configs, state, set(), set())], ['primary__a', 'h4__a'])
        self.assertEqual([c['id'] for c in night.pending_attacks(configs, state, {'primary'}, set())], ['h4__a'])
        self.assertEqual(night.pending_attacks(configs, state, {'primary'}, {'h4'}), [])


if __name__ == '__main__':
    unittest.main()
