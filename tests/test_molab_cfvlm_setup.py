import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import types
import unittest
import uuid
from unittest.mock import patch
from zipfile import ZipFile

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / 'notebooks/attacks'
spec = importlib.util.spec_from_file_location('cfsetup', BASE / 'molab_cfvlm_setup.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class SetupTests(unittest.TestCase):
    def test_interrupted_legacy_campaign_uses_variant_markers_and_never_old_bank(self):
        folder = REPO / 'tmp' / ('cfsetup_qa_' + uuid.uuid4().hex)
        folder.mkdir(parents=True)
        try:
            package, root, eval_root = folder / 'package', folder / 'attack', folder / 'eval'
            package.mkdir()
            names = ('molab_hypothesis_campaign.py', 'molab_remaining_tasks.py', 'molab_overnight.py',
                     'kltn-attack-v4.3.3-h1-cfvlm.ipynb', 'cfvlm_negatives.py')
            for name in names:
                shutil.copy2(BASE / name, package / name)
            for name in ('molab_eval_completed.py', 'text_gen_eval_v3.1.ipynb'):
                shutil.copy2(REPO / 'notebooks' / name, package / name)
            shutil.copy2(BASE / 'molab_hypotheses_n200.json', package / 'original_plan.json')
            original = setup.read(package / 'original_plan.json')
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.iterdir()}
            (package / 'checksums.json').write_text(json.dumps(hashes), encoding='utf-8')
            texts = {str(i): {'source': 'source', 'target': 'target', 'negatives': [{'caption': 'OLD NEGATIVE'}]}
                     for i in range(200)}
            legacy = io.BytesIO()
            with ZipFile(legacy, 'w') as z:
                z.writestr('text_records.json', json.dumps(texts))
                z.writestr('run_config.json', json.dumps({'model_revisions': original['attack_model_revisions']}))
                for variant in ('foa_reference', 'mpc_clip'):
                    z.writestr(variant + '/status.json', json.dumps({'status': 'complete', 'completed': 200}))
                    z.writestr(variant + '/manifest.jsonl', '\n'.join(json.dumps({'index': i}) for i in range(200)))
                # No root summary.json: real timeout checkpoints only have the
                # per-variant status/manifest files until all variants finish.
            data = io.BytesIO()
            with ZipFile(data, 'w') as z:
                z.writestr('pairs_200.jsonl', '')
            blobs = {setup.PARENT + '/results/primary_ablations/root_latest.zip': legacy.getvalue(),
                     setup.PARENT + '/attack_data_n200.zip': data.getvalue()}
            class FakeFS:
                def __init__(self, **kwargs): pass
                def get_file(self, remote, local): Path(local).write_bytes(blobs[remote])
                def put_file(self, local, remote): blobs[remote] = Path(local).read_bytes()
                def open(self, remote, mode): return io.BytesIO(blobs[remote])
                def exists(self, remote): return remote in blobs
                def makedirs(self, *args, **kwargs): pass
            def write(path, value):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(value), encoding='utf-8')
            driver = types.SimpleNamespace(validate_package=lambda *a: None, write_json=write)
            actual_load = setup.load
            with patch.dict(sys.modules, {'gdrive_fsspec': types.SimpleNamespace(GoogleDriveFileSystem=FakeFS)}), \
                 patch.object(setup, 'DATA_SHA', hashlib.sha256(data.getvalue()).hexdigest()), \
                 patch.object(setup, 'load', side_effect=lambda path, name: driver if Path(path).name == 'molab_hypothesis_campaign.py' else actual_load(path, name)):
                plan = setup.initialize(package, root, eval_root)
                self.assertEqual(sum(len(j['variants']) for j in plan['attack_jobs']), 8)
                self.assertFalse(plan['legacy_provenance']['original_bank_imported'])
                self.assertNotIn('OLD NEGATIVE', (root / 'molab_hypotheses_n200.json').read_text())
                self.assertEqual(setup.read(root / 'campaign_state.json')['attack_runs'], {})
                self.assertEqual(setup.initialize(package, root, eval_root), plan)
        finally:
            shutil.rmtree(folder)


if __name__ == '__main__':
    unittest.main()
