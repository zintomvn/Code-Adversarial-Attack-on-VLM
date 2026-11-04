"""Small recovery contracts independent of model and transport implementations."""
import json
from pathlib import Path
import time

from storage import write_json


class AttemptLedger:
    """Immutable attempt identities, counting interrupted/failed calls too."""
    def __init__(self, folder, limit):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.limit = limit

    def records(self):
        return sorted(self.folder.glob('attempt_*.json'))

    def completed(self, key):
        for path in self.records():
            record = json.loads(path.read_text())
            if record['key'] == key and record.get('state') == 'COMPLETE':
                return record['result']
        return None

    def start(self, key, metadata):
        records = self.records()
        if len(records) >= self.limit:
            raise RuntimeError('Cumulative approved generation attempt budget exhausted')
        for path in records:
            if json.loads(path.read_text())['key'] == key:
                raise RuntimeError('Existing generation attempt cannot be silently repeated')
        path = self.folder / f'attempt_{len(records) + 1:04d}.json'
        write_json(path, {'key': key, 'metadata': metadata, 'state': 'STARTED', 'started': time.time()})
        return path

    def finish(self, path, result):
        record = json.loads(Path(path).read_text())
        if record['state'] != 'STARTED':
            raise RuntimeError('Completed generation record is immutable')
        write_json(path, {**record, 'state': 'COMPLETE', 'result': result, 'finished': time.time()})


def validate_review(key, decision, evidence_sha256):
    if decision.get('evidence_sha256') != evidence_sha256 or decision.get('status') != 'PASS':
        raise ValueError('Required review failed or belongs to stale evidence')
    if not decision.get('inspector') or not decision.get('notes'):
        raise ValueError('Named inspector and substantive notes are required')
    if key.endswith('final_checklist'):
        names = ['augmented_data', 'concepts', 'negatives', 'loss', 'crops', 'attack_outcome', 'adversarial_captions']
        for name in names:
            item = decision.get('checklist', {}).get(name, {})
            if item.get('status') not in ('PASS', 'N/A') or not item.get('artifacts'):
                raise ValueError(f'Required checklist item lacks passing linked evidence: {name}')
            if item['status'] == 'N/A' and not item.get('justification'):
                raise ValueError(f'N/A requires justification: {name}')
        for arm in ('mpc_original', 'mpc_negative10'):
            diagnostic = decision.get('diagnostic', {}).get(arm, {})
            if not all(diagnostic.get(field) for field in ('target_alignment', 'source_persistence')):
                raise ValueError('Frozen caption diagnostic is incomplete')


def pending_report(root, run_id, state, fields):
    payload = {'run_id': run_id, 'state': state, 'official_asr': 'PENDING',
               'time': time.time(), 'synchronization': 'PENDING_CONNECTOR_READBACK', **fields}
    write_json(Path(root) / 'sheet_update_pending.json', payload)
    (Path(root) / 'run_results_pending.md').write_text(
        '## Run Results\n\n' + json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
