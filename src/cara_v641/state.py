"""Durable stage identities, explicit audit gates and conservative API reservations."""
import hashlib
import json
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require_gate(path, identity):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f'Audit gate pending: {path.name}')
    gate = json.loads(path.read_text(encoding='utf-8'))
    if gate.get('identity') != identity or gate.get('status') != 'PASS' or not gate.get('evidence'):
        raise RuntimeError(f'Audit gate not passed for this identity: {path.name}')


class StageStore:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = fingerprint(identity)
        path = self.root / 'identity.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != identity:
                raise ValueError('Immutable run identity mismatch')
        else:
            atomic_json(path, identity)

    def get(self, stage):
        path = self.root / f'{stage}.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

    def save(self, stage, value):
        atomic_json(self.root / f'{stage}.json', value)


class Budget:
    """Single worker; unknown request usage retains its entire reservation on disk."""
    def __init__(self, path, cap=1.0, attempts=28):
        self.path = Path(path)
        self.cap = cap
        self.limit = attempts
        self.data = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {'spent': 0.0, 'reserved': 0.0, 'attempts': 0}

    def reserve(self, maximum, request_key=None):
        if maximum <= 0 or self.data['attempts'] >= self.limit:
            raise RuntimeError('API attempt budget exhausted')
        if self.data['spent'] + self.data['reserved'] + maximum > self.cap:
            raise RuntimeError('API dollar budget exhausted')
        if request_key is not None:
            requests = self.data.setdefault('requests', {})
            if requests.get(request_key, 0) >= 2:
                raise RuntimeError('Logical request exhausted its two durable attempts')
            requests[request_key] = requests.get(request_key, 0) + 1
        self.data['reserved'] += maximum
        self.data['attempts'] += 1
        atomic_json(self.path, self.data)

    def settle(self, reservation, actual):
        if not 0 <= actual <= reservation or self.data['reserved'] + 1e-9 < reservation:
            raise ValueError('Invalid usage settlement')
        self.data['reserved'] -= reservation
        self.data['spent'] += actual
        atomic_json(self.path, self.data)
