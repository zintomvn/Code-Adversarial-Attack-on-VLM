"""Bounded per-stage artifact snapshots in MoLab's native persistent cache."""
from pathlib import Path
import hashlib
import io
import json
import zipfile


def read_snapshot(snapshot_path: str, stage_key: str) -> bytes:
    # Deliberately no model calls on cache miss: absent scratch -> explicit error.
    return Path(snapshot_path).read_bytes()


class StagePersistence:
    def __init__(self, run_root, cache_root=None):
        self.root = Path(run_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_root = Path(cache_root) if cache_root else None
        self.index = self.root.parent / (self.root.name + '_restore_index.py')
        self.entries = {}
        if self.index.exists():
            # A JSON literal in a retained source file, never executable input.
            text = self.index.read_text(encoding='utf-8')
            self.entries = json.loads(text.split(' = ', 1)[1])
        self.reader = read_snapshot
        if self.cache_root:
            import marimo as mo
            self.reader = mo.persistent_cache(save_path=str(self.cache_root), method='pickle')(read_snapshot)

    @staticmethod
    def archive(folder):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(Path(folder).rglob('*')):
                if path.is_file() and path.name != '.env' and '__pycache__' not in path.parts:
                    archive.write(path, path.relative_to(folder).as_posix())
        return stream.getvalue()

    def save(self, stage_key, folder):
        folder = Path(folder)
        relative = folder.relative_to(self.root).as_posix()
        payload = self.archive(folder)
        digest = hashlib.sha256(payload).hexdigest()
        stage_key = stage_key + '_' + digest[:16]
        scratch = self.root.parent / '.snapshots' / f'{self.root.name}_{stage_key}.zip'
        scratch.parent.mkdir(parents=True, exist_ok=True)
        scratch.write_bytes(payload)
        restored = self.reader(str(scratch), stage_key)
        if hashlib.sha256(restored).hexdigest() != digest:
            raise RuntimeError('Native cache round-trip hash mismatch')
        if len(restored) >= 950_000_000:
            raise RuntimeError('Per-stage snapshot exceeds MoLab file-size allowance')
        # One latest snapshot per folder; previous optimizer snapshots stay cached but are not restored.
        self.entries[relative] = {'key': stage_key, 'scratch': str(scratch), 'sha256': digest,
                                 'sequence': max([e.get('sequence', 0) for e in self.entries.values()] + [0]) + 1}
        temporary = self.index.with_suffix('.tmp')
        temporary.write_text('RESTORE_INDEX = ' + json.dumps(self.entries, indent=2), encoding='utf-8')
        temporary.replace(self.index)
        return {'stage': stage_key, 'bytes': len(payload), 'sha256': digest,
                'native_cache_enabled': bool(self.cache_root)}

    def restore(self):
        restored = []
        for relative, entry in sorted(self.entries.items(), key=lambda item: item[1].get('sequence', 0)):
            payload = self.reader(entry['scratch'], entry['key'])
            if hashlib.sha256(payload).hexdigest() != entry['sha256']:
                raise RuntimeError('Retained snapshot hash mismatch; no automatic regeneration')
            target = self.root / relative
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                if archive.testzip() is not None:
                    raise RuntimeError('Invalid retained ZIP')
                for name in archive.namelist():
                    path = (target / name).resolve()
                    if not path.is_relative_to(target.resolve()):
                        raise ValueError('Unsafe snapshot path')
                archive.extractall(target)
            restored.append(relative)
        return restored
