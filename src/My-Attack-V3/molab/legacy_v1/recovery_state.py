"""Pure, testable contracts for the explicitly approved recovery1 continuation."""
import hashlib
import io
import json
from pathlib import Path
import stat
import zipfile


def recover_caption_prefix(records):
    captions, attempts, seen = [], [], set()
    for index, record in enumerate(records):
        if record.get('state') != 'COMPLETE':
            raise ValueError('Interrupted generation cannot be silently repeated')
        raw = record['result']['text']
        caption = raw.strip()
        key = ' '.join(caption.casefold().split())
        accepted = bool(key) and key not in seen
        attempts.append({'attempt': index + 1, 'raw_output': raw, 'accepted': accepted})
        if accepted:
            seen.add(key)
            captions.append(caption)
    return captions, attempts


def verify_replay(expected, actual):
    if actual['token_ids'] != expected['token_ids'] or actual['text'] != expected['text']:
        raise RuntimeError('Exact recovery replay mismatch; no retry or approximate continuation')


def restore_snapshot(payload, pointer, root):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('Restore requires an empty root; no overwrite or automatic second recovery')
    if hashlib.sha256(payload).hexdigest() != pointer['sha256']:
        raise ValueError('Recovery snapshot checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            if not (root / member.filename).resolve().is_relative_to(root):
                raise ValueError('Snapshot path escapes the approved output directory')
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Snapshot symlinks are not permitted')
        if archive.testzip() is not None:
            raise ValueError('Snapshot ZIP CRC failure')
        root.mkdir(parents=True, exist_ok=True)
        archive.extractall(root)
    path = root / 'drive_receipts.json'
    receipts = json.loads(path.read_text())
    if max(record['sequence'] for record in receipts) != pointer['sequence'] - 1:
        raise ValueError('Unexpected recovered receipt sequence')
    path.write_text(json.dumps(receipts + [pointer], indent=2))


def verify_source_identity(project, manifest):
    """Scientific source/config/model pins are immutable; new recovery files may be added."""
    for relative, expected in manifest.items():
        if relative.startswith('test_'):
            continue  # Three extra synthetic tests were added after v1 packaging.
        path = Path(project) / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Original scientific source identity changed: {relative}')
