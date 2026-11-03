"""Fail-closed, verified Drive snapshots for the approved pipeline."""
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import re
import stat
import uuid
import zipfile


class DriveStorageError(RuntimeError):
    """A failed durable commit stops work; it must never become a sample failure/retry."""


class GoogleDriveTransport:
    """Reuse MoLab's existing gdrive-fsspec OAuth credentials, scoped to an exact folder ID."""
    def __init__(self, folder_id, prefix):
        from gdrive_fsspec import GoogleDriveFileSystem
        if not re.fullmatch(r'[A-Za-z0-9_-]+', folder_id):
            raise ValueError('Use an exact Google Drive folder ID')
        self.fs = GoogleDriveFileSystem(root_file_id=folder_id, token='cache', skip_instance_cache=True)
        self.prefix = prefix.strip('/')
        self.path('validation')
        self._folder_cache = {'': folder_id}
        root = self.fs.files.get(fileId=folder_id, supportsAllDrives=True,
                                 fields='id,name,mimeType,capabilities(canAddChildren)').execute(num_retries=0)
        if root.get('mimeType') != 'application/vnd.google-apps.folder':
            raise DriveStorageError('Checkpoint destination is not a Drive folder')
        if not root.get('capabilities', {}).get('canAddChildren', False):
            raise DriveStorageError('No write permission on checkpoint destination')
        self.folder_id = self._folder(self.prefix, create=True)

    def path(self, name):
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('Unsafe Drive-relative path')
        return self.prefix + '/' + name

    def exists(self, name):
        try:
            parent, leaf = self.path(name).rsplit('/', 1)
            return self._find(self._folder(parent), leaf) is not None
        except FileNotFoundError:
            return False

    def get(self, name):
        parent, leaf = self.path(name).rsplit('/', 1)
        record = self._find(self._folder(parent), leaf)
        if record is None:
            raise FileNotFoundError(name)
        return self.fs.files.get_media(fileId=record['id'], supportsAllDrives=True).execute(num_retries=0)

    def _find(self, parent_id, name):
        escaped = name.replace('\\', '\\\\').replace("'", "\\'")
        result = self.fs.files.list(q=f"'{parent_id}' in parents and name = '{escaped}' and trashed = false",
                                    fields='files(id,name,mimeType,size)', pageSize=100,
                                    supportsAllDrives=True, includeItemsFromAllDrives=True).execute(num_retries=0)
        records = result.get('files', [])
        if len(records) > 1:
            raise DriveStorageError(f'Duplicate Drive paths require review: {name}')
        return records[0] if records else None

    def _folder(self, path, create=False):
        if path in self._folder_cache:
            return self._folder_cache[path]
        parent, _, name = path.rpartition('/')
        parent_id = self._folder(parent, create=create)
        record = self._find(parent_id, name)
        if record is None:
            if not create:
                raise FileNotFoundError(path)
            record = self.fs.files.create(body={'name': name, 'parents': [parent_id],
                                                'mimeType': 'application/vnd.google-apps.folder'},
                                          fields='id,mimeType', supportsAllDrives=True).execute(num_retries=0)
        if record.get('mimeType') != 'application/vnd.google-apps.folder':
            raise DriveStorageError(f'Expected a Drive folder: {path}')
        self._folder_cache[path] = record['id']
        return record['id']

    def put(self, name, data):
        # gdrive-fsspec's generic wb creates duplicate names on overwrite.
        # Use its authenticated official Drive API for atomic file-content updates.
        from googleapiclient.http import MediaIoBaseUpload
        path = self.path(name)
        parent = path.rsplit('/', 1)[0]
        parent_id = self._folder(parent, create=True)
        existing = self._find(parent_id, path.rsplit('/', 1)[-1])
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype='application/octet-stream',
                                 chunksize=8 * 1024**2, resumable=True)
        if existing:
            request = self.fs.files.update(fileId=existing['id'], media_body=media,
                                           supportsAllDrives=True)
        else:
            request = self.fs.files.create(body={'name': path.rsplit('/', 1)[-1],
                                                'parents': [parent_id]},
                                           media_body=media, supportsAllDrives=True)
        request.execute(num_retries=0)


def write_json(path, value):
    """Replace a JSON file atomically on its local filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str),
                         encoding='utf-8')
    temporary.replace(path)


class RunStorage:
    """Publish immutable snapshots, then a verified latest pointer; never retry failures.

    Instances are single-writer. Reuse the local root when resuming; an existing
    remote snapshot is an error rather than permission to overwrite it.
    """

    def __init__(self, root: Path, folder_id: str, prefix: str, transport=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.healthy = True
        self.transport = transport if transport is not None else GoogleDriveTransport(folder_id, prefix)
        self.folder_id = self.transport.folder_id
        receipt_path = self.root / 'drive_receipts.json'
        self.receipts = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else []
        self.sequence = max((r['sequence'] for r in self.receipts), default=0)

    def _check(self):
        if not self.healthy:
            raise DriveStorageError('Previous sync failed; refusing further work')

    def _verified_put(self, name, payload):
        self.transport.put(name, payload)
        digest = hashlib.sha256(payload).hexdigest()
        if hashlib.sha256(self.transport.get(name)).hexdigest() != digest:
            raise DriveStorageError(f'Drive readback SHA256 mismatch: {name}')
        return digest

    @staticmethod
    def _allowed(relative):
        for part in relative.parts:
            name = part.lower()
            if (name in {'.git', '__pycache__', '.drive_staging', 'tmp', 'temp'}
                    or name.startswith('.env') or name.endswith(('.tmp', '.temp', '.pyc'))
                    or any(word in name for word in ('token', 'secret', 'credential', 'api_key'))):
                return False
        return True

    def commit(self, stage: str):
        self._check()
        try:
            safe_stage = re.sub(r'[^A-Za-z0-9_-]+', '_', stage).strip('_')[:100] or 'stage'
            sequence = self.sequence + 1
            snapshot = f'snapshots/{sequence:04d}_{safe_stage}.zip'
            if self.transport.exists(snapshot):
                raise DriveStorageError(f'Refusing to overwrite immutable snapshot: {snapshot}')
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(self.root.rglob('*')):
                    relative = path.relative_to(self.root)
                    if not self._allowed(relative):
                        continue
                    # Never follow directory or file symlinks outside the output tree.
                    if any(p.is_symlink() for p in (path, *path.parents) if p != self.root.parent):
                        continue
                    if stat.S_ISREG(path.lstat().st_mode):
                        archive.write(path, relative.as_posix())
            payload = buffer.getvalue()
            digest = self._verified_put(snapshot, payload)
            receipt = {'folder_id': self.folder_id, 'sha256': digest,
                       'bytes': len(payload), 'stage': stage, 'sequence': sequence,
                       'snapshot': snapshot}
            pointer = json.dumps(receipt, sort_keys=True, indent=2).encode('utf-8')
            self._verified_put('status/latest.json', pointer)
            write_json(self.root / 'drive_receipts.json', self.receipts + [receipt])
            self.receipts.append(receipt)
            self.sequence = sequence
            return receipt
        except Exception as exc:
            self.healthy = False
            raise DriveStorageError(f'Drive commit failed at stage {stage!r}: {exc}') from exc

    def sentinel(self):
        self._check()
        try:
            name = f'status/sentinel_{uuid.uuid4().hex}.txt'
            payload = b'Harmless pipeline storage upload/readback check.\n'
            digest = self._verified_put(name, payload)
            return {'folder_id': self.folder_id, 'sha256': digest,
                    'bytes': len(payload), 'stage': 'sentinel', 'path': name}
        except Exception as exc:
            self.healthy = False
            raise DriveStorageError(f'Drive sentinel failed: {exc}') from exc
