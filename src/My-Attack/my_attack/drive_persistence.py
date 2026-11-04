"""Verified incremental checkpoints on the user's Google Drive, independent of MoLab scratch."""
from pathlib import Path, PurePosixPath
import copy
import hashlib
import io
import json
import re
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


class DriveStagePersistence:
    """Append verified changed-file ZIPs, then commit an immutable index and latest pointer."""
    def __init__(self, run_root, transport, sync_every_samples=1):
        self.root = Path(run_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.transport = transport
        self.sync_every_samples = int(sync_every_samples)
        if self.sync_every_samples < 1:
            raise ValueError('sync_every_samples must be positive')
        self.pending_folders = set()
        self.completed_since_sync = set()
        self.healthy = True
        self.index = {'schema': 1, 'files': {}, 'sequence': 0, 'stage': None}
        self.remote_groups = {}
        self.group_files = {}
        if self.transport.exists('latest.json'):
            self.index = self.read_index()

    @staticmethod
    def sha(data):
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def encoded(value):
        return json.dumps(value, sort_keys=True, indent=2).encode('utf-8')

    @staticmethod
    def allowed(relative):
        parts = PurePosixPath(relative).parts
        forbidden = {'.env', '.git', '__pycache__', '.drive_staging'}
        return not any(p in forbidden for p in parts) and not any(
            token in PurePosixPath(relative).name.lower()
            for token in ['credentials', 'token.json', 'api_keys', 'secret']) and not relative.endswith('.tmp')

    def verified_put(self, name, data):
        self.transport.put(name, data)
        if self.sha(self.transport.get(name)) != self.sha(data):
            raise DriveStorageError(f'Drive readback SHA256 mismatch: {name}')

    def read_index(self):
        pointer = json.loads(self.transport.get('latest.json'))
        data = self.transport.get(pointer['index'])
        if self.sha(data) != pointer['sha256']:
            raise DriveStorageError('Drive index checksum mismatch; no automatic regeneration')
        index = json.loads(data)
        if index.get('schema') == 2:
            self.remote_groups = index['groups']
            self.group_files = {}
            files = {}
            for group, record in self.remote_groups.items():
                payload = self.transport.get(record['manifest'])
                if self.sha(payload) != record['sha256']:
                    raise DriveStorageError('Sample manifest checksum mismatch')
                group_files = json.loads(payload)
                if any(self.group(name) != group for name in group_files):
                    raise DriveStorageError('Checkpoint manifest group mismatch')
                self.group_files[group] = group_files
                files.update(group_files)
            return {'schema': 1, 'files': files, 'sequence': index['sequence'], 'stage': index['stage']}
        if index.get('schema') != 1 or not isinstance(index.get('files'), dict):
            raise DriveStorageError('Unknown Drive index schema')
        for name, record in index['files'].items():
            self.group_files.setdefault(self.group(name), {})[name] = record
        return index

    @staticmethod
    def group(name):
        parts = PurePosixPath(name).parts
        return parts[0] if len(parts) > 1 else '_root'

    def save(self, stage_key, folder):
        if not self.healthy:
            raise DriveStorageError('Previous durable commit failed; refusing further work')
        try:
            if self.sync_every_samples > 1:
                relative = Path(folder).resolve().relative_to(self.root.resolve())
                if relative.parts:
                    self.pending_folders.add(relative.parts[0])
                if stage_key.endswith('_complete'):
                    self.completed_since_sync.add(stage_key[:-9])
                final = stage_key.startswith('run_metadata_')
                if len(self.completed_since_sync) >= self.sync_every_samples or final:
                    return self.flush(stage_key)
                return {'backend': 'LOCAL_PENDING_DRIVE', 'stage': stage_key,
                        'sync_every_samples': self.sync_every_samples,
                        'completed_since_sync': len(self.completed_since_sync),
                        'verified_readback': False}
            return self._save(stage_key, Path(folder))
        except Exception as error:
            self.healthy = False
            if isinstance(error, DriveStorageError):
                raise
            raise DriveStorageError(f'Drive commit failed ({type(error).__name__}); work stopped') from None

    def flush(self, stage_key):
        """Upload one changed-file chunk per sample at a batch boundary, keeping chunks bounded."""
        result = None
        for name in sorted(self.pending_folders):
            folder = self.root / name
            if folder.is_dir():
                result = self._save(f'{stage_key}_batch_{name}', folder)
        self.pending_folders.clear()
        self.completed_since_sync.clear()
        return result or {'backend': 'LOCAL_PENDING_DRIVE', 'stage': stage_key}

    def _save(self, stage_key, folder):
        relative_folder = folder.resolve().relative_to(self.root.resolve()).as_posix()
        selected = {p.relative_to(self.root).as_posix(): p for p in folder.rglob('*') if p.is_file()}
        # Budget, identity, latest heartbeat/errors/metrics travel with every stage commit.
        selected.update({p.name: p for p in self.root.glob('*.json') if p.is_file()})
        selected.update({p.name: p for p in self.root.glob('*.txt') if p.is_file()})
        selected = {name: path for name, path in selected.items() if self.allowed(name)}
        candidate = copy.deepcopy(self.index)
        removed = []
        # Record deletions (e.g. completed generation markers), so restore cannot resurrect them.
        for name in list(candidate['files']):
            within = relative_folder == '.' or name == relative_folder or name.startswith(relative_folder + '/')
            if within and name not in selected:
                del candidate['files'][name]
                removed.append(name)
        changed = {}
        for name, path in selected.items():
            data = path.read_bytes()
            digest = self.sha(data)
            if self.index['files'].get(name, {}).get('sha256') != digest:
                changed[name] = (data, digest)
        sequence = self.index['sequence'] + 1
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, (data, _) in changed.items():
                archive.writestr(name, data)
        payload = stream.getvalue()
        archive_sha = self.sha(payload)
        safe_key = re.sub(r'[^A-Za-z0-9_.-]', '_', stage_key)[:100]
        chunk = f'chunks/{sequence:06d}_{safe_key}_{archive_sha[:16]}.zip'
        if len(payload) >= 900_000_000:
            raise DriveStorageError('Stage chunk too large; split before uploading')
        self.verified_put(chunk, payload)
        for name, (data, digest) in changed.items():
            candidate['files'][name] = {'chunk': chunk, 'chunk_sha256': archive_sha,
                                        'sha256': digest, 'bytes': len(data)}
        candidate.update(sequence=sequence, stage=stage_key)
        # Per-sample manifests keep root index O(number of samples), not O(all crop/checkpoint files).
        touched = {self.group(name) for name in [*changed, *removed]}
        if not self.remote_groups and self.group_files:
            touched.update(self.group_files)
        next_groups = dict(self.remote_groups)
        updated_files = {}
        for group in touched:
            files = dict(self.group_files.get(group, {}))
            for name in removed:
                files.pop(name, None)
            for name in changed:
                if self.group(name) == group:
                    files[name] = candidate['files'][name]
            data = self.encoded(files)
            digest = self.sha(data)
            name = f'manifests/{group}/{digest[:24]}.json'
            self.verified_put(name, data)
            next_groups[group] = {'manifest': name, 'sha256': digest}
            updated_files[group] = files
        index_data = self.encoded({'schema': 2, 'groups': next_groups,
                                   'sequence': sequence, 'stage': stage_key})
        index_sha = self.sha(index_data)
        index_path = f'indices/{sequence:06d}_{index_sha[:16]}.json'
        self.verified_put(index_path, index_data)
        # HEAD points only at a fully uploaded/readback-verified generation.
        self.verified_put('latest.json', self.encoded({'index': index_path, 'sha256': index_sha,
                                                     'sequence': sequence, 'stage': stage_key}))
        self.index = candidate
        self.remote_groups = next_groups
        self.group_files.update(updated_files)
        progress = {'stage': stage_key, 'sequence': sequence, 'indexed_files': len(candidate['files'])}
        for name in ['run_status.json', 'heartbeat.json', 'budget.json', 'metrics_confirmation.json']:
            path = self.root / name
            if path.exists():
                progress[name] = json.loads(path.read_text(encoding='utf-8'))
        self.verified_put('progress.json', self.encoded(progress))
        if hasattr(self.transport, 'folder_id'):
            folder_id = self.transport.folder_id
        else:
            folder_id = None
        result = {'backend': 'GOOGLE_DRIVE', 'stage': stage_key, 'sequence': sequence,
                  'changed_files': len(changed), 'uploaded_bytes': len(payload),
                  'sha256': archive_sha, 'verified_readback': True, 'folder_id': folder_id}
        (self.root / 'drive_commit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result

    def restore(self):
        self.index = self.read_index()
        groups = {}
        for name, record in self.index['files'].items():
            if (PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
                    or not self.allowed(name)):
                raise DriveStorageError('Unsafe path in remote checkpoint')
            groups.setdefault(record['chunk'], []).append((name, record))
        restored = []
        for chunk, records in groups.items():
            payload = self.transport.get(chunk)
            if any(self.sha(payload) != record['chunk_sha256'] for _, record in records):
                raise DriveStorageError('Checkpoint ZIP checksum mismatch')
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                if archive.testzip() is not None:
                    raise DriveStorageError('Corrupt checkpoint ZIP')
                for name, record in records:
                    data = archive.read(name)
                    if self.sha(data) != record['sha256'] or len(data) != record['bytes']:
                        raise DriveStorageError(f'Checkpoint file checksum mismatch: {name}')
                    path = self.root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    restored.append(name)
        return restored
