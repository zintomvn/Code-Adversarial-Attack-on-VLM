"""Portable orchestration. Author attack files are verified, never edited.

Commands: prepare, run, status. The notebook supplies a JSON configuration.
Drive contains immutable, checksum-verified sample archives and per-image evaluation
records. A commit is considered durable only after a successful Drive upload.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if z.testzip():
            raise ValueError('Corrupt ZIP')
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if '\\' in info.filename or not target.is_relative_to(destination):
                raise ValueError('Unsafe ZIP path')
            if info.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(info)
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Conflicting existing artifact: {target}')
            if not target.exists():
                target.write_bytes(data)


def credentials():
    """Use an environment secret or private pydata OAuth cache."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google.oauth2 import service_account
    import base64

    raw = os.environ.get('BASELINE_DRIVE_CREDENTIALS_JSON')
    if not raw and os.environ.get('MOLAB_DRIVE_CREDENTIALS_B64'):
        raw = base64.b64decode(os.environ['MOLAB_DRIVE_CREDENTIALS_B64']).decode()
    if raw:
        info = json.loads(raw)
        cred = (service_account.Credentials.from_service_account_info(
            info, scopes=['https://www.googleapis.com/auth/drive'])
            if info.get('type') == 'service_account' else Credentials.from_authorized_user_info(info))
    else:
        from pydata_google_auth.cache import ReadWriteCredentialsCache
        cred = ReadWriteCredentialsCache().load()
        if cred is None:
            raise RuntimeError('Connect Google Drive in the notebook before starting.')
    if not cred.valid:
        cred.refresh(Request())
    return cred


class Drive:
    """Drive API by IDs, avoiding ambiguous path-based uploads and duplicate files."""

    def __init__(self, parent_id):
        from googleapiclient.discovery import build
        self.api = build('drive', 'v3', credentials=credentials(), cache_discovery=False)
        self.parent = parent_id
        self.api.files().get(fileId=parent_id, fields='id,name').execute(num_retries=5)
        self.folders = {'': parent_id}

    def children(self, parent):
        rows, token = [], None
        while True:
            page = self.api.files().list(q=f"'{parent}' in parents and trashed=false", pageSize=1000,
                pageToken=token, fields='nextPageToken,files(id,name,mimeType,md5Checksum,size)').execute(num_retries=5)
            rows.extend(page.get('files', []))
            token = page.get('nextPageToken')
            if not token:
                return rows

    def find(self, parent, name):
        rows = [r for r in self.children(parent) if r['name'] == name]
        if len(rows) > 1:
            raise RuntimeError(f'Duplicate Drive names: {name}; reconcile duplicates first.')
        return rows[0] if rows else None

    def folder(self, relative, create=True):
        relative = str(relative).strip('/')
        if relative in self.folders:
            return self.folders[relative]
        parent, _, name = relative.rpartition('/')
        parent_id = self.folder(parent, create)
        if parent_id is None:
            return None
        row = self.find(parent_id, name)
        if row is None:
            if not create:
                return None
            file_id = self.api.files().generateIds(count=1).execute(num_retries=5)['ids'][0]
            body = dict(id=file_id, name=name, parents=[parent_id], mimeType='application/vnd.google-apps.folder')
            try:
                row = self.api.files().create(body=body, fields='id').execute(num_retries=0)
            except Exception:
                # A lost response must not cause duplicate folders.
                row = self.api.files().get(fileId=file_id, fields='id').execute(num_retries=5)
        self.folders[relative] = row['id']
        return row['id']

    def put(self, local, remote):
        from googleapiclient.http import MediaFileUpload
        local = Path(local)
        parent, _, name = str(remote).rpartition('/')
        folder = self.folder(parent)
        current = self.find(folder, name)
        expected = hashlib.md5(local.read_bytes()).hexdigest()
        if current and current.get('md5Checksum') == expected:
            return current['id']
        media = MediaFileUpload(str(local), resumable=True, chunksize=4 << 20)
        if current:
            row = self.api.files().update(fileId=current['id'], media_body=media,
                fields='id,md5Checksum,size').execute(num_retries=5)
        else:
            file_id = self.api.files().generateIds(count=1).execute(num_retries=5)['ids'][0]
            try:
                row = self.api.files().create(body=dict(id=file_id, name=name, parents=[folder]),
                    media_body=media, fields='id,md5Checksum,size').execute(num_retries=0)
            except Exception:
                row = self.api.files().get(fileId=file_id, fields='id,md5Checksum,size').execute(num_retries=5)
                if row.get('md5Checksum') != expected:
                    row = self.api.files().update(fileId=file_id, media_body=media,
                        fields='id,md5Checksum,size').execute(num_retries=5)
        if row.get('md5Checksum') != expected or int(row.get('size', -1)) != local.stat().st_size:
            raise IOError('Drive upload checksum mismatch')
        return row['id']

    def download(self, file_id, destination, md5=None):
        from googleapiclient.http import MediaIoBaseDownload
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(destination.name + '.download')
        with temp.open('wb') as f:
            dl = MediaIoBaseDownload(f, self.api.files().get_media(fileId=file_id), chunksize=4 << 20)
            done = False
            while not done:
                _, done = dl.next_chunk(num_retries=5)
        if md5 and hashlib.md5(temp.read_bytes()).hexdigest() != md5:
            raise IOError('Drive download checksum mismatch')
        temp.replace(destination)

    def restore_tree(self, remote, local):
        folder = self.folder(remote, create=False)
        if folder is None:
            return
        local = Path(local)
        for row in self.children(folder):
            if Path(row['name']).name != row['name']:
                raise ValueError('Unsafe Drive artifact name')
            child = local / row['name']
            if row['mimeType'] == 'application/vnd.google-apps.folder':
                self.restore_tree(remote + '/' + row['name'], child)
            elif not child.exists():
                self.download(row['id'], child, row.get('md5Checksum'))
            elif hashlib.md5(child.read_bytes()).hexdigest() != row.get('md5Checksum'):
                raise ValueError(f'Local/Drive conflict: {child}')


def run_command(command, **kwargs):
    print('COMMAND', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def verify_sources(root):
    root = Path(root)
    lock = read_json(root / 'author' / 'SOURCE_LOCK.json')
    for method, info in lock.items():
        for relative, expected in info['files'].items():
            if sha(root / 'author' / method / relative) != expected:
                raise ValueError(f'Author file changed: {method}/{relative}')
    return lock


def prepare_data(root, cfg, drive):
    root = Path(root)
    archive = Path(cfg['data_archive']) if cfg.get('data_archive') else root / 'attack_data_n200.zip'
    if not archive.is_file():
        if not drive or not cfg.get('data_drive_file_id'):
            raise FileNotFoundError('Set DATA_ARCHIVE or DATA_DRIVE_FILE_ID.')
        drive.download(cfg['data_drive_file_id'], archive)
    if sha(archive) != cfg['data_sha256']:
        raise ValueError('Dataset SHA-256 mismatch')
    extract(archive, root / 'data')
    rows = [json.loads(s) for s in (root / 'data' / 'pairs_200.jsonl').read_text().splitlines() if s.strip()]
    if not 0 < cfg['n'] <= len(rows):
        raise ValueError(f'Requested N={cfg["n"]}; available={len(rows)}')
    rows = rows[:cfg['n']]
    if len({r['index'] for r in rows}) != len(rows):
        raise ValueError('Duplicate pair indices')
    from PIL import Image
    captions = {Path(r['file_name']).stem: r['caption'] for r in
        map(json.loads, (root / 'author/saga/input/coco.jsonl').read_text().splitlines())}
    records = []
    for r in rows:
        stem = Path(r['source']).stem
        if Path(r['target']).stem != stem or stem not in captions:
            raise ValueError(f'Source/target ID or SAGA caption mismatch: {stem}')
        info = dict(r, stem=stem, saga_target=captions[stem])
        for role in ('source', 'target'):
            source = (root / 'data' / r[role]).resolve()
            if not source.is_relative_to((root / 'data').resolve()):
                raise ValueError('Dataset path traversal')
            with Image.open(source) as im:
                im.verify()
            info[role + '_sha256'] = sha(source)
            # Preserve original bytes, extension and author's transforms.
            dest = root / 'inputs' / role / 'images' / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and sha(dest) != sha(source):
                raise ValueError('Input changed')
            if not dest.exists():
                shutil.copyfile(source, dest)
        records.append(info)
    write_json(root / 'selected_pairs.json', records)
    return records


def interpreter(root, kind):
    root = Path(root)
    uv = shutil.which('uv')
    if uv is None:
        run_command([sys.executable, '-m', 'pip', 'install', 'uv'])
        uv = shutil.which('uv')
    pyver = '3.10' if kind in ('foa', 'mpc') else '3.11'
    env = root.parent / 'environments' / kind
    executable = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    stamp = env / 'baseline-ready-v2.json'
    if stamp.exists():
        return executable
    run_command([uv, 'python', 'install', pyver])
    if not executable.is_file():
        run_command([uv, 'venv', '--python', pyver, env])
    run_command([uv, 'pip', 'install', '--python', executable, 'torch==2.8.0',
        'torchvision==0.23.0', '--index-url', 'https://download.pytorch.org/whl/cu128'])
    specs = ['transformers==4.57.1', 'accelerate==1.11.0', 'pillow==11.3.0',
        'tqdm==4.67.1', 'qwen-vl-utils==0.0.14', 'google-api-python-client',
        'pydata-google-auth', 'python-dotenv', 'sentencepiece==0.2.1']
    specs += ['numpy==2.2.4', 'loguru==0.7.3'] if kind == 'saga' else ['numpy==1.26.4']
    if kind in ('foa', 'mpc'):
        specs += ['hydra-core==1.3.2', 'wandb==0.19.6', 'scipy==1.15.3',
            'kmeans-pytorch @ git+https://github.com/subhadarship/kmeans_pytorch.git@f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805',
            'POT==0.9.6.post1', 'numba==0.63.1']
    if kind == 'foa':
        specs += ['pytorch-lightning==2.5.0.post0', 'torchmetrics==1.6.1']
    if kind == 'evaluation':
        specs += ['openai>=2.20,<3', 'google-genai>=1.41,<2', 'tenacity>=9,<10', 'bitsandbytes>=0.48,<1']
    run_command([uv, 'pip', 'install', '--python', executable, *specs])
    freeze = subprocess.check_output([uv, 'pip', 'freeze', '--python', str(executable)], text=True)
    write_json(stamp, dict(python=pyver, packages=freeze.splitlines(), torch_runtime='CUDA 12.8 for Blackwell'))
    return executable


def validate_config(cfg):
    if set(cfg['methods']) - {'foa', 'mpc', 'saga'} or not cfg['methods']:
        raise ValueError('Select foa, mpc, saga')
    if not 0 <= cfg['epsilon'] <= 255 or cfg['steps'] < 1 or cfg['alpha'] <= 0:
        raise ValueError('Invalid epsilon, steps, alpha')
    if 'mpc' in cfg['methods'] and cfg['steps'] != 300:
        raise ValueError('Original MPC ignores --steps and hardcodes range(300). Choose 300 or disable MPC.')
    if cfg['backup_drive'] and not cfg.get('drive_parent_id'):
        raise ValueError('Set Drive parent folder ID')
    if Path(cfg['run_name']).name != cfg['run_name'] or cfg['run_name'] in ('.', '..', ''):
        raise ValueError('RUN_NAME must be one directory name')


def fingerprint(root, cfg):
    # Transport/account/storage location does not alter the experiment identity.
    keys = ('n', 'epsilon', 'steps', 'alpha', 'methods', 'mpc', 'saga', 'model_revisions')
    identity = dict(config={k: cfg[k] for k in keys}, sources=verify_sources(root),
        pairs=read_json(Path(root) / 'selected_pairs.json'),
        workers={p.name: sha(p) for p in Path(root).glob('*worker.py')})
    identity['runtime'] = sha(Path(root) / 'baseline_runtime.py')
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(), identity


def stage(root, name, **extra):
    write_json(Path(root) / 'status.json', dict(stage=name, updated_unix=time.time(), **extra))
    print('STAGE', name, json.dumps(extra), flush=True)


def rebuild_manifests(root, cfg, run_id):
    """A fully restored attack must also be evaluable without rerunning its models."""
    root = Path(root)
    expected = {p['index'] for p in read_json(root / 'selected_pairs.json')}
    for method in cfg['methods']:
        directory = root / 'results' / method
        rows = []
        for marker in sorted((directory / 'samples').glob('*/complete.json')):
            record = read_json(marker)
            if record['run_id'] != run_id or record['index'] not in expected:
                raise ValueError('Restored checkpoint belongs to a different experiment')
            for name, checksum in record['artifacts_sha256'].items():
                if Path(name).name != name or sha(marker.parent / name) != checksum:
                    raise ValueError('Corrupt restored sample')
            base = marker.parent.relative_to(directory).as_posix() + '/'
            rows.append(dict(index=record['index'], method=method,
                **{role: base + role + '.png' for role in ('source', 'target', 'adversarial')}))
        if rows:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def controller(root, cfg):
    """One controller per run; attack processes finish before evaluation loads 8B VLMs."""
    import fcntl
    root = Path(root).resolve()
    lock = (root / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    validate_config(cfg)
    drive = Drive(cfg['drive_parent_id']) if cfg['backup_drive'] else None
    prefix = cfg['drive_prefix'].strip('/') + '/' + cfg['run_name']
    stage(root, 'preparing_data')
    pairs = prepare_data(root, cfg, drive)
    run_id, identity = fingerprint(root, cfg)
    identity_file = root / 'identity.json'
    if identity_file.exists() and read_json(identity_file)['fingerprint'] != run_id:
        raise ValueError('Experiment identity changed. Choose a new RUN_NAME.')
    write_json(identity_file, dict(fingerprint=run_id, identity=identity))
    if drive:
        # Check remote identity before accepting any checkpoint.
        remote_folder = drive.folder(prefix)
        row = drive.find(remote_folder, 'identity.json')
        if row:
            remote_identity = root / 'remote_identity.json'
            drive.download(row['id'], remote_identity, row.get('md5Checksum'))
            if read_json(remote_identity)['fingerprint'] != run_id:
                raise ValueError('Drive run identity differs. Choose a new RUN_NAME.')
        drive.put(identity_file, prefix + '/identity.json')
        for method in cfg['methods']:
            drive.restore_tree(prefix + '/checkpoints/' + method, root / 'checkpoints' / method)
            for archive in sorted((root / 'checkpoints' / method).glob('*.zip')):
                extract(archive, root)
    rebuild_manifests(root, cfg, run_id)
    deadline = time.time() + cfg['session_hours'] * 3600
    child_env = os.environ.copy()
    child_env.update(WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS=str(cfg['cpu_threads']), MKL_NUM_THREADS=str(cfg['cpu_threads']),
        HF_HOME=str(root.parent / 'hf_cache'), BASELINE_DEADLINE=str(deadline),
        HF_HUB_DISABLE_XET='1')
    (root / 'logs').mkdir(exist_ok=True)
    for method in cfg['methods']:
        count = len(list((root / 'results' / method / 'samples').glob('*/complete.json')))
        if count == len(pairs):
            stage(root, 'attack_complete', method=method, completed=count)
            continue
        stage(root, 'installing', method=method)
        exe = interpreter(root, method)
        stage(root, 'attacking', method=method, completed=count, total=len(pairs))
        with (root / 'logs' / f'{method}.log').open('a') as log:
            result = subprocess.run([str(exe), '-u', str(root / 'author_worker.py'), str(root), method],
                env=child_env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            stage(root, 'paused_session' if result.returncode == 75 else 'attack_failed',
                method=method, exit_code=result.returncode)
            return result.returncode
    pending_models = []
    if cfg['run_evaluation']:
        stage(root, 'installing_evaluation')
        exe = interpreter(root, 'evaluation')
        for model in cfg['evaluation']['models']:
            stage(root, 'evaluating', model=model)
            with (root / 'logs/evaluation.log').open('a') as log:
                result = subprocess.run([str(exe), '-u', str(root / 'eval_worker.py'), str(root), model],
                    env=child_env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 75:
                stage(root, 'paused_session', model=model)
                return 75
            if result.returncode:
                pending_models.append(dict(model=model, exit_code=result.returncode))
    stage(root, 'evaluation_pending' if pending_models else 'complete' if cfg['run_evaluation']
        else 'attacks_complete', total=len(pairs), pending_models=pending_models)
    if drive:
        drive.put(root / 'status.json', prefix + '/status.json')
    return 76 if pending_models else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    root = args.config.resolve().parent
    try:
        return controller(root, read_json(args.config))
    except Exception as exc:
        # Provider credentials or request headers must never be written into notebook output.
        stage(root, 'failed', error_type=type(exc).__name__,
            detail=str(exc)[:500] if isinstance(exc, (ValueError, FileNotFoundError)) else 'See private process logs.')
        raise


if __name__ == '__main__':
    sys.exit(main())
