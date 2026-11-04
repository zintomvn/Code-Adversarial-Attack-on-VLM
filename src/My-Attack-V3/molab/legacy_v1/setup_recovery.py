"""Foreground recovery bootstrap with an allocation-based hard deadline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile

from recovery_state import restore_snapshot, verify_source_identity

SNAPSHOT_ID = '10vd8BcTOGi4_g66S4UI1fpuFMU17hvm8'
SNAPSHOT_SHA = '2d806181ba2d70be0212a2569e81a9985c415a7ca42af73bc6edcef0266ac9e0'
POINTER_ID = '16lG6bSCgCABiF_ty655cBTEh9OgJ5NZ0'


def remaining_seconds(deadline, now=None):
    remaining = deadline - (time.time() if now is None else now)
    if remaining <= 0:
        raise TimeoutError('Approved recovery allocation deadline reached')
    return remaining


def freeze_command(uv, python):
    return [uv, 'pip', 'freeze', '--python', str(python)]


def require_source_archive(path, project=None):
    if not path.is_file():
        raise RuntimeError('Recovery source archive must be present before launch')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Recovery source archive CRC failure')
        if project is not None:
            for name in archive.namelist():
                relative = Path(name).relative_to('My-Attack-V3')
                if archive.read(name) != (project / relative).read_bytes():
                    raise RuntimeError(f'Recovery archive/source mismatch: {relative}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(file_id, deadline):
    with urllib.request.urlopen(f'https://drive.google.com/uc?export=download&id={file_id}',
                                timeout=min(90, remaining_seconds(deadline))) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation-start', type=float, required=True)
    parser.add_argument('--root', type=Path, default=Path('/marimo/MY-ATTACK-V3-20261102-N2-v1'))
    args = parser.parse_args()
    root, project = args.root, Path(__file__).resolve().parent
    deadline = args.allocation_start + 4500
    def hard_deadline(signum, frame):
        raise TimeoutError('Hard recovery allocation deadline reached')
    signal.signal(signal.SIGALRM, hard_deadline)
    signal.setitimer(signal.ITIMER_REAL, remaining_seconds(deadline))
    archive = project.parent / 'my_attack_v3_recovery1_source.zip'
    archive_sha = require_source_archive(archive, project)
    pointer = json.loads(download(POINTER_ID, deadline))
    if pointer['sequence'] != 59 or pointer['sha256'] != SNAPSHOT_SHA:
        raise RuntimeError('Remote run changed; refusing duplicate recovery')
    restore_snapshot(download(SNAPSHOT_ID, deadline), pointer, root)
    verify_source_identity(project, json.loads((root / 'source_manifest.json').read_text()))
    recovery = root / 'recovery1'
    recovery.mkdir()
    clock = {'started': args.allocation_start, 'setup_started': time.time(),
             'disk_initial': shutil.disk_usage(root).used, 'max_allocation_seconds': 4500,
             'approval': 'User2026-11-03: run attack, approving published recovery plan',
             'source_snapshot_sha256': SNAPSHOT_SHA, 'continuation_id': 'recovery1',
             'recovery_source_sha256': archive_sha}
    (recovery / 'clock.json').write_text(json.dumps(clock, indent=2))
    packages = json.loads((root / 'runtime.json').read_text())['packages']
    (recovery / 'requirements_frozen.txt').write_text('\n'.join(packages) + '\n')
    shutil.copy2(archive, recovery / 'source.zip')
    environment = Path('/marimo/my_attack_v3_recovery1_env')
    python = environment / 'bin/python'
    uv = shutil.which('uv')
    if not uv:
        raise RuntimeError('MoLab uv executable unavailable')

    def run(command, **kwargs):
        if time.time() >= deadline - 120:
            raise TimeoutError('Approved recovery allocation limit reached')
        process = subprocess.Popen(command, start_new_session=True, **kwargs)
        try:
            while process.poll() is None:
                if time.time() >= deadline - 120:
                    raise TimeoutError('Recovery allocation limit reached; reserve export time')
                if shutil.disk_usage(root).used - clock['disk_initial'] > 120 * 2**30:
                    raise RuntimeError('Reviewed120GiB disk limit exceeded')
                time.sleep(2)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=max(1, min(120, deadline - time.time())))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            raise

    try:
        run([uv, 'venv', '--python', '3.11.15', str(environment)])
        run([uv, 'pip', 'install', '--python', str(python), '-r', str(recovery / 'requirements_frozen.txt'),
             '--extra-index-url', 'https://download.pytorch.org/whl/cu128', '--index-strategy', 'unsafe-best-match'])
        installed = subprocess.check_output(freeze_command(uv, python), text=True,
                                            timeout=remaining_seconds(deadline)).splitlines()
        if sorted(installed) != sorted(packages):
            raise RuntimeError('Recovery environment differs from the recorded v1 freeze')
        (recovery / 'packages_verified.json').write_text(json.dumps(installed, indent=2))
        run([str(python), '-X', 'utf8', '-m', 'unittest', 'discover', '-s', str(project), '-p', 'test_*.py', '-v'], cwd=project)
        run([str(python), '-u', str(project / 'recover_pipeline.py'), '--root', str(root)], cwd=project,
            env={**os.environ, 'TOKENIZERS_PARALLELISM': 'false'})
    except BaseException as error:
        (recovery / 'setup_error.json').write_text(json.dumps({'type': type(error).__name__, 'error': str(error),
                                                            'allocation_elapsed': time.time() - args.allocation_start}))
        raise


if __name__ == '__main__':
    main()
