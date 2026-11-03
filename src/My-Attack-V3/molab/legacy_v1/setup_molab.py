"""Install locked dependencies and run the approved source under a hard watchdog."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/marimo/MY-ATTACK-V3-20261102-N2-v1'))
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    clock = root / 'clock.json'
    if not clock.exists():
        clock.write_text(json.dumps({'started': time.time(), 'disk_initial': shutil.disk_usage(root).used}))
    identity = json.loads(clock.read_text())
    deadline = identity['started'] + 7200
    uv = shutil.which('uv')
    if not uv:
        raise RuntimeError('MoLab uv executable unavailable')
    environment = Path('/marimo/my_attack_v3_env')
    python = environment / 'bin/python'

    def run(command, **kwargs):
        if time.time() >= deadline:
            raise TimeoutError('Approved cumulative setup/compute budget exhausted')
        process = subprocess.Popen(command, start_new_session=True, **kwargs)
        try:
            while process.poll() is None:
                if time.time() >= deadline - 120:
                    raise TimeoutError('Approved cumulative setup/compute budget exhausted')
                if shutil.disk_usage(root).used - identity.get('disk_initial', 0) > 120 * 2**30:
                    raise RuntimeError('Approved 120 GiB disk budget exceeded')
                time.sleep(2)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            raise

    try:
        if not python.exists():
            run([uv, 'venv', '--python', '3.11', str(environment)])
        run([uv, 'pip', 'install', '--python', str(python), 'torch==2.8.0', 'torchvision==0.23.0',
             '--index-url', 'https://download.pytorch.org/whl/cu128'])
        run([uv, 'pip', 'install', '--python', str(python), '-r', str(project / 'requirements.txt')])
        archive = project.parent / 'my_attack_v3_source.zip'
        if archive.exists():
            shutil.copy2(archive, root / 'source.zip')
        # This preflight only uses random tensors and fake generators/transports.
        run([str(python), '-X', 'utf8', '-m', 'unittest', 'discover', '-s', str(project), '-p', 'test_*.py', '-v'], cwd=project)
        run([str(python), '-u', str(project / 'run_pipeline.py'), '--root', str(root)], cwd=project,
            env={**os.environ, 'TOKENIZERS_PARALLELISM': 'false'})
    except BaseException as error:
        (root / 'setup_error.json').write_text(json.dumps({'type': type(error).__name__, 'error': str(error),
                                                         'elapsed_seconds': time.time() - identity['started']}))
        raise


if __name__ == '__main__':
    main()
