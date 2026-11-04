"""Install the reviewed environment and invoke readable source, never notebook cells."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time


def main():
    package = Path(__file__).resolve().parents[2]
    root = Path('/marimo/CARA641-20261027-N2-4V-v1')
    root.mkdir(parents=True, exist_ok=True)
    clock = root / 'timing.json'
    if not clock.exists():
        clock.write_text(json.dumps({'started': time.time(), 'gpu_started': time.time(),
                                     'disk_used_initial': shutil.disk_usage(root).used}))
    clocks = json.loads(clock.read_text())
    deadline = min(clocks['started'] + 3 * 3600, clocks['gpu_started'] + 2 * 3600)
    uv = shutil.which('uv')
    if not uv:
        raise RuntimeError('MoLab uv binary unavailable')
    environment = root.parent / 'cara641_env311'
    python = environment / 'bin/python'
    def run(command, **kwargs):
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError('Reviewed setup budget exhausted')
        process = subprocess.Popen(command, **kwargs)
        try:
            while process.poll() is None:
                if time.time() >= deadline:
                    raise TimeoutError('Reviewed cumulative GPU/wall deadline exhausted')
                if shutil.disk_usage(root).used - clocks['disk_used_initial'] > 150 * 2**30:
                    raise RuntimeError('Reviewed 150GiB disk growth budget exhausted')
                time.sleep(2)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException as error:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            (root / 'setup_watchdog.json').write_text(json.dumps({'error_type': type(error).__name__,
                'time': time.time(), 'returncode': process.returncode}))
            raise
    if not python.exists():
        run([uv, 'venv', '--python', '3.11', str(environment)])
    run([uv, 'pip', 'install', '--python', str(python), 'torch==2.8.0', 'torchvision==0.23.0',
         '--index-url', 'https://download.pytorch.org/whl/cu128'])
    run([uv, 'pip', 'install', '--python', str(python), '-r', str(package / 'src/cara_v641/requirements.txt')])
    with (root / 'source.zip').open('wb') as output:
        output.write((package / 'source.zip').read_bytes())
    run([str(python), '-u', '-m', 'src.cara_v641', 'remote', '--root', str(root),
         '--package', str(package)], cwd=package,
        env={**os.environ, 'MPLBACKEND': 'Agg', 'TOKENIZERS_PARALLELISM': 'false'})


if __name__ == '__main__':
    main()
