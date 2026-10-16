"""Resume the remaining campaign with independent job workers and queued backups.

Preserves the original attack driver, its identity, and FOA sample order. GPU job
concurrency is explicit: workers=2 is a benchmark candidate, not image batching.
Finish the priority evaluation before starting this launcher on the same GPU.
"""
import argparse
import builtins
from contextlib import redirect_stdout
import importlib.util
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid
from zipfile import ZipFile, ZIP_DEFLATED


def load_driver(root):
    spec = importlib.util.spec_from_file_location('original_campaign', root / 'molab_hypothesis_campaign.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def merged_state(root, module):
    state = module.read_json(root / 'campaign_state.json')
    for p in sorted((root / 'parallel_tasks').glob('*/campaign_state.json')):
        other = module.read_json(p)
        for section in ('attack_runs', 'evaluations', 'eval_model_revisions'):
            for key, value in other.get(section, {}).items():
                existing = state.setdefault(section, {}).get(key)
                if existing is not None and existing != value:
                    raise ValueError(f'Conflicting task state: {section}/{key}')
                state[section][key] = value
    return state


class QueuedWriter:
    def __init__(self, target):
        self.target = target

    def write(self, message):
        return self.target.write(message.replace('DURABLE SAMPLE:', 'LOCAL SAMPLE (backup queued):'))

    def flush(self):
        return self.target.flush()


def task_worker(root, name, prefix, hours, stop_after_variant=None, pause_file=None):
    module = load_driver(root)
    plan = module.read_json(root / 'molab_hypotheses_n200.json')
    job = next(j for j in plan['attack_jobs'] if j['name'] == name)
    if stop_after_variant and stop_after_variant not in job['variants']:
        raise ValueError('Stop variant must belong to this job')
    pause_file = Path(pause_file) if pause_file else None
    if stop_after_variant or pause_file:
        def execute_with_boundary(code, namespace):
            builtins.exec(code, namespace)
            if 'run_experiment' not in namespace or namespace.get('_variant_boundary_installed'):
                return
            namespace['_variant_boundary_installed'] = True
            original_write = namespace['write_json']

            def boundary_write(path, value):
                original_write(path, value)
                path = Path(path)
                run = namespace.get('RUN_DIR')
                if not run or path.name != 'status.json' or path.parent.parent != Path(run):
                    return
                if path.parent.name not in job['variants']:
                    return
                if pause_file and pause_file.exists() and value.get('status') in ('running', 'complete'):
                    pause_file.unlink(missing_ok=True)
                    raise module.BoundaryPause('GPU handoff after a complete sample and rebuilt manifest')
                if path.parent.name == stop_after_variant and value.get('status') == 'complete':
                    raise module.BoundaryPause('Ablation complete; hand off immediately to evaluation')

            namespace['write_json'] = boundary_write
        # Operational hook only. Original notebook cells, full RUN_VARIANTS,
        # run identity, numerical updates and prefix resume checks remain intact.
        module.exec = execute_with_boundary
    state = merged_state(root, module)
    task_root = root / 'parallel_tasks' / name
    task_root.mkdir(parents=True, exist_ok=True)
    state_path = task_root / 'campaign_state.json'
    module.write_json(state_path, state)
    original_backup = module.DriveBackup
    instances = []

    class AsyncBackup(original_backup):
        def __init__(self, *args, **kwargs):
            self.network_lock = threading.Lock()
            super().__init__(*args, **kwargs)
            self.pending = queue.Queue(maxsize=2)
            self.failure = None
            self.thread = threading.Thread(target=self.consume, daemon=True)
            self.thread.start()
            instances.append(self)

        def put(self, local, relative):
            with self.network_lock:
                return super().put(local, relative)

        def check(self):
            if self.failure:
                raise module.BackupFailure('Background backup failed; stopping before another sample.')

        def consume(self):
            while True:
                item = self.pending.get()
                try:
                    if item is None:
                        return
                    archive, relative = item
                    if self.failure is None:
                        self.put(archive, relative)
                        archive.unlink()
                except BaseException as exc:
                    self.failure = type(exc).__name__
                finally:
                    self.pending.task_done()

        def enqueue(self, archive, relative):
            self.check()
            while True:
                try:
                    self.pending.put((archive, relative), timeout=1)
                    break
                except queue.Full:
                    self.check()
            self.check()

        def snapshot(self, run, sample=None):
            self.check()
            run = Path(run)
            staging = task_root / 'immutable_snapshots'
            staging.mkdir(exist_ok=True)
            if sample is not None:
                sample = Path(sample)
                archive = staging / (uuid.uuid4().hex + '.zip')
                with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
                    for p in sorted(sample.rglob('*')):
                        if p.is_file() and not p.name.endswith('.tmp'):
                            z.write(p, p.relative_to(run).as_posix())
                self.enqueue(archive, sample.relative_to(run).as_posix() + '.zip')
            archive = staging / (uuid.uuid4().hex + '.zip')
            with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
                for p in sorted(run.rglob('*')):
                    if p.is_file() and 'samples' not in p.relative_to(run).parts and not p.name.endswith('.tmp'):
                        z.write(p, p.relative_to(run).as_posix())
            self.enqueue(archive, 'root_latest.zip')

        def close(self):
            self.pending.join()
            self.pending.put(None)
            self.thread.join()
            self.check()
            print('DRIVE BACKUP DRAINED AND VERIFIED:', name, flush=True)

    module.DriveBackup = AsyncBackup
    try:
        with redirect_stdout(QueuedWriter(sys.stdout)):
            return module.attack_worker(root, plan, job, state_path, prefix, hours)
    finally:
        try:
            saved = module.read_json(state_path)
            run = saved.get('attack_runs', {}).get(name)
            if run:
                module.write_json(Path(run) / 'parallel_execution_provenance.json', {
                    'launcher_sha256': module.digest(__file__), 'job': name,
                    'sample_batch_size': 1, 'backup': 'bounded_queue_2_immutable_archives_checksum_readback',
                    'original_driver_sha256': module.digest(root / 'molab_hypothesis_campaign.py'),
                    'stop_after_variant': stop_after_variant, 'pause_file_enabled': pause_file is not None,
                    'algorithm': 'unchanged; preserve per-variant FOA/RNG state and image order'})
                for instance in instances:
                    instance.snapshot(run)
        finally:
            for instance in instances:
                instance.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['list', 'run', '_task'])
    parser.add_argument('--root', type=Path, default=Path('/marimo/hypotheses_n200'))
    parser.add_argument('--backup-prefix', default='KLTN_Molab_Hypotheses_n200_v432_h1/results')
    parser.add_argument('--workers', type=int, choices=[1, 2], default=1)
    parser.add_argument('--max-hours', type=float, default=8)
    parser.add_argument('--task')
    parser.add_argument('--stop-after-variant')
    parser.add_argument('--pause-file', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    module = load_driver(root)
    if args.max_hours <= 0:
        parser.error('--max-hours must be positive')
    if args.mode == '_task':
        return task_worker(root, args.task, args.backup_prefix, args.max_hours,
                           args.stop_after_variant, args.pause_file)
    state = merged_state(root, module)
    plan = module.read_json(root / 'molab_hypotheses_n200.json')
    jobs = []
    for job in plan['attack_jobs']:
        run = state['attack_runs'].get(job['name'])
        status = Path(run) / 'status.json' if run else None
        complete = status and status.exists() and module.read_json(status).get('status') == 'complete'
        print(job['name'], 'complete' if complete else 'pending/partial', flush=True)
        if not complete:
            jobs.append(job)
    if args.mode == 'list':
        return 0
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Enable the Molab GPU')
    if args.workers == 2 and torch.cuda.get_device_properties(0).total_memory < 80 * 2**30:
        raise RuntimeError('Two-worker benchmark is restricted to >=80 GiB VRAM; use --workers 1')
    print('Job concurrency:', args.workers, '| Per-variant image batch: 1 (preserves FOA semantics)', flush=True)
    print('Do not run the legacy attack launcher or evaluation concurrently with these jobs.', flush=True)
    exe = root / '.venvs/attack' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not exe.is_file():
        raise RuntimeError('Restore the original campaign and run its setup first')
    started = time.monotonic()
    pending = list(jobs)
    active = []
    failed = False
    paused = False
    last_report = 0.0
    try:
        while pending or active:
            remaining = args.max_hours - (time.monotonic() - started) / 3600
            while pending and len(active) < args.workers and remaining > 0 and not failed:
                job = pending.pop(0)
                task_dir = root / 'parallel_tasks' / job['name']
                task_dir.mkdir(parents=True, exist_ok=True)
                log = (task_dir / 'console.log').open('a', encoding='utf-8')
                command = [str(exe), '-u', str(Path(__file__).resolve()), '_task', '--root', str(root),
                           '--task', job['name'], '--max-hours', str(remaining),
                           '--backup-prefix', args.backup_prefix]
                p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                active.append((p, job, log))
                print('STARTED', job['name'], 'PID', p.pid, '| log', log.name, flush=True)
            for p, job, log in active[:]:
                if p.poll() is not None:
                    code = p.returncode
                    log.close()
                    active.remove((p, job, log))
                    failed = failed or code not in (0, 75)
                    paused = paused or code == 75
                    print('FINISHED', job['name'], 'exit', code, flush=True)
            if failed or remaining <= 0:
                pending.clear()
                paused = paused or remaining <= 0
            if active and time.monotonic() - last_report >= 60:
                for _, job, _ in active:
                    state_file = root / 'parallel_tasks' / job['name'] / 'campaign_state.json'
                    task_state = module.read_json(state_file) if state_file.exists() else {}
                    run_path = task_state.get('attack_runs', {}).get(job['name'])
                    count = len(list(Path(run_path).glob('*/samples/*/complete.json'))) if run_path else 0
                    print('PROGRESS', job['name'], 'local completed samples', count, flush=True)
                telemetry = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                                            '--format=csv,noheader'], capture_output=True, text=True)
                print('GPU util / MiB / W:', telemetry.stdout.strip(), flush=True)
                last_report = time.monotonic()
            if active:
                time.sleep(5)
        module.write_json(root / 'campaign_state.json', merged_state(root, module))
        print('Task states merged. Durable backups can be restored with the original package.', flush=True)
        return 1 if failed else (75 if paused else 0)
    finally:
        for p, _, log in active:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=30)
            log.close()


if __name__ == '__main__':
    raise SystemExit(main())
