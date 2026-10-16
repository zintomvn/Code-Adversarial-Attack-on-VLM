"""Interleave ablations with immediate batched captions and asynchronous GPT judging.

Original attack/evaluation engines are frozen. The controller only schedules
independent processes and hands GPU ownership over at flushed sample boundaries.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configurations(plan):
    return [{'id': job['name'] + '__' + variant, 'job': job['name'], 'variant': variant}
            for job in plan['attack_jobs'] for variant in job['variants']]


def complete_count(run, variant):
    path = Path(run) / variant / 'manifest.jsonl' if run else None
    if not path or not path.is_file():
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    indices = [r['index'] for r in rows]
    if indices != list(range(len(rows))):
        raise ValueError('Attack completion must be an ordered prefix')
    return len(rows)


def pending_attacks(configs, state, active_jobs, failed_jobs, expected=200):
    pending, seen = [], set()
    for config in configs:
        job = config['job']
        if job in seen or job in active_jobs or job in failed_jobs:
            continue
        if complete_count(state['attack_runs'].get(job), config['variant']) < expected:
            pending.append(config)
            seen.add(job)
    return pending


def fresh_environment():
    env = os.environ.copy()
    try:
        from dotenv import dotenv_values
        key = dotenv_values('/marimo/.env').get('OPENAI_API_KEY')
        if key:
            env['OPENAI_API_KEY'] = key
    except ImportError:
        pass
    return env


def factual_captions_ready(state, expected):
    run = state.get('attack_runs', {}).get('primary_ablations')
    path = Path(run) / 'text_records.json' if run else None
    if not path or not path.is_file():
        return False
    records = json.loads(path.read_text(encoding='utf-8'))
    return len(records) == expected and all(row.get('source') and row.get('target') and not row.get('caption_error')
                                            for row in records.values())


def process_state(pid):
    path = Path('/proc') / str(pid) / 'stat'
    try:
        content = path.read_text()
    except FileNotFoundError:
        return None
    fields = content[content.rfind(')') + 2:].split()
    return {'state': fields[0], 'parent': int(fields[1]), 'start_ticks': fields[19]}


class AdoptedProcess:
    """Observe a worker surviving controller replacement without relaunching it."""
    def __init__(self, pid, start_ticks, result):
        self.pid, self.start_ticks, self.result = pid, start_ticks, result

    def poll(self):
        state = process_state(self.pid)
        if state and state['start_ticks'] == self.start_ticks and state['state'] != 'Z':
            return None
        return self.result()


class Controller:
    def __init__(self, root, eval_root, control_root, hours, workers, batch_size, backup_prefix, eval_backup_prefix):
        self.root, self.eval_root, self.control_root = root, eval_root, control_root
        control_root.mkdir(parents=True, exist_ok=True)
        self.attack = load(root / 'molab_hypothesis_campaign.py', 'frozen_campaign')
        self.tasks = load(root / 'molab_remaining_tasks.py', 'task_launcher')
        self.eval = load(eval_root / 'molab_eval_completed.py', 'completed_eval')
        self.plan = self.attack.read_json(root / 'molab_hypotheses_n200.json')
        self.configs = configurations(self.plan)
        self.deadline = time.monotonic() + hours * 3600
        self.workers, self.batch_size = workers, batch_size
        self.backup_prefix = backup_prefix
        self.eval_backup_prefix = eval_backup_prefix
        self.active = {}
        self.judge = None
        self.failed_jobs = set()
        self.caption_failures = set()
        self.judge_deferred_until = 0.
        self.started = time.time()
        self.last_report = 0.
        self.fs = self.eval.drive_fs()
        self.state_path = control_root / 'controller_state.json'

    def remaining(self):
        return max(0., (self.deadline - time.monotonic()) / 3600)

    def adopt(self, path):
        saved = self.attack.read_json(path)
        for item in saved['workers']:
            config = next(c for c in self.configs if c['id'] == item['config_id'])
            state = process_state(item['pid'])
            if not state or state['state'] == 'Z' or state['start_ticks'] != item['start_ticks']:
                raise RuntimeError('Adopted worker no longer matches its recorded process identity')
            directory = self.root / 'parallel_tasks' / config['job']
            log_path = directory / 'overnight_attack.log'
            def result(config=config, log_path=log_path):
                run = self.attack_state()['attack_runs'].get(config['job'])
                if complete_count(run, config['variant']) == self.plan['num_samples']:
                    return 0
                text = log_path.read_text(encoding='utf-8', errors='replace')[-10000:]
                if 'DRIVE BACKUP DRAINED AND VERIFIED' in text and 'PAUSED AT CHECKPOINT:' in text:
                    return 75
                return 1
            self.active[config['job']] = {
                'process': AdoptedProcess(item['pid'], item['start_ticks'], result),
                'config': config, 'log': log_path.open('a', encoding='utf-8'),
                'pause': directory / 'gpu_handoff.request', 'started': time.monotonic() - 120}
            print('ADOPTED EXISTING WORKER', config['id'], 'PID', item['pid'], flush=True)

    def attack_state(self):
        state = self.tasks.merged_state(self.root, self.attack)
        self.attack.write_json(self.root / 'campaign_state.json', state)
        return state

    def evaluation_state(self):
        path = self.eval_root / 'evaluation_state.json'
        return self.eval.read_json(path) if path.exists() else {'tasks': {}, 'model_revisions': {}}

    def task_status(self, config, model, stage):
        task = config['id'] + '__' + model
        run = self.evaluation_state()['tasks'].get(task)
        file = Path(run) / ('status.json' if stage == 'caption' else 'evaluation/status.json') if run else None
        return self.eval.read_json(file).get('status') if file and file.exists() else 'pending'

    def all_stage_complete(self, config, stage):
        return all(self.task_status(config, model, stage) == 'complete'
                   for model in self.plan['evaluation']['models'])

    def ready_for_captions(self):
        state = self.attack_state()
        return [c for c in self.configs
                if complete_count(state['attack_runs'].get(c['job']), c['variant']) == self.plan['num_samples']
                and not self.all_stage_complete(c, 'caption') and c['id'] not in self.caption_failures]

    def ready_for_judge(self):
        return [c for c in self.configs if self.all_stage_complete(c, 'caption')
                and not self.all_stage_complete(c, 'judge')]

    def record(self, event):
        state = self.attack_state()
        report = {'event': event, 'timestamp_unix': time.time(), 'session_started_unix': self.started,
                  'controller_pid': os.getpid(),
                  'attack_pids': {job: item['process'].pid for job, item in self.active.items()},
                  'remaining_hours': round(self.remaining(), 3), 'active_attacks': list(self.active),
                  'active_judge': self.judge['config']['id'] if self.judge else None,
                  'failed_jobs': sorted(self.failed_jobs), 'caption_failures': sorted(self.caption_failures),
                  'judge_deferred_until_unix': self.judge_deferred_until,
                  'configurations': [{**c, 'attack_completed': complete_count(state['attack_runs'].get(c['job']), c['variant']),
                                      'captions_complete': self.all_stage_complete(c, 'caption'),
                                      'judge_complete': self.all_stage_complete(c, 'judge')} for c in self.configs]}
        self.eval.write_json(self.state_path, report)
        self.eval.verified_upload(self.fs, self.state_path,
                                  self.backup_prefix.rsplit('/', 1)[0] + '/overnight/controller_state.json')
        self.eval.report(self.eval_root)
        print('OVERNIGHT', event, '| hours left', report['remaining_hours'],
              '| attack jobs', list(self.active), '| judge', report['active_judge'], flush=True)

    def launch_attack(self, config):
        job = next(j for j in self.plan['attack_jobs'] if j['name'] == config['job'])
        if job.get('reuse_primary_captions') and not factual_captions_ready(self.attack_state(), self.plan['num_samples']):
            return False
        directory = self.root / 'parallel_tasks' / config['job']
        directory.mkdir(parents=True, exist_ok=True)
        pause = directory / 'gpu_handoff.request'
        pause.unlink(missing_ok=True)
        executable = self.root / '.venvs/attack/bin/python'
        command = [str(executable), '-u', str(self.root / 'molab_remaining_tasks.py'), '_task',
                   '--root', str(self.root), '--task', config['job'],
                   '--stop-after-variant', config['variant'], '--pause-file', str(pause),
                   '--max-hours', str(max(.01, self.remaining() - 5 / 60)),
                   '--backup-prefix', self.backup_prefix]
        log = (directory / 'overnight_attack.log').open('a', encoding='utf-8')
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=fresh_environment())
        self.active[config['job']] = {'process': process, 'config': config, 'log': log,
                                      'pause': pause, 'started': time.monotonic()}
        print('ATTACK START', config['id'], '| PID', process.pid, flush=True)
        return True

    def reconcile_job(self, config):
        state = self.attack_state()
        run = state['attack_runs'].get(config['job'])
        if not run:
            return
        job = next(j for j in self.plan['attack_jobs'] if j['name'] == config['job'])
        statuses = {}
        for variant in job['variants']:
            count = complete_count(run, variant)
            statuses[variant] = {'status': 'complete' if count == self.plan['num_samples'] else 'interrupted',
                                 'completed': count, 'total': self.plan['num_samples']}
        self.attack.write_json(Path(run) / 'summary.json', statuses)
        self.attack.write_json(Path(run) / 'status.json', {
            'status': 'complete' if all(s['status'] == 'complete' for s in statuses.values()) else 'interrupted',
            'variants': statuses})
        backup = self.attack.DriveBackup(self.backup_prefix, config['job'])
        backup.snapshot(run)

    def poll_attacks(self):
        for job, item in list(self.active.items()):
            code = item['process'].poll()
            if code is None:
                continue
            item['log'].close()
            del self.active[job]
            self.reconcile_job(item['config'])
            if code not in (0, 75):
                self.failed_jobs.add(job)
                print('ATTACK ERROR', item['config']['id'], 'exit', code, '| inspect', item['log'].name, flush=True)
            else:
                print('ATTACK CHECKPOINT', item['config']['id'], 'exit', code, flush=True)
            self.record('attack_checkpoint')

    def pause_attacks_for_gpu(self):
        if not self.active:
            return
        print('GPU HANDOFF: waiting for attack sample boundaries and verified backups.', flush=True)
        for item in self.active.values():
            item['pause'].write_text('caption GPU handoff\n', encoding='utf-8')
        while self.active:
            self.poll_attacks()
            self.poll_judge()
            if self.active:
                time.sleep(5)

    def caption(self, config):
        # No local caption model may contend with attack model allocations.
        if self.active:
            raise RuntimeError('Caption handoff requires all attack workers to be paused')
        self.eval.prepare(self.eval_root, self.root, self.backup_prefix)
        exe = self.eval_root / '.venv/bin/python'
        command = [str(exe), '-u', str(self.eval_root / 'molab_eval_completed.py'), 'run',
                   '--root', str(self.eval_root), '--only', config['id'], '--captions-only',
                   '--backup-prefix', self.eval_backup_prefix,
                   '--batch-size', str(self.batch_size), '--max-hours', str(max(.01, self.remaining() - 5 / 60))]
        path = self.control_root / (config['id'] + '_captions.log')
        print('CAPTION NOW', config['id'], '| batch', self.batch_size, flush=True)
        with path.open('a', encoding='utf-8') as log:
            with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=fresh_environment()) as child:
                code = child.wait()
        if code or not self.all_stage_complete(config, 'caption'):
            self.caption_failures.add(config['id'])
            print('CAPTION PARTIAL/ERROR', config['id'], '| inspect', path, flush=True)
        self.record('caption_checkpoint')

    def launch_judge(self):
        if self.judge or time.time() < self.judge_deferred_until or self.remaining() < .12:
            return
        pending = self.ready_for_judge()
        if not pending:
            return
        env = fresh_environment()
        if not env.get('OPENAI_API_KEY'):
            return
        config = pending[0]
        # Judge and caption workers both persist evaluation_state.json. Give the
        # API worker a private state copy, while its completed caption/run paths
        # still point at the original immutable tasks. This avoids lost updates
        # when new caption tasks are registered during API requests.
        judge_root = self.eval_root / '.judge_state' / config['id']
        judge_root.mkdir(parents=True, exist_ok=True)
        for name in ('molab_eval_completed.py', 'text_gen_eval_v3.1.ipynb', 'molab_hypotheses_n200.json',
                     'ready_configurations.json', 'evaluation_state.json'):
            shutil.copy2(self.eval_root / name, judge_root / name)
        # Every caption is already complete. The frozen engine loads no VLM in
        # this mode; GPT API requests can overlap with the next attack ablation.
        command = [str(self.eval_root / '.venv/bin/python'), '-u',
                   str(judge_root / 'molab_eval_completed.py'), 'run', '--root', str(judge_root),
                   '--backup-prefix', self.eval_backup_prefix,
                   '--only', config['id'], '--batch-size', str(self.batch_size),
                   '--max-hours', str(max(.01, self.remaining() - 3 / 60))]
        log = (self.control_root / (config['id'] + '_judge.log')).open('a', encoding='utf-8')
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        self.judge = {'process': process, 'config': config, 'log': log, 'started_unix': time.time(), 'root': judge_root}
        print('GPT JUDGE NOW', config['id'], '| PID', process.pid, '| rate governor shared within one process', flush=True)

    def poll_judge(self):
        if not self.judge:
            return
        code = self.judge['process'].poll()
        if code is None:
            return
        old = self.judge
        old['log'].close()
        self.judge = None
        if code or not self.all_stage_complete(old['config'], 'judge'):
            quota = old['root'] / 'judge_quota_pause.json'
            if quota.exists() and quota.stat().st_mtime >= old['started_unix']:
                self.judge_deferred_until = self.eval.read_json(quota)['not_before_unix']
            else:
                # Do not spam retries for invalid credentials, data or exhausted
                # session budgets. Attack and GPU captions can still progress.
                self.judge_deferred_until = time.time() + self.remaining() * 3600
            print('GPT PAUSED/PARTIAL', old['config']['id'], '| inspect', old['log'].name, flush=True)
        else:
            print('GPT COMPLETE', old['config']['id'], flush=True)
        self.record('judge_checkpoint')

    def free_vram_gib(self):
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True)
        return float(result.stdout.strip().splitlines()[0]) / 1024

    def run(self):
        self.record('started')
        if not fresh_environment().get('OPENAI_API_KEY'):
            print('KEY MISSING: attacks/captions continue; judge remains queued until key is available.', flush=True)
        try:
            while self.remaining() > 5 / 60:
                self.poll_attacks()
                self.poll_judge()
                ready = self.ready_for_captions()
                if ready and self.remaining() > .2:
                    self.pause_attacks_for_gpu()
                    for config in self.ready_for_captions():
                        if self.remaining() <= .2:
                            break
                        self.caption(config)
                    self.launch_judge()
                self.launch_judge()
                state = self.attack_state()
                pending = pending_attacks(self.configs, state, set(self.active), self.failed_jobs)
                for config in pending:
                    if len(self.active) >= self.workers:
                        break
                    if self.active:
                        # Give the first model time to allocate before admitting
                        # another job, and preserve headroom for its activations.
                        age = min(time.monotonic() - x['started'] for x in self.active.values())
                        if age < 120 or self.free_vram_gib() < 32:
                            break
                    self.launch_attack(config)
                if time.monotonic() - self.last_report >= 120:
                    self.record('running')
                    telemetry = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                                                '--format=csv,noheader'], capture_output=True, text=True)
                    print('GPU util / MiB / W:', telemetry.stdout.strip(), flush=True)
                    self.last_report = time.monotonic()
                if not self.active and not pending and not self.ready_for_captions() and not self.judge:
                    break
                time.sleep(5)
            self.pause_attacks_for_gpu()
            while self.judge:
                self.poll_judge()
                if self.judge:
                    time.sleep(5)
            self.record('finished_or_time_paused')
            print('OVERNIGHT CHECKPOINTS SAVED. Review controller_state.json and evaluation summary.', flush=True)
            return 0
        finally:
            # Graceful scheduled exits use handoff above. Manual/forced exits
            # retain the previous verified Drive checkpoints for safe resume.
            for item in self.active.values():
                item['pause'].write_text('controller interrupted\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/marimo/hypotheses_n200'))
    parser.add_argument('--eval-root', type=Path, default=Path('/marimo/eval_completed_n200'))
    parser.add_argument('--control-root', type=Path, default=Path('/marimo/overnight_n200'))
    parser.add_argument('--max-hours', type=float, default=9)
    parser.add_argument('--attack-workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--caption-batch-size', type=int, choices=range(1, 65), default=64)
    parser.add_argument('--backup-prefix', default='KLTN_Molab_Hypotheses_n200_v432_h1/results')
    parser.add_argument('--eval-backup-prefix', default='KLTN_Molab_Hypotheses_n200_v432_h1/evaluation_completed_v1')
    parser.add_argument('--adopt-state', type=Path, help='Verified surviving workers from a replaced controller')
    args = parser.parse_args()
    if args.max_hours <= 0:
        parser.error('--max-hours must be positive')
    # The lock is held for the controller lifetime; duplicate clicks cannot
    # launch competing writers against the same experiment state.
    import fcntl
    args.control_root.mkdir(parents=True, exist_ok=True)
    with (args.control_root / 'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('An overnight controller is already running for this root')
        controller = Controller(args.root.resolve(), args.eval_root.resolve(), args.control_root.resolve(),
                                args.max_hours, args.attack_workers, args.caption_batch_size,
                                args.backup_prefix, args.eval_backup_prefix)
        if args.adopt_state:
            controller.adopt(args.adopt_state)
        return controller.run()


if __name__ == '__main__':
    raise SystemExit(main())
