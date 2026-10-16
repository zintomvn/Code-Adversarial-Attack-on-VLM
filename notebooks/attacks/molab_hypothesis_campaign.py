"""Sequential, resumable execution of the original attack and evaluation notebooks.

Run from the extracted campaign directory. Original notebook algorithms are left
unchanged. Google Drive credentials must already be connected in the Molab UI.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED


class BoundaryPause(BaseException):
    """Bypass notebook sample-error handling at a completed checkpoint boundary."""


class BackupFailure(BaseException):
    """A storage failure must stop execution instead of becoming a sample error."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def notebook_cells(path):
    return [(i, ''.join(cell['source'])) for i, cell in enumerate(read_json(path)['cells'])
            if cell['cell_type'] == 'code']


class DriveBackup:
    def __init__(self, prefix, job):
        from gdrive_fsspec import GoogleDriveFileSystem
        self.fs = GoogleDriveFileSystem(token='cache', skip_instance_cache=True)
        self.prefix = prefix.rstrip('/') + '/' + job
        self.fs.makedirs(self.prefix, exist_ok=True)

    def put(self, local, relative):
        try:
            remote = self.prefix + '/' + relative
            parent = remote.rsplit('/', 1)[0]
            self.fs.makedirs(parent, exist_ok=True)
            self.fs.put_file(str(local), remote)
            # Verify remote bytes before declaring the checkpoint durable.
            check = hashlib.sha256()
            with self.fs.open(remote, 'rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    check.update(block)
            if check.hexdigest() != digest(local):
                raise RuntimeError(f'Remote checkpoint checksum mismatch: {relative}')
        except Exception as exc:
            raise BackupFailure(f'Checkpoint storage failed ({type(exc).__name__}); stopping the campaign.') from None

    def snapshot(self, run, sample=None):
        run = Path(run)
        staging = run.parent / 'backup_staging'
        staging.mkdir(exist_ok=True)
        if sample is not None:
            sample = Path(sample)
            relative = sample.relative_to(run)
            archive = staging / 'sample.zip'
            with ZipFile(archive, 'w', ZIP_DEFLATED) as output:
                for path in sorted(sample.rglob('*')):
                    if path.is_file() and not path.name.endswith('.tmp'):
                        output.write(path, path.relative_to(run).as_posix())
            self.put(archive, relative.as_posix() + '.zip')
        metadata = staging / 'root_latest.zip'
        with ZipFile(metadata, 'w', ZIP_DEFLATED) as output:
            for path in sorted(run.rglob('*')):
                if path.is_file() and 'samples' not in path.relative_to(run).parts and not path.name.endswith('.tmp'):
                    output.write(path, path.relative_to(run).as_posix())
        self.put(metadata, 'root_latest.zip')


def setup_environments(root, plan):
    base_python = getattr(sys, '_base_executable', sys.executable)
    for stage in ('attack', 'eval'):
        env = root / '.venvs' / stage
        executable = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if not executable.is_file():
            subprocess.run([base_python, '-m', 'venv', '--system-site-packages', str(env)], check=True)
        if stage == 'attack':
            tree = ast.parse(notebook_cells(root / plan['attack_notebook'])[1][1])
            pins = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'PINNED_PACKAGES' for t in node.targets))
            pins += ['numpy', 'Pillow', 'gdrive-fsspec']
        else:
            pins = ['transformers==4.57.6', 'accelerate==1.10.1', 'bitsandbytes==0.48.1',
                    'openai==1.109.1', 'tenacity==9.0.0', 'tqdm==4.67.1',
                    'sentencepiece==0.2.1', 'Pillow', 'ipython', 'gdrive-fsspec']
        subprocess.run([str(executable), '-m', 'pip', 'install', *pins], check=True)
        print(f'{stage} interpreter: {executable}', flush=True)


def attack_worker(root, plan, job, state_path, backup_prefix, max_hours):
    state = read_json(state_path) if state_path.is_file() else {'attack_runs': {}, 'evaluations': {}}
    main_run = state['attack_runs'].get('primary_ablations')
    overrides = dict(job.get('overrides', {}))
    revisions = plan.get('attack_model_revisions')
    if job.get('reuse_primary_captions'):
        if not main_run:
            raise RuntimeError('Primary run is required before H4 controls.')
        texts = read_json(Path(main_run) / 'text_records.json')
        overrides['MANUAL_TEXT_OVERRIDES'] = {sid: {key: row[key] for key in ('source', 'target')}
                                             for sid, row in texts.items() if not row.get('caption_error')}
        revisions = read_json(Path(main_run) / 'run_config.json')['model_revisions']
    ns = {'__name__': '__main__', '__file__': str(root / plan['attack_notebook'])}
    cells = notebook_cells(root / plan['attack_notebook'])
    exec(compile(cells[0][1], f'{plan["attack_notebook"]}:cell_{cells[0][0]}', 'exec'), ns)
    ns.update(DATASET_ROOT=root / 'data', PAIR_MANIFEST=root / plan['dataset_manifest'],
              WORK_ROOT=root / 'work', OUTPUT_ROOT=root / 'work' / job['name'],
              NUM_SAMPLES=plan['num_samples'], SAMPLE_START=0, STEPS=plan['steps'], SEED=plan['seed'],
              EPSILON_255=plan['epsilon_255'], ALPHA_255=plan['alpha_255'],
              RUN_VARIANTS=job['variants'], RUN_ATTACK=False, SETUP_ONLY=False,
              INSTALL_DEPENDENCIES=False, ATTACK_DEVICE='cuda:0', ATTENTION_DEVICE='cuda:0', CAPTION_DEVICE='cuda:0',
              SAVED_TENSORS_ON_CPU=False, OFFLOAD_CLIP_DURING_QWEN=False,
              RESUME_RUN_DIR=state['attack_runs'].get(job['name']), **overrides)
    ns['validate_configuration']()
    os.environ['HF_HOME'] = str(root / 'hf_cache')
    for index, source in cells[1:]:
        exec(compile(source, f'{plan["attack_notebook"]}:cell_{index}', 'exec'), ns)
    labels = ns['BUILD_CELL_FIRST_LINES']
    ns['BUILD_CELL_HASHES'] = {labels[s.strip().splitlines()[0]]: hashlib.sha256(s.strip().encode()).hexdigest()
                              for _, s in cells if s.strip().splitlines()[0] in labels}
    ns['verify_devices_and_runtime']()
    if len(ns['prepare_pairs']()) != plan['num_samples']:
        raise RuntimeError('Expected exactly 200 selected pairs.')
    backup = DriveBackup(backup_prefix, job['name'])
    probe = root / 'work' / 'backup_probe.json'
    write_json(probe, {'experiment': plan['experiment'], 'job': job['name'], 'check': 'write_readback'})
    backup.put(probe, 'backup_probe.json')
    original_identity = ns['run_identity']
    driver_sha = digest(Path(__file__))
    ns['run_identity'] = lambda pairs: {**original_identity(pairs), 'campaign_driver_sha256': driver_sha,
                                       'plan_sha256': digest(root / 'molab_hypotheses_n200.json')}
    if revisions is not None:
        original_resolve = ns['resolve_revisions']
        ns['resolve_revisions'] = lambda ids, existing=None: original_resolve(ids, existing or {k: revisions[k] for k in ids})
    original_write = ns['write_json']
    original_run_one = ns['run_one']
    deadline = time.monotonic() + max_hours * 3600
    text_writes = 0

    def checkpoint_write(path, value):
        nonlocal text_writes
        original_write(path, value)
        path = Path(path)
        run = ns.get('RUN_DIR')
        if path.name == 'run_config.json' and run is not None and path.parent == Path(run):
            state['attack_runs'][job['name']] = str(run)
            write_json(state_path, state)
            backup.put(state_path, 'campaign_state.json')
            backup.snapshot(run)
        elif run is not None and path.name == 'complete.json':
            backup.snapshot(run, path.parent)
            print(f'DURABLE SAMPLE: {path.parent.relative_to(run)}', flush=True)
            if time.monotonic() >= deadline:
                raise BoundaryPause('Time limit reached after a durable sample checkpoint.')
        elif run is not None and path.name == 'text_records.json':
            text_writes += 1
            if text_writes % 5 == 0 or time.monotonic() >= deadline:
                backup.snapshot(run)
            if time.monotonic() >= deadline:
                raise BoundaryPause('Time limit reached after saving text preparation progress.')

    def boundary_run_one(*args, **kwargs):
        backup.snapshot(ns['RUN_DIR'])
        if time.monotonic() >= deadline:
            raise BoundaryPause('Time limit reached before starting another sample.')
        return original_run_one(*args, **kwargs)

    ns['write_json'], ns['run_one'], ns['RUN_ATTACK'] = checkpoint_write, boundary_run_one, True
    try:
        run = ns['run_experiment']()
    except BoundaryPause as exc:
        run = ns.get('RUN_DIR')
        if run:
            backup.snapshot(run)
        print(f'PAUSED AT CHECKPOINT: {exc}. Restore and rerun with identical configuration.', flush=True)
        return 75
    finally:
        run = ns.get('RUN_DIR')
        if run and Path(run).exists():
            backup.snapshot(run)
    status = read_json(Path(run) / 'status.json')
    if status['status'] != 'complete':
        raise RuntimeError(f'Attack incomplete: inspect {run}/summary.json and variant errors.jsonl.')
    return 0


def eval_worker(root, plan, task, state_path, backup_prefix):
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('Evaluation requires OPENAI_API_KEY in Molab Secrets/environment. Never place it in this package.')
    state = read_json(state_path)
    attack_run = Path(state['attack_runs'][task['job']])
    settings = plan['evaluation']
    ns = {'__name__': '__main__', '__file__': str(root / plan['eval_notebook'])}
    cells = notebook_cells(root / plan['eval_notebook'])
    exec(compile(cells[0][1], f'{plan["eval_notebook"]}:config', 'exec'), ns)
    ns.update(INPUT_DIR=attack_run / task['variant'], OUTPUT_DIR=root / 'evaluation' / task['name'],
              NUM_SAMPLES=plan['num_samples'], RUN_PIPELINE=True, RUN_EVALUATION=True,
              MODEL=task['model'], SEED=plan['seed'], GPU_MODE='single', MAX_GPU_REPLICAS=1,
              LOCAL_BATCH_SIZE=settings['local_batch_size'], JUDGE_RPM=settings['judge_rpm'],
              JUDGE_TPM=settings['judge_tpm'], JUDGE_MODEL=settings['judge_model'],
              SUCCESS_THRESHOLD=settings['success_threshold'], SUCCESS_OPERATOR=settings['success_operator'],
              RESUME_CAPTION_RUN_DIR=state['evaluations'].get(task['name']), LOCAL_MODEL=True)
    os.environ['HF_HOME'] = str(root / 'hf_cache')
    backup = DriveBackup(backup_prefix, 'evaluation/' + task['name'])
    original_atomic_write = None
    last_checkpoint = 0.0

    def evaluation_checkpoint(path, value):
        nonlocal last_checkpoint
        original_atomic_write(path, value)
        run = ns.get('CAPTION_RUN_DIR')
        if not run:
            return
        state['evaluations'][task['name']] = str(run)
        write_json(state_path, state)
        if Path(path).name == 'run_config.json' or time.monotonic() - last_checkpoint >= 60:
            backup.put(state_path, 'campaign_state.json')
            backup.snapshot(run)
            last_checkpoint = time.monotonic()

    for index, source in cells[1:]:
        exec(compile(source, f'{plan["eval_notebook"]}:cell_{index}', 'exec'), ns)
        if 'cfg' in ns and index == 4:
            from huggingface_hub import HfApi
            locked = state.setdefault('eval_model_revisions', {})
            model_id = ns['cfg'].caption.model_id
            if model_id not in locked:
                locked[model_id] = HfApi().model_info(model_id, revision='main').sha
            ns['cfg'].caption.revision = locked[model_id]
            ns['CAPTION_CONFIG']['campaign_driver_sha256'] = digest(Path(__file__))
            ns['CAPTION_CONFIG_SHA256'] = hashlib.sha256(
                json.dumps(ns['CAPTION_CONFIG'], sort_keys=True).encode()).hexdigest()
            write_json(state_path, state)
        if original_atomic_write is None and 'write_json_atomic' in ns:
            original_atomic_write = ns['write_json_atomic']
            ns['write_json_atomic'] = evaluation_checkpoint
        run = ns.get('CAPTION_RUN_DIR')
        if run:
            state['evaluations'][task['name']] = str(run)
            write_json(state_path, state)
            backup.put(state_path, 'campaign_state.json')
            # Evaluation stores captions and scores without copying attack images.
            backup.snapshot(run)
    if (ns.get('evaluation_status') or {}).get('status') != 'complete':
        raise RuntimeError('Evaluation is partial; restore the checkpoint and retry before interpreting metrics.')
    return 0


def restore_campaign(root, plan, backup_prefix):
    from gdrive_fsspec import GoogleDriveFileSystem
    fs = GoogleDriveFileSystem(token='cache', skip_instance_cache=True)
    state = {'attack_runs': {}, 'evaluations': {}, 'eval_model_revisions': {}}
    tasks = [('attack_runs', j['name'], j['name']) for j in plan['attack_jobs']]
    tasks += [('evaluations', t['name'], 'evaluation/' + t['name']) for t in evaluation_tasks(plan)]
    # Each task keeps a small, cumulative state file. Merge all observed snapshots.
    for _, _, remote_job in tasks:
        remote_state = backup_prefix.rstrip('/') + '/' + remote_job + '/campaign_state.json'
        if fs.exists(remote_state):
            with fs.open(remote_state, 'rb') as stream:
                saved = json.load(stream)
            for stage in state:
                for name, path in saved.get(stage, {}).items():
                    if name in state[stage] and state[stage][name] != path:
                        raise ValueError(f'Conflicting remote run pointers for {name}')
                    state[stage][name] = path
    if not any(state.values()):
        raise FileNotFoundError('No remote campaign checkpoints found.')
    staging = root / '.restore_staging'
    staging.mkdir(exist_ok=True)
    for stage, name, remote_job in tasks:
        if name not in state[stage]:
            continue
        run = Path(state[stage][name]).resolve()
        if not run.is_relative_to(root):
            raise ValueError(f'Restore under the original absolute project root: {root}; saved run is {run}')
        run.mkdir(parents=True, exist_ok=True)
        remote_root = backup_prefix.rstrip('/') + '/' + remote_job
        archives = [p for p in fs.find(remote_root) if p.endswith('.zip')]
        for remote in archives:
            local = staging / (hashlib.sha256(remote.encode()).hexdigest() + '.zip')
            fs.get_file(remote, str(local))
            with ZipFile(local) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f'Corrupt checkpoint archive: {remote}')
                for member in archive.infolist():
                    if not (run / member.filename).resolve().is_relative_to(run):
                        raise ValueError(f'Unsafe archive path: {member.filename}')
                archive.extractall(run)
            local.unlink()
        print('RESTORED:', name, run, flush=True)
    write_json(root / 'campaign_state.json', state)
    print('Restore complete. Run setup, then attack/evaluate with the same package and runtime.', flush=True)


def evaluation_tasks(plan):
    for job in plan['attack_jobs']:
        for variant in job['variants']:
            for model in plan['evaluation']['models']:
                name = f'{job["name"]}__{variant}__{model}'
                yield {'name': name, 'job': job['name'], 'variant': variant, 'model': model}


def validate_package(root, plan):
    for name in (plan['attack_notebook'], plan['eval_notebook']):
        for index, source in notebook_cells(root / name):
            compile(source, f'{name}:cell_{index}', 'exec')
    manifest = root / plan['dataset_manifest']
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != 200 or [r['index'] for r in rows] != list(range(200)):
        raise ValueError('Expected the predefined 200 image pairs, in order.')
    provenance = read_json(root / 'data' / 'data_provenance.json')
    for row in rows:
        for role in ('source', 'target'):
            relative = row[role]
            path = (manifest.parent / relative).resolve()
            if not path.is_relative_to(manifest.parent.resolve()) or digest(path) != provenance['sha256'][relative]:
                raise ValueError(f'Input image changed: {relative}')
    print('PACKAGE VALID: 200 pairs, 400 original images, both notebook code sets compile.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'setup', 'restore', 'attack', 'evaluate', '_attack_worker', '_eval_worker'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--backup-prefix', default='KLTN_Molab_Hypotheses_n200_v432_h1/results')
    parser.add_argument('--max-hours', type=float, default=10.0)
    parser.add_argument('--task')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = read_json(root / 'molab_hypotheses_n200.json')
    state_path = root / 'campaign_state.json'
    if args.mode == 'check':
        validate_package(root, plan)
        print('Attack configurations:', sum(len(j['variants']) for j in plan['attack_jobs']))
        print('Evaluation tasks:', len(list(evaluation_tasks(plan))))
        return 0
    if args.mode == 'setup':
        validate_package(root, plan)
        setup_environments(root, plan)
        return 0
    if args.mode == 'restore':
        restore_campaign(root, plan, args.backup_prefix)
        return 0
    if args.mode == '_attack_worker':
        job = next(j for j in plan['attack_jobs'] if j['name'] == args.task)
        return attack_worker(root, plan, job, state_path, args.backup_prefix, args.max_hours)
    if args.mode == '_eval_worker':
        task = next(t for t in evaluation_tasks(plan) if t['name'] == args.task)
        return eval_worker(root, plan, task, state_path, args.backup_prefix)
    validate_package(root, plan)
    state = read_json(state_path) if state_path.is_file() else {'attack_runs': {}, 'evaluations': {}}
    write_json(state_path, state)
    stage = 'attack' if args.mode == 'attack' else 'eval'
    executable = root / '.venvs' / stage / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not executable.is_file():
        raise RuntimeError('Run setup first.')
    jobs = plan['attack_jobs'] if stage == 'attack' else list(evaluation_tasks(plan))
    started = time.monotonic()
    for job in jobs:
        state = read_json(state_path)
        existing = state['attack_runs' if stage == 'attack' else 'evaluations'].get(job['name'])
        status_path = Path(existing) / ('status.json' if stage == 'attack' else 'evaluation/status.json') if existing else None
        if status_path and status_path.is_file() and read_json(status_path).get('status') == 'complete':
            print(f'Skip completed task: {job["name"]}', flush=True)
            continue
        remaining = args.max_hours - (time.monotonic() - started) / 3600
        if remaining <= 0:
            print('Session time limit reached. Resume in a new GPU session.', flush=True)
            return 75
        command = [str(executable), '-u', str(Path(__file__).resolve()), '_' + stage + '_worker',
                   '--root', str(root), '--task', job['name'], '--backup-prefix', args.backup_prefix,
                   '--max-hours', str(remaining)]
        print('Starting:', job['name'], flush=True)
        logs = root / 'logs'
        logs.mkdir(exist_ok=True)
        with (logs / (job['name'] + '.log')).open('a', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
            if code:
                return code
    print(args.mode.upper() + ' CAMPAIGN COMPLETE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
