"""Evaluate completed attack configurations without waiting for the full campaign.

Uses the unchanged text_gen_eval_v3.1.ipynb protocol in an isolated interpreter.
This runner never writes into the attack run or its campaign_state.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from zipfile import ZipFile, ZIP_DEFLATED

VERSION = 'molab_completed_eval_v1'
# Audited compatibility: the first release differs only in quota/deadline
# orchestration. Caption/judge protocols, manifests and model locks are unchanged.
COMPATIBLE_ENGINE_HASHES = {'cc9de38eb2debb89c991bf4d1cd20d5db8ee9adf9899952cfe54a5efda30cc5a'}
SOURCE_PREFIX = 'KLTN_Molab_Hypotheses_n200_v432_h1/results'
EVAL_PREFIX = 'KLTN_Molab_Hypotheses_n200_v432_h1/evaluation_completed_v1'
PINS = ['transformers==4.57.6', 'accelerate==1.10.1', 'bitsandbytes==0.48.1',
        'openai==1.109.1', 'tenacity==9.0.0', 'tqdm==4.67.1',
        'sentencepiece==0.2.1', 'Pillow', 'ipython', 'gdrive-fsspec', 'python-dotenv']


class SafePause(BaseException):
    pass


class QuotaPauseError(RuntimeError):
    pass


def install_quota_guard(ns, deadline, pause_file):
    """Abort waiting workers cooperatively; do not shorten provider cooldowns."""
    limiter = ns['JUDGE_LIMITER']
    cancelled = threading.Event()
    guard_lock = threading.Lock()
    original_cooldown = limiter.cooldown
    original_fatal = ns['judge_fatal']

    def cooldown(duration, *, count_429=False):
        original_cooldown(duration, count_429=count_429)
        if duration > 120 or time.monotonic() + duration >= deadline:
            with guard_lock:
                if not cancelled.is_set():
                    write_json(pause_file, {'reason': 'provider_cooldown', 'http_429': count_429,
                                           'retry_after_seconds': duration,
                                           'not_before_unix': time.time() + duration})
                    cancelled.set()
                    print(f'QUOTA PAUSE: provider requires {duration:.1f}s. Save and resume later; no new requests.', flush=True)

    def acquire(predicted_tokens):
        predicted_tokens = int(predicted_tokens)
        if predicted_tokens > limiter.tpm:
            raise ValueError('One request exceeds JUDGE_TPM')
        while True:
            if cancelled.is_set() or time.monotonic() >= deadline:
                raise QuotaPauseError('Judge paused for quota/time budget; successful scores remain resumable')
            with limiter.lock:
                now = time.monotonic()
                while limiter.events and now - limiter.events[0][0] >= 60:
                    limiter.events.popleft()
                delay = max(0., limiter.cooldown_until - now, limiter.next_start - now)
                if len(limiter.events) >= limiter.rpm:
                    delay = max(delay, 60 - (now - limiter.events[0][0]) + 0.05)
                used = sum(item[1] for item in limiter.events)
                if used + predicted_tokens > limiter.tpm:
                    reclaimed = 0
                    for stamp, tokens in limiter.events:
                        reclaimed += tokens
                        if used - reclaimed + predicted_tokens <= limiter.tpm:
                            delay = max(delay, 60 - (now - stamp) + 0.05)
                            break
                if delay <= 0:
                    limiter.events.append((now, predicted_tokens))
                    limiter.next_start = now + 60 / limiter.rpm
                    limiter.requests_reserved += 1
                    return
            cancelled.wait(min(5., max(0.02, delay)))

    limiter.cooldown = cooldown
    limiter.acquire = acquire
    ns['judge_fatal'] = lambda error: isinstance(error, QuotaPauseError) or original_fatal(error)
    return cancelled


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def contained(root, relative):
    if not isinstance(relative, str) or '\\' in relative or ':' in relative:
        raise ValueError('Invalid relative artifact path')
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if Path(relative).is_absolute() or not target.is_relative_to(root):
        raise ValueError('Artifact path escapes its root')
    return target


def configuration_jobs(plan):
    return [{'id': job['name'] + '__' + variant, 'job': job['name'], 'variant': variant}
            for job in plan['attack_jobs'] for variant in job['variants']]


def drive_fs():
    from gdrive_fsspec import GoogleDriveFileSystem
    return GoogleDriveFileSystem(token='cache', skip_instance_cache=True)


def verified_upload(fs, local, remote):
    fs.makedirs(remote.rsplit('/', 1)[0], exist_ok=True)
    fs.put_file(str(local), remote)
    h = hashlib.sha256()
    with fs.open(remote, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != sha(local):
        raise RuntimeError('Drive backup readback checksum mismatch')


def unpack_metadata(archive, target):
    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Corrupt metadata ZIP')
        for item in z.infolist():
            path = contained(target, item.filename)
            if item.is_dir():
                continue
            # Root snapshots contain configuration manifests, text and status.
            if 'samples' in Path(item.filename).parts:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(item))


def setup(root):
    exe = root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not exe.exists():
        subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(root / '.venv')], check=True)
    subprocess.run([str(exe), '-m', 'pip', 'install', *PINS], check=True)
    return exe


def validate_manifest(root, expected=200):
    manifest = root / 'manifest.jsonl'
    records = rows(manifest)
    indices = [r['index'] for r in records]
    if len(records) != expected or indices != list(range(expected)):
        raise ValueError(f'Expected the full ordered cohort of {expected} pairs; got {len(records)}')
    fingerprints = {r['run_fingerprint'] for r in records}
    if len(fingerprints) != 1:
        raise ValueError('Mixed attack identities')
    for row in records:
        if row['sample_id'] != f"{row['index']:05d}":
            raise ValueError('Inconsistent sample ID')
        for role in ('source', 'target', 'adversarial'):
            p = contained(root, row[role])
            if not p.is_file() or sha(p) != row['image_sha256'][role]:
                raise ValueError(f"Missing/changed image: {row['sample_id']} / {role}")
        if row.get('saved_linf_255', float('inf')) > row.get('epsilon_255', -1):
            raise ValueError('Saved perturbation exceeds attack budget')
        complete = read_json(root / 'samples' / row['sample_id'] / 'complete.json')
        for field in ('index', 'sample_id', 'run_fingerprint', 'image_sha256', 'source', 'target', 'adversarial'):
            if complete.get(field) != row.get(field):
                raise ValueError(f'Completion marker differs at {field}')
    return records


def restore_sample(fs, remote, local_zip, destination, item, variant):
    fs.get_file(remote, str(local_zip))
    with ZipFile(local_zip) as z:
        if z.testzip() is not None:
            raise ValueError('Corrupt sample ZIP')
        prefix = variant + '/'
        allowed = {item[role] for role in ('source', 'target', 'adversarial')}
        allowed.add('samples/' + item['sample_id'] + '/complete.json')
        members = {name[len(prefix):]: name for name in z.namelist() if name.startswith(prefix)}
        for relative in allowed:
            target = contained(destination, relative)
            if relative not in members:
                raise ValueError('Sample ZIP lacks a required evaluation artifact')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(members[relative]))
    local_zip.unlink()


def prepare(root, source_root=None, source_prefix=SOURCE_PREFIX, only=None):
    plan = read_json(root / 'molab_hypotheses_n200.json')
    fs = None
    local_state = source_root / 'campaign_state.json' if source_root else None
    state = read_json(local_state) if local_state and local_state.is_file() else {}
    metadata_roots = {}
    inventory, ready = [], []
    for config in configuration_jobs(plan):
        if only and config['id'] not in only:
            continue
        job, variant = config['job'], config['variant']
        local_run = Path(state.get('attack_runs', {}).get(job, '/__missing_attack_run__'))
        metadata = local_run
        if not (metadata / variant / 'manifest.jsonl').is_file():
            if job not in metadata_roots:
                if fs is None:
                    fs = drive_fs()
                remote = source_prefix + '/' + job + '/root_latest.zip'
                if not fs.exists(remote):
                    metadata_roots[job] = None
                else:
                    metadata = root / 'source_metadata' / job
                    archive = root / (job + '_metadata.zip')
                    fs.get_file(remote, str(archive))
                    unpack_metadata(archive, metadata)
                    archive.unlink()
                    metadata_roots[job] = metadata
            metadata = metadata_roots[job]
        manifest = metadata / variant / 'manifest.jsonl' if metadata else None
        records = rows(manifest) if manifest and manifest.is_file() else []
        entry = {**config, 'completed_samples': len(records), 'expected_samples': plan['num_samples'],
                 'eligible': len(records) == plan['num_samples']}
        inventory.append(entry)
        print(f"{config['id']}: {len(records)}/{plan['num_samples']}", flush=True)
        if not entry['eligible']:
            continue
        destination = root / 'inputs' / config['id']
        destination.mkdir(parents=True, exist_ok=True)
        frozen_manifest = destination / 'manifest.jsonl'
        if frozen_manifest.is_file() and sha(frozen_manifest) != sha(manifest):
            raise ValueError('Frozen evaluation input changed; use a new evaluation root')
        for item in records:
            roles = [item[r] for r in ('source', 'target', 'adversarial')]
            marker = 'samples/' + item['sample_id'] + '/complete.json'
            if all(contained(destination, r).is_file() for r in [*roles, marker]):
                # Full checksums are verified below, not inferred from file presence.
                continue
            source_variant = local_run / variant
            if all(contained(source_variant, r).is_file() for r in [*roles, marker]):
                for relative in [*roles, marker]:
                    target = contained(destination, relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(contained(source_variant, relative), target)
            else:
                if fs is None:
                    fs = drive_fs()
                remote = f"{source_prefix}/{job}/{variant}/samples/{item['sample_id']}.zip"
                restore_sample(fs, remote, root / 'sample_download.zip', destination, item, variant)
        shutil.copy2(manifest, frozen_manifest)
        if (metadata / 'run_config.json').is_file():
            shutil.copy2(metadata / 'run_config.json', destination / 'run_config.json')
        validate_manifest(destination, plan['num_samples'])
        ready.append({**config, 'input_dir': str(destination), 'manifest_sha256': sha(frozen_manifest),
                      'samples': len(records)})
    write_json(root / 'inventory.json', inventory)
    write_json(root / 'ready_configurations.json', ready)
    print(f'READY: {len(ready)} complete configurations; partial configurations excluded.', flush=True)
    return ready


def backup_task(root, state, task, fs):
    run = Path(state['tasks'][task])
    archive = root / 'eval_backup.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
        for p in sorted(run.rglob('*')):
            if p.is_file() and not p.name.endswith('.tmp'):
                z.write(p, p.relative_to(run).as_posix())
    prefix = state['backup_prefix'] + '/' + task
    # Pointer is published only after a verified archive upload.
    archive_name = 'checkpoint_' + sha(archive) + '.zip'
    verified_upload(fs, archive, prefix + '/' + archive_name)
    pointer = root / 'eval_pointer.json'
    write_json(pointer, {'version': VERSION, 'task': task, 'run': str(run),
                         'archive': archive_name, 'archive_sha256': sha(archive),
                         'model_revisions': state['model_revisions']})
    verified_upload(fs, pointer, prefix + '/pointer.json')


def restore_evaluations(root, backup_prefix=EVAL_PREFIX):
    fs = drive_fs()
    state_path = root / 'evaluation_state.json'
    state = read_json(state_path) if state_path.is_file() else {'tasks': {}, 'model_revisions': {}}
    state['backup_prefix'] = backup_prefix
    for config in read_json(root / 'ready_configurations.json'):
        for model in read_json(root / 'molab_hypotheses_n200.json')['evaluation']['models']:
            task = config['id'] + '__' + model
            prefix = backup_prefix + '/' + task
            if not fs.exists(prefix + '/pointer.json'):
                continue
            with fs.open(prefix + '/pointer.json', 'rb') as stream:
                pointer = json.load(stream)
            run = Path(pointer['run']).resolve()
            if not run.is_relative_to(root):
                raise ValueError('Restore requires the original absolute evaluation root')
            archive = root / 'restore_eval.zip'
            archive_name = pointer['archive']
            if '/' in archive_name or '\\' in archive_name or not archive_name.startswith('checkpoint_'):
                raise ValueError('Unsafe remote evaluation archive name')
            fs.get_file(prefix + '/' + archive_name, str(archive))
            if sha(archive) != pointer['archive_sha256']:
                raise ValueError('Evaluation archive differs from its durable pointer')
            with ZipFile(archive) as z:
                if z.testzip() is not None:
                    raise ValueError('Corrupt evaluation ZIP')
                for item in z.infolist():
                    target = contained(run, item.filename)
                    if not item.is_dir():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(z.read(item))
            for model_id, revision in pointer['model_revisions'].items():
                if model_id in state['model_revisions'] and state['model_revisions'][model_id] != revision:
                    raise ValueError('Conflicting victim model revision locks')
                state['model_revisions'][model_id] = revision
            state['tasks'][task] = str(run)
            print('RESTORED', task, flush=True)
    write_json(state_path, state)


def caption_signature(config):
    return {key: config[key] for key in ('model', 'caption', 'roles', 'seed')}


def cached_items(root, state, signature, wanted):
    cached = {}
    for directory in state['tasks'].values():
        run = Path(directory)
        if not (run / 'run_config.json').is_file():
            continue
        prior = read_json(run / 'run_config.json')['caption_config']
        if caption_signature(prior) != signature:
            continue
        if (run / 'caption_items.jsonl').is_file():
            # Only complete JSONL rows; never repair another task's checkpoint.
            for line in (run / 'caption_items.jsonl').read_bytes().splitlines(keepends=True):
                if not line.endswith(b'\n'):
                    continue
                row = json.loads(line)
                if row.get('sha256') in wanted and row.get('status') == 'ok':
                    cached.setdefault(row['sha256'], row)
    return cached


def worker(root, config_id, model, batch_size=32, judge=True, max_hours=8, backup=True):
    plan = read_json(root / 'molab_hypotheses_n200.json')
    config = next(c for c in read_json(root / 'ready_configurations.json') if c['id'] == config_id)
    input_dir = Path(config['input_dir'])
    validate_manifest(input_dir, plan['num_samples'])
    if sha(input_dir / 'manifest.jsonl') != config['manifest_sha256']:
        raise ValueError('Evaluation input manifest changed')
    state_path = root / 'evaluation_state.json'
    state = read_json(state_path) if state_path.is_file() else {'tasks': {}, 'model_revisions': {}}
    state.setdefault('backup_prefix', EVAL_PREFIX)
    task = config_id + '__' + model
    ns = {'__name__': '__main__', '__file__': str(root / 'text_gen_eval_v3.1.ipynb')}
    notebook = read_json(root / 'text_gen_eval_v3.1.ipynb')
    cells = {i: ''.join(c['source']) for i, c in enumerate(notebook['cells']) if c['cell_type'] == 'code'}
    exec(compile(cells[2], 'evaluation_v3.1:config', 'exec'), ns)
    ns.update(INPUT_DIR=input_dir, OUTPUT_DIR=root / 'runs' / task, NUM_SAMPLES=plan['num_samples'],
              RUN_PIPELINE=True, RUN_EVALUATION=judge, MODEL=model, SEED=plan['seed'],
              GPU_MODE='single', MAX_GPU_REPLICAS=1, LOCAL_BATCH_SIZE=batch_size,
              LOCAL_MODEL=True, JUDGE_WORKERS=4, JUDGE_RPM=plan['evaluation']['judge_rpm'],
              JUDGE_TPM=plan['evaluation']['judge_tpm'], JUDGE_MODEL='gpt-4o',
              SUCCESS_THRESHOLD=0.3, SUCCESS_OPERATOR='>=',
              EXPERIMENT_NAME=VERSION, RESUME_CAPTION_RUN_DIR=state['tasks'].get(task))
    os.environ['HF_HOME'] = str(root / 'hf_cache')
    fs = drive_fs() if backup else None
    if backup:
        probe = root / 'backup_probe.json'
        write_json(probe, {'runner': VERSION, 'check': 'upload_readback'})
        verified_upload(fs, probe, state['backup_prefix'] + '/backup_probe.json')
    deadline = time.monotonic() + max_hours * 3600
    last_backup = 0.0
    original_write = None

    def checkpoint(path, value):
        nonlocal last_backup
        original_write(path, value)
        run = ns.get('CAPTION_RUN_DIR')
        if not run:
            return
        state['tasks'][task] = str(run)
        write_json(state_path, state)
        path = Path(path)
        if path.name == 'run_config.json' and path.parent == Path(run):
            wanted = {digest for s in ns['selected_samples'] for digest in s['image_sha256'].values()}
            cache_path = Path(run) / 'caption_items.jsonl'
            if not cache_path.exists():
                cache = cached_items(root, state, caption_signature(ns['CAPTION_CONFIG']), wanted)
                if cache:
                    cache_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                                  for row in cache.values()), encoding='utf-8')
                    print(f'REUSED {len(cache)} identical-image captions from the same victim protocol.', flush=True)
        expired = time.monotonic() >= deadline
        if backup and (time.monotonic() - last_backup >= 120 or expired):
            backup_task(root, state, task, fs)
            last_backup = time.monotonic()
        if expired and path.name == 'status.json':
            raise SafePause('Time budget reached after a flushed caption/judge checkpoint')

    try:
        for index in (3, 4, 5, 6, 7, 9, 10, 11):
            if index == 10 and ns.get('caption_status', {}).get('status') != 'complete':
                print('Captions incomplete; resume before judging the full cohort.', flush=True)
                return 75
            if index == 10 and not judge:
                break
            exec(compile(cells[index], f'evaluation_v3.1:cell_{index}', 'exec'), ns)
            if index == 4:
                from huggingface_hub import HfApi
                model_id = ns['cfg'].caption.model_id
                locks = state['model_revisions']
                if model_id not in locks:
                    locks[model_id] = HfApi().model_info(model_id, revision='main').sha
                ns['cfg'].caption.revision = locks[model_id]
                # cfg.caption is copied into CAPTION_CONFIG in the original notebook.
                ns['CAPTION_CONFIG']['caption']['revision'] = locks[model_id]
                ns['CAPTION_CONFIG']['evaluation_notebook_sha256'] = sha(root / 'text_gen_eval_v3.1.ipynb')
                ns['CAPTION_CONFIG']['runner_version'] = VERSION
                engine_hash = sha(__file__)
                effective_hash = engine_hash
                if state['tasks'].get(task):
                    prior_config = read_json(Path(state['tasks'][task]) / 'run_config.json')['caption_config']
                    prior_hash = prior_config.get('runner_sha256')
                    if prior_hash in COMPATIBLE_ENGINE_HASHES:
                        effective_hash = prior_hash
                ns['CAPTION_CONFIG']['runner_sha256'] = effective_hash
                ns['CAPTION_CONFIG_SHA256'] = hashlib.sha256(
                    json.dumps(ns['CAPTION_CONFIG'], sort_keys=True).encode()).hexdigest()
                write_json(state_path, state)
                if state['tasks'].get(task):
                    write_json(Path(state['tasks'][task]) / 'execution_engine.json', {
                        'engine_sha256': engine_hash, 'protocol_engine_sha256': effective_hash,
                        'compatibility': 'quota/deadline guard only; all other fingerprint fields still verified'})
            if index == 5:
                original_write = ns['write_json_atomic']
                ns['write_json_atomic'] = checkpoint
            if index == 7:
                install_quota_guard(ns, deadline, root / 'judge_quota_pause.json')
        if judge and ns.get('evaluation_status', {}).get('status') != 'complete':
            return 75
        return 0
    except SafePause as exc:
        print('EVALUATION PAUSED SAFELY:', str(exc), flush=True)
        return 75
    finally:
        if ns.get('CAPTION_RUN_DIR') and backup:
            backup_task(root, state, task, fs)


def report(root):
    state_path = root / 'evaluation_state.json'
    state = read_json(state_path) if state_path.exists() else {'tasks': {}}
    summary = []
    for task, run in state['tasks'].items():
        run = Path(run)
        caption = read_json(run / 'status.json') if (run / 'status.json').exists() else {}
        scored = read_json(run / 'evaluation' / 'summary.json') if (run / 'evaluation' / 'summary.json').exists() else {}
        checkpoint = run / 'evaluation' / 'scores.jsonl'
        if checkpoint.exists():
            latest = {}
            for line in checkpoint.read_bytes().splitlines(keepends=True):
                if line.endswith(b'\n'):
                    row = json.loads(line)
                    latest[row['sample_id']] = row
            valid = [r for r in latest.values() if r.get('status') == 'ok']
            n = caption.get('selected_samples', 200)
            if len(valid) != scored.get('total_evaluated'):
                scored = {'status': 'partial', 'total_evaluated': len(valid), 'coverage': len(valid) / n,
                          'full_cohort_success_rate': None,
                          'average_similarity_score': sum(r['score'] for r in valid) / len(valid) if valid else None}
        summary.append({'task': task, 'caption_status': caption.get('status'),
                        'ready_pairs': caption.get('completed_samples', 0),
                        'judge_status': scored.get('status', 'pending'),
                        'judged_pairs': scored.get('total_evaluated', 0),
                        'coverage': scored.get('coverage', 0),
                        'ASR': scored.get('full_cohort_success_rate'),
                        'AvgSim': scored.get('average_similarity_score'), 'run': str(run)})
    write_json(root / 'summary_completed.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['setup', 'prepare', 'restore', 'run', 'worker', 'report'])
    parser.add_argument('--root', type=Path, default=Path('/marimo/eval_completed_n200'))
    parser.add_argument('--source-root', type=Path, default=Path('/marimo/hypotheses_n200'))
    parser.add_argument('--source-prefix', default=SOURCE_PREFIX)
    parser.add_argument('--backup-prefix', default=EVAL_PREFIX)
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--models', nargs='+', default=['LLaVA-1.5-7B', 'InternVL3-8B'])
    parser.add_argument('--batch-size', type=int, default=32, choices=range(1, 65))
    parser.add_argument('--max-hours', type=float, default=8)
    parser.add_argument('--captions-only', action='store_true')
    parser.add_argument('--no-backup', action='store_true')
    parser.add_argument('--task')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mode in ('run', 'worker') and not args.captions_only:
        if not os.environ.get('OPENAI_API_KEY'):
            try:
                from dotenv import load_dotenv
                load_dotenv('/marimo/.env', override=False)
            except ImportError:
                pass
        if not os.environ.get('OPENAI_API_KEY'):
            try:
                from kaggle_secrets import UserSecretsClient
                os.environ['OPENAI_API_KEY'] = UserSecretsClient().get_secret('OPENAI_API_KEY')
            except Exception:
                pass
        if not os.environ.get('OPENAI_API_KEY'):
            print('OPENAI_API_KEY missing: generating GPU captions only. Add the secret and rerun for GPT judging.', flush=True)
            args.captions_only = True
    if args.max_hours <= 0:
        parser.error('--max-hours must be positive')
    if any(m not in ('LLaVA-1.5-7B', 'InternVL3-8B') for m in args.models):
        parser.error('Use the two prespecified held-out victim checkpoints')
    if args.mode == 'setup':
        setup(root)
        return 0
    if args.mode == 'prepare':
        prepare(root, args.source_root, args.source_prefix, args.only)
        return 0
    if args.mode == 'restore':
        restore_evaluations(root, args.backup_prefix)
        return 0
    if args.mode == 'report':
        report(root)
        return 0
    if args.mode == 'worker':
        return worker(root, args.task, args.models[0], args.batch_size,
                      not args.captions_only, args.max_hours, not args.no_backup)
    state_path = root / 'evaluation_state.json'
    state = read_json(state_path) if state_path.exists() else {'tasks': {}, 'model_revisions': {}}
    state['backup_prefix'] = args.backup_prefix
    write_json(state_path, state)
    exe = root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not exe.is_file():
        raise RuntimeError('Run setup first')
    started = time.monotonic()
    for model in args.models:
        for config in read_json(root / 'ready_configurations.json'):
            if args.only and config['id'] not in args.only:
                continue
            task = config['id'] + '__' + model
            state = read_json(state_path)
            prior = state['tasks'].get(task)
            status_file = Path(prior) / ('status.json' if args.captions_only else 'evaluation/status.json') if prior else None
            if status_file and status_file.exists() and read_json(status_file).get('status') == 'complete':
                print('SKIP COMPLETE:', task, flush=True)
                continue
            remaining = args.max_hours - (time.monotonic() - started) / 3600
            if remaining <= 0:
                return 75
            command = [str(exe), '-u', str(Path(__file__).resolve()), 'worker', '--root', str(root),
                       '--task', config['id'], '--models', model, '--batch-size', str(args.batch_size),
                       '--max-hours', str(remaining)]
            if args.captions_only:
                command.append('--captions-only')
            if args.no_backup:
                command.append('--no-backup')
            log = root / (task + '.log')
            print('EVALUATING:', task, 'batch=', args.batch_size, flush=True)
            with log.open('a', encoding='utf-8') as output:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        output.write(line)
                        output.flush()
                    code = process.wait()
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=30)
            report(root)
            if code != 0:
                return code
    report(root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
