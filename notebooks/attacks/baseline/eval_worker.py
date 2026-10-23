"""Portable evaluation runner using classes extracted verbatim from v3.2.

The generated eval_protocol_v32.py preserves CaptionGenerator, GPTScorer, rubric,
retry logic, rate limiter and safe_error from text_gen_eval_v3.2.ipynb cells 5-7.
Only scheduling, storage paths, batch size and configuration are supplied here.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import mimetypes
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from baseline_runtime import Drive, read_json, sha, write_json
from author_worker import pin_model_downloads


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_protocol(root, cfg, model):
    from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError
    from tqdm.auto import tqdm
    settings = cfg['evaluation']
    file = root / 'eval_protocol_v32.py'
    spec = importlib.util.spec_from_file_location('baseline_v32', file)
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(dict(Path=Path, os=os, json=json, hashlib=hashlib, random=random,
        time=time, math=math, threading=threading, mimetypes=mimetypes, gc=gc,
        datetime=datetime, timezone=timezone, SimpleNamespace=SimpleNamespace, tqdm=tqdm,
        OpenAI=OpenAI, APIConnectionError=APIConnectionError, APITimeoutError=APITimeoutError,
        APIStatusError=APIStatusError, ThreadPoolExecutor=ThreadPoolExecutor,
        RUN_PIPELINE=False, RUN_EVALUATION=False, TRACE_VERBOSE=False,
        JUDGE_MODEL=settings['judge_model'], JUDGE_RPM=settings['judge_rpm'],
        JUDGE_TPM=settings['judge_tpm'], JUDGE_MAX_OUTPUT_TOKENS=100,
        JUDGE_RETRY_BACKOFF_CAP=180, JUDGE_SMOOTH_REQUESTS=False,
        JUDGE_TIMEOUT_SECONDS=45, JUDGE_MAX_ATTEMPTS=4,
        MODEL_LOCAL_DIR=None, SUCCESS_THRESHOLD=settings['success_threshold'],
        SUCCESS_OPERATOR=settings['success_operator']))
    import torch
    module.torch = torch
    spec.loader.exec_module(module)
    module.TRACE_LOG_PATH = root / 'logs' / 'evaluation_trace.jsonl'
    return module


def evaluate(root, model):
    root = Path(root).resolve()
    cfg = read_json(root / 'config.json')
    setting = cfg['evaluation']
    model_cfg = setting['model_settings'][model]
    slug = ''.join(c if c.isalnum() or c in '-_' else '_' for c in model)
    output = root / 'evaluation' / slug
    output.mkdir(parents=True, exist_ok=True)
    drive = Drive(cfg['drive_parent_id']) if cfg['backup_drive'] else None
    remote = cfg['drive_prefix'].strip('/') + '/' + cfg['run_name'] + '/evaluation/' + slug
    if drive:
        # Merge a newer local batch after an interrupted upload; never replace a valid
        # caption/score with an error or silently resolve two different successful results.
        restored = root / 'evaluation_restore' / (slug + '_' + str(time.time_ns()))
        drive.restore_tree(remote, restored)
        for name in ('captions.json', 'scores.json'):
            remote_file, local_file = restored / name, output / name
            if not remote_file.exists():
                continue
            merged = read_json(remote_file)
            if local_file.exists():
                for key, row in read_json(local_file).items():
                    previous = merged.get(key)
                    if previous and previous.get('status') == row.get('status') == 'ok' and previous != row:
                        raise ValueError('Conflicting successful evaluation checkpoints')
                    if not previous or row.get('status') == 'ok':
                        merged[key] = row
            write_json(local_file, merged)
    protocol = load_protocol(root, cfg, model)
    pin_model_downloads(cfg['model_revisions'])
    caption_cfg = SimpleNamespace(**model_cfg, prompt=setting['prompt'],
        max_new_tokens=setting['max_new_tokens'], max_image_pixels=512 * 512,
        min_image_pixels=256 * 256, attention='sdpa',
        revision=cfg['model_revisions'].get(model_cfg['model_id'], 'main'),
        timeout=90, max_output_tokens=2048, reasoning_effort='minimal')
    identity = dict(model=model, caption=vars(caption_cfg), protocol_sha256=sha(root / 'eval_protocol_v32.py'))
    image_jobs, pairs = {}, []
    for method in cfg['methods']:
        directory = root / 'results' / method
        rows = [json.loads(s) for s in (directory / 'manifest.jsonl').read_text().splitlines() if s.strip()]
        if len(rows) != cfg['n']:
            raise ValueError(f'Incomplete attack manifest: {method}')
        for row in rows:
            pair = dict(method=method, index=row['index'])
            for role in ('target', 'adversarial'):
                file = directory / row[role]
                key = digest(dict(identity=identity, image_sha256=sha(file)))
                image_jobs[key] = file
                pair[role] = key
            pairs.append(pair)
    cap_path, scores_path = output / 'captions.json', output / 'scores.json'
    captions = read_json(cap_path) if cap_path.exists() else {}
    scores = read_json(scores_path) if scores_path.exists() else {}
    for key, row in captions.items():
        if key not in image_jobs or row.get('image_sha256') != sha(image_jobs[key]):
            raise ValueError('Caption checkpoint identity mismatch')
    def save(path, value):
        write_json(path, value)
        if drive:
            drive.put(path, remote + '/' + path.name)
    pending = [key for key in image_jobs if captions.get(key, {}).get('status') != 'ok']
    if pending and model_cfg['provider'] in ('openai', 'gemini'):
        key_name = 'OPENAI_API_KEY' if model_cfg['provider'] == 'openai' else 'GEMINI_API_KEY'
        try:
            protocol.KaggleRuntime.secret(key_name)
        except RuntimeError:
            save(output / 'status.json', dict(stage='caption_key_pending', model=model, key_name=key_name))
            return 76
    try:
        generator = protocol.CaptionGenerator(SimpleNamespace(caption=caption_cfg)) if pending else None
    except RuntimeError:
        # Missing credentials are resumable; do not fabricate empty captions.
        save(output / 'status.json', dict(stage='caption_credentials_or_runtime_pending', model=model,
            completed=len(image_jobs) - len(pending), expected=len(image_jobs)))
        raise
    def receive(key, row):
        captions[key] = dict(row, image_sha256=sha(image_jobs[key]))
    if model_cfg['provider'] == 'huggingface':
        size = setting['local_batch_size']
        for start in range(0, len(pending), size):
            if time.time() >= float(os.environ.get('BASELINE_DEADLINE', 'inf')):
                return 75
            keys = pending[start:start + size]
            results = generator.generate_batch([image_jobs[k] for k in keys])
            for key, row in zip(keys, results, strict=True):
                receive(key, row)
            save(cap_path, captions)
            print('CAPTIONS', model, start + len(keys), '/', len(pending), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=setting['api_workers']) as pool:
            # Submit one bounded group at a time so a session boundary does not leave a long queue.
            for start in range(0, len(pending), setting['api_workers']):
                if time.time() >= float(os.environ.get('BASELINE_DEADLINE', 'inf')):
                    return 75
                jobs = {pool.submit(generator._api_safe, image_jobs[k]): k
                    for k in pending[start:start + setting['api_workers']]}
                for future in as_completed(jobs):
                    receive(jobs[future], future.result())
                save(cap_path, captions)
    if generator is not None:
        del generator
        gc.collect()
        if protocol.torch.cuda.is_available():
            protocol.torch.cuda.empty_cache()
    if not setting['run_judge']:
        save(output / 'status.json', dict(stage='captions_only', model=model))
        return 0
    try:
        api_key = protocol.KaggleRuntime.secret('OPENAI_API_KEY')
    except RuntimeError:
        save(output / 'status.json', dict(stage='judge_key_pending', model=model))
        return 76
    judge_identity = dict(protocol_sha256=sha(root / 'eval_protocol_v32.py'),
        model=setting['judge_model'], temperature=0, max_tokens=100,
        threshold=setting['success_threshold'], operator=setting['success_operator'])
    todo = []
    for pair in pairs:
        target, adv = (captions.get(pair[r], {}) for r in ('target', 'adversarial'))
        if target.get('status') != 'ok' or adv.get('status') != 'ok':
            continue
        key = digest(dict(judge=judge_identity, target=target['caption'], adversarial=adv['caption']))
        pair['score_key'] = key
        if scores.get(key, {}).get('status') != 'ok':
            todo.append((key, target['caption'], adv['caption']))
    # Deduplicate identical text pairs; each completion is reused only for exactly identical judge inputs.
    todo = list({r[0]: r for r in todo}.values())
    def score(job):
        key, target, adv = job
        scorer = protocol.GPTScorer(api_key, model=setting['judge_model'])
        try:
            value = scorer.compute_similarity(target, adv)
            return key, dict(status='ok', score=value, target=target, adversarial=adv,
                response=scorer.last_response, judge=judge_identity)
        except Exception as exc:
            return key, dict(status='error', error=protocol.safe_error(exc),
                target=target, adversarial=adv, judge=judge_identity)
    with ThreadPoolExecutor(max_workers=setting['judge_workers']) as pool:
        for start in range(0, len(todo), setting['judge_workers']):
            if time.time() >= float(os.environ.get('BASELINE_DEADLINE', 'inf')):
                return 75
            for key, row in pool.map(score, todo[start:start + setting['judge_workers']]):
                scores[key] = row
            save(scores_path, scores)
            print('SCORED', model, start + setting['judge_workers'], '/', len(todo), flush=True)
    summaries, exported = [], []
    for method in cfg['methods']:
        group = [p for p in pairs if p['method'] == method]
        values = []
        for pair in group:
            row = scores.get(pair.get('score_key'), {})
            if row.get('status') == 'ok':
                value = row['score']
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Invalid score in checkpoint')
                values.append(value)
                exported.append(dict(method=method, index=pair['index'], model=model, **row))
        successes = sum(protocol.is_success(v) for v in values)
        summaries.append(dict(method=method, model=model, expected=cfg['n'], judged=len(values),
            pending=cfg['n'] - len(values), successes=successes,
            asr=successes / len(values) if values else None,
            avg_sim=sum(values) / len(values) if values else None,
            complete=len(values) == cfg['n'], judge=judge_identity))
    save(output / 'summary.json', summaries)
    jsonl = output / 'scores.jsonl'
    jsonl.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in exported), encoding='utf-8')
    if drive:
        drive.put(jsonl, remote + '/scores.jsonl')
    table = output / 'summary.csv'
    with table.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in summaries[0] if k != 'judge'])
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != 'judge'} for row in summaries)
    if drive:
        drive.put(table, remote + '/summary.csv')
    save(output / 'status.json', dict(stage='complete' if all(s['complete'] for s in summaries) else 'partial',
        model=model, summaries=summaries))
    return 0 if all(s['complete'] for s in summaries) else 76


if __name__ == '__main__':
    sys.exit(evaluate(sys.argv[1], sys.argv[2]))
