"""Resume an unchanged controller after a bounded, identified Drive network failure.

This optional supervisor does not edit the attack runner or checkpoint identity.
Usage: python storage_watchdog.py /path/to/run --max-retries 5
It exits on algorithm/CUDA/API errors, normal completion, or session-budget pause.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path


def storage_timeout(trace):
    """Classify only the latest terminal traceback; never match an older appended error."""
    start = trace.rfind('Traceback (most recent call last):')
    if start < 0:
        return False
    current = trace[start:]
    lines = [line.strip() for line in current.splitlines() if line.strip()]
    if not lines:
        return False
    transport = ('googleapiclient' in current or 'httplib2' in current)
    terminal = lines[-1].startswith(('TimeoutError:', 'ConnectionResetError:', 'BrokenPipeError:',
        'http.client.RemoteDisconnected:', 'RemoteDisconnected:', 'ssl.SSLError:'))
    return transport and terminal


def alive(record):
    try:
        raw = Path(f'/proc/{int(record["pid"])}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return fields[0] != 'Z' and fields[19] == record['start_ticks']
    except (OSError, KeyError, ValueError, TypeError):
        return False


def tail(path, size=24000):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(max(0, Path(path).stat().st_size - size))
            return stream.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def main(root, max_retries=5):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    from baseline_runtime import fingerprint, read_json, sha, write_json
    from notebook_session import Session

    lock = (root / 'storage_watchdog.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Storage watchdog already running.', flush=True)
        return 0
    cfg = read_json(root / 'config.json')
    expected = read_json(root / 'identity.json')['fingerprint']
    actual, _ = fingerprint(root, cfg)
    if actual != expected:
        raise ValueError('Runner identity changed; automatic resume is prohibited.')
    previous = read_json(root / 'storage_watchdog_status.json') if (root / 'storage_watchdog_status.json').exists() else {}
    retries = int(previous.get('retries', 0))
    started = time.time()
    deadline = started + float(cfg['session_hours']) * 3600

    def status(stage, **extra):
        record = dict(stage=stage, updated_unix=time.time(), retries=retries,
            max_retries=max_retries, supervisor_pid=os.getpid(),
            supervisor_source_sha256=sha(Path(__file__)), run_fingerprint=expected, **extra)
        write_json(root / 'storage_watchdog_status.json', record)
        print(json.dumps(record), flush=True)

    status('watching_existing_controller')
    while time.time() < deadline:
        pointer = read_json(root / 'controller_pid.json') if (root / 'controller_pid.json').exists() else {}
        if alive(pointer):
            time.sleep(10)
            continue
        run = read_json(root / 'status.json') if (root / 'status.json').exists() else {}
        stage = run.get('stage')
        if stage not in ('attack_failed', 'failed'):
            status('stopped', controller_stage=stage, reason='No retryable controller failure')
            return 0
        name = run.get('method') if stage == 'attack_failed' else 'controller'
        if name not in ('foa', 'mpc', 'saga', 'controller'):
            raise ValueError('Unexpected method in controller status')
        if not storage_timeout(tail(root / 'logs' / (name + '.log'))):
            status('needs_review', controller_stage=stage, method=name)
            return 1
        if retries >= max_retries:
            status('retry_limit_reached', method=name)
            return 1
        retries += 1
        delay = min(30 * 2 ** (retries - 1), 300)
        status('backoff', delay_seconds=delay, method=name)
        for _ in range(delay // 5):
            if time.time() >= deadline:
                status('session_budget_reached')
                return 75
            time.sleep(5)
        # A user may have resumed manually during the backoff.
        pointer = read_json(root / 'controller_pid.json')
        if alive(pointer):
            continue
        actual, _ = fingerprint(root, cfg)
        if actual != expected:
            raise ValueError('Runner identity changed during backoff')
        pid = Session(root, cfg).launch()
        status('resumed_after_drive_timeout', controller_pid=pid, method=name)
    status('session_budget_reached')
    return 75


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--max-retries', type=int, default=5)
    args = parser.parse_args()
    if not 0 <= args.max_retries <= 10:
        parser.error('--max-retries must be between 0 and 10')
    sys.exit(main(args.root, args.max_retries))
