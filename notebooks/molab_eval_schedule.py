"""Schedule GPU captions for all ready tasks before the API judge phase.

The evaluation engine and its fingerprints stay unchanged, including runs already
started using molab_eval_completed.py directly. This file only orders subprocesses.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run'])
    parser.add_argument('--root', type=Path, default=Path('/marimo/eval_completed_n200'))
    parser.add_argument('--max-hours', type=float, default=8)
    args, forwarded = parser.parse_known_args()
    if args.max_hours <= 0:
        parser.error('--max-hours must be positive')
    root = args.root.resolve()
    exe = root / '.venv' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not exe.is_file():
        raise RuntimeError('Run evaluation setup first')
    captions_only = '--captions-only' in forwarded
    forwarded = [arg for arg in forwarded if arg != '--captions-only']
    started = time.monotonic()
    phases = ['captions'] if captions_only else ['captions', 'judge']
    for phase in phases:
        remaining = args.max_hours - (time.monotonic() - started) / 3600
        if remaining <= 0:
            return 75
        command = [str(exe), '-u', str(root / 'molab_eval_completed.py'), 'run', '--root', str(root),
                   '--max-hours', str(remaining), *forwarded]
        if phase == 'captions':
            command.append('--captions-only')
        print('EVALUATION PHASE:', phase, '| Remaining hours:', round(remaining, 3), flush=True)
        with subprocess.Popen(command) as child:
            try:
                code = child.wait()
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=60)
        if code:
            return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
