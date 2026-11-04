"""Bounded Linux worker processes, suitable for Molab and Kaggle."""
import os
import signal
import subprocess
import time


def command(args, env, log_path, budget_seconds):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + budget_seconds
    with log_path.open("a", encoding="utf-8") as log:
        child = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            while child.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Worker exceeded {budget_seconds}s budget")
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f"Worker exit {child.returncode}; see {log_path}")
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
