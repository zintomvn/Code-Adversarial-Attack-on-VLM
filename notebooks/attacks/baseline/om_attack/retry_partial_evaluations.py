"""Retry incomplete evaluation models once after the active controller exits.

Deploy under validation/, outside the frozen runner bundle. Uses the original
worker, configuration, caption limits, rubric and checkpoint merge rules.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path


def process_alive(record: dict) -> bool:
    try:
        value = Path(f'/proc/{record["pid"]}/stat').read_text()
    except FileNotFoundError:
        return False
    fields = value[value.rfind(")") + 2:].split()
    return fields[0] != "Z" and fields[19] == record["start_ticks"]


def main(root: Path, wait_hours: float, retry_minutes: float) -> int:
    sys.path.insert(0, str(root))
    from campaign_runtime import Drive, read_json, stage, write_json

    cfg = read_json(root / "config.json")
    original = read_json(root / "controller_pid.json")
    deadline = time.time() + wait_hours * 3600
    print("WAIT_FOR_ACTIVE_CONTROLLER", original["pid"], flush=True)
    while process_alive(original):
        if time.time() >= deadline:
            print("FOLLOWUP_WAIT_LIMIT_REACHED", flush=True)
            return 75
        time.sleep(20)

    # A session boundary must be handled by the normal resume workflow.
    status = read_json(root / "status.json")
    if status["stage"] != "evaluation_pending":
        print("FOLLOWUP_NOT_NEEDED", status["stage"], flush=True)
        return 0
    with (root / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("ANOTHER_CONTROLLER_IS_RUNNING", flush=True)
            return 0
        drive = Drive(cfg["drive_parent_id"]) if cfg["backup_drive"] else None
        remote = cfg["drive_prefix"].strip("/") + "/" + cfg["run_name"]
        pending, attempts = [], []
        exe = root.parent / "environments/evaluation/bin/python"
        stat = Path(f"/proc/{os.getpid()}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        write_json(root / "controller_pid.json",
                   {"pid": os.getpid(), "start_ticks": fields[19]})
        for row in status.get("pending_models", []):
            model = row["model"]
            provider = cfg["evaluation"]["model_settings"][model]["provider"]
            key = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
            if key and not os.environ.get(key):
                pending.append(row)
                continue
            env = os.environ.copy()
            env.update(PYTHONPATH=str(root), HF_HOME=str(root.parent / "hf_cache"),
                       HF_HUB_DISABLE_XET="1", TOKENIZERS_PARALLELISM="false",
                       OMP_NUM_THREADS=str(cfg["cpu_threads"]),
                       BASELINE_DEADLINE=str(time.time() + retry_minutes * 60))
            stage(root, "evaluating", model=model, bounded_retry=True)
            print("RETRY_INCOMPLETE_MODEL_ONCE", model, flush=True)
            with (root / "logs/evaluation.log").open("a") as log:
                code = subprocess.run(
                    [str(exe), "-u", str(root / "campaign_worker.py"), str(root),
                     "evaluation", "--model", model], env=env, stdout=log,
                    stderr=subprocess.STDOUT, check=False,
                ).returncode
            attempts.append({"model": model, "exit_code": code})
            if code:
                pending.append({"model": model, "exit_code": code})
        report = root / "validation/evaluation_retry_report.json"
        write_json(report, {"attempts": attempts, "pending_models": pending,
                            "configuration_changed": False})
        stage(root, "evaluation_pending" if pending else "complete", pending_models=pending)
        if drive:
            drive.put(report, remote + "/validation/" + report.name)
            drive.put(root / "status.json", remote + "/status.json")
        return 76 if pending else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--wait-hours", type=float, default=6)
    parser.add_argument("--retry-minutes", type=float, default=45)
    args = parser.parse_args()
    raise SystemExit(main(args.root.resolve(), args.wait_hours, args.retry_minutes))
