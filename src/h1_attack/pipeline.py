"""File-backed stage execution; importing this module never starts an experiment."""
import hashlib
import json
import time
import traceback
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_stage(directory, action, previous):
    """Reuse verified completed output; failed/interrupted stages need review."""
    record_path = directory / "stage.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["state"] != "completed":
            raise RuntimeError(f"{directory.name}: prior failed/interrupted attempt; no automatic retry")
        for relative, expected in record["files"].items():
            path = directory / relative
            if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != expected:
                raise ValueError(f"{directory.name}: saved artifact missing/changed: {relative}")
        return record

    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json(record_path, {"state": "started"})
    try:
        result = action(directory, previous)
        if result.get("state") != "completed":
            raise RuntimeError("Stage did not confirm completion")
        files = {
            path.relative_to(directory).as_posix(): digest(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path != record_path
        }
        if not files:
            raise RuntimeError("Stage produced no durable evidence")
        record = {"state": "completed", "result": result, "files": files,
                  "wall_seconds": time.time() - started}
        write_json(record_path, record)
        return record
    except BaseException as exc:
        write_json(record_path, {"state": "failed", "error": str(exc),
                                "traceback": traceback.format_exc(),
                                "wall_seconds": time.time() - started})
        raise


def run_pipeline(root, identity, actions):
    """Run ordered attack → captions → evaluation callbacks with independent records."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "identity.json"
    if lock.exists():
        if json.loads(lock.read_text(encoding="utf-8"))["fingerprint"] != identity:
            raise ValueError("Run identity differs; use a reviewed new run root")
    else:
        write_json(lock, {"fingerprint": identity})
    records = {}
    summary = {"state": "running", "official_asr": None, "stages": records}
    try:
        for name, action in actions.items():
            print(f"H1_STAGE {name}", flush=True)
            records[name] = run_stage(root / name, action, records)
        summary["state"] = "completed_caption_diagnostic"
    except Exception as exc:
        summary.update(state="failed", error=str(exc), traceback=traceback.format_exc())
    write_json(root / "summary.json", summary)
    return summary
