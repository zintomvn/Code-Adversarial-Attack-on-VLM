"""H1 entry point: validate by default; explicitly execute only a reviewed config."""
import argparse
import json
import os
from pathlib import Path

from config import load_config, source_fingerprint, validate_assets
from pipeline import run_pipeline, write_json


def main():
    source = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=source / "configs/pilot.json")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--execute", action="store_true", help="Run only after explicit plan/config approval")
    parser.add_argument("--install-env", action="store_true", help="Install pinned environment within attack budget")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_assets(source)
    fingerprint, files = source_fingerprint(source, config)
    if not args.execute:
        print(json.dumps({"state": "validated_not_run", "run_id": config["run_id"],
                          "fingerprint": fingerprint, "files": len(files),
                          "approval": config["approval"], "official_asr": None}, indent=2))
        return 0
    approval = config["approval"]
    if approval.get("status") != "approved" or not approval.get("reviewed_by") or not approval.get("reviewed_at"):
        parser.error("Experiment pending explicit user approval; do not self-approve the config")
    if os.name != "posix":
        parser.error("GPU execution requires Linux (Molab/Kaggle); local validation works on Windows")
    root = args.output.resolve() / config["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    from runner import build_actions, collect_results
    import fcntl
    with (root / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actions = build_actions(source, root, config, args.install_env)
        result = run_pipeline(root, fingerprint, actions)
        write_json(root / "config.json", config)
        write_json(root / "source_manifest.json", files)
        collect_results(root, root / "results.zip")
    print(json.dumps(result, indent=2))
    print("H1_RESULTS", str(root / "results.zip"), flush=True)
    return 0 if result["state"] == "completed_caption_diagnostic" else 1


if __name__ == "__main__":
    raise SystemExit(main())
