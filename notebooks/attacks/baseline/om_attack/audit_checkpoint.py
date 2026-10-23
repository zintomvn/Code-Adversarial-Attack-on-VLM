"""Download one real Drive checkpoint and validate restoration in a fresh directory.

Run with the attack's own Python interpreter; this never runs or alters an attack.
The script belongs under validation/ on the remote machine, outside the frozen bundle.
"""

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def audit(root: Path, method: str) -> dict:
    sys.path.insert(0, str(root))
    from campaign_runtime import Drive, extract, read_json, write_json
    from campaign_state import State
    from author_worker import rng_state

    config = read_json(root / "config.json")
    remote = config["drive_prefix"].strip("/") + "/" + config["run_name"]
    drive = Drive(config["drive_parent_id"])
    folder = drive.folder(remote + "/checkpoints/" + method, create=False)
    if not folder:
        raise FileNotFoundError("No completed checkpoint on Drive: " + method)
    archives = sorted(
        (item for item in drive.children(folder) if item["name"].endswith(".zip")),
        key=lambda item: item["name"],
    )
    if not archives:
        raise FileNotFoundError("No completed batch archive: " + method)
    first = archives[0]
    validation = root / "validation"
    validation.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="restore_", dir=validation) as directory:
        fresh = Path(directory)
        archive = fresh / first["name"]
        drive.download(first["id"], archive, first.get("md5Checksum"))
        for name in ("config.json", "identity.json", "selected_pairs.json"):
            (fresh / name).write_bytes((root / name).read_bytes())
        isolated_config = read_json(fresh / "config.json")
        isolated_config["backup_drive"] = False
        write_json(fresh / "config.json", isolated_config)
        extract(archive, fresh)

        state = State(fresh, method)
        expected = min(config["method_settings"][method]["batch_size"], config["n"])
        if len(state.done) != expected:
            raise AssertionError("First batch has unexpected sample count")
        expected_rng = state.resume_rng
        if expected_rng is None:
            raise AssertionError("Batch RNG checkpoint is missing")
        state.restore()
        if json.dumps(rng_state(), sort_keys=True) != json.dumps(expected_rng, sort_keys=True):
            raise AssertionError("RNG state differs after restoration")
        report = {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "run_name": config["run_name"],
            "run_fingerprint": state.run_id,
            "archive_id": first["id"],
            "archive_md5": first.get("md5Checksum"),
            "restored_samples": len(state.done),
            "png_linf_max": max(row["saved_png_linf_0_255"] for row in state.done.values()),
            "restored_from_drive": True,
            "artifact_hashes_valid": True,
            "rng_roundtrip_exact": True,
            "environment": state.environment,
        }

    output = validation / (method + "_drive_restore.json")
    write_json(output, report)
    drive.put(output, remote + "/validation/" + output.name)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("method", choices=("o_attack", "m_attack_v2"))
    args = parser.parse_args()
    print(json.dumps(audit(args.root.resolve(), args.method), indent=2))
