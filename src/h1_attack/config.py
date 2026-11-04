"""Explicit configuration and source/data identity; no embedded runtime source."""
import hashlib
import json
import re
from pathlib import Path

from pipeline import digest

MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["run_id"]):
        raise ValueError("Unsafe run ID")
    if config["variants"] != ["global", "concept"]:
        raise ValueError("Pilot requires matched global/concept arms")
    settings = config["attack"]
    for key, expected in {"steps": 300, "epsilon": 16, "seed": 2023, "alpha": .005}.items():
        if settings[key] != expected:
            raise ValueError(f"Unsupported pilot attack setting: {key}")
    if not 0 < settings["budget_seconds_per_variant"] <= 3600:
        raise ValueError("Attack budget must be 1..3600 seconds per variant")
    caption = config["caption"]
    if not 0 < caption["budget_seconds"] <= 1800:
        raise ValueError("Caption budget must be 1..1800 seconds")
    expected = {"model": "qwen", "revision": MODEL_REVISION, "dtype": "float16", "device": "cuda"}
    if any(caption[key] != value for key, value in expected.items()):
        raise ValueError("Caption protocol differs from fixed pilot")
    if config["evaluation"]["mode"] != "caption_diagnostic":
        raise ValueError("Official/paid judge has no configured protocol")
    return config


def source_fingerprint(source, config):
    files = {}
    for path in sorted(Path(source).rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file() or "__pycache__" in relative.parts:
            continue
        if relative.parts[0] in {"baseline", "data"} or (len(relative.parts) == 1 and path.suffix in {".py", ".txt"}):
            files[relative.as_posix()] = digest(path)
    payload = {"config": config, "source_and_data": files}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), files


def validate_assets(source):
    """Read-only source-lock and pilot-data validation, without inference."""
    import staging
    baseline = source / "baseline/author"
    lock = json.loads((baseline / "SOURCE_LOCK.json").read_text(encoding="utf-8"))
    for relative, expected in lock["m_attack_v2"]["files"].items():
        path = baseline / "m_attack_v2" / relative
        if digest(path) != expected:
            raise ValueError(f"Frozen baseline source changed: {relative}")
    pilot = source / "data/pilot"
    manifest = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    if [pair["stem"] for pair in manifest["pairs"]] != ["0", "1"]:
        raise ValueError("Only reviewed fixed pilot pair IDs 0,1 are supported")
    for pair in manifest["pairs"]:
        for role in ("source", "target"):
            staging.checked_asset(pilot, pair[role])
        for references in pair["variants"].values():
            for asset in references:
                staging.checked_asset(pilot, asset)
