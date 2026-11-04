"""Stage and invoke the frozen campaign M-V2 worker with a verified retrieval hook."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(p)


def checked_asset(kit, record):
    path = (Path(kit) / record["path"]).resolve()
    if not path.is_relative_to(Path(kit).resolve()) or sha(path) != record["sha256"]:
        raise ValueError("Asset missing, changed or outside kit: " + record["path"])
    return path


def copy_verified(src, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha(src) != sha(dest):
            raise ValueError("Refusing to overwrite changed staged file: " + str(dest))
    else:
        shutil.copyfile(src, dest)


def stage_assets(kit, manifest, variant, root):
    if variant not in ("global", "concept") or not 2 <= len(manifest["pairs"]) <= 5:
        raise ValueError("Only global/concept with 2..5 pairs supported; single-pair squeeze is unsafe")
    root = Path(root)
    rows, retrieval, stems = [], {}, set()
    forbidden = {p[role]["sha256"] for p in manifest["pairs"] for role in ("source", "target")}
    for pair in manifest["pairs"]:
        stem = pair["stem"]
        if stem in stems or not isinstance(stem, str) or not stem.isdecimal():
            raise ValueError("Pair stems must be unique numeric author IDs")
        stems.add(stem)
        row = {"index": pair["index"], "stem": stem}
        if pair["source"]["sha256"] == pair["target"]["sha256"]:
            raise ValueError("Primary source and target are identical")
        for role in ("source", "target"):
            asset = pair[role]
            rel = Path(asset["path"])
            if rel.parts[0] != role or rel.stem != stem:
                raise ValueError("Source-target role/stem mismatch; swaps are prohibited")
            src = checked_asset(kit, asset)
            copy_verified(src, root / "inputs" / role / "images" / src.name)
            row[role] = "inputs/" + role + "/images/" + src.name
            row[role + "_sha256"] = asset["sha256"]
        auxiliaries = pair["variants"][variant]
        hashes = [a["sha256"] for a in auxiliaries]
        if len(auxiliaries) != 2 or len(set(hashes)) != 2 or forbidden.intersection(hashes):
            raise ValueError("Two distinct auxiliary references required; no clean source/target padding")
        retrieval[stem] = {}
        for rank, asset in enumerate(auxiliaries, 1):
            copy_verified(checked_asset(kit, asset), root / "retrieval" / stem / f"{rank}.jpg")
            retrieval[stem][str(rank)] = asset["sha256"]
        rows.append(row)
    rows.sort(key=lambda r: Path(r["source"]).name)
    if len({r["index"] for r in rows}) != len(rows):
        raise ValueError("Pair indices must be unique")
    return rows, retrieval


def validate_retrieval(root, expected):
    root = Path(root)
    for stem, ranks in expected.items():
        for rank, digest in ranks.items():
            p = root / "retrieval" / stem / f"{rank}.jpg"
            if not p.is_file() or sha(p) != digest:
                raise ValueError("Experimental auxiliary changed: " + str(p))


def fingerprint(manifest, variant, files):
    value = {"manifest": manifest, "variant": variant, "files": files,
             "settings": {"configured_batch_size": 10, "steps": 300, "epsilon": 16,
                          "alpha": .005, "seed": 2023, "retrieval_hook": "verify_only_no_download"}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def run_config(manifest, variant):
    return {"run_name": "full_mv2_ata_" + variant + "_pilot", "methods": ["m_attack_v2"],
            "n": len(manifest["pairs"]), "epsilon": 16, "steps": 300, "cpu_threads": 4,
            "session_hours": 7.5, "backup_drive": False, "drive_parent_id": "", "drive_prefix": "",
            "data_archive": "", "data_sha256": "inputs_locked_in_kit_manifest", "run_evaluation": False,
            "model_revisions": manifest["model_revisions"],
            "method_settings": {"m_attack_v2": {"seed": 2023, "alpha": .005, "batch_size": 10, "overrides": {}}},
            "retrieval": {"allow_download": False, "reference_root": ""},
            "evaluation": {"success_threshold": .5, "legacy_secondary_threshold": .3}}


def prepare(kit, root, variant):
    kit, root = Path(kit).resolve(), Path(root).resolve()
    lock = read(kit / "KIT_LOCK.json")
    for rel, digest in lock["files"].items():
        checked_asset(kit, {"path": rel, "sha256": digest})
    manifest = read(kit / "manifest.json")
    run_id = fingerprint(manifest, variant, lock["files"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "identity.json").exists() and read(root / "identity.json")["fingerprint"] != run_id:
        raise ValueError("Existing run identity differs; choose another run root")
    for rel in lock["files"]:
        if rel.startswith("runtime/"):
            copy_verified(kit / rel, root / Path(rel).relative_to("runtime"))
    rows, expected = stage_assets(kit, manifest, variant, root)
    write(root / "selected_pairs.json", rows)
    write(root / "config.json", run_config(manifest, variant))
    write(root / "experimental_retrieval.json", expected)
    write(root / "identity.json", {"fingerprint": run_id, "variant": variant,
        "kit_lock": lock, "manifest": manifest, "actual_batch_size": len(rows), "configured_batch_size": 10})
    if not (root / "status.json").exists():
        write(root / "status.json", {"status": "prepared_not_run", "full_m_v2_attack_completed": False,
            "asr": None, "evaluation": "not_run", "configured_batch_size": 10, "actual_batch_size": len(rows)})
    return root


def execute(root):
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Original config_schema requires Python 3.10; use the supplied frozen environment installer")
    expected_versions = {"torch": "2.10.0", "torchvision": "0.25.0", "transformers": "4.51.3"}
    for package, expected in expected_versions.items():
        if importlib.metadata.version(package).split("+")[0] != expected:
            raise RuntimeError("Pinned runtime required: " + package + "==" + expected)
    sys.path.insert(0, str(root))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Full original M-V2 requires CUDA")
    import retrieval_worker
    expected = read(root / "experimental_retrieval.json")
    original = retrieval_worker.ensure_retrieval

    def experimental_retrieval(actual_root, cfg, drive):
        if Path(actual_root).resolve() != root or drive is not None:
            raise ValueError("Unexpected retrieval hook root/storage")
        validate_retrieval(root, expected)
        write(root / "retrieval_manifest.json", {"reference": "locked_experimental_ATA_no_download", "inputs": expected})
        print("EXPERIMENTAL_ATA_VERIFIED_NO_DOWNLOAD", len(expected), flush=True)

    retrieval_worker.ensure_retrieval = experimental_retrieval
    import campaign_worker
    previous = sys.argv[:]
    try:
        sys.argv = [str(root / "campaign_worker.py"), str(root), "m_attack_v2"]
        code = campaign_worker.main()
        done = list((root / "results/m_attack_v2/samples").glob("*/complete.json"))
        if code != 0 or len(done) != len(read(root / "selected_pairs.json")):
            raise RuntimeError("Frozen worker did not commit the entire two-pair batch")
        write(root / "status.json", {"status": "full_m_v2_attack_completed_evaluation_pending",
            "full_m_v2_attack_completed": True, "completed_pairs": len(done), "asr": None,
            "evaluation": "not_run", "configured_batch_size": 10, "actual_batch_size": len(done)})
    except Exception as exc:
        write(root / "status.json", {"status": "full_m_v2_failed", "full_m_v2_attack_completed": False,
                                    "error": type(exc).__name__ + ": " + str(exc), "asr": None})
        raise
    finally:
        retrieval_worker.ensure_retrieval = original
        sys.argv = previous


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kit", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--variant", choices=("global", "concept"), required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--install-env", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = p.parse_args()
    root = prepare(args.kit, args.run_root, args.variant)
    os.environ.update(WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_XET="1",
                      HF_HOME=str(root.parent / "hf_cache"), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    if args.install_env:
        sys.path.insert(0, str(root))
        from campaign_runtime import interpreter
        print("PINNED_PYTHON", interpreter(root, "m_attack_v2"), flush=True)
    elif args.execute:
        execute(root)
    else:
        print(json.dumps({"status": "prepared_not_run", "run_root": str(root), "variant": args.variant,
                          "fingerprint": read(root / "identity.json")["fingerprint"]}))


if __name__ == "__main__":
    main()
