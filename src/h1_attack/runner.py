"""Concrete H1 stages. All implementation lives in ordinary source files."""
import hashlib
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

from pipeline import digest, write_json
from process import command


def prepare_kit(source, runtime):
    """Stage readable baseline/data into the existing frozen worker's kit format."""
    kit = runtime / "kit"
    kit.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "baseline", kit / "runtime", dirs_exist_ok=True)
    shutil.copytree(source / "data/pilot", kit, dirs_exist_ok=True)
    shutil.copyfile(source / "staging.py", kit / "launch.py")
    manifest = json.loads((kit / "manifest.json").read_text(encoding="utf-8"))
    files = {path.relative_to(kit).as_posix(): digest(path)
             for path in sorted(kit.rglob("*"))
             if path.is_file() and path.name != "KIT_LOCK.json" and "__pycache__" not in path.parts}
    write_json(kit / "KIT_LOCK.json", {
        "files": files, "no_models_embedded": True,
        "original_release_bundle_sha256": manifest["original_release_bundle_sha256"],
    })
    # Validate role pairing, selected references and every asset hash before GPU use.
    import staging
    for variant in ("global", "concept"):
        staging.prepare(kit, runtime / ("validate_" + variant), variant)
    return kit


def archive_attack(work, variant, destination):
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(work.rglob("*")):
            if path.is_file() and not any(part in {"__pycache__", "checkpoints", "staging", "author_output"} for part in path.relative_to(work).parts):
                archive.write(path, variant + "/" + path.relative_to(work).as_posix())


def build_actions(source, root, config, install_environment=False):
    """Construct callbacks; no execution or model loading occurs here."""
    runtime = root / "_runtime"
    pinned = runtime / "environments/om_v2_m_attack_v2_py310_tf4513/bin/python"
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="2023",
               H1_SHARED_HF_CACHE=str(runtime / "hf_cache"), HF_HOME=str(runtime / "hf_cache"),
               WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_XET="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")

    def attack_stage(variant):
        def execute(directory, previous):
            started = time.monotonic()
            budget = config["attack"]["budget_seconds_per_variant"]
            kit = prepare_kit(source, runtime)

            def bounded(args, log):
                remaining = budget - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Attack/setup budget exhausted")
                command(args, env, directory / log, remaining)

            if not pinned.is_file():
                if not install_environment:
                    raise RuntimeError("Pinned environment missing; use reviewed --install-env launch")
                bounded([sys.executable, str(source / "staging.py"), "--kit", str(kit),
                         "--run-root", str(runtime / "environment_setup"), "--variant", "global",
                         "--install-env"], "setup.log")
            if not pinned.is_file():
                raise RuntimeError("Pinned interpreter missing after setup")
            bounded([str(pinned), str(source / "attack.py"), "--kit", str(kit),
                     "--run-root", str(directory / "selftest"), "--selftest",
                     "--selftest-device", "cuda"], "selftest.log")
            if "H1_OVERLAY_SYNTHETIC_INTEGRATION_PASS" not in (directory / "selftest.log").read_text():
                raise RuntimeError("Synthetic worker integration did not pass")
            work = directory / "work"
            bounded([str(pinned), "-u", str(source / "attack.py"), "--kit", str(kit),
                     "--run-root", str(work), "--variant", variant], "attack.log")
            from verification import verify
            verification = verify(work)
            archive = directory / "attack.zip"
            archive_attack(work, variant, archive)
            summary = {**verification, "state": "completed_verified", "variant": variant,
                       "archive_sha256": digest(archive), "run_id": config["run_id"]}
            write_json(directory / "attack_summary.json", summary)
            return {"state": "completed", "attack": summary, "official_asr": None}
        return execute

    def captions(directory, previous):
        from data import build_caption_bundle
        records = {}
        for variant in config["variants"]:
            stage = root / ("attack_" + variant)
            records[variant] = {
                "summary": previous["attack_" + variant]["result"]["attack"],
                "archive": (stage / "attack.zip").read_bytes(),
            }
        bundle = build_caption_bundle(records, config["attack"]["epsilon"])
        inputs = directory / "inputs"
        inputs.mkdir()
        with zipfile.ZipFile(io.BytesIO(bundle["archive"])) as archive:
            for member in archive.infolist():
                if not (inputs / member.filename).resolve().is_relative_to(inputs.resolve()):
                    raise ValueError("Unsafe caption input path")
            archive.extractall(inputs)
        settings = config["caption"]
        command([str(pinned), "-u", str(source / "caption.py"), "--manifest", str(inputs / "manifest.json"),
                 "--output", str(directory / "output"), "--model", settings["model"],
                 "--revision", settings["revision"], "--dtype", settings["dtype"],
                 "--device", settings["device"]], env, directory / "caption.log", settings["budget_seconds"])
        raw = json.loads((directory / "output/captions.json").read_text(encoding="utf-8"))
        if len(raw["rows"]) != len(bundle["manifest"]["rows"]) or any(row["status"] != "captioned" for row in raw["rows"]):
            raise RuntimeError("Incomplete caption results")
        return {"state": "completed", "unique_images": len(raw["rows"]), "official_asr": None}

    def evaluation(directory, previous):
        from evaluation import make_review
        raw = json.loads((root / "captions/output/captions.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "captions/inputs/manifest.json").read_text(encoding="utf-8"))
        review = make_review(raw, manifest, directory, config["evaluation"]["term_map"])
        write_json(directory / "diagnostic.json", review)
        return {"state": "completed", "diagnostic": review, "official_asr": None,
                "official_judge": "PENDING", "human_semantic_review": "PENDING"}

    return {"attack_global": attack_stage("global"), "attack_concept": attack_stage("concept"),
            "captions": captions, "evaluation": evaluation}


def collect_results(root, destination):
    """Bundle completed/partial evidence, excluding environments and model downloads."""
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if path.is_file() and path != destination and not any(part in {"_runtime", "__pycache__"} for part in relative.parts):
                archive.write(path, relative.as_posix())
