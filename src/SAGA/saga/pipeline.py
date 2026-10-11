"""SAGA orchestration with two images and one metadata file per sample."""

import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from .attention import extract_attention
from .data import sha256
from .hotspots import build_schedule
from .optimization import optimize
from .surrogates import load_surrogates


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_environment(seed=2023):
    """Match OurAttack._initialize_models() and its separate RNG reset exactly."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_image(image):
    """Preserve the original contiguous CHW tensor and default floating dtype."""
    image = image.convert("RGB")
    tensor = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
    tensor = tensor.view(image.size[1], image.size[0], len(image.getbands()))
    tensor = tensor.permute((2, 0, 1)).contiguous()
    return tensor.to(dtype=torch.get_default_dtype()).unsqueeze(0)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def run(args, pairs):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Image generation requires an NVIDIA CUDA GPU. Use --check_inputs for CPU validation."
        )
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Logical GPU {args.gpu} unavailable; visible count={torch.cuda.device_count()}")
    device = f"cuda:{args.gpu}"
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "INFO")
    output = args.save_path / args.note
    configuration = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in {"resume", "check_inputs", "debug"}
    }
    configuration["effective_hotspot_prob"] = 1.0
    samples = [
        {"id": path.stem, "source": str(path), "source_sha256": sha256(path), "target": caption}
        for path, caption in pairs
    ]
    identity = {"configuration": configuration, "samples": samples}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    completed_ids = set()
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise FileExistsError(f"{output} already contains a run. Choose --note or pass --resume.")
        # A directory with trace files from an older release is not silently rewritten.
        expected_ids = {path.stem for path, _ in pairs}
        for sample_dir in output.iterdir():
            if not sample_dir.is_dir() or sample_dir.name not in expected_ids:
                raise ValueError("Output layout differs from this run. Choose a new --note.")
            names = {p.name for p in sample_dir.iterdir()}
            if not names <= {"original.png", "adversarial.png", "metadata.json", "metadata.json.tmp"}:
                raise ValueError(f"Unexpected output files in {sample_dir}; choose a new --note.")
            done = sample_dir / "metadata.json"
            temporary = sample_dir / "metadata.json.tmp"
            if not done.is_file():
                if not names:
                    # Interrupted after mkdir, before the initial metadata write.
                    continue
                if not temporary.is_file():
                    raise ValueError(f"Incomplete sample {sample_dir.name}; choose a new --note.")
                try:
                    pending = json.loads(temporary.read_text())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if names == {temporary.name}:
                        # No image outputs exist yet; discard a partial initial write.
                        temporary.unlink()
                        continue
                    raise ValueError(
                        f"Cannot recover metadata in {sample_dir}; choose a new --note."
                    ) from None
                if not isinstance(pending, dict) or pending.get("run_id") != run_id:
                    raise ValueError("Existing run has different inputs/settings. Choose a new --note.")
                temporary.replace(done)
            metadata = json.loads(done.read_text())
            if metadata.get("run_id") != run_id:
                raise ValueError("Existing run has different inputs/settings. Choose a new --note.")
            if temporary.is_file():
                # The canonical file is authoritative until atomic replacement.
                temporary.unlink()
            if metadata.get("completed", False):
                files = metadata.get("artifacts_sha256", {})
                if set(files) != {"original.png", "adversarial.png"} or not set(files) <= names:
                    raise RuntimeError(f"Completed sample has missing files: {sample_dir}")
                if any(sha256(sample_dir / name) != digest for name, digest in files.items()):
                    raise RuntimeError(f"Completed sample has changed files: {sample_dir}")
                completed_ids.add(sample_dir.name)
        if completed_ids == expected_ids:
            logger.info(f"All {len(pairs)} samples are complete and verified. Outputs: {output}")
            return
        logger.warning(
            "Resume preserves completed files, but skipped samples change the original sequential RNG stream."
        )
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(args.gpu),
        "packages": {p: version(p) for p in ["torch", "torchvision", "transformers", "qwen-vl-utils"]},
    }
    set_seed(args.seed)
    # Preserve the original OurAttack._initialize_models() reset before model loading.
    set_environment(args.optimization_seed)
    extractor, objective = load_surrogates(device)
    for index, (path, target) in enumerate(pairs, 1):
        sample_dir = output / path.stem
        done = sample_dir / "metadata.json"
        if args.resume and done.exists() and json.loads(done.read_text()).get("completed", False):
            logger.info(f"Skipping verified sample {path.stem}")
            continue
        sample_dir.mkdir(exist_ok=True)
        write_json(
            done,
            {
                "run_id": run_id,
                "completed": False,
                "configuration": configuration,
                "source": str(path),
                "source_sha256": sha256(path),
                "target": target,
            },
        )
        logger.info(f"[{index}/{len(pairs)}] {path.name}")
        with Image.open(path) as source:
            image = source.convert("RGB")
        attention, caption = extract_attention(image, device, args.layer_idx, args.question_prompt)
        schedule = build_schedule(
            attention, args.progressive_num_phases, args.hotspot_top_k, args.hotspot_iou_threshold
        )
        tensor = preprocess_image(image)
        clean, adversarial, _ = optimize(
            tensor,
            target,
            schedule,
            extractor,
            objective,
            device=device,
            steps=args.attack_steps,
            alpha=args.attack_alpha,
            epsilon=args.attack_epsilon,
            scale=(args.hotspot_scale_min, args.hotspot_scale_max),
        )
        clean_pil = to_pil_image(clean.squeeze(0).cpu())
        adv_pil = to_pil_image(adversarial.squeeze(0).cpu())
        saved_linf = int(
            np.abs(np.asarray(adv_pil, dtype=np.int16) - np.asarray(clean_pil, dtype=np.int16)).max()
        )
        if saved_linf > np.ceil(args.attack_epsilon):
            raise AssertionError(f"Saved PNG L-infinity exceeds quantized budget: {saved_linf}")
        clean_pil.save(sample_dir / "original.png")
        adv_pil.save(sample_dir / "adversarial.png")
        write_json(
            done,
            {
                "run_id": run_id,
                "completed": True,
                "configuration": configuration,
                "environment": environment,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(path),
                "source_sha256": sha256(path),
                "target": target,
                "clean_caption": caption,
                "saved_png_linf_0_255": saved_linf,
                "schedule_items": len(schedule),
                "steps_per_item": args.attack_steps // len(schedule),
                "last_item_steps": args.attack_steps
                - (len(schedule) - 1) * (args.attack_steps // len(schedule)),
                "artifacts_sha256": {
                    name: sha256(sample_dir / name) for name in ["original.png", "adversarial.png"]
                },
            },
        )
    logger.info(f"Completed {len(pairs)} samples. Outputs: {output}")
