"""Strict one-to-one source-image / target-caption loading."""

import hashlib
import json
import os
import random
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pairs(image_dir, jsonl_path, text_path=None, num_images=1000, seed=43):
    image_dir, jsonl_path = Path(image_dir), Path(jsonl_path)
    captions = {}
    for number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        entry = json.loads(line)
        stem = Path(entry["file_name"]).stem
        if stem in captions:
            raise ValueError(f"Duplicate target image ID {stem} at line {number}")
        if not isinstance(entry.get("caption"), str) or not entry["caption"].strip():
            raise ValueError(f"Empty caption at line {number}")
        captions[stem] = entry["caption"]
    if text_path:
        texts = [s.strip() for s in Path(text_path).read_text(encoding="utf-8").splitlines() if s.strip()]
        if texts != list(captions.values()):
            raise ValueError(
                "Target TXT and JSONL captions differ; use matching files or omit --target_text_file"
            )
    # Match DataLoader._get_image_files: preserve filesystem enumeration before sampling.
    paths = [
        image_dir / name for name in os.listdir(image_dir) if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Source image stems must be unique")
    if num_images < 1 or num_images > len(paths):
        raise ValueError(f"Requested {num_images} images, found {len(paths)} in {image_dir}")
    # Original one-to-one pairing sorts by stem only after random selection.
    selected = random.Random(seed).sample(paths, num_images)
    selected.sort(key=lambda p: p.stem)
    missing = [p.name for p in selected if p.stem not in captions]
    if missing:
        raise ValueError(f"Missing JSONL captions for source IDs: {missing[:10]}")
    return [(p, captions[p.stem]) for p in selected]
