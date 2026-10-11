"""SAGA-only CLI; accepts every option from the supplied main command."""

import argparse
from pathlib import Path

from .data import load_pairs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SAGA main method (Qwen3-VL + three CLIP surrogates)")
    parser.add_argument("--dataset", default="aadcd")
    parser.add_argument("--image_dir", type=Path, help="Default: input/images/<dataset>")
    parser.add_argument("--target_text_file", type=Path, default=None)
    parser.add_argument("--target_text_jsonl", type=Path, default=Path("input/coco.jsonl"))
    parser.add_argument("--one_to_one", action="store_true", default=True, help="Always enabled for SAGA")
    parser.add_argument("--seed", type=int, default=43, help="Dataset sampling seed")
    parser.add_argument(
        "--optimization_seed",
        type=int,
        default=2023,
        help="Original model-init RNG reset (2023); recorded in metadata.json",
    )
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--note", default="saga")
    parser.add_argument("--gpu", type=int, default=0, help="Logical index within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--save_path", type=Path, default=Path("results"))
    parser.add_argument(
        "--resume", action="store_true", help="Skip verified completed samples; RNG stream may differ"
    )
    parser.add_argument(
        "--check_inputs", action="store_true", help="Validate images and targets without loading models"
    )
    parser.add_argument("--debug", action="store_true", help="Verbose local logs (no remote logging)")
    # Retain original command spelling, with only the released method permitted.
    for flag, value in [
        ("method", "ours"),
        ("hotspot_attack_type", "progressive_hotspot_attack"),
        ("hotspot_metric_type", "open_vlm_attention"),
        ("vlm_attention_extractor", "qwen3_vl"),
        ("attention_extraction_mode", "generation"),
        ("attention_norm_mode", "minmax"),
        ("hotspot_ordering_mode", "saga"),
    ]:
        parser.add_argument("--" + flag, choices=[value], default=value)
    parser.add_argument("--layer_idx", type=int, default=29)
    parser.add_argument("--use_content_words", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_smart_resize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progressive_mode", action="store_true", default=True)
    parser.add_argument("--progressive_num_phases", type=int, default=10)
    parser.add_argument("--hotspot_top_k", type=int, default=3)
    parser.add_argument("--hotspot_iou_threshold", type=float, default=0.3)
    parser.add_argument("--hotspot_scale_min", type=float, default=0.5)
    parser.add_argument("--hotspot_scale_max", type=float, default=0.9)
    parser.add_argument(
        "--hotspot_prob",
        type=float,
        default=0.5,
        help="Compatibility argument: original SAGA always crops within hotspot (effective 1.0)",
    )
    parser.add_argument("--attack_steps", type=int, default=300)
    parser.add_argument("--attack_alpha", type=float, default=1.0)
    parser.add_argument(
        "--attack_epsilon", type=float, default=16.0, help="Pixel units, divided by 255 in [0,1] space"
    )
    parser.add_argument("--question_prompt", default="Describe this image, no longer than 25 words.")
    args = parser.parse_args(argv)
    if args.use_smart_resize or not args.use_content_words:
        parser.error("Main release requires --no-use_smart_resize and --use_content_words")
    if not args.note or Path(args.note).name != args.note or args.note in {".", ".."}:
        parser.error("--note must be a single nonempty directory name")
    if args.gpu < 0 or args.layer_idx < 0 or args.progressive_num_phases < 1 or args.hotspot_top_k < 1:
        parser.error("Invalid device, layer, or phase configuration")
    if args.attack_steps < args.progressive_num_phases * args.hotspot_top_k:
        parser.error("--attack_steps must be at least phases * top_k (30 by default)")
    if not (0 < args.hotspot_scale_min <= args.hotspot_scale_max <= 1):
        parser.error("Crop scales must satisfy 0 < min <= max <= 1")
    if not (0 <= args.hotspot_iou_threshold <= 1 and 0 <= args.hotspot_prob <= 1):
        parser.error("IoU threshold and hotspot probability must be in [0,1]")
    if not (0 <= args.attack_epsilon <= 255 and 0 < args.attack_alpha <= 255):
        parser.error("Epsilon must be in [0,255] and alpha in (0,255]")
    args.image_dir = args.image_dir or Path("input/images") / args.dataset
    return args


def main():
    args = parse_args()
    pairs = load_pairs(
        args.image_dir, args.target_text_jsonl, args.target_text_file, args.num_images, args.seed
    )
    from PIL import Image

    for path, _ in pairs:
        with Image.open(path) as image:
            image.verify()
    if args.check_inputs:
        print(f"Validated {len(pairs)} source images and one-to-one captions. First ID: {pairs[0][0].stem}")
        return
    from .pipeline import run

    run(args, pairs)
