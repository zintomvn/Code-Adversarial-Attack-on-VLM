#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
uv run --frozen python -m attack \
    --dataset aadcd \
    --method ours \
    --target_text_file input/coco.txt \
    --target_text_jsonl input/coco.jsonl \
    --one_to_one \
    --seed 43 \
    --num_images 1000 \
    --note saga \
    --gpu 0 \
    --hotspot_attack_type progressive_hotspot_attack \
    --hotspot_metric_type open_vlm_attention \
    --vlm_attention_extractor qwen3_vl \
    --layer_idx 29 \
    --attention_extraction_mode generation \
    --attention_norm_mode minmax \
    --use_content_words \
    --no-use_smart_resize \
    --progressive_mode \
    --progressive_num_phases 10 \
    --hotspot_ordering_mode saga \
    --hotspot_top_k 3 \
    --hotspot_iou_threshold 0.3 \
    --hotspot_scale_min 0.5 \
    --hotspot_scale_max 0.9 \
    --hotspot_prob 0.5 \
    --attack_steps 300 \
    --attack_alpha 1 \
    --attack_epsilon 16 \
    "$@"
