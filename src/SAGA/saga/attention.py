"""Load Qwen3-VL, extract one clean-image map, then release the extractor."""

import base64
import gc
from io import BytesIO

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from .attention_map import get_attention_map_from_generated_text_vlm

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


@torch.no_grad()
def extract_attention(image, device, layer_idx=29, prompt="Describe this image, no longer than 25 words."):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    image_processor = processor.image_processor
    if getattr(image_processor, "min_pixels", None) is None:
        image_processor.min_pixels = 200704
    if getattr(image_processor, "max_pixels", None) is None:
        image_processor.max_pixels = 1003520
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="auto"
    ).eval()
    try:
        # Preserve the supplied --no-use_smart_resize preprocessing path.
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        images, videos, _ = process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
        )
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=images, videos=videos, do_resize=False, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        attention, caption = get_attention_map_from_generated_text_vlm(
            model,
            processor,
            inputs,
            vlm_type="qwen3_vl",
            layer_idx=layer_idx,
            use_content_words=True,
            do_sample=False,
            norm_mode="minmax",
        )
        # Original extractor normalizes once more after averaging its single map.
        attention = attention.detach().float().cpu().numpy()
        mn, mx = attention.min(), attention.max()
        attention = (attention - mn) / (mx - mn) if mx > mn else np.ones_like(attention) * 0.5
        if not np.isfinite(attention).all():
            raise ValueError("Qwen3-VL returned a non-finite attention map")
        return attention, caption
    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
