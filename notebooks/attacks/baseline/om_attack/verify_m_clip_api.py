"""Reproduce the M-Attack-V2 CLIP API contract without downloading any weights."""

import json

import torch
import transformers
from transformers import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer


def verify():
    if not torch.cuda.is_available():
        raise RuntimeError("Run with the M-Attack-V2 GPU environment")
    torch.manual_seed(2023)
    config = CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, image_size=32, patch_size=16,
    )
    model = CLIPVisionTransformer(config).eval().cuda().requires_grad_(False)
    image = torch.rand(2, 3, 32, 32, device="cuda", requires_grad=True)
    with torch.amp.autocast(device_type="cuda"):
        output = model(
            pixel_values=image, output_attentions=False, output_hidden_states=True,
            interpolate_pos_encoding=False, return_dict=True,
        )
        loss = output.pooler_output[0, 0]
    loss.backward()
    assert output.pooler_output.shape == (2, 32)
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0
    return {
        "transformers": transformers.__version__, "torch": torch.__version__,
        "author_forward_arguments_accepted": True, "amp_input_gradient_finite": True,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
