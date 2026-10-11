"""Original Qwen3-VL generation-time extraction (content-token filtering)."""

import torch
from loguru import logger
from torch import Tensor

from .attention_utils import build_attention_matrix, get_token_img_shape, get_token_indices
from .text import filter_generated_tokens


def get_attention_map_from_generated_text_vlm(
    model,
    processor,
    inputs,
    vlm_type: str,
    use_vis_attention: bool = False,
    temperature: float = 1.0,
    do_sample: bool = False,
    layer_idx: int | None = None,
    use_content_words: bool = False,
    norm_mode: str = "minmax",
    **kwargs,
) -> tuple[Tensor, str]:
    """Extract generation-time Qwen3-VL attention using the original causal row alignment."""

    if vlm_type != "qwen3_vl":
        raise ValueError("This release supports Qwen3-VL only.")

    tokenizer = processor.tokenizer

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        return_dict_in_generate=True,
        output_attentions=True,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
    )

    decoded_text = processor.batch_decode(
        outputs["sequences"], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]

    token_indices = get_token_indices(decoded_text, tokenizer, outputs, inputs, vlm_type=vlm_type)
    generated_token_ids = outputs.sequences[0][token_indices["output_start"] : token_indices["output_end"]]
    valid_token_indices = filter_generated_tokens(tokenizer, generated_token_ids, use_content_words)

    # Build attention matrix and get token indices
    # Keep every generation step: compacting filtered rows breaks causal row alignment.
    llm_attn_matrix = build_attention_matrix(outputs=outputs, layer_idx=layer_idx)

    vision_token_start = token_indices["vision_start"]
    vision_token_end = token_indices["vision_end"]
    num_vision_tokens = vision_token_end - vision_token_start

    grid_h, grid_w = get_token_img_shape(inputs=inputs, processor=processor, vlm_type=vlm_type)
    assert grid_h * grid_w == num_vision_tokens, (
        f"Mismatch in vision tokens ({num_vision_tokens}) and grid size: {num_vision_tokens} vs {grid_h}x{grid_w}"
    )

    # Average attention maps over all valid generated tokens
    attn_maps = []
    for token_offset in valid_token_indices:
        token_row = token_indices["output_start"] + token_offset
        if token_row >= llm_attn_matrix.shape[0]:
            logger.warning(
                f"Token row {token_row} out of range for attention matrix with shape {llm_attn_matrix.shape}"
            )
            continue

        attn_weights_over_vis_tokens = llm_attn_matrix[token_row][vision_token_start:vision_token_end]
        if attn_weights_over_vis_tokens.sum() == 0:
            continue
        attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()
        attn_maps.append(attn_weights_over_vis_tokens.reshape(grid_h, grid_w))

    if attn_maps:
        attn_over_image = torch.stack(attn_maps).mean(dim=0)
    else:
        attn_over_image = torch.zeros((grid_h, grid_w), dtype=llm_attn_matrix.dtype)

    mn = attn_over_image.min()
    mx = attn_over_image.max()
    attn_over_image = (attn_over_image - mn) / (mx - mn + 1e-8)
    logger.debug(f"Attention map shape: {attn_over_image.shape}")

    generated_text = tokenizer.decode(
        outputs.sequences[0][token_indices["output_start"] : token_indices["output_end"]],
        skip_special_tokens=True,
    )

    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return attn_over_image, generated_text
