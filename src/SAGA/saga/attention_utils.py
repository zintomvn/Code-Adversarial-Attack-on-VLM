"""Generation-attention aggregation retained from the research implementation."""

import torch
from torch import Tensor


def get_token_indices(decoded_text: str, tokenizer, outputs: dict, inputs: dict, vlm_type="qwen3_vl") -> dict:
    """Extract indices for vision tokens, prompt tokens, and output tokens."""
    if vlm_type == "qwen3_vl":
        marker_start = "<|vision_start|>"
        marker_end = "<|vision_end|>"

    len_prompt_tokens = len(inputs["input_ids"][0])
    len_all_tokens = len(outputs["sequences"][0])

    # +1 to include <|vision_start|> token
    vision_token_start = (
        len(tokenizer(decoded_text.split(marker_start)[0], return_tensors="pt")["input_ids"][0]) + 1
    )
    vision_token_end = len(tokenizer(decoded_text.split(marker_end)[0], return_tensors="pt")["input_ids"][0])

    return {
        "vision_start": vision_token_start,
        "vision_end": vision_token_end,
        "output_start": len_prompt_tokens,
        "output_end": len_all_tokens,
        "output_len": len_all_tokens - len_prompt_tokens,
    }


def aggregate_llm_attention(attn: Tensor, layer_idx: int | None = None) -> Tensor:
    """
    Extract average attention vector from attention outputs.

    Nullifies attention over the first token (<bos>), which is a common practice
    for null attention (https://aclanthology.org/W19-4808.pdf).

    Args:
        attn: Attention tensor from model outputs
        layer_idx: If provided, extract from this layer only. Otherwise average across all layers.
    """
    if layer_idx is not None and layer_idx != -100:
        layer = attn[layer_idx]
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat(
            [
                torch.tensor([0.0]),  # Zero out null attention on <bos>
                attns_per_head[-1][1:].cpu(),
                torch.tensor([0.0]),  # Zero for final generated token
            ]
        )
        return vec / vec.sum()

    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat(
            [
                torch.tensor([0.0]),  # Zero out null attention on <bos>
                attns_per_head[-1][1:].cpu(),
                torch.tensor([0.0]),  # Zero for final generated token
            ]
        )
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)


def heterogenous_stack(vecs: list[torch.Tensor]) -> torch.Tensor:
    """Pad vectors with zeros to same length then stack."""
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([torch.concat((v, torch.zeros(max_length - v.shape[0]))) for v in vecs])


def aggregate_prompt_attention(
    first_generated_token_attn: Tensor,
    layer_idx: int | None = None,
) -> Tensor:
    """
    Aggregate attention for the prompt/first generated token.

    Args:
        first_generated_token_attn: Attention tensor from first generation step
        layer_idx: If provided, extract from this layer only. Otherwise average across all layers.
    """
    if layer_idx is not None and layer_idx != -100:
        layer_attn = first_generated_token_attn[layer_idx]
        layer_attns = layer_attn.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        # last row corresponds to the generated token, exclude it
        cur = attns_per_head[:-1].cpu().clone()

        # Zero out attention to the first <bos> token (null attention)
        # Skip first row since <bos> is the only token it can attend to
        cur[1:, 0] = 0.0
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)

        return cur

    aggregated = []
    for layer_attn in first_generated_token_attn:
        layer_attns = layer_attn.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        # last row corresponds to the generated token, exclude it
        cur = attns_per_head[:-1].cpu().clone()

        # Zero out attention to the first <bos> token (null attention)
        # Skip first row since <bos> is the only token it can attend to
        cur[1:, 0] = 0.0
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)

        aggregated.append(cur)

    # Average across all layers
    return torch.stack(aggregated).mean(dim=0)


def build_attention_matrix(outputs, layer_idx: int | None = None, valid_token_indices=None) -> torch.Tensor:
    """
    Build a complete attention matrix from generation outputs.

    Args:
        outputs: Model generation outputs with attentions
        layer_idx: If provided, extract from this layer only. Otherwise average across all layers.
        valid_token_indices: Optional set of valid token indices to include in the attention matrix.

    Returns:
        Tensor of shape (total_tokens, max_seq_len) representing attention
        across all tokens (prompt + generated)
        entry (i, j): how much token i attends to token j
    """
    attention_vectors = []

    # Add <bos> token (attends only to itself)
    attention_vectors.append(torch.tensor([1.0]))

    # Add prompt token attentions
    prompt_attention = aggregate_prompt_attention(outputs.attentions[0], layer_idx=layer_idx)
    attention_vectors.extend(list(prompt_attention))

    # Add generated token attentions
    for step_idx, step_attention in enumerate(outputs.attentions):
        if valid_token_indices is not None and step_idx not in valid_token_indices:
            continue
        gen_attention = aggregate_llm_attention(step_attention, layer_idx=layer_idx)
        attention_vectors.append(gen_attention)

    # Pad all vectors to same length and stack
    return heterogenous_stack(attention_vectors)


def get_token_img_shape(inputs, processor, vlm_type="qwen3_vl"):
    if vlm_type == "qwen3_vl":
        """
        Qwen3-VL: Calculate the grid dimensions (height, width) of image tokens after spatial merging.
        NOTE: Processor outputs containing 'image_grid_thw'.
        """
        # Get the raw grid from the processor inputs
        grid_h = inputs["image_grid_thw"][0][1].item()
        grid_w = inputs["image_grid_thw"][0][2].item()

        # 2x2 spatial merging/pooling
        merge_factor = processor.image_processor.merge_size
        grid_h = grid_h // merge_factor
        grid_w = grid_w // merge_factor
        return grid_h, grid_w
