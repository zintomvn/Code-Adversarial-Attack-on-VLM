"""Migrated directly from CARA v6.4.1; see SOURCE_MAP.json and ATTRIBUTION.md."""
from .config import INSTALL_PACKAGES, RUN_ATTACK, RUN_EVALUATION, RUN_VISUALIZATION, NUM_SAMPLES, START_INDEX, SEED, DATA_ROOT, SOURCE_DIR, TARGET_DIR, PAIR_MANIFEST, OUTPUT_DIR, K, STEPS, XAI_METHOD, EPSILON, ALPHA, MOMENTUM, TEMPERATURE, OMEGA, CROP_AREA, ALPHA_GLOBAL, ALPHA_CROP, SAVE_BRANCH_IMAGES, SAVE_EVERY, SAVE_CROP_EVERY, ATTENTION_LAYER, NEGATIVE_RETRIES, CAPTION_ATTEMPTS, IMAGE_SIZE, CLIP_G14_ID, CLIP_B16_ID, CLIP_B32_ID, QWEN_ID, EVALUATION_ID, CLIP_DEVICES, ATTACK_DEVICE, QWEN_MAX_MEMORY_GIB, IMAGE_WEIGHT, TEXT_WEIGHT, CAPTION_TEMPERATURE, CAPTION_TOP_P, EVALUATION_PROMPT, RUN_ID, MODEL_REVISIONS
from .common import *
import os
import string  # required by the original SAGA clean_token helper
from loguru import logger

DEBUG = os.environ.get("DEBUG", "0") == "1"

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

STOP_WORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "she's",
    "her",
    "hers",
    "herself",
    "it",
    "it's",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "that'll",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "don't",
    "should",
    "should've",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "aren't",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "isn",
    "isn't",
    "ma",
    "mightn",
    "mightn't",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "shan",
    "shan't",
    "shouldn",
    "shouldn't",
    "wasn",
    "wasn't",
    "weren",
    "weren't",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
}

def clean_token(token: str) -> str:
    """Lowercase, strip, and drop punctuation to normalize tokens."""
    return token.strip().lower().translate(str.maketrans("", "", string.punctuation))

def is_content_word(token: str) -> bool:
    """Return True if token is non-empty after cleaning and not a stop word."""
    cleaned = clean_token(token)
    return bool(cleaned) and cleaned not in STOP_WORDS

def filter_generated_tokens(tokenizer, token_ids: Sequence[int], use_content_words: bool) -> list[int]:
    """
    Return indices of tokens to use. If use_content_words is True, keep only content
    words; otherwise return all indices. Falls back to all tokens if filtering empties.
    """
    if not use_content_words:
        return list(range(len(token_ids)))

    logger.debug(f"[Use content words: {use_content_words}] Only keeping content words")
    valid_indices = []
    for idx, token_id in enumerate(token_ids):
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        if is_content_word(token_str):
            valid_indices.append(idx)
        elif DEBUG:
            logger.debug(f"Filtering out token '{token_str}' at index {idx}")

    if not valid_indices:
        return list(range(len(token_ids)))

    return valid_indices


class AttentionConceptMatchError(ValueError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


class QwenAttentionXAI:
    def __init__(self, captioner, layer=29):
        self.captioner = captioner
        self.layer = layer

    @staticmethod
    def concept_offsets(tokenizer, generated_ids, concepts):
        ids = [int(x) for x in generated_ids]
        prefixes = [tokenizer.decode(ids[:i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in range(len(ids) + 1)]
        if any(not prefixes[-1].startswith(prefix) for prefix in prefixes):
            raise AttentionConceptMatchError('Non-monotonic token decoding', {'decoded_prefixes':prefixes})
        spans = [(len(prefixes[i]), len(prefixes[i + 1])) for i in range(len(ids))]
        generated = prefixes[-1]
        result = {}
        for concept in concepts:
            words = [word for word in re.findall(r"\w+", concept.lower()) if is_content_word(word)]
            matches = [m.span() for m in re.finditer(r"\w+", generated.lower())
                       if m.group() in words]
            offsets = [i for i, (start, end) in enumerate(spans)
                       if any(start < b and end > a for a, b in matches)]
            if offsets:
                result[concept] = offsets
        return result

    @torch.no_grad()
    def maps(self, image, concepts):
        captioner = self.captioner
        inputs = captioner.attention_inputs(image)
        outputs = captioner.model.generate(
            **inputs,
            max_new_tokens=50,
            return_dict_in_generate=True,
            output_attentions=True,
            do_sample=False,
        )
        decoded = captioner.processor.batch_decode(outputs.sequences,
                      skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        indices = get_token_indices(decoded, captioner.processor.tokenizer, outputs, inputs, "qwen3_vl")
        generated = outputs.sequences[0][indices["output_start"]:indices["output_end"]]
        offsets = self.concept_offsets(captioner.processor.tokenizer, generated, concepts)
        content = set(filter_generated_tokens(captioner.processor.tokenizer, generated, True))
        attention = build_attention_matrix(outputs, layer_idx=self.layer)
        height, width = get_token_img_shape(inputs, captioner.processor, "qwen3_vl")
        start, end = indices["vision_start"], indices["vision_end"]
        if end - start != height * width:
            raise ValueError("Vision-token count does not match Qwen image grid")

        concept_maps = {}
        matched_offsets = {}
        matched_rows = {}
        for concept, positions in offsets.items():
            rows = []
            used_positions = []
            for position in positions:
                row_index = indices["output_start"] + position
                if position not in content or row_index >= attention.shape[0]:
                    continue
                weights = attention[row_index, start:end].float()
                if torch.isfinite(weights).all() and weights.sum() > 0:
                    rows.append((weights / weights.sum()).reshape(height, width))
                    used_positions.append(position)
            if rows:
                concept_maps[concept] = torch.stack(rows).mean(0)
                matched_offsets[concept] = used_positions
                matched_rows[concept] = [indices['output_start']+position for position in used_positions]

        caption = captioner.processor.tokenizer.decode(generated, skip_special_tokens=True)
        metadata = {'generated_caption':caption,'layer':self.layer,
                    'concepts':list(concepts),'matched_concepts':list(concept_maps),
                    'unmatched_concepts':[c for c in concepts if c not in concept_maps],
                    'concept_token_offsets':matched_offsets,'concept_attention_rows':matched_rows,
                    'generated_token_ids':[int(x) for x in generated],
                    'generated_tokens':[captioner.processor.tokenizer.decode([int(x)],skip_special_tokens=False,
                        clean_up_tokenization_spaces=False) for x in generated],
                    'content_token_offsets':sorted(content),
                    'vision_grid':[height,width],'vision_span':[start,end],
                    'output_start':indices['output_start'],
                    'matching':'whole content words; no synonyms or stemming'}
        del outputs,attention
        if not concept_maps:
            raise AttentionConceptMatchError("No generated Qwen tokens matched source concepts",metadata)
        return concept_maps, metadata
