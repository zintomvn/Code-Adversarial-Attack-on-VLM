"""Migrated directly from CARA v6.4.1; see SOURCE_MAP.json and ATTRIBUTION.md."""
from .config import INSTALL_PACKAGES, RUN_ATTACK, RUN_EVALUATION, RUN_VISUALIZATION, NUM_SAMPLES, START_INDEX, SEED, DATA_ROOT, SOURCE_DIR, TARGET_DIR, PAIR_MANIFEST, OUTPUT_DIR, K, STEPS, XAI_METHOD, EPSILON, ALPHA, MOMENTUM, TEMPERATURE, OMEGA, CROP_AREA, ALPHA_GLOBAL, ALPHA_CROP, SAVE_BRANCH_IMAGES, SAVE_EVERY, SAVE_CROP_EVERY, ATTENTION_LAYER, NEGATIVE_RETRIES, CAPTION_ATTEMPTS, IMAGE_SIZE, CLIP_G14_ID, CLIP_B16_ID, CLIP_B32_ID, QWEN_ID, EVALUATION_ID, CLIP_DEVICES, ATTACK_DEVICE, QWEN_MAX_MEMORY_GIB, IMAGE_WEIGHT, TEXT_WEIGHT, CAPTION_TEMPERATURE, CAPTION_TOP_P, EVALUATION_PROMPT, RUN_ID, MODEL_REVISIONS
from .common import *
SYSTEM_MSG_OBJECTS = (
    "You are an information extraction assistant.\n"
    "From the caption, select up to FIVE main objects. A main object must have at least one descriptive attribute in the caption "
    "(e.g., color, size, material, possession/with-phrase, appositive, relative clause, or an explicit number).\n"
    "Rules:\n"
    "• Output only object names that appear in the caption and are part of the described scene. Never invent or infer.\n"
    "• Prefer plain object names in the output (omit adjectives), but KEEP explicit numbers/quantifiers if the caption states them "
    "(e.g., 'two cats', 'five chickens', 'a pair of skis').\n"
    "• If multiple mentions share the same name and no explicit number is given, output the plural form (e.g., 'dogs'). \n"
    "• List 1–5 main objects; if only one is present, output just that one.\n"
    "Do not add quantifiers like 'some' unless present in the caption.\n"
    "Format:\n"
    "Return EXACTLY one line:\n"
    "PHRASE=<comma-separated list with 'and' before the last item, e.g., 'a dog, a cat and two birds'>\n"
    "No trailing period. No extra text."
)

FEW_SHOT_OBJECTS = [
    (
        "A bustling street scene with three blue buses at the curb, two red cars in traffic, "
        "a tall glass office building behind them, and a bronze statue with a raised arm in the plaza.",
        "PHRASE=three buses, two cars, an office building and a statue"
    ),
    (
        "On a wooden table, a cast-iron skillet sits beside a blue ceramic vase of sunflowers, "
        "with red apples and green pears piled high in a stainless steel bowl.",
        "PHRASE=a skillet, a vase, apples, pears and a bowl"
    ),
]

SYSTEM_MSG_NEGOBJ = (
    "You are a negative object creator.\n"
    "You will receive a caption, an object list as PHRASE=..., and REPLACE_INDEX=k (1-based).\n"
    "Replace EXACTLY the k-th object in PHRASE with a distinctly different NEGATIVE object.\n"
    "\n"
    "Keep ALL other objects unchanged and preserve their order and punctuation. Keep the same quantifier/determiner "
    "for the replaced slot (e.g., 'two cats' -> 'two bicycles').\n"
    "\n"
    "Constraints for the NEGATIVE object:\n"
    "• It must be distinctly different from the replaced object (not a synonym; not just singular/plural).\n"
    "• It must NOT be a synonym or near-equivalent of ANY object that appears in the caption.\n"
    "• It must NOT appear anywhere in the caption (as a whole word, singular or plural).\n"
    "• Do not modify any other items; do not reorder items; do not add or remove items.\n"
    "\n"
    "Edge cases (must follow):\n"
    "• If REPLACE_INDEX is greater than the number of objects in PHRASE, replace the LAST object.\n"
    "• If REPLACE_INDEX is less than 1, replace the FIRST object.\n"
    "\n"
    "Self-check (must hold):\n"
    "• Same number of items as input; exactly one item (the k-th per the rule above) differs.\n"
    "\n"
    "Output EXACTLY one line:\n"
    "PHRASE=<the new list, same format>\n"
    "No extra text. No quotes. No trailing period."
)

# Few-shots cover different list lengths and positions; each uses REPLACE_INDEX to avoid a last-item bias.
FEW_SHOT_NEGOBJ = [
    (
        # Replace FIRST of three
        "Caption:\nA small yellow school bus is parked next to two red cars in front of a brick office building.\n"
        "PHRASE=a bus, two cars and an office building\n"
        "REPLACE_INDEX=1",
        "PHRASE=a boat, two cars and an office building"
    ),
    (
        "Caption:\nTwo hikers stand on a wooden bridge with their dog beside them over a calm river.\n"
        "PHRASE=two hikers, a dog, a bridge and a river\n"
        "REPLACE_INDEX=2",
        "PHRASE=two hikers, a bicycle, a bridge and a river"
    ),
    (
        "Caption:\nOn a side table there is a bowl of fruit, a ceramic vase, and a brass lamp.\n"
        "PHRASE=a bowl of fruit, a vase and a lamp\n"
        "REPLACE_INDEX=3",
        "PHRASE=a bowl of fruit, a vase and a clock"
    ),
    (
        "Caption:\nA dashboard card labeled 'Latency' shows a graph with a menu icon and a legend for several lines.\n"
        "PHRASE=a card, a graph, a menu icon, a legend and lines\n"
        "REPLACE_INDEX=3",
        "PHRASE=a card, a graph, a slider, a legend and lines"
    ),
    (
        "Caption:\nA booking screen shows a highlighted 12:05–12:30 time slot, a search box, an entry for Animal Kingdom Villas, a restaurant icon, and a market link.\n"
        "PHRASE=a time slot, a box, an entry, a restaurant and a market\n"
        "REPLACE_INDEX=4",
        "PHRASE=a time slot, a box, an entry, a ticket counter and a market"
    ),
]

CATEGORIES = ["natural_image", "chart_graph", "screenshot_ui", "document_text"]

SYSTEM_MSG_CATEGORY = (
    "You are a strict classifier. Based ONLY on the caption text, assign EXACTLY ONE label.\n"
    "Allowed labels:\n"
    "  - natural_image\n"
    "  - chart_graph   (includes diagrams, flowcharts, schematics, illustrations)\n"
    "  - screenshot_ui (UI/app/website/IDE/terminal screenshots)\n"
    "  - document_text (scans, printed pages, receipts, invoices, whiteboards, handwriting)\n"
    "\n"
    "Rules:\n"
    "• Return ONLY the label string, nothing else.\n"
    "• If multiple seem plausible, pick the single best one.\n"
    "• If unsure, choose natural_image.\n"
)

FEW_SHOT_CATEGORY = [
    ("A line chart comparing quarterly revenue for three products with a legend and axes labels.",
     "chart_graph"),
    ("A screenshot of a mobile banking app showing the account balance and a transfer button.",
     "screenshot_ui"),
    ("A scanned receipt from a grocery store listing items and totals.",
     "document_text"),
    ("A photo of a dog running on a beach at sunset.",
     "natural_image"),
    ("A flowchart illustrating an authentication pipeline with decision nodes and arrows.",
     "chart_graph"),
]


import re
from typing import List

def truncate_first_tokens(tokenizer, text: str, max_tokens: int = 512) -> str:
    """
    Keep only the first `max_tokens` tokenizer tokens.
    """
    ids = tokenizer(text or "", add_special_tokens=False).input_ids
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()

def parse_tagged_output(
    text: str,
    *,
    allowed_tags: Sequence[str],
    max_chars: int = 512,
    split_on_semicolon: bool = False,
    strip_trailing_pattern: str = r"[.;:\s]+$",
) -> str:
    """
    Parse a generated single-line output like:
      PHRASE=...
      SENTENCE=...
      ACC_PHRASE=...
    by scanning for the first matching tagged line.

    Fallback:
      - if no tagged line is found, use the first non-empty line.

    Normalization:
      - strip outer quotes
      - optional split on first ';'
      - strip trailing punctuation via regex
      - collapse internal whitespace
      - clip to max_chars
    """
    raw = (text or "").strip()
    lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and line.strip() not in {"```", "~~~"}
    ]

    if not lines:
        return ""

    tag_pattern = "|".join(re.escape(tag) for tag in allowed_tags)
    pattern = re.compile(
        rf"^\s*(?:{tag_pattern})\s*[:=]\s*(.+)$",
        flags=re.IGNORECASE,
    )

    value = ""
    for line in lines:
        match = pattern.match(line)
        if match:
            value = match.group(1).strip()
            break

    if not value:
        value = lines[0].strip()

    value = (
        value.strip()
        .strip('"')
        .strip("'")
        .strip("“”‘’")
        .strip()
    )

    if split_on_semicolon:
        value = value.split(";", 1)[0].strip()

    if strip_trailing_pattern:
        value = re.sub(strip_trailing_pattern, "", value)

    value = re.sub(r"\s+", " ", value).strip()

    if max_chars is not None:
        value = value[:max_chars]

    return value


_truncate_caption_first_tokens = truncate_first_tokens
def _split_objects_phrase(phrase: str) -> List[str]:
    """
    Split 'a dog, a cat and two birds' -> ['a dog','a cat','two birds'].
    Heuristic: split on the last ' and ' and commas on the left.
    Used only for lightweight validation; storage remains a single string.
    """
    s = (phrase or "").strip()
    if not s:
        return []
    if " and " in s:
        left, last = s.rsplit(" and ", 1)
        left_parts = [p.strip() for p in left.split(",") if p.strip()]
        items = left_parts + [last.strip()]
    else:
        items = [p.strip() for p in s.split(",") if p.strip()]
    return items

def _messages_objects(caption_text: str) -> List[dict]:
    """Chat messages for the 'objects' mode with few-shot."""
    msgs = [{"role": "system", "content": SYSTEM_MSG_OBJECTS}]

    for cap, ans in FEW_SHOT_OBJECTS:
        demo_user = (
            "Caption:\n" + cap + "\n\n"
            "Task: From this caption, select up to five main objects that have at least one descriptive attribute in the caption. "
            "Prefer plain object names; keep explicit numbers/quantifiers if present; pluralize repeated names without numbers. "
            "Return EXACTLY one line:\nPHRASE=<...>"
        )
        msgs.append({"role": "user", "content": demo_user})
        msgs.append({"role": "assistant", "content": ans})

    task_user = (
        "Caption:\n" + caption_text + "\n\n"
        "Task: Select up to five main objects that have at least one descriptive attribute in the caption. "
        "Prefer plain object names; keep explicit numbers/quantifiers if present; pluralize repeated names without numbers. "
        "Return EXACTLY one line:\nPHRASE=<...>"
    )
    msgs.append({"role": "user", "content": task_user})
    return msgs

def build_objects_prompt(tokenizer, caption: str) -> str:
    """Final chat prompt using the FIRST 512 caption tokens."""
    cap_trunc = _truncate_caption_first_tokens(tokenizer, caption, 512)
    return tokenizer.apply_chat_template(
        _messages_objects(cap_trunc),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def parse_objects_output(text: str) -> str:
    return parse_tagged_output(
        text,
        allowed_tags=("PHRASE",),
        max_chars=512,
    )


def _validate_single_replacement(original_phrase: str, new_phrase: str) -> bool:
    """
    Light sanity: same number of items and exactly one item differs (string-compare after strip).
    """
    old = _split_objects_phrase(original_phrase)
    new = _split_objects_phrase(new_phrase)
    if not old or not new or len(old) != len(new):
        return False
    diffs = sum(1 for a, b in zip(old, new) if a.strip() != b.strip())
    return diffs == 1


def _messages_negobj(caption_text: str, obj_phrase: str, replace_index_1based: int) -> List[dict]:
    """
    Chat messages for negative replacement with an explicit slot to replace.
    The model must replace exactly REPLACE_INDEX (1-based).
    """
    msgs = [{"role": "system", "content": SYSTEM_MSG_NEGOBJ}]
    for demo_user, demo_assistant in FEW_SHOT_NEGOBJ:
        msgs.append({"role": "user", "content": demo_user})
        msgs.append({"role": "assistant", "content": demo_assistant})
    task_user = f"Caption:\n{caption_text}\nPHRASE={obj_phrase}\nREPLACE_INDEX={replace_index_1based}"
    msgs.append({"role": "user", "content": task_user})
    return msgs


def build_negobj_prompt(tokenizer, caption: str, obj_phrase: str, replace_index_1based: int) -> str:
    """Final chat prompt for negative replacement, with an explicit index (1-based) to replace."""
    cap_trunc = _truncate_caption_first_tokens(tokenizer, caption, 512)
    return tokenizer.apply_chat_template(
        _messages_negobj(cap_trunc, obj_phrase, replace_index_1based),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


class CaptionBank:
    def __init__(self, captioner, k=5):
        self.captioner = captioner
        self.k = k

    def generate_captions(self, image):
        captions = []
        for _ in range(CAPTION_ATTEMPTS):
            caption = self.captioner.caption(image, sample=True).strip()
            if caption.casefold() not in {x.casefold() for x in captions}:
                captions.append(caption)
            if len(captions) == self.k:
                break
        if len(captions) != self.k:
            raise RuntimeError(f"Only {len(captions)} distinct captions; need {self.k}")
        return captions

    def extract_objects(self, caption):
        prompt = build_objects_prompt(self.captioner.processor.tokenizer, caption)
        response = self.captioner.generate(prompt, max_new_tokens=80)
        phrase = parse_objects_output(response)
        objects = _split_objects_phrase(phrase)
        if not (1 <= len(objects) <= 5):
            raise ValueError(f"Invalid FINER object list: {response}")
        return {"caption": caption, "phrase": phrase, "objects": objects}

    @staticmethod
    def reconstruct_caption(caption, old_item, new_item):
        old_words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", old_item)
        new_words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", new_item)
        common = 0
        while common < min(len(old_words), len(new_words)):
            if old_words[common].lower() != new_words[common].lower():
                break
            common += 1
        old_core = old_words[common:]
        new_core = new_words[common:]
        if not old_core or not new_core:
            return None

        for index in range(len(old_core)):
            old = " ".join(old_core[index:])
            new = " ".join(new_core[index:]) if len(new_core) > index else " ".join(new_core)
            pattern = re.compile(r"\b" + re.escape(old).replace(r"\ ", r"\s+") + r"\b", re.I)
            hits = list(pattern.finditer(caption))
            if len(hits) == 1 and new:
                hit = hits[0]
                changed = caption[:hit.start()] + new + caption[hit.end():]
                if changed.casefold() != caption.casefold():
                    return changed
        return None

    def generate_negative(self, object_record):
        caption = object_record["caption"]
        phrase = object_record["phrase"]
        objects = object_record["objects"]
        for _ in range(NEGATIVE_RETRIES):
            index = random.randint(1, len(objects))  # Uniform object slot, as in FINER
            prompt = build_negobj_prompt(self.captioner.processor.tokenizer, caption, phrase, index)
            raw = self.captioner.generate(prompt, max_new_tokens=80)
            new_phrase = parse_objects_output(raw)
            new_objects = _split_objects_phrase(new_phrase)
            valid = (
                _validate_single_replacement(phrase, new_phrase)
                and len(new_objects) == len(objects)
                and all(objects[j] == new_objects[j] for j in range(len(objects)) if j != index - 1)
            )
            if not valid:
                continue
            new_caption = self.reconstruct_caption(caption, objects[index - 1], new_objects[index - 1])
            if new_caption and new_caption.casefold() != caption.casefold():
                return {"caption": new_caption, "phrase": new_phrase, "replaced_position": index}
        raise RuntimeError(f"No validated FINER negative for: {caption}")

    def create(self, source_image, target_image):
        source = self.generate_captions(source_image)
        target = self.generate_captions(target_image)
        object_records = [self.extract_objects(caption) for caption in source]
        target_records = [self.extract_objects(caption) for caption in target]
        negatives = [self.generate_negative(record) for record in object_records]
        negative_captions = [record["caption"] for record in negatives]
        if len(set(x.casefold() for x in negative_captions)) != self.k:
            raise ValueError("Negative captions contain duplicates")

        source_concepts = list(dict.fromkeys(obj for record in object_records for obj in record["objects"]))
        target_concepts = list(dict.fromkeys(obj for record in target_records for obj in record["objects"]))
        return {
            "source_captions": source,
            "target_captions": target,
            "negative_captions": negative_captions,
            "source_concepts": source_concepts,
            "target_concepts": target_concepts,
            "source_objects": object_records,
            "target_objects": target_records,
            "negative_details": negatives,
        }
