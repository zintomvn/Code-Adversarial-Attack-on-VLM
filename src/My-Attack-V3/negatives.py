"""Source-only counterfactual bank ported from v6.4.1 cells 10, 12, 14.

String checks are conservative sanity checks, not evidence of grounding.
Model loading, decoding configuration and token provenance belong to the caller.
"""
import random
import re
from typing import List, Sequence

CAPTION_ATTEMPTS = 80
NEGATIVE_RETRIES = 10

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
    ("A scanned receipt from a grocery store listing items and totals.", "document_text"),
    ("A photo of a dog running on a beach at sunset.", "natural_image"),
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
    text: str, *, allowed_tags: Sequence[str], max_chars: int = 512, split_on_semicolon: bool = False,
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
    pattern = re.compile(rf"^\s*(?:{tag_pattern})\s*[:=]\s*(.+)$", flags=re.IGNORECASE)

    value = ""
    for line in lines:
        match = pattern.match(line)
        if match:
            value = match.group(1).strip()
            break

    if not value:
        value = lines[0].strip()

    value = (value.strip() .strip('"') .strip("'") .strip("“”‘’") .strip())

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
        _messages_objects(cap_trunc), tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

def parse_objects_output(text: str) -> str:
    return parse_tagged_output(text, allowed_tags=("PHRASE",), max_chars=512)


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
        _messages_negobj(cap_trunc, obj_phrase, replace_index_1based), tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )


# Deliberately small, explicit equivalence list; this is not a semantic oracle.
_EQUIVALENTS = (
    {'dog', 'canine', 'hound'}, {'cat', 'feline'},
    {'car', 'automobile', 'auto'}, {'bicycle', 'bike'},
    {'sofa', 'couch'}, {'airplane', 'aeroplane', 'plane'},
    {'person', 'human'}, {'motorcycle', 'motorbike'},
)
_PREFIX = re.compile(
    r'^(?:(?:a pair of|a couple of|a|an|the|some|any|each|every|this|that|these|those|'
    r'one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several|many|few|'
    r'no|both|all|[0-9]+)\b\s*)+', re.I)


def _singular(word):
    word = word.casefold()
    irregular = {'people': 'person', 'men': 'man', 'women': 'woman',
                 'children': 'child', 'mice': 'mouse', 'geese': 'goose'}
    if word in irregular:
        return irregular[word]
    if word.endswith('ies'):
        return word[:-3] + 'y'
    if word.endswith(('ches', 'shes', 'xes', 'zes', 'sses')):
        return word[:-2]
    if word.endswith('s') and not word.endswith(('ss', 'us')):
        return word[:-1]
    return word


def _core(item):
    return _PREFIX.sub('', item.casefold()).strip()


def _equivalent(a, b):
    a, b = _singular(a), _singular(b)
    return a == b or any(a in group and b in group for group in _EQUIVALENTS)


def _validate_single_replacement(original_phrase, new_phrase,
                                 replace_index_1based=None, caption=''):
    old = _split_objects_phrase(original_phrase)
    new = _split_objects_phrase(new_phrase)
    if not old or len(old) != len(new):
        return False
    differences = [i for i, pair in enumerate(zip(old, new)) if pair[0] != pair[1]]
    if len(differences) != 1:
        return False
    index = differences[0]
    if replace_index_1based is not None:
        if index != max(0, min(len(old) - 1, replace_index_1based - 1)):
            return False
    before = _PREFIX.match(old[index])
    after = _PREFIX.match(new[index])
    if (before.group().strip().casefold() if before else '') != (
            after.group().strip().casefold() if after else ''):
        return False
    old_core, new_core = _core(old[index]), _core(new[index])
    if not old_core or not new_core:
        return False
    if any(_equivalent(new_core, _core(item))
           or _equivalent(new_core.split()[-1], _core(item).split()[-1])
           for item in old if _core(item)):
        return False
    words = re.findall(r'[A-Za-z]+(?:-[A-Za-z]+)*', caption)
    new_words = new_core.split()
    for i in range(len(words) - len(new_words) + 1):
        if all(_equivalent(a, b) for a, b in zip(new_words, words[i:i + len(new_words)])):
            return False
    # Detect obvious number changes even when no leading determiner was parsed.
    if re.findall(r'\d+', old[index]) != re.findall(r'\d+', new[index]):
        return False
    return True


class NegativeBank:
    """Generate ten frozen negatives with a private slot-selection RNG.

    On failure, partial raw evidence remains available in ``provenance``.
    Caption wrapper owns caption prompts/token IDs and model/decoding metadata.
    """

    def __init__(self, captioner, seed=2026, k=10, caption_attempts=80, negative_retries=10):
        self.captioner = captioner
        self.seed = seed
        self.k = k
        self.caption_attempts = caption_attempts
        self.negative_retries = negative_retries
        self.rng = random.Random(seed)
        self.provenance = {}

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

    def generate_captions(self, image):
        captions = []
        seen = set()
        for attempt in range(self.caption_attempts):
            raw = self.captioner.caption(image, sample=True)
            caption = raw.strip()
            key = ' '.join(caption.casefold().split())
            accepted = bool(key) and key not in seen
            self.provenance['caption_attempts'].append({
                'attempt': attempt + 1, 'raw_output': raw, 'accepted': accepted})
            if accepted:
                captions.append(caption)
                seen.add(key)
            if len(captions) == self.k:
                return captions
        raise RuntimeError(f'Only {len(captions)} distinct captions; need {self.k}')

    def extract_objects(self, caption):
        prompt = build_objects_prompt(self.captioner.processor.tokenizer, caption)
        raw = self.captioner.generate(prompt, max_new_tokens=80)
        phrase = parse_objects_output(raw)
        objects = _split_objects_phrase(phrase)
        record = {'caption': caption, 'phrase': phrase, 'objects': objects,
                  'raw_prompt': prompt, 'raw_output': raw}
        self.provenance['object_attempts'].append(record)
        if not 1 <= len(objects) <= 5:
            raise ValueError(f'Invalid FINER object list: {raw}')
        return record

    def generate_negative(self, object_record, forbidden=()):
        caption = object_record['caption']
        phrase = object_record['phrase']
        objects = object_record['objects']
        for attempt in range(self.negative_retries):
            index = self.rng.randint(1, len(objects))
            prompt = build_negobj_prompt(self.captioner.processor.tokenizer, caption, phrase, index)
            raw = self.captioner.generate(prompt, max_new_tokens=80)
            new_phrase = parse_objects_output(raw)
            new_objects = _split_objects_phrase(new_phrase)
            valid = _validate_single_replacement(phrase, new_phrase, index, caption)


            new_item = new_objects[index - 1] if len(new_objects) >= index else None
            changed = self.reconstruct_caption(caption, objects[index - 1], new_item) if valid else None
            key = ' '.join(changed.casefold().split()) if changed else ''
            accepted = bool(key) and key not in forbidden
            record = {
                'source_caption': caption, 'caption': changed, 'phrase': new_phrase,
                'replaced_position': index, 'old_item': objects[index - 1],
                'new_item': new_item, 'raw_prompt': prompt, 'raw_output': raw,
                'attempt': attempt + 1, 'accepted': accepted,
                'rejection_reason': None if accepted else (
                    'invalid_replacement' if not valid else
                    'ambiguous_reconstruction' if not changed else 'duplicate_caption'),
            }
            self.provenance['negative_attempts'].append(record)
            if accepted:
                return record
        raise RuntimeError(f'No validated FINER negative for: {caption}')

    def create(self, source_image):
        self.provenance = {
            'seed': self.seed, 'rng_state_before': self.rng.getstate(),
            'caption_attempts': [], 'object_attempts': [], 'negative_attempts': [],
            'grounding_verified': False, 'caption_generation_provenance': 'external wrapper',
            'caption_attempt_limit': self.caption_attempts, 'negative_retry_limit': self.negative_retries,
        }
        try:
            source = self.generate_captions(source_image)
            objects = [self.extract_objects(caption) for caption in source]
            forbidden = {' '.join(c.casefold().split()) for c in source}
            
            negatives = []
            for record in objects:
                negative = self.generate_negative(record, forbidden)
                negatives.append(negative)
                forbidden.add(' '.join(negative['caption'].casefold().split()))
            return {
                'source_captions': source, 'source_objects': objects,
                'negative_captions': [record['caption'] for record in negatives],
                'negative_details': negatives,
                'source_concepts': list(dict.fromkeys(obj for record in objects for obj in record['objects'])),
                'provenance': self.provenance,
            }
        except Exception as error:
            self.provenance['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            self.provenance['rng_state_after'] = self.rng.getstate()


class NegativeCaptionGenerator:
    """Pinned Qwen wrapper used only to prepare the notebook's negative samples.

    Each call saves its exact prompt, output, token IDs and RNG state. Existing
    incomplete generation folders require an explicitly reviewed recovery.
    """
    def __init__(self, model_id, output_dir, max_calls=190, device='cuda:0', deadline=None):
        import torch
        from pathlib import Path
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from model_lock import MODEL_REVISIONS
        self.output_dir = Path(output_dir)
        self.deadline = deadline
        self._check_deadline()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if list(self.output_dir.glob('call_*.json')):
            raise RuntimeError('Partial bank generation exists; do not silently restart it')
        if model_id not in MODEL_REVISIONS:
            raise ValueError('Pin the negative generator revision in model_revisions.json first')
        self.model_id = model_id
        self.revision = MODEL_REVISIONS[model_id]
        self.max_calls, self.calls = max_calls, 0
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_id, revision=self.revision)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=self.revision, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa').eval().to(self.device).requires_grad_(False)
        self._check_deadline()

    def _check_deadline(self):
        import time
        if self.deadline is not None and time.time() >= self.deadline:
            raise TimeoutError('Reviewed budget exhausted during bank preparation')

    def _generate(self, prompt, image=None, sample=False, max_new_tokens=80):
        import hashlib
        import torch
        from utils import atomic_write_json
        self._check_deadline()
        if self.calls >= self.max_calls:
            raise RuntimeError('Configured bank-generation call limit reached')
        self.calls += 1
        path = self.output_dir / f'call_{self.calls:04d}.json'
        settings = {'do_sample': sample, 'max_new_tokens': max_new_tokens, 'num_beams': 1}
        if sample:
            settings.update(temperature=.8, top_p=.9)
        record = {'model': self.model_id, 'revision': self.revision, 'prompt': prompt,
                  'settings': settings, 'state': 'STARTED', 'seed': torch.initial_seed(),
                  'image_sha256': hashlib.sha256(image.tobytes()).hexdigest() if image is not None else None,
                  'cpu_rng_before': torch.get_rng_state().tolist(),
                  'cuda_rng_before': torch.cuda.get_rng_state(self.device).cpu().tolist() if self.device.type == 'cuda' else None}
        atomic_write_json(path, record)
        inputs = self.processor(text=[prompt], images=[image] if image is not None else None,
                                return_tensors='pt').to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, **settings)[:, inputs.input_ids.shape[1]:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        record.update(state='COMPLETE', text=text, token_ids=generated[0].cpu().tolist(),
                      cpu_rng_after=torch.get_rng_state().tolist(),
                      cuda_rng_after=torch.cuda.get_rng_state(self.device).cpu().tolist() if self.device.type == 'cuda' else None,
                      token_limit_reached=len(generated[0]) >= max_new_tokens)
        atomic_write_json(path, record)
        return text

    def caption(self, image, sample=False):
        messages = [{'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': 'Describe this image in one concise sentence, no longer than 20 words.'}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._generate(prompt, image=image, sample=sample, max_new_tokens=50)

    def generate(self, prompt, max_new_tokens=80):
        return self._generate(prompt, max_new_tokens=max_new_tokens)

    def close(self):
        import gc
        import torch
        del self.model
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
