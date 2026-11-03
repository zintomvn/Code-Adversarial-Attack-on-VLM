"""Original MPC ensemble plus a separately loaded caption-bank/evaluation model."""
import gc
import json
from pathlib import Path

import torch

from model_lock import MODEL_REVISIONS
from losses import make_negative_anchors
from runtime import AttemptLedger

CAPTION_PROMPT = 'Describe this image in one concise sentence, no longer than 20 words.'


class Captioner:
    """Record every generation before returning it; no external inference API."""
    def __init__(self, model_id, record_dir, max_calls, max_tokens=50, persist=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model_id = model_id
        self.revision = MODEL_REVISIONS[model_id]
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.ledger = AttemptLedger(self.record_dir, max_calls)
        self.calls = len(self.ledger.records())
        self.persist = persist or (lambda stage: None)
        self.processor = AutoProcessor.from_pretrained(model_id, revision=self.revision)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=self.revision, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', device_map='cuda:0').eval().requires_grad_(False)

    @torch.no_grad()
    def _generate(self, prompt, image=None, sample=False, max_new_tokens=80, key=None):
        key = key or f'bank_call_{self.calls + 1:04d}'
        cached = self.ledger.completed(key)
        if cached is not None:
            return cached['text']
        if self.calls >= self.max_calls:
            raise RuntimeError('Reviewed caption generation call cap reached')
        attempt = self.ledger.start(key, {'model': self.model_id, 'revision': self.revision, 'prompt': prompt})
        self.calls += 1
        self.persist(f'generation_attempt_{self.calls:04d}')
        inputs = self.processor(text=[prompt], images=[image] if image is not None else None,
                                return_tensors='pt').to(self.model.device)
        settings = {'max_new_tokens': max_new_tokens, 'do_sample': sample, 'num_beams': 1}
        if sample:
            settings.update(temperature=.8, top_p=.9)
        cpu_rng = torch.get_rng_state().tolist()
        cuda_rng = torch.cuda.get_rng_state().cpu().tolist()
        generated = self.model.generate(**inputs, **settings)[:, inputs.input_ids.shape[1]:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        record = {'model': self.model_id, 'revision': self.revision, 'prompt': prompt,
                  'settings': settings, 'text': text, 'token_ids': generated[0].cpu().tolist(),
                  'input_ids': inputs.input_ids[0].cpu().tolist(), 'seed': torch.initial_seed(),
                  'cpu_rng_before': cpu_rng, 'cuda_rng_before': cuda_rng,
                  'truncated': len(generated[0]) >= max_new_tokens and int(generated[0, -1]) not in eos,
                  'dtype': 'bfloat16', 'attention': 'sdpa',
                  'processor': self.processor.to_dict() if hasattr(self.processor, 'to_dict') else str(type(self.processor))}
        self.ledger.finish(attempt, record)
        self.persist(f'generation_complete_{self.calls:04d}')
        return text

    def caption(self, image, sample=False, key=None):
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': CAPTION_PROMPT}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._generate(prompt, image, sample, self.max_tokens, key=key)

    def generate(self, prompt, max_new_tokens=80):
        return self._generate(prompt, max_new_tokens=max_new_tokens)

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class MPCEncoder:
    """The same three CLIPs, InternVL3-1B and DINOv2 as the MPC entry point."""
    def __init__(self, device='cuda:0'):
        from surrogates import (ClipB16FeatureExtractor, ClipB32FeatureExtractor, ClipLaionFeatureExtractor,
                                InternVL3_1B_FeatureExtractor, DINOv2FeatureExtractor, EnsembleFeatureExtractor_our)
        self.device = device
        clips = [factory().eval().requires_grad_(False).to(device) for factory in
                 (ClipB16FeatureExtractor, ClipB32FeatureExtractor, ClipLaionFeatureExtractor)]
        self.clip = EnsembleFeatureExtractor_our(clips)
        self.internvl = InternVL3_1B_FeatureExtractor().eval().requires_grad_(False).to(device)
        self.dino = DINOv2FeatureExtractor().eval().requires_grad_(False).to(device)

    def components(self, image):
        return self.clip(image), self.internvl(image).mean(dim=1).float(), self.dino(image)

    def __call__(self, image):
        clip, internvl, dino = self.components(image)
        return torch.cat([*clip.values(), internvl, dino], dim=-1)

    @torch.no_grad()
    def anchors(self, source, target, source_caption, target_caption, bank):
        source_clip, source_iv, source_dino = self.components(source)
        target_clip, target_iv, target_dino = self.components(target)
        source_text = self.clip.get_text_features(source_caption)
        target_text = self.clip.get_text_features(target_caption)
        positive = torch.cat([
            *[.6 * target_clip[key] + .4 * target_text[key] for key in target_clip], target_iv, target_dino
        ], dim=-1).detach()
        baseline = torch.cat([
            *[.6 * source_clip[key] + .4 * source_text[key] for key in source_clip], source_iv, source_dino
        ], dim=-1).detach()
        negative_text = self.clip.get_text_features(bank['negative_captions'])
        variant = make_negative_anchors(source_clip, negative_text, source_iv, source_dino)
        return positive, baseline, variant

    def close(self):
        del self.clip, self.internvl, self.dino
        gc.collect()
        torch.cuda.empty_cache()
