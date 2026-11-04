from .common import *
from .config_schema import cfg
from .utils import V53Store

class V53CaptionSelector:
    @staticmethod
    def select(blocks):
        embeddings = torch.cat([F.normalize(block.float(), dim=-1) for block in blocks], dim=-1)
        distances = (embeddings-embeddings.mean(0, keepdim=True)).square().sum(-1)
        return int(distances.argmin()), distances.tolist()



class V53HFVision:
    _instances = {}

    def __new__(cls, model_id):
        if torch.cuda.get_device_properties(cfg.V53_DEVICE).total_memory >= 80 * 2**30:
            key = (model_id, cfg.V53_REVISIONS[model_id], cfg.V53_DEVICE)
            if key not in cls._instances:
                cls._instances[key] = super().__new__(cls)
            return cls._instances[key]
        return super().__new__(cls)

    def __init__(self, model_id):
        if getattr(self, '_ready', False):
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model_id = model_id
        self.revision = cfg.V53_REVISIONS[model_id]
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.processor = AutoProcessor.from_pretrained(model_id, revision=self.revision)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=self.revision, torch_dtype=self.dtype,
            attn_implementation="sdpa", device_map={"": cfg.V53_DEVICE}).eval()
        self.model.requires_grad_(False)
        self._ready = True

    @torch.no_grad()
    def caption(self, image, sample=False, max_new_tokens=None):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": cfg.V53_CAPTION_PROMPT}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[prompt], images=[image], return_tensors="pt").to(cfg.V53_DEVICE)
        for key in batch:
            if torch.is_floating_point(batch[key]):
                batch[key] = batch[key].to(self.dtype)
        kwargs = {"do_sample": sample, "max_new_tokens": max_new_tokens or cfg.V53_VICTIM_TOKENS}
        if sample:
            kwargs.update(temperature=cfg.V53_CAPTION_TEMPERATURE, top_p=cfg.V53_CAPTION_TOP_P)
        output = self.model.generate(**batch, **kwargs)
        generated = output[:, batch["input_ids"].shape[1]:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if not text:
            raise ValueError("Empty caption; no automatic retry")
        return {"text": text, "model": self.model_id, "revision": self.revision,
                "prompt": cfg.V53_CAPTION_PROMPT, "rendered_prompt": prompt,
                "decoding": kwargs, "generated_ids": generated[0].cpu().tolist(),
                "tokens": generated.shape[-1],
                "possible_truncation": generated.shape[-1] >= kwargs["max_new_tokens"]}



class V53CaptionBank:
    def __init__(self, captioner, checkpoint_callback=None):
        self.captioner = captioner
        self.checkpoint_callback = checkpoint_callback

    def create(self, source, target, seed, budget, heartbeat, folder):
        partial = Path(folder)/"bank_partial.json"
        started = Path(folder)/"caption_started.json"
        records = V53Store.read(partial) if partial.exists() else {"source": [], "target": []}
        if started.exists():
            raise RuntimeError("Uncertain prior caption generation; inspect partial bank before repeating")
        for role, image in [("source", source), ("target", target)]:
            for index in range(len(records[role]), cfg.V53_K):
                budget.check()
                sample_seed = seed + index + (1000 if role == "target" else 0)
                V53Store.write(started, {"role": role, "index": index, "seed": sample_seed})
                if self.checkpoint_callback:
                    self.checkpoint_callback(f'bank_{role}_{index}_started', folder)
                with torch.random.fork_rng(devices=[torch.device(cfg.V53_DEVICE).index or 0]):
                    torch.manual_seed(sample_seed)
                    torch.cuda.manual_seed_all(sample_seed)
                    record = self.captioner.caption(image, sample=True, max_new_tokens=cfg.V53_CAPTION_TOKENS)
                records[role].append({**record, "seed": sample_seed})
                V53Store.write(partial, records)
                started.unlink()
                if self.checkpoint_callback:
                    self.checkpoint_callback(f'bank_{role}_{index}_saved', folder)
                heartbeat.update(role=role, caption=index+1, total=cfg.V53_K)
        records["audit_flags"] = {role: {
            "duplicate_count": len(rows)-len({r["text"].strip().lower() for r in rows}),
            "possible_truncation": any(row["possible_truncation"] for row in rows),
            "grounding": "PENDING_HUMAN_REVIEW"} for role, rows in records.items()}
        records["source_target_exact_overlap"] = sorted(
            {r["text"].strip().lower() for r in records["source"]}
            & {r["text"].strip().lower() for r in records["target"]})
        return records
