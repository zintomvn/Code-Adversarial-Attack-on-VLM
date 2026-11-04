from .common import *
from .config_schema import cfg
from .data import ImagePairDataset
from .utils import V53Store, V53Pixels
from .caption_bank import V53HFVision, V53CaptionBank, V53CaptionSelector
from .surrogates.clip import V53CLIPEnsemble
from .diffusion import V53DiffusionTarget

class V53Preparation:
    def __init__(self):
        self.root = cfg.V53_ROOT / cfg.V53_RUN_ID
        self.dataset = ImagePairDataset(cfg.V53_SOURCE_DIR)


    @staticmethod
    def inspect_pairs():
        # Read-only identity inspection; does not load pretrained models or transform images.
        rows = []
        for sample_id in cfg.V53_PAIR_IDS:
            source, target = cfg.V53_SOURCE_DIR/f"{sample_id}.png", cfg.V53_TARGET_DIR/f"{sample_id}.jpg"
            rows.append({"id": sample_id, "source": str(source), "target": str(target),
                         "source_sha256": V53Store.sha(source), "target_sha256": V53Store.sha(target)})
        return rows


    def prepare(self, pair, budget, heartbeat):
        shared = self.root/pair["id"]/"shared"
        shared.mkdir(parents=True, exist_ok=True)
        if (shared/"integrity.json").exists():
            V53Store.verify_seal(shared)
            return shared
        # Keep exact raw pair bytes on Drive as well as the224px attack inputs.
        for role, extension in [('source', '.png'), ('target', '.jpg')]:
            raw = shared / (role + '_original' + extension)
            if not raw.exists():
                shutil.copy2(pair[role], raw)
        # Reuse the base preprocessing exactly: Resize224 + CenterCrop224, RGB.
        with Image.open(pair["source"]) as handle:
            source = self.dataset.transform(ImageOps.exif_transpose(handle).convert("RGB"))
        with Image.open(pair["target"]) as handle:
            target = self.dataset.transform(ImageOps.exif_transpose(handle).convert("RGB"))
        source.save(shared/"source.png")
        target.save(shared/"target.png")
        bank_path = shared/"bank.json"
        if not bank_path.exists():
            captioner = V53HFVision(cfg.V53_VICTIMS["qwen"])
            try:
                bank = V53CaptionBank(captioner, getattr(budget, 'durable_callback', None)).create(
                    source, target, cfg.V53_SEED+int(pair["id"]), budget, heartbeat, shared)
                V53Store.write(bank_path, bank)
            finally:
                del captioner
                release_memory()
        bank = V53Store.read(bank_path)
        selection_path = shared/"selection.json"
        if not selection_path.exists():
            encoder = V53CLIPEnsemble()
            try:
                selection = {}
                for role in ["source", "target"]:
                    blocks = encoder.text_blocks([row["text"] for row in bank[role]])
                    index, distances = V53CaptionSelector.select(blocks)
                    selection[role+"_index"] = index
                    selection[role+"_squared_distances"] = distances
                    torch.save(blocks, shared/f"{role}_caption_embeddings.pt")
                V53Store.write(selection_path, selection)
            finally:
                del encoder
                release_memory()
        selection = V53Store.read(selection_path)
        if (shared/"effective_target.png").exists() != (shared/"diffusion.json").exists():
            raise RuntimeError("Incomplete diffusion persistence; review artifacts before another attempt")
        if not (shared/"effective_target.png").exists():
            marker = shared / 'diffusion_started.json'
            if marker.exists():
                raise RuntimeError('Prior diffusion attempt uncertain; do not regenerate automatically')
            budget.check()
            prompt = bank["target"][selection["target_index"]]["text"]
            # Use original source RGB resized512 as diffusion conditioning; attack source remains224.
            with Image.open(pair["source"]) as handle:
                conditioning = ImageOps.exif_transpose(handle).convert("RGB")
            V53Store.write(marker, {'prompt': prompt, 'seed': cfg.V53_DIFFUSION_SEED + int(pair['id'])})
            callback = getattr(budget, 'durable_callback', None)
            if callback:
                callback(f'{pair["id"]}_diffusion_started', shared)
            image, provenance = V53DiffusionTarget.generate_with_fallback(
                V53DiffusionTarget.generate, conditioning, target, prompt,
                cfg.V53_DIFFUSION_SEED+int(pair["id"]))
            image.save(shared/"generated_or_fallback_native.png")
            effective = image.resize((cfg.V53_SIZE, cfg.V53_SIZE), Image.Resampling.BICUBIC)
            effective.save(shared/"effective_target.png")
            provenance.update(model=cfg.V53_DIFFUSION_ID, revision=cfg.V53_REVISIONS[cfg.V53_DIFFUSION_ID],
                              strength=cfg.V53_DIFFUSION_STRENGTH, guidance=cfg.V53_DIFFUSION_GUIDANCE,
                              requested_steps=cfg.V53_DIFFUSION_STEPS,
                              effective_sha256=V53Store.sha(shared/"effective_target.png"))
            V53Store.write(shared/"diffusion.json", provenance)
            marker.unlink()
        V53Store.seal(shared, ["source.png", "target.png", "bank.json", "selection.json",
                              "effective_target.png", "diffusion.json", "generated_or_fallback_native.png",
                              "source_caption_embeddings.pt", "target_caption_embeddings.pt"])
        print("Inspect bank, pairing, effective target and negatives:", shared, flush=True)
        return shared
