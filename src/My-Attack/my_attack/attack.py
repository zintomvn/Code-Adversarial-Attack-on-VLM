from .common import *
from .config_schema import cfg
from .utils import V53Store, V53Pixels
from .losses import V53CropViews, V53Objective
from .visualization import V53Preview

class V53Attack:
    def __init__(self, encoder, budget, heartbeat, checkpoint_callback=None):
        self.checkpoint_callback = checkpoint_callback
        self.encoder = encoder
        self.budget = budget
        self.heartbeat = heartbeat
        self.cropper = V53CropViews(cfg.V53_CROPS, cfg.V53_CROP_SCALE, cfg.V53_SIZE, cfg.V53_CROP_RATIO)
        self.objective = V53Objective(cfg.V53_TEMPERATURE, cfg.V53_OMEGA)

    @torch.no_grad()
    def anchors(self, source, effective_target, bank, selection, variant):
        target_caption = bank["target"][selection["target_index"]]["text"]
        source_captions = [row["text"] for row in bank["source"]]
        if variant == "negative_1":
            source_captions = [source_captions[selection["source_index"]]]
        positive = self.encoder.fused(effective_target, [target_caption]).detach()
        negatives = self.encoder.fused(source, source_captions).detach()
        return positive, negatives

    def run(self, source, effective_target, bank, selection, folder, sample_id, variant):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        final = folder / "adversarial.png"
        if final.exists():
            V53Store.verify_seal(folder)
            V53Store.write(folder / "bounds.json", V53Pixels.verify(final, source, cfg.V53_EPSILON))
            return
        positive, negatives = self.anchors(source, effective_target, bank, selection, variant)
        checkpoint = folder / "checkpoint.pt"
        if checkpoint.exists():
            if V53Store.sha(checkpoint) != V53Store.read(folder/"checkpoint_hash.json")["sha256"]:
                raise RuntimeError("Checkpoint modified")
            state = torch.load(checkpoint, map_location=source.device, weights_only=True)
            delta, momentum = state["delta"], state["momentum"]
            start_step = state["step"]+1
            if cfg.INTERACTIVE_REVIEW:
                V53Store.require_review(folder, "step1")
        else:
            delta, momentum = torch.zeros_like(source), torch.zeros_like(source)
            start_step = 1
        for step in range(start_step, cfg.V53_STEPS+1):
            self.budget.check()
            delta = delta.detach().requires_grad_(True)
            current = (source+delta).clamp(0, 255)
            crop_seed = cfg.V53_SEED + int(sample_id)*100000 + step
            crops, boxes = self.cropper.make(current, crop_seed)
            features = self.encoder.encode_image(crops)
            loss, record = self.objective(features, positive, negatives, variant)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            gradient, = torch.autograd.grad(loss, delta)
            if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
                raise FloatingPointError("Nonfinite or zero full-image gradient")
            record.update(step=step, seed=crop_seed, boxes=boxes,
                          gradient_shape=list(gradient.shape), gradient_l1=float(gradient.abs().sum()),
                          gradient_nonzero_fraction=float((gradient != 0).float().mean()),
                          variant=variant, update_rule="full_image_MI_FGSM_descent",
                          temperature=cfg.V53_TEMPERATURE, omega=cfg.V53_OMEGA,
                          image_weight=cfg.V53_IMAGE_WEIGHT, text_weight=cfg.V53_TEXT_WEIGHT,
                          crop_raw_min=float(crops.detach().min()), crop_raw_max=float(crops.detach().max()),
                          clip_input_range="base adapter clamps resized pixels/255 to [0,1]")
            with torch.no_grad():
                normalized = gradient/gradient.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
                momentum = cfg.V53_MOMENTUM*momentum+normalized
                adv = V53Pixels.project(source, delta-cfg.V53_ALPHA*momentum.sign(), cfg.V53_EPSILON)
                delta = (adv-source).detach()
            self.heartbeat.update(sample=sample_id, variant=variant, step=step, loss=record["loss"])
            with (folder / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record)+"\n")
            # Persist optimizer state every step; resume preserves full delta and momentum.
            torch.save({"step": step, "delta": delta.cpu(), "momentum": momentum.cpu()}, checkpoint)
            V53Store.write(folder/"checkpoint_hash.json", {"sha256": V53Store.sha(checkpoint)})
            if step == 1 or step % cfg.V53_SAVE_EVERY == 0 or step == cfg.V53_STEPS:
                # Save the exact views fed to CLIP before this update, not resampled views.
                view_dir = folder / "views" / f"step_{step:03d}"
                V53Pixels.save(current, view_dir / "input_adv.png")
                for index, crop in enumerate(crops):
                    V53Pixels.save(crop[None], view_dir / f"crop_{index}.png")
                V53Store.write(view_dir / "record.json", record)
                if step == 1:
                    torch.save({"actual_float_crops": crops.detach().cpu(),
                                "full_image_gradient": gradient.detach().cpu(),
                                "positive": positive.cpu(), "negatives": negatives.cpu()},
                               view_dir/"first_step_tensors.pt")
                V53Pixels.save(adv, folder / f"step_{step:03d}.png")
            if self.checkpoint_callback and (step == 1 or step % cfg.V53_SAVE_EVERY == 0):
                self.checkpoint_callback(f"optimizer_{sample_id}_{variant}_{step}", folder)
            if step == 1:
                V53Preview.show(folder.parent/"shared", folder)
                if cfg.INTERACTIVE_REVIEW:
                    V53Store.require_review(folder, "step1")
        V53Pixels.save(source+delta, final)
        V53Store.write(folder / "bounds.json", V53Pixels.verify(final, source, cfg.V53_EPSILON))
        V53Store.write(folder / "attack_status.json", {"status": "COMPLETED", "steps": cfg.V53_STEPS})
        V53Store.seal(folder, ["adversarial.png", "attack_status.json", "checkpoint.pt", "checkpoint_hash.json"])
