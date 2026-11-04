"""H1 execution/debug policy. Frozen attack inputs and optimization math stay intact."""
import contextlib
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def correct_log_metrics(metrics, k, batch_size):
    result = dict(metrics)
    if "local_similarity" in result:
        result["legacy_local_similarity"] = float(result["local_similarity"])
        result["local_similarity"] = float(result["local_similarity"]) * k
        result["similarity_per_image"] = result["local_similarity"] / batch_size
    return result


def seed_attack(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class Recorder:
    def __init__(self, directory, seed):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.views = 0
        self.losses = []
        self.gradients = []
        self.calls = 0

    def view(self, kind, source, result, **metadata):
        if self.views >= 8:
            return
        index = self.views
        self.views += 1
        value = result.detach().float().cpu()
        if not torch.isfinite(value).all():
            raise RuntimeError("Nonfinite augmented view")
        record = {"index": index, "kind": kind, "seed": self.seed,
                  "shape": list(value.shape), "input_shape": list(source.shape),
                  "range": [value.min().item(), value.max().item()], **metadata}
        if "crop_ijhw" in record:
            i, j, h, w = record["crop_ijhw"]
            if min(i, j) < 0 or min(h, w) <= 0 or i+h > source.shape[-2] or j+w > source.shape[-1]:
                raise RuntimeError("Crop outside input bounds")
        for sample, tensor in enumerate(value if value.ndim == 4 else value[None]):
            F.to_pil_image(tensor.clamp(0, 1)).save(self.directory/f"view_{index:02d}_{sample:02d}.png")
        with (self.directory/"views.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record)+"\n")

    @contextlib.contextmanager
    def crop_hooks(self):
        crop_forward = T.RandomResizedCrop.forward
        crop_params = T.RandomResizedCrop.get_params
        flip_forward = T.RandomHorizontalFlip.forward
        rotate_forward = T.RandomRotation.forward
        rotate_params = T.RandomRotation.get_params
        metadata = {}

        def params(*args, **kwargs):
            result = crop_params(*args, **kwargs)
            metadata["crop_ijhw"] = list(result)
            return result

        def crop(module, image):
            # CUDA AA resize backward is nondeterministic in pinned torch2.10.
            # Preserve crop sampling/interpolation while using deterministic CPU autograd.
            cpu_gradient_resize = image.is_cuda and image.requires_grad
            working = image.to("cpu") if cpu_gradient_resize else image
            result = crop_forward(module, working)
            if cpu_gradient_resize:
                result = result.to(image.device)
            self.view("RandomResizedCrop", image, result, scale=list(module.scale),
                      interpolation_device="cpu" if cpu_gradient_resize else str(image.device), **metadata)
            return result

        def flip(module, image):
            result = flip_forward(module, image)
            self.view("RandomHorizontalFlip", image, result, flipped=not torch.equal(image, result))
            return result

        def angle(*args, **kwargs):
            result = rotate_params(*args, **kwargs)
            metadata["rotation_degrees"] = result
            return result

        def rotate(module, image):
            result = rotate_forward(module, image)
            self.view("RandomRotation", image, result, rotation_degrees=metadata.get("rotation_degrees"))
            return result

        T.RandomResizedCrop.get_params = staticmethod(params)
        T.RandomResizedCrop.forward = crop
        T.RandomHorizontalFlip.forward = flip
        T.RandomRotation.get_params = staticmethod(angle)
        T.RandomRotation.forward = rotate
        try:
            yield
        finally:
            T.RandomResizedCrop.get_params = staticmethod(crop_params)
            T.RandomResizedCrop.forward = crop_forward
            T.RandomHorizontalFlip.forward = flip_forward
            T.RandomRotation.get_params = staticmethod(rotate_params)
            T.RandomRotation.forward = rotate_forward

    def check_gradient(self, gradient):
        value = gradient.detach().float()
        finite = bool(torch.isfinite(value).all())
        maximum = float(value.abs().max())
        per_sample = value.reshape(value.shape[0], -1).norm(dim=1) if value.ndim == 4 else value.reshape(1, -1).norm(dim=1)
        record = {"finite": finite, "abs_max": maximum, "l2": float(value.norm()),
                  "per_sample_l2": per_sample.cpu().tolist(), "amp_scaled": True, "shape": list(value.shape)}
        self.gradients.append(record)
        write(self.directory/"gradients.json", self.gradients)
        if not finite or maximum == 0 or bool((per_sample == 0).any()):
            raise RuntimeError("Invalid or zero adversarial-image gradient in first-step gate")
        return gradient

    def gate(self):
        if not self.losses or not self.gradients or self.views == 0:
            raise RuntimeError("First-step gate lacks loss/gradient/crop evidence")
        write(self.directory/"first_step_gate.json", {"technical_checks": "PASS", "seed": self.seed,
              "actual_views_saved": self.views, "loss_checks": len(self.losses),
              "gradient_checks": len(self.gradients), "caption_success": "PENDING", "negative_data": "N/A",
              "concept_grounding": "known limitation; unchanged author keyword bank"})


def install(run, directory, seed=2023):
    """Install in a dedicated worker process, before frozen campaign_worker runs."""
    import attack.base as base
    import surrogates.loss as losses
    recorder = Recorder(directory, seed)
    original_seed = run.set_environment
    original_attack = base.PGDAttackMutiPass.attack
    original_compute = base.AttackFramework.compute_loss_and_metrics
    original_loss = losses.EnsWeightedMultiAlignmentLoss.forward
    original_log = base.log_metrics
    context = {"active": False, "k": 10, "batch": 2, "gate": False}

    def set_environment(value=seed):
        original_seed(value)
        seed_attack(value)

    def attack(module, *args, **kwargs):
        seed_attack(seed)
        context.update(active=True, k=kwargs.get("multi_pass_num", 5),
                       batch=kwargs.get("image_org", args[1] if len(args)>1 else None).shape[0])
        write(recorder.directory/"execution_policy.json", {"seed": seed, "deterministic_algorithms": True,
              "cudnn_benchmark": False, "tf32": False, "reset_at_attack_entry": True,
              "source_gradient_crop_resize": "CPU antialiased interpolation; original RNG/parameters; differentiable transfers",
              "K": context["k"], "batch_size": context["batch"], "torch": torch.__version__})
        with recorder.crop_hooks():
            try:
                result = original_attack(module, *args, **kwargs)
                recorder.gate()
                return result
            finally:
                context["active"] = False

    def compute(module, adv_image, *args, **kwargs):
        if recorder.calls < 2 and context["active"]:
            if list(adv_image.shape) != [2, 3, 224, 224]:
                raise RuntimeError("Expected two-pair adversarial batch")
            adv_image.register_hook(recorder.check_gradient)
        return original_compute(module, adv_image, *args, **kwargs)

    def loss(module, source_img, target_img, target_num=1):
        capture = context["active"] and recorder.calls < 2
        recorder.calls += int(context["active"])
        if not capture:
            return original_loss(module, source_img, target_img, target_num)
        if target_num != 3 or target_img.shape[0] != 3*source_img.shape[0]:
            raise RuntimeError("Expected primary plus two auxiliary targets per source")
        features = [[] for _ in module.extractors]
        handles = [extractor.register_forward_hook(lambda _m, _i, out, index=i: features[index].append(out.detach().float()))
                   for i, extractor in enumerate(module.extractors)]
        try:
            result = original_loss(module, source_img, target_img, target_num)
        finally:
            for handle in handles:
                handle.remove()
        if not torch.isfinite(result):
            raise RuntimeError("Nonfinite alignment loss")
        components = []
        for target, source in features:
            target = target.reshape(source_img.shape[0], target_num, -1)
            source = source.reshape(source_img.shape[0], -1)
            cosine = torch.nn.functional.cosine_similarity(source[:, None], target, dim=-1)
            components.append(cosine.cpu().tolist())
        values = torch.tensor(components)
        reconstructed = (values[:, :, 0].sum() + module.beta*values[:, :, 1:].sum())/len(features)/target_num
        scalar = float(result.detach())
        error = abs(scalar-float(reconstructed))
        if error > .02:
            raise RuntimeError("Alignment loss/component mismatch")
        recorder.losses.append({"loss_scalar_batch_sum": scalar, "per_model_pair_primary_aux_cosines": components,
              "reconstructed_float32": float(reconstructed), "amp_reconstruction_abs_error": error,
              "target_num": target_num, "beta": module.beta, "source_shape": list(source_img.shape),
              "target_shape": list(target_img.shape), "optimization_sign": "negative similarity / K"})
        write(recorder.directory/"loss_components.json", recorder.losses)
        return result

    def log(progress, metrics, *args, **kwargs):
        epoch = args[1] if len(args)>1 else kwargs.get("epoch")
        # Final metrics have no local_similarity and need no scaling correction.
        corrected = correct_log_metrics(metrics, context["k"], context["batch"])
        if "local_similarity" in metrics:
            if not context["gate"]:
                recorder.gate()
                context["gate"] = True
                print("H1_FIRST_STEP_TECHNICAL_GATE_PASS", flush=True)
            with (recorder.directory/"metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"epoch": epoch, **{k: float(v) for k,v in corrected.items()}})+"\n")
        return original_log(progress, corrected, *args, **kwargs)

    run.set_environment = set_environment
    base.PGDAttackMutiPass.attack = attack
    base.AttackFramework.compute_loss_and_metrics = compute
    losses.EnsWeightedMultiAlignmentLoss.forward = loss
    base.log_metrics = log
    write(recorder.directory/"overlay_identity.json", {"sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "frozen_kit_modified": False, "policy": "H1 deterministic/instrumented rerun"})
    return recorder
