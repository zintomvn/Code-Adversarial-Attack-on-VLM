"""SAGA's original sequential crop-and-sign-gradient optimization."""

import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from tqdm import tqdm

from .crops import RegionBiasedRandomResizedCrop


def optimize(
    image,
    target,
    schedule,
    extractor,
    objective,
    *,
    device,
    steps=300,
    alpha=1.0,
    epsilon=16.0,
    scale=(0.5, 0.9),
):
    """Return clean/adversarial tensors in [0,1], and scalar loss history.

    Pixel-space optimization uses [0,255]. Steps are divided by the actual
    number of schedule entries; the last entry receives the remainder, as in
    the original progressive_hotspot_attack.
    """
    if not schedule or steps < len(schedule):
        raise ValueError(f"Need at least {len(schedule)} steps to visit the full hotspot schedule")
    clean = F.resize(
        image.to(device), [224, 224], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
    )
    delta = torch.zeros_like(clean, requires_grad=True)
    block_steps = steps // len(schedule)
    pixel_rects = [
        (int(t * 224), int(left * 224), max(1, int(h * 224)), max(1, int(w * 224)))
        for t, left, h, w in (entry["box"] for entry in schedule)
    ]
    losses = []
    progress = tqdm(range(steps), desc="SAGA", leave=False)
    for step in progress:
        phase = min(step // block_steps, len(schedule) - 1)
        # The original main path overrides --hotspot_prob with 1.0.
        cropper = RegionBiasedRandomResizedCrop(
            size=(224, 224), scale=scale, hotspot_rect=pixel_rects[phase], hotspot_prob=1.0
        )
        crop, _ = cropper(clean + delta)
        with torch.no_grad():
            objective.set_ground_truth(target, "text")
        loss = objective(extractor(crop, "image")) # crops -> optimize
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite surrogate objective at step {step}")
        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
        with torch.no_grad():
            delta.copy_((delta + alpha * grad.sign()).clamp(-epsilon, epsilon))
        losses.append(float(loss.detach()))
        progress.set_postfix(similarity=f"{losses[-1]:.4f}", phase=phase + 1)
    if delta.detach().abs().max().item() > epsilon + 1e-5:
        raise AssertionError("L-infinity constraint violated")
    return (clean / 255).clamp(0, 1).detach(), ((clean + delta) / 255).clamp(0, 1).detach(), losses
