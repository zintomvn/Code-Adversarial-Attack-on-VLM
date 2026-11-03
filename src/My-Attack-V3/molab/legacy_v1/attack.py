"""MPC random-crop MI-FGSM; the objective accepts a frozen negative bank."""
import math
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop, InterpolationMode

from losses import info_nce_loss


def project_delta(proposal, source, epsilon):
    """L-infinity projection, also ensuring valid RGB before serialization."""
    proposal = proposal.clamp(-epsilon, epsilon)
    return torch.maximum(torch.minimum(proposal, 255 - source), -source)


def make_crop_schedule(size, steps, seed):
    """Sample MPC's original crop distribution without changing global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        image = torch.empty(1, 3, size, size)
        return [RandomResizedCrop.get_params(image, (0.5, 0.9), (3 / 4, 4 / 3)) for _ in range(steps)]


def save_png(tensor, path):
    """Round once at export so integer source pixels retain the exact bound."""
    from PIL import Image
    pixels = tensor.detach().cpu()[0].permute(1, 2, 0).clamp(0, 255).round().byte().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def verify_png(path, source, epsilon=16):
    import numpy as np
    from PIL import Image
    pixels = torch.from_numpy(np.array(Image.open(path).convert('RGB'), copy=True)).permute(2, 0, 1)[None]
    bound = (pixels.float() - source.detach().cpu()).abs().max().item()
    if bound > epsilon:
        raise RuntimeError(f'Saved PNG perturbation {bound} exceeds {epsilon}')
    return {'saved_png_linf': bound, 'rgb_range': [int(pixels.min()), int(pixels.max())]}


def run_attack(encoder, source, positive, negatives, initial_delta, crops, output, checkpoint,
               first_step_gate, deadline, epsilon=16, alpha=1, tau=.2, omega=2):
    """One attempt. Checkpoint callback durably commits before progression.

    Encoder exposes the original MPC concatenated image features. Evaluation
    never participates in gradients or checkpoint selection.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'attack_started.json'
    if marker.exists():
        raise RuntimeError('Attack already attempted; automatic rerun is disabled')
    marker.write_text('{"attempt": 1}')
    delta = initial_delta.detach().clone().to(source.device).requires_grad_(True)
    momentum = torch.zeros_like(delta)
    checkpoint(0, source, {'stage': 'initialized'}, delta, momentum)
    history = []
    for index, box in enumerate(crops):
        step = index + 1
        if time.time() >= deadline:
            raise TimeoutError('Cumulative approved compute limit exhausted')
        top, left, height, width = box
        # Same bilinear RandomResizedCrop behavior as original MPC.
        view = TF.resized_crop(source + delta, top, left, height, width,
                               [224, 224], InterpolationMode.BILINEAR, antialias=True)
        query = encoder(view)

        # loss function 
        score = info_nce_loss(query, positive, negatives, tau, omega)

        # gradient 
        gradient, = torch.autograd.grad(score, delta)


        if not torch.isfinite(score) or not torch.isfinite(gradient).all() or not gradient.abs().sum() > 0:
            raise FloatingPointError('Nonfinite loss/gradient or disconnected image gradient')
        with torch.no_grad():
            momentum = .9 * momentum + gradient
            delta = project_delta(delta + alpha * momentum.sign(), source, epsilon)
        delta = delta.detach().requires_grad_(True)
        q = torch.nn.functional.normalize(query.detach(), dim=-1)
        p = torch.nn.functional.normalize(positive.detach(), dim=-1)
        n = torch.nn.functional.normalize(negatives.detach(), dim=-1)
        record = {
            'step': step, 'cross_entropy': float(-score.detach()), 'score': float(score.detach()),
            'target_similarity': float((q * p).sum(-1).mean()),
            'negative_similarities': (q @ n.T).cpu().tolist(),
            'gradient_l1': float(gradient.abs().sum()), 'gradient_linf': float(gradient.abs().max()),
            'query_shape': list(query.shape), 'negative_shape': list(negatives.shape),
            'crop': list(box), 'crop_scale': height * width / (224 * 224),
            'direction': 'ascent_negative_cross_entropy', 'tau': tau, 'omega': omega,
        }
        history.append(record)
        if step == 1:
            save_png(view, output / 'actual_crop_step001.png')
            torch.save(gradient.detach().cpu(), output / 'gradient_step001.pt')
        if step == 1 or step % 30 == 0 or step == len(crops):
            checkpoint(step, source + delta, record, delta, momentum)
        if step == 1:
            first_step_gate(record)
    final = output / 'adversarial.png'
    save_png(source + delta, final)
    verification = verify_png(final, source, epsilon)
    return history, verification
