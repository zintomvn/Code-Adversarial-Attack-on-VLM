"""MPC's negative cross-entropy, extended to K caption-derived anchors."""
import torch
import torch.nn.functional as F


def info_nce_loss(query, positive, negatives, temperature=0.2, omega=2.0):
    """Maximize this score with MPC's original gradient-ascent update.

    Query/positive are [B,D]; negatives are [K,D]. The reviewed pilot uses
    batch one, so negatives never accidentally mix data pairs.
    """
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if query.ndim != 2 or positive.shape != query.shape:
        raise ValueError('query and positive must have matching [B,D] shapes')
    if negatives.ndim != 2 or not len(negatives) or negatives.shape[1] != query.shape[1]:
        raise ValueError('negatives must have shape [K,D], K >= 1')
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive.detach(), dim=-1)
    negatives = F.normalize(negatives.detach(), dim=-1)
    positive_logit = omega * (query * positive).sum(-1, keepdim=True)
    logits = torch.cat([positive_logit, query @ negatives.T], dim=-1) / temperature
    return -F.cross_entropy(logits, torch.zeros(len(query), dtype=torch.long, device=query.device))


def make_negative_anchors(source_clip, negative_text, source_internvl, source_dino, lam=0.6):
    """Change only MPC's CLIP text component; retain source visual features."""
    count = next(iter(negative_text.values())).shape[0]
    clip = torch.cat([
        lam * source_clip[key] + (1 - lam) * negative_text[key]
        for key in source_clip
    ], dim=-1)
    return torch.cat([
        clip, source_internvl.expand(count, -1), source_dino.expand(count, -1)
    ], dim=-1).detach()
