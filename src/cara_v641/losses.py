"""Migrated directly from CARA v6.4.1; see SOURCE_MAP.json and ATTRIBUTION.md."""
from .config import INSTALL_PACKAGES, RUN_ATTACK, RUN_EVALUATION, RUN_VISUALIZATION, NUM_SAMPLES, START_INDEX, SEED, DATA_ROOT, SOURCE_DIR, TARGET_DIR, PAIR_MANIFEST, OUTPUT_DIR, K, STEPS, XAI_METHOD, EPSILON, ALPHA, MOMENTUM, TEMPERATURE, OMEGA, CROP_AREA, ALPHA_GLOBAL, ALPHA_CROP, SAVE_BRANCH_IMAGES, SAVE_EVERY, SAVE_CROP_EVERY, ATTENTION_LAYER, NEGATIVE_RETRIES, CAPTION_ATTEMPTS, IMAGE_SIZE, CLIP_G14_ID, CLIP_B16_ID, CLIP_B32_ID, QWEN_ID, EVALUATION_ID, CLIP_DEVICES, ATTACK_DEVICE, QWEN_MAX_MEMORY_GIB, IMAGE_WEIGHT, TEXT_WEIGHT, CAPTION_TEMPERATURE, CAPTION_TOP_P, EVALUATION_PROMPT, RUN_ID, MODEL_REVISIONS
from .common import *
def info_nce_loss(feat_adv, feat_tgt, feat_src, temperature=0.2, mode='img', omega=2.0):
    """
    InfoNCE loss for pulling adv->tgt close, pushing adv->src away.

    Args:
        feat_adv: Tensor [B, D] adversarial features
        feat_tgt: Tensor [B, D] target features (positive)
        feat_src: Tensor [B, D] source features (negative)
        temperature: float, softmax temperature

    Returns:
        loss: scalar tensor
    """
    # L2 normalize
    feat_adv = F.normalize(feat_adv, dim=-1)
    feat_tgt = F.normalize(feat_tgt, dim=-1)
    feat_src = F.normalize(feat_src, dim=-1)
    # Cosine similarities
    if mode == 'text':
        sim_pos = torch.sum(feat_adv * feat_tgt, dim=-1).mean(dim=1) * omega / temperature  # [B]
        sim_neg = torch.sum(feat_adv * feat_src, dim=-1).mean(dim=1) / temperature  # [B]
    else:
        sim_pos = torch.sum(feat_adv * feat_tgt, dim=-1) * omega / temperature  # [B]
        sim_neg = torch.sum(feat_adv * feat_src, dim=-1) / temperature  # [B]
    # Stack positive and negative: shape [B, 2]
    logits = torch.stack([sim_pos, sim_neg], dim=1)
    labels = torch.zeros(feat_adv.size(0), dtype=torch.long, device=feat_adv.device)  # positives index=0
    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    return - loss


class ContrastiveLoss:
    def __init__(self, temperature=0.2, omega=2.0):
        self.temperature = temperature
        self.omega = omega

    def __call__(self, adversarial, positive, negatives):
        query = l2_normalize(adversarial)
        positive = l2_normalize(positive)
        negatives = l2_normalize(negatives)
        if query.ndim != 2 or positive.shape != query.shape:
            raise ValueError("Expected query and positive of shape [B, D]")
        if negatives.ndim != 2 or negatives.shape[1] != query.shape[1]:
            raise ValueError("Expected K negative anchors of shape [K, D]")

        positive_sim = (query * positive).sum(dim=-1, keepdim=True)
        negative_sim = query @ negatives.T
        logits = torch.cat([self.omega * positive_sim, negative_sim], dim=-1)
        labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        cross_entropy = F.cross_entropy(logits / self.temperature, labels)
        return -cross_entropy, positive_sim.mean(), negative_sim.max()


class CropGlobalLoss(ContrastiveLoss):
    def gradients(self, encoder, source, delta, box, positive, negatives):
        """Evaluate both branches at the SAME merged perturbation; free graphs sequentially."""
        top, left, height, width = box
        region = (slice(None), slice(None), slice(top, top+height), slice(left, left+width))
        global_delta = delta.detach().clone().requires_grad_(True)
        global_features = encoder.encode_image((source + global_delta).clamp(0, 255))
        global_score, global_pos, global_neg = self(global_features, positive, negatives)
        if not torch.isfinite(global_score): raise FloatingPointError('Nonfinite global loss')
        global_gradient, = torch.autograd.grad(global_score, global_delta)
        global_metrics = (float(-global_score.detach()), float(global_pos.detach()), float(global_neg.detach()))
        del global_features, global_score, global_pos, global_neg, global_delta

        crop_delta = delta[region].detach().clone().requires_grad_(True)
        crop_features = encoder.encode_image((source[region] + crop_delta).clamp(0, 255))
        crop_score, crop_pos, crop_neg = self(crop_features, positive, negatives)
        if not torch.isfinite(crop_score): raise FloatingPointError('Nonfinite crop loss')
        crop_gradient, = torch.autograd.grad(crop_score, crop_delta)
        crop_metrics = (float(-crop_score.detach()), float(crop_pos.detach()), float(crop_neg.detach()))
        for name, gradient in [('global',global_gradient),('crop',crop_gradient)]:
            if not torch.isfinite(gradient).all() or gradient.abs().sum()==0:
                raise RuntimeError(f'Invalid {name} gradient')
        record = {'loss':global_metrics[0]+crop_metrics[0],
                  'global_loss':global_metrics[0], 'crop_loss':crop_metrics[0],
                  'target_similarity':global_metrics[1], 'max_negative_similarity':global_metrics[2],
                  'crop_target_similarity':crop_metrics[1], 'crop_max_negative_similarity':crop_metrics[2],
                  'global_gradient_l1':float(global_gradient.abs().sum()),
                  'crop_gradient_l1':float(crop_gradient.abs().sum())}
        return global_gradient.detach(), crop_gradient.detach(), record
