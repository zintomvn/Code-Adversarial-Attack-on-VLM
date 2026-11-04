"""Migrated directly from CARA v6.4.1; see SOURCE_MAP.json and ATTRIBUTION.md."""
from .config import INSTALL_PACKAGES, RUN_ATTACK, RUN_EVALUATION, RUN_VISUALIZATION, NUM_SAMPLES, START_INDEX, SEED, DATA_ROOT, SOURCE_DIR, TARGET_DIR, PAIR_MANIFEST, OUTPUT_DIR, K, STEPS, XAI_METHOD, EPSILON, ALPHA, MOMENTUM, TEMPERATURE, OMEGA, CROP_AREA, ALPHA_GLOBAL, ALPHA_CROP, SAVE_BRANCH_IMAGES, SAVE_EVERY, SAVE_CROP_EVERY, ATTENTION_LAYER, NEGATIVE_RETRIES, CAPTION_ATTEMPTS, IMAGE_SIZE, CLIP_G14_ID, CLIP_B16_ID, CLIP_B32_ID, QWEN_ID, EVALUATION_ID, CLIP_DEVICES, ATTACK_DEVICE, QWEN_MAX_MEMORY_GIB, IMAGE_WEIGHT, TEXT_WEIGHT, CAPTION_TEMPERATURE, CAPTION_TOP_P, EVALUATION_PROMPT, RUN_ID, MODEL_REVISIONS
from .common import *
class RegionSelector:
    def __init__(self, height=224, width=224):
        self.height = height
        self.width = width

    @staticmethod
    def stage_area(step):
        return CROP_AREA

    def select(self, heatmap, step):
        area = self.stage_area(step)
        height = max(1, min(self.height, round(self.height * math.sqrt(area))))
        width = max(1, min(self.width, round(self.width * math.sqrt(area))))
        if height == self.height and width == self.width:
            return 0, 0, height, width, "xai"

        heat = F.interpolate(heatmap[None, None].float(), size=(self.height, self.width),
                             mode="bilinear", align_corners=False)[0, 0]
        integral = F.pad(heat.double(), (1, 0, 1, 0)).cumsum(0).cumsum(1)
        scores = (integral[height:, width:] - integral[:-height, width:]
                  - integral[height:, :-width] + integral[:-height, :-width])
        index = int(scores.flatten().argmax())
        top, left = divmod(index, self.width - width + 1)
        return top, left, height, width, "xai"


class XAIMap:
    @staticmethod
    def normalize(maps):
        valid = {}
        for name, value in maps.items():
            value = torch.as_tensor(value).float().cpu()
            if value.ndim != 2 or not torch.isfinite(value).all():
                continue
            if value.min() < 0 or value.max() - value.min() <= 1e-8:
                continue
            small = F.interpolate(value[None, None], (32, 32), mode="bilinear",
                                  align_corners=False)[0, 0]
            if small.max() - small.min() <= 1e-8:
                continue
            valid[name] = small / small.sum()
        if not valid:
            raise ValueError("No informative XAI heatmap")
        return torch.stack(list(valid.values())).mean(0), valid

    @staticmethod
    def dispersion(heatmap):
        probability = heatmap / heatmap.sum()
        positive = probability[probability > 0]
        entropy = -(positive * positive.log()).sum()
        yy, xx = torch.meshgrid(torch.linspace(0, 1, probability.shape[0]),
                                torch.linspace(0, 1, probability.shape[1]), indexing="ij")
        cy = (probability * yy).sum()
        cx = (probability * xx).sum()
        variance = (probability * ((yy - cy).square() + (xx - cx).square())).sum()
        return {
            "entropy": float(entropy / math.log(probability.numel())),
            "effective_support": float(entropy.exp() / probability.numel()),
            "variance": float(variance),
        }
