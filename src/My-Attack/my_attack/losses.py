from .common import *
from .config_schema import cfg

class V53CropViews:
    def __init__(self, count=5, scale=(0.5, 0.9), size=224, ratio=(0.75, 4/3)):
        self.count = count
        self.scale = scale
        self.size = size
        self.ratio = ratio

    def make(self, image, seed):
        crops, boxes = [], []
        # Crop parameter draws on CPU; inspection does not consume attack RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            for _ in range(self.count):
                box = None
                for attempt in range(100):
                    candidate = RandomResizedCrop.get_params(image, self.scale, self.ratio)
                    top, left, height, width = candidate
                    fraction = height*width/(image.shape[-2]*image.shape[-1])
                    if self.scale[0] <= fraction <= self.scale[1]:
                        box = candidate
                        break
                if box is None:
                    raise RuntimeError("No crop satisfying strict area bounds")
                top, left, height, width = box
                crop = image[:, :, top:top+height, left:left+width]
                crops.append(F.interpolate(crop, (self.size, self.size), mode="bicubic", align_corners=False))
                boxes.append(list(box))
        return torch.cat(crops, dim=0), boxes



class V53Objective:
    def __init__(self, temperature=0.2, omega=2.):
        self.temperature = temperature
        self.omega = omega

    def __call__(self, query, positive, negatives, variant):
        if variant not in {"negatives_10", "negative_1"}:
            raise ValueError(f"Unknown variant: {variant}")
        if negatives is None or negatives.shape[0] < 1:
            raise ValueError("InfoNCE requires negatives; user requested 10 versus 1")
        expected = 10 if variant == "negatives_10" else 1
        if negatives.shape[0] != expected or positive.shape != (1, query.shape[-1]):
            raise ValueError("Incorrect positive/negative shapes for the reviewed variant")
        q, p, n = [F.normalize(x.float(), dim=-1) for x in (query, positive, negatives)]
        positive_sim = q @ p.T
        negative_sim = q @ n.T
        logits = torch.cat([self.omega*positive_sim, negative_sim], dim=-1)/self.temperature
        loss = F.cross_entropy(logits, torch.zeros(q.shape[0], device=q.device, dtype=torch.long))
        return loss, {"loss": float(loss.detach()), "positive_cosine": float(positive_sim.mean().detach()),
                      "max_negative_cosine": float(negative_sim.max().detach()),
                      "query_shape": list(q.shape), "negative_shape": list(n.shape),
                      "direction": "minimize_CE", "reduction": "mean_over_5_crops"}
