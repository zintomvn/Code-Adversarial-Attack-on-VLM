"""Three CLIP surrogates used by the supplied main experiment."""

from .base import EnsembleFeatureExtractor, EnsembleFeatureLoss
from .clipb16 import ClipB16FeatureExtractor
from .clipb32 import ClipB32FeatureExtractor
from .cliplaion import ClipLaionFeatureExtractor


def load_surrogates(device):
    models = [
        cls().eval().to(device).requires_grad_(False)
        for cls in (ClipB16FeatureExtractor, ClipB32FeatureExtractor, ClipLaionFeatureExtractor)
    ]
    return EnsembleFeatureExtractor(models), EnsembleFeatureLoss(models)
