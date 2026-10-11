import random

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as F


class RegionBiasedRandomResizedCrop(nn.Module):
    """
    RandomResizedCrop with biased sampling of hotspot region.
    """

    def __init__(
        self,
        size,
        scale=(0.8, 1.0),
        hotspot_rect: tuple[int, int, int, int] = None,
        hotspot_prob: float = 0.8,
        antialias=True,
    ):
        super().__init__()
        # Use standard RandomResizedCrop internally
        self.standard_cropper = transforms.RandomResizedCrop(size=size, scale=scale, antialias=antialias)

        if hotspot_rect is None:
            # If hotspot is not provided, always use standard crop
            self.hotspot_prob = 0.0
        else:
            self.hotspot_prob = hotspot_prob
            self.top, self.left, self.height, self.width = hotspot_rect

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # With p_hot probability, crop from the hotspot region
        if random.random() < self.hotspot_prob:
            # 1. Crop the hotspot region from the original image
            img_hotspot = F.crop(img, self.top, self.left, self.height, self.width)

            # 2. Apply standard RandomResizedCrop to the cropped hotspot region
            #    (If the hotspot is too small to crop, an error may occur, so exception handling)
            try:
                return self.standard_cropper(img_hotspot), "hotspot"
            except ValueError:
                # If the hotspot is too small to crop, just resize
                return F.resize(img_hotspot, self.standard_cropper.size, antialias=True), "hotspot"
        else:
            # With the remaining probability, use standard RandomResizedCrop on the entire image
            return self.standard_cropper(img), "global"
