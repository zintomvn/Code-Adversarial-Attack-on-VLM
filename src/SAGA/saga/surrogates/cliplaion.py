import torch
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

from .base import BaseFeatureExtractor


class ClipLaionFeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(ClipLaionFeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-G-14-laion2B-s12B-b42K")
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-G-14-laion2B-s12B-b42K")
        self.normalizer = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                ),
            ]
        )

    def forward(self, x, type):
        if type == "image":
            inputs = dict(pixel_values=self.normalizer(x))
            image_features = self.model.get_image_features(**inputs)
            if not torch.is_tensor(image_features):
                image_features = image_features.pooler_output
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            return image_features
        elif type == "text":
            inputs = self.processor(text=x, return_tensors="pt", padding=True, truncation=True).to(
                self.model.device
            )
            text_features = self.model.get_text_features(**inputs)
            if not torch.is_tensor(text_features):
                text_features = text_features.pooler_output
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            return text_features
