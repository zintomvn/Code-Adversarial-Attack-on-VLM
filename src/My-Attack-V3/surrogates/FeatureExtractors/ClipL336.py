"""MPC's optional L336 backbone, with pinned loading and the shared text API."""
import torch
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms

from .Base import BaseFeatureExtractor
from model_lock import MODEL_REVISIONS


class ClipL336FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        model_id = 'openai/clip-vit-large-patch14-336'
        revision = MODEL_REVISIONS[model_id]
        self.model = CLIPModel.from_pretrained(model_id, revision=revision)
        self.processor = CLIPProcessor.from_pretrained(model_id, revision=revision)
        self.normalizer = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda image: torch.clamp(image, 0, 255) / 255),
            transforms.CenterCrop(336),
            transforms.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711)),
        ])

    def forward(self, image):
        features = self.model.get_image_features(pixel_values=self.normalizer(image))
        return features / features.norm(dim=1, keepdim=True)

    def text_features_get(self, text):
        inputs = self.processor(text=text, padding=True, return_tensors='pt').to(self.model.device)
        features = self.model.get_text_features(**inputs)
        return features / features.norm(dim=1, keepdim=True)
