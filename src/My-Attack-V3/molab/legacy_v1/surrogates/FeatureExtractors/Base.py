import torch
from torch import nn, Tensor
from abc import abstractmethod
from typing import List

class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

class EnsembleFeatureExtractor_our(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor_our, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        # features = []
        # for model in self.extractors:
        #     features.append(model(x).squeeze())
        # features = torch.cat(features, dim=0)
        features = {}
        for i, model in enumerate(self.extractors):
            # features[i] = model(x).squeeze()
            features[i] = model(x)
        return features

    def get_text_features(self, text):
        text_features = {}  # 不拼接，改为字典存储
        for i, model in enumerate(self.extractors):
            # text_features[i] = model.text_features_get(text).squeeze()
            text_features[i] = model.text_features_get(text)
        return text_features
