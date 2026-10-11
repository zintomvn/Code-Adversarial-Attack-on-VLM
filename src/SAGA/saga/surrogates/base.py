from abc import abstractmethod
from typing import Any

import torch
from torch import Tensor, nn


class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class EnsembleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: list[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor, type: str) -> Tensor:
        features = {}
        for i, model in enumerate(self.extractors):
            features[i] = model(x, type).squeeze()
        return features


class EnsembleFeatureLoss(nn.Module):
    def __init__(self, extractors: list[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor, type: str):
        self.ground_truth.clear()
        for model in self.extractors:
            self.ground_truth.append(model(x, type))

    def __call__(self, feature_dict: dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            loss += torch.mean(torch.sum(feature * gt, dim=1))

        loss = loss / len(self.extractors)

        return loss
