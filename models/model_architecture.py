import torch.nn as nn
from torchvision import models


class DeepfakeDetector(nn.Module):
    """ResNet-18 with a single-logit head: fake (1) versus real (0)."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        # ImageNet weights are only useful as a starting point for training.
        # Inference loads a full checkpoint, so skip the download there.
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        # Raw logits. Apply sigmoid to get the probability of "fake".
        return self.backbone(x)
