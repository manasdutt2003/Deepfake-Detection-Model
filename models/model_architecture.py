import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone
        self.backbone = models.resnet18(pretrained=True)

        # Replace classifier
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, 1
        )

    def forward(self, x):
        # IMPORTANT: return RAW LOGITS (NO SIGMOID)
        return self.backbone(x)

