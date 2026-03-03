from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNet18Backbone(nn.Module):
    def __init__(self, num_classes: int = 10, proj_dim: int | None = None) -> None:
        super().__init__()
        net = resnet18(num_classes=num_classes)
        self.encoder = nn.Sequential(*list(net.children())[:-1])
        self.feature_dim = net.fc.in_features
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.projector = None
        if proj_dim is not None:
            self.projector = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, proj_dim),
            )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.encoder(x).flatten(1)
        logits = self.classifier(feats)
        if return_features:
            proj = self.projector(feats) if self.projector is not None else None
            return logits, feats, proj
        return logits


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)


def infer_proj_dim(state_dict: dict[str, torch.Tensor]) -> int | None:
    """Infer projector output dimension from a saved model state dict.

    Returns None for checkpoints trained without a projector head.
    """
    proj_weight = state_dict.get("projector.2.weight")
    if proj_weight is None:
        return None
    return int(proj_weight.shape[0])
