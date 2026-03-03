from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class ResNetCifar(nn.Module):
    """CIFAR-style ResNet (3x3 stem, no maxpool) used in KNN-OOD paper setups."""

    def __init__(self, block: type[BasicBlock], layers: list[int], num_classes: int = 10, proj_dim: int | None = None) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.feature_dim = 512 * block.expansion
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.projector = None
        if proj_dim is not None:
            self.projector = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, proj_dim),
            )

        self._init_weights()

    def _make_layer(self, block: type[BasicBlock], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.encode(x)
        logits = self.classifier(feats)
        if return_features:
            proj = self.projector(feats) if self.projector is not None else None
            return logits, feats, proj
        return logits


class ResNet18Backbone(ResNetCifar):
    def __init__(self, num_classes: int = 10, proj_dim: int | None = None) -> None:
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, proj_dim=proj_dim)


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
