from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


def _groups(channels: int) -> int:
    for value in (32, 16, 8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class CNN2GN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNN4GN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "CNN2_GN":
        return CNN2GN(num_classes)
    if model_name == "CNN4_GN":
        return CNN4GN(num_classes)
    if model_name == "ResNet18_GN":
        model = resnet18(
            weights=None,
            num_classes=num_classes,
            norm_layer=lambda channels: nn.GroupNorm(_groups(channels), channels),
        )
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.avgpool = SpatialMean()
        return model
    raise ValueError(f"Unknown model_name={model_name}")


class SpatialMean(nn.Module):
    """Deterministic global spatial mean with ResNet-compatible output shape."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-2, -1), keepdim=True)


def model_payload_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())
