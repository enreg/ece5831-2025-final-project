from typing import Tuple

import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int = 43, pretrained: bool = True) -> Tuple[nn.Module, nn.Module]:
    name = name.lower()
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        target_layer = model.features[-1]
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        target_layer = model.layer4[-1]
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model, target_layer


def freeze_backbone(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in get_classifier(model).parameters():
        param.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, num_blocks: int = 2):
    for block in get_backbone_blocks(model)[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in get_classifier(model).parameters():
        param.requires_grad = True


def get_classifier(model: nn.Module) -> nn.Module:
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "fc"):
        return model.fc
    raise AttributeError("Model missing classifier attribute")


def get_backbone_blocks(model: nn.Module):
    if hasattr(model, "features"):
        return list(model.features.children())
    if hasattr(model, "layer1"):
        return [model.layer1, model.layer2, model.layer3, model.layer4]
    raise AttributeError("Model missing recognizable backbone modules")
