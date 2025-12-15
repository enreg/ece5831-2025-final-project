from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _denormalize(image: torch.Tensor):
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(-1, 1, 1)
    return image * std + mean


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hooks = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        if isinstance(class_idx, int):
            class_idx = torch.tensor([class_idx], device=input_tensor.device)

        selected = logits.gather(1, class_idx.view(-1, 1)).sum()
        selected.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze(0).squeeze(0).detach().cpu().numpy()


def overlay_heatmap(image: torch.Tensor, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    image = _denormalize(image).clamp(0, 1)
    base = TF.to_pil_image(image)
    cmap = (plt_colormap(heatmap) * 255).astype(np.uint8)
    heat = Image.fromarray(cmap, mode="RGBA").resize(base.size)
    base_rgba = base.convert("RGBA")
    blended = Image.blend(base_rgba, heat, alpha=alpha)
    return blended.convert("RGB")


def plt_colormap(arr: np.ndarray):
    import matplotlib.cm as cm

    colormap = cm.get_cmap("jet")
    colored = colormap(arr)
    return colored
