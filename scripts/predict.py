"""Run inference and optional Grad-CAM on a single image."""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from data import GTSRB_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from gradcam import GradCAM, overlay_heatmap
from models import build_model


def build_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Predict a traffic sign")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pt", help="Checkpoint path")
    parser.add_argument("--model", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet50"], help="Model architecture")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--gradcam", action="store_true", help="Save Grad-CAM overlay next to the image")
    parser.add_argument("--output", type=str, default="outputs/demo_gradcam.jpg", help="Grad-CAM output path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = build_transform(args.img_size)

    image = Image.open(args.image).convert("RGB")
    tensor = tfm(image).unsqueeze(0).to(device)

    model, target_layer = build_model(args.model, num_classes=len(GTSRB_CLASSES), pretrained=False)
    state = torch.load(args.checkpoint, map_location=device)
    payload = state.get("model", state)
    model.load_state_dict(payload)
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    label = GTSRB_CLASSES[pred_idx.item()]
    print(f"Prediction: {label} (conf={conf.item():.3f})")

    if args.gradcam:
        cam = GradCAM(model, target_layer)
        heatmap = cam.generate(tensor)
        vis = overlay_heatmap(tensor[0].cpu(), heatmap)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vis.save(out_path)
        print(f"Saved Grad-CAM to {out_path}")


if __name__ == "__main__":
    main()
