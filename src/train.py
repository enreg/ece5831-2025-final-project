import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DataConfig, GTSRB_CLASSES, get_dataloaders
from gradcam import GradCAM, overlay_heatmap
from models import build_model, freeze_backbone, unfreeze_last_blocks
from utils import (
    accuracy,
    classification_report,
    expected_calibration_error,
    gather_predictions,
    plot_metrics_summary,
    plot_confusion,
    plot_reliability,
)


def mixup_batch(x, y, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def rand_bbox(size, lam):
    _, _, H, W = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_batch(x, y, alpha: float):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[index]
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        use_mixup = mixup_alpha > 0
        use_cutmix = not use_mixup and cutmix_alpha > 0
        if use_mixup:
            images, targets_a, targets_b, lam = mixup_batch(images, labels, mixup_alpha)
        elif use_cutmix:
            images, targets_a, targets_b, lam = cutmix_batch(images, labels, cutmix_alpha)
        outputs = model(images)

        if use_mixup or use_cutmix:
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
        else:
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

        loss.backward()
        optimizer.step()
        if scheduler and isinstance(scheduler, OneCycleLR):
            scheduler.step()

        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total
    acc = total_correct / total
    if scheduler and isinstance(scheduler, CosineAnnealingLR):
        scheduler.step()
    return avg_loss, acc


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs, preds, labs = gather_predictions(outputs, labels)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labs)
    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = accuracy(preds, labels)
    report = classification_report(preds, labels)
    ece, bin_acc, bin_conf, bin_counts = expected_calibration_error(probs, labels)
    return {
        "accuracy": acc,
        "macro_f1": report["macro_f1"],
        "ece": ece,
        "per_class_precision": report["per_class_precision"].tolist(),
        "per_class_recall": report["per_class_recall"].tolist(),
        "probs": probs,
        "preds": preds,
        "labels": labels,
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
        "bin_counts": bin_counts,
    }


def save_gradcam_samples(model, target_layer, loader, device, output_dir: Path, class_names, max_images: int = 4):
    cam = GradCAM(model, target_layer)
    saved = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        for i in range(images.size(0)):
            if saved >= max_images:
                cam.clear_hooks()
                return
            heatmap = cam.generate(images[i : i + 1])
            overlay = overlay_heatmap(images[i].cpu(), heatmap)
            name = f"gradcam_{saved}_pred-{class_names[preds[i]]}_true-{class_names[labels[i]]}.jpg"
            overlay.save(output_dir / name)
            saved += 1
    cam.clear_hooks()


def fit(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = DataConfig(
        data_root=Path(args.data_root),
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        balanced_sampler=args.balanced_sampler,
        download=not args.no_download,
    )
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(data_cfg)
    ds = train_loader.dataset.dataset if hasattr(train_loader.dataset, "dataset") else train_loader.dataset
    class_names = getattr(ds, "classes", GTSRB_CLASSES[:num_classes])

    model, target_layer = build_model(args.model, num_classes=num_classes, pretrained=not args.no_pretrained)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    freeze_backbone(model)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.head_lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_acc = 0.0
    history = []

    for epoch in range(args.head_epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scheduler=None,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
        )
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_metrics["accuracy"], "val_f1": val_metrics["macro_f1"]})
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {"model": model.state_dict(), "epoch": epoch + 1}
        print(f"[Head] Epoch {epoch+1}: train_acc={train_acc:.4f}, val_acc={val_metrics['accuracy']:.4f}")

    unfreeze_last_blocks(model, args.unfreeze_blocks)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scheduler=scheduler,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
        )
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": args.head_epochs + epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_metrics["accuracy"], "val_f1": val_metrics["macro_f1"]})
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {"model": model.state_dict(), "epoch": args.head_epochs + epoch + 1}
        print(f"[Finetune] Epoch {epoch+1}: train_acc={train_acc:.4f}, val_acc={val_metrics['accuracy']:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_state:
        torch.save(best_state, output_dir / "best_model.pt")
        model.load_state_dict(best_state["model"])

    final_metrics = evaluate(model, val_loader, device)
    from sklearn.metrics import confusion_matrix as sk_cm

    cm = sk_cm(final_metrics["labels"], final_metrics["preds"])
    plot_confusion(cm, class_names, output_dir / "confusion_matrix.png")
    plot_reliability(final_metrics["bin_conf"], final_metrics["bin_acc"], final_metrics["bin_counts"], output_dir / "reliability.png")

    if not args.skip_test:
        test_metrics = evaluate(model, test_loader, device)
    else:
        test_metrics = {}

    if args.run_gradcam:
        save_gradcam_samples(model, target_layer, val_loader, device, output_dir / "gradcam", class_names)

    payload = {
        "best_val_acc": best_val_acc,
        "history": history,
        "val": {k: v for k, v in final_metrics.items() if k in ["accuracy", "macro_f1", "ece"]},
        "test": {k: v for k, v in test_metrics.items() if k in ["accuracy", "macro_f1", "ece"]},
        "config": vars(args),
    }
    plot_metrics_summary(payload["val"], payload["test"], output_dir / "metrics_summary.png")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    print("Training complete. Artifacts saved to", output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="GTSRB Transfer Learning Trainer")
    parser.add_argument("--data-root", type=str, default="data", help="Path to store GTSRB data")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for checkpoints/plots")
    parser.add_argument("--model", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet50"], help="Backbone model")
    parser.add_argument("--img-size", type=int, default=96, help="Square image size")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--head-epochs", type=int, default=3, help="Epochs to train classifier head only")
    parser.add_argument("--epochs", type=int, default=12, help="Fine-tuning epochs after head warmup")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--unfreeze-blocks", type=int, default=2, help="How many last backbone blocks to unfreeze")
    parser.add_argument("--scheduler", type=str, default="onecycle", choices=["onecycle", "cosine", "none"], help="LR scheduler for fine-tuning")
    parser.add_argument("--balanced-sampler", action="store_true", help="Use class-balanced sampler")
    parser.add_argument("--run-gradcam", action="store_true", help="Export Grad-CAM overlays on validation samples")
    parser.add_argument("--skip-test", action="store_true", help="Skip running on test split")
    parser.add_argument("--no-download", action="store_true", help="Assume data already present; do not download")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet initialization")
    return parser.parse_args()


if __name__ == "__main__":
    fit(parse_args())
