from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    return float((preds == targets).mean())


def gather_predictions(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    preds = probs.argmax(axis=1)
    labels = targets.detach().cpu().numpy()
    return probs, preds, labels


def classification_report(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    macro_f1 = f1_score(labels, preds, average="macro")
    per_class = precision_recall_fscore_support(labels, preds, average=None)
    return {"macro_f1": float(macro_f1), "per_class_precision": per_class[0], "per_class_recall": per_class[1]}


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accuracies = preds == labels
    bin_bounds = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_acc = []
    bin_conf = []
    bin_counts = []
    for i in range(n_bins):
        mask = (confidences > bin_bounds[i]) & (confidences <= bin_bounds[i + 1])
        count = mask.sum()
        bin_counts.append(int(count))
        if count > 0:
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()
        else:
            acc = 0.0
            conf = 0.0
        bin_acc.append(acc)
        bin_conf.append(conf)
        ece += (count / len(confidences)) * abs(acc - conf)
    return float(ece), np.array(bin_acc), np.array(bin_conf), np.array(bin_counts)


def plot_confusion(cm: np.ndarray, class_names, save_path: Path):
    # Simpler, readable heatmap for 43 classes (no per-cell text to avoid clutter).
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="GTSRB Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    ax.tick_params(axis="both", which="major", labelsize=6)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_reliability(bin_conf: np.ndarray, bin_acc: np.ndarray, bin_counts: np.ndarray, save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.bar(bin_conf, bin_acc, width=1.0 / len(bin_conf), alpha=0.6, edgecolor="black", label="Observed")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(val_metrics: Dict, test_metrics: Dict, save_path: Path):
    entries = []
    if val_metrics:
        entries.append(("Val Acc", val_metrics.get("accuracy")))
        entries.append(("Val Macro-F1", val_metrics.get("macro_f1")))
        entries.append(("Val ECE", val_metrics.get("ece")))
    if test_metrics:
        entries.append(("Test Acc", test_metrics.get("accuracy")))
        entries.append(("Test Macro-F1", test_metrics.get("macro_f1")))
        entries.append(("Test ECE", test_metrics.get("ece")))
    entries = [(k, v) for k, v in entries if v is not None]
    if not entries:
        return

    labels, values = zip(*entries)
    colors = ["#4a90e2" if "Val" in lbl else "#50c878" for lbl in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, max(values) * 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Validation/Test Metrics")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
