"""Small utilities shared by training/evaluation scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    predictions = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "support": len(labels),
    }


def multiclass_metrics(
    labels: list[int],
    predictions: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """Accuracy, macro F1, per-class precision/recall and confusion matrix."""
    n = len(class_names)
    confusion = [[0 for _ in range(n)] for _ in range(n)]
    for truth, pred in zip(labels, predictions):
        if 0 <= truth < n and 0 <= pred < n:
            confusion[truth][pred] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1s: list[float] = []
    for index, name in enumerate(class_names):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(n) if row != index)
        fn = sum(confusion[index][col] for col in range(n) if col != index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(confusion[index])
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        f1s.append(f1)
    correct = sum(confusion[i][i] for i in range(n))
    return {
        "accuracy": correct / len(labels) if labels else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "class_names": class_names,
        "support": len(labels),
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is false. "
            "Install the CUDA 12.8+ PyTorch build and check the NVIDIA driver."
        )
    if requested == "cuda":
        capability = torch.cuda.get_device_capability()
        print(f"CUDA device: {torch.cuda.get_device_name(0)} sm_{capability[0]}{capability[1]}")
        if capability[0] < 12:
            print("WARNING: this GPU is older than Blackwell; the CUDA 12.8 build is still fine.")
    return torch.device(requested)
