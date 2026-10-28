"""DINOv2 ViT-S/14 three-class taste classifier.

ImageNet normalization is used by the official DINOv2 checkpoints.  The
preprocessing here must stay byte-for-byte identical to inference.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
from torchvision import transforms

BACKBONE_NAME = "vit_small_patch14_dinov2.lvd142m"
CLASS_NAMES = ("skip", "suspicious", "collect")
POSITIVE_INDEX = 2
IMAGE_SIZE = 224
RESIZE_SIZE = 256
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MODEL_VERSION = "1.0.0"
PREPROCESS_DESCRIPTION = (
    "convert to RGB; resize shortest edge to 256 with bicubic; center crop 224x224; "
    "scale to [0,1]; normalize with ImageNet mean/std; NCHW float32"
)


class DinoTasteModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = BACKBONE_NAME,
        *,
        pretrained: bool = True,
        num_classes: int = len(CLASS_NAMES),
        dropout: float = 0.2,
        img_size: int = IMAGE_SIZE,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=True,
        )
        self.feature_dim = int(getattr(self.backbone, "num_features", 0) or 384)
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim == 3:
            # Defensive fallback if a timm version returns tokens.
            features = features[:, 0]
        return self.head(features)


class ExportWrapper(nn.Module):
    """ONNX wrapper: fixed NCHW input -> logits [N, 3]."""

    def __init__(self, model: DinoTasteModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def build_model(
    *,
    backbone_name: str = BACKBONE_NAME,
    pretrained: bool = True,
    num_classes: int = len(CLASS_NAMES),
    dropout: float = 0.2,
    img_size: int = IMAGE_SIZE,
) -> DinoTasteModel:
    return DinoTasteModel(
        backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
        img_size=img_size,
    )


def set_trainable(model: DinoTasteModel, unfreeze_last_blocks: int = 0, full_finetune: bool = False) -> None:
    """Freeze everything except the head, then optionally unfreeze blocks.

    unfreeze_last_blocks=0 means linear probe only.  -1/full_finetune unfreezes
    the entire backbone.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True

    if full_finetune:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
        return
    if unfreeze_last_blocks <= 0:
        return

    blocks = getattr(model.backbone, "blocks", None)
    if blocks is None:
        raise ValueError("backbone has no .blocks; cannot selectively unfreeze")
    for block in list(blocks)[-unfreeze_last_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    norm = getattr(model.backbone, "norm", None)
    if norm is not None:
        for parameter in norm.parameters():
            parameter.requires_grad = True


def train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def eval_transform(image_size: int = IMAGE_SIZE, resize_size: int = RESIZE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"model", "class_names"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"checkpoint missing keys: {sorted(missing)}")
    return checkpoint


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    pretrained: bool = False,
) -> tuple[DinoTasteModel, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})
    model = build_model(
        backbone_name=config.get("backbone", BACKBONE_NAME),
        pretrained=pretrained,
        num_classes=len(checkpoint["class_names"]),
        dropout=float(config.get("dropout", 0.2)),
        img_size=int(config.get("image_size", IMAGE_SIZE)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, checkpoint


def write_model_config(
    *,
    output_path: Path,
    checkpoint: dict[str, Any],
    threshold: float,
    version: str = MODEL_VERSION,
) -> dict[str, Any]:
    config = checkpoint.get("config", {})
    payload = {
        "model_name": "dinov2_vits14",
        "backbone": config.get("backbone", BACKBONE_NAME),
        "version": version,
        "input_size": [IMAGE_SIZE, IMAGE_SIZE],
        "resize_size": RESIZE_SIZE,
        "mean": list(MEAN),
        "std": list(STD),
        "class_names": list(checkpoint.get("class_names") or CLASS_NAMES),
        "positive_class_index": list(checkpoint.get("class_names") or CLASS_NAMES).index("collect"),
        "threshold": float(threshold),
        "suspicious_policy": {
            "save_at_saved_count": 5,
            "cache_below_saved_count": 3,
        },
        "preprocess": PREPROCESS_DESCRIPTION,
        "onnx": {
            "input_name": "input",
            "output_name": "logits",
            "opset": 17,
        },
        "training": {
            "epoch": checkpoint.get("epoch"),
            "val_metrics": checkpoint.get("val_metrics"),
            "args": config,
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
