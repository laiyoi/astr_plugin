"""Dataset scanning, exact+near duplicate removal, and torch Datasets.

Split layout (required):

    data_root/
      train/collect/*.jpg
      train/skip/*.jpg
      val/collect/*.jpg
      val/skip/*.jpg
      test/collect/*.jpg
      test/skip/*.jpg

`collect` is positive (label 1), `skip` is negative (label 0).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_TO_INDEX = {"skip": 0, "collect": 1}
INDEX_TO_CLASS = {0: "skip", 1: "collect"}
SPLITS = ("train", "val", "test")


@dataclass
class Record:
    path: Path
    label: int
    split: str
    sha256: str = ""
    dhash: int = 0
    removed: bool = False
    reason: str = ""


@dataclass
class DedupeReport:
    removed: list[dict[str, Any]] = field(default_factory=list)
    kept: dict[str, dict[str, int]] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path, hash_size: int = 8) -> int:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        image = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return int(a ^ b).bit_count()


def discover_records(data_root: Path) -> list[Record]:
    records: list[Record] = []
    for split in SPLITS:
        for class_name, label in CLASS_TO_INDEX.items():
            directory = data_root / split / class_name
            if not directory.is_dir():
                raise FileNotFoundError(f"missing directory: {directory}")
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                records.append(Record(path=path, label=label, split=split))
    if not records:
        raise ValueError("no images found")
    for record in records:
        record.sha256 = sha256_file(record.path)
        record.dhash = dhash(record.path)
    return records


def dedupe_records(records: Iterable[Record], threshold: int = 6) -> tuple[list[Record], DedupeReport]:
    """Remove exact/near duplicates, never removing a train image for a val/test one.

    Within a split the first path wins.  Across splits, train beats val, val
    beats test; the lower-priority record is removed and recorded.
    """
    records = list(records)
    report = DedupeReport()
    kept: list[Record] = []

    # 1) Within-split duplicates.
    for split in SPLITS:
        split_records = [r for r in records if r.split == split]
        split_kept: list[Record] = []
        for record in split_records:
            duplicate = None
            for prior in split_kept:
                if record.sha256 == prior.sha256 or hamming(record.dhash, prior.dhash) <= threshold:
                    duplicate = prior
                    break
            if duplicate is None:
                split_kept.append(record)
            else:
                record.removed = True
                record.reason = f"duplicate within {split} of {duplicate.path}"
                report.removed.append({"path": str(record.path), "split": split, "reason": record.reason})
        kept.extend(split_kept)

    # 2) Cross-split duplicates: drop the lower-priority side.
    priority = {"train": 0, "val": 1, "test": 2}
    result: list[Record] = []
    for record in kept:
        victim = None
        for other in result:
            if other.split == record.split:
                continue
            if record.sha256 != other.sha256 and hamming(record.dhash, other.dhash) > threshold:
                continue
            if priority[record.split] >= priority[other.split]:
                victim = record
                keeper = other
            else:
                victim = other
                keeper = record
            break
        if victim is None:
            result.append(record)
        elif victim is record:
            record.removed = True
            record.reason = f"near-duplicate across split of {keeper.path} in {keeper.split}"
            report.removed.append({"path": str(record.path), "split": record.split, "reason": record.reason})
        else:
            victim.removed = True
            victim.reason = f"near-duplicate across split of {record.path} in {record.split}"
            report.removed.append({"path": str(victim.path), "split": victim.split, "reason": victim.reason})
            result.remove(victim)
            result.append(record)

    final = [r for r in result if not r.removed]
    for split in SPLITS:
        split_records = [r for r in final if r.split == split]
        report.kept[split] = {
            "total": len(split_records),
            "collect": sum(1 for r in split_records if r.label == 1),
            "skip": sum(1 for r in split_records if r.label == 0),
        }
    return final, report


def write_dedupe_report(path: Path, report: DedupeReport) -> None:
    path.write_text(json.dumps(report.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")


class TasteDataset(Dataset):
    def __init__(self, records: list[Record], split: str, transform: transforms.Compose):
        self.records = [r for r in records if r.split == split and not r.removed]
        if not self.records:
            raise ValueError(f"no records left for split {split}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return tensor, record.label, str(record.path)


def class_counts(records: list[Record], split: str) -> tuple[int, int]:
    subset = [r for r in records if r.split == split and not r.removed]
    return sum(1 for r in subset if r.label == 0), sum(1 for r in subset if r.label == 1)


def class_weights(records: list[Record], split: str = "train") -> torch.Tensor:
    negative, positive = class_counts(records, split)
    if negative == 0 or positive == 0:
        raise ValueError(f"split {split} needs both classes, got skip={negative}, collect={positive}")
    total = negative + positive
    return torch.tensor([total / (2 * negative), total / (2 * positive)], dtype=torch.float32)


def make_loader(
    dataset: TasteDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    balanced_sampler: bool = False,
    seed: int = 42,
) -> DataLoader:
    sampler = None
    if balanced_sampler:
        labels = [record.label for record in dataset.records]
        counts = np.bincount(labels, minlength=2)
        sample_weights = [1.0 / counts[label] for label in labels]
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=generator)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
