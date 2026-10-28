"""Build the split training dataset from raw skip/suspicious/collect media.

Raw layout:

    training/dataset_raw/
      collect/      # images, GIF, videos
      skip/

Output layout:

    training/dataset/
      train/collect/*.jpg
      train/skip/*.jpg
      val/collect/*.jpg
      val/skip/*.jpg
      test/collect/*.jpg
      test/skip/*.jpg

Images/GIF are copied as-is. Videos are decoded with the same high-change,
uniform-window frame selector used by the runtime detector:

    core.video.extract_frames(...)

All selected frames from one video stay in the same split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _SCRIPT_DIR.parent / "astrbot_plugin_img_curator"
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from core.video import VIDEO_SUFFIXES, extract_frames, ffmpeg_available  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
SPLITS = ("train", "val", "test")
CLASSES = {"skip": 0, "suspicious": 1, "collect": 2}
CLASS_NAMES = {index: name for name, index in CLASSES.items()}


@dataclass
class Item:
    path: Path
    label: int
    group: str
    sha256: str
    split: str = "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess skip/suspicious/collect media into train/val/test")
    parser.add_argument("--raw-root", type=Path, default=_SCRIPT_DIR / "dataset_raw")
    parser.add_argument("--output-root", type=Path, default=_SCRIPT_DIR / "dataset")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--video-max-frames", type=int, default=8)
    parser.add_argument("--video-max-seconds", type=int, default=15)
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--clean", action="store_true", help="delete output-root first")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(raw_root: Path) -> list[Item]:
    items: list[Item] = []
    seen: dict[str, Item] = {}
    for class_name, label in CLASSES.items():
        directory = raw_root / class_name
        if not directory.is_dir():
            raise FileNotFoundError(f"missing raw directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in IMAGE_SUFFIXES and suffix not in VIDEO_SUFFIXES:
                continue
            digest = sha256_file(path)
            if digest in seen:
                print(f"skip exact duplicate: {path} -> {seen[digest].path}")
                continue
            is_video = suffix in VIDEO_SUFFIXES
            group = f"video:{digest}" if is_video else f"image:{digest}"
            item = Item(path=path, label=label, group=group, sha256=digest)
            seen[digest] = item
            items.append(item)
    if not items:
        raise ValueError(f"no images/videos found under {raw_root}")
    return items


def assign_splits(items: list[Item], *, val_ratio: float, test_ratio: float, seed: int) -> None:
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("val/test ratios must be in [0, 1)")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1")

    rng = random.Random(seed)
    for label in CLASSES.values():
        label_items = [item for item in items if item.label == label]
        groups: dict[str, list[Item]] = {}
        for item in label_items:
            groups.setdefault(item.group, []).append(item)
        keys = list(groups)
        if len(keys) < 3:
            raise ValueError(
                f"class {label} needs at least 3 image/video groups for train/val/test, got {len(keys)}"
            )
        rng.shuffle(keys)
        n = len(keys)
        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * test_ratio))
        while n_val + n_test >= n:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        for index, key in enumerate(keys):
            if index < n_val:
                split = "val"
            elif index < n_val + n_test:
                split = "test"
            else:
                split = "train"
            for item in groups[key]:
                item.split = split


def preprocess_image(item: Item, output_root: Path) -> Path:
    target_dir = output_root / item.split / CLASS_NAMES[item.label]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{item.sha256[:16]}{item.path.suffix.lower()}"
    if not target.exists():
        shutil.copy2(item.path, target)
    return target


def preprocess_video(item: Item, output_root: Path, args: argparse.Namespace) -> list[Path]:
    target_dir = output_root / item.split / CLASS_NAMES[item.label]
    target_dir.mkdir(parents=True, exist_ok=True)
    work = output_root / ".work" / item.sha256[:20]
    try:
        frames = extract_frames(
            item.path,
            work,
            max_frames=args.video_max_frames,
            max_seconds=args.video_max_seconds,
            fps=args.video_fps,
        )
        outputs: list[Path] = []
        for index, frame in enumerate(frames):
            target = target_dir / f"{item.sha256[:12]}_{index:03d}_{frame.time_sec:06.2f}.jpg"
            if not target.exists():
                shutil.copy2(frame.path, target)
            outputs.append(target)
        return outputs
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    args = parse_args()
    if not args.raw_root.is_dir():
        raise SystemExit(f"raw root not found: {args.raw_root}")
    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)

    items = discover(args.raw_root)
    assign_splits(
        items,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )
    for split in SPLITS:
        for class_name in CLASSES:
            (args.output_root / split / class_name).mkdir(parents=True, exist_ok=True)

    report: dict = {"raw_root": str(args.raw_root), "output_root": str(args.output_root), "splits": {}, "errors": []}
    for item in items:
        try:
            if item.path.suffix.lower() in VIDEO_SUFFIXES:
                if not ffmpeg_available():
                    raise RuntimeError("ffmpeg is not installed or not on PATH")
                outputs = preprocess_video(item, args.output_root, args)
                count = len(outputs)
            else:
                preprocess_image(item, args.output_root)
                count = 1
            bucket = report["splits"].setdefault(
                item.split,
                {name: 0 for name in CLASSES},
            )
            bucket[CLASS_NAMES[item.label]] += count
        except Exception as exc:  # noqa: BLE001
            report["errors"].append({"path": str(item.path), "error": str(exc)})
            print(f"error: {item.path}: {exc}")

    report_path = args.output_root / "preprocess_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))
    if report["errors"]:
        print(f"errors: {len(report['errors'])}, see {report_path}")
        return 1
    print(f"done: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
