"""Evaluate a trained DINOv2 classifier on val or test.

The threshold must come from calibration on val (or an explicit argument for
diagnostics).  This script never changes the threshold.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import TasteDataset, dedupe_records, discover_records, make_loader  # noqa: E402
from model import POSITIVE_INDEX, eval_transform, load_model_from_checkpoint  # noqa: E402
from utils import binary_metrics, load_json, resolve_device, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 skip/suspicious/collect classifier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dedupe-threshold", type=int, default=6)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    threshold = args.threshold
    if args.config:
        threshold = float(load_json(args.config).get("threshold", 0.5))
    if threshold is None:
        threshold = 0.5
    if args.threshold is not None:
        print("WARNING: explicit --threshold is for diagnostics; do not tune it on test.")

    records, report = dedupe_records(discover_records(args.data_root), threshold=args.dedupe_threshold)
    print("Dedupe kept:", report.kept)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)
    image_size = int(checkpoint.get("config", {}).get("image_size", 224))
    dataset = TasteDataset(records, args.split, eval_transform(image_size=image_size))
    loader = make_loader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    labels: list[int] = []
    scores: list[float] = []
    paths: list[str] = []
    with torch.no_grad():
        for images, targets, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)[:, POSITIVE_INDEX]
            labels.extend(int(x) for x in targets.tolist())
            scores.extend(float(x) for x in probs.cpu().tolist())
            paths.extend(batch_paths)

    collect_labels = [1 if label == POSITIVE_INDEX else 0 for label in labels]
    metrics = binary_metrics(collect_labels, scores, threshold)
    false_positives = [
        {"path": p, "score": s}
        for p, y, s in zip(paths, collect_labels, scores)
        if y == 0 and s >= threshold
    ]
    false_negatives = [
        {"path": p, "score": s}
        for p, y, s in zip(paths, collect_labels, scores)
        if y == 1 and s < threshold
    ]
    output = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": {
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        },
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "dedupe": report.__dict__,
        "class_names": checkpoint.get("class_names"),
    }
    out_path = args.output or args.checkpoint.parent / f"evaluation_{args.split}.json"
    save_json(out_path, output)

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "score", "prediction"], lineterminator="\n")
        writer.writeheader()
        for path, label, score in zip(paths, labels, scores):
            writer.writerow(
                {
                    "path": path,
                    "label": checkpoint.get("class_names", ["skip", "suspicious", "collect"])[label] if label < len(checkpoint.get("class_names", [])) else str(label),
                    "score": f"{score:.8f}",
                    "prediction": "collect" if score >= threshold else "skip",
                }
            )

    print(f"split={args.split} threshold={threshold:.6f}")
    print(
        f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']} | "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
    )
    print(f"false positives: {len(false_positives)} (listed in {out_path})")
    print(f"false negatives: {len(false_negatives)} (listed in {out_path})")
    print(f"wrote {out_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
