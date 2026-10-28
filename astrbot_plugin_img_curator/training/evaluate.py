"""Evaluate the trained DINOv2 classifier on the configured split.

All paths/settings come from training/config.json; run with no arguments.
The threshold must come from calibration on val; this script never changes it.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config_utils  # noqa: E402
from dataset import TasteDataset, dedupe_records, discover_records, make_loader  # noqa: E402
from model import POSITIVE_INDEX, eval_transform, load_model_from_checkpoint  # noqa: E402
from utils import binary_metrics, load_json, resolve_device, save_json  # noqa: E402


def main() -> int:
    cfg = config_utils.section("evaluate")
    p = config_utils.paths()
    device = resolve_device(str(cfg.device))
    threshold = float(load_json(p.model_config).get("threshold", 0.5))
    split = str(cfg.split)

    records, report = dedupe_records(
        discover_records(p.data_root, max_image_pixels=int(cfg.max_image_pixels)),
        threshold=int(cfg.dedupe_threshold),
    )
    print("Dedupe kept:", report.kept)
    model, checkpoint = load_model_from_checkpoint(p.checkpoint, device=device)
    image_size = int(checkpoint.get("config", {}).get("image_size", 224))
    dataset = TasteDataset(records, split, eval_transform(image_size=image_size))
    loader = make_loader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )

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
        {"path": path, "score": score}
        for path, label, score in zip(paths, collect_labels, scores)
        if label == 0 and score >= threshold
    ]
    false_negatives = [
        {"path": path, "score": score}
        for path, label, score in zip(paths, collect_labels, scores)
        if label == 1 and score < threshold
    ]
    class_names = list(checkpoint.get("class_names") or ["skip", "suspicious", "collect"])
    output = {
        "split": split,
        "checkpoint": str(p.checkpoint),
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
        "class_names": class_names,
    }
    out_path = p.run_dir / f"evaluation_{split}.json"
    save_json(out_path, output)

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "label", "score", "prediction"],
            lineterminator="\n",
        )
        writer.writeheader()
        for path, label, score in zip(paths, labels, scores):
            writer.writerow(
                {
                    "path": path,
                    "label": class_names[label] if label < len(class_names) else str(label),
                    "score": f"{score:.8f}",
                    "prediction": "collect" if score >= threshold else "not_collect",
                }
            )

    print(f"split={split} threshold={threshold:.6f}")
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
