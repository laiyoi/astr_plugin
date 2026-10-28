"""Choose the collect threshold on the validation split only.

Policy: maximize recall subject to validation precision >= target.  If the
target is unreachable, choose the highest-precision operating point and warn.
The test split is never loaded here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import TasteDataset, dedupe_records, discover_records, make_loader  # noqa: E402
from model import POSITIVE_INDEX, eval_transform, load_model_from_checkpoint  # noqa: E402
from utils import binary_metrics, load_json, resolve_device, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate collect threshold on validation data")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="model_config.json to update")
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dedupe-threshold", type=int, default=6)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--update-config", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    records, report = dedupe_records(discover_records(args.data_root), threshold=args.dedupe_threshold)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)
    image_size = int(checkpoint.get("config", {}).get("image_size", 224))
    dataset = TasteDataset(records, "val", eval_transform(image_size=image_size))
    loader = make_loader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    labels: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)[:, POSITIVE_INDEX]
            labels.extend(int(x) for x in targets.tolist())
            scores.extend(float(x) for x in probs.cpu().tolist())

    collect_labels = [1 if label == POSITIVE_INDEX else 0 for label in labels]
    unique_scores = sorted(set(scores))
    candidates = [0.0] + unique_scores + [1.0]
    best = None
    fallback = None
    for threshold in candidates:
        metrics = binary_metrics(collect_labels, scores, threshold)
        if fallback is None or (metrics["precision"], metrics["recall"]) > (
            fallback["precision"],
            fallback["recall"],
        ):
            fallback = metrics
        if metrics["precision"] >= args.target_precision:
            if best is None or (metrics["recall"], threshold) > (best["recall"], best["threshold"]):
                best = metrics

    if best is None:
        chosen = fallback
        print(
            f"WARNING: no threshold reaches precision {args.target_precision:.3f}; "
            "using the highest-precision point instead."
        )
    else:
        chosen = best

    threshold = float(chosen["threshold"])
    report_payload = {
        "split": "val",
        "target_precision": args.target_precision,
        "threshold": threshold,
        "metrics": chosen,
        "all_candidates": len(candidates),
        "dedupe": report.__dict__,
    }
    out_path = args.output or args.checkpoint.parent / "threshold.json"
    save_json(out_path, report_payload)
    print(
        f"chosen threshold={threshold:.6f} precision={chosen['precision']:.4f} "
        f"recall={chosen['recall']:.4f} f1={chosen['f1']:.4f}"
    )
    print(f"wrote {out_path}")

    if args.config and args.update_config:
        config = load_json(args.config)
        config["threshold"] = threshold
        config["calibration"] = {
            "split": "val",
            "target_precision": args.target_precision,
            "metrics": chosen,
        }
        args.config.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"updated {args.config}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
