"""Choose the collect threshold on the validation split only.

All paths/settings come from training/config.json; run with no arguments.

Policy: maximize recall subject to validation precision >= target. If the
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

import config_utils  # noqa: E402
from dataset import TasteDataset, dedupe_records, discover_records, make_loader  # noqa: E402
from model import POSITIVE_INDEX, eval_transform, load_model_from_checkpoint  # noqa: E402
from utils import binary_metrics, load_json, resolve_device, save_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate collect threshold")
    parser.add_argument("--ckpt", type=int, default=None, help="use ckpt/epoch_NNNN.pt")
    cli = parser.parse_args()
    cfg = config_utils.section("calibrate")
    p = config_utils.paths()
    checkpoint_path = config_utils.checkpoint_path(cli.ckpt)
    tag = f"_epoch{cli.ckpt:04d}" if cli.ckpt is not None else ""
    device = resolve_device(str(cfg.device))
    records, report = dedupe_records(
        discover_records(p.data_root, max_image_pixels=int(cfg.max_image_pixels)),
        threshold=int(cfg.dedupe_threshold),
    )
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    image_size = int(checkpoint.get("config", {}).get("image_size", 224))
    dataset = TasteDataset(records, "val", eval_transform(image_size=image_size))
    loader = make_loader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )

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
        if metrics["precision"] >= float(cfg.target_precision):
            if best is None or (metrics["recall"], threshold) > (best["recall"], best["threshold"]):
                best = metrics

    if best is None:
        chosen = fallback
        print(
            f"WARNING: no threshold reaches precision {float(cfg.target_precision):.3f}; "
            "using the highest-precision point instead."
        )
    else:
        chosen = best

    threshold = float(chosen["threshold"])
    report_payload = {
        "split": "val",
        "checkpoint": str(checkpoint_path),
        "target_precision": float(cfg.target_precision),
        "threshold": threshold,
        "metrics": chosen,
        "all_candidates": len(candidates),
        "dedupe": report.__dict__,
    }
    out_path = p.run_dir / f"threshold{tag}.json"
    save_json(out_path, report_payload)
    print(
        f"chosen threshold={threshold:.6f} precision={chosen['precision']:.4f} "
        f"recall={chosen['recall']:.4f} f1={chosen['f1']:.4f}"
    )
    print(f"wrote {out_path}")

    if bool(cfg.update_model_config) and cli.ckpt is None:
        config = load_json(p.model_config)
        config["threshold"] = threshold
        config["calibration"] = {
            "split": "val",
            "target_precision": float(cfg.target_precision),
            "metrics": chosen,
        }
        p.model_config.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"updated {p.model_config}")
    elif cli.ckpt is not None:
        print("epoch checkpoint calibration: model_config.json was not modified")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
