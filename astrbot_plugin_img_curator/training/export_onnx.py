"""Export the DINOv2 backbone + three-class head to a fixed-shape ONNX model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (  # noqa: E402
    ExportWrapper,
    IMAGE_SIZE,
    load_model_from_checkpoint,
    write_model_config,
)
from utils import load_json, resolve_device, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DINOv2 taste classifier to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    config = load_json(args.config) if args.config else None
    if config is None:
        config = write_model_config(
            output_path=args.output_dir / "model_config.json",
            checkpoint=checkpoint,
            threshold=0.5,
        )
    threshold = float(args.threshold if args.threshold is not None else config.get("threshold", 0.5))
    input_size = int(config.get("input_size", [IMAGE_SIZE, IMAGE_SIZE])[0])
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    onnx_path = args.output_dir / "taste_classifier.onnx"

    wrapper = ExportWrapper(model).to(device).eval()
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        dynamo=False,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX exported: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MiB)")

    config["threshold"] = threshold
    config["onnx"] = {
        "path": onnx_path.name,
        "input_name": "input",
        "output_name": "logits",
        "opset": args.opset,
        "input_size": [input_size, input_size],
        "fixed_batch": 1,
    }
    config.setdefault("version", checkpoint.get("config", {}).get("version", "1.0.0"))
    config["export"] = {
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "opset": args.opset,
    }
    config_path = args.output_dir / "model_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    save_json(
        args.output_dir / "export_report.json",
        {
            "onnx": str(onnx_path),
            "bytes": onnx_path.stat().st_size,
            "input_size": [input_size, input_size],
            "threshold": threshold,
            "opset": args.opset,
        },
    )
    print(f"config: {config_path} (threshold={threshold:.6f})")
    print("Copy taste_classifier.onnx + model_config.json + inference.py to Ubuntu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
