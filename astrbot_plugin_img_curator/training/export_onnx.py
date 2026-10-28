"""Export the DINOv2 backbone + three-class head to a fixed-shape ONNX model.

All paths/settings come from training/config.json; run with no arguments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config_utils  # noqa: E402
from model import (  # noqa: E402
    ExportWrapper,
    IMAGE_SIZE,
    load_model_from_checkpoint,
    write_model_config,
)
from utils import load_json, resolve_device, save_json  # noqa: E402


def main() -> int:
    cfg = config_utils.section("export")
    p = config_utils.paths()
    device = resolve_device(str(cfg.device))
    p.deploy_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model_from_checkpoint(p.checkpoint, device=device)
    model.eval()

    config = load_json(p.model_config) if p.model_config.is_file() else None
    if config is None:
        config = write_model_config(
            output_path=p.model_config,
            checkpoint=checkpoint,
            threshold=0.5,
        )
    threshold = float(config.get("threshold", 0.5))
    input_size = int(config.get("input_size", [IMAGE_SIZE, IMAGE_SIZE])[0])
    dummy = torch.randn(1, 3, input_size, input_size, device=device)

    wrapper = ExportWrapper(model).to(device).eval()
    torch.onnx.export(
        wrapper,
        dummy,
        str(p.onnx),
        export_params=True,
        opset_version=int(cfg.opset),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        dynamo=False,
    )
    onnx_model = onnx.load(str(p.onnx))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX exported: {p.onnx} ({p.onnx.stat().st_size / 1024 / 1024:.1f} MiB)")

    config["threshold"] = threshold
    config["onnx"] = {
        "path": p.onnx.name,
        "input_name": "input",
        "output_name": "logits",
        "opset": int(cfg.opset),
        "input_size": [input_size, input_size],
        "fixed_batch": 1,
    }
    config.setdefault("version", checkpoint.get("config", {}).get("version", "1.0.0"))
    config["export"] = {
        "checkpoint": str(p.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "opset": int(cfg.opset),
    }
    p.model_config.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    save_json(
        p.deploy_dir / "export_report.json",
        {
            "onnx": str(p.onnx),
            "bytes": p.onnx.stat().st_size,
            "input_size": [input_size, input_size],
            "threshold": threshold,
            "opset": int(cfg.opset),
        },
    )
    print(f"config: {p.model_config} (threshold={threshold:.6f})")
    print("Copy taste_classifier.onnx + model_config.json + inference.py to Ubuntu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
