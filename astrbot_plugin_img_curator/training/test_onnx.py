"""Verify PyTorch vs ONNX parity on Windows, using the exact deployment preprocessing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config_utils  # noqa: E402
from dataset import IMAGE_SUFFIXES  # noqa: E402
from model import eval_transform, load_model_from_checkpoint  # noqa: E402
from utils import load_json, resolve_device, save_json  # noqa: E402


def collect_images(data_root: Path, num_samples: int) -> list[Path]:
    images: list[Path] = []
    for split in ("test", "val", "train"):
        for class_name in ("collect", "suspicious", "skip"):
            directory = data_root / split / class_name
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    images.append(path)
                    if len(images) >= num_samples:
                        return images
    return images


def preprocess(path: Path, input_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = eval_transform(image_size=input_size)(image)
    return tensor.unsqueeze(0).numpy().astype(np.float32)


def main() -> int:
    cfg = config_utils.section("test_onnx")
    p = config_utils.paths()
    device = resolve_device(str(cfg.device))
    config = load_json(p.model_config)
    input_size = int(config.get("input_size", [224, 224])[0])
    model, _ = load_model_from_checkpoint(p.checkpoint, device=device)
    session = ort.InferenceSession(str(p.onnx), providers=["CPUExecutionProvider"])
    input_name = config.get("onnx", {}).get("input_name", "input")
    output_name = config.get("onnx", {}).get("output_name", "logits")

    images = collect_images(p.data_root, int(cfg.num_samples))
    if not images:
        raise FileNotFoundError("no test images found")
    rows = []
    all_ok = True
    for path in images:
        tensor = preprocess(path, input_size)
        if tensor.shape[1:] != (3, input_size, input_size):
            raise ValueError(f"unexpected tensor shape {tensor.shape}")
        with torch.no_grad():
            torch_logits = model(torch.from_numpy(tensor).to(device)).float().cpu().numpy()
        onnx_logits = session.run([output_name], {input_name: tensor})[0]
        diff = np.abs(torch_logits - onnx_logits)
        max_diff = float(diff.max())
        ok = bool(np.allclose(torch_logits, onnx_logits, atol=1e-4, rtol=1e-4))
        all_ok = all_ok and ok
        rows.append({"path": str(path), "max_abs_diff": max_diff, "allclose": ok})
        print(f"{path.name}: max_abs_diff={max_diff:.8f} allclose={ok}")

    report = {
        "onnx": str(p.onnx),
        "checkpoint": str(p.checkpoint),
        "input_size": input_size,
        "samples": len(rows),
        "allclose": all_ok,
        "max_abs_diff": max((r["max_abs_diff"] for r in rows), default=0.0),
        "rows": rows,
    }
    output = p.onnx.parent / "onnx_parity.json"
    save_json(output, report)
    print(f"max_abs_diff={report['max_abs_diff']:.8f} allclose={all_ok}")
    print(f"wrote {output}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
