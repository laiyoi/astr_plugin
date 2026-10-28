"""CPU-only DINOv2 taste classifier for Ubuntu / AstrBot.

This module intentionally imports only numpy, Pillow and onnxruntime.  It must
not import torch or any training code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


def _bicubic() -> int:
    return getattr(Image, "Resampling", Image).BICUBIC


class TasteClassifier:
    def __init__(
        self,
        onnx_path: str | Path,
        config_path: str | Path | None = None,
        *,
        providers: list[str] | None = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.config_path = Path(config_path) if config_path else self.onnx_path.with_name("model_config.json")
        if not self.onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"model_config.json not found: {self.config_path}")
        self.config: dict[str, Any] = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.input_size = int(self.config["input_size"][0])
        self.resize_size = int(self.config.get("resize_size", 256))
        self.mean = np.asarray(self.config["mean"], dtype=np.float32)
        self.std = np.asarray(self.config["std"], dtype=np.float32)
        self.class_names = list(self.config["class_names"])
        self.positive_index = int(self.config.get("positive_class_index", 1))
        self.threshold = float(self.config.get("threshold", 0.5))
        if len(self.class_names) not in (2, 3):
            raise ValueError("expected 2 or 3 class_names")
        if self.class_names[self.positive_index] != "collect":
            raise ValueError("positive_class_index must point to collect")
        if self.mean.shape != (3,) or self.std.shape != (3,):
            raise ValueError("mean/std must have 3 values")

        available = ort.get_available_providers()
        requested = providers or ["CPUExecutionProvider"]
        selected = [p for p in requested if p in available]
        if not selected:
            selected = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.onnx_path), providers=selected)
        self.input_name = self.config.get("onnx", {}).get("input_name", "input")
        self.output_name = self.config.get("onnx", {}).get("output_name", "logits")
        shape = self.session.get_inputs()[0].shape
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] != self.input_size:
            raise ValueError(f"ONNX input {shape} does not match config size {self.input_size}")
        self.version = str(self.config.get("version", "unknown"))

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("empty image")
        if width < height:
            new_width, new_height = self.resize_size, max(1, round(height * self.resize_size / width))
        else:
            new_width, new_height = max(1, round(width * self.resize_size / height)), self.resize_size
        image = image.resize((new_width, new_height), _bicubic())
        left = int(round((new_width - self.input_size) / 2.0))
        top = int(round((new_height - self.input_size) / 2.0))
        image = image.crop((left, top, left + self.input_size, top + self.input_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - self.mean) / self.std
        return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        tensor = self.preprocess(image)
        logits = np.asarray(self.session.run([self.output_name], {self.input_name: tensor})[0], dtype=np.float32)[0]
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp)
        positive_score = float(probs[self.positive_index])
        predicted_index = int(np.argmax(probs))
        label = self.class_names[predicted_index]
        return {
            "label": label,
            "collect": label == "collect",
            "score": positive_score,
            "collect_score": positive_score,
            "suspicious_score": float(probs[self.class_names.index("suspicious")]) if "suspicious" in self.class_names else 0.0,
            "threshold": self.threshold,
            "class_scores": {name: float(probs[i]) for i, name in enumerate(self.class_names)},
            "logits": [float(x) for x in logits],
            "version": self.version,
        }

    def predict_path(self, path: str | Path) -> dict[str, Any]:
        with Image.open(path) as image:
            return self.predict(image)


def decide_save(label: str, publisher_saved_count: int, policy: dict[str, Any] | None = None) -> str:
    """Return save/cache/skip using publisher history, never the model.

    suspicious:
      count < cache_below_saved_count -> cache
      count >= save_at_saved_count   -> save
      otherwise                      -> skip
    collect -> save, skip -> skip.
    """
    if label == "collect":
        return "save"
    if label == "skip":
        return "skip"
    policy = policy or {}
    save_at = int(policy.get("save_at_saved_count", 5))
    cache_below = int(policy.get("cache_below_saved_count", 3))
    if publisher_saved_count < cache_below:
        return "cache"
    if publisher_saved_count >= save_at:
        return "save"
    return "skip"


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU ONNX inference for the DINOv2 taste classifier")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--publisher-saved-count", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    classifier = TasteClassifier(args.onnx, args.config)
    result = classifier.predict_path(args.image)
    decision = decide_save(
        str(result["label"]),
        args.publisher_saved_count,
        classifier.config.get("suspicious_policy"),
    )
    result["publisher_saved_count"] = args.publisher_saved_count
    result["decision"] = decision
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            f"label={result['label']} decision={result['decision']} "
            f"collect={result['collect']} score={result['score']:.4f} threshold={result['threshold']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
