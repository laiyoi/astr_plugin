"""Stage a sanitized Hugging Face release folder from the Windows training run.

Copies only the artifacts listed in HUGGINGFACE_HANDOFF.md, rewrites local
absolute paths out of the JSON/CSV reports, and writes the model card.

Usage:  python stage_hf_release.py
Output: <training>/hf_release/
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parent
PLUGIN_ROOT = TRAINING.parent
RUN = TRAINING / "runs" / "taste_dinov2_v1"
DEPLOY = RUN / "deploy"
OUT = Path(os.environ.get("HF_RELEASE_DIR") or (TRAINING / "hf_release"))

DATASET_ROOT = str(TRAINING / "dataset")

MODEL_CARD = """---
license: other
library_name: pytorch
tags:
  - image-classification
  - dinov2
  - onnx
  - astrbot
---

# femboydetector (AstrBot Img Curator Taste DINOv2 v1)

The repo name is a joke; the model does not detect femboys, gender or identity.
It is a three-class "taste" classifier used by the AstrBot image-curator plugin to
decide whether a picture/video frame is worth collecting. It must not be used to
infer anything about real people.

```text
0 skip          do not collect
1 suspicious    resolved by the publisher's saved-image count, not by the model
2 collect       collect
```

Architecture: DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`) with a frozen
backbone and a linear probe head (LayerNorm + Dropout 0.2 + Linear(384, 3)).
Training used 224x224 inputs, batch 32, AdamW (head lr 1e-3), 20+ epochs on a
private dataset of collect / suspicious / skip favourites.

- Input: `[1, 3, 224, 224]` float32, ImageNet normalization
- Output: `logits [1, 3]`
- `class_names = ["skip", "suspicious", "collect"]`, `positive_class_index = 2`
- `collect` threshold: `0.1164180263876915` (calibrated on the validation split for 0.95 precision)

## Preprocessing (must match `inference.py`)

```text
RGB
-> resize shortest edge to 256, bicubic
-> center crop 224x224
-> /255
-> ImageNet mean [0.485, 0.456, 0.406]
-> std [0.229, 0.224, 0.225]
-> NCHW float32
```

## The `suspicious` class

The model only outputs probabilities. Turning `suspicious` into a save/no-save
decision is a product policy applied outside the model, based on how many images
the publisher already has saved:

```text
publisher_saved_count < 3   -> cache (pending)
3 <= count < 5              -> skip
count >= 5                  -> save
```

Policy keys: `suspicious_policy.save_at_saved_count` (5) and
`suspicious_policy.cache_below_saved_count` (3) in `model_config.json`.

## Evaluation (private test split, n=76)

`collect` decision at the calibrated threshold:

| precision | recall | F1 | accuracy |
|---|---|---|---|
| 0.943 | 0.985 | 0.964 | 0.934 |

Three-class accuracy: 0.816 (macro F1 0.648); the `suspicious` class is tiny
(n=2) so its per-class numbers are not meaningful.

PyTorch vs ONNX parity (`onnx_parity.json`): `allclose = true`, max abs diff
`3.2e-6` over 8 samples.

## Files

- `best.pt` - PyTorch checkpoint (state dict, class names, training config, metrics)
- `taste_classifier.onnx` - fixed-shape ONNX (opset 17) for CPU inference
- `model_config.json` - preprocessing, class order, threshold, suspicious policy
- `inference.py` - onnxruntime CPU inference (`TasteClassifier`) and `decide_save()`
- `threshold.json`, `threshold_epoch0020.json` - threshold calibration sweeps
- `evaluation_test.json`, `evaluation_test.csv` - test-split metrics
- `onnx_parity.json` - PyTorch vs ONNX parity check

## Usage

```python
from inference import TasteClassifier, decide_save

clf = TasteClassifier("taste_classifier.onnx", "model_config.json")
result = clf.predict(pil_image)          # label, collect_score, class_scores, logits
decision = decide_save(
    result["label"],
    publisher_saved_count,
    clf.config.get("suspicious_policy"),
)                                        # "save" | "cache" | "skip"
```

Runtime dependencies: `onnxruntime`, `numpy`, `Pillow` only - no torch needed.

## Privacy

Released without any training images, dataset files, QQ numbers, group IDs or
nicknames. Paths inside the metrics files were rewritten to dataset-relative
form before publishing.
"""


def sanitize_text(text: str) -> str:
    return (
        text.replace(DATASET_ROOT + "\\\\", "")
        .replace(DATASET_ROOT + "\\", "")
        .replace(str(RUN) + "\\\\", "")
        .replace(str(RUN) + "\\", "")
        .replace(str(TRAINING) + "\\\\", "")
        .replace(str(TRAINING) + "\\", "")
        .replace("K:\\astr_plugin\\", "")
    )


def copy_sanitized_json(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))

    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str):
            return sanitize_text(node).replace("\\", "/")
        return node

    dst.write_text(json.dumps(walk(data), ensure_ascii=False, indent=2), encoding="utf-8")


def copy_sanitized_csv(src: Path, dst: Path) -> None:
    # The training pipeline writes this CSV with a UTF-8 BOM; utf-8-sig strips it
    # so the "path" header check below actually matches.
    with src.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if rows:
        header, body = rows[0], rows[1:]
        if header and header[0] == "path":
            for row in body:
                if row:
                    row[0] = sanitize_text(row[0]).replace("\\", "/")
            rows = [header] + body
    with dst.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)


def main() -> int:
    required = [
        RUN / "ckpt" / "best.pt",
        DEPLOY / "taste_classifier.onnx",
        RUN / "model_config.json",
        PLUGIN_ROOT / "astrbot_plugin_img_curator" / "inference.py",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        print("missing required artifacts:")
        for m in missing:
            print("  " + m)
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    shutil.copy2(RUN / "ckpt" / "best.pt", OUT / "best.pt")
    shutil.copy2(DEPLOY / "taste_classifier.onnx", OUT / "taste_classifier.onnx")
    copy_sanitized_json(RUN / "model_config.json", OUT / "model_config.json")
    shutil.copy2(PLUGIN_ROOT / "astrbot_plugin_img_curator" / "inference.py", OUT / "inference.py")
    (OUT / "README.md").write_text(MODEL_CARD, encoding="utf-8")

    for name in ("threshold.json", "threshold_epoch0020.json", "evaluation_test.json"):
        src = RUN / name
        if src.is_file():
            copy_sanitized_json(src, OUT / name)

    src = RUN / "evaluation_test.csv"
    if src.is_file():
        copy_sanitized_csv(src, OUT / "evaluation_test.csv")

    parity = DEPLOY / "onnx_parity.json"
    if parity.is_file():
        data = json.loads(parity.read_text(encoding="utf-8"))
        data["onnx"] = "taste_classifier.onnx"
        data["checkpoint"] = "best.pt"
        for i, row in enumerate(data.get("rows") or []):
            data["rows"][i] = {
                "sample": i,
                "max_abs_diff": row.get("max_abs_diff"),
                "allclose": row.get("allclose"),
            }
        (OUT / "onnx_parity.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    for path in sorted(OUT.iterdir()):
        print(f"{path.name:28s} {path.stat().st_size / 1024 / 1024:8.2f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
