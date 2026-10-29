#!/usr/bin/env python3
"""一键发布 / 更新 Hugging Face 模型仓库。

把原来单独的 stage_hf_release.py 合并进来了：生成清洗过的发布目录 + 上传 + 远程校验，
一条命令搞定。仓库可见性**完全不动**，脚本从不调用任何修改可见性的接口。

用法:
    python publish_hf.py                 # 生成发布目录并上传（最常用）
    python publish_hf.py --dry-run       # 只生成发布目录，不上传
    python publish_hf.py --readme-only   # 只更新模型卡
    python publish_hf.py --export        # 先跑 export_onnx.py + test_onnx.py，再发布
    python publish_hf.py --skip-stage    # 直接复用已有发布目录上传

发布前的硬性检查：
    - deploy/taste_classifier.onnx 存在；
    - deploy/onnx_parity.json 的 allclose 为 true（没有这个文件会警告并中止，除非 --force）；
    - ONNX 不旧于 ckpt/best.pt（旧了说明忘了重新导出，中止，除非 --force）。

上传内容固定白名单，训练图片、dataset、QQ/群/昵称、内部交接文档都不会进去。
"""

from __future__ import annotations

import os

# 必须在 import huggingface_hub 之前设置：本机 Xet 后端会报 I/O error: 拒绝访问
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parent
PLUGIN_ROOT = TRAINING.parent
RUN = TRAINING / "runs" / "taste_dinov2_v1"
CKPT = RUN / "ckpt" / "best.pt"
DEPLOY = RUN / "deploy"
ONNX = DEPLOY / "taste_classifier.onnx"
PARITY = DEPLOY / "onnx_parity.json"
MODEL_CONFIG = RUN / "model_config.json"
INFERENCE = PLUGIN_ROOT / "astrbot_plugin_img_curator" / "inference.py"

DEFAULT_REPO = os.environ.get("HF_REPO_ID", "laiyoi/femboydetector")
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
- `collect` threshold: from `model_config.json` (calibrated on validation for 0.95 precision)

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

Metrics for the shipped checkpoint are in `evaluation_test.json` (collect
precision/recall/F1 at the calibrated threshold, plus per-class numbers).

## Files

- `best.pt` - PyTorch checkpoint (state dict, class names, training config, metrics)
- `taste_classifier.onnx` - fixed-shape ONNX (opset 17) for CPU inference
- `model_config.json` - preprocessing, class order, threshold, suspicious policy
- `inference.py` - onnxruntime CPU inference (`TasteClassifier`) and `decide_save()`
- `threshold*.json`, `evaluation_test.*` - calibration sweeps and test-split metrics
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
nicknames. Paths inside the metrics files are rewritten to dataset-relative form
before publishing.
"""


# ---------------------------------------------------------------- staging


def sanitize_text(text: str) -> str:
    for prefix in (DATASET_ROOT, str(RUN), str(TRAINING)):
        text = text.replace(prefix + "\\\\", "").replace(prefix + "\\", "")
    return text.replace("K:\\astr_plugin\\", "")


def copy_sanitized_json(src: Path, dst: Path) -> None:
    def walk(node):
        if isinstance(node, dict):
            return {key: walk(value) for key, value in node.items()}
        if isinstance(node, list):
            return [walk(value) for value in node]
        if isinstance(node, str):
            return sanitize_text(node).replace("\\", "/")
        return node

    data = json.loads(src.read_text(encoding="utf-8"))
    dst.write_text(json.dumps(walk(data), ensure_ascii=False, indent=2), encoding="utf-8")


def copy_sanitized_csv(src: Path, dst: Path) -> None:
    # 训练产出的 CSV 带 UTF-8 BOM，用 utf-8-sig 读，否则 "path" 表头匹配不上
    with src.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if rows and rows[0] and rows[0][0] == "path":
        for row in rows[1:]:
            if row:
                row[0] = sanitize_text(row[0]).replace("\\", "/")
    with dst.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)


def check_artifacts(force: bool) -> bool:
    problems: list[str] = []
    for path in (CKPT, ONNX, MODEL_CONFIG, INFERENCE):
        if not path.is_file():
            problems.append(f"缺少文件: {path}")

    if PARITY.is_file():
        try:
            parity = json.loads(PARITY.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            parity = {}
        if not parity.get("allclose"):
            problems.append(f"onnx_parity.json 的 allclose 不是 true: {PARITY}")
    else:
        problems.append(f"缺少 {PARITY}（先跑 test_onnx.py）")

    if ONNX.is_file() and CKPT.is_file() and ONNX.stat().st_mtime < CKPT.stat().st_mtime:
        problems.append(
            "ONNX 比 best.pt 旧，说明换了权重但没重新导出，先跑 export_onnx.py"
        )

    if not problems:
        return True
    for problem in problems:
        print(f"[检查失败] {problem}")
    if force:
        print("已指定 --force，继续发布")
        return True
    return False


def stage(out_dir: Path) -> dict[str, Path]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copy2(CKPT, out_dir / "best.pt")
    shutil.copy2(ONNX, out_dir / "taste_classifier.onnx")
    copy_sanitized_json(MODEL_CONFIG, out_dir / "model_config.json")
    shutil.copy2(INFERENCE, out_dir / "inference.py")
    (out_dir / "README.md").write_text(MODEL_CARD, encoding="utf-8")

    for name in ("threshold.json", "threshold_epoch0020.json", "evaluation_test.json"):
        src = RUN / name
        if src.is_file():
            copy_sanitized_json(src, out_dir / name)

    if (RUN / "evaluation_test.csv").is_file():
        copy_sanitized_csv(RUN / "evaluation_test.csv", out_dir / "evaluation_test.csv")

    if PARITY.is_file():
        data = json.loads(PARITY.read_text(encoding="utf-8"))
        data["onnx"] = "taste_classifier.onnx"
        data["checkpoint"] = "best.pt"
        data["rows"] = [
            {"sample": index, "max_abs_diff": row.get("max_abs_diff"), "allclose": row.get("allclose")}
            for index, row in enumerate(data.get("rows") or [])
        ]
        (out_dir / "onnx_parity.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    files = {path.name: path for path in sorted(out_dir.iterdir()) if path.is_file()}
    print(f"发布目录: {out_dir}")
    for name, path in files.items():
        print(f"  {name:24s} {path.stat().st_size / 1024 / 1024:8.2f} MiB")

    leaks = []
    for name, path in files.items():
        if path.suffix.lower() in {".json", ".csv", ".md", ".py"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "astr_plugin" in text or "K:\\" in text:
                leaks.append(name)
    if leaks:
        raise SystemExit(f"发布目录仍有本机绝对路径，拒绝上传: {', '.join(leaks)}")
    print("路径清洗检查: 通过")
    return files


# ---------------------------------------------------------------- upload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upload(repo_id: str, folder: Path, message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    info = None
    try:
        info = api.repo_info(repo_id, repo_type="model")
    except Exception:
        info = None

    if info is None:
        print(f"仓库不存在，新建公开仓库: {repo_id}")
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        commit_message=message,
    )
    print(f"上传完成: https://huggingface.co/{repo_id}")


def upload_readme(repo_id: str, folder: Path, message: str) -> None:
    from huggingface_hub import HfApi

    readme = folder / "README.md"
    if not readme.is_file():
        raise SystemExit(f"找不到模型卡: {readme}（先跑一次不带 --skip-stage 的发布）")
    HfApi().upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=message,
    )
    print(f"模型卡已更新: https://huggingface.co/{repo_id}")


def verify(repo_id: str, folder: Path) -> bool:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(repo_id, repo_type="model")
    # 只读展示，脚本不修改可见性
    print(f"仓库可见性: {'private' if info.private else 'public'}（脚本不会修改）")

    remote_names = api.list_repo_files(repo_id, repo_type="model")
    entries = api.get_paths_info(repo_id, remote_names, repo_type="model") if remote_names else []
    remote = {entry.path: entry for entry in entries}

    ok = True
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        entry = remote.get(path.name)
        if entry is None:
            print(f"  缺失 {path.name}")
            ok = False
        elif entry.size != path.stat().st_size:
            print(f"  大小不一致 {path.name}: 本地 {path.stat().st_size} / 远程 {entry.size}")
            ok = False
        else:
            print(f"  ok {path.name} ({entry.size} bytes)")
    print("远程校验: 通过" if ok else "远程校验: 失败")
    return ok


# ---------------------------------------------------------------- main


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=TRAINING)
    if result.returncode != 0:
        raise SystemExit(f"命令失败: {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="一键生成清洗后的发布目录并上传到 Hugging Face（不改仓库可见性）。"
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"模型仓库 ID，默认 {DEFAULT_REPO}")
    parser.add_argument("--out", type=Path, default=TRAINING / "hf_release", help="发布目录，默认 training/hf_release")
    parser.add_argument("--dry-run", action="store_true", help="只生成发布目录，不上传")
    parser.add_argument("--skip-stage", action="store_true", help="复用已有发布目录，只上传")
    parser.add_argument("--readme-only", action="store_true", help="只上传模型卡")
    parser.add_argument("--export", action="store_true", help="上传前先跑 export_onnx.py 和 test_onnx.py")
    parser.add_argument("--force", action="store_true", help="忽略产物检查失败，强制发布")
    parser.add_argument("--message", default=None, help="commit message")
    args = parser.parse_args()

    out_dir = args.out.expanduser().resolve()

    if not args.skip_stage and not args.readme_only:
        check_artifacts(args.force) or sys.exit(1)
        if args.export:
            run([sys.executable, "export_onnx.py"])
            run([sys.executable, "test_onnx.py"])
            check_artifacts(args.force) or sys.exit(1)
        stage(out_dir)

    if args.dry_run:
        print("--dry-run: 不上传")
        return 0

    message = args.message or "Update model artifacts"
    if args.readme_only:
        upload_readme(args.repo, out_dir, message)
        return 0

    upload(args.repo, out_dir, message)
    return 0 if verify(args.repo, out_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
