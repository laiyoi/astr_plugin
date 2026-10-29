"""Shared config/path helper for training-side scripts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

TRAINING_DIR = Path(__file__).resolve().parent


def resolve(value: str | Path | None, default: str | Path | None = None) -> Path | None:
    if value is None:
        value = default
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (TRAINING_DIR / path).resolve()


def load_config() -> dict:
    path = TRAINING_DIR / "config.json"
    if not path.is_file():
        raise SystemExit(f"config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def section(name: str) -> SimpleNamespace:
    return SimpleNamespace(**dict(load_config().get(name) or {}))


def checkpoint_path(epoch: int | None = None) -> Path:
    p = paths()
    if epoch is None:
        path = p.checkpoint
    else:
        path = p.checkpoint_dir / f"epoch_{int(epoch):04d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def paths() -> SimpleNamespace:
    cfg = load_config()
    train = dict(cfg.get("train") or {})
    run_dir = resolve(train.get("output_dir"), "runs/taste_dinov2_v1")
    assert run_dir is not None
    checkpoint_dir = Path(train.get("checkpoint_dir", "ckpt")).expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = run_dir / checkpoint_dir
    data_root = resolve(train.get("data_root"), "dataset")
    assert data_root is not None
    model_config = run_dir / "model_config.json"
    deploy_dir = run_dir / "deploy"
    return SimpleNamespace(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint=checkpoint_dir / "best.pt",
        last_checkpoint=checkpoint_dir / "last.pt",
        data_root=data_root,
        model_config=model_config,
        deploy_dir=deploy_dir,
        onnx=deploy_dir / "taste_classifier.onnx",
    )
