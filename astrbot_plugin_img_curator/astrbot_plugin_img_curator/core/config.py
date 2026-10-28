"""Configuration parsing with safe defaults.

No group whitelist: every group message the adapter delivers is eligible.
No API keys, no cloud endpoints, no GPU settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _int(raw: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(raw.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _float(raw: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(raw.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _bool(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _path(raw: dict[str, Any], key: str, default_rel: str, data_dir: Path) -> Path:
    value = str(raw.get(key, "") or "").strip()
    if not value:
        value = default_rel
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        path = data_dir / path
    return path.resolve()


@dataclass(frozen=True)
class PluginConfig:
    data_dir: Path
    save_dir: Path
    pending_dir: Path
    tmp_dir: Path
    forward_dir: Path
    model_dir: Path
    csv_path: Path
    db_path: Path
    enabled: bool
    recognition_enabled: bool
    save_pending: bool
    strip_metadata: bool
    dedupe: bool
    max_file_bytes: int
    max_video_bytes: int
    video_max_seconds: int
    video_max_frames: int
    video_fps: float
    video_extract_timeout_sec: int
    max_image_pixels: int
    min_image_side: int
    queue_size: int
    rate_limit_count: int
    rate_limit_window_sec: int
    max_pending_files: int
    forward_max_minutes: int
    forward_max_images: int
    forward_max_nodes: int
    forward_max_depth: int
    cpu_threads: int
    model_name: str
    model_variant: str
    auto_download_model: bool
    download_timeout_sec: int
    image_download_timeout_sec: int
    high_threshold: float
    pending_threshold: float
    min_positive_score: float
    min_pending_score: float
    #debug
    debug_dir: Path = Path("/astr_plugin/debug")

    @classmethod
    def load(cls, raw: dict[str, Any] | None, data_dir: Path) -> "PluginConfig":
        raw = dict(raw or {})
        data_dir = data_dir.resolve()
        variant = str(raw.get("model_variant", "fp32") or "fp32").strip().lower()
        if variant not in {"fp32", "int8"}:
            variant = "fp32"
        high = _float(raw, "high_threshold", 0.011, 0.0, 1.0)
        pending = _float(raw, "pending_threshold", 0.005, 0.0, 1.0)
        # Keep the bands ordered even if a user enters them backwards.
        if pending > high:
            pending = high
        min_pos = _float(raw, "min_positive_score", 0.240, 0.0, 1.0)
        min_pending = _float(raw, "min_pending_score", 0.210, 0.0, 1.0)
        if min_pending > min_pos:
            min_pending = min_pos

        return cls(
            data_dir=data_dir,
            save_dir=_path(raw, "save_dir", "images", data_dir),
            pending_dir=_path(raw, "pending_dir", "pending", data_dir),
            tmp_dir=_path(raw, "tmp_dir", "tmp", data_dir),
            forward_dir=_path(raw, "forward_dir", "training_data/forwarded", data_dir),
            model_dir=_path(raw, "model_dir", "models", data_dir),
            csv_path=_path(raw, "csv_path", "metadata.csv", data_dir),
            db_path=_path(raw, "db_path", "state.sqlite3", data_dir),
            enabled=_bool(raw, "enabled", True),
            recognition_enabled=_bool(raw, "recognition_enabled", False),
            save_pending=_bool(raw, "save_pending", True),
            strip_metadata=_bool(raw, "strip_metadata", True),
            dedupe=_bool(raw, "dedupe", True),
            max_file_bytes=_int(raw, "max_file_mb", 12, 1, 100) * 1024 * 1024,
            max_video_bytes=_int(raw, "max_video_mb", 50, 1, 500) * 1024 * 1024,
            video_max_seconds=_int(raw, "video_max_seconds", 15, 1, 300),
            video_max_frames=_int(raw, "video_max_frames", 8, 1, 32),
            video_fps=_float(raw, "video_fps", 1.0, 0.1, 10.0),
            video_extract_timeout_sec=_int(raw, "video_extract_timeout_sec", 60, 5, 600),
            max_image_pixels=_int(raw, "max_image_pixels_million", 40, 1, 200) * 1_000_000,
            min_image_side=_int(raw, "min_image_side", 64, 1, 4096),
            queue_size=_int(raw, "queue_size", 64, 1, 4096),
            rate_limit_count=_int(raw, "rate_limit_count", 20, 1, 10_000),
            rate_limit_window_sec=_int(raw, "rate_limit_window_sec", 60, 1, 86_400),
            max_pending_files=_int(raw, "max_pending_files", 500, 0, 100_000),
            forward_max_minutes=_int(raw, "forward_max_minutes", 60, 1, 1440),
            forward_max_images=_int(raw, "forward_max_images", 500, 1, 10_000),
            forward_max_nodes=_int(raw, "forward_max_nodes", 2000, 1, 100_000),
            forward_max_depth=_int(raw, "forward_max_depth", 8, 1, 32),
            cpu_threads=_int(raw, "cpu_threads", min(4, os.cpu_count() or 4), 1, 64),
            model_name=str(raw.get("model_name", "CLIP-ViT-B-32") or "CLIP-ViT-B-32"),
            model_variant=variant,
            auto_download_model=_bool(raw, "auto_download_model", False),
            download_timeout_sec=_int(raw, "download_timeout_sec", 600, 30, 3600),
            image_download_timeout_sec=_int(raw, "image_download_timeout_sec", 30, 5, 300),
            high_threshold=high,
            pending_threshold=pending,
            min_positive_score=min_pos,
            min_pending_score=min_pending,
        )

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.save_dir, self.pending_dir, self.tmp_dir, self.forward_dir, self.model_dir):
            path.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(path, 0o700)
            except OSError:
                pass
