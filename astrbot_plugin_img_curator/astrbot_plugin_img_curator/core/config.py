"""Plugin configuration for the DINOv2 femboy detector collector."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _bool(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _int(raw: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _float(raw: dict[str, Any], key: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(raw.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _path(raw: dict[str, Any], key: str, default: Path) -> Path:
    value = str(raw.get(key, "") or "").strip()
    return Path(os.path.expanduser(value)).resolve() if value else default


@dataclass
class PluginConfig:
    enabled: bool
    recognition_enabled: bool
    save_pending: bool
    dedupe: bool
    strip_metadata: bool
    max_file_mb: int
    max_video_mb: int
    video_max_seconds: int
    video_max_frames: int
    video_fps: float
    video_extract_timeout_sec: int
    max_image_pixels: int
    min_image_side: int
    image_download_timeout_sec: int
    max_pending_files: int
    queue_size: int
    rate_limit_count: int
    rate_limit_window_sec: int
    cpu_threads: int
    hf_repo_id: str
    hf_model_filename: str
    hf_config_filename: str
    download_timeout_sec: int
    suspicious_save_at: int
    suspicious_cache_below: int
    data_dir: Path
    save_dir: Path
    pending_dir: Path
    tmp_dir: Path
    model_dir: Path
    db_path: Path
    csv_path: Path

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    @property
    def max_video_bytes(self) -> int:
        return self.max_video_mb * 1024 * 1024

    @classmethod
    def load(cls, raw: dict[str, Any] | None, data_dir: Path) -> "PluginConfig":
        raw = dict(raw or {})
        data_dir = data_dir.resolve()
        return cls(
            enabled=_bool(raw, "enabled", True),
            recognition_enabled=_bool(raw, "recognition_enabled", True),
            save_pending=_bool(raw, "save_pending", True),
            dedupe=_bool(raw, "dedupe", True),
            strip_metadata=_bool(raw, "strip_metadata", True),
            max_file_mb=_int(raw, "max_file_mb", 12, 1, 200),
            max_video_mb=_int(raw, "max_video_mb", 50, 1, 500),
            video_max_seconds=_int(raw, "video_max_seconds", 15, 1, 300),
            video_max_frames=_int(raw, "video_max_frames", 8, 1, 32),
            video_fps=_float(raw, "video_fps", 1.0, 0.1, 10.0),
            video_extract_timeout_sec=_int(raw, "video_extract_timeout_sec", 60, 1, 600),
            max_image_pixels=_int(raw, "max_image_pixels", 40_000_000, 1_000_000, 100_000_000),
            min_image_side=_int(raw, "min_image_side", 64, 1, 4096),
            image_download_timeout_sec=_int(raw, "image_download_timeout_sec", 30, 1, 300),
            max_pending_files=_int(raw, "max_pending_files", 500, 0, 100_000),
            queue_size=_int(raw, "queue_size", 64, 1, 4096),
            rate_limit_count=_int(raw, "rate_limit_count", 20, 1, 10_000),
            rate_limit_window_sec=_int(raw, "rate_limit_window_sec", 60, 1, 3600),
            cpu_threads=_int(raw, "cpu_threads", min(4, os.cpu_count() or 4), 1, 64),
            hf_repo_id=str(raw.get("hf_repo_id", "laiyoi/femboydetector") or "laiyoi/femboydetector").strip(),
            hf_model_filename=str(raw.get("hf_model_filename", "taste_classifier.onnx") or "taste_classifier.onnx").strip(),
            hf_config_filename=str(raw.get("hf_config_filename", "model_config.json") or "model_config.json").strip(),
            download_timeout_sec=_int(raw, "download_timeout_sec", 600, 10, 3600),
            suspicious_save_at=_int(raw, "suspicious_save_at", 5, 1, 1000),
            suspicious_cache_below=_int(raw, "suspicious_cache_below", 3, 0, 1000),
            data_dir=data_dir,
            save_dir=_path(raw, "save_dir", data_dir / "images"),
            pending_dir=_path(raw, "pending_dir", data_dir / "pending"),
            tmp_dir=_path(raw, "tmp_dir", data_dir / "tmp"),
            model_dir=_path(raw, "model_dir", data_dir / "models" / "femboydetector"),
            db_path=_path(raw, "db_path", data_dir / "state.sqlite3"),
            csv_path=_path(raw, "csv_path", data_dir / "metadata.csv"),
        )

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.save_dir, self.pending_dir, self.tmp_dir, self.model_dir):
            path.mkdir(parents=True, exist_ok=True)
