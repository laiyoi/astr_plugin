"""Local DINOv2 femboy detector ONNX backend.

The model is published at ``laiyoi/femboydetector``. It is a three-class
classifier (skip / suspicious / collect); no CLIP prompts or text encoder.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PluginConfig

try:
    from ..inference import TasteClassifier
except ImportError:  # direct/core imports in tests
    from inference import TasteClassifier  # type: ignore


class ModelUnavailable(Exception):
    pass


@dataclass
class TasteResult:
    label: str
    score: float
    class_scores: dict[str, float]
    inference_ms: float
    raw: dict[str, Any] = field(default_factory=dict)


class TasteOnnxModel:
    def __init__(self, cfg: PluginConfig, logger):
        self.cfg = cfg
        self.logger = logger
        self.model_path = cfg.model_dir / cfg.hf_model_filename
        self.config_path = cfg.model_dir / cfg.hf_config_filename
        self.classifier: TasteClassifier | None = None
        self.error = ""
        self.load_ms = 0.0
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self.classifier is not None

    @property
    def key(self) -> str:
        return f"{self.cfg.hf_repo_id}/{self.cfg.hf_model_filename}"

    @property
    def display(self) -> str:
        return f"DINOv2 femboydetector · {self.cfg.hf_model_filename}"

    def _download(self) -> None:
        from huggingface_hub import hf_hub_download

        self.cfg.model_dir.mkdir(parents=True, exist_ok=True)
        for filename in (self.cfg.hf_model_filename, self.cfg.hf_config_filename):
            target = self.cfg.model_dir / filename
            if target.is_file():
                continue
            hf_hub_download(
                repo_id=self.cfg.hf_repo_id,
                filename=filename,
                local_dir=str(self.cfg.model_dir),
            )

    async def ensure_ready(self) -> None:
        if self.classifier is not None:
            return
        async with self._lock:
            if self.classifier is not None:
                return
            self.error = ""
            try:
                await asyncio.to_thread(self._download)
                started = time.perf_counter()
                self.classifier = await asyncio.to_thread(
                    TasteClassifier,
                    self.model_path,
                    self.config_path,
                    providers=["CPUExecutionProvider"],
                )
                self.load_ms = (time.perf_counter() - started) * 1000
                self.logger.info(
                    "模型就绪: %s | 加载 %.0f ms | 类别 %s",
                    self.display,
                    self.load_ms,
                    self.classifier.class_names,
                )
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
                self.classifier = None
                raise ModelUnavailable(str(exc)) from exc

    def classify_sync(self, path: Path) -> TasteResult:
        if self.classifier is None:
            raise ModelUnavailable("model is not loaded")
        started = time.perf_counter()
        payload = self.classifier.predict_path(path)
        elapsed = (time.perf_counter() - started) * 1000
        return TasteResult(
            label=str(payload["label"]),
            score=float(payload.get("collect_score", payload.get("score", 0.0))),
            class_scores={k: float(v) for k, v in payload.get("class_scores", {}).items()},
            inference_ms=elapsed,
            raw=payload,
        )

    async def classify(self, path: Path) -> TasteResult:
        await self.ensure_ready()
        return await asyncio.to_thread(self.classify_sync, path)

    def status(self) -> dict[str, Any]:
        return {
            "model": self.display,
            "state": "ready" if self.ready else "not_loaded",
            "weights_mb": round(
                (self.model_path.stat().st_size / 1024 / 1024) if self.model_path.is_file() else 0.0,
                1,
            ),
            "dir": str(self.cfg.model_dir),
            "files_present": sum(
                1 for p in (self.model_path, self.config_path) if p.is_file()
            ),
            "files_expected": 2,
            "load_ms": round(self.load_ms, 1),
            "error": self.error,
            "repo": self.cfg.hf_repo_id,
            "class_names": list(self.classifier.class_names) if self.classifier else [],
        }
