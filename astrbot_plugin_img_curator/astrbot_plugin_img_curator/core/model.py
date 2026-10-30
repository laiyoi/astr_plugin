"""Local DINOv2 femboy detector ONNX backend.

The model is published at ``laiyoi/femboydetector``. It is a three-class
classifier (skip / suspicious / collect); no CLIP prompts or text encoder.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import uuid
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
        self.last_check = ""
        self.last_update = ""
        self.update_error = ""
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

    @property
    def _meta_path(self) -> Path:
        return self.cfg.model_dir / ".model_meta.json"

    def _token(self) -> str | None:
        return os.environ.get("HF_TOKEN") or None

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
                revision=self.cfg.hf_revision,
                local_dir=str(self.cfg.model_dir),
                token=self._token(),
            )

    def _remote_meta(self) -> dict[str, dict[str, Any]]:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        result = {}
        for filename in (self.cfg.hf_model_filename, self.cfg.hf_config_filename):
            url = hf_hub_url(
                repo_id=self.cfg.hf_repo_id, filename=filename, revision=self.cfg.hf_revision
            )
            meta = get_hf_file_metadata(url, token=self._token())
            result[filename] = {"etag": meta.etag, "size": int(meta.size or 0)}
        return result

    def _read_meta(self) -> dict[str, Any]:
        try:
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}

    async def update(self, force: bool = False) -> bool:
        """Check/download a new HF revision and hot-swap the classifier."""
        async with self._lock:
            self.last_check = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            try:
                remote = await asyncio.to_thread(self._remote_meta)
            except Exception as exc:  # noqa: BLE001
                self.update_error = str(exc)
                raise ModelUnavailable(f"检查模型更新失败: {exc}") from exc
            local = self._read_meta()
            files_ok = self.model_path.is_file() and self.config_path.is_file()
            if not force and files_ok and local.get("remote") == remote:
                self.update_error = ""
                return False

            staging = self.cfg.model_dir / f".staging-{uuid.uuid4().hex}"
            staging.mkdir(parents=True, exist_ok=True)
            try:
                from huggingface_hub import hf_hub_download

                for filename in (self.cfg.hf_model_filename, self.cfg.hf_config_filename):
                    hf_hub_download(
                        repo_id=self.cfg.hf_repo_id,
                        filename=filename,
                        revision=self.cfg.hf_revision,
                        local_dir=str(staging),
                        force_download=True,
                        token=self._token(),
                    )
                staged_model = staging / self.cfg.hf_model_filename
                staged_config = staging / self.cfg.hf_config_filename
                classifier = await asyncio.to_thread(
                    TasteClassifier,
                    staged_model,
                    staged_config,
                    providers=["CPUExecutionProvider"],
                )
                for filename in (self.cfg.hf_model_filename, self.cfg.hf_config_filename):
                    os.replace(staging / filename, self.cfg.model_dir / filename)
                self.classifier = classifier
                self.load_ms = 0.0
                self.error = ""
                self.update_error = ""
                self.last_update = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                self._meta_path.write_text(
                    json.dumps({"remote": remote, "updated_at": self.last_update}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self.logger.info("模型已更新: %s | 类别 %s", self.display, classifier.class_names)
                return True
            except Exception as exc:  # noqa: BLE001
                self.update_error = str(exc)
                raise ModelUnavailable(f"模型更新失败: {exc}") from exc
            finally:
                shutil.rmtree(staging, ignore_errors=True)

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
            "revision": self.cfg.hf_revision,
            "auto_update": bool(self.cfg.auto_update_model),
            "last_check": self.last_check,
            "last_update": self.last_update,
            "update_error": self.update_error,
            "class_names": list(self.classifier.class_names) if self.classifier else [],
        }
