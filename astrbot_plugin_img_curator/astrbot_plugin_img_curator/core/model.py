"""Resident ONNX CLIP encoder.

Default: OpenCLIP ViT-B/32 exported to ONNX, dynamically quantized to int8
(153 MB of weights).  The fp32 variant and Apple's MobileCLIP-S0 are kept as
selectable presets so the choice can be re-measured on the real N100 instead
of being taken on faith.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from tokenizers import Tokenizer

from .scoring import PROMPTS, Assessment, assess

HF = "https://huggingface.co"


@dataclass(frozen=True)
class FileSpec:
    name: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    family: str
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    pad_id: int
    files: tuple[FileSpec, ...]
    note: str

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


def _files(repo: str, commit: str, vision: tuple[str, str, int], text: tuple[str, str, int],
           tokenizer: tuple[str, str, int]) -> tuple[FileSpec, ...]:
    base = f"{HF}/{repo}/resolve/{commit}"
    return (
        FileSpec("vision_model.onnx", f"{base}/{vision[0]}", vision[1], vision[2]),
        FileSpec("text_model.onnx", f"{base}/{text[0]}", text[1], text[2]),
        FileSpec("tokenizer.json", f"{base}/tokenizer.json", tokenizer[1], tokenizer[2]),
    )


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_SPECS: dict[tuple[str, str], ModelSpec] = {
    ("CLIP-ViT-B-32", "int8"): ModelSpec(
        key="clip_vit_b32_int8",
        display="OpenCLIP ViT-B/32 · ONNX int8",
        family="clip_vit_b32",
        image_size=224,
        mean=CLIP_MEAN,
        std=CLIP_STD,
        pad_id=49407,
        note="轻量备选：156 MB 权重，CPU 动态量化；内存紧张时使用。",
        files=_files(
            "Xenova/clip-vit-base-patch32",
            "d15189d7028b43f1d3e65039190477f6af591c2a",
            ("onnx/vision_model_quantized.onnx", "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299", 89117001),
            ("onnx/text_model_quantized.onnx", "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a", 64504507),
            ("tokenizer.json", "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd", 2224119),
        ),
    ),
    ("CLIP-ViT-B-32", "fp32"): ModelSpec(
        key="clip_vit_b32_fp32",
        display="OpenCLIP ViT-B/32 · ONNX fp32",
        family="clip_vit_b32",
        image_size=224,
        mean=CLIP_MEAN,
        std=CLIP_STD,
        pad_id=49407,
        note="推荐默认值：608 MB 权重，召回率更高。",
        files=_files(
            "Xenova/clip-vit-base-patch32",
            "d15189d7028b43f1d3e65039190477f6af591c2a",
            ("onnx/vision_model.onnx", "fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40", 351685709),
            ("onnx/text_model.onnx", "3f6571f5bad13a97c469c1622e1cfc4d9aef78b79fdbfcff804ca357bfada8cc", 254058553),
            ("tokenizer.json", "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd", 2224119),
        ),
    ),
    ("MobileCLIP-S0", "fp32"): ModelSpec(
        key="mobileclip_s0_fp32",
        display="Apple MobileCLIP-S0 · ONNX fp32",
        family="mobileclip_s0",
        image_size=256,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        pad_id=256,
        note="更小但在这类细粒度目标上的区分度不足；仅作为轻量备选。",
        files=_files(
            "Xenova/mobileclip_s0",
            "757d59c9c6870a76a4b0306f05f5061bca15c39f",
            ("onnx/vision_model.onnx", "17d3c037b1d488c10c50e09f6009ea5a198caef4e0e8f4ea5617b7cb2d067ac0", 45543630),
            ("onnx/text_model.onnx", "f6e9bd5742bfc515889e901634d8a2ff2a57fab8564e4ad3760e800b1a51b77c", 169807789),
            ("tokenizer.json", "72ed5c96db5729294468543e4bc75fce14ca63f58e37300290189ba1c1e52b85", 2224081),
        ),
    ),
    ("MobileCLIP-S0", "int8"): ModelSpec(
        key="mobileclip_s0_int8",
        display="Apple MobileCLIP-S0 · ONNX int8",
        family="mobileclip_s0",
        image_size=256,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        pad_id=256,
        note="55 MB 权重，但在无 AVX-VNNI 的虚拟 CPU 上反而更慢。",
        files=_files(
            "Xenova/mobileclip_s0",
            "757d59c9c6870a76a4b0306f05f5061bca15c39f",
            ("onnx/vision_model_int8.onnx", "7a1b45f57fb9f3cde9d325759883e9451d7281336caeb9c576ae918e72080f0b", 11846808),
            ("onnx/text_model_int8.onnx", "fc8d87978623385c17a46331ffb9cb5ab7fe8b61c513c094602b85f08edd0a0b", 42799230),
            ("tokenizer.json", "72ed5c96db5729294468543e4bc75fce14ca63f58e37300290189ba1c1e52b85", 2224081),
        ),
    ),
}

DEFAULT_KEY = ("CLIP-ViT-B-32", "fp32")


def resolve_spec(model_name: str, variant: str) -> ModelSpec:
    variant = variant if variant in {"fp32", "int8"} else "int8"
    if (model_name, variant) in _SPECS:
        return _SPECS[(model_name, variant)]
    # Accept the model key itself.
    for spec in _SPECS.values():
        if spec.key == model_name:
            return spec
    return _SPECS[DEFAULT_KEY]


@dataclass
class ClassifyResult:
    assessment: Assessment
    preprocess_ms: float
    inference_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.assessment.as_dict(),
            "preprocess_ms": round(self.preprocess_ms, 2),
            "inference_ms": round(self.inference_ms, 2),
        }


class ModelUnavailable(Exception):
    pass


class ClipOnnxModel:
    def __init__(
        self,
        *,
        model_name: str,
        variant: str,
        model_root: Path,
        cpu_threads: int,
        auto_download: bool,
        download_timeout: int,
        high_threshold: float,
        pending_threshold: float,
        min_positive_score: float,
        min_pending_score: float,
        logger,
    ):
        self.spec = resolve_spec(model_name, variant)
        self.model_dir = Path(model_root) / self.spec.key
        self.cpu_threads = cpu_threads
        self.auto_download = auto_download
        self.download_timeout = download_timeout
        self.high_threshold = high_threshold
        self.pending_threshold = pending_threshold
        self.min_positive_score = min_positive_score
        self.min_pending_score = min_pending_score
        self.logger = logger

        self._state = "not_loaded"
        self._last_error = ""
        self._last_failure = 0.0
        self._lock = asyncio.Lock()
        self._run_lock = threading.Lock()
        self._vision: ort.InferenceSession | None = None
        self._text: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._text_emb: np.ndarray | None = None
        self._load_ms = 0.0
        self._text_ms = 0.0
        self._warmup_ms = 0.0

    @property
    def state(self) -> str:
        return self._state

    @property
    def ready(self) -> bool:
        return self._state == "ready"

    def status(self) -> dict[str, Any]:
        present = [f for f in self.spec.files if (self.model_dir / f.name).is_file()]
        return {
            "state": self._state,
            "model": self.spec.display,
            "key": self.spec.key,
            "family": self.spec.family,
            "dir": str(self.model_dir),
            "files_present": len(present),
            "files_expected": len(self.spec.files),
            "weights_mb": round(self.spec.total_bytes / 1024 / 1024, 1),
            "threads": self.cpu_threads,
            "load_ms": round(self._load_ms, 1),
            "text_encode_ms": round(self._text_ms, 1),
            "warmup_ms": round(self._warmup_ms, 1),
            "prompts": len(PROMPTS),
            "providers": self._vision.get_providers() if self._vision else [],
            "error": self._last_error,
        }

    async def ensure_ready(self) -> None:
        if self.ready:
            return
        if self._state == "error" and time.monotonic() - self._last_failure < 60:
            raise ModelUnavailable(f"model previously failed; retry in <60s: {self._last_error}")
        async with self._lock:
            if self.ready:
                return
            self._state = "loading"
            try:
                await asyncio.to_thread(self._prepare_sync)
            except Exception as exc:
                self._state = "error"
                self._last_failure = time.monotonic()
                self._last_error = f"{type(exc).__name__}: {exc}"
                self.logger.error("模型加载失败: %s", self._last_error)
                raise
            self._state = "ready"
            self._last_error = ""

    def _prepare_sync(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        missing = [f for f in self.spec.files if not self._file_ok(self.model_dir / f.name, f)]
        if missing and not self.auto_download:
            names = ", ".join(f.name for f in missing)
            raise ModelUnavailable(f"model files missing and auto_download_model=false: {names}")
        for f in missing:
            self._download(f)

        t0 = time.perf_counter()
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.cpu_threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self._vision = ort.InferenceSession(
            str(self.model_dir / "vision_model.onnx"), so, providers=["CPUExecutionProvider"]
        )
        self._text = ort.InferenceSession(
            str(self.model_dir / "text_model.onnx"), so, providers=["CPUExecutionProvider"]
        )

        self._tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=77)
        self._tokenizer.enable_padding(length=77, pad_id=self.spec.pad_id)

        t1 = time.perf_counter()
        enc = self._tokenizer.encode_batch(list(PROMPTS))
        ids = np.asarray([e.ids for e in enc], dtype=np.int64)
        emb = self._text.run(None, {"input_ids": ids})[0].astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        self._text_emb = emb
        self._text_ms = (time.perf_counter() - t1) * 1000

        # One warm-up pass so the first real image does not pay allocation cost.
        zeros = np.zeros((1, 3, self.spec.image_size, self.spec.image_size), dtype=np.float32)
        t2 = time.perf_counter()
        self._vision.run(None, {"pixel_values": zeros})
        self._warmup_ms = (time.perf_counter() - t2) * 1000
        self._load_ms = (time.perf_counter() - t0) * 1000
        self.logger.info(
            "模型就绪: %s | 权重 %.1f MB | 加载 %.0f ms | 文本编码 %.0f ms | 线程 %d",
            self.spec.display,
            self.spec.total_bytes / 1024 / 1024,
            self._load_ms,
            self._text_ms,
            self.cpu_threads,
        )

    def _file_ok(self, path: Path, spec: FileSpec) -> bool:
        if not path.is_file():
            return False
        try:
            if path.stat().st_size != spec.size:
                return False
            return _sha256(path) == spec.sha256
        except OSError:
            return False

    def _download(self, spec: FileSpec) -> None:
        target = self.model_dir / spec.name
        if target.exists() and not self._file_ok(target, spec):
            self.logger.warning("删除校验失败的旧文件: %s", target.name)
            target.unlink(missing_ok=True)
        if self._file_ok(target, spec):
            return
        part = target.with_name(target.name + ".part")
        self.logger.info("下载模型文件 %s (%.1f MB)...", spec.name, spec.size / 1024 / 1024)
        last_log = 0
        h = hashlib.sha256()
        downloaded = 0
        req = urllib.request.Request(spec.url, headers={"User-Agent": "AstrBot-ImgCurator/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self.download_timeout) as resp, open(part, "wb") as f:
                total = int(resp.headers.get("Content-Length") or spec.size)
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > spec.size:
                        raise ModelUnavailable(f"{spec.name} larger than expected")
                    h.update(chunk)
                    f.write(chunk)
                    if downloaded - last_log >= 32 * 1024 * 1024:
                        last_log = downloaded
                        self.logger.info(
                            "  %s: %.0f/%.0f MB", spec.name, downloaded / 1024 / 1024, total / 1024 / 1024
                        )
            if downloaded != spec.size:
                raise ModelUnavailable(f"{spec.name} size {downloaded} != {spec.size}")
            digest = h.hexdigest()
            if digest != spec.sha256:
                raise ModelUnavailable(f"{spec.name} sha256 mismatch")
            os.replace(part, target)
        except Exception:
            part.unlink(missing_ok=True)
            raise

    def _preprocess(self, path: Path) -> np.ndarray:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            width, height = im.size
            size = self.spec.image_size
            if width < height:
                new_w, new_h = size, max(size, round(height * size / width))
            else:
                new_w, new_h = max(size, round(width * size / height)), size
            im = im.resize((new_w, new_h), Image.BICUBIC)
            left = (new_w - size) // 2
            top = (new_h - size) // 2
            im = im.crop((left, top, left + size, top + size))
            array = np.asarray(im, dtype=np.float32) / 255.0
        mean = np.asarray(self.spec.mean, dtype=np.float32)
        std = np.asarray(self.spec.std, dtype=np.float32)
        array = (array - mean) / std
        return np.transpose(array, (2, 0, 1))[None, ...]

    def classify_sync(self, path: Path) -> ClassifyResult:
        if not self.ready or self._vision is None or self._text_emb is None:
            raise ModelUnavailable("model is not loaded")
        t0 = time.perf_counter()
        pixel_values = self._preprocess(path)
        t1 = time.perf_counter()
        with self._run_lock:
            image_emb = self._vision.run(None, {"pixel_values": pixel_values})[0].astype(np.float32)
        t2 = time.perf_counter()
        image_emb /= np.linalg.norm(image_emb, axis=1, keepdims=True)
        similarities = (image_emb @ self._text_emb.T)[0]
        assessment = assess(
            similarities,
            high_threshold=self.high_threshold,
            pending_threshold=self.pending_threshold,
            min_positive_score=self.min_positive_score,
            min_pending_score=self.min_pending_score,
        )
        return ClassifyResult(
            assessment=assessment,
            preprocess_ms=(t1 - t0) * 1000,
            inference_ms=(t2 - t1) * 1000,
        )

    async def classify(self, path: Path) -> ClassifyResult:
        await self.ensure_ready()
        return await asyncio.to_thread(self.classify_sync, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
