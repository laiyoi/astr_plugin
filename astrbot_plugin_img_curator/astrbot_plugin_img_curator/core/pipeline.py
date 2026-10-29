"""Bounded image/video queue with the DINOv2 three-class decision layer."""

from __future__ import annotations

import asyncio
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .config import PluginConfig
from .model import ModelUnavailable, TasteOnnxModel, TasteResult
from .sources import ImageFetchError, ImageFetcher, sanitize_image
from .store import Store
from .video import FrameExtractionError, extract_frames

try:
    from ..inference import decide_save
except ImportError:  # tests
    from inference import decide_save  # type: ignore


@dataclass
class Job:
    image: object
    sender_id: str = ""
    sender_name: str = ""
    group_id: str = ""
    group_name: str = ""
    message_id: str = ""
    media_type: str = "image"
    enqueued_at: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds")
    )


class CuratorPipeline:
    def __init__(self, cfg: PluginConfig, store: Store, model: TasteOnnxModel, logger):
        self.cfg = cfg
        self.store = store
        self.model = model
        self.logger = logger
        self.paused = store.get_paused()
        self._queue: asyncio.Queue[Job | None] | None = None
        self._worker: asyncio.Task | None = None
        self._stopping = False
        self._fetcher = ImageFetcher(
            max_bytes=cfg.max_file_bytes,
            max_pixels=cfg.max_image_pixels,
            min_side=cfg.min_image_side,
            timeout_sec=cfg.image_download_timeout_sec,
        )

    async def start(self) -> None:
        if self._queue is not None:
            return
        self.cfg.ensure_directories()
        self._queue = asyncio.Queue(maxsize=self.cfg.queue_size)
        await self._fetcher.start()
        self._worker = asyncio.create_task(self._run(), name="img-curator-worker")
        self.logger.info("图片筛选队列已启动 (容量 %d, 目录 %s)", self.cfg.queue_size, self.cfg.save_dir)

    async def stop(self) -> None:
        self._stopping = True
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        await self._fetcher.close()
        self._queue = None

    def queue_size(self) -> int:
        return self._queue.qsize() if self._queue is not None else 0

    async def enqueue(self, job: Job) -> bool:
        if self._stopping or self._queue is None:
            return False
        if self.paused or not self.cfg.enabled:
            return False
        try:
            self._queue.put_nowait(job)
            return True
        except asyncio.QueueFull:
            self.store.incr_meta("queue_dropped")
            self.logger.warning("队列已满 (%d)，丢弃群 %s 的一张媒体", self.cfg.queue_size, job.group_id)
            return False

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            job = await self._queue.get()
            try:
                if job is not None:
                    await self._process(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.store.incr_meta("errors")
                self.logger.exception("处理媒体时发生未预期错误")
            finally:
                self._queue.task_done()

    async def _process(self, job: Job) -> None:
        if self.paused or not self.cfg.enabled:
            return
        if job.media_type == "video":
            await self._process_video(job)
            return

        temp = None
        try:
            temp = await self._fetcher.fetch(job.image, self.cfg.tmp_dir)
            if self.cfg.dedupe and self.store.find_by_sha(temp.sha256) is not None:
                self.store.incr_meta("duplicates")
                return
            result = await self._classify_or_error(temp.path, "图片")
            if result is None:
                return
            with Image.open(temp.path) as image:
                width, height = image.size
            await self._finalize(
                job,
                result=result,
                source_path=temp.path,
                sha256=temp.sha256,
                ext=temp.ext,
                mime=temp.mime,
                width=width,
                height=height,
                source=temp.source,
                source_type="image",
                frame_time=None,
            )
        except ImageFetchError as exc:
            self.store.incr_meta("errors")
            self.logger.warning("图片获取失败，未保存: %s", exc)
        finally:
            if temp is not None:
                temp.cleanup()

    async def _process_video(self, job: Job) -> None:
        temp = None
        frames_dir = self.cfg.tmp_dir / f"frames_{uuid.uuid4().hex}"
        try:
            temp = await self._fetcher.fetch_video(
                job.image, self.cfg.tmp_dir, max_bytes=self.cfg.max_video_bytes
            )
            if self.cfg.dedupe and self.store.find_by_sha(temp.sha256) is not None:
                self.store.incr_meta("duplicates")
                return
            frames = await asyncio.to_thread(
                extract_frames,
                temp.path,
                frames_dir,
                max_frames=self.cfg.video_max_frames,
                max_seconds=self.cfg.video_max_seconds,
                fps=self.cfg.video_fps,
                timeout_sec=self.cfg.video_extract_timeout_sec,
            )
            best: tuple[TasteResult, Any] | None = None
            for frame in frames:
                result = await self._classify_or_error(frame.path, "视频帧")
                if result is None:
                    continue
                rank = result.class_scores.get("collect", 0.0) + result.class_scores.get("suspicious", 0.0)
                if best is None or rank > (
                    best[0].class_scores.get("collect", 0.0)
                    + best[0].class_scores.get("suspicious", 0.0)
                ):
                    best = (result, frame)
            if best is None:
                self.store.incr_meta("errors")
                self.logger.warning("视频没有可识别的帧，未保存: %s", temp.sha256[:12])
                return
            result, frame = best
            with Image.open(frame.path) as image:
                width, height = image.size
            await self._finalize(
                job,
                result=result,
                source_path=frame.path,
                sha256=temp.sha256,
                ext=".jpg",
                mime="image/jpeg",
                width=width,
                height=height,
                source=f"video({temp.source})",
                source_type="video_frame",
                frame_time=round(frame.time_sec, 2),
            )
        except ImageFetchError as exc:
            self.store.incr_meta("errors")
            self.logger.warning("视频获取失败，未保存: %s", exc)
        except FrameExtractionError as exc:
            self.store.incr_meta("errors")
            self.logger.warning("视频抽帧失败，未保存: %s", exc)
        finally:
            if temp is not None:
                temp.cleanup()
            shutil.rmtree(frames_dir, ignore_errors=True)

    async def _classify_or_error(self, path: Path, label: str) -> TasteResult | None:
        try:
            return await self.model.classify(path)
        except ModelUnavailable as exc:
            self.store.incr_meta("model_errors")
            self.store.incr_meta("errors")
            self.logger.error("模型不可用，%s未保存: %s", label, exc)
            return None
        except Exception:
            self.store.incr_meta("model_errors")
            self.store.incr_meta("errors")
            self.logger.exception("模型推理失败，%s未保存", label)
            return None

    def _decision(self, job: Job, result: TasteResult) -> str:
        if result.label == "collect":
            return "saved"
        if result.label == "skip":
            return "rejected"
        if result.label != "suspicious":
            return "rejected"
        count = self.store.count_saved_by_sender(job.sender_id)
        action = decide_save(
            "suspicious",
            count,
            {
                "save_at_saved_count": self.cfg.suspicious_save_at,
                "cache_below_saved_count": self.cfg.suspicious_cache_below,
            },
        )
        if action == "save":
            return "saved"
        if action == "cache":
            return "pending" if self.cfg.save_pending else "rejected"
        return "rejected"

    async def _finalize(
        self,
        job: Job,
        *,
        result: TasteResult,
        source_path: Path,
        sha256: str,
        ext: str,
        mime: str,
        width: int,
        height: int,
        source: str,
        source_type: str,
        frame_time: float | None,
    ) -> None:
        status = self._decision(job, result)
        if status == "rejected":
            self.store.mark_seen(
                sha256,
                status="rejected",
                score=result.score,
                margin=self._margin(result),
                latency_ms=result.inference_ms,
            )
            self.logger.info(
                "不收藏 %s sha=%s label=%s score=%.3f",
                source_type,
                sha256[:12],
                result.label,
                result.score,
            )
            return

        if status == "pending":
            if (
                self.cfg.max_pending_files > 0
                and self.store.count_status("pending") >= self.cfg.max_pending_files
            ):
                self.store.mark_seen(
                    sha256,
                    status="rejected",
                    score=result.score,
                    margin=self._margin(result),
                    latency_ms=result.inference_ms,
                )
                self.logger.warning("待确认目录已达上限，媒体未保存")
                return

        target_dir = self.cfg.save_dir if status == "saved" else self.cfg.pending_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        image_id = str(uuid.uuid4())
        target = target_dir / f"{image_id}{ext}"
        try:
            if self.cfg.strip_metadata:
                await asyncio.to_thread(sanitize_image, source_path, target, ext)
            else:
                await asyncio.to_thread(_move_file, source_path, target)
            size_bytes = target.stat().st_size
        except (OSError, ImageFetchError):
            target.unlink(missing_ok=True)
            self.store.incr_meta("errors")
            self.logger.exception("写入媒体文件失败")
            return
        if size_bytes > self.cfg.max_file_bytes:
            target.unlink(missing_ok=True)
            self.store.incr_meta("errors")
            self.logger.warning("清理元数据后的媒体超过体积上限，未保存")
            return

        record = {
            "id": image_id,
            "sha256": sha256,
            "status": status,
            "source_type": source_type,
            "frame_time": frame_time,
            "file_name": target.name,
            "stored_path": str(target),
            "source": source,
            "mime": mime,
            "width": width,
            "height": height,
            "size_bytes": size_bytes,
            "sender_id": job.sender_id,
            "sender_name": job.sender_name,
            "group_id": job.group_id,
            "group_name": job.group_name,
            "message_id": job.message_id,
            "score": result.score,
            "margin": self._margin(result),
            "top_group": result.label,
            "top_prompt": "",
            "model": self.model.key,
            "latency_ms": result.inference_ms,
            "review_label": "",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if not self.store.record_image(record):
            target.unlink(missing_ok=True)
            self.store.incr_meta("duplicates")
            return
        await asyncio.to_thread(self.store.export_csv)
        self.logger.info(
            "已%s %s | %s | label=%s | score=%.3f | %.0f ms",
            "收藏" if status == "saved" else "转入缓存",
            image_id,
            source_type,
            result.label,
            result.score,
            result.inference_ms,
        )

    @staticmethod
    def _margin(result: TasteResult) -> float:
        values = sorted(result.class_scores.values(), reverse=True)
        return float(values[0] - values[1]) if len(values) >= 2 else 0.0

    async def preview(self, media: object, media_type: str = "image") -> dict[str, Any]:
        if media_type == "video":
            temp = await self._fetcher.fetch_video(
                media, self.cfg.tmp_dir, max_bytes=self.cfg.max_video_bytes
            )
            frames_dir = self.cfg.tmp_dir / f"preview_{uuid.uuid4().hex}"
            try:
                frames = await asyncio.to_thread(
                    extract_frames,
                    temp.path,
                    frames_dir,
                    max_frames=self.cfg.video_max_frames,
                    max_seconds=self.cfg.video_max_seconds,
                    fps=self.cfg.video_fps,
                    timeout_sec=self.cfg.video_extract_timeout_sec,
                )
                if not frames:
                    raise ImageFetchError("video has no frames")
                result = await self.model.classify(frames[0].path)
                return {
                    "label": result.label,
                    "score": result.score,
                    "class_scores": result.class_scores,
                    "inference_ms": result.inference_ms,
                }
            finally:
                temp.cleanup()
                shutil.rmtree(frames_dir, ignore_errors=True)

        temp = await self._fetcher.fetch(media, self.cfg.tmp_dir)
        try:
            result = await self.model.classify(temp.path)
            return {
                "label": result.label,
                "score": result.score,
                "class_scores": result.class_scores,
                "inference_ms": result.inference_ms,
            }
        finally:
            temp.cleanup()

    def review(self, prefix: str, accept: bool) -> tuple[str, dict[str, Any] | None, str]:
        rows = self.store.find_pending_prefix(prefix)
        if not rows:
            return "not_found", None, ""
        if len(rows) > 1:
            return "ambiguous", None, ""
        row = rows[0]
        path = Path(str(row.get("stored_path") or ""))
        try:
            if accept:
                target_dir = self.cfg.save_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / path.name
                shutil.move(str(path), str(target))
                self.store.update_review(
                    row["id"], label="collect", status="saved",
                    file_name=target.name, stored_path=str(target),
                )
                return "ok", {**row, "file_name": target.name}, ""
            path.unlink(missing_ok=True)
            self.store.update_review(
                row["id"], label="skip", status="rejected",
                file_name="", stored_path="",
            )
            return "ok", row, ""
        except Exception as exc:  # noqa: BLE001
            return "error", row, str(exc)

    def snapshot(self) -> dict[str, Any]:
        return {
            "queue": self.queue_size(),
            "queue_capacity": self.cfg.queue_size,
            "paused": self.paused,
            "counts": self.store.counts(),
            "latency": self.store.latency_stats(200),
        }


def _move_file(src: Path, dst: Path) -> None:
    shutil.move(str(src), str(dst))
