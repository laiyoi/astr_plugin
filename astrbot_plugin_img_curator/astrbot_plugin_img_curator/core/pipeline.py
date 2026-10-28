"""Bounded async queue + per-image pipeline."""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PluginConfig
from .model import ClipOnnxModel, ModelUnavailable
from PIL import Image

from .sources import ImageFetchError, ImageFetcher, sanitize_image
from .store import Store
from .video import FrameExtractionError, extract_frames


@dataclass
class Job:
    image: object
    sender_id: str = ""
    sender_name: str = ""
    group_id: str = ""
    group_name: str = ""
    message_id: str = ""
    media_type: str = "image"
    enqueued_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))


class CuratorPipeline:
    def __init__(self, cfg: PluginConfig, store: Store, model: ClipOnnxModel, logger):
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
        self.logger.info(
            "图片筛选队列已启动 (容量 %d, 目录 %s)", self.cfg.queue_size, self.cfg.save_dir
        )

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
            self.logger.warning(
                "队列已满 (%d)，丢弃群 %s 的一张图片", self.cfg.queue_size, job.group_id
            )
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
                self.logger.exception("处理图片时发生未预期错误")
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
                self.logger.debug("重复图片，跳过: %s", temp.sha256[:12])
                return
            result = await self._classify_or_error(temp.path, "图片")
            if result is None:
                return
            await self._finalize(
                job,
                result=result,
                source_path=temp.path,
                sha256=temp.sha256,
                ext=temp.ext,
                mime=temp.mime,
                width=temp.width,
                height=temp.height,
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
        frames_dir = self.cfg.tmp_dir / f"video_{uuid.uuid4().hex}"
        try:
            temp = await self._fetcher.fetch_video(
                job.image, self.cfg.tmp_dir, max_bytes=self.cfg.max_video_bytes
            )
            if self.cfg.dedupe and self.store.find_by_sha(temp.sha256) is not None:
                self.store.incr_meta("duplicates")
                self.logger.debug("重复视频，跳过: %s", temp.sha256[:12])
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
            best: tuple[Any, Any] | None = None
            for frame in frames:
                result = await self._classify_or_error(frame.path, "视频帧")
                if result is None:
                    continue
                if best is None or (
                    result.assessment.margin,
                    result.assessment.positive_score,
                ) > (
                    best[0].assessment.margin,
                    best[0].assessment.positive_score,
                ):
                    best = (result, frame)
            if best is None:
                self.store.incr_meta("errors")
                self.logger.warning("视频没有可识别的帧，未保存: %s", temp.sha256[:12])
                return
            result, frame = best
            with Image.open(frame.path) as im:
                width, height = im.size
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

    async def _classify_or_error(self, path: Path, label: str):
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

    async def _finalize(
        self,
        job: Job,
        *,
        result,
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
        assessment = result.assessment
        if assessment.decision == "rejected":
            self.store.mark_seen(
                sha256,
                status="rejected",
                score=assessment.positive_score,
                margin=assessment.margin,
                latency_ms=result.inference_ms,
            )
            self.logger.info(
                "不收藏 %s sha=%s top=%s score=%.3f margin=%+.3f",
                source_type,
                sha256[:12],
                assessment.top_group,
                assessment.positive_score,
                assessment.margin,
            )
            return

        if assessment.decision == "pending":
            if not self.cfg.save_pending:
                self.store.mark_seen(
                    sha256,
                    status="rejected",
                    score=assessment.positive_score,
                    margin=assessment.margin,
                    latency_ms=result.inference_ms,
                )
                return
            if (
                self.cfg.max_pending_files > 0
                and self.store.count_status("pending") >= self.cfg.max_pending_files
            ):
                self.store.mark_seen(
                    sha256,
                    status="rejected",
                    score=assessment.positive_score,
                    margin=assessment.margin,
                    latency_ms=result.inference_ms,
                )
                self.logger.warning("待确认目录已达上限，媒体未保存")
                return

        status = "saved" if assessment.decision == "saved" else "pending"
        target_dir = self.cfg.save_dir if status == "saved" else self.cfg.pending_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        image_id = str(uuid.uuid4())
        target = target_dir / f"{image_id}{ext}"
        try:
            if self.cfg.strip_metadata:
                await asyncio.to_thread(sanitize_image, source_path, target, ext)
            else:
                await asyncio.to_thread(_move_file, source_path, target)
        except (OSError, ImageFetchError):
            target.unlink(missing_ok=True)
            self.store.incr_meta("errors")
            self.logger.exception("写入/清理媒体元数据失败")
            return
        try:
            size_bytes = target.stat().st_size
        except OSError:
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
            "score": assessment.positive_score,
            "margin": assessment.margin,
            "top_group": assessment.top_group,
            "top_prompt": assessment.top_prompt,
            "model": self.model.spec.key,
            "latency_ms": result.inference_ms,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if not self.store.record_image(record):
            target.unlink(missing_ok=True)
            self.store.incr_meta("duplicates")
            return
        await asyncio.to_thread(self.store.export_csv)
        self.logger.info(
            "已%s %s | %s | %s | score=%.3f margin=%+.3f | %.0f ms",
            "收藏" if status == "saved" else "转入待确认",
            image_id,
            source_type,
            assessment.top_group,
            assessment.positive_score,
            assessment.margin,
            result.inference_ms,
        )

    async def preview(self, media: object, media_type: str = "image") -> dict[str, Any]:
        """Classify one media item without touching dedupe or the archive."""
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
                best = None
                for frame in frames:
                    try:
                        result = await self.model.classify(frame.path)
                    except Exception:
                        continue
                    if best is None or (
                        result.assessment.margin,
                        result.assessment.positive_score,
                    ) > (
                        best[0].assessment.margin,
                        best[0].assessment.positive_score,
                    ):
                        best = (result, frame)
                if best is None:
                    raise FrameExtractionError("no frame could be classified")
                data = best[0].as_dict()
                data.update(
                    {
                        "sha256": temp.sha256,
                        "media_type": "video",
                        "frames": len(frames),
                        "frame_time": round(best[1].time_sec, 2),
                    }
                )
                return data
            finally:
                temp.cleanup()
                shutil.rmtree(frames_dir, ignore_errors=True)

        temp = await self._fetcher.fetch(media, self.cfg.tmp_dir)
        try:
            result = await self.model.classify(temp.path)
            data = result.as_dict()
            data.update({"sha256": temp.sha256, "media_type": "image"})
            return data
        finally:
            temp.cleanup()

    def review(self, prefix: str, accept: bool) -> tuple[str, dict[str, Any] | None, str]:
        matches = self.store.find_pending_prefix(prefix, limit=2)
        if not matches:
            return "not_found", None, ""
        if len(matches) > 1:
            return "ambiguous", None, ""
        row = matches[0]
        image_id = row["id"]
        src = Path(row["stored_path"])
        if accept:
            target = self.cfg.save_dir / row["file_name"]
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                _move_file(src, target)
            except OSError as exc:
                return "error", row, str(exc)
            self.store.update_review(
                image_id,
                label="positive",
                status="saved",
                file_name=target.name,
                stored_path=str(target),
            )
        else:
            try:
                src.unlink(missing_ok=True)
            except OSError as exc:
                return "error", row, str(exc)
            self.store.update_review(
                image_id,
                label="negative",
                status="rejected",
                file_name=None,
                stored_path=None,
            )
        self.store.export_csv()
        return "ok", row, ""

    def snapshot(self) -> dict[str, Any]:
        counts = self.store.counts()
        counts["duplicates"] = int(self.store.get_meta("duplicates", "0") or 0)
        counts["errors"] = int(self.store.get_meta("errors", "0") or 0)
        counts["model_errors"] = int(self.store.get_meta("model_errors", "0") or 0)
        counts["queue_dropped"] = int(self.store.get_meta("queue_dropped", "0") or 0)
        return {
            "paused": self.paused,
            "queue": self.queue_size(),
            "queue_capacity": self.cfg.queue_size,
            "counts": counts,
            "latency": self.store.latency_stats(),
        }


def _move_file(src: Path, dst: Path) -> None:
    try:
        os.replace(src, dst)
    except OSError:
        shutil.move(str(src), str(dst))
