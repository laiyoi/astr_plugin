"""Collect images from merged/forwarded chat records into a training set.

A short-lived session is armed by an admin command.  While armed, forwarded
message components are queued to a separate worker.  The worker walks nested
nodes (and fetches ``Forward`` ids through OneBot when only an id is present),
downloads images, and writes UUID files plus metadata.jsonl.

This is a training-data collector, not the normal archive pipeline.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from astrbot.api.message_components import Forward, Image as MessageImage
from astrbot.api.message_components import Json, Node, Nodes, Video as MessageVideo

from .config import PluginConfig
from .sources import ImageFetcher, sanitize_image
from .store import now_iso


def _load_json_data(value: Any) -> Any:
    if isinstance(value, dict):
        if "data" in value and isinstance(value["data"], (str, dict, list)):
            return _load_json_data(value["data"])
        return value
    if isinstance(value, str):
        text = value.replace("&#44;", ",").strip()
        if not text or text[0] not in "[{":
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        return _load_json_data(parsed)
    if isinstance(value, list):
        return value
    return None


def _as_message_list(raw: Any) -> list[Any]:
    if isinstance(raw, dict):
        for key in ("message", "messages"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
        value = raw.get("data")
        if isinstance(value, dict):
            return _as_message_list(value)
        if isinstance(value, list):
            return value
    value = getattr(raw, "message", None)
    if isinstance(value, list):
        return value
    return []


def _raw_forward_content(raw: Any, forward_id: str) -> list[Any]:
    for segment in _as_message_list(raw):
        if not isinstance(segment, dict):
            continue
        data = segment.get("data") if isinstance(segment.get("data"), dict) else {}
        if segment.get("type") not in {"forward", "forward_msg", "nodes"}:
            continue
        candidate = str(data.get("id") or data.get("message_id") or "")
        if candidate and candidate != forward_id:
            continue
        content = _segment_content(data)
        if content:
            return content
    return []


def _segment_content(data: Any) -> list[Any]:
    """Return the already-included nested segments from a forward/node object."""
    if not isinstance(data, dict):
        return []
    for key in ("content", "messages", "message"):
        value = data.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            parsed = _load_json_data(value)
            if isinstance(parsed, list):
                return parsed
    return []


def _normalize_forward_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("messages", "message"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        value = payload.get("data")
        if isinstance(value, dict):
            return _normalize_forward_payload(value)
    return []


@dataclass
class ForwardJob:
    components: list[Any]
    group_id: str
    group_name: str
    sender_id: str
    sender_name: str
    message_id: str
    bot: object | None = None
    self_id: str = ""
    raw_message: object | None = None


class ForwardCollector:
    def __init__(self, cfg: PluginConfig, logger):
        self.cfg = cfg
        self.logger = logger
        self.root = cfg.forward_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, float] = {}
        self._queue: asyncio.Queue[ForwardJob | None] | None = None
        self._worker: asyncio.Task | None = None
        self._fetcher = ImageFetcher(
            max_bytes=cfg.max_file_bytes,
            max_pixels=cfg.max_image_pixels,
            min_side=cfg.min_image_side,
            timeout_sec=cfg.image_download_timeout_sec,
        )
        self._seen: set[str] = set()
        self._sessions_path = self.root / "sessions.json"
        self._index_lock = asyncio.Lock()
        self._load_sessions()
        self._stats = {
            "batches": 0,
            "images": 0,
            "duplicates": 0,
            "skipped_videos": 0,
            "errors": 0,
            "queue_dropped": 0,
            "last_error": "",
        }
        self._load_index()

    async def start(self) -> None:
        if self._queue is not None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self._queue = asyncio.Queue(maxsize=8)
        await self._fetcher.start()
        self._worker = asyncio.create_task(self._run(), name="img-curator-forward-worker")

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        await self._fetcher.close()
        self._queue = None
        # Keep sessions on disk so a plugin reload does not silently disarm
        # a time window the admin just opened.

    def arm(self, scope: str, minutes: int) -> int:
        minutes = max(1, min(minutes, self.cfg.forward_max_minutes))
        self._sessions[scope] = time.time() + minutes * 60
        self._save_sessions()
        return minutes

    def cancel(self, scope: str) -> bool:
        existed = self._sessions.pop(scope, None) is not None
        if existed:
            self._save_sessions()
        return existed

    def is_armed(self, scope: str) -> bool:
        expires = self._sessions.get(scope)
        if expires is None:
            return False
        if time.time() >= expires:
            self._sessions.pop(scope, None)
            self._save_sessions()
            return False
        return True

    def status(self) -> dict[str, Any]:
        now = time.time()
        sessions = []
        for scope, expires in list(self._sessions.items()):
            remaining = max(0, int(expires - now))
            if remaining == 0:
                self._sessions.pop(scope, None)
            else:
                sessions.append({"scope": scope, "remaining_sec": remaining})
        self._save_sessions()
        return {
            "root": str(self.root),
            "queue": self._queue.qsize() if self._queue is not None else 0,
            "sessions": sessions,
            "seen": len(self._seen),
            **self._stats,
        }

    async def maybe_collect(self, event, scope: str, group_id: str, group_name: str) -> bool:
        if not scope or not self.is_armed(scope):
            return False
        components = list(getattr(event.message_obj, "message", None) or [])
        if not any(isinstance(c, (Forward, Nodes, Node, Json)) for c in components):
            return False
        job = ForwardJob(
            components=components,
            group_id=group_id,
            group_name=group_name,
            sender_id=event.get_sender_id(),
            sender_name=event.get_sender_name(),
            message_id=str(getattr(event.message_obj, "message_id", "") or ""),
            bot=getattr(event, "bot", None),
            self_id=event.get_self_id(),
            raw_message=getattr(event.message_obj, "raw_message", None),
        )
        if self._queue is None:
            return False
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            self._stats["queue_dropped"] += 1
            self.logger.warning("合并聊天记录队列已满，丢弃一批")
            return False
        return True

    def _load_sessions(self) -> None:
        if not self._sessions_path.exists():
            return
        try:
            payload = json.loads(self._sessions_path.read_text(encoding="utf-8"))
            now = time.time()
            for scope, expires in payload.items():
                if isinstance(expires, (int, float)) and expires > now:
                    self._sessions[str(scope)] = float(expires)
        except Exception:
            self.logger.exception("读取合并记录会话失败")

    def _save_sessions(self) -> None:
        try:
            tmp = self._sessions_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._sessions, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._sessions_path)
        except OSError:
            self.logger.exception("保存合并记录会话失败")

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
                self._stats["errors"] += 1
                self.logger.exception("解析合并聊天记录失败")
            finally:
                self._queue.task_done()

    async def _process(self, job: ForwardJob) -> None:
        batch_id = str(uuid.uuid4())
        batch_dir = self.root / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        # Keep all training images in one flat folder; only per-batch metadata
        # stays under the batch UUID directory.
        images_dir = self.root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        started_at = now_iso()
        metadata: list[dict[str, Any]] = []
        started_ts = time.time()
        state = {
            "images": 0,
            "index": 0,
            "nodes": 0,
            "videos": 0,
            "stopped": False,
            "unresolved_nodes": 0,
        }

        try:
            self.logger.debug(
                "forward batch components=%s",
                [type(component).__name__ for component in job.components],
            )
            await self._walk_astrbot(
                job.components,
                images_dir=images_dir,
                metadata=metadata,
                state=state,
                depth=0,
                path=[],
                node_sender_id="",
                node_sender_name="",
                node_time=0,
                batch_id=batch_id,
                job=job,
            )
        finally:
            records = metadata
            metadata_path = batch_dir / "metadata.jsonl"
            with open(metadata_path, "w", encoding="utf-8") as handle:
                for row in records:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary = {
                "batch_id": batch_id,
                "images_dir": str(images_dir),
                "started_at": started_at,
                "finished_at": now_iso(),
                "elapsed_ms": round((time.time() - started_ts) * 1000),
                "group_id": job.group_id,
                "group_name": job.group_name,
                "sender_id": job.sender_id,
                "sender_name": job.sender_name,
                "message_id": job.message_id,
                "saved_images": state["images"],
                "candidate_images": state["index"],
                "nodes_seen": state["nodes"],
                "skipped_videos": state["videos"],
                "unresolved_nodes": state["unresolved_nodes"],
                "stopped_by_limit": state["stopped"],
                "label": "unlabeled",
            }
            (batch_dir / "batch.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._stats["batches"] += 1
            self._stats["images"] += state["images"]
            self._stats["skipped_videos"] += state["videos"]
            duplicates = sum(1 for row in records if row.get("status") == "duplicate")
            self.logger.info(
                "合并聊天记录入库: batch=%s saved=%d nodes=%d unresolved=%d "
                "duplicates=%d videos=%d elapsed=%dms",
                batch_id,
                state["images"],
                state["nodes"],
                state["unresolved_nodes"],
                duplicates,
                state["videos"],
                round((time.time() - started_ts) * 1000),
            )

    async def _walk_astrbot(
        self,
        components: list[Any],
        *,
        images_dir: Path,
        metadata: list[dict[str, Any]],
        state: dict[str, Any],
        depth: int,
        path: list[int],
        node_sender_id: str,
        node_sender_name: str,
        node_time: int,
        batch_id: str,
        job: ForwardJob,
    ) -> None:
        if depth > self.cfg.forward_max_depth or state["stopped"]:
            return
        for index, component in enumerate(components):
            if state["stopped"]:
                return
            if isinstance(component, MessageImage):
                await self._save_image(
                    component,
                    images_dir=images_dir,
                    metadata=metadata,
                    state=state,
                    depth=depth,
                    path=path + [index],
                    node_sender_id=node_sender_id,
                    node_sender_name=node_sender_name,
                    node_time=node_time,
                    batch_id=batch_id,
                    job=job,
                )
            elif isinstance(component, MessageVideo):
                state["videos"] += 1
            elif isinstance(component, Node):
                if state["nodes"] >= self.cfg.forward_max_nodes:
                    state["stopped"] = True
                    self.logger.warning("合并转发节点数达到上限 %d", self.cfg.forward_max_nodes)
                    return
                state["nodes"] += 1
                await self._walk_astrbot(
                    list(component.content or []),
                    images_dir=images_dir,
                    metadata=metadata,
                    state=state,
                    depth=depth + 1,
                    path=path + [index],
                    node_sender_id=str(component.uin or ""),
                    node_sender_name=str(component.name or ""),
                    node_time=int(component.time or 0),
                    batch_id=batch_id,
                    job=job,
                )
            elif isinstance(component, Nodes):
                for node_index, node in enumerate(component.nodes or []):
                    if state["stopped"]:
                        return
                    if state["nodes"] >= self.cfg.forward_max_nodes:
                        state["stopped"] = True
                        self.logger.warning("合并转发节点数达到上限 %d", self.cfg.forward_max_nodes)
                        return
                    state["nodes"] += 1
                    await self._walk_astrbot(
                        list(node.content or []),
                        images_dir=images_dir,
                        metadata=metadata,
                        state=state,
                        depth=depth + 1,
                        path=path + [index, node_index],
                        node_sender_id=str(node.uin or ""),
                        node_sender_name=str(node.name or ""),
                        node_time=int(node.time or 0),
                        batch_id=batch_id,
                        job=job,
                    )
            elif isinstance(component, Json):
                parsed = _load_json_data(getattr(component, "data", None))
                if parsed is not None:
                    await self._walk_multimsg_json(
                        parsed,
                        images_dir=images_dir,
                        metadata=metadata,
                        state=state,
                        depth=depth + 1,
                        path=path + [index],
                        node_sender_id=node_sender_id,
                        node_sender_name=node_sender_name,
                        node_time=node_time,
                        batch_id=batch_id,
                        job=job,
                    )
            elif isinstance(component, Forward):
                # Some adapters include the forwarded payload in the raw
                # event even when get_forward_msg refuses to fetch it.  Prefer
                # that local copy; fall back to the OneBot API only if absent.
                raw_content = _raw_forward_content(job.raw_message, str(component.id))
                if raw_content:
                    await self._walk_onebot(
                        raw_content,
                        images_dir=images_dir,
                        metadata=metadata,
                        state=state,
                        depth=depth + 1,
                        path=path + [index],
                        node_sender_id="",
                        node_sender_name="",
                        node_time=0,
                        batch_id=batch_id,
                        job=job,
                    )
                else:
                    payload = await self._fetch_forward(job, component.id)
                    await self._walk_onebot(
                        payload,
                        images_dir=images_dir,
                        metadata=metadata,
                        state=state,
                        depth=depth + 1,
                        path=path + [index],
                        node_sender_id="",
                        node_sender_name="",
                        node_time=0,
                        batch_id=batch_id,
                        job=job,
                    )

    async def _walk_onebot(
        self,
        segments: list[Any],
        *,
        images_dir: Path,
        metadata: list[dict[str, Any]],
        state: dict[str, Any],
        depth: int,
        path: list[int],
        node_sender_id: str,
        node_sender_name: str,
        node_time: int,
        batch_id: str,
        job: ForwardJob,
    ) -> None:
        if depth > self.cfg.forward_max_depth or state["stopped"]:
            return
        if not segments:
            return
        for index, segment in enumerate(segments):
            if state["stopped"]:
                return
            if not isinstance(segment, dict):
                continue

            # A full message object already contains its real segments.
            if isinstance(segment.get("message"), list) and "type" not in segment:
                if not self._take_node(state):
                    return
                sender = segment.get("sender") if isinstance(segment.get("sender"), dict) else {}
                await self._walk_onebot(
                    segment["message"],
                    images_dir=images_dir,
                    metadata=metadata,
                    state=state,
                    depth=depth + 1,
                    path=path + [index],
                    node_sender_id=str(segment.get("user_id") or sender.get("user_id") or ""),
                    node_sender_name=str(
                        segment.get("nickname") or sender.get("nickname") or sender.get("card") or ""
                    ),
                    node_time=int(segment.get("time") or sender.get("time") or 0),
                    batch_id=batch_id,
                    job=job,
                )
                continue

            # A node-like object with content directly attached.
            if "content" in segment and "type" not in segment:
                content = segment.get("content")
                if isinstance(content, list) and content:
                    if not self._take_node(state):
                        return
                    sender = segment.get("sender") if isinstance(segment.get("sender"), dict) else {}
                    await self._walk_onebot(
                        content,
                        images_dir=images_dir,
                        metadata=metadata,
                        state=state,
                        depth=depth + 1,
                        path=path + [index],
                        node_sender_id=str(segment.get("user_id") or sender.get("user_id") or ""),
                        node_sender_name=str(
                            segment.get("nickname") or sender.get("nickname") or sender.get("card") or ""
                        ),
                        node_time=int(segment.get("time") or sender.get("time") or 0),
                        batch_id=batch_id,
                        job=job,
                    )
                    continue

            seg_type = segment.get("type")
            data = segment.get("data") if isinstance(segment.get("data"), dict) else {}
            if seg_type == "image":
                await self._save_image(
                    SimpleNamespace(
                        url=str(data.get("url") or ""),
                        path=str(data.get("path") or ""),
                        file=str(data.get("file") or ""),
                    ),
                    images_dir=images_dir,
                    metadata=metadata,
                    state=state,
                    depth=depth,
                    path=path + [index],
                    node_sender_id=node_sender_id,
                    node_sender_name=node_sender_name,
                    node_time=node_time,
                    batch_id=batch_id,
                    job=job,
                )
            elif seg_type == "video":
                state["videos"] += 1
            elif seg_type in {"node", "nodes"}:
                content = _segment_content(data) or _segment_content(segment)
                if not content:
                    continue
                if not self._take_node(state):
                    return
                node_sender = data.get("sender") if isinstance(data.get("sender"), dict) else {}
                await self._walk_onebot(
                    content,
                    images_dir=images_dir,
                    metadata=metadata,
                    state=state,
                    depth=depth + 1,
                    path=path + [index],
                    node_sender_id=str(data.get("user_id") or node_sender.get("user_id") or ""),
                    node_sender_name=str(
                        data.get("nickname") or node_sender.get("nickname") or node_sender.get("card") or ""
                    ),
                    node_time=int(data.get("time") or node_sender.get("time") or 0),
                    batch_id=batch_id,
                    job=job,
                )
            elif seg_type in {"forward", "forward_msg"}:
                # The first get_forward_msg response usually contains the
                # complete nested tree.  Recurse into included content; do
                # not fetch nested forward ids again.
                content = _segment_content(data) or _segment_content(segment)
                if content:
                    await self._walk_onebot(
                        content,
                        images_dir=images_dir,
                        metadata=metadata,
                        state=state,
                        depth=depth + 1,
                        path=path + [index],
                        node_sender_id=node_sender_id,
                        node_sender_name=node_sender_name,
                        node_time=node_time,
                        batch_id=batch_id,
                        job=job,
                    )
                else:
                    state["unresolved_nodes"] += 1
            elif seg_type in {"json", "xml"}:
                parsed = _load_json_data(data.get("data") if seg_type == "json" else data)
                if parsed is not None:
                    await self._walk_multimsg_json(
                        parsed,
                        images_dir=images_dir,
                        metadata=metadata,
                        state=state,
                        depth=depth + 1,
                        path=path + [index],
                        node_sender_id=node_sender_id,
                        node_sender_name=node_sender_name,
                        node_time=node_time,
                        batch_id=batch_id,
                        job=job,
                    )
        return

    def _take_node(self, state: dict[str, Any]) -> bool:
        if state["nodes"] >= self.cfg.forward_max_nodes:
            state["stopped"] = True
            self.logger.warning("合并转发节点数达到上限 %d", self.cfg.forward_max_nodes)
            return False
        state["nodes"] += 1
        return True

    async def _walk_multimsg_json(
        self,
        parsed: Any,
        *,
        images_dir: Path,
        metadata: list[dict[str, Any]],
        state: dict[str, Any],
        depth: int,
        path: list[int],
        node_sender_id: str,
        node_sender_name: str,
        node_time: int,
        batch_id: str,
        job: ForwardJob,
    ) -> None:
        """Walk a Tencent multimsg JSON payload or a generic JSON wrapper."""
        if not isinstance(parsed, dict) or depth > self.cfg.forward_max_depth:
            return
        # Some payloads are {"messages":[...]} or {"message":[...]}.
        for key in ("messages", "message", "content"):
            value = parsed.get(key)
            if isinstance(value, list):
                await self._walk_onebot(
                    value,
                    images_dir=images_dir,
                    metadata=metadata,
                    state=state,
                    depth=depth + 1,
                    path=path + [0],
                    node_sender_id=node_sender_id,
                    node_sender_name=node_sender_name,
                    node_time=node_time,
                    batch_id=batch_id,
                    job=job,
                )
                return
        meta = parsed.get("meta") if isinstance(parsed.get("meta"), dict) else {}
        detail = meta.get("detail") if isinstance(meta.get("detail"), dict) else {}
        news = detail.get("news")
        if isinstance(news, list):
            # Multimsg preview items usually contain only text; some clients
            # include a picture URL.  Handle both without treating every
            # avatar as an image.
            if news:
                self.logger.debug(
                    "forward JSON news count=%d item_keys=%s",
                    len(news),
                    sorted(news[0].keys()) if isinstance(news[0], dict) else type(news[0]).__name__,
                )
            for item_index, item in enumerate(news):
                if not isinstance(item, dict):
                    continue
                for key in ("content", "message", "messages"):
                    value = item.get(key)
                    if isinstance(value, list):
                        await self._walk_onebot(
                            value,
                            images_dir=images_dir,
                            metadata=metadata,
                            state=state,
                            depth=depth + 1,
                            path=path + [item_index],
                            node_sender_id=str(item.get("user_id") or ""),
                            node_sender_name=str(item.get("nickname") or item.get("name") or ""),
                            node_time=int(item.get("time") or 0),
                            batch_id=batch_id,
                            job=job,
                        )
                picture = item.get("picture") or item.get("image")
                if isinstance(picture, str) and picture.startswith(("http://", "https://")):
                    await self._save_image(
                        SimpleNamespace(url=picture, path="", file=""),
                        images_dir=images_dir,
                        metadata=metadata,
                        state=state,
                        depth=depth + 1,
                        path=path + [item_index],
                        node_sender_id=str(item.get("user_id") or ""),
                        node_sender_name=str(item.get("nickname") or item.get("name") or ""),
                        node_time=int(item.get("time") or 0),
                        batch_id=batch_id,
                        job=job,
                    )
            return
        # Generic JSON wrappers: recurse into lists/dicts that look like
        # message containers, with a hard cap from the normal node limit.
        for key in ("data", "detail", "meta"):
            value = parsed.get(key)
            if isinstance(value, (dict, list)):
                nested = {"messages": value} if isinstance(value, list) else value
                await self._walk_multimsg_json(
                    nested,
                    images_dir=images_dir,
                    metadata=metadata,
                    state=state,
                    depth=depth + 1,
                    path=path + [0],
                    node_sender_id=node_sender_id,
                    node_sender_name=node_sender_name,
                    node_time=node_time,
                    batch_id=batch_id,
                    job=job,
                )

    async def _fetch_forward(self, job: ForwardJob, forward_id: str) -> list[Any]:
        bot = job.bot
        if bot is None:
            return []
        fid = str(forward_id)
        attempts: list[tuple[str, dict[str, Any]]] = [
            ("get_forward_msg", {"id": fid}),
            ("get_forward_msg", {"message_id": fid}),
            ("get_msg", {"message_id": fid}),
        ]
        for action, kwargs in attempts:
            try:
                payload = await bot.call_action(action, **kwargs)
            except Exception:
                continue
            content = _normalize_forward_payload(payload)
            if content:
                return content
            if isinstance(payload, dict):
                nested = _segment_content(payload.get("data"))
                if nested:
                    return nested
        self.logger.debug("拉取合并转发失败 id=%s", forward_id)
        return []

    async def _save_image(
        self,
        image: Any,
        *,
        images_dir: Path,
        metadata: list[dict[str, Any]],
        state: dict[str, Any],
        depth: int,
        path: list[int],
        node_sender_id: str,
        node_sender_name: str,
        node_time: int,
        batch_id: str,
        job: ForwardJob,
    ) -> None:
        if state["stopped"]:
            return
        if state["index"] >= self.cfg.forward_max_images:
            state["stopped"] = True
            self.logger.warning("单批合并转发图片数达到上限 %d", self.cfg.forward_max_images)
            return
        state["index"] += 1
        image_index = state["index"]
        temp = None
        try:
            temp = await self._fetcher.fetch(image, self.cfg.tmp_dir)
            if temp.sha256 in self._seen:
                self._stats["duplicates"] += 1
                metadata.append(
                    {
                        "batch_id": batch_id,
                        "image_index": image_index,
                        "status": "duplicate",
                        "sha256": temp.sha256,
                        "group_id": job.group_id,
                        "group_name": job.group_name,
                        "sender_id": job.sender_id,
                        "sender_name": job.sender_name,
                        "node_sender_id": node_sender_id,
                        "node_sender_name": node_sender_name,
                        "depth": depth,
                        "forward_path": ".".join(str(x) for x in path),
                        "collected_at": now_iso(),
                    }
                )
                return
            image_id = str(uuid.uuid4())
            target = images_dir / f"{image_id}{temp.ext}"
            if self.cfg.strip_metadata:
                await asyncio.to_thread(sanitize_image, temp.path, target, temp.ext)
            else:
                await asyncio.to_thread(shutil.move, str(temp.path), str(target))
            size_bytes = target.stat().st_size
            row = {
                "batch_id": batch_id,
                "image_id": image_id,
                "image_index": image_index,
                "status": "saved",
                "label": "unlabeled",
                "file": str(target.relative_to(self.root)),
                "sha256": temp.sha256,
                "mime": temp.mime,
                "width": temp.width,
                "height": temp.height,
                "size_bytes": size_bytes,
                "group_id": job.group_id,
                "group_name": job.group_name,
                "sender_id": job.sender_id,
                "sender_name": job.sender_name,
                "node_sender_id": node_sender_id,
                "node_sender_name": node_sender_name,
                "node_time": node_time,
                "depth": depth,
                "forward_path": ".".join(str(x) for x in path),
                "collected_at": now_iso(),
            }
            metadata.append(row)
            state["images"] += 1
            self._seen.add(temp.sha256)
            await self._append_index({"sha256": temp.sha256, "path": str(target.relative_to(self.root)), "batch_id": batch_id})
        except Exception as exc:  # noqa: BLE001
            self._stats["errors"] += 1
            self.logger.warning("合并转发中的图片保存失败: %s", exc)
            metadata.append(
                {
                    "batch_id": batch_id,
                    "image_index": image_index,
                    "status": "error",
                    "error": str(exc),
                    "group_id": job.group_id,
                    "sender_id": job.sender_id,
                    "depth": depth,
                    "forward_path": ".".join(str(x) for x in path),
                    "collected_at": now_iso(),
                }
            )
        finally:
            if temp is not None:
                temp.cleanup()

    async def _append_index(self, row: dict[str, Any]) -> None:
        async with self._index_lock:
            with open(self.root / "index.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _load_index(self) -> None:
        path = self.root / "index.jsonl"
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("sha256"):
                        self._seen.add(str(row["sha256"]))
        except Exception:
            self.logger.exception("读取合并转发索引失败")
