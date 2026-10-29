"""AstrBot plugin: femboydetector image/video collector with local web review."""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import mimetypes
import re
import time
import zipfile
from collections import defaultdict, deque
from pathlib import Path

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Forward, Image as MessageImage
from astrbot.api.message_components import Json, Node, Nodes, Reply
from astrbot.api.message_components import Video as MessageVideo
from astrbot.api.star import Context, Star, StarTools
from quart import jsonify, request, send_file

from .core.config import PluginConfig
from .core.forward_media import ForwardMediaExtractor
from .core.model import TasteOnnxModel
from .core.pipeline import CuratorPipeline, Job
from .core.store import Store

PLUGIN_NAME = "astrbot_plugin_img_curator"
PLUGIN_VERSION = "2.0.0"


class GroupRateLimiter:
    def __init__(self, count: int, window_sec: int):
        self.count = count
        self.window = window_sec
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, group_id: str) -> bool:
        now = time.monotonic()
        hits = self._hits[group_id]
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.count:
            return False
        hits.append(now)
        return True


class Main(Star):
    """DINOv2 femboydetector collector with merged-forward support."""

    def __init__(
        self,
        context: Context,
        config: AstrBotConfig | None = None,
        data_dir: Path | None = None,
    ):
        super().__init__(context)
        self.cfg = PluginConfig.load(dict(config or {}), Path(data_dir) if data_dir else StarTools.get_data_dir(PLUGIN_NAME))
        self.cfg.ensure_directories()
        self.store = Store(self.cfg.db_path, self.cfg.csv_path)
        self.model = TasteOnnxModel(self.cfg, self.logger)
        self.pipeline = CuratorPipeline(self.cfg, self.store, self.model, self.logger)
        self.rate_limiter = GroupRateLimiter(self.cfg.rate_limit_count, self.cfg.rate_limit_window_sec)
        self._group_cache: dict[str, tuple[float, str]] = {}
        self._warm_task: asyncio.Task | None = None
        self.logger.info(
            "插件已初始化: %s | 保存目录 %s | 模型 %s",
            PLUGIN_VERSION,
            self.cfg.save_dir,
            self.model.display,
        )

    async def initialize(self) -> None:
        await self.pipeline.start()
        self._register_web_apis()
        if self.cfg.recognition_enabled:
            self._warm_task = asyncio.create_task(self._warm_model(), name="img-curator-warmup")
        else:
            self.logger.info("识别未启用，不加载模型")

    async def terminate(self) -> None:
        if self._warm_task:
            self._warm_task.cancel()
            try:
                await self._warm_task
            except asyncio.CancelledError:
                pass
            self._warm_task = None
        await self.pipeline.stop()
        self.store.close()

    async def _warm_model(self) -> None:
        try:
            await self.model.ensure_ready()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.error("模型预热失败: %s", exc)

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    async def on_group_image(self, event: AstrMessageEvent) -> None:
        if not self.cfg.enabled or _looks_like_our_command(event.message_str):
            return
        if event.get_self_id() and event.get_sender_id() == event.get_self_id():
            return
        group_id = str(event.get_group_id() or "")
        if not group_id:
            return
        group_name = await self._resolve_group_name(event, group_id)
        await self._enqueue_event_media(event, group_id, group_name, private=False)

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    async def on_private_forward(self, event: AstrMessageEvent) -> None:
        if not self.cfg.enabled or _looks_like_our_command(event.message_str):
            return
        if event.get_self_id() and event.get_sender_id() == event.get_self_id():
            return
        if not _has_forward(event):
            return
        await self._enqueue_event_media(event, "", "私聊", private=True)

    async def _enqueue_event_media(
        self,
        event: AstrMessageEvent,
        group_id: str,
        group_name: str,
        *,
        private: bool,
    ) -> None:
        if not self.cfg.recognition_enabled or self.pipeline.paused:
            return
        direct = _collect_media(event)
        refs = []
        if _has_forward(event):
            extractor = ForwardMediaExtractor(event.bot, self.logger)
            try:
                refs = await extractor.from_event(event)
            except Exception:
                self.logger.exception("解析合并聊天记录失败")
        items = [(component, kind, event.get_sender_id(), event.get_sender_name()) for component, kind in direct]
        items += [
            (ref.component, ref.media_type, ref.sender_id or event.get_sender_id(), ref.sender_name or event.get_sender_name())
            for ref in refs
        ]
        if not items:
            return
        for _component, _kind, _sid, _sname in items:
            if not self.rate_limiter.allow(group_id or f"private:{event.get_sender_id()}"):
                self.logger.warning("触发频率限制，媒体已跳过")
                return
        message_id = str(getattr(event.message_obj, "message_id", "") or "")
        for component, media_type, sender_id, sender_name in items:
            await self.pipeline.enqueue(
                Job(
                    image=component,
                    media_type=media_type,
                    sender_id=str(sender_id or ""),
                    sender_name=str(sender_name or ""),
                    group_id=group_id,
                    group_name=group_name,
                    message_id=message_id,
                )
            )

    def _register_web_apis(self) -> None:
        routes = (
            ("curator/images", self.api_images, ["GET"]),
            ("curator/image/<image_id>", self.api_image, ["GET"]),
            ("curator/label", self.api_label, ["POST"]),
            ("curator/users", self.api_users, ["GET"]),
            ("curator/export", self.api_export, ["GET"]),
        )
        for path, handler, methods in routes:
            self.context.register_web_api(
                f"/{PLUGIN_NAME}/{path}", handler, methods, f"Femboy dataset: {path}"
            )

    async def api_images(self):
        rows = self.store.list_images(
            user=request.args.get("user", ""),
            label=request.args.get("label", ""),
            status=request.args.get("status", ""),
            limit=int(request.args.get("limit", 120)),
            offset=int(request.args.get("offset", 0)),
        )
        for row in rows:
            row["label"] = row.get("review_label") or row.get("top_group") or "unlabeled"
        total = self.store.count_images(
            user=request.args.get("user", ""),
            label=request.args.get("label", ""),
            status=request.args.get("status", ""),
        )
        return jsonify({"items": rows, "total": total})

    async def api_image(self, image_id: str):
        row = self.store.get_image(image_id)
        if not row:
            return jsonify({"error": "not found"}), 404
        path = Path(str(row.get("stored_path") or "")).resolve()
        root = self.cfg.data_dir.resolve()
        if not path.is_file() or root not in path.parents:
            return jsonify({"error": "file missing"}), 404
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return jsonify({"data_url": f"data:{mime};base64,{data}"})

    async def api_label(self):
        body = await request.get_json() or {}
        try:
            ok = self.store.set_review_label(str(body.get("id", "")), str(body.get("label", "")))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": ok})

    async def api_users(self):
        return jsonify({"items": self.store.user_ranking()})

    async def api_export(self):
        user = request.args.get("user", "")
        label = request.args.get("label", "")
        status = request.args.get("status", "")
        rows = self.store.list_images(user=user, label=label, status=status, limit=100000)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            meta = io.StringIO()
            writer = csv.writer(meta)
            writer.writerow(["image_id", "label", "status", "sender_id", "sender_name", "group_id", "group_name", "score", "created_at"])
            for row in rows:
                row_label = row.get("review_label") or row.get("top_group") or "unlabeled"
                path = Path(str(row.get("stored_path") or ""))
                if path.is_file():
                    zf.write(path, f"{row_label}/{path.name}")
                writer.writerow([
                    row.get("id"), row_label, row.get("status"), row.get("sender_id"),
                    row.get("sender_name"), row.get("group_id"), row.get("group_name"),
                    row.get("score"), row.get("created_at"),
                ])
            zf.writestr("metadata.csv", meta.getvalue())
        buffer.seek(0)
        return await send_file(
            buffer,
            mimetype="application/zip",
            as_attachment=True,
            attachment_filename="training_dataset.zip",
        )

    async def _resolve_group_name(self, event: AstrMessageEvent, group_id: str) -> str:
        now = time.monotonic()
        cached = self._group_cache.get(group_id)
        if cached and now - cached[0] < 3600:
            return cached[1]
        group = None
        try:
            group = await asyncio.wait_for(event.get_group(), timeout=5)
        except Exception:
            group = getattr(event.message_obj, "group", None)
        name = str(getattr(group, "group_name", "") or "") if group is not None else ""
        self._group_cache[group_id] = (now, name)
        return name

    @filter.command_group("imgfilter")
    def imgfilter(self):
        """本地图片筛选管理（仅管理员）。"""

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("status")
    async def cmd_status(self, event: AstrMessageEvent):
        event.stop_event()
        snap = self.pipeline.snapshot()
        ms = self.model.status()
        state = "已暂停" if snap["paused"] else "运行中"
        yield event.plain_result(
            f"图片筛选：{state} | 识别：{'开' if self.cfg.recognition_enabled else '关'}\n"
            f"队列：{snap['queue']}/{snap['queue_capacity']}\n"
            f"模型：{ms['model']} [{ms['state']}] {ms['weights_mb']} MB\n"
            f"审核/导出：AstrBot 管理界面 → 插件 → 可视化视图\n"
            f"目录：{self.cfg.save_dir}"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("model")
    async def cmd_model(self, event: AstrMessageEvent, action: str = ""):
        event.stop_event()
        if action.strip().lower() in {"load", "reload", "加载"}:
            try:
                await self.model.ensure_ready()
            except Exception as exc:  # noqa: BLE001
                yield event.plain_result(f"模型加载失败：{exc}")
                return
        ms = self.model.status()
        yield event.plain_result(
            f"模型：{ms['model']} [{ms['state']}]\n"
            f"仓库：{ms['repo']} | 文件：{ms['files_present']}/{ms['files_expected']}\n"
            f"目录：{ms['dir']}\n"
            f"加载：{ms['load_ms']} ms | 类别：{', '.join(ms['class_names']) or '未加载'}\n"
            f"错误：{ms['error'] or '无'}"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("stats")
    async def cmd_stats(self, event: AstrMessageEvent):
        event.stop_event()
        snap = self.pipeline.snapshot()
        c = snap["counts"]
        lat = snap["latency"]
        yield event.plain_result(
            f"已识别 {c.get('recognized', 0)}：收藏 {c.get('saved', 0)}，缓存 {c.get('pending', 0)}，"
            f"不收藏 {c.get('rejected', 0)}\n"
            f"重复 {c.get('duplicates', 0)}，错误 {c.get('errors', 0)}\n"
            f"推理：平均 {lat['avg_ms']:.1f} ms，P95 {lat['p95_ms']:.1f} ms"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("pause")
    async def cmd_pause(self, event: AstrMessageEvent):
        event.stop_event()
        self.pipeline.paused = True
        self.store.set_paused(True)
        yield event.plain_result("已暂停识别。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("resume")
    async def cmd_resume(self, event: AstrMessageEvent):
        event.stop_event()
        self.pipeline.paused = False
        self.store.set_paused(False)
        yield event.plain_result("已恢复识别。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("test")
    async def cmd_test(self, event: AstrMessageEvent):
        event.stop_event()
        if not self.cfg.recognition_enabled:
            yield event.plain_result("识别当前未启用。")
            return
        media = _collect_media(event)
        if not media:
            yield event.plain_result("请附带一张图片/短视频。")
            return
        try:
            result = await self.pipeline.preview(media[0][0], media[0][1])
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"测试失败：{exc}")
            return
        scores = " ".join(f"{k}={v:.3f}" for k, v in result["class_scores"].items())
        yield event.plain_result(
            f"label={result['label']} score={result['score']:.3f} {scores}\n"
            f"推理 {result['inference_ms']:.0f} ms；未保存。"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("pending")
    async def cmd_pending(self, event: AstrMessageEvent):
        event.stop_event()
        rows = self.store.recent_pending(10)
        if not rows:
            yield event.plain_result("没有缓存中的图片。")
            return
        lines = [f"缓存 {self.store.count_status('pending')} 张，最近："]
        for row in rows:
            lines.append(f"{row['id'][:8]}  {row['score']:.3f}  {row['created_at']}")
        lines.append("用网页或 /imgfilter confirm|reject <ID前缀> 审核。")
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("confirm")
    async def cmd_confirm(self, event: AstrMessageEvent, image_id: str = ""):
        event.stop_event()
        yield event.plain_result(self._do_review(image_id, True))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("reject")
    async def cmd_reject(self, event: AstrMessageEvent, image_id: str = ""):
        event.stop_event()
        yield event.plain_result(self._do_review(image_id, False))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("help")
    async def cmd_help(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(
            "Femboy Detector 收藏\n"
            "自动处理群图片/视频和合并聊天记录；私聊处理合并记录。\n"
            "/imgfilter status | model [load] | stats | pending\n"
            "/imgfilter pause | resume\n"
            "/imgfilter test（附带图片/视频）\n"
            "/imgfilter confirm|reject <缓存ID前缀>\n"
            "审核/导出：AstrBot 管理界面 → 插件 → 可视化视图。"
        )

    def _do_review(self, prefix: str, accept: bool) -> str:
        prefix = prefix.strip()
        if not re.fullmatch(r"[0-9a-fA-F-]{4,36}", prefix):
            return "请输入至少 4 位缓存 ID 前缀。"
        status, row, error = self.pipeline.review(prefix, accept)
        if status == "not_found":
            return f"没有找到缓存图片：{prefix}"
        if status == "ambiguous":
            return "前缀匹配多张，请提供更长 ID。"
        if status == "error":
            return f"审核失败：{error}"
        return f"已{'收藏' if accept else '拒绝'} {row['id']}"


def _collect_media(event: AstrMessageEvent) -> list[tuple[object, str]]:
    message = getattr(event.message_obj, "message", None) or []
    media: list[tuple[object, str]] = []
    for component in message:
        if isinstance(component, MessageImage):
            media.append((component, "image"))
        elif isinstance(component, MessageVideo):
            media.append((component, "video"))
        elif isinstance(component, Reply):
            for nested in getattr(component, "chain", None) or []:
                if isinstance(nested, MessageImage):
                    media.append((nested, "image"))
                elif isinstance(nested, MessageVideo):
                    media.append((nested, "video"))
    return media


def _has_forward(event: AstrMessageEvent) -> bool:
    message = getattr(event.message_obj, "message", None) or []
    return any(isinstance(component, (Forward, Nodes, Node, Json)) for component in message)


def _looks_like_our_command(text: str) -> bool:
    stripped = (text or "").lstrip()
    for prefix in ("/", "!", "！"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):].lstrip()
            break
    return stripped.lower().startswith("imgfilter")
