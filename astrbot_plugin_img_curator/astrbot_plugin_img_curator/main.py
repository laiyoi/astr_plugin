"""AstrBot local image curator plugin (AstrBot 4.29.x, aiocqhttp).

Listens to every group image, runs a resident ONNX CLIP encoder on CPU, saves
high-confidence matches under a UUID filename, and keeps uncertain images in
a separate review directory.  Rejected images leave only a SHA-256 row.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict, deque
from pathlib import Path

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Forward, Image as MessageImage
from astrbot.api.message_components import Json, Node, Nodes, Reply
from astrbot.api.message_components import Video as MessageVideo
from astrbot.api.star import Context, Star, StarTools

from .core.config import PluginConfig
from .core.forward_collector import ForwardCollector
from .core.model import ClipOnnxModel
from .core.pipeline import CuratorPipeline, Job
from .core.store import Store

#debug
import json
#

PLUGIN_NAME = "astrbot_plugin_img_curator"
PLUGIN_VERSION = "1.0.0"


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
    """本地图片筛选收藏：MobileCLIP / ViT-B/32 ONNX，纯 CPU，无云端 API。"""

    def __init__(
        self,
        context: Context,
        config: AstrBotConfig | None = None,
        data_dir: Path | None = None,
    ):
        super().__init__(context)
        raw = dict(config or {})
        base_dir = Path(data_dir) if data_dir else StarTools.get_data_dir(PLUGIN_NAME)
        self.cfg = PluginConfig.load(raw, base_dir)
        self.cfg.ensure_directories()

        self.store = Store(self.cfg.db_path, self.cfg.csv_path)
        self.model = ClipOnnxModel(
            model_name=self.cfg.model_name,
            variant=self.cfg.model_variant,
            model_root=self.cfg.model_dir,
            cpu_threads=self.cfg.cpu_threads,
            auto_download=self.cfg.auto_download_model,
            download_timeout=self.cfg.download_timeout_sec,
            high_threshold=self.cfg.high_threshold,
            pending_threshold=self.cfg.pending_threshold,
            min_positive_score=self.cfg.min_positive_score,
            min_pending_score=self.cfg.min_pending_score,
            logger=self.logger,
        )
        self.pipeline = CuratorPipeline(self.cfg, self.store, self.model, self.logger)
        self.forward_collector = ForwardCollector(self.cfg, self.logger)
        self.rate_limiter = GroupRateLimiter(self.cfg.rate_limit_count, self.cfg.rate_limit_window_sec)
        self._group_cache: dict[str, tuple[float, str]] = {}
        self._warm_task: asyncio.Task | None = None
        self.logger.info(
            "插件已初始化: %s | 保存目录 %s | 模型 %s",
            PLUGIN_VERSION,
            self.cfg.save_dir,
            self.model.spec.display,
        )

    async def initialize(self) -> None:
        await self.pipeline.start()
        await self.forward_collector.start()
        if self.cfg.recognition_enabled and self.cfg.auto_download_model:
            self._warm_task = asyncio.create_task(self._warm_model(), name="img-curator-warmup")
        elif not self.cfg.recognition_enabled:
            self.logger.info("识别未启用：仅接收合并记录训练集，不加载/读取视觉模型")
        elif not self.model.ready:
            self.logger.warning(
                "识别已启用但模型未预加载且 auto_download_model=false；图片会因模型缺失而跳过"
            )

    async def terminate(self) -> None:
        if self._warm_task:
            self._warm_task.cancel()
            try:
                await self._warm_task
            except asyncio.CancelledError:
                pass
            self._warm_task = None
        await self.forward_collector.stop()
        await self.pipeline.stop()
        self.store.close()

    async def _warm_model(self) -> None:
        try:
            await self.model.ensure_ready()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.error("模型预热失败: %s", exc)

    # -- group image listener -------------------------------------------
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    async def on_group_image(self, event: AstrMessageEvent) -> None:
        """监听所有群聊中的图片/短视频消息。没有群号白名单。"""
        if not self.cfg.enabled:
            return
        self.logger.debug(
            "群消息: group=%s recognition=%s paused=%s components=%s",
            event.get_group_id(),
            self.cfg.recognition_enabled,
            self.pipeline.paused,
            [type(c).__name__ for c in (getattr(event.message_obj, "message", None) or [])],
        )
        if _looks_like_our_command(event.message_str):
            # Do not archive media attached to /imgfilter test.
            return
        if event.get_self_id() and event.get_sender_id() == event.get_self_id():
            return
        group_id = str(event.get_group_id() or "")
        if not group_id:
            return

        scope = f"group:{group_id}"
        with open(self.cfg.forward_dir / "last_event.txt", "w", encoding="utf-8") as f:
            json.dump(event and event.message_obj and event.message_str, f, ensure_ascii=False, indent=2)
        if self.forward_collector.is_armed(scope) and _has_forward(event):
            group_name = await self._resolve_group_name(event, group_id)
            if await self.forward_collector.maybe_collect(event, scope, group_id, group_name):
                return

        if not self.cfg.recognition_enabled or self.pipeline.paused:
            return

        media = _collect_media(event)
        if not media:
            return

        for _ in media:
            if not self.rate_limiter.allow(group_id):
                self.logger.warning("群 %s 触发频率限制，媒体已跳过", group_id)
                return

        group_name = await self._resolve_group_name(event, group_id)
        sender_id = event.get_sender_id()
        sender_name = event.get_sender_name()
        message_id = str(getattr(event.message_obj, "message_id", "") or "")
        for component, media_type in media:
            await self.pipeline.enqueue(
                Job(
                    image=component,
                    media_type=media_type,
                    sender_id=sender_id,
                    sender_name=sender_name,
                    group_id=group_id,
                    group_name=group_name,
                    message_id=message_id,
                )
            )

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    async def on_private_forward(self, event: AstrMessageEvent) -> None:
        """私聊中只处理合并聊天记录，不处理普通私聊图片。"""
        if not self.cfg.enabled:
            return
        self.logger.debug(
            "群消息: group=%s recognition=%s paused=%s components=%s",
            event.get_group_id(),
            self.cfg.recognition_enabled,
            self.pipeline.paused,
            [type(c).__name__ for c in (getattr(event.message_obj, "message", None) or [])],
        )
        if _looks_like_our_command(event.message_str):
            return
        if event.get_self_id() and event.get_sender_id() == event.get_self_id():
            return
        sender_id = event.get_sender_id()
        scope = f"private:{sender_id}"
        self.logger.debug(
            "私聊消息: qq=%s armed=%s forward=%s components=%s",
            sender_id,
            self.forward_collector.is_armed(scope),
            _has_forward(event),
            [type(c).__name__ for c in (getattr(event.message_obj, "message", None) or [])],
        )
        if not self.forward_collector.is_armed(scope) or not _has_forward(event):
            return
        await self.forward_collector.maybe_collect(event, scope, "", "私聊")

    async def _scope_for_event(self, event: AstrMessageEvent) -> tuple[str, str, str, str]:
        group_id = str(event.get_group_id() or "")
        if group_id:
            name = await self._resolve_group_name(event, group_id)
            return f"group:{group_id}", group_id, name, f"群 {group_id}"
        sender_id = event.get_sender_id() or event.get_session_id()
        return f"private:{sender_id}", "", "私聊", f"私聊 {sender_id}"

    async def _resolve_group_name(self, event: AstrMessageEvent, group_id: str) -> str:
        now = time.monotonic()
        cached = self._group_cache.get(group_id)
        if cached and now - cached[0] < 3600:
            return cached[1]
        name = ""
        group = None
        try:
            group = await asyncio.wait_for(event.get_group(), timeout=5)
        except Exception:  # noqa: BLE001 - group name is optional metadata
            group = None
        if group is None:
            group = getattr(event.message_obj, "group", None)
        if group is not None:
            name = str(getattr(group, "group_name", "") or "")
        if not name:
            inbound = getattr(event.message_obj, "group", None)
            name = str(getattr(inbound, "group_name", "") or "") if inbound else ""
        # Cache positive names longer than failures.
        self._group_cache[group_id] = (now, name)
        return name

    # -- commands --------------------------------------------------------
    @filter.command_group("imgfilter")
    def imgfilter(self):
        """本地图片筛选收藏管理（仅管理员）。"""

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("status")
    async def cmd_status(self, event: AstrMessageEvent):
        """查看插件运行状态、模型状态与累计计数。"""
        event.stop_event()
        snap = self.pipeline.snapshot()
        counts = snap["counts"]
        ms = self.model.status()
        state = "已暂停" if snap["paused"] else "运行中"
        lines = [
            f"图片筛选：{state} | 启用：{'是' if self.cfg.enabled else '否'} | "
            f"识别：{'开' if self.cfg.recognition_enabled else '关（仅训练采集）'}",
            f"队列：{snap['queue']}/{snap['queue_capacity']}",
            f"模型：{ms['model']} [{ms['state']}] {ms['weights_mb']} MB",
            f"已识别：{counts.get('recognized', 0)} | 收藏：{counts.get('saved', 0)} | "
            f"待确认：{counts.get('pending', 0)} | 不收藏：{counts.get('rejected', 0)}",
            f"重复：{counts.get('duplicates', 0)} | 丢弃：{counts.get('queue_dropped', 0)} | "
            f"错误：{counts.get('errors', 0)}",
            f"目录：{self.cfg.save_dir}",
        ]
        fs = self.forward_collector.status()
        if fs["sessions"]:
            lines.append(f"合并记录采集：{len(fs['sessions'])} 个会话，队列 {fs['queue']}")
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("model")
    async def cmd_model(self, event: AstrMessageEvent, action: str = ""):
        """查看模型加载状态；`/imgfilter model load` 可主动下载并加载。"""
        event.stop_event()
        if action.strip().lower() in {"load", "reload", "加载"}:
            try:
                await self.model.ensure_ready()
            except Exception as exc:  # noqa: BLE001
                yield event.plain_result(f"模型加载失败：{exc}")
                return
        ms = self.model.status()
        lines = [
            f"模型：{ms['model']}",
            f"状态：{ms['state']} | 权重：{ms['weights_mb']} MB | 线程：{ms['threads']}",
            f"文件：{ms['files_present']}/{ms['files_expected']} | 目录：{ms['dir']}",
            f"加载：{ms['load_ms']} ms | 文本编码：{ms['text_encode_ms']} ms | "
            f"预热：{ms['warmup_ms']} ms",
            f"提示词：{ms['prompts']} 条 | Provider：{', '.join(ms['providers']) or '未加载'}",
        ]
        if ms["error"]:
            lines.append(f"错误：{ms['error']}")
        if not ms["files_present"]:
            lines.append("模型文件缺失；auto_download_model=true 时会在首张图片或 load 时下载。")
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("stats")
    async def cmd_stats(self, event: AstrMessageEvent):
        """查看累计数量与最近推理耗时。"""
        event.stop_event()
        snap = self.pipeline.snapshot()
        counts = snap["counts"]
        lat = snap["latency"]
        lines = [
            f"已识别 {counts.get('recognized', 0)} 张：收藏 {counts.get('saved', 0)}，"
            f"待确认 {counts.get('pending', 0)}，不收藏 {counts.get('rejected', 0)}",
            f"重复 {counts.get('duplicates', 0)}，队列丢弃 {counts.get('queue_dropped', 0)}，"
            f"错误 {counts.get('errors', 0)}，模型错误 {counts.get('model_errors', 0)}",
            f"单张推理耗时（最近 {lat['count']} 张）：平均 {lat['avg_ms']:.1f} ms，"
            f"P95 {lat['p95_ms']:.1f} ms，最大 {lat['max_ms']:.1f} ms",
            "说明：CLIP 相似度不是概率；阈值为保守初值，需用授权样本校准。",
        ]
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("pending")
    async def cmd_pending(self, event: AstrMessageEvent):
        """查看最近的待确认图片 ID。"""
        event.stop_event()
        rows = self.store.recent_pending(10)
        if not rows:
            yield event.plain_result("没有待确认图片。")
            return
        lines = [f"待确认 {self.store.count_status('pending')} 张，最近："]
        for row in rows:
            lines.append(
                f"{row['id'][:8]}  {row['score']:.3f}/{row['margin']:+.3f}  {row['created_at']}"
            )
        lines.append("使用 /imgfilter confirm <ID前缀> 或 reject <ID前缀> 审核。")
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("pause")
    async def cmd_pause(self, event: AstrMessageEvent):
        """暂停识别（不停止已启动的模型进程）。"""
        event.stop_event()
        self.pipeline.paused = True
        self.store.set_paused(True)
        yield event.plain_result("已暂停图片识别。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("resume")
    async def cmd_resume(self, event: AstrMessageEvent):
        """恢复识别。"""
        event.stop_event()
        self.pipeline.paused = False
        self.store.set_paused(False)
        yield event.plain_result("已恢复图片识别。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("test")
    async def cmd_test(self, event: AstrMessageEvent):
        """对当前消息中的图片做一次分类，不保存、不去重。"""
        event.stop_event()
        if not self.cfg.recognition_enabled:
            yield event.plain_result("识别当前未启用（recognition_enabled=false），不会加载模型。")
            return
        media = _collect_media(event)
        if not media:
            yield event.plain_result("请在发送本命令时附带一张图片/短视频（或引用一条含媒体消息）。")
            return
        try:
            component, media_type = media[0]
            result = await self.pipeline.preview(component, media_type)
        except Exception as exc:  # noqa: BLE001
            yield event.plain_result(f"测试失败：{exc}")
            return
        decision = {"saved": "自动收藏", "pending": "待确认", "rejected": "不收藏"}.get(
            result["decision"], result["decision"]
        )
        lines = [
            f"媒体：{'视频（抽帧）' if result.get('media_type') == 'video' else '图片'}",
            f"结果：{decision}",
            f"正分 {result['positive_score']:.3f} | 负分 {result['negative_score']:.3f} | "
            f"margin {result['margin']:+.3f}",
            f"预处理 {result['preprocess_ms']:.0f} ms | 推理 {result['inference_ms']:.0f} ms",
        ]
        if result.get("media_type") == "video":
            lines.append(f"抽帧 {result.get('frames', 0)} 张，最佳帧约 {result.get('frame_time', 0)} s")
        lines += [
            "前 5 个提示词：",
        ]
        for prompt, score in result["top5"]:
            lines.append(f"  {score:.3f}  {prompt}")
        lines.append("注意：这是相对相似度，不是概率；未保存本次测试媒体。")
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("confirm")
    async def cmd_confirm(self, event: AstrMessageEvent, image_id: str = ""):
        """把待确认图片移入收藏：/imgfilter confirm <ID前缀>。"""
        event.stop_event()
        yield event.plain_result(self._do_review(image_id, True))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("reject")
    async def cmd_reject(self, event: AstrMessageEvent, image_id: str = ""):
        """删除待确认图片并标记负样本：/imgfilter reject <ID前缀>。"""
        event.stop_event()
        yield event.plain_result(self._do_review(image_id, False))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("forward")
    async def cmd_forward(self, event: AstrMessageEvent, minutes: int = 10):
        """开始接收合并聊天记录并保存图片作为训练集。"""
        event.stop_event()
        scope, _group_id, _group_name, label = await self._scope_for_event(event)
        duration = self.forward_collector.arm(scope, minutes)
        yield event.plain_result(
            f"已开始接收{label}的合并聊天记录，持续 {duration} 分钟。\n"
            f"请发送合并转发；图片保存到 {self.cfg.forward_dir}，标签为 unlabeled。"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("forward_status")
    async def cmd_forward_status(self, event: AstrMessageEvent):
        """查看合并聊天记录采集状态。"""
        event.stop_event()
        status = self.forward_collector.status()
        lines = [
            f"目录：{status['root']}",
            f"队列：{status['queue']} | 已见图片：{status['seen']}",
            f"批次：{status['batches']} | 保存图片：{status['images']} | "
            f"重复：{status['duplicates']} | 错误：{status['errors']}",
        ]
        if status.get("last_error"):
            lines.append(f"最近错误：{status['last_error'][:300]}")
        if status["sessions"]:
            for item in status["sessions"]:
                lines.append(f"{item['scope']} 剩余 {item['remaining_sec']} 秒")
        else:
            lines.append("当前没有接收中的会话。")
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("forward_cancel")
    async def cmd_forward_cancel(self, event: AstrMessageEvent):
        """停止当前群/私聊的合并聊天记录采集。"""
        event.stop_event()
        scope, _group_id, _group_name, label = await self._scope_for_event(event)
        if self.forward_collector.cancel(scope):
            yield event.plain_result(f"已停止{label}的合并聊天记录采集。")
        else:
            yield event.plain_result(f"{label}当前没有接收中的合并聊天记录采集。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @imgfilter.command("help")
    async def cmd_help(self, event: AstrMessageEvent):
        """显示帮助。"""
        event.stop_event()
        yield event.plain_result(
            "本地图片筛选收藏\n"
            "/imgfilter status | model [load] | stats | pending\n"
            "/imgfilter pause | resume\n"
            "/imgfilter test（附带图片/视频）\n"
            "/imgfilter forward [分钟] | forward_status | forward_cancel\n"
            "/imgfilter confirm|reject <待确认ID前缀>\n"
            "仅管理员可用；识别与审核均在本机完成。"
        )

    def _do_review(self, prefix: str, accept: bool) -> str:
        prefix = prefix.strip()
        if not re.fullmatch(r"[0-9a-fA-F-]{4,36}", prefix):
            return "请输入至少 4 位、仅含十六进制字符或连字符的待确认 ID 前缀。"
        status, row, error = self.pipeline.review(prefix, accept)
        if status == "not_found":
            return f"没有找到待确认图片：{prefix}"
        if status == "ambiguous":
            return "该前缀匹配多张待确认图片，请提供更长 ID。"
        if status == "error":
            return f"审核失败：{error}"
        assert row is not None
        if accept:
            return f"已收藏 {row['id']}（文件名保持不变：{row['file_name']}）。"
        return f"已拒绝并删除 {row['id']}。"


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
    for prefix in ("/", "!", "！", ""):
        candidate = stripped[len(prefix):].lstrip() if prefix and stripped.startswith(prefix) else stripped
        if candidate.lower().startswith("imgfilter"):
            return True
    return False
