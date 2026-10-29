"""AstrBot plugin: femboydetector image/video collector with local web review."""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import mimetypes
import re
import shutil
import time
import uuid
import zipfile
from collections import defaultdict, deque
from pathlib import Path

from PIL import Image, ImageOps

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
        self._stopping = False
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
        self._stopping = True
        if self._warm_task:
            self._warm_task.cancel()
            try:
                await self._warm_task
            except asyncio.CancelledError:
                pass
            self._warm_task = None
        await self.pipeline.stop()
        # AstrBot may still have an in-flight listener from the about-to-be
        # replaced instance.  Do not close SQLite underneath it immediately.
        asyncio.create_task(self._close_store_later(), name="img-curator-close-store")

    async def _close_store_later(self) -> None:
        await asyncio.sleep(60)
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
        if self._stopping:
            return
        if not self.cfg.enabled or _looks_like_our_command(event.message_str):
            return
        if event.get_self_id() and event.get_sender_id() == event.get_self_id():
            return
        group_id = str(event.get_group_id() or "")
        if not group_id:
            return
        group_name = await self._resolve_group_name(event, group_id)
        self._record_group_member(event, group_id, group_name)
        await self._enqueue_event_media(event, group_id, group_name, private=False)

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    async def on_private_message(self, event: AstrMessageEvent) -> None:
        if self._stopping:
            return
        if not self.cfg.enabled or _looks_like_our_command(event.message_str):
            return
        if event.get_self_id() and event.get_sender_id() == event.get_self_id():
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
        if direct and event.get_sender_name() and not self._stopping:
            fixed = self.store.backfill_sender_by_name(
                str(event.get_sender_name() or ""), str(event.get_sender_id() or "")
            )
            if fixed:
                self.logger.info("按昵称补全历史合并记录 QQ: %s -> %s（%d 条）", event.get_sender_name(), event.get_sender_id(), fixed)
        forward_items = []
        for ref in refs:
            name = ref.sender_name or event.get_sender_name()
            uid = ref.sender_id or event.get_sender_id()
            alias = self.store.sender_id_for_alias(name) if name else ""
            if alias and (not uid or uid == "1094950020"):
                uid = alias
            forward_items.append((ref.component, ref.media_type, uid, name))
        items += forward_items
        if not items:
            return
        # One token per incoming message, not per media.  A single merged
        # record often contains dozens of images; counting each image as a
        # separate request made bulk re-imports hit the limiter immediately.
        # Private messages are owner-driven and are not rate limited.
        if group_id and not self.rate_limiter.allow(group_id):
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
            ("curator/thumb/<image_id>", self.api_thumb, ["GET"]),
            ("curator/label", self.api_label, ["POST"]),
            ("curator/labels/batch", self.api_batch_label, ["POST"]),
            ("curator/sender", self.api_set_sender, ["POST"]),
            ("curator/resolve_names", self.api_resolve_names, ["POST"]),
            ("curator/users", self.api_users, ["GET"]),
            ("curator/ranking", self.api_ranking, ["GET"]),
            ("curator/aliases", self.api_aliases, ["GET"]),
            ("curator/aliases", self.api_set_alias, ["POST"]),
            ("curator/export/summary", self.api_export_summary, ["GET"]),
            ("curator/export", self.api_export, ["GET"]),
        )
        for path, handler, methods in routes:
            self.context.register_web_api(
                f"/{PLUGIN_NAME}/{path}", handler, methods, f"Femboy dataset: {path}"
            )

    @staticmethod
    def _int_arg(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            return max(minimum, min(int(request.args.get(name, default)), maximum))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _float_arg(name: str, default: float | None = None) -> float | None:
        raw = request.args.get(name)
        if raw in (None, ""):
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _image_url(self, image_id: str) -> str:
        return f"/api/v1/plugins/extensions/{PLUGIN_NAME}/curator/image/{image_id}"

    async def api_images(self):
        args = request.args
        limit = self._int_arg("limit", 24, 1, 96)
        offset = self._int_arg("offset", 0, 0, 10_000_000)
        sort = args.get("sort", "created_desc")
        if sort not in {"created_desc", "created_asc", "score_desc", "score_asc", "sender_asc"}:
            sort = "created_desc"
        kwargs = {
            "user": args.get("user", ""),
            "label": args.get("label", ""),
            "status": args.get("status", ""),
            "q": args.get("q", ""),
            "since": args.get("since", ""),
            "until": args.get("until", ""),
            "min_score": self._float_arg("min_score"),
            "max_score": self._float_arg("max_score"),
            "limit": limit,
            "offset": offset,
            "sort": sort,
        }
        rows = self.store.list_images(**kwargs)
        safe_fields = (
            "id", "label", "status", "score", "sender_id", "sender_name",
            "group_id", "group_name", "created_at", "reviewed_at", "source_type",
            "width", "height", "size_bytes", "top_group", "review_label",
        )
        rows = [
            {**{key: row.get(key) for key in safe_fields}, "url": self._image_url(row["id"])}
            for row in rows
        ]
        total = self.store.count_images(
            user=kwargs["user"], label=kwargs["label"], status=kwargs["status"],
            q=kwargs["q"], since=kwargs["since"], until=kwargs["until"],
            min_score=kwargs["min_score"], max_score=kwargs["max_score"],
        )
        return jsonify({"items": rows, "total": total, "limit": limit, "offset": offset})

    def _image_payload(self, image_id: str, max_side: int | None) -> dict:
        row = self.store.get_image(image_id)
        if not row:
            raise FileNotFoundError("not found")
        path = Path(str(row.get("stored_path") or "")).resolve()
        root = self.cfg.data_dir.resolve()
        if not path.is_file() or root not in path.parents:
            raise FileNotFoundError("file missing")
        if max_side is None:
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return {"data_url": f"data:{mime};base64,{data}"}
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_side, max_side))
            out = io.BytesIO()
            im.save(out, "JPEG", quality=82, optimize=True)
        data = base64.b64encode(out.getvalue()).decode("ascii")
        return {"data_url": f"data:image/jpeg;base64,{data}"}

    async def api_image(self, image_id: str):
        try:
            return jsonify(await asyncio.to_thread(self._image_payload, image_id, None))
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception:
            return jsonify({"error": "图片无法读取"}), 422

    async def api_thumb(self, image_id: str):
        try:
            return jsonify(await asyncio.to_thread(self._image_payload, image_id, 520))
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception:
            return jsonify({"error": "图片无法读取"}), 422

    def _apply_label(self, row: dict, label: str) -> dict:
        if label not in {"collect", "suspicious", "skip", "unlabeled"}:
            raise ValueError("invalid label")
        image_id = str(row["id"])
        path = Path(str(row.get("stored_path") or "")).resolve()
        root = self.cfg.data_dir.resolve()
        has_file = path.is_file() and root in path.parents

        if label == "skip":
            # A manual skip is a negative training label, not a delete.
            # Keep the bytes in skip/ so the dataset export can include them.
            target = path
            if has_file:
                self.cfg.skip_dir.mkdir(parents=True, exist_ok=True)
                target = self.cfg.skip_dir / path.name
                if path != target:
                    shutil.move(str(path), str(target))
            self.store.update_review(
                image_id, label="skip", status="rejected",
                file_name=target.name if has_file else "",
                stored_path=str(target) if has_file else "",
            )
            return {
                "ok": True, "id": image_id, "label": "skip", "status": "rejected",
                "removed": False, "moved": has_file,
            }

        if label == "collect" and has_file and row.get("status") != "saved":
            self.cfg.save_dir.mkdir(parents=True, exist_ok=True)
            target = self.cfg.save_dir / path.name
            if path != target:
                shutil.move(str(path), str(target))
            self.store.update_review(
                image_id, label="collect", status="saved",
                file_name=target.name, stored_path=str(target),
            )
            return {"ok": True, "id": image_id, "label": "collect", "status": "saved", "moved": True}

        if label == "collect" and not has_file:
            raise ValueError("image file is not available")

        self.store.set_review_label(image_id, label)
        return {
            "ok": True, "id": image_id, "label": label,
            "status": row.get("status") or "", "removed": False,
        }

    async def api_label(self):
        body = await request.get_json() or {}
        row = self.store.get_image(str(body.get("id", "")))
        if not row:
            return jsonify({"error": "not found"}), 404
        try:
            result = await asyncio.to_thread(self._apply_label, row, str(body.get("label", "")))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    async def api_batch_label(self):
        body = await request.get_json() or {}
        label = str(body.get("label", ""))
        ids = body.get("ids")
        if label not in {"collect", "suspicious", "skip", "unlabeled"}:
            return jsonify({"error": "invalid label"}), 400
        if not isinstance(ids, list) or not ids:
            return jsonify({"error": "ids must be a non-empty array"}), 400
        if len(ids) > 500:
            return jsonify({"error": "最多一次修改 500 张图片"}), 400
        ok = 0
        failed: list[dict] = []
        for raw_id in ids:
            image_id = str(raw_id)
            row = self.store.get_image(image_id)
            if not row:
                failed.append({"id": image_id, "error": "not found"})
                continue
            try:
                await asyncio.to_thread(self._apply_label, row, label)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed.append({"id": image_id, "error": str(exc)})
        return jsonify({"ok": ok, "failed": failed, "total": len(ids), "label": label})

    async def api_set_sender(self):
        body = await request.get_json() or {}
        ids = body.get("ids")
        if not isinstance(ids, list) or not ids:
            return jsonify({"error": "ids must be a non-empty array"}), 400
        if len(ids) > 500:
            return jsonify({"error": "最多一次修改 500 张图片"}), 400
        try:
            updated = await asyncio.to_thread(
                self.store.set_sender_for_images,
                [str(item) for item in ids],
                str(body.get("sender_id", "")),
                str(body.get("sender_name", "")),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True, "updated": updated, "total": len(ids)})

    async def api_users(self):
        """Backward-compatible endpoint: collect-only ranking."""
        return jsonify(self.store.ranking(limit=50, min_samples=1))

    async def api_ranking(self):
        threshold_raw = self._float_arg("threshold", 0.6)
        threshold = 0.6 if threshold_raw is None else max(0.0, min(1.0, threshold_raw))
        range_days = self._int_arg("range_days", 0, 0, 3650)
        min_samples = self._int_arg("min_samples", 0, 0, 100000)
        sort = request.args.get("sort", "high_count")
        if sort not in {"collect_count", "high_count", "high_ratio", "avg_score"}:
            sort = "high_count"
        limit = self._int_arg("limit", 50, 1, 200)
        offset = self._int_arg("offset", 0, 0, 1_000_000)
        result = self.store.ranking(
            threshold=threshold, range_days=range_days, min_samples=min_samples,
            sort=sort, q=request.args.get("q", ""), limit=limit, offset=offset,
        )
        for item in result.get("items", []):
            uid = str(item.get("sender_id") or "")
            item["avatar_url"] = (
                f"https://q1.qlogo.cn/g?b=qq&nk={uid}&s=100" if uid.isdigit() and len(uid) >= 5 else ""
            )
        result.update({
            "range_days": range_days, "sort": sort, "limit": limit, "offset": offset,
            "legacy_unknown": self.store.legacy_unknown_senders(),
        })
        return jsonify(result)

    @staticmethod
    def _readable_image(path: Path) -> bool:
        try:
            with Image.open(path) as im:
                im.verify()
            return True
        except Exception:
            return False

    @staticmethod
    def _export_labels() -> list[str]:
        raw = request.args.get("labels", "collect,suspicious,skip")
        allowed = {"collect", "suspicious", "skip"}
        return [item for item in raw.split(",") if item in allowed]

    def _onebot_client(self):
        manager = getattr(self.context, "platform_manager", None)
        for platform in getattr(manager, "platform_insts", []) or []:
            bot = getattr(platform, "bot", None)
            if bot is None and hasattr(platform, "get_client"):
                try:
                    bot = platform.get_client()
                except Exception:
                    bot = None
            if bot is not None and hasattr(bot, "call_action"):
                return bot
        return None

    async def api_resolve_names(self):
        bot = self._onebot_client()
        if bot is None:
            return jsonify({"error": "没有可用的 OneBot 适配器"}), 503
        users = self.store.users_missing_nickname()
        resolved = []
        failed = []
        updated = 0
        for uid in users:
            if not uid.isdigit():
                continue
            try:
                response = await asyncio.wait_for(
                    bot.call_action("get_stranger_info", user_id=int(uid), no_cache=False),
                    timeout=10,
                )
                data = response.get("data") if isinstance(response, dict) else None
                info = data if isinstance(data, dict) else response
                nickname = ""
                if isinstance(info, dict):
                    nickname = str(info.get("nick") or info.get("nickname") or info.get("card") or "").strip()
                if not nickname:
                    failed.append({"sender_id": uid, "error": "empty nickname"})
                    continue
                n = self.store.set_sender_nickname(uid, nickname)
                if nickname not in {"QQ用户", "未知用户"}:
                    n += self.store.set_sender_alias(nickname, uid)
                updated += n
                resolved.append({"sender_id": uid, "nickname": nickname, "updated": n})
            except Exception as exc:  # noqa: BLE001
                failed.append({"sender_id": uid, "error": str(exc)})
        return jsonify({"ok": True, "resolved": resolved, "failed": failed, "updated": updated})

    async def api_aliases(self):
        return jsonify(self.store.list_sender_aliases())

    async def api_set_alias(self):
        body = await request.get_json() or {}
        try:
            updated = await asyncio.to_thread(
                self.store.set_sender_alias,
                str(body.get("nickname", "")), str(body.get("sender_id", "")),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True, "updated": updated})

    async def api_export_summary(self):
        labels = self._export_labels()
        rows = self.store.export_rows(
            labels=labels,
            since=request.args.get("since", ""),
            until=request.args.get("until", ""),
            include_missing=True,
        )
        dedupe = request.args.get("dedupe", "1") not in {"0", "false", "False"}
        seen: set[str] = set()
        counts = {label: 0 for label in labels}
        available = 0
        missing = 0
        total_bytes = 0
        duplicates = 0
        root = self.cfg.data_dir.resolve()
        for row in rows:
            label = row.get("review_label") or row.get("top_group") or ""
            if label not in counts:
                continue
            if dedupe and row.get("sha256"):
                if row["sha256"] in seen:
                    duplicates += 1
                    continue
                seen.add(row["sha256"])
            path = Path(str(row.get("stored_path") or "")).resolve()
            if not path.is_file() or root not in path.parents or not self._readable_image(path):
                missing += 1
                continue
            counts[label] += 1
            available += 1
            total_bytes += path.stat().st_size
        return jsonify({
            "labels": labels, "counts": counts, "total": available,
            "missing": missing, "duplicates": duplicates, "size_bytes": total_bytes,
        })

    async def api_export(self):
        labels = self._export_labels()
        if not labels:
            return jsonify({"error": "请至少选择一个标注"}), 400
        rows = self.store.export_rows(
            labels=labels,
            since=request.args.get("since", ""),
            until=request.args.get("until", ""),
            include_missing=True,
        )
        dedupe = request.args.get("dedupe", "1") not in {"0", "false", "False"}
        include_missing = request.args.get("include_missing", "0") in {"1", "true", "True"}
        root = self.cfg.data_dir.resolve()
        seen: set[str] = set()
        selected: list[tuple[dict, Path, str]] = []
        missing = 0
        for row in rows:
            label = row.get("review_label") or row.get("top_group") or ""
            if label not in labels:
                continue
            if dedupe and row.get("sha256"):
                if row["sha256"] in seen:
                    continue
                seen.add(row["sha256"])
            path = Path(str(row.get("stored_path") or "")).resolve()
            if not path.is_file() or root not in path.parents or not self._readable_image(path):
                missing += 1
                continue
            selected.append((row, path, label))
        if not selected and not include_missing:
            return jsonify({"error": "没有可导出的有效图片"}), 400

        self.cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
        target = self.cfg.tmp_dir / f"dataset_{uuid.uuid4().hex}.zip"
        def build_zip() -> None:
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
                meta = io.StringIO()
                writer = csv.writer(meta)
                writer.writerow(["image_id", "label", "status", "sender_id", "sender_name", "group_id", "group_name", "score", "created_at"])
                for index, (row, path, label) in enumerate(selected, 1):
                    safe_name = f"{index:05d}_{row['id'][:8]}{path.suffix.lower() or '.jpg'}"
                    zf.write(path, f"{label}/{safe_name}")
                    writer.writerow([
                        row.get("id"), label, row.get("status"), row.get("sender_id"),
                        row.get("sender_name"), row.get("group_id"), row.get("group_name"),
                        row.get("score"), row.get("created_at"),
                    ])
                zf.writestr("metadata.csv", meta.getvalue())

        try:
            await asyncio.to_thread(build_zip)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        asyncio.create_task(self._cleanup_export(target))
        return await send_file(
            str(target), mimetype="application/zip",
            as_attachment=True, attachment_filename="dataset.zip",
        )

    @staticmethod
    async def _cleanup_export(path: Path) -> None:
        await asyncio.sleep(600)
        path.unlink(missing_ok=True)

    def _record_group_member(self, event: AstrMessageEvent, group_id: str, group_name: str) -> None:
        sender_id = str(event.get_sender_id() or "")
        raw = getattr(event.message_obj, "raw_message", None)
        raw_sender = raw.get("sender") if hasattr(raw, "get") else None
        if not isinstance(raw_sender, dict):
            raw_sender = {}
        nickname = str(raw_sender.get("nickname") or event.get_sender_name() or "")
        card = str(raw_sender.get("card") or "")
        raw_group = raw.get("group_name") if hasattr(raw, "get") else ""
        try:
            self.store.record_group_member(
                sender_id=sender_id,
                group_id=group_id,
                group_name=group_name or str(raw_group or ""),
                nickname=nickname,
                card=card,
            )
        except Exception:
            if not self._stopping:
                self.logger.exception("保存群成员信息失败")

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
