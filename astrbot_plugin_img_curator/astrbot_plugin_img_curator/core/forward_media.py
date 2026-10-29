"""Extract image/video refs from merged/forwarded chat records.

The first ``get_forward_msg`` response normally contains the complete nested
tree.  This module only walks that tree; it does not save training files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from astrbot.api.message_components import Forward, Image, Json, Node, Nodes, Video


@dataclass
class MediaRef:
    component: object
    sender_id: str = ""
    sender_name: str = ""
    media_type: str = "image"
    message_id: str = ""


def _load_json(value: Any) -> Any:
    if isinstance(value, dict):
        if isinstance(value.get("data"), (str, dict, list)):
            return _load_json(value["data"])
        return value
    if isinstance(value, str):
        text = value.replace("&#44;", ",").strip()
        if not text or text[0] not in "[{":
            return None
        try:
            return _load_json(json.loads(text))
        except Exception:
            return None
    return value


def _sender(obj: dict[str, Any], fallback_id: str = "", fallback_name: str = "") -> tuple[str, str]:
    sender = obj.get("sender") if isinstance(obj.get("sender"), dict) else {}
    # In a forward node the top-level user_id can be the container/bot id;
    # the actual image sender is inside ``sender``.  Prefer sender fields.
    raw_id = (
        sender.get("user_id") or sender.get("uin") or sender.get("qq")
        or sender.get("uid") or sender.get("sender_id") or sender.get("senderId")
        or obj.get("user_id") or obj.get("uin") or obj.get("qq") or obj.get("uid")
        or obj.get("sender_id") or obj.get("senderId")
        or fallback_id or ""
    )
    raw_name = (
        sender.get("nickname") or sender.get("card") or sender.get("name")
        or sender.get("sender_name") or sender.get("senderName")
        or obj.get("nickname") or obj.get("card") or obj.get("name")
        or obj.get("sender_name") or obj.get("senderName")
        or fallback_name or ""
    )
    return str(raw_id), str(raw_name)


class ForwardMediaExtractor:
    def __init__(self, bot: object, logger, max_depth: int = 8):
        self.bot = bot
        self.logger = logger
        self.max_depth = max_depth
        self._seen_forwards: set[str] = set()
        self._message_sender_cache: dict[str, tuple[str, str]] = {}

    async def from_event(self, event) -> list[MediaRef]:
        refs: list[MediaRef] = []
        for component in getattr(event.message_obj, "message", None) or []:
            await self._walk_astrbot(component, refs, 0, "", "", str(getattr(event.message_obj, "message_id", "") or ""))
        return refs

    async def _walk_astrbot(
        self,
        component: object,
        out: list[MediaRef],
        depth: int,
        sender_id: str,
        sender_name: str,
        message_id: str,
    ) -> None:
        if depth > self.max_depth:
            return
        if isinstance(component, Image):
            out.append(MediaRef(component, sender_id, sender_name, "image", message_id))
        elif isinstance(component, Video):
            out.append(MediaRef(component, sender_id, sender_name, "video", message_id))
        elif isinstance(component, Node):
            for child in component.content or []:
                await self._walk_astrbot(
                    child, out, depth + 1, str(component.uin or sender_id), str(component.name or sender_name), message_id
                )
        elif isinstance(component, Nodes):
            for node in component.nodes or []:
                for child in node.content or []:
                    await self._walk_astrbot(
                        child, out, depth + 1, str(node.uin or sender_id), str(node.name or sender_name), message_id
                    )
        elif isinstance(component, Json):
            parsed = _load_json(getattr(component, "data", None))
            if parsed is not None:
                await self._walk_payload(parsed, out, depth + 1, sender_id, sender_name, message_id)
        elif isinstance(component, Forward):
            embedded = getattr(component, "content", None)
            if isinstance(embedded, list) and embedded:
                for child in embedded:
                    await self._walk_astrbot(child, out, depth + 1, sender_id, sender_name, message_id)
                return
            payload = await self._fetch_forward(component.id)
            await self._walk_payload(payload, out, depth + 1, sender_id, sender_name, message_id)

    async def _walk_payload(
        self,
        payload: Any,
        out: list[MediaRef],
        depth: int,
        sender_id: str,
        sender_name: str,
        message_id: str,
    ) -> None:
        if depth > self.max_depth:
            return
        if isinstance(payload, list):
            for item in payload:
                await self._walk_payload(item, out, depth + 1, sender_id, sender_name, message_id)
            return
        if not isinstance(payload, dict):
            return

        node_id, node_name = _sender(payload, sender_id, sender_name)
        if node_id in {"", "1094950020"}:
            node_id, node_name = await self._resolve_message_sender(payload, node_id, node_name)
        if isinstance(payload.get("message"), list):
            for item in payload["message"]:
                await self._walk_payload(item, out, depth + 1, node_id, node_name, message_id)
            return
        if isinstance(payload.get("content"), list) and "type" not in payload:
            for item in payload["content"]:
                await self._walk_payload(item, out, depth + 1, node_id, node_name, message_id)
            return

        seg_type = payload.get("type")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        if seg_type == "image":
            out.append(
                MediaRef(
                    SimpleNamespace(
                        url=str(data.get("url") or ""),
                        path=str(data.get("path") or ""),
                        file=str(data.get("file") or ""),
                    ),
                    node_id,
                    node_name,
                    "image",
                    message_id,
                )
            )
        elif seg_type == "video":
            out.append(
                MediaRef(
                    SimpleNamespace(
                        url=str(data.get("url") or ""),
                        path=str(data.get("path") or ""),
                        file=str(data.get("file") or ""),
                    ),
                    node_id,
                    node_name,
                    "video",
                    message_id,
                )
            )
        elif seg_type in {"node", "nodes"}:
            # OneBot forward nodes put the original sender in data.sender /
            # data.user_id (NapCat also exposes uin/qq); do not inherit the
            # forwarder's fallback ID for every nested image.
            node_id2, node_name2 = _sender(
                data if data else payload, node_id, node_name
            )
            if node_id2 in {"", "1094950020"}:
                node_id2, node_name2 = await self._resolve_message_sender(data, node_id2, node_name2)
            content = data.get("content") or data.get("messages") or data.get("message") or []
            if isinstance(content, dict):
                content = content.get("message") or content.get("content") or content.get("messages") or []
            if isinstance(content, list):
                for item in content:
                    await self._walk_payload(item, out, depth + 1, node_id2, node_name2, message_id)
        elif seg_type in {"forward", "forward_msg"}:
            content = data.get("content") or data.get("messages") or []
            # Prefer the nested forward id. Some NapCat responses embed a
            # content snapshot whose node sender.user_id is a bot/fallback
            # account, while get_forward_msg can return the real senders.
            fetch_id = data.get("id") or data.get("message_id")
            nested = await self._fetch_forward(fetch_id) if fetch_id else []
            if nested:
                await self._walk_payload(nested, out, depth + 1, node_id, node_name, message_id)
            elif isinstance(content, list) and content:
                for item in content:
                    await self._walk_payload(item, out, depth + 1, node_id, node_name, message_id)
        elif seg_type in {"json", "xml"}:
            parsed = _load_json(data)
            if parsed is not None:
                await self._walk_payload(parsed, out, depth + 1, node_id, node_name, message_id)

    async def _resolve_message_sender(
        self, payload: dict[str, Any], fallback_id: str, fallback_name: str
    ) -> tuple[str, str]:
        """Ask OneBot for the original message when a forward node is anonymous."""
        mid = payload.get("message_id") or payload.get("real_id")
        if not mid or self.bot is None:
            return fallback_id, fallback_name
        key = str(mid)
        if key in self._message_sender_cache:
            return self._message_sender_cache[key]
        resolved = (fallback_id, fallback_name)
        try:
            response = await self.bot.call_action("get_msg", message_id=key)
            data = response.get("data") if isinstance(response, dict) else None
            source = data if isinstance(data, dict) else response
            if isinstance(source, dict):
                sender = source.get("sender") if isinstance(source.get("sender"), dict) else {}
                uid = str(sender.get("user_id") or source.get("user_id") or "")
                name = str(sender.get("card") or sender.get("nickname") or source.get("nickname") or "")
                if uid and uid != "1094950020":
                    resolved = (uid, name or fallback_name)
        except Exception:
            pass
        self._message_sender_cache[key] = resolved
        return resolved

    async def _fetch_forward(self, forward_id: Any) -> list[Any]:
        fid = str(forward_id or "")
        if not fid or fid in self._seen_forwards or self.bot is None:
            return []
        self._seen_forwards.add(fid)
        for action, kwargs in (
            ("get_forward_msg", {"id": fid}),
            ("get_forward_msg", {"message_id": fid}),
            ("get_msg", {"message_id": fid}),
        ):
            try:
                payload = await self.bot.call_action(action, **kwargs)
            except Exception:
                continue
            if isinstance(payload, dict):
                for key in ("messages", "message"):
                    if isinstance(payload.get(key), list):
                        return payload[key]
                data = payload.get("data")
                if isinstance(data, dict):
                    for key in ("messages", "message"):
                        if isinstance(data.get(key), list):
                            return data[key]
        return []
