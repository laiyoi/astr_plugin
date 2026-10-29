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
    return (
        str(obj.get("user_id") or sender.get("user_id") or fallback_id or ""),
        str(obj.get("nickname") or sender.get("nickname") or sender.get("card") or fallback_name or ""),
    )


class ForwardMediaExtractor:
    def __init__(self, bot: object, logger, max_depth: int = 8):
        self.bot = bot
        self.logger = logger
        self.max_depth = max_depth
        self._seen_forwards: set[str] = set()

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
            content = data.get("content") or data.get("messages") or []
            if isinstance(content, list):
                for item in content:
                    await self._walk_payload(item, out, depth + 1, node_id, node_name, message_id)
        elif seg_type in {"forward", "forward_msg"}:
            content = data.get("content") or data.get("messages") or []
            if isinstance(content, list) and content:
                for item in content:
                    await self._walk_payload(item, out, depth + 1, node_id, node_name, message_id)
            elif data.get("id") or data.get("message_id"):
                nested = await self._fetch_forward(data.get("id") or data.get("message_id"))
                await self._walk_payload(nested, out, depth + 1, node_id, node_name, message_id)
        elif seg_type in {"json", "xml"}:
            parsed = _load_json(data)
            if parsed is not None:
                await self._walk_payload(parsed, out, depth + 1, node_id, node_name, message_id)

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
        self.logger.debug("forward no payload: %s", fid)
        return []
