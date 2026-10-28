from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from core.config import PluginConfig
from core.forward_collector import ForwardCollector

try:
    from astrbot.api.message_components import Forward, Image as MessageImage, Json, Node, Nodes
except Exception:  # pragma: no cover - only runs inside AstrBot
    Forward = MessageImage = Json = Node = Nodes = None


@unittest.skipIf(MessageImage is None, "AstrBot components unavailable")
class ForwardCollectorTests(unittest.TestCase):
    def test_json_multimsg_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image_path = base / "json.jpg"
            Image.new("RGB", (90, 90), (10, 200, 10)).save(image_path, "JPEG")
            cfg = PluginConfig.load({"forward_dir": "training_data/forwarded"}, base)
            cfg.ensure_directories()
            collector = ForwardCollector(cfg, SimpleNamespace(info=lambda *a, **k: None,
                                                                 warning=lambda *a, **k: None,
                                                                 error=lambda *a, **k: None,
                                                                 exception=lambda *a, **k: None,
                                                                 debug=lambda *a, **k: None))
            payload = {
                "app": "com.tencent.multimsg",
                "meta": {
                    "detail": {
                        "news": [
                            {
                                "user_id": "888",
                                "nickname": "JSON节点",
                                "content": [
                                    {"type": "image", "data": {"file": str(image_path)}},
                                ],
                            },
                        ],
                    },
                },
            }
            component = Json(data={"data": json.dumps(payload, ensure_ascii=False)})
            event = SimpleNamespace(
                message_obj=SimpleNamespace(message=[component], message_id="m2", raw_message=None),
                bot=None,
                get_sender_id=lambda: "10001",
                get_sender_name=lambda: "发起人",
                get_self_id=lambda: "999",
            )

            async def run():
                await collector.start()
                try:
                    collector.arm("private:10001", 1)
                    self.assertTrue(await collector.maybe_collect(event, "private:10001", "", "私聊"))
                    await collector._queue.join()
                finally:
                    await collector.stop()

            asyncio.run(run())
            metas = list((cfg.forward_dir).rglob("metadata.jsonl"))
            self.assertEqual(len(metas), 1)
            rows = [json.loads(line) for line in metas[0].read_text(encoding="utf-8").splitlines() if line]
            saved = [r for r in rows if r["status"] == "saved"]
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["node_sender_name"], "JSON节点")
            self.assertEqual(len(list((cfg.forward_dir).rglob("*.jpg"))), 1)

    def test_message_array_node_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image_path = base / "message-array.jpg"
            Image.new("RGB", (80, 80), (3, 4, 5)).save(image_path, "JPEG")
            cfg = PluginConfig.load({"forward_dir": "training_data/forwarded"}, base)
            cfg.ensure_directories()

            class FakeBot:
                async def call_action(self, action, **kwargs):
                    return {
                        "messages": [
                            {
                                "self_id": 1,
                                "user_id": 1094950020,
                                "time": 1793160840,
                                "message_id": 1234226688,
                                "message_seq": 1234226688,
                                "real_id": 1234226688,
                                "message_type": "group",
                                "sender": {
                                    "user_id": 1094950020,
                                    "nickname": "妖",
                                    "card": "",
                                },
                                "message": [
                                    {"type": "text", "data": {"text": "@全体成员 野生男娘奶"}},
                                    {"type": "image", "data": {"file": str(image_path)}},
                                ],
                            },
                        ],
                    }

            collector = ForwardCollector(cfg, SimpleNamespace(info=lambda *a, **k: None,
                                                                 warning=lambda *a, **k: None,
                                                                 error=lambda *a, **k: None,
                                                                 exception=lambda *a, **k: None,
                                                                 debug=lambda *a, **k: None))
            event = SimpleNamespace(
                message_obj=SimpleNamespace(message=[Forward(id="99")], message_id="outer", raw_message=None),
                bot=FakeBot(),
                get_sender_id=lambda: "10001",
                get_sender_name=lambda: "发起人",
                get_self_id=lambda: "999",
            )

            async def run():
                await collector.start()
                try:
                    collector.arm("private:10001", 1)
                    self.assertTrue(await collector.maybe_collect(event, "private:10001", "", "私聊"))
                    await collector._queue.join()
                finally:
                    await collector.stop()

            asyncio.run(run())
            metas = list((cfg.forward_dir).rglob("metadata.jsonl"))
            self.assertEqual(len(metas), 1)
            rows = [json.loads(line) for line in metas[0].read_text(encoding="utf-8").splitlines() if line]
            saved = [r for r in rows if r["status"] == "saved"]
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["node_sender_name"], "妖")
            self.assertTrue(saved[0]["file"].startswith("images/"))
            self.assertEqual(len(list((cfg.forward_dir / "images").glob("*.jpg"))), 1)

    def test_nested_forward_content_does_not_fetch_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image_path = base / "nested-content.jpg"
            Image.new("RGB", (64, 64), (9, 8, 7)).save(image_path, "JPEG")
            cfg = PluginConfig.load({"forward_dir": "training_data/forwarded"}, base)
            cfg.ensure_directories()
            calls = []

            class FakeBot:
                async def call_action(self, action, **kwargs):
                    calls.append((action, kwargs))
                    if len(calls) > 1:
                        raise AssertionError("nested content must not be fetched again")
                    return {
                        "messages": [
                            {
                                "message_id": 123,
                                "sender": {"user_id": 1, "nickname": "外层"},
                                "message": [
                                    {
                                        "type": "forward",
                                        "data": {
                                            "id": "inner",
                                            "content": [
                                                {
                                                    "type": "node",
                                                    "data": {
                                                        "user_id": 2,
                                                        "nickname": "内层",
                                                        "content": [
                                                            {"type": "image", "data": {"file": str(image_path)}},
                                                        ],
                                                    },
                                                },
                                            ],
                                        },
                                    },
                                ],
                            },
                        ],
                    }

            collector = ForwardCollector(cfg, SimpleNamespace(info=lambda *a, **k: None,
                                                                 warning=lambda *a, **k: None,
                                                                 error=lambda *a, **k: None,
                                                                 exception=lambda *a, **k: None,
                                                                 debug=lambda *a, **k: None))
            event = SimpleNamespace(
                message_obj=SimpleNamespace(message=[Forward(id="outer")], message_id="m", raw_message=None),
                bot=FakeBot(),
                get_sender_id=lambda: "10001",
                get_sender_name=lambda: "发起人",
                get_self_id=lambda: "999",
            )

            async def run():
                await collector.start()
                try:
                    collector.arm("private:10001", 1)
                    self.assertTrue(await collector.maybe_collect(event, "private:10001", "", "私聊"))
                    await collector._queue.join()
                finally:
                    await collector.stop()

            asyncio.run(run())
            self.assertEqual(len(calls), 1)
            rows = [
                json.loads(line)
                for path in cfg.forward_dir.rglob("metadata.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len([r for r in rows if r["status"] == "saved"]), 1)
            self.assertEqual(len(list(cfg.forward_dir.rglob("*.jpg"))), 1)

    def test_nested_nodes_and_forward_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image_path = base / "a.jpg"
            nested_image_path = base / "b.jpg"
            Image.new("RGB", (96, 128), (30, 60, 90)).save(image_path, "JPEG")
            Image.new("RGB", (96, 128), (200, 100, 20)).save(nested_image_path, "JPEG")
            cfg = PluginConfig.load(
                {"forward_dir": "training_data/forwarded", "auto_download_model": False},
                base,
            )
            cfg.ensure_directories()

            class FakeBot:
                async def call_action(self, action, **kwargs):
                    return {
                        "messages": [
                            {
                                "type": "node",
                                "data": {
                                    "user_id": "777",
                                    "nickname": "嵌套发送者",
                                    "content": [
                                        {"type": "image", "data": {"file": str(nested_image_path)}},
                                    ],
                                },
                            },
                        ],
                    }

            collector = ForwardCollector(cfg, SimpleNamespace(info=lambda *a, **k: None,
                                                                 warning=lambda *a, **k: None,
                                                                 error=lambda *a, **k: None,
                                                                 exception=lambda *a, **k: None,
                                                                 debug=lambda *a, **k: None))
            outer = Node(
                content=[
                    MessageImage(file=str(image_path)),
                    Forward(id="99"),
                ],
                uin="123",
                name="外层",
            )
            event = SimpleNamespace(
                message_obj=SimpleNamespace(
                    message=[Nodes([outer])],
                    message_id="m1",
                ),
                bot=FakeBot(),
                get_sender_id=lambda: "10001",
                get_sender_name=lambda: "发起人",
                get_self_id=lambda: "999",
            )

            async def run():
                await collector.start()
                try:
                    self.assertEqual(collector.arm("group:g1", 1), 1)
                    self.assertTrue(await collector.maybe_collect(event, "group:g1", "g1", "测试群"))
                    await collector._queue.join()
                finally:
                    await collector.stop()

            asyncio.run(run())
            metas = list((cfg.forward_dir).rglob("metadata.jsonl"))
            self.assertEqual(len(metas), 1)
            rows = [json.loads(line) for line in metas[0].read_text(encoding="utf-8").splitlines() if line]
            saved = [r for r in rows if r["status"] == "saved"]
            self.assertEqual(len(saved), 2)
            self.assertEqual({r["node_sender_name"] for r in saved if r["depth"] > 0}, {"外层", "嵌套发送者"})
            self.assertEqual(len(list((cfg.forward_dir).rglob("*.jpg"))), 2)
            self.assertTrue((cfg.forward_dir / "index.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
