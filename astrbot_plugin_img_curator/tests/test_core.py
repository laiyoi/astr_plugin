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

from core.config import PluginConfig
from core.model import ClassifyResult
from core.pipeline import CuratorPipeline, Job
from core.scoring import NEG_START, PROMPTS, assess
from core.sources import ImageFetcher, ImageFetchError
from core.store import Store
from core.video import extract_frames, ffmpeg_available


def make_assessment(decision="saved"):
    sims = [0.0] * len(PROMPTS)
    if decision == "saved":
        sims[0] = 0.32
        sims[NEG_START] = 0.25
    elif decision == "pending":
        sims[0] = 0.26
        sims[NEG_START] = 0.25
    else:
        sims[0] = 0.20
        sims[NEG_START] = 0.30
    return assess(sims, high_threshold=0.015, pending_threshold=0.005,
                  min_positive_score=0.24, min_pending_score=0.21)


class ScoringTests(unittest.TestCase):
    def test_saved_pending_rejected(self):
        self.assertEqual(make_assessment("saved").decision, "saved")
        self.assertEqual(make_assessment("pending").decision, "pending")
        self.assertEqual(make_assessment("rejected").decision, "rejected")

    def test_min_positive_gate(self):
        sims = [0.0] * len(PROMPTS)
        sims[0] = 0.20  # margin over negative is large, but absolute score too low
        sims[NEG_START] = 0.01
        result = assess(sims, high_threshold=0.015, pending_threshold=0.005,
                        min_positive_score=0.24, min_pending_score=0.21)
        self.assertNotEqual(result.decision, "saved")

    def test_wrong_length(self):
        with self.assertRaises(ValueError):
            assess([0.0])


class ConfigTests(unittest.TestCase):
    def test_defaults_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PluginConfig.load({}, Path(tmp))
            self.assertEqual(cfg.model_name, "CLIP-ViT-B-32")
            self.assertEqual(cfg.model_variant, "fp32")
            self.assertEqual(cfg.save_dir, Path(tmp).resolve() / "images")
            self.assertEqual(cfg.max_file_bytes, 12 * 1024 * 1024)
            self.assertEqual(cfg.high_threshold, 0.011)
            cfg.ensure_directories()
            self.assertTrue(cfg.save_dir.is_dir())

    def test_threshold_order_is_clamped(self):
        cfg = PluginConfig.load({"high_threshold": 0.001, "pending_threshold": 0.5}, Path("/tmp"))
        self.assertLessEqual(cfg.pending_threshold, cfg.high_threshold)


class StoreTests(unittest.TestCase):
    def _row(self, image_id="11111111-1111-1111-1111-111111111111", sha="a" * 64, status="saved"):
        return {
            "id": image_id, "sha256": sha, "status": status, "file_name": image_id + ".jpg",
            "stored_path": "/tmp/" + image_id + ".jpg", "source": "local", "mime": "image/jpeg",
            "width": 10, "height": 10, "size_bytes": 100, "sender_id": "10001",
            "sender_name": "=HYPERLINK(\"x\")", "group_id": "123", "group_name": "测试群",
            "message_id": "m1", "score": 0.3, "margin": 0.02, "top_group": "anime_feminine",
            "top_prompt": "x", "model": "fake", "latency_ms": 12.0,
            "created_at": "2026-01-01T00:00:00+08:00",
        }

    def test_record_export_and_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "s.db", Path(tmp) / "m.csv")
            row = self._row()
            self.assertTrue(store.record_image(row))
            self.assertEqual(store.counts()["saved"], 1)
            self.assertIsNotNone(store.find_by_sha(row["sha256"]))
            store.export_csv()
            text = (Path(tmp) / "m.csv").read_text(encoding="utf-8-sig")
            header = text.splitlines()[0]
            self.assertTrue(header.startswith("image_id,"))
            self.assertIn(row["id"], text)
            self.assertIn("'=HYPERLINK", text)  # CSV injection neutralised
            store.update_review(row["id"], label="positive", status="saved",
                                file_name=row["file_name"], stored_path=row["stored_path"])
            self.assertEqual(store.counts()["saved"], 1)
            store.close()

    def test_duplicate_hash_is_allowed_when_dedupe_is_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "s.db", Path(tmp) / "m.csv")
            self.assertTrue(store.record_image(self._row()))
            second = self._row(image_id="22222222-2222-2222-2222-222222222222")
            self.assertTrue(store.record_image(second))
            self.assertEqual(store.counts()["saved"], 2)
            store.close()


class SourceTests(unittest.TestCase):
    def test_local_file_and_sha(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jpg"
            Image.new("RGB", (80, 80), (20, 30, 40)).save(path, "JPEG")
            fetcher = ImageFetcher(max_bytes=1024 * 1024, max_pixels=4_000_000, min_side=10, timeout_sec=5)
            image = SimpleNamespace(path=str(path), url="", file="")
            temp = asyncio.run(fetcher.fetch(image, Path(tmp) / "tmp"))
            try:
                self.assertEqual(temp.ext, ".jpg")
                self.assertEqual(len(temp.sha256), 64)
                self.assertEqual((temp.width, temp.height), (80, 80))
            finally:
                temp.cleanup()

    @unittest.skipUnless(ffmpeg_available(), "ffmpeg not installed")
    def test_fetch_video_local(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "src.mp4"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
                 "-pix_fmt", "yuv420p", str(video)],
                check=True,
            )
            fetcher = ImageFetcher(max_bytes=10 * 1024 * 1024, max_pixels=4_000_000, min_side=10, timeout_sec=5)
            temp = asyncio.run(fetcher.fetch_video(SimpleNamespace(path=str(video), url="", file=""),
                                                    Path(tmp) / "tmp", max_bytes=10 * 1024 * 1024))
            try:
                self.assertEqual(temp.size_bytes, video.stat().st_size)
                self.assertEqual(len(temp.sha256), 64)
            finally:
                temp.cleanup()

    def test_metadata_is_stripped(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.jpg"
            exif = Image.Exif()
            exif[0x010F] = "TestCamera"
            Image.new("RGB", (80, 80), (20, 30, 40)).save(src, "JPEG", exif=exif)
            dst = Path(tmp) / "b.jpg"
            from core.sources import sanitize_image
            sanitize_image(src, dst, ".jpg")
            with Image.open(dst) as im:
                self.assertEqual(dict(im.getexif()), {})
                self.assertEqual(im.size, (80, 80))

    def test_rejects_non_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            path.write_text("not an image")
            fetcher = ImageFetcher(max_bytes=1024, max_pixels=4_000_000, min_side=10, timeout_sec=5)
            with self.assertRaises(ImageFetchError):
                asyncio.run(fetcher.fetch(SimpleNamespace(path=str(path), url="", file=""), Path(tmp) / "tmp"))


def _quiet_logger():
    return SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
                           error=lambda *a, **k: None, exception=lambda *a, **k: None,
                           debug=lambda *a, **k: None)


class FakeModel:
    spec = SimpleNamespace(key="fake")
    def __init__(self, decision="saved"):
        self.decision = decision
        self.calls = 0
    async def classify(self, path):
        self.calls += 1
        return ClassifyResult(assessment=make_assessment(self.decision), preprocess_ms=1.0, inference_ms=2.0)


class VideoTests(unittest.TestCase):
    @unittest.skipUnless(ffmpeg_available(), "ffmpeg not installed")
    def test_extract_frames(self):
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "src.mp4"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
                 "-pix_fmt", "yuv420p", str(video)],
                check=True,
            )
            frames = extract_frames(video, Path(tmp) / "frames", max_frames=3, max_seconds=10, fps=1.0)
            self.assertGreaterEqual(len(frames), 1)
            self.assertLessEqual(len(frames), 3)
            self.assertTrue(all(f.path.is_file() for f in frames))
            self.assertEqual([f.time_sec for f in frames], sorted(f.time_sec for f in frames))
            self.assertTrue(all(f.change_score >= 0 for f in frames))


class PipelineTests(unittest.TestCase):
    def test_save_dedupe_and_csv(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            Image.new("RGB", (80, 80), (1, 2, 3)).save(base / "in.jpg", "JPEG")
            cfg = PluginConfig.load({"save_dir": "images", "pending_dir": "pending", "model_dir": "models",
                                     "csv_path": "m.csv", "db_path": "s.db", "auto_download_model": False},
                                    base)
            cfg.ensure_directories()
            store = Store(cfg.db_path, cfg.csv_path)
            model = FakeModel()
            pipeline = CuratorPipeline(cfg, store, model, SimpleNamespace(info=lambda *a, **k: None,
                                                                           warning=lambda *a, **k: None,
                                                                           error=lambda *a, **k: None,
                                                                           exception=lambda *a, **k: None,
                                                                           debug=lambda *a, **k: None))
            asyncio.run(self._run(pipeline, cfg, base, store))
            store.close()

    def test_dedupe_off_saves_twice(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            Image.new("RGB", (80, 80), (5, 6, 7)).save(base / "in.jpg", "JPEG")
            cfg = PluginConfig.load({"save_dir": "images", "pending_dir": "pending", "model_dir": "models",
                                     "csv_path": "m.csv", "db_path": "s.db", "auto_download_model": False,
                                     "dedupe": False}, base)
            cfg.ensure_directories()
            store = Store(cfg.db_path, cfg.csv_path)
            pipeline = CuratorPipeline(cfg, store, FakeModel(), _quiet_logger())
            asyncio.run(self._run_twice_no_dedupe(pipeline, cfg, base, store))
            store.close()

    async def _run_twice_no_dedupe(self, pipeline, cfg, base, store):
        await pipeline.start()
        try:
            for _ in range(2):
                await pipeline.enqueue(Job(image=SimpleNamespace(path=str(base / "in.jpg")),
                                           sender_id="1", sender_name="n", group_id="2",
                                           group_name="g", message_id="m"))
                await pipeline._queue.join()
            self.assertEqual(len(list(cfg.save_dir.iterdir())), 2)
            self.assertEqual(store.counts()["saved"], 2)
        finally:
            await pipeline.stop()

    async def _run(self, pipeline, cfg, base, store):
        await pipeline.start()
        try:
            for _ in range(2):
                await pipeline.enqueue(Job(image=SimpleNamespace(path=str(base / "in.jpg")),
                                           sender_id="1", sender_name="n", group_id="2",
                                           group_name="g", message_id="m"))
                await pipeline._queue.join()
            files = list(cfg.save_dir.iterdir())
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].name.endswith(".jpg"))
            self.assertEqual(store.counts()["saved"], 1)
            self.assertEqual(store.get_meta("duplicates"), "1")
            text = cfg.csv_path.read_text(encoding="utf-8-sig")
            self.assertIn(files[0].stem, text)
        finally:
            await pipeline.stop()


if __name__ == "__main__":
    unittest.main()
