#!/usr/bin/env python3
"""Offline benchmark for the ONNX curator model.

Usage (inside the AstrBot environment, from the plugin directory):

  python benchmark/benchmark.py \
      --dataset /path/to/labeled/images \
      --model-root /path/to/plugin_data/models \
      --model CLIP-ViT-B-32 --variant int8 \
      --json-out benchmark/results.json

The dataset can be either::

  dataset/real_crossdress/xxx.jpg  # label = directory name
  dataset/anime_crossdress/yyy.jpg
  dataset/negative/zzz.jpg

or a flat directory with ``--manifest manifest.jsonl`` where each line has
``{"path": "samples/a.jpg", "label": "anime_crossdress"}``.  Positive labels
default to ``real_crossdress,anime_crossdress`` and can be overridden with
``--positive-labels``.

This script never writes into the archive and never downloads weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.model import ClipOnnxModel  # noqa: E402
from core.video import VIDEO_SUFFIXES, extract_frames  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES


def collect(dataset: Path, manifest: Path | None) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    if manifest:
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                path = Path(row["path"])
                if not path.is_absolute():
                    candidates = [
                        manifest.parent / path,
                        manifest.parent.parent / path,
                        ROOT / path,
                        Path.cwd() / path,
                    ]
                    path = next((p for p in candidates if p.is_file()), candidates[0])
                if path.is_file():
                    items.append((path, str(row.get("label", "unknown"))))
        return items
    for path in sorted(dataset.rglob("*")):
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES:
            items.append((path, path.parent.name))
    return items


def auc(y: list[int], score: list[float]) -> float | None:
    pos = sum(y)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(score)), key=lambda i: score[i])
    ranks = [0.0] * len(score)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and score[order[j + 1]] == score[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(ranks[i] for i in range(len(y)) if y[i])
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "samples")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--model-root", type=Path, required=True, help="plugin data models/ directory")
    ap.add_argument("--model", default="CLIP-ViT-B-32")
    ap.add_argument("--variant", default="fp32", choices=["int8", "fp32"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--download", action="store_true", help="allow downloading the ONNX weights into --model-root")
    ap.add_argument("--high", type=float, default=None)
    ap.add_argument("--pending", type=float, default=None)
    ap.add_argument("--video-frames", type=int, default=8)
    ap.add_argument("--video-seconds", type=int, default=15)
    ap.add_argument("--video-fps", type=float, default=1.0)
    ap.add_argument("--min-positive", type=float, default=0.240)
    ap.add_argument("--min-pending", type=float, default=0.210)
    ap.add_argument("--positive-labels", default="real_crossdress,anime_crossdress")
    ap.add_argument("--json-out", type=Path, default=ROOT / "benchmark" / "results.json")
    ap.add_argument("--csv-out", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    items = collect(args.dataset, args.manifest)
    if not items:
        print("no labelled images found", file=sys.stderr)
        return 2
    positives = {x.strip() for x in args.positive_labels.split(",") if x.strip()}
    high = args.high if args.high is not None else 0.011
    pending = args.pending if args.pending is not None else 0.005

    model = ClipOnnxModel(
        model_name=args.model,
        variant=args.variant,
        model_root=args.model_root,
        cpu_threads=args.threads,
        auto_download=args.download,
        download_timeout=600,
        high_threshold=high,
        pending_threshold=pending,
        min_positive_score=args.min_positive,
        min_pending_score=args.min_pending,
        logger=logging.getLogger("benchmark"),
    )
    t0 = time.perf_counter()
    import asyncio

    asyncio.run(model.ensure_ready())
    load_seconds = time.perf_counter() - t0

    def make_row(path: Path, label: str, result, *, media_type: str, frames: int = 1, frame_time: float | None = None):
        a = result.assessment
        return {
            "path": str(path),
            "label": label,
            "target": label in positives,
            "media_type": media_type,
            "frames": frames,
            "frame_time": frame_time,
            "decision": a.decision,
            "margin": a.margin,
            "positive_score": a.positive_score,
            "negative_score": a.negative_score,
            "top_group": a.top_group,
            "top_prompt": a.top_prompt,
            "inference_ms": result.inference_ms,
            "preprocess_ms": result.preprocess_ms,
        }

    rows = []
    for path, label in items:
        try:
            if path.suffix.lower() in VIDEO_SUFFIXES:
                with tempfile.TemporaryDirectory(prefix="imgcurator-video-") as tmp:
                    frames = extract_frames(
                        path,
                        Path(tmp),
                        max_frames=args.video_frames,
                        max_seconds=args.video_seconds,
                        fps=args.video_fps,
                    )
                    best = None
                    for frame in frames:
                        result = model.classify_sync(frame.path)
                        if best is None or (
                            result.assessment.margin,
                            result.assessment.positive_score,
                        ) > (
                            best[0].assessment.margin,
                            best[0].assessment.positive_score,
                        ):
                            best = (result, frame)
                    if best is None:
                        raise RuntimeError("video produced no classifiable frames")
                    rows.append(
                        make_row(
                            path,
                            label,
                            best[0],
                            media_type="video",
                            frames=len(frames),
                            frame_time=round(best[1].time_sec, 2),
                        )
                    )
            else:
                rows.append(make_row(path, label, model.classify_sync(path), media_type="image"))
        except Exception as exc:  # noqa: BLE001
            rows.append({"path": str(path), "label": label, "error": str(exc)})

    ok = [r for r in rows if "error" not in r]
    y = [1 if r["target"] else 0 for r in ok]
    margins = [r["margin"] for r in ok]
    latencies = sorted(r["inference_ms"] for r in ok)
    tp = sum(1 for r in ok if r["target"] and r["decision"] == "saved")
    fp = sum(1 for r in ok if not r["target"] and r["decision"] == "saved")
    fn = sum(1 for r in ok if r["target"] and r["decision"] != "saved")
    tn = sum(1 for r in ok if not r["target"] and r["decision"] != "saved")
    auto_precision = tp / (tp + fp) if tp + fp else None
    auto_recall = tp / (tp + fn) if tp + fn else None
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))] if latencies else 0.0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    summary = {
        "model": model.spec.display,
        "model_key": model.spec.key,
        "thresholds": {"high": high, "pending": pending, "min_positive": args.min_positive, "min_pending": args.min_pending},
        "count": len(rows),
        "videos": sum(1 for r in ok if r.get("media_type") == "video"),
        "errors": len(rows) - len(ok),
        "targets": sum(y),
        "auto_saved": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "auto_precision": auto_precision,
        "auto_recall": auto_recall,
        "auc_margin": auc(y, margins),
        "pending": sum(1 for r in ok if r["decision"] == "pending"),
        "rejected": sum(1 for r in ok if r["decision"] == "rejected"),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p95": p95,
            "max": max(latencies) if latencies else 0.0,
        },
        "load_seconds": load_seconds,
        "model_status": model.status(),
        "peak_rss_mb": peak_mb,
        "rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_out, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "path", "label", "target", "media_type", "frames", "frame_time",
                    "decision", "margin", "positive_score", "negative_score",
                    "top_group", "top_prompt", "inference_ms", "preprocess_ms",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows({k: r.get(k, "") for k in writer.fieldnames} for r in rows)

    print(json.dumps({k: v for k, v in summary.items() if k not in {"rows", "model_status"}}, ensure_ascii=False, indent=2))
    print(f"results -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
