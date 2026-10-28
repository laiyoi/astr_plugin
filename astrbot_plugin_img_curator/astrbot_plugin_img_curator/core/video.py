"""Short-video frame extraction with ffmpeg.

Frames are first sampled at a higher candidate rate, then each temporal window
contributes the frame with the largest visual change.  This gives both uniform
coverage and preference for "reference-like" frames without trying to identify
actual reference frames.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv"}


class FrameExtractionError(Exception):
    pass


@dataclass(frozen=True)
class VideoFrame:
    path: Path
    time_sec: float
    change_score: float = 0.0


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _frame_signature(path: Path, size: int = 64) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def _select_uniform_high_change(frames: list[VideoFrame], scores: list[float], max_frames: int) -> list[VideoFrame]:
    """One highest-change frame per equal temporal window.

    The windows are based on candidate order (approximately time).  If there
    are fewer candidates than max_frames, all are kept.
    """
    if not frames:
        return []
    count = min(max_frames, len(frames))
    selected: list[VideoFrame] = []
    for window_index in range(count):
        start = round(window_index * len(frames) / count)
        end = round((window_index + 1) * len(frames) / count)
        end = max(end, start + 1)
        best_index = max(range(start, min(end, len(frames))), key=lambda i: (scores[i], -i))
        frame = frames[best_index]
        selected.append(VideoFrame(frame.path, frame.time_sec, float(scores[best_index])))
    return selected


def extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    max_frames: int = 8,
    max_seconds: int = 15,
    fps: float = 1.0,
    width: int = 1280,
    timeout_sec: int = 60,
    candidate_fps: float = 4.0,
    max_candidates: int = 64,
    change_size: int = 64,
) -> list[VideoFrame]:
    """Extract up to max_frames, one per equal window, preferring visual change.

    fps is kept for compatibility and is the minimum candidate rate: the
    actual candidate rate is max(fps, candidate_fps).  The output frame count is
    still capped by max_frames.
    """
    if not ffmpeg_available():
        raise FrameExtractionError("ffmpeg is not installed")
    if max_frames < 1:
        raise FrameExtractionError("max_frames must be >= 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    rate = max(float(fps), float(candidate_fps))
    vf = f"fps={rate:g},scale='min({width},iw)':-2"
    pattern = out_dir / "candidate_%04d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-t",
        str(max_seconds),
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-frames:v",
        str(max_candidates),
        "-q:v",
        "2",
        str(pattern),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FrameExtractionError("ffmpeg timed out") from exc
    except OSError as exc:
        raise FrameExtractionError(f"ffmpeg failed to start: {exc}") from exc
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", "replace").strip()[-300:]
        raise FrameExtractionError(f"ffmpeg exited {proc.returncode}: {message}")
    files = sorted(out_dir.glob("candidate_*.jpg"))
    if not files:
        raise FrameExtractionError("ffmpeg produced no frames")

    frames: list[VideoFrame] = []
    previous: np.ndarray | None = None
    scores: list[float] = []
    for index, path in enumerate(files):
        signature = _frame_signature(path, change_size)
        score = float(np.mean(np.abs(signature - previous))) if previous is not None else 0.0
        previous = signature
        frames.append(VideoFrame(path=path, time_sec=index / max(rate, 0.001), change_score=score))
        scores.append(score)

    selected = _select_uniform_high_change(frames, scores, max_frames)
    # Keep only selected files; this keeps temp cleanup cheap and avoids
    # accidentally passing unselected candidates to the model.
    selected_paths = {frame.path for frame in selected}
    for frame in frames:
        if frame.path not in selected_paths:
            frame.path.unlink(missing_ok=True)
    return selected
