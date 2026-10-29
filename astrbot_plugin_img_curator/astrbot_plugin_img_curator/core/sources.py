"""Fetch a QQ image into a bounded local temporary file.

Only rasters accepted by the configured allow-list reach the model.  The
downloader enforces the byte limit while streaming, so a malicious or huge
image cannot fill the disk or RAM.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from html import unescape
from urllib.parse import unquote, urlparse

import aiohttp
from PIL import Image, ImageOps, UnidentifiedImageError

# GIF/BMP are decoded to their first frame and re-encoded as JPEG.  This
# keeps animated GIFs from becoming unbounded frame sequences.
ALLOWED_FORMATS = {
    "JPEG": (".jpg", "image/jpeg"),
    "PNG": (".png", "image/png"),
    "WEBP": (".webp", "image/webp"),
    "GIF": (".jpg", "image/jpeg"),
    "BMP": (".jpg", "image/jpeg"),
}
TRAINING_FORMATS = {
    **ALLOWED_FORMATS,
    "GIF": (".gif", "image/gif"),
    "BMP": (".bmp", "image/bmp"),
}


class ImageFetchError(Exception):
    pass


@dataclass
class TempVideo:
    path: Path
    sha256: str
    size_bytes: int
    source: str
    ext: str = ".mp4"
    mime: str = "video/mp4"

    def cleanup(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


@dataclass
class TempImage:
    path: Path
    sha256: str
    mime: str
    ext: str
    width: int
    height: int
    size_bytes: int
    source: str

    def cleanup(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def _check_pixel_budget(max_pixels: int) -> None:
    Image.MAX_IMAGE_PIXELS = max_pixels


def _decode_base64(value: str, max_bytes: int) -> bytes:
    if value.startswith("base64://"):
        value = value[9:]
    elif value.startswith("data:"):
        _, _, value = value.partition(",")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageFetchError("invalid base64 image") from exc
    if len(data) > max_bytes:
        raise ImageFetchError(f"base64 image exceeds {max_bytes} bytes")
    return data


def _local_path(value: str) -> Path | None:
    if value.startswith("file://"):
        parsed = urlparse(value)
        return Path(unquote(parsed.path))
    if value.startswith("/"):
        return Path(value)
    return None


class ImageFetcher:
    def __init__(self, *, max_bytes: int, max_pixels: int, min_side: int, timeout_sec: int):
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels
        self.min_side = min_side
        self.timeout = aiohttp.ClientTimeout(total=timeout_sec, connect=10, sock_read=timeout_sec)
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=4, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout,
                headers={"User-Agent": "AstrBot-ImgCurator/1.0"},
                trust_env=True,
            )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    @staticmethod
    def _candidates(image: object) -> list[str]:
        values: list[str] = []
        for attr in ("url", "path", "file"):
            value = getattr(image, attr, None)
            if isinstance(value, str) and value.strip():
                values.append(unescape(value.strip()))
        # Preserve order but remove duplicates.
        seen: set[str] = set()
        result = []
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result

    async def _http_bytes(self, url: str, max_bytes: int | None = None) -> bytes:
        if self._session is None or self._session.closed:
            await self.start()
        assert self._session is not None
        limit = max_bytes if max_bytes is not None else self.max_bytes
        chunks: list[bytes] = []
        total = 0
        try:
            async with self._session.get(url, allow_redirects=True) as resp:
                if resp.status != 200:
                    raise ImageFetchError(f"HTTP {resp.status}")
                length = resp.content_length
                if length and length > limit:
                    raise ImageFetchError(f"content-length {length} exceeds limit")
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    total += len(chunk)
                    if total > limit:
                        raise ImageFetchError(f"media exceeds {limit} bytes")
                    chunks.append(chunk)
        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            raise ImageFetchError(f"download failed: {exc}") from exc
        return b"".join(chunks)

    def _validate(self, path: Path) -> tuple[str, str, int, int]:
        _check_pixel_budget(self.max_pixels)
        try:
            with Image.open(path) as im:
                im.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageFetchError("not a readable image") from exc
        try:
            with Image.open(path) as im:
                fmt = (im.format or "").upper()
                width, height = im.size
                frames = int(getattr(im, "n_frames", 1) or 1)
                # Apply EXIF orientation to the dimensions we record.
                try:
                    oriented = ImageOps.exif_transpose(im)
                except Exception:  # noqa: BLE001 - orientation is advisory
                    oriented = im
                width, height = oriented.size
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageFetchError("not a readable image") from exc
        if fmt not in ALLOWED_FORMATS:
            raise ImageFetchError(f"unsupported format {fmt or 'unknown'}")
        if width * height > self.max_pixels:
            raise ImageFetchError(f"image pixels {width}x{height} exceed limit {self.max_pixels}")
        if frames > 1 and fmt != "GIF":
            raise ImageFetchError("animated images are not supported")
        if width < self.min_side or height < self.min_side:
            raise ImageFetchError(f"image too small ({width}x{height})")
        ext, mime = ALLOWED_FORMATS[fmt]
        return ext, mime, width, height

    async def fetch_video(self, video: object, tmp_dir: Path, *, max_bytes: int) -> TempVideo:
        """Materialize a video reference without decoding it.

        ffmpeg does the decoding later, with its own frame/duration bounds.
        """
        tmp_dir.mkdir(parents=True, exist_ok=True)
        raw: bytes | None = None
        local: Path | None = None
        source_value = ""
        for value in self._candidates(video):
            if value.startswith(("base64://", "data:")):
                raw = _decode_base64(value, max_bytes)
                source_value = value
                break
            if value.startswith(("http://", "https://")):
                raw = await self._http_bytes(value, max_bytes=max_bytes)
                source_value = value
                break
            candidate = _local_path(value)
            if candidate and candidate.is_file():
                local = candidate
                source_value = value
                break
        if raw is None and local is None:
            raise ImageFetchError("no usable video URL, base64 payload, or local path")
        video_ext = ".mp4"
        video_mime = "video/mp4"
        suffix = Path(urlparse(source_value).path).suffix.lower() if source_value else ""
        known = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska"}
        if suffix in known:
            video_ext, video_mime = suffix, known[suffix]

        tmp_path = tmp_dir / f"{uuid.uuid4().hex}.video"
        try:
            if raw is not None:
                if len(raw) > max_bytes:
                    raise ImageFetchError(f"video exceeds {max_bytes} bytes")
                await asyncio.to_thread(tmp_path.write_bytes, raw)
            else:
                assert local is not None
                if local.stat().st_size > max_bytes:
                    raise ImageFetchError(f"local video exceeds {max_bytes} bytes")
                await asyncio.to_thread(_copy_file, local, tmp_path)
            size = tmp_path.stat().st_size
            digest = await asyncio.to_thread(_sha256_file, tmp_path)
            return TempVideo(
                path=tmp_path,
                sha256=digest,
                size_bytes=size,
                source="http" if raw is not None else "local",
                ext=video_ext,
                mime=video_mime,
            )
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    async def fetch(self, image: object, tmp_dir: Path) -> TempImage:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        raw: bytes | None = None
        local: Path | None = None
        last_error: ImageFetchError | None = None
        for value in self._candidates(image):
            try:
                if value.startswith(("base64://", "data:")):
                    raw = _decode_base64(value, self.max_bytes)
                    break
                if value.startswith(("http://", "https://")):
                    raw = await self._http_bytes(value)
                    break
                candidate = _local_path(value)
                if candidate and candidate.is_file():
                    local = candidate
                    break
            except ImageFetchError as exc:
                last_error = exc
                continue

        if raw is None and local is None:
            if last_error is not None:
                raise last_error
            raise ImageFetchError("no usable image URL, base64 payload, or local path")

        tmp_path = tmp_dir / f"{uuid.uuid4().hex}.download"
        try:
            if raw is not None:
                if len(raw) > self.max_bytes:
                    raise ImageFetchError(f"image exceeds {self.max_bytes} bytes")
                await asyncio.to_thread(tmp_path.write_bytes, raw)
            else:
                assert local is not None
                size = local.stat().st_size
                if size > self.max_bytes:
                    raise ImageFetchError(f"local image exceeds {self.max_bytes} bytes")
                await asyncio.to_thread(_copy_file, local, tmp_path)

            ext, mime, width, height = await asyncio.to_thread(self._validate, tmp_path)
            size = tmp_path.stat().st_size
            digest = await asyncio.to_thread(_sha256_file, tmp_path)
            # All temp files end in .download while being validated; the
            # pipeline moves them to a UUID name ending in the real extension.
            final = tmp_path.with_suffix(".img")
            os.replace(tmp_path, final)
            return TempImage(
                path=final,
                sha256=digest,
                mime=mime,
                ext=ext,
                width=width,
                height=height,
                size_bytes=size,
                source="http" if raw is not None else "local",
            )
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


    async def fetch_raw(self, image: object, tmp_dir: Path) -> TempImage:
        """Fetch an image for dataset storage without model-side validation.

        Training collection keeps the original raster, including animated GIF
        and WebP; it does not re-encode or extract a single frame.
        """
        tmp_dir.mkdir(parents=True, exist_ok=True)
        raw: bytes | None = None
        local: Path | None = None
        for value in self._candidates(image):
            if value.startswith(("base64://", "data:")):
                raw = _decode_base64(value, self.max_bytes)
                break
            if value.startswith(("http://", "https://")):
                raw = await self._http_bytes(value)
                break
            candidate = _local_path(value)
            if candidate and candidate.is_file():
                local = candidate
                break

        if raw is None and local is None:
            raise ImageFetchError("no usable image URL, base64 payload, or local path")

        tmp_path = tmp_dir / f"{uuid.uuid4().hex}.download"
        try:
            if raw is not None:
                if len(raw) > self.max_bytes:
                    raise ImageFetchError(f"image exceeds {self.max_bytes} bytes")
                await asyncio.to_thread(tmp_path.write_bytes, raw)
            else:
                assert local is not None
                size = local.stat().st_size
                if size > self.max_bytes:
                    raise ImageFetchError(f"local image exceeds {self.max_bytes} bytes")
                await asyncio.to_thread(_copy_file, local, tmp_path)

            try:
                with Image.open(tmp_path) as im:
                    fmt = im.format
                    width, height = im.size
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise ImageFetchError("not a readable image") from exc
            if fmt not in TRAINING_FORMATS:
                raise ImageFetchError(f"unsupported image format {fmt or 'unknown'}")
            ext, mime = TRAINING_FORMATS[fmt]
            digest = await asyncio.to_thread(_sha256_file, tmp_path)
            size = tmp_path.stat().st_size
            final = tmp_path.with_suffix(".img")
            os.replace(tmp_path, final)
            return TempImage(
                path=final,
                sha256=digest,
                mime=mime,
                ext=ext,
                width=width,
                height=height,
                size_bytes=size,
                source="http" if raw is not None else "local",
            )
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


def sanitize_image(src: Path, dst: Path, ext: str) -> None:
    """Re-encode without EXIF/GPS/text metadata (privacy default)."""
    fmt = {"jpg": "JPEG", "png": "PNG", "webp": "WEBP"}.get(ext.lstrip(".").lower())
    if fmt is None:
        raise ImageFetchError(f"unsupported sanitize target {ext}")
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            if fmt == "JPEG":
                im = im.convert("RGB")
                im.save(dst, format="JPEG", quality=95, optimize=True)
            elif fmt == "PNG":
                im.save(dst, format="PNG", optimize=True)
            else:
                im.save(dst, format="WEBP", quality=95, method=4)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        dst.unlink(missing_ok=True)
        raise ImageFetchError("failed to sanitize image metadata") from exc


def _copy_file(src: Path, dst: Path) -> None:
    with open(src, "rb") as rf, open(dst, "wb") as wf:
        while True:
            chunk = rf.read(256 * 1024)
            if not chunk:
                break
            wf.write(chunk)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(256 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
