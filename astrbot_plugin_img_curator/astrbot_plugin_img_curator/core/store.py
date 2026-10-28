"""SQLite state + CSV metadata export.

SQLite is the source of truth (dedupe, status, counters, latency samples).
The CSV is regenerated atomically from SQLite so a review updates the row
instead of appending a second, contradictory one.
"""

from __future__ import annotations

import csv
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

CSV_COLUMNS = [
    "image_id",
    "file_name",
    "status",
    "source_type",
    "frame_time",
    "review_label",
    "score",
    "margin",
    "top_group",
    "top_prompt",
    "sender_id",
    "sender_name",
    "group_id",
    "group_name",
    "message_id",
    "sha256",
    "mime",
    "width",
    "height",
    "size_bytes",
    "model",
    "latency_ms",
    "created_at",
    "reviewed_at",
    "stored_path",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _csv_safe(value: Any) -> str:
    """Neutralise spreadsheet formulas in free-form fields."""
    if value is None:
        return ""
    text = str(value)
    if text and text[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + text
    return text


class Store:
    def __init__(self, db_path: Path, csv_path: Path):
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        self._ensure_columns()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_type TEXT,
                    frame_time REAL,
                    file_name TEXT,
                    stored_path TEXT,
                    source TEXT,
                    mime TEXT,
                    width INTEGER,
                    height INTEGER,
                    size_bytes INTEGER,
                    sender_id TEXT,
                    sender_name TEXT,
                    group_id TEXT,
                    group_name TEXT,
                    message_id TEXT,
                    score REAL,
                    margin REAL,
                    top_group TEXT,
                    top_prompt TEXT,
                    model TEXT,
                    latency_ms REAL,
                    review_label TEXT,
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
                CREATE INDEX IF NOT EXISTS idx_images_created ON images(created_at);
                CREATE INDEX IF NOT EXISTS idx_images_sha ON images(sha256);
                CREATE TABLE IF NOT EXISTS seen (
                    sha256 TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score REAL,
                    margin REAL,
                    latency_ms REAL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    PRIMARY KEY (sha256, image_id)
                );
                CREATE INDEX IF NOT EXISTS idx_seen_status ON seen(status);
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def _ensure_columns(self) -> None:
        """Forward-compatible migration for the two video columns."""
        with self._lock:
            columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(images)")}
            if "source_type" not in columns:
                self._conn.execute("ALTER TABLE images ADD COLUMN source_type TEXT")
            if "frame_time" not in columns:
                self._conn.execute("ALTER TABLE images ADD COLUMN frame_time REAL")

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    # -- meta / counters -------------------------------------------------
    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )

    def get_meta(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def incr_meta(self, key: str, amount: int = 1) -> int:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=CAST(CAST(meta.value AS INTEGER) + ? AS TEXT)",
                (key, str(amount), amount),
            )
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return 0

    def get_paused(self) -> bool:
        return self.get_meta("paused", "0") == "1"

    def set_paused(self, paused: bool) -> None:
        self.set_meta("paused", "1" if paused else "0")

    # -- images ----------------------------------------------------------
    def find_by_sha(self, sha256: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM seen WHERE sha256=?", (sha256,)).fetchone()
        return dict(row) if row else None

    def mark_seen(
        self,
        sha256: str,
        *,
        status: str,
        image_id: str | None = None,
        score: float | None = None,
        margin: float | None = None,
        latency_ms: float | None = None,
    ) -> bool:
        """Record a hash event. Returns False only when the same (hash, id) exists."""
        if image_id is None:
            import uuid as _uuid

            image_id = str(_uuid.uuid4())
        stamp = now_iso()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO seen(sha256,image_id,status,score,margin,latency_ms,first_seen,last_seen) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (sha256, image_id, status, score, margin, latency_ms, stamp, stamp),
                )
            except sqlite3.IntegrityError:
                self._conn.execute(
                    "UPDATE seen SET last_seen=? WHERE sha256=? AND image_id=?", (stamp, sha256, image_id)
                )
                return False
        return True

    def record_image(self, record: dict[str, Any]) -> bool:
        """Atomically store an image row and its seen hash."""
        columns = [
            "id", "sha256", "status", "source_type", "frame_time", "file_name", "stored_path", "source", "mime",
            "width", "height", "size_bytes", "sender_id", "sender_name", "group_id",
            "group_name", "message_id", "score", "margin", "top_group", "top_prompt",
            "model", "latency_ms", "created_at",
        ]
        values = [record.get(c) for c in columns]
        stamp = now_iso()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    f"INSERT INTO images({','.join(columns)}) VALUES({','.join('?' * len(columns))})",
                    values,
                )
                self._conn.execute(
                    "INSERT INTO seen(sha256,image_id,status,score,margin,latency_ms,first_seen,last_seen) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        record["sha256"],
                        record["id"],
                        record["status"],
                        record.get("score"),
                        record.get("margin"),
                        record.get("latency_ms"),
                        stamp,
                        stamp,
                    ),
                )
                self._conn.execute("COMMIT")
            except Exception:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                return False
        return True

    def find_pending_prefix(self, prefix: str, limit: int = 2) -> list[dict[str, Any]]:
        prefix = prefix.strip().lower()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM images WHERE status='pending' AND lower(id) LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (prefix + "%", limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_review(
        self,
        image_id: str,
        *,
        label: str,
        status: str,
        file_name: str | None,
        stored_path: str | None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE images SET review_label=?, status=?, file_name=?, stored_path=?, "
                "reviewed_at=? WHERE id=?",
                (label, status, file_name, stored_path, now_iso(), image_id),
            )
            self._conn.execute("UPDATE seen SET status=? WHERE image_id=?", (status, image_id))

    def count_status(self, status: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM images WHERE status=?", (status,)
            ).fetchone()
        return int(row["n"])

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM seen GROUP BY status"
            ).fetchall()
        result = {str(r["status"]): int(r["n"]) for r in rows}
        result["recognized"] = sum(
            result.get(s, 0) for s in ("saved", "pending", "rejected")
        )
        return result

    def recent_pending(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, score, margin, top_group, created_at, group_id FROM images "
                "WHERE status='pending' ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def latency_stats(self, limit: int = 300) -> dict[str, float]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT latency_ms FROM seen WHERE latency_ms IS NOT NULL "
                "ORDER BY last_seen DESC LIMIT ?",
                (limit,),
            ).fetchall()
        values = sorted(float(r["latency_ms"]) for r in rows)
        if not values:
            return {"count": 0, "avg_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        p95 = values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]
        return {
            "count": len(values),
            "avg_ms": sum(values) / len(values),
            "p95_ms": p95,
            "max_ms": values[-1],
        }

    # -- CSV -------------------------------------------------------------
    def export_csv(self) -> None:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM images ORDER BY created_at ASC"
            ).fetchall()
        tmp = self.csv_path.with_name(self.csv_path.name + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                data = {}
                for column in CSV_COLUMNS:
                    if column == "image_id":
                        data[column] = row["id"]
                    else:
                        data[column] = row[column] if column in row.keys() else ""
                for key in ("sender_name", "group_name", "top_prompt", "review_label"):
                    data[key] = _csv_safe(data[key])
                writer.writerow(data)
        os.replace(tmp, self.csv_path)

    def total_latency_samples(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM seen WHERE latency_ms IS NOT NULL"
            ).fetchone()
        return int(row["n"])
