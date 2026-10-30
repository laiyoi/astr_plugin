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
    "poster_path",
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
                    reviewed_at TEXT,
                    poster_path TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
                CREATE INDEX IF NOT EXISTS idx_images_created ON images(created_at);
                CREATE INDEX IF NOT EXISTS idx_images_sha ON images(sha256);
                CREATE INDEX IF NOT EXISTS idx_images_sender ON images(sender_id, status, created_at);
                CREATE INDEX IF NOT EXISTS idx_images_score ON images(score);
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
                CREATE TABLE IF NOT EXISTS sender_aliases (
                    nickname TEXT PRIMARY KEY,
                    sender_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS group_members (
                    sender_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    group_name TEXT,
                    nickname TEXT,
                    card TEXT,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    PRIMARY KEY (sender_id, group_id)
                );
                INSERT OR IGNORE INTO group_members
                    (sender_id, group_id, group_name, nickname, card, first_seen, last_seen)
                SELECT sender_id, group_id, COALESCE(group_name,''), '', sender_name,
                       MIN(created_at), MAX(created_at)
                FROM images
                WHERE sender_id NOT IN ('', '1094950020') AND group_id <> ''
                GROUP BY sender_id, group_id;
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
            if "poster_path" not in columns:
                self._conn.execute("ALTER TABLE images ADD COLUMN poster_path TEXT")

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
            "model", "latency_ms", "created_at", "poster_path",
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
        poster_path: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE images SET review_label=?, status=?, file_name=?, stored_path=?, "
                "poster_path=COALESCE(?, poster_path), reviewed_at=? WHERE id=?",
                (label, status, file_name, stored_path, poster_path, now_iso(), image_id),
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

    # -- web / review ----------------------------------------------------
    def update_prediction(
        self,
        image_id: str,
        *,
        label: str,
        score: float,
        margin: float,
        latency_ms: float,
        model: str,
    ) -> bool:
        """Fill a model prediction for an imported row without a human label."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE images SET top_group=?, score=?, margin=?, latency_ms=?, model=? "
                "WHERE id=? AND COALESCE(review_label,'')=''",
                (label, score, margin, latency_ms, model, image_id),
            )
        return cur.rowcount > 0

    def skip_images(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM images WHERE COALESCE(NULLIF(review_label,''), top_group)='skip'"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_images(self, ids: list[str]) -> int:
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            cur = self._conn.execute(
                f"DELETE FROM images WHERE id IN ({placeholders})",
                [str(i) for i in ids],
            )
            self._conn.execute(
                f"DELETE FROM seen WHERE image_id IN ({placeholders})",
                [str(i) for i in ids],
            )
        return int(cur.rowcount)

    def set_sender_for_images(self, ids: list[str], sender_id: str, sender_name: str) -> int:
        """Manually attribute selected images to a concrete QQ/nickname."""
        uid = (sender_id or "").strip()
        name = (sender_name or "").strip()
        if not uid.isdigit() or not (5 <= len(uid) <= 20):
            raise ValueError("QQ 必须是 5-20 位数字")
        if not name:
            raise ValueError("昵称不能为空")
        if not ids:
            return 0
        if len(ids) > 500:
            raise ValueError("最多一次修改 500 张图片")
        updated = 0
        with self._lock:
            for image_id in ids:
                cur = self._conn.execute(
                    "UPDATE images SET sender_id=?, sender_name=? WHERE id=?",
                    (uid, name, str(image_id)),
                )
                updated += int(cur.rowcount)
        return updated

    def record_group_member(
        self,
        *,
        sender_id: str,
        group_id: str,
        group_name: str,
        nickname: str,
        card: str,
    ) -> None:
        """Persist a user's group membership and group-specific display name."""
        uid = str(sender_id or "").strip()
        gid = str(group_id or "").strip()
        if not uid or gid == "" or uid == "0":
            return
        now = now_iso()
        with self._lock:
            self._conn.execute(
                "INSERT INTO group_members (sender_id, group_id, group_name, nickname, card, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(sender_id, group_id) DO UPDATE SET "
                "group_name=excluded.group_name, nickname=excluded.nickname, card=excluded.card, last_seen=excluded.last_seen",
                (uid, gid, group_name or "", nickname or "", card or "", now, now),
            )

    def list_group_members(self, sender_id: str = "", limit: int = 200) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if sender_id:
            clauses.append("sender_id=?")
            params.append(sender_id)
        params.append(max(1, min(limit, 1000)))
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM group_members WHERE " + " AND ".join(clauses) +
                " ORDER BY last_seen DESC LIMIT ?", params
            ).fetchall()
        return [dict(r) for r in rows]

    def sender_id_for_alias(self, nickname: str) -> str:
        name = (nickname or "").strip()
        if not name:
            return ""
        with self._lock:
            row = self._conn.execute(
                "SELECT sender_id FROM sender_aliases WHERE nickname=?", (name,)
            ).fetchone()
        return str(row["sender_id"]) if row else ""

    def set_sender_alias(self, nickname: str, sender_id: str) -> int:
        """Save a manual nickname→QQ mapping and apply it to legacy rows."""
        name = (nickname or "").strip()
        uid = (sender_id or "").strip()
        if not name:
            raise ValueError("nickname is required")
        if not uid.isdigit() or not (5 <= len(uid) <= 20):
            raise ValueError("QQ 必须是 5-20 位数字")
        with self._lock:
            self._conn.execute(
                "INSERT INTO sender_aliases (nickname, sender_id, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(nickname) DO UPDATE SET sender_id=excluded.sender_id, updated_at=excluded.updated_at",
                (name, uid, now_iso()),
            )
            cur = self._conn.execute(
                "UPDATE images SET sender_id=?, "
                "source_type=CASE WHEN source_type='forward_training' THEN 'image' ELSE source_type END "
                "WHERE sender_name=? AND sender_id IN ('', '1094950020')",
                (uid, name),
            )
        return int(cur.rowcount)

    def list_sender_aliases(self) -> dict[str, Any]:
        with self._lock:
            unknown = self._conn.execute(
                "SELECT i.sender_name AS nickname, COUNT(*) AS total, "
                "SUM(CASE WHEN i.sender_id = '' THEN 1 ELSE 0 END) AS unknown, "
                "MAX(i.created_at) AS last_at, a.sender_id AS mapped_id "
                "FROM images i LEFT JOIN sender_aliases a ON a.nickname=i.sender_name "
                "WHERE i.sender_name<>'' AND COALESCE(i.stored_path,'')<>'' "
                "GROUP BY i.sender_name HAVING unknown > 0 "
                "ORDER BY total DESC, nickname ASC"
            ).fetchall()
            aliases = self._conn.execute(
                "SELECT nickname, sender_id, updated_at FROM sender_aliases ORDER BY updated_at DESC"
            ).fetchall()
        items = []
        for row in unknown:
            item = dict(row)
            item["groups"] = []
            with self._lock:
                if item.get("mapped_id"):
                    groups = self._conn.execute(
                        "SELECT group_id, group_name, nickname, card, last_seen FROM group_members "
                        "WHERE sender_id=? ORDER BY last_seen DESC LIMIT 20",
                        (item["mapped_id"],),
                    ).fetchall()
                else:
                    groups = self._conn.execute(
                        "SELECT group_id, group_name, nickname, card, last_seen FROM group_members "
                        "WHERE card=? OR nickname=? ORDER BY last_seen DESC LIMIT 20",
                        (item["nickname"], item["nickname"]),
                    ).fetchall()
            item["groups"] = [dict(g) for g in groups]
            items.append(item)
        return {
            "items": items,
            "aliases": [dict(r) for r in aliases],
        }

    def groups_for_identity(self, sender_id: str = "", nickname: str = "") -> list[dict[str, Any]]:
        uid = str(sender_id or "").strip()
        name = str(nickname or "").strip()
        with self._lock:
            if uid:
                rows = self._conn.execute(
                    "SELECT group_id, group_name, nickname, card, last_seen FROM group_members "
                    "WHERE sender_id=? ORDER BY last_seen DESC LIMIT 20",
                    (uid,),
                ).fetchall()
            elif name:
                rows = self._conn.execute(
                    "SELECT group_id, group_name, nickname, card, last_seen FROM group_members "
                    "WHERE card=? OR nickname=? ORDER BY last_seen DESC LIMIT 20",
                    (name, name),
                ).fetchall()
            else:
                rows = []
        return [dict(r) for r in rows]

    def users_missing_nickname(self, limit: int = 200) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT sender_id FROM images WHERE sender_id<>'' "
                "AND COALESCE(sender_name,'') IN ('', '未知用户', 'QQ用户') "
                "ORDER BY sender_id LIMIT ?",
                (max(1, min(limit, 1000)),),
            ).fetchall()
        return [str(r["sender_id"]) for r in rows]

    def set_sender_nickname(self, sender_id: str, nickname: str) -> int:
        uid = str(sender_id or "").strip()
        name = str(nickname or "").strip()
        if not uid or not name:
            return 0
        with self._lock:
            cur = self._conn.execute(
                "UPDATE images SET sender_name=? WHERE sender_id=? "
                "AND COALESCE(sender_name,'') IN ('', '未知用户', 'QQ用户')",
                (name, uid),
            )
            self._conn.execute(
                "UPDATE group_members SET nickname=? WHERE sender_id=? AND COALESCE(nickname,'')=''",
                (name, uid),
            )
        return int(cur.rowcount)

    def backfill_sender_by_name(self, sender_name: str, sender_id: str) -> int:
        """Use a direct message's real QQ to repair same-nickname legacy rows.

        Merged-forward nodes sometimes expose only a nickname plus a fallback
        id.  When the same nickname later sends a direct image, that direct
        event has a reliable QQ and can fill the historical rows.
        """
        name = (sender_name or "").strip()
        uid = (sender_id or "").strip()
        if not name or not uid or uid == "1094950020":
            return 0
        with self._lock:
            self._conn.execute(
                "INSERT INTO sender_aliases (nickname, sender_id, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(nickname) DO UPDATE SET sender_id=excluded.sender_id, updated_at=excluded.updated_at",
                (name, uid, now_iso()),
            )
            cur = self._conn.execute(
                "UPDATE images SET sender_id=?, "
                "source_type=CASE WHEN source_type='forward_training' THEN 'image' ELSE source_type END, "
                "reviewed_at=COALESCE(reviewed_at, ?) "
                "WHERE sender_name=? AND sender_id IN ('', '1094950020')",
                (uid, now_iso(), name),
            )
        return int(cur.rowcount)

    def refresh_sender_metadata(
        self,
        sha256: str,
        *,
        sender_id: str,
        sender_name: str,
        group_id: str = "",
        group_name: str = "",
        message_id: str = "",
    ) -> bool:
        """Correct a legacy/imported row's sender when the same image is resent.

        Only records with a known fallback/import provenance are touched; a
        previously valid sender is never overwritten by a later forward.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT id, sender_id, source_type FROM images WHERE sha256=? "
                "ORDER BY created_at ASC LIMIT 1",
                (sha256,),
            ).fetchone()
            if not row:
                return False
            current_id = str(row["sender_id"] or "")
            source_type = str(row["source_type"] or "")
            if not sender_id:
                return False
            if source_type != "forward_training" and current_id != "" and current_id != sender_id:
                return False
            self._conn.execute(
                "UPDATE images SET sender_id=?, sender_name=?, group_id=?, group_name=?, "
                "message_id=?, source_type=CASE WHEN source_type='forward_training' THEN 'image' ELSE source_type END "
                "WHERE id=?",
                (
                    sender_id, sender_name, group_id, group_name, message_id,
                    row["id"],
                ),
            )
        return True

    def get_image(self, image_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        return dict(row) if row else None

    def set_review_label(self, image_id: str, label: str) -> bool:
        if label not in {"collect", "skip", "suspicious", "unlabeled"}:
            raise ValueError("invalid label")
        with self._lock:
            cur = self._conn.execute(
                "UPDATE images SET review_label=?, reviewed_at=? WHERE id=?",
                ("unlabeled" if label == "unlabeled" else label, now_iso(), image_id),
            )
        return cur.rowcount > 0

    @staticmethod
    def _build_image_filter(
        *,
        user: str = "",
        label: str = "",
        status: str = "",
        q: str = "",
        since: str = "",
        until: str = "",
        min_score: float | None = None,
        max_score: float | None = None,
        include_missing: bool = False,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if not include_missing:
            clauses.append("COALESCE(stored_path, '') <> ''")
        if user:
            clauses.append("sender_id = ?")
            params.append(user)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if label:
            clauses.append("COALESCE(NULLIF(review_label,''), top_group) = ?")
            params.append(label)
        if q:
            like = f"%{q}%"
            clauses.append("(sender_id LIKE ? OR sender_name LIKE ? OR group_name LIKE ?)")
            params.extend([like, like, like])
        if since:
            clauses.append("created_at >= ?")
            params.append(since)
        if until:
            clauses.append("created_at <= ?")
            params.append(until)
        if min_score is not None:
            clauses.append("score IS NOT NULL AND score >= ?")
            params.append(min_score)
        if max_score is not None:
            clauses.append("score IS NOT NULL AND score <= ?")
            params.append(max_score)
        return (" AND ".join(clauses) if clauses else "1=1"), params

    def list_images(
        self,
        *,
        user: str = "",
        label: str = "",
        status: str = "",
        q: str = "",
        since: str = "",
        until: str = "",
        min_score: float | None = None,
        max_score: float | None = None,
        sort: str = "created_desc",
        include_missing: bool = False,
        limit: int = 60,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where, params = self._build_image_filter(
            user=user, label=label, status=status, q=q, since=since, until=until,
            min_score=min_score, max_score=max_score, include_missing=include_missing,
        )
        orders = {
            "created_desc": "created_at DESC, rowid DESC",
            "created_asc": "created_at ASC, rowid ASC",
            "score_desc": "CASE WHEN score IS NULL THEN 1 ELSE 0 END ASC, score DESC, created_at DESC",
            "score_asc": "CASE WHEN score IS NULL THEN 1 ELSE 0 END ASC, score ASC, created_at DESC",
            "sender_asc": "sender_name ASC, created_at DESC",
        }
        order = orders.get(sort, orders["created_desc"])
        params.extend([max(1, min(limit, 200)), max(0, offset)])
        sql = f"SELECT * FROM images WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["label"] = item.get("review_label") or item.get("top_group") or "unlabeled"
            result.append(item)
        return result

    def count_images(
        self,
        *,
        user: str = "",
        label: str = "",
        status: str = "",
        q: str = "",
        since: str = "",
        until: str = "",
        min_score: float | None = None,
        max_score: float | None = None,
        include_missing: bool = False,
    ) -> int:
        where, params = self._build_image_filter(
            user=user, label=label, status=status, q=q, since=since, until=until,
            min_score=min_score, max_score=max_score, include_missing=include_missing,
        )
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM images WHERE {where}", params
            ).fetchone()
        return int(row["n"])

    def label_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT COALESCE(NULLIF(review_label,''), top_group) AS label, COUNT(*) AS n "
                "FROM images WHERE COALESCE(stored_path, '') <> '' GROUP BY label"
            ).fetchall()
        return {str(r["label"] or "unlabeled"): int(r["n"]) for r in rows}

    def export_rows(
        self,
        *,
        labels: list[str],
        since: str = "",
        until: str = "",
        include_missing: bool = True,
    ) -> list[dict[str, Any]]:
        if not labels:
            return []
        placeholders = ",".join("?" for _ in labels)
        clauses = [f"COALESCE(NULLIF(review_label,''), top_group) IN ({placeholders})"]
        params: list[Any] = list(labels)
        if not include_missing:
            clauses.append("COALESCE(stored_path, '') <> ''")
        if since:
            clauses.append("created_at >= ?")
            params.append(since)
        if until:
            clauses.append("created_at <= ?")
            params.append(until)
        sql = "SELECT * FROM images WHERE " + " AND ".join(clauses) + " ORDER BY created_at ASC, id ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def ranking(
        self,
        *,
        threshold: float = 0.8,
        range_days: int = 0,
        min_samples: int = 1,
        sort: str = "high_count",
        q: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Ranking over current human-confirmed collect images only.

        ``score`` is the model's collect probability.  Rows with no score
        (imported training data) are excluded from confidence denominators.
        """
        from datetime import datetime, timedelta, timezone

        cutoff = ""
        if range_days > 0:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=range_days)).isoformat(timespec="seconds")
        clauses = [
            "COALESCE(NULLIF(review_label,''), top_group) = 'collect'",
            "COALESCE(stored_path, '') <> ''",
            "status IN ('saved', 'pending')",
            # Historical imports and private-forward legacy rows all carry
            # a fallback node id instead of the image sender's QQ.  They are
            # grouped by their visible nickname and marked QQ-unknown rather
            # than being attributed to one fake account.
        ]
        params: list[Any] = []
        if cutoff:
            clauses.append("created_at >= ?")
            params.append(cutoff)
        if q:
            like = f"%{q}%"
            clauses.append("(sender_id LIKE ? OR sender_name LIKE ?)")
            params.extend([like, like])
        where = " AND ".join(clauses)
        base = (
            "WITH base AS ("
            "SELECT "
            "CASE WHEN sender_id = '' THEN 'name:' || COALESCE(sender_name, '') "
            "ELSE 'id:' || sender_id END AS user_key, "
            "CASE WHEN sender_id = '' THEN '' ELSE sender_id END AS sender_id, "
            "MAX(sender_name) AS sender_name, COUNT(*) AS collect_count, "
            "SUM(CASE WHEN score IS NOT NULL THEN 1 ELSE 0 END) AS confidence_samples, "
            "SUM(CASE WHEN score IS NOT NULL AND score >= ? THEN 1 ELSE 0 END) AS high_count, "
            "AVG(CASE WHEN score IS NOT NULL THEN score END) AS avg_score, "
            "MAX(score) AS max_score, MAX(created_at) AS last_at "
            f"FROM images WHERE {where} GROUP BY user_key)"
        )
        # insert threshold before where params
        query_params: list[Any] = [threshold] + params
        ratio = "CASE WHEN confidence_samples > 0 THEN high_count * 1.0 / confidence_samples ELSE 0 END"
        orders = {
            "collect_count": "collect_count DESC, high_count DESC, sender_name ASC",
            "high_count": f"high_count DESC, {ratio} DESC, avg_score DESC, collect_count DESC, sender_name ASC",
            "high_ratio": f"{ratio} DESC, high_count DESC, avg_score DESC, sender_name ASC",
            "avg_score": f"avg_score DESC, high_count DESC, {ratio} DESC, sender_name ASC",
        }
        order = orders.get(sort, orders["high_count"])
        min_clause = "WHERE confidence_samples >= ?" if min_samples > 0 else ""
        count_where = " AND ".join([min_clause] if min_clause else [])
        sql = (
            base +
            ", filtered AS (SELECT * FROM base " + (min_clause or "") + ") "
            "SELECT *, " + ratio + " AS high_ratio FROM filtered "
            "ORDER BY " + order + " LIMIT ? OFFSET ?"
        )
        if min_samples > 0:
            query_params.append(max(1, min_samples))
        query_params.extend([max(1, min(limit, 200)), max(0, offset)])
        with self._lock:
            rows = self._conn.execute(sql, query_params).fetchall()
            count_sql = base + ", filtered AS (SELECT * FROM base " + (min_clause or "") + ") SELECT COUNT(*) AS n FROM filtered"
            count_params = [threshold] + params + ([max(1, min_samples)] if min_samples > 0 else [])
            total = int(self._conn.execute(count_sql, count_params).fetchone()["n"])
        items = []
        for row in rows:
            item = dict(row)
            item["high_ratio"] = float(item.get("high_ratio") or 0.0)
            item["avg_score"] = float(item["avg_score"]) if item.get("avg_score") is not None else None
            item["max_score"] = float(item["max_score"]) if item.get("max_score") is not None else None
            items.append(item)
        return {"items": items, "total": total, "threshold": threshold, "min_samples": min_samples}

    def legacy_unknown_senders(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM images "
                "WHERE COALESCE(stored_path,'')<>'' AND sender_id = ''"
            ).fetchone()
        return int(row["n"])

    def user_ranking(self, limit: int = 50) -> list[dict[str, Any]]:
        """Backward-compatible simple collect-only ranking."""
        return self.ranking(limit=limit)["items"]

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

    def count_saved_by_sender(self, sender_id: str) -> int:
        """Number of accepted images already saved for this publisher."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM images WHERE sender_id = ? AND status = 'saved'",
                (str(sender_id),),
            ).fetchone()
        return int(row["n"])

    def total_latency_samples(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM seen WHERE latency_ms IS NOT NULL"
            ).fetchone()
        return int(row["n"])
