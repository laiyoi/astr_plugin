#!/usr/bin/env python3
"""按文件内容哈希去重：同内容只保留最新的一个，删掉创建时间早的。

默认只预览（dry-run），加 --apply 才真正删除。
判断"更新"看创建时间（Windows 上就是文件的"创建日期"），创建时间相同则按路径字典序取最后一个；
想改成看修改时间就加 --by modified。

用法:
    python dedupe_files.py D:\\pics                   # 预览
    python dedupe_files.py D:\\pics --apply           # 真正删除
    python dedupe_files.py D:\\pics --by modified     # 改成按修改时间排序
    python dedupe_files.py D:\\pics --trash D:\\dup    # 移到回收目录而不是删除
    python dedupe_files.py D:\\pics --ext .jpg .png   # 只处理指定后缀

空文件会被跳过（所有 0 字节文件哈希相同，删了没意义）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def file_time(path: Path, by: str) -> float:
    """返回用于比较的"新旧"时间戳。

    created: 优先 st_birthtime（macOS/部分 BSD），其次 Windows 的 st_ctime（创建时间），
             都没有时退回 st_mtime 并在 stderr 提示一次（Linux 的 st_ctime 是 inode
             变更时间，会随着 chmod/改名变化，不适合代表"创建")。
    modified: st_mtime。
    """
    stat = path.stat()
    if by == "modified":
        return stat.st_mtime
    birth = getattr(stat, "st_birthtime", None)
    if birth is not None:
        return float(birth)
    return float(stat.st_ctime) if sys.platform == "win32" else float(stat.st_mtime)


def collect_files(roots: list[Path], exts: set[str] | None) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if not path.is_file():
                continue
            if exts and path.suffix.lower() not in exts:
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    return files


def sort_key(path: Path, by: str) -> tuple[float, str]:
    """最新的排前面；时间相同时用路径字典序保证结果稳定。"""
    return (file_time(path, by), str(path).lower())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按 sha256 内容哈希去重，同内容保留创建时间最新的那个，删掉创建日期较早的副本。"
    )
    parser.add_argument("paths", nargs="+", type=Path, help="要扫描的目录或文件（可多个）")
    parser.add_argument("--apply", action="store_true", help="真正删除；不加此参数只预览")
    parser.add_argument("--trash", type=Path, default=None, help="把重复文件移动到这个目录，而不是直接删除")
    parser.add_argument("--ext", nargs="+", default=None, help="只处理这些后缀，例如 --ext .jpg .png")
    parser.add_argument("--json-out", type=Path, default=None, help="把结果写入 JSON 报告")
    parser.add_argument("--no-progress", action="store_true", help="不打印每个文件的哈希进度")
    parser.add_argument(
        "--by",
        choices=("created", "modified"),
        default="created",
        help="按哪个时间判断新旧，默认 created（创建日期）",
    )
    args = parser.parse_args()

    roots = [p.resolve() for p in args.paths]
    missing = [str(p) for p in roots if not p.exists()]
    if missing:
        parser.error("路径不存在: " + ", ".join(missing))

    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext} if args.ext else None

    files = collect_files(roots, exts)
    if not files:
        print("没有找到可处理的文件")
        return 0

    by_size: dict[int, list[Path]] = defaultdict(list)
    empty: list[Path] = []
    for path in files:
        size = path.stat().st_size
        if size == 0:
            empty.append(path)
            continue
        by_size[size].append(path)

    # 只有大小相同的文件才可能同内容，先按大小分组再算哈希
    groups: dict[str, list[Path]] = defaultdict(list)
    hashed = 0
    for size, same_size in sorted(by_size.items()):
        if len(same_size) < 2:
            continue
        for path in same_size:
            digest = sha256_file(path)
            groups[digest].append(path)
            hashed += 1
            if not args.no_progress:
                print(f"  hash {digest[:12]}  {path}")

    duplicates = {digest: paths for digest, paths in groups.items() if len(paths) > 1}

    print()
    print(f"扫描文件: {len(files)}  计算哈希: {hashed}  重复组: {len(duplicates)}")
    if empty:
        print(f"跳过空文件: {len(empty)}")

    if not duplicates:
        print("没有重复文件")
        return 0

    time_label = "创建时间" if args.by == "created" else "修改时间"
    to_delete: list[dict] = []
    for digest, paths in duplicates.items():
        ordered = sorted(paths, key=lambda p: sort_key(p, args.by), reverse=True)
        keep, drop = ordered[0], ordered[1:]
        keep_time = file_time(keep, args.by)
        for path in drop:
            stat = path.stat()
            to_delete.append(
                {
                    "path": path,
                    "keep": keep,
                    "digest": digest,
                    "size": stat.st_size,
                    "time": file_time(path, args.by),
                    "keep_time": keep_time,
                }
            )

    total_bytes = 0
    print()
    for item in sorted(to_delete, key=lambda entry: entry["time"]):
        path, keep, digest = item["path"], item["keep"], item["digest"]
        total_bytes += item["size"]
        print(f"重复 {digest[:12]}  {time_label} {fmt_time(item['time'])}  {path}")
        print(f"     保留 {time_label} {fmt_time(item['keep_time'])}  {keep}")
        if args.apply:
            try:
                if args.trash:
                    args.trash.mkdir(parents=True, exist_ok=True)
                    target = args.trash / path.name
                    index = 1
                    while target.exists():
                        target = args.trash / f"{path.stem}.{index}{path.suffix}"
                        index += 1
                    shutil.move(str(path), str(target))
                else:
                    path.unlink()
            except OSError as exc:
                print(f"     失败: {exc}")
                item["failed"] = str(exc)

    print()
    done = len([item for item in to_delete if not item.get("failed")])
    action = "已删除" if args.apply and not args.trash else ("已移动" if args.apply else "将删除")
    print(f"{action} {done} 个文件，释放约 {total_bytes / 1024 / 1024:.2f} MiB")
    if not args.apply:
        print("这是预览，加 --apply 才会真的删除")

    if args.json_out:
        report = {
            "scanned": len(files),
            "hashed": hashed,
            "empty_skipped": [str(p) for p in empty],
            "duplicate_groups": len(duplicates),
            "order_by": args.by,
            "order_by_label": time_label,
            "applied": bool(args.apply),
            "trash": str(args.trash) if args.trash else None,
            "removed": [
                {
                    "removed": str(item["path"]),
                    "removed_time": fmt_time(item["time"]),
                    "kept": str(item["keep"]),
                    "kept_time": fmt_time(item["keep_time"]),
                    "sha256": item["digest"],
                    "bytes": item["size"],
                    "trashed": bool(args.trash) if args.apply else None,
                    "error": item.get("failed"),
                }
                for item in to_delete
            ],
            "freed_bytes": total_bytes,
        }
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
