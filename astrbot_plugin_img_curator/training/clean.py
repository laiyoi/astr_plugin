
#!/usr/bin/env python3
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="删除 dir1 中与 dir2 有相同相对路径的文件"
    )
    parser.add_argument("dir1", type=Path, help="待清理的目录")
    parser.add_argument("dir2", type=Path, help="用于比对的目录")
    args = parser.parse_args()

    dir1 = args.dir1.resolve()
    dir2 = args.dir2.resolve()

    if not dir1.is_dir():
        parser.error(f"dir1 不是有效目录: {dir1}")
    if not dir2.is_dir():
        parser.error(f"dir2 不是有效目录: {dir2}")

    # 收集 dir2 中所有文件的相对路径
    files2 = {
        p.relative_to(dir2)
        for p in dir2.rglob("*")
        if p.is_file()
    }

    deleted = 0
    for p in dir1.rglob("*"):
        if p.is_file() and p.relative_to(dir1) in files2:
            print(f"删除: {p}")
            p.unlink()
            deleted += 1

    print(f"完成，共删除 {deleted} 个文件")


if __name__ == "__main__":
    main()
