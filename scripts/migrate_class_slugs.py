"""
一次性迁移脚本：把 raw_capture 中"hex 兜底"的类别 slug 迁回拼音 slug。

背景：在没安装 pypinyin 的机器上跑 capture_sample.py，中文类名会 fallback
到 zh_<utf8 hex> 的目录名（虽然能跑，但极不可读）。本脚本用 pypinyin 重算
slug，并安全地：
  - 重命名 pending/<old_slug>/ -> pending/<new_slug>/
  - 重命名 pending/<new_slug>/<old_slug>_NNNN/ -> .../<new_slug>_NNNN/
  - 更新每条 capture_meta.json 的 class_name 字段
  - 写回 classes.json

默认是 dry-run（只打印计划），加 --apply 才真改。

用法：
  cd RobotMustFindAnything
  uv run python scripts/migrate_class_slugs.py                # 预演
  uv run python scripts/migrate_class_slugs.py --apply        # 执行
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _dataset_common import (  # noqa: E402
    allocate_unique_slug,
    load_classes,
    load_json,
    make_class_slug,
    save_json,
)


DEFAULT_RAW_ROOT = Path(__file__).resolve().parents[1] / "raw_capture"


def is_hex_fallback_slug(slug: str) -> bool:
    """判断 slug 是否是 hex 兜底形式（zh_ 开头 + 全 hex 字符）。"""
    if not slug.startswith("zh_"):
        return False
    tail = slug[3:]
    if not tail:
        return False
    return all(ch in "0123456789abcdef" for ch in tail)


def plan_class_renames(classes_doc: Dict[str, Any]) -> List[Tuple[Dict[str, Any], str]]:
    """
    为每个 hex 兜底类规划新 slug。返回 [(class_obj, new_slug), ...]。
    用一份临时 doc 做唯一性分配，避免对 classes_doc 现场改动影响后续判定。
    """
    plans: List[Tuple[Dict[str, Any], str]] = []

    # 工作副本：包含所有不需要重命名的类（占着原 name），
    # 然后逐个为需要重命名的类分配新 slug。
    working_doc = {
        "version": classes_doc.get("version", 1),
        "next_id": classes_doc.get("next_id", 1),
        "classes": [
            {"name": c["name"]}  # 仅保留 name 用于撞名检测
            for c in classes_doc.get("classes", [])
            if not is_hex_fallback_slug(c.get("name", ""))
        ],
    }

    for c in classes_doc.get("classes", []):
        old = c.get("name", "")
        if not is_hex_fallback_slug(old):
            continue

        # 用 name_zh 作为输入；若 name_zh 为空则用首个 alias
        seed = c.get("name_zh") or (c.get("aliases") or [""])[0]
        if not seed:
            print(f"  [warn] 类 id={c['id']} 没有可用的中文名 / alias，跳过")
            continue

        base = make_class_slug(seed)
        new_slug = allocate_unique_slug(working_doc, base)
        # 占位写回 working_doc 防止两个类被分配到同一个 slug
        working_doc["classes"].append({"name": new_slug})
        plans.append((c, new_slug))

    return plans


def rename_sample_dirs_within(class_dir: Path, old_slug: str, new_slug: str,
                              dry_run: bool) -> int:
    """
    把 <class_dir>/<old_slug>_NNNN 子目录全部改名为 <new_slug>_NNNN。
    返回修改条数。
    """
    if not class_dir.exists():
        return 0
    count = 0
    for sample_dir in sorted(class_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        name = sample_dir.name
        if not name.startswith(old_slug + "_"):
            continue
        suffix = name[len(old_slug) + 1:]  # 例如 "0001"
        new_name = f"{new_slug}_{suffix}"
        new_path = sample_dir.with_name(new_name)
        if new_path.exists():
            print(f"    [conflict] {new_path} 已存在，跳过")
            continue
        if dry_run:
            print(f"    rename: {sample_dir.name} -> {new_name}")
        else:
            sample_dir.rename(new_path)
        count += 1
    return count


def update_capture_meta_files(class_dir: Path, new_slug: str, dry_run: bool) -> int:
    """
    遍历 class_dir 下每个样本子目录的 capture_meta.json，
    把 class_name 字段改成 new_slug。
    """
    count = 0
    for sample_dir in sorted(class_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        meta_path = sample_dir / "capture_meta.json"
        if not meta_path.exists():
            continue
        meta = load_json(meta_path, default=None)
        if not isinstance(meta, dict):
            continue
        old_value = meta.get("class_name")
        if old_value == new_slug:
            continue
        meta["class_name"] = new_slug
        # 也把 sample_id 内嵌字段（如有）一起改
        if "sample_id" in meta and isinstance(meta["sample_id"], str):
            old_id = meta["sample_id"]
            # 仅在前缀确实匹配旧 slug 时改，避免误伤
            for c in (old_value,):
                if c and old_id.startswith(c + "_"):
                    meta["sample_id"] = new_slug + old_id[len(c):]
                    break
        if dry_run:
            print(f"    update meta: {meta_path.relative_to(class_dir.parent.parent)} "
                  f"class_name {old_value} -> {new_slug}")
        else:
            save_json(meta_path, meta)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=str, default=str(DEFAULT_RAW_ROOT),
        help="raw_capture 根目录（默认仓库根 ./raw_capture）"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="真正执行迁移；默认仅 dry-run 打印计划"
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    if not raw_root.exists():
        print(f"[migrate] raw_root 不存在: {raw_root}")
        return 1

    classes_path = raw_root / "classes.json"
    pending_root = raw_root / "pending"
    if not classes_path.exists():
        print(f"[migrate] classes.json 不存在: {classes_path}")
        return 1

    classes_doc = load_classes(classes_path)
    plans = plan_class_renames(classes_doc)

    if not plans:
        print("[migrate] 没有需要迁移的 hex 兜底 slug。")
        return 0

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[migrate] 模式: {mode}")
    print(f"[migrate] 计划迁移 {len(plans)} 个类:")
    for c, new_slug in plans:
        print(f"  - id={c['id']} name_zh={c.get('name_zh','')!s} "
              f"old={c['name']} -> new={new_slug}")
    print()

    dry_run = not args.apply
    total_dirs_renamed = 0
    total_metas_updated = 0

    for c, new_slug in plans:
        old_slug = c["name"]
        old_class_dir = pending_root / old_slug
        new_class_dir = pending_root / new_slug

        print(f"[migrate] === class id={c['id']} ({c.get('name_zh','')}) ===")

        # 1) 先重命名内部样本目录（在原父目录下做，避免跨 rename 路径问题）
        if old_class_dir.exists():
            n = rename_sample_dirs_within(old_class_dir, old_slug, new_slug, dry_run)
            total_dirs_renamed += n
            print(f"  样本子目录改名: {n} 个")

            # 2) 更新每个 capture_meta.json
            #    （在 old_class_dir 上扫，因为此时父目录还没重命名）
            n_meta = update_capture_meta_files(old_class_dir, new_slug, dry_run)
            total_metas_updated += n_meta
            print(f"  capture_meta.json 更新: {n_meta} 个")

            # 3) 重命名父类目录 old_class_dir -> new_class_dir
            if new_class_dir.exists():
                print(f"  [conflict] 目标父目录已存在: {new_class_dir}（跳过此步，需手工处理）")
            else:
                if dry_run:
                    print(f"  rename class dir: {old_class_dir.name} -> {new_class_dir.name}")
                else:
                    old_class_dir.rename(new_class_dir)
        else:
            print(f"  [warn] {old_class_dir} 不存在，仅更新 classes.json")

        # 4) 更新 classes.json 的 name 字段
        if dry_run:
            print(f"  classes.json: name {old_slug} -> {new_slug}")
        else:
            c["name"] = new_slug

    if not dry_run:
        save_json(classes_path, classes_doc)
        print(f"\n[migrate] 已写回 {classes_path}")

    print(f"\n[migrate] 总结：样本目录改名 {total_dirs_renamed} 个，"
          f"capture_meta 更新 {total_metas_updated} 个，"
          f"类目录改名 {len(plans)} 个")
    if dry_run:
        print("[migrate] 这是 DRY-RUN。确认无误后追加 --apply 参数真正执行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
