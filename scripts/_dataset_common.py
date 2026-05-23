"""
RoFA-SemEval 数据集脚本公共工具。

被 capture_sample.py / annotate_sample.py / finalize_dataset.py 共用。
本文件不依赖 RealSense / OpenCV 之外的重型库，便于在不同阶段独立运行。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CHINA_TZ = timezone(timedelta(hours=8))


# --------------------------------------------------------------------------- #
# 时间 / 路径辅助
# --------------------------------------------------------------------------- #

def now_iso() -> str:
    """ISO 8601 时间戳，带 +08:00 时区。"""
    return datetime.now(CHINA_TZ).strftime("%Y-%m-%dT%H:%M:%S+08:00")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# JSON 安全读写（原子写）
# --------------------------------------------------------------------------- #

def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """原子写：先写到 .tmp，再 rename，避免半写文件。"""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# 文件指纹
# --------------------------------------------------------------------------- #

def sha1_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 中文 → 拼音 slug
# --------------------------------------------------------------------------- #

def _has_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _try_chinese_to_pinyin(text: str) -> Optional[str]:
    """
    优先用 pypinyin（如安装），否则返回 None 让上层降级。
    """
    try:
        from pypinyin import lazy_pinyin, Style  # type: ignore

        parts = lazy_pinyin(text, style=Style.NORMAL, errors="ignore")
        return "".join(parts)
    except Exception:
        return None


def _ascii_slugify(text: str) -> str:
    """
    通用 slug：去重音 → 仅保留 [a-zA-Z0-9_]，其余转下划线，去多余下划线。
    """
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    ascii_only = ascii_only.encode("ascii", "ignore").decode("ascii")
    ascii_only = ascii_only.lower()
    ascii_only = re.sub(r"[^a-z0-9]+", "_", ascii_only).strip("_")
    return ascii_only


def make_class_slug(user_input: str) -> str:
    """
    把用户输入归一化为目录名 slug。
    - 含中文：先转拼音再 ascii slugify
    - 全英文：直接 ascii slugify
    - 转换失败：用 utf8-hex 兜底，确保不返回空串
    """
    raw = user_input.strip()
    if not raw:
        return "unknown"

    if _has_chinese(raw):
        pinyin = _try_chinese_to_pinyin(raw)
        if pinyin:
            slug = _ascii_slugify(pinyin)
            if slug:
                return slug
        # pypinyin 不可用 → 用 hex 兜底
        return "zh_" + raw.encode("utf-8").hex()[:16]

    slug = _ascii_slugify(raw)
    return slug if slug else "unknown"


def normalize_alias(text: str) -> str:
    """用于 alias 比较：去前后空格、转小写、压缩空白；保留中文原样。"""
    return re.sub(r"\s+", " ", text.strip()).lower()


# --------------------------------------------------------------------------- #
# classes.json 数据结构封装
# --------------------------------------------------------------------------- #

def empty_classes_doc() -> Dict[str, Any]:
    return {
        "version": 1,
        "managed_by": "capture_sample.py",
        "next_id": 1,
        "classes": [],
    }


def load_classes(path: Path) -> Dict[str, Any]:
    doc = load_json(path, default=None)
    if doc is None:
        return empty_classes_doc()
    # 旧文件兜底字段
    doc.setdefault("version", 1)
    doc.setdefault("managed_by", "capture_sample.py")
    doc.setdefault("classes", [])
    if "next_id" not in doc:
        existing = [c.get("id", 0) for c in doc["classes"]]
        doc["next_id"] = (max(existing) + 1) if existing else 1
    return doc


def find_class_by_alias(doc: Dict[str, Any], user_input: str) -> Optional[Dict[str, Any]]:
    target = normalize_alias(user_input)
    for c in doc["classes"]:
        for a in c.get("aliases", []):
            if normalize_alias(a) == target:
                return c
    return None


def find_class_by_slug(doc: Dict[str, Any], slug: str) -> Optional[Dict[str, Any]]:
    for c in doc["classes"]:
        if c.get("name") == slug:
            return c
    return None


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def fuzzy_match_classes(
    doc: Dict[str, Any], user_input: str, top_k: int = 3, max_edit: int = 3
) -> List[Tuple[Dict[str, Any], int]]:
    """返回 [(class_obj, edit_distance)]，按相似度升序，最多 top_k 个。"""
    target = normalize_alias(user_input)
    candidates: List[Tuple[Dict[str, Any], int]] = []
    for c in doc["classes"]:
        best = 10**9
        for a in c.get("aliases", []) + [c.get("name", ""), c.get("name_zh", "")]:
            if not a:
                continue
            na = normalize_alias(a)
            if not na:
                continue
            dist = _levenshtein(na, target)
            # 子串关系视作 1 距离
            if target and (target in na or na in target):
                dist = min(dist, 1)
            best = min(best, dist)
        if best <= max_edit:
            candidates.append((c, best))
    candidates.sort(key=lambda x: x[1])
    return candidates[:top_k]


def allocate_unique_slug(doc: Dict[str, Any], base_slug: str) -> str:
    """如果 base_slug 已被占用（不同类）则追加 _2 / _3 ..."""
    if find_class_by_slug(doc, base_slug) is None:
        return base_slug
    i = 2
    while True:
        candidate = f"{base_slug}_{i}"
        if find_class_by_slug(doc, candidate) is None:
            return candidate
        i += 1


def append_alias(class_obj: Dict[str, Any], raw_input: str) -> bool:
    """如果输入是新 alias 就追加，返回是否新增。"""
    target = normalize_alias(raw_input)
    aliases = class_obj.setdefault("aliases", [])
    for a in aliases:
        if normalize_alias(a) == target:
            return False
    aliases.append(raw_input.strip())
    class_obj["last_updated_at"] = now_iso()
    return True


def create_class(
    doc: Dict[str, Any], user_input: str
) -> Dict[str, Any]:
    base_slug = make_class_slug(user_input)
    slug = allocate_unique_slug(doc, base_slug)
    name_zh = user_input.strip() if _has_chinese(user_input) else ""

    new_class: Dict[str, Any] = {
        "id": int(doc["next_id"]),
        "name": slug,
        "name_zh": name_zh,
        "aliases": [user_input.strip()],
        "first_seen_at": now_iso(),
        "last_updated_at": now_iso(),
        "captured_count": 0,
    }
    doc["next_id"] = int(doc["next_id"]) + 1
    doc["classes"].append(new_class)
    return new_class


def bump_captured_count(class_obj: Dict[str, Any]) -> None:
    class_obj["captured_count"] = int(class_obj.get("captured_count", 0)) + 1
    class_obj["last_updated_at"] = now_iso()


# --------------------------------------------------------------------------- #
# 样本 ID 分配
# --------------------------------------------------------------------------- #

_SAMPLE_ID_RE = re.compile(r"^(?P<slug>.+)_(?P<idx>\d{4,})$")


def next_sample_id(class_dir: Path, class_slug: str) -> str:
    """
    扫描 class_dir 下已有 <slug>_NNNN 子目录，返回下一个未被占用的 id。
    NNNN 至少 4 位，超过 9999 自动扩位（_10000 等）。
    """
    used = set()
    if class_dir.exists():
        for child in class_dir.iterdir():
            if not child.is_dir():
                continue
            m = _SAMPLE_ID_RE.match(child.name)
            if m and m.group("slug") == class_slug:
                used.add(int(m.group("idx")))
    idx = 1
    while idx in used:
        idx += 1
    width = max(4, len(str(idx)))
    return f"{class_slug}_{idx:0{width}d}"


# --------------------------------------------------------------------------- #
# 文件移动 / 链接
# --------------------------------------------------------------------------- #

def move_dir(src: Path, dst: Path) -> None:
    """跨设备时 shutil.move 自动 fallback 到拷贝+删除。"""
    ensure_dir(dst.parent)
    if dst.exists():
        raise FileExistsError(f"目标已存在: {dst}")
    shutil.move(str(src), str(dst))


def hardlink_or_copy_file(src: Path, dst: Path) -> str:
    """
    优先硬链接；跨设备失败则降级为 copy2。返回 'hardlink' 或 'copy'。
    """
    ensure_dir(dst.parent)
    if dst.exists():
        return "skip"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def hardlink_or_copy_dir(src_dir: Path, dst_dir: Path) -> Dict[str, int]:
    """递归把 src_dir 下文件硬链接（或拷贝）到 dst_dir。"""
    stats = {"hardlink": 0, "copy": 0, "skip": 0}
    for src_path in src_dir.rglob("*"):
        if src_path.is_dir():
            continue
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel
        action = hardlink_or_copy_file(src_path, dst_path)
        stats[action] = stats.get(action, 0) + 1
    return stats


# --------------------------------------------------------------------------- #
# 退避锁（极简）：classes.json 频繁修改时避免被并发写覆盖
# --------------------------------------------------------------------------- #

class FileLock:
    """
    极简文件锁。仅用于多进程同时操作 classes.json 的小概率场景。
    若锁文件存在则等待最多 timeout_sec。
    """

    def __init__(self, target: Path, timeout_sec: float = 5.0, poll_sec: float = 0.05):
        self.lock_path = target.with_suffix(target.suffix + ".lock")
        self.timeout_sec = timeout_sec
        self.poll_sec = poll_sec

    def __enter__(self):
        deadline = time.monotonic() + self.timeout_sec
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    # 超时则强行接管，假设上一持有者异常退出
                    try:
                        self.lock_path.unlink()
                    except Exception:
                        pass
                    continue
                time.sleep(self.poll_sec)

    def __exit__(self, exc_type, exc, tb):
        try:
            self.lock_path.unlink()
        except Exception:
            pass
