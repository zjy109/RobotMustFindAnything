"""
RoFA-SemEval 阶段 3：数据集发布脚本

把 raw_capture/annotated/ 经过完整性校验后，发布到 dataset/。
- 不做主观删减（这一步在阶段 2 就已完成）
- 仅做硬性完整性兜底；不通过的样本写入 rejected.csv，不入最终数据集
- 输出 dataset_report.json + dataset_report.html，展示与大纲推荐刻度的对照

依赖：仅 Python 标准库 + numpy + opencv（用于尺寸 / dtype 校验）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _dataset_common import (  # noqa: E402
    ensure_dir,
    hardlink_or_copy_dir,
    load_classes,
    load_json,
    now_iso,
    save_json,
    sha1_of_file,
)


# --------------------------------------------------------------------------- #
# 默认参数
# --------------------------------------------------------------------------- #
DEFAULT_RAW_ROOT = Path(__file__).resolve().parents[1] / "RoFA-SemEval" / "raw_capture"
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "RoFA-SemEval" / "dataset"
MIN_RGB_W, MIN_RGB_H = 640, 480
MIN_MASK_PIXELS = 200
MIN_POINTS = 200
MIN_AABB_VOLUME = 1e-6  # m^3

# 大纲推荐刻度（仅用于报告对照，不做强制约束）
OUTLINE_TARGETS = {
    "num_classes": 20,
    "num_samples": 1000,
    "per_class_min": 25,
    "per_class_max": 75,
}

REQUIRED_FILES = (
    "rgb.jpg",
    "depth.png",
    "mask.png",
    "intrinsics.json",
    "pose.txt",
    "points.ply",
    "aabb.json",
    "sample.json",
    "capture_meta.json",
)


# --------------------------------------------------------------------------- #
# 单样本校验
# --------------------------------------------------------------------------- #

def _check_sample(sample_dir: Path) -> Tuple[bool, List[str]]:
    """
    返回 (passed, reasons)。passed=False 时 reasons 至少有一项。
    """
    reasons: List[str] = []

    # 1) 必备文件齐全
    for fname in REQUIRED_FILES:
        if not (sample_dir / fname).exists():
            reasons.append(f"missing_file:{fname}")
    if reasons:
        return False, reasons

    # 2) 加载关键文件
    try:
        rgb = cv2.imread(str(sample_dir / "rgb.jpg"), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(sample_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(sample_dir / "mask.png"), cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        return False, [f"read_error:{exc}"]

    if rgb is None:
        reasons.append("rgb_unreadable")
    if depth is None:
        reasons.append("depth_unreadable")
    elif depth.dtype != np.uint16:
        reasons.append(f"depth_dtype:{depth.dtype}")
    if mask is None:
        reasons.append("mask_unreadable")
    if reasons:
        return False, reasons

    # 3) 分辨率 + 尺寸一致
    h, w = rgb.shape[:2]
    if w < MIN_RGB_W or h < MIN_RGB_H:
        reasons.append(f"rgb_too_small:{w}x{h}")
    if depth.shape[:2] != (h, w):
        reasons.append(f"depth_shape_mismatch:{depth.shape[:2]} vs {(h, w)}")
    if mask.shape[:2] != (h, w):
        reasons.append(f"mask_shape_mismatch:{mask.shape[:2]} vs {(h, w)}")
    if reasons:
        return False, reasons

    # 4) mask 非空
    mask_bin = mask if mask.ndim == 2 else mask[..., 0]
    if int((mask_bin > 0).sum()) < MIN_MASK_PIXELS:
        reasons.append(f"mask_too_small:{int((mask_bin > 0).sum())}")

    # 5) intrinsics
    intrinsics = load_json(sample_dir / "intrinsics.json", default=None)
    if not isinstance(intrinsics, dict) or not all(
        k in intrinsics for k in ("fx", "fy", "cx", "cy")
    ):
        reasons.append("intrinsics_invalid")

    # 6) pose 4x4 / 有限值
    try:
        pose = np.loadtxt(sample_dir / "pose.txt", dtype=np.float32)
        if pose.shape == (16,):
            pose = pose.reshape(4, 4)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            reasons.append("pose_invalid")
    except Exception as exc:
        reasons.append(f"pose_read_error:{exc}")

    # 7) AABB
    aabb = load_json(sample_dir / "aabb.json", default=None)
    if not isinstance(aabb, dict) or not all(
        k in aabb for k in ("min", "max", "extent", "center", "num_points")
    ):
        reasons.append("aabb_invalid")
    else:
        try:
            extent = [float(v) for v in aabb["extent"]]
            if any(e <= 0 for e in extent):
                reasons.append(f"aabb_zero_extent:{extent}")
            if extent[0] * extent[1] * extent[2] < MIN_AABB_VOLUME:
                reasons.append(f"aabb_volume_too_small:{extent}")
            num_points = int(aabb.get("num_points", 0))
            if num_points < MIN_POINTS:
                reasons.append(f"too_few_points:{num_points}")
        except Exception as exc:
            reasons.append(f"aabb_parse_error:{exc}")

    # 8) sample.json：class_id 合法性留到外层（需要 classes_doc）
    sample_json = load_json(sample_dir / "sample.json", default=None)
    if not isinstance(sample_json, dict):
        reasons.append("sample_json_invalid")

    # 9) checksum 校验（如有）
    if isinstance(sample_json, dict):
        ck = sample_json.get("checksums") or {}
        if "rgb_sha1" in ck:
            if sha1_of_file(sample_dir / "rgb.jpg") != ck["rgb_sha1"]:
                reasons.append("rgb_sha1_mismatch")
        if "depth_sha1" in ck:
            if sha1_of_file(sample_dir / "depth.png") != ck["depth_sha1"]:
                reasons.append("depth_sha1_mismatch")
        if "mask_sha1" in ck:
            if sha1_of_file(sample_dir / "mask.png") != ck["mask_sha1"]:
                reasons.append("mask_sha1_mismatch")

    return (len(reasons) == 0), reasons


def _validate_class_id(
    sample_dir: Path, classes_doc: Dict[str, Any]
) -> Optional[str]:
    sample_json = load_json(sample_dir / "sample.json", default=None)
    if not isinstance(sample_json, dict):
        return "sample_json_missing"
    cls_id = sample_json.get("class_id")
    cls_name = sample_json.get("class_name")
    valid_ids = {c.get("id") for c in classes_doc.get("classes", [])}
    valid_names = {c.get("name") for c in classes_doc.get("classes", [])}
    if cls_id not in valid_ids:
        return f"class_id_unknown:{cls_id}"
    if cls_name not in valid_names:
        return f"class_name_unknown:{cls_name}"
    return None


# --------------------------------------------------------------------------- #
# 遍历 raw_capture/annotated/
# --------------------------------------------------------------------------- #

def list_annotated_samples(raw_root: Path) -> List[Path]:
    annotated = raw_root / "annotated"
    if not annotated.exists():
        return []
    out: List[Path] = []
    for cdir in sorted(p for p in annotated.iterdir() if p.is_dir()):
        for sdir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            out.append(sdir)
    return out


# --------------------------------------------------------------------------- #
# Levenshtein（轻量复用，避免拖入 _dataset_common 的内部函数）
# --------------------------------------------------------------------------- #

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


def find_class_merge_suggestions(
    classes_doc: Dict[str, Any], in_dataset_classes: List[str]
) -> List[Dict[str, Any]]:
    """检测在最终数据集里同时存在的"高度相似"类对。"""
    classes = [
        c for c in classes_doc.get("classes", [])
        if c.get("name") in set(in_dataset_classes)
    ]
    suggestions: List[Dict[str, Any]] = []
    seen_pairs = set()
    for i, ca in enumerate(classes):
        for cb in classes[i + 1:]:
            pair_key = tuple(sorted([ca["name"], cb["name"]]))
            if pair_key in seen_pairs:
                continue

            reasons: List[str] = []
            # 1) alias 重叠
            aa = {a.strip().lower() for a in ca.get("aliases", [])}
            ab = {a.strip().lower() for a in cb.get("aliases", [])}
            if aa & ab:
                reasons.append(f"alias_overlap={sorted(aa & ab)}")

            # 2) name 极相似
            d_name = _levenshtein(ca["name"], cb["name"])
            if d_name <= 2 and ca["name"] != cb["name"]:
                reasons.append(f"name_levenshtein={d_name}")

            # 3) 互为子串
            if (ca["name"] != cb["name"]) and (
                ca["name"] in cb["name"] or cb["name"] in ca["name"]
            ):
                reasons.append("name_substring")

            # 4) 中文名互为子串
            za, zb = ca.get("name_zh", ""), cb.get("name_zh", "")
            if za and zb and za != zb and (za in zb or zb in za):
                reasons.append("zh_substring")

            if reasons:
                suggestions.append({
                    "class_a": ca["name"],
                    "class_b": cb["name"],
                    "name_zh_a": za,
                    "name_zh_b": zb,
                    "reasons": reasons,
                })
                seen_pairs.add(pair_key)

    return suggestions


# --------------------------------------------------------------------------- #
# 拷贝（硬链接优先）到最终数据集
# --------------------------------------------------------------------------- #

def export_sample(
    sample_dir: Path, raw_root: Path, dataset_samples_root: Path
) -> Tuple[Path, Dict[str, int]]:
    rel = sample_dir.relative_to(raw_root / "annotated")  # <class>/<id>
    dst = dataset_samples_root / rel
    if dst.exists():
        # 已存在 → 跳过（多次运行 finalize 安全）
        return dst, {"skip_dir": 1}
    stats = hardlink_or_copy_dir(sample_dir, dst)
    return dst, stats


# --------------------------------------------------------------------------- #
# 报告渲染
# --------------------------------------------------------------------------- #

def render_report_html(report: Dict[str, Any]) -> str:
    """轻量 HTML 报告。无任何外部依赖。"""
    totals = report["totals"]
    per_class = report["per_class"]
    suggestions = report.get("merge_suggestions", [])
    targets = report["outline_targets"]
    gap = report["outline_gap"]

    rows = []
    for c in per_class:
        row = (
            f"<tr><td>{c['name']}</td><td>{c.get('name_zh','')}</td>"
            f"<td>{c['captured']}</td><td>{c['annotated']}</td>"
            f"<td>{c['discarded']}</td><td>{c['rejected']}</td>"
            f"<td><b>{c['in_dataset']}</b></td>"
            f"<td>{c['vs_outline']}</td></tr>"
        )
        rows.append(row)

    sug_rows = []
    for s in suggestions:
        sug_rows.append(
            f"<tr><td>{s['class_a']}</td><td>{s['class_b']}</td>"
            f"<td>{', '.join(s['reasons'])}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"/>
<title>RoFA-SemEval Dataset Report</title>
<style>
body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
       max-width: 980px; margin: 24px auto; padding: 0 18px; color: #1f2937;
       line-height: 1.6; }}
h1 {{ color: #1e40af; border-bottom: 2px solid #1e40af; padding-bottom: 6px; }}
h2 {{ color: #1f2937; margin-top: 28px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 10px; text-align: left; }}
th {{ background: #f1f5f9; }}
.kv span {{ display: inline-block; min-width: 220px; }}
.note {{ background: #fff7ed; border-left: 4px solid #fdba74;
         padding: 10px 14px; margin: 12px 0; font-size: 14px; }}
</style></head><body>

<h1>RoFA-SemEval Dataset Report</h1>
<div class="note">
  报告生成时间：{report['validated_at']}<br>
  本报告仅作为 <b>事实陈述</b> 与 <b>大纲对照</b>，不会自动剔除类别或样本。
</div>

<h2>1. 总体规模</h2>
<div class="kv">
  <span>已标注样本：<b>{totals['annotated_samples']}</b></span>
  <span>通过完整性校验：<b>{totals['passed_integrity']}</b></span><br>
  <span>未通过被拒：<b>{totals['rejected_for_integrity']}</b></span>
  <span>标注员主动删除：<b>{totals['discarded_by_annotator']}</b></span><br>
  <span>累计抓拍：<b>{totals['captured_total']}</b></span>
  <span>最终类别数：<b>{totals['classes_in_dataset']}</b></span>
</div>

<h2>2. 各类别统计</h2>
<table>
  <thead><tr><th>name</th><th>name_zh</th><th>captured</th><th>annotated</th>
  <th>discarded</th><th>rejected</th><th>in_dataset</th><th>vs_outline</th></tr></thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>

<h2>3. 类别合并建议（不自动执行）</h2>
{
    "<p>无可疑相似类别。</p>"
    if not sug_rows
    else f'''<table><thead><tr><th>class_a</th><th>class_b</th><th>reasons</th></tr></thead>
<tbody>{''.join(sug_rows)}</tbody></table>'''
}

<h2>4. 与大纲推荐刻度对照</h2>
<table>
  <thead><tr><th>维度</th><th>大纲推荐</th><th>实际</th><th>差距</th></tr></thead>
  <tbody>
    <tr><td>类别数</td><td>{targets['num_classes']}</td>
        <td>{totals['classes_in_dataset']}</td>
        <td>{gap['classes_short_by']}</td></tr>
    <tr><td>样本数</td><td>{targets['num_samples']}</td>
        <td>{totals['passed_integrity']}</td>
        <td>{gap['samples_short_by']}</td></tr>
    <tr><td>每类下限</td><td>{targets['per_class_min']}</td><td>—</td><td>—</td></tr>
    <tr><td>每类上限</td><td>{targets['per_class_max']}</td><td>—</td><td>—</td></tr>
  </tbody>
</table>

</body></html>
"""
    return html


# --------------------------------------------------------------------------- #
# vs_outline 文案
# --------------------------------------------------------------------------- #

def vs_outline_label(in_dataset_count: int, targets: Dict[str, int]) -> str:
    lo = targets["per_class_min"]
    hi = targets["per_class_max"]
    if in_dataset_count == 0:
        return "空类（已被自动从最终类别表移除）"
    if in_dataset_count < 5:
        return "极稀缺类，建议人工决定保留 / 补采 / 删除"
    if in_dataset_count < lo:
        return f"低于推荐下限 {lo}，仅作参考"
    if in_dataset_count > hi:
        return f"高于推荐上限 {hi}，仅作参考"
    return f"在推荐区间 {lo}~{hi} 内"


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RoFA-SemEval 发布脚本（阶段 3）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument(
        "--exclude-class",
        action="append",
        default=[],
        help="排除指定类别（可重复指定）。例：--exclude-class door_handle",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验、生成报告，不实际拷贝/链接到 dataset/",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    excluded = set(args.exclude_class or [])

    if not raw_root.exists():
        print(f"[finalize] raw_root 不存在: {raw_root}")
        return 1
    print(f"[finalize] raw_root     = {raw_root}")
    print(f"[finalize] dataset_root = {dataset_root}")
    if excluded:
        print(f"[finalize] excluded classes = {sorted(excluded)}")
    if args.dry_run:
        print("[finalize] dry-run mode: 不会写入 dataset/")

    classes_doc = load_classes(raw_root / "classes.json")

    # 1) 校验所有 annotated 样本
    annotated_samples = list_annotated_samples(raw_root)
    print(f"[finalize] 共 {len(annotated_samples)} 个 annotated 样本待校验")

    rejected_rows: List[Tuple[str, str, str]] = []   # (class, sample_id, reasons)
    accepted: List[Path] = []

    for sdir in annotated_samples:
        cls_slug = sdir.parent.name
        sample_id = sdir.name
        if cls_slug in excluded:
            rejected_rows.append((cls_slug, sample_id, "excluded_by_user"))
            continue

        passed, reasons = _check_sample(sdir)
        if not passed:
            rejected_rows.append((cls_slug, sample_id, ";".join(reasons)))
            continue

        cls_err = _validate_class_id(sdir, classes_doc)
        if cls_err:
            rejected_rows.append((cls_slug, sample_id, cls_err))
            continue

        accepted.append(sdir)

    # 2) 拷贝/硬链到 dataset/samples/
    dataset_samples_root = dataset_root / "samples"
    samples_index: List[Dict[str, Any]] = []
    in_dataset_per_class: Dict[str, int] = defaultdict(int)
    link_stats = defaultdict(int)

    if not args.dry_run:
        ensure_dir(dataset_samples_root)

    for sdir in accepted:
        cls_slug = sdir.parent.name
        sample_id = sdir.name
        rel_dir = f"samples/{cls_slug}/{sample_id}"

        if not args.dry_run:
            dst, stats = export_sample(sdir, raw_root, dataset_samples_root)
            for k, v in stats.items():
                link_stats[k] += v

        cls_obj = next(
            (c for c in classes_doc.get("classes", []) if c.get("name") == cls_slug),
            None,
        )
        cls_id = cls_obj["id"] if cls_obj else -1
        cls_name_zh = cls_obj.get("name_zh", "") if cls_obj else ""

        samples_index.append({
            "sample_id": sample_id,
            "class_id": cls_id,
            "class_name": cls_slug,
            "class_name_zh": cls_name_zh,
            "sample_dir": rel_dir,
        })
        in_dataset_per_class[cls_slug] += 1

    # 3) 写 dataset/classes.json：仅保留至少有 1 条样本的类
    final_classes = [
        {
            "id": c["id"],
            "name": c["name"],
            "name_zh": c.get("name_zh", ""),
            "aliases": c.get("aliases", []),
            "in_dataset": int(in_dataset_per_class.get(c["name"], 0)),
        }
        for c in classes_doc.get("classes", [])
        if in_dataset_per_class.get(c["name"], 0) > 0
    ]

    # 4) 统计每类（用于报告）
    annotated_per_class = defaultdict(int)
    for sdir in annotated_samples:
        annotated_per_class[sdir.parent.name] += 1

    discarded_per_class = defaultdict(int)
    discarded_root = raw_root / "discarded"
    if discarded_root.exists():
        for cdir in discarded_root.iterdir():
            if cdir.is_dir():
                discarded_per_class[cdir.name] = sum(
                    1 for x in cdir.iterdir() if x.is_dir()
                )

    rejected_per_class = defaultdict(int)
    for cls_slug, _sid, _reason in rejected_rows:
        rejected_per_class[cls_slug] += 1

    per_class_report: List[Dict[str, Any]] = []
    for c in classes_doc.get("classes", []):
        in_ds = int(in_dataset_per_class.get(c["name"], 0))
        per_class_report.append({
            "name": c["name"],
            "name_zh": c.get("name_zh", ""),
            "captured": int(c.get("captured_count", 0)),
            "annotated": int(annotated_per_class.get(c["name"], 0)),
            "discarded": int(discarded_per_class.get(c["name"], 0)),
            "rejected": int(rejected_per_class.get(c["name"], 0)),
            "in_dataset": in_ds,
            "vs_outline": vs_outline_label(in_ds, OUTLINE_TARGETS),
        })

    suggestions = find_class_merge_suggestions(
        classes_doc, [c["name"] for c in final_classes]
    )

    captured_total = sum(
        int(c.get("captured_count", 0)) for c in classes_doc.get("classes", [])
    )

    report = {
        "validated_at": now_iso(),
        "totals": {
            "annotated_samples": len(annotated_samples),
            "passed_integrity": len(accepted),
            "rejected_for_integrity": len(rejected_rows),
            "discarded_by_annotator": sum(discarded_per_class.values()),
            "captured_total": captured_total,
            "classes_in_dataset": len(final_classes),
        },
        "per_class": per_class_report,
        "merge_suggestions": suggestions,
        "outline_targets": OUTLINE_TARGETS,
        "outline_gap": {
            "classes_short_by": max(0, OUTLINE_TARGETS["num_classes"] - len(final_classes)),
            "samples_short_by": max(0, OUTLINE_TARGETS["num_samples"] - len(accepted)),
        },
        "link_stats": dict(link_stats),
        "raw_root": str(raw_root),
        "dataset_root": str(dataset_root),
        "excluded_classes": sorted(excluded),
        "dry_run": bool(args.dry_run),
    }

    # 5) 落盘 dataset/dataset.json + classes.json + samples.json + report
    if not args.dry_run:
        dataset_json = {
            "name": "RoFA-SemEval",
            "version": "1.0",
            "task": "indicator_4.2_env_semantic_accuracy",
            "world_frame": "camera",
            "pose_source": "dummy_identity",
            "depth_unit": "millimeter",
            "depth_scale": 0.001,
            "camera_model": "Intel RealSense D435",
            "image": {"width": 640, "height": 480, "fps": 30},
            "depth": {"width": 640, "height": 480, "aligned_to": "color"},
            "iou_thresholds": [0.25, 0.50],
            "reference_scale_from_outline": OUTLINE_TARGETS,
            "actual_stats": {
                "num_classes": len(final_classes),
                "num_samples": len(accepted),
                "per_class_distribution": {
                    c["name"]: c["in_dataset"] for c in per_class_report if c["in_dataset"] > 0
                },
                "validated_at": report["validated_at"],
            },
            "created_at": now_iso(),
        }
        save_json(dataset_root / "dataset.json", dataset_json)
        save_json(dataset_root / "classes.json", {
            "version": 1,
            "managed_by": "finalize_dataset.py",
            "classes": final_classes,
        })
        save_json(dataset_root / "samples.json", samples_index)
        save_json(dataset_root / "dataset_report.json", report)
        (dataset_root / "dataset_report.html").write_text(
            render_report_html(report), encoding="utf-8"
        )

    # 6) rejected.csv 始终输出
    target_rejected = (
        dataset_root / "rejected.csv" if not args.dry_run
        else raw_root / "rejected.csv"
    )
    ensure_dir(target_rejected.parent)
    with target_rejected.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "sample_id", "reasons"])
        writer.writerows(rejected_rows)

    # 7) 打印简要总结
    print("\n=== finalize summary ===")
    print(f"  annotated_samples       : {len(annotated_samples)}")
    print(f"  passed_integrity        : {len(accepted)}")
    print(f"  rejected_for_integrity  : {len(rejected_rows)}")
    print(f"  discarded_by_annotator  : {sum(discarded_per_class.values())}")
    print(f"  captured_total          : {captured_total}")
    print(f"  classes_in_dataset      : {len(final_classes)}")
    if suggestions:
        print(f"  merge_suggestions       : {len(suggestions)} (见 dataset_report)")
    if not args.dry_run:
        print(f"  link_stats              : {dict(link_stats)}")
        print(f"  → wrote {dataset_root}")
    else:
        print(f"  (dry-run) rejected csv  : {target_rejected}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[finalize] interrupted")
        sys.exit(130)
    except Exception as exc:
        print(f"[finalize][fatal] {exc}")
        traceback.print_exc()
        sys.exit(2)
