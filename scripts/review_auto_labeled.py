"""
RoFA-SemEval 阶段 2.6：批量后审 AI 预标
=========================================

读 raw_capture/auto_labeled/<class>/<sample>/ 下每个样本，复用
annotate_sample.preview_and_decide 的 3D 多视角决策面板，让你按 y/n/d/s/q
快速过一遍预标结果。

按键结果（与 annotate_sample.py 完全一致）：
    y → 接受 → 整目录搬到 raw_capture/annotated/<class>/<sample>/
                同时把 sample.json.annotation.method 改为
                'vlm_predicted_accepted'
    n → 退回 → 整目录搬回 raw_capture/pending/<class>/<sample>/
                **并清理 AI 写的产物**（mask.png / aabb.json / sample.json /
                points.ply / viz_*.png），让你下次跑 annotate_sample.py 时
                这条样本看起来与从未标过完全一样
    d → 删除 → 整目录搬到 raw_capture/discarded/<class>/<sample>/
                附 discard_reason.txt
    s → 跳过 → 留在 auto_labeled/，下次再审
    q → 退出 → 已审过的不丢

设计原则：
- 完全 import 复用 annotate_sample.py 的逻辑，**零代码重复**。
- 不会调用 VLM server，纯本地决策。
- 安全：n 退回时只删"AI 产物"，从不动 rgb.jpg / depth.png / intrinsics.json /
  pose.txt / capture_meta.json 等原始采集数据。

用法：
    python scripts/review_auto_labeled.py \\
        --raw-root ./RoFA-SemEval/raw_capture \\
        --annotator alice
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from annotate_sample import (  # noqa: E402
    DEFAULT_RAW_ROOT,
    preview_and_decide,
    prompt_discard_reason,
    write_discard_reason,
)
from _dataset_common import load_json, move_dir, now_iso, save_json  # noqa: E402


# 与 auto_prelabel.PRELABEL_OUTPUT_FILES 同步：n（退回）时要清理掉的 AI 产物
AI_PRODUCT_FILES = (
    "mask.png", "points.ply", "aabb.json", "sample.json",
    "viz_mask.png", "viz_aabb.png", "viz_aabb_3d.png",
)


def list_auto_labeled_samples(
    raw_root: Path, only_class: Optional[str] = None
) -> List[Path]:
    base = raw_root / "auto_labeled"
    if not base.exists():
        return []
    classes = (
        [base / only_class] if only_class
        else sorted(p for p in base.iterdir() if p.is_dir())
    )
    out: List[Path] = []
    for cdir in classes:
        if not cdir.exists():
            continue
        for sdir in sorted(cdir.iterdir()):
            if not sdir.is_dir():
                continue
            # 必须有完整产物
            if not (sdir / "viz_mask.png").exists():
                continue
            if not (sdir / "sample.json").exists():
                continue
            out.append(sdir)
    return out


def move_between(raw_root: Path, sample_dir: Path, src_subdir: str, dst_subdir: str) -> Path:
    rel = sample_dir.relative_to(raw_root / src_subdir)  # <class>/<id>
    target = raw_root / dst_subdir / rel
    move_dir(sample_dir, target)
    return target


def revert_to_pending(raw_root: Path, sample_dir: Path) -> Path:
    """
    把 sample 从 auto_labeled/ 退回 pending/，删掉 AI 产物。
    这样它在 pending/ 里看起来与刚采集时完全一致，标注员可继续走人工流程。
    """
    target = move_between(raw_root, sample_dir, "auto_labeled", "pending")
    for fn in AI_PRODUCT_FILES:
        fp = target / fn
        if fp.exists():
            try:
                fp.unlink()
            except Exception as exc:
                print(f"[review] 清理 {fp.name} 失败（已忽略）: {exc}")
    return target


def accept_to_annotated(
    raw_root: Path, sample_dir: Path, reviewer_id: str
) -> Path:
    """
    把 sample 从 auto_labeled/ 接受到 annotated/，并更新 sample.json：
        annotation.method  = "vlm_predicted_accepted"
        annotation.reviewer = <reviewer_id>
        annotation.reviewed_at = now_iso()
    """
    target = move_between(raw_root, sample_dir, "auto_labeled", "annotated")

    sj_path = target / "sample.json"
    sj = load_json(sj_path, default=None)
    if isinstance(sj, dict):
        ann = sj.setdefault("annotation", {})
        ann["method"] = "vlm_predicted_accepted"
        ann["reviewer"] = reviewer_id
        ann["reviewed_at"] = now_iso()
        save_json(sj_path, sj)
    return target


def discard_sample(
    raw_root: Path, sample_dir: Path, reviewer_id: str, reason: str
) -> Path:
    target = move_between(raw_root, sample_dir, "auto_labeled", "discarded")
    write_discard_reason(target, reason, reviewer_id)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RoFA-SemEval 阶段 2.6：审核 AI 预标结果",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--annotator", type=str, default="anonymous",
                        help="审核员标识，会写进 annotation.reviewer")
    parser.add_argument("--class", dest="only_class", type=str, default=None,
                        help="只审核指定类别（slug，可选）")
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    if not raw_root.exists():
        print(f"[review] raw_root 不存在: {raw_root}")
        return 1

    samples = list_auto_labeled_samples(raw_root, args.only_class)
    if not samples:
        print(f"[review] 没有待审样本: {raw_root / 'auto_labeled'}")
        return 0
    print(f"[review] 共 {len(samples)} 个待审样本")

    counters = {"accept": 0, "revert": 0, "discard": 0, "skip": 0}
    rc = 0
    try:
        for idx, sdir in enumerate(samples, start=1):
            sample_id = sdir.name
            class_slug = sdir.parent.name
            print(f"\n[{idx}/{len(samples)}] {class_slug}/{sample_id}")

            decision = preview_and_decide(sdir, sample_id, class_slug)

            if decision == "accept":
                target = accept_to_annotated(raw_root, sdir, args.annotator)
                counters["accept"] += 1
                print(f"  ✓ accept → {target.relative_to(raw_root)}")
            elif decision == "redo":
                target = revert_to_pending(raw_root, sdir)
                counters["revert"] += 1
                print(f"  ↩ revert → {target.relative_to(raw_root)}  (AI 产物已清，待人工重做)")
            elif decision == "discard":
                reason = prompt_discard_reason()
                target = discard_sample(raw_root, sdir, args.annotator, reason)
                counters["discard"] += 1
                print(f"  ✗ discard → {target.relative_to(raw_root)} ({reason})")
            elif decision == "skip":
                counters["skip"] += 1
                print(f"  · skip (留 auto_labeled/)")
            elif decision == "quit":
                print("[review] q 退出")
                break

    except KeyboardInterrupt:
        print("\n[review] interrupted (已审过的不丢)")
    except Exception as exc:
        print(f"[review][fatal] {exc}")
        traceback.print_exc()
        rc = 2
    finally:
        cv2.destroyAllWindows()

    print("\n=== review session summary ===")
    for k, v in counters.items():
        print(f"  {k:<10s}: {v}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
