"""
RoFA-SemEval 阶段 2.5：批量 AI 预标
=====================================

把 raw_capture/pending/ 下所有样本拿去问 VLM server (RynnBrain + SAM2)：
- 模型成功 → 走完整的 反投影 / 去噪 / AABB / bbox_2d 流程，**整目录搬到
  raw_capture/auto_labeled/<class>/<sample>/**，等你后审。
- 模型失败 / mask 太小 / 点云退化 → 留在 pending/，回头你用 annotate_sample.py
  走人工流程。

设计要点
--------
1. **不打开任何 OpenCV 窗口**，所以可以纯 headless 长跑、断点续标。
2. **断点续标**：检测到 sample_dir 已经有 mask.png/sample.json 之类的"产物
   文件"就跳过，直接复用。这意味着你可以 Ctrl-C 后接着跑。
3. **零代码改动到 annotate_sample.py**：完全 import 复用现成函数。
4. **不影响另一台机器的人工标注**：predict 失败的样本原地不动，留给那台机器。
5. **每条预标失败都会写一行日志**到 raw_capture/auto_prelabel.log，便于事后查
   原因（超时 / 未找到 / mask 太小 等）。

用法
----
    # server 必须先起来：python real_main_on_server.py
    python scripts/auto_prelabel.py \\
        --raw-root ./RoFA-SemEval/raw_capture \\
        --vlm-host 192.168.x.y --vlm-port 5555 \\
        --annotator alice_auto

    # 只跑某一类（slug）
    python scripts/auto_prelabel.py --class shuihu ...

    # 看一遍统计但不真正落盘
    python scripts/auto_prelabel.py --dry-run ...

样本流转图
----------
    raw_capture/pending/<class>/<id>/
        ├ rgb.jpg, depth.png, intrinsics.json, pose.txt, capture_meta.json
        ▼   (本脚本：模型成功)
    raw_capture/auto_labeled/<class>/<id>/
        ├ ...上面 5 个 + mask.png + points.ply + aabb.json + sample.json
        └ viz_mask.png / viz_aabb.png / viz_aabb_3d.png
        ▼   (再跑 review_auto_labeled.py：人按 y)
    raw_capture/annotated/<class>/<id>/  (与正常人工标注完全同构)

    raw_capture/pending/<class>/<id>/  (本脚本：模型失败 → 留这里)
        ▼   (你用 annotate_sample.py 走人工)
    raw_capture/annotated/<class>/<id>/
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_DIR))

# 复用 annotate_sample.py 的所有逻辑，确保产物 100% 同构
from annotate_sample import (  # noqa: E402
    DEFAULT_DBSCAN_EPS,
    DEFAULT_DBSCAN_MIN_POINTS,
    DEFAULT_DEPTH_MAX_M,
    DEFAULT_DEPTH_MIN_M,
    DEFAULT_RAW_ROOT,
    DEFAULT_SOR_NB,
    DEFAULT_SOR_STD,
    DEFAULT_VLM_HOST,
    DEFAULT_VLM_PORT,
    DEFAULT_VLM_TIMEOUT_MS,
    MIN_MASK_PIXELS,
    MIN_POINTS_AFTER_DENOISE,
    build_sample_json,
    denoise_pointcloud,
    filter_depth_range,
    list_pending_samples,
    load_sample,
    make_search_engine_stub,
    write_annotation_products,
)
from rofa.roimap.search_engine import SearchEngine  # noqa: E402
from _dataset_common import load_classes, load_json, move_dir, save_json  # noqa: E402
from vlm_seg_client import VLMSegClient  # noqa: E402


# 标记一个样本"已经被预标过且产物齐全"的 sentinel 文件
PRELABEL_OUTPUT_FILES = (
    "mask.png", "points.ply", "aabb.json", "sample.json",
    "viz_mask.png",
)


def already_prelabeled(sample_dir: Path) -> bool:
    """
    判断目录是否已经走过完整预标流程（断点续标用）。
    标准：mask.png + sample.json 都在，且 sample.json 里 method ==
    'vlm_predicted_pending_review'。
    """
    if not (sample_dir / "mask.png").exists():
        return False
    sj = load_json(sample_dir / "sample.json", default=None)
    if not isinstance(sj, dict):
        return False
    method = sj.get("annotation", {}).get("method")
    return method == "vlm_predicted_pending_review"


def predict_to_mask(
    client: VLMSegClient,
    rgb_bgr: np.ndarray,
    prompt: str,
    sample_id: str,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], str]:
    """
    给一个样本调一次 server，返回：
        mask (bool [H, W] 或 None)，meta (dict 或 None)，reason
    reason 在 mask=None 时表明失败原因（用于日志）。
    """
    try:
        result = client.predict(rgb_bgr, prompt, anchor_id=sample_id)
    except Exception as exc:
        return None, None, f"client_exception: {exc!r}"

    if result is None:
        return None, None, "predict_returned_none"

    mask = result.get("mask")
    if mask is None:
        if result.get("bbox_pixel"):
            return None, None, "vlm_found_but_sam2_failed"
        return None, None, "object_not_found"

    n_fg = int(mask.sum())
    if n_fg < MIN_MASK_PIXELS:
        return None, None, f"mask_too_small({n_fg}<{MIN_MASK_PIXELS})"

    meta = {
        "host": f"{client.host}:{client.port}",
        "prompt": prompt,
        "bbox_pixel": result.get("bbox_pixel"),
        "fg_pixels": n_fg,
    }
    return mask, meta, "ok"


def auto_prelabel_one(
    sample_dir: Path,
    classes_doc: Dict[str, Any],
    prompt_resolver,
    client: VLMSegClient,
    annotator_id: str,
    args: argparse.Namespace,
) -> Tuple[str, Optional[str]]:
    """
    返回 (status, reason)：
        status ∈ {"ok", "skip_already_done", "fail_predict",
                  "fail_load", "fail_postprocess"}
    """
    sample_id = sample_dir.name
    class_slug = sample_dir.parent.name

    if already_prelabeled(sample_dir):
        return "skip_already_done", "已有完整预标产物"

    bundle = load_sample(sample_dir)
    if bundle is None:
        return "fail_load", "load_sample 返回 None（数据缺失）"

    rgb = bundle["rgb"]
    depth = bundle["depth"]
    intrinsics = bundle["intrinsics"]
    pose = bundle["pose"]
    capture_meta = bundle["capture_meta"]

    prompt = prompt_resolver(class_slug)

    # ---------- 1) 调 server ----------
    mask, vlm_meta, reason = predict_to_mask(client, rgb, prompt, sample_id)
    if mask is None:
        return "fail_predict", reason

    # ---------- 2) 反投影 + 去噪 + AABB + 写产物（纯函数复用） ----------
    try:
        engine = make_search_engine_stub(intrinsics, intrinsics.get("depth_scale", 0.001))
        points_cam = engine._mask_to_world_points(mask, depth, pose)  # noqa: SLF001
        points_cam = filter_depth_range(points_cam, args.depth_min_m, args.depth_max_m)
        if points_cam.shape[0] < MIN_POINTS_AFTER_DENOISE:
            return "fail_postprocess", (
                f"depth_filtered_points_too_few({points_cam.shape[0]}"
                f"<{MIN_POINTS_AFTER_DENOISE})"
            )
        cleaned, denoise_info = denoise_pointcloud(
            points_cam,
            sor_nb=args.sor_nb, sor_std=args.sor_std,
            dbscan_eps=args.dbscan_eps, dbscan_min_points=args.dbscan_min_points,
        )
        if cleaned.shape[0] < MIN_POINTS_AFTER_DENOISE:
            return "fail_postprocess", (
                f"denoised_points_too_few({cleaned.shape[0]}"
                f"<{MIN_POINTS_AFTER_DENOISE})"
            )
        aabb = SearchEngine._compute_aabb(cleaned)  # noqa: SLF001
        if any(e <= 0 for e in aabb["extent"]):
            return "fail_postprocess", f"aabb_degenerate({aabb['extent']})"

        write_annotation_products(
            sample_dir, rgb, mask, cleaned, aabb, engine, pose, denoise_info,
        )
        sample_json = build_sample_json(
            sample_dir, sample_id, class_slug, annotator_id,
            classes_doc, capture_meta, denoise_info,
            used_vlm_mask=True, vlm_meta=vlm_meta,
        )
        # 关键：把 method 改成 "pending_review"，与 review 后的 "vlm_predicted_accepted"
        # 区分，方便区分『AI 标了但还没被人审』和『AI 标完且人审通过了』。
        sample_json["annotation"]["method"] = "vlm_predicted_pending_review"
        save_json(sample_dir / "sample.json", sample_json)

    except Exception as exc:
        # 出错时清理已写的部分产物，避免把 sample 弄成半成品
        for fn in PRELABEL_OUTPUT_FILES:
            fp = sample_dir / fn
            if fp.exists():
                try:
                    fp.unlink()
                except Exception:
                    pass
        return "fail_postprocess", f"exception: {exc!r}\n{traceback.format_exc()}"

    return "ok", None


def move_to_auto_labeled(raw_root: Path, sample_dir: Path) -> Path:
    """把 sample_dir 从 pending/<class>/<id> 整目录搬到 auto_labeled/<class>/<id>"""
    rel = sample_dir.relative_to(raw_root / "pending")  # <class>/<id>
    target = raw_root / "auto_labeled" / rel
    move_dir(sample_dir, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RoFA-SemEval 阶段 2.5：批量 AI 预标（headless）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--annotator", type=str, default="auto_prelabel",
                        help="标注员标识，会写进 sample.json.annotation.annotator")
    parser.add_argument("--class", dest="only_class", type=str, default=None,
                        help="只预标指定类别（slug，可选）")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="最多处理 N 个样本（-1 = 全部），便于试跑")
    parser.add_argument("--dry-run", action="store_true",
                        help="只跑流程不真正搬目录、不真正写产物（连 server 也不调）")

    parser.add_argument("--depth-min-m", type=float, default=DEFAULT_DEPTH_MIN_M)
    parser.add_argument("--depth-max-m", type=float, default=DEFAULT_DEPTH_MAX_M)
    parser.add_argument("--sor-nb", type=int, default=DEFAULT_SOR_NB)
    parser.add_argument("--sor-std", type=float, default=DEFAULT_SOR_STD)
    parser.add_argument("--dbscan-eps", type=float, default=DEFAULT_DBSCAN_EPS)
    parser.add_argument("--dbscan-min-points", type=int, default=DEFAULT_DBSCAN_MIN_POINTS)

    parser.add_argument("--vlm-host", type=str, default=DEFAULT_VLM_HOST)
    parser.add_argument("--vlm-port", type=int, default=DEFAULT_VLM_PORT)
    parser.add_argument("--vlm-timeout-ms", type=int, default=DEFAULT_VLM_TIMEOUT_MS)
    parser.add_argument("--vlm-prompt-map", type=str, default=None,
                        help="可选：JSON 文件覆盖 slug→prompt（默认用 classes.json 的 name_zh）")

    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    if not raw_root.exists():
        print(f"[auto_prelabel] raw_root 不存在: {raw_root}")
        return 1

    classes_path = raw_root / "classes.json"
    classes_doc = load_classes(classes_path)

    samples = list_pending_samples(raw_root, args.only_class)
    if not samples:
        print(f"[auto_prelabel] 没有 pending 样本: {raw_root / 'pending'}")
        return 0
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"[auto_prelabel] 共 {len(samples)} 个 pending 样本待预标")

    # ---------- prompt 解析器（与 annotate_sample.py 一致）----------
    prompt_map: Dict[str, str] = {}
    for c in classes_doc.get("classes", []):
        slug = c.get("name")
        if slug:
            prompt_map[slug] = str(c.get("name_zh") or slug)
    if args.vlm_prompt_map:
        try:
            user_map = load_json(Path(args.vlm_prompt_map).expanduser())
            if isinstance(user_map, dict):
                prompt_map.update({str(k): str(v) for k, v in user_map.items()})
                print(f"[auto_prelabel] 已加载自定义 prompt 映射: {args.vlm_prompt_map}")
        except Exception as exc:
            print(f"[auto_prelabel] 读取 --vlm-prompt-map 失败: {exc}")

    def prompt_resolver(slug: str, _m: Dict[str, str] = prompt_map) -> str:
        return _m.get(slug, slug)

    print(f"[auto_prelabel] slug→prompt: {prompt_map}")

    # ---------- 起 client ----------
    if args.dry_run:
        print("[auto_prelabel] --dry-run 模式：跳过 server 连接和落盘")
        client = None
    else:
        try:
            client = VLMSegClient(
                host=args.vlm_host,
                port=args.vlm_port,
                timeout_ms=args.vlm_timeout_ms,
            )
            print(
                f"[auto_prelabel] ✓ VLM client connected → "
                f"tcp://{args.vlm_host}:{args.vlm_port} "
                f"(timeout={args.vlm_timeout_ms}ms)"
            )
        except Exception as exc:
            print(f"[auto_prelabel] VLM client 初始化失败: {exc}")
            return 2

    # ---------- 主循环 ----------
    log_path = raw_root / "auto_prelabel.log"
    counters = {"ok": 0, "skip_already_done": 0, "fail_predict": 0,
                "fail_load": 0, "fail_postprocess": 0, "moved": 0}
    rc = 0
    t_start = time.time()

    try:
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(
                f"\n=== auto_prelabel session @ {datetime.now().isoformat()}  "
                f"raw_root={raw_root} dry_run={args.dry_run} ===\n"
            )

            for idx, sdir in enumerate(samples, start=1):
                sample_id = sdir.name
                class_slug = sdir.parent.name
                print(f"\n[{idx}/{len(samples)}] {class_slug}/{sample_id}")

                if args.dry_run:
                    print(f"  [dry-run] would call server with prompt='{prompt_resolver(class_slug)}'")
                    continue

                t0 = time.time()
                status, reason = auto_prelabel_one(
                    sdir, classes_doc, prompt_resolver, client,
                    args.annotator, args,
                )
                elapsed = time.time() - t0
                counters[status] = counters.get(status, 0) + 1

                line = (
                    f"{datetime.now().isoformat()}  {class_slug}/{sample_id}  "
                    f"{status}  elapsed={elapsed:.1f}s  reason={reason}\n"
                )
                logf.write(line)
                logf.flush()

                if status == "ok":
                    target = move_to_auto_labeled(raw_root, sdir)
                    counters["moved"] += 1
                    print(f"  ✓ ok ({elapsed:.1f}s) → {target.relative_to(raw_root)}")
                elif status == "skip_already_done":
                    print(f"  · skip (already prelabeled)")
                else:
                    print(f"  ✗ {status}: {reason}  ({elapsed:.1f}s)  → 留 pending/")

    except KeyboardInterrupt:
        print("\n[auto_prelabel] interrupted (已处理的产物已保存)")
    except Exception as exc:
        print(f"[auto_prelabel][fatal] {exc}")
        traceback.print_exc()
        rc = 2
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    total = time.time() - t_start
    print("\n=== auto_prelabel session summary ===")
    for k, v in counters.items():
        print(f"  {k:<22s}: {v}")
    print(f"  total elapsed       : {total:.1f}s "
          f"({total / max(1, len(samples)):.2f}s/sample)")
    print(f"  log file            : {log_path}")
    print()
    print(
        "下一步：\n"
        "  1) 后审：    python scripts/review_auto_labeled.py --raw-root <root>\n"
        "  2) 处理失败：python scripts/annotate_sample.py --raw-root <root>"
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
