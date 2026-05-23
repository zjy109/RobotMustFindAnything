"""
RoFA-SemEval 阶段 1：采集脚本

采集员视角：输入物品名 → 实时预览 → 空格抓拍 → 退出。
- 不画 mask，不计算点云 / AABB
- 类别由用户运行时输入并动态扩展，写入 raw_capture/classes.json
- 每条样本以 {slug}_{NNNN} 命名，落盘到 raw_capture/pending/<slug>/
- 位姿统一占位为 4x4 单位阵（dummy）

依赖：
    pip install pyrealsense2 opencv-python "numpy<2.0"
    可选：pip install pypinyin   # 中文类别名转拼音 slug

退出码：
    0  正常退出
    1  RealSense 初始化失败
    2  其他运行时异常
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None  # 启动时再检查，方便在无 D435 机器上做 --help

# 允许直接 `python scripts/capture_sample.py` 运行
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dataset_common import (  # noqa: E402
    FileLock,
    append_alias,
    bump_captured_count,
    create_class,
    ensure_dir,
    find_class_by_alias,
    fuzzy_match_classes,
    load_classes,
    next_sample_id,
    now_iso,
    save_json,
)


# --------------------------------------------------------------------------- #
# 默认参数
# --------------------------------------------------------------------------- #
DEFAULT_RAW_ROOT = Path(__file__).resolve().parents[1] / "RoFA-SemEval" / "raw_capture"
DEFAULT_RGB_W, DEFAULT_RGB_H = 640, 480
DEFAULT_DEPTH_W, DEFAULT_DEPTH_H = 640, 480
DEFAULT_FPS = 30
JPEG_QUALITY = 95


# --------------------------------------------------------------------------- #
# 终端交互：解析类别输入
# --------------------------------------------------------------------------- #

def prompt_class(classes_doc: Dict[str, Any], prompt_msg: str) -> Optional[Dict[str, Any]]:
    """
    交互式输入类别名。返回类别 obj，None 表示用户放弃。
    会就地修改 classes_doc（追加 alias / 新建类）。
    """
    while True:
        try:
            raw = input(prompt_msg).strip()
        except EOFError:
            return None
        if not raw:
            print("  [skip] 输入为空，请重新输入（或 Ctrl-C 中止）")
            continue

        # 1) 精确 alias 命中
        existing = find_class_by_alias(classes_doc, raw)
        if existing is not None:
            print(
                f"  [match] 复用已有类别: id={existing['id']} "
                f"name={existing['name']} (zh={existing.get('name_zh','')})"
            )
            return existing

        # 2) 模糊匹配
        candidates = fuzzy_match_classes(classes_doc, raw, top_k=3, max_edit=3)
        if candidates:
            print("  [fuzzy] 发现相似类别：")
            for i, (c, dist) in enumerate(candidates):
                print(
                    f"    [{i}] id={c['id']} name={c['name']} "
                    f"zh={c.get('name_zh','')} aliases={c.get('aliases', [])} "
                    f"(distance={dist})"
                )
            print("  输入 0/1/2 选择复用，或输入 N 新建该类，或输入 R 重新输入：")
            try:
                choice = input("  > ").strip().lower()
            except EOFError:
                return None
            if choice == "r":
                continue
            if choice == "n":
                pass  # fall through 到新建
            elif choice.isdigit() and 0 <= int(choice) < len(candidates):
                chosen = candidates[int(choice)][0]
                if append_alias(chosen, raw):
                    print(f"  [alias] 已为 {chosen['name']} 追加新写法: {raw!r}")
                return chosen
            else:
                print("  [err] 无效选择，请重试")
                continue

        # 3) 新建
        new_class = create_class(classes_doc, raw)
        print(
            f"  [new] 新建类别 id={new_class['id']} name={new_class['name']} "
            f"zh={new_class.get('name_zh','')}"
        )
        return new_class


# --------------------------------------------------------------------------- #
# RealSense 初始化
# --------------------------------------------------------------------------- #

def init_realsense(rgb_size: Tuple[int, int], depth_size: Tuple[int, int], fps: int):
    if rs is None:
        raise RuntimeError(
            "未安装 pyrealsense2。请先 `pip install pyrealsense2`，或确认 D435 已连接。"
        )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, rgb_size[0], rgb_size[1], rs.format.bgr8, fps
    )
    config.enable_stream(
        rs.stream.depth, depth_size[0], depth_size[1], rs.format.z16, fps
    )
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())  # 通常 0.001

    device_serial = ""
    try:
        device_serial = profile.get_device().get_info(rs.camera_info.serial_number)
    except Exception:
        pass

    return pipeline, align, intr, depth_scale, device_serial


def intrinsics_to_dict(intr, depth_scale: float) -> Dict[str, Any]:
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "model": "pinhole",
        "distortion": [float(c) for c in intr.coeffs],
        "depth_scale": float(depth_scale),
    }


# --------------------------------------------------------------------------- #
# 深度质量检查
# --------------------------------------------------------------------------- #

def depth_quality_ok(
    depth_u16: np.ndarray,
    min_valid_ratio: float = 0.5,
    center_box: float = 0.4,
) -> Tuple[bool, str]:
    """
    返回 (ok, reason)。
    - 整图有效深度像素占比 ≥ min_valid_ratio
    - 中心 center_box 区域至少有非零深度
    """
    if depth_u16 is None or depth_u16.size == 0:
        return False, "depth empty"
    h, w = depth_u16.shape[:2]
    valid = depth_u16 > 0
    valid_ratio = float(valid.mean())
    if valid_ratio < min_valid_ratio:
        return False, f"valid_ratio={valid_ratio:.2f} < {min_valid_ratio}"
    cy0 = int(h * (0.5 - center_box / 2))
    cy1 = int(h * (0.5 + center_box / 2))
    cx0 = int(w * (0.5 - center_box / 2))
    cx1 = int(w * (0.5 + center_box / 2))
    if not (depth_u16[cy0:cy1, cx0:cx1] > 0).any():
        return False, "center region all-zero"
    return True, ""


def depth_to_vis(depth_u16: np.ndarray) -> np.ndarray:
    """伪彩色深度图，用于预览。"""
    d = depth_u16.astype(np.float32)
    pos = d[d > 0]
    vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
    scaled = np.clip(d / max(vmax, 1.0) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


# --------------------------------------------------------------------------- #
# UI overlay
# --------------------------------------------------------------------------- #

def render_status_panel(
    color_bgr: np.ndarray,
    depth_vis: np.ndarray,
    current_class: Optional[Dict[str, Any]],
    total_captured: int,
    last_msg: str,
) -> np.ndarray:
    panel = color_bgr.copy()
    h = panel.shape[0]

    cls_line = (
        f"class: {current_class['name']} "
        f"(zh={current_class.get('name_zh','')}, "
        f"captured={current_class.get('captured_count', 0)})"
        if current_class
        else "class: <not set>"
    )
    lines = [
        cls_line,
        f"total captured (all classes): {total_captured}",
        "keys: SPACE capture | n switch class | q quit",
    ]
    if last_msg:
        lines.append(f"> {last_msg}")
    for i, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (10, 24 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return np.hstack([panel, depth_vis])


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #

def write_pose_identity(path: Path) -> None:
    np.savetxt(path, np.eye(4, dtype=np.float32), fmt="%.6f")


def save_sample(
    raw_root: Path,
    class_obj: Dict[str, Any],
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    intrinsics: Dict[str, Any],
    operator: str,
    device_info: Dict[str, Any],
) -> Path:
    pending_root = raw_root / "pending"
    class_dir = pending_root / class_obj["name"]
    ensure_dir(class_dir)

    sample_id = next_sample_id(class_dir, class_obj["name"])
    sample_dir = class_dir / sample_id
    ensure_dir(sample_dir)

    # RGB
    cv2.imwrite(
        str(sample_dir / "rgb.jpg"),
        color_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    # Depth (uint16, 单位 mm，假设 depth_scale=0.001)
    cv2.imwrite(str(sample_dir / "depth.png"), depth_u16)
    # Intrinsics
    save_json(sample_dir / "intrinsics.json", intrinsics)
    # Pose: dummy 单位阵
    write_pose_identity(sample_dir / "pose.txt")
    # capture_meta
    save_json(
        sample_dir / "capture_meta.json",
        {
            "sample_id": sample_id,
            "class_id": class_obj["id"],
            "class_name": class_obj["name"],
            "class_name_zh": class_obj.get("name_zh", ""),
            "class_input_aliases_at_capture": list(class_obj.get("aliases", [])),
            "captured_at": now_iso(),
            "captured_by": operator,
            "device": device_info,
            "image": {
                "width": int(color_bgr.shape[1]),
                "height": int(color_bgr.shape[0]),
            },
            "depth": {
                "width": int(depth_u16.shape[1]),
                "height": int(depth_u16.shape[0]),
                "aligned_to": "color",
                "depth_scale": float(intrinsics.get("depth_scale", 0.001)),
            },
            "pose_source": "dummy_identity",
        },
    )
    return sample_dir


# --------------------------------------------------------------------------- #
# 主循环
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RoFA-SemEval 采集脚本（阶段 1）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=str(DEFAULT_RAW_ROOT),
        help="raw_capture 根目录",
    )
    parser.add_argument("--operator", type=str, default="", help="采集员标识，可选")
    parser.add_argument("--rgb-w", type=int, default=DEFAULT_RGB_W)
    parser.add_argument("--rgb-h", type=int, default=DEFAULT_RGB_H)
    parser.add_argument("--depth-w", type=int, default=DEFAULT_DEPTH_W)
    parser.add_argument("--depth-h", type=int, default=DEFAULT_DEPTH_H)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--no-quality-check",
        action="store_true",
        help="关闭抓拍前的深度质量检查（调试用）",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    ensure_dir(raw_root)
    classes_path = raw_root / "classes.json"

    print(f"[capture] raw root = {raw_root}")
    print(f"[capture] classes file = {classes_path}")

    # 1) 启动 RealSense
    try:
        pipeline, align, intr, depth_scale, device_serial = init_realsense(
            (args.rgb_w, args.rgb_h),
            (args.depth_w, args.depth_h),
            args.fps,
        )
    except Exception as exc:
        print(f"[capture][fatal] RealSense 初始化失败: {exc}")
        traceback.print_exc()
        return 1

    intrinsics = intrinsics_to_dict(intr, depth_scale)
    device_info = {"model": "Intel RealSense D435", "serial": device_serial}
    print(f"[capture] intrinsics = {json.dumps(intrinsics, ensure_ascii=False)}")
    print(f"[capture] device = {device_info}")

    # 2) 读取/初始化 classes.json
    with FileLock(classes_path):
        classes_doc = load_classes(classes_path)
        save_json(classes_path, classes_doc)  # 保证文件存在

    # 3) 第一次输入类别
    print("\n=== 输入第一个采集物品的名称（中英文均可），Ctrl-C 中止 ===")
    current_class = None
    while current_class is None:
        with FileLock(classes_path):
            classes_doc = load_classes(classes_path)
            current_class = prompt_class(classes_doc, "class> ")
            if current_class is None:
                print("[capture] 已放弃，退出。")
                pipeline.stop()
                return 0
            save_json(classes_path, classes_doc)

    # 4) 主循环
    cv2.namedWindow("RoFA Capture", cv2.WINDOW_AUTOSIZE)
    last_msg = ""
    session_count = 0
    rc = 0
    try:
        while True:
            try:
                frames = align.process(pipeline.wait_for_frames())
            except Exception as exc:
                last_msg = f"frame error: {exc}"
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_u16 = np.asanyarray(depth_frame.get_data())  # uint16, mm

            depth_vis = depth_to_vis(depth_u16)
            # 计算总样本数（粗略：直接用所有类的 captured_count 之和）
            total_captured = sum(
                int(c.get("captured_count", 0)) for c in classes_doc.get("classes", [])
            )
            panel = render_status_panel(
                color_bgr, depth_vis, current_class, total_captured, last_msg
            )
            cv2.imshow("RoFA Capture", panel)

            key = cv2.waitKey(1) & 0xFF
            if key == 0xFF:
                continue

            if key == ord("q"):
                print("[capture] q 退出")
                break

            if key == ord("n"):
                # 切换类别
                with FileLock(classes_path):
                    classes_doc = load_classes(classes_path)
                    new_cls = prompt_class(classes_doc, "switch to class> ")
                    save_json(classes_path, classes_doc)
                if new_cls is not None:
                    current_class = new_cls
                    last_msg = f"switched to {current_class['name']}"
                else:
                    last_msg = "switch cancelled"
                continue

            if key == ord(" "):
                if not args.no_quality_check:
                    ok, reason = depth_quality_ok(depth_u16)
                    if not ok:
                        last_msg = f"depth quality fail: {reason}"
                        print(f"[capture] reject capture: {reason}")
                        continue

                with FileLock(classes_path):
                    classes_doc = load_classes(classes_path)
                    # 把 current_class 同步成磁盘最新版本
                    fresh = next(
                        (c for c in classes_doc["classes"]
                         if c["id"] == current_class["id"]),
                        None,
                    )
                    if fresh is None:
                        # 如果磁盘上找不到（被外部修改），回退用本地副本写回
                        classes_doc["classes"].append(current_class)
                        fresh = current_class
                    bump_captured_count(fresh)
                    sample_dir = save_sample(
                        raw_root,
                        fresh,
                        color_bgr,
                        depth_u16,
                        intrinsics,
                        args.operator,
                        device_info,
                    )
                    save_json(classes_path, classes_doc)
                    current_class = fresh

                session_count += 1
                last_msg = f"saved {sample_dir.name}"
                print(
                    f"[capture] saved {sample_dir.relative_to(raw_root)} "
                    f"(class={fresh['name']}, count={fresh['captured_count']})"
                )
                continue

    except KeyboardInterrupt:
        print("\n[capture] interrupted")
    except Exception as exc:
        print(f"[capture][fatal] {exc}")
        traceback.print_exc()
        rc = 2
    finally:
        cv2.destroyAllWindows()
        try:
            pipeline.stop()
        except Exception:
            pass

    # 5) 退出统计
    print("\n=== capture session summary ===")
    print(f"  saved this session : {session_count}")
    classes_doc = load_classes(classes_path)
    grand_total = sum(
        int(c.get("captured_count", 0)) for c in classes_doc.get("classes", [])
    )
    print(f"  cumulative captured: {grand_total}")
    print(f"  classes seen so far: {len(classes_doc.get('classes', []))}")
    for c in classes_doc.get("classes", []):
        print(
            f"    - {c['name']:<24s} "
            f"zh={c.get('name_zh',''):<8s} captured={c.get('captured_count', 0)}"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
