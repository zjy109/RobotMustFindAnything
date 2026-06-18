#!/usr/bin/env python3
"""使用 Intel RealSense D4xx（如 D435）采集 RGBD 样本，并附加随机模拟位姿。

每个样本包含：
    rgb.jpg          已对齐到彩色坐标系的 RGB
    depth.png        16-bit 深度（与彩色对齐，单位由 intrinsics.depth_scale 决定）
    intrinsics.json  {fx, fy, cx, cy, depth_scale, width, height}
    pose.json        随机模拟的相机->世界位姿

采集结果可直接用 `python ui.py --capture-dir <output>` 加载查看与检索。

用法：
    # 交互式（弹预览窗，空格/回车采一帧，q/ESC 退出）
    python sample_rsd4xx.py --output ./captures

    # 无人值守，自动采集 20 帧，每帧间隔 1 秒（无窗口，适合 SSH/无显示环境）
    python sample_rsd4xx.py --output ./captures --num 20 --auto --interval 1.0

    # 没有真机？用 RealSense 官方 .bag 录制文件回放采集（无需任何硬件）
    python sample_rsd4xx.py --from-bag ./sample.bag --output ./captures --num 10

依赖：pyrealsense2（见 requirements.txt / install.sh）。

获取测试用 .bag（无相机时）：
    Intel 官方样例数据 https://github.com/IntelRealSense/librealsense/blob/master/doc/sample-data.md
    例如： wget https://librealsense.intel.com/rs-tests/TestData/outdoors.bag
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# 让脚本能 import 项目内 src/benchmark
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.capture_io import random_pose, save_sample  # noqa: E402


def _import_realsense():
    try:
        import pyrealsense2 as rs  # noqa: WPS433
        return rs
    except ImportError:
        print(
            "[error] 未安装 pyrealsense2。请先安装：\n"
            "    pip install pyrealsense2\n"
            "（Linux x86_64 / Windows 有官方 wheel；其它平台请参考 RealSense SDK 文档）",
            file=sys.stderr,
        )
        sys.exit(2)


def build_intrinsics(color_intr, depth_scale: float) -> dict:
    """RealSense 彩色流内参 + depth_scale -> 评测程序使用的 intrinsics dict。"""
    return {
        "fx": float(color_intr.fx),
        "fy": float(color_intr.fy),
        "cx": float(color_intr.ppx),
        "cy": float(color_intr.ppy),
        "depth_scale": float(depth_scale),
        "width": int(color_intr.width),
        "height": int(color_intr.height),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="RealSense D4xx RGBD 采集 + 随机位姿",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output", type=str, default="./captures",
                    help="采集结果输出目录（默认 ./captures）")
    ap.add_argument("--num", type=int, default=-1,
                    help="采集样本数上限；-1 表示不限制（交互式手动结束）")
    ap.add_argument("--auto", action="store_true",
                    help="自动连续采集（无预览窗口），配合 --num / --interval")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="--auto 模式下每帧之间的间隔秒数（默认 1.0）")
    ap.add_argument("--width", type=int, default=640, help="分辨率宽（默认 640）")
    ap.add_argument("--height", type=int, default=480, help="分辨率高（默认 480）")
    ap.add_argument("--fps", type=int, default=30, help="帧率（默认 30）")
    ap.add_argument("--warmup", type=int, default=15,
                    help="启动后丢弃的预热帧数，等待自动曝光稳定（默认 15；bag 回放时忽略）")
    ap.add_argument("--seed", type=int, default=None,
                    help="随机位姿种子（可选，便于复现）")
    ap.add_argument("--from-bag", type=str, default=None,
                    help="从 RealSense .bag 录制文件回放采集（无需真机）；指定后忽略预览窗")
    ap.add_argument("--bag-stride", type=int, default=30,
                    help="--from-bag 模式下每隔多少帧采一帧（默认 30，即约每秒 1 帧）")
    args = ap.parse_args()

    rs = _import_realsense()
    rng = np.random.default_rng(args.seed)
    from_bag = args.from_bag is not None

    capture_dir = Path(args.output).expanduser().resolve()
    capture_dir.mkdir(parents=True, exist_ok=True)
    print(f"[capture] 输出目录: {capture_dir}")

    # ---- 配置并启动 pipeline ----
    pipeline = rs.pipeline()
    config = rs.config()
    if from_bag:
        bag_path = Path(args.from_bag).expanduser().resolve()
        if not bag_path.exists():
            print(f"[error] .bag 文件不存在: {bag_path}", file=sys.stderr)
            return 2
        # 让设备来源于录制文件；流由 bag 内容决定，不手动 enable_stream
        config.enable_device_from_file(str(bag_path), repeat_playback=False)
        print(f"[capture] 从 .bag 回放: {bag_path}")
    else:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    print("[capture] 启动 RealSense pipeline ...")
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # 米 / 单位
    align = rs.align(rs.stream.color)  # 把深度对齐到彩色
    print(f"[capture] depth_scale = {depth_scale} (米/单位)")

    if from_bag:
        # 非实时回放：逐帧处理，不丢帧
        try:
            playback = profile.get_device().as_playback()
            playback.set_real_time(False)
        except Exception:
            pass

    # 模式：bag 回放 / 自动定时 / 交互预览
    interactive = (not from_bag) and (not args.auto)
    cv2 = None
    if interactive:
        import cv2 as _cv2  # noqa: WPS433
        cv2 = _cv2

    saved = 0
    frame_idx = 0
    try:
        # 预热：丢弃前若干帧（bag 回放时帧数有限，跳过预热）
        if not from_bag:
            for _ in range(max(0, args.warmup)):
                pipeline.wait_for_frames()

        if interactive:
            print("[capture] 交互模式：在预览窗口中按 <空格>/<回车> 采集一帧，按 q/ESC 退出。")
        elif from_bag:
            print(f"[capture] bag 回放模式：每隔 {args.bag_stride} 帧采一帧。")

        last_auto = 0.0
        while True:
            if args.num >= 0 and saved >= args.num:
                break

            if from_bag:
                # 回放结束时 try_wait_for_frames 返回 False
                ok, frames = pipeline.try_wait_for_frames(2000)
                if not ok:
                    print("[capture] .bag 已播放完毕。")
                    break
            else:
                frames = pipeline.wait_for_frames()

            frame_idx += 1
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_intr = color_frame.profile.as_video_stream_profile().intrinsics
            color_image = np.asanyarray(color_frame.get_data())          # BGR uint8
            depth_image = np.asanyarray(depth_frame.get_data()).astype(np.uint16)

            do_capture = False
            if from_bag:
                do_capture = (frame_idx % max(1, args.bag_stride) == 0)
            elif args.auto:
                now = time.time()
                if now - last_auto >= args.interval:
                    do_capture = True
                    last_auto = now
            else:
                # 预览：彩色 + 深度伪彩拼接
                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET,
                )
                banner = color_image.copy()
                cv2.putText(banner, f"saved={saved}  SPACE/ENTER=capture  q/ESC=quit",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("RealSense D4xx 采集 (color | depth)",
                           np.hstack([banner, depth_vis]))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q / ESC
                    break
                if key in (ord(" "), 13, 10):  # space / enter
                    do_capture = True

            if not do_capture:
                continue

            sample_id = f"rsd_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            intrinsics = build_intrinsics(color_intr, depth_scale)
            pose = random_pose(rng)
            save_sample(capture_dir, sample_id, color_image, depth_image, intrinsics, pose)
            saved += 1
            print(f"[capture] ✓ {saved} -> {sample_id}")

    except KeyboardInterrupt:
        print("\n[capture] 用户中断。")
    finally:
        pipeline.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()

    print(f"[capture] 完成，共采集 {saved} 个样本，索引: {capture_dir / 'samples.json'}")
    print(f"[capture] 下一步：python ui.py --capture-dir {capture_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
