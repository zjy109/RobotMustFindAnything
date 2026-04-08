import argparse
from pathlib import Path

import cv2
import numpy as np

from main_on_robot import MainOnRobot

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STREAM_ROOT = SCRIPT_DIR / "benchmark" / "office2"
DEFAULT_ROIMAP_ROOT = SCRIPT_DIR / "roimap_fixed_data"


class PosedRGBDStream:
    def __init__(self, root):
        self.root = Path(root)
        self.ids = sorted(p.stem.replace(".color", "") for p in self.root.glob("*.color.jpg"))
        self.i = 0

    @staticmethod
    def _load_camera_pose(path):
        pose = np.loadtxt(path, dtype=np.float32)
        pose = np.asarray(pose, dtype=np.float32)
        if pose.shape == (16,):
            pose = pose.reshape(4, 4)
        if pose.shape != (4, 4):
            return None
        if not np.all(np.isfinite(pose)):
            return None
        return pose

    def next_posed_rgbd(self):
        if not self.ids:
            raise RuntimeError(f"No frames found in {self.root}")
        while self.i < len(self.ids):
            frame_id = self.ids[self.i]
            self.i += 1

            rgb = cv2.imread(str(self.root / f"{frame_id}.color.jpg"))
            depth = cv2.imread(str(self.root / f"{frame_id}.depth.png"), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None:
                raise RuntimeError(f"Failed to read frame {frame_id}")

            camera_pose = self._load_camera_pose(self.root / f"{frame_id}.pose.txt")
            if camera_pose is None:
                print(f"Skip frame with invalid camera pose: {frame_id}")
                continue

            return {
                "FrameId": frame_id,
                "RGB": cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                "Depth": depth,
                "CameraPose": camera_pose,
            }
        return None


def _build_argparser():
    parser = argparse.ArgumentParser(description="MainOnRobot test runner")
    parser.add_argument("--stream-root", type=str, default=str(DEFAULT_STREAM_ROOT), help="Posed RGBD 数据目录")
    parser.add_argument("--roimap-root", type=str, default=str(DEFAULT_ROIMAP_ROOT), help="ROI map 输出目录")
    parser.add_argument("--server-host", type=str, default="219.223.200.92", help="远端检索服务 host")
    parser.add_argument("--server-port", type=int, default=5555, help="远端检索服务端口")
    parser.add_argument("--fx", type=float, default=525.0, help="相机内参 fx")
    parser.add_argument("--fy", type=float, default=525.0, help="相机内参 fy")
    parser.add_argument("--cx", type=float, default=319.5, help="相机内参 cx")
    parser.add_argument("--cy", type=float, default=239.5, help="相机内参 cy")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="深度缩放")
    parser.add_argument("--search-timeout-seconds", type=float, default=10.0, help="搜索服务超时秒数")
    parser.add_argument("--search-hold-seconds", type=float, default=5.0, help="搜索结果停留秒数")
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    stream_root = Path(args.stream_root).expanduser().resolve()
    roimap_root = Path(args.roimap_root).expanduser().resolve()

    robot = MainOnRobot(
        roimap_root=roimap_root,
        server_host=args.server_host,
        server_port=args.server_port,
        camera_intrinsics={
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
        },
        depth_scale=args.depth_scale,
        search_timeout_seconds=args.search_timeout_seconds,
        search_hold_seconds=args.search_hold_seconds,
    )
    stream = PosedRGBDStream(stream_root)
    robot.run_stream(stream)
