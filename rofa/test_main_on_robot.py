import argparse
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from main_on_robot import MainOnRobot
from roimap.roimap_fixed import ROIMapFixed
from roimap.search_engine import SearchEngine

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STREAM_ROOT = SCRIPT_DIR / "benchmark" / "office2"
DEFAULT_ROIMAP_ROOT = SCRIPT_DIR / "roimap_fixed_data"


def _make_camera_pose(x, y, yaw_deg):
    yaw = np.deg2rad(float(yaw_deg))
    pose = np.eye(4, dtype=np.float32)
    pose[0, 0] = np.cos(yaw)
    pose[0, 1] = -np.sin(yaw)
    pose[1, 0] = np.sin(yaw)
    pose[1, 1] = np.cos(yaw)
    pose[0, 3] = float(x)
    pose[1, 3] = float(y)
    return pose


def _make_posed_rgbd(frame_id, rgb_value, depth_value, x=0.0, y=0.0, yaw_deg=0.0):
    rgb = np.full((4, 5, 3), int(rgb_value), dtype=np.uint8)
    depth = np.full((4, 5), int(depth_value), dtype=np.uint16)
    return {
        "FrameId": frame_id,
        "RGB": rgb,
        "Depth": depth,
        "CameraPose": _make_camera_pose(x, y, yaw_deg),
    }


class _DummySocket:
    def close(self, linger=0):
        return None


class _TestSearchEngine(SearchEngine):
    def _create_socket(self):
        return _DummySocket()


class TestAnchorStorage(unittest.TestCase):
    def test_roimap_anchor_keeps_first_and_updates_last(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            roimap_root = Path(tmp_dir) / "roimap"
            roimap = ROIMapFixed(roimap_root)
            first_rgbd = _make_posed_rgbd("first", rgb_value=11, depth_value=101, x=1.0, y=2.0, yaw_deg=0.0)
            first_pose2d = roimap.camera_pose_to_pose2d(first_rgbd["CameraPose"])
            anchor = roimap.set_anchor(first_rgbd, first_pose2d)

            anchor_dir = roimap_root / anchor["id"]
            first_dir = anchor_dir / "first"
            last_dir = anchor_dir / "last"

            self.assertTrue(first_dir.exists())
            self.assertFalse(last_dir.exists())

            refreshed_rgbd = _make_posed_rgbd("latest", rgb_value=77, depth_value=909, x=1.02, y=2.01, yaw_deg=8.0)
            refreshed_pose2d = roimap.camera_pose_to_pose2d(refreshed_rgbd["CameraPose"])
            refreshed = roimap.refresh_anchor(anchor["id"], refreshed_rgbd, refreshed_pose2d)

            self.assertIsNotNone(refreshed)
            self.assertTrue(last_dir.exists())

            first_rgb = cv2.imread(str(first_dir / "rgb.png"), cv2.IMREAD_COLOR)
            last_rgb = cv2.imread(str(last_dir / "rgb.png"), cv2.IMREAD_COLOR)
            self.assertEqual(int(first_rgb[0, 0, 0]), 11)
            self.assertEqual(int(last_rgb[0, 0, 0]), 77)

            first_depth = cv2.imread(str(first_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
            last_depth = cv2.imread(str(last_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
            self.assertEqual(int(first_depth[0, 0]), 101)
            self.assertEqual(int(last_depth[0, 0]), 909)

            np.testing.assert_allclose(np.load(first_dir / "camera_pose.npy"), first_rgbd["CameraPose"])
            np.testing.assert_allclose(np.load(last_dir / "camera_pose.npy"), refreshed_rgbd["CameraPose"])

            stored_first_pose2d = json.loads((first_dir / "pose2d.json").read_text(encoding="utf-8"))
            stored_last_pose2d = json.loads((last_dir / "pose2d.json").read_text(encoding="utf-8"))
            self.assertEqual(stored_first_pose2d["yaw"], first_pose2d["yaw"])
            self.assertEqual(stored_last_pose2d["yaw"], refreshed_pose2d["yaw"])

    def test_search_engine_prefers_last_then_first_then_legacy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            roimap_root = Path(tmp_dir) / "roimap"
            roimap = ROIMapFixed(roimap_root)

            latest_anchor = roimap.set_anchor(
                _make_posed_rgbd("a0", rgb_value=10, depth_value=100, x=0.0, y=0.0, yaw_deg=0.0)
            )
            roimap.refresh_anchor(
                latest_anchor["id"],
                _make_posed_rgbd("a0_new", rgb_value=20, depth_value=200, x=0.02, y=0.01, yaw_deg=5.0),
            )

            first_only_anchor = roimap.set_anchor(
                _make_posed_rgbd("a1", rgb_value=30, depth_value=300, x=1.0, y=0.0, yaw_deg=0.0)
            )

            legacy_anchor_id = "anchor_legacy"
            legacy_anchor_dir = roimap_root / legacy_anchor_id
            legacy_anchor_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(legacy_anchor_dir / "rgb.png"), np.full((3, 4, 3), 40, dtype=np.uint8))
            cv2.imwrite(str(legacy_anchor_dir / "depth.png"), np.full((3, 4), 400, dtype=np.uint16))
            np.save(legacy_anchor_dir / "camera_pose.npy", _make_camera_pose(2.0, 0.0, 0.0))
            roimap.index["anchors"].append({"id": legacy_anchor_id, "x": 2.0, "y": 0.0, "yaw": 0.0})
            roimap._save_index()

            engine = _TestSearchEngine(anchor_map_dir=roimap_root)
            entries = {entry["anchor_id"]: entry for entry in engine._get_anchor_entries()}

            self.assertEqual(entries[latest_anchor["id"]]["snapshot_dir"].name, "last")
            self.assertEqual(entries[first_only_anchor["id"]]["snapshot_dir"].name, "first")
            self.assertEqual(entries[legacy_anchor_id]["snapshot_dir"], roimap_root / legacy_anchor_id)

            latest_rgb = cv2.imread(str(entries[latest_anchor["id"]]["rgb_path"]), cv2.IMREAD_COLOR)
            first_rgb = cv2.imread(str(entries[first_only_anchor["id"]]["rgb_path"]), cv2.IMREAD_COLOR)
            legacy_rgb = cv2.imread(str(entries[legacy_anchor_id]["rgb_path"]), cv2.IMREAD_COLOR)
            self.assertEqual(int(latest_rgb[0, 0, 0]), 20)
            self.assertEqual(int(first_rgb[0, 0, 0]), 30)
            self.assertEqual(int(legacy_rgb[0, 0, 0]), 40)


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
    # parser.add_argument("--server-host", type=str, default="219.223.200.92", help="远端检索服务 host")
    parser.add_argument("--server-host", type=str, default="192.168.1.103", help="远端检索服务 host")
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
