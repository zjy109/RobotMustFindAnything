import argparse
from pathlib import Path

import cv2
import numpy as np

from main_on_robot import MainOnRobot, MainOnRobotState

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


def _depth_to_vis(depth):
    depth = np.asarray(depth)
    depth_float = depth.astype(np.float32)
    positive = depth_float[depth_float > 0]
    vmax = np.percentile(positive, 99) if positive.size else 1.0
    scaled = np.clip(depth_float / max(vmax, 1.0) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def _project_points_to_canvas(points, size=720, margin=70):
    if not points:
        points = [(0.0, 0.0)]

    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    span = max(max_x - min_x, max_y - min_y, 1.0)
    scale = (size - 2 * margin) / span
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5

    def project(x, y):
        px = int((x - center_x) * scale + size * 0.5)
        py = int(size * 0.5 - (y - center_y) * scale)
        return px, py

    return project


def _build_search_lines(snapshot):
    search_result = snapshot["search_result"]
    if search_result is None:
        return []
    if search_result["status"] == "failed":
        return [f"search failed: {search_result['error']}"]

    center = search_result["center"]
    extent = search_result["extent"]
    return [
        f"query: {search_result['query']}",
        f"anchor: {search_result['anchor_id']}",
        f"center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})",
        f"extent=({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})",
    ]


def _draw_pose2d_map(snapshot, size=720, margin=70):
    canvas = np.full((size, size, 3), 250, dtype=np.uint8)
    anchors = snapshot["anchors"]
    current_pose = snapshot["pose2d"]
    search_result = snapshot["search_result"]

    points = []
    if current_pose is not None:
        points.append((current_pose["x"], current_pose["y"]))
    points.extend((anchor["x"], anchor["y"]) for anchor in anchors)
    if search_result and search_result.get("target_xy") is not None:
        points.append(search_result["target_xy"])

    project = _project_points_to_canvas(points, size=size, margin=margin)
    cv2.putText(canvas, "MainOnRobot 2D Map", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.line(canvas, (margin, size // 2), (size - margin, size // 2), (220, 220, 220), 1)
    cv2.line(canvas, (size // 2, margin), (size // 2, size - margin), (220, 220, 220), 1)

    highlighted_anchor_id = None if search_result is None else search_result.get("anchor_id")
    for anchor in anchors:
        px, py = project(anchor["x"], anchor["y"])
        dx = int(20 * np.cos(anchor["yaw"]))
        dy = int(20 * np.sin(anchor["yaw"]))
        color = (0, 90, 255) if anchor["id"] == highlighted_anchor_id else (0, 0, 220)
        cv2.circle(canvas, (px, py), 6, color, -1)
        cv2.arrowedLine(canvas, (px, py), (px + dx, py - dy), color, 2, tipLength=0.3)
        cv2.putText(canvas, anchor["id"], (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    if current_pose is not None:
        px, py = project(current_pose["x"], current_pose["y"])
        dx = int(26 * np.cos(current_pose["yaw"]))
        dy = int(26 * np.sin(current_pose["yaw"]))
        cv2.circle(canvas, (px, py), 8, (0, 160, 0), -1)
        cv2.arrowedLine(canvas, (px, py), (px + dx, py - dy), (0, 120, 0), 2, tipLength=0.3)
        cv2.putText(canvas, "robot", (px + 10, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1, cv2.LINE_AA)

    if search_result is not None and search_result.get("target_xy") is not None:
        px, py = project(*search_result["target_xy"])
        cv2.circle(canvas, (px, py), 10, (255, 140, 0), 2)
        cv2.line(canvas, (px - 12, py), (px + 12, py), (255, 140, 0), 2)
        cv2.line(canvas, (px, py - 12), (px, py + 12), (255, 140, 0), 2)
        cv2.putText(
            canvas,
            f"target: {search_result['query']}",
            (px + 12, py - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 140, 0),
            2,
            cv2.LINE_AA,
        )

    for i, line in enumerate(_build_search_lines(snapshot)):
        cv2.putText(
            canvas,
            line,
            (20, size - 60 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    return canvas


def _build_mapping_lines(snapshot, autoplay):
    pose = snapshot["pose2d"]
    frame_id = snapshot["frame_id"] or "-"
    if not snapshot["camera_pose_valid"]:
        return [
            f"{frame_id} state={snapshot['state']} autoplay={autoplay}",
            "invalid camera pose, frame skipped",
            f"anchors={len(snapshot['anchors'])}",
            "keys: q quit | c autoplay | s step | Enter set anchor | f search",
        ]

    lines = [
        f"{frame_id} state={snapshot['state']} autoplay={autoplay}",
        f"x={pose['x']:.2f} y={pose['y']:.2f} yaw={np.degrees(pose['yaw']):.1f}",
        f"anchors={len(snapshot['anchors'])} refreshed={len(snapshot['refreshed'])}",
        "keys: q quit | c autoplay | s step | Enter set anchor | f search",
    ]
    event = snapshot["event"]
    if event is not None and event["event"] == "manual_set_anchor":
        lines.append(f"set anchor {event['anchor']['id']}")
    elif snapshot["refreshed"]:
        lines.append(f"refreshed {len(snapshot['refreshed'])} anchor(s)")
    elif event is not None and event["event"] == "search_failed":
        lines.append(f"search failed: {event['error']}")
    return lines


def _build_search_result_lines(snapshot, robot):
    remaining = 0.0
    if snapshot["resume_mapping_at"] is not None:
        remaining = max(0.0, snapshot["resume_mapping_at"] - robot._now())
    query = "-" if snapshot["last_query"] is None else snapshot["last_query"]
    return [
        f"state={snapshot['state']} query={query}",
        f"resume in {remaining:.1f}s",
        "keys: q quit | r resume now | f search again",
    ]


def _show_views(snapshot, rgb_bgr, depth_vis, autoplay, robot):
    if rgb_bgr is None or depth_vis is None:
        return -1

    if snapshot["state"] == MainOnRobotState.SHOWING_SEARCH_RESULT.value:
        status_lines = _build_search_result_lines(snapshot, robot)
    else:
        status_lines = _build_mapping_lines(snapshot, autoplay)

    rgb_panel = rgb_bgr.copy()
    for i, line in enumerate(status_lines):
        cv2.putText(rgb_panel, line, (10, 28 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("MainOnRobot RGBD", np.hstack([rgb_panel, depth_vis]))
    cv2.imshow("MainOnRobot 2D Map", _draw_pose2d_map(snapshot))
    return cv2.waitKey(30 if autoplay else 0) & 0xFF


def _prompt_search_query():
    print("\nSearch mode: 在终端输入要检索的内容，直接回车提交，空输入取消。")
    try:
        query = input("search> ")
    except EOFError:
        query = ""
    return "" if query is None else str(query).strip()


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

    autoplay = True
    last_rgb_bgr = None
    last_depth_vis = None
    snapshot = robot.update(None)

    try:
        while robot.state != MainOnRobotState.STOPPED:
            if robot.state == MainOnRobotState.SHOWING_SEARCH_RESULT:
                snapshot = robot.tick()
            else:
                posed_rgbd = stream.next_posed_rgbd()
                if posed_rgbd is None:
                    break
                last_rgb_bgr = cv2.cvtColor(posed_rgbd["RGB"], cv2.COLOR_RGB2BGR)
                last_depth_vis = _depth_to_vis(posed_rgbd["Depth"])
                snapshot = robot.process_frame(posed_rgbd)

            key = _show_views(snapshot, last_rgb_bgr, last_depth_vis, autoplay, robot)
            if key == ord("q"):
                snapshot = robot.stop()
            elif key == ord("c"):
                autoplay = True
            elif key == ord("s"):
                autoplay = False
            elif key in (13, 10):
                snapshot = robot.set_anchor_from_last_frame()
            elif key == ord("f"):
                snapshot = robot.search(_prompt_search_query())
            elif key == ord("r"):
                snapshot = robot.resume_mapping()
    finally:
        robot.close()
        cv2.destroyAllWindows()
