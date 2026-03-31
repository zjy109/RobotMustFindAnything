from pathlib import Path

import cv2
import numpy as np

from roimap.roimap_fixed import ROIMapFixed


class PosedRGBDStream:
    def __init__(self, root):
        self.root = Path(root)
        self.ids = sorted(p.stem.replace(".color", "") for p in self.root.glob("*.color.jpg"))
        self.i = 0

    def next_posed_rgbd(self):
        if not self.ids:
            raise RuntimeError(f"No frames found in {self.root}")
        if self.i >= len(self.ids):
            return None
        fid = self.ids[self.i]
        self.i += 1
        rgb = cv2.imread(str(self.root / f"{fid}.color.jpg"))
        depth = cv2.imread(str(self.root / f"{fid}.depth.png"), cv2.IMREAD_UNCHANGED)
        matrix = np.loadtxt(self.root / f"{fid}.pose.txt", dtype=np.float32)
        if rgb is None or depth is None:
            raise RuntimeError(f"Failed to read frame {fid}")
        return {
            "FrameId": fid,
            "RGB": cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
            "Depth": depth,
            "CameraPose": matrix,
        }


def draw_pose2d_map(current_pose, anchors, size=600, margin=60):
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    points = [(current_pose["x"], current_pose["y"])] + [(a["x"], a["y"]) for a in anchors]
    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    span = max(max_x - min_x, max_y - min_y, 1.0)
    scale = (size - 2 * margin) / span
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5

    def project(x, y):
        px = int((x - cx) * scale + size * 0.5)
        py = int(size * 0.5 - (y - cy) * scale)
        return px, py

    cv2.putText(canvas, "2D Map", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (margin, size // 2), (size - margin, size // 2), (220, 220, 220), 1)
    cv2.line(canvas, (size // 2, margin), (size // 2, size - margin), (220, 220, 220), 1)

    for anchor in anchors:
        px, py = project(anchor["x"], anchor["y"])
        dx = int(18 * np.cos(anchor["yaw"]))
        dy = int(18 * np.sin(anchor["yaw"]))
        cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1)
        cv2.arrowedLine(canvas, (px, py), (px + dx, py - dy), (0, 0, 180), 2, tipLength=0.3)
        cv2.putText(canvas, anchor["id"], (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1, cv2.LINE_AA)

    px, py = project(current_pose["x"], current_pose["y"])
    dx = int(24 * np.cos(current_pose["yaw"]))
    dy = int(24 * np.sin(current_pose["yaw"]))
    cv2.circle(canvas, (px, py), 8, (0, 180, 0), -1)
    cv2.arrowedLine(canvas, (px, py), (px + dx, py - dy), (0, 120, 0), 2, tipLength=0.3)
    cv2.putText(canvas, "current", (px + 10, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1, cv2.LINE_AA)
    return canvas


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    stream = PosedRGBDStream(root / "benchmark" / "office2")
    roimap = ROIMapFixed(root / "roimap_fixed_data")
    autoplay = False

    print("Controls: Enter=set anchor, c=continue autoplay, s=single-step, q=quit")

    while True:
        posed_rgbd = stream.next_posed_rgbd()
        if posed_rgbd is None:
            break

        pose2d = roimap.camera_pose_to_pose2d(posed_rgbd["CameraPose"])
        refreshed, matches = roimap.refresh_matching_anchors(posed_rgbd, pose2d)
        for anchor, match in zip(refreshed, matches):
            print(
                f"Refreshed {anchor['id']} | xy={match['xy']:.3f} m | yaw={match['yaw_deg']:.2f} deg"
            )

        rgb = cv2.cvtColor(posed_rgbd["RGB"], cv2.COLOR_RGB2BGR)
        depth = posed_rgbd["Depth"].astype(np.float32)
        vmax = np.percentile(depth[depth > 0], 99) if np.any(depth > 0) else 1.0
        depth_vis = cv2.applyColorMap(np.clip(depth / max(vmax, 1.0) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)

        status = f"{posed_rgbd['FrameId']} x={pose2d['x']:.2f} y={pose2d['y']:.2f} yaw={np.degrees(pose2d['yaw']):.1f}"
        anchor_text = f"anchors={len(roimap.index['anchors'])} refreshed={len(refreshed)} autoplay={autoplay}"
        cv2.putText(rgb, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(rgb, anchor_text, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("ROIMapFixed", np.hstack([rgb, depth_vis]))
        cv2.imshow("ROIMapFixed 2D Map", draw_pose2d_map(pose2d, roimap.index["anchors"]))

        key = cv2.waitKey(30 if autoplay else 0) & 0xFF
        if key == ord("q"):
            break
        if key in (13, 10):
            anchor = roimap.set_anchor(posed_rgbd, pose2d)
            print(f"Set anchor: {anchor['id']} at x={anchor['x']:.3f}, y={anchor['y']:.3f}, yaw={np.degrees(anchor['yaw']):.2f} deg")
        if key == ord("c"):
            autoplay = True
        if key == ord("s"):
            autoplay = False

    cv2.destroyAllWindows()
