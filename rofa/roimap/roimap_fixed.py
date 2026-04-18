import json
from pathlib import Path

import cv2
import numpy as np


class ROIMapFixed:
    FIRST_DIR_NAME = "first"
    LAST_DIR_NAME = "last"

    def __init__(self, root, xy_thresh=0.1, yaw_thresh_deg=10.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "anchors.json"
        self.xy_thresh = float(xy_thresh)
        self.yaw_thresh_deg = float(yaw_thresh_deg)
        self.index = self._load_index()

    def _load_index(self):
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {"current_anchor_id": None, "anchors": []}

    def _save_index(self):
        self.index_path.write_text(json.dumps(self.index, indent=2), encoding="utf-8")

    @staticmethod
    def camera_pose_to_pose2d(camera_pose):
        camera_pose = np.asarray(camera_pose, dtype=np.float32)
        return {
            "x": float(camera_pose[0, 3]),
            "y": float(camera_pose[1, 3]),
            "yaw": float(np.arctan2(camera_pose[1, 0], camera_pose[0, 0])),
        }

    @staticmethod
    def _yaw_diff_deg(a, b):
        diff = np.arctan2(np.sin(a - b), np.cos(a - b))
        return float(abs(np.degrees(diff)))

    def _anchor_dir(self, anchor_id):
        return self.root / anchor_id

    def _anchor_snapshot_dir(self, anchor_id, snapshot_name):
        return self._anchor_dir(anchor_id) / snapshot_name

    def _write_anchor_rgbd(self, anchor_id, posed_rgbd, snapshot_name):
        anchor_dir = self._anchor_snapshot_dir(anchor_id, snapshot_name)
        anchor_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(anchor_dir / "rgb.png"), cv2.cvtColor(posed_rgbd["RGB"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(anchor_dir / "depth.png"), posed_rgbd["Depth"])

    def _write_anchor_pose(self, anchor_id, camera_pose, pose2d, snapshot_name):
        anchor_dir = self._anchor_snapshot_dir(anchor_id, snapshot_name)
        anchor_dir.mkdir(parents=True, exist_ok=True)
        np.save(anchor_dir / "camera_pose.npy", camera_pose)
        (anchor_dir / "pose2d.json").write_text(json.dumps(pose2d, indent=2), encoding="utf-8")

    def _write_anchor_snapshot(self, anchor_id, posed_rgbd, pose2d, snapshot_name):
        self._write_anchor_rgbd(anchor_id, posed_rgbd, snapshot_name)
        self._write_anchor_pose(anchor_id, posed_rgbd["CameraPose"], pose2d, snapshot_name)

    def set_anchor(self, posed_rgbd, pose2d=None):
        pose2d = pose2d or self.camera_pose_to_pose2d(posed_rgbd["CameraPose"])
        anchor_id = f"anchor_{len(self.index['anchors']):04d}"
        anchor = {"id": anchor_id, **pose2d}
        self._write_anchor_snapshot(anchor_id, posed_rgbd, pose2d, self.FIRST_DIR_NAME)
        self.index["anchors"].append(anchor)
        self.index["current_anchor_id"] = anchor_id
        self._save_index()
        return anchor

    def get_current_anchor(self):
        anchor_id = self.index["current_anchor_id"]
        if anchor_id is None:
            return None
        for anchor in self.index["anchors"]:
            if anchor["id"] == anchor_id:
                return anchor
        return None

    def find_close_anchors(self, camera_pose, pose2d=None):
        pose2d = pose2d or self.camera_pose_to_pose2d(camera_pose)
        matches = []
        for anchor in self.index["anchors"]:
            xy = float(np.hypot(pose2d["x"] - anchor["x"], pose2d["y"] - anchor["y"]))
            yaw = self._yaw_diff_deg(pose2d["yaw"], anchor["yaw"])
            if xy <= self.xy_thresh and yaw <= self.yaw_thresh_deg:
                matches.append({"anchor": anchor, "xy": xy, "yaw_deg": yaw})
        return matches

    def refresh_anchor(self, anchor_id, posed_rgbd, pose2d=None):
        pose2d = pose2d or self.camera_pose_to_pose2d(posed_rgbd["CameraPose"])
        for anchor in self.index["anchors"]:
            if anchor["id"] != anchor_id:
                continue
            self._write_anchor_snapshot(anchor_id, posed_rgbd, pose2d, self.LAST_DIR_NAME)
            return anchor
        return None

    def refresh_matching_anchors(self, posed_rgbd, pose2d=None):
        pose2d = pose2d or self.camera_pose_to_pose2d(posed_rgbd["CameraPose"])
        matches = self.find_close_anchors(posed_rgbd["CameraPose"], pose2d)
        refreshed = []
        for match in matches:
            refreshed.append(self.refresh_anchor(match["anchor"]["id"], posed_rgbd, pose2d))
        return [anchor for anchor in refreshed if anchor is not None], matches
