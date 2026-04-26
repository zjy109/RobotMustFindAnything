import time
from enum import Enum
from pathlib import Path

import numpy as np

from .roimap.roimap_fixed import ROIMapFixed
from .roimap.search_engine import SearchEngine


class MainOnRobotState(str, Enum):
    MAPPING = "mapping"
    SEARCHING = "searching"
    SHOWING_SEARCH_RESULT = "showing_search_result"
    STOPPED = "stopped"


class MainOnRobot:
    def __init__(
        self,
        roimap_root,
        server_host="127.0.0.1",
        server_port=5555,
        camera_intrinsics=None,
        depth_scale=0.001,
        search_timeout_seconds=10.0,
        search_hold_seconds=5.0,
    ):
        roimap_root = Path(roimap_root)

        self.roimap = ROIMapFixed(roimap_root)
        self.search_engine = SearchEngine(
            anchor_map_dir=roimap_root,
            server_host=server_host,
            server_port=server_port,
            recv_timeout_ms=int(float(search_timeout_seconds) * 1000),
            send_timeout_ms=int(float(search_timeout_seconds) * 1000),
            camera_intrinsics=camera_intrinsics,
            depth_scale=depth_scale,
        )

        self.state = MainOnRobotState.MAPPING
        self.search_timeout_seconds = float(search_timeout_seconds)
        self.search_hold_seconds = float(search_hold_seconds)

        self.last_pose2d = None
        self.last_posed_rgbd = None
        self.last_frame_pose_valid = False
        self.last_search_result = None
        self.last_query = None
        self.resume_mapping_at = None

    @staticmethod
    def _normalize_camera_pose(camera_pose):
        pose = np.asarray(camera_pose, dtype=np.float32)
        if pose.shape == (16,):
            pose = pose.reshape(4, 4)
        if pose.shape != (4, 4):
            return None
        if not np.all(np.isfinite(pose)):
            return None
        return pose

    def close(self):
        self.search_engine.close()

    def _now(self):
        return time.monotonic()

    def _snapshot(self, event=None, refreshed=None, matches=None):
        frame_id = None
        if self.last_posed_rgbd is not None:
            frame_id = self.last_posed_rgbd.get("FrameId")

        return {
            "state": self.state.value,
            "frame_id": frame_id,
            "pose2d": self.last_pose2d,
            "anchors": list(self.roimap.index["anchors"]),
            "refreshed": [] if refreshed is None else refreshed,
            "matches": [] if matches is None else matches,
            "event": event,
            "camera_pose_valid": self.last_frame_pose_valid,
            "search_result": self.last_search_result,
            "last_query": self.last_query,
            "resume_mapping_at": self.resume_mapping_at,
        }

    def snapshot(self):
        return self._snapshot()

    def stop(self):
        self.state = MainOnRobotState.STOPPED
        self.resume_mapping_at = None
        return self._snapshot(event={"event": "quit"})

    def _clear_search_result(self):
        self.last_search_result = None
        self.resume_mapping_at = None

    def _build_search_result(self, query, search_output=None, error_message=None):
        if error_message is not None:
            return {
                "status": "failed",
                "query": query,
                "anchor_id": None,
                "target_xy": None,
                "center": None,
                "extent": None,
                "error": error_message,
                "raw_output": None,
            }

        localization = search_output["localization"]
        aabb = localization["aabb"]
        center = tuple(float(v) for v in aabb["center"])
        extent = tuple(float(v) for v in aabb["extent"])
        return {
            "status": "success",
            "query": query,
            "anchor_id": localization["anchor_id"],
            "target_xy": (center[0], center[1]),
            "center": center,
            "extent": extent,
            "error": None,
            "raw_output": search_output,
        }

    def process_frame(self, posed_rgbd):
        if self.state == MainOnRobotState.STOPPED:
            return self._snapshot(event={"event": "stopped"})

        if self.state == MainOnRobotState.SHOWING_SEARCH_RESULT:
            return self._snapshot(event={"event": "search_result_pending"})

        if posed_rgbd is None:
            return self._snapshot(event={"event": "no_frame"})

        self.last_posed_rgbd = posed_rgbd
        camera_pose = self._normalize_camera_pose(posed_rgbd["CameraPose"])
        if camera_pose is None:
            self.last_frame_pose_valid = False
            return self._snapshot(
                event={"event": "invalid_pose_skipped", "frame_id": posed_rgbd["FrameId"]},
                refreshed=[],
                matches=[],
            )

        posed_rgbd["CameraPose"] = camera_pose
        self.last_pose2d = self.roimap.camera_pose_to_pose2d(camera_pose)
        self.last_frame_pose_valid = True
        refreshed, matches = self.roimap.refresh_matching_anchors(posed_rgbd, self.last_pose2d)
        return self._snapshot(refreshed=refreshed, matches=matches)

    def set_anchor_from_last_frame(self):
        if self.state != MainOnRobotState.MAPPING:
            return self._snapshot(event={"event": "manual_set_anchor_rejected", "reason": "invalid_state"})

        if self.last_posed_rgbd is None:
            return self._snapshot(event={"event": "manual_set_anchor_rejected", "reason": "no_frame"})

        if not self.last_frame_pose_valid or self.last_pose2d is None:
            return self._snapshot(event={"event": "manual_set_anchor_rejected", "reason": "invalid_pose"})

        anchor = self.roimap.set_anchor(self.last_posed_rgbd, self.last_pose2d)
        return self._snapshot(event={"event": "manual_set_anchor", "anchor": anchor})

    def search(self, query, current_time=None):
        if self.state == MainOnRobotState.STOPPED:
            return self._snapshot(event={"event": "stopped"})

        query = "" if query is None else str(query).strip()
        if not query:
            return self._snapshot(event={"event": "search_cancelled"})

        self.last_query = query
        self.state = MainOnRobotState.SEARCHING

        try:
            search_output = self.search_engine.search_by_language_instruction(query)
            self.last_search_result = self._build_search_result(query, search_output=search_output)
            now = self._now() if current_time is None else float(current_time)
            self.resume_mapping_at = now + self.search_hold_seconds
            self.state = MainOnRobotState.SHOWING_SEARCH_RESULT
            return self._snapshot(
                event={"event": "search_success", "query": query, "search_output": search_output}
            )
        except Exception as exc:
            self.last_search_result = self._build_search_result(query, error_message=str(exc))
            self.resume_mapping_at = None
            self.state = MainOnRobotState.MAPPING
            return self._snapshot(event={"event": "search_failed", "query": query, "error": str(exc)})

    def resume_mapping(self):
        if self.state != MainOnRobotState.SHOWING_SEARCH_RESULT:
            return self._snapshot(event={"event": "resume_ignored"})

        self.state = MainOnRobotState.MAPPING
        self._clear_search_result()
        return self._snapshot(event={"event": "resume_now"})

    def tick(self, current_time=None):
        if self.state == MainOnRobotState.STOPPED:
            return self._snapshot(event={"event": "stopped"})

        if self.state != MainOnRobotState.SHOWING_SEARCH_RESULT:
            return self._snapshot()

        now = self._now() if current_time is None else float(current_time)
        if self.resume_mapping_at is not None and now >= self.resume_mapping_at:
            self.state = MainOnRobotState.MAPPING
            self._clear_search_result()
            return self._snapshot(event={"event": "resume_timeout"})

        return self._snapshot()

    def update(self, posed_rgbd):
        if posed_rgbd is None:
            return self.tick()
        return self.process_frame(posed_rgbd)
