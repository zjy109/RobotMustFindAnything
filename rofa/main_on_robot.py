import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from roimap.roimap_fixed import ROIMapFixed
from roimap.search_engine import SearchEngine


class MainOnRobotState(str, Enum):
    MAPPING = "mapping"
    QUERY_INPUT = "query_input"
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
        sam2_model_id="facebook/sam2.1-hiera-small",
        sam2_cache_dir=None,
        sam2_device=None,
        preload_search_models=True,
        search_timeout_seconds=10.0,
        continue_show=True,
        search_hold_seconds=5.0,
        enable_visualization=True,
        query_prompt_fn=None,
    ):
        roimap_root = Path(roimap_root)
        sam2_cache_dir = Path(sam2_cache_dir) if sam2_cache_dir else SearchEngine.default_sam2_cache_dir(sam2_model_id)

        self.roimap = ROIMapFixed(roimap_root)
        self.search_engine = SearchEngine(
            anchor_map_dir=roimap_root,
            server_host=server_host,
            server_port=server_port,
            recv_timeout_ms=int(float(search_timeout_seconds) * 1000),
            send_timeout_ms=int(float(search_timeout_seconds) * 1000),
            camera_intrinsics=camera_intrinsics,
            depth_scale=depth_scale,
            sam2_model_id=sam2_model_id,
            sam2_cache_dir=sam2_cache_dir,
            sam2_device=sam2_device,
        )

        self.state = MainOnRobotState.MAPPING
        self.continue_show = bool(continue_show)
        self.preload_search_models = bool(preload_search_models)
        self.search_timeout_seconds = float(search_timeout_seconds)
        self.search_hold_seconds = float(search_hold_seconds)
        self.enable_visualization = bool(enable_visualization)
        self.query_prompt_fn = query_prompt_fn or input
        self.sam2_cache_dir = sam2_cache_dir

        self.rgb_window_name = "MainOnRobot RGBD"
        self.map_window_name = "MainOnRobot 2D Map"

        self.last_pose2d = None
        self.last_rgb_bgr = None
        self.last_depth_vis = None
        self.last_status_lines = []
        self.last_search_visualization = None
        self.last_query = None
        self.resume_mapping_at = None
        self.quit_requested = False

        if self.preload_search_models:
            self._preload_search_models()

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
        if self.enable_visualization:
            cv2.destroyAllWindows()

    def _preload_search_models(self):
        print(f"Preloading SAM2 from local cache: {self.sam2_cache_dir}")
        try:
            self.search_engine.warmup_models()
        except Exception as exc:
            print(f"SAM2 preload failed: {exc}")

    def _depth_to_vis(self, depth):
        depth = np.asarray(depth)
        depth_float = depth.astype(np.float32)
        positive = depth_float[depth_float > 0]
        vmax = np.percentile(positive, 99) if positive.size else 1.0
        scaled = np.clip(depth_float / max(vmax, 1.0) * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)

    def _project_points_to_canvas(self, points, size=720, margin=70):
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

    def _draw_pose2d_map(self, current_pose, anchors, search_visualization=None, size=720, margin=70):
        canvas = np.full((size, size, 3), 250, dtype=np.uint8)
        points = []
        if current_pose is not None:
            points.append((current_pose["x"], current_pose["y"]))
        points.extend((anchor["x"], anchor["y"]) for anchor in anchors)
        if search_visualization and search_visualization.get("target_xy") is not None:
            tx, ty = search_visualization["target_xy"]
            points.append((tx, ty))

        project = self._project_points_to_canvas(points, size=size, margin=margin)

        cv2.putText(canvas, "MainOnRobot 2D Map", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.line(canvas, (margin, size // 2), (size - margin, size // 2), (220, 220, 220), 1)
        cv2.line(canvas, (size // 2, margin), (size // 2, size - margin), (220, 220, 220), 1)

        highlighted_anchor_id = None
        if search_visualization is not None:
            highlighted_anchor_id = search_visualization.get("anchor_id")

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

        if search_visualization is not None:
            target_xy = search_visualization.get("target_xy")
            if target_xy is not None:
                px, py = project(target_xy[0], target_xy[1])
                cv2.circle(canvas, (px, py), 10, (255, 140, 0), 2)
                cv2.line(canvas, (px - 12, py), (px + 12, py), (255, 140, 0), 2)
                cv2.line(canvas, (px, py - 12), (px, py + 12), (255, 140, 0), 2)
                label = f"target: {search_visualization.get('query', 'unknown')}"
                cv2.putText(canvas, label, (px + 12, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2, cv2.LINE_AA)

            search_lines = search_visualization.get("text_lines", [])
            for i, line in enumerate(search_lines):
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

    def _build_rgb_panel(self, rgb_bgr):
        panel = rgb_bgr.copy()
        for i, line in enumerate(self.last_status_lines):
            cv2.putText(panel, line, (10, 28 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        return panel

    def _show_views(self, rgb_bgr, depth_vis, map_image, delay_ms):
        if not self.enable_visualization:
            return -1
        cv2.imshow(self.rgb_window_name, np.hstack([self._build_rgb_panel(rgb_bgr), depth_vis]))
        cv2.imshow(self.map_window_name, map_image)
        return cv2.waitKey(delay_ms) & 0xFF

    def _handle_common_key(self, key, posed_rgbd=None):
        if key == ord("q"):
            self.quit_requested = True
            self.state = MainOnRobotState.STOPPED
            return {"event": "quit"}
        if key == ord("c"):
            self.continue_show = True
            return {"event": "continue_show"}
        if key == ord("s"):
            self.continue_show = False
            return {"event": "single_step"}
        if key in (13, 10) and posed_rgbd is not None:
            anchor = self.roimap.set_anchor(posed_rgbd, self.last_pose2d)
            return {"event": "manual_set_anchor", "anchor": anchor}
        if key == ord("f"):
            return self._run_search_flow()
        if key == ord("r") and self.state == MainOnRobotState.SHOWING_SEARCH_RESULT:
            self.resume_mapping_at = time.monotonic()
            return {"event": "resume_now"}
        return None

    def _prompt_search_query(self):
        self.state = MainOnRobotState.QUERY_INPUT
        print("\nSearch mode: 在终端输入要检索的内容，直接回车提交，空输入取消。")
        try:
            query = self.query_prompt_fn("search> ")
        except EOFError:
            query = ""
        query = "" if query is None else str(query).strip()
        if not query:
            self.state = MainOnRobotState.MAPPING
            return None
        self.last_query = query
        return query

    def _build_search_visualization(self, query, search_output=None, error_message=None):
        if error_message is not None:
            return {
                "query": query,
                "anchor_id": None,
                "target_xy": None,
                "text_lines": [f"search failed: {error_message}"],
            }

        localization = search_output["localization"]
        aabb = localization["aabb"]
        center = aabb["center"]
        target_xy = (float(center[0]), float(center[1]))
        extent = aabb["extent"]
        return {
            "query": query,
            "anchor_id": localization["anchor_id"],
            "target_xy": target_xy,
            "text_lines": [
                f"query: {query}",
                f"anchor: {localization['anchor_id']}",
                f"center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})",
                f"extent=({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})",
            ],
        }

    def _run_search_flow(self):
        query = self._prompt_search_query()
        if query is None:
            return {"event": "search_cancelled"}

        self.state = MainOnRobotState.SEARCHING
        try:
            search_output = self.search_engine.search_by_language_instruction(query)
            self.last_search_visualization = self._build_search_visualization(query, search_output=search_output)
            self.resume_mapping_at = time.monotonic() + self.search_hold_seconds
            self.state = MainOnRobotState.SHOWING_SEARCH_RESULT
            result = {"event": "search_success", "query": query, "search_output": search_output}
        except Exception as exc:
            self.last_search_visualization = None
            self.resume_mapping_at = None
            self.state = MainOnRobotState.MAPPING
            print(f"Search failed for '{query}': {exc}. Resume mapping.")
            result = {"event": "search_failed", "query": query, "error": str(exc)}

        return result

    def _status_lines_for_mapping(self, posed_rgbd, refreshed_count, anchor_event=None):
        pose = self.last_pose2d
        status = [
            f"{posed_rgbd['FrameId']} state={self.state.value} autoplay={self.continue_show}",
            f"x={pose['x']:.2f} y={pose['y']:.2f} yaw={np.degrees(pose['yaw']):.1f}",
            f"anchors={len(self.roimap.index['anchors'])} refreshed={refreshed_count}",
            "keys: q quit | c autoplay | s step | Enter set anchor | f search",
        ]
        if anchor_event is not None:
            status.append(anchor_event)
        return status

    def _status_lines_for_search_result(self):
        remaining = 0.0
        if self.resume_mapping_at is not None:
            remaining = max(0.0, self.resume_mapping_at - time.monotonic())
        query = self.last_query or "-"
        return [
            f"state={self.state.value} query={query}",
            f"resume in {remaining:.1f}s",
            "keys: q quit | r resume now | f search again",
        ]

    def _update_mapping(self, posed_rgbd):
        camera_pose = self._normalize_camera_pose(posed_rgbd["CameraPose"])
        if camera_pose is None:
            self.last_rgb_bgr = cv2.cvtColor(posed_rgbd["RGB"], cv2.COLOR_RGB2BGR)
            self.last_depth_vis = self._depth_to_vis(posed_rgbd["Depth"])
            self.last_status_lines = [
                f"{posed_rgbd['FrameId']} state={self.state.value} autoplay={self.continue_show}",
                "invalid camera pose, frame skipped",
                f"anchors={len(self.roimap.index['anchors'])}",
                "keys: q quit | c autoplay | s step | Enter set anchor | f search",
            ]
            map_image = self._draw_pose2d_map(self.last_pose2d, self.roimap.index["anchors"])
            delay_ms = 30 if self.continue_show else 0
            key = self._show_views(self.last_rgb_bgr, self.last_depth_vis, map_image, delay_ms)
            event = self._handle_common_key(key)
            return {
                "state": self.state.value,
                "pose2d": self.last_pose2d,
                "refreshed": [],
                "matches": [],
                "event": event or {"event": "invalid_pose_skipped", "frame_id": posed_rgbd["FrameId"]},
            }

        posed_rgbd["CameraPose"] = camera_pose
        self.last_pose2d = self.roimap.camera_pose_to_pose2d(camera_pose)
        refreshed, matches = self.roimap.refresh_matching_anchors(posed_rgbd, self.last_pose2d)
        anchor_event = None

        if refreshed:
            anchor_event = f"refreshed {len(refreshed)} anchor(s)"

        self.last_rgb_bgr = cv2.cvtColor(posed_rgbd["RGB"], cv2.COLOR_RGB2BGR)
        self.last_depth_vis = self._depth_to_vis(posed_rgbd["Depth"])
        self.last_status_lines = self._status_lines_for_mapping(posed_rgbd, len(refreshed), anchor_event=anchor_event)
        map_image = self._draw_pose2d_map(self.last_pose2d, self.roimap.index["anchors"])

        delay_ms = 30 if self.continue_show else 0
        key = self._show_views(self.last_rgb_bgr, self.last_depth_vis, map_image, delay_ms)
        event = self._handle_common_key(key, posed_rgbd=posed_rgbd)

        return {
            "state": self.state.value,
            "pose2d": self.last_pose2d,
            "refreshed": refreshed,
            "matches": matches,
            "event": event,
        }

    def _update_show_search_result(self):
        if self.resume_mapping_at is not None and time.monotonic() >= self.resume_mapping_at:
            self.state = MainOnRobotState.MAPPING
            self.resume_mapping_at = None
            self.last_search_visualization = None
            return {"state": self.state.value, "event": {"event": "resume_timeout"}}

        if self.last_rgb_bgr is None or self.last_depth_vis is None:
            return {"state": self.state.value, "event": None}

        self.last_status_lines = self._status_lines_for_search_result()
        map_image = self._draw_pose2d_map(
            self.last_pose2d,
            self.roimap.index["anchors"],
            search_visualization=self.last_search_visualization,
        )
        key = self._show_views(self.last_rgb_bgr, self.last_depth_vis, map_image, 30)
        event = self._handle_common_key(key)
        return {"state": self.state.value, "event": event}

    def update(self, posed_rgbd):
        if self.state == MainOnRobotState.STOPPED:
            return {"state": self.state.value, "event": {"event": "stopped"}}

        if self.state == MainOnRobotState.SHOWING_SEARCH_RESULT:
            return self._update_show_search_result()

        if posed_rgbd is None:
            return {"state": self.state.value, "event": None}

        return self._update_mapping(posed_rgbd)

    def run_stream(self, stream):
        while not self.quit_requested:
            if self.state == MainOnRobotState.SHOWING_SEARCH_RESULT:
                self.update(None)
                continue

            posed_rgbd = stream.next_posed_rgbd()
            if posed_rgbd is None:
                break
            self.update(posed_rgbd)

        self.close()
