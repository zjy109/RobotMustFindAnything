import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import zmq


class SearchEngine:
    """
    读取 ROIMapFixed 保存的 anchor 地图，将全部 anchor RGB 图像和语言指令发给远端服务。
    远端直接返回目标所在 anchor 与 mask；本地仅负责：
    1. 校验 server 返回数据
    2. 保存 mask 可视化
    3. 结合深度图、相机内参和 camera pose 反投影到世界坐标
    4. 计算目标点云的 AABB

    请求协议（client -> server）：
    - multipart[0]：UTF-8 JSON metadata
    - multipart[1:]：与 anchors 顺序一致的 JPEG 图片 bytes

    metadata JSON 格式：
    {
      "type": "search_request",
      "instruction": "<str>",
      "anchors": [
        {
          "anchor_id": "<str>",
          "image_name": "rgb.jpg",
          "image_format": "jpeg",
          "height": <int>,
          "width": <int>
        }
      ],
      "response_format": {
        "type": "search_result",
        "mask_encoding": "png_base64",
        "mask_semantics": "nonzero_is_foreground"
      }
    }

    响应协议（server -> client）：
    - UTF-8 JSON 字符串
    - 顶层必须是 JSON object / Python dict

    推荐且正式支持的 server 返回格式：
    {
      "success": true,
      "found": true,
      "anchor_id": "anchor_0003",
      "image_index": 3,
      "bbox": [x1, y1, x2, y2],
      "score": 0.93,
      "mask": {
        "encoding": "png_base64",
        "height": 480,
        "width": 640,
        "data": "<base64 encoded PNG bytes>"
      }
    }

    字段要求：
    - success：bool。若为 false，必须额外返回 error_message: str
    - found：bool。表示检索是否命中目标；success=true 只代表服务执行成功
    - anchor_id：str。与 image_index 至少提供一个
    - image_index：int。0-based，下标对应本次请求中的 anchors 顺序
    - bbox：长度为 4 的数组，格式为 [x1, y1, x2, y2]，像素坐标，xyxy；可选
    - mask.encoding：当前约定为 "png_base64"
    - mask.height / mask.width：int，必须与对应 anchor 的 RGB/Depth 分辨率一致
    - mask.data：str，base64 编码后的单通道 PNG bytes；像素值 0 表示背景，非 0 表示前景
    - 当 found=false 时，不应再返回 anchor_id / image_index / bbox / mask；可选返回 message 或 error_message 说明原因

    兼容格式（只作为调试保底，不建议 server 正式使用）：
    - mask 直接是二维数组 / 二维 list，元素可以是 bool、0/1、0~255
    - mask = {"encoding": "array", "data": [[...], [...]]}
    """

    MASK_FILE_NAME = "remote_mask.png"
    MASK_OVERLAY_FILE_NAME = "remote_mask_overlay.png"
    AABB_OVERLAY_FILE_NAME = "remote_aabb_overlay.png"
    AABB_3D_FILE_NAME = "remote_aabb_3d.png"
    SNAPSHOT_PRIORITY = ("last", "first", None)

    def __init__(
        self,
        anchor_map_dir,
        server_host="127.0.0.1",
        server_port=5555,
        jpeg_quality=90,
        recv_timeout_ms=60000,
        send_timeout_ms=60000,
        camera_intrinsics=None,
        depth_scale=0.001,
    ):
        self.anchor_map_dir = Path(anchor_map_dir)
        self.index_path = self.anchor_map_dir / "anchors.json"
        self.server_host = str(server_host)
        self.server_port = int(server_port)
        self.jpeg_quality = int(jpeg_quality)
        self.recv_timeout_ms = int(recv_timeout_ms)
        self.send_timeout_ms = int(send_timeout_ms)
        self.depth_scale = float(depth_scale)
        self.camera_intrinsics = self._normalize_camera_intrinsics(camera_intrinsics)

        self.anchor_map_dir.mkdir(parents=True, exist_ok=True)

        self.index = self._load_index()
        self.context = zmq.Context.instance()
        self.socket = self._create_socket()

    @staticmethod
    def _normalize_camera_intrinsics(camera_intrinsics):
        if camera_intrinsics is None:
            return None
        required = ("fx", "fy", "cx", "cy")
        missing = [key for key in required if key not in camera_intrinsics]
        if missing:
            raise ValueError(f"camera_intrinsics 缺少字段: {missing}")
        return {key: float(camera_intrinsics[key]) for key in required}

    def _load_index(self):
        if not self.index_path.exists():
            return {"current_anchor_id": None, "anchors": []}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _create_socket(self):
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.send_timeout_ms)
        socket.connect(f"tcp://{self.server_host}:{self.server_port}")
        return socket

    def reset_socket(self):
        self.close()
        self.socket = self._create_socket()

    def close(self):
        if getattr(self, "socket", None) is not None:
            self.socket.close(linger=0)
            self.socket = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_anchor_entries(self):
        self.index = self._load_index()
        entries = []
        for anchor_order, anchor in enumerate(self.index.get("anchors", [])):
            anchor_id = anchor["id"]
            anchor_dir = self.anchor_map_dir / anchor_id
            (
                snapshot_dir,
                rgb_path,
                depth_path,
                camera_pose_path,
                snapshot_name,
                snapshot_timestamp,
            ) = self._resolve_anchor_files(anchor_dir, anchor)

            entries.append(
                {
                    "anchor_id": anchor_id,
                    "anchor_dir": anchor_dir,
                    "snapshot_dir": snapshot_dir,
                    "snapshot_name": snapshot_name,
                    "snapshot_timestamp": snapshot_timestamp,
                    "anchor_order": anchor_order,
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "camera_pose_path": camera_pose_path,
                }
            )
        entries.sort(key=self._anchor_entry_sort_key)
        return entries

    def _resolve_anchor_files(self, anchor_dir, anchor=None):
        missing_by_candidate = []
        for snapshot_name in self.SNAPSHOT_PRIORITY:
            snapshot_dir = anchor_dir if snapshot_name is None else anchor_dir / snapshot_name
            rgb_path = snapshot_dir / "rgb.png"
            depth_path = snapshot_dir / "depth.png"
            camera_pose_path = snapshot_dir / "camera_pose.npy"
            required_paths = (
                ("RGB 图片", rgb_path),
                ("深度图", depth_path),
                ("camera pose", camera_pose_path),
            )
            missing = [f"{label}: {path}" for label, path in required_paths if not path.exists()]
            if not missing:
                return (
                    snapshot_dir,
                    rgb_path,
                    depth_path,
                    camera_pose_path,
                    snapshot_name,
                    self._resolve_snapshot_timestamp(
                        anchor,
                        snapshot_name,
                        rgb_path,
                        depth_path,
                        camera_pose_path,
                    ),
                )
            if snapshot_dir.exists():
                missing_by_candidate.append((snapshot_dir, missing))

        if missing_by_candidate:
            snapshot_dir, missing = missing_by_candidate[0]
            raise FileNotFoundError(f"anchor 快照不完整: {snapshot_dir} | 缺少 {', '.join(missing)}")

        raise FileNotFoundError(
            f"anchor 不存在可用快照: {anchor_dir}，期望优先检查 {anchor_dir / 'last'}、{anchor_dir / 'first'} 或旧格式根目录"
        )

    def _anchor_entry_sort_key(self, entry):
        priority = self.SNAPSHOT_PRIORITY.index(entry["snapshot_name"])
        return (priority, -float(entry["snapshot_timestamp"]), int(entry["anchor_order"]))

    @staticmethod
    def _coerce_timestamp(value):
        if value is None:
            return None
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(timestamp):
            return None
        return timestamp

    def _resolve_snapshot_timestamp(self, anchor, snapshot_name, *paths):
        anchor = anchor or {}
        timestamp_keys = {
            "last": ("last_updated_at",),
            "first": ("first_created_at", "created_at"),
            None: ("last_updated_at", "first_created_at", "created_at"),
        }

        for key in timestamp_keys.get(snapshot_name, ()):
            timestamp = self._coerce_timestamp(anchor.get(key))
            if timestamp is not None:
                return timestamp

        mtimes = []
        for path in paths:
            if path.exists():
                mtimes.append(path.stat().st_mtime)
        if mtimes:
            return max(float(mtime) for mtime in mtimes)
        return 0.0

    def _encode_anchor_images(self, anchor_entries):
        anchors = []
        image_frames = []
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

        for entry in anchor_entries:
            image = cv2.imread(str(entry["rgb_path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法读取 anchor RGB 图片: {entry['rgb_path']}")

            ok, encoded = cv2.imencode(".jpg", image, encode_params)
            if not ok:
                raise ValueError(f"无法编码 anchor RGB 图片: {entry['rgb_path']}")

            height, width = image.shape[:2]
            anchors.append(
                {
                    "anchor_id": entry["anchor_id"],
                    "image_name": "rgb.jpg",
                    "image_format": "jpeg",
                    "height": int(height),
                    "width": int(width),
                }
            )
            image_frames.append(encoded.tobytes())

        return anchors, image_frames

    def _send_search_request(self, language_instruction, anchors, image_frames):
        metadata = {
            "type": "search_request",
            "instruction": language_instruction,
            "anchors": anchors,
            "response_format": {
                "type": "search_result",
                "mask_encoding": "png_base64",
                "mask_semantics": "nonzero_is_foreground",
            },
        }
        request_frames = [json.dumps(metadata, ensure_ascii=False).encode("utf-8"), *image_frames]
        try:
            self.socket.send_multipart(request_frames)
            response = self.socket.recv()
        except zmq.error.Again as exc:
            self.reset_socket()
            raise TimeoutError(
                f"搜索服务 {self.server_host}:{self.server_port} 在 {self.recv_timeout_ms / 1000.0:.1f}s 内无响应"
            ) from exc
        except zmq.ZMQError as exc:
            self.reset_socket()
            raise RuntimeError(f"搜索服务通信失败: {exc}") from exc

        try:
            search_result = json.loads(response.decode("utf-8"))
        except Exception as exc:
            raise ValueError("server 返回不是合法的 UTF-8 JSON 字符串") from exc

        if not isinstance(search_result, dict):
            raise TypeError(f"server 返回顶层必须是 JSON object/dict，实际为: {type(search_result).__name__}")
        return search_result

    @staticmethod
    def _clip_bbox(bbox, width, height):
        if len(bbox) != 4:
            raise ValueError(f"bbox 格式错误，期望长度为 4，实际为: {bbox}")
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 < x1 or y2 < y1:
            raise ValueError(f"bbox 无效，裁剪后结果为 {[x1, y1, x2, y2]}")
        return [x1, y1, x2, y2]

    @staticmethod
    def _normalize_mask(mask_array, expected_shape=None):
        mask_array = np.asarray(mask_array)

        if mask_array.ndim == 3:
            if mask_array.shape[2] == 1:
                mask_array = mask_array[:, :, 0]
            else:
                mask_array = np.any(mask_array > 0, axis=2)

        if mask_array.ndim != 2:
            raise ValueError(f"mask 必须是二维数组，实际 shape={mask_array.shape}")

        mask = mask_array.astype(np.float32) > 0
        if expected_shape is not None and tuple(mask.shape) != tuple(expected_shape):
            raise ValueError(f"mask 尺寸不匹配，期望 {expected_shape}，实际 {mask.shape}")
        return mask

    @classmethod
    def _decode_mask_payload(cls, mask_payload, expected_shape=None):
        if isinstance(mask_payload, dict):
            encoding = mask_payload.get("encoding")

            if encoding == "png_base64":
                if "data" not in mask_payload:
                    raise KeyError("mask.data 缺失")
                encoded_bytes = base64.b64decode(mask_payload["data"])
                image_array = np.frombuffer(encoded_bytes, dtype=np.uint8)
                decoded = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
                if decoded is None:
                    raise ValueError("mask.data 不是合法的 PNG bytes")

                mask = cls._normalize_mask(decoded, expected_shape=expected_shape)

                height = mask_payload.get("height")
                width = mask_payload.get("width")
                if height is not None and int(height) != int(mask.shape[0]):
                    raise ValueError(f"mask.height 与解码结果不一致: {height} vs {mask.shape[0]}")
                if width is not None and int(width) != int(mask.shape[1]):
                    raise ValueError(f"mask.width 与解码结果不一致: {width} vs {mask.shape[1]}")
                return mask

            if encoding == "array":
                if "data" not in mask_payload:
                    raise KeyError("mask.data 缺失")
                return cls._normalize_mask(mask_payload["data"], expected_shape=expected_shape)

            if "data" in mask_payload and encoding is None:
                return cls._normalize_mask(mask_payload["data"], expected_shape=expected_shape)

            raise ValueError(
                "当前仅支持 mask.encoding 为 'png_base64' 或 'array'，"
                f"实际收到: {encoding!r}"
            )

        if isinstance(mask_payload, list):
            return cls._normalize_mask(mask_payload, expected_shape=expected_shape)

        raise TypeError(
            "mask 字段格式错误；期望 dict 或二维 list。"
            f"实际类型: {type(mask_payload).__name__}"
        )

    @staticmethod
    def _bbox_from_mask(mask):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise ValueError("server 返回的 mask 为空，无法计算 bbox")
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    @classmethod
    def _save_mask_visualization(cls, anchor_dir, rgb_image_bgr, mask, bbox):
        overlay = rgb_image_bgr.copy()
        overlay[mask] = (0.4 * overlay[mask] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(str(anchor_dir / cls.MASK_OVERLAY_FILE_NAME), overlay)
        cv2.imwrite(str(anchor_dir / cls.MASK_FILE_NAME), mask.astype(np.uint8) * 255)

    @staticmethod
    def _aabb_corners(aabb_min, aabb_max):
        x0, y0, z0 = aabb_min
        x1, y1, z1 = aabb_max
        return np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ],
            dtype=np.float32,
        )

    # 立方体的 12 条边（顶点索引参考 _aabb_corners 顺序）
    _AABB_EDGES = (
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face (z=z0)
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face (z=z1)
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    )

    def _project_world_points_to_image(self, points_world, camera_pose, image_shape):
        """
        将世界坐标点投影到给定 anchor 图像的像素坐标。
        - camera_pose: 4x4，T_world_camera（与 _mask_to_world_points 中使用的方向一致）
        - image_shape: (H, W)
        """
        rotation = camera_pose[:3, :3].astype(np.float32)
        translation = camera_pose[:3, 3].astype(np.float32)
        # T_camera_world = inverse(T_world_camera)
        rotation_inv = rotation.T
        translation_inv = -rotation_inv @ translation

        points_world = np.asarray(points_world, dtype=np.float32)
        points_cam = points_world @ rotation_inv.T + translation_inv[None, :]

        fx = self.camera_intrinsics["fx"]
        fy = self.camera_intrinsics["fy"]
        cx = self.camera_intrinsics["cx"]
        cy = self.camera_intrinsics["cy"]

        z_cam = points_cam[:, 2]
        # 防止除零；z<=0 的点视为不可见
        eps = 1e-6
        valid = z_cam > eps
        u = np.full(points_cam.shape[0], np.nan, dtype=np.float32)
        v = np.full(points_cam.shape[0], np.nan, dtype=np.float32)
        z_safe = np.where(valid, z_cam, 1.0)
        u_all = points_cam[:, 0] * fx / z_safe + cx
        v_all = points_cam[:, 1] * fy / z_safe + cy
        u[valid] = u_all[valid]
        v[valid] = v_all[valid]

        height, width = image_shape[:2]
        # 单独再返回每个点是否在图像内的标志，便于绘制时做截断
        in_image = (
            valid
            & (u >= 0) & (u < width)
            & (v >= 0) & (v < height)
        )
        return np.stack([u, v], axis=1), valid, in_image

    @classmethod
    def _save_aabb_2d_overlay(
        cls, anchor_dir, rgb_image_bgr, aabb_corners_uv, corner_valid, bbox=None
    ):
        """
        把 3D AABB 的 8 个顶点投影后的 12 条边画在 anchor 图像上。
        无效（z<=0 或 NaN）的顶点对应的边不会绘制。
        """
        overlay = rgb_image_bgr.copy()

        # 先画 mask 的 2D bbox 作为参考（红色），便于和 3D AABB 投影对照
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 画 3D AABB 的 12 条边（绿色）
        height, width = overlay.shape[:2]
        for i, j in cls._AABB_EDGES:
            if not (corner_valid[i] and corner_valid[j]):
                continue
            pt1 = aabb_corners_uv[i]
            pt2 = aabb_corners_uv[j]
            if not (np.all(np.isfinite(pt1)) and np.all(np.isfinite(pt2))):
                continue
            # 直接在大画布上裁剪：cv2.line 自身会处理超出图像范围的部分
            x1i = int(round(float(pt1[0])))
            y1i = int(round(float(pt1[1])))
            x2i = int(round(float(pt2[0])))
            y2i = int(round(float(pt2[1])))
            cv2.line(overlay, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2, cv2.LINE_AA)

        # 画顶点
        for i, pt in enumerate(aabb_corners_uv):
            if not corner_valid[i] or not np.all(np.isfinite(pt)):
                continue
            xi = int(round(float(pt[0])))
            yi = int(round(float(pt[1])))
            if 0 <= xi < width and 0 <= yi < height:
                cv2.circle(overlay, (xi, yi), 4, (0, 255, 255), -1, cv2.LINE_AA)

        cv2.putText(
            overlay,
            "3D AABB (green) / 2D bbox (red)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = anchor_dir / cls.AABB_OVERLAY_FILE_NAME
        cv2.imwrite(str(out_path), overlay)
        return out_path

    @classmethod
    def _save_aabb_3d_plot(cls, anchor_dir, points_world, aabb_min, aabb_max):
        """
        使用 matplotlib 离线渲染 2x2 多视角点云 + AABB 立方体线框，保存为 PNG。

        子图布局：
            [左上] 自由 3D 视角           [右上] 俯视 (X-Z 平面，从 +Y 看下来)
            [左下] 正视 (X-Y 平面，相机视角)  [右下] 侧视 (Z-Y 平面，从 +X 看过去)

        每个 2D 子图的 AABB 框是该视角投影下的矩形（轴对齐），
        让标注员一眼判断点云是否被背景污染、AABB 是否合理。
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  仅用于注册 3d 投影
        except Exception as exc:
            print(f"[search_engine] matplotlib 不可用，跳过 3D AABB 图: {exc}")
            return None

        points_world = np.asarray(points_world, dtype=np.float32)
        aabb_min = np.asarray(aabb_min, dtype=np.float32)
        aabb_max = np.asarray(aabb_max, dtype=np.float32)
        center = (aabb_min + aabb_max) * 0.5
        extent = aabb_max - aabb_min

        # 点云下采样，避免渲染过慢
        max_points = 5000
        if points_world.shape[0] > max_points:
            idx = np.random.default_rng(0).choice(
                points_world.shape[0], size=max_points, replace=False
            )
            sampled = points_world[idx]
        else:
            sampled = points_world

        ext_cm = extent * 100.0
        fig = plt.figure(figsize=(14, 11))
        fig.suptitle(
            f"3D AABB Multi-View   "
            f"center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m   "
            f"extent (W x H x D) = {ext_cm[0]:.1f} x {ext_cm[1]:.1f} x {ext_cm[2]:.1f} cm   "
            f"num_points={int(points_world.shape[0])}",
            fontsize=12,
        )

        # ===== 左上：自由 3D 视角 =====
        ax3d = fig.add_subplot(2, 2, 1, projection="3d")
        ax3d.scatter(
            sampled[:, 0], sampled[:, 1], sampled[:, 2],
            s=2, c="tab:blue", alpha=0.5, label="points",
        )
        corners = cls._aabb_corners(aabb_min, aabb_max)
        for i, j in cls._AABB_EDGES:
            ax3d.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color="green", linewidth=2,
            )
        ax3d.set_xlabel("X (m)")
        ax3d.set_ylabel("Y (m)")
        ax3d.set_zlabel("Z (m)")
        ax3d.set_title("Free 3D view")
        max_range = float(extent.max()) if float(extent.max()) > 0 else 1.0
        ax3d.set_xlim(center[0] - max_range, center[0] + max_range)
        ax3d.set_ylim(center[1] - max_range, center[1] + max_range)
        ax3d.set_zlim(center[2] - max_range, center[2] + max_range)
        ax3d.legend(loc="upper right", fontsize=8)

        def _draw_2d_view(ax, pts_a, pts_b, lo_a, hi_a, lo_b, hi_b,
                          xlabel, ylabel, title, invert_y=False):
            ax.scatter(pts_a, pts_b, s=2, c="tab:blue", alpha=0.5)
            # AABB 在该视角的矩形
            ax.plot(
                [lo_a, hi_a, hi_a, lo_a, lo_a],
                [lo_b, lo_b, hi_b, hi_b, lo_b],
                color="green", linewidth=2,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.3)
            if invert_y:
                ax.invert_yaxis()

        # ===== 右上：俯视 X-Z（看物体的"长 x 深"），从 +Y 看下来 =====
        ax_top = fig.add_subplot(2, 2, 2)
        _draw_2d_view(
            ax_top, sampled[:, 0], sampled[:, 2],
            aabb_min[0], aabb_max[0], aabb_min[2], aabb_max[2],
            xlabel="X (m)", ylabel="Z (m, depth)",
            title=f"Top view (X-Z)  W={ext_cm[0]:.1f}cm  D={ext_cm[2]:.1f}cm",
        )

        # ===== 左下：正视 X-Y（相机视角），Y 轴反转使图像方向直观 =====
        ax_front = fig.add_subplot(2, 2, 3)
        _draw_2d_view(
            ax_front, sampled[:, 0], sampled[:, 1],
            aabb_min[0], aabb_max[0], aabb_min[1], aabb_max[1],
            xlabel="X (m)", ylabel="Y (m, image down)",
            title=f"Front view (X-Y, camera)  W={ext_cm[0]:.1f}cm  H={ext_cm[1]:.1f}cm",
            invert_y=True,
        )

        # ===== 右下：侧视 Z-Y（看物体的"深 x 高"），从 +X 看过去 =====
        ax_side = fig.add_subplot(2, 2, 4)
        _draw_2d_view(
            ax_side, sampled[:, 2], sampled[:, 1],
            aabb_min[2], aabb_max[2], aabb_min[1], aabb_max[1],
            xlabel="Z (m, depth)", ylabel="Y (m, image down)",
            title=f"Side view (Z-Y)  D={ext_cm[2]:.1f}cm  H={ext_cm[1]:.1f}cm",
            invert_y=True,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = anchor_dir / cls.AABB_3D_FILE_NAME
        fig.savefig(str(out_path), dpi=110)
        plt.close(fig)
        return out_path

    def _mask_to_world_points(self, mask, depth_image, camera_pose):
        if self.camera_intrinsics is None:
            raise RuntimeError(
                "未提供相机内参，无法从 mask + depth 恢复 3D。"
                "请在 SearchEngine 初始化时传入 camera_intrinsics={'fx':..., 'fy':..., 'cx':..., 'cy':...}"
            )

        if depth_image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"depth 和 mask 尺寸不一致: depth={depth_image.shape[:2]}, mask={mask.shape[:2]}"
            )

        valid_mask = mask & (depth_image > 0)
        v_coords, u_coords = np.nonzero(valid_mask)
        if len(u_coords) == 0:
            raise ValueError("mask 区域内没有有效深度，无法恢复 3D 点云")

        depth_m = depth_image[v_coords, u_coords].astype(np.float32) * self.depth_scale
        fx = self.camera_intrinsics["fx"]
        fy = self.camera_intrinsics["fy"]
        cx = self.camera_intrinsics["cx"]
        cy = self.camera_intrinsics["cy"]

        x_cam = (u_coords.astype(np.float32) - cx) * depth_m / fx
        y_cam = (v_coords.astype(np.float32) - cy) * depth_m / fy
        z_cam = depth_m
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        rotation = camera_pose[:3, :3].astype(np.float32)
        translation = camera_pose[:3, 3].astype(np.float32)
        points_world = points_cam @ rotation.T + translation[None, :]
        return points_world

    @staticmethod
    def _compute_aabb(points_world):
        if len(points_world) == 0:
            raise ValueError("点云为空，无法计算 AABB")

        aabb_min = points_world.min(axis=0)
        aabb_max = points_world.max(axis=0)
        center = (aabb_min + aabb_max) * 0.5
        extent = aabb_max - aabb_min
        return {
            "min": aabb_min.tolist(),
            "max": aabb_max.tolist(),
            "center": center.tolist(),
            "extent": extent.tolist(),
            "num_points": int(points_world.shape[0]),
        }

    def _resolve_search_target(self, search_result, anchor_entries):
        anchor_by_id = {entry["anchor_id"]: entry for entry in anchor_entries}

        anchor_id = search_result.get("anchor_id")
        if anchor_id is None and "image_index" in search_result:
            image_index = int(search_result["image_index"])
            if image_index < 0 or image_index >= len(anchor_entries):
                raise IndexError(f"image_index 越界: {image_index}")
            anchor_id = anchor_entries[image_index]["anchor_id"]

        if anchor_id is None:
            raise KeyError(f"server 返回中缺少 anchor_id 或 image_index: {search_result}")
        if anchor_id not in anchor_by_id:
            raise KeyError(f"server 返回的 anchor_id 不存在于本地地图中: {anchor_id}")

        mask_payload = search_result.get("mask")
        if mask_payload is None and "segmentation_mask" in search_result:
            mask_payload = search_result["segmentation_mask"]
        if mask_payload is None:
            raise KeyError(f"server 返回中缺少 mask: {search_result}")

        return anchor_by_id[anchor_id], search_result.get("bbox"), mask_payload

    def _load_anchor_observation(self, anchor_entry):
        rgb_image = cv2.imread(str(anchor_entry["rgb_path"]), cv2.IMREAD_COLOR)
        depth_image = cv2.imread(str(anchor_entry["depth_path"]), cv2.IMREAD_UNCHANGED)
        camera_pose = np.load(anchor_entry["camera_pose_path"])

        if rgb_image is None:
            raise ValueError(f"无法读取 RGB 图片: {anchor_entry['rgb_path']}")
        if depth_image is None:
            raise ValueError(f"无法读取深度图: {anchor_entry['depth_path']}")
        if camera_pose.shape != (4, 4):
            raise ValueError(f"camera pose 形状异常: {anchor_entry['camera_pose_path']} -> {camera_pose.shape}")

        return rgb_image, depth_image, camera_pose

    def localize_from_search_result(self, search_result, anchor_entries=None):
        anchor_entries = anchor_entries or self._get_anchor_entries()
        anchor_entry, bbox, mask_payload = self._resolve_search_target(search_result, anchor_entries)
        rgb_image, depth_image, camera_pose = self._load_anchor_observation(anchor_entry)

        expected_shape = rgb_image.shape[:2]
        mask = self._decode_mask_payload(mask_payload, expected_shape=expected_shape)
        if not mask.any():
            raise ValueError("server 返回的 mask 为空，无法执行 3D 定位")

        if bbox is None:
            bbox = self._bbox_from_mask(mask)
        clipped_bbox = self._clip_bbox(bbox, rgb_image.shape[1], rgb_image.shape[0])

        self._save_mask_visualization(anchor_entry["anchor_dir"], rgb_image, mask, clipped_bbox)

        points_world = self._mask_to_world_points(mask, depth_image, camera_pose)
        aabb = self._compute_aabb(points_world)

        # 生成 3D AABB 可视化
        aabb_overlay_path = None
        aabb_3d_path = None
        try:
            corners_world = self._aabb_corners(
                np.asarray(aabb["min"], dtype=np.float32),
                np.asarray(aabb["max"], dtype=np.float32),
            )
            corners_uv, corner_valid, _ = self._project_world_points_to_image(
                corners_world, camera_pose, rgb_image.shape
            )
            aabb_overlay_path = self._save_aabb_2d_overlay(
                anchor_entry["anchor_dir"],
                rgb_image,
                corners_uv,
                corner_valid,
                bbox=clipped_bbox,
            )
        except Exception as exc:
            print(f"[search_engine] 保存 3D AABB 投影叠加图失败: {exc}")

        try:
            aabb_3d_path = self._save_aabb_3d_plot(
                anchor_entry["anchor_dir"],
                points_world,
                aabb["min"],
                aabb["max"],
            )
        except Exception as exc:
            print(f"[search_engine] 保存 3D AABB 立体图失败: {exc}")

        print(f"Matched anchor: {anchor_entry['anchor_id']}")
        print(f"Segmentation bbox: {clipped_bbox}")
        print(f"Point cloud size: {aabb['num_points']}")
        print("World AABB:", json.dumps(aabb, ensure_ascii=False, indent=2))

        return {
            "anchor_id": anchor_entry["anchor_id"],
            "bbox": clipped_bbox,
            "mask_num_pixels": int(mask.sum()),
            "mask_shape": [int(mask.shape[0]), int(mask.shape[1])],
            "points_world": points_world,
            "aabb": aabb,
            "mask_path": str(anchor_entry["anchor_dir"] / self.MASK_FILE_NAME),
            "mask_overlay_path": str(anchor_entry["anchor_dir"] / self.MASK_OVERLAY_FILE_NAME),
            "aabb_overlay_path": str(aabb_overlay_path) if aabb_overlay_path else None,
            "aabb_3d_path": str(aabb_3d_path) if aabb_3d_path else None,
        }

    def search_by_language_instruction(self, language_instruction: str):
        if not isinstance(language_instruction, str) or not language_instruction.strip():
            raise ValueError("language_instruction 必须是非空字符串")

        anchor_entries = self._get_anchor_entries()
        if not anchor_entries:
            raise ValueError(f"anchor 地图为空: {self.anchor_map_dir}")

        anchors, image_frames = self._encode_anchor_images(anchor_entries)
        search_result = self._send_search_request(language_instruction, anchors, image_frames)
        print("Remote search result:", search_result)

        if search_result.get("success") is not True:
            raise RuntimeError(f"远端搜索失败: {search_result.get('error_message', '未知错误')}")
        if "found" not in search_result:
            raise KeyError(f"server 返回中缺少 found: {search_result}")
        if not isinstance(search_result["found"], bool):
            raise TypeError(f"server 返回的 found 必须是 bool，实际为: {type(search_result['found']).__name__}")
        if not search_result["found"]:
            raise LookupError(f"远端搜索成功，但未找到目标: {search_result.get('message', 'no match')}")

        localization = self.localize_from_search_result(search_result, anchor_entries=anchor_entries)
        return {"search_result": search_result, "localization": localization}


def _build_argparser():
    parser = argparse.ArgumentParser(description="SearchEngine end-to-end test (server returns mask)")
    parser.add_argument("anchor_map_dir", type=str, help="ROIMapFixed 保存的 anchor 地图目录")
    parser.add_argument("language_instruction", type=str, help="语言检索指令")
    parser.add_argument("--server-host", type=str, default="219.223.200.92", help="远端 server IP/hostname")
    parser.add_argument("--server-port", type=int, default=5555, help="远端 server 端口")
    parser.add_argument("--fx", type=float, default=525.0, help="相机内参 fx")
    parser.add_argument("--fy", type=float, default=525.0, help="相机内参 fy")
    parser.add_argument("--cx", type=float, default=319.5, help="相机内参 cx")
    parser.add_argument("--cy", type=float, default=239.5, help="相机内参 cy")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="深度缩放，默认毫米转米")
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    engine = SearchEngine(
        anchor_map_dir=args.anchor_map_dir,
        server_host=args.server_host,
        server_port=args.server_port,
        camera_intrinsics={
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
        },
        depth_scale=args.depth_scale,
    )

    try:
        engine.search_by_language_instruction(args.language_instruction)
    finally:
        engine.close()
