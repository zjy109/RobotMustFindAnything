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
