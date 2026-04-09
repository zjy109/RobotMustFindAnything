import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import zmq


class SearchEngine:
    """
    读取 ROIMapFixed 保存的 anchor 地图，将全部 anchor RGB 图像和语言指令发给远端服务。
    远端返回目标所在 anchor 和 2D bbox 后，本地继续：
    1. 使用 SAM2 + bbox prompt 分割目标 mask
    2. 结合深度图、相机内参和 camera pose 反投影到世界坐标
    3. 计算目标点云的 AABB 并打印

    请求协议（client -> server）：
    - multipart[0]：UTF-8 JSON metadata
    - multipart[1:]：与 anchors 顺序一致的 JPEG 图片 bytes

    响应协议（server -> client）：
    - UTF-8 JSON 字符串，至少包含：
      {
        "anchor_id": "anchor_0003",
        "bbox": [x1, y1, x2, y2]
      }
    也兼容：
      {
        "image_index": 3,
        "bbox": [x1, y1, x2, y2]
      }
    """

    @staticmethod
    def default_sam2_cache_dir(model_id):
        cache_name = str(model_id).replace("/", "--")
        return Path(__file__).resolve().parent.parent / ".model_cache" / cache_name

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
        sam2_model_id="facebook/sam2.1-hiera-small",
        sam2_cache_dir=None,
        sam2_device=None,
    ):
        self.anchor_map_dir = Path(anchor_map_dir)
        self.index_path = self.anchor_map_dir / "anchors.json"
        self.server_host = str(server_host)
        self.server_port = int(server_port)
        self.jpeg_quality = int(jpeg_quality)
        self.recv_timeout_ms = int(recv_timeout_ms)
        self.send_timeout_ms = int(send_timeout_ms)
        self.depth_scale = float(depth_scale)
        self.sam2_model_id = str(sam2_model_id)
        self.sam2_cache_dir = Path(sam2_cache_dir) if sam2_cache_dir else self.default_sam2_cache_dir(self.sam2_model_id)
        self.sam2_device = sam2_device
        self.camera_intrinsics = self._normalize_camera_intrinsics(camera_intrinsics)

        self.sam2_model = None
        self.sam2_processor = None
        self.torch = None

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
        for anchor in self.index.get("anchors", []):
            anchor_id = anchor["id"]
            anchor_dir = self.anchor_map_dir / anchor_id
            rgb_path = anchor_dir / "rgb.png"
            depth_path = anchor_dir / "depth.png"
            camera_pose_path = anchor_dir / "camera_pose.npy"

            if not rgb_path.exists():
                raise FileNotFoundError(f"anchor 缺少 RGB 图片: {rgb_path}")
            if not depth_path.exists():
                raise FileNotFoundError(f"anchor 缺少深度图: {depth_path}")
            if not camera_pose_path.exists():
                raise FileNotFoundError(f"anchor 缺少 camera pose: {camera_pose_path}")

            entries.append(
                {
                    "anchor_id": anchor_id,
                    "anchor_dir": anchor_dir,
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "camera_pose_path": camera_pose_path,
                }
            )
        return entries

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

            anchors.append({"anchor_id": entry["anchor_id"], "image_name": "rgb.jpg"})
            image_frames.append(encoded.tobytes())

        return anchors, image_frames

    def _send_search_request(self, language_instruction, anchors, image_frames):
        metadata = {
            "type": "search_request",
            "instruction": language_instruction,
            "anchors": anchors,
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
        return json.loads(response.decode("utf-8"))

    def _ensure_sam2_model(self):
        if self.sam2_model is not None and self.sam2_processor is not None:
            return

        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise RuntimeError(
                "SAM2 依赖未安装。请先安装 torch、transformers、accelerate、huggingface_hub、pillow。"
            ) from exc

        self.torch = torch
        device = self.sam2_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam2_cache_dir.mkdir(parents=True, exist_ok=True)

        self.sam2_model = Sam2Model.from_pretrained(
            self.sam2_model_id,
            cache_dir=str(self.sam2_cache_dir),
        ).to(device)
        self.sam2_model.eval()
        self.sam2_processor = Sam2Processor.from_pretrained(
            self.sam2_model_id,
            cache_dir=str(self.sam2_cache_dir),
        )
        self.sam2_device = device
        print(
            f"SAM2 loaded: model_id={self.sam2_model_id}, device={self.sam2_device}, "
            f"cache_dir={self.sam2_cache_dir}"
        )

    def warmup_models(self):
        self._ensure_sam2_model()

    @staticmethod
    def _clip_bbox(bbox, width, height):
        if len(bbox) != 4:
            raise ValueError(f"bbox 格式错误，期望长度为 4，实际为: {bbox}")
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox 无效，裁剪后结果为 {[x1, y1, x2, y2]}")
        return [x1, y1, x2, y2]

    @staticmethod
    def _extract_best_mask(mask_output, iou_scores=None):
        mask_array = np.asarray(mask_output)

        if mask_array.ndim == 2:
            return mask_array.astype(bool)

        if mask_array.ndim == 4:
            if mask_array.shape[0] != 1:
                raise ValueError(f"当前仅支持单图单目标分割，mask shape={mask_array.shape}")
            mask_array = mask_array[0]

        if mask_array.ndim != 3:
            raise ValueError(f"无法解析 SAM2 mask 输出，shape={mask_array.shape}")

        if iou_scores is not None:
            score_array = np.asarray(iou_scores)
            while score_array.ndim > 1 and score_array.shape[0] == 1:
                score_array = score_array[0]
            best_index = int(np.argmax(score_array))
        else:
            best_index = 0

        return mask_array[best_index].astype(bool)

    def _segment_with_sam2(self, rgb_image_bgr, bbox):
        self._ensure_sam2_model()

        rgb_image = cv2.cvtColor(rgb_image_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb_image.shape[:2]
        bbox = self._clip_bbox(bbox, width, height)

        inputs = self.sam2_processor(
            images=rgb_image,
            input_boxes=[[bbox]],
            return_tensors="pt",
        ).to(self.sam2_device)

        with self.torch.no_grad():
            outputs = self.sam2_model(**inputs, multimask_output=False)

        processed_masks = self.sam2_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )
        mask = self._extract_best_mask(
            processed_masks[0],
            outputs.iou_scores.cpu().numpy() if hasattr(outputs, "iou_scores") else None,
        )
        return mask, bbox

    @staticmethod
    def _save_mask_visualization(anchor_dir, rgb_image_bgr, mask, bbox):
        overlay = rgb_image_bgr.copy()
        overlay[mask] = (0.4 * overlay[mask] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(str(anchor_dir / "sam2_mask_overlay.png"), overlay)
        cv2.imwrite(str(anchor_dir / "sam2_mask.png"), (mask.astype(np.uint8) * 255))

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
        if "bbox" not in search_result:
            raise KeyError(f"server 返回中缺少 bbox: {search_result}")

        return anchor_by_id[anchor_id], search_result["bbox"]

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
        anchor_entry, bbox = self._resolve_search_target(search_result, anchor_entries)
        rgb_image, depth_image, camera_pose = self._load_anchor_observation(anchor_entry)

        mask, clipped_bbox = self._segment_with_sam2(rgb_image, bbox)
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
            "points_world": points_world,
            "aabb": aabb,
            "mask_path": str(anchor_entry["anchor_dir"] / "sam2_mask.png"),
            "mask_overlay_path": str(anchor_entry["anchor_dir"] / "sam2_mask_overlay.png"),
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
        if search_result['success'] is not True:
            raise RuntimeError(f"远端搜索失败: {search_result.get('error_message', '未知错误')}")

        localization = self.localize_from_search_result(search_result, anchor_entries=anchor_entries)
        return {"search_result": search_result, "localization": localization}


def _build_argparser():
    parser = argparse.ArgumentParser(description="SearchEngine end-to-end test")
    parser.add_argument("anchor_map_dir", type=str, help="ROIMapFixed 保存的 anchor 地图目录")
    parser.add_argument("language_instruction", type=str, help="语言检索指令")
    parser.add_argument("--server-host", type=str, default="219.223.200.92", help="远端 server IP/hostname")
    parser.add_argument("--server-port", type=int, default=5555, help="远端 server 端口")
    parser.add_argument("--fx", type=float, default=525.0, help="相机内参 fx")
    parser.add_argument("--fy", type=float, default=525.0, help="相机内参 fy")
    parser.add_argument("--cx", type=float, default=319.5, help="相机内参 cx")
    parser.add_argument("--cy", type=float, default=239.5, help="相机内参 cy")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="深度缩放，默认毫米转米")
    parser.add_argument(
        "--sam2-model-id",
        type=str,
        default="facebook/sam2.1-hiera-small",
        help="Hugging Face 上的 SAM2 模型 ID",
    )
    parser.add_argument(
        "--sam2-cache-dir",
        type=str,
        default=None,
        help="SAM2 模型本地缓存目录，不传则默认放到 rofa/.model_cache/<model-id>",
    )
    parser.add_argument(
        "--sam2-device",
        type=str,
        default=None,
        help="SAM2 推理设备，默认自动选择 cuda/cpu",
    )
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
        sam2_model_id=args.sam2_model_id,
        sam2_cache_dir=args.sam2_cache_dir,
        sam2_device=args.sam2_device,
    )

    try:
        engine.search_by_language_instruction(args.language_instruction)
    finally:
        engine.close()
