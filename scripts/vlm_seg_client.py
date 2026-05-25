"""
VLM 预分割 ZMQ 客户端
========================

这是一个轻量客户端，作用就是：
    给一张 RGB 图 + 一个文本 prompt
        → 通过 ZMQ 调用 rofa.vlm_server.server.VLMSearchServer
        → 拿回一张 (H, W) bool mask（SAM2 在服务端生成）
        → 失败/未找到 时返回 None

设计原则：
- 客户端**不依赖任何模型/torch**，只依赖 zmq + numpy + opencv + Pillow，
  方便标注员的工作机直接装。
- 服务端协议见 rofa/vlm_server/server.py 的 handle_request / search()。
  消息格式：multipart = [metadata_json_bytes, jpg_bytes_1, jpg_bytes_2, ...]
  响应：JSON，关键字段：
      success: bool
      object_found / found: bool
      bbox: [x1, y1, x2, y2]  # 像素，相对于发送的那张图
      image_width / image_height: int
      mask: { encoding: "png_base64", height, width, data }  # 可选，SAM2 失败时缺失

CLI 自测：
    python scripts/vlm_seg_client.py \\
        --host 127.0.0.1 --port 5555 \\
        --image raw_capture/pending/guochan/guochan_0001/rgb.jpg \\
        --prompt "spatula" \\
        --out /tmp/vlm_pred_mask.png
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import zmq
from PIL import Image


# --------------------------------------------------------------------------- #
# 默认连接参数
# --------------------------------------------------------------------------- #
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555
DEFAULT_TIMEOUT_MS = 90_000  # RynnBrain + SAM2 首次推理可能 ~30s，给 90s


class VLMSegClientError(RuntimeError):
    """所有可预期的客户端错误都包装成它，调用方可以选择吞掉降级。"""


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #

class VLMSegClient:
    """
    单连接 ZMQ REQ 客户端。

    用法（推荐 with）:
        with VLMSegClient(host, port) as client:
            mask = client.predict(rgb_bgr, prompt="spatula")
    或：
        client = VLMSegClient(host, port)
        try:
            mask = client.predict(rgb_bgr, "spatula")
        finally:
            client.close()

    每次 predict 是阻塞调用，超时由 timeout_ms 控制。
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        jpeg_quality: int = 92,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.jpeg_quality = int(jpeg_quality)

        self._ctx: Optional[zmq.Context] = None
        self._sock: Optional[zmq.Socket] = None
        self._endpoint = f"tcp://{self.host}:{self.port}"

        self._connect()

    # ---------- 连接管理 ---------- #

    def _connect(self) -> None:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self._endpoint)
        self._ctx = ctx
        self._sock = sock

    def _reconnect(self) -> None:
        """
        REQ socket 一旦在错误状态（比如超时未 recv 完成）就不能继续 send，
        这里直接重建一个 socket。Context 复用 zmq.Context.instance()。
        """
        try:
            if self._sock is not None:
                self._sock.close(linger=0)
        finally:
            self._sock = None
        self._connect()

    def close(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close(linger=0)
        finally:
            self._sock = None
        # 不调用 ctx.term()，避免破坏全局 instance()

    def __enter__(self) -> "VLMSegClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------- 编码工具 ---------- #

    @staticmethod
    def _bgr_to_jpeg_bytes(rgb_bgr: np.ndarray, quality: int) -> bytes:
        """
        cv2 用 BGR，server 端 PIL.Image.open + .convert("RGB") 会自动按 JPEG
        色彩空间还原，所以我们走 cv2.imencode(".jpg") 即可。
        """
        if rgb_bgr.dtype != np.uint8:
            raise VLMSegClientError(f"rgb_bgr 必须是 uint8，实际 dtype={rgb_bgr.dtype}")
        if rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
            raise VLMSegClientError(f"rgb_bgr 必须是 HxWx3，实际 shape={rgb_bgr.shape}")
        ok, buf = cv2.imencode(
            ".jpg", rgb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not ok:
            raise VLMSegClientError("cv2.imencode JPEG 失败")
        return buf.tobytes()

    @staticmethod
    def _decode_mask_payload(
        payload: Dict[str, Any], expected_hw: Tuple[int, int]
    ) -> np.ndarray:
        """
        解析 server 返回的 mask 子结构 -> bool ndarray (H, W)。
        若尺寸与原图不匹配，抛出 VLMSegClientError。
        """
        encoding = payload.get("encoding")
        if encoding != "png_base64":
            raise VLMSegClientError(f"未知 mask encoding: {encoding!r}")
        b64 = payload.get("data")
        if not isinstance(b64, str) or not b64:
            raise VLMSegClientError("mask payload 缺少 data 字段")
        try:
            png_bytes = base64.b64decode(b64)
        except Exception as exc:
            raise VLMSegClientError(f"mask base64 解码失败: {exc}") from exc

        arr = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise VLMSegClientError("mask PNG 解码失败")

        # SAM2 输出可能是 (H, W) 或 (H, W, 1)，统一成 (H, W) bool
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.shape != expected_hw:
            raise VLMSegClientError(
                f"mask 尺寸 {arr.shape} 与原图 {expected_hw} 不一致，丢弃"
            )

        return arr > 0

    # ---------- 主接口 ---------- #

    def predict(
        self,
        rgb_bgr: np.ndarray,
        prompt: str,
        anchor_id: str = "annotate_query",
    ) -> Optional[Dict[str, Any]]:
        """
        发送一张图给 server，等待响应，返回：
            {
                "mask": np.ndarray (H, W) bool,
                "bbox_pixel": [x1, y1, x2, y2],
                "prompt": prompt,
                "raw_response": <dict>,   # 服务端原始 JSON，便于落盘
            }
        若 server 报错 / 未找到目标 / 没生成 mask / 尺寸不匹配，则返回 None
        （原因会通过 print 输出，方便排查；不抛异常以方便降级）。
        """
        if self._sock is None:
            self._connect()

        h, w = rgb_bgr.shape[:2]
        try:
            jpg = self._bgr_to_jpeg_bytes(rgb_bgr, self.jpeg_quality)
        except VLMSegClientError as exc:
            print(f"[vlm_client] 图片编码失败: {exc}")
            return None

        metadata = {
            "type": "search_request",
            "instruction": prompt,
            "anchors": [{"anchor_id": anchor_id, "image_name": "rgb.jpg"}],
            "save_results": False,
        }

        message_parts = [
            json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
            jpg,
        ]

        t0 = time.time()
        try:
            self._sock.send_multipart(message_parts)
            resp_bytes = self._sock.recv()
        except zmq.error.Again:
            elapsed = time.time() - t0
            print(f"[vlm_client] 超时 ({elapsed:.1f}s, timeout={self.timeout_ms}ms)")
            self._reconnect()
            return None
        except Exception as exc:
            print(f"[vlm_client] ZMQ 通信异常: {exc}")
            self._reconnect()
            return None

        try:
            resp = json.loads(resp_bytes.decode("utf-8"))
        except Exception as exc:
            print(f"[vlm_client] 响应 JSON 解析失败: {exc}")
            return None

        if not resp.get("success"):
            print(f"[vlm_client] server 返回 success=False: {resp.get('error')}")
            return None

        if not (resp.get("object_found") or resp.get("found")):
            print("[vlm_client] 模型未找到目标 (object_found=False)")
            return {"mask": None, "bbox_pixel": None, "prompt": prompt, "raw_response": resp}

        if "mask" not in resp:
            print("[vlm_client] 找到 bbox 但无 mask（SAM2 失败）")
            return {
                "mask": None,
                "bbox_pixel": resp.get("bbox"),
                "prompt": prompt,
                "raw_response": resp,
            }

        try:
            mask_bool = self._decode_mask_payload(resp["mask"], (h, w))
        except VLMSegClientError as exc:
            print(f"[vlm_client] mask 解码失败: {exc}")
            return None

        elapsed = time.time() - t0
        n_fg = int(mask_bool.sum())
        print(
            f"[vlm_client] ✓ 预标成功 prompt='{prompt}' "
            f"fg_pixels={n_fg} ({n_fg / mask_bool.size * 100:.1f}%) "
            f"elapsed={elapsed:.1f}s"
        )

        return {
            "mask": mask_bool,
            "bbox_pixel": resp.get("bbox"),
            "prompt": prompt,
            "raw_response": resp,
        }


# --------------------------------------------------------------------------- #
# CLI 自测
# --------------------------------------------------------------------------- #

def _cli_main() -> int:
    parser = argparse.ArgumentParser(
        description="VLM 预分割 ZMQ 客户端自测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--image", type=str, required=True, help="本地 RGB 图路径")
    parser.add_argument("--prompt", type=str, required=True, help="目标物体描述")
    parser.add_argument(
        "--out", type=str, default=None,
        help="可选：将 mask 保存为 PNG（uint8 0/255）到该路径",
    )
    parser.add_argument(
        "--out-overlay", type=str, default=None,
        help="可选：将 mask 红色叠加到 RGB 上保存为 PNG",
    )
    args = parser.parse_args()

    img_path = Path(args.image).expanduser().resolve()
    if not img_path.exists():
        print(f"[error] image 不存在: {img_path}")
        return 2

    rgb_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        print(f"[error] 无法读取图片: {img_path}")
        return 2

    with VLMSegClient(args.host, args.port, args.timeout_ms) as client:
        result = client.predict(rgb_bgr, args.prompt)

    if result is None:
        print("[result] 预标失败（详见上方日志）")
        return 1

    mask = result["mask"]
    bbox = result["bbox_pixel"]

    if mask is None:
        print(f"[result] 模型未找到 / 无 mask；bbox={bbox}")
        return 1

    print(f"[result] ✓ mask shape={mask.shape}  fg={int(mask.sum())}  bbox={bbox}")

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), (mask.astype(np.uint8) * 255))
        print(f"  mask saved -> {out_path}")

    if args.out_overlay:
        ov_path = Path(args.out_overlay).expanduser().resolve()
        ov_path.parent.mkdir(parents=True, exist_ok=True)
        overlay = rgb_bgr.copy()
        red_layer = np.zeros_like(overlay)
        red_layer[..., 2] = 255  # BGR 的 R
        overlay[mask] = cv2.addWeighted(
            overlay[mask], 0.5, red_layer[mask], 0.5, 0
        )
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(ov_path), overlay)
        print(f"  overlay saved -> {ov_path}")

    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
