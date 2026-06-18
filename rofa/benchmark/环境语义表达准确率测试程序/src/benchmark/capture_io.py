"""采集样本的读写、随机位姿与点云/AABB 工具。

供 ``sample_rsd4xx.py``（RealSense 采集）与 ``ui.py``（查看 / 推理）共用，
**不依赖** PyTorch / 大模型，方便在任意环境单独使用。

采集样本目录结构（与评测数据集尽量同构，但用随机模拟 ``pose.json``
取代评测数据集中的 GT ``aabb.json``）::

    <capture_dir>/
        samples.json                 # 样本索引: [{sample_id, sample_dir, created_at}, ...]
        samples/<sample_id>/
            rgb.jpg                  # RGB 彩色图
            depth.png                # 16-bit 深度（已对齐到彩色，单位由 depth_scale 决定）
            intrinsics.json          # {fx, fy, cx, cy, depth_scale, width, height}
            pose.json                # 随机模拟的相机->世界位姿

位姿 ``pose.json`` 字段::

    {
      "frame": "camera_to_world",
      "translation": [x, y, z],            # 米
      "quaternion_xyzw": [qx, qy, qz, qw], # 单位四元数
      "matrix": [[..4x4..]]                # 齐次变换矩阵（行优先）
    }
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 随机位姿
# ---------------------------------------------------------------------------

def quaternion_to_matrix(q_xyzw: List[float]) -> np.ndarray:
    """单位四元数 (qx, qy, qz, qw) -> 3x3 旋转矩阵。"""
    x, y, z, w = q_xyzw
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def random_quaternion(rng: np.random.Generator) -> List[float]:
    """均匀分布的随机单位四元数 (Shoemake)，返回 [qx, qy, qz, qw]。"""
    u1, u2, u3 = rng.random(3)
    qw = float(np.sqrt(1 - u1) * np.sin(2 * np.pi * u2))
    qx = float(np.sqrt(1 - u1) * np.cos(2 * np.pi * u2))
    qy = float(np.sqrt(u1) * np.sin(2 * np.pi * u3))
    qz = float(np.sqrt(u1) * np.cos(2 * np.pi * u3))
    return [qx, qy, qz, qw]


def random_pose(
    rng: Optional[np.random.Generator] = None,
    trans_range: Tuple[float, float, float] = (1.0, 1.0, 2.0),
) -> Dict[str, Any]:
    """生成一个随机的"相机->世界"位姿。

    - 旋转：均匀随机
    - 平移：x,y ∈ [-r, r]，z ∈ [0, r]（默认 r=(1,1,2) 米）
    """
    if rng is None:
        rng = np.random.default_rng()
    rx, ry, rz = trans_range
    t = [
        float(rng.uniform(-rx, rx)),
        float(rng.uniform(-ry, ry)),
        float(rng.uniform(0.0, rz)),
    ]
    q = random_quaternion(rng)
    R = quaternion_to_matrix(q)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return {
        "frame": "camera_to_world",
        "translation": t,
        "quaternion_xyzw": q,
        "matrix": T.tolist(),
    }


def pose_to_matrix(pose: Dict[str, Any]) -> np.ndarray:
    """pose.json -> 4x4 齐次矩阵。优先用 matrix 字段，否则用四元数+平移重建。"""
    if isinstance(pose.get("matrix"), list):
        return np.array(pose["matrix"], dtype=np.float64).reshape(4, 4)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_matrix(pose["quaternion_xyzw"])
    T[:3, 3] = pose["translation"]
    return T


# ---------------------------------------------------------------------------
# AABB / 点云
# ---------------------------------------------------------------------------

def transform_aabb(aabb: List[float], T: np.ndarray) -> List[float]:
    """把相机系 6 维 AABB [xmin,ymin,zmin,xmax,ymax,zmax] 用 4x4 变换到目标系。

    做法：取 8 个角点变换后重新取轴对齐包围盒。
    """
    mn = np.asarray(aabb[:3], dtype=np.float64)
    mx = np.asarray(aabb[3:], dtype=np.float64)
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
    ], dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    world = corners @ R.T + t
    wmn = world.min(axis=0)
    wmx = world.max(axis=0)
    return [float(wmn[0]), float(wmn[1]), float(wmn[2]),
            float(wmx[0]), float(wmx[1]), float(wmx[2])]


def depth_to_points(
    depth_map: np.ndarray,
    intrinsics: Dict[str, float],
    rgb: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    stride: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """深度图反投影为相机系点云 (N,3)；可选返回对应颜色 (N,3) [0,1]。

    - ``mask`` 给定时只取 mask>0 的像素
    - ``stride`` 用于整图下采样（仅在无 mask 时生效，加速 3D 预览）
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    depth_scale = float(intrinsics.get("depth_scale", 0.001))

    if mask is not None:
        ys, xs = np.where(mask > 0)
    else:
        h, w = depth_map.shape[:2]
        yy, xx = np.mgrid[0:h:stride, 0:w:stride]
        ys, xs = yy.ravel(), xx.ravel()

    depths = depth_map[ys, xs].astype(np.float32) * depth_scale
    valid = depths > 0
    xs, ys, depths = xs[valid], ys[valid], depths[valid]
    if len(depths) == 0:
        return np.zeros((0, 3), np.float32), (None if rgb is None else np.zeros((0, 3), np.float32))

    X = (xs - cx) * depths / fx
    Y = (ys - cy) * depths / fy
    Z = depths
    points = np.stack((X, Y, Z), axis=-1).astype(np.float32)

    colors = None
    if rgb is not None:
        colors = rgb[ys, xs].astype(np.float32) / 255.0
    return points, colors


# ---------------------------------------------------------------------------
# 样本索引 / 读写
# ---------------------------------------------------------------------------

def index_path(capture_dir: Path) -> Path:
    return capture_dir / "samples.json"


def load_index(capture_dir: Path) -> List[Dict[str, Any]]:
    """读取采集目录的 samples.json，没有则返回空列表。"""
    p = index_path(capture_dir)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict) and isinstance(doc.get("samples"), list):
        return doc["samples"]
    return []


def write_index(capture_dir: Path, records: List[Dict[str, Any]]) -> None:
    p = index_path(capture_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def save_sample(
    capture_dir: Path,
    sample_id: str,
    rgb_bgr: np.ndarray,
    depth_u16: np.ndarray,
    intrinsics: Dict[str, Any],
    pose: Dict[str, Any],
) -> Dict[str, Any]:
    """保存单个采集样本，并把记录追加进 samples.json，返回该记录。

    ``rgb_bgr``  : (H,W,3) BGR uint8（OpenCV 习惯）
    ``depth_u16``: (H,W)   uint16 原始深度（单位由 intrinsics.depth_scale 决定）
    """
    import cv2  # 局部导入，避免无 cv2 环境 import 本模块即失败

    rel_dir = Path("samples") / sample_id
    sample_dir = capture_dir / rel_dir
    sample_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(sample_dir / "rgb.jpg"), rgb_bgr)
    cv2.imwrite(str(sample_dir / "depth.png"), depth_u16.astype(np.uint16))
    with open(sample_dir / "intrinsics.json", "w", encoding="utf-8") as f:
        json.dump(intrinsics, f, ensure_ascii=False, indent=2)
    with open(sample_dir / "pose.json", "w", encoding="utf-8") as f:
        json.dump(pose, f, ensure_ascii=False, indent=2)

    record = {
        "sample_id": sample_id,
        "sample_dir": str(rel_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    records = load_index(capture_dir)
    records = [r for r in records if r.get("sample_id") != sample_id]
    records.append(record)
    write_index(capture_dir, records)
    return record


def load_capture_sample(capture_dir: Path, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """加载单条采集样本：rgb / depth / intrinsics / pose（无 GT aabb）。

    返回 None 表示文件缺失。
    """
    import cv2
    from PIL import Image

    sample_dir = capture_dir / record["sample_dir"]
    rgb_path = sample_dir / "rgb.jpg"
    depth_path = sample_dir / "depth.png"
    intr_path = sample_dir / "intrinsics.json"
    pose_path = sample_dir / "pose.json"

    if not (rgb_path.exists() and depth_path.exists() and intr_path.exists()):
        return None

    rgb_img = Image.open(rgb_path).convert("RGB")
    depth_map = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    with open(intr_path, "r", encoding="utf-8") as f:
        intrinsics = json.load(f)

    pose: Optional[Dict[str, Any]] = None
    pose_matrix = np.eye(4, dtype=np.float64)
    if pose_path.exists():
        with open(pose_path, "r", encoding="utf-8") as f:
            pose = json.load(f)
        pose_matrix = pose_to_matrix(pose)

    return {
        "sample_id": record.get("sample_id"),
        "rgb_img": rgb_img,
        "depth_map": depth_map,
        "intrinsics": intrinsics,
        "pose": pose,
        "pose_matrix": pose_matrix,
        "img_width": rgb_img.size[0],
        "img_height": rgb_img.size[1],
    }
