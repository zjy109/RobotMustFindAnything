"""几何与点云工具，无 PyTorch / 大模型依赖。

包含：
- compute_2d_iou / compute_3d_iou：xyxy / xyzxyz 格式的 IoU
- denormalize_bbox：[0,1000] → 像素坐标
- denoise_pointcloud：Open3D SOR 去离群点（缺 open3d 时直通）
- mask_to_3d_aabb：mask + depth + 内参 → 世界系 AABB（min,max 拼成 6 维）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# bbox 工具
# ---------------------------------------------------------------------------

def denormalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    """[0,1000] 归一化坐标 → 像素整数坐标 (xyxy)。"""
    x1, y1, x2, y2 = bbox
    return [
        round(x1 / 1000 * width),
        round(y1 / 1000 * height),
        round(x2 / 1000 * width),
        round(y2 / 1000 * height),
    ]


def compute_2d_iou(box1: Optional[List[float]], box2: Optional[List[float]]) -> float:
    """两个 xyxy 矩形的 IoU。任一为空 / 无面积返回 0。"""
    if not box1 or not box2:
        return 0.0
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def compute_3d_iou(box1: Optional[List[float]], box2: Optional[List[float]]) -> float:
    """两个 xyzxyz AABB（[xmin,ymin,zmin, xmax,ymax,zmax]）的 IoU。"""
    if not box1 or not box2:
        return 0.0
    x_l = max(box1[0], box2[0])
    y_t = max(box1[1], box2[1])
    z_f = max(box1[2], box2[2])
    x_r = min(box1[3], box2[3])
    y_b = min(box1[4], box2[4])
    z_back = min(box1[5], box2[5])
    if x_r < x_l or y_b < y_t or z_back < z_f:
        return 0.0
    inter = (x_r - x_l) * (y_b - y_t) * (z_back - z_f)
    vol1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    vol2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    union = vol1 + vol2 - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 点云去噪 + 反投影
# ---------------------------------------------------------------------------

def denoise_pointcloud(
    points_cam: np.ndarray, sor_nb: int, sor_std: float
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """对相机系点云做 Statistical Outlier Removal。

    Open3D 不可用时直通返回（不报错），方便没装 open3d 的临时环境。
    """
    info: Dict[str, Any] = {
        "input_points": int(points_cam.shape[0]),
        "params": {"sor_nb": sor_nb, "sor_std": sor_std},
    }
    try:
        import open3d as o3d  # noqa: WPS433  (lazy import)
    except ImportError:
        info["method"] = "numpy_fallback (no-op)"
        info["fallback_warning"] = "open3d 未安装，跳过 SOR 去噪。"
        return points_cam.astype(np.float32), info

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_cam.astype(np.float64))
    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=int(sor_nb), std_ratio=float(sor_std),
    )
    cleaned = np.asarray(pcd_clean.points, dtype=np.float32)
    info["method"] = "open3d_sor"
    info["after_sor"] = int(cleaned.shape[0])
    return cleaned, info


def mask_to_3d_aabb(
    depth_map: np.ndarray,
    mask: np.ndarray,
    intrinsics: Dict[str, float],
    sor_nb: int,
    sor_std: float,
) -> Optional[List[float]]:
    """根据 mask + depth + 内参，得到相机系 6 维 AABB；空返回 None。

    返回值：[xmin, ymin, zmin, xmax, ymax, zmax]
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    depth_scale = float(intrinsics.get("depth_scale", 0.001))

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None

    depths = depth_map[ys, xs].astype(np.float32) * depth_scale
    valid = depths > 0
    xs, ys, depths = xs[valid], ys[valid], depths[valid]
    if len(depths) == 0:
        return None

    X = (xs - cx) * depths / fx
    Y = (ys - cy) * depths / fy
    Z = depths
    points_3d = np.stack((X, Y, Z), axis=-1)

    cleaned, _info = denoise_pointcloud(points_3d, sor_nb, sor_std)
    if cleaned.shape[0] == 0:
        return None

    mn = cleaned.min(axis=0)
    mx = cleaned.max(axis=0)
    return [float(mn[0]), float(mn[1]), float(mn[2]),
            float(mx[0]), float(mx[1]), float(mx[2])]
