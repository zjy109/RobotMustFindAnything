"""
RoFA-SemEval 阶段 2：标注脚本

标注员视角：从 raw_capture/pending/ 取出未标注样本 → 鼠标画多边形 → 空格闭合填充
得到 mask → 自动反投影/去噪/AABB → 决策 (y/n/d/s/q)。

决策键：
    y  接受 → 整目录从 pending/ 移到 annotated/
    n  重做 mask
    d  删除 → 整目录从 pending/ 移到 discarded/，写 discard_reason.txt
    s  跳过 → 留在 pending/
    q  退出标注

依赖：
    pip install opencv-python "numpy<2.0"
    可选：pip install open3d   # 强烈建议；没有则 fallback 到纯 numpy 去噪
            pip install matplotlib  # 用于生成 viz_aabb.png 的 3D 立体图

Notes：
- 反投影使用 rofa.roimap.search_engine.SearchEngine 的纯算法部分，
  不会启动 ZMQ。
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# 允许 `python scripts/annotate_sample.py` 直接运行
THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from _dataset_common import (  # noqa: E402
    ensure_dir,
    load_classes,
    load_json,
    move_dir,
    now_iso,
    save_json,
    sha1_of_file,
)
from rofa.roimap.search_engine import SearchEngine  # noqa: E402


# --------------------------------------------------------------------------- #
# 默认参数
# --------------------------------------------------------------------------- #
DEFAULT_RAW_ROOT = Path(__file__).resolve().parents[1] / "RoFA-SemEval" / "raw_capture"
DEFAULT_DEPTH_MIN_M = 0.2
DEFAULT_DEPTH_MAX_M = 5.0
DEFAULT_SOR_NB = 30
DEFAULT_SOR_STD = 2.0
DEFAULT_DBSCAN_EPS = 0.03
DEFAULT_DBSCAN_MIN_POINTS = 50
MIN_MASK_PIXELS = 200
MIN_POINTS_AFTER_DENOISE = 200


# --------------------------------------------------------------------------- #
# 复用 SearchEngine 的反投影与可视化（不启动 ZMQ）
# --------------------------------------------------------------------------- #

def make_search_engine_stub(intrinsics: Dict[str, float], depth_scale: float) -> SearchEngine:
    """
    创建一个仅用于调用 SearchEngine 内部算法（_mask_to_world_points /
    _project_world_points_to_image / _save_aabb_2d_overlay / _save_aabb_3d_plot
    / _compute_aabb / _aabb_corners）的 stub，不会调用 __init__、不会启动 socket。
    """
    engine = object.__new__(SearchEngine)
    engine.camera_intrinsics = {
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "cx": float(intrinsics["cx"]),
        "cy": float(intrinsics["cy"]),
    }
    engine.depth_scale = float(depth_scale)
    return engine


# --------------------------------------------------------------------------- #
# 点云去噪
# --------------------------------------------------------------------------- #

def denoise_pointcloud(
    points_cam: np.ndarray,
    sor_nb: int = DEFAULT_SOR_NB,
    sor_std: float = DEFAULT_SOR_STD,
    dbscan_eps: float = DEFAULT_DBSCAN_EPS,
    dbscan_min_points: int = DEFAULT_DBSCAN_MIN_POINTS,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    返回 (cleaned_points, info_dict)。优先使用 open3d；不可用则 fallback。
    """
    info: Dict[str, Any] = {
        "input_points": int(points_cam.shape[0]),
        "method": "",
        "params": {
            "sor_nb": sor_nb,
            "sor_std": sor_std,
            "dbscan_eps": dbscan_eps,
            "dbscan_min_points": dbscan_min_points,
        },
    }

    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"[annotate] open3d 不可用（{exc}），fallback 到纯 numpy 去噪")
        return _numpy_fallback_denoise(points_cam, info)

    info["method"] = "open3d_sor + dbscan"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_cam.astype(np.float64))

    # 1) 统计离群点
    pcd_clean, sor_idx = pcd.remove_statistical_outlier(
        nb_neighbors=int(sor_nb), std_ratio=float(sor_std)
    )
    info["after_sor"] = int(np.asarray(pcd_clean.points).shape[0])

    # 2) DBSCAN 选最大簇
    labels = np.array(
        pcd_clean.cluster_dbscan(
            eps=float(dbscan_eps), min_points=int(dbscan_min_points), print_progress=False
        )
    )
    if labels.size == 0:
        info["clusters"] = 0
        return np.asarray(pcd_clean.points, dtype=np.float32), info

    valid_labels = labels[labels >= 0]
    if valid_labels.size == 0:
        # 没有有效簇 → 全部点都是噪声，回退到 SOR 结果
        info["clusters"] = 0
        info["dbscan_fallback"] = "no_valid_cluster"
        return np.asarray(pcd_clean.points, dtype=np.float32), info

    counts = np.bincount(valid_labels)
    largest = int(np.argmax(counts))
    keep_mask = labels == largest
    cleaned = np.asarray(pcd_clean.points)[keep_mask]
    info["clusters"] = int(counts.size)
    info["largest_cluster"] = int(counts[largest])
    return cleaned.astype(np.float32), info


def _numpy_fallback_denoise(
    points_cam: np.ndarray, info: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """没有 open3d 时的简化兜底：仅做轴向 3 西格玛过滤。"""
    info["method"] = "numpy_3sigma"
    if points_cam.shape[0] == 0:
        return points_cam, info
    median = np.median(points_cam, axis=0)
    mad = np.median(np.abs(points_cam - median), axis=0) + 1e-6
    mask = np.all(np.abs(points_cam - median) <= 5.0 * mad, axis=1)
    cleaned = points_cam[mask]
    info["after_filter"] = int(cleaned.shape[0])
    return cleaned.astype(np.float32), info


def filter_depth_range(
    points_cam: np.ndarray, dmin: float, dmax: float
) -> np.ndarray:
    """相机系下，z 即深度（米）。"""
    if points_cam.shape[0] == 0:
        return points_cam
    z = points_cam[:, 2]
    mask = (z >= dmin) & (z <= dmax)
    return points_cam[mask]


# --------------------------------------------------------------------------- #
# PLY 写出（不依赖 open3d，避免硬绑定）
# --------------------------------------------------------------------------- #

def write_ply_xyz(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = int(points.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(points.tobytes())


# --------------------------------------------------------------------------- #
# 多边形标注 UI
# --------------------------------------------------------------------------- #

class PolygonAnnotator:
    """
    OpenCV 鼠标交互：左键加点 → 空格/右键闭合 → fillPoly 得 mask。
    支持 z 撤销最近一个点、r 清空多边形。
    """

    WIN = "RoFA Annotate"

    def __init__(self, rgb_bgr: np.ndarray, depth_vis: np.ndarray):
        self.rgb = rgb_bgr
        self.depth_vis = depth_vis
        self.h, self.w = rgb_bgr.shape[:2]
        self.points: List[Tuple[int, int]] = []
        self.cursor: Optional[Tuple[int, int]] = None

    def _on_mouse(self, event, x, y, flags, _userdata):
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((int(x), int(y)))
        elif event == cv2.EVENT_MOUSEMOVE:
            self.cursor = (int(x), int(y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键 = 闭合（与空格等价）
            self.points.append(("__close__",))  # 哨兵

    def _render(self, status: str) -> np.ndarray:
        canvas = self.rgb.copy()

        # 绘制多边形顶点 + 折线
        pts_xy = [p for p in self.points if isinstance(p, tuple) and len(p) == 2]
        for i, (x, y) in enumerate(pts_xy):
            cv2.circle(canvas, (x, y), 4, (0, 255, 255), -1, cv2.LINE_AA)
            if i > 0:
                cv2.line(
                    canvas,
                    pts_xy[i - 1],
                    (x, y),
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        # 当前光标到最后一个点的预览线
        if pts_xy and self.cursor is not None:
            cv2.line(
                canvas,
                pts_xy[-1],
                self.cursor,
                (0, 200, 200),
                1,
                cv2.LINE_AA,
            )

        # 状态栏
        for i, line in enumerate(status.split("\n")):
            cv2.putText(
                canvas,
                line,
                (10, 24 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return np.hstack([canvas, self.depth_vis])

    def run(self, header: str) -> Optional[np.ndarray]:
        """
        阻塞运行直到用户按下空格/右键闭合多边形。
        返回 (H, W) 的 bool mask，或 None 表示放弃当前样本（按 r 全清后再按 q）。
        """
        cv2.namedWindow(self.WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WIN, self._on_mouse)
        try:
            while True:
                # 处理右键哨兵
                if self.points and isinstance(self.points[-1], tuple) and self.points[-1] == ("__close__",):
                    self.points.pop()
                    if len(self.points) >= 3:
                        return self._build_mask()
                    else:
                        # 顶点太少，忽略
                        pass

                status = (
                    f"{header}\n"
                    f"polygon points: {len([p for p in self.points if len(p)==2])}\n"
                    "L-click add | R-click/SPACE close | z undo | r reset | q skip"
                )
                cv2.imshow(self.WIN, self._render(status))
                key = cv2.waitKey(20) & 0xFF
                if key == 0xFF:
                    continue
                if key == ord(" "):
                    if len(self.points) >= 3:
                        return self._build_mask()
                elif key == ord("z"):
                    pts_xy = [p for p in self.points if len(p) == 2]
                    if pts_xy:
                        # 弹出最后一个真正的顶点
                        for i in range(len(self.points) - 1, -1, -1):
                            if len(self.points[i]) == 2:
                                self.points.pop(i)
                                break
                elif key == ord("r"):
                    self.points = []
                elif key == ord("q"):
                    return None
        finally:
            cv2.setMouseCallback(self.WIN, lambda *args, **kwargs: None)

    def _build_mask(self) -> np.ndarray:
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        pts = np.array(
            [p for p in self.points if len(p) == 2], dtype=np.int32
        ).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
        return mask > 0


# --------------------------------------------------------------------------- #
# 单样本处理流水线
# --------------------------------------------------------------------------- #

def depth_to_vis(depth_u16: np.ndarray) -> np.ndarray:
    d = depth_u16.astype(np.float32)
    pos = d[d > 0]
    vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
    scaled = np.clip(d / max(vmax, 1.0) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def list_pending_samples(raw_root: Path, only_class: Optional[str]) -> List[Path]:
    pending = raw_root / "pending"
    if not pending.exists():
        return []
    classes = (
        [pending / only_class] if only_class else sorted(p for p in pending.iterdir() if p.is_dir())
    )
    out: List[Path] = []
    for cdir in classes:
        if not cdir.exists():
            continue
        for sdir in sorted(cdir.iterdir()):
            if not sdir.is_dir():
                continue
            # 必须含 rgb / depth
            if not (sdir / "rgb.jpg").exists() or not (sdir / "depth.png").exists():
                continue
            out.append(sdir)
    return out


def load_sample(sample_dir: Path) -> Optional[Dict[str, Any]]:
    rgb_path = sample_dir / "rgb.jpg"
    depth_path = sample_dir / "depth.png"
    intr_path = sample_dir / "intrinsics.json"
    pose_path = sample_dir / "pose.txt"
    cap_meta_path = sample_dir / "capture_meta.json"

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        print(f"[annotate] 无法读取 rgb: {rgb_path}")
        return None
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        print(f"[annotate] 无法读取 depth 或类型不为 uint16: {depth_path}")
        return None
    intrinsics = load_json(intr_path)
    if intrinsics is None:
        print(f"[annotate] 缺少 intrinsics.json: {intr_path}")
        return None
    try:
        pose = np.loadtxt(pose_path, dtype=np.float32)
        if pose.shape == (16,):
            pose = pose.reshape(4, 4)
        if pose.shape != (4, 4):
            raise ValueError(f"pose shape={pose.shape}")
    except Exception as exc:
        print(f"[annotate] 无法读取 pose: {pose_path} ({exc})")
        return None
    capture_meta = load_json(cap_meta_path, default={}) or {}

    return {
        "rgb": rgb,
        "depth": depth,
        "intrinsics": intrinsics,
        "pose": pose,
        "capture_meta": capture_meta,
    }


def process_one_sample(
    sample_dir: Path,
    annotator_id: str,
    classes_doc: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    """
    返回决策结果："accept" / "discard" / "skip" / "quit" / "redo_failed"。
    """
    sample_id = sample_dir.name
    class_slug = sample_dir.parent.name
    print(f"\n=== [{sample_id}] class={class_slug} ===")

    bundle = load_sample(sample_dir)
    if bundle is None:
        print(f"[annotate] 跳过 {sample_id}（数据缺失）")
        return "skip"

    rgb = bundle["rgb"]
    depth = bundle["depth"]
    intrinsics = bundle["intrinsics"]
    pose = bundle["pose"]
    capture_meta = bundle["capture_meta"]

    depth_vis = depth_to_vis(depth)
    engine = make_search_engine_stub(intrinsics, intrinsics.get("depth_scale", 0.001))

    while True:
        annot = PolygonAnnotator(rgb, depth_vis)
        header = f"sample={sample_id}  class={class_slug}"
        mask_bool = annot.run(header)

        if mask_bool is None:
            # 用户在画 mask 时按 q
            return _post_decision_prompt(sample_dir, sample_id)

        if int(mask_bool.sum()) < MIN_MASK_PIXELS:
            print(f"[annotate] mask 太小（{int(mask_bool.sum())} px < {MIN_MASK_PIXELS}），重画")
            continue

        # 反投影 + 去噪 + AABB
        try:
            points_cam = engine._mask_to_world_points(mask_bool, depth, pose)  # noqa: SLF001
        except Exception as exc:
            print(f"[annotate] 反投影失败: {exc}")
            print("  按 r 重画 / d 删除 / s 跳过 / q 退出")
            decision = _wait_yes_no_etc(["r", "d", "s", "q"])
            if decision == "r":
                continue
            return _decision_to_action(decision, sample_dir, sample_id)

        points_cam = filter_depth_range(
            points_cam, args.depth_min_m, args.depth_max_m
        )
        if points_cam.shape[0] < MIN_POINTS_AFTER_DENOISE:
            print(
                f"[annotate] 深度范围过滤后点云太少（{points_cam.shape[0]}），"
                "可能 mask 区域无有效深度。"
            )
            print("  按 r 重画 / d 删除 / s 跳过 / q 退出")
            decision = _wait_yes_no_etc(["r", "d", "s", "q"])
            if decision == "r":
                continue
            return _decision_to_action(decision, sample_dir, sample_id)

        cleaned, denoise_info = denoise_pointcloud(
            points_cam,
            sor_nb=args.sor_nb,
            sor_std=args.sor_std,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_points=args.dbscan_min_points,
        )
        if cleaned.shape[0] < MIN_POINTS_AFTER_DENOISE:
            print(
                f"[annotate] 去噪后点云太少（{cleaned.shape[0]} < {MIN_POINTS_AFTER_DENOISE}）"
            )
            print("  按 r 重画 / d 删除 / s 跳过 / q 退出")
            decision = _wait_yes_no_etc(["r", "d", "s", "q"])
            if decision == "r":
                continue
            return _decision_to_action(decision, sample_dir, sample_id)

        aabb = SearchEngine._compute_aabb(cleaned)  # noqa: SLF001
        if any(e <= 0 for e in aabb["extent"]):
            print(f"[annotate] AABB extent 退化为 0 维: {aabb['extent']}")
            print("  按 r 重画 / d 删除 / s 跳过 / q 退出")
            decision = _wait_yes_no_etc(["r", "d", "s", "q"])
            if decision == "r":
                continue
            return _decision_to_action(decision, sample_dir, sample_id)

        # 写出标注产物（写在 sample_dir 内，迁移时整目录搬走）
        write_annotation_products(
            sample_dir,
            rgb,
            mask_bool,
            cleaned,
            aabb,
            engine,
            pose,
            denoise_info,
        )

        # 展示结果给标注员判断
        decision = preview_and_decide(sample_dir, sample_id, class_slug)
        if decision == "redo":
            # 删除已写产物，重画
            for fname in (
                "mask.png",
                "points.ply",
                "aabb.json",
                "viz_mask.png",
                "viz_aabb.png",
                "sample.json",
            ):
                fp = sample_dir / fname
                if fp.exists():
                    fp.unlink()
            continue

        if decision == "accept":
            # 写 sample.json（含 annotator 信息 + checksums）
            sample_json = build_sample_json(
                sample_dir, sample_id, class_slug, annotator_id,
                classes_doc, capture_meta, denoise_info,
            )
            save_json(sample_dir / "sample.json", sample_json)
            return "accept"

        return decision  # discard / skip / quit


def write_annotation_products(
    sample_dir: Path,
    rgb_bgr: np.ndarray,
    mask_bool: np.ndarray,
    points_cam: np.ndarray,
    aabb: Dict[str, Any],
    engine: SearchEngine,
    pose: np.ndarray,
    denoise_info: Dict[str, Any],
) -> None:
    # mask.png
    cv2.imwrite(str(sample_dir / "mask.png"), mask_bool.astype(np.uint8) * 255)

    # points.ply
    write_ply_xyz(sample_dir / "points.ply", points_cam)

    # aabb.json
    save_json(sample_dir / "aabb.json", aabb)

    # viz_mask.png：复用 SearchEngine._save_mask_visualization 风格，但写到子目录
    overlay = rgb_bgr.copy()
    overlay[mask_bool] = (
        0.4 * overlay[mask_bool] + 0.6 * np.array([0, 255, 0])
    ).astype(np.uint8)
    ys, xs = np.where(mask_bool)
    if len(xs):
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    cv2.imwrite(str(sample_dir / "viz_mask.png"), overlay)

    # viz_aabb.png：复用 _save_aabb_2d_overlay
    try:
        corners = SearchEngine._aabb_corners(  # noqa: SLF001
            np.asarray(aabb["min"], dtype=np.float32),
            np.asarray(aabb["max"], dtype=np.float32),
        )
        corners_uv, valid, _ = engine._project_world_points_to_image(  # noqa: SLF001
            corners, pose, rgb_bgr.shape
        )
        out_path = SearchEngine._save_aabb_2d_overlay(  # noqa: SLF001
            sample_dir, rgb_bgr, corners_uv, valid, bbox=None
        )
        # 重命名 remote_aabb_overlay.png → viz_aabb.png（数据集里使用更通用的名字）
        if out_path.exists():
            target = sample_dir / "viz_aabb.png"
            if target.exists():
                target.unlink()
            out_path.rename(target)
    except Exception as exc:
        print(f"[annotate] 生成 viz_aabb.png 失败: {exc}")

    # 3D 立体图（matplotlib 可选）
    try:
        path3d = SearchEngine._save_aabb_3d_plot(  # noqa: SLF001
            sample_dir, points_cam, aabb["min"], aabb["max"]
        )
        if path3d is not None and path3d.exists():
            target = sample_dir / "viz_aabb_3d.png"
            if target.exists():
                target.unlink()
            path3d.rename(target)
    except Exception as exc:
        print(f"[annotate] 生成 viz_aabb_3d.png 失败: {exc}")

    # 把过程信息打到 stdout，便于审阅
    print(
        f"[annotate] AABB: min={aabb['min']} max={aabb['max']} "
        f"extent={aabb['extent']} num_points={aabb['num_points']}"
    )
    print(f"[annotate] denoise: {denoise_info}")


def build_sample_json(
    sample_dir: Path,
    sample_id: str,
    class_slug: str,
    annotator_id: str,
    classes_doc: Dict[str, Any],
    capture_meta: Dict[str, Any],
    denoise_info: Dict[str, Any],
) -> Dict[str, Any]:
    class_obj = next(
        (c for c in classes_doc.get("classes", []) if c.get("name") == class_slug),
        None,
    )
    class_id = capture_meta.get("class_id") if class_obj is None else class_obj["id"]
    class_name_zh = (
        capture_meta.get("class_name_zh")
        if class_obj is None
        else class_obj.get("name_zh", "")
    )

    rgb_sha = sha1_of_file(sample_dir / "rgb.jpg")
    depth_sha = sha1_of_file(sample_dir / "depth.png")
    mask_sha = sha1_of_file(sample_dir / "mask.png")

    return {
        "sample_id": sample_id,
        "class_id": class_id,
        "class_name": class_slug,
        "class_name_zh": class_name_zh,
        "rgb_path": "rgb.jpg",
        "depth_path": "depth.png",
        "mask_path": "mask.png",
        "intrinsics_path": "intrinsics.json",
        "pose_path": "pose.txt",
        "points_path": "points.ply",
        "aabb_path": "aabb.json",
        "world_frame": "camera",
        "pose_source": "dummy_identity",
        "queries": {
            "category": class_slug,
            "name_zh": class_name_zh,
            "language": [],  # 标注员可以后续手工填
        },
        "annotation": {
            "annotator": annotator_id,
            "tool": "polygon_floodfill",
            "annotated_at": now_iso(),
            "denoise": denoise_info,
        },
        "capture_meta": capture_meta,
        "checksums": {
            "rgb_sha1": rgb_sha,
            "depth_sha1": depth_sha,
            "mask_sha1": mask_sha,
        },
    }


# --------------------------------------------------------------------------- #
# 决策提示 / 终端等键
# --------------------------------------------------------------------------- #

def _wait_yes_no_etc(allowed: List[str]) -> str:
    """阻塞式读单字符（OpenCV 窗口需聚焦）。"""
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 0xFF:
            continue
        ch = chr(key)
        if ch in allowed:
            return ch


def _post_decision_prompt(sample_dir: Path, sample_id: str) -> str:
    """
    在标注员未画完 mask 就按 q 时调用。提供 d / s / q 三个可选。
    """
    print("\n  画 mask 已放弃。请选择本样本的处理方式：")
    print("    d  删除（移到 discarded/）")
    print("    s  跳过（留在 pending/）")
    print("    q  退出标注程序")
    decision = _wait_yes_no_etc(["d", "s", "q"])
    return _decision_to_action(decision, sample_dir, sample_id)


def _decision_to_action(decision: str, sample_dir: Path, sample_id: str) -> str:
    if decision == "d":
        return "discard"
    if decision == "s":
        return "skip"
    if decision == "q":
        return "quit"
    return "skip"


def preview_and_decide(sample_dir: Path, sample_id: str, class_slug: str) -> str:
    """
    展示 viz_mask + viz_aabb，等待标注员决定。
    返回 "accept" / "redo" / "discard" / "skip" / "quit"。
    """
    viz_mask = cv2.imread(str(sample_dir / "viz_mask.png"))
    viz_aabb = (
        cv2.imread(str(sample_dir / "viz_aabb.png"))
        if (sample_dir / "viz_aabb.png").exists()
        else None
    )
    if viz_mask is None:
        print("[annotate] 无 viz_mask.png，直接返回 redo")
        return "redo"

    panel = viz_mask if viz_aabb is None else np.hstack(
        [viz_mask, _resize_to(viz_aabb, viz_mask.shape[:2])]
    )
    status = (
        f"sample={sample_id}  class={class_slug}\n"
        "y accept | n redo | d discard | s skip | q quit"
    )
    for i, line in enumerate(status.split("\n")):
        cv2.putText(
            panel,
            line,
            (10, 24 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imshow(PolygonAnnotator.WIN, panel)
    decision = _wait_yes_no_etc(["y", "n", "d", "s", "q"])
    if decision == "y":
        return "accept"
    if decision == "n":
        return "redo"
    if decision == "d":
        return "discard"
    if decision == "s":
        return "skip"
    return "quit"


def _resize_to(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# 样本目录迁移
# --------------------------------------------------------------------------- #

def move_sample_to(raw_root: Path, sample_dir: Path, target_subdir: str) -> Path:
    """
    把 sample_dir 从 pending/<class>/<id> 移到 <target_subdir>/<class>/<id>。
    """
    rel = sample_dir.relative_to(raw_root / "pending")  # <class>/<id>
    target = raw_root / target_subdir / rel
    move_dir(sample_dir, target)
    return target


def write_discard_reason(target_dir: Path, reason: str, annotator_id: str) -> None:
    text = (
        f"discarded_at: {now_iso()}\n"
        f"annotator: {annotator_id}\n"
        f"reason: {reason}\n"
    )
    (target_dir / "discard_reason.txt").write_text(text, encoding="utf-8")


def prompt_discard_reason() -> str:
    print("  请输入删除原因（一行；回车结束；空则记为 'unspecified'）：")
    try:
        line = input("  reason> ").strip()
    except EOFError:
        line = ""
    return line or "unspecified"


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RoFA-SemEval 标注脚本（阶段 2）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-root", type=str, default=str(DEFAULT_RAW_ROOT),
        help="raw_capture 根目录"
    )
    parser.add_argument("--annotator", type=str, default="anonymous", help="标注员标识")
    parser.add_argument("--class", dest="only_class", type=str, default=None,
                        help="只标注指定类别（slug，可选）")
    parser.add_argument("--depth-min-m", type=float, default=DEFAULT_DEPTH_MIN_M)
    parser.add_argument("--depth-max-m", type=float, default=DEFAULT_DEPTH_MAX_M)
    parser.add_argument("--sor-nb", type=int, default=DEFAULT_SOR_NB)
    parser.add_argument("--sor-std", type=float, default=DEFAULT_SOR_STD)
    parser.add_argument("--dbscan-eps", type=float, default=DEFAULT_DBSCAN_EPS)
    parser.add_argument("--dbscan-min-points", type=int, default=DEFAULT_DBSCAN_MIN_POINTS)
    args = parser.parse_args()

    raw_root = Path(args.raw_root).expanduser().resolve()
    if not raw_root.exists():
        print(f"[annotate] raw_root 不存在: {raw_root}")
        return 1

    classes_path = raw_root / "classes.json"
    classes_doc = load_classes(classes_path)

    samples = list_pending_samples(raw_root, args.only_class)
    if not samples:
        print(f"[annotate] 没有 pending 样本: {raw_root / 'pending'}")
        return 0
    print(f"[annotate] 共 {len(samples)} 个 pending 样本")

    counters = {"accept": 0, "discard": 0, "skip": 0, "redo_failed": 0}
    rc = 0
    try:
        for sdir in samples:
            decision = process_one_sample(sdir, args.annotator, classes_doc, args)
            if decision == "accept":
                target = move_sample_to(raw_root, sdir, "annotated")
                counters["accept"] += 1
                print(f"[annotate] ✓ accept → {target.relative_to(raw_root)}")
            elif decision == "discard":
                reason = prompt_discard_reason()
                target = move_sample_to(raw_root, sdir, "discarded")
                write_discard_reason(target, reason, args.annotator)
                counters["discard"] += 1
                print(f"[annotate] ✗ discard → {target.relative_to(raw_root)} ({reason})")
            elif decision == "skip":
                counters["skip"] += 1
                print(f"[annotate] · skip {sdir.name}")
            elif decision == "quit":
                print("[annotate] q 退出")
                break
            else:
                counters["redo_failed"] += 1

    except KeyboardInterrupt:
        print("\n[annotate] interrupted")
    except Exception as exc:
        print(f"[annotate][fatal] {exc}")
        traceback.print_exc()
        rc = 2
    finally:
        cv2.destroyAllWindows()

    print("\n=== annotation session summary ===")
    for k, v in counters.items():
        print(f"  {k:<10s}: {v}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
