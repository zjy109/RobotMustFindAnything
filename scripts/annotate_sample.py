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

# VLM 预分割客户端（可选，仅 --use-vlm 时启用）
try:
    from vlm_seg_client import VLMSegClient  # noqa: E402
except Exception as _vlm_import_exc:  # pragma: no cover - 仅在缺 zmq 等依赖时触发
    VLMSegClient = None  # type: ignore[assignment]
    _VLM_IMPORT_ERROR: Optional[str] = str(_vlm_import_exc)
else:
    _VLM_IMPORT_ERROR = None


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
# VLM 预标：slug → 模型 prompt 解析规则
# --------------------------------------------------------------------------- #
# RynnBrain 支持中文输入，所以默认直接用 classes.json 里的 name_zh
# （比如 "锅铲"、"水壶"）作为 prompt，不再维护英文映射表。
# 命中优先级：
#     1. --vlm-prompt-map JSON 里显式给出的 slug → prompt
#     2. classes.json 中该 slug 的 name_zh
#     3. slug 本身（最后兜底，例如 classes.json 缺失时）
DEFAULT_VLM_HOST = "127.0.0.1"
DEFAULT_VLM_PORT = 5555
DEFAULT_VLM_TIMEOUT_MS = 90_000


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
    vlm_client: Optional["VLMSegClient"] = None,
    prompt_resolver: Optional[Any] = None,
) -> str:
    """
    返回决策结果："accept" / "discard" / "skip" / "quit" / "redo_failed"。

    若传入 vlm_client（即 --use-vlm 启用且初始化成功），则在让人手画前
    先尝试一次模型预标；标注员可以选择直接接受或转入手画。
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

    # ------------------------------------------------------------------ #
    # 阶段 0：模型预标（可选）
    # ------------------------------------------------------------------ #
    pred_mask: Optional[np.ndarray] = None
    pred_meta: Optional[Dict[str, Any]] = None
    if vlm_client is not None and prompt_resolver is not None:
        prompt = prompt_resolver(class_slug)
        print(f"[annotate] VLM 预标中... prompt='{prompt}'")
        try:
            pred = vlm_client.predict(rgb, prompt, anchor_id=sample_id)
        except Exception as exc:
            print(f"[annotate] VLM 预标异常（已忽略，转人工）: {exc}")
            pred = None

        if pred is not None and pred.get("mask") is not None:
            mask_pred_raw = pred["mask"]
            n_fg = int(mask_pred_raw.sum())
            if n_fg < MIN_MASK_PIXELS:
                print(
                    f"[annotate] VLM 预标 mask 太小 ({n_fg} < {MIN_MASK_PIXELS})，丢弃"
                )
            else:
                pred_mask = mask_pred_raw
                pred_meta = {
                    "host": f"{vlm_client.host}:{vlm_client.port}",
                    "prompt": prompt,
                    "bbox_pixel": pred.get("bbox_pixel"),
                    "fg_pixels": n_fg,
                }

    while True:
        # ------------------------------------------------------------------ #
        # 阶段 1：决定 mask 来源 —— 接受预标 / 进入手画
        # ------------------------------------------------------------------ #
        used_vlm_mask = False
        if pred_mask is not None:
            choice = preview_vlm_prediction(rgb, pred_mask, pred_meta, sample_id, class_slug)
            if choice == "accept":
                mask_bool = pred_mask
                used_vlm_mask = True
            elif choice == "manual":
                # 转人工：丢掉 pred_mask，让标注员重画一次（之后样本里该决策也不再尝试预标）
                pred_mask = None
                continue  # 回到 while 顶端，下一轮走人工分支
            elif choice == "skip":
                return "skip"
            elif choice == "discard":
                return _post_decision_prompt_returning_known("d", sample_dir, sample_id)
            else:  # quit
                return "quit"
        else:
            annot = PolygonAnnotator(rgb, depth_vis)
            header = f"sample={sample_id}  class={class_slug}"
            mask_bool_or_none = annot.run(header)

            if mask_bool_or_none is None:
                # 用户在画 mask 时按 q
                return _post_decision_prompt(sample_dir, sample_id)
            mask_bool = mask_bool_or_none

        if int(mask_bool.sum()) < MIN_MASK_PIXELS:
            print(f"[annotate] mask 太小（{int(mask_bool.sum())} px < {MIN_MASK_PIXELS}），重画")
            if used_vlm_mask:
                pred_mask = None  # 别再用预标了
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
                "viz_aabb_3d.png",
                "sample.json",
            ):
                fp = sample_dir / fname
                if fp.exists():
                    fp.unlink()
            # redo 一律转人工：标注员看完产物还按 n，说明预标 mask 也不行
            pred_mask = None
            continue

        if decision == "accept":
            # 写 sample.json（含 annotator 信息 + checksums + 标注方法）
            sample_json = build_sample_json(
                sample_dir, sample_id, class_slug, annotator_id,
                classes_doc, capture_meta, denoise_info,
                used_vlm_mask=used_vlm_mask,
                vlm_meta=pred_meta if used_vlm_mask else None,
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
    used_vlm_mask: bool = False,
    vlm_meta: Optional[Dict[str, Any]] = None,
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

    annotation_method = "vlm_predicted_accepted" if used_vlm_mask else "manual_polygon"
    tool_name = "vlm_rynnbrain_sam2" if used_vlm_mask else "polygon_floodfill"

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
            "method": annotation_method,
            "tool": tool_name,
            "annotated_at": now_iso(),
            "denoise": denoise_info,
            "vlm": vlm_meta,  # None 或 {host, prompt, bbox_pixel, fg_pixels}
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


def _post_decision_prompt_returning_known(
    decision: str, sample_dir: Path, sample_id: str
) -> str:
    """
    `process_one_sample` 内部用：把已经拿到的 decision 字符（'d'/'s'/'q'）
    走与 `_post_decision_prompt` 相同的语义映射。当 decision == 'd' 时还会
    走 main() 里的 prompt_discard_reason，所以这里只需返回 action 字符串。
    """
    return _decision_to_action(decision, sample_dir, sample_id)


def preview_vlm_prediction(
    rgb_bgr: np.ndarray,
    pred_mask: np.ndarray,
    pred_meta: Optional[Dict[str, Any]],
    sample_id: str,
    class_slug: str,
) -> str:
    """
    在 OpenCV 窗口里预览 VLM 预标 mask（红色叠加 + 绿色 bbox），
    标注员按键决定下一步。

    返回字符串：
        "accept"   接受预标 mask，跳过手画
        "manual"   预标不行，转人工手画
        "skip"     跳过当前样本（留 pending）
        "discard"  删除当前样本（移到 discarded/）
        "quit"     退出标注程序
    """
    overlay = rgb_bgr.copy()
    # 半透明红色覆盖
    if pred_mask.any():
        red = np.zeros_like(overlay)
        red[..., 2] = 255  # BGR 红
        overlay[pred_mask] = cv2.addWeighted(
            overlay[pred_mask], 0.5, red[pred_mask], 0.5, 0
        )

    # bbox（mask 自身推出来一遍，比 server 给的 bbox 更准——SAM2 的 mask 边界
    # 才是最终用的）
    ys, xs = np.where(pred_mask)
    if len(xs):
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    fg = int(pred_mask.sum())
    fg_pct = fg / pred_mask.size * 100.0
    prompt = pred_meta.get("prompt") if pred_meta else "?"

    status_lines = [
        f"sample={sample_id}  class={class_slug}  prompt='{prompt}'",
        f"VLM PREDICTION  fg={fg}px ({fg_pct:.1f}%)",
        "y accept | n manual draw | d discard | s skip | q quit",
    ]
    for i, line in enumerate(status_lines):
        cv2.putText(
            overlay, line, (10, 24 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )

    cv2.imshow(PolygonAnnotator.WIN, overlay)
    decision = _wait_yes_no_etc(["y", "n", "d", "s", "q"])

    if decision == "y":
        return "accept"
    if decision == "n":
        return "manual"
    if decision == "d":
        return "discard"
    if decision == "s":
        return "skip"
    return "quit"


def preview_and_decide(sample_dir: Path, sample_id: str, class_slug: str) -> str:
    """
    展示 viz_mask + viz_aabb（上排）+ viz_aabb_3d（下排），等待标注员决定。
    返回 "accept" / "redo" / "discard" / "skip" / "quit"。

    布局：
        +----------------+----------------+
        |   viz_mask     |   viz_aabb     |   <- 上排，2D 视角
        +----------------+----------------+
        |        viz_aabb_3d (4 子图)     |   <- 下排，3D 多视角
        +---------------------------------+

    若 viz_aabb_3d.png 不存在（matplotlib 缺失等），则自动退回到仅显示上排。
    """
    viz_mask = cv2.imread(str(sample_dir / "viz_mask.png"))
    viz_aabb = (
        cv2.imread(str(sample_dir / "viz_aabb.png"))
        if (sample_dir / "viz_aabb.png").exists()
        else None
    )
    viz_aabb_3d = (
        cv2.imread(str(sample_dir / "viz_aabb_3d.png"))
        if (sample_dir / "viz_aabb_3d.png").exists()
        else None
    )
    if viz_mask is None:
        print("[annotate] 无 viz_mask.png，直接返回 redo")
        return "redo"

    # ===== 上排：2D 视角拼图 =====
    if viz_aabb is None:
        top_row = viz_mask
    else:
        top_row = np.hstack(
            [viz_mask, _resize_to(viz_aabb, viz_mask.shape[:2])]
        )

    # ===== 下排：3D 多视角图（缩放到上排同宽） =====
    if viz_aabb_3d is not None:
        top_w = top_row.shape[1]
        scale = top_w / viz_aabb_3d.shape[1]
        new_h = max(1, int(round(viz_aabb_3d.shape[0] * scale)))
        bottom_row = cv2.resize(
            viz_aabb_3d, (top_w, new_h), interpolation=cv2.INTER_AREA
        )
        # 整体面板若超过屏幕高度上限，等比缩小
        panel = np.vstack([top_row, bottom_row])
        max_h = 950  # 给状态栏与窗口装饰留出空间
        if panel.shape[0] > max_h:
            s = max_h / panel.shape[0]
            panel = cv2.resize(
                panel,
                (int(round(panel.shape[1] * s)), max_h),
                interpolation=cv2.INTER_AREA,
            )
    else:
        panel = top_row

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

    # ----- VLM 预标（可选） -----
    parser.add_argument(
        "--use-vlm", action="store_true",
        help="启用 RynnBrain+SAM2 服务做预标，标注员只需 y/n 决策；"
             "失败/拒绝时自动 fallback 到手画。",
    )
    parser.add_argument("--vlm-host", type=str, default=DEFAULT_VLM_HOST,
                        help="VLM ZMQ server 地址")
    parser.add_argument("--vlm-port", type=int, default=DEFAULT_VLM_PORT,
                        help="VLM ZMQ server 端口")
    parser.add_argument("--vlm-timeout-ms", type=int, default=DEFAULT_VLM_TIMEOUT_MS,
                        help="单次 predict 超时时间（毫秒）")
    parser.add_argument(
        "--vlm-prompt-map", type=str, default=None,
        help="可选：JSON 文件路径，覆盖每个 slug 默认使用的 name_zh 作为 prompt。"
             '例如 \'{"shuihu": "不锈钢保温水壶", "guochan": "厨房铲子"}\'',
    )

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

    # ------------------------------------------------------------------ #
    # 初始化 VLM client（如果启用）
    # ------------------------------------------------------------------ #
    vlm_client = None
    prompt_resolver = None
    if args.use_vlm:
        if VLMSegClient is None:
            print(
                f"[annotate] --use-vlm 启用但 vlm_seg_client 不可用 "
                f"({_VLM_IMPORT_ERROR})，将 fallback 到纯人工标注。"
            )
        else:
            # ----- 构建 slug → prompt 解析器 -----
            # 默认：classes.json 里的 name_zh（RynnBrain 直接支持中文）
            prompt_map: Dict[str, str] = {}
            for c in classes_doc.get("classes", []):
                slug = c.get("name")
                if not slug:
                    continue
                name_zh = c.get("name_zh") or slug
                prompt_map[slug] = str(name_zh)

            # 用户自定义 JSON 覆盖（最高优先级）
            if args.vlm_prompt_map:
                try:
                    user_map = load_json(Path(args.vlm_prompt_map).expanduser())
                    if isinstance(user_map, dict):
                        prompt_map.update({str(k): str(v) for k, v in user_map.items()})
                        print(f"[annotate] 已加载自定义 prompt 映射: {args.vlm_prompt_map}")
                    else:
                        print(f"[annotate] --vlm-prompt-map 不是 JSON object，已忽略")
                except Exception as exc:
                    print(f"[annotate] 读取 --vlm-prompt-map 失败: {exc}")

            def prompt_resolver(slug: str, _m: Dict[str, str] = prompt_map) -> str:
                # 都没有时回退 slug 本身（极端兜底）
                return _m.get(slug, slug)

            try:
                vlm_client = VLMSegClient(
                    host=args.vlm_host,
                    port=args.vlm_port,
                    timeout_ms=args.vlm_timeout_ms,
                )
                print(
                    f"[annotate] ✓ VLM client connected → "
                    f"tcp://{args.vlm_host}:{args.vlm_port} "
                    f"(timeout={args.vlm_timeout_ms}ms)"
                )
                print(f"[annotate]   slug→prompt: {prompt_map}")
            except Exception as exc:
                print(
                    f"[annotate] VLM client 初始化失败 ({exc})，"
                    "将 fallback 到纯人工标注。"
                )
                vlm_client = None
                prompt_resolver = None

    counters = {"accept": 0, "discard": 0, "skip": 0, "redo_failed": 0,
                "vlm_accepted": 0}
    rc = 0
    try:
        for sdir in samples:
            decision = process_one_sample(
                sdir, args.annotator, classes_doc, args,
                vlm_client=vlm_client, prompt_resolver=prompt_resolver,
            )
            if decision == "accept":
                target = move_sample_to(raw_root, sdir, "annotated")
                counters["accept"] += 1
                # 看一眼 sample.json 的 method 字段，便于 session 末尾打统计
                try:
                    sj = load_json(target / "sample.json")
                    if sj and sj.get("annotation", {}).get("method") == "vlm_predicted_accepted":
                        counters["vlm_accepted"] += 1
                except Exception:
                    pass
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
        if vlm_client is not None:
            try:
                vlm_client.close()
            except Exception:
                pass
        cv2.destroyAllWindows()

    print("\n=== annotation session summary ===")
    for k, v in counters.items():
        print(f"  {k:<14s}: {v}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
