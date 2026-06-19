#!/usr/bin/env python3
"""环境语义检索 UI（Web 版，单文件）。

使用方式很简单：
    1. SSH 到（有 GPU 的）服务器
    2. 运行：python ui.py --capture-dir ./captures
    3. 浏览器打开终端里打印的网址（默认 http://127.0.0.1:7860）

检索逻辑：
    用户只输入"要查找的物体名称"，**系统自动扫描采集目录下的所有样本**，
    用 RynnBrain 定位 + SAM2 分割 + 深度反投影，把命中的样本以图集形式返回；
    点击任意命中结果即可查看其 2D bbox / 掩码 / 相机系 & 世界系 3D AABB，
    以及（可选）浏览器内的三维点云 + AABB。

依赖：gradio；三维点云预览还需 open3d。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# 让脚本能 import 项目内 src/benchmark
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.capture_io import (  # noqa: E402
    depth_to_points,
    load_capture_sample,
    load_index,
    transform_aabb,
)
from benchmark.config import ENABLE_EXISTENCE_CHECK, SOR_NB, SOR_STD  # noqa: E402
from benchmark.geometry import denormalize_bbox, mask_to_3d_aabb  # noqa: E402
from benchmark.viz import render_overlay  # noqa: E402


# ---------------------------------------------------------------------------
# 模型封装（惰性加载）
# ---------------------------------------------------------------------------

class Pipeline:
    """RynnBrain + SAM2 惰性加载与单帧推理。"""

    def __init__(self, cuda_devices: str = "0"):
        self.cuda_devices = cuda_devices
        self.rynn = None
        self.sam2 = None

    def ensure_loaded(self, log=print) -> None:
        if self.rynn is not None and self.sam2 is not None:
            return
        from benchmark.model_resolver import ensure_rynnbrain, ensure_sam2
        from benchmark.models import (
            RynnBrainDetector, SAM2Segmenter, setup_cuda_devices,
        )

        setup_cuda_devices(self.cuda_devices)
        log("准备模型权重（首次会自动下载到 ./models/）...")
        rynn_path = ensure_rynnbrain()
        sam2_path = ensure_sam2()
        log("加载 RynnBrain ...")
        self.rynn = RynnBrainDetector(
            model_path=str(rynn_path),
            enable_existence_check=ENABLE_EXISTENCE_CHECK,
        )
        log("加载 SAM2 ...")
        self.sam2 = SAM2Segmenter(model_path=str(sam2_path))
        log("模型加载完成。")

    def infer(self, bundle: Dict[str, Any], target: str, log=print) -> Dict[str, Any]:
        """对单个样本检索 target，返回中间结果（found / bbox / mask / aabb）。"""
        self.ensure_loaded(log)

        rgb_img = bundle["rgb_img"]
        depth_map = bundle["depth_map"]
        intrinsics = bundle["intrinsics"]
        W, H = bundle["img_width"], bundle["img_height"]

        bbox_norm = self.rynn.detect(rgb_img, target)
        if bbox_norm is None:
            return {"found": False}

        pred_2d = denormalize_bbox(bbox_norm, W, H)
        mask = self.sam2.segment(rgb_img, pred_2d)

        aabb_cam = mask_to_3d_aabb(depth_map, mask, intrinsics, SOR_NB, SOR_STD)
        aabb_world = transform_aabb(aabb_cam, bundle["pose_matrix"]) if aabb_cam else None

        return {
            "found": True,
            "pred_2d": pred_2d,
            "mask": mask,
            "aabb_cam": aabb_cam,
            "aabb_world": aabb_world,
        }


# ---------------------------------------------------------------------------
# 可视化工具
# ---------------------------------------------------------------------------

def fmt_aabb(name: str, aabb: Optional[List[float]]) -> str:
    if aabb is None:
        return f"{name}: 无（点云为空 / 深度无效）"
    mn, mx = aabb[:3], aabb[3:]
    ext = [mx[i] - mn[i] for i in range(3)]
    return (
        f"{name}:\n"
        f"  min  = [{mn[0]:.3f}, {mn[1]:.3f}, {mn[2]:.3f}] m\n"
        f"  max  = [{mx[0]:.3f}, {mx[1]:.3f}, {mx[2]:.3f}] m\n"
        f"  尺寸 = [{ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f}] m"
    )


def _sample_aabb_edges(aabb: List[float], n_per_edge: int = 40) -> np.ndarray:
    """沿 AABB 的 12 条棱采点，用于在点云里把包围盒画成红色"线"。"""
    mn = np.array(aabb[:3]); mx = np.array(aabb[3:])
    c = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    pts, ts = [], np.linspace(0, 1, n_per_edge)[:, None]
    for a, b in edges:
        pts.append(c[a][None, :] * (1 - ts) + c[b][None, :] * ts)
    return np.concatenate(pts, axis=0)


def export_ply(bundle: Dict[str, Any], mask: np.ndarray,
               aabb_cam: Optional[List[float]]) -> Optional[str]:
    """导出 场景点云 + 目标点云(红) + AABB红框 为 .ply，供浏览器三维预览。"""
    try:
        import open3d as o3d
    except ImportError:
        return None

    rgb = np.array(bundle["rgb_img"])
    depth_map = bundle["depth_map"]
    intrinsics = bundle["intrinsics"]

    scene_pts, scene_col = depth_to_points(depth_map, intrinsics, rgb=rgb, stride=3)
    obj_pts, _ = depth_to_points(depth_map, intrinsics, mask=mask)

    pts_list, col_list = [], []
    if len(scene_pts) > 0:
        pts_list.append(scene_pts)
        col_list.append(scene_col if scene_col is not None
                        else np.full((len(scene_pts), 3), 0.6, np.float32))
    if len(obj_pts) > 0:
        pts_list.append(obj_pts)
        col_list.append(np.tile([1.0, 0.2, 0.2], (len(obj_pts), 1)))
    if aabb_cam is not None:
        edge = _sample_aabb_edges(aabb_cam)
        pts_list.append(edge.astype(np.float32))
        col_list.append(np.tile([1.0, 0.0, 0.0], (len(edge), 1)))

    if not pts_list:
        return None

    P = np.concatenate(pts_list).astype(np.float64)
    C = np.concatenate(col_list).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    out = Path(tempfile.gettempdir()) / f"pcd_{bundle.get('sample_id', 'x')}.ply"
    o3d.io.write_point_cloud(str(out), pcd)
    return str(out)


# ---------------------------------------------------------------------------
# Gradio 应用
# ---------------------------------------------------------------------------

def build_app(capture_dir: Path, cuda_devices: str):
    import gradio as gr

    pipeline = Pipeline(cuda_devices=cuda_devices)
    state: Dict[str, Any] = {"capture_dir": capture_dir, "matches": []}

    def search_all(cdir_str: str, target: str, progress=gr.Progress()):
        """扫描采集目录下所有样本，返回命中目标的样本图集 + 概要。"""
        cdir = Path(cdir_str).expanduser().resolve()
        state["capture_dir"] = cdir
        state["matches"] = []

        target = (target or "").strip()
        if not target:
            return [], "请输入要查找的物体名称。"

        records = load_index(cdir)
        if not records:
            return [], f"目录中没有样本（缺 samples.json）：{cdir}"

        # 先把模型加载好，再逐样本推理（这样进度条只反映扫描进度）
        progress(0, desc="加载模型 ...")
        try:
            pipeline.ensure_loaded(log=print)
        except Exception as exc:  # noqa: BLE001
            return [], f"模型加载失败：{exc}"

        gallery, matches = [], []
        scanned = 0
        for r in progress.tqdm(records, desc=f"检索『{target}』"):
            b = load_capture_sample(cdir, r)
            if b is None:
                continue
            scanned += 1
            try:
                res = pipeline.infer(b, target, log=lambda *_: None)
            except Exception as exc:  # noqa: BLE001
                print(f"[ui] 样本 {r.get('sample_id')} 推理出错: {exc}")
                continue
            if not res.get("found"):
                continue

            overlay = render_overlay(
                np.array(b["rgb_img"]), res["pred_2d"], res["mask"],
                gt_2d_bbox=None, label=target, mask_alpha=0.5,
            )
            overlay_rgb = np.ascontiguousarray(overlay[:, :, ::-1])
            sid = b["sample_id"]
            gallery.append((overlay_rgb, sid))
            matches.append({"sample_id": sid, "bundle": b, "res": res})

        state["matches"] = matches
        summary = (
            f"已扫描 {scanned} 个样本，命中『{target}』的有 **{len(matches)}** 个。\n\n"
            + ("点击下方任意结果查看 3D AABB 与点云。" if matches
               else "没有任何样本包含该物体（RynnBrain 判定不存在 / 无法定位）。")
        )
        return gallery, summary

    def on_select(evt: gr.SelectData):
        """点击图集某个命中结果 -> 显示其 AABB 信息 + 三维点云。"""
        matches = state.get("matches", [])
        if evt.index is None or evt.index >= len(matches):
            return "未选中有效结果。", None
        m = matches[evt.index]
        b, res = m["bundle"], m["res"]
        info = (
            f"样本: {m['sample_id']}\n"
            f"2D bbox(px): {res['pred_2d']}\n\n"
            + fmt_aabb("相机系 3D AABB", res["aabb_cam"]) + "\n\n"
            + fmt_aabb("世界系 3D AABB(已应用随机位姿)", res["aabb_world"])
        )
        ply = export_ply(b, res["mask"], res["aabb_cam"])
        if ply is None:
            info += "\n\n（未安装 open3d，跳过三维点云导出）"
        return info, ply

    with gr.Blocks(title="环境语义检索 UI") as demo:
        gr.Markdown(
            "## 环境语义检索 UI\n"
            "输入要查找的物体，系统会**自动扫描采集目录下的所有样本**并返回命中的结果。"
        )
        with gr.Row():
            cdir_box = gr.Textbox(value=str(capture_dir), label="采集目录", scale=3)
            query_box = gr.Textbox(label="查找物体（支持中文）",
                                   placeholder="例如：水杯 / 键盘 / 椅子", scale=3)
            search_btn = gr.Button("检索全部样本", variant="primary", scale=1)

        summary = gr.Markdown()
        gallery = gr.Gallery(label="命中的样本（绿框=bbox，蓝=掩码）",
                             columns=4, height=420, object_fit="contain",
                             allow_preview=True)

        with gr.Row():
            detail = gr.Textbox(label="所选样本的 3D AABB 信息", lines=10, scale=1)
            model3d = gr.Model3D(label="3D 点云 + AABB（红框）",
                                 clear_color=[0, 0, 0, 1], scale=1)

        search_btn.click(search_all, inputs=[cdir_box, query_box],
                         outputs=[gallery, summary])
        query_box.submit(search_all, inputs=[cdir_box, query_box],
                         outputs=[gallery, summary])
        gallery.select(on_select, inputs=None, outputs=[detail, model3d])

    return demo


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

def _ensure_localhost_no_proxy(host: str) -> None:
    """让本机地址绕过 HTTP 代理，避免 gradio 启动自检请求被代理拦截而 503。"""
    targets = {"localhost", "127.0.0.1", "::1", host}
    for var in ("no_proxy", "NO_PROXY"):
        items = [s.strip() for s in os.environ.get(var, "").split(",") if s.strip()]
        for t in targets:
            if t and t not in items:
                items.append(t)
        os.environ[var] = ",".join(items)


def main() -> int:
    ap = argparse.ArgumentParser(description="环境语义检索 UI（Web 版）")
    ap.add_argument("--capture-dir", type=str, default="./captures",
                    help="sample_rsd4xx.py 的采集输出目录（默认 ./captures）")
    ap.add_argument("--cuda-devices", type=str, default="0",
                    help="CUDA_VISIBLE_DEVICES（默认 0）")
    ap.add_argument("--host", type=str, default="127.0.0.1",
                    help="监听地址（默认 127.0.0.1；如需局域网直接访问可设 0.0.0.0）")
    ap.add_argument("--port", type=int, default=7860, help="监听端口（默认 7860）")
    ap.add_argument("--share", action="store_true",
                    help="生成 gradio 公网临时链接（需外网，一般用不到）")
    args = ap.parse_args()

    try:
        import gradio  # noqa: F401
    except ImportError:
        print("[error] 未安装 gradio。请先安装：pip install gradio", file=sys.stderr)
        return 2

    _ensure_localhost_no_proxy(args.host)

    capture_dir = Path(args.capture_dir).expanduser().resolve()
    demo = build_app(capture_dir, args.cuda_devices)
    print(f"[ui] 启动 Web 服务: http://{args.host}:{args.port}")
    print("[ui] 在浏览器打开上面的网址即可使用。")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
