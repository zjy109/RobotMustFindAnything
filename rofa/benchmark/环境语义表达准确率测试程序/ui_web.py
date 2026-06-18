#!/usr/bin/env python3
"""采集数据可视化检索 —— Web 版（适合跳板机 / 无显示远程服务器）。

与 `ui.py` 功能一致，但用浏览器渲染，**不依赖 X11 / 本地显示 / 服务器 OpenGL**：

    - 在服务器上启动一个本地 Web 服务（默认 127.0.0.1:7860）
    - 你通过 SSH 端口转发把它映射到本地，浏览器打开即可
    - 2D 叠加（bbox + 半透明掩码）直接出图；3D 点云 + AABB 在浏览器里用 three.js 渲染

用法（服务器上）：
    python ui_web.py --capture-dir ./captures --port 7860

跳板机端口转发（本地执行；<bastion> 是跳板机，<server> 是 GPU 服务器）：
    ssh -N -L 7860:localhost:7860 -J user@<bastion> user@<server>
    # 然后本地浏览器打开 http://localhost:7860

依赖：gradio（见 requirements.txt）。3D 预览还需 open3d 导出 .ply。
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# 让脚本能 import 项目内 src/benchmark 与同目录 ui.py
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.capture_io import (  # noqa: E402
    depth_to_points,
    load_capture_sample,
    load_index,
)
from benchmark.viz import render_overlay  # noqa: E402
from ui import Pipeline  # noqa: E402  复用 ui.py 里的推理封装（不会触发 tkinter import）


# ---------------------------------------------------------------------------
# 工具
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
    pts = []
    ts = np.linspace(0, 1, n_per_edge)[:, None]
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
    state: Dict[str, Any] = {"capture_dir": capture_dir}

    def list_ids(cdir_str: str) -> List[str]:
        cdir = Path(cdir_str).expanduser().resolve()
        state["capture_dir"] = cdir
        return [r.get("sample_id", "?") for r in load_index(cdir)]

    def refresh(cdir_str: str):
        ids = list_ids(cdir_str)
        info = f"已加载 {len(ids)} 个样本：{state['capture_dir']}" if ids \
            else f"目录中没有样本（缺 samples.json）：{state['capture_dir']}"
        first = ids[0] if ids else None
        preview = preview_sample(first) if first else None
        return gr.update(choices=ids, value=first), info, preview

    def _bundle(sample_id: Optional[str]):
        if not sample_id:
            return None
        for r in load_index(state["capture_dir"]):
            if r.get("sample_id") == sample_id:
                return load_capture_sample(state["capture_dir"], r)
        return None

    def preview_sample(sample_id: Optional[str]):
        b = _bundle(sample_id)
        return None if b is None else np.array(b["rgb_img"])

    def run_query(sample_id: Optional[str], target: str):
        b = _bundle(sample_id)
        if b is None:
            return None, "请先选择一个样本。", None
        target = (target or "").strip()
        if not target:
            return np.array(b["rgb_img"]) if b else None, "请输入要查找的物体名称。", None

        logs: List[str] = []
        res = pipeline.infer(b, target, log=lambda m: logs.append(m))
        if not res.get("found"):
            return (np.array(b["rgb_img"]),
                    f"目标『{target}』未找到（RynnBrain 判定不存在 / 无法定位）。\n\n"
                    + "\n".join(logs), None)

        overlay_bgr = render_overlay(
            np.array(b["rgb_img"]), res["pred_2d"], res["mask"],
            gt_2d_bbox=None, label=target, mask_alpha=0.5,
        )
        overlay_rgb = np.ascontiguousarray(overlay_bgr[:, :, ::-1])
        info = (
            f"目标: {target}\n"
            f"2D bbox(px): {res['pred_2d']}\n\n"
            + fmt_aabb("相机系 3D AABB", res["aabb_cam"]) + "\n\n"
            + fmt_aabb("世界系 3D AABB(已应用随机位姿)", res["aabb_world"])
        )
        ply = export_ply(b, res["mask"], res["aabb_cam"])
        if ply is None:
            info += "\n\n（未安装 open3d，跳过三维点云导出）"
        return overlay_rgb, info, ply

    with gr.Blocks(title="环境语义检索 UI (Web)") as demo:
        gr.Markdown("## 环境语义检索 UI（Web 版）\n选择采集样本 → 输入要查找的物体 → 查看 bbox + 掩码 + 3D AABB")
        with gr.Row():
            cdir_box = gr.Textbox(value=str(capture_dir), label="采集目录", scale=4)
            refresh_btn = gr.Button("刷新", scale=1)
        with gr.Row():
            with gr.Column(scale=1):
                sample_dd = gr.Dropdown(choices=list_ids(str(capture_dir)),
                                        label="样本", interactive=True)
                query_box = gr.Textbox(label="查找物体（支持中文）", placeholder="例如：水杯")
                run_btn = gr.Button("查找", variant="primary")
                status = gr.Textbox(label="状态 / 结果信息", lines=10)
            with gr.Column(scale=2):
                img_out = gr.Image(label="可视化（绿框=bbox，蓝=掩码）", type="numpy")
                model3d = gr.Model3D(label="3D 点云 + AABB（红框）", clear_color=[0, 0, 0, 1])

        refresh_btn.click(refresh, inputs=cdir_box,
                          outputs=[sample_dd, status, img_out])
        sample_dd.change(preview_sample, inputs=sample_dd, outputs=img_out)
        run_btn.click(run_query, inputs=[sample_dd, query_box],
                      outputs=[img_out, status, model3d])

    return demo


def main() -> int:
    ap = argparse.ArgumentParser(description="采集数据可视化检索 UI（Web 版）")
    ap.add_argument("--capture-dir", type=str, default="./captures",
                    help="sample_rsd4xx.py 的采集输出目录（默认 ./captures）")
    ap.add_argument("--cuda-devices", type=str, default="0",
                    help="CUDA_VISIBLE_DEVICES（默认 0）")
    ap.add_argument("--host", type=str, default="127.0.0.1",
                    help="监听地址（默认 127.0.0.1，配合 SSH 端口转发最安全）")
    ap.add_argument("--port", type=int, default=7860, help="监听端口（默认 7860）")
    ap.add_argument("--share", action="store_true",
                    help="生成 gradio 公网临时链接（需外网；一般用端口转发即可，不建议开）")
    args = ap.parse_args()

    try:
        import gradio  # noqa: F401
    except ImportError:
        print("[error] 未安装 gradio。请先安装：pip install gradio", file=sys.stderr)
        return 2

    capture_dir = Path(args.capture_dir).expanduser().resolve()
    demo = build_app(capture_dir, args.cuda_devices)
    print(f"[ui_web] 启动 Web 服务: http://{args.host}:{args.port}")
    print("[ui_web] 跳板机端口转发示例：")
    print(f"    ssh -N -L {args.port}:localhost:{args.port} -J user@<bastion> user@<server>")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
