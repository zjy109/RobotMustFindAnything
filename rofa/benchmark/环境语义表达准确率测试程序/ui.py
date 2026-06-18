#!/usr/bin/env python3
"""采集数据可视化检索 UI。

读取 `sample_rsd4xx.py` 采集的样本，输入要查找的物体名称，调用
RynnBrain（定位）+ SAM2（分割）+ 深度反投影流水线，在界面上可视化：

    - 目标 2D bbox（绿框）
    - 目标掩码（半透明蓝色叠加）
    - 由掩码 + 深度反投影得到的相机系 3D AABB
    - 应用样本随机位姿后的世界系 3D AABB
    - （可选）点云 + AABB 的 Open3D 三维窗口

用法：
    python ui.py --capture-dir ./captures
    python ui.py --capture-dir ./captures --cuda-devices 0

说明：
    - 模型在首次检索时惰性加载（与评测程序共用 model_resolver，自动下载到 ./models/）。
    - 推理在后台线程进行，不阻塞界面。
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import traceback
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
# 模型封装（惰性加载，供 UI 后台线程使用）
# ---------------------------------------------------------------------------

class Pipeline:
    """RynnBrain + SAM2 惰性加载与单帧推理。"""

    def __init__(self, cuda_devices: str = "0"):
        self.cuda_devices = cuda_devices
        self.rynn = None
        self.sam2 = None

    def ensure_loaded(self, log) -> None:
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

    def infer(self, bundle: Dict[str, Any], target: str, log) -> Dict[str, Any]:
        """对单个样本检索 target，返回可视化所需的中间结果。"""
        self.ensure_loaded(log)

        rgb_img = bundle["rgb_img"]
        depth_map = bundle["depth_map"]
        intrinsics = bundle["intrinsics"]
        W, H = bundle["img_width"], bundle["img_height"]

        log(f"RynnBrain 定位 '{target}' ...")
        bbox_norm = self.rynn.detect(rgb_img, target)
        if bbox_norm is None:
            return {"found": False}

        pred_2d = denormalize_bbox(bbox_norm, W, H)
        log("SAM2 分割 ...")
        mask = self.sam2.segment(rgb_img, pred_2d)

        log("深度反投影 + SOR 去噪 -> AABB ...")
        aabb_cam = mask_to_3d_aabb(depth_map, mask, intrinsics, SOR_NB, SOR_STD)
        aabb_world = None
        if aabb_cam is not None:
            aabb_world = transform_aabb(aabb_cam, bundle["pose_matrix"])

        return {
            "found": True,
            "pred_2d": pred_2d,
            "mask": mask,
            "aabb_cam": aabb_cam,
            "aabb_world": aabb_world,
        }


# ---------------------------------------------------------------------------
# Open3D 三维显示（可选）
# ---------------------------------------------------------------------------

def show_point_cloud(bundle: Dict[str, Any], mask: np.ndarray,
                     aabb_cam: Optional[List[float]]) -> None:
    """在 Open3D 窗口展示场景点云 + 目标 AABB（红框）。"""
    try:
        import open3d as o3d
    except ImportError:
        raise RuntimeError("未安装 open3d，无法显示三维点云：pip install open3d")

    rgb = np.array(bundle["rgb_img"])
    depth_map = bundle["depth_map"]
    intrinsics = bundle["intrinsics"]

    # 场景点云（下采样）+ 目标点云（原分辨率）
    scene_pts, scene_col = depth_to_points(depth_map, intrinsics, rgb=rgb, stride=2)
    obj_pts, _ = depth_to_points(depth_map, intrinsics, mask=mask)

    geoms = []
    if len(scene_pts) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene_pts)
        if scene_col is not None:
            pcd.colors = o3d.utility.Vector3dVector(scene_col)
        geoms.append(pcd)
    if len(obj_pts) > 0:
        opcd = o3d.geometry.PointCloud()
        opcd.points = o3d.utility.Vector3dVector(obj_pts)
        opcd.paint_uniform_color([1.0, 0.2, 0.2])
        geoms.append(opcd)
    if aabb_cam is not None:
        box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array(aabb_cam[:3]), max_bound=np.array(aabb_cam[3:]),
        )
        box.color = (1.0, 0.0, 0.0)
        geoms.append(box)

    if not geoms:
        raise RuntimeError("没有可显示的有效点（深度全为 0？）")
    o3d.visualization.draw_geometries(geoms, window_name="点云 + 目标 AABB")


# ---------------------------------------------------------------------------
# Tkinter UI
# ---------------------------------------------------------------------------

def run_ui(capture_dir: Path, cuda_devices: str) -> int:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from PIL import Image, ImageTk

    pipeline = Pipeline(cuda_devices=cuda_devices)
    msg_queue: "queue.Queue[tuple]" = queue.Queue()

    state: Dict[str, Any] = {
        "capture_dir": capture_dir,
        "records": [],
        "bundle": None,      # 当前样本
        "result": None,      # 最近一次推理结果
        "busy": False,
        "photo": None,       # 防止 PhotoImage 被 GC
    }

    root = tk.Tk()
    root.title("环境语义检索 UI — 采集数据可视化")
    root.geometry("1180x720")

    # ===== 顶部：采集目录 =====
    top = ttk.Frame(root, padding=8)
    top.pack(side=tk.TOP, fill=tk.X)
    ttk.Label(top, text="采集目录:").pack(side=tk.LEFT)
    dir_var = tk.StringVar(value=str(capture_dir))
    dir_entry = ttk.Entry(top, textvariable=dir_var, width=70)
    dir_entry.pack(side=tk.LEFT, padx=4)

    def choose_dir() -> None:
        d = filedialog.askdirectory(initialdir=dir_var.get() or ".")
        if d:
            dir_var.set(d)
            reload_index()

    ttk.Button(top, text="选择", command=choose_dir).pack(side=tk.LEFT, padx=2)
    ttk.Button(top, text="刷新", command=lambda: reload_index()).pack(side=tk.LEFT, padx=2)

    # ===== 主体：左列表 / 右画布 =====
    body = ttk.Frame(root, padding=8)
    body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    left = ttk.Frame(body)
    left.pack(side=tk.LEFT, fill=tk.Y)
    ttk.Label(left, text="样本列表").pack(anchor=tk.W)
    listbox = tk.Listbox(left, width=28, height=28, exportselection=False)
    listbox.pack(side=tk.LEFT, fill=tk.Y)
    lb_scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=listbox.yview)
    lb_scroll.pack(side=tk.LEFT, fill=tk.Y)
    listbox.config(yscrollcommand=lb_scroll.set)

    right = ttk.Frame(body, padding=(10, 0))
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    canvas = tk.Label(right, background="#222", anchor=tk.CENTER)
    canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # ===== 检索栏 =====
    query_bar = ttk.Frame(right)
    query_bar.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
    ttk.Label(query_bar, text="查找物体:").pack(side=tk.LEFT)
    query_var = tk.StringVar()
    query_entry = ttk.Entry(query_bar, textvariable=query_var, width=30)
    query_entry.pack(side=tk.LEFT, padx=4)
    search_btn = ttk.Button(query_bar, text="查找")
    search_btn.pack(side=tk.LEFT, padx=2)
    view3d_btn = ttk.Button(query_bar, text="查看3D点云")
    view3d_btn.pack(side=tk.LEFT, padx=2)

    # ===== 结果信息 =====
    info = tk.Text(right, height=8, wrap=tk.WORD)
    info.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))

    status_var = tk.StringVar(value="就绪。请选择左侧样本并输入要查找的物体。")
    status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ---- 工具函数 ----
    def set_status(text: str) -> None:
        status_var.set(text)

    def set_info(text: str) -> None:
        info.delete("1.0", tk.END)
        info.insert(tk.END, text)

    def display_image(rgb_arr_or_pil) -> None:
        if isinstance(rgb_arr_or_pil, np.ndarray):
            img = Image.fromarray(rgb_arr_or_pil)
        else:
            img = rgb_arr_or_pil
        cw = max(canvas.winfo_width(), 320)
        ch = max(canvas.winfo_height(), 240)
        disp = img.copy()
        disp.thumbnail((cw, ch))
        photo = ImageTk.PhotoImage(disp)
        state["photo"] = photo
        canvas.configure(image=photo)

    def fmt_aabb(name: str, aabb: Optional[List[float]]) -> str:
        if aabb is None:
            return f"{name}: 无（点云为空 / 深度无效）"
        mn = aabb[:3]
        mx = aabb[3:]
        ext = [mx[i] - mn[i] for i in range(3)]
        return (
            f"{name}:\n"
            f"  min = [{mn[0]:.3f}, {mn[1]:.3f}, {mn[2]:.3f}] m\n"
            f"  max = [{mx[0]:.3f}, {mx[1]:.3f}, {mx[2]:.3f}] m\n"
            f"  尺寸 = [{ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f}] m"
        )

    # ---- 索引加载 / 样本选择 ----
    def reload_index() -> None:
        cdir = Path(dir_var.get()).expanduser().resolve()
        state["capture_dir"] = cdir
        records = load_index(cdir)
        state["records"] = records
        listbox.delete(0, tk.END)
        for r in records:
            listbox.insert(tk.END, r.get("sample_id", "?"))
        if records:
            set_status(f"已加载 {len(records)} 个样本：{cdir}")
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(0)
            on_select()
        else:
            set_status(f"目录中没有样本（缺 samples.json）：{cdir}")
            canvas.configure(image="")
            state["photo"] = None
            set_info("")

    def current_record() -> Optional[Dict[str, Any]]:
        sel = listbox.curselection()
        if not sel:
            return None
        return state["records"][sel[0]]

    def on_select(_evt=None) -> None:
        rec = current_record()
        if rec is None:
            return
        bundle = load_capture_sample(state["capture_dir"], rec)
        if bundle is None:
            set_status(f"样本文件缺失：{rec.get('sample_id')}")
            return
        state["bundle"] = bundle
        state["result"] = None
        display_image(bundle["rgb_img"])
        pose = bundle.get("pose") or {}
        t = pose.get("translation")
        set_info(
            f"样本: {bundle['sample_id']}\n"
            f"分辨率: {bundle['img_width']}x{bundle['img_height']}\n"
            f"随机位姿平移: {t}\n"
            "（输入物体名称后点『查找』）"
        )
        set_status(f"已选择样本 {bundle['sample_id']}")

    listbox.bind("<<ListboxSelect>>", on_select)

    # ---- 检索（后台线程） ----
    def do_search() -> None:
        if state["busy"]:
            return
        bundle = state["bundle"]
        if bundle is None:
            messagebox.showwarning("提示", "请先在左侧选择一个样本。")
            return
        target = query_var.get().strip()
        if not target:
            messagebox.showwarning("提示", "请输入要查找的物体名称。")
            return

        state["busy"] = True
        search_btn.config(state=tk.DISABLED)

        def worker() -> None:
            try:
                res = pipeline.infer(
                    bundle, target,
                    log=lambda m: msg_queue.put(("status", m)),
                )
                msg_queue.put(("result", {"bundle": bundle, "target": target, "res": res}))
            except Exception as exc:  # noqa: BLE001
                msg_queue.put(("error", f"{exc}\n{traceback.format_exc()}"))

        threading.Thread(target=worker, daemon=True).start()

    def do_view3d() -> None:
        bundle = state["bundle"]
        result = state["result"]
        if bundle is None or result is None or not result.get("found"):
            messagebox.showinfo("提示", "请先成功检索出一个目标，再查看三维点云。")
            return

        def worker() -> None:
            try:
                show_point_cloud(bundle, result["mask"], result["aabb_cam"])
            except Exception as exc:  # noqa: BLE001
                msg_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    search_btn.config(command=do_search)
    view3d_btn.config(command=do_view3d)
    query_entry.bind("<Return>", lambda e: do_search())

    # ---- 处理后台线程消息 ----
    def poll_queue() -> None:
        try:
            while True:
                kind, payload = msg_queue.get_nowait()
                if kind == "status":
                    set_status(payload)
                elif kind == "error":
                    state["busy"] = False
                    search_btn.config(state=tk.NORMAL)
                    set_status("出错")
                    messagebox.showerror("错误", payload)
                elif kind == "result":
                    state["busy"] = False
                    search_btn.config(state=tk.NORMAL)
                    _on_result(payload)
        except queue.Empty:
            pass
        root.after(100, poll_queue)

    def _on_result(payload: Dict[str, Any]) -> None:
        bundle = payload["bundle"]
        target = payload["target"]
        res = payload["res"]
        state["result"] = res

        if not res.get("found"):
            display_image(bundle["rgb_img"])
            set_info(f"目标『{target}』未在图中找到（RynnBrain 判定不存在 / 无法定位）。")
            set_status("未找到目标")
            return

        overlay_bgr = render_overlay(
            np.array(bundle["rgb_img"]),
            res["pred_2d"], res["mask"], gt_2d_bbox=None,
            label=f"{target}", mask_alpha=0.5,
        )
        overlay_rgb = overlay_bgr[:, :, ::-1]  # BGR -> RGB
        display_image(np.ascontiguousarray(overlay_rgb))

        set_info(
            f"目标: {target}\n"
            f"2D bbox(px): {res['pred_2d']}\n\n"
            + fmt_aabb("相机系 3D AABB", res["aabb_cam"]) + "\n\n"
            + fmt_aabb("世界系 3D AABB(已应用随机位姿)", res["aabb_world"])
        )
        set_status("检索完成。可点『查看3D点云』查看三维结果。")

    # 启动
    reload_index()
    root.after(100, poll_queue)
    root.mainloop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="采集数据可视化检索 UI")
    ap.add_argument("--capture-dir", type=str, default="./captures",
                    help="sample_rsd4xx.py 的采集输出目录（默认 ./captures）")
    ap.add_argument("--cuda-devices", type=str, default="0",
                    help="CUDA_VISIBLE_DEVICES（默认 0）")
    args = ap.parse_args()
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    return run_ui(capture_dir, args.cuda_devices)


if __name__ == "__main__":
    sys.exit(main())
