"""主流水线：加载数据集 → 逐样本推理 → 落盘结果。

设计要点：
- 进度条（tqdm）显示总体精度滚动均值
- 每跑完一个类就 flush results.json，崩了能续
- 断点续跑（--resume）：results.json 已有的样本直接跳过
- 失败样本会单独保存可视化到 visualizations/，便于后审
- 推理总耗时 / 各阶段耗时都记到 process.log
"""
from __future__ import annotations

import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from . import __version__
from .config import BenchmarkConfig
from .geometry import (
    compute_2d_iou,
    compute_3d_iou,
    denormalize_bbox,
    mask_to_3d_aabb,
)
from .model_resolver import ensure_rynnbrain, ensure_sam2
from .models import RynnBrainDetector, SAM2Segmenter, setup_cuda_devices
from .viz import save_overlay


# ---------------------------------------------------------------------------
# 数据集加载与索引
# ---------------------------------------------------------------------------

def load_dataset_index(dataset_root: Path) -> List[Dict[str, Any]]:
    """读 samples.json，返回样本列表（与 finalize_dataset/build_subset 兼容）。"""
    samples_path = dataset_root / "samples.json"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"数据集缺少 samples.json: {samples_path}\n"
            "请确认 --dataset 指向 finalize_dataset.py 或 build_subset_dataset.py 的产物目录。"
        )
    with open(samples_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict) and isinstance(doc.get("samples"), list):
        return doc["samples"]
    raise ValueError(f"samples.json 格式无法识别: {type(doc).__name__}")


def load_sample(dataset_root: Path, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """加载单条样本的 rgb / depth / intrinsics / aabb（GT）。

    返回 None 表示某文件缺失。
    """
    sample_dir = dataset_root / item["sample_dir"]
    rgb_path = sample_dir / "rgb.jpg"
    depth_path = sample_dir / "depth.png"
    intrinsics_path = sample_dir / "intrinsics.json"
    aabb_path = sample_dir / "aabb.json"

    missing = [p for p in (rgb_path, depth_path, intrinsics_path, aabb_path) if not p.exists()]
    if missing:
        return None

    rgb_img = Image.open(rgb_path).convert("RGB")
    depth_map = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    with open(intrinsics_path, "r", encoding="utf-8") as f:
        intrinsics = json.load(f)
    with open(aabb_path, "r", encoding="utf-8") as f:
        aabb_data = json.load(f)

    # GT 3D AABB：min + max 拼成 6 维
    gt_3d_bbox = list(aabb_data["min"]) + list(aabb_data["max"])

    # GT 2D bbox（finalize 后才会有 bbox_2d.from_mask）
    gt_2d_bbox = [0, 0, 0, 0]
    bbox_2d = aabb_data.get("bbox_2d", {})
    if isinstance(bbox_2d, dict):
        from_mask = bbox_2d.get("from_mask")
        if isinstance(from_mask, list) and len(from_mask) == 4:
            gt_2d_bbox = [int(v) for v in from_mask]

    return {
        "rgb_img": rgb_img,
        "depth_map": depth_map,
        "intrinsics": intrinsics,
        "gt_3d_bbox": gt_3d_bbox,
        "gt_2d_bbox": gt_2d_bbox,
        "img_width": rgb_img.size[0],
        "img_height": rgb_img.size[1],
    }


# ---------------------------------------------------------------------------
# 断点续跑
# ---------------------------------------------------------------------------

def load_existing_results(results_path: Path) -> Dict[str, Dict[str, Any]]:
    """读历史 results.json 中已经处理过的样本，返回 {sample_id: result_dict}。"""
    if not results_path.exists():
        return {}
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return {}

    by_id: Dict[str, Dict[str, Any]] = {}
    for cls_block in (doc.get("sample_results") or {}).values():
        if not isinstance(cls_block, dict):
            continue
        for s in cls_block.get("samples", []):
            sid = s.get("sample_id")
            if sid:
                by_id[sid] = s
    return by_id


# ---------------------------------------------------------------------------
# 单样本推理
# ---------------------------------------------------------------------------

def evaluate_one_sample(
    item: Dict[str, Any],
    cfg: BenchmarkConfig,
    rynn: RynnBrainDetector,
    sam2: SAM2Segmenter,
) -> Dict[str, Any]:
    """对单个样本跑完整推理流程，返回结果 dict。"""
    sample_id = item["sample_id"]
    # 与 benchmark.py 一致：优先 class_name_zh，回退 class_name
    class_zh = item.get("class_name_zh") or item.get("class_name")

    bundle = load_sample(cfg.dataset_root, item)
    if bundle is None:
        return {
            "sample_id": sample_id,
            "class_name": item.get("class_name"),
            "object_prompt": class_zh,
            "error": "missing_files",
            "is_correct": False,
        }

    rgb_img = bundle["rgb_img"]
    depth_map = bundle["depth_map"]
    intrinsics = bundle["intrinsics"]
    gt_3d_bbox = bundle["gt_3d_bbox"]
    gt_2d_bbox = bundle["gt_2d_bbox"]
    W, H = bundle["img_width"], bundle["img_height"]

    pred_2d_bbox: Optional[List[int]] = None
    pred_3d_bbox: Optional[List[float]] = None
    pred_mask: Optional[np.ndarray] = None
    iou_2d = 0.0
    iou_3d = 0.0
    is_correct = False

    t0 = time.time()
    bbox_norm = rynn.detect(rgb_img, class_zh)
    t_rynn = time.time() - t0

    t_sam = 0.0
    t_3d = 0.0
    if bbox_norm is not None:
        pred_2d_bbox = denormalize_bbox(bbox_norm, W, H)
        iou_2d = compute_2d_iou(pred_2d_bbox, gt_2d_bbox)

        t1 = time.time()
        pred_mask = sam2.segment(rgb_img, pred_2d_bbox)
        t_sam = time.time() - t1

        t2 = time.time()
        pred_3d_bbox = mask_to_3d_aabb(
            depth_map, pred_mask, intrinsics, cfg.sor_nb, cfg.sor_std,
        )
        t_3d = time.time() - t2

        if pred_3d_bbox is not None:
            iou_3d = compute_3d_iou(pred_3d_bbox, gt_3d_bbox)
            is_correct = iou_3d >= cfg.iou_threshold

    # 可视化
    if cfg.save_visualizations and pred_2d_bbox is not None:
        viz_path = cfg.visualizations_dir / f"det_seg_{sample_id}.jpg"
        label = f"{sample_id} | 3D IoU={iou_3d:.3f} | {'OK' if is_correct else 'FAIL'}"
        save_overlay(
            np.array(rgb_img), pred_2d_bbox, pred_mask, gt_2d_bbox, viz_path, label,
        )

    return {
        "sample_id": sample_id,
        "class_name": item.get("class_name"),
        "object_prompt": class_zh,
        "pred_2d_bbox": pred_2d_bbox,
        "gt_2d_bbox": gt_2d_bbox,
        "2d_iou": round(iou_2d, 4),
        "pred_3d_bbox": pred_3d_bbox,
        "gt_3d_bbox": gt_3d_bbox,
        "3d_iou": round(iou_3d, 4),
        "is_correct": is_correct,
        "timing": {
            "rynnbrain_s": round(t_rynn, 3),
            "sam2_s": round(t_sam, 3),
            "lift_3d_s": round(t_3d, 3),
        },
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def collect_env_snapshot(cfg: BenchmarkConfig) -> Dict[str, Any]:
    """记录运行环境，便于复现。"""
    snap = {
        "benchmark_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_devices": cfg.cuda_devices,
        "config": cfg.to_dict(),
    }
    try:
        import torch
        snap["torch"] = torch.__version__
        snap["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            snap["cuda_device_count"] = torch.cuda.device_count()
            snap["cuda_device_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        snap["torch_error"] = str(exc)
    try:
        import transformers
        snap["transformers"] = transformers.__version__
    except Exception:
        pass
    return snap


def write_results_atomic(path: Path, doc: Dict[str, Any]) -> None:
    """先写到 .tmp，再 rename，保证 results.json 永远是完整的。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def group_samples_by_class(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_cls: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        # 与 benchmark.py 一致：优先 class_name_zh，回退 class_name
        key = s.get("class_name_zh") or s.get("class_name")
        by_cls.setdefault(key, []).append(s)
    return by_cls


def run_benchmark(cfg: BenchmarkConfig) -> Dict[str, Any]:
    """执行主流水线。返回最终 results dict。"""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_visualizations:
        cfg.visualizations_dir.mkdir(parents=True, exist_ok=True)

    # 日志
    logging.basicConfig(
        filename=str(cfg.process_log),
        filemode="a",
        format="%(asctime)s  %(message)s",
        level=logging.INFO,
        force=True,
    )

    # 环境快照
    env_snap = collect_env_snapshot(cfg)
    with open(cfg.env_snapshot, "w", encoding="utf-8") as f:
        json.dump(env_snap, f, ensure_ascii=False, indent=2)

    # 1) 数据集加载
    samples = load_dataset_index(cfg.dataset_root)
    if cfg.only_class:
        samples = [s for s in samples if s.get("class_name") == cfg.only_class]
    if cfg.max_samples > 0:
        samples = samples[: cfg.max_samples]
    if not samples:
        raise RuntimeError("过滤后没有可评测的样本")

    samples_by_class = group_samples_by_class(samples)
    total = sum(len(v) for v in samples_by_class.values())
    print(f"[benchmark] 数据集: {cfg.dataset_root}")
    print(f"[benchmark] 类别数: {len(samples_by_class)}, 样本数: {total}")
    print(
        "[benchmark] 锁死参数: "
        f"iou_threshold={cfg.iou_threshold}  "
        f"sor_nb={cfg.sor_nb}  sor_std={cfg.sor_std}  "
        f"existence_check={cfg.enable_existence_check}  "
        f"sam2={cfg.sam2_model_id}"
    )
    logging.info(f"[start] dataset={cfg.dataset_root} total={total} classes={len(samples_by_class)}")
    logging.info(
        f"[frozen] iou={cfg.iou_threshold} sor_nb={cfg.sor_nb} sor_std={cfg.sor_std} "
        f"existence_check={cfg.enable_existence_check} sam2={cfg.sam2_model_id} "
        f"gen_kw={cfg.rynnbrain_generate_kw}"
    )

    # 2) 断点续跑
    existing = load_existing_results(cfg.results_json) if cfg.resume else {}
    if existing:
        print(f"[benchmark] resume: 已有结果中含 {len(existing)} 个样本，将跳过")
        logging.info(f"[resume] reuse_existing_count={len(existing)}")

    # 3) 模型加载
    setup_cuda_devices(cfg.cuda_devices)

    # 3.1) 准备模型权重（统一下载到 <项目根>/models/）
    print("[benchmark] 准备模型权重 ...")
    rynn_path = ensure_rynnbrain()
    sam2_path = ensure_sam2()
    cfg.rynnbrain_model_path = rynn_path  # 回填，便于审计与日志
    cfg.sam2_model_path = sam2_path
    logging.info(f"[model] rynnbrain_path={rynn_path}")
    logging.info(f"[model] sam2_path={sam2_path}")

    print("[benchmark] 加载 RynnBrain ...")
    rynn = RynnBrainDetector(
        model_path=str(rynn_path),
        enable_existence_check=cfg.enable_existence_check,
    )
    print("[benchmark] 加载 SAM2 ...")
    sam2 = SAM2Segmenter(model_path=str(sam2_path))

    # 4) 推理
    results_by_class: Dict[str, Dict[str, Any]] = {}
    failed_global: List[str] = []
    success_count = 0
    processed_count = 0

    pbar = tqdm(total=total, desc="评测进度", unit="sample")

    for class_zh, items in samples_by_class.items():
        cls_block = results_by_class.setdefault(class_zh, {"samples": [], "stats": {}})
        ious_2d: List[float] = []
        ious_3d: List[float] = []
        cls_success = 0

        for item in items:
            sid = item["sample_id"]

            # 断点续跑
            if sid in existing:
                rec = existing[sid]
                cls_block["samples"].append(rec)
                ious_2d.append(rec.get("2d_iou", 0.0))
                ious_3d.append(rec.get("3d_iou", 0.0))
                if rec.get("is_correct"):
                    success_count += 1
                    cls_success += 1
                else:
                    failed_global.append(sid)
                processed_count += 1
                pbar.update(1)
                continue

            try:
                rec = evaluate_one_sample(item, cfg, rynn, sam2)
            except Exception as exc:
                logging.exception(f"[error] sample={sid}: {exc}")
                rec = {
                    "sample_id": sid,
                    "class_name": item.get("class_name"),
                    "object_prompt": class_zh,
                    "error": str(exc),
                    "is_correct": False,
                    "3d_iou": 0.0,
                    "2d_iou": 0.0,
                }

            cls_block["samples"].append(rec)
            iou_3d = float(rec.get("3d_iou", 0.0))
            iou_2d = float(rec.get("2d_iou", 0.0))
            ious_2d.append(iou_2d)
            ious_3d.append(iou_3d)
            if rec.get("is_correct"):
                success_count += 1
                cls_success += 1
            else:
                failed_global.append(sid)
            processed_count += 1

            timing = rec.get("timing", {})
            logging.info(
                f"[sample] {processed_count}/{total} {sid} cls={class_zh} "
                f"2d_iou={iou_2d:.4f} 3d_iou={iou_3d:.4f} "
                f"correct={rec.get('is_correct')} "
                f"t_rynn={timing.get('rynnbrain_s', 0)}s "
                f"t_sam={timing.get('sam2_s', 0)}s "
                f"t_3d={timing.get('lift_3d_s', 0)}s",
            )

            cur_acc = success_count / max(1, processed_count)
            pbar.set_postfix({"acc": f"{cur_acc:.4f}"})
            pbar.update(1)

        # 类内汇总
        n = len(ious_3d)
        if n > 0:
            cls_block["stats"] = {
                "total_samples": n,
                "avg_2d_iou": round(sum(ious_2d) / n, 4),
                "avg_3d_iou": round(sum(ious_3d) / n, 4),
                "accuracy_3d": round(cls_success / n, 4),
            }

        # 增量 flush
        cur_doc = build_results_doc(
            cfg, results_by_class, processed_count, total,
            success_count, failed_global,
        )
        write_results_atomic(cfg.results_json, cur_doc)

    pbar.close()
    final = build_results_doc(
        cfg, results_by_class, processed_count, total, success_count, failed_global,
    )
    write_results_atomic(cfg.results_json, final)

    # 总结打印
    acc = success_count / max(1, processed_count)
    print()
    print("=" * 60)
    print("评测完成")
    print(f"  总样本数: {processed_count} / {total}")
    print(f"  成功 (3D IoU >= {cfg.iou_threshold}): {success_count}")
    print(f"  最终精度: {acc:.4f}  ({acc:.2%})")
    print(f"  结果文件: {cfg.results_json}")
    print(f"  日志:     {cfg.process_log}")
    print("=" * 60)
    logging.info(f"[done] success={success_count}/{processed_count} acc={acc:.4f}")

    return final


def build_results_doc(
    cfg: BenchmarkConfig,
    results_by_class: Dict[str, Dict[str, Any]],
    processed: int,
    total: int,
    success: int,
    failed: List[str],
) -> Dict[str, Any]:
    """results.json 的标准结构（与原 benchmark.py 一致）。"""
    return {
        "schema_version": 1,
        "config": cfg.to_dict(),
        "sample_results": results_by_class,
        "summary": {
            "total_processed": processed,
            "total_target": total,
            "iou_threshold": cfg.iou_threshold,
            "success_count": success,
            "accuracy": round(success / max(1, processed), 4),
            "failed_cases": failed,
        },
    }
