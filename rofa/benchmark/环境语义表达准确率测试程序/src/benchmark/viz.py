"""单样本叠加可视化（pred bbox + pred mask + GT bbox）。

每张图体积 ~50KB，1000 张总共约 50MB，按需关闭（--no-viz）。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


COLOR_PRED_BBOX = (0, 255, 0)     # 绿
COLOR_PRED_MASK = (255, 0, 0)     # 蓝（BGR 中的蓝色通道；OpenCV 是 BGR）
COLOR_GT_BBOX = (0, 200, 255)     # 黄（更醒目，避免与绿混）
COLOR_CONTOUR = (255, 0, 0)


def render_overlay(
    rgb_arr: np.ndarray,
    pred_bbox_pixel: Optional[List[int]],
    pred_mask: Optional[np.ndarray],
    gt_2d_bbox: Optional[List[int]] = None,
    label: str = "",
    mask_alpha: float = 0.5,
) -> np.ndarray:
    """在 RGB 图上画 pred mask 蓝色半透明 + pred bbox 绿框 + GT bbox 黄框 + label。

    返回 (H, W, 3) **BGR** uint8 数组（可直接 cv2.imwrite，或转 RGB 给 PIL/Tk 显示）。

    rgb_arr: (H, W, 3) RGB uint8
    pred_mask: (H, W) bool, 可为 None
    pred_bbox_pixel / gt_2d_bbox: [x1, y1, x2, y2]，可为 None
    mask_alpha: 掩码叠加透明度（0~1）
    """
    img = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR).copy()

    # 1) pred mask 半透明叠加（蓝色）
    if pred_mask is not None and pred_mask.any():
        mask_u8 = (pred_mask.astype(np.uint8) * 255)
        colored = np.zeros_like(img)
        colored[:, :, 0] = mask_u8  # B 通道
        cv2.addWeighted(colored, float(mask_alpha), img, 1.0, 0, img)
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(img, contours, -1, COLOR_CONTOUR, 2)

    # 2) GT 2D bbox 黄框
    if gt_2d_bbox is not None and len(gt_2d_bbox) == 4 and any(v > 0 for v in gt_2d_bbox):
        x1, y1, x2, y2 = [int(v) for v in gt_2d_bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_GT_BBOX, 2)
        cv2.putText(img, "GT", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GT_BBOX, 1, cv2.LINE_AA)

    # 3) pred bbox 绿框
    if pred_bbox_pixel is not None and len(pred_bbox_pixel) == 4:
        x1, y1, x2, y2 = [int(v) for v in pred_bbox_pixel]
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PRED_BBOX, 2)
        cv2.putText(img, "PRED", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PRED_BBOX, 1, cv2.LINE_AA)

    # 4) 顶部 label
    if label:
        cv2.putText(img, label, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return img


def save_overlay(
    rgb_arr: np.ndarray,
    pred_bbox_pixel: Optional[List[int]],
    pred_mask: Optional[np.ndarray],
    gt_2d_bbox: Optional[List[int]],
    save_path: Path,
    label: str = "",
) -> None:
    """渲染叠加图并写盘（行为与原实现一致，内部复用 render_overlay）。"""
    img = render_overlay(rgb_arr, pred_bbox_pixel, pred_mask, gt_2d_bbox, label)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)
