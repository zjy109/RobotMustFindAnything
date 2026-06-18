"""模型推理类：RynnBrainDetector + SAM2Segmenter。

每个类只负责自己的领域，签名干净：
    RynnBrainDetector.detect(image, target_zh) -> Optional[List[int]]
        返回 [x1,y1,x2,y2] 归一化 [0,1000] 坐标，未找到返回 None。
    SAM2Segmenter.segment(image, bbox_pixel) -> np.ndarray
        返回 (H, W) bool mask。

模型加载惰性：构造函数会真去加载权重；只有 runner 真正需要推理时才会构造。
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from .config import RYNNBRAIN_GENERATE_KW

# torch 是重依赖，构造模型时才真正需要。这里延迟 import，让单元测试 / --help
# 在没装 torch 的环境下也能 import 本模块。
_torch = None  # 由各类 __init__ 通过 _get_torch() 取


def _get_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch


# ---------------------------------------------------------------------------
# CUDA 设备
# ---------------------------------------------------------------------------

def setup_cuda_devices(cuda_devices: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices


# ---------------------------------------------------------------------------
# RynnBrain：VLM 检测器
# ---------------------------------------------------------------------------

class RynnBrainDetector:
    """RynnBrain-8B 包装。

    工作流（与原 benchmark.py 一致）：
        1. （可选）存在性预筛：问『图里有没有 X』，回答 'no' 则直接返回 None
        2. 定位：要求 VLM 输出 <object> (x1,y1), (x2,y2) </object>，正则解析

    支持中文 prompt（这是 RynnBrain 的能力，不需要做翻译）。
    """

    BBOX_REGEX = re.compile(
        r"<object>.*?\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\).*?</object>",
        re.DOTALL,
    )

    EXISTENCE_FORMAT = (
        "Use this strict checklist:\n"
        "- Color matches exactly? (If no -> 'No')\n"
        "- 3D Shape matches exactly? (If flat instead of 3D -> 'No')\n"
        "- Category matches exactly? (If look-alike -> 'No')\n"
        "Answer 'No' if ANY check fails. Output exactly one word: 'Yes' or 'No'."
    )
    BBOX_FORMAT = (
        "RULES:\n"
        "1. MUST match exact color, 3D shape, and category.\n"
        "2. Tightly enclose ONLY the true target.\n"
        "Generate coordinates for exactly one object. x1,y1,x2,y2 ∈ [0,1000].\n"
        "Output strictly: <object> (x1, y1), (x2, y2) </object>"
    )

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        enable_existence_check: bool = True,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"RynnBrain 模型路径不存在: {model_path}\n"
                "正常情况下 runner 会调用 model_resolver 自动下载；如手动构造，"
                "请先运行 `python scripts/download_models.py`。"
            )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype="auto", device_map=device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.enable_existence_check = enable_existence_check

    def _query(self, image: Image.Image, prompt: str) -> str:
        """对单张图 + prompt 做一次贪婪解码，返回纯文本回答。

        生成参数被锁定为 ``RYNNBRAIN_GENERATE_KW``（即与原 benchmark.py 完全一致：
        ``do_sample=False, max_new_tokens=128``），任何调用方都无法覆盖。
        """
        torch = _get_torch()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            out_ids = self.model.generate(**inputs, **RYNNBRAIN_GENERATE_KW)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

    def detect(self, image: Image.Image, target_zh: str) -> Optional[List[int]]:
        """返回 [0,1000] 归一化的 xyxy bbox，找不到时返回 None。"""
        if self.enable_existence_check:
            prompt = (
                f"Verify if the EXACT '{target_zh}' is present in any of these images.\n"
                f"{self.EXISTENCE_FORMAT}"
            )
            ans = self._query(image, prompt)
            if "yes" not in ans.lower():
                return None

        prompt = (
            f"Localize the EXACT '{target_zh}' in the images.\n{self.BBOX_FORMAT}"
        )
        ans = self._query(image, prompt)
        m = self.BBOX_REGEX.search(ans)
        if not m:
            return None
        return [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))]


# ---------------------------------------------------------------------------
# SAM2：分割器
# ---------------------------------------------------------------------------

class SAM2Segmenter:
    """SAM2 通过 bbox prompt 出 mask。

    模型权重一律从本地路径加载（由 ``model_resolver.ensure_sam2()`` 准备）。
    """

    def __init__(self, model_path: str):
        from transformers import AutoModelForMaskGeneration, AutoProcessor
        torch = _get_torch()

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"SAM2 模型路径不存在: {model_path}\n"
                "正常情况下 runner 会调用 model_resolver 自动下载；如手动构造，"
                "请先运行 `python scripts/download_models.py`。"
            )

        self.model_path = str(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForMaskGeneration.from_pretrained(
            self.model_path,
        ).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def segment(self, image: Image.Image, bbox_pixel: List[int]) -> np.ndarray:
        """返回 (H, W) bool mask。"""
        torch = _get_torch()
        inputs = self.processor(
            images=image,
            input_boxes=[[bbox_pixel]],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks.cpu(),
            original_sizes=inputs.original_sizes.cpu(),
            reshaped_input_sizes=inputs.reshaped_input_sizes.cpu(),
        )[0]

        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            best_idx = int(outputs.iou_scores[0, 0].argmax().item())
        else:
            best_idx = 0

        return masks[0][best_idx].numpy() > 0
