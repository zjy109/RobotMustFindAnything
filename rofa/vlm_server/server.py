"""
VLM Search Server
接收客户端搜索请求，使用 RynnBrain 模型进行物体检测和定位
"""
import base64
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import io
from datetime import datetime

import cv2
import numpy as np
import torch
import zmq
from PIL import Image

from transformers import AutoModelForImageTextToText, AutoProcessor, Sam2Model, Sam2Processor
# ============ GPU 配置辅助函数 ============
def setup_cuda_devices(cuda_devices: str = "0") -> None:
    """设置可见的 CUDA 设备
    
    Args:
        cuda_devices: CUDA 设备 ID，例如 "0" 或 "0,1,2"
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    logger_temp = logging.getLogger(__name__)
    logger_temp.info(f"Set CUDA_VISIBLE_DEVICES={cuda_devices}")
# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 日志记录器需要在这里初始化以便在 setup_cuda_devices 中使用


# ============ 辅助函数 ============
def denormalize_points(point: List[int], width: int, height: int) -> List[int]:
    """将归一化坐标 [0, 1000] 反归一化到像素坐标"""
    x_norm, y_norm = point[0], point[1]
    x = round(x_norm / 1000 * width)
    y = round(y_norm / 1000 * height)
    return [x, y]


def denormalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    """将归一化 bbox [0, 1000] 反归一化到像素坐标"""
    x1_norm, y1_norm, x2_norm, y2_norm = bbox
    x1 = round(x1_norm / 1000 * width)
    y1 = round(y1_norm / 1000 * height)
    x2 = round(x2_norm / 1000 * width)
    y2 = round(y2_norm / 1000 * height)
    return [x1, y1, x2, y2]


def normalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    """将像素坐标 bbox 归一化到 [0, 1000]"""
    x1, y1, x2, y2 = bbox
    x1_norm = int(round(x1 / (width - 1) * 1000)) if width > 1 else 0
    y1_norm = int(round(y1 / (height - 1) * 1000)) if height > 1 else 0
    x2_norm = int(round(x2 / (width - 1) * 1000)) if width > 1 else 1000
    y2_norm = int(round(y2 / (height - 1) * 1000)) if height > 1 else 1000
    # 确保坐标在 [0, 1000] 范围内
    return [max(0, min(1000, v)) for v in [x1_norm, y1_norm, x2_norm, y2_norm]]


def parse_object_bbox(response_str: str) -> Optional[List[int]]:
    """从模型响应中解析物体 bbox
    
    期望格式: <object> (x1, y1), (x2, y2) </object>
    坐标范围: [0, 1000]
    """
    pattern = re.compile(r"<object>.*?\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\).*?</object>", re.DOTALL)
    match = pattern.search(response_str)
    if match:
        return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
    return None


def save_detection_results(
    save_dir: str,
    instruction: str,
    images: List[Image.Image],
    anchors: List[Dict],
    found: bool = False,
    found_image_index: Optional[int] = None,
    bbox_norm: Optional[List[int]] = None,
    found_indices_with_bboxes: Optional[Dict[int, List[int]]] = None,
) -> None:
    """保存所有接收到的图片和检测结果
    
    Args:
        save_dir: 保存目录（应为 server_results_dir / search_时间戳_物体名 格式）
        instruction: 搜索指令（目标物体名称）
        images: 所有接收到的图片列表
        anchors: 对应的 anchor 信息列表
        found: 是否找到物体
        found_image_index: 第一次找到时该物体所在图片的索引
        bbox_norm: 第一次找到时的 bbox [x1, y1, x2, y2]（归一化 [0,1000]）
        found_indices_with_bboxes: 所有找到的图片的字典 {图片索引: bbox_norm} - 包容多个找到的对象
    """
    # 如果提供了multiple detected objects的信息，使用它；否则从单个参数构建
    found_detections = found_indices_with_bboxes or {}
    if found and found_image_index is not None and bbox_norm is not None:
        found_detections[found_image_index] = bbox_norm
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(images)} images to {save_path}")
    
    # 保存所有图片
    for img_idx, img in enumerate(images):
        anchor_id = anchors[img_idx]["anchor_id"] if img_idx < len(anchors) else f"anchor_{img_idx:04d}"
        
        # 图片命名：anchor_id + 是否找到
        if img_idx in found_detections:
            # 这是找到物体的那张图
            img_filename = f"{anchor_id}_found.jpg"
        else:
            # 其他图片
            img_filename = f"{anchor_id}_notfound.jpg"
        
        img_path = save_path / img_filename
        img.save(img_path)
        logger.info(f"  Saved: {img_filename}")
    
    # 对所有找到物体的图片进行标注
    for found_idx, bbox_norm_detected in found_detections.items():
        found_img = images[found_idx]
        img_width, img_height = found_img.size
        bbox_pixel = denormalize_bbox(bbox_norm_detected, img_width, img_height)
        
        # 绘制 bbox
        annotated_img = found_img.copy().convert("RGB")
        draw = __import__('PIL.ImageDraw', fromlist=['ImageDraw']).Draw(annotated_img)
        x1, y1, x2, y2 = bbox_pixel
        draw.rectangle([x1, y1, x2, y2], outline='lime', width=5)
        
        # 添加标记
        try:
            font = __import__('PIL.ImageFont', fromlist=['ImageFont']).truetype("arial.ttf", size=15)
        except:
            font = __import__('PIL.ImageFont', fromlist=['ImageFont']).load_default()
        draw.text((x1 + 2, y1 - 15), "Found", fill='lime', font=font)
        
        # 保存标注版本
        anchor_id = anchors[found_idx]["anchor_id"] if found_idx < len(anchors) else f"anchor_{found_idx:04d}"
        annotated_filename = f"{anchor_id}_found_annotated.jpg"
        annotated_path = save_path / annotated_filename
        annotated_img.save(annotated_path)
        logger.info(f"  Saved annotated: {annotated_filename}")
    
    # 保存元数据
    metadata = {
        "instruction": instruction,
        "object_found": found,
        "num_images": len(images),
        "found_indices": list(found_detections.keys()),  # 所有找到的图片索引列表
        "found_detections": {str(k): v for k, v in found_detections.items()},  # 所有找到的 bbox
        "first_found_image_index": found_image_index,  # 兼容性：第一次找到的索引
        "first_found_bbox_normalized": bbox_norm,  # 兼容性：第一次找到的 bbox
        "anchors": anchors,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    metadata_path = save_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Metadata saved to {metadata_path}")


# ============ VLM 搜索引擎服务 ============
class VLMSearchServer:
    """
    VLM 搜索服务器
    接收客户端请求，使用 RynnBrain 模型进行物体检测和定位
    """
    
    def __init__(
        self,
        server_port: int = 5555,
        model_path: str = "../models/RynnBrain-8B",
        device: str = "auto",
        cuda_devices: str = "0",
        save_results: bool = False,
        sam2_model_id: str = "facebook/sam2.1-hiera-small",
        sam2_cache_dir: Optional[str] = None,
        sam2_device: Optional[str] = None,
    ):
        """
        初始化 VLM 搜索服务器
        
        Args:
            server_port: 服务器监听的端口
            model_path: RynnBrain 模型路径
            device: 推理设备 (auto/cuda/cpu)
            cuda_devices: 可见的 CUDA 设备 ID (默认: "0")
            save_results: 是否保存检测结果（启用后在 result_images 下创建目录）
            sam2_model_id: SAM2 模型 ID (默认: facebook/sam2.1-hiera-small)
            sam2_cache_dir: SAM2 模型缓存目录（可选，默认使用 .model_cache/<model_id>）
            sam2_device: SAM2 推理设备（可选，默认使用 cuda 如果可用）
        """
        # 设置 CUDA 设备（在模型加载之前）
        setup_cuda_devices(cuda_devices)
        
        self.server_port = server_port
        self.cuda_devices = cuda_devices
        self.save_results = save_results
        
        # 初始化结果保存目录结构
        self.results_base_dir = Path(".") / "result_images"
        self.results_base_dir.mkdir(exist_ok=True)
        
        # 为每次 Server 启动创建时间戳文件夹
        server_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.server_results_dir = self.results_base_dir / f"server_{server_timestamp}"
        self.server_results_dir.mkdir(exist_ok=True)
        
        # 初始化 Server 统一日志文件
        self.logs_dir = Path(".") / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.server_log_path = self.logs_dir / f"server_{timestamp}.log"
        
        # 创建 Server 启动日志
        with open(self.server_log_path, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write(f"VLM Search Server Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Server Port: {self.server_port}\n")
            f.write(f"Model Path: {model_path}\n")
            f.write(f"CUDA Devices: {cuda_devices}\n")
            f.write("=" * 100 + "\n\n")
        
        logger.info(f"Server log file: {self.server_log_path}")
        
        # 初始化模型
        logger.info(f"Loading RynnBrain model from {model_path}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype="auto",
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        logger.info("Model loaded successfully")
        
        # 初始化 SAM2 模型
        self.sam2_model_id = str(sam2_model_id)
        self.sam2_cache_dir = Path(sam2_cache_dir) if sam2_cache_dir else self.default_sam2_cache_dir(self.sam2_model_id)
        self.sam2_device = sam2_device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading SAM2 model: {self.sam2_model_id}")
        self.sam2_cache_dir.mkdir(parents=True, exist_ok=True)
        self.sam2_model = Sam2Model.from_pretrained(
            self.sam2_model_id,
            cache_dir=str(self.sam2_cache_dir),
        ).to(self.sam2_device)
        self.sam2_model.eval()
        self.sam2_processor = Sam2Processor.from_pretrained(
            self.sam2_model_id,
            cache_dir=str(self.sam2_cache_dir),
        )
        logger.info(
            f"SAM2 loaded: model_id={self.sam2_model_id}, device={self.sam2_device}, "
            f"cache_dir={self.sam2_cache_dir}"
        )

        # ====== 预热 SAM2 模型 (Warm-up) ======
        # 使用随机生成的图片进行一次完整推理，触发 CUDA 显存分配和内核初始化
        # 这将显著降低第一次真实网络请求的延迟
        logger.info("Starting SAM2 model warm-up...")
        try:
            # 1. 生成一张随机的彩色图片 (640x480 RGB 格式) 和一个任意的 BBox
            dummy_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            dummy_bbox = [50, 50, 200, 200]

            # 2. 模拟 Processor 处理
            inputs = self.sam2_processor(
                images=dummy_image,
                input_boxes=[[[dummy_bbox[0], dummy_bbox[1], dummy_bbox[2], dummy_bbox[3]]]],
                return_tensors="pt",
            )
            
            # 3. 将 Tensors 移动到对应设备 (GPU/CPU)
            inputs = {k: v.to(self.sam2_device) for k, v in inputs.items()}

            # 4. 执行无梯度前向推理
            with torch.no_grad():
                outputs = self.sam2_model(**inputs)

            # 5. 执行一次后处理逻辑，确保整个 Pipeline 都被跑通
            # 使用 getattr 或 hasattr 兼容部分 transformers 版本中 original_sizes 返回 list 的情况
            original_sizes = inputs["original_sizes"]
            if hasattr(original_sizes, "cpu"):
                original_sizes = original_sizes.cpu()
                
            _ = self.sam2_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                original_sizes,
            )
            
            logger.info("SAM2 model warm-up completed successfully. Ready for inference.")
            
        except Exception as e:
            # 加上 try-except 以防万一预热失败（比如版本兼容性小问题）导致整个 Server 无法启动
            logger.warning(f"SAM2 warm-up skipped or failed: {e}. The first request might be slightly slower.")
        
        # 初始化 ZMQ - Server 端监听所有接口
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.server_port}")
        logger.info(f"Server listening on tcp://*:{self.server_port}")
    
    def _check_object_existence(self, images: List[Image.Image], target_object: str) -> bool:
        """
        步骤1：检查目标物体是否在图片中存在
        
        Args:
            images: PIL Image 列表
            target_object: 目标物体描述
            
        Returns:
            物体是否存在
        """
        # 构建消息
        messages = [{
            "role": "user",
            "content": []
        }]
        
        # 添加所有图片
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})
        
        # 添加检查指令
        existence_instruction = f"Verify if the EXACT '{target_object}' is present in any of these images."
        existence_format = (
            "Use this strict checklist:\n"
            "- Color matches exactly? (If no -> 'No')\n"
            "- 3D Shape matches exactly? (If flat instead of 3D -> 'No')\n"
            "- Category matches exactly? (If look-alike -> 'No')\n"
            "Answer 'No' if ANY check fails. Output exactly one word: 'Yes' or 'No'."
        )
        
        messages[0]["content"].append({
            "type": "text",
            "text": f"{existence_instruction}\n{existence_format}"
        })
        
        # 推理
        logger.info("Checking object existence...")
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=128
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        logger.info(f"Existence check response: {response}")
        return "yes" in response.lower()
    
    def _localize_object_bbox(self, images: List[Image.Image], target_object: str) -> Optional[List[int]]:
        """
        步骤2：定位目标物体的 bbox
        
        Args:
            images: PIL Image 列表
            target_object: 目标物体描述
            
        Returns:
            Bbox [x1, y1, x2, y2] in [0, 1000] 范围，如果失败返回 None
        """
        # 构建消息
        messages = [{
            "role": "user",
            "content": []
        }]
        
        # 添加所有图片
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})
        
        # 添加定位指令
        bbox_instruction = f"Localize the EXACT '{target_object}' in the images."
        bbox_format_prompt = (
            "RULES:\n"
            "1. MUST match exact color, 3D shape, and category.\n"
            "2. IGNORE look-alikes (e.g., PC cases, scanners, lotion bottles, flat notebooks).\n"
            "3. Tightly enclose ONLY the true target.\n"
            "Generate coordinates for exactly one object. x1,y1,x2,y2 ∈ [0,1000].\n"
            "Output strictly: <object> (x1, y1), (x2, y2) </object>"
        )
        
        messages[0]["content"].append({
            "type": "text",
            "text": f"{bbox_instruction}\n{bbox_format_prompt}"
        })
        
        # 推理
        logger.info("Localizing object bbox...")
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                top_k=10
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        logger.info(f"Bbox localization response: {response}")
        
        # 解析 bbox
        bbox = parse_object_bbox(response)
        if bbox:
            logger.info(f"Detected bbox: {bbox}")
        else:
            logger.warning("Failed to parse bbox from response")
        
        return bbox
    
    def _process_single_image(
        self,
        image: Image.Image,
        image_index: int,
        target_object: str,
    ) -> Dict:
        """
        处理单张图片，进行物体检测和定位
        
        Args:
            image: PIL Image 对象
            image_index: 图片索引
            target_object: 目标物体描述
            
        Returns:
            处理结果 {"found": bool, "bbox_norm": [x1, y1, x2, y2] or None}
        """
        # 步骤1：检查物体是否存在
        object_exists = self._check_object_existence([image], target_object)
        
        if not object_exists:
            logger.info(f"Image {image_index}: Object not found")
            return {"found": False, "bbox_norm": None}
        
        # 步骤2：如果存在，定位物体 bbox
        logger.info(f"Image {image_index}: Object found, localizing bbox...")
        bbox_norm = self._localize_object_bbox([image], target_object)
        
        if bbox_norm:
            logger.info(f"Image {image_index}: Bbox detected: {bbox_norm}")
            return {"found": True, "bbox_norm": bbox_norm}
        else:
            logger.warning(f"Image {image_index}: Object found but bbox extraction failed")
            return {"found": False, "bbox_norm": None}
    
    @staticmethod
    def default_sam2_cache_dir(model_id: str) -> Path:
        """返回 SAM2 模型的默认缓存目录"""
        cache_name = str(model_id).replace("/", "--")
        return Path(__file__).resolve().parent.parent / ".model_cache" / cache_name

    def _segment_with_sam2(self, image: Image.Image, bbox_pixel: List[int]) -> Optional[Dict]:
        """使用 SAM2 根据 bbox 生成分割 mask
        
        Args:
            image: PIL Image 对象（RGB）
            bbox_pixel: 像素坐标 bbox [x1, y1, x2, y2]
            
        Returns:
            mask payload dict (png_base64 编码)，失败返回 None
        """
        try:
            x1, y1, x2, y2 = bbox_pixel
            inputs = self.sam2_processor(
                images=image,
                input_boxes=[[[x1, y1, x2, y2]]],
                return_tensors="pt",
            )
            inputs = {k: v.to(self.sam2_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.sam2_model(**inputs)

            masks = self.sam2_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
            )

            # masks[0]: [num_objects, num_masks_per_object, H, W]
            # 根据 IOU 分数选最优 mask
            if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
                best_idx = int(outputs.iou_scores[0, 0].argmax().item())
            else:
                best_idx = 0

            mask_tensor = masks[0][0, best_idx]  # [H, W] bool tensor
            mask_np = mask_tensor.numpy().astype(np.uint8) * 255  # [H, W] uint8

            _, buf = cv2.imencode(".png", mask_np)
            mask_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            logger.info(
                f"SAM2 segmentation succeeded: mask_shape={mask_np.shape}, "
                f"foreground_pixels={int((mask_np > 0).sum())}"
            )
            return {
                "encoding": "png_base64",
                "height": int(mask_np.shape[0]),
                "width": int(mask_np.shape[1]),
                "data": mask_b64,
            }
        except Exception as e:
            logger.error(f"SAM2 segmentation failed: {e}", exc_info=True)
            return None

    def _process_remaining_images_background(
        self,
        images: List[Image.Image],
        image_names: List[str],
        anchors: List[Dict],
        found_image_index: int,
        found_bbox_norm: List[int],
        target_object: str,
        save_results: bool,
        search_timestamp: str,
    ) -> None:
        """
        后台线程：处理剩余图片并保存完整日志
        
        Args:
            images: 所有 PIL Image 对象
            image_names: 对应的图片名称
            anchors: anchor 信息列表
            found_image_index: 找到物体的图片索引
            found_bbox_norm: 找到的 bbox（归一化）
            target_object: 目标物体描述
            save_results: 是否保存结果
            search_timestamp: 搜索时间戳（HHMMSS 格式）
        """
        try:
            logger.info(f"[Background] Starting to process complete log for all {len(images)} images...")
            
            # 初始化完整的处理日志和找到对象的记录
            process_log = []
            found_indices_with_bboxes = {}  # {img_idx: bbox_norm}
            process_log.append("\n" + "█" * 100)
            process_log.append(f"Search Task Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            process_log.append(f"Target Object: {target_object}")
            process_log.append(f"Total Images Received: {len(images)}")
            process_log.append("█" * 100)
            
            # 记录第一次找到的对象
            found_indices_with_bboxes[found_image_index] = found_bbox_norm
            
            # 处理所有图片并记录完整信息
            for img_idx, image in enumerate(images):
                anchor_id = anchors[img_idx]["anchor_id"] if img_idx < len(anchors) else f"anchor_{img_idx:04d}"
                
                log_line = f"\n{'─'*100}"
                log_line += f"\nProcessing image {img_idx + 1}/{len(images)}: {image_names[img_idx]} | Anchor: {anchor_id}"
                log_line += f"\nImage size: {image.size[0]}x{image.size[1]}"
                logger.info(log_line)
                process_log.append(log_line)
                
                if img_idx < found_image_index:
                    # 这是在快速扫描时已处理但未找到的图片
                    log_line = f"  Object not found in this image"
                    logger.info(log_line)
                    process_log.append(log_line)
                    
                elif img_idx == found_image_index:
                    # 这是找到的那张
                    image_width, image_height = image.size
                    bbox_pixel = denormalize_bbox(found_bbox_norm, image_width, image_height)
                    
                    log_line = f"✓ FOUND!"
                    log_line += f"\n  Bbox (pixel): {bbox_pixel}"
                    log_line += f"\n  Bbox (normalized): {found_bbox_norm}"
                    log_line += f"\n  [Already returned to client immediately]"
                    logger.info(log_line)
                    process_log.append(log_line)
                    
                else:  # img_idx > found_image_index
                    # 这是后续处理的图片
                    process_result = self._process_single_image(image, img_idx, target_object)
                    
                    if process_result["found"]:
                        # 记录额外找到的对象
                        found_indices_with_bboxes[img_idx] = process_result['bbox_norm']
                        
                        log_line = f"✓ Also found!"
                        log_line += f"\n  Bbox (normalized): {process_result['bbox_norm']}"
                        log_line += f"\n  [Not returned to client, already found in image {found_image_index + 1}]"
                        logger.info(log_line)
                        process_log.append(log_line)
                    else:
                        log_line = f"  Object not found in this image"
                        logger.info(log_line)
                        process_log.append(log_line)
            
            # 生成最终结果对象用于日志
            found_image_width, found_image_height = images[found_image_index].size
            final_result = {
                "success": True,
                "object_found": True,
                "anchor_id": anchors[found_image_index]["anchor_id"] if found_image_index < len(anchors) else f"anchor_{found_image_index:04d}",
                "bbox": denormalize_bbox(found_bbox_norm, found_image_width, found_image_height),
                "target_object": target_object,
                "num_images_searched": found_image_index + 1,
                "total_images_received": len(images),
                "image_width": found_image_width,
                "image_height": found_image_height,
            }
            
            # 保存检测结果（如果启用）
            if save_results:
                try:
                    # 创建本次搜索的专属文件夹：search_HHMMSS_物体名
                    request_save_dir = self.server_results_dir / f"search_{search_timestamp}_{target_object}"
                    
                    save_detection_results(
                        save_dir=str(request_save_dir),
                        instruction=target_object,
                        images=images,  # 传递所有图片
                        anchors=anchors,  # 传递所有 anchor 信息
                        found=True,
                        found_image_index=found_image_index,
                        bbox_norm=found_bbox_norm,
                        found_indices_with_bboxes=found_indices_with_bboxes,  # 传递所有找到的对象！
                    )
                    log_line = f"\nDetection results saved to: {request_save_dir}"
                    logger.info(log_line)
                    process_log.append(log_line)
                except Exception as e:
                    logger.error(f"[Background] Failed to save results: {e}")
                    process_log.append(f"Failed to save results: {e}")
            
            # 保存处理日志
            self._save_process_log(target_object, process_log, final_result)
            
            process_log.append("█" * 100)
            process_log.append(f"Search Task Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            logger.info("[Background] Complete log saved")
            
        except Exception as e:
            logger.error(f"[Background] Error in background processing: {e}", exc_info=True)
    
    def search(
        self,
        anchors: List[Dict],
        image_data: List[bytes],
        target_object: str,
        save_results_override: Optional[bool] = None,
    ) -> Dict:
        """
        执行物体搜索 - 找到后立即返回结果给client，同时后台继续处理剩余图片
        
        Args:
            anchors: anchor 信息列表，每个包含 {"anchor_id": "...", "image_name": "..."}
            image_data: 图片二进制数据列表 (JPEG bytes)
            target_object: 目标物体描述
            save_results_override: 覆盖保存设置
            
        Returns:
            搜索结果字典:
            {
                "success": true/false,       # 搜索是否成功执行（不代表一定找到物体，除非发生错误）
                "found": true/false,         # 是否找到物体（与 object_found 向后兼容）
                "object_found": true/false,  # 是否找到物体
                "image_index": N,            # 找到时该物体所在图片的索引，未找到时为null
                "anchor_id": "anchor_0003",  # 找到时使用该anchor的ID，未找到时为null
                "bbox": [x1, y1, x2, y2],    # 像素坐标（与该张图尺寸相对应），未找到时为null
                "target_object": "...",
                "num_images_searched": N,    # 直到找到时已搜索的图片数
                "total_images_received": N,  # 本次请求接收的总图片数
                "image_width": W,            # 找到物体时该图的宽度
                "image_height": H,           # 找到物体时该图的高度
                "mask": mask_payload         # 分割 mask（如果生成）
            }
        """
        logger.info(f"Starting search for: {target_object}")
        logger.info(f"Total images received: {len(image_data)}")
        
        # 解码所有图片
        images = []
        image_names = []
        for i, img_bytes in enumerate(image_data):
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(img)
                image_names.append(anchors[i].get("image_name", f"image_{i}.jpg") if i < len(anchors) else f"image_{i}.jpg")
            except Exception as e:
                logger.error(f"Failed to decode image {i}: {e}")
                return {
                    "error": f"Failed to decode image {i}",
                    "success": False
                }
        
        if not images:
            logger.error("No valid images decoded")
            return {
                "error": "No valid images could be decoded",
                "success": False
            }
        
        # ============ FAST PATH: 逐张处理直到找到 ============
        logger.info("Scanning images for target object...")
        
        for img_idx, image in enumerate(images):
            logger.info(f"Processing image {img_idx + 1}/{len(images)}: {image_names[img_idx]} (size: {image.size[0]}x{image.size[1]})")
            
            process_result = self._process_single_image(image, img_idx, target_object)
            
            if process_result["found"]:
                # ✓ FOUND! 立即返回给 client
                logger.info(f"✓ Object found in image {img_idx + 1}! Returning to client...")
                
                image_width, image_height = image.size
                bbox_norm = process_result["bbox_norm"]
                bbox_pixel = denormalize_bbox(bbox_norm, image_width, image_height)
                anchor_id = anchors[img_idx]["anchor_id"] if img_idx < len(anchors) else f"anchor_{img_idx:04d}"
                
                # 使用 SAM2 生成分割 mask
                mask_payload = self._segment_with_sam2(image, bbox_pixel)
                
                result = {
                    "success": True,
                    "found": True,
                    "object_found": True,  # 向后兼容
                    "anchor_id": anchor_id,
                    "image_index": img_idx,
                    "bbox": bbox_pixel,  # 像素坐标
                    "target_object": target_object,
                    "num_images_searched": img_idx + 1,  # 已搜索的图片数
                    "total_images_received": len(images),
                    "image_width": image_width,
                    "image_height": image_height,
                }
                if mask_payload is not None:
                    result["mask"] = mask_payload
                
                # ============ BACKGROUND TASK: 继续处理剩余图片并保存日志 ============
                should_save = save_results_override if save_results_override is not None else self.save_results
                
                # 生成搜索时间戳（用于结果保存）
                search_timestamp = datetime.now().strftime("%H%M%S")
                
                if img_idx + 1 < len(images):
                    # 还有剩余图片，创建后台线程处理
                    logger.info(f"Starting background thread to process remaining {len(images) - img_idx - 1} images...")
                    background_thread = threading.Thread(
                        target=self._process_remaining_images_background,
                        args=(
                            images,
                            image_names,
                            anchors,
                            img_idx,
                            bbox_norm,
                            target_object,
                            should_save,
                            search_timestamp,
                        ),
                        daemon=True,
                        name=f"bg_search_{target_object}_{img_idx}"
                    )
                    background_thread.start()
                else:
                    # 这是最后一张，直接保存日志（无需后台线程）
                    logger.info("No remaining images, saving log directly...")
                    
                    # 完整记录所有图片（包括找到的这一张）
                    process_log = []
                    process_log.append("\n" + "█" * 100)
                    process_log.append(f"Search Task Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    process_log.append(f"Target Object: {target_object}")
                    process_log.append(f"Total Images Received: {len(images)}")
                    process_log.append("█" * 100)
                    
                    for idx, img_name in enumerate(image_names):
                        anchor_id = anchors[idx]["anchor_id"] if idx < len(anchors) else f"anchor_{idx:04d}"
                        log_line = f"\n{'─'*100}"
                        log_line += f"\nProcessing image {idx + 1}/{len(images)}: {img_name} | Anchor: {anchor_id}"
                        log_line += f"\nImage size: {images[idx].size[0]}x{images[idx].size[1]}"
                        
                        if idx == img_idx:
                            # 这是找到的那张
                            log_line += f"\n✓ FOUND!"
                            log_line += f"\n  Bbox (pixel): {bbox_pixel}"
                            log_line += f"\n  Bbox (normalized): {bbox_norm}"
                        else:
                            # 其他图片都是未找到
                            log_line += f"\n  Object not found in this image"
                        
                        process_log.append(log_line)
                    
                    # 保存检测结果（如果启用）
                    if should_save:
                        try:
                            # 创建本次搜索的专属文件夹：search_HHMMSS_物体名
                            request_save_dir = self.server_results_dir / f"search_{search_timestamp}_{target_object}"
                            
                            save_detection_results(
                                save_dir=str(request_save_dir),
                                instruction=target_object,
                                images=images,
                                anchors=anchors,
                                found=True,
                                found_image_index=img_idx,
                                bbox_norm=bbox_norm,
                            )
                            log_line = f"\nDetection results saved to: {request_save_dir}"
                            logger.info(log_line)
                            process_log.append(log_line)
                        except Exception as e:
                            logger.error(f"Failed to save results: {e}")
                            process_log.append(f"Failed to save results: {e}")
                    
                    self._save_process_log(target_object, process_log, result)
                
                # 立即返回结果给 client（不等待后台线程）
                return result
        
        # ============ 如果所有图片都处理完都没找到 ============
        logger.info(f"Object not found in any of the {len(images)} images")
        
        # 生成搜索时间戳
        search_timestamp = datetime.now().strftime("%H%M%S")
        
        # 保存 not found 的日志
        process_log = [
            "\n" + "█" * 100,
            f"Search Task Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Target Object: {target_object}",
            f"Total Images Received: {len(images)}",
            "█" * 100,
        ]
        
        for img_idx, image_name in enumerate(image_names):
            anchor_id = anchors[img_idx]["anchor_id"] if img_idx < len(anchors) else f"anchor_{img_idx:04d}"
            log_line = f"\n{'─'*100}"
            log_line += f"\nProcessing image {img_idx + 1}/{len(images)}: {image_name} | Anchor: {anchor_id}"
            log_line += f"\nImage size: {images[img_idx].size[0]}x{images[img_idx].size[1]}"
            log_line += "\n  Object not found in this image"
            process_log.append(log_line)
        
        process_log.append(f"\n✗ Object not found in any of the {len(images)} images")
        
        result = {
            "success": True,
            "found": False,
            "object_found": False,  # 向后兼容
            "anchor_id": None,
            "bbox": None,
            "score": 1.0,  # 预留一个score字段，表示物体置信度（目前都默认为1.0）
            "target_object": target_object,
            "num_images_searched": len(images),
            "total_images_received": len(images),
            "image_width": None,
            "image_height": None,
        }
        
        # 保存结果图片（如果启用）
        should_save = save_results_override if save_results_override is not None else self.save_results
        if should_save:
            try:
                # 创建本次搜索的专属文件夹：search_HHMMSS_物体名
                request_save_dir = self.server_results_dir / f"search_{search_timestamp}_{target_object}"
                
                save_detection_results(
                    save_dir=str(request_save_dir),
                    instruction=target_object,
                    images=images,  # 传递所有图片
                    anchors=anchors,  # 传递所有 anchor 信息
                    found=False,
                    found_image_index=None,
                    bbox_norm=None,
                )
                log_line = f"Detection results saved to: {request_save_dir}"
                logger.info(log_line)
                process_log.append(f"\n{log_line}")
            except Exception as e:
                logger.error(f"Failed to save results: {e}")
                process_log.append(f"\nFailed to save results: {e}")
        
        self._save_process_log(target_object, process_log, result)
        
        return result
    
    def _save_process_log(self, target_object: str, process_log: List[str], result: Dict) -> None:
        """保存处理过程日志到 Server 统一日志文件"""
        try:
            with open(self.server_log_path, "a", encoding="utf-8") as f:
                # 写入处理过程
                for line in process_log:
                    f.write(line + "\n")
                
                # 写入最终结果
                f.write("\n" + "=" * 100 + "\n")
                f.write("Final Result:\n")
                f.write("=" * 100 + "\n")
                f.write(json.dumps(result, ensure_ascii=False, indent=2))
                f.write("\n\n")
            
            logger.info(f"Process log appended to: {self.server_log_path}")
        except Exception as e:
            logger.error(f"Failed to save process log: {e}")
    
    def handle_request(self) -> None:
        """处理单个客户端请求"""
        try:
            # 接收 multipart 消息
            message_parts = self.socket.recv_multipart()
            
            if len(message_parts) < 2:
                logger.error(f"Invalid message format: expected at least 2 parts, got {len(message_parts)}")
                self.socket.send(json.dumps({
                    "error": "Invalid message format",
                    "success": False
                }).encode("utf-8"))
                return
            
            # 解析 metadata (第一部分)
            try:
                metadata = json.loads(message_parts[0].decode("utf-8"))
            except Exception as e:
                logger.error(f"Failed to parse metadata: {e}")
                self.socket.send(json.dumps({
                    "error": f"Failed to parse metadata: {e}",
                    "success": False
                }).encode("utf-8"))
                return
            
            # 提取关键信息
            instruction = metadata.get("instruction", "")
            anchors = metadata.get("anchors", [])
            save_results = metadata.get("save_results", self.save_results)
            
            if not instruction:
                logger.error("Missing instruction in metadata")
                self.socket.send(json.dumps({
                    "error": "Missing instruction in metadata",
                    "success": False
                }).encode("utf-8"))
                return
            
            # 收集图片数据 (message_parts[1:])
            image_data = message_parts[1:]
            
            logger.info(f"Received request: instruction='{instruction}', num_images={len(image_data)}, num_anchors={len(anchors)}")
            
            # 执行搜索
            result = self.search(
                anchors=anchors,
                image_data=image_data,
                target_object=instruction,
                save_results_override=save_results,
            )
            
            # 发送响应
            response_json = json.dumps(result, ensure_ascii=False)
            self.socket.send(response_json.encode("utf-8"))
            
            logger.info(f"Response sent: {response_json}")
            
        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            self.socket.send(json.dumps({
                "error": str(e),
                "success": False
            }).encode("utf-8"))
    
    def start(self, num_workers: int = 1) -> None:
        """
        启动服务器
        
        Args:
            num_workers: 处理请求的循环次数 (-1 表示无限循环)
        """
        logger.info(f"Starting VLM Search Server (num_workers={num_workers})")
        
        try:
            if num_workers == -1:
                # 无限循环
                while True:
                    self.handle_request()
            else:
                # 处理指定数量的请求
                for i in range(num_workers):
                    logger.info(f"Handling request {i+1}/{num_workers}")
                    self.handle_request()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            self.close()
    
    def close(self) -> None:
        """关闭服务器"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        logger.info("Server closed")


# ============ 命令行接口 ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VLM Search Server")
    parser.add_argument("--port", type=int, default=5555, help="Server port (default: 5555)")
    parser.add_argument("--model-path", type=str, default="/home/zjy/RynnBrain/models/RynnBrain-8B", help="RynnBrain model path")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--cuda-devices", type=str, default="0", help="Visible CUDA devices (default: 0). Use '0,1,2' for multiple GPUs")
    parser.add_argument("--save-results", action="store_true", help="Enable saving detection results to result_images/server_*/search_*/ folders")
    parser.add_argument("--num-workers", type=int, default=-1, help="Number of requests to handle (-1 for infinite)")
    parser.add_argument("--sam2-model-id", type=str, default="facebook/sam2.1-hiera-small", help="SAM2 model ID (default: facebook/sam2.1-hiera-small)")
    parser.add_argument("--sam2-cache-dir", type=str, default=None, help="SAM2 model cache directory (default: .model_cache/<model_id>)")
    parser.add_argument("--sam2-device", type=str, default=None, help="SAM2 inference device (default: cuda if available)")
    
    args = parser.parse_args()
    
    server = VLMSearchServer(
        server_port=args.port,
        model_path=args.model_path,
        device=args.device,
        cuda_devices=args.cuda_devices,
        save_results=args.save_results,
        sam2_model_id=args.sam2_model_id,
        sam2_cache_dir=args.sam2_cache_dir,
        sam2_device=args.sam2_device,
    )
    
    server.start(num_workers=args.num_workers)
