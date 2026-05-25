import os
import re
import json
import torch
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from PIL import Image
from pathlib import Path

# ================= 全局配置 =================
# 硬件设备配置
CUDA_DEVICES = "1"  # 设置可见的 CUDA 设备

# RynnBrain 配置
RYNNBRAIN_MODEL_PATH = "/home/zjy/RynnBrain/models/RynnBrain-8B"
ENABLE_EXISTENCE_CHECK = True  # 是否在定位前进行存在性判断

# SAM2 配置
SAM2_MODEL_ID = "facebook/sam2.1-hiera-small"
SAM2_CACHE_DIR_GLOBAL = None  

# 评估配置
IOU_THRESHOLD_3D = 0.25

# 存储与可视化配置
# DATASET_DIR = "../annotated"
DATASET_DIR = "..//RoFA-SemEval/dataset/samples"
SAVE_VISUALIZATIONS = True
VISUALIZATION_DIR = "./det_seg_images"
RESULTS_JSON_PATH = "results.json"

# 点云去噪默认参数
DEFAULT_SOR_NB = 30
DEFAULT_SOR_STD = 2.0
DEFAULT_DBSCAN_EPS = 0.03
DEFAULT_DBSCAN_MIN_POINTS = 50


# ================= 辅助函数 =================
def setup_cuda_devices(cuda_devices: str = "0") -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    print(f"Set CUDA_VISIBLE_DEVICES={cuda_devices}")

def denormalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    return [
        round(x1 / 1000 * width), round(y1 / 1000 * height),
        round(x2 / 1000 * width), round(y2 / 1000 * height)
    ]

def compute_3d_iou(box1: List[float], box2: List[float]) -> float:
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    z_front = max(box1[2], box2[2])
    x_right = min(box1[3], box2[3])
    y_bottom = min(box1[4], box2[4])
    z_back = min(box1[5], box2[5])

    if x_right < x_left or y_bottom < y_top or z_back < z_front:
        return 0.0

    intersection_vol = (x_right - x_left) * (y_bottom - y_top) * (z_back - z_front)
    vol1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    vol2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    
    union_vol = vol1 + vol2 - intersection_vol
    if union_vol <= 0:
        return 0.0
        
    return intersection_vol / union_vol

# --- 点云去噪逻辑 ---
def _numpy_fallback_denoise(points_cam: np.ndarray, info: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """由于未提供具体实现，这里做一个简单的容错 Fallback，直接返回原点云"""
    info["method"] = "numpy_fallback (no-op)"
    info["fallback_warning"] = "Open3D not installed. Returning raw points."
    return points_cam.astype(np.float32), info

def denoise_pointcloud(
    points_cam: np.ndarray,
    sor_nb: int = DEFAULT_SOR_NB,
    sor_std: float = DEFAULT_SOR_STD,
    dbscan_eps: float = DEFAULT_DBSCAN_EPS,
    dbscan_min_points: int = DEFAULT_DBSCAN_MIN_POINTS,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """返回 (cleaned_points, info_dict)。优先使用 open3d；不可用则 fallback。"""
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

    # 1) 统计离群点 (SOR)
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

def mask_to_3d_aabb(depth_map: np.ndarray, mask: np.ndarray, intrinsics: Dict[str, float]) -> Optional[List[float]]:
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    depth_scale = intrinsics.get("depth_scale", 0.001)

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None

    depths = depth_map[ys, xs] * depth_scale
    valid_mask = depths > 0
    xs, ys, depths = xs[valid_mask], ys[valid_mask], depths[valid_mask]

    if len(depths) == 0:
        return None

    X = (xs - cx) * depths / fx
    Y = (ys - cy) * depths / fy
    Z = depths

    points_3d = np.stack((X, Y, Z), axis=-1)
    
    # 引入新增的去噪步骤
    cleaned_points, _ = denoise_pointcloud(points_3d)
    
    if cleaned_points.shape[0] == 0:
        return None
    
    min_pt = cleaned_points.min(axis=0)
    max_pt = cleaned_points.max(axis=0)
    
    # 强制转换为原生 float，避免 JSON 序列化报错
    return [
        float(min_pt[0]), float(min_pt[1]), float(min_pt[2]), 
        float(max_pt[0]), float(max_pt[1]), float(max_pt[2])
    ]

def save_overlay(image_arr: np.ndarray, bbox: List[int], mask: np.ndarray, save_path: str):
    img_cv2 = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
    
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    mask_uint8 = (mask * 255).astype(np.uint8)
    colored_mask = np.zeros_like(img_cv2)
    colored_mask[:, :, 0] = mask_uint8
    cv2.addWeighted(colored_mask, 0.5, img_cv2, 1.0, 0, img_cv2)
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_cv2, contours, -1, (255, 0, 0), 2)

    cv2.imwrite(save_path, img_cv2)


# ================= 模型加载 =================
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import AutoModelForMaskGeneration as Sam2Model
from transformers import AutoProcessor as Sam2Processor

# ================= 模型类定义 =================
class RynnBrainDetector:
    def __init__(self, model_path: str, device: str = "auto"):
        print(f"Loading RynnBrain model from {model_path}...")
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", device_map=device)
        self.model.eval()  # 优化点：显式设置为 eval 模式
        self.processor = AutoProcessor.from_pretrained(model_path)

    def process_image(self, image: Image.Image, target_object: str) -> Optional[List[int]]:
        if ENABLE_EXISTENCE_CHECK:
            existence_instruction = f"Verify if the EXACT '{target_object}' is present in any of these images."
            existence_format = (
                "Use this strict checklist:\n"
                "- Color matches exactly? (If no -> 'No')\n"
                "- 3D Shape matches exactly? (If flat instead of 3D -> 'No')\n"
                "- Category matches exactly? (If look-alike -> 'No')\n"
                "Answer 'No' if ANY check fails. Output exactly one word: 'Yes' or 'No'."
            )
            
            messages_exist = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{existence_instruction}\n{existence_format}"}
            ]}]
            
            inputs_exist = self.processor.apply_chat_template(
                messages_exist, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(self.model.device)
            
            with torch.no_grad():
                resp_exist = self.model.generate(**inputs_exist, do_sample=False, max_new_tokens=128)
            
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_exist.input_ids, resp_exist)]
            answer = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            if "yes" not in answer.lower():
                return None

        bbox_instruction = f"Localize the EXACT '{target_object}' in the images."
        bbox_format_prompt = (
            "RULES:\n"
            "1. MUST match exact color, 3D shape, and category.\n"
            "2. Tightly enclose ONLY the true target.\n"
            "Generate coordinates for exactly one object. x1,y1,x2,y2 ∈ [0,1000].\n"
            "Output strictly: <object> (x1, y1), (x2, y2) </object>"
        )
        
        messages_loc = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"{bbox_instruction}\n{bbox_format_prompt}"}
        ]}]
        
        inputs_loc = self.processor.apply_chat_template(
            messages_loc, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            # 优化点：使用 do_sample=False (贪心解码) 消除不稳定输出和参数警告
            resp_loc = self.model.generate(
                **inputs_loc, max_new_tokens=128, do_sample=False
            )
            
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_loc.input_ids, resp_loc)]
        answer_loc = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        pattern = re.compile(r"<object>.*?\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\).*?</object>", re.DOTALL)
        match = pattern.search(answer_loc)
        if match:
            return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
        return None


class SAM2Segmenter:
    def __init__(self, model_id: str, sam2_cache_dir: Optional[str] = None):
        self.sam2_model_id = str(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.sam2_cache_dir = Path(sam2_cache_dir) if sam2_cache_dir else self.default_sam2_cache_dir(self.sam2_model_id)
        self.sam2_cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading SAM2 model: {self.sam2_model_id}")
        self.model = Sam2Model.from_pretrained(
            self.sam2_model_id,
            cache_dir=str(self.sam2_cache_dir)
        ).to(self.device)
        
        self.processor = Sam2Processor.from_pretrained(
            self.sam2_model_id,
            cache_dir=str(self.sam2_cache_dir)
        )

    @staticmethod
    def default_sam2_cache_dir(model_id: str) -> Path:
        cache_name = str(model_id).replace("/", "--")
        return Path(__file__).resolve().parent / ".model_cache" / cache_name

    def segment(self, image: Image.Image, bbox_pixel: List[int]) -> np.ndarray:
        inputs = self.processor(
            images=image, 
            input_boxes=[[bbox_pixel]], 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            self.model.eval()  # 优化点：确保推理模式
            outputs = self.model(**inputs)

        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks.cpu(),
            original_sizes=inputs.original_sizes.cpu(),
            reshaped_input_sizes=inputs.reshaped_input_sizes.cpu()
        )[0]
        
        # 优化点：根据 iou_scores 动态挑选质量最好的掩码
        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            best_idx = int(outputs.iou_scores[0, 0].argmax().item())
        else:
            best_idx = 0
            
        best_mask = masks[0][best_idx].numpy() > 0
        return best_mask


# ================= 主流程 =================
def main():
    setup_cuda_devices(CUDA_DEVICES)

    if SAVE_VISUALIZATIONS:
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    rynn_brain = RynnBrainDetector(model_path=RYNNBRAIN_MODEL_PATH)
    sam2_segmenter = SAM2Segmenter(model_id=SAM2_MODEL_ID, sam2_cache_dir=SAM2_CACHE_DIR_GLOBAL)

    results_data = []
    failed_cases = []
    success_count = 0
    total_count = 0

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset root directory not found: {DATASET_DIR}")

    for class_name in os.listdir(DATASET_DIR):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir): 
            continue
            
        for sample_id in os.listdir(class_dir):
            sample_dir = os.path.join(class_dir, sample_id)
            if not os.path.isdir(sample_dir): 
                continue

            print(f"\n--- Processing Sample {sample_id} ({class_name}) ---")
            
            rgb_path = os.path.join(sample_dir, "rgb.jpg")
            depth_path = os.path.join(sample_dir, "depth.png")
            aabb_path = os.path.join(sample_dir, "aabb.json")
            intrinsics_path = os.path.join(sample_dir, "intrinsics.json")
            sample_meta_path = os.path.join(sample_dir, "sample.json")
            
            missing_files = [f for f in [rgb_path, depth_path, aabb_path, intrinsics_path, sample_meta_path] if not os.path.exists(f)]
            if missing_files:
                print(f"[{sample_id}] Missing essential files: {missing_files}. Skipping.")
                continue

            # 1. 载入 Ground Truth 和 元数据
            with open(aabb_path, "r", encoding="utf-8") as f:
                aabb_data = json.load(f)
                gt_3d_bbox = aabb_data["min"] + aabb_data["max"]

            with open(intrinsics_path, "r", encoding="utf-8") as f:
                intrinsics = json.load(f)

            with open(sample_meta_path, "r", encoding="utf-8") as f:
                sample_info = json.load(f)
                obj_name = sample_info.get("class_name_zh", sample_info.get("class_name", class_name))

            gt_2d_bbox = [0, 0, 0, 0]
            
            # 2. 载入图像
            rgb_img = Image.open(rgb_path).convert("RGB")
            img_width, img_height = rgb_img.size
            depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) 

            total_count += 1
            is_correct = False
            pred_2d_bbox_pixel = None
            pred_3d_bbox = None
            iou_3d = 0.0

            # 3. 模型预测
            bbox_norm = rynn_brain.process_image(rgb_img, obj_name)
            
            if bbox_norm is not None:
                pred_2d_bbox_pixel = denormalize_bbox(bbox_norm, img_width, img_height)
                mask = sam2_segmenter.segment(rgb_img, pred_2d_bbox_pixel)
                
                pred_3d_bbox = mask_to_3d_aabb(depth_map, mask, intrinsics)
                
                if pred_3d_bbox is not None:
                    # 优化点：强制转换为原生类型防止 JSON 报错
                    iou_3d = float(compute_3d_iou(pred_3d_bbox, gt_3d_bbox))
                    is_correct = bool(iou_3d >= IOU_THRESHOLD_3D)

                    if SAVE_VISUALIZATIONS:
                        save_path = os.path.join(VISUALIZATION_DIR, f"det_seg_{sample_id}.jpg")
                        save_overlay(np.array(rgb_img), pred_2d_bbox_pixel, mask, save_path)
                else:
                    print(f"[{sample_id}] Failed to extract valid 3D points from depth map (or removed by denoise).")
            else:
                print(f"[{sample_id}] Object '{obj_name}' not detected by RynnBrain.")

            if is_correct:
                success_count += 1
            else:
                failed_cases.append(sample_id)

            print(f"Result for {sample_id}: IoU = {iou_3d:.4f} | Correct = {is_correct}")

            # 4. 记录数据
            results_data.append({
                "sample_id": sample_id,
                "class_name": class_name,
                "object_prompt": obj_name,
                "pred_2d_bbox": pred_2d_bbox_pixel,
                "gt_2d_bbox": gt_2d_bbox,
                "pred_3d_bbox": pred_3d_bbox,
                "gt_3d_bbox": gt_3d_bbox,
                "3d_iou": round(iou_3d, 4),
                "is_correct": is_correct
            })

    # 3. 汇总统计结果并写入 JSON
    accuracy = float(success_count / total_count if total_count > 0 else 0.0)
    final_output = {
        "sample_results": results_data,
        "summary": {
            "total_processed": total_count,
            "iou_threshold": IOU_THRESHOLD_3D,
            "success_count": success_count,
            "accuracy": round(accuracy, 4),
            "failed_cases": failed_cases
        }
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

    print("\n" + "="*50)
    print("Testing Complete!")
    print(f"Total Samples: {total_count}")
    print(f"Accuracy (IoU >= {IOU_THRESHOLD_3D}): {accuracy:.2%}")
    print(f"Results saved to {RESULTS_JSON_PATH}")
    if SAVE_VISUALIZATIONS:
        print(f"Visualizations saved to {VISUALIZATION_DIR}/")
    print("="*50)

if __name__ == "__main__":
    main()