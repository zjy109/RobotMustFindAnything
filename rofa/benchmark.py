import os
import re
import json
import torch
import cv2
import numpy as np
import shutil
import time
import logging
from tqdm import tqdm
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
DATASET_ROOT = "../RoFA-SemEval/dataset"  # 数据集根目录
SAMPLES_JSON_PATH = os.path.join(DATASET_ROOT, "samples.json")
MAP_DIR = "./map"                 # 复制构建的 map 文件夹
PROCESS_LOG_PATH = "process.log"  # 逐样本处理日志输出路径
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

def compute_2d_iou(box1: List[float], box2: List[float]) -> float:
    if not box1 or not box2: return 0.0
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - intersection_area
    if union_area <= 0:
        return 0.0
        
    return intersection_area / union_area

def compute_3d_iou(box1: List[float], box2: List[float]) -> float:
    if not box1 or not box2: return 0.0
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
    cleaned_points, _ = denoise_pointcloud(points_3d)
    
    if cleaned_points.shape[0] == 0:
        return None
    
    min_pt = cleaned_points.min(axis=0)
    max_pt = cleaned_points.max(axis=0)
    
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


# ================= 模型类定义 =================
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import AutoModelForMaskGeneration as Sam2Model
from transformers import AutoProcessor as Sam2Processor

class RynnBrainDetector:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", device_map=device)
        self.model.eval() 
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
            self.model.eval()
            outputs = self.model(**inputs)

        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks.cpu(),
            original_sizes=inputs.original_sizes.cpu(),
            reshaped_input_sizes=inputs.reshaped_input_sizes.cpu()
        )[0]
        
        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            best_idx = int(outputs.iou_scores[0, 0].argmax().item())
        else:
            best_idx = 0
            
        best_mask = masks[0][best_idx].numpy() > 0
        return best_mask


# ================= 数据集映射构建 =================
def build_map_directory() -> Dict[str, List[Dict[str, Any]]]:
    """
    根据 samples.json 构建 map 目录，返回按 class_name_zh 分组的字典。
    """
    if not os.path.exists(SAMPLES_JSON_PATH):
        raise FileNotFoundError(f"未找到 {SAMPLES_JSON_PATH}")

    with open(SAMPLES_JSON_PATH, 'r', encoding='utf-8') as f:
        samples_info = json.load(f)

    os.makedirs(MAP_DIR, exist_ok=True)
    
    mapped_data_by_class = {}
    anchor_idx = 0
    
    print("正在构建/验证 Map 目录和数据集映射...")
    for item in tqdm(samples_info, desc="构建 Map 映射"):
        class_zh = item.get("class_name_zh", item.get("class_name"))
        if class_zh not in mapped_data_by_class:
            mapped_data_by_class[class_zh] = []
            
        anchor_dir_name = f"anchor_{anchor_idx:04d}"
        anchor_dir = os.path.join(MAP_DIR, anchor_dir_name)
        os.makedirs(anchor_dir, exist_ok=True)
        
        src_sample_dir = os.path.join(DATASET_ROOT, item['sample_dir'])
        
        # 复制必需文件
        files_to_copy = ['rgb.jpg', 'depth.png', 'intrinsics.json', 'pose.txt']
        for file_name in files_to_copy:
            src_file = os.path.join(src_sample_dir, file_name)
            dst_file = os.path.join(anchor_dir, file_name)
            if os.path.exists(src_file) and not os.path.exists(dst_file):
                shutil.copy(src_file, dst_file)
                
        new_item = item.copy()
        new_item['anchor_dir'] = anchor_dir
        new_item['src_sample_dir'] = src_sample_dir
        mapped_data_by_class[class_zh].append(new_item)
        
        anchor_idx += 1
        
    return mapped_data_by_class


# ================= 主流程 =================
def main():
    setup_cuda_devices(CUDA_DEVICES)
    
    # 初始化日志记录
    logging.basicConfig(
        filename=PROCESS_LOG_PATH, 
        filemode='w', 
        format='%(message)s', 
        level=logging.INFO
    )

    if SAVE_VISUALIZATIONS:
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    # 1. 构建目录映射
    mapped_data_by_class = build_map_directory()

    print("\n加载模型中，请稍候...")
    rynn_brain = RynnBrainDetector(model_path=RYNNBRAIN_MODEL_PATH)
    sam2_segmenter = SAM2Segmenter(model_id=SAM2_MODEL_ID, sam2_cache_dir=SAM2_CACHE_DIR_GLOBAL)

    # 统计相关
    results_by_class = {}
    failed_cases_global = []
    global_success_count = 0
    global_total_count = sum(len(items) for items in mapped_data_by_class.values())
    current_processed = 0

    print(f"\n开始评测总样本数：{global_total_count}")
    
    # 设置主进度条
    pbar = tqdm(total=global_total_count, desc="总体处理进度", unit="sample")

    for class_zh, samples in mapped_data_by_class.items():
        results_by_class[class_zh] = {
            "samples": [],
            "stats": {}
        }
        class_success = 0
        class_2d_ious = []
        class_3d_ious = []

        for item in samples:
            sample_id = item['sample_id']
            anchor_dir = item['anchor_dir']
            src_sample_dir = item['src_sample_dir']
            
            # Map 中的路径
            rgb_path = os.path.join(anchor_dir, "rgb.jpg")
            depth_path = os.path.join(anchor_dir, "depth.png")
            intrinsics_path = os.path.join(anchor_dir, "intrinsics.json")
            
            # 原始数据集获取 GT
            aabb_path = os.path.join(src_sample_dir, "aabb.json")
            
            missing_files = [f for f in [rgb_path, depth_path, intrinsics_path, aabb_path] if not os.path.exists(f)]
            if missing_files:
                logging.info(f"Progress: {current_processed+1}/{global_total_count} | [{sample_id}] Error: 缺失文件 {missing_files} 跳过。")
                pbar.update(1)
                current_processed += 1
                continue

            # 载入数据
            with open(aabb_path, "r", encoding="utf-8") as f:
                aabb_data = json.load(f)
                gt_3d_bbox = aabb_data["min"] + aabb_data["max"]
                # 读取 2D bbox GT
                gt_2d_bbox = aabb_data.get("bbox_2d", {}).get("from_mask", [0, 0, 0, 0])

            with open(intrinsics_path, "r", encoding="utf-8") as f:
                intrinsics = json.load(f)

            rgb_img = Image.open(rgb_path).convert("RGB")
            img_width, img_height = rgb_img.size
            depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) 

            # 初始化变量
            is_correct = False
            pred_2d_bbox_pixel = None
            pred_3d_bbox = None
            iou_2d = 0.0
            iou_3d = 0.0

            time_rynn = time_sam = time_3d = 0.0

            # 预测流程与计时
            t_start = time.time()
            bbox_norm = rynn_brain.process_image(rgb_img, class_zh)
            time_rynn = time.time() - t_start
            
            if bbox_norm is not None:
                pred_2d_bbox_pixel = denormalize_bbox(bbox_norm, img_width, img_height)
                iou_2d = float(compute_2d_iou(pred_2d_bbox_pixel, gt_2d_bbox))

                t_sam_start = time.time()
                mask = sam2_segmenter.segment(rgb_img, pred_2d_bbox_pixel)
                time_sam = time.time() - t_sam_start
                
                t_3d_start = time.time()
                pred_3d_bbox = mask_to_3d_aabb(depth_map, mask, intrinsics)
                time_3d = time.time() - t_3d_start
                
                if pred_3d_bbox is not None:
                    iou_3d = float(compute_3d_iou(pred_3d_bbox, gt_3d_bbox))
                    is_correct = bool(iou_3d >= IOU_THRESHOLD_3D)

                    if SAVE_VISUALIZATIONS:
                        save_path = os.path.join(VISUALIZATION_DIR, f"det_seg_{sample_id}.jpg")
                        save_overlay(np.array(rgb_img), pred_2d_bbox_pixel, mask, save_path)
            
            # 数据累加
            current_processed += 1
            if is_correct:
                global_success_count += 1
                class_success += 1
            else:
                failed_cases_global.append(sample_id)
                
            class_2d_ious.append(iou_2d)
            class_3d_ious.append(iou_3d)

            # 写入日志文件
            log_str = (
                f"Progress: {current_processed}/{global_total_count} | "
                f"Sample: {sample_id} | Class: {class_zh} | "
                f"2D IoU: {iou_2d:.4f} | 3D IoU: {iou_3d:.4f} | Correct: {is_correct} | "
                f"Times: Rynn={time_rynn:.2f}s, SAM2={time_sam:.2f}s, 3D={time_3d:.2f}s"
            )
            logging.info(log_str)

            # 写入结果记录
            results_by_class[class_zh]["samples"].append({
                "sample_id": sample_id,
                "class_name": item["class_name"],
                "object_prompt": class_zh,
                "pred_2d_bbox": pred_2d_bbox_pixel,
                "gt_2d_bbox": gt_2d_bbox,
                "2d_iou": round(iou_2d, 4),
                "pred_3d_bbox": pred_3d_bbox,
                "gt_3d_bbox": gt_3d_bbox,
                "3d_iou": round(iou_3d, 4),
                "is_correct": is_correct
            })

            # 更新控制台进度条
            current_acc = global_success_count / current_processed if current_processed > 0 else 0
            pbar.set_postfix({"Current Acc": f"{current_acc:.2%}"})
            pbar.update(1)

        # 当前类别处理结束，汇总该类别数据
        total_class_samples = len(class_2d_ious)
        if total_class_samples > 0:
            avg_2d = sum(class_2d_ious) / total_class_samples
            avg_3d = sum(class_3d_ious) / total_class_samples
            acc_3d = class_success / total_class_samples
        else:
            avg_2d = avg_3d = acc_3d = 0.0

        results_by_class[class_zh]["stats"] = {
            "total_samples": total_class_samples,
            "avg_2d_iou": round(avg_2d, 4),
            "avg_3d_iou": round(avg_3d, 4),
            "accuracy_3d": round(acc_3d, 4)
        }

        # ================= 新增：每处理完一个类别后，实时保存当前的 JSON 结果 =================
        current_overall_accuracy = float(global_success_count / current_processed if current_processed > 0 else 0.0)
        current_output = {
            "sample_results": results_by_class,
            "summary": {
                "total_processed": current_processed,  # 当前已处理的总数
                "total_target": global_total_count,    # 预计要处理的总数
                "iou_threshold": IOU_THRESHOLD_3D,
                "success_count": global_success_count,
                "accuracy": round(current_overall_accuracy, 4),
                "failed_cases": failed_cases_global
            }
        }
        with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(current_output, f, ensure_ascii=False, indent=4)

    pbar.close()

    print("\n" + "="*50)
    print("Testing Complete!")
    print(f"Total Samples Processed: {current_processed} / {global_total_count}")
    
    final_accuracy = float(global_success_count / global_total_count if global_total_count > 0 else 0.0)
    print(f"Overall Accuracy (IoU >= {IOU_THRESHOLD_3D}): {final_accuracy:.2%}")
    print(f"Final Results saved to {RESULTS_JSON_PATH}")
    print(f"Processing logs saved to {PROCESS_LOG_PATH}")
    if SAVE_VISUALIZATIONS:
        print(f"Visualizations saved to {VISUALIZATION_DIR}/")
    print("="*50)

if __name__ == "__main__":
    main()