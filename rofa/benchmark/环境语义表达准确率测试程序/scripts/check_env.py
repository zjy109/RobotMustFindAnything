#!/usr/bin/env python3
"""环境自检：在第三方机器上跑测试前先确认环境就绪。

依次检查：
  1. Python 版本 >= 3.10
  2. 关键 Python 包：torch / transformers / opencv / numpy / PIL / tqdm / open3d
  3. CUDA 是否可用、显存是否够
  4. 模型权重（<项目根>/models/RynnBrain-8B 与 sam2.1-hiera-small；缺失时仅警告，跑测试时会自动下载）
  5. 数据集结构有效（samples.json + 抽样校验单样本必备文件）
  6. 磁盘空间

使用：
    python scripts/check_env.py --dataset /path/to/dataset_1000
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# 让脚本能 import 项目内 src/benchmark
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def status(ok: bool, msg: str, warn: bool = False) -> None:
    if ok and not warn:
        print(f"  {GREEN}✓{RESET} {msg}")
    elif warn:
        print(f"  {YELLOW}!{RESET} {msg}")
    else:
        print(f"  {RED}✗{RESET} {msg}")


# ---------------------------------------------------------------------------
# 检查项
# ---------------------------------------------------------------------------

def check_python() -> bool:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    status(ok, f"Python {v.major}.{v.minor}.{v.micro}  (要求 >= 3.10)")
    return ok


REQUIRED_PACKAGES = [
    ("numpy", None),
    ("cv2", "opencv-python"),
    ("PIL", "Pillow"),
    ("tqdm", None),
    ("torch", None),
    ("transformers", None),
]
OPTIONAL_PACKAGES = [
    ("open3d", "open3d"),
]


def check_packages() -> Tuple[bool, List[str]]:
    print(f"\n{BOLD}[2] 关键依赖{RESET}")
    all_ok = True
    missing: List[str] = []
    for mod_name, pip_name in REQUIRED_PACKAGES:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "?")
            status(True, f"{mod_name}  {ver}")
        except ImportError:
            install = pip_name or mod_name
            status(False, f"{mod_name}  缺失（pip install {install}）")
            missing.append(install)
            all_ok = False

    print(f"\n{BOLD}[3] 可选依赖{RESET}")
    for mod_name, pip_name in OPTIONAL_PACKAGES:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "?")
            status(True, f"{mod_name}  {ver}")
        except ImportError:
            status(True, f"{mod_name}  未安装（点云去噪将退化为直通；建议 pip install {pip_name}）", warn=True)

    return all_ok, missing


def check_cuda() -> bool:
    print(f"\n{BOLD}[4] CUDA / GPU{RESET}")
    try:
        import torch
    except ImportError:
        status(False, "torch 未安装")
        return False

    if not torch.cuda.is_available():
        status(False, "CUDA 不可用！测试程序需要 NVIDIA GPU。")
        print(f"    {YELLOW}如果服务器有显卡，请检查 NVIDIA 驱动 / CUDA toolkit / pytorch CUDA 版本是否匹配。{RESET}")
        return False

    n = torch.cuda.device_count()
    status(True, f"CUDA 可用，设备数 = {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / 1e9
        ok = gb >= 12  # RynnBrain-8B + SAM2 在 fp16 下大约需要 18GB；给 12GB 是底线警告
        if gb < 12:
            status(True, f"  device {i}: {name}  ({gb:.1f} GB)  显存可能不足，建议 >=24GB", warn=True)
        else:
            status(True, f"  device {i}: {name}  ({gb:.1f} GB)")
    return True


def check_models() -> bool:
    """检查 RynnBrain-8B 与 SAM2 是否就绪。

    缺失/不完整不视为致命错误：runner 启动时会自动从 ModelScope/HuggingFace 下载。
    本检查只是给用户一个『要不要先预热』的提示。
    """
    from benchmark.model_resolver import (
        rynnbrain_dir, sam2_dir, is_rynnbrain_ready, is_sam2_ready,
    )

    print(f"\n{BOLD}[5] 模型权重{RESET}")

    rynn_ready = is_rynnbrain_ready()
    sam2_ready = is_sam2_ready()

    if rynn_ready:
        status(True, f"RynnBrain-8B  已就绪: {rynnbrain_dir()}")
    else:
        status(True, f"RynnBrain-8B  未下载（首次跑测试时自动获取，~16GB）", warn=True)

    if sam2_ready:
        status(True, f"SAM2           已就绪: {sam2_dir()}")
    else:
        status(True, f"SAM2           未下载（首次跑测试时自动获取，~200MB）", warn=True)

    if not (rynn_ready and sam2_ready):
        print(f"    {YELLOW}如希望先预热（推荐），运行: python scripts/download_models.py{RESET}")

    return True  # 不阻断


def check_dataset(dataset_root: Path) -> bool:
    print(f"\n{BOLD}[6] 数据集{RESET}")
    if not dataset_root.exists():
        status(False, f"数据集目录不存在: {dataset_root}")
        return False
    samples_json = dataset_root / "samples.json"
    samples_dir = dataset_root / "samples"

    if not samples_json.exists():
        status(False, f"缺 samples.json: {samples_json}")
        return False
    if not samples_dir.exists():
        status(False, f"缺 samples/ 目录: {samples_dir}")
        return False
    status(True, f"samples.json 与 samples/ 都在")

    try:
        with open(samples_json, "r", encoding="utf-8") as f:
            doc = json.load(f)
        items = doc if isinstance(doc, list) else doc.get("samples", [])
        n = len(items)
        status(True, f"samples.json 共 {n} 条")
    except Exception as exc:
        status(False, f"读取 samples.json 失败: {exc}")
        return False

    # 抽 3 个样本看必备文件齐不齐
    REQ = ["rgb.jpg", "depth.png", "intrinsics.json", "aabb.json"]
    sample_ok = True
    for it in items[:3]:
        sd = dataset_root / it.get("sample_dir", "")
        miss = [f for f in REQ if not (sd / f).exists()]
        if miss:
            status(False, f"  抽检 {it.get('sample_id')} 缺: {miss}")
            sample_ok = False
        else:
            status(True, f"  抽检 {it.get('sample_id')} 必备文件齐全")
    return sample_ok


def check_disk(output_dir: Path) -> bool:
    print(f"\n{BOLD}[7] 磁盘空间{RESET}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output_dir.parent).free / 1e9
    if free_gb < 5:
        status(False, f"磁盘剩余 {free_gb:.1f} GB（< 5GB），不够保存可视化和模型缓存")
        return False
    status(True, f"磁盘剩余 {free_gb:.1f} GB")
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="环境自检")
    ap.add_argument("--dataset", type=str, default=None,
                    help="数据集根目录（可选，但强烈建议在跑测试前先校验）")
    ap.add_argument("--output", type=str, default="./run_output",
                    help="输出目录（仅用于检查磁盘）")
    args = ap.parse_args()

    print(f"{BOLD}=== 环境自检 ==={RESET}\n")
    print(f"{BOLD}[1] Python{RESET}")
    py_ok = check_python()
    pkg_ok, missing = check_packages()
    cuda_ok = check_cuda()
    models_ok = check_models()

    ds_ok = True
    if args.dataset:
        ds_ok = check_dataset(Path(args.dataset).expanduser().resolve())
    else:
        print(f"\n{BOLD}[6] 数据集{RESET}")
        status(True, "未传入 --dataset，跳过数据集检查（建议跑测试前再校验一次）", warn=True)

    disk_ok = check_disk(Path(args.output).expanduser().resolve())

    print(f"\n{BOLD}=== 总结 ==={RESET}")
    overall = py_ok and pkg_ok and cuda_ok and models_ok and ds_ok and disk_ok
    if overall:
        print(f"  {GREEN}✓ 所有检查通过，可以运行测试。{RESET}")
        ds_arg = args.dataset or '<dataset_path>'
        print(f"\n  下一步:")
        print(f"    bash scripts/run_benchmark.sh {ds_arg}")
        return 0

    print(f"  {RED}✗ 有检查未通过，请按上方提示修复。{RESET}")
    if missing:
        print(f"\n  快速修复缺失的依赖：")
        print(f"    pip install {' '.join(missing)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
