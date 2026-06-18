#!/usr/bin/env bash
# 环境语义表达准确率测试程序 — 一键环境安装
#
# 适用于 Ubuntu 20.04 / 22.04 + NVIDIA GPU 服务器。
# 创建一个独立的 conda 环境 `rofa-bench`，装好所有 Python 依赖。
#
# 用法：
#   bash scripts/install.sh              # 完整安装
#   bash scripts/install.sh --skip-conda # 已经在某个 conda 环境里，只装 pip 包
#   bash scripts/install.sh --cpu-only   # 只装 CPU 版 PyTorch（一般用于本地调试，
#                                          运行真实测试必须用 GPU 版）
#
set -euo pipefail

# ----- 颜色 -----
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
RESET="\033[0m"
BOLD="\033[1m"

log()  { echo -e "${GREEN}[install]${RESET} $*"; }
warn() { echo -e "${YELLOW}[install]${RESET} $*"; }
err()  { echo -e "${RED}[install]${RESET} $*" >&2; }

# ----- 默认参数 -----
ENV_NAME="rofa-bench"
PYTHON_VERSION="3.10.19"
SKIP_CONDA=0
CPU_ONLY=0
TORCH_CUDA="cu121"   # 与 CUDA 12.1 匹配；若服务器是 CUDA 11.8 改成 cu118

# ----- 解析参数 -----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-conda) SKIP_CONDA=1; shift ;;
        --cpu-only)   CPU_ONLY=1; shift ;;
        --env-name)   ENV_NAME="$2"; shift 2 ;;
        --torch-cuda) TORCH_CUDA="$2"; shift 2 ;;
        -h|--help)
            cat <<EOF
用法: bash scripts/install.sh [OPTIONS]

OPTIONS:
  --skip-conda          跳过 conda 环境创建（已激活某环境时用）
  --cpu-only            安装 CPU 版 PyTorch（仅本地调试用）
  --env-name <NAME>     conda 环境名（默认 rofa-bench）
  --torch-cuda <VER>    PyTorch CUDA 版本：cu121 / cu118（默认 cu121）
  -h, --help            打印此帮助
EOF
            exit 0
            ;;
        *) err "未知参数: $1"; exit 1 ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log "项目根目录: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# ===== Step 1: 系统包（可选） =====
log "${BOLD}[1/4] 系统级依赖（可选 — 如果没有 sudo 权限会跳过）${RESET}"
if command -v apt-get &>/dev/null && [[ $EUID -eq 0 || -n "${SUDO_USER:-}" ]]; then
    sudo apt-get update -y || true
    # python3-tk: ui.py 的 Tkinter 界面需要；libusb-1.0-0: RealSense 采集需要
    sudo apt-get install -y --no-install-recommends \
        build-essential libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        python3-tk libusb-1.0-0 \
        wget curl ca-certificates git || warn "apt-get 部分失败，但通常不致命"
else
    warn "  非 root 或 apt-get 不可用，跳过系统包安装。"
    warn "  如果运行时报 'libGL.so.1' 之类错，请手动安装："
    warn "    sudo apt-get install -y libgl1 libglib2.0-0"
    warn "  如果 ui.py 报 'No module named _tkinter'，请安装：sudo apt-get install -y python3-tk"
    warn "  如果 RealSense 采集报 USB/libusb 相关错，请安装：sudo apt-get install -y libusb-1.0-0"
fi

# ===== Step 2: conda 环境 =====
log "${BOLD}[2/4] Python 环境${RESET}"
if [[ $SKIP_CONDA -eq 0 ]]; then
    if ! command -v conda &>/dev/null; then
        err "未找到 conda；请先安装 Miniconda/Anaconda 后再运行：https://docs.conda.io/projects/miniconda/"
        exit 1
    fi

    # 检查环境是否已存在
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        warn "  conda 环境 '$ENV_NAME' 已存在，复用之"
    else
        log "  创建 conda 环境 '$ENV_NAME' (python=$PYTHON_VERSION)"
        conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
    fi
    # 激活
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    log "  当前环境: $(python -c 'import sys; print(sys.executable)')"
else
    log "  --skip-conda：使用当前 Python: $(which python)"
fi

# ===== Step 3: PyTorch（GPU 或 CPU） =====
log "${BOLD}[3/4] PyTorch${RESET}"
# 重要：torch 的 GPU/CPU wheel 只在 PyTorch 官方源或其镜像上有；
# 不能用 `-i 清华源` 覆盖 --index-url（-i 是 --index-url 的简写，后者会被覆盖，
# 导致从普通 PyPI 镜像拉 torch，常见 403 / 拉到错误的 CUDA 版本）。
# 这里按「官方源 -> 阿里云 PyTorch 镜像」顺序回退。
if [[ $CPU_ONLY -eq 1 ]]; then
    log "  安装 CPU 版 PyTorch（仅本地调试可用，运行测试必须用 GPU）"
    TORCH_VARIANT="cpu"
else
    log "  安装 GPU 版 PyTorch (CUDA $TORCH_CUDA)"
    TORCH_VARIANT="$TORCH_CUDA"
fi

TORCH_SOURCES=(
    "https://download.pytorch.org/whl/${TORCH_VARIANT}"
    "https://mirrors.aliyun.com/pytorch-wheels/${TORCH_VARIANT}"
)
torch_ok=0
for src in "${TORCH_SOURCES[@]}"; do
    log "  尝试 PyTorch 源: $src"
    if pip install --index-url "$src" torch==2.5.1 torchvision==0.20.1; then
        torch_ok=1
        break
    fi
    warn "  该源安装失败，尝试下一个 ..."
done
if [[ $torch_ok -ne 1 ]]; then
    err "PyTorch 安装失败（所有源都失败）。请检查网络，或手动安装后重跑 --skip-conda："
    err "    pip install --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} torch==2.5.1 torchvision==0.20.1"
    exit 1
fi

# 校验 torch 安装
python - <<'PY'
import torch
print(f"  torch       {torch.__version__}")
print(f"  cuda avail  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device 0    {torch.cuda.get_device_name(0)}")
PY

# ===== Step 4: 业务依赖 =====
log "${BOLD}[4/4] 业务 Python 依赖${RESET}"
# 主用清华源，腾讯云源作为 extra（个别大 wheel 某个镜像 403 时可自动换源）。
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_EXTRA="https://mirrors.cloud.tencent.com/pypi/simple"
pip install -U pip setuptools wheel -i "$PIP_MIRROR" --extra-index-url "$PIP_EXTRA"
pip install -r requirements.txt -i "$PIP_MIRROR" --extra-index-url "$PIP_EXTRA"

# ===== 完成 =====
log ""
log "${BOLD}${GREEN}✓ 安装完成${RESET}"
log ""
log "下一步："
if [[ $SKIP_CONDA -eq 0 ]]; then
    log "  conda activate $ENV_NAME"
fi
log "  python scripts/download_models.py        # (可选) 预热模型到 ./models/"
log "  python scripts/check_env.py --dataset /path/to/dataset_1000"
log "  bash scripts/run_benchmark.sh /path/to/dataset_1000"
log ""
log "采集 + 可视化检索（RealSense D4xx）："
log "  python sample_rsd4xx.py --output ./captures   # 采集 RGBD + 随机位姿"
log "  python ui.py --capture-dir ./captures         # 打开 UI 检索并可视化"
