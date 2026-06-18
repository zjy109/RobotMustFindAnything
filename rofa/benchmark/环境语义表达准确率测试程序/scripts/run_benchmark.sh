#!/usr/bin/env bash
# 一键运行测试 — 简单封装 main.py，给第三方一个最少参数的入口。
#
# 用法：
#   bash scripts/run_benchmark.sh <dataset_path> [output_dir]
#
# 示例：
#   bash scripts/run_benchmark.sh /data/dataset_1000
#   bash scripts/run_benchmark.sh /data/dataset_1000 ./run_20260528
#
# 设计：
#   - 模型权重统一在 <项目根>/models/ 下：
#       models/RynnBrain-8B/         (~16GB，自动下载)
#       models/sam2.1-hiera-small/   (~200MB，自动下载)
#   - 核心评测参数（IoU=0.25、SOR=20/0.75、存在性预筛、SAM2 模型 ID、解码参数）
#     已在 src/benchmark/config.py 中锁死，与课题二原 benchmark.py 完全一致。
#   - 不需要任何环境变量；第三方测试时只输入数据集路径即可。
#
set -euo pipefail

GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
RESET="\033[0m"
BOLD="\033[1m"

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<EOF
用法: bash scripts/run_benchmark.sh <dataset_path> [output_dir]

参数:
  <dataset_path>    数据集根目录（含 samples.json）
  [output_dir]      输出目录（默认 ./run_<timestamp>）

可选环境变量:
  CUDA_DEVICES      可见的 CUDA 设备（默认 0）
  MAX_SAMPLES       仅跑前 N 个样本（默认 -1 = 全集；试跑用）
  EXTRA_ARGS        额外传给 main.py 的参数

锁死参数（不可改，与原 benchmark.py 一致）:
  3D IoU 阈值       0.25
  SOR 去噪          nb=20, std=0.75
  存在性预筛        启用
  SAM2 模型         facebook/sam2.1-hiera-small
  RynnBrain 解码    do_sample=False, max_new_tokens=128

模型权重（自动下载到 <项目根>/models/）:
  RynnBrain-8B          ~16GB（仅首次）
  sam2.1-hiera-small    ~200MB（仅首次）

示例:
  bash scripts/run_benchmark.sh /data/dataset_1000
  CUDA_DEVICES=1 bash scripts/run_benchmark.sh /data/dataset_1000
  MAX_SAMPLES=10 bash scripts/run_benchmark.sh /data/dataset_1000 ./trial_run
EOF
    exit 0
fi

DATASET_PATH="$1"
OUTPUT_DIR="${2:-./run_$(date +%Y%m%d_%H%M%S)}"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ----- 校验 -----
if [[ ! -d "$DATASET_PATH" ]]; then
    echo -e "${RED}错误: 数据集目录不存在: $DATASET_PATH${RESET}"
    exit 2
fi
if [[ ! -f "$DATASET_PATH/samples.json" ]]; then
    echo -e "${RED}错误: $DATASET_PATH 里没有 samples.json${RESET}"
    exit 2
fi

# 模型存在性提示（runner 会自动下载）
RYNN_DIR="$PROJECT_ROOT/models/RynnBrain-8B"
SAM2_DIR="$PROJECT_ROOT/models/sam2.1-hiera-small"
if [[ ! -f "$RYNN_DIR/config.json" || ! -f "$SAM2_DIR/config.json" ]]; then
    echo -e "${YELLOW}提示: 模型权重未就绪，运行时会自动下载到 <项目根>/models/${RESET}"
    [[ ! -f "$RYNN_DIR/config.json" ]] && echo -e "${YELLOW}      - RynnBrain-8B  ~16GB${RESET}"
    [[ ! -f "$SAM2_DIR/config.json" ]] && echo -e "${YELLOW}      - sam2.1-hiera-small  ~200MB${RESET}"
    echo -e "${YELLOW}      也可先单独跑：python scripts/download_models.py 预热${RESET}"
fi

echo -e "${BOLD}=== 运行参数 ===${RESET}"
echo "  数据集:       $DATASET_PATH"
echo "  输出目录:     $OUTPUT_DIR"
echo "  CUDA 设备:    $CUDA_DEVICES"
echo "  样本数限制:   $MAX_SAMPLES"
echo "  锁死参数:     IoU=0.25  SOR(nb=20,std=0.75)  existence_check=on  SAM2=sam2.1-hiera-small"
echo

# ----- 跑 -----
mkdir -p "$OUTPUT_DIR"

# shellcheck disable=SC2086
python main.py \
    --dataset "$DATASET_PATH" \
    --output "$OUTPUT_DIR" \
    --cuda-devices "$CUDA_DEVICES" \
    --max-samples "$MAX_SAMPLES" \
    $EXTRA_ARGS

echo
echo -e "${GREEN}${BOLD}✓ 运行结束${RESET}"
echo "  打开报告：  file://$(realpath "$OUTPUT_DIR/report.html")"
