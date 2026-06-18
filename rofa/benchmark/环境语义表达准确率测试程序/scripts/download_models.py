#!/usr/bin/env python3
"""下载所有评测所需模型到 <项目根>/models/。

零参数运行：
    python scripts/download_models.py

下载内容（路径固定，不可改）：
    <项目根>/models/RynnBrain-8B/         (~16GB)
    <项目根>/models/sam2.1-hiera-small/   (~200MB)

下载源策略：先 ModelScope（国内零配置），失败 fallback HuggingFace。
首次运行如果机器上未装 modelscope/huggingface_hub，脚本会自动 pip 安装。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让脚本能 import 项目内 src/benchmark
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.model_resolver import (  # noqa: E402
    ensure_all,
    is_rynnbrain_ready,
    is_sam2_ready,
    rynnbrain_dir,
    sam2_dir,
)


def main() -> int:
    print("=" * 60)
    print("准备评测所需模型")
    print("=" * 60)
    print(f"  RynnBrain-8B  -> {rynnbrain_dir()}")
    print(f"  SAM2          -> {sam2_dir()}")
    print()

    # 已就绪的不重下
    if is_rynnbrain_ready() and is_sam2_ready():
        print("✓ 所有模型已就绪，无需下载。")
        return 0

    try:
        paths = ensure_all()
    except Exception as exc:
        print(f"\n[error] 模型准备失败: {exc}", file=sys.stderr)
        return 2

    print()
    print("=" * 60)
    print("✓ 全部模型准备完毕")
    for k, v in paths.items():
        print(f"  {k:12s} -> {v}")
    print("=" * 60)
    print("\n下一步：bash scripts/run_benchmark.sh /path/to/dataset_1000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
