#!/usr/bin/env python3
"""环境语义表达准确率测试程序 — 入口脚本。

用法：
    python main.py --dataset /path/to/dataset_1000

详细参数见 `python main.py --help`，或参考 README.md。
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

# 让 `python main.py` 直接能导入 src/benchmark
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.config import add_arguments, from_argparse  # noqa: E402
from benchmark.runner import run_benchmark, collect_env_snapshot  # noqa: E402
from benchmark.report import write_html_report, write_json_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description=(
            "环境语义表达准确率测试程序\n"
            "对给定数据集运行 RynnBrain + SAM2 + 3D 反投影流水线，"
            "输出准确率指标与可视化报告。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_arguments(parser)
    args = parser.parse_args()

    cfg = from_argparse(args)

    print(f"[main] 输出目录: {cfg.output_dir}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        results_doc = run_benchmark(cfg)
    except KeyboardInterrupt:
        print("\n[main] 用户中断，已保存已处理的结果。")
        print(f"  下次运行同样命令会自动续跑：{cfg.results_json}")
        return 130
    except Exception as exc:
        print(f"\n[main][fatal] {exc}")
        traceback.print_exc()
        return 2

    # 报告
    env_snap = collect_env_snapshot(cfg)
    env_snap["generated_at"] = datetime.now().isoformat(timespec="seconds")

    print("[main] 生成 HTML 报告 ...")
    html_path = write_html_report(cfg, results_doc, env_snap)
    json_path = write_json_report(cfg, results_doc)
    print(f"  ✓ {html_path}")
    print(f"  ✓ {json_path}")
    print()
    print(f"在浏览器打开报告：file://{html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
