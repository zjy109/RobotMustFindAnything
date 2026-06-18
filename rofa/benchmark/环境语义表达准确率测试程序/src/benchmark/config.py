"""测评配置：dataclass + CLI 解析。

设计原则
---------
**核心评测参数（影响"环境语义表达准确率"指标的参数）一律锁死**，
与课题二原始 `RobotMustFindAnything/rofa/benchmark.py` 保持完全一致，
不允许通过 CLI 修改，确保不同环境/第三方复现时指标可比。

锁死项（FROZEN，定义为模块级常量，不进入 argparse）：
- ``IOU_THRESHOLD_3D = 0.25``
- ``SOR_NB = 20``、``SOR_STD = 0.75``        (点云 SOR 去噪)
- ``ENABLE_EXISTENCE_CHECK = True``           (RynnBrain 先做存在性预筛)
- ``SAM2_MODEL_ID = "facebook/sam2.1-hiera-small"``
- ``RYNNBRAIN_GENERATE_KW = {do_sample: False, max_new_tokens: 128}``

模型路径全部由 ``benchmark.model_resolver`` 自动管理，统一下载到
``<项目根>/models/`` 下，**用户无需也无法**通过 CLI / 环境变量指定。

可调项（仅环境/调试相关，不影响指标语义）：
- ``--dataset``：数据集根目录（必填）
- ``--cuda-devices``：``CUDA_VISIBLE_DEVICES``
- ``--output``：输出目录
- ``--no-viz`` / ``--max-samples`` / ``--class`` / ``--no-resume`` / ``--seed``
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# FROZEN：核心评测参数（与 benchmark.py 一致，禁止从 CLI 改）
# ---------------------------------------------------------------------------
IOU_THRESHOLD_3D: float = 0.25
"""3D IoU 判定为成功的阈值。"""

SOR_NB: int = 20
"""统计离群点滤波（SOR）的近邻数。"""

SOR_STD: float = 0.75
"""统计离群点滤波（SOR）的标准差比例。"""

ENABLE_EXISTENCE_CHECK: bool = True
"""RynnBrain 在定位前先做"是否存在"预筛。"""

SAM2_MODEL_ID: str = "facebook/sam2.1-hiera-small"
"""SAM2 模型 HuggingFace ID。固定为原 benchmark 使用的 small 版。"""

RYNNBRAIN_GENERATE_KW: Dict[str, Any] = {
    "do_sample": False,
    "max_new_tokens": 128,
}
"""RynnBrain `model.generate` 关键字参数（确定性贪婪解码）。"""


# ---------------------------------------------------------------------------
# 仅作 CLI 默认值的"环境/调试"参数（不影响指标）
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "./run_output"
DEFAULT_CUDA_DEVICES = "0"


@dataclass
class BenchmarkConfig:
    """单次评测运行的所有配置。

    核心评测参数已经写死在模块级常量中，模型路径由 model_resolver 管理，
    这里只保留输出/调试类字段。
    """

    # ---- 数据集 ----
    dataset_root: Path
    """已发布数据集根目录（含 samples.json / classes.json / samples/）"""

    # ---- 硬件 ----
    cuda_devices: str = DEFAULT_CUDA_DEVICES

    # ---- 输出 ----
    output_dir: Path = Path(DEFAULT_OUTPUT_DIR)
    save_visualizations: bool = True

    # ---- 调试 ----
    max_samples: int = -1
    only_class: Optional[str] = None
    resume: bool = True
    seed: int = 42

    # ---- 运行时回填（由 runner 通过 model_resolver 设置，便于审计） ----
    rynnbrain_model_path: Optional[Path] = None
    """运行时由 model_resolver.ensure_rynnbrain() 回填，仅用于日志/审计。"""

    sam2_model_path: Optional[Path] = None
    """运行时由 model_resolver.ensure_sam2() 回填，仅用于日志/审计。"""

    # ---- 其它 ----
    extra: Dict[str, Any] = field(default_factory=dict)

    # ===== 锁死参数（只读 property，不允许覆盖） =====
    @property
    def iou_threshold(self) -> float:
        return IOU_THRESHOLD_3D

    @property
    def sor_nb(self) -> int:
        return SOR_NB

    @property
    def sor_std(self) -> float:
        return SOR_STD

    @property
    def enable_existence_check(self) -> bool:
        return ENABLE_EXISTENCE_CHECK

    @property
    def sam2_model_id(self) -> str:
        return SAM2_MODEL_ID

    @property
    def rynnbrain_generate_kw(self) -> Dict[str, Any]:
        return dict(RYNNBRAIN_GENERATE_KW)

    # -------- 派生路径 --------
    @property
    def results_json(self) -> Path:
        return self.output_dir / "results.json"

    @property
    def process_log(self) -> Path:
        return self.output_dir / "process.log"

    @property
    def visualizations_dir(self) -> Path:
        return self.output_dir / "visualizations"

    @property
    def report_html(self) -> Path:
        return self.output_dir / "report.html"

    @property
    def report_json(self) -> Path:
        return self.output_dir / "report.json"

    @property
    def env_snapshot(self) -> Path:
        return self.output_dir / "env_snapshot.json"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        # 把锁死参数也写进去，便于结果文件溯源
        d["frozen"] = {
            "iou_threshold": IOU_THRESHOLD_3D,
            "sor_nb": SOR_NB,
            "sor_std": SOR_STD,
            "enable_existence_check": ENABLE_EXISTENCE_CHECK,
            "sam2_model_id": SAM2_MODEL_ID,
            "rynnbrain_generate_kw": dict(RYNNBRAIN_GENERATE_KW),
        }
        return d


# ---------------------------------------------------------------------------
# CLI 解析（只暴露环境/输出/调试参数；模型路径全部内置）
# ---------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> None:
    """给 argparse parser 注入 CLI 参数。

    注意：
    - 核心评测参数（IoU / SOR / 存在性预筛 / SAM2 模型 ID）已锁死，不在此暴露。
    - 模型路径（RynnBrain-8B / SAM2）固定为 ``<项目根>/models/``，不在此暴露。
    """
    g_data = parser.add_argument_group("数据集")
    g_data.add_argument(
        "--dataset", type=str, required=True,
        help="数据集根目录（含 samples.json / classes.json / samples/...）",
    )

    g_hw = parser.add_argument_group("硬件")
    g_hw.add_argument(
        "--cuda-devices", type=str, default=DEFAULT_CUDA_DEVICES,
        help='CUDA_VISIBLE_DEVICES，例如 "0" 或 "0,1"',
    )

    g_out = parser.add_argument_group("输出")
    g_out.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_DIR,
        help="本次运行的输出目录（结果 json / 报告 / 可视化都在这里）",
    )
    g_out.add_argument(
        "--no-viz", action="store_true",
        help="不保存逐样本叠加图（节省时间和磁盘）",
    )

    g_dbg = parser.add_argument_group("调试")
    g_dbg.add_argument(
        "--max-samples", type=int, default=-1,
        help="只跑前 N 个样本（试跑用），-1 = 全集",
    )
    g_dbg.add_argument(
        "--class", dest="only_class", type=str, default=None,
        help="仅评测指定类别（class_name slug，可选）",
    )
    g_dbg.add_argument(
        "--no-resume", action="store_true",
        help="禁用断点续跑（默认会跳过 results.json 中已存在的样本）",
    )
    g_dbg.add_argument("--seed", type=int, default=42)


def from_argparse(ns: argparse.Namespace) -> BenchmarkConfig:
    """argparse Namespace -> BenchmarkConfig。"""
    return BenchmarkConfig(
        dataset_root=Path(ns.dataset).expanduser().resolve(),
        cuda_devices=ns.cuda_devices,
        output_dir=Path(ns.output).expanduser().resolve(),
        save_visualizations=not ns.no_viz,
        max_samples=ns.max_samples,
        only_class=ns.only_class,
        resume=not ns.no_resume,
        seed=ns.seed,
    )
