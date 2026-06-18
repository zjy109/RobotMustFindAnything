"""环境语义表达准确率测试程序 — 核心包

模块组织：
    config.py    配置 dataclass + CLI 解析
    geometry.py  IoU / 反投影 / 点云去噪等几何工具（无模型依赖）
    models.py    RynnBrainDetector + SAM2Segmenter 两个推理类
    viz.py       单样本叠加图（失败 case 可视化用）
    report.py    HTML 报告生成（不依赖 matplotlib，纯 SVG）
    runner.py    主流水线：加载数据集 → 逐样本推理 → 写结果与报告
"""

__version__ = "1.0.0"
