# RoFA-SemEval 数据集采集工具

面向《指标 4.2 环境语义表达准确率》的数据集采集 / 标注 / 发布工具集。

详细方案见 [`dataset_plan.html`](./dataset_plan.html)（用浏览器打开）。

## 目录结构

```
scripts/
├── README.md                # 本文件
├── requirements.txt         # 依赖清单（按阶段分组）
├── dataset_plan.html        # 数据集设计方案（v5）
├── _dataset_common.py       # 三个脚本共用的工具函数
├── capture_sample.py        # 阶段 1：采集
├── annotate_sample.py       # 阶段 2：标注 + 删减
└── finalize_dataset.py      # 阶段 3：完整性校验 + 发布
```

## 三阶段工作流

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  阶段 1 采集  │ →  │  阶段 2 标注  │ →  │  阶段 3 发布  │
│  采集员      │    │  标注员      │    │  数据负责人  │
│  capture_*   │    │  annotate_*  │    │  finalize_*  │
└──────────────┘    └──────────────┘    └──────────────┘
        ↓                  ↓                   ↓
  raw_capture/       raw_capture/         dataset/
   pending/         annotated/           samples/
                    discarded/           dataset.json
                                         samples.json
                                         dataset_report.html
```

## 快速开始

### 1. 创建并激活 conda 环境

推荐用一个独立的轻量环境 `rofa-data`，与项目原有的重型 `rofa`（含 SAM3/RynnBrain/ROS2）解耦：

```bash
conda create -n rofa-data python=3.10 -y
conda activate rofa-data
```

### 2. 安装依赖

机器人本机（采集 + 标注 + 发布全装）：

```bash
# 基础（必需）
conda install -y "numpy<2.0"
pip install opencv-python

# 阶段 1（采集）
pip install pyrealsense2
pip install pypinyin                # 中文类别名 → 拼音 slug，强烈建议

# 阶段 2（标注，可选但强烈建议）
pip install open3d                   # 点云 SOR + DBSCAN 去噪
pip install matplotlib               # 3D AABB 立体可视化
```

或者一键装齐：

```bash
pip install -r RobotMustFindAnything/scripts/requirements.txt
```

按角色精简版：

| 角色 | 必需 | 可选 |
|---|---|---|
| 采集员（D435 主机） | `numpy<2.0` `opencv-python` `pyrealsense2` | `pypinyin` |
| 标注员（任意机器） | `numpy<2.0` `opencv-python` | `open3d` `matplotlib` |
| 数据负责人（发布机） | `numpy<2.0` `opencv-python` | — |

> 若想复用工程原有的 `rofa` 环境而不新建：`conda activate rofa && pip install pypinyin open3d matplotlib pyrealsense2`（其余依赖原 `rofa` 环境已具备）。

### 3. 阶段 1：采集

```bash
python scripts/capture_sample.py \
    --raw-root ./RoFA-SemEval/raw_capture \
    --operator alice
```

操作：
- 启动后在终端输入物品名（中英文均可），如 `瓶子`、`bottle`、`轮椅`、`door handle`
- 实时窗口聚焦后，按 **空格** 抓拍
- 按 **n** 切换 / 新建类别（终端再次提示输入）
- 按 **q** 退出

每次抓拍会在 `raw_capture/pending/<class_slug>/<class>_<NNNN>/` 下落盘 7 个文件：
`rgb.jpg / depth.png / intrinsics.json / pose.txt / capture_meta.json`。

类别表 `raw_capture/classes.json` 由脚本累积维护，包含 id / 英文 slug / 中文名 / aliases / captured_count。

### 4. 阶段 2：标注

```bash
python scripts/annotate_sample.py \
    --raw-root ./RoFA-SemEval/raw_capture \
    --annotator alice \
    --class bottle           # 可选：只标某一类
```

操作（OpenCV 窗口聚焦时）：
- **左键** 沿物体轮廓依次点击，画一圈闭合多边形
- **空格** 或 **右键** 闭合 → 自动填充 mask → 自动反投影 / 去噪 / 算 AABB
- 看完结果窗口后按：
  - **y** 接受 → 整目录从 `pending/` 移到 `annotated/`
  - **n** 重做（重新画 mask）
  - **d** 删除 → 移到 `discarded/`，附 `discard_reason.txt`
  - **s** 跳过 → 留在 `pending/`
  - **q** 退出
- 画 mask 过程中：**z** 撤销最近一个顶点，**r** 清空重画

### 5. 阶段 3：发布

```bash
# 干跑（只校验、生成报告，不写 dataset/）
python scripts/finalize_dataset.py \
    --raw-root ./RoFA-SemEval/raw_capture \
    --dataset-root ./RoFA-SemEval/dataset \
    --dry-run

# 正式发布
python scripts/finalize_dataset.py \
    --raw-root ./RoFA-SemEval/raw_capture \
    --dataset-root ./RoFA-SemEval/dataset

# 发布时排除某些类
python scripts/finalize_dataset.py \
    --exclude-class door_handle \
    --exclude-class water_bottle
```

产物：
- `dataset/dataset.json` — 全局元数据
- `dataset/classes.json` — 最终类别表（仅保留至少 1 条样本的类）
- `dataset/samples.json` — 全集索引（所有有效样本，**无 train/test 划分**）
- `dataset/samples/<class>/<sample_id>/` — 完整样本目录（硬链接或拷贝）
- `dataset/dataset_report.json` + `.html` — 数据集报告
- `dataset/rejected.csv` — 校验未通过的样本清单（不入最终数据集）

## 设计要点

- **位姿**：所有 `pose.txt` 写为 4×4 单位阵（dummy）。设备只有 RealSense D435，无底盘 / 无 SLAM。坐标系约定为"相机系即参考系"，IoU3D 不受影响。
- **类别**：完全由采集员运行时输入并归并；中文输入会转拼音作为目录名（依赖 pypinyin，缺失时降级到 utf8-hex 兜底）。
- **规模**：采集端零约束，最终多少类、多少条由阶段 3 报告事后呈现。大纲推荐刻度（20 类 / 1000 样本 / 每类 25-75）只在报告里对照展示。
- **删减权**：只有标注员在阶段 2 能主动删除样本（按 d 键）；阶段 3 的 finalize 不做主观删减，仅做硬性完整性兜底。
- **可重入**：阶段 3 多次运行安全（已存在的目标目录会跳过）；阶段 1 与阶段 2 都不会破坏已有数据。

## 单脚本帮助

```bash
python scripts/capture_sample.py  --help
python scripts/annotate_sample.py --help
python scripts/finalize_dataset.py --help
```
