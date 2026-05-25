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

#### 4.1 （可选）AI 预标加速：RynnBrain + SAM2

如果你想"先让模型猜一遍，不行再人工画"，可以用 `--use-vlm` 接到已经在跑的 VLM 服务（`real_main_on_server.py`）。

**原理**：标注脚本会把每张待标的 `rgb.jpg` 通过 ZMQ 发给 server；server 端用 RynnBrain 出 bbox + SAM2 出 mask，把 mask base64 回传。标注脚本拿到 mask 后弹一个红色叠加预览，你只需 y/n 决策——

- **y** 直接接受预标 mask，跳过手画，进入正常的 反投影/AABB/决策 流程；
- **n** 预标不行，进入手画分支（与原流程完全一致）；
- **d/s/q** 同上：删除 / 跳过 / 退出。

**先把 server 起来**（在有 GPU 和 RynnBrain 权重的机器上）：
```bash
# 在 server 机器
python real_main_on_server.py
# 或自定义参数
python rofa/vlm_server/server.py --port 5555 --model-path /path/to/RynnBrain-8B
```

**标注端启用预标**（标注员的工作机，可以是另一台机器）：
```bash
python scripts/annotate_sample.py \
    --raw-root ./RoFA-SemEval/raw_capture \
    --annotator alice \
    --use-vlm \
    --vlm-host 192.168.x.y \
    --vlm-port 5555
```

**slug → prompt 映射**：RynnBrain 直接支持中文 prompt，所以脚本默认用 `classes.json` 里每类的 `name_zh`（如 `锅铲`、`水壶`）做检索文本，**无需任何额外配置**。如果某一类的中文名描述太宽泛（例如同一个 `name_zh` 下其实包含多个子品类），可以用 JSON 文件单点覆盖：

```bash
echo '{"shuihu": "不锈钢保温水壶", "guochan": "厨房铲子"}' > my_prompts.json
python scripts/annotate_sample.py --use-vlm --vlm-prompt-map my_prompts.json ...
```

优先级：`--vlm-prompt-map` > `classes.json.name_zh` > slug 本身（兜底）。

**`sample.json` 中的标注溯源**：每条样本会记录 `annotation.method`：
- `manual_polygon` — 标注员从空白手画
- `vlm_predicted_accepted` — 标注员看了预标后按 `y` 接受

下游训练 / 评测里可以按需筛选"纯人工"或"AI 辅助"样本。

**先单独自测 client**（可以不跑 annotate，先确认 server 能正常分割）：
```bash
python scripts/vlm_seg_client.py \
    --host 192.168.x.y --port 5555 \
    --image raw_capture/pending/guochan/guochan_0001/rgb.jpg \
    --prompt "锅铲" \
    --out /tmp/pred_mask.png \
    --out-overlay /tmp/pred_overlay.png
```

**降级行为**：当 server 连不上 / 预标超时 / 模型未找到目标 / 返回 mask 尺寸异常时，脚本只会 print 一行警告，然后无缝 fallback 到原本的手画流程。**`--use-vlm` 的存在本身是无副作用的**。

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
