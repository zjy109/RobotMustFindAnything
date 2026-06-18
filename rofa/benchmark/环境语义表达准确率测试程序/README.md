# 环境语义表达准确率测试程序

> 对给定语义评测数据集（RoFA-SemEval 格式），运行 **RynnBrain (VLM) + SAM2 + 3D 反投影** 流水线，
> 计算 **3D IoU 准确率** 这一核心指标，并产出人类可读的 HTML 报告。

适用场景：在第三方机器（Ubuntu + NVIDIA GPU）上独立完成测试与验收。

---

## 目录

1. [核心指标定义](#1-核心指标定义)
2. [测试流水线](#2-测试流水线)
3. [前置要求](#3-前置要求)
4. [安装](#4-安装)
5. [准备模型](#5-准备模型)
6. [运行测试](#6-运行测试)
7. [理解报告](#7-理解报告)
8. [项目结构](#8-项目结构)
9. [配置项详解](#9-配置项详解)
10. [常见问题](#10-常见问题)

---

## 1. 核心指标定义

**3D IoU 成功率（accuracy@IoU=θ）**：在数据集上，预测 3D 包围盒与 GT 3D 包围盒的 IoU ≥ θ 的样本比例。

```
accuracy(θ) = | { sample : IoU3D(pred, gt) ≥ θ } | / | dataset |
```

**θ 锁定为 0.25**（与课题二原始 `RobotMustFindAnything/rofa/benchmark.py` 一致）。

> **本测试程序的合格标准**：在指定数据集（1000 样本）下，θ=0.25 时准确率 **≥ 0.836**。

> ⚠ **关于参数固定**：为保证不同环境/第三方机器上结果可复现、可对照，
> 所有"会影响指标语义"的参数都已**写死在 `src/benchmark/config.py` 顶部**，
> 不允许通过 CLI 修改。锁定项见下表，与原 `benchmark.py` 完全一致：
>
> | 参数 | 锁定值 |
> |---|---|
> | `IOU_THRESHOLD_3D` | `0.25` |
> | `SOR_NB` / `SOR_STD` | `20` / `0.75` |
> | `ENABLE_EXISTENCE_CHECK` | `True` |
> | `SAM2_MODEL_ID` | `facebook/sam2.1-hiera-small` |
> | `RYNNBRAIN_GENERATE_KW` | `do_sample=False, max_new_tokens=128` |

---

## 2. 测试流水线

```
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│ 单帧 RGB   │→ │ RynnBrain  │→ │ 像素 bbox  │→ │   SAM2     │→ │ 物体 mask  │
│ + 类别提示 │  │  VLM 检测  │  │ [x1,y1,x2,y2]│ │  分割      │  │ (H, W) bool│
└────────────┘  └────────────┘  └────────────┘  └────────────┘  └─────┬──────┘
                                                                       │
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│ 3D IoU    │← │ 6-D AABB   │← │ SOR 去噪   │← │ 像素+深度  │←────────┘
│ vs GT      │  │ (xyz min/max)│ │           │  │ → 点云     │
└────────────┘  └────────────┘  └────────────┘  └────────────┘
```

输入：每个样本目录包含
- `rgb.jpg`           640×480 RGB
- `depth.png`         16-bit 深度（毫米）
- `intrinsics.json`   `{fx, fy, cx, cy, depth_scale}`
- `aabb.json`         GT，含 `min/max/extent/...` 与 `bbox_2d.from_mask`

数据集索引：`samples.json`（每条记录含 `sample_id / class_name / class_name_zh / sample_dir`）。

---

## 3. 前置要求

| 项 | 要求 |
|---|---|
| 操作系统 | Ubuntu 20.04 / 22.04（其他 Linux 通常也可以） |
| GPU | NVIDIA GPU，**显存 ≥ 24GB 推荐**（RynnBrain-8B + SAM2 在 fp16 下约 18GB） |
| CUDA | 11.8 或 12.1（与 PyTorch 匹配即可） |
| Python | 3.10 / 3.11 |
| 磁盘 | 至少 **50 GB** 空闲（RynnBrain-8B ~16GB + 数据集 + 输出） |
| 内存 | ≥ 32 GB 推荐 |
| 网络 | 首次跑需下载 SAM2（≈ 200 MB） |

---

## 4. 安装

### 4.1 一键脚本

```bash
git clone <本仓库地址> rofa-bench && cd rofa-bench
bash scripts/install.sh
```

脚本会：
1. 安装系统级依赖（如有 sudo 权限：`libgl1`, `libglib2.0-0` 等）
2. 创建 conda 环境 `rofa-bench` (Python 3.10)
3. 装 GPU 版 PyTorch（默认 CUDA 12.1，如需 11.8 用 `--torch-cuda cu118`）
4. 装 `requirements.txt` 里的所有包

### 4.2 手动安装（如果你已有 conda 环境）

```bash
conda activate <你的环境>
bash scripts/install.sh --skip-conda
# 或
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
pip install -r requirements.txt
```

### 4.3 校验

```bash
conda activate rofa-bench
python scripts/check_env.py --dataset /path/to/dataset_1000
```

期望输出（示意）：

```
=== 环境自检 ===

[1] Python
  ✓ Python 3.10.13  (要求 >= 3.10)

[2] 关键依赖
  ✓ numpy  1.26.4
  ✓ cv2  4.8.1
  ✓ PIL  10.2.0
  ✓ tqdm  4.66.1
  ✓ torch  2.3.0+cu121
  ✓ transformers  4.45.0

[4] CUDA / GPU
  ✓ CUDA 可用，设备数 = 1
  ✓   device 0: NVIDIA RTX 4090  (24.0 GB)

[5] 模型权重
  ✓ RynnBrain-8B  已就绪: ./models/RynnBrain-8B
  ✓ SAM2           已就绪: ./models/sam2.1-hiera-small
  （或：未下载 — 首次运行测试时会自动下载）

[6] 数据集
  ✓ samples.json 与 samples/ 都在
  ✓ samples.json 共 1000 条
  ✓   抽检 zh_xxxx_0001 必备文件齐全
  ...

=== 总结 ===
  ✓ 所有检查通过，可以运行测试。
```

---

## 5. 准备模型

### 总览：模型自动下载

**默认无需任何操作**。第一次跑 `bash scripts/run_benchmark.sh ...` 时，
程序会自动把所需权重下载到 `<项目根>/models/`：

| 模型 | 目录 | 大小 | 来源 |
|---|---|---|---|
| RynnBrain-8B | `models/RynnBrain-8B/` | ~16GB | ModelScope `DAMO_Academy/RynnBrain-8B` → HuggingFace `Alibaba-DAMO-Academy/RynnBrain-8B` |
| SAM2 (hiera-small) | `models/sam2.1-hiera-small/` | ~200MB | ModelScope `AI-ModelScope/sam2.1-hiera-small` → HuggingFace `facebook/sam2.1-hiera-small` |

下载策略：**先试 ModelScope（国内零配置可用），失败兜底 HuggingFace**。
两个仓位的下载都会在 runner 启动时一次性完成，**无需配置任何环境变量、镜像或代理**。

> ⚠ **路径与下载源都是固定的**，用户不能也不需要通过命令行/环境变量修改。
> 这是为了保证第三方测试结果的可复现性。

### 显式预热（可选，推荐在评测前先单独跑一次）

```bash
python scripts/download_models.py
```

这会把上面两个模型一次性下完。后续 `bash scripts/run_benchmark.sh ...` 直接复用，不再下载。

### 没有外网怎么办？

如果机器既访问不到 ModelScope 也访问不到 HuggingFace，请在**有外网的机器**上跑一次
`python scripts/download_models.py`，然后把整个 `models/` 目录拷贝到测试机的项目根下即可。

---

## 6. 运行测试

### 6.1 最简用法

```bash
bash scripts/run_benchmark.sh /path/to/dataset_1000
```

输出位于 `./run_<时间戳>/`：

```
run_20260528_103025/
├── results.json          原始逐样本结果（与原 benchmark.py 同构）
├── report.html           人话报告（核心交付物，浏览器打开看）
├── report.json           精简数值摘要
├── process.log           逐样本日志
├── env_snapshot.json     运行环境快照（Python/PyTorch/GPU 型号）
└── visualizations/       逐样本叠加图（蓝 mask + 绿 pred + 黄 GT）
    ├── det_seg_xxx_0001.jpg
    └── ...
```

### 6.2 进阶用法

```bash
# 试跑前 10 个样本（验证流程，~3 分钟）
MAX_SAMPLES=10 bash scripts/run_benchmark.sh /path/to/dataset_1000

# 用第二张卡
CUDA_DEVICES=1 bash scripts/run_benchmark.sh /path/to/dataset_1000

# 只测某一类
EXTRA_ARGS="--class zh_e6b0b4e5a3b6" bash scripts/run_benchmark.sh /path/to/dataset_1000

# 关掉可视化加速 5~10%
EXTRA_ARGS="--no-viz" bash scripts/run_benchmark.sh /path/to/dataset_1000
```

### 6.3 直接调 main.py

如果只想自定义路径/调试参数（**核心评测参数已锁死，无法修改；RynnBrain 模型路径由 model_resolver 自动管理，无需也不允许 CLI 指定**）：

```bash
python main.py \
    --dataset /path/to/dataset_1000 \
    --output ./run_test \
    --cuda-devices 0
```

完整参数：`python main.py --help`。

### 6.4 断点续跑

测试默认开启 `--resume`：

- 中途 Ctrl-C 不会丢失已处理的结果（每跑完一个类就 flush 一次 `results.json`）
- 重新执行**完全相同的命令**，会自动跳过 `results.json` 中已存在的样本，从断点继续

如果想强制重跑：删掉 `results.json` 或加 `EXTRA_ARGS="--no-resume"`。

### 6.5 预期耗时

在 RTX 4090 单卡上：

| 阶段 | 单样本耗时 |
|---|---|
| RynnBrain（含存在性预筛 + 定位） | 1.5 ~ 3 s |
| SAM2 | 0.1 ~ 0.3 s |
| 3D 反投影 + SOR 去噪 | 0.05 ~ 0.2 s |
| **合计** | **~2 ~ 4 s/sample** |

1000 样本约 **40 ~ 70 分钟**。

---

## 7. 理解报告

`report.html` 用浏览器打开，包含 4 个区块：

### ① 核心指标卡片
4 张大数字卡片：**总精度（IoU≥0.25）** / 成功数 / 总样本数 / 类别覆盖。

### ② 类别精度柱状图（SVG）
每个类一根横向柱子，颜色编码：
- 🟢 ≥ 0.8（合格）
- 🟡 0.5 ~ 0.8（待优化）
- 🔴 < 0.5（差类）

可展开查看『类别精度数值表』，每行包含样本数 / 2D IoU 均值 / 3D IoU 均值 / 类内精度。

### ③ 失败案例展板
**Top-30 失败样本**（按 3D IoU 升序）展示原图 + 预测叠加缩略图。
其余失败样本只列 ID，避免报告过大。

### ④ 运行环境快照
完整记录 Python / PyTorch / CUDA 设备 / 配置参数，便于复现。

---

## 8. 项目结构

```
环境语义表达准确率测试程序/
├── README.md               本文件（详细教程）
├── QUICKSTART.md           30 秒上手卡
├── pyproject.toml          uv / pip 兼容的项目元信息
├── requirements.txt        Python 依赖（torch 除外）
├── .gitignore
│
├── main.py                 入口：python main.py --dataset ...
│
├── sample_rsd4xx.py        采集：RealSense D4xx 采 RGBD + 随机位姿 → 样本目录
├── ui.py                   可视化检索 UI（桌面/Tkinter）：bbox+掩码+AABB
├── ui_web.py               可视化检索 UI（Web/Gradio）：跳板机/无显示远程服务器用
│
├── src/benchmark/          核心代码（按职责拆分）
│   ├── __init__.py
│   ├── config.py           BenchmarkConfig dataclass + CLI 解析（核心评测参数全部锁死）
│   ├── geometry.py         IoU / 反投影 / SOR 去噪（无模型依赖，可单测）
│   ├── capture_io.py       采集样本读写 + 随机位姿 + 点云/AABB 工具（采集与 UI 共用）
│   ├── model_resolver.py   模型自动下载（RynnBrain-8B + SAM2，统一到 <项目根>/models/）
│   ├── models.py           RynnBrainDetector + SAM2Segmenter
│   ├── viz.py              单样本叠加图（render_overlay 渲染 + save_overlay 写盘）
│   ├── report.py           HTML/JSON 报告生成
│   └── runner.py           主流水线（断点续跑、日志、增量 flush）
│
└── scripts/
    ├── install.sh          一键环境安装（conda + pip）
    ├── check_env.py        环境自检（GPU/包/数据集/磁盘）
    ├── download_models.py  零参数预热模型（RynnBrain-8B + SAM2 一次下完）
    └── run_benchmark.sh    一键跑测试（封装 main.py）
```

---

## 9. 配置项详解

完整列表：`python main.py --help`。

> **核心评测参数已锁死，不在 CLI 暴露**（IoU 阈值、SOR、存在性预筛、SAM2 模型 ID、RynnBrain 解码参数）。
> 见 [§1 核心指标定义](#1-核心指标定义) 中的"参数固定"说明，或直接看 `src/benchmark/config.py` 顶部的常量。
>
> **模型路径也不在 CLI 暴露**，由 `model_resolver` 统一管理，固定下载到 `<项目根>/models/`：
> - `models/RynnBrain-8B/`
> - `models/sam2.1-hiera-small/`
>
> 程序不读任何环境变量来覆盖模型路径或下载源，确保第三方测试结果可复现。

下面是**可调**的环境/输出/调试类参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | （必填） | 数据集根目录 |
| `--output` | `./run_output` | 本次输出目录 |
| `--cuda-devices` | `0` | `CUDA_VISIBLE_DEVICES` |
| `--max-samples` | `-1` | 仅跑前 N 个样本（试跑） |
| `--class` | （无） | 仅评测指定类别 slug |
| `--no-viz` | （未开） | 不保存逐样本可视化 |
| `--no-resume` | （未开） | 强制全量重跑 |
| `--seed` | `42` | 随机种子（仅影响数据加载顺序，不影响推理结果） |

`run_benchmark.sh` 还接受少量便利环境变量：`CUDA_DEVICES` / `MAX_SAMPLES` / `EXTRA_ARGS`，详见 `bash scripts/run_benchmark.sh --help`。

### 关于 SOR 去噪参数

`SOR_NB / SOR_STD` 控制 mask→点云→AABB 的精度，原项目实测组合：

| 参数组合 | 0.25 阈值精度 |
|---|---|
| `nb=20, std=0.75` ★（**已锁定**） | 0.837 |
| `nb=30, std=0.75` | 0.842 |
| `nb=5,  std=0.5`  | 0.862 |

虽然其它组合在该 1000 样本子集上数值更高，但**测试程序的目的是评测，不是为了刷分**。
为保证不同环境结果可比，本程序锁定为 `nb=20, std=0.75`。如果你确实需要研究/调优，
请直接修改 `src/benchmark/config.py` 顶部的 `SOR_NB / SOR_STD` 并明确标注。

---

## 10. 常见问题

### Q1: 安装时 `libGL.so.1: cannot open shared object file`

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

### Q2: 出现 `CUDA out of memory`

- 检查显存：`nvidia-smi`
- 关闭其它占用显存的进程
- 用更大显存的卡（建议 ≥ 24GB）
- 暂未支持显存优化（量化 / offload），如需要请告知。

### Q3: `KeyError: 'sample_results'` 或类似的 `results.json` 解析错误

可能是上一次运行被异常打断、`results.json` 写到一半。删掉它重跑即可：

```bash
rm ./run_*/results.json
bash scripts/run_benchmark.sh /path/to/dataset_1000
```

### Q4: 评测中途服务器断电了，怎么办？

直接重新执行同样的命令，会自动从断点续跑（`--resume` 默认开）。`results.json` 是「先写 .tmp 再 rename」原子写入，断电不会损坏。

### Q5: 想看看模型在某一张图上具体的预测

跑完测试后看 `run_<时间戳>/visualizations/det_seg_<sample_id>.jpg`，包含：

- 蓝色半透明：SAM2 mask
- 绿色框：RynnBrain pred bbox（标 "PRED"）
- 黄色框：GT 2D bbox（标 "GT"）
- 顶部文字：sample_id + 3D IoU + OK/FAIL

### Q6: 数据集路径里有中文怎么办？

测试程序对中文路径友好（脚本里都用 `Path` 与 utf-8 读写）。但建议第三方机器上把数据集放到纯英文路径下，避免某些系统配置导致的诡异问题。

### Q7: 想以编程方式调用，不通过 CLI？

```python
from pathlib import Path
import sys
sys.path.insert(0, "src")
from benchmark.config import BenchmarkConfig
from benchmark.runner import run_benchmark
from benchmark.report import write_html_report, write_json_report

cfg = BenchmarkConfig(
    dataset_root=Path("/path/to/dataset_1000"),
    output_dir=Path("./run_api"),
    max_samples=10,  # 试跑
)
# 注意：
# - 模型路径由 model_resolver 全权管理，固定下载到 <项目根>/models/，
#   编程时也不应传 rynnbrain_model_path / sam2_model_path。
# - iou_threshold / sor_nb / sor_std / enable_existence_check / sam2_model_id
#   是只读 property，由 config.py 顶部常量决定，无法在构造时覆盖。
results = run_benchmark(cfg)
write_html_report(cfg, results, env_snap={})
write_json_report(cfg, results)
```

### Q8: 报告里的 SVG 柱状图想导出 PDF / PNG

最简单：浏览器打开 `report.html` → 打印 → 另存为 PDF。SVG 是矢量，缩放不失真。

---

## 11. 采集与可视化检索（RealSense D4xx + UI）

> 📄 **完整教程见 [`README_CAPTURE_UI.md`](./README_CAPTURE_UI.md)**（含参数表、数据格式、坐标系约定与排错）。下面是速览。

除了对已发布数据集跑评测外，本程序还提供一条「**现场采集 → 交互检索**」的链路，
用 Intel RealSense D435（或其它 D4xx）现采 RGBD，输入想找的物体，实时可视化
**2D bbox + 半透明掩码 + 3D AABB**。这条链路与评测链路**完全独立**，不影响任何评测功能。

### 11.1 采集数据 `sample_rsd4xx.py`

接好 RealSense 相机后：

```bash
# 交互式：弹预览窗，按 <空格>/<回车> 采一帧，按 q/ESC 退出
python sample_rsd4xx.py --output ./captures

# 无人值守：自动采 20 帧，每帧间隔 1 秒（无窗口，适合 SSH/无显示环境）
python sample_rsd4xx.py --output ./captures --num 20 --auto --interval 1.0
```

每个样本会保存到 `./captures/samples/<sample_id>/`：

| 文件 | 说明 |
|---|---|
| `rgb.jpg` | 已对齐到彩色坐标系的 RGB |
| `depth.png` | 16-bit 深度（与彩色对齐，单位由 `depth_scale` 决定） |
| `intrinsics.json` | `{fx, fy, cx, cy, depth_scale, width, height}`（直接取自相机） |
| `pose.json` | **随机模拟**的相机→世界位姿（平移 + 四元数 + 4×4 矩阵） |

并在 `./captures/samples.json` 维护样本索引。

> 说明：采集样本用随机模拟的 `pose.json` 取代了评测数据集里的 GT `aabb.json`，
> 因此不参与「3D IoU 准确率」评测；它服务于下面的交互检索可视化。

### 11.2 可视化检索 `ui.py`

```bash
python ui.py --capture-dir ./captures
# 指定显卡： python ui.py --capture-dir ./captures --cuda-devices 0
```

界面操作：

1. 左侧选择一个采集样本（自动显示 RGB）；
2. 在「查找物体」输入框输入名称（支持中文，如 `水杯`），点「查找」或回车；
3. 后台跑 **RynnBrain 定位 → SAM2 分割 → 深度反投影 + SOR 去噪 → AABB**，
   结果直接叠加在图上：绿框=2D bbox，半透明蓝=掩码；
4. 右下信息区显示 **相机系 3D AABB** 与 **应用随机位姿后的世界系 3D AABB**（min/max/尺寸）；
5. 点「查看3D点云」用 Open3D 弹窗查看场景点云 + 目标点云（红）+ AABB（红框）。

> - 模型在**首次检索时惰性加载**（与评测共用 `model_resolver`，自动下载到 `./models/`），
>   首次点击「查找」会稍慢，之后复用。
> - 推理在后台线程进行，界面不卡顿。
> - 反投影与去噪参数与评测一致（`SOR_NB=20, SOR_STD=0.75`），见 `src/benchmark/config.py`。

### 11.3 依赖

`requirements.txt` 已包含：

- `pyrealsense2`（采集端；Linux x86_64 / Windows 有官方 wheel）
- `open3d`（UI 的三维点云显示，同时也是评测的点云去噪库）
- UI 用的 Tkinter 是 Python 标准库；若系统 Python 报 `No module named _tkinter`，
  执行 `sudo apt-get install -y python3-tk`（`install.sh` 已自动尝试安装）。

---

## 联系与反馈

- 提交 Issue 时请附上 `run_<时间戳>/env_snapshot.json` 和最后 50 行 `process.log`。
- 对结果有疑问，可以从 `results.json` 找到对应 `sample_id`，再看 `visualizations/det_seg_<sample_id>.jpg` 一眼定位问题（mask 错？bbox 错？还是反投影错？）。
