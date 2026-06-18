# 采集与可视化检索（RealSense D4xx + UI）

> 本文档只讲「**现场采集 → 交互检索**」这条链路：用 Intel RealSense D435（或其它 D4xx）
> 现采 RGBD，输入想找的物体，实时可视化 **2D bbox + 半透明掩码 + 3D AABB**。
>
> 这条链路与评测链路（`main.py` / `scripts/run_benchmark.sh`）**完全独立**，
> 互不影响。评测相关说明见 [`README.md`](./README.md)。

---

## 目录

1. [流程总览](#1-流程总览)
2. [前置要求](#2-前置要求)
3. [采集数据 `sample_rsd4xx.py`](#3-采集数据-sample_rsd4xxpy)
4. [可视化检索 `ui.py`](#4-可视化检索-uipy)
5. [数据格式与坐标系约定](#5-数据格式与坐标系约定)
6. [常见问题](#6-常见问题)

---

## 1. 流程总览

```
┌───────────────┐   sample_rsd4xx.py    ┌──────────────────────┐
│ RealSense D435│ ────────────────────▶ │ 采集样本目录 captures/ │
│  (RGBD 数据)  │   采 RGBD + 随机位姿  │  rgb/depth/intr/pose  │
└───────────────┘                       └──────────┬───────────┘
                                                    │  ui.py 读取
                                                    ▼
   用户输入物体名 ──▶ RynnBrain 定位 ──▶ SAM2 分割 ──▶ 深度反投影 + SOR 去噪
                                                    │
                                                    ▼
        UI 可视化：2D bbox(绿) + 掩码(半透明蓝) + 3D AABB(相机系/世界系) + 点云
```

- **采集端**（`sample_rsd4xx.py`）：只依赖相机 + `pyrealsense2`，不加载大模型。
- **检索端**（`ui.py`）：复用评测同款 `RynnBrain + SAM2 + 反投影`，参数与评测一致
  （`SOR_NB=20, SOR_STD=0.75`，见 `src/benchmark/config.py`）。
- 模型由 `model_resolver` 自动下载到 `<项目根>/models/`，与评测共用，零配置。

---

## 2. 前置要求

| 项 | 说明 |
|---|---|
| 相机 | Intel RealSense D4xx（D435 / D435i / D455 等） |
| 采集依赖 | `pyrealsense2`（Linux x86_64 / Windows 有官方 wheel） |
| UI 依赖 | Tkinter（Python 标准库）+ `Pillow`；三维显示用 `open3d` |
| 推理依赖 | 与评测相同：`torch` + `transformers` + NVIDIA GPU（建议显存 ≥ 24GB） |
| 系统包(Ubuntu) | `python3-tk`、`libusb-1.0-0`、`libgl1`（`scripts/install.sh` 会自动尝试） |

安装（已并入主安装脚本，无需额外操作）：

```bash
bash scripts/install.sh          # 含 pyrealsense2 / open3d / python3-tk
# 或在已有环境中只装 Python 依赖
pip install -r requirements.txt
```

> 采集端与检索端**可以分机器**：在有相机的机器上采集，把 `captures/` 拷到有 GPU 的机器上跑 `ui.py`。

---

## 3. 采集数据 `sample_rsd4xx.py`

接好 RealSense 相机后运行：

```bash
# 交互式：弹预览窗，按 <空格>/<回车> 采一帧，按 q/ESC 退出
python sample_rsd4xx.py --output ./captures

# 无人值守：自动采 20 帧，每帧间隔 1 秒（无窗口，适合 SSH / 无显示环境）
python sample_rsd4xx.py --output ./captures --num 20 --auto --interval 1.0
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--output` | `./captures` | 采集结果输出目录 |
| `--num` | `-1` | 采集样本数上限；`-1` 表示不限制（交互式手动结束） |
| `--auto` | （未开） | 自动连续采集，无预览窗口，配合 `--num` / `--interval` |
| `--interval` | `1.0` | `--auto` 模式每帧间隔秒数 |
| `--width` / `--height` | `640` / `480` | 采集分辨率 |
| `--fps` | `30` | 帧率 |
| `--warmup` | `15` | 启动后丢弃的预热帧数（等自动曝光稳定） |
| `--seed` | （无） | 随机位姿种子，便于复现 |

### 输出

每个样本保存到 `./captures/samples/<sample_id>/`：

```
captures/
├── samples.json                       # 样本索引
└── samples/
    └── rsd_20260618_161200_123456/
        ├── rgb.jpg                     # 已对齐到彩色坐标系的 RGB
        ├── depth.png                   # 16-bit 深度（与彩色对齐）
        ├── intrinsics.json             # {fx, fy, cx, cy, depth_scale, width, height}
        └── pose.json                   # 随机模拟的相机->世界位姿
```

> 深度图已通过 `rs.align(rs.stream.color)` **对齐到彩色坐标系**，
> 因此 `rgb.jpg` 与 `depth.png` 像素一一对应，可直接配合彩色内参做反投影。

---

## 4. 可视化检索 `ui.py`

```bash
python ui.py --capture-dir ./captures
# 指定显卡
python ui.py --capture-dir ./captures --cuda-devices 0
```

### 界面布局

```
┌──────────────────────────────────────────────────────────────┐
│ 采集目录: [ ./captures            ] [选择] [刷新]              │
├──────────────┬───────────────────────────────────────────────┤
│ 样本列表      │                                               │
│ rsd_..._01   │          [ 图像显示区：RGB / 叠加结果 ]        │
│ rsd_..._02   │                                               │
│ ...          │                                               │
│              ├───────────────────────────────────────────────┤
│              │ 查找物体: [ 水杯        ] [查找] [查看3D点云]   │
│              ├───────────────────────────────────────────────┤
│              │ 结果信息：2D bbox / 相机系 AABB / 世界系 AABB   │
├──────────────┴───────────────────────────────────────────────┤
│ 状态栏：就绪 / 加载模型 / 检索中 / 完成                        │
└──────────────────────────────────────────────────────────────┘
```

### 操作流程

1. 左侧选择一个采集样本（自动显示该样本 RGB）。
2. 在「查找物体」输入名称（**支持中文**，如 `水杯` / `键盘`），点「查找」或回车。
3. 后台跑 **RynnBrain 定位 → SAM2 分割 → 深度反投影 + SOR 去噪 → AABB**：
   - 图上叠加：**绿框 = 2D bbox**，**半透明蓝 = 目标掩码**；
   - 右下信息区显示 **相机系 3D AABB** 与 **应用随机位姿后的世界系 3D AABB**（min / max / 尺寸）。
4. 点「查看3D点云」用 Open3D 弹窗查看：场景点云 + **目标点云(红)** + **AABB(红框)**。

### 行为说明

- **模型惰性加载**：首次点「查找」时才加载模型（自动下载到 `./models/`），稍慢；之后复用。
- **后台线程推理**：界面不卡顿，状态栏实时显示进度。
- **未找到目标**：若 RynnBrain 判定物体不存在或无法定位，会提示「未找到」，不画框。
- 反投影 / 去噪参数与评测一致，保证结果可对照。

---

## 5. 数据格式与坐标系约定

### `intrinsics.json`

```json
{
  "fx": 615.0, "fy": 615.0,
  "cx": 320.0, "cy": 240.0,
  "depth_scale": 0.001,
  "width": 640, "height": 480
}
```

- `depth_scale`：每个深度单位对应的米数（RealSense `get_depth_scale()` 直接给出，通常 `0.001`）。
- 反投影：`Z = depth_png[v,u] * depth_scale`，`X = (u-cx)*Z/fx`，`Y = (v-cy)*Z/fy`（相机系，单位米）。

### `pose.json`

```json
{
  "frame": "camera_to_world",
  "translation": [x, y, z],
  "quaternion_xyzw": [qx, qy, qz, qw],
  "matrix": [[r11,r12,r13,tx], [r21,r22,r23,ty], [r31,r32,r33,tz], [0,0,0,1]]
}
```

- 表示**相机系 → 世界系**的刚体变换（4×4 齐次矩阵，行优先）。
- 旋转为均匀随机四元数（Shoemake 法），平移在 `x,y∈[-1,1]`、`z∈[0,2]` 米内随机。
- UI 中「世界系 AABB」= 把相机系 AABB 的 8 个角点用该矩阵变换后，重新取轴对齐包围盒。

> 这是**模拟位姿**：真实场景中应替换为机器人/SLAM 给出的相机外参。
> 代码中位姿读写、`相机系↔世界系` 变换都集中在 `src/benchmark/capture_io.py`，便于替换。

---

## 6. 常见问题

### Q1: 运行 `sample_rsd4xx.py` 报 `No module named pyrealsense2`

```bash
pip install pyrealsense2
```
非 Linux/Windows 平台没有官方 wheel，请参考 [RealSense SDK 文档](https://github.com/IntelRealSense/librealsense) 从源码编译。

### Q2: 采集报 USB / 设备相关错误

```bash
sudo apt-get install -y libusb-1.0-0
```
并确认相机插在 **USB 3.0** 口、未被其它程序（如 realsense-viewer）占用。

### Q3: `ui.py` 报 `No module named _tkinter`

系统 Python 缺 Tk：
```bash
sudo apt-get install -y python3-tk
```
（conda 环境通常自带 Tkinter；`scripts/install.sh` 已尝试自动安装。）

### Q4: 「查看3D点云」没反应 / 报 open3d 错误

```bash
pip install open3d
```
无显示环境（纯 SSH 无 X11）下，Open3D 窗口无法弹出；请在带桌面或 X11 转发的环境运行，
或只看 UI 信息区里的 AABB 数值。

### Q5: 检索很慢 / 第一次卡住

首次检索会**下载并加载** RynnBrain-8B(~16GB) + SAM2(~200MB)。可先预热：
```bash
python scripts/download_models.py
```
推理需要 NVIDIA GPU；CPU 上会非常慢甚至 OOM。

### Q6: 这些采集样本能用 `main.py` 评测吗？

不能直接评测。采集样本用**随机模拟 `pose.json`** 取代了评测集的 GT `aabb.json`，
没有 3D 真值，因此不参与「3D IoU 准确率」评测；它服务于本文档的交互检索可视化。
