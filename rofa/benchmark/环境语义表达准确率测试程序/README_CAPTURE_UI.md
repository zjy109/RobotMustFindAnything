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
4. [可视化检索 `ui.py`（Web 版）](#4-可视化检索-uipyweb-版)
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
| UI 依赖 | `gradio`（网页界面）+ `open3d`（导出三维点云供浏览器预览） |
| 推理依赖 | 与评测相同：`torch` + `transformers` + NVIDIA GPU（建议显存 ≥ 24GB） |
| 系统包(Ubuntu) | 一般无需（用 `opencv-python-headless`）；RealSense 采集需 `libusb-1.0-0` |

安装（已并入主安装脚本，无需额外操作）：

```bash
bash scripts/install.sh          # 含 gradio / pyrealsense2 / open3d
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

### 3.1 没有真机？用官方 `.bag` 录制回放（无需任何硬件）

pyrealsense2 可以把 RealSense `.bag` 录制文件当成一台「虚拟相机」回放，
内参 / 深度 / 对齐与真机完全一致，因此**没有相机也能跑通整条链路**。

下载 Intel 官方样例 `.bag`（任选其一）：

```bash
# Intel 官方样例数据列表：
#   https://github.com/IntelRealSense/librealsense/blob/master/doc/sample-data.md
wget https://librealsense.intel.com/rs-tests/TestData/outdoors.bag
# 其它示例：stairs.bag / depth_under_water.bag 等（同一目录下）
```

> 也可以用别人电脑上的 `realsense-viewer` 录一段 `.bag` 发给你；
> 或从 Intel 论坛 / Roboflow / Kaggle 搜索 "realsense bag" 获取社区录制。

回放采集成样本：

```bash
python sample_rsd4xx.py --from-bag ./outdoors.bag --output ./captures --num 10
# --bag-stride 控制每隔多少帧采一帧（默认 30）
```

然后照常 `python ui.py --capture-dir ./captures` 即可。

> 注意：`.bag` 里不一定同时含彩色与深度流；若只有深度，`ui.py` 仍可反投影出 AABB，
> 但 RynnBrain 需要彩色图才能定位，建议选含 **Color + Depth** 的录制（如 `outdoors.bag`）。

### 3.2 已有 RGB+深度图片，想直接喂给 UI？

如果你手上是普通的 RGBD 数据集（一对 `rgb.png` + `depth.png` + 内参，如
TUM RGB-D / NYU Depth V2 / Redwood），只要按
[§5 数据格式](#5-数据格式与坐标系约定) 摆成
`samples/<id>/{rgb.jpg, depth.png, intrinsics.json, pose.json}` 并写一个
`samples.json` 索引即可被 `ui.py` 读取（`pose.json` 可随便给个单位矩阵）。
如需要，我可以补一个「通用 RGBD → 采集样本」的转换脚本。

---

## 4. 可视化检索 `ui.py`（Web 版）

`ui.py` 是一个**单文件 Web 应用**（基于 Gradio）。最简单的用法就是：
SSH 到（有 GPU 的）服务器 → 运行它 → 浏览器打开终端里打印的网址。

```bash
python ui.py --capture-dir ./captures
# 指定显卡 / 端口
python ui.py --capture-dir ./captures --cuda-devices 0 --port 7860
```

启动后终端会打印 `http://127.0.0.1:7860`，浏览器打开即可。

### 检索方式：输入物体名，系统自动扫描全部样本

**不需要手动选样本**。你只在输入框里写要找的东西（支持中文），点「检索全部样本」，
系统会**遍历采集目录下的所有样本**，对每个样本跑
**RynnBrain 定位 → SAM2 分割 → 深度反投影 + SOR 去噪 → AABB**，
然后把**命中目标的样本**以图集（Gallery）形式返回：

1. 顶部输入「采集目录 + 查找物体」，点「检索全部样本」（或回车）；
2. 进度条显示扫描进度，结束后概要给出「扫描 N 个 / 命中 M 个」；
3. 图集展示所有命中样本（每张图叠加 **绿框 = 2D bbox**、**半透明蓝 = 掩码**，标题为样本号）；
4. **点击图集中任意一张**，下方显示该样本的 **相机系 3D AABB** 与 **应用随机位姿后的世界系 3D AABB**（min / max / 尺寸），
   以及可在浏览器内旋转缩放的 **3D 点云 + AABB（红框）**。

### 行为说明

- **纯浏览器渲染**，不依赖服务器显示 / X11 / 服务器 OpenGL；3D 点云用浏览器端 three.js 渲染。
- **模型惰性加载**：第一次检索时才加载 RynnBrain + SAM2（自动下载到 `./models/`），之后复用。
- 样本较多时，扫描是逐样本推理（每个约 1.5~3s），进度条会显示进度。
- 未命中：若 RynnBrain 判定某样本不含该物体，则该样本不出现在结果里。
- 反投影 / 去噪参数与评测一致（`SOR_NB=20, SOR_STD=0.75`），保证结果可对照。

### 远程访问（SSH 端口转发）

`ui.py` 默认 `--host 127.0.0.1`，直接 SSH 连服务器时，用端口转发把网页带回本地浏览器：

```bash
# 在本地电脑执行（保持窗口开着）
ssh -N -L 7860:localhost:7860 root@<服务器IP>
```

保持隧道窗口开着，本地浏览器打开 **http://localhost:7860**。

> 若本地能直接访问服务器 IP，也可在服务器上用 `--host 0.0.0.0` 启动，
> 然后浏览器直接开 `http://<服务器IP>:7860`（需放开防火墙/安全组的该端口；
> `0.0.0.0` 会暴露给所有能访问该 IP 的人，内网可控再用）。
>
> 启动时 `ui.py` 会自动把 localhost 加入 `no_proxy`，避免代理拦截本地自检请求导致 503。

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

### Q3: `ui.py` 报 `No module named gradio`

```bash
pip install gradio
```

### Q4: 浏览器打不开网址 / `ui.py` 启动报 503

- 503 通常是服务器设了 `http_proxy/https_proxy`，本地自检请求被代理拦截。
  `ui.py` 启动时已自动把 localhost 加入 `no_proxy`；若仍报错，手动执行：
  ```bash
  export no_proxy="localhost,127.0.0.1,::1"; export NO_PROXY="localhost,127.0.0.1,::1"
  ```
- 远程访问要做 SSH 端口转发：`ssh -N -L 7860:localhost:7860 root@<服务器IP>`，
  再打开本地 `http://localhost:7860`。

### Q5: 「3D 点云 + AABB」面板空白 / 没有三维

三维点云需要 `open3d` 导出 `.ply`：
```bash
pip install open3d
```
缺失时仍可正常看 2D 叠加图与 AABB 数值，只是没有三维预览。

### Q6: 检索很慢 / 第一次卡住

首次检索会**下载并加载** RynnBrain-8B(~16GB) + SAM2(~200MB)。可先预热：
```bash
python scripts/download_models.py
```
推理需要 NVIDIA GPU；CPU 上会非常慢甚至 OOM。另外「检索全部样本」是逐样本推理，
样本越多越慢（每个约 1.5~3s），属正常现象，进度条会显示进度。

### Q7: 这些采集样本能用 `main.py` 评测吗？

不能直接评测。采集样本用**随机模拟 `pose.json`** 取代了评测集的 GT `aabb.json`，
没有 3D 真值，因此不参与「3D IoU 准确率」评测；它服务于本文档的交互检索可视化。
