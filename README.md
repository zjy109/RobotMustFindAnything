# RobotMustFindAnything

## 项目简介

RobotMustFindAnything 是一个机器人感知项目，集成了 RealSense 深度相机、ArUco 标记检测以及 Livox 激光雷达 + FAST-LIO2 建图等功能模块。

## 项目结构

```
RobotMustFindAnything/
├── main.py                 # 主程序入口
├── real_main_on_robot.py   # 启动机器人端的交互式语义地图构建和检索程序
├── real_main_on_server.py  # 启动vlm服务端并设置相关端口信息
├── README.md               # 本文件 - 环境配置说明
├── doc/                    # 各功能模块的详细使用文档
│   ├── realsense.md        # RealSense 相机使用说明
│   ├── aruco.md            # ArUco 标记检测使用说明
│   └── ros_ws.md           # ROS2 工作空间 & 激光雷达使用说明
└── rofa/                   # 功能模块源码
    ├── realsense/          # RealSense 相机模块 (D435)
    ├── aruco/              # ArUco 标记检测模块
    ├── sam3/               # SAM3 语义分割模块
    ├── llm/                # LLM 推理模块（基于 SiliconFlow）
    ├── dinox/              # DINO-X 模块（基于 deepdataspace的DINO-X 开放平台）
    └── ros_ws/             # ROS2 工作空间 (Livox 驱动 + FAST-LIO2)
    └── roimap/             # 环境语义检索机器人端（维护语义地图，发起检索请求）
    └── vlm_server/         # 环境语义检索服务端（基于vlm模型）
```

## 环境配置

### 1. Python 环境（Conda）

创建并激活名为 `rofa` 的 Conda 环境：

```bash
conda create -n rofa python=3.10.19
conda activate rofa
```

安装 Python 依赖：

```bash
pip install pyrealsense2 "numpy<2.0" opencv-python scipy openai
```

VLM服务端（rynnbrain模型依赖 + socket包） 环境配置
```bash
pip install --index-url "https://download.pytorch.org/whl/cu121" \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        torch==2.5.1 torchvision==0.20.1
pip install numpy<2.0 opencv-python>=4.8 Pillow>=10.0 tqdm>=4.65 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.57.1 accelerate==1.12.0 huggingface-hub>=0.20 safetensors>=0.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install pyzmq
```

rynnbrain 权重下载 (Modelscope)

```bash
modelscope download --model DAMO_Academy/RynnBrain-8B --local_dir ./models/RynnBrain-8B
```

SAM3 环境配置
```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
cd rofa/sam3/sam3_lib/sam3/
pip install -e .
pip install einops decord pycocotools psutil
pip install httpx[socks]
```


SAM3 权重下载 (Modelscope)

```bash
pip install modelscope
modelscope download --model facebook/sam3  # /home/user/.cache/modelscope/hub/models/facebook/sam3
```

LLM 模块配置

> LLM 模块用于基于场景描述进行智能位置推理，仅在需要语义位置查询功能时需要配置。

##### 获取 SiliconFlow API Key

1. 访问 [SiliconFlow 官网](https://siliconflow.cn/)
2. 注册账户并获取 API Key
3. 设置环境变量：

```bash
export SILICONFLOW_API_KEY="your_api_key_here"
```

DINO-X 模块配置

> DINO-X 模块用于强鲁棒性和高精度的开放词汇物体检测。

##### 获取 deepdataspace 的 DINO-X 开放平台 API Key

1. 访问 [DINO-X 开放平台 官网](https://cloud.deepdataspace.com/zh/playground/dino-x)
2. 注册账户并获取 API Key
3. 设置环境变量：

```bash
export DINO_API_TOKEN="your_api_key_here"
```

### 2. ROS 2 环境

> ROS 2 环境仅在使用激光雷达（Livox MID360）和 FAST-LIO2 建图时需要。如果仅使用 RealSense 相机或 ArUco 功能，可以跳过此部分。

#### 2.1 系统要求

- **操作系统**：Ubuntu 22.04
- **ROS 版本**：ROS 2 Humble

#### 2.2 安装 ROS 2 Humble

参考教程：[ROS 2 Humble 安装教程](https://comate.baidu.com/zh/page/w4rrweyik8h)

安装完成后验证：

```bash
echo $ROS_DISTRO
# 输出 humble 即为成功
```

#### 2.3 empy 依赖

ROS 2 编译需要 `empy` 库（版本需 < 4）：

```bash
pip install empy==3.3.4
```

如果使用系统级 Python 而非 Conda 环境，编译时需指定解释器：

```bash
colcon build --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3
```

#### 2.4 安装 Livox SDK

```bash
cd rofa/ros_ws/src/Livox-SDK2
mkdir build && cd build
cmake .. && make -j
sudo make install
```

#### 2.5 安装 livox_ros_driver2

```bash
cd rofa/ros_ws/src/livox_ros_driver2
./build.sh humble
```

#### 2.6 配置雷达网络

将雷达网口接入电脑，设置有线连接的静态 IP：

| 项目 | 值 |
|------|------|
| IP 地址 | `192.168.1.50` |
| 子网掩码 | `255.255.255.0` |
| 网关 | `192.168.1.255` |

编辑 MID360 配置文件 `rofa/ros_ws/src/livox_ros_driver2/config/MID360_config.json`：

- `host_net_info` 下的 4 个 IP 均改为 `192.168.1.50`
- `lidar_configs` 下的 `ip` 改为 `192.168.1.172`（最后两位对应雷达 SN 码）

#### 2.7 编译 ROS2 工作空间

```bash
cd rofa/ros_ws
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.sh
```

## 各模块详细文档

各功能模块的具体用法和 API 说明请参考 `doc/` 目录下的文档：

- [RealSense 相机](doc/realsense.md) — D435 相机的初始化、帧获取与深度对齐
- [ArUco 标记](doc/aruco.md) — ArUco 标记的检测与位姿估计
- [ROS2 工作空间](doc/ros_ws.md) — Livox 雷达数据获取与 FAST-LIO2 建图

## 参考教程

- [CSDN 教程 1](https://blog.csdn.net/2401_83327355/article/details/157468981?spm=1001.2014.3001.5506)
- [CSDN 教程 2](https://blog.csdn.net/2401_83327355/article/details/157511186?spm=1001.2014.3001.5506)
- [51CTO 教程](https://blog.51cto.com/u_12968/14213494)