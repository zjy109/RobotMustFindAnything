# Benchmark 模块

## 概述

本模块提供项目的**基准测试数据集**和**硬件控制工具**，用于评估目标检测与语义分割模型（DINO-X、SAM3 等）的检测能力，并通过步进电机控制实现空间定位基准平台的搭建。

### 模块组成

| 子模块 | 路径 | 说明 |
|--------|------|------|
| 测试样例 | `benchmark/example/` | 12 张标注场景图片 + 结构化描述文件 |
| 步进电机控制 | `benchmark/spatial_localization/zdt_ttl.py` | ZDT 步进电机 TTL 串口通讯控制类 |

---

## 测试样例数据集

### 目录结构

```
rofa/benchmark/example/
├── 1.jpg ~ 12.jpg        # 12 张场景测试图片
└── discription.json       # 场景与物体的结构化描述文件
```

### 数据格式 (`discription.json`)

JSON 文件为数组，每个元素描述一张图片：

```json
{
    "image_filename": "1.jpg",
    "scene_understanding": {
        "description_zh": "场景中文描述",
        "description_en": "Scene English description"
    },
    "object_descriptions": [
        {
            "object_id": 1,
            "difficulty": "容易找到 (Easy)",
            "short_zh": "简短中文描述",
            "detailed_zh": "详细中文描述",
            "short_en": "short English description",
            "detailed_en": "Detailed English description with context."
        }
    ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `image_filename` | `str` | 图片文件名 |
| `scene_understanding` | `dict` | 场景级别的语义描述（中英双语） |
| `object_descriptions` | `list` | 该场景中需要检测的物体列表（每张图 5 个） |
| `object_id` | `int` | 物体序号 |
| `difficulty` | `str` | 难度等级：`容易找到 (Easy)` / `中等 (Medium)` / `不容易找到 (Hard)` |
| `short_en` / `short_zh` | `str` | 物体简短描述（名称级别） |
| `detailed_en` / `detailed_zh` | `str` | 物体详细描述（含外观、位置、上下文特征） |

### 场景覆盖

12 张测试图片涵盖以下场景类型：

- 冷藏展示柜 / 自动售货机
- 卧室（人体模型 + 床铺）
- 杂乱台面（微波炉、路由器等）
- 办公桌（多屏、键盘、零食）
- 洗衣机角落
- 卫浴设备混搭
- 办公室打印区
- 搬迁中的杂物堆放

### 评测方式

测试脚本（`test_dinox.py`、`test_sam3.py`）从 `discription.json` 读取物体描述，分别使用 `short_en`（简短描述）和 `detailed_en`（详细描述）作为 prompt 进行检测，并统计检测成功率。

---

## 步进电机控制 (`ZDTTTL`)

### 概述

`ZDTTTL` 类封装了 ZDT 步进电机的 TTL 串口通讯协议（Emm_V5 协议），通过 USB 转 TTL（CH340 芯片）控制步进电机，用于空间定位基准测试平台的构建。

### 安装依赖

```bash
pip install pyserial
```

### 快速开始

```python
from benchmark.spatial_localization.zdt_ttl import ZDTTTL

# 自动检测 CH340 串口并初始化（也可手动指定 port='/dev/ttyUSB0'）
motor = ZDTTTL()

# 使能电机
motor.enable(addr=1, state=True)

# 顺时针旋转 45°
motor.move_angle(addr=1, angle=45.0, vel=300, acc=50)

# 读取位置
pos = motor.read_position(addr=1)
print(f"当前角度: {pos:.1f}°")

# 失能并关闭
motor.enable(addr=1, state=False)
motor.close()
```

支持 `with` 上下文管理器：

```python
with ZDTTTL() as motor:
    motor.enable(1, True)
    motor.move_angle(1, 90.0)
    motor.enable(1, False)
```

### 命令行测试

```bash
python zdt_ttl.py --port /dev/ttyUSB0 --addr 1 --speed 300 --acc 50
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 自动检测 | 串口设备路径 |
| `--addr` | `1` | 电机地址 (1-247) |
| `--baudrate` | `115200` | 波特率 |
| `--speed` | `300` | 运动速度 (RPM) |
| `--acc` | `50` | 加速度 |
| `--pulses-per-rev` | `3200` | 每圈脉冲数 |

### API 参考

#### 构造函数

```python
ZDTTTL(port=None, baudrate=115200, timeout=0.2)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `port` | `str \| None` | `None` | 串口路径，`None` 时自动检测 CH340 |
| `baudrate` | `int` | `115200` | 波特率 |
| `timeout` | `float` | `0.2` | 读取超时 (秒) |

#### 运动控制方法

| 方法 | 说明 |
|------|------|
| `enable(addr, state, sync)` | 电机使能/失能 |
| `vel_control(addr, direction, vel, acc, sync)` | 速度模式控制 |
| `pos_control(addr, direction, vel, acc, pulses, relative, sync)` | 位置模式控制 |
| `move_angle(addr, angle, vel, acc, pulses_per_rev, wait, timeout)` | 按角度运动（便捷方法） |
| `stop(addr, sync)` | 立即停止电机 |
| `sync_motion(addr)` | 多机同步运动 |

#### 状态读取方法

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `read_position(addr)` | `float` | 实时位置角度 (度) |
| `read_velocity(addr)` | `float` | 实时转速 (RPM) |
| `read_bus_voltage(addr)` | `float` | 总线电压 (V) |
| `read_status_flags(addr)` | `dict \| None` | 使能/到位/堵转状态 |
| `read_sys_params(addr, param)` | — | 读取驱动板参数 |

#### 回零相关方法

| 方法 | 说明 |
|------|------|
| `origin_set_zero(addr, save)` | 设置当前位置为零点 |
| `origin_modify_params(addr, ...)` | 修改回零参数 |
| `origin_trigger(addr, o_mode, sync)` | 触发回零 |
| `origin_interrupt(addr)` | 中断回零 |

#### 等待方法

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `wait_for_arrival(addr, timeout, poll_interval)` | `bool` | 轮询等待到位 |
| `wait_for_origin_done(addr, timeout, poll_interval)` | `bool` | 轮询等待回零完成 |

#### 方向常量

| 常量 | 值 | 说明 |
|------|----|------|
| `ZDTTTL.CW` | `0` | 顺时针 |
| `ZDTTTL.CCW` | `1` | 逆时针 |



https://graphics.stanford.edu/projects/bundlefusion/#data