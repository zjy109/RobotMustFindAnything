# 30 秒上手卡（QUICKSTART）

```
[1] 安装环境
    bash scripts/install.sh

[2] (可选) 预热模型权重
    python scripts/download_models.py
    # 一次性下载 RynnBrain-8B (~16GB) + SAM2 (~200MB) 到 <项目根>/models/
    # 默认先试 ModelScope（国内零配置），失败再走 HuggingFace
    # 跳过这步也可以：runner 缺权重时会自动下载

[3] 自检
    conda activate rofa-bench
    python scripts/check_env.py --dataset /path/to/dataset_1000

[4] 跑测试
    bash scripts/run_benchmark.sh /path/to/dataset_1000

[5] 看报告
    在浏览器打开:  ./run_<时间戳>/report.html
```

## 常见问题（一行答案）

- **CUDA 不可用？** → 检查显卡驱动 + `nvidia-smi`；安装时若 CUDA 是 11.8 用 `bash scripts/install.sh --torch-cuda cu118`。
- **显存不够？** → RynnBrain-8B + SAM2 大约需 ~18GB，建议 ≥24GB；可以 `CUDA_DEVICES=0,1` 多卡。
- **模型下载失败？** → 程序会先后尝试 ModelScope 和 HuggingFace，覆盖国内/海外网络；都失败请检查网络。
- **完全没外网？** → 在有网机器上跑 `python scripts/download_models.py`，把整个 `models/` 目录拷到测试机的项目根下即可。
- **跑到一半中断了？** → 直接重新执行同样的命令，会自动续跑（默认 `--resume` 开）。
- **想快速试跑？** → `MAX_SAMPLES=10 bash scripts/run_benchmark.sh /path/to/dataset_1000`
- **结果在哪？** → `./run_<时间戳>/report.html`（人话报告）+ `results.json`（原始数据）+ `visualizations/`（每张图的预测叠加）

详细教程见 [README.md](./README.md)。
