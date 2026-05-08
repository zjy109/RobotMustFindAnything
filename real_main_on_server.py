from pathlib import Path

from rofa.vlm_server.server import VLMSearchServer, logger



SERVER_PORT = 5555  # VLM 服务监听端口。
MODEL_PATH = "/home/zjy/RynnBrain/models/RynnBrain-8B"  # RynnBrain 模型目录；需要按本机实际模型路径修改。
MODEL_DEVICE = "auto"  # 主模型推理设备，可选 auto/cuda/cpu。
CUDA_DEVICES = "0"  # 可见 CUDA 设备编号，单卡示例为 "0"，多卡示例为 "0,1"。
SAVE_RESULTS = False  # 是否保存检索结果图片和元数据到 result_images 目录。
NUM_WORKERS = -1  # 主循环处理请求次数，-1 表示无限循环处理。
SAM2_MODEL_ID = "facebook/sam2.1-hiera-small"  # SAM2 模型 ID。
SAM2_CACHE_DIR = None  # SAM2 模型缓存目录；为 None 时使用 server.py 内部默认目录。
SAM2_DEVICE = None  # SAM2 推理设备；为 None 时自动选择 cuda 或 cpu。


# 初始化函数
def initialize_server():
    model_path = str(Path(MODEL_PATH).expanduser())
    sam2_cache_dir = str(Path(SAM2_CACHE_DIR).expanduser()) if SAM2_CACHE_DIR else None

    server = VLMSearchServer(
        server_port=SERVER_PORT,
        model_path=model_path,
        device=MODEL_DEVICE,
        cuda_devices=CUDA_DEVICES,
        save_results=SAVE_RESULTS,
        sam2_model_id=SAM2_MODEL_ID,
        sam2_cache_dir=sam2_cache_dir,
        sam2_device=SAM2_DEVICE,
    )

    logger.info(
        "Server runtime initialized: "
        f"port={SERVER_PORT}, model_path={model_path}, device={MODEL_DEVICE}, "
        f"cuda_devices={CUDA_DEVICES}, save_results={SAVE_RESULTS}, num_workers={NUM_WORKERS}"
    )

    return {
        "server": server,
    }


# 功能函数
def handle_server_request(runtime):
    server = runtime["server"]
    server.handle_request()


# 主函数
def main():
    runtime = initialize_server()
    server = runtime["server"]

    try:
        logger.info(f"Starting wrapped VLM server main loop (num_workers={NUM_WORKERS})")
        if NUM_WORKERS == -1:
            while True:
                handle_server_request(runtime)
        else:
            for worker_index in range(NUM_WORKERS):
                logger.info(f"Handling request {worker_index + 1}/{NUM_WORKERS}")
                handle_server_request(runtime)
    except KeyboardInterrupt:
        logger.info("Wrapped VLM server interrupted by user")
    except Exception as exc:
        logger.error(f"Wrapped VLM server error: {exc}", exc_info=True)
        raise
    finally:
        server.close()


if __name__ == "__main__":
    main()
