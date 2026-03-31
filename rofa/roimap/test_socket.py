# client.py
import zmq
import cv2
import numpy as np
import time
import unicodedata
import os

# --- 辅助函数：中文对齐 ---
def get_display_width(s):
    width = 0
    for char in s:
        if unicodedata.east_asian_width(char) in ('W', 'F'): width += 2
        else: width += 1
    return width

def align_text(text, width, align='left'):
    text = str(text)
    current_width = get_display_width(text)
    space_count = max(0, width - current_width)
    if align == 'left': return text + ' ' * space_count
    elif align == 'right': return ' ' * space_count + text
    else:
        l = space_count // 2
        return ' ' * l + text + ' ' * (space_count - l)

def run_benchmark(server_ip, n_trials=30):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{server_ip}:5555")

    # --- 阶段 1：预处理（不计入耗时统计） ---
    print("正在预处理图片（读取、Resize、保存本地副本）...")
    original_img = cv2.imread("test.jpg")
    if original_img is None:
        print("错误：目录下未找到 test.jpg"); return

    save_dir = "client_results"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    res_map = {
        "640x480": cv2.resize(original_img, (640, 480)),
        "1280x720": cv2.resize(original_img, (1280, 720))
    }
    
    # 提前保存客户端的处理后副本
    qualities = [50, 80, 95, "RAW"]
    for res_name, img_data in res_map.items():
        for q in qualities:
            mode_name = f"JPEG_{q}" if q != "RAW" else "RAW"
            path = os.path.join(save_dir, f"client_{res_name}_{mode_name}.jpg")
            if q == "RAW":
                cv2.imwrite(path, img_data)
            else:
                cv2.imwrite(path, img_data, [int(cv2.IMWRITE_JPEG_QUALITY), q])

    # --- 阶段 2：正式压测 ---
    cols = [12, 10, 10, 10, 10, 10, 8]
    print(f"\n全链路耗时分析 (每种配置运行 {n_trials} 次取均值)")
    print("-" * (sum(cols) + 18))
    header = (
        align_text("分辨率", cols[0]) + " | " + align_text("模式", cols[1]) + " | " +
        align_text("压缩/ms", cols[2], 'right') + " | " + align_text("网络/ms", cols[3], 'right') + " | " +
        align_text("解码/ms", cols[4], 'right') + " | " + align_text("总计/ms", cols[5], 'right') + " | " +
        align_text("大小/KB", cols[6], 'right')
    )
    print(header)
    print("-" * (sum(cols) + 18))

    for res_name, base_img in res_map.items():
        for q in qualities:
            list_comp, list_trans, list_decode, list_sizes = [], [], [], []
            mode_label = f"JPEG-{q}" if q != "RAW" else "RAW"
            config_id = f"{res_name}_{mode_label}"

            for _ in range(n_trials):
                # 1. 客户端压缩计时
                t1 = time.perf_counter()
                if q == "RAW":
                    # RAW 模式为了能让服务端 imdecode，我们转为不压缩的 BMP 格式
                    _, encoded = cv2.imencode('.bmp', base_img)
                    data = encoded.tobytes()
                else:
                    _, encoded = cv2.imencode('.jpg', base_img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
                    data = encoded.tobytes()
                t2 = time.perf_counter()
                
                # 2. 传输计时（发送：配置名 + 图片数据）
                t3 = time.perf_counter()
                socket.send_multipart([config_id.encode('utf-8'), data])
                server_resp = socket.recv_string()
                t4 = time.perf_counter()

                # 3. 统计
                decode_ms = float(server_resp)
                comp_ms = (t2 - t1) * 1000
                net_ms = ((t4 - t3) * 1000 - decode_ms) / 2 # 估算单程网络延时
                
                list_comp.append(comp_ms); list_trans.append(net_ms)
                list_decode.append(decode_ms); list_sizes.append(len(data)/1024)
                time.sleep(0.005)

            # 打印均值
            m_comp, m_net, m_decode, m_size = np.mean(list_comp), np.mean(list_trans), np.mean(list_decode), np.mean(list_sizes)
            row = (
                align_text(res_name, cols[0]) + " | " + align_text(mode_label, cols[1]) + " | " +
                align_text(f"{m_comp:.2f}", cols[2], 'right') + " | " + align_text(f"{m_net:.2f}", cols[3], 'right') + " | " +
                align_text(f"{m_decode:.2f}", cols[4], 'right') + " | " + align_text(f"{m_comp+m_net+m_decode:.2f}", cols[5], 'right') + " | " +
                align_text(f"{m_size:.1f}", cols[6], 'right')
            )
            print(row)

    socket.close(); context.term()

if __name__ == "__main__":
    run_benchmark("219.223.200.92", n_trials=30) # 替换为你的4090 IP
