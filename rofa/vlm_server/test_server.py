"""
VLM Search Server 测试脚本

演示如何向 VLM Search Server 发送搜索请求
"""
import json
import time
import zmq
from pathlib import Path
import argparse


def load_test_image(image_path):
    """加载测试图片并转换为 JPEG bytes"""
    from PIL import Image
    import io
    
    img = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def test_search_request(
    server_host="127.0.0.1",
    server_port=5555,
    image_paths=None,
    target_object="yellow cup",
    anchor_ids=None,
    save_results=False,
    timeout_ms=60000,
):
    """
    测试物体搜索请求
    
    Args:
        server_host: 服务器地址
        server_port: 服务器端口
        image_paths: 图片文件路径列表
        target_object: 目标物体描述
        anchor_ids: anchor ID 列表（如果为 None，自动生成）
        save_results: 是否保存结果
        timeout_ms: 接收超时时间（毫秒）
    """
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    
    print(f"连接到服务器: tcp://{server_host}:{server_port}")
    socket.connect(f"tcp://{server_host}:{server_port}")
    
    try:
        # 默认测试图片
        if image_paths is None:
            print("使用默认测试图片...")
            # 创建简单的测试图片
            from PIL import Image
            import io
            
            test_images_bytes = []
            for i in range(2):
                img = Image.new('RGB', (640, 480), color=(73 * i, 109 + i * 20, 137))
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=90)
                test_images_bytes.append(buffer.getvalue())
        else:
            test_images_bytes = [load_test_image(p) for p in image_paths]
        
        # 生成 anchor IDs
        if anchor_ids is None:
            anchor_ids = [f"anchor_{i:04d}" for i in range(len(test_images_bytes))]
        
        # 构建请求
        metadata = {
            "type": "search_request",
            "instruction": target_object,
            "anchors": [
                {"anchor_id": aid, "image_name": f"rgb_{i}.jpg"}
                for i, aid in enumerate(anchor_ids)
            ],
            "save_results": save_results,
        }
        
        # 发送 multipart 消息
        print(f"\n发送请求:")
        print(f"  目标物体: {target_object}")
        print(f"  图片数量: {len(test_images_bytes)}")
        print(f"  保存结果: {save_results}")
        print(f"  Anchors: {anchor_ids}")
        
        message_parts = [
            json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
            *test_images_bytes
        ]
        
        socket.send_multipart(message_parts)
        print("\n请求已发送，等待响应...")
        
        # 接收响应
        response = socket.recv()
        response_data = json.loads(response.decode("utf-8"))
        
        print("\n收到响应:")
        print(json.dumps(response_data, ensure_ascii=False, indent=2))
        
        # 解析结果
        if response_data.get("success"):
            if response_data.get("object_found"):
                print(f"\n✓ 物体找到！")
                print(f"  Anchor ID: {response_data.get('anchor_id')}")
                bbox = response_data.get("bbox")
                if bbox:
                    print(f"  BBox (归一化 [0, 1000]): {bbox}")
                    x1, y1, x2, y2 = bbox
                    print(f"    左上角: ({x1}, {y1})")
                    print(f"    右下角: ({x2}, {y2})")
                    print(f"    宽度: {x2 - x1}, 高度: {y2 - y1}")
            else:
                print(f"\n✗ 物体未找到")
        else:
            print(f"\n✗ 搜索失败: {response_data.get('error')}")
        
        return response_data
        
    finally:
        socket.close()
        context.term()


def main():
    parser = argparse.ArgumentParser(description="VLM Search Server 测试脚本")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器地址")
    parser.add_argument("--port", type=int, default=5555, help="服务器端口")
    parser.add_argument("--images", nargs="+", help="图片文件路径")
    parser.add_argument("--object", type=str, default="yellow cup", help="目标物体描述")
    parser.add_argument("--save-results", action="store_true", help="保存检测结果")
    parser.add_argument("--timeout", type=int, default=60000, help="接收超时时间（毫秒）")
    parser.add_argument("--test-mode", action="store_true", help="使用默认生成的测试图片")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VLM Search Server 测试")
    print("=" * 60)
    
    try:
        result = test_search_request(
            server_host=args.host,
            server_port=args.port,
            image_paths=args.images if not args.test_mode else None,
            target_object=args.object,
            save_results=args.save_results,
            timeout_ms=args.timeout,
        )
        
        print("\n" + "=" * 60)
        if result.get("success"):
            print("✓ 测试成功")
        else:
            print("✗ 测试失败")
        print("=" * 60)
        
    except zmq.error.Again:
        print("\n✗ 错误: 服务器响应超时")
        print("请确保服务器正在运行: python server.py")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
