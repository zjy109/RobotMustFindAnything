import json
import threading
import time
from pathlib import Path

import zmq

from rofa.main_on_robot import MainOnRobot, MainOnRobotState
from rofa.sensor import Sensor
from rofa.utils import build_search_response, camera_info_to_intrinsics, snapshot_summary, to_builtin



ROIMAP_ROOT = "rofa/roimap_fixed_data"  # ROI 地图输出目录，用于保存/读取机器人建图数据。
SERVER_HOST = "219.223.193.175"  # 远端检索服务 IP 或域名。
SERVER_PORT = 5555  # 远端检索服务端口。
COMMAND_HOST = "0.0.0.0"  # 本地 RPC 服务监听地址，0.0.0.0 表示监听所有网卡。
COMMAND_PORT = 6000  # 本地 RPC 服务监听端口。
CAMERA_NS = "/camera"  # 相机对应的 ROS namespace。
MAP_FRAME = "map"  # TF 中的全局地图坐标系名称。
BASE_FRAME = "base_footprint"  # TF 中的机器人底盘坐标系名称。
FRAME_ID_PREFIX = "frame"  # 传感器帧编号前缀。
DEPTH_SCALE = 0.001  # 深度图缩放系数，通常用于毫米到米的转换。
SEARCH_TIMEOUT_SECONDS = 10.0  # 单次搜索请求的超时时间，单位秒。
SEARCH_HOLD_SECONDS = 0.0  # 搜索结果展示保持时间，单位秒，0.0 表示不额外停留。
SENSOR_TIMEOUT_SECONDS = 3.0  # 读取传感器数据的超时时间，单位秒。
LOOP_SLEEP_SECONDS = 0.0  # 建图线程每轮循环后的额外休眠时间，单位秒，0.0 表示不主动休眠。


def _mapping_loop(robot, sensor, robot_lock, stop_event, sensor_timeout_sec, loop_sleep_sec):
    last_frame_id = None
    while not stop_event.is_set():
        try:
            posed_rgbd = sensor.get_current_posed_rgbd(timeout_sec=sensor_timeout_sec)
        except Exception as exc:
            print(f"[mapping] sensor read failed: {exc}")
            time.sleep(max(loop_sleep_sec, 0.05))
            continue

        frame_id = posed_rgbd.get("FrameId")
        if frame_id == last_frame_id:
            time.sleep(max(loop_sleep_sec, 0.01))
            continue
        last_frame_id = frame_id

        with robot_lock:
            if robot.state == MainOnRobotState.STOPPED:
                stop_event.set()
                break
            if robot.state == MainOnRobotState.SHOWING_SEARCH_RESULT:
                robot.tick()
            else:
                robot.process_frame(posed_rgbd)

        if loop_sleep_sec > 0:
            time.sleep(loop_sleep_sec)


# 初始化函数
def initialize_robot_system():
    roimap_root = Path(ROIMAP_ROOT).expanduser().resolve()
    sensor = Sensor(
        camera_ns=CAMERA_NS,
        map_frame=MAP_FRAME,
        base_frame=BASE_FRAME,
        frame_id_prefix=FRAME_ID_PREFIX,
    )

    print("[startup] waiting for first sensor frame...")
    first_frame = sensor.get_current_posed_rgbd(timeout_sec=SENSOR_TIMEOUT_SECONDS)
    camera_intrinsics = camera_info_to_intrinsics(first_frame.get("CameraInfo"))
    print(f"[startup] camera intrinsics: {camera_intrinsics}")

    robot = MainOnRobot(
        roimap_root=roimap_root,
        server_host=SERVER_HOST,
        server_port=SERVER_PORT,
        camera_intrinsics=camera_intrinsics,
        depth_scale=DEPTH_SCALE,
        search_timeout_seconds=SEARCH_TIMEOUT_SECONDS,
        search_hold_seconds=SEARCH_HOLD_SECONDS,
    )

    robot_lock = threading.Lock()
    stop_event = threading.Event()

    with robot_lock:
        initial_snapshot = robot.process_frame(first_frame)
    print(f"[startup] initial state: {snapshot_summary(initial_snapshot)}")

    mapping_thread = threading.Thread(
        target=_mapping_loop,
        args=(
            robot,
            sensor,
            robot_lock,
            stop_event,
            float(SENSOR_TIMEOUT_SECONDS),
            float(LOOP_SLEEP_SECONDS),
        ),
        daemon=True,
        name="mapping-loop",
    )
    mapping_thread.start()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{COMMAND_HOST}:{COMMAND_PORT}")
    print(f"[rpc] listening on tcp://{COMMAND_HOST}:{COMMAND_PORT}")
    print('[rpc] supported commands: {"command":"status"} | {"command":"set_anchor"} | {"command":"search","query":"..."} | {"command":"stop"}')

    return {
        "robot": robot,
        "robot_lock": robot_lock,
        "stop_event": stop_event,
        "mapping_thread": mapping_thread,
        "context": context,
        "socket": socket,
    }


# 功能函数
def handle_rpc_request(request_raw, runtime):
    robot = runtime["robot"]
    robot_lock = runtime["robot_lock"]
    stop_event = runtime["stop_event"]

    try:
        request = json.loads(request_raw.decode("utf-8"))
        if not isinstance(request, dict):
            raise TypeError("request must be a JSON object")
    except Exception as exc:
        return {"success": False, "error": f"invalid request: {exc}"}

    command = str(request.get("command", "")).strip().lower()

    try:
        with robot_lock:
            if command == "status":
                return {
                    "success": True,
                    "command": command,
                    "status": snapshot_summary(robot.snapshot()),
                }
            if command == "set_anchor":
                snapshot = robot.set_anchor_from_last_frame()
                event = snapshot["event"] or {}
                success = event.get("event") == "manual_set_anchor"
                response = {
                    "success": success,
                    "command": command,
                    "status": snapshot_summary(snapshot),
                }
                if not success:
                    response["error"] = event.get("reason", "set_anchor failed")
                return response
            if command == "search":
                query = request.get("query", request.get("instruction", ""))
                snapshot = robot.search(query)
                search_payload = build_search_response(snapshot)
                if robot.state == MainOnRobotState.SHOWING_SEARCH_RESULT:
                    robot.resume_mapping()
                return {
                    "success": True,
                    "command": command,
                    "result": search_payload,
                    "status": snapshot_summary(robot.snapshot()),
                }
            if command == "stop":
                snapshot = robot.stop()
                stop_event.set()
                return {
                    "success": True,
                    "command": command,
                    "status": snapshot_summary(snapshot),
                }
            raise ValueError(f"unsupported command: {command}")
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        return {
            "success": False,
            "command": command,
            "error": str(exc),
        }


# 主函数
def main():
    runtime = initialize_robot_system()
    robot = runtime["robot"]
    robot_lock = runtime["robot_lock"]
    stop_event = runtime["stop_event"]
    mapping_thread = runtime["mapping_thread"]
    socket = runtime["socket"]
    context = runtime["context"]

    try:
        while not stop_event.is_set():
            request_raw = socket.recv()
            response = handle_rpc_request(request_raw, runtime)
            socket.send(json.dumps(to_builtin(response), ensure_ascii=False).encode("utf-8"))
    except KeyboardInterrupt:
        print("[shutdown] interrupted by user")
    finally:
        stop_event.set()
        with robot_lock:
            if robot.state != MainOnRobotState.STOPPED:
                robot.stop()
            robot.close()
        mapping_thread.join(timeout=2.0)
        socket.close(linger=0)
        context.term()
        print("[shutdown] exited cleanly")


if __name__ == "__main__":
    main()
