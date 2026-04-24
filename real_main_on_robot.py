import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import zmq

from rofa.main_on_robot import MainOnRobot, MainOnRobotState
from rofa.sensor import Sensor


def _camera_info_to_intrinsics(camera_info):
    if camera_info is None:
        raise ValueError("CameraInfo is required to build camera intrinsics")
    k = list(camera_info.K)
    if len(k) != 9:
        raise ValueError(f"CameraInfo.K length must be 9, got {len(k)}")
    return {
        "fx": float(k[0]),
        "fy": float(k[4]),
        "cx": float(k[2]),
        "cy": float(k[5]),
    }


def _to_builtin(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    return value


def _snapshot_summary(snapshot):
    pose2d = snapshot["pose2d"]
    return {
        "state": snapshot["state"],
        "frame_id": snapshot["frame_id"],
        "pose2d": _to_builtin(pose2d),
        "camera_pose_valid": bool(snapshot["camera_pose_valid"]),
        "anchor_count": len(snapshot["anchors"]),
        "last_query": snapshot["last_query"],
        "search_result": _to_builtin(snapshot["search_result"]),
        "event": _to_builtin(snapshot["event"]),
    }


def _build_search_response(snapshot):
    search_result = snapshot["search_result"]
    if search_result is None:
        raise RuntimeError("search completed without search_result")
    if search_result["status"] != "success":
        raise RuntimeError(search_result["error"])
    return {
        "query": search_result["query"],
        "anchor_id": search_result["anchor_id"],
        "target_xy": _to_builtin(search_result["target_xy"]),
        "center": _to_builtin(search_result["center"]),
        "extent": _to_builtin(search_result["extent"]),
    }


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


def _build_argparser():
    parser = argparse.ArgumentParser(description="Deploy MainOnRobot on a real robot with socket RPC")
    parser.add_argument("--roimap-root", type=str, default="rofa/roimap_fixed_data", help="ROI map 输出目录")
    parser.add_argument("--server-host", type=str, default="127.0.0.1", help="远端检索服务 host")
    parser.add_argument("--server-port", type=int, default=5555, help="远端检索服务端口")
    parser.add_argument("--command-host", type=str, default="0.0.0.0", help="本地命令服务监听 host")
    parser.add_argument("--command-port", type=int, default=6000, help="本地命令服务监听端口")
    parser.add_argument("--camera-ns", type=str, default="/camera", help="RealSense ROS namespace")
    parser.add_argument("--map-frame", type=str, default="map", help="TF map frame")
    parser.add_argument("--base-frame", type=str, default="base_footprint", help="TF base frame")
    parser.add_argument("--frame-id-prefix", type=str, default="frame", help="FrameId 前缀")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="深度缩放")
    parser.add_argument("--search-timeout-seconds", type=float, default=10.0, help="搜索服务超时秒数")
    parser.add_argument("--search-hold-seconds", type=float, default=0.0, help="搜索结果停留秒数")
    parser.add_argument("--sensor-timeout-seconds", type=float, default=3.0, help="读取传感器超时秒数")
    parser.add_argument("--loop-sleep-seconds", type=float, default=0.0, help="建图线程每轮额外 sleep 秒数")
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    roimap_root = Path(args.roimap_root).expanduser().resolve()
    sensor = Sensor(
        camera_ns=args.camera_ns,
        map_frame=args.map_frame,
        base_frame=args.base_frame,
        frame_id_prefix=args.frame_id_prefix,
    )

    print("[startup] waiting for first sensor frame...")
    first_frame = sensor.get_current_posed_rgbd(timeout_sec=args.sensor_timeout_seconds)
    camera_intrinsics = _camera_info_to_intrinsics(first_frame.get("CameraInfo"))
    print(f"[startup] camera intrinsics: {camera_intrinsics}")

    robot = MainOnRobot(
        roimap_root=roimap_root,
        server_host=args.server_host,
        server_port=args.server_port,
        camera_intrinsics=camera_intrinsics,
        depth_scale=args.depth_scale,
        search_timeout_seconds=args.search_timeout_seconds,
        search_hold_seconds=args.search_hold_seconds,
    )

    robot_lock = threading.Lock()
    stop_event = threading.Event()

    with robot_lock:
        initial_snapshot = robot.process_frame(first_frame)
    print(f"[startup] initial state: {_snapshot_summary(initial_snapshot)}")

    mapping_thread = threading.Thread(
        target=_mapping_loop,
        args=(
            robot,
            sensor,
            robot_lock,
            stop_event,
            float(args.sensor_timeout_seconds),
            float(args.loop_sleep_seconds),
        ),
        daemon=True,
        name="mapping-loop",
    )
    mapping_thread.start()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.command_host}:{args.command_port}")
    print(f"[rpc] listening on tcp://{args.command_host}:{args.command_port}")
    print('[rpc] supported commands: {"command":"status"} | {"command":"set_anchor"} | {"command":"search","query":"..."} | {"command":"stop"}')

    try:
        while not stop_event.is_set():
            request_raw = socket.recv()
            try:
                request = json.loads(request_raw.decode("utf-8"))
                if not isinstance(request, dict):
                    raise TypeError("request must be a JSON object")
            except Exception as exc:
                socket.send(
                    json.dumps({"success": False, "error": f"invalid request: {exc}"}, ensure_ascii=False).encode("utf-8")
                )
                continue

            command = str(request.get("command", "")).strip().lower()
            response = None

            try:
                with robot_lock:
                    if command == "status":
                        response = {
                            "success": True,
                            "command": command,
                            "status": _snapshot_summary(robot.snapshot()),
                        }
                    elif command == "set_anchor":
                        snapshot = robot.set_anchor_from_last_frame()
                        event = snapshot["event"] or {}
                        success = event.get("event") == "manual_set_anchor"
                        response = {
                            "success": success,
                            "command": command,
                            "status": _snapshot_summary(snapshot),
                        }
                        if not success:
                            response["error"] = event.get("reason", "set_anchor failed")
                    elif command == "search":
                        query = request.get("query", request.get("instruction", ""))
                        snapshot = robot.search(query)
                        search_payload = _build_search_response(snapshot)
                        if robot.state == MainOnRobotState.SHOWING_SEARCH_RESULT:
                            robot.resume_mapping()
                        response = {
                            "success": True,
                            "command": command,
                            "result": search_payload,
                            "status": _snapshot_summary(robot.snapshot()),
                        }
                    elif command == "stop":
                        snapshot = robot.stop()
                        stop_event.set()
                        response = {
                            "success": True,
                            "command": command,
                            "status": _snapshot_summary(snapshot),
                        }
                    else:
                        raise ValueError(f"unsupported command: {command}")
            except Exception as exc:
                response = {
                    "success": False,
                    "command": command,
                    "error": str(exc),
                }

            socket.send(json.dumps(_to_builtin(response), ensure_ascii=False).encode("utf-8"))
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
