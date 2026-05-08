import numpy as np


def camera_info_to_intrinsics(camera_info):
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


def to_builtin(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    return value


def snapshot_summary(snapshot):
    pose2d = snapshot["pose2d"]
    return {
        "state": snapshot["state"],
        "frame_id": snapshot["frame_id"],
        "pose2d": to_builtin(pose2d),
        "camera_pose_valid": bool(snapshot["camera_pose_valid"]),
        "anchor_count": len(snapshot["anchors"]),
        "last_query": snapshot["last_query"],
        "search_result": to_builtin(snapshot["search_result"]),
        "event": to_builtin(snapshot["event"]),
    }


def build_search_response(snapshot):
    search_result = snapshot["search_result"]
    if search_result is None:
        raise RuntimeError("search completed without search_result")
    if search_result["status"] != "success":
        raise RuntimeError(search_result["error"])
    return {
        "query": search_result["query"],
        "anchor_id": search_result["anchor_id"],
        "target_xy": to_builtin(search_result["target_xy"]),
        "center": to_builtin(search_result["center"]),
        "extent": to_builtin(search_result["extent"]),
    }
