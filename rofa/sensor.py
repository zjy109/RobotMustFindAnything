import time

import cv2
import numpy as np

try:
    from sensors.chassis import Chassis
    from sensors.head_realsense import HeadRealsense
except ImportError:
    from rofa.sensors.chassis import Chassis
    from rofa.sensors.head_realsense import HeadRealsense


class Sensor:
    CAMERA_LINK_X_M = 0.4
    CAMERA_LINK_Y_M = 0.03
    CAMERA_LINK_Z_M = 1.5

    def __init__(
        self,
        camera_ns="/camera",
        map_frame="map",
        base_frame="base_footprint",
        frame_id_prefix="frame",
    ):
        self.camera_ns = camera_ns
        self.frame_id_prefix = str(frame_id_prefix)

        self.head_realsense = HeadRealsense(camera_ns=camera_ns)
        self.chassis = Chassis(map_frame=map_frame, base_frame=base_frame)

        self._camera_to_base = np.eye(4, dtype=np.float32)
        self._camera_to_base[0, 3] = float(self.CAMERA_LINK_X_M)
        self._camera_to_base[1, 3] = float(self.CAMERA_LINK_Y_M)
        self._camera_to_base[2, 3] = float(self.CAMERA_LINK_Z_M)

    @staticmethod
    def _quaternion_to_rotation_matrix(x, y, z, w):
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )

    def _base_pose_to_matrix(self, base_pose):
        q = base_pose.pose.orientation
        t = base_pose.pose.position

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = self._quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        transform[0, 3] = float(t.x)
        transform[1, 3] = float(t.y)
        transform[2, 3] = float(t.z)
        return transform

    def get_current_camera_pose(self, timeout_sec=3.0):
        base_pose = self.chassis.get_current_base_footprint(
            timeout_sec=timeout_sec,
            raise_on_error=True,
        )
        map_to_base = self._base_pose_to_matrix(base_pose)
        return np.matmul(map_to_base, self._camera_to_base)

    def get_current_posed_rgbd(self, timeout_sec=3.0):
        bundle = self.head_realsense.get_current_aligned_rgbd(
            timeout_sec=timeout_sec,
            raise_on_timeout=True,
        )
        camera_pose = self.get_current_camera_pose(timeout_sec=timeout_sec)

        stamp = bundle["stamp"].to_sec()
        frame_id = "{}_{:.6f}".format(self.frame_id_prefix, stamp if stamp > 0 else time.time())

        return {
            "FrameId": frame_id,
            "RGB": np.asarray(bundle["rgb"], dtype=np.uint8),
            "Depth": np.asarray(bundle["depth"]),
            "CameraPose": np.asarray(camera_pose, dtype=np.float32),
            "CameraInfo": bundle["camera_info"],
        }


def _depth_to_vis(depth):
    depth = np.asarray(depth)
    depth_float = depth.astype(np.float32)
    positive = depth_float[depth_float > 0]
    vmax = np.percentile(positive, 99) if positive.size else 1.0
    scaled = np.clip(depth_float / max(vmax, 1.0) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


if __name__ == "__main__":
    sensor = Sensor()
    print("Showing latest posed RGBD. Press any key in the image window to exit.")

    try:
        while True:
            posed_rgbd = sensor.get_current_posed_rgbd(timeout_sec=5.0)
            rgb_bgr = cv2.cvtColor(posed_rgbd["RGB"], cv2.COLOR_RGB2BGR)
            depth_vis = _depth_to_vis(posed_rgbd["Depth"])

            cv2.putText(
                rgb_bgr,
                posed_rgbd["FrameId"],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Sensor RGB", rgb_bgr)
            cv2.imshow("Sensor Depth", depth_vis)

            if cv2.waitKey(1) != -1:
                break
    finally:
        cv2.destroyAllWindows()
