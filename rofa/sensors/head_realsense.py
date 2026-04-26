import threading
import time
import argparse

import numpy as np

try:
    import rospy
    from message_filters import ApproximateTimeSynchronizer, Subscriber
    from sensor_msgs.msg import CameraInfo, Image
except ModuleNotFoundError as exc:
    missing_module = getattr(exc, "name", None)
    if missing_module in {"rospy", "message_filters", "sensor_msgs"}:
        raise ModuleNotFoundError(
            "ROS1 RealSense dependencies are missing. "
            "Please source ROS Noetic first. "
            "Example: `conda activate rofa && source /opt/ros/noetic/setup.bash`."
        ) from exc
    raise


class HeadRealsense:
    def __init__(
        self,
        camera_ns="/camera",
        color_topic=None,
        depth_topic=None,
        camera_info_topic=None,
        queue_size=10,
        slop=0.1,
        node_name="head_realsense_reader",
        anonymous=True,
    ):
        """
        订阅 ROS1 RealSense 驱动发布的对齐 RGB、Depth 和 CameraInfo。

        默认订阅以下话题:
            {camera_ns}/color/image_raw
            {camera_ns}/aligned_depth_to_color/image_raw
            {camera_ns}/color/camera_info

        多相机场景下，可通过 camera_ns 区分，例如:
            /head_camera
            /wrist_camera
        """
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=anonymous, disable_signals=True)

        self.camera_ns = camera_ns.rstrip("/")
        self.color_topic = color_topic or (self.camera_ns + "/color/image_raw")
        self.depth_topic = depth_topic or (self.camera_ns + "/aligned_depth_to_color/image_raw")
        self.camera_info_topic = camera_info_topic or (self.camera_ns + "/color/camera_info")

        self._lock = threading.Lock()
        self._latest_bundle = None

        self._color_sub = Subscriber(self.color_topic, Image)
        self._depth_sub = Subscriber(self.depth_topic, Image)
        self._info_sub = Subscriber(self.camera_info_topic, CameraInfo)

        self._sync = ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub, self._info_sub],
            queue_size=int(queue_size),
            slop=float(slop),
        )
        self._sync.registerCallback(self._sync_callback)

    @staticmethod
    def _image_msg_to_numpy(image_msg):
        encoding = image_msg.encoding.lower()
        height = int(image_msg.height)
        width = int(image_msg.width)

        if encoding in ("bgr8", "rgb8"):
            channels = 3
            dtype = np.uint8
        elif encoding in ("bgra8", "rgba8"):
            channels = 4
            dtype = np.uint8
        elif encoding in ("mono8", "8uc1"):
            channels = 1
            dtype = np.uint8
        elif encoding in ("mono16", "16uc1", "16sc1"):
            channels = 1
            dtype = np.uint16 if encoding != "16sc1" else np.int16
        elif encoding in ("32fc1",):
            channels = 1
            dtype = np.float32
        else:
            raise ValueError("Unsupported image encoding: {}".format(image_msg.encoding))

        bytes_per_channel = np.dtype(dtype).itemsize
        row_stride = int(image_msg.step) // bytes_per_channel
        image = np.frombuffer(image_msg.data, dtype=dtype).reshape(height, row_stride)
        image = image[:, : width * channels]
        if channels == 1:
            return image.reshape(height, width)
        return image.reshape(height, width, channels)

    @staticmethod
    def _normalize_color_to_rgb(color_image, encoding):
        encoding = str(encoding).lower()
        if encoding == "rgb8":
            return color_image
        if encoding == "bgr8":
            return color_image[:, :, ::-1]
        if encoding == "rgba8":
            return color_image[:, :, :3]
        if encoding == "bgra8":
            return color_image[:, :, :3][:, :, ::-1]
        raise ValueError("Unsupported color image encoding: {}".format(encoding))

    def _sync_callback(self, color_msg, depth_msg, camera_info_msg):
        try:
            color_image = self._image_msg_to_numpy(color_msg)
            depth_image = self._image_msg_to_numpy(depth_msg)
            color_image = self._normalize_color_to_rgb(color_image, color_msg.encoding)
        except Exception as exc:
            rospy.logerr("Failed to decode RealSense image messages: %s", exc)
            return

        bundle = {
            "rgb": np.asarray(color_image),
            "depth": np.asarray(depth_image),
            "camera_info": camera_info_msg,
            "rgb_msg": color_msg,
            "depth_msg": depth_msg,
            "stamp": color_msg.header.stamp,
        }

        with self._lock:
            self._latest_bundle = bundle

    def has_frame(self):
        with self._lock:
            return self._latest_bundle is not None

    def get_current_aligned_rgbd(self, timeout_sec=3.0, copy=True, raise_on_timeout=False):
        """
        返回当前最新的一组同步数据。

        :param timeout_sec: 等待首帧/新帧可用的超时时间。
        :param copy: True 时返回 numpy 副本，避免外部误改内部缓存。
        :param raise_on_timeout: 超时是否抛异常。
        :return: dict，包含 rgb / depth / camera_info / stamp；失败返回 None。
        """
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)

        while not rospy.is_shutdown():
            with self._lock:
                if self._latest_bundle is not None:
                    bundle = self._latest_bundle
                    if copy:
                        bundle = {
                            "rgb": bundle["rgb"].copy(),
                            "depth": bundle["depth"].copy(),
                            "camera_info": bundle["camera_info"],
                            "rgb_msg": bundle["rgb_msg"],
                            "depth_msg": bundle["depth_msg"],
                            "stamp": bundle["stamp"],
                        }
                    return bundle

            if time.monotonic() >= deadline:
                break
            rospy.sleep(0.02)

        message = (
            "Timeout waiting for RealSense topics: {}, {}, {}".format(
                self.color_topic,
                self.depth_topic,
                self.camera_info_topic,
            )
        )
        if raise_on_timeout:
            raise TimeoutError(message)
        rospy.logwarn(message)
        return None

    def get_current_rgb_depth_camera_info(self, timeout_sec=3.0, copy=True, raise_on_timeout=False):
        """
        简化接口，直接返回 (rgb, depth, camera_info)。
        """
        bundle = self.get_current_aligned_rgbd(
            timeout_sec=timeout_sec,
            copy=copy,
            raise_on_timeout=raise_on_timeout,
        )
        if bundle is None:
            return None, None, None
        return bundle["rgb"], bundle["depth"], bundle["camera_info"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-ns", default="/camera", help="RealSense camera namespace, e.g. /camera or /head_camera")
    args = parser.parse_args()

    camera_ns = args.camera_ns
    reader = HeadRealsense(camera_ns=camera_ns)
    bundle = reader.get_current_aligned_rgbd(timeout_sec=5.0, raise_on_timeout=True)

    rgb = bundle["rgb"]
    depth = bundle["depth"]
    camera_info = bundle["camera_info"]

    print("Got aligned RealSense frame:")
    print("camera_ns: {}".format(camera_ns))
    print("rgb shape: {}, dtype: {}".format(rgb.shape, rgb.dtype))
    print("depth shape: {}, dtype: {}".format(depth.shape, depth.dtype))
    print("camera_info size: {}x{}".format(camera_info.width, camera_info.height))
    print("camera_info K: {}".format(list(camera_info.K)))
    print("stamp: {:.6f}".format(bundle["stamp"].to_sec()))


if __name__ == "__main__":
    main()
