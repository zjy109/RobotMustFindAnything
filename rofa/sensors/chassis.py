import time

try:
    import rospy
    import tf2_ros
    from geometry_msgs.msg import PoseStamped
except ModuleNotFoundError as exc:
    missing_module = getattr(exc, "name", None)
    if missing_module in {"rospy", "tf2_ros", "geometry_msgs"}:
        raise ModuleNotFoundError(
            "ROS1 Python dependencies are missing. "
            "Please install/source ROS Noetic first, then run this file in the same shell. "
            "Example: `conda activate rofa && source /opt/ros/noetic/setup.bash`."
        ) from exc
    raise


class Chassis:
    def __init__(
        self,
        map_frame="map",
        odom_frame="odom",
        base_frame="base_footprint",
        cache_time_sec=10.0,
        node_name="chassis_tf_listener",
        anonymous=True,
    ):
        """
        底盘位姿读取类，通过 TF 获取机器人在 map 坐标系下的 base_footprint 位姿。

        TF 树约定为:
            map -> odom -> base_footprint

        实际查询时直接 lookup_transform(map, base_footprint)，
        tf2 会自动沿 TF 树完成链式拼接。
        """
        self.map_frame = map_frame
        self.odom_frame = odom_frame
        self.base_frame = base_frame

        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=anonymous, disable_signals=True)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(float(cache_time_sec)))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def get_current_base_footprint(self, timeout_sec=0.5, raise_on_error=False):
        """
        获取当前时刻下 base_footprint 在 map 坐标系中的位姿。

        :param timeout_sec: 等待 TF 可用的超时时间，单位秒。
        :param raise_on_error: 若为 True，查询失败时抛异常；否则返回 None。
        :return: PoseStamped，frame_id 为 map；失败时返回 None。
        """
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        last_error = None

        while not rospy.is_shutdown():
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.base_frame,
                    rospy.Time(0),
                    timeout=rospy.Duration(0.05),
                )

                pose = PoseStamped()
                pose.header = transform.header
                pose.pose.position.x = transform.transform.translation.x
                pose.pose.position.y = transform.transform.translation.y
                pose.pose.position.z = transform.transform.translation.z
                pose.pose.orientation = transform.transform.rotation
                return pose
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
                tf2_ros.TransformException,
            ) as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    break

        message = (
            "Failed to lookup TF {} -> {}. Expected TF chain like {} -> {} -> {}. "
            "Last error: {}"
        ).format(
            self.map_frame,
            self.base_frame,
            self.map_frame,
            self.odom_frame,
            self.base_frame,
            last_error,
        )

        if raise_on_error:
            raise RuntimeError(message)

        rospy.logwarn(message)
        return None


def main():
    chassis = Chassis()
    pose = chassis.get_current_base_footprint(timeout_sec=2.0, raise_on_error=True)
    print("base_footprint pose in map:")
    print(
        "position=({:.3f}, {:.3f}, {:.3f})".format(
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )
    )
    print(
        "orientation=({:.4f}, {:.4f}, {:.4f}, {:.4f})".format(
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        )
    )


if __name__ == "__main__":
    main()
