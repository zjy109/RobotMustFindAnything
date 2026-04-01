import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CmdVelTest(Node):
    def __init__(self):
        super().__init__('cmd_vel_test')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_cmd)
        self.start_time = self.get_clock().now().nanoseconds / 1e9

    def publish_cmd(self):
        now = self.get_clock().now().nanoseconds / 1e9
        t = now - self.start_time

        msg = Twist()

        if t < 5.0:
            msg.linear.x = 0.2
            msg.angular.z = 0.0
        elif t < 10.0:
            msg.linear.x = 0.0
            msg.angular.z = 0.5
        elif t < 15.0:
            msg.linear.x = 0.2
            msg.angular.z = 0.3
        else:
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()