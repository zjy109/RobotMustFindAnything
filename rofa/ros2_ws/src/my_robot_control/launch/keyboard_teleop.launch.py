from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='teleop_twist_keyboard',
            executable='teleop_twist_keyboard',
            name='teleop_twist_keyboard',
            prefix='xterm -e',
            output='screen',
            parameters=[
                {'speed': 0.3},
                {'turn': 0.8},
            ],
            remappings=[
                ('/cmd_vel', '/cmd_vel')
            ]
        )
    ])