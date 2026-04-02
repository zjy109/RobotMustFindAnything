from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    robot_description_content = Command([
        'xacro ',
        PathJoinSubstitution([
            FindPackageShare('my_robot_description'),
            'urdf',
            'robot.urdf.xacro'
        ])
    ])

    robot_description = {
        'robot_description': robot_description_content
    }

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[robot_description]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_broadcaster',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
            output='screen'
        ),

        ExecuteProcess(
            cmd=[
                'gazebo',
                '--verbose',
                '-s', 'libgazebo_ros_factory.so'
            ],
            output='screen'
        ),

        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', 'my_robot',
                '-topic', 'robot_description',
                '-x', '0.0',
                '-y', '0.0',
                '-z', '0.1'
            ],
            output='screen'
        )
    ])
