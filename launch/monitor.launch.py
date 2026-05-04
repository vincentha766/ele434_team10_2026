import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ele434_team10_2026',
            executable='monitor_node.py',
            name='robot_monitor',
            output='screen',
            parameters=[
                {'log_dir': os.path.join(os.getcwd(), 'robot_logs')},
            ],
            emulate_tty=True,
        )
    ])
