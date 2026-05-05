import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    default_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'robot_logs'))
    return LaunchDescription([
        Node(
            package='ele434_team10_2026',
            executable='monitor_node.py',
            name='robot_monitor',
            output='screen',
            parameters=[
                {'log_dir': default_log_dir},
            ],
            emulate_tty=True,
        )
    ])
