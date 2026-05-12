import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    launch_dir = os.path.dirname(__file__)
    workspace_root = os.path.abspath(os.path.join(launch_dir, '..', '..', '..', '..', '..'))
    src_log_dir = os.path.join(workspace_root, 'src', 'ele434_team10_2026', 'robot_logs')
    fallback_log_dir = os.path.abspath(os.path.join(launch_dir, '..', 'robot_logs'))
    default_log_dir = src_log_dir if os.path.isdir(os.path.join(workspace_root, 'src', 'ele434_team10_2026')) else fallback_log_dir
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
