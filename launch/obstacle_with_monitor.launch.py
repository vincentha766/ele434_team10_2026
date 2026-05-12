import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_monitor = LaunchConfiguration('use_monitor')
    log_dir_arg = LaunchConfiguration('log_dir')
    launch_dir = os.path.dirname(__file__)
    workspace_root = os.path.abspath(os.path.join(launch_dir, '..', '..', '..', '..', '..'))
    src_log_dir = os.path.join(workspace_root, 'src', 'ele434_team10_2026', 'robot_logs')
    fallback_log_dir = os.path.abspath(os.path.join(launch_dir, '..', 'robot_logs'))
    default_log_dir = src_log_dir if os.path.isdir(os.path.join(workspace_root, 'src', 'ele434_team10_2026')) else fallback_log_dir

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_monitor',
            default_value='true',
            description='Whether to start robot_monitor node',
        ),
        DeclareLaunchArgument(
            'log_dir',
            default_value=default_log_dir,
            description='Directory to save robot logs',
        ),
        Node(
            package='ele434_team10_2026',
            executable='obstacle.py',
            name='coverage_navigation_node',
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='ele434_team10_2026',
            executable='monitor_node.py',
            name='robot_monitor',
            output='screen',
            condition=IfCondition(use_monitor),
            parameters=[
                {'log_dir': log_dir_arg},
            ],
            emulate_tty=True,
        ),
    ])
