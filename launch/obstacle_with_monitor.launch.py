import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    log_dir = os.path.join(os.getcwd(), 'robot_logs')
    use_monitor = LaunchConfiguration('use_monitor')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_monitor',
            default_value='true',
            description='Whether to start robot_monitor node',
        ),
        Node(
            package='ele434_team10_2026',
            executable='obstacle.py',
            name='coverage_navigation_node',
            output='screen',
        ),
        Node(
            package='ele434_team10_2026',
            executable='monitor_node.py',
            name='robot_monitor',
            output='screen',
            condition=IfCondition(use_monitor),
            parameters=[
                {'log_dir': log_dir},
            ],
        ),
    ])
