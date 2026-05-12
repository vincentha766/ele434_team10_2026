#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('tuos_task_sims'),
                'launch',
                'obstacle_avoidance.launch.py',
            )
        )
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('tuos_tb3_tools'),
                'launch',
                'slam.launch.py',
            )
        )
    )

    work_node = Node(
        package='ele434_team10_2026',
        executable='work.py',
        name='work',
        output='screen',
    )

    return LaunchDescription([
        sim_launch,
        TimerAction(period=5.0, actions=[slam_launch]),
        TimerAction(period=8.0, actions=[work_node]),
    ])
