#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    env_arg = DeclareLaunchArgument('environment', default_value='sim')

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('tuos_tb3_tools'),
                'launch',
                'slam.launch.py',
            )
        ),
        launch_arguments={'environment': LaunchConfiguration('environment')}.items(),
    )

    work_node = Node(
        package='ele434_team10_2026',
        executable='work.py',
        name='work',
        output='screen',
    )

    return LaunchDescription([
        env_arg,
        slam_launch,
        TimerAction(period=2.0, actions=[work_node]),
    ])
