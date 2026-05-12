import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, Shutdown, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    slam_launch = os.path.join(
        get_package_share_directory('slam_toolbox'),
        'launch',
        'online_async_launch.py'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(slam_launch),
            launch_arguments={
                'use_sim_time': 'true',
                'autostart': 'true',
            }.items()
        ),
        TimerAction(
            period=0.5,
            actions=[
                Node(
                    package='ele434_team10_2026',
                    executable='explore.py',
                    name='explore_node',
                    output='screen',
                    parameters=[{'use_sim_time': True}],
                    on_exit=Shutdown(reason='exploration node finished')
                )
            ]
        )
    ])
