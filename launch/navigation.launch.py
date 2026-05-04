import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. 启动航点管理器
        Node(
            package='ele434_team10_2026',
            executable='waypoint_manager.py',
            name='waypoint_manager',
            output='screen'
        ),
        
        # 2. 启动 APF 控制器
        Node(
            package='ele434_team10_2026',
            executable='apf_controller.py',
            name='apf_controller',
            output='screen'
        )
    ])
