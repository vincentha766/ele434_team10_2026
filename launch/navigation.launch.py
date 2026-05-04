import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    log_dir = os.path.join(os.getcwd(), 'robot_logs')

    return LaunchDescription([
        # 1. 启动航点管理器
        Node(
            package='ele434_team10_2026',
            executable='waypoint_manager.py',
            name='waypoint_manager',
            output='screen',
            emulate_tty=True
        ),
        
        # 2. 启动 APF 控制器
        Node(
            package='ele434_team10_2026',
            executable='apf_controller.py',
            name='apf_controller',
            output='screen',
            emulate_tty=True
        ),

        # 3. 启动 监控节点
        Node(
            package='ele434_team10_2026',
            executable='monitor_node.py',
            name='robot_monitor',
            output='screen',
            parameters=[
                {'log_dir': log_dir},
            ],
            emulate_tty=True
        )
    ])
