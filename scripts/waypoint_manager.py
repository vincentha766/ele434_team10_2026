#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import math

class WaypointManager(Node):
    """
    航点管理类：负责 12 宫格任务的状态机管理。
    """
    def __init__(self):
        super().__init__('waypoint_manager')
        
        # 1. 参数与状态
        self.waypoints = [ 
            (1.5,  0.5), (1.5, -0.5), (1.5, -1.5), (0.5, -1.5), 
            (-0.5, -1.5), (-1.5, -1.5), (-1.5, -0.5), (-1.5, 0.5),
            (-1.5,  1.5), (-0.5,  1.5), (0.5,  1.5), (1.5,  1.5)
        ]
        self.current_idx = 0
        self.BOX_TOL = 0.28  # 容差范围 [m]
        
        # 2. 发布者与订阅者
        self.goal_pub = self.create_publisher(Point, '/cmd_goal', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # 定时器：以 2Hz 频率广播当前目标点
        self.timer = self.create_timer(0.5, self.publish_goal)
        self.get_logger().info("航点管理器已启动，正在派发任务...")

    def odom_callback(self, msg):
        # 提取当前坐标
        curr_x = msg.pose.pose.position.x
        curr_y = msg.pose.pose.position.y
        
        # 获取目标坐标
        tx, ty = self.waypoints[self.current_idx]
        
        # 判定是否到达 (曼哈顿距离或欧氏距离均可)
        if abs(tx - curr_x) < self.BOX_TOL and abs(ty - curr_y) < self.BOX_TOL:
            self.current_idx = (self.current_idx + 1) % len(self.waypoints)
            self.get_logger().info(f"目标点已达成！切换至航点 {self.current_idx}: ({tx}, {ty})")

    def publish_goal(self):
        tx, ty = self.waypoints[self.current_idx]
        msg = Point()
        msg.x, msg.y, msg.z = float(tx), float(ty), 0.0
        self.goal_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = WaypointManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
