#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math

class APFController(Node):
    """
    人工势场法控制器：订阅目标点、雷达、里程计，发布速度指令。
    """
    def __init__(self):
        super().__init__('apf_controller')
        
        # 1. 核心超参数
        self.K_ATT = 1.6
        self.K_REP = 6.0
        self.D_SAFE = 0.65
        self.D_STOP = 0.22
        self.MAX_V = 0.26
        self.MAX_W = 1.82
        self.MIN_V = 0.06

        # 2. 实时状态
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.lidar_ranges = []
        self.target_x = None
        self.target_y = None

        # 3. 订阅者与发布者
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_subscription(Point, '/cmd_goal', self.goal_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # 4. 控制循环 (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("APF 控制器已就绪，等待目标指令...")

    # --- 回调函数集 ---
    def goal_callback(self, msg):
        self.target_x = msg.x
        self.target_y = msg.y

    def odom_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)

    def scan_callback(self, msg):
        self.lidar_ranges = [(3.5 if (math.isnan(r) or math.isinf(r) or r < 0.05) else r) for r in msg.ranges]

    # --- 辅助工具 ---
    def wrap_to_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    # --- 核心势场循环 ---
    def control_loop(self):
        if self.target_x is None or not self.lidar_ranges:
            return

        # 1. 引力分量
        dx, dy = self.target_x - self.odom_x, self.target_y - self.odom_y
        angle_to_goal = math.atan2(dy, dx)
        angle_error = self.wrap_to_pi(angle_to_goal - self.odom_yaw)

        # 2. 扇区分析 (提取关键视角)
        num_bins = len(self.lidar_ranges)
        idx_45 = num_bins // 8
        f_slice = self.lidar_ranges[-idx_45//2:] + self.lidar_ranges[:idx_45//2]
        fl_slice = self.lidar_ranges[idx_45//2 : 3*idx_45//2]
        fr_slice = self.lidar_ranges[-3*idx_45//2 : -idx_45//2]
        
        min_f, min_fl, min_fr = min(f_slice), min(fl_slice), min(fr_slice)
        obs_nearest = min(min_f, min_fl, min_fr)

        # 3. 动态引力权重
        track_weight = max(0.0, (obs_nearest - self.D_STOP) / (self.D_SAFE - self.D_STOP)) if obs_nearest < self.D_SAFE else 1.0
        omega_att = self.K_ATT * angle_error * track_weight

        # 4. 斥力分量
        omega_rep = 0.0
        if min_fl < self.D_SAFE: omega_rep -= (self.D_SAFE - min_fl) * self.K_REP
        if min_fr < self.D_SAFE: omega_rep += (self.D_SAFE - min_fr) * self.K_REP
        
        # 打破正前死锁
        if min_f < self.D_SAFE and abs(min_fl - min_fr) < 0.1:
            omega_rep += 1.5 * (self.D_SAFE - min_f)

        # 5. 速度合成
        final_w = max(min(omega_att + omega_rep, self.MAX_W), -self.MAX_W)
        
        alignment_factor = 1.0 - (abs(angle_error) / math.pi)
        obstacle_factor = max(0.1, (min_f - self.D_STOP) / (self.D_SAFE - self.D_STOP)) if min_f < self.D_SAFE else 1.0
        
        final_v = max(self.MAX_V * alignment_factor * obstacle_factor, self.MIN_V)

        # 6. 指令发布
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = float(final_v)
        cmd.twist.angular.z = float(final_w)
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = APFController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 停车
        stop_cmd = TwistStamped()
        node.cmd_pub.publish(stop_cmd)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
