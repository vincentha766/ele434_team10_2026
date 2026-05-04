#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class RobotMonitor(Node):
    def __init__(self):
        super().__init__('robot_monitor')

        # 参数设置
        self.declare_parameter('log_dir', 'robot_logs')
        self.log_dir = self.get_parameter('log_dir').get_parameter_value().string_value
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = os.path.join(self.log_dir, f'odom_data_{now_str}.csv')
        self.plot_file = os.path.join(self.log_dir, f'velocity_plot_{now_str}.png')
        self.logic_log_file = os.path.join(self.log_dir, f'avoidance_logic_{now_str}.txt')

        # 数据缓存
        self.timestamps = []
        self.linear_v = []
        self.angular_v = []
        self.min_dist = 3.5

        # 订阅话题
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(String, '/avoidance_log', self.avoidance_log_callback, 10)
        self.start_time = self.get_clock().now()

        # 初始化 CSV 文件
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'x', 'y', 'yaw', 'v_lin', 'v_ang', 'min_scan'])

        self.get_logger().info(f'监控节点启动。数据将保存至: {self.log_dir}')
        with open(self.logic_log_file, 'w') as f:
            f.write(f'避障逻辑记录启动时间: {datetime.now()}\n')
            f.write('-' * 50 + '\n')

    def odom_callback(self, msg):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9

        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        # 四元数转偏航角
        siny_cosp = 2 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1 - 2 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        v_lin = msg.twist.twist.linear.x
        v_ang = msg.twist.twist.angular.z

        self.timestamps.append(t)
        self.linear_v.append(v_lin)
        self.angular_v.append(v_ang)

        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([t, pos.x, pos.y, yaw, v_lin, v_ang, self.min_dist])

    def scan_callback(self, msg):
        ranges = [r for r in msg.ranges if not math.isnan(r) and not math.isinf(r) and r > 0.05]
        if not ranges:
            return

        new_min_dist = min(ranges)
        # 简单的避障逻辑判定记录 (保留作为雷达直接触发的保底记录)
        if new_min_dist < 0.5 and new_min_dist < self.min_dist - 0.05:
            t = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            with open(self.logic_log_file, 'a') as f:
                f.write(f'[{t:.2f}s] [雷达警告] 距离极近: {new_min_dist:.2f}m\n')

        self.min_dist = new_min_dist

    def avoidance_log_callback(self, msg):
        """
        接收来自导航节点的详细避障状态
        """
        t = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        with open(self.logic_log_file, 'a') as f:
            f.write(f'[{t:.2f}s] [避障逻辑] {msg.data}\n')
        self.get_logger().info(f'[避障逻辑] {msg.data}')

    def finalize(self):
        self.get_logger().info('正在生成速度折线图...')
        if not self.timestamps:
            self.get_logger().warn('没有接收到足够的数据，无法生成图表。')
            return

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.timestamps, self.linear_v, 'b-', label='Linear Velocity (m/s)')
        plt.ylabel('Linear V')
        plt.legend()
        plt.grid(True)
        plt.title('Robot Velocity Monitor')

        plt.subplot(2, 1, 2)
        plt.plot(self.timestamps, self.angular_v, 'r-', label='Angular Velocity (rad/s)')
        plt.ylabel('Angular V')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.plot_file)
        plt.close()
        self.get_logger().info(f'折线图已保存至: {self.plot_file}')
        self.get_logger().info(f'Odom数据已保存至: {self.csv_file}')
        self.get_logger().info(f'避障日志已保存至: {self.logic_log_file}')


def main(args=None):
    rclpy.init(args=args)
    node = RobotMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
