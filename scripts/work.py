#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math
import time

# =====================================================================
# [1] 全局变量定义
# =====================================================================
odom_ready = False
lidar_ready = False

odom_x = 0.0
odom_y = 0.0
odom_yaw = 0.0
lidar_ranges = []

# =====================================================================
# [2] 回调函数定义
# =====================================================================
def odom_callback(msg):
    global odom_ready, odom_x, odom_y, odom_yaw
    odom_x = msg.pose.pose.position.x
    odom_y = msg.pose.pose.position.y
    
    q = msg.pose.pose.orientation
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    odom_yaw = math.atan2(siny_cosp, cosy_cosp)
    
    odom_ready = True

def scan_callback(msg):
    global lidar_ready, lidar_ranges
    cleaned_ranges = []
    for r in msg.ranges:
        if math.isnan(r) or math.isinf(r) or r < 0.05:
            cleaned_ranges.append(3.5)
        else:
            cleaned_ranges.append(r)
    
    lidar_ranges = cleaned_ranges
    lidar_ready = True

def wrap_to_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

# =====================================================================
# [3] 主程序入口
# =====================================================================
def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('coverage_navigation_node')
    
    cmd_pub = node.create_publisher(TwistStamped, '/cmd_vel', 10)
    node.create_subscription(Odometry, '/odom', odom_callback, 10)
    node.create_subscription(LaserScan, '/scan', scan_callback, 10)

    # 算法参数配置
    p_max_v = 0.25           
    p_max_w = 1.5            
    p_safe_dist = 0.35       
    p_corner_weight = 1.0    
    enable_repeat = True     
    
    STOP_DIST = 0.22         
    BOX_LIMIT = 0.28         
    MIN_FORWARD_SPEED = 0.06 

    # 12 个外围得分区块坐标 (1号位于右上角，顺时针排列)
    waypoints = [
         (1.5,  1.5),  (1.5,  0.5),  (1.5, -0.5),  (1.5, -1.5), # 0,1,2,3
         (0.5, -1.5), (-0.5, -1.5), (-1.5, -1.5), (-1.5, -0.5), # 4,5,6,7
         (-1.5, 0.5), (-1.5,  1.5), (-0.5,  1.5),  (0.5,  1.5)  # 8,9,10,11
    ]
    num_waypoints = len(waypoints)
    current_idx = 0 
    
    initialized_start_point = False
    
    node.get_logger().info("覆盖导航节点已启动，正在等待传感器数据...")

    try:
        # =================================================================
        # [4] 控制循环 (20Hz)
        # =================================================================
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            
            if not odom_ready or not lidar_ready:
                continue
                
            if not initialized_start_point:
                node.get_logger().info(f"传感器就绪！前往 1 号区域: {waypoints[current_idx]}")
                initialized_start_point = True

            # --- A. 状态机与到达判定 ---
            target_x, target_y = waypoints[current_idx]
            dx = target_x - odom_x
            dy = target_y - odom_y
            
            if abs(dx) < BOX_LIMIT and abs(dy) < BOX_LIMIT:
                current_idx += 1 
                node.get_logger().info(f"成功打卡，前往下一区域 (索引: {current_idx})")
                
                if current_idx >= num_waypoints:
                    if enable_repeat:
                        current_idx = 0 
                        node.get_logger().info("完成一圈，重新开始循环！")
                    else:
                        node.get_logger().info("任务全部完成！申请停车。")
                        break
                
                target_x, target_y = waypoints[current_idx]
                dx = target_x - odom_x
                dy = target_y - odom_y

            # --- B. 激光雷达 8 扇区划分 ---
            num_scans = len(lidar_ranges)
            if num_scans >= 360:
                r_f  = lidar_ranges[337:360] + lidar_ranges[0:22] 
                r_fl = lidar_ranges[22:67]   
                r_l  = lidar_ranges[67:112]  
                r_r  = lidar_ranges[247:292] 
                r_fr = lidar_ranges[292:337] 
            else:
                r_f = r_fl = r_l = r_r = r_fr = lidar_ranges

            min_f  = min(r_f)
            min_fl = min(r_fl)
            min_fr = min(r_fr)
            min_l  = min(r_l)
            min_r  = min(r_r)

            # --- C. 核心势场计算 ---
            target_angle = math.atan2(dy, dx)
            angle_error = wrap_to_pi(target_angle - odom_yaw)
            
            min_scan = min(min_f, min_fl, min_fr, min_l, min_r)
            if min_scan < p_safe_dist:
                track_weight = max(0.0, (min_scan - STOP_DIST) / (p_safe_dist - STOP_DIST))
            else:
                track_weight = 1.0 
                
            omega_track = angle_error * 1.6 * track_weight
            repel_omega = 0.0
            
            is_in_corner = (min_fl < p_safe_dist) and (min_fr < p_safe_dist)
            
            if is_in_corner:
                is_major_corner = current_idx in [0, 3, 6, 9]
                if is_major_corner:
                    repel_omega = -p_corner_weight * 2.2
                else:
                    if (min_fl + min_l) > (min_fr + min_r) + 0.15:
                        repel_omega = p_corner_weight * 1.8  
                    else:
                        repel_omega = -p_corner_weight * 1.8 
            else:
                if min_fl < p_safe_dist: repel_omega -= (p_safe_dist - min_fl) * 6.0 
                if min_fr < p_safe_dist: repel_omega += (p_safe_dist - min_fr) * 6.0 
                if min_l < 0.28: repel_omega -= (0.28 - min_l) * 4.0
                if min_r < 0.28: repel_omega += (0.28 - min_r) * 4.0

            if min_f < p_safe_dist and not is_in_corner:
                bias_dir = 1.5 if (min_fl + min_l) > (min_fr + min_r) + 0.15 else -1.5
                repel_omega += bias_dir * (p_safe_dist - min_f)

            # --- D. 速度合成与约束 ---
            final_w = omega_track + repel_omega
            angular_vel = max(min(final_w, p_max_w), -p_max_w)

            speed_factor = 1.0 - (abs(angle_error) / math.pi)
            if min_f < p_safe_dist:
                obs_limit = max(0.05, (min_f - STOP_DIST) / (p_safe_dist - STOP_DIST))
                speed_factor = min(speed_factor, obs_limit)
            
            linear_vel = p_max_v * max(speed_factor, 0.2)
            
            if linear_vel < MIN_FORWARD_SPEED:
                linear_vel = MIN_FORWARD_SPEED
            if min_f <= STOP_DIST:
                linear_vel = MIN_FORWARD_SPEED

            # --- E. 发布 TwistStamped 控制指令 ---
            cmd = TwistStamped()
            
            cmd.header.stamp = node.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            cmd.twist.linear.x = float(linear_vel)
            cmd.twist.angular.z = float(angular_vel)
            
            cmd_pub.publish(cmd)

    except KeyboardInterrupt:
        node.get_logger().info("\n收到键盘中断信号！")
    
    finally:
        # =================================================================
        # [5] 程序退出清理
        # =================================================================
        node.get_logger().info("正在停车...")
        stop_cmd = TwistStamped()
        stop_cmd.header.stamp = node.get_clock().now().to_msg()
        stop_cmd.header.frame_id = 'base_link'
        stop_cmd.twist.linear.x = 0.0
        stop_cmd.twist.angular.z = 0.0
        
        cmd_pub.publish(stop_cmd)
        time.sleep(0.1) 
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()