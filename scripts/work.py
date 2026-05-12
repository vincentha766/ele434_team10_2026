#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================================================================
# 节点名称: coverage_navigation_node (SLAM 升级版)
# 适用场景: ROS 2 (TurtleBot3 Waffle), 结合 slam_toolbox 或 cartographer
# 核心变更: 
#   1. 弃用 /odom，改用 tf2 监听 map -> base_link 的坐标变换，彻底消除累积漂移。
#   2. 依然保留 LaserScan 的高频人工势场避障，保证对动态障碍物的极速响应。
# =========================================================================

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import math
import time

# 引入 TF2 相关库，用于获取 SLAM 的全局坐标
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException

# =====================================================================
# [1] 全局变量定义
# =====================================================================
pose_ready = False  # SLAM 全局定位就绪标志
lidar_ready = False # 激光雷达数据就绪标志

map_x = 0.0         # 机器人实时 X 坐标 (SLAM map 坐标系)
map_y = 0.0         # 机器人实时 Y 坐标 (SLAM map 坐标系)
map_yaw = 0.0       # 机器人实时偏航角 (SLAM map 坐标系)
lidar_ranges = []   # 清洗后的 360 度雷达距离数据

# =====================================================================
# [2] 回调函数定义 (处理传感器底层输入)
# =====================================================================
def scan_callback(msg):
    """
    激光雷达回调: 数据清洗与滤波。
    真实雷达会有盲区噪点或无穷大值，这里进行统一的安全替换。
    """
    global lidar_ready, lidar_ranges
    cleaned_ranges = []
    for r in msg.ranges:
        # 过滤: 无效值(nan), 无穷大(inf), 或贴脸噪点(<5cm)
        if math.isnan(r) or math.isinf(r) or r < 0.05:
            # 视为极其空旷，赋予 3.5m 的最大安全视距
            cleaned_ranges.append(3.5)
        else:
            cleaned_ranges.append(r)
    
    lidar_ranges = cleaned_ranges
    lidar_ready = True

def wrap_to_pi(angle):
    """
    角度标准化函数: 将任意夹角收敛至 [-pi, pi] 之间。
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi

# =====================================================================
# [3] 主程序入口
# =====================================================================
def main(args=None):
    # 1. 初始化 ROS 2 节点
    rclpy.init(args=args)
    node = rclpy.create_node('coverage_navigation_node')
    
    # 2. 声明发布者与订阅者 (注: 移除了 /odom 订阅)
    cmd_pub = node.create_publisher(TwistStamped, '/cmd_vel', 10)
    avoidance_pub = node.create_publisher(String, '/avoidance_log', 10)
    node.create_subscription(LaserScan, '/scan', scan_callback, 10)

    # 初始化 TF2 Buffer 和 Listener，用于接收 SLAM 的定位数据
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    # 3. 核心算法参数配置
    p_max_v = 0.26           # 最高线速度 (m/s)
    p_max_w = 1.82           # 最高角速度 (rad/s)
    p_safe_dist = 0.5       # 避障预警距离: < 0.35m 时开始产生排斥力
    p_corner_weight = 1.0    # 墙角逃逸权重: 越大，被卡在死角时甩尾越猛烈
    enable_repeat = False     # 是否无限跑圈
    
    # 4. 物理防撞极限参数
    STOP_DIST = 0.22         # 物理极限距离: Waffle 包络圆半径为 0.22m
    BOX_LIMIT = 0.28         # 得分判定容差
    MIN_FORWARD_SPEED = 0.1 # 最低前进速度

    # 5. 场地拓扑与航点坐标 (这里现在是基于 SLAM 建图起点的绝对坐标)
    waypoints = [
         (1.5,  1.5),  (1.5,  0.5),  (1.5, -0.5),  (1.5, -1.5), 
         (0.5, -1.5), (-0.5, -1.5), (-1.5, -1.5), (-1.5, -0.5), 
         (-1.5, 0.5), (-1.5,  1.5), (-0.5,  1.5),  (0.5,  1.5)  
    ]
    num_waypoints = len(waypoints)
    current_idx = 0  
    
    initialized_start_point = False
    
    node.get_logger().info("SLAM覆盖导航节点已启动，正在等待 tf 树与雷达数据...")

    try:
        # =================================================================
        # [4] 核心控制循环 (约 20Hz 频率)
        # =================================================================
        while rclpy.ok():
            global pose_ready, map_x, map_y, map_yaw

            # 关键阻塞函数
            rclpy.spin_once(node, timeout_sec=0.05)
            
            # 实时获取 SLAM 的 map -> base_link 坐标变换
            try:
                t = tf_buffer.lookup_transform(
                    'map',
                    'base_link',
                    rclpy.time.Time()
                )
                map_x = t.transform.translation.x
                map_y = t.transform.translation.y
                
                # 四元数转欧拉角 (Yaw)
                q = t.transform.rotation
                siny_cosp = 2 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
                map_yaw = math.atan2(siny_cosp, cosy_cosp)
                
                pose_ready = True
            except TransformException as ex:
                # SLAM TF 树尚未建立或丢失时，跳过本帧控制
                pose_ready = False
                continue

            # 数据未就绪前，跳过本次循环运算
            if not pose_ready or not lidar_ready:
                continue
                
            if not initialized_start_point:
                msg_str = f"SLAM定位就绪！前往 1 号区域: {waypoints[current_idx]}"
                node.get_logger().info(msg_str)
                avoidance_pub.publish(String(data=msg_str))
                initialized_start_point = True

            # -------------------------------------------------------------
            # --- A. 状态机与到达判定 (使用 SLAM 坐标 map_x, map_y) ---
            # -------------------------------------------------------------
            target_x, target_y = waypoints[current_idx]
            dx = target_x - map_x
            dy = target_y - map_y
            
            # 判定条件: X偏差和Y偏差同时小于 BOX_LIMIT 时，确认车体 100% 进入目标区域
            if abs(dx) < BOX_LIMIT and abs(dy) < BOX_LIMIT:
                current_idx += 1 
                msg_str = f"成功打卡，前往下一区域 (索引: {current_idx})"
                node.get_logger().info(msg_str)
                avoidance_pub.publish(String(data=msg_str))
                
                # 越界检查与循环逻辑
                if current_idx >= num_waypoints:
                    if enable_repeat:
                        current_idx = 0 
                        msg_str = "完成一圈，重新开始循环！"
                        node.get_logger().info(msg_str)
                        avoidance_pub.publish(String(data=msg_str))
                    else:
                        msg_str = "任务全部完成！申请停车。"
                        node.get_logger().info(msg_str)
                        avoidance_pub.publish(String(data=msg_str))
                        break 
                
                target_x, target_y = waypoints[current_idx]
                dx = target_x - map_x
                dy = target_y - map_y

            # -------------------------------------------------------------
            # --- B. 激光雷达 8 扇区高精度划分 ---
            # -------------------------------------------------------------
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

            # -------------------------------------------------------------
            # --- C. 核心势场计算 (使用 SLAM 偏航角 map_yaw) ---
            # -------------------------------------------------------------
            target_angle = math.atan2(dy, dx)
            angle_error = wrap_to_pi(target_angle - map_yaw)
            
            # C.1 注意力转移机制
            min_scan = min(min_f, min_fl, min_fr, min_l, min_r)
            if min_scan < p_safe_dist:
                track_weight = max(0.0, (min_scan - STOP_DIST) / (p_safe_dist - STOP_DIST))
            else:
                track_weight = 1.0 
                
            omega_track = angle_error * 1.6 * track_weight
            repel_omega = 0.0
            
            # C.2 墙角检测与破局
            is_in_corner = (min_fl < p_safe_dist) and (min_fr < p_safe_dist)
            
            if is_in_corner:
                is_major_corner = current_idx in [0, 3, 6, 9]
                avoidance_pub.publish(String(data=f"检测到墙角，执行避障行为 (大角: {is_major_corner})"))
                if is_major_corner:
                    repel_omega = -p_corner_weight * 2.2
                else:
                    if (min_fl + min_l) > (min_fr + min_r) + 0.15:
                        repel_omega = p_corner_weight * 1.8  
                    else:
                        repel_omega = -p_corner_weight * 1.8 
            else:
                # C.3 常规平滑斥力
                if min_fl < p_safe_dist: repel_omega -= (p_safe_dist - min_fl) * 6.0 
                if min_fr < p_safe_dist: repel_omega += (p_safe_dist - min_fr) * 6.0 
                if min_l < 0.28: repel_omega -= (0.28 - min_l) * 4.0
                if min_r < 0.28: repel_omega += (0.28 - min_r) * 4.0

            # C.4 正前方圆筒避障
            if min_f < p_safe_dist and not is_in_corner:
                bias_dir = 1.5 if (min_fl + min_l) > (min_fr + min_r) + 0.15 else -1.5
                repel_omega += bias_dir * (p_safe_dist - min_f)

            # -------------------------------------------------------------
            # --- D. 速度合成与物理约束 ---
            # -------------------------------------------------------------
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

            # -------------------------------------------------------------
            # --- E. 组装并发布 TwistStamped 控制指令 ---
            # -------------------------------------------------------------
            cmd = TwistStamped()
            cmd.header.stamp = node.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            cmd.twist.linear.x = float(linear_vel)
            cmd.twist.angular.z = float(angular_vel)
            
            cmd_pub.publish(cmd)

    except KeyboardInterrupt:
        node.get_logger().info("\n收到键盘中断信号！准备安全退出。")
    
    finally:
        # =================================================================
        # [5] 程序退出清理 (断电刹车机制)
        # =================================================================
        node.get_logger().info("正在发送停车指令...")
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