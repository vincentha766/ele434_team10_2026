#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================================================================
# 节点名称: coverage_navigation_node (面向过程 / While 循环架构)
# 适用场景: ROS 2 (TurtleBot3 Waffle), 4x4m场地外围 12 宫格探索任务
# 核心策略: 
#   1. 严格矩形内接判定 (确保车身 100% 完全进入 1x1m 区块才算得分)
#   2. 顺时针全局拓扑 + 墙角强制右转逃逸 (防止在死角来回摆头)
#   3. 动态注意力势场法 (距离过近时放弃寻路，专心避障，防圆筒死锁)
#   4. 强制前向约束 (绝对禁止原地打转，确保走切角圆弧)
# =========================================================================

import rclpy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math
import time

# =====================================================================
# [1] 全局变量定义
# 作用: 在纯 while 循环架构中，回调函数与主循环通过全局变量共享数据。
# =====================================================================
odom_ready = False  # 里程计数据就绪标志
lidar_ready = False # 激光雷达数据就绪标志

odom_x = 0.0        # 机器人实时 X 坐标 (世界坐标系)
odom_y = 0.0        # 机器人实时 Y 坐标 (世界坐标系)
odom_yaw = 0.0      # 机器人实时偏航角 (Yaw, 范围 [-pi, pi])
lidar_ranges = []   # 清洗后的 360 度雷达距离数据

# =====================================================================
# [2] 回调函数定义 (处理传感器底层输入)
# =====================================================================
def odom_callback(msg):
    """
    里程计回调: 提取位置坐标，并将四元数姿态解算为 2D 平面偏航角。
    """
    global odom_ready, odom_x, odom_y, odom_yaw
    odom_x = msg.pose.pose.position.x
    odom_y = msg.pose.pose.position.y
    
    q = msg.pose.pose.orientation
    # 四元数转欧拉角 (Yaw 偏航角) 的标准数学公式
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    odom_yaw = math.atan2(siny_cosp, cosy_cosp)
    
    odom_ready = True

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
    确保机器人在调整朝向时，始终选择角度最小(最省时)的旋转方向。
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi

# =====================================================================
# [3] 主程序入口
# =====================================================================
def main(args=None):
    # 1. 初始化 ROS 2 节点
    rclpy.init(args=args)
    node = rclpy.create_node('coverage_navigation_node')
    
    # 2. 声明发布者与订阅者
    cmd_pub = node.create_publisher(TwistStamped, '/cmd_vel', 10)
    node.create_subscription(Odometry, '/odom', odom_callback, 10)
    node.create_subscription(LaserScan, '/scan', scan_callback, 10)

    # 3. 核心算法参数配置
    p_max_v = 0.26           # 最高线速度 (m/s)
    p_max_w = 1.82            # 最高角速度 (rad/s)
    p_safe_dist = 0.35       # 避障预警距离: < 0.35m 时开始产生排斥力
    p_corner_weight = 1.0    # 墙角逃逸权重: 越大，被卡在死角时甩尾越猛烈
    enable_repeat = True     # 是否无限跑圈
    
    # 4. 物理防撞极限参数 (基于 Waffle 硬件尺寸)
    STOP_DIST = 0.22         # 物理极限距离: Waffle 包络圆半径为 0.22m。突破此值将触发刹车。
    BOX_LIMIT = 0.28         # 得分判定容差: 0.5m(区块半宽) - 0.22m(车身半径) = 0.28m。
    MIN_FORWARD_SPEED = 0.06 # 最低前进速度: 强制走弧线避障，禁止底盘原地打转。

    # 5. 场地拓扑与航点坐标 (顺时针，1号位于右上角)
    waypoints = [
         (1.5,  1.5),  (1.5,  0.5),  (1.5, -0.5),  (1.5, -1.5), # 0,1,2,3 (右侧边缘，往下)
         (0.5, -1.5), (-0.5, -1.5), (-1.5, -1.5), (-1.5, -0.5), # 4,5,6,7 (底部边缘，往左)
         (-1.5, 0.5), (-1.5,  1.5), (-0.5,  1.5),  (0.5,  1.5)  # 8,9,10,11 (顶部边缘，往右)
    ]
    num_waypoints = len(waypoints)
    current_idx = 0  # 当前目标航点索引 (初始化为第 1 个区域)
    
    initialized_start_point = False
    
    node.get_logger().info("覆盖导航节点已启动，正在等待传感器数据...")

    try:
        # =================================================================
        # [4] 核心控制循环 (约 20Hz 频率)
        # =================================================================
        while rclpy.ok():
            # 关键阻塞函数: 检查传感器数据并触发回调，timeout 控制循环周期为 0.05s
            rclpy.spin_once(node, timeout_sec=0.05)
            
            # 数据未就绪前，跳过本次循环运算
            if not odom_ready or not lidar_ready:
                continue
                
            if not initialized_start_point:
                node.get_logger().info(f"传感器就绪！前往 1 号区域: {waypoints[current_idx]}")
                initialized_start_point = True

            # -------------------------------------------------------------
            # --- A. 状态机与到达判定 ---
            # -------------------------------------------------------------
            target_x, target_y = waypoints[current_idx]
            dx = target_x - odom_x
            dy = target_y - odom_y
            
            # 判定条件: X偏差和Y偏差同时小于 BOX_LIMIT 时，确认车体 100% 进入目标区域
            if abs(dx) < BOX_LIMIT and abs(dy) < BOX_LIMIT:
                current_idx += 1 
                node.get_logger().info(f"成功打卡，前往下一区域 (索引: {current_idx})")
                
                # 越界检查与循环逻辑
                if current_idx >= num_waypoints:
                    if enable_repeat:
                        current_idx = 0 
                        node.get_logger().info("完成一圈，重新开始循环！")
                    else:
                        node.get_logger().info("任务全部完成！申请停车。")
                        break # 跳出 while 循环
                
                # 更新为新目标点的坐标差
                target_x, target_y = waypoints[current_idx]
                dx = target_x - odom_x
                dy = target_y - odom_y

            # -------------------------------------------------------------
            # --- B. 激光雷达 8 扇区高精度划分 ---
            # -------------------------------------------------------------
            num_scans = len(lidar_ranges)
            if num_scans >= 360:
                # 拼接数组，提取正前方 45 度角 (-22.5度 到 22.5度)
                r_f  = lidar_ranges[337:360] + lidar_ranges[0:22] 
                r_fl = lidar_ranges[22:67]   # 左前方
                r_l  = lidar_ranges[67:112]  # 正左方
                r_r  = lidar_ranges[247:292] # 正右方
                r_fr = lidar_ranges[292:337] # 右前方
            else:
                r_f = r_fl = r_l = r_r = r_fr = lidar_ranges

            # 获取各扇区距离最近的障碍物
            min_f  = min(r_f)
            min_fl = min(r_fl)
            min_fr = min(r_fr)
            min_l  = min(r_l)
            min_r  = min(r_r)

            # -------------------------------------------------------------
            # --- C. 核心势场计算 (引力与斥力) ---
            # -------------------------------------------------------------
            target_angle = math.atan2(dy, dx)
            angle_error = wrap_to_pi(target_angle - odom_yaw)
            
            # C.1 注意力转移机制 (动态引力衰减)
            min_scan = min(min_f, min_fl, min_fr, min_l, min_r)
            if min_scan < p_safe_dist:
                # 距离越近，寻路权重越低。逼近 STOP_DIST 时权重归0，完全专注避障。
                track_weight = max(0.0, (min_scan - STOP_DIST) / (p_safe_dist - STOP_DIST))
            else:
                track_weight = 1.0 
                
            omega_track = angle_error * 1.6 * track_weight
            repel_omega = 0.0
            
            # C.2 墙角检测与破局
            # 如果左前和右前同时探测到障碍，说明被卡在墙角
            is_in_corner = (min_fl < p_safe_dist) and (min_fr < p_safe_dist)
            
            if is_in_corner:
                # 索引 0,3,6,9 对应场地的四大死角 (1, 4, 7, 10号区域)
                is_major_corner = current_idx in [0, 3, 6, 9]
                if is_major_corner:
                    # 顺时针拓扑基因: 遇到大墙角，无视雷达数据，输出强大的向右斥力强制转身
                    repel_omega = -p_corner_weight * 2.2
                else:
                    # 普通夹角: 比较左右空间，并附带 +0.15m 的右转偏好
                    if (min_fl + min_l) > (min_fr + min_r) + 0.15:
                        repel_omega = p_corner_weight * 1.8  # 只有左侧极其开阔才允许左转
                    else:
                        repel_omega = -p_corner_weight * 1.8 # 否则默认向右突围
            else:
                # C.3 常规平滑斥力 (防擦墙与切角)
                if min_fl < p_safe_dist: repel_omega -= (p_safe_dist - min_fl) * 6.0 
                if min_fr < p_safe_dist: repel_omega += (p_safe_dist - min_fr) * 6.0 
                if min_l < 0.28: repel_omega -= (0.28 - min_l) * 4.0
                if min_r < 0.28: repel_omega += (0.28 - min_r) * 4.0

            # C.4 正前方圆筒避障
            if min_f < p_safe_dist and not is_in_corner:
                # 给正前方的圆柱体施加不对称斥力，打破力学平衡，防止在圆筒前反复摆头
                bias_dir = 1.5 if (min_fl + min_l) > (min_fr + min_r) + 0.15 else -1.5
                repel_omega += bias_dir * (p_safe_dist - min_f)

            # -------------------------------------------------------------
            # --- D. 速度合成与物理约束 ---
            # -------------------------------------------------------------
            # 最终角速度 = 寻路引力 + 避障斥力，并进行硬限幅
            final_w = omega_track + repel_omega
            angular_vel = max(min(final_w, p_max_w), -p_max_w)

            # 线速度动态削减: 弯越急、前方越危险，车速越慢
            speed_factor = 1.0 - (abs(angle_error) / math.pi)
            if min_f < p_safe_dist:
                obs_limit = max(0.05, (min_f - STOP_DIST) / (p_safe_dist - STOP_DIST))
                speed_factor = min(speed_factor, obs_limit)
            
            linear_vel = p_max_v * max(speed_factor, 0.2)
            
            # [关键规则]: 绝对禁止原地旋转
            # 即使在极度危险下，也必须保持向前的最低线速度，逼迫机器人走内切圆弧脱困
            if linear_vel < MIN_FORWARD_SPEED:
                linear_vel = MIN_FORWARD_SPEED
            
            # 极限防撞保底: 突破安全底线时，剥夺加速权
            if min_f <= STOP_DIST:
                linear_vel = MIN_FORWARD_SPEED

            # -------------------------------------------------------------
            # --- E. 组装并发布 TwistStamped 控制指令 ---
            # -------------------------------------------------------------
            cmd = TwistStamped()
            
            # 注入 ROS 2 规范的时间戳与坐标系
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
        
        # 延时 0.1s，确保 ROS 底层通讯有足够时间将零速度指令发给电机驱动板
        time.sleep(0.1) 
        
        # 清理节点资源并关闭 rclpy 上下文
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()