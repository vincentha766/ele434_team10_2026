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
#
# -------------------------------------------------------------------------
# 人工势场避障 (Artificial Potential Field, APF) — 给团队看的原理摘要
# -------------------------------------------------------------------------
# 经典 APF 把运动规划写成“虚拟力”：F = -∇U，势能 U = U_att(目标) + Σ U_rep(障碍)。
#   · 引力：离目标越远，U_att 越大，机器人被“拉”向目标（本代码用航向误差 angle_error
#     产生角速度项 ω_track，相当于在航向维度上把车头拧向目标）。
#   · 斥力：离障碍越近，U_rep 越大，机器人被“推”离障碍（本代码用各扇区距离 min_* 生成
#     repel_omega，相当于在航向维度上推开墙壁/柱子；不是严格的 -∇U_rep，但工程上等价
#     于“哪侧更近就往反方向转一点”）。
# 已知问题：目标与障碍之间可能出现势场“局部极小”，车体振荡或卡住。本节点的补丁：
#   · 注意力衰减：太近时减小 track_weight，暂时弱化“去目标”，优先避障。
#   · 墙角特判：左右前同时近时，用规则角速度突围，而不是单纯线性叠加斥力。
#   · 圆筒前不对称偏置：打破左右对称导致的来回摆头。
#   · 最低前进速度：禁止 v≈0 原地转，逼车走弧线脱困。
# 日志里 [势场分支] 仅在**刚进入**某条件时打印一次（边沿触发），便于对照代码分支。
# =========================================================================

import rclpy
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
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
    log_pub = node.create_publisher(String, '/avoidance_log', 10)
    node.create_subscription(Odometry, '/odom', odom_callback, 10)
    node.create_subscription(LaserScan, '/scan', scan_callback, 10)

    # 3. 关键参数安全初始化（支持外部参数覆盖 + 本地默认兜底 + 范围校验）
    # 目的：防止误配置导致除零、超速、或“看起来像卡死”的异常行为。
    node.declare_parameter('p_max_v', 0.26)
    node.declare_parameter('p_max_w', 1.82)
    node.declare_parameter('p_safe_dist', 0.35)
    node.declare_parameter('p_corner_weight', 1.0)
    node.declare_parameter('p_track_gain', 1.6)
    node.declare_parameter('enable_repeat', True)
    node.declare_parameter('stop_dist', 0.22)
    node.declare_parameter('box_limit', 0.28)
    node.declare_parameter('min_forward_speed', 0.06)
    node.declare_parameter('reset_odom_on_start', True)

    def clamp(value, lo, hi):
        return max(lo, min(hi, value))

    p_max_v = float(node.get_parameter('p_max_v').value)
    p_max_w = float(node.get_parameter('p_max_w').value)
    p_safe_dist = float(node.get_parameter('p_safe_dist').value)
    p_corner_weight = float(node.get_parameter('p_corner_weight').value)
    p_track_gain = float(node.get_parameter('p_track_gain').value)
    enable_repeat = bool(node.get_parameter('enable_repeat').value)
    STOP_DIST = float(node.get_parameter('stop_dist').value)
    BOX_LIMIT = float(node.get_parameter('box_limit').value)
    MIN_FORWARD_SPEED = float(node.get_parameter('min_forward_speed').value)
    reset_odom_on_start = bool(node.get_parameter('reset_odom_on_start').value)

    # 基础物理范围（面向 TurtleBot3 Waffle 的保守约束）
    p_max_v = clamp(p_max_v, 0.05, 0.30)            # 最高线速度 (m/s)
    p_max_w = clamp(p_max_w, 0.30, 2.84)            # 最高角速度 (rad/s)
    p_safe_dist = clamp(p_safe_dist, 0.20, 1.00)    # 避障预警距离
    p_corner_weight = clamp(p_corner_weight, 0.10, 3.00)
    p_track_gain = clamp(p_track_gain, 0.50, 3.00)
    STOP_DIST = clamp(STOP_DIST, 0.16, 0.45)        # 物理极限距离
    BOX_LIMIT = clamp(BOX_LIMIT, 0.10, 0.45)        # 到达判定容差
    MIN_FORWARD_SPEED = clamp(MIN_FORWARD_SPEED, 0.03, p_max_v)

    # 强制保证 p_safe_dist > STOP_DIST，避免分母为0或负值
    if p_safe_dist <= STOP_DIST + 0.02:
        p_safe_dist = STOP_DIST + 0.02
        node.get_logger().warn(
            "参数保护触发: p_safe_dist 过小，已自动调整为 STOP_DIST + 0.02"
        )

    node.get_logger().info(
        "参数初始化完成: "
        f"p_max_v={p_max_v:.3f}, p_max_w={p_max_w:.3f}, "
        f"p_safe_dist={p_safe_dist:.3f}, stop_dist={STOP_DIST:.3f}, "
        f"min_forward_speed={MIN_FORWARD_SPEED:.3f}, p_track_gain={p_track_gain:.3f}, "
        f"box_limit={BOX_LIMIT:.3f}, p_corner_weight={p_corner_weight:.3f}, "
        f"enable_repeat={enable_repeat}, reset_odom_on_start={reset_odom_on_start}"
    )

    # 5. 场地拓扑与航点坐标 (顺时针，1号位于右上角)
    waypoints = [
         (1.5,  1.5),  (1.5,  0.5),  (1.5, -0.5),  (1.5, -1.5), # 0,1,2,3 (右侧边缘，往下)
         (0.5, -1.5), (-0.5, -1.5), (-1.5, -1.5), (-1.5, -0.5), # 4,5,6,7 (底部边缘，往左)
         (-1.5, 0.5), (-1.5,  1.5), (-0.5,  1.5),  (0.5,  1.5)  # 8,9,10,11 (顶部边缘，往右)
    ]
    num_waypoints = len(waypoints)
    current_idx = 0  # 当前目标航点索引 (初始化为第 1 个区域)
    
    initialized_start_point = False
    origin_initialized = False
    origin_x = 0.0
    origin_y = 0.0
    origin_yaw = 0.0
    # 避障 / 状态调试：按时间节流打印，避免 20Hz 刷屏
    debug_log_interval = 0.35  # 秒
    last_debug_log_mono = 0.0
    param_echoed_once = False
    # [势场分支] 边沿检测：只在“刚进入/刚离开”某分支时打印，避免每周期重复
    edge_prev = {
        'attention': False,   # min_scan < p_safe_dist，引力衰减区
        'in_corner': False,   # 左右前同时近，墙角特判
        'cylinder': False,  # 正前近且非墙角，圆筒不对称绕障
        'brake_front': False, # min_f <= STOP_DIST，前向极限保底
        'repel_fl': False,  # C.3 左前扇区斥力
        'repel_fr': False,  # C.3 右前扇区斥力
        'repel_l': False,   # C.3 左侧斥力
        'repel_r': False,   # C.3 右侧斥力
    }
    corner_mode_prev = None  # None | 'major' | 'left' | 'right'，墙角子策略

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
                if reset_odom_on_start and not origin_initialized:
                    origin_x = odom_x
                    origin_y = odom_y
                    origin_yaw = odom_yaw
                    origin_initialized = True
                    node.get_logger().info(
                        f"已初始化局部坐标原点: origin=({origin_x:.3f},{origin_y:.3f}), "
                        f"yaw0={math.degrees(origin_yaw):.1f}deg"
                    )
                elif not reset_odom_on_start:
                    node.get_logger().info("未启用 reset_odom_on_start，使用全局 /odom 坐标")
                node.get_logger().info(f"传感器就绪！前往 1 号区域: {waypoints[current_idx]}")
                initialized_start_point = True
            
            # 仅在进入主循环后打印一次关键参数，便于日志复盘时定位运行配置
            if not param_echoed_once:
                node.get_logger().info(
                    "[运行参数] "
                    f"safe={p_safe_dist:.3f}, stop={STOP_DIST:.3f}, box={BOX_LIMIT:.3f}, "
                    f"v_max={p_max_v:.3f}, w_max={p_max_w:.3f}, v_min={MIN_FORWARD_SPEED:.3f}, "
                    f"track_gain={p_track_gain:.3f}, corner_weight={p_corner_weight:.3f}, "
                    f"repeat={enable_repeat}, reset_odom={reset_odom_on_start}"
                )
                param_echoed_once = True

            if reset_odom_on_start:
                odom_x_nav = odom_x - origin_x
                odom_y_nav = odom_y - origin_y
                odom_yaw_nav = wrap_to_pi(odom_yaw - origin_yaw)
            else:
                odom_x_nav = odom_x
                odom_y_nav = odom_y
                odom_yaw_nav = odom_yaw

            # -------------------------------------------------------------
            # --- A. 状态机与到达判定 ---
            # -------------------------------------------------------------
            target_x, target_y = waypoints[current_idx]
            dx = target_x - odom_x_nav
            dy = target_y - odom_y_nav
            
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
                dx = target_x - odom_x_nav
                dy = target_y - odom_y_nav

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
            # 连续势场在这里离散实现：ω = ω_track(引力/朝向目标) + repel_omega(斥力/躲障碍)。
            # 详见文件头部「人工势场避障」说明；下列 C.1~C.4 对应不同“势能补丁”分支。
            # -------------------------------------------------------------
            target_angle = math.atan2(dy, dx)
            angle_error = wrap_to_pi(target_angle - odom_yaw_nav)
            
            # 用于记录当前周期的所有激活逻辑，输出到 log_pub
            active_log_branches = []

            # C.1 注意力转移机制 (动态引力衰减)
            # 等价于：离障碍越近，U_att 对总势能的影响越小，避免“既要赶去目标又要猛躲墙”打架。
            min_scan = min(min_f, min_fl, min_fr, min_l, min_r)
            attention_zone = min_scan < p_safe_dist
            if attention_zone:
                # 距离越近，寻路权重越低。逼近 STOP_DIST 时权重归0，完全专注避障。
                safe_gap = max(p_safe_dist - STOP_DIST, 1e-6)
                track_weight = max(0.0, (min_scan - STOP_DIST) / safe_gap)
                active_log_branches.append(f"C.1 注意力衰减(w={track_weight:.2f})")
            else:
                track_weight = 1.0

            if attention_zone and not edge_prev['attention']:
                node.get_logger().info(
                    f"[势场分支] 进入 C.1 注意力衰减: min_scan={min_scan:.3f} < p_safe_dist={p_safe_dist}，"
                    f"track_weight 随距离缩小，弱化朝向目标的引力项"
                )
            elif (not attention_zone) and edge_prev['attention']:
                node.get_logger().info(
                    f"[势场分支] 离开 C.1 注意力衰减: min_scan={min_scan:.3f} >= p_safe_dist={p_safe_dist}"
                )
            edge_prev['attention'] = attention_zone
                
            omega_track = angle_error * p_track_gain * track_weight
            repel_omega = 0.0
            
            # C.2 墙角检测与破局
            # 若左前、右前同时很近，经典 APF 易在凹角形成局部极小；这里改用语义规则直接给角速度。
            is_in_corner = (min_fl < p_safe_dist) and (min_fr < p_safe_dist)
            
            if is_in_corner:
                if not edge_prev['in_corner']:
                    node.get_logger().info(
                        f"[势场分支] 进入 C.2 墙角判定: min_fl={min_fl:.3f}, min_fr={min_fr:.3f} 均 < "
                        f"p_safe_dist={p_safe_dist}（左右前同时受压，启用墙角突围逻辑）"
                    )
                # 墙角阶段不叠加 C.3；清零 C.3 边沿记忆，出角后若仍贴墙可再次打印「进入 C.3」
                edge_prev['repel_fl'] = edge_prev['repel_fr'] = False
                edge_prev['repel_l'] = edge_prev['repel_r'] = False
                # 索引 0,3,6,9 对应场地的四大死角 (1, 4, 7, 10号区域)
                is_major_corner = current_idx in [0, 3, 6, 9]
                if is_major_corner:
                    # 顺时针拓扑基因: 遇到大墙角，无视雷达数据，输出强大的向右斥力强制转身
                    repel_omega = -p_corner_weight * 2.2
                    corner_mode = 'major'
                else:
                    # 普通夹角: 比较左右空间，并附带 +0.15m 的右转偏好
                    if (min_fl + min_l) > (min_fr + min_r) + 0.15:
                        repel_omega = p_corner_weight * 1.8  # 只有左侧极其开阔才允许左转
                        corner_mode = 'left'
                    else:
                        repel_omega = -p_corner_weight * 1.8 # 否则默认向右突围
                        corner_mode = 'right'
                
                active_log_branches.append(f"C.2 墙角突围({corner_mode})")

                if corner_mode != corner_mode_prev:
                    if corner_mode == 'major':
                        node.get_logger().info(
                            "[势场分支] C.2 子策略: 赛场大死角 (idx∈{0,3,6,9})，固定强右转 repel_omega "
                            f"= {-p_corner_weight * 2.2:.3f}"
                        )
                    elif corner_mode == 'left':
                        node.get_logger().info(
                            f"[势场分支] C.2 子策略: 普通墙角，左侧更空 "
                            f"(FL+L={min_fl + min_l:.3f} > FR+R+0.15={min_fr + min_r + 0.15:.3f})，左转突围"
                        )
                    else:
                        node.get_logger().info(
                            f"[势场分支] C.2 子策略: 普通墙角，默认右转突围 "
                            f"(FL+L={min_fl + min_l:.3f}, FR+R={min_fr + min_r:.3f})"
                        )
                    corner_mode_prev = corner_mode
                edge_prev['in_corner'] = True
            else:
                if edge_prev['in_corner']:
                    node.get_logger().info("[势场分支] 离开 C.2 墙角判定，恢复常规扇区斥力 (C.3)")
                edge_prev['in_corner'] = False
                corner_mode_prev = None
                # C.3 常规平滑斥力 (防擦墙与切角)
                # 各向距离越近，对 repel_omega 的增量越大，相当于在航向维推开该侧障碍。
                act_fl = min_fl < p_safe_dist
                if act_fl:
                    repel_omega -= (p_safe_dist - min_fl) * 6.0
                    active_log_branches.append("C.3 左前斥力")
                    if not edge_prev['repel_fl']:
                        node.get_logger().info(
                            f"[势场分支] 进入 C.3 左前斥力: min_fl={min_fl:.3f} < p_safe_dist={p_safe_dist}"
                        )
                elif edge_prev['repel_fl']:
                    node.get_logger().info("[势场分支] 离开 C.3 左前斥力区 (min_fl 已 ≥ p_safe_dist)")
                edge_prev['repel_fl'] = act_fl

                act_fr = min_fr < p_safe_dist
                if act_fr:
                    repel_omega += (p_safe_dist - min_fr) * 6.0
                    active_log_branches.append("C.3 右前斥力")
                    if not edge_prev['repel_fr']:
                        node.get_logger().info(
                            f"[势场分支] 进入 C.3 右前斥力: min_fr={min_fr:.3f} < p_safe_dist={p_safe_dist}"
                        )
                elif edge_prev['repel_fr']:
                    node.get_logger().info("[势场分支] 离开 C.3 右前斥力区 (min_fr 已 ≥ p_safe_dist)")
                edge_prev['repel_fr'] = act_fr

                act_l = min_l < 0.28
                if act_l:
                    repel_omega -= (0.28 - min_l) * 4.0
                    active_log_branches.append("C.3 左侧斥力")
                    if not edge_prev['repel_l']:
                        node.get_logger().info(
                            f"[势场分支] 进入 C.3 左侧斥力: min_l={min_l:.3f} < 0.28"
                        )
                elif edge_prev['repel_l']:
                    node.get_logger().info("[势场分支] 离开 C.3 左侧斥力区 (min_l 已 ≥ 0.28)")
                edge_prev['repel_l'] = act_l

                act_r = min_r < 0.28
                if act_r:
                    repel_omega += (0.28 - min_r) * 4.0
                    active_log_branches.append("C.3 右侧斥力")
                    if not edge_prev['repel_r']:
                        node.get_logger().info(
                            f"[势场分支] 进入 C.3 右侧斥力: min_r={min_r:.3f} < 0.28"
                        )
                elif edge_prev['repel_r']:
                    node.get_logger().info("[势场分支] 离开 C.3 右侧斥力区 (min_r 已 ≥ 0.28)")
                edge_prev['repel_r'] = act_r

            # C.4 正前方圆筒避障
            # 正对柱子时左右对称斥力会抵消，加 bias_dir 打破对称，等价于人为偏置 U_rep。
            in_cylinder_logic = min_f < p_safe_dist and not is_in_corner
            if in_cylinder_logic:
                bias_dir = 1.5 if (min_fl + min_l) > (min_fr + min_r) + 0.15 else -1.5
                repel_omega += bias_dir * (p_safe_dist - min_f)
                active_log_branches.append(f"C.4 正前绕障(bias={bias_dir})")
                if not edge_prev['cylinder']:
                    node.get_logger().info(
                        f"[势场分支] 进入 C.4 正前绕障: min_f={min_f:.3f} < p_safe_dist 且非墙角，"
                        f"不对称偏置 bias_dir={bias_dir:+.1f}，Δω += {bias_dir * (p_safe_dist - min_f):.3f}"
                    )
                edge_prev['cylinder'] = True
            else:
                if edge_prev['cylinder']:
                    node.get_logger().info("[势场分支] 离开 C.4 正前绕障逻辑")
                edge_prev['cylinder'] = False

            # 发布逻辑状态
            if active_log_branches:
                log_msg = String()
                log_msg.data = " | ".join(active_log_branches)
                log_pub.publish(log_msg)

            # -------------------------------------------------------------
            # --- D. 速度合成与物理约束 ---
            # -------------------------------------------------------------
            # 最终角速度 = 寻路引力 + 避障斥力，并进行硬限幅
            final_w = omega_track + repel_omega
            angular_vel = max(min(final_w, p_max_w), -p_max_w)
            angular_saturated = abs(final_w - angular_vel) > 1e-6

            # 线速度动态削减: 弯越急、前方越危险，车速越慢
            heading_factor = 1.0 - (abs(angle_error) / math.pi)
            speed_factor = heading_factor
            obs_limit = 1.0
            if min_f < p_safe_dist:
                safe_gap = max(p_safe_dist - STOP_DIST, 1e-6)
                obs_limit = max(0.05, (min_f - STOP_DIST) / safe_gap)
                speed_factor = min(speed_factor, obs_limit)
            
            linear_vel = p_max_v * max(speed_factor, 0.2)
            
            # [关键规则]: 绝对禁止原地旋转
            # 即使在极度危险下，也必须保持向前的最低线速度，逼迫机器人走内切圆弧脱困
            if linear_vel < MIN_FORWARD_SPEED:
                linear_vel = MIN_FORWARD_SPEED
            
            # 极限防撞保底: 突破安全底线时，剥夺加速权
            if min_f <= STOP_DIST:
                if not edge_prev['brake_front']:
                    node.get_logger().info(
                        f"[势场分支] 前向极限 D: min_f={min_f:.3f} <= STOP_DIST={STOP_DIST}，"
                        f"线速度压至 MIN_FORWARD_SPEED={MIN_FORWARD_SPEED}"
                    )
                edge_prev['brake_front'] = True
                linear_vel = MIN_FORWARD_SPEED
            else:
                edge_prev['brake_front'] = False
            linear_floor_applied = linear_vel <= MIN_FORWARD_SPEED + 1e-9

            # -------------------------------------------------------------
            # --- E. 组装并发布 TwistStamped 控制指令 ---
            # -------------------------------------------------------------
            cmd = TwistStamped()
            
            # 注入 ROS 2 规范的时间戳与坐标系
            cmd.header.stamp = node.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            
            cmd.twist.linear.x = float(linear_vel)
            cmd.twist.angular.z = float(angular_vel)

            # --- 调试输出：当前位姿、雷达扇区、势场分项、最终速度 ---
            now_mono = time.monotonic()
            if now_mono - last_debug_log_mono >= debug_log_interval:
                last_debug_log_mono = now_mono
                dist_wp = math.hypot(dx, dy)
                yaw_err_deg = math.degrees(angle_error)
                attention = min_scan < p_safe_dist
                brake_zone = min_f <= STOP_DIST
                branch_text = " / ".join(active_log_branches) if active_log_branches else "None"
                node.get_logger().info(
                    f"[避障调试] idx={current_idx} "
                    f"origin=({origin_x:.2f},{origin_y:.2f},{math.degrees(origin_yaw):.1f}deg) "
                    f"pos_raw=({odom_x:.3f},{odom_y:.3f}) yaw_raw_deg={math.degrees(odom_yaw):.1f} | "
                    f"pos_nav=({odom_x_nav:.3f},{odom_y_nav:.3f}) yaw_nav_deg={math.degrees(odom_yaw_nav):.1f} | "
                    f"目标=({target_x:.2f},{target_y:.2f}) dist_wp={dist_wp:.3f} yaw_err={yaw_err_deg:.1f}° | "
                    f"扇区 F/FL/FR/L/R={min_f:.2f}/{min_fl:.2f}/{min_fr:.2f}/{min_l:.2f}/{min_r:.2f} "
                    f"min_scan={min_scan:.2f} | "
                    f"墙角={is_in_corner} 专注避障={attention} 刹车区={brake_zone} | "
                    f"track_w={track_weight:.2f} ω_跟踪={omega_track:.3f} ω_斥力={repel_omega:.3f} "
                    f"ω_合成={final_w:.3f} sat_w={angular_saturated} | "
                    f"head_factor={heading_factor:.2f} obs_limit={obs_limit:.2f} speed_factor={speed_factor:.2f} "
                    f"v_floor={linear_floor_applied} | "
                    f"cmd v={linear_vel:.3f} w={angular_vel:.3f} | "
                    f"branches={branch_text}"
                )
            
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
