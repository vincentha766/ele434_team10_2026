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
import tf2_ros
import heapq
import numpy as np
from rclpy.duration import Duration
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
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

# Cartographer 发布的占据栅格
map_ready = False
map_grid = None          # 2D int8 np.array, shape (h, w): -1=未知, 0..100=占据概率
map_res = 0.05
map_origin_x = 0.0
map_origin_y = 0.0
map_width = 0
map_height = 0

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

def map_callback(msg):
    """OccupancyGrid 回调: 缓存地图到 numpy 数组"""
    global map_ready, map_grid, map_res, map_origin_x, map_origin_y, map_width, map_height
    w = msg.info.width
    h = msg.info.height
    map_grid = np.array(msg.data, dtype=np.int8).reshape(h, w)
    map_res = msg.info.resolution
    map_origin_x = msg.info.origin.position.x
    map_origin_y = msg.info.origin.position.y
    map_width = w
    map_height = h
    map_ready = True

def world_to_grid(x, y):
    col = int((x - map_origin_x) / map_res)
    row = int((y - map_origin_y) / map_res)
    return col, row

def grid_to_world(col, row):
    x = map_origin_x + (col + 0.5) * map_res
    y = map_origin_y + (row + 0.5) * map_res
    return x, y

def inflate_obstacles(grid, radius):
    """对占据格 (值>=50) 做 Chebyshev 距离为 radius 的膨胀, 返回 bool 2D 数组"""
    occ = (grid >= 50)
    inflated = occ.copy()
    for _ in range(radius):
        new = inflated.copy()
        new[:-1] |= inflated[1:]
        new[1:]  |= inflated[:-1]
        new[:, :-1] |= inflated[:, 1:]
        new[:, 1:]  |= inflated[:, :-1]
        inflated = new
    return inflated

def astar(blocked, start, goal):
    """在 2D bool 栅格上做 8 连通 A*. blocked[row, col]=True 为障碍. 返回 [(col,row),...] 或 None"""
    h, w = blocked.shape
    if not (0 <= start[0] < w and 0 <= start[1] < h): return None
    if not (0 <= goal[0] < w and 0 <= goal[1] < h): return None
    if blocked[goal[1], goal[0]]: return None
    if start == goal: return [start]
    DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    def hd(p, q): return math.hypot(p[0]-q[0], p[1]-q[1])
    open_heap = [(hd(start, goal), 0.0, start)]
    came = {start: None}
    g = {start: 0.0}
    while open_heap:
        _, gc, cur = heapq.heappop(open_heap)
        if gc > g.get(cur, float('inf')): continue
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur); cur = came[cur]
            return list(reversed(path))
        for dc, dr in DIRS:
            nc, nr = cur[0]+dc, cur[1]+dr
            if nc < 0 or nc >= w or nr < 0 or nr >= h: continue
            if blocked[nr, nc]: continue
            step = 1.41421 if (dc and dr) else 1.0
            ng = gc + step
            n = (nc, nr)
            if ng < g.get(n, float('inf')):
                g[n] = ng; came[n] = cur
                heapq.heappush(open_heap, (ng + hd(n, goal), ng, n))
    return None

def _unblock_region(blocked, cx, cy, r):
    """把 (cx,cy) 周围 ±r 栅格解封, 处理机器人/目标恰好在膨胀区内的情况"""
    h, w = blocked.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                blocked[ny, nx] = False

def plan_path(start_xy, goal_xy, inflate_r=10):
    """A* 规划 world 坐标路径, 返回下采样后的 [(x,y),...] 或 None
    渐进回退: 先用 inflate_r 膨胀, 若无解就减半再试, 直到成功或 r=1."""
    if not map_ready or map_grid is None:
        return None
    start = world_to_grid(*start_xy)
    goal  = world_to_grid(*goal_xy)

    # 渐进放松膨胀半径
    radii = [inflate_r]
    r = inflate_r
    while r > 1:
        r = max(1, r // 2)
        radii.append(r)

    path_cells = None
    for r in radii:
        blocked = inflate_obstacles(map_grid, r)
        # 不管膨胀半径多少, 起点/终点周围 2 格强制解封 (机器人已经在那了)
        _unblock_region(blocked, start[0], start[1], 2)
        _unblock_region(blocked, goal[0], goal[1], 2)
        path_cells = astar(blocked, start, goal)
        if path_cells is not None:
            break

    if path_cells is None:
        return None
    path_world = [grid_to_world(c, r_) for c, r_ in path_cells]
    if len(path_world) > 8:
        step = max(1, len(path_world) // 6)
        ds = path_world[::step]
        if ds[-1] != path_world[-1]:
            ds.append(path_world[-1])
        path_world = ds
    return path_world

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
    # 访问 odom_callback 填写的全局位姿; 使用 TF 时也覆盖这些名字
    global odom_x, odom_y, odom_yaw

    # 1. 初始化 ROS 2 节点
    rclpy.init(args=args)
    node = rclpy.create_node('coverage_navigation_node')
    
    # 2. 声明发布者与订阅者
    cmd_pub = node.create_publisher(TwistStamped, '/cmd_vel', 10)
    node.create_subscription(Odometry, '/odom', odom_callback, 10)
    node.create_subscription(LaserScan, '/scan', scan_callback, 10)
    node.create_subscription(OccupancyGrid, '/map', map_callback, 10)

    # TF listener: 查询 Cartographer 校正后的 map -> base_footprint
    # 轮滑漂移被吸收进 map -> odom TF, 这里拿到的是全局真值, 不会因打滑飞走
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, node)

    # 3. 核心算法参数配置
    # Bang-bang 控制参数: ALIGN (原地对齐) / DRIVE (对齐后直行) / AVOID (前方危险原地转)
    # Waffle 的 LIDAR 在 base_link 原点, 差速轮轴也过原点 -> 激光中心 = 旋转中心
    # 外接圆半径 ~0.21m (外形 281×306 mm), 保留 0.07m 裕量 -> SAFE_RADIUS = 0.28
    p_max_v = 0.22           # 直行最高线速度
    p_max_w = 1.60            # 原地旋转最高角速度
    SAFE_RADIUS = 0.32       # 外接圆 0.21 + 11cm 安全余量 -> front_danger 更早触发
    ALIGN_TOL = 0.15         # 航向容差 (~9°): 小于此才进入 DRIVE
    K_HEAD = 3.0             # ALIGN 航向比例系数
    K_DRIVE = 1.2            # DRIVE 阶段航向微调系数
    enable_repeat = True

    # 得分判定容差
    BOX_LIMIT = 0.28

    # 5. 17 点路径: 12 得分格 + 5 内缩枢纽 (避开 8 面 L 型斜墙)
    waypoints = [
        (1.5,  1.5),   # 0  NE corner (45° 直取, 离 beacon 0.78m)
        (1.5,  0.5),   # 1  east-top (南下)
        (1.5, -0.5),   # 2  east-bot
        (0.5, -0.5),   # 3  inner SE
        (0.5, -1.5),   # 4  south-east
        (1.5, -1.5),   # 5  SE corner
        (0.5, -1.5),   # 6  回到 4
        (-0.5, -1.5),  # 7  south-west
        (-0.5, -0.5),  # 8  inner SW
        (-1.5, -0.5),  # 9  west-bot
        (-1.5, -1.5),  # 10 SW corner
        (-1.5,  0.5),  # 11 west-top
        (-0.5,  0.5),  # 12 inner NW
        (-0.5,  1.5),  # 13 north-west
        (-1.5,  1.5),  # 14 NW corner
        (-0.5,  1.5),  # 15 回到 13
        ( 0.5,  1.5),  # 16 north-east
    ]
    num_waypoints = len(waypoints)
    current_idx = 0
    lap_count = 0
    max_laps = 1
    # 12 得分格的机会式标记 (独立于 waypoint 索引)
    score_cells = [
        (1.5, 1.5), (1.5, 0.5), (1.5, -0.5), (1.5, -1.5),
        (0.5, -1.5), (-0.5, -1.5),
        (-1.5, -1.5), (-1.5, -0.5), (-1.5, 0.5), (-1.5, 1.5),
        (-0.5, 1.5), (0.5, 1.5),
    ]
    scored = [False] * 12

    # Bug-0 风格 AVOID 状态: 进入后承诺方向直到绕过障碍
    avoid_active = False
    avoid_dir = 0            # +1 = 左转(障碍在右), -1 = 右转(障碍在左)
    avoid_start_yaw = 0.0
    avoid_start_x = 0.0
    avoid_start_y = 0.0
    AVOID_MIN_ROT = 0.5      # rad (~30°), 最小提交转角
    AVOID_EXIT_SIDE = 0.45   # 障碍所在侧清空阈值
    SIDESTEP_V = 0.06        # 弧形绕障的前进速度 (慢 -> 弧半径大, 贴墙不擦)
    SIDESTEP_W = 1.20        # 弧形绕障的角速度

    # stuck 检测
    stuck_history = []
    STUCK_WINDOW = 20.0      # s (绕障可能较慢, 放宽到 20s)
    STUCK_DIST = 0.05
    
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

            # 优先使用 SLAM 校正的 map -> base_footprint 位姿
            # (Cartographer 把轮滑漂移吸收进 map -> odom TF, 所以这个位姿是全局真值)
            # SLAM 未上线或 TF 尚不可用时回落到原始 /odom
            try:
                t = tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time(),
                                               timeout=Duration(seconds=0.0))
                odom_x = t.transform.translation.x
                odom_y = t.transform.translation.y
                q = t.transform.rotation
                odom_yaw = math.atan2(2*(q.w*q.z + q.x*q.y),
                                      1 - 2*(q.y*q.y + q.z*q.z))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                pass  # 回落到 odom_callback 填写的 odom_x/y/yaw

            if not initialized_start_point:
                node.get_logger().info(f"传感器就绪, 首点: {waypoints[current_idx]}")
                initialized_start_point = True

            # stuck 检测: 采样当前 (x,y), 维护滑动窗口, 检测 STUCK_WINDOW 内位移
            now_t = time.time()
            stuck_history.append((now_t, odom_x, odom_y))
            stuck_history = [(t, x, y) for (t, x, y) in stuck_history if now_t - t <= STUCK_WINDOW]
            if len(stuck_history) > 10 and now_t - stuck_history[0][0] >= STUCK_WINDOW - 1:
                dx_ = odom_x - stuck_history[0][1]
                dy_ = odom_y - stuck_history[0][2]
                if math.hypot(dx_, dy_) < STUCK_DIST:
                    _tgt = waypoints[current_idx]
                    node.get_logger().warn(
                        f"[STUCK] {STUCK_WINDOW:.0f}s 内位移 < {STUCK_DIST*100:.0f}cm. "
                        f"航点 {current_idx}={_tgt} pose=({odom_x:.2f},{odom_y:.2f}) "
                        f"已打卡 {sum(scored)}/12"
                    )
                    break

            # -------------------------------------------------------------
            # --- A. 17 航点顺序推进 + 12 格机会式打分 ---
            # -------------------------------------------------------------
            # 机会式: 任何一格只要车身进入 1x1m 内就标记
            for _i, (_cx, _cy) in enumerate(score_cells):
                if not scored[_i] and abs(odom_x - _cx) <= 0.5 and abs(odom_y - _cy) <= 0.5:
                    scored[_i] = True
                    node.get_logger().info(f"[进入] 格 {_i} ({_cx:+.1f},{_cy:+.1f}) ({sum(scored)}/12)")

            if all(scored):
                node.get_logger().info("全部 12 格已覆盖, 停车.")
                break

            target_x, target_y = waypoints[current_idx]
            dx = target_x - odom_x
            dy = target_y - odom_y

            if abs(dx) < BOX_LIMIT and abs(dy) < BOX_LIMIT:
                current_idx += 1
                if current_idx >= num_waypoints:
                    lap_count += 1
                    node.get_logger().info(f"第 {lap_count}/{max_laps} 圈.")
                    if lap_count >= max_laps:
                        if all(scored):
                            node.get_logger().info("达成, 停车.")
                        else:
                            node.get_logger().info(f"圈数到, 未打卡 = {[i for i,s in enumerate(scored) if not s]}. 停车.")
                        break
                    current_idx = 0
                target_x, target_y = waypoints[current_idx]
                dx = target_x - odom_x
                dy = target_y - odom_y

            # -------------------------------------------------------------
            # --- B. 激光雷达 8 扇区高精度划分 ---
            # -------------------------------------------------------------
            num_scans = len(lidar_ranges)
            if num_scans >= 360:
                # 8 个扇区, 每个 45°. 角度以 base_scan 正前为 0, 逆时针为正
                r_f  = lidar_ranges[337:360] + lidar_ranges[0:22]   # 正前
                r_fl = lidar_ranges[22:67]    # 左前
                r_l  = lidar_ranges[67:112]   # 正左
                r_bl = lidar_ranges[112:157]  # 左后
                r_b  = lidar_ranges[157:202]  # 正后
                r_br = lidar_ranges[202:247]  # 右后
                r_r  = lidar_ranges[247:292]  # 正右
                r_fr = lidar_ranges[292:337]  # 右前
            else:
                r_f = r_fl = r_l = r_bl = r_b = r_br = r_r = r_fr = lidar_ranges

            min_f  = min(r_f)
            min_fl = min(r_fl)
            min_fr = min(r_fr)
            min_l  = min(r_l)
            min_r  = min(r_r)
            min_bl = min(r_bl)
            min_b  = min(r_b)
            min_br = min(r_br)

            # -------------------------------------------------------------
            # --- C+D. 状态机: AVOID (Bug-0 弧形绕障) / ALIGN / DRIVE ---
            # AVOID 一旦触发就承诺方向, 弧形绕过障碍直到 "跟随侧" 清空或转够角度
            # -------------------------------------------------------------
            target_angle = math.atan2(dy, dx)
            angle_error = wrap_to_pi(target_angle - odom_yaw)
            front_danger = min_f < SAFE_RADIUS

            # AVOID 进入/维持判断
            if avoid_active:
                rotated = abs(wrap_to_pi(odom_yaw - avoid_start_yaw))
                traveled = math.hypot(odom_x - avoid_start_x, odom_y - avoid_start_y)
                # 跟随侧: avoid_dir > 0 (左转) 意味着障碍在右 -> 监控 R/FR 侧
                if avoid_dir > 0:
                    side_min = min(min_r, min_fr)
                else:
                    side_min = min(min_l, min_fl)
                # 退出条件: 已转够角度且前方清空, 或跟随侧开阔 (障碍过去了)
                exit_on_rotate = (rotated > AVOID_MIN_ROT and min_f > SAFE_RADIUS + 0.05)
                exit_on_pass   = (traveled > 0.25 and side_min > AVOID_EXIT_SIDE
                                  and min_f > SAFE_RADIUS + 0.05)
                if exit_on_rotate or exit_on_pass:
                    avoid_active = False
                    current_path = None   # 退出避障后强制重规划, 用更新的地图绕行

            if (not avoid_active) and front_danger:
                # 进入 AVOID: 选方向 = target 侧优先, 两侧相差显著时让更空的一侧
                avoid_active = True
                toward_left = angle_error > 0
                left_clear  = min_fl + min_l
                right_clear = min_fr + min_r
                if toward_left and left_clear > right_clear - 0.30:
                    avoid_dir = 1   # 左转, 障碍保持在右
                elif (not toward_left) and right_clear > left_clear - 0.30:
                    avoid_dir = -1  # 右转, 障碍保持在左
                elif left_clear > right_clear:
                    avoid_dir = 1
                else:
                    avoid_dir = -1
                avoid_start_yaw = odom_yaw
                avoid_start_x = odom_x
                avoid_start_y = odom_y

            # 根据状态计算速度
            if avoid_active:
                # SIDESTEP: 弧形前进绕障. 距离过近时停下纯转, 否则带线速度走弧
                if min_f < 0.22:
                    linear_vel = 0.0
                    angular_vel = avoid_dir * p_max_w
                else:
                    linear_vel = SIDESTEP_V
                    angular_vel = avoid_dir * SIDESTEP_W
            elif abs(angle_error) > ALIGN_TOL:
                linear_vel = 0.0
                angular_vel = max(-p_max_w, min(p_max_w, K_HEAD * angle_error))
            else:
                linear_vel = p_max_v
                angular_vel = max(-0.5, min(0.5, K_DRIVE * angle_error))

            # 尾部保护: 纯原地旋转时 (v≈0, |ω|大) 才启用
            if linear_vel < 0.05 and abs(angular_vel) > 0.2:
                rear_min = min(min_bl, min_b, min_br, min_l, min_r)
                if rear_min < SAFE_RADIUS:
                    gain = max(0.0, (rear_min - 0.21) / (SAFE_RADIUS - 0.21))
                    angular_vel *= gain

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
