#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================================================================
# 节点: astar_navigator
#
# 任务背景:
#   4x4m 场地, 12 个 1x1m 得分格在外圈 (中央 2x2 是非得分区).
#   场内有 4 个圆筒 beacon + 4 个 L 挡板 (位置随机, 数量类型固定).
#   目标: 90s 内尽可能多走过得分格, 不撞挡板.
#
# 实测最终配置 (V_max=0.26, W_max=1.82, 0 碰撞):
#   - 12/12 in ~98s walltime
#   - RTF=1.0 时 sim 时间 ≈ walltime, 上实机预估 100s 量级
#
# ─────────────────────── 策略层级 ───────────────────────
#
# 全局: 贪心最近 cell 选目标
#   - 每帧选: 未打卡 + 未 deferred 的 cell 中, 欧氏距离最近的
#   - 不预设 zigzag 顺序 -> 适配 L 挡板任意布局
#   - STUCK 3 次的 cell 进 deferred 集合, 一圈后再解封
#
# 全局规划: A* + 软膨胀代价场 (cost_field)
#   - HARD_R 内 (25cm): 不可走 (硬禁区)
#   - HARD_R..SOFT_R (25-45cm): 可走但 cost 渐增, 倾向远离障碍但不绕远
#   - 渐进 fallback: 起点/终点恰好在硬禁区 -> 局部解封 2 格
#   - REPLAN_PERIOD=1.0s 周期重规划, 适应实时建图
#
# 路径跟随: 纯追踪
#   - LOOKAHEAD 0.20m 取路径上最近的前瞻点
#   - cos(yaw_err) 调速度, K_YAW * yaw_err 调角速度
#   - 偏差 > YAW_HARD 1.2rad 时只转不走
#
# 反应式安全层 (覆盖纯追踪输出):
#   - SAFE_FRONT < 0.28m: 立即刹停 + 重规划 (撞前救援)
#   - SLOW_FRONT < 0.50m: 按距离比例减速到 0.3*v
#   - SAFE_SIDE < 0.22m (侧前 ±20-60°): v ×= 0.4 (L 挡板斜面常侧蹭)
#
# 兜底机制:
#   - WARMUP 4s 启动自旋: 让 SLAM 在出发前看清 4 个 L + 4 beacon
#                         (省 17s — 否则第一次过 L 时 A* 无图凭空规划)
#   - STUCK 检测: 3s 内位移 < 8cm -> 后退 + 反向转 1s
#   - DEFER: 同格 STUCK 3 次 -> 跳到下个最近, 一圈后回头
#   - opportunistic 打卡: 进入任何 cell 0.29m 圈就记分, 路过即可
#
# ─────────────────────── 历史踩坑 ───────────────────────
#
# 试过但更慢/更危险的方案 (按时间顺序):
#   1. 左墙跟随       6/12, NW 角后过冲乱套
#   2. APF 势场法     0/12, 起步在 4 beacon 中心被斥力围困打转
#   3. VFH histogram  6/12, L 挡板下慢传 (42s 走 1m)
#   4. 硬编 bypass    2/12, 中转点选错就堵死, 障碍变了完全失效
#   5. zigzag 预设序  9/12 → 12/12 (V=0.55), 11/12 (V=0.26 hw 限速)
#   6. 二元膨胀 35cm  12/12, 但绕 L 远 (107s)
#   7. 软膨胀 + zigzag 12/12 in 107s
#   8. 软膨胀 + 贪心  12/12 in 98s   <-- 当前
#
# =========================================================================

import rclpy
import tf2_ros
import heapq
import numpy as np
import os
import struct
import zlib
from rclpy.duration import Duration
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
import math
import time

odom_ready = False
lidar_ready = False
map_ready   = False
odom_x = 0.0
odom_y = 0.0
odom_yaw = 0.0
lidar_ranges = []
map_grid = None
map_res = 0.05
map_origin_x = 0.0
map_origin_y = 0.0
map_w = 0
map_h = 0

MAX_LINEAR_X = 0.26
MAX_ANGULAR_Z = 1.82


def odom_callback(msg):
    global odom_ready, odom_x, odom_y, odom_yaw
    odom_x = msg.pose.pose.position.x
    odom_y = msg.pose.pose.position.y
    q = msg.pose.pose.orientation
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    odom_yaw = math.atan2(siny, cosy)
    odom_ready = True


def scan_callback(msg):
    global lidar_ready, lidar_ranges
    cleaned = []
    for r in msg.ranges:
        if math.isnan(r) or math.isinf(r) or r < 0.05:
            cleaned.append(3.5)
        else:
            cleaned.append(r)
    lidar_ranges = cleaned
    lidar_ready = True


def map_callback(msg):
    global map_ready, map_grid, map_res, map_origin_x, map_origin_y, map_w, map_h
    map_w = msg.info.width
    map_h = msg.info.height
    map_grid = np.array(msg.data, dtype=np.int8).reshape(map_h, map_w)
    map_res = msg.info.resolution
    map_origin_x = msg.info.origin.position.x
    map_origin_y = msg.info.origin.position.y
    map_ready = True


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def world_to_grid(x, y):
    c = int((x - map_origin_x) / map_res)
    r = int((y - map_origin_y) / map_res)
    return c, r


def grid_to_world(c, r):
    return (map_origin_x + (c + 0.5) * map_res,
            map_origin_y + (r + 0.5) * map_res)


def inflate(grid, radius):
    """Chebyshev 膨胀, 障碍 (>=50) 周围 radius 格全标 True (二元)"""
    occ = (grid >= 50)
    out = occ.copy()
    for _ in range(radius):
        new = out.copy()
        new[:-1] |= out[1:]
        new[1:]  |= out[:-1]
        new[:, :-1] |= out[:, 1:]
        new[:, 1:]  |= out[:, :-1]
        out = new
    return out


def cost_field(grid, hard_r, soft_r):
    """
    返回 (blocked, cost):
      blocked: hard_r 格内的视为不可走 (二元)
      cost:    超出 hard_r 但在 soft_r 内, 按到障碍距离给附加代价
               距离越近代价越高, A* 会本能远离障碍但允许靠近
    """
    h, w = grid.shape
    occ = (grid >= 50)
    # 用 BFS / 多次膨胀算最近障碍距离 (Chebyshev)
    dist = np.full((h, w), 99, dtype=np.int8)
    dist[occ] = 0
    cur = occ.copy()
    for d in range(1, soft_r + 1):
        new = cur.copy()
        new[:-1] |= cur[1:]
        new[1:]  |= cur[:-1]
        new[:, :-1] |= cur[:, 1:]
        new[:, 1:]  |= cur[:, :-1]
        newly = new & ~cur
        dist[newly] = d
        cur = new
    blocked = dist <= hard_r
    # cost: dist == hard_r+1 给最高额外代价, 离得远递减
    extra = np.zeros((h, w), dtype=np.float32)
    for d in range(hard_r + 1, soft_r + 1):
        extra[dist == d] = (soft_r - d + 1) * 0.6   # 离障碍越近 cost 越高
    return blocked, extra


def unblock(blocked, c, r, radius):
    h, w = blocked.shape
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nc, nr = c + dc, r + dr
            if 0 <= nc < w and 0 <= nr < h:
                blocked[nr, nc] = False


def astar(blocked, start, goal, extra_cost=None):
    """8-连通 A*, blocked 是 bool 2D, extra_cost 是 [h,w] float (可选, 软膨胀代价)"""
    h, w = blocked.shape
    if not (0 <= start[0] < w and 0 <= start[1] < h): return None
    if not (0 <= goal[0] < w and 0 <= goal[1] < h):   return None
    if blocked[goal[1], goal[0]]: return None
    if start == goal: return [start]
    DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    def hd(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
    open_h = [(hd(start, goal), 0.0, start)]
    came = {start: None}
    g = {start: 0.0}
    while open_h:
        _, gc, cur = heapq.heappop(open_h)
        if gc > g.get(cur, float('inf')): continue
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return list(reversed(path))
        for dc, dr in DIRS:
            nc, nr = cur[0] + dc, cur[1] + dr
            if nc < 0 or nc >= w or nr < 0 or nr >= h: continue
            if blocked[nr, nc]: continue
            step = 1.41421 if (dc and dr) else 1.0
            if extra_cost is not None:
                step += float(extra_cost[nr, nc])
            ng = gc + step
            n = (nc, nr)
            if ng < g.get(n, float('inf')):
                g[n] = ng
                came[n] = cur
                heapq.heappush(open_h,
                               (ng + hd(n, goal), ng, n))
    return None


def plan_path(start_xy, goal_xy, hard_r=4, soft_r=8):
    """
    A* 软膨胀:
      hard_r 格内 = 不可走 (~20cm 物理半径)
      hard_r..soft_r 格 = 可走但代价增加 (倾向远离障碍但允许靠近)
    若 hard 起点/终点恰好被堵, 解锁 2 格.
    """
    if not map_ready or map_grid is None:
        return None, None
    s = world_to_grid(*start_xy)
    g = world_to_grid(*goal_xy)
    blocked, extra = cost_field(map_grid, hard_r, soft_r)
    unblock(blocked, s[0], s[1], 2)
    unblock(blocked, g[0], g[1], 2)
    cells = astar(blocked, s, g, extra_cost=extra)
    if cells is None:
        # fallback: 更小的 hard radius
        blocked2, _ = cost_field(map_grid, max(1, hard_r // 2), soft_r)
        unblock(blocked2, s[0], s[1], 2)
        unblock(blocked2, g[0], g[1], 2)
        cells = astar(blocked2, s, g, extra_cost=extra)
        if cells is None:
            return None, None
    pts = [grid_to_world(c, r) for c, r in cells]
    if len(pts) > 6:
        step = max(1, len(pts) // 5)
        ds = pts[::step]
        if ds[-1] != pts[-1]:
            ds.append(pts[-1])
        pts = ds
    return pts, hard_r


def sector_min(ranges, a_deg, b_deg):
    n = len(ranges)
    a = int(round(a_deg * n / 360.0)) % n
    b = int(round(b_deg * n / 360.0)) % n
    if a <= b:
        return min(ranges[a:b + 1])
    return min(ranges[a:] + ranges[:b + 1])


def publish_cmd(pub, node, v, w):
    cmd = TwistStamped()
    cmd.header.stamp = node.get_clock().now().to_msg()
    cmd.header.frame_id = 'base_link'
    cmd.twist.linear.x = float(max(-MAX_LINEAR_X, min(MAX_LINEAR_X, v)))
    cmd.twist.angular.z = float(max(-MAX_ANGULAR_Z, min(MAX_ANGULAR_Z, w)))
    pub.publish(cmd)


def write_png_gray(path, image):
    """Write an 8-bit grayscale PNG using only the Python standard library."""
    height, width = image.shape
    raw_rows = b''.join(b'\x00' + bytes(row.tolist()) for row in image)

    def chunk(tag, data):
        checksum = zlib.crc32(tag + data) & 0xffffffff
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', checksum)

    png = [
        b'\x89PNG\r\n\x1a\n',
        chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0)),
        chunk(b'IDAT', zlib.compress(raw_rows, 9)),
        chunk(b'IEND', b''),
    ]
    with open(path, 'wb') as f:
        f.write(b''.join(png))


def package_maps_dir():
    cwd_package = os.path.join(os.getcwd(), 'src', 'ele434_team10_2026')
    source_package = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for package_dir in (cwd_package, source_package):
        if os.path.exists(os.path.join(package_dir, 'package.xml')):
            return os.path.join(package_dir, 'maps')
    return os.path.join(os.getcwd(), 'maps')


def save_map_files(node):
    if not map_ready or map_grid is None or map_w <= 0 or map_h <= 0:
        node.get_logger().warn('No SLAM map received; maps/explore_map.* not saved.')
        return

    maps_dir = package_maps_dir()
    os.makedirs(maps_dir, exist_ok=True)
    png_path = os.path.join(maps_dir, 'explore_map.png')
    yaml_path = os.path.join(maps_dir, 'explore_map.yaml')

    image = np.full(map_grid.shape, 205, dtype=np.uint8)
    image[(map_grid >= 0) & (map_grid < 50)] = 254
    image[map_grid >= 50] = 0
    image = np.flipud(image)
    write_png_gray(png_path, image)

    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write('image: explore_map.png\n')
        f.write(f'resolution: {map_res:.6f}\n')
        f.write(f'origin: [{map_origin_x:.6f}, {map_origin_y:.6f}, 0.000000]\n')
        f.write('negate: 0\n')
        f.write('occupied_thresh: 0.65\n')
        f.write('free_thresh: 0.25\n')
        f.write('mode: trinary\n')

    node.get_logger().info(f'Saved SLAM map to {png_path} and {yaml_path}')


def main(args=None):
    global odom_x, odom_y, odom_yaw

    rclpy.init(args=args)
    node = rclpy.create_node('astar_navigator')
    cmd_pub = node.create_publisher(TwistStamped, '/cmd_vel', 10)
    node.create_subscription(Odometry, '/odom', odom_callback, 10)
    node.create_subscription(LaserScan, '/scan', scan_callback, 10)
    node.create_subscription(OccupancyGrid, '/map', map_callback, 10)

    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, node)

    # ----- 调参 -----
    V_MAX        = MAX_LINEAR_X
    W_MAX        = MAX_ANGULAR_Z
    K_YAW        = 2.8
    YAW_HARD     = 1.45
    SCORE_MARGIN = 0.22
    TARGET_MARGIN = 0.26
    REACH_TOL    = 0.22
    MAX_RUN_T    = 90.0       # 作业评分窗口

    HARD_R       = 5          # 25cm 硬禁区 (robot 半径 21cm + 4cm)
    SOFT_R       = 7          # 35cm 软膨胀; 硬禁区仍保留, 路径更短
    REPLAN_PERIOD = 0.8

    # (C) 侧前方也保护; SAFE 触发刹停, SLOW 触发减速
    SAFE_FRONT   = 0.28
    SLOW_FRONT   = 0.44
    SAFE_SIDE    = 0.22       # 侧前 ±20-60° 比这近就强减速

    STUCK_WINDOW = 3.0        # 放宽 stuck 检测 (慢走时 5cm/3s 不算 stuck)
    STUCK_DIST   = 0.08
    ESCAPE_DUR   = 1.0
    DEFER_STUCK  = 3          # 给更多机会

    LOOKAHEAD    = 0.35

    STARTUP_WAIT_T = 3.0      # launch 已先启动 SLAM, 不再白等 12s
    WARMUP_T     = 2.0        # 快速自旋建图, 省下更多有效移动时间

    zone_bounds = [
        (-2.0, -1.0,  1.0,  2.0), (-1.0,  0.0,  1.0,  2.0),
        ( 0.0,  1.0,  1.0,  2.0), ( 1.0,  2.0,  1.0,  2.0),
        ( 1.0,  2.0,  0.0,  1.0), ( 1.0,  2.0, -1.0,  0.0),
        ( 1.0,  2.0, -2.0, -1.0), ( 0.0,  1.0, -2.0, -1.0),
        (-1.0,  0.0, -2.0, -1.0), (-2.0, -1.0, -2.0, -1.0),
        (-2.0, -1.0, -1.0,  0.0), (-2.0, -1.0,  0.0,  1.0),
    ]

    def inner_target(bounds):
        xmin, xmax, ymin, ymax = bounds
        x = max(xmin + TARGET_MARGIN, min(xmax - TARGET_MARGIN, 0.0))
        y = max(ymin + TARGET_MARGIN, min(ymax - TARGET_MARGIN, 0.0))
        return x, y

    score_cells = [inner_target(bounds) for bounds in zone_bounds]
    # 贪心最近策略: 每步选最近的"未打卡且未 deferred"格
    # 不依赖 zigzag 预设 -> 适配任意 L 挡板布局变化
    scored = [False] * 12
    completed_once = False
    deferred = set()
    cur_idx = 0      # 临时, 第一帧会被替换为最近格
    cur_stuck = 0

    cur_path     = None
    last_replan_t = 0.0
    pose_hist = []
    escape_until = 0.0
    escape_dir   = 1
    last_status_t = 0.0
    # 防 opportunistic 死循环: 每个 cell 上次被 defer 的时间, 冷却期内不重试
    last_defer_t = {}     # cell_idx -> time
    DEFER_COOLDOWN = 10.0

    node.get_logger().info("A* 节点已启动, 等待传感器 + map + SLAM ...")

    # Wait briefly for sensor data. SLAM continues building during warmup.
    wait_t0 = time.time()
    while rclpy.ok() and time.time() - wait_t0 < STARTUP_WAIT_T:
        rclpy.spin_once(node, timeout_sec=0.05)
        if not (odom_ready and lidar_ready):
            continue
        try:
            t = tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.0))
            tx, ty = t.transform.translation.x, t.transform.translation.y
            if math.hypot(tx, ty) < 0.4 and map_ready:
                node.get_logger().info(
                    f"启动确认: pose=({tx:+.2f},{ty:+.2f}), map={map_w}x{map_h}")
                break
        except Exception:
            pass
        if odom_ready and lidar_ready and time.time() - wait_t0 > 0.8:
            break

    # (D) WARMUP: 原地快转一小段时间, 给 SLAM 初始观测但不浪费太多评分窗口.
    node.get_logger().info(f"WARMUP: 原地转 {WARMUP_T}s 让 SLAM 建图...")
    warmup_t0 = time.time()
    while rclpy.ok() and time.time() - warmup_t0 < WARMUP_T:
        rclpy.spin_once(node, timeout_sec=0.05)
        publish_cmd(cmd_pub, node, 0.0, W_MAX)
    publish_cmd(cmd_pub, node, 0.0, 0.0)
    node.get_logger().info("WARMUP 完成, 开始任务.")
    start_time = time.time()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if not (odom_ready and lidar_ready):
                continue
            try:
                t = tf_buffer.lookup_transform(
                    'map', 'base_footprint', rclpy.time.Time(),
                    timeout=Duration(seconds=0.0))
                odom_x = t.transform.translation.x
                odom_y = t.transform.translation.y
                q = t.transform.rotation
                odom_yaw = math.atan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                pass

            now_t = time.time()

            # 机会式打卡
            for i, bounds in enumerate(zone_bounds):
                xmin, xmax, ymin, ymax = bounds
                fully_inside = (
                    xmin + SCORE_MARGIN <= odom_x <= xmax - SCORE_MARGIN
                    and ymin + SCORE_MARGIN <= odom_y <= ymax - SCORE_MARGIN
                )
                if not scored[i] and fully_inside:
                    scored[i] = True
                    elapsed = now_t - start_time
                    cx, cy = score_cells[i]
                    node.get_logger().info(
                        f"[打卡 t={elapsed:5.1f}s] 格 {i + 1} "
                        f"target=({cx:+.2f},{cy:+.2f})  {sum(scored)}/12")
                    if i == cur_idx:
                        cur_path = None  # 强制重规划下一个

            if all(scored) and not completed_once:
                completed_once = True
                node.get_logger().info(
                    f"12 格全覆盖 t={now_t-start_time:.1f}s, 继续运行到 90s.")
            if now_t - start_time > MAX_RUN_T:
                node.get_logger().warn(
                    f"超时 90s, 已打卡 {sum(scored)}/12, 停车.")
                break

            # 贪心最近: 选最近的未打卡且未 deferred 的格
            prev_idx = cur_idx
            unscored = [i for i in range(12) if not scored[i]]
            if not unscored:
                unscored = list(range(12))
            if unscored:
                # 优先未 deferred 的; 全 deferred 才考虑 deferred (解封最近)
                avail = [i for i in unscored if i not in deferred]
                if not avail:
                    # 全部 unscored 都已 deferred -> 解封最近
                    avail = unscored
                    nearest = min(avail, key=lambda i: math.hypot(
                        score_cells[i][0] - odom_x,
                        score_cells[i][1] - odom_y))
                    if nearest in deferred:
                        node.get_logger().info(
                            f"全 deferred, 解封最近格 {nearest+1} 重试.")
                        deferred.discard(nearest)
                cur_idx = min(avail, key=lambda i: math.hypot(
                    score_cells[i][0] - odom_x,
                    score_cells[i][1] - odom_y))

            if cur_idx != prev_idx:
                cur_path = None
                cur_stuck = 0
                pose_hist = []

            target_x, target_y = score_cells[cur_idx]
            d_goal = math.hypot(target_x - odom_x, target_y - odom_y)
            if d_goal < REACH_TOL and not scored[cur_idx]:
                # 已到附近但还没被打卡 (理论上 SCORE_TOL > REACH_TOL 不会), 强制 score
                scored[cur_idx] = True
                continue

            # ---- 重规划 ----
            need_replan = (cur_path is None
                           or now_t - last_replan_t > REPLAN_PERIOD)
            if need_replan and map_ready:
                pp, rr_used = plan_path((odom_x, odom_y),
                                         (target_x, target_y),
                                         hard_r=HARD_R, soft_r=SOFT_R)
                if pp is None:
                    node.get_logger().warn(
                        f"A* 失败 cell{cur_idx+1}, 直线兜底")
                    cur_path = [(target_x, target_y)]
                else:
                    cur_path = pp
                last_replan_t = now_t

            # 前瞻点
            if cur_path is None or len(cur_path) == 0:
                lh_x, lh_y = target_x, target_y
            else:
                # 沿 cur_path 找第一个距离 robot > LOOKAHEAD 的点
                lh_x, lh_y = cur_path[-1]
                for px, py in cur_path:
                    if math.hypot(px - odom_x, py - odom_y) > LOOKAHEAD:
                        lh_x, lh_y = px, py
                        break

            target_yaw = math.atan2(lh_y - odom_y, lh_x - odom_x)
            yaw_err = wrap_to_pi(target_yaw - odom_yaw)

            # ---- stuck (旋转期不算; 只在我们实际命令前进时积累 pose_hist) ----
            # 注: yaw_err > YAW_HARD 时 v 会被设成 0, 那不是真 stuck
            if abs(yaw_err) <= YAW_HARD:
                pose_hist.append((now_t, odom_x, odom_y))
                pose_hist = [p for p in pose_hist
                             if now_t - p[0] <= STUCK_WINDOW]
            else:
                pose_hist = [(now_t, odom_x, odom_y)]
            if now_t < escape_until:
                publish_cmd(cmd_pub, node, -0.10, escape_dir * 1.5)
                continue
            if (len(pose_hist) > 30
                    and now_t - pose_hist[0][0] >= STUCK_WINDOW - 0.3):
                d_t = math.hypot(odom_x - pose_hist[0][1],
                                 odom_y - pose_hist[0][2])
                if d_t < STUCK_DIST:
                    escape_until = now_t + ESCAPE_DUR
                    escape_dir = -escape_dir
                    cur_stuck += 1
                    pose_hist = []
                    cur_path = None  # 重规划
                    node.get_logger().warn(
                        f"[STUCK #{cur_stuck}] 后退{ESCAPE_DUR}s "
                        f"@({odom_x:+.2f},{odom_y:+.2f})")
                    if cur_stuck >= DEFER_STUCK:
                        node.get_logger().warn(
                            f"[DEFER] 格 {cur_idx+1} STUCK {cur_stuck}x, 跳过.")
                        deferred.add(cur_idx)
                        last_defer_t[cur_idx] = now_t
                        # 下一帧贪心选下一最近 cell, 不需要在这里强切
                        cur_stuck = 0
                    continue

            d_front = float('inf')
            d_fl = float('inf')
            d_fr = float('inf')
            d_side = float('inf')
            if len(lidar_ranges) >= 36:
                d_front = sector_min(lidar_ranges, -20, 20)
                d_fl = sector_min(lidar_ranges, 20, 60)
                d_fr = sector_min(lidar_ranges, -60, -20)
                d_side = min(d_fl, d_fr)

            # ---- 速度 ----
            if abs(yaw_err) > YAW_HARD:
                if d_front > SLOW_FRONT and d_side > SAFE_SIDE and abs(yaw_err) < 1.85:
                    v = 0.08
                else:
                    v = 0.0
                w = max(-W_MAX, min(W_MAX, K_YAW * yaw_err))
            else:
                turn_factor = max(0.68, math.cos(yaw_err))
                if abs(yaw_err) > 1.10:
                    turn_factor = 0.52
                v = V_MAX * turn_factor
                w = max(-W_MAX, min(W_MAX, K_YAW * yaw_err))

            # 反应式刹车 + 减速
            if len(lidar_ranges) >= 36:
                if d_front < SAFE_FRONT:
                    v = 0.0
                    cur_path = None
                elif d_front < SLOW_FRONT:
                    factor = (d_front - SAFE_FRONT) / (SLOW_FRONT - SAFE_FRONT)
                    v *= max(0.55, factor)
                # 侧前方近 -> 强制减速 (但不刹停, 否则过 L 时卡死)
                if d_side < SAFE_SIDE:
                    v *= 0.65

            publish_cmd(cmd_pub, node, v, w)

            # 状态日志
            if now_t - last_status_t >= 1.0:
                last_status_t = now_t
                path_len = len(cur_path) if cur_path else 0
                df = (sector_min(lidar_ranges, -20, 20)
                      if len(lidar_ranges) >= 36 else 0)
                node.get_logger().info(
                    f"t={now_t-start_time:5.1f} "
                    f"pos=({odom_x:+.2f},{odom_y:+.2f}) yaw={math.degrees(odom_yaw):+6.1f} "
                    f"-> cell{cur_idx+1}({target_x:+.1f},{target_y:+.1f}) "
                    f"path={path_len} lh=({lh_x:+.1f},{lh_y:+.1f}) "
                    f"v={v:+.2f} w={w:+.2f} d_f={df:.2f}")

    except KeyboardInterrupt:
        node.get_logger().info("中断, 停车.")
    finally:
        publish_cmd(cmd_pub, node, 0.0, 0.0)
        save_map_files(node)
        time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
