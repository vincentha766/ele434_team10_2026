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
#   - HARD_R 内 (35cm): 不可走 (硬禁区, robot 21cm + 障碍 10cm + 4cm 跟踪余量)
#   - HARD_R..SOFT_R (35-55cm): 可走但 cost 渐增, 倾向远离障碍但不绕远
#   - 终点贴障碍时不解封, 改 snap 到最近合法 cell (find_nearest_free)
#   - 渐进 fallback HARD_R 7→6→5, 保留安全底线
#   - REPLAN_PERIOD=1.0s 周期重规划, 适应实时建图
#
# 路径跟随: 纯追踪 + 跟踪稳定性
#   - LOOKAHEAD 0.29m 取路径上最近的前瞻点
#   - v = V_MAX * cos(yaw_err), 大偏差 (>YAW_HARD) 时只转不走
#   - w = K_YAW * yaw_err 限幅在 ±W_MAX
#
# 兜底机制:
#   - WARMUP 4s 启动自旋: 让 SLAM 在出发前看清 4 个 L + 4 beacon
#   - STUCK 检测: 3s 内位移 < 8cm -> 后退 + 反向转 1s
#   - DEFER: 同格 STUCK 3 次 -> 跳到下个最近, 一圈后回头
#   - opportunistic 打卡: 进入任何 cell SCORE_TOL=0.45m 方圆就记分
#     (大到能让障碍中心 cell 也从 cell 边缘合法打卡, 又不会跨 cell)
#   - 分段紧急刹停 (lidar):
#     ±15° 前向 < 0.22m: 即将正撞
#     ±15-45° 侧前 < 0.16m: 侧前贴擦 (50cm 走廊不误触)
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
import numpy as np
from rclpy.duration import Duration
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
import math
import os
import time

from ele434_team10_2026_modules.nav_math import (
    wrap_to_pi, yaw_from_quaternion,
)
from ele434_team10_2026_modules.path_planner import plan_path
from ele434_team10_2026_modules.motion_control import (
    find_lookahead_point,
)
from ele434_team10_2026_modules.reactive_safety import (
    sector_min,
)
from ele434_team10_2026_modules.debug_logger import RunLogger


def publish_cmd(pub, node, v, w):
    cmd = TwistStamped()
    cmd.header.stamp = node.get_clock().now().to_msg()
    cmd.header.frame_id = 'base_link'
    cmd.twist.linear.x = float(v)
    cmd.twist.angular.z = float(w)
    pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('astar_navigator')
    cmd_pub = node.create_publisher(TwistStamped, '/cmd_vel', 10)

    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, node)

    # ---- sensor state (replaces globals) ----
    state = {
        'odom_ready': False,
        'lidar_ready': False,
        'map_ready': False,
        'odom_x': 0.0,
        'odom_y': 0.0,
        'odom_yaw': 0.0,
        'lidar_ranges': [],
        'map_grid': None,
        'map_res': 0.05,
        'map_origin_x': 0.0,
        'map_origin_y': 0.0,
        'map_w': 0,
        'map_h': 0,
    }

    def odom_callback(msg):
        state['odom_x'] = msg.pose.pose.position.x
        state['odom_y'] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        state['odom_yaw'] = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        state['odom_ready'] = True

    def scan_callback(msg):
        cleaned = []
        for r in msg.ranges:
            if math.isnan(r) or math.isinf(r) or r < 0.05:
                cleaned.append(3.5)
            else:
                cleaned.append(r)
        state['lidar_ranges'] = cleaned
        state['lidar_ready'] = True

    def map_callback(msg):
        state['map_w'] = msg.info.width
        state['map_h'] = msg.info.height
        state['map_grid'] = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        state['map_res'] = msg.info.resolution
        state['map_origin_x'] = msg.info.origin.position.x
        state['map_origin_y'] = msg.info.origin.position.y
        state['map_ready'] = True

    node.create_subscription(Odometry, '/odom', odom_callback, 10)
    node.create_subscription(LaserScan, '/scan', scan_callback, 10)
    node.create_subscription(OccupancyGrid, '/map', map_callback, 10)

    # ----- 调参 -----
    V_MAX        = 0.26       # hw 标称速度
    W_MAX        = 1.82
    K_YAW        = 2.4
    SCORE_TOL    = 0.45      # 拉宽到 cell 半宽附近, 容许障碍在中心时从 cell 边缘打卡
    REACH_TOL    = 0.22
    MAX_RUN_T    = 240.0      # 放松时限, 优先零碰撞

    HARD_R       = 7          # 35cm 硬禁区
    SOFT_R       = 11         # 55cm 软膨胀
    REPLAN_PERIOD = 1.0

    # 紧急刹停: 极近距离才触发, 比 robot 半径稍小, 不在窄角落误触发
    EMERG_FRONT  = 0.22       # lidar(中心) 到障碍表面 < 22cm = 即将贴 (robot 半径 21cm)
    YAW_HARD     = 1.2        # 大偏差只转不走 (路径跟踪稳定性, 不是减速)

    STUCK_WINDOW = 3.0        # 放宽 stuck 检测 (慢走时 5cm/3s 不算 stuck)
    STUCK_DIST   = 0.08
    ESCAPE_DUR   = 1.0
    DEFER_STUCK  = 3          # 给更多机会

    LOOKAHEAD    = 0.29

    WARMUP_T     = 4.0        # 启动原地自旋 N 秒让 SLAM 建图

    score_cells = [
        (-1.5,  1.5), (-0.5,  1.5), ( 0.5,  1.5), ( 1.5,  1.5),  # 1-4
        ( 1.5,  0.5), ( 1.5, -0.5), ( 1.5, -1.5),                # 5-7
        ( 0.5, -1.5), (-0.5, -1.5), (-1.5, -1.5),                # 8-10
        (-1.5, -0.5), (-1.5,  0.5),                              # 11-12
    ]
    # 贪心最近策略: 每步选最近的"未打卡且未 deferred"格
    # 不依赖 zigzag 预设 -> 适配任意 L 挡板布局变化
    scored = [False] * 12
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
    # 周期 dump SLAM 占据栅格 (诊断: 圆筒是否被 SLAM 看到 / cost_field 是什么样)
    MAP_DUMP_PERIOD = 10.0
    last_map_dump_t = 0.0

    # ---- 数据采集 / 调试日志 ----
    dbg = RunLogger(
        run_name='work',
        trace_fields=[
            'x', 'y', 'yaw_deg',
            'cell', 'tgt_x', 'tgt_y',
            'lh_x', 'lh_y', 'yaw_err_deg', 'd_goal',
            'v_cmd', 'w_cmd',
            'd_front', 'd_fl', 'd_fr', 'd_min', 'd_min_deg',
            'path_len', 'in_escape',
            'scored', 'cur_stuck', 'deferred_n',
            'phase',
        ],
        params={
            'V_MAX': V_MAX, 'W_MAX': W_MAX, 'K_YAW': K_YAW,
            'SCORE_TOL': SCORE_TOL,
            'REACH_TOL': REACH_TOL, 'MAX_RUN_T': MAX_RUN_T,
            'HARD_R': HARD_R, 'SOFT_R': SOFT_R,
            'REPLAN_PERIOD': REPLAN_PERIOD,
            'EMERG_FRONT': EMERG_FRONT, 'YAW_HARD': YAW_HARD,
            'STUCK_WINDOW': STUCK_WINDOW, 'STUCK_DIST': STUCK_DIST,
            'ESCAPE_DUR': ESCAPE_DUR, 'DEFER_STUCK': DEFER_STUCK,
            'LOOKAHEAD': LOOKAHEAD, 'WARMUP_T': WARMUP_T,
        },
    )
    node.get_logger().info(f"调试日志目录: {dbg.dir}")
    dbg.event('INIT', f'log_dir={dbg.dir}')

    node.get_logger().info("A* 节点已启动, 等待传感器 + map + SLAM ...")

    # 等 SLAM TF 报告靠近 (0,0)
    wait_t0 = time.time()
    while rclpy.ok() and time.time() - wait_t0 < 12.0:
        rclpy.spin_once(node, timeout_sec=0.05)
        if not (state['odom_ready'] and state['lidar_ready']):
            continue
        try:
            t = tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.0))
            tx, ty = t.transform.translation.x, t.transform.translation.y
            if math.hypot(tx, ty) < 0.4 and state['map_ready']:
                node.get_logger().info(
                    f"启动确认: pose=({tx:+.2f},{ty:+.2f}), "
                    f"map={state['map_w']}x{state['map_h']}")
                break
        except Exception:
            pass

    # (D) WARMUP: 原地慢转 WARMUP_T 秒, 让 SLAM 把周围 4 个 L + 4 beacon 都扫到
    node.get_logger().info(f"WARMUP: 原地转 {WARMUP_T}s 让 SLAM 建图...")
    dbg.event('WARMUP_START', f'duration={WARMUP_T}s')
    warmup_t0 = time.time()
    while rclpy.ok() and time.time() - warmup_t0 < WARMUP_T:
        rclpy.spin_once(node, timeout_sec=0.05)
        publish_cmd(cmd_pub, node, 0.0, 1.6)   # 中等角速度自旋一圈半
    publish_cmd(cmd_pub, node, 0.0, 0.0)
    dbg.event('WARMUP_DONE',
              f'pose=({state["odom_x"]:+.2f},{state["odom_y"]:+.2f}) '
              f'map_ready={state["map_ready"]}')
    node.get_logger().info("WARMUP 完成, 开始任务.")

    start_time = time.time()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if not (state['odom_ready'] and state['lidar_ready']):
                continue
            try:
                t = tf_buffer.lookup_transform(
                    'map', 'base_footprint', rclpy.time.Time(),
                    timeout=Duration(seconds=0.0))
                state['odom_x'] = t.transform.translation.x
                state['odom_y'] = t.transform.translation.y
                q = t.transform.rotation
                state['odom_yaw'] = yaw_from_quaternion(q.x, q.y, q.z, q.w)
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                pass

            odom_x = state['odom_x']
            odom_y = state['odom_y']
            odom_yaw = state['odom_yaw']
            lidar_ranges = state['lidar_ranges']
            now_t = time.time()

            # 周期 dump map_grid (.npy) + 元信息. SCORED 后强制 dump 一份.
            if (state['map_ready']
                    and (now_t - last_map_dump_t >= MAP_DUMP_PERIOD
                         or last_map_dump_t == 0.0)):
                last_map_dump_t = now_t
                elapsed_dump = now_t - start_time
                try:
                    np.save(
                        os.path.join(dbg.dir, f'map_t{int(elapsed_dump):04d}.npy'),
                        state['map_grid'])
                    dbg.event(
                        'MAP_DUMP',
                        f'elapsed={elapsed_dump:.1f}s '
                        f'shape={state["map_grid"].shape} '
                        f'res={state["map_res"]:.3f} '
                        f'origin=({state["map_origin_x"]:+.2f},{state["map_origin_y"]:+.2f}) '
                        f'pose=({odom_x:+.2f},{odom_y:+.2f})')
                except Exception as e:
                    dbg.event('MAP_DUMP_FAIL', f'err={e!r}')

            # 机会式打卡
            for i, (cx, cy) in enumerate(score_cells):
                if (not scored[i]
                        and abs(odom_x - cx) <= SCORE_TOL
                        and abs(odom_y - cy) <= SCORE_TOL):
                    scored[i] = True
                    elapsed = now_t - start_time
                    node.get_logger().info(
                        f"[打卡 t={elapsed:5.1f}s] 格 {i + 1} "
                        f"({cx:+.1f},{cy:+.1f})  {sum(scored)}/12")
                    dbg.event(
                        'SCORED',
                        f'cell={i + 1} at=({cx:+.1f},{cy:+.1f}) '
                        f'elapsed={elapsed:.1f}s total={sum(scored)}/12')
                    if i == cur_idx:
                        cur_path = None  # 强制重规划下一个

            if all(scored):
                node.get_logger().info(
                    f"12 格全覆盖 t={now_t-start_time:.1f}s, 停车.")
                dbg.event('ALL_DONE', f'elapsed={now_t-start_time:.1f}s')
                break
            if now_t - start_time > MAX_RUN_T:
                node.get_logger().warn(
                    f"超时 90s, 已打卡 {sum(scored)}/12, 停车.")
                dbg.event('TIMEOUT',
                          f'elapsed={now_t-start_time:.1f}s '
                          f'scored={sum(scored)}/12')
                break

            # 贪心最近: 选最近的未打卡且未 deferred 的格
            prev_idx = cur_idx
            unscored = [i for i in range(12) if not scored[i]]
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
            if need_replan and state['map_ready']:
                pp, rr_used = plan_path(
                    (odom_x, odom_y), (target_x, target_y),
                    state['map_grid'], state['map_res'],
                    state['map_origin_x'], state['map_origin_y'],
                    hard_r=HARD_R, soft_r=SOFT_R)
                if pp is None:
                    node.get_logger().warn(
                        f"A* 失败 cell{cur_idx+1}, 直线兜底")
                    dbg.event(
                        'REPLAN_FAIL',
                        f'cell={cur_idx+1} from=({odom_x:+.2f},{odom_y:+.2f}) '
                        f'to=({target_x:+.1f},{target_y:+.1f})')
                    cur_path = [(target_x, target_y)]
                else:
                    cur_path = pp
                    dbg.event(
                        'REPLAN',
                        f'cell={cur_idx+1} waypoints={len(pp)} rr={rr_used}')
                last_replan_t = now_t

            # 前瞻点 + yaw_err (stuck 检测需要 yaw_err)
            if cur_path is None or len(cur_path) == 0:
                lh_x, lh_y = target_x, target_y
            else:
                lh_x, lh_y = find_lookahead_point(
                    cur_path, odom_x, odom_y, LOOKAHEAD)

            target_yaw = math.atan2(lh_y - odom_y, lh_x - odom_x)
            yaw_err = wrap_to_pi(target_yaw - odom_yaw)

            # ---- stuck (旋转期不算; YAW_HARD 时 v=0, 不是真 stuck) ----
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
                    dbg.event(
                        'STUCK',
                        f'cell={cur_idx+1} n={cur_stuck} '
                        f'pose=({odom_x:+.2f},{odom_y:+.2f}) '
                        f'yaw={math.degrees(odom_yaw):+.1f} d_t={d_t:.3f}')
                    # 把 lidar + 最近路径快照下来 — 没现场的话这是事后唯一线索
                    if lidar_ranges:
                        dbg.snapshot(
                            f'stuck_cell{cur_idx+1}_n{cur_stuck}_lidar',
                            [f'# pose=({odom_x:+.3f},{odom_y:+.3f}) '
                             f'yaw_deg={math.degrees(odom_yaw):+.2f}']
                            + [f'{i} {r:.3f}' for i, r in enumerate(lidar_ranges)])
                    if cur_path:
                        dbg.snapshot(
                            f'stuck_cell{cur_idx+1}_n{cur_stuck}_path',
                            [f'{px:.3f} {py:.3f}' for px, py in cur_path])
                    if cur_stuck >= DEFER_STUCK:
                        node.get_logger().warn(
                            f"[DEFER] 格 {cur_idx+1} STUCK {cur_stuck}x, 跳过.")
                        dbg.event('DEFER', f'cell={cur_idx+1} after_stuck={cur_stuck}')
                        deferred.add(cur_idx)
                        last_defer_t[cur_idx] = now_t
                        cur_stuck = 0
                    continue

            # ---- 速度 ----
            # 路径跟踪: 大偏差只转不走, 否则 cos(yaw_err) 稳定弧线
            if abs(yaw_err) > YAW_HARD:
                v = 0.0
            else:
                v = V_MAX * math.cos(yaw_err)
            w = max(-W_MAX, min(W_MAX, K_YAW * yaw_err))

            # 紧急刹停: 分段防撞.
            #   ±15° 前向 < 0.22m: 即将正撞 (机器人前向冲入障碍)
            #   ±15-45° 侧前 < 0.16m: 侧前贴擦, 仅极近时触发 (50cm 走廊不误触)
            has_scan = len(lidar_ranges) >= 36
            if has_scan:
                d_front_emerg = sector_min(lidar_ranges, -15, 15)
                d_left = sector_min(lidar_ranges, 15, 45)
                d_right = sector_min(lidar_ranges, -45, -15)
                if d_front_emerg < EMERG_FRONT or min(d_left, d_right) < 0.16:
                    v = 0.0
                    cur_path = None  # 强制重规划

            publish_cmd(cmd_pub, node, v, w)

            # 全量 lidar 三向汇总 + 360° 全周最小 + 状态写入 CSV
            has_scan = len(lidar_ranges) >= 36
            d_front = sector_min(lidar_ranges, -20, 20) if has_scan else -1.0
            d_fl = sector_min(lidar_ranges, 20, 60) if has_scan else -1.0
            d_fr = sector_min(lidar_ranges, -60, -20) if has_scan else -1.0
            if has_scan:
                d_min_idx = min(range(len(lidar_ranges)),
                                key=lambda i: lidar_ranges[i])
                d_min = lidar_ranges[d_min_idx]
                # 等价于 sector_min 的角度约定: idx 0 = 0°, 顺序 0..n-1 = 0..360°,
                # 折回 [-180,180]
                d_min_deg = d_min_idx * 360.0 / len(lidar_ranges)
                if d_min_deg > 180.0:
                    d_min_deg -= 360.0
            else:
                d_min = -1.0
                d_min_deg = 0.0
            dbg.trace(
                x=odom_x, y=odom_y, yaw_deg=math.degrees(odom_yaw),
                cell=cur_idx + 1, tgt_x=target_x, tgt_y=target_y,
                lh_x=lh_x, lh_y=lh_y,
                yaw_err_deg=math.degrees(yaw_err), d_goal=d_goal,
                v_cmd=v, w_cmd=w,
                d_front=d_front, d_fl=d_fl, d_fr=d_fr,
                d_min=d_min, d_min_deg=d_min_deg,
                path_len=(len(cur_path) if cur_path else 0),
                in_escape=(now_t < escape_until),
                scored=sum(scored),
                cur_stuck=cur_stuck,
                deferred_n=len(deferred),
                phase='run',
            )

            # 状态日志
            if now_t - last_status_t >= 1.0:
                last_status_t = now_t
                path_len = len(cur_path) if cur_path else 0
                node.get_logger().info(
                    f"t={now_t-start_time:5.1f} "
                    f"pos=({odom_x:+.2f},{odom_y:+.2f}) yaw={math.degrees(odom_yaw):+6.1f} "
                    f"-> cell{cur_idx+1}({target_x:+.1f},{target_y:+.1f}) "
                    f"path={path_len} lh=({lh_x:+.1f},{lh_y:+.1f}) "
                    f"v={v:+.2f} w={w:+.2f} d_f={d_front:.2f}")

    except KeyboardInterrupt:
        node.get_logger().info("中断, 停车.")
        dbg.event('INTERRUPT')
    finally:
        publish_cmd(cmd_pub, node, 0.0, 0.0)
        time.sleep(0.1)
        try:
            dbg.close(summary={
                'scored_total': sum(scored),
                'scored_cells': ','.join(str(i + 1) for i, s in enumerate(scored) if s),
                'deferred_cells': ','.join(str(i + 1) for i in sorted(deferred)),
                'elapsed_s': f'{time.time() - start_time:.2f}',
            })
            node.get_logger().info(f"调试日志写入: {dbg.dir}")
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
