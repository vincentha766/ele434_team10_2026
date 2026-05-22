#!/usr/bin/env python3

import rclpy
import tf2_ros
import numpy as np
from rclpy.duration import Duration
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
import math
import os
import signal
import subprocess
import time

from ele434_team10_2026_modules.core import (
    wrap_to_pi, yaw_from_quaternion, plan_path,
    find_lookahead_point, pure_pursuit,
    sector_min, apply_safety, RunLogger,
)


def publish_cmd(pub, node, v, w):
    cmd = TwistStamped()
    cmd.header.stamp = node.get_clock().now().to_msg()
    cmd.header.frame_id = 'base_link'
    cmd.twist.linear.x = float(v)
    cmd.twist.angular.z = float(w)
    pub.publish(cmd)


# A* navigator: greedy nearest cell + soft-inflation A* + pure pursuit + reactive safety
def main(args=None):
    rclpy.init(args=args, signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    node = rclpy.create_node('astar_navigator')
    cmd_pub = node.create_publisher(TwistStamped, '/cmd_vel', 10)

    stop_requested = {'v': False}

    def _sigint_handler(signum, frame):
        if stop_requested['v']:
            return
        stop_requested['v'] = True
        try:
            cmd = TwistStamped()
            cmd.header.stamp = node.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            cmd_pub.publish(cmd)
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=0.02)
                cmd_pub.publish(cmd)
        except Exception:
            pass
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, node)

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
        state['lidar_ranges'] = [
            3.5 if (math.isnan(r) or math.isinf(r) or r < 0.05) else r
            for r in msg.ranges]
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

    V_MAX = 0.26;  W_MAX = 1.82;  K_YAW = 2.4;  YAW_HARD = 1.2
    SCORE_TOL = 0.29;  REACH_TOL = 0.22;  MAX_RUN_T = 240.0
    HARD_R = 5;  SOFT_R = 9;  REPLAN_PERIOD = 1.0
    SAFE_FRONT = 0.28;  SLOW_FRONT = 0.50;  SAFE_SIDE = 0.23
    STUCK_WINDOW = 3.0;  STUCK_DIST = 0.08;  ESCAPE_DUR = 1.0;  DEFER_STUCK = 3
    LOOKAHEAD = 0.29;  WARMUP_T = 4.0

    score_cells = [
        (-1.5, 1.5), (-0.5, 1.5), (0.5, 1.5), (1.5, 1.5),
        (1.5, 0.5), (1.5, -0.5), (1.5, -1.5),
        (0.5, -1.5), (-0.5, -1.5), (-1.5, -1.5),
        (-1.5, -0.5), (-1.5, 0.5),
    ]
    scored = [False] * 12
    deferred = set()
    cur_idx = 0;  cur_stuck = 0
    cur_path = None;  last_replan_t = 0.0
    pose_hist = [];  escape_until = 0.0;  escape_dir = 1;  last_status_t = 0.0
    dbg = RunLogger(
        run_name='work',
        trace_fields=[
            'x', 'y', 'yaw_deg',
            'cell', 'tgt_x', 'tgt_y',
            'lh_x', 'lh_y', 'yaw_err_deg', 'd_goal',
            'v_cmd', 'w_cmd',
            'd_front', 'd_fl', 'd_fr',
            'path_len', 'braked', 'in_escape',
            'scored', 'cur_stuck', 'deferred_n',
            'phase',
        ],
        params={
            'V_MAX': V_MAX, 'W_MAX': W_MAX, 'K_YAW': K_YAW,
            'YAW_HARD': YAW_HARD, 'SCORE_TOL': SCORE_TOL,
            'REACH_TOL': REACH_TOL, 'MAX_RUN_T': MAX_RUN_T,
            'HARD_R': HARD_R, 'SOFT_R': SOFT_R,
            'REPLAN_PERIOD': REPLAN_PERIOD,
            'SAFE_FRONT': SAFE_FRONT, 'SLOW_FRONT': SLOW_FRONT,
            'SAFE_SIDE': SAFE_SIDE,
            'STUCK_WINDOW': STUCK_WINDOW, 'STUCK_DIST': STUCK_DIST,
            'ESCAPE_DUR': ESCAPE_DUR, 'DEFER_STUCK': DEFER_STUCK,
            'LOOKAHEAD': LOOKAHEAD, 'WARMUP_T': WARMUP_T,
        },
    )
    node.get_logger().info(f"Log dir: {dbg.dir}")
    dbg.event('INIT', f'log_dir={dbg.dir}')

    node.get_logger().info("A* node started, waiting for sensors + map + SLAM ...")

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
                    f"Ready: pose=({tx:+.2f},{ty:+.2f}), "
                    f"map={state['map_w']}x{state['map_h']}")
                break
        except Exception:
            pass

    node.get_logger().info(f"WARMUP: spinning {WARMUP_T}s for SLAM ...")
    dbg.event('WARMUP_START', f'duration={WARMUP_T}s')
    warmup_t0 = time.time()
    while rclpy.ok() and time.time() - warmup_t0 < WARMUP_T:
        rclpy.spin_once(node, timeout_sec=0.05)
        publish_cmd(cmd_pub, node, 0.0, 1.6)
    publish_cmd(cmd_pub, node, 0.0, 0.0)
    dbg.event('WARMUP_DONE',
              f'pose=({state["odom_x"]:+.2f},{state["odom_y"]:+.2f}) '
              f'map_ready={state["map_ready"]}')
    node.get_logger().info("WARMUP done, starting task.")

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

            for i, (cx, cy) in enumerate(score_cells):
                if (not scored[i]
                        and abs(odom_x - cx) <= SCORE_TOL
                        and abs(odom_y - cy) <= SCORE_TOL):
                    scored[i] = True
                    elapsed = now_t - start_time
                    node.get_logger().info(
                        f"[SCORED t={elapsed:5.1f}s] cell {i + 1} "
                        f"({cx:+.1f},{cy:+.1f})  {sum(scored)}/12")
                    dbg.event(
                        'SCORED',
                        f'cell={i + 1} at=({cx:+.1f},{cy:+.1f}) '
                        f'elapsed={elapsed:.1f}s total={sum(scored)}/12')
                    if i == cur_idx:
                        cur_path = None

            if all(scored):
                node.get_logger().info(
                    f"All 12 cells scored t={now_t-start_time:.1f}s, stopping.")
                dbg.event('ALL_DONE', f'elapsed={now_t-start_time:.1f}s')
                break
            if now_t - start_time > MAX_RUN_T:
                node.get_logger().warn(
                    f"Timeout, scored {sum(scored)}/12, stopping.")
                dbg.event('TIMEOUT',
                          f'elapsed={now_t-start_time:.1f}s '
                          f'scored={sum(scored)}/12')
                break

            prev_idx = cur_idx
            unscored = [i for i in range(12) if not scored[i]]
            if unscored:
                avail = [i for i in unscored if i not in deferred]
                if not avail:
                    avail = unscored
                    nearest = min(avail, key=lambda i: math.hypot(
                        score_cells[i][0] - odom_x,
                        score_cells[i][1] - odom_y))
                    if nearest in deferred:
                        node.get_logger().info(
                            f"All deferred, unblocking nearest cell {nearest+1}.")
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
                scored[cur_idx] = True
                continue

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
                        f"A* failed cell{cur_idx+1}, falling back to straight line")
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

            if cur_path is None or len(cur_path) == 0:
                lh_x, lh_y = target_x, target_y
            else:
                lh_x, lh_y = find_lookahead_point(
                    cur_path, odom_x, odom_y, LOOKAHEAD)

            target_yaw = math.atan2(lh_y - odom_y, lh_x - odom_x)
            yaw_err = wrap_to_pi(target_yaw - odom_yaw)

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
                    cur_path = None
                    node.get_logger().warn(
                        f"[STUCK #{cur_stuck}] reversing {ESCAPE_DUR}s "
                        f"@({odom_x:+.2f},{odom_y:+.2f})")
                    dbg.event(
                        'STUCK',
                        f'cell={cur_idx+1} n={cur_stuck} '
                        f'pose=({odom_x:+.2f},{odom_y:+.2f}) '
                        f'yaw={math.degrees(odom_yaw):+.1f} d_t={d_t:.3f}')
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
                            f"[DEFER] cell {cur_idx+1} stuck {cur_stuck}x, skipping.")
                        dbg.event('DEFER', f'cell={cur_idx+1} after_stuck={cur_stuck}')
                        deferred.add(cur_idx)
                        cur_stuck = 0
                    continue

            v, w = pure_pursuit(
                odom_x, odom_y, odom_yaw, cur_path, target_x, target_y,
                LOOKAHEAD, V_MAX, W_MAX, K_YAW, YAW_HARD)

            v_pp, w_pp = v, w
            v, w, braked = apply_safety(
                lidar_ranges, v, w,
                safe_front=SAFE_FRONT, slow_front=SLOW_FRONT,
                safe_side=SAFE_SIDE)
            if braked:
                cur_path = None
                dbg.event(
                    'BRAKE',
                    f'cell={cur_idx+1} pose=({odom_x:+.2f},{odom_y:+.2f}) '
                    f'v={v_pp:.2f}->{v:.2f} w={w_pp:.2f}->{w:.2f}')

            publish_cmd(cmd_pub, node, v, w)

            has_scan = len(lidar_ranges) >= 36
            d_front = sector_min(lidar_ranges, -20, 20) if has_scan else -1.0
            d_fl = sector_min(lidar_ranges, 20, 60) if has_scan else -1.0
            d_fr = sector_min(lidar_ranges, -60, -20) if has_scan else -1.0
            dbg.trace(
                x=odom_x, y=odom_y, yaw_deg=math.degrees(odom_yaw),
                cell=cur_idx + 1, tgt_x=target_x, tgt_y=target_y,
                lh_x=lh_x, lh_y=lh_y,
                yaw_err_deg=math.degrees(yaw_err), d_goal=d_goal,
                v_cmd=v, w_cmd=w,
                d_front=d_front, d_fl=d_fl, d_fr=d_fr,
                path_len=(len(cur_path) if cur_path else 0),
                braked=braked,
                in_escape=(now_t < escape_until),
                scored=sum(scored),
                cur_stuck=cur_stuck,
                deferred_n=len(deferred),
                phase='run',
            )

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
        node.get_logger().info("Interrupted, stopping.")
        dbg.event('INTERRUPT')
    finally:
        publish_cmd(cmd_pub, node, 0.0, 0.0)
        time.sleep(0.1)
        try:
            map_dir = os.path.expanduser('~/ros2_ws/src/ele434_team10_2026/maps')
            os.makedirs(map_dir, exist_ok=True)
            map_path = os.path.join(map_dir, 'explore_map')
            result = subprocess.run(
                ['ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                 '-f', map_path, '--fmt', 'png'],
                capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                node.get_logger().info(f"Map saved: {map_path}.png")
            else:
                node.get_logger().warn(f"Map save failed: {result.stderr.strip()}")
        except Exception as e:
            node.get_logger().warn(f"Map save error: {e}")
        try:
            dbg.close(summary={
                'scored_total': sum(scored),
                'scored_cells': ','.join(str(i + 1) for i, s in enumerate(scored) if s),
                'deferred_cells': ','.join(str(i + 1) for i in sorted(deferred)),
                'elapsed_s': f'{time.time() - start_time:.2f}',
            })
            node.get_logger().info(f"Log saved: {dbg.dir}")
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
