#!/usr/bin/env python3
"""
Drive to a single point using pure pursuit only (no A*, no safety layer).
Useful for verifying the motion controller works correctly on hardware.

Parameters:
  goal_x (float, default 0.5)
  goal_y (float, default 0.0)

Usage:
  ros2 run ele434_team10_2026 test_motion.py
  ros2 run ele434_team10_2026 test_motion.py --ros-args -p goal_x:=1.0 -p goal_y:=0.5
"""

import math
import time
import rclpy
import tf2_ros
from rclpy.duration import Duration
from geometry_msgs.msg import TwistStamped
from ele434_team10_2026_modules.motion_control import pure_pursuit
from ele434_team10_2026_modules.nav_math import yaw_from_quaternion
from ele434_team10_2026_modules.debug_logger import RunLogger

V_MAX = 0.26
W_MAX = 1.82
K_YAW = 2.4
YAW_HARD = 1.2
LOOKAHEAD = 0.20
REACH_TOL = 0.15


def main():
    rclpy.init()
    node = rclpy.create_node('test_motion')

    node.declare_parameter('goal_x', 0.5)
    node.declare_parameter('goal_y', 0.0)
    goal_x = node.get_parameter('goal_x').value
    goal_y = node.get_parameter('goal_y').value

    cmd_pub = node.create_publisher(TwistStamped, '/cmd_vel', 10)
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, node)

    node.get_logger().info(
        f"test_motion: driving to ({goal_x}, {goal_y}) with pure pursuit, "
        f"LOOKAHEAD={LOOKAHEAD}")

    dbg = RunLogger(
        run_name='test_motion',
        trace_fields=['x', 'y', 'yaw_deg', 'dist', 'v_cmd', 'w_cmd'],
        params={
            'goal_x': goal_x, 'goal_y': goal_y,
            'V_MAX': V_MAX, 'W_MAX': W_MAX, 'K_YAW': K_YAW,
            'YAW_HARD': YAW_HARD, 'LOOKAHEAD': LOOKAHEAD,
            'REACH_TOL': REACH_TOL,
        },
    )
    node.get_logger().info(f"调试日志目录: {dbg.dir}")
    dbg.event('START', f'goal=({goal_x},{goal_y})')

    last_log_t = 0.0
    t_start = time.time()

    def publish_cmd(v, w):
        cmd = TwistStamped()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = float(v)
        cmd.twist.angular.z = float(w)
        cmd_pub.publish(cmd)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

            try:
                t = tf_buffer.lookup_transform(
                    'map', 'base_footprint', rclpy.time.Time(),
                    timeout=Duration(seconds=0.0))
                rx = t.transform.translation.x
                ry = t.transform.translation.y
                q = t.transform.rotation
                ryaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
            except Exception:
                continue

            d = math.hypot(goal_x - rx, goal_y - ry)
            if d < REACH_TOL:
                publish_cmd(0.0, 0.0)
                node.get_logger().info(
                    f"Reached goal ({goal_x},{goal_y}), dist={d:.3f}")
                dbg.event('REACHED', f'dist={d:.3f} elapsed={time.time()-t_start:.2f}s')
                break

            # Straight-line "path" — just the goal point
            path = [(goal_x, goal_y)]
            v, w = pure_pursuit(
                rx, ry, ryaw, path, goal_x, goal_y,
                LOOKAHEAD, V_MAX, W_MAX, K_YAW, YAW_HARD)
            publish_cmd(v, w)

            dbg.trace(x=rx, y=ry, yaw_deg=math.degrees(ryaw),
                      dist=d, v_cmd=v, w_cmd=w)

            now = time.time()
            if now - last_log_t >= 0.5:
                last_log_t = now
                node.get_logger().info(
                    f"pos=({rx:+.2f},{ry:+.2f}) yaw={math.degrees(ryaw):+.1f} "
                    f"dist={d:.2f} v={v:+.2f} w={w:+.2f}")

    except KeyboardInterrupt:
        dbg.event('INTERRUPT')
    finally:
        publish_cmd(0.0, 0.0)
        time.sleep(0.1)
        try:
            dbg.close(summary={'elapsed_s': f'{time.time()-t_start:.2f}'})
            node.get_logger().info(f"调试日志写入: {dbg.dir}")
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
