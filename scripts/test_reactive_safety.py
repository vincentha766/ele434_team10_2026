#!/usr/bin/env python3
"""
Subscribe to /scan, continuously log reactive safety decisions.

Shows what apply_safety would do if the robot were commanding V_MAX forward.
Useful for verifying that safety thresholds trigger at the right distances.

Usage:
  ros2 run ele434_team10_2026 test_reactive_safety.py
"""

import math
import rclpy
from sensor_msgs.msg import LaserScan
from ele434_team10_2026_modules.reactive_safety import sector_min, apply_safety
from ele434_team10_2026_modules.debug_logger import RunLogger

V_MAX = 0.26
SAFE_FRONT = 0.28
SLOW_FRONT = 0.50
SAFE_SIDE = 0.22


def main():
    rclpy.init()
    node = rclpy.create_node('test_reactive_safety')

    dbg = RunLogger(
        run_name='test_reactive_safety',
        trace_fields=['d_front', 'd_fl', 'd_fr', 'v_out', 'v_scale', 'braked'],
        params={
            'V_MAX': V_MAX, 'SAFE_FRONT': SAFE_FRONT,
            'SLOW_FRONT': SLOW_FRONT, 'SAFE_SIDE': SAFE_SIDE,
        },
    )
    node.get_logger().info(f"调试日志目录: {dbg.dir}")

    last_braked = [False]

    def scan_cb(msg):
        ranges = []
        for r in msg.ranges:
            if math.isnan(r) or math.isinf(r) or r < 0.05:
                ranges.append(3.5)
            else:
                ranges.append(r)

        if len(ranges) < 36:
            return

        d_front = sector_min(ranges, -20, 20)
        d_fl = sector_min(ranges, 20, 60)
        d_fr = sector_min(ranges, -60, -20)

        v_out, _, braked = apply_safety(
            ranges, V_MAX, 0.0,
            safe_front=SAFE_FRONT, slow_front=SLOW_FRONT,
            safe_side=SAFE_SIDE)

        status = "BRAKE" if braked else (
            f"v_scale={v_out / V_MAX:.0%}" if v_out < V_MAX else "clear")

        node.get_logger().info(
            f"d_front={d_front:.2f} d_fl={d_fl:.2f} d_fr={d_fr:.2f} | "
            f"v={v_out:.3f}/{V_MAX} {status}")

        # 只在 brake 状态切换时记 event, 避免事件 log 被刷屏
        if braked != last_braked[0]:
            dbg.event(
                'BRAKE_ON' if braked else 'BRAKE_OFF',
                f'd_front={d_front:.2f} d_fl={d_fl:.2f} d_fr={d_fr:.2f}')
            # brake 触发时把整圈 lidar 留个快照 (排查 false positive)
            if braked:
                dbg.snapshot('brake_lidar',
                             [f'{i} {r:.3f}' for i, r in enumerate(ranges)])
            last_braked[0] = braked

        dbg.trace(d_front=d_front, d_fl=d_fl, d_fr=d_fr,
                  v_out=v_out, v_scale=v_out / V_MAX, braked=braked)

    node.create_subscription(LaserScan, '/scan', scan_cb, 10)
    node.get_logger().info(
        f"test_reactive_safety: SAFE_FRONT={SAFE_FRONT} "
        f"SLOW_FRONT={SLOW_FRONT} SAFE_SIDE={SAFE_SIDE}")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        dbg.event('INTERRUPT')
    finally:
        try:
            dbg.close()
            node.get_logger().info(f"调试日志写入: {dbg.dir}")
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
