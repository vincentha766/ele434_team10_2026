#!/usr/bin/env python3
"""
Subscribe to /map, build cost field, publish debug topics for RViz.

Topics published:
  /debug/blocked    — OccupancyGrid (hard inflation layer)
  /debug/soft_cost  — OccupancyGrid (soft cost layer, scaled 0-100)

Usage:
  ros2 run ele434_team10_2026 test_cost_field.py
"""

import rclpy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from ele434_team10_2026_modules.cost_field import cost_field

HARD_R = 5
SOFT_R = 9


def main():
    rclpy.init()
    node = rclpy.create_node('test_cost_field')

    pub_blocked = node.create_publisher(OccupancyGrid, '/debug/blocked', 1)
    pub_soft = node.create_publisher(OccupancyGrid, '/debug/soft_cost', 1)

    def map_cb(msg):
        w = msg.info.width
        h = msg.info.height
        grid = np.array(msg.data, dtype=np.int8).reshape(h, w)

        blocked, extra = cost_field(grid, HARD_R, SOFT_R)

        n_occ = int(np.sum(grid >= 50))
        n_blocked = int(np.sum(blocked))
        n_soft = int(np.sum(extra > 0))
        node.get_logger().info(
            f"map {w}x{h} | occ={n_occ} blocked={n_blocked} soft={n_soft}")

        # Publish blocked layer
        out_blocked = OccupancyGrid()
        out_blocked.header = msg.header
        out_blocked.info = msg.info
        out_blocked.data = (blocked.astype(np.int8) * 100).flatten().tolist()
        pub_blocked.publish(out_blocked)

        # Publish soft cost layer (scale to 0-100)
        max_extra = extra.max() if extra.max() > 0 else 1.0
        soft_scaled = (extra / max_extra * 100).astype(np.int8)
        out_soft = OccupancyGrid()
        out_soft.header = msg.header
        out_soft.info = msg.info
        out_soft.data = soft_scaled.flatten().tolist()
        pub_soft.publish(out_soft)

    node.create_subscription(OccupancyGrid, '/map', map_cb, 1)
    node.get_logger().info(
        f"test_cost_field: HARD_R={HARD_R} SOFT_R={SOFT_R}, "
        f"publishing /debug/blocked and /debug/soft_cost")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
