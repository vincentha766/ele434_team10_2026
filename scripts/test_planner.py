#!/usr/bin/env python3
"""
Subscribe to /map + TF, plan a path to a goal, publish as nav_msgs/Path.

Topics published:
  /debug/planned_path — Path (visualize in RViz)

Parameters:
  goal_x (float, default 1.5)
  goal_y (float, default 1.5)

Usage:
  ros2 run ele434_team10_2026 test_planner.py
  ros2 run ele434_team10_2026 test_planner.py --ros-args -p goal_x:=-1.5 -p goal_y:=1.5
"""

import math
import time
import rclpy
import tf2_ros
import numpy as np
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from ele434_team10_2026_modules.path_planner import plan_path
from ele434_team10_2026_modules.nav_math import yaw_from_quaternion

HARD_R = 5
SOFT_R = 9
REPLAN_INTERVAL = 2.0


def main():
    rclpy.init()
    node = rclpy.create_node('test_planner')

    node.declare_parameter('goal_x', 1.5)
    node.declare_parameter('goal_y', 1.5)
    goal_x = node.get_parameter('goal_x').value
    goal_y = node.get_parameter('goal_y').value

    path_pub = node.create_publisher(Path, '/debug/planned_path', 1)

    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, node)

    map_grid = None
    map_info = None

    def map_cb(msg):
        nonlocal map_grid, map_info
        map_grid = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        map_info = msg.info

    node.create_subscription(OccupancyGrid, '/map', map_cb, 1)
    node.get_logger().info(
        f"test_planner: goal=({goal_x}, {goal_y}), "
        f"publishing /debug/planned_path every {REPLAN_INTERVAL}s")

    last_plan_t = 0.0

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.time()
            if now - last_plan_t < REPLAN_INTERVAL:
                continue
            if map_grid is None or map_info is None:
                continue

            try:
                t = tf_buffer.lookup_transform(
                    'map', 'base_footprint', rclpy.time.Time(),
                    timeout=Duration(seconds=0.0))
                rx = t.transform.translation.x
                ry = t.transform.translation.y
            except Exception:
                continue

            last_plan_t = now

            pts, _ = plan_path(
                (rx, ry), (goal_x, goal_y),
                map_grid, map_info.resolution,
                map_info.origin.position.x, map_info.origin.position.y,
                hard_r=HARD_R, soft_r=SOFT_R)

            if pts is None:
                node.get_logger().warn(
                    f"A* failed from ({rx:.2f},{ry:.2f}) "
                    f"to ({goal_x},{goal_y})")
                continue

            path_msg = Path()
            path_msg.header.stamp = node.get_clock().now().to_msg()
            path_msg.header.frame_id = 'map'
            for px, py in pts:
                ps = PoseStamped()
                ps.header = path_msg.header
                ps.pose.position.x = px
                ps.pose.position.y = py
                path_msg.poses.append(ps)
            path_pub.publish(path_msg)

            d = math.hypot(goal_x - rx, goal_y - ry)
            node.get_logger().info(
                f"pos=({rx:+.2f},{ry:+.2f}) -> goal=({goal_x},{goal_y}) "
                f"dist={d:.2f} waypoints={len(pts)}")

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
