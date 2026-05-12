#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan


class Explore(Node):
    def __init__(self):
        super().__init__('explore')

        self.publisher_ = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        self.front_distance = float('inf')
        self.left_distance = float('inf')
        self.right_distance = float('inf')

        self.safe_distance = 0.70

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info('Explore node started.')

    def get_valid_min(self, data, range_min, range_max):
        valid = [r for r in data if range_min < r < range_max]
        if valid:
            return min(valid)
        return float('inf')

    def scan_callback(self, msg):
        ranges = msg.ranges

        front_arc = list(ranges[:20]) + list(ranges[-20:])
        left_arc = list(ranges[20:60])
        right_arc = list(ranges[-60:-20])

        self.front_distance = self.get_valid_min(
            front_arc, msg.range_min, msg.range_max
        )
        self.left_distance = self.get_valid_min(
            left_arc, msg.range_min, msg.range_max
        )
        self.right_distance = self.get_valid_min(
            right_arc, msg.range_min, msg.range_max
        )

    def timer_callback(self):
        msg = TwistStamped()

        if self.front_distance > self.safe_distance:
            msg.twist.linear.x = 0.10
            msg.twist.angular.z = 0.0
            state = 'FORWARD'
        else:
            msg.twist.linear.x = 0.0

            if self.left_distance > self.right_distance:
                msg.twist.angular.z = 0.7
                state = 'TURN_LEFT'
            else:
                msg.twist.angular.z = -0.7
                state = 'TURN_RIGHT'

        self.publisher_.publish(msg)

        self.get_logger().info(
            f'State: {state} | '
            f'Front: {self.front_distance:.2f} m | '
            f'Left: {self.left_distance:.2f} m | '
            f'Right: {self.right_distance:.2f} m',
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = Explore()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_msg = TwistStamped()
        node.publisher_.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()