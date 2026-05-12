#!/usr/bin/env python3
"""Offline unit tests for nav_math — no ROS or hardware needed."""

import math
import sys

from ele434_team10_2026_modules.nav_math import (
    wrap_to_pi, world_to_grid, grid_to_world, yaw_from_quaternion,
)


def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        print(f"FAIL: {a} != {b} (tol={tol}) {msg}")
        sys.exit(1)


def test_wrap_to_pi():
    assert_close(wrap_to_pi(0.0), 0.0)
    assert_close(wrap_to_pi(math.pi), -math.pi, tol=1e-5)
    assert_close(wrap_to_pi(-math.pi), -math.pi, tol=1e-5)
    assert_close(wrap_to_pi(3.5), 3.5 - 2 * math.pi)
    assert_close(wrap_to_pi(-3.5), -3.5 + 2 * math.pi)
    assert_close(wrap_to_pi(2 * math.pi), 0.0, tol=1e-5)
    print("  wrap_to_pi: OK")


def test_grid_roundtrip():
    ox, oy, res = -2.0, -2.0, 0.05
    for wx, wy in [(0.0, 0.0), (1.5, -1.5), (-1.9, 1.9)]:
        c, r = world_to_grid(wx, wy, ox, oy, res)
        bx, by = grid_to_world(c, r, ox, oy, res)
        assert_close(bx, wx, tol=res, msg=f"x roundtrip ({wx},{wy})")
        assert_close(by, wy, tol=res, msg=f"y roundtrip ({wx},{wy})")
    print("  grid roundtrip: OK")


def test_yaw_from_quaternion():
    # Identity quaternion -> yaw = 0
    assert_close(yaw_from_quaternion(0, 0, 0, 1), 0.0)
    # 90 deg yaw -> qz = sin(45deg), qw = cos(45deg)
    assert_close(yaw_from_quaternion(0, 0, math.sin(math.pi / 4),
                                     math.cos(math.pi / 4)),
                 math.pi / 2, tol=1e-5)
    # 180 deg yaw
    assert_close(yaw_from_quaternion(0, 0, 1, 0), math.pi, tol=1e-5)
    print("  yaw_from_quaternion: OK")


if __name__ == '__main__':
    print("test_nav_math")
    test_wrap_to_pi()
    test_grid_roundtrip()
    test_yaw_from_quaternion()
    print("ALL PASSED")
