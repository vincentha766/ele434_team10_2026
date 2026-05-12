import math

from ele434_team10_2026_modules.nav_math import wrap_to_pi


def find_lookahead_point(path, robot_x, robot_y, lookahead):
    """Return the first point on path farther than `lookahead` from the robot."""
    if not path:
        return None, None
    for px, py in path:
        if math.hypot(px - robot_x, py - robot_y) > lookahead:
            return px, py
    return path[-1][0], path[-1][1]


def pure_pursuit(robot_x, robot_y, robot_yaw, path, target_x, target_y,
                 lookahead, v_max, w_max, k_yaw, yaw_hard):
    """
    Pure pursuit controller.

    Returns (v, w) velocity commands.
    Does NOT apply reactive safety — that is layered on top by the caller.
    """
    if path:
        lh_x, lh_y = find_lookahead_point(path, robot_x, robot_y, lookahead)
    else:
        lh_x, lh_y = target_x, target_y

    target_yaw = math.atan2(lh_y - robot_y, lh_x - robot_x)
    yaw_err = wrap_to_pi(target_yaw - robot_yaw)

    if abs(yaw_err) > yaw_hard:
        v = 0.0
        w = max(-w_max, min(w_max, k_yaw * yaw_err))
    else:
        v = v_max * math.cos(yaw_err)
        w = max(-w_max, min(w_max, k_yaw * yaw_err))

    return v, w
