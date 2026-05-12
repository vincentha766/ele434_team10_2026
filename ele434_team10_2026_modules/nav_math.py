import math


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def world_to_grid(x, y, origin_x, origin_y, resolution):
    c = int((x - origin_x) / resolution)
    r = int((y - origin_y) / resolution)
    return c, r


def grid_to_world(c, r, origin_x, origin_y, resolution):
    return (origin_x + (c + 0.5) * resolution,
            origin_y + (r + 0.5) * resolution)


def yaw_from_quaternion(qx, qy, qz, qw):
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)
