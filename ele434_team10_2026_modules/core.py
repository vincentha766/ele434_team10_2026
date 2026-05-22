import csv
import math
import heapq
import os
import time
from datetime import datetime

import numpy as np


# --nav_math --

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


# --cost_field --

def cost_field(grid, hard_r, soft_r):
    """
    Returns (blocked, extra_cost):
      blocked:    bool[h,w] --True within hard_r cells of any obstacle
      extra_cost: float[h,w] --gradual cost increase between hard_r and soft_r
    """
    h, w = grid.shape
    occ = (grid >= 50)
    dist = np.full((h, w), 99, dtype=np.int8)
    dist[occ] = 0
    cur = occ.copy()
    for d in range(1, soft_r + 1):
        new = cur.copy()
        new[:-1] |= cur[1:]
        new[1:]  |= cur[:-1]
        new[:, :-1] |= cur[:, 1:]
        new[:, 1:]  |= cur[:, :-1]
        newly = new & ~cur
        dist[newly] = d
        cur = new
    blocked = dist <= hard_r
    extra = np.zeros((h, w), dtype=np.float32)
    for d in range(hard_r + 1, soft_r + 1):
        extra[dist == d] = (soft_r - d + 1) * 0.6
    return blocked, extra


# --path_planner --

def _unblock(blocked, c, r, radius):
    h, w = blocked.shape
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nc, nr = c + dc, r + dr
            if 0 <= nc < w and 0 <= nr < h:
                blocked[nr, nc] = False


def _astar(blocked, start, goal, extra_cost=None):
    h, w = blocked.shape
    if not (0 <= start[0] < w and 0 <= start[1] < h):
        return None
    if not (0 <= goal[0] < w and 0 <= goal[1] < h):
        return None
    if blocked[goal[1], goal[0]]:
        return None
    if start == goal:
        return [start]
    DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1)]

    def hd(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_h = [(hd(start, goal), 0.0, start)]
    came = {start: None}
    g = {start: 0.0}
    while open_h:
        _, gc, cur = heapq.heappop(open_h)
        if gc > g.get(cur, float('inf')):
            continue
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return list(reversed(path))
        for dc, dr in DIRS:
            nc, nr = cur[0] + dc, cur[1] + dr
            if nc < 0 or nc >= w or nr < 0 or nr >= h:
                continue
            if blocked[nr, nc]:
                continue
            step = 1.41421 if (dc and dr) else 1.0
            if extra_cost is not None:
                step += float(extra_cost[nr, nc])
            ng = gc + step
            n = (nc, nr)
            if ng < g.get(n, float('inf')):
                g[n] = ng
                came[n] = cur
                heapq.heappush(open_h, (ng + hd(n, goal), ng, n))
    return None


def plan_path(start_xy, goal_xy, grid, resolution, origin_x, origin_y,
              hard_r=4, soft_r=8):
    if grid is None:
        return None, None
    s = world_to_grid(start_xy[0], start_xy[1], origin_x, origin_y, resolution)
    g = world_to_grid(goal_xy[0], goal_xy[1], origin_x, origin_y, resolution)
    blocked, extra = cost_field(grid, hard_r, soft_r)
    _unblock(blocked, s[0], s[1], 2)
    _unblock(blocked, g[0], g[1], 2)
    cells = _astar(blocked, s, g, extra_cost=extra)
    if cells is None:
        blocked2, _ = cost_field(grid, max(1, hard_r // 2), soft_r)
        _unblock(blocked2, s[0], s[1], 2)
        _unblock(blocked2, g[0], g[1], 2)
        cells = _astar(blocked2, s, g, extra_cost=extra)
        if cells is None:
            return None, None
    pts = [grid_to_world(c, r, origin_x, origin_y, resolution)
           for c, r in cells]
    if len(pts) > 6:
        step = max(1, len(pts) // 5)
        ds = pts[::step]
        if ds[-1] != pts[-1]:
            ds.append(pts[-1])
        pts = ds
    return pts, hard_r


# --motion_control --

def find_lookahead_point(path, robot_x, robot_y, lookahead):
    if not path:
        return None, None
    for px, py in path:
        if math.hypot(px - robot_x, py - robot_y) > lookahead:
            return px, py
    return path[-1][0], path[-1][1]


def pure_pursuit(robot_x, robot_y, robot_yaw, path, target_x, target_y,
                 lookahead, v_max, w_max, k_yaw, yaw_hard):
    if path:
        lh_x, lh_y = find_lookahead_point(path, robot_x, robot_y, lookahead)
    else:
        lh_x, lh_y = target_x, target_y

    target_yaw = math.atan2(lh_y - robot_y, lh_x - robot_x)
    yaw_err = wrap_to_pi(target_yaw - robot_yaw)

    if abs(yaw_err) > yaw_hard:
        v = 0.0
    else:
        v = v_max * math.cos(yaw_err)
    w = max(-w_max, min(w_max, k_yaw * yaw_err))
    return v, w


# --reactive_safety --

def sector_min(ranges, a_deg, b_deg):
    n = len(ranges)
    a = int(round(a_deg * n / 360.0)) % n
    b = int(round(b_deg * n / 360.0)) % n
    if a <= b:
        return min(ranges[a:b + 1])
    return min(ranges[a:] + ranges[:b + 1])


def apply_safety(ranges, v, w, safe_front=0.28, slow_front=0.50,
                 safe_side=0.22):
    if len(ranges) < 36:
        return v, w, False

    d_front = sector_min(ranges, -20, 20)
    d_fl = sector_min(ranges, 20, 60)
    d_fr = sector_min(ranges, -60, -20)

    braked = False
    if d_front < safe_front:
        v = 0.0
        braked = True
    elif d_front < slow_front:
        factor = (d_front - safe_front) / (slow_front - safe_front)
        v *= max(0.3, factor)

    d_side = min(d_fl, d_fr)
    if d_side < safe_side:
        v *= 0.4

    return v, w, braked


# --debug_logger --

class RunLogger:
    def __init__(self, run_name, trace_fields, params=None, log_dir=None):
        base = log_dir or os.environ.get('EL434_LOG_DIR', '.tmp_logs')
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dir = os.path.join(base, f'{run_name}_{ts}')
        try:
            os.makedirs(self.dir, exist_ok=True)
        except Exception:
            self.dir = os.path.join('/tmp', f'{run_name}_{ts}')
            os.makedirs(self.dir, exist_ok=True)

        self.t0 = time.time()
        self._trace_fields = list(trace_fields)
        self._counters = {}
        self._snap_n = 0

        self._trace_f = open(os.path.join(self.dir, 'trace.csv'), 'w', newline='')
        self._trace_w = csv.writer(self._trace_f)
        self._trace_w.writerow(['t'] + self._trace_fields)

        self._events_f = open(os.path.join(self.dir, 'events.log'), 'w')
        self._events_f.write(f'# run={run_name} started={ts}\n')
        self._events_f.flush()

        if params:
            try:
                with open(os.path.join(self.dir, 'params.txt'), 'w') as f:
                    for k, v in params.items():
                        f.write(f'{k}={v}\n')
            except Exception:
                pass

    def trace(self, **kwargs):
        try:
            row = [f'{time.time() - self.t0:.3f}']
            for k in self._trace_fields:
                v = kwargs.get(k, '')
                if isinstance(v, float):
                    row.append(f'{v:.4f}')
                elif isinstance(v, bool):
                    row.append('1' if v else '0')
                else:
                    row.append(v)
            self._trace_w.writerow(row)
            self._trace_f.flush()
        except Exception:
            pass

    def event(self, kind, message=''):
        try:
            t = time.time() - self.t0
            self._events_f.write(f'{t:7.2f}  {kind:<14}  {message}\n')
            self._events_f.flush()
            self._counters[kind] = self._counters.get(kind, 0) + 1
        except Exception:
            pass

    def snapshot(self, tag, content):
        try:
            self._snap_n += 1
            name = f'snap_{self._snap_n:03d}_{tag}.txt'
            path = os.path.join(self.dir, name)
            with open(path, 'w') as f:
                t = time.time() - self.t0
                f.write(f'# t={t:.2f} tag={tag}\n')
                if isinstance(content, str):
                    f.write(content)
                    if not content.endswith('\n'):
                        f.write('\n')
                else:
                    for line in content:
                        f.write(str(line) + '\n')
        except Exception:
            pass

    def close(self, summary=None):
        try:
            with open(os.path.join(self.dir, 'summary.txt'), 'w') as f:
                f.write(f'duration_s={time.time() - self.t0:.2f}\n')
                for k, v in self._counters.items():
                    f.write(f'event_{k}={v}\n')
                if summary:
                    for k, v in summary.items():
                        f.write(f'{k}={v}\n')
        except Exception:
            pass
        for fh in (getattr(self, '_trace_f', None), getattr(self, '_events_f', None)):
            try:
                if fh:
                    fh.close()
            except Exception:
                pass
