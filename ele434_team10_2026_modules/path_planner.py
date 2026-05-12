import math
import heapq

from ele434_team10_2026_modules.nav_math import world_to_grid, grid_to_world
from ele434_team10_2026_modules.cost_field import cost_field


def unblock(blocked, c, r, radius):
    """Clear blocked cells in a square around (c, r)."""
    h, w = blocked.shape
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nc, nr = c + dc, r + dr
            if 0 <= nc < w and 0 <= nr < h:
                blocked[nr, nc] = False


def astar(blocked, start, goal, extra_cost=None):
    """8-connected A* on a boolean grid with optional soft cost overlay."""
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
    """
    A* with soft inflation.
    Returns (waypoints, hard_r) or (None, None) on failure.
    """
    if grid is None:
        return None, None
    s = world_to_grid(start_xy[0], start_xy[1], origin_x, origin_y, resolution)
    g = world_to_grid(goal_xy[0], goal_xy[1], origin_x, origin_y, resolution)
    blocked, extra = cost_field(grid, hard_r, soft_r)
    unblock(blocked, s[0], s[1], 2)
    unblock(blocked, g[0], g[1], 2)
    cells = astar(blocked, s, g, extra_cost=extra)
    if cells is None:
        blocked2, _ = cost_field(grid, max(1, hard_r // 2), soft_r)
        unblock(blocked2, s[0], s[1], 2)
        unblock(blocked2, g[0], g[1], 2)
        cells = astar(blocked2, s, g, extra_cost=extra)
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
