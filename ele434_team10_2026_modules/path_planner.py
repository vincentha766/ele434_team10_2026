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


def find_nearest_free(blocked, c, r, max_radius):
    """Return the nearest non-blocked cell to (c, r) within Chebyshev max_radius.

    Used when the planning goal lies inside an obstacle's inflation zone —
    instead of carving a tunnel through, we snap the goal to the closest
    legal cell. Returns None if no free cell exists within range."""
    h, w = blocked.shape
    if 0 <= c < w and 0 <= r < h and not blocked[r, c]:
        return (c, r)
    best = None
    best_d2 = 10 ** 9
    for radius in range(1, max_radius + 1):
        if best is not None:
            break
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dc), abs(dr)) != radius:
                    continue
                nc, nr = c + dc, r + dr
                if 0 <= nc < w and 0 <= nr < h and not blocked[nr, nc]:
                    d2 = dc * dc + dr * dr
                    if d2 < best_d2:
                        best_d2 = d2
                        best = (nc, nr)
    return best


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
              hard_r=4, soft_r=8, goal_snap_radius=10):
    """
    A* with soft inflation.

    If the goal cell falls inside an obstacle's hard zone (which happens when
    a beacon sits at the score-cell center), we snap the goal to the closest
    free cell within Chebyshev `goal_snap_radius` cells. We DO NOT unblock
    around the goal — that previously carved a tunnel through the obstacle
    and made the controller drive straight into it.

    Returns (waypoints, hard_r) or (None, None) on failure.
    """
    if grid is None:
        return None, None
    s = world_to_grid(start_xy[0], start_xy[1], origin_x, origin_y, resolution)
    g = world_to_grid(goal_xy[0], goal_xy[1], origin_x, origin_y, resolution)

    # Progressive fallback — keep a safety floor of 5 cells (25cm) inflation,
    # which is the minimum that prevents the planner from carving a path
    # straight up against an obstacle (robot radius 21cm).
    h_seq = []
    h = hard_r
    while h >= 5:
        h_seq.append(h)
        h -= 1
    if not h_seq or h_seq[-1] != 5:
        h_seq.append(5)

    for h_r in h_seq:
        blocked, extra = cost_field(grid, h_r, soft_r)
        unblock(blocked, s[0], s[1], 2)  # robot near a wall is OK to escape
        g_eff = find_nearest_free(blocked, g[0], g[1], goal_snap_radius)
        if g_eff is None:
            continue
        cells = astar(blocked, s, g_eff, extra_cost=extra)
        if cells is not None:
            pts = [grid_to_world(c, r, origin_x, origin_y, resolution)
                   for c, r in cells]
            if len(pts) > 6:
                step = max(1, len(pts) // 5)
                ds = pts[::step]
                if ds[-1] != pts[-1]:
                    ds.append(pts[-1])
                pts = ds
            return pts, h_r
    return None, None
