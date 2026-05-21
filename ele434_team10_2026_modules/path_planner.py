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
              hard_r=4, soft_r=8, goal_snap_radius=10, hard_r_wall=None,
              detour_ratio_threshold=1.6):
    """
    A* with soft inflation, differential wall vs beacon膨胀, 自适应回退.

    膨胀分类:
      hard_r: 柱状障碍 (beacon) 的硬禁区半径 (cells)
      hard_r_wall: 长条墙的硬禁区半径; None 则与 hard_r 相同
        Walls 是平面只需 robot_radius 缓冲; beacons 是圆柱需 robot_r + beacon_r.
        默认让 walls 用更小膨胀, 避免 wall 旁通道挤窄.

    Goal snap: 终点贴障碍时 snap 到最近合法 cell, 不解封 (避免凿穿).

    自适应回退: 用主膨胀 (hard_r, hard_r_wall) 试规划, 若绕行 > 直线距离 *
    detour_ratio_threshold, 逐步降级膨胀直到绕行可接受; 都不行用最短一条.

    Returns (waypoints, used_hard_r) or (None, None) on failure.
    """
    if grid is None:
        return None, None
    s = world_to_grid(start_xy[0], start_xy[1], origin_x, origin_y, resolution)
    g = world_to_grid(goal_xy[0], goal_xy[1], origin_x, origin_y, resolution)

    if hard_r_wall is None:
        hard_r_wall = hard_r

    # Fallback: 只降 wall 膨胀, beacon 膨胀绝不降. wall 底线 6 (30cm = robot 21cm
    # + 9cm 跟踪余量). 低于此跟踪误差能让 robot 边进 wall 表面 → scrape.
    # 找不到安全路径时返回 None → work.py defer 这 cell (绝不直线兜底防撞).
    h_seq = [(hard_r, hard_r_wall)]
    if hard_r_wall > 6:
        h_seq.append((hard_r, 6))

    # 直线距离参考
    euclid_cells = math.hypot(s[0] - g[0], s[1] - g[1])

    best_short = None  # (cells_path, hr_beacon, hr_wall, ratio) for shortest-found
    for hb, hw in h_seq:
        blocked, extra = cost_field(
            grid, hb, soft_r, hard_r_wall=hw,
            origin_x=origin_x, origin_y=origin_y, resolution=resolution)
        unblock(blocked, s[0], s[1], 2)
        g_eff = find_nearest_free(blocked, g[0], g[1], goal_snap_radius)
        if g_eff is None:
            continue
        cells = astar(blocked, s, g_eff, extra_cost=extra)
        if cells is None:
            continue

        # 计算路径长度 (8 连通, 对角 1.41)
        path_len_cells = 0.0
        for i in range(1, len(cells)):
            dx = cells[i][0] - cells[i - 1][0]
            dy = cells[i][1] - cells[i - 1][1]
            path_len_cells += 1.41421 if (dx and dy) else 1.0
        ratio = path_len_cells / max(euclid_cells, 1.0)

        # 当前膨胀下绕路可接受 -> 用它 (优先大膨胀 = 更安全)
        if ratio <= detour_ratio_threshold:
            pts = [grid_to_world(c, r, origin_x, origin_y, resolution)
                   for c, r in cells]
            if len(pts) > 6:
                step = max(1, len(pts) // 5)
                ds = pts[::step]
                if ds[-1] != pts[-1]:
                    ds.append(pts[-1])
                pts = ds
            return pts, hb

        # 否则记下最短的, 继续降级试
        if best_short is None or ratio < best_short[3]:
            best_short = (cells, hb, hw, ratio)

    # 所有膨胀都绕路过多 -> 用最短的
    if best_short is not None:
        cells, hb, _, _ = best_short
        pts = [grid_to_world(c, r, origin_x, origin_y, resolution)
               for c, r in cells]
        if len(pts) > 6:
            step = max(1, len(pts) // 5)
            ds = pts[::step]
            if ds[-1] != pts[-1]:
                ds.append(pts[-1])
            pts = ds
        return pts, hb

    return None, None
