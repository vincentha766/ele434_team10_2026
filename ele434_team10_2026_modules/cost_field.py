import numpy as np


def inflate(grid, radius):
    """Chebyshev inflation: mark all cells within `radius` of occupied (>=50) as True."""
    occ = (grid >= 50)
    out = occ.copy()
    for _ in range(radius):
        new = out.copy()
        new[:-1] |= out[1:]
        new[1:]  |= out[:-1]
        new[:, :-1] |= out[:, 1:]
        new[:, 1:]  |= out[:, :-1]
        out = new
    return out


def cost_field(grid, hard_r, soft_r):
    """
    Returns (blocked, extra_cost):
      blocked:    bool[h,w] — True within hard_r cells of any obstacle
      extra_cost: float[h,w] — gradual cost increase between hard_r and soft_r
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
