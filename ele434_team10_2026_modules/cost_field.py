import numpy as np
from collections import deque


def inflate(occ, radius):
    """Chebyshev inflation: mark all cells within `radius` of any True cell."""
    out = occ.copy()
    for _ in range(radius):
        new = out.copy()
        new[:-1] |= out[1:]
        new[1:]  |= out[:-1]
        new[:, :-1] |= out[:, 1:]
        new[:, 1:]  |= out[:, :-1]
        out = new
    return out


def _label_components(occ):
    """4-connected BFS labeling. Returns int label array + count."""
    h, w = occ.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    for i in range(h):
        for j in range(w):
            if occ[i, j] and labels[i, j] == 0:
                cur += 1
                q = deque([(i, j)])
                while q:
                    ci, cj = q.popleft()
                    if not (0 <= ci < h and 0 <= cj < w):
                        continue
                    if not occ[ci, cj] or labels[ci, cj] != 0:
                        continue
                    labels[ci, cj] = cur
                    q.append((ci - 1, cj))
                    q.append((ci + 1, cj))
                    q.append((ci, cj - 1))
                    q.append((ci, cj + 1))
    return labels, cur


def _classify_components(labels, n_comp, origin_x=None, origin_y=None,
                          resolution=0.05, field_half=2.0):
    """分类连通块.

    'wall' 条件 (任一):
      A. 大部分 cells 在场地外墙附近 (within 15cm of x=±2 or y=±2 in world)
         -> 阵地外墙, 长直线
      B. bbox aspect ≥ 1.6 且 cells ≥ 5 -> 直线段
      C. bbox 长边 ≥ 5 且 density < 0.55 -> L 形等稀疏长形

    'beacon' 否则 (圆柱状, 紧密小簇).

    No origin/resolution -> 退化只用 shape 判定 (无 perimeter 检测).
    """
    classes = {}
    for k in range(1, n_comp + 1):
        coords = np.argwhere(labels == k)
        if len(coords) == 0:
            continue
        ys = coords[:, 0]
        xs = coords[:, 1]

        # A: 外墙位置检测
        if origin_x is not None and origin_y is not None:
            ys_w = origin_y + ys * resolution
            xs_w = origin_x + xs * resolution
            perim_frac = (
                (np.abs(xs_w) >= field_half - 0.15)
                | (np.abs(ys_w) >= field_half - 0.15)
            ).sum() / len(coords)
            if perim_frac > 0.5:
                classes[k] = 'wall'
                continue

        # B + C: 形状检测
        bbh = int(ys.max() - ys.min() + 1)
        bbw = int(xs.max() - xs.min() + 1)
        long_side = max(bbh, bbw)
        short_side = max(1, min(bbh, bbw))
        aspect = long_side / short_side
        density = len(coords) / max(1, bbh * bbw)
        n_cells = len(coords)

        is_line = (aspect >= 1.6) and (n_cells >= 5)
        is_L_shape = (long_side >= 5) and (density < 0.55)

        classes[k] = 'wall' if (is_line or is_L_shape) else 'beacon'
    return classes


def cost_field(grid, hard_r, soft_r, hard_r_wall=None,
                origin_x=None, origin_y=None, resolution=0.05):
    """
    Inflate占据栅格为代价场, 区分墙和柱子.
      blocked:    bool[h,w] — 硬禁区, robot 不能进
      extra_cost: float[h,w] — 软膨胀梯度, 倾向远离障碍但不阻断

    Args:
      hard_r: 柱状障碍 (beacon) 的膨胀半径 (cells, Chebyshev)
      hard_r_wall: 长条墙 (wall_*) 的膨胀半径; None 则与 hard_r 相同
      soft_r: 软膨胀外边界, 软梯度从 hard_r+1 延伸到 soft_r
      origin_x/origin_y/resolution: SLAM 占据栅格原点 + 分辨率, 用于按
        世界坐标判断 cell 是否在场地外墙附近. 不传只用形状判定.

    Why differ: walls 是平面, robot 只需 robot_radius (~21cm) 缓冲即可不撞;
    beacons 是圆柱, robot 需 robot_radius + beacon_radius (~31cm) 缓冲.
    一律取大值会让 wall 旁通道挤窄到 robot 过不去.
    """
    if hard_r_wall is None:
        hard_r_wall = hard_r
    h, w = grid.shape
    occ = (grid >= 50)

    # 连通块标记 + 分类 (带位置感知)
    labels, n_comp = _label_components(occ)
    classes = _classify_components(labels, n_comp,
                                    origin_x=origin_x, origin_y=origin_y,
                                    resolution=resolution)

    wall_mask = np.zeros((h, w), dtype=bool)
    beacon_mask = np.zeros((h, w), dtype=bool)
    for k, c in classes.items():
        m = (labels == k)
        if c == 'wall':
            wall_mask |= m
        else:
            beacon_mask |= m

    # 分别膨胀
    wall_blocked = inflate(wall_mask, hard_r_wall)
    beacon_blocked = inflate(beacon_mask, hard_r)
    blocked = wall_blocked | beacon_blocked

    # 软膨胀: 用 min(hard_r) 起算, 距离量从任意障碍计
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
    extra = np.zeros((h, w), dtype=np.float32)
    hr_min = min(hard_r, hard_r_wall)
    for d in range(hr_min + 1, soft_r + 1):
        extra[dist == d] = (soft_r - d + 1) * 0.6
    return blocked, extra
