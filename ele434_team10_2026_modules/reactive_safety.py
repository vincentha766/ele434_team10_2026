def sector_min(ranges, a_deg, b_deg):
    """Minimum range in the angular sector [a_deg, b_deg] (wrapping-safe)."""
    n = len(ranges)
    a = int(round(a_deg * n / 360.0)) % n
    b = int(round(b_deg * n / 360.0)) % n
    if a <= b:
        return min(ranges[a:b + 1])
    return min(ranges[a:] + ranges[:b + 1])


def apply_safety(ranges, v, w, safe_front=0.28, slow_front=0.50,
                 safe_side=0.22):
    """
    Reactive safety layer on top of velocity commands.

    Returns (v_out, w_out, braked):
      braked=True means emergency front stop was triggered (caller should replan).
    """
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
