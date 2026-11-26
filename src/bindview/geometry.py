"""Pure 2D geometry. No chemistry, no drawing, no state -- unit-tested on its own.

Scalars and small tuples are used for one-off computations; the layout engine's
hot loops use the NumPy variants at the bottom of the file.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "Point", "Box", "dist", "sub", "add", "mul", "dot", "cross", "norm", "unit", "perp", "lerp", "rotate",
    "angle_of", "angle_diff", "box_of", "box_union", "box_expand", "boxes_overlap", "point_in_box",
    "point_segment_distance", "segment_segment_distance", "segments_intersect",
    "segment_box_entry", "largest_angular_gap", "free_direction",
    "capsule_gaps", "segments_distance", "points_segments_distance", "marching_squares", "polyline_length", "resample_closed",
    "spread_on_circle", "spread_on_line", "polygon_area",
]

Point = tuple[float, float]
Box = tuple[float, float, float, float]          # x0, y0, x1, y1 with x0 <= x1, y0 <= y1

_EPS = 1e-12


# -- vectors ---------------------------------------------------------------

def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def mul(a: Point, k: float) -> Point:
    return (a[0] * k, a[1] * k)


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(a: Point, fallback: Point = (1.0, 0.0)) -> Point:
    n = math.hypot(a[0], a[1])
    if n < _EPS:
        return fallback
    return (a[0] / n, a[1] / n)


def perp(a: Point) -> Point:
    """Rotate by +90 degrees (counter-clockwise in a y-up frame)."""
    return (-a[1], a[0])


def lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def rotate(a: Point, angle: float) -> Point:
    c, s = math.cos(angle), math.sin(angle)
    return (a[0] * c - a[1] * s, a[0] * s + a[1] * c)


def angle_of(a: Point) -> float:
    return math.atan2(a[1], a[0])


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference ``a - b`` wrapped to (-pi, pi]."""
    d = (a - b) % (2.0 * math.pi)
    if d > math.pi:
        d -= 2.0 * math.pi
    return d


# -- boxes -----------------------------------------------------------------

def box_of(points: Iterable[Point]) -> Box:
    xs, ys = zip(*points)
    return (min(xs), min(ys), max(xs), max(ys))


def box_union(boxes: Iterable[Box]) -> Box:
    bs = list(boxes)
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def box_expand(b: Box, pad: float) -> Box:
    return (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)


def boxes_overlap(a: Box, b: Box, pad: float = 0.0) -> bool:
    return a[0] < b[2] + pad and b[0] < a[2] + pad and a[1] < b[3] + pad and b[1] < a[3] + pad


def point_in_box(p: Point, b: Box) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


# -- segments --------------------------------------------------------------

def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ab = sub(b, a)
    l2 = dot(ab, ab)
    if l2 < _EPS:
        return dist(p, a)
    t = max(0.0, min(1.0, dot(sub(p, a), ab) / l2))
    return dist(p, (a[0] + ab[0] * t, a[1] + ab[1] * t))


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Proper intersection: the open segments cross at a single interior point."""
    d1 = cross(sub(b, a), sub(c, a))
    d2 = cross(sub(b, a), sub(d, a))
    d3 = cross(sub(d, c), sub(a, c))
    d4 = cross(sub(d, c), sub(b, c))
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and \
        abs(d1) > _EPS and abs(d2) > _EPS and abs(d3) > _EPS and abs(d4) > _EPS


def segment_segment_distance(a: Point, b: Point, c: Point, d: Point) -> float:
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(point_segment_distance(a, c, d), point_segment_distance(b, c, d),
               point_segment_distance(c, a, b), point_segment_distance(d, a, b))


def segment_box_entry(outside: Point, inside: Point, box: Box) -> Point:
    """Walking from ``outside`` toward ``inside`` (a point in ``box``), the point
    where the segment first meets the box. Returns ``outside`` if it already
    lies in the box, ``inside`` if the segment never meets it."""
    if point_in_box(outside, box):
        return outside
    dx, dy = inside[0] - outside[0], inside[1] - outside[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, outside[0] - box[0]), (dx, box[2] - outside[0]),
                 (-dy, outside[1] - box[1]), (dy, box[3] - outside[1])):
        if abs(p) < _EPS:
            if q < 0:
                return inside
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return inside
            t0 = max(t0, r)
        else:
            if r < t0:
                return inside
            t1 = min(t1, r)
    return (outside[0] + dx * t0, outside[1] + dy * t0)


# -- directions around a point ---------------------------------------------

def largest_angular_gap(angles: Sequence[float]) -> tuple[float, float]:
    """``(mid_angle, width)`` of the widest gap between the given directions."""
    if not angles:
        return (0.0, 2.0 * math.pi)
    srt = sorted(a % (2.0 * math.pi) for a in angles)
    best_w, best_mid = -1.0, 0.0
    for i, a in enumerate(srt):
        b = srt[(i + 1) % len(srt)]
        w = (b - a) % (2.0 * math.pi)
        if len(srt) == 1:
            w = 2.0 * math.pi
        if w > best_w + 1e-9:
            best_w, best_mid = w, a + w / 2.0
    return (best_mid % (2.0 * math.pi), best_w)


def free_direction(centre: Point, neighbours: Sequence[Point]) -> Point:
    """Unit vector pointing into the widest gap between ``neighbours`` as seen from ``centre``."""
    angles = [angle_of(sub(n, centre)) for n in neighbours if dist(n, centre) > _EPS]
    mid, _ = largest_angular_gap(angles)
    return (math.cos(mid), math.sin(mid))


# -- NumPy kernels ---------------------------------------------------------

# (The kernels below work on x and y apart. The layout engine spends most of its time in them, and a trailing
# axis of two costs several times what the arithmetic does. Every sum is of two terms, so the numbers are the
# same to the last bit as with ``(v * w).sum(-1)``; ``tests/test_layout.py`` holds them to that.)

def points_segments_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each of ``points`` ``(P, 2)`` to each segment ``a[i]->b[i]`` ``(S, 2)``: ``(P, S)``."""
    ax, ay = a[:, 0], a[:, 1]
    abx, aby = b[:, 0] - ax, b[:, 1] - ay                                # (S,)
    l2 = np.maximum(abx * abx + aby * aby, _EPS)
    px, py = points[:, 0][:, None], points[:, 1][:, None]                # (P, 1)
    t = ((px - ax) * abx + (py - ay) * aby) / l2
    np.clip(t, 0.0, 1.0, out=t)
    dx, dy = px - (ax + abx * t), py - (ay + aby * t)
    return np.sqrt(dx * dx + dy * dy)


def _seg_seg_dist_arrays(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Pairwise segment-segment distance, broadcast over leading dimensions.

    Closest-points formulation (Ericson, *Real-Time Collision Detection*, 5.1.9)
    with every degenerate case handled explicitly: either segment, or both, may
    have zero length -- a disc is a capsule whose segment is a point.
    """
    p1x, p1y, p2x, p2y = p1[..., 0], p1[..., 1], p2[..., 0], p2[..., 1]
    d1x, d1y = q1[..., 0] - p1x, q1[..., 1] - p1y
    d2x, d2y = q2[..., 0] - p2x, q2[..., 1] - p2y
    rx, ry = p1x - p2x, p1y - p2y
    a = d1x * d1x + d1y * d1y             # squared length of segment 1
    e = d2x * d2x + d2y * d2y             # squared length of segment 2
    f = d2x * rx + d2y * ry
    c = d1x * rx + d1y * ry
    b = d1x * d2x + d1y * d2y
    a_ok = a > _EPS
    e_ok = e > _EPS
    safe_a = np.where(a_ok, a, 1.0)
    safe_e = np.where(e_ok, e, 1.0)
    denom = a * e - b * b
    # General case: closest point on line 1 to line 2, clamped to segment 1.
    s = np.where(denom > _EPS, np.clip((b * f - c * e) / np.where(denom > _EPS, denom, 1.0), 0.0, 1.0), 0.0)
    # Segment 2 is a point: the answer is the closest point of segment 1 to it.
    s = np.where(e_ok, s, np.clip(-c / safe_a, 0.0, 1.0))
    t = np.where(e_ok, (b * s + f) / safe_e, 0.0)
    # Where t leaves [0, 1], clamp it and recompute s for the clamped end point.
    s = np.where(t < 0.0, np.clip(-c / safe_a, 0.0, 1.0), s)
    s = np.where(t > 1.0, np.clip((b - c) / safe_a, 0.0, 1.0), s)
    t = np.clip(t, 0.0, 1.0)
    # Segment 1 is a point: s is irrelevant, t is the closest point of segment 2 to it.
    s = np.where(a_ok, s, 0.0)
    t = np.where(a_ok, t, np.where(e_ok, np.clip(f / safe_e, 0.0, 1.0), 0.0))
    dx, dy = (p1x + d1x * s) - (p2x + d2x * t), (p1y + d1y * s) - (p2y + d2y * t)
    return np.sqrt(dx * dx + dy * dy)


def segments_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Segment-to-segment distance, broadcasting over leading dimensions."""
    return _seg_seg_dist_arrays(p1, q1, p2, q2)


def capsule_gaps(a0: np.ndarray, a1: np.ndarray, ra: np.ndarray,
                 b0: np.ndarray, b1: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Clearance between every capsule of set A and every capsule of set B.

    A capsule is a segment ``p0->p1`` swept by a disc of radius ``r`` (a disc is
    a capsule with ``p0 == p1``). Returns ``(len(A), len(B))``; negative values
    are penetration depths.
    """
    if len(a0) == 0 or len(b0) == 0:
        return np.zeros((len(a0), len(b0)))
    d = _seg_seg_dist_arrays(a0[:, None, :], a1[:, None, :], b0[None, :, :], b1[None, :, :])
    return d - ra[:, None] - rb[None, :]


# -- contours --------------------------------------------------------------

def polyline_length(points: np.ndarray, closed: bool = False) -> float:
    if len(points) < 2:
        return 0.0
    seg = np.diff(points, axis=0)
    total = float(np.linalg.norm(seg, axis=1).sum())
    if closed:
        total += float(np.linalg.norm(points[0] - points[-1]))
    return total


def polygon_area(points: np.ndarray) -> float:
    """Signed area (positive when counter-clockwise in a y-up frame)."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def resample_closed(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a closed polyline at (approximately) uniform arc-length ``spacing``."""
    pts = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = max(8, int(round(total / spacing)))
    targets = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(targets, cum, pts[:, 0])
    y = np.interp(targets, cum, pts[:, 1])
    return np.column_stack([x, y])


def marching_squares(field: np.ndarray, level: float, x0: float, y0: float, step: float) -> list[np.ndarray]:
    """Closed iso-contours of ``field`` (indexed ``[iy, ix]``) at ``level``.

    Grid node ``(ix, iy)`` sits at ``(x0 + ix*step, y0 + iy*step)``. Contours
    that would leave the grid are dropped: callers pad the grid so the level
    set of interest closes inside it. Returned loops are ``(n, 2)`` arrays,
    ordered longest first, deterministic for a given field.
    """
    ny, nx = field.shape
    inside = field < level
    edges: dict[tuple[int, int, int], tuple[float, float]] = {}

    def crossing(ix: int, iy: int, horizontal: bool) -> tuple[float, float]:
        # Edge from node (ix, iy) to its +x (horizontal) or +y neighbour.
        key = (ix, iy, 0 if horizontal else 1)
        hit = edges.get(key)
        if hit is None:
            v0 = field[iy, ix]
            v1 = field[iy, ix + 1] if horizontal else field[iy + 1, ix]
            t = 0.5 if abs(v1 - v0) < _EPS else float((level - v0) / (v1 - v0))
            t = min(1.0, max(0.0, t))
            hit = (x0 + (ix + (t if horizontal else 0.0)) * step, y0 + (iy + (0.0 if horizontal else t)) * step)
            edges[key] = hit
        return hit

    # Each cell contributes 0, 1 or 2 directed segments between edge crossings.
    # Edge ids within a cell: 0 bottom, 1 right, 2 top, 3 left.
    table = {
        1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),), 6: ((0, 2),), 7: ((3, 2),),
        8: ((2, 3),), 9: ((2, 0),), 11: ((2, 1),), 12: ((1, 3),), 13: ((1, 0),), 14: ((0, 3),),
    }
    nxt: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    def edge_key(ix: int, iy: int, e: int) -> tuple[int, int, int]:
        if e == 0:
            return (ix, iy, 0)
        if e == 1:
            return (ix + 1, iy, 1)
        if e == 2:
            return (ix, iy + 1, 0)
        return (ix, iy, 1)

    for iy in range(ny - 1):
        row0, row1 = inside[iy], inside[iy + 1]
        for ix in range(nx - 1):
            code = (1 if row0[ix] else 0) | (2 if row0[ix + 1] else 0) | (4 if row1[ix + 1] else 0) | (8 if row1[ix] else 0)
            if code == 0 or code == 15:
                continue
            segs: tuple[tuple[int, int], ...]
            if code == 5 or code == 10:
                centre = 0.25 * (field[iy, ix] + field[iy, ix + 1] + field[iy + 1, ix] + field[iy + 1, ix + 1])
                joined = centre < level
                if code == 5:
                    segs = ((3, 2), (1, 0)) if joined else ((3, 0), (1, 2))
                else:
                    segs = ((0, 3), (2, 1)) if joined else ((0, 1), (2, 3))
            else:
                segs = table[code]
            for e_from, e_to in segs:
                nxt[edge_key(ix, iy, e_from)] = edge_key(ix, iy, e_to)

    loops: list[np.ndarray] = []
    visited: set[tuple[int, int, int]] = set()
    for start in sorted(nxt):
        if start in visited:
            continue
        chain = [start]
        visited.add(start)
        cur = nxt[start]
        closed = False
        while True:
            if cur == start:
                closed = True
                break
            if cur in visited or cur not in nxt:
                break
            visited.add(cur)
            chain.append(cur)
            cur = nxt[cur]
        if closed and len(chain) >= 4:
            loops.append(np.array([crossing(k[0], k[1], k[2] == 0) for k in chain], dtype=float))
    loops.sort(key=lambda p: (-polyline_length(p, closed=True), float(p[0, 0]), float(p[0, 1])))
    return loops


# -- one-dimensional spreading ----------------------------------------------

def _pava(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    """Weighted isotonic regression (non-decreasing) by pooling adjacent violators."""
    blocks: list[list[float]] = []            # [weighted sum, weight, count]
    for v, w in zip(values, weights):
        blocks.append([v * w, w, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1] + 1e-15:
            s, w2, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += w2
            blocks[-1][2] += c
    out: list[float] = []
    for s, w, c in blocks:
        out.extend([s / w] * int(c))
    return out


def spread_on_line(targets: Sequence[float], widths: Sequence[float],
                   weights: Sequence[float] | None = None) -> list[float]:
    """Centres for intervals on a line, in the given order, without overlap.

    Exact minimiser of the weighted squared displacement from ``targets``
    subject to ``x[i+1] - x[i] >= (widths[i] + widths[i+1]) / 2``.
    """
    n = len(targets)
    if n == 0:
        return []
    w = list(weights) if weights is not None else [1.0] * n
    offs = [0.0]
    for i in range(1, n):
        offs.append(offs[-1] + (widths[i - 1] + widths[i]) / 2.0)
    y = _pava([t - o for t, o in zip(targets, offs)], w)
    return [yi + o for yi, o in zip(y, offs)]


def spread_on_circle(targets: Sequence[float], widths: Sequence[float], period: float,
                     weights: Sequence[float] | None = None) -> list[float]:
    """Place intervals on a circle of circumference ``period`` without overlap.

    Interval ``i`` has width ``widths[i]`` and would like its centre at
    ``targets[i]``. The cyclic order of the targets is preserved. The circle is
    cut at each of the ``n`` possible places, the resulting line problem is
    solved exactly, and the cheapest solution that also clears the cut is
    returned. If the intervals cannot fit they are packed evenly in order.
    """
    n = len(targets)
    if n == 0:
        return []
    if n == 1:
        return [targets[0] % period]
    w = [max(v, 1e-9) for v in (weights if weights is not None else [1.0] * n)]
    order = sorted(range(n), key=lambda i: (targets[i] % period, i))
    t = [targets[i] % period for i in order]
    wd = [widths[i] for i in order]
    wt = [w[i] for i in order]
    out = [0.0] * n
    if sum(wd) >= period - 1e-9:
        scale = period / sum(wd)
        pos, acc = [], 0.0
        for width in wd:
            pos.append(acc + width * scale / 2.0)
            acc += width * scale
        shift = sum(angle_diff_period(ti, pi, period) * wi for ti, pi, wi in zip(t, pos, wt)) / sum(wt)
        for k, i in enumerate(order):
            out[i] = (pos[k] + shift) % period
        return out
    best: tuple[float, list[float], int] | None = None
    for cut in range(n):
        idx = list(range(cut, n)) + list(range(cut))
        tt = [t[i] + (period if i < cut else 0.0) for i in idx]
        xs = spread_on_line(tt, [wd[i] for i in idx], [wt[i] for i in idx])
        if xs[0] + period - xs[-1] < (wd[idx[0]] + wd[idx[-1]]) / 2.0 - 1e-9:
            continue
        cost = sum(wt[i] * (x - ti) ** 2 for i, x, ti in zip(idx, xs, tt))
        if best is None or cost < best[0] - 1e-12:
            best = (cost, xs, cut)
    if best is None:
        # Fits in total but every cut is blocked: tighten from the least bad cut.
        cut = 0
        idx = list(range(n))
        xs = spread_on_line(t, wd, wt)
        over = (wd[0] + wd[-1]) / 2.0 - (xs[0] + period - xs[-1])
        xs = [x - over * (k / (n - 1) - 0.5) for k, x in enumerate(xs)]
        xs = spread_on_line(xs, wd, wt)
        best = (0.0, xs, cut)
    _, xs, cut = best
    idx = list(range(cut, n)) + list(range(cut))
    for k, x in zip(idx, xs):
        out[order[k]] = x % period
    return out


def angle_diff_period(a: float, b: float, period: float) -> float:
    d = (a - b) % period
    if d > period / 2.0:
        d -= period
    return d
