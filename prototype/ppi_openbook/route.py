"""Lines that go around boxes: the shortest way between two places that passes through no box.

A SKETCH, not part of bindview: nothing here is tested or promised.

A line between two residues that runs under the box of a third reads as a contact of the third. Here a
line is led around every box but the two it joins: the corners of the boxes, set a little off them, are
the places a line may turn at; two places see each other when the straight way between them crosses no
box; and a line is the shortest chain of such ways (Dijkstra). Lines in a second and a third lane turn a
little farther out, so that lines which share a passage lie side by side and not on each other.
"""

from __future__ import annotations

import math

import numpy as np

LANES = (3.5, 6.0, 8.5)        # how far off a box a line turns, by lane (the units of the drawing)
CLEAR = 1.2                    # no line comes nearer to a box than this
TURN = 3.0                     # what a turn costs, so that of two ways of about a length the straighter is taken


def crosses(p: np.ndarray, q: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """For every straight way ``p[k]`` to ``q[k]`` and every box (centre x, centre y, half width, half
    height): does the way pass through the inside of the box? Comes back as ways by boxes."""
    if not len(rects):
        return np.zeros((len(p), 0), dtype=bool)
    d = (q - p)[:, None, :]                                           # ways, 1, 2
    lo = (rects[:, :2] - rects[:, 2:])[None, :, :] - p[:, None, :]     # ways, boxes, 2
    hi = (rects[:, :2] + rects[:, 2:])[None, :, :] - p[:, None, :]
    still = np.abs(d) < 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        t0, t1 = lo / np.where(still, 1.0, d), hi / np.where(still, 1.0, d)
    enter, leave = np.minimum(t0, t1), np.maximum(t0, t1)
    inside = (lo < 0.0) & (hi > 0.0)                                   # where the way does not move along an axis
    enter = np.where(still, np.where(inside, -np.inf, np.inf), enter)
    leave = np.where(still, np.where(inside, np.inf, -np.inf), leave)
    first = np.maximum(enter.max(axis=2), 0.0)
    last = np.minimum(leave.min(axis=2), 1.0)
    return first < last - 1e-9


class Router:
    """Ways around a set of boxes, in one lane."""

    def __init__(self, rects, margin: float = LANES[0], clear: float = CLEAR):
        self.rects = np.asarray(rects, dtype=float).reshape(-1, 4)
        self.walls = self.rects + np.array([0.0, 0.0, clear, clear])
        r = self.rects
        corners = np.concatenate([np.stack([r[:, 0] + sx * (r[:, 2] + margin), r[:, 1] + sy * (r[:, 3] + margin)], axis=1)
                                  for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]) if len(r) else np.zeros((0, 2))
        if len(corners):                                               # a corner inside another box is no place to turn
            inside = ((np.abs(corners[:, None, 0] - self.walls[None, :, 0]) < self.walls[None, :, 2])
                      & (np.abs(corners[:, None, 1] - self.walls[None, :, 1]) < self.walls[None, :, 3])).any(axis=1)
            corners = corners[~inside]
        self.nodes = corners
        n = len(corners)
        self.cost = np.full((n, n), np.inf)
        if n:
            i, j = np.triu_indices(n, 1)
            free = ~crosses(corners[i], corners[j], self.walls).any(axis=1)
            length = np.linalg.norm(corners[i] - corners[j], axis=1)
            self.cost[i[free], j[free]] = length[free]
            self.cost[j[free], i[free]] = length[free]

    def _sight(self, p: np.ndarray, own) -> np.ndarray:
        """How far it is from ``p`` to every corner it sees; its own boxes do not stand in the way."""
        if not len(self.nodes):
            return np.zeros(0)
        keep = np.ones(len(self.walls), dtype=bool)
        keep[list(own)] = False
        blocked = crosses(np.repeat(p[None, :], len(self.nodes), axis=0), self.nodes, self.walls[keep]).any(axis=1)
        return np.where(blocked, np.inf, np.linalg.norm(self.nodes - p, axis=1))

    def way(self, a, b, own=()) -> "list[np.ndarray] | None":
        """The places a line from ``a`` to ``b`` goes through, both ends included. ``own`` are the boxes
        of its two ends (their numbers), which it may leave and enter. ``None`` where no way is free."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        keep = np.ones(len(self.walls), dtype=bool)
        keep[list(own)] = False
        if not crosses(a[None, :], b[None, :], self.walls[keep]).any():
            return [a, b]
        n = len(self.nodes)
        if not n:
            return None
        from_a, to_b = self._sight(a, own), self._sight(b, own)
        best = from_a + TURN
        via = np.full(n, -1)
        done = np.zeros(n, dtype=bool)
        while True:
            open_ = np.where(done, np.inf, best)
            k = int(np.argmin(open_))
            if not np.isfinite(open_[k]):
                break
            done[k] = True
            step = best[k] + self.cost[k] + TURN
            better = step < best
            best[better], via[better] = step[better], k
        total = best + to_b
        k = int(np.argmin(total))
        if not np.isfinite(total[k]):
            return None
        chain = [b]
        while k >= 0:
            chain.append(self.nodes[k])
            k = int(via[k])
        chain.append(a)
        return chain[::-1]


def routers(rects) -> "list[Router]":
    return [Router(rects, margin) for margin in LANES]


def leave(centre, towards, half_w: float, half_h: float, round_: bool) -> np.ndarray:
    """Where a line from the middle of a mark towards a place leaves the mark: a box, or an ellipse."""
    centre, towards = np.asarray(centre, dtype=float), np.asarray(towards, dtype=float)
    d = towards - centre
    if float(np.hypot(*d)) < 1e-9:
        return centre
    if round_:
        t = 1.0 / math.sqrt((d[0] / half_w) ** 2 + (d[1] / half_h) ** 2)
    else:
        t = min(half_w / abs(d[0]) if abs(d[0]) > 1e-9 else np.inf, half_h / abs(d[1]) if abs(d[1]) > 1e-9 else np.inf)
    return centre + min(t, 1.0) * d


def path(points, radius: float = 5.0) -> str:
    """The line through these places as the ``d`` of an SVG path, its turns rounded."""
    pts = [np.asarray(p, dtype=float) for p in points]
    out = [f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"]
    for before, here, after in zip(pts, pts[1:], pts[2:]):
        u, v = here - before, after - here
        lu, lv = float(np.hypot(*u)), float(np.hypot(*v))
        if lu < 1e-6 or lv < 1e-6:
            continue
        r = min(radius, lu / 2.0, lv / 2.0)
        p0, p1 = here - u / lu * r, here + v / lv * r
        out.append(f"L {p0[0]:.1f} {p0[1]:.1f} Q {here[0]:.1f} {here[1]:.1f} {p1[0]:.1f} {p1[1]:.1f}")
    out.append(f"L {pts[-1][0]:.1f} {pts[-1][1]:.1f}")
    return " ".join(out)
