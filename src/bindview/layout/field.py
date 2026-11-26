"""The ligand as an obstacle: a sampled distance field plus bond segments.

Every placement question about the ligand -- is this spot free, which way is
outward, does this line cross a bond -- is answered from here. The distance
field is sampled once on a regular grid (a few hundred thousand cheap NumPy
operations) so that the search over candidate poses can test thousands of
positions with array lookups.
"""

from __future__ import annotations

import hashlib
from functools import cached_property
from typing import Sequence

import numpy as np

from ..depict import Depiction
from ..geometry import marching_squares, points_segments_distance

__all__ = ["LigandField"]


class LigandField:
    """Distance to the nearest ligand ink, sampled on a grid.

    Parameters
    ----------
    dep:
        The ligand depiction.
    reach:
        How far beyond the ligand's bounding box the grid extends.
    step:
        Grid spacing in user units.
    extra:
        Additional obstacle discs ``(x, y, radius)``.
    """

    def __init__(self, dep: Depiction, reach: float, step: float = 2.0,
                 extra: "Sequence[tuple[float, float, float]]" = ()) -> None:
        self.dep = dep
        self.step = float(step)
        m = dep.metrics

        seg_a: list[tuple[float, float]] = []
        seg_b: list[tuple[float, float]] = []
        rad: list[float] = []
        bond_a: list[tuple[float, float]] = []
        bond_b: list[tuple[float, float]] = []
        straight = dep.arranged == "plane"       # ... where a link between two boxes is that straight line
        for b in dep.bonds:
            if b.kind in ("closure", "crosslink") and not straight:
                continue                 # drawn as a bracket beside the chain, not as this straight line
            pa, pb = dep.pos(b.a), dep.pos(b.b)
            if b.kind in ("covalent", "chain", "gap", "closure", "crosslink"):
                bond_a.append(pa)
                bond_b.append(pb)
            wide = m.double_offset if (b.order > 1 or b.aromatic) else (m.wedge_width / 2.0 if b.stereo != "none" else 0.0)
            seg_a.append(pa)
            seg_b.append(pb)
            rad.append(m.bond_width / 2.0 + wide)
        for a in dep.atoms:
            if a.label is None:
                # A bare vertex still occupies its point (single-atom fragments, chain ends).
                seg_a.append(a.pos)
                seg_b.append(a.pos)
                rad.append(m.bond_width / 2.0)
                continue
            for box in a.label.boxes:
                h = box[3] - box[1]
                w = box[2] - box[0]
                r = min(w, h) / 2.0
                cy = (box[1] + box[3]) / 2.0
                cx = (box[0] + box[2]) / 2.0
                if w >= h:
                    seg_a.append((box[0] + r, cy))
                    seg_b.append((box[2] - r, cy))
                else:
                    seg_a.append((cx, box[1] + r))
                    seg_b.append((cx, box[3] - r))
                rad.append(r)
        for x, y, r in extra:
            # Ink the style will add on the ligand (contact spokes): keep bodies off it.
            seg_a.append((x, y))
            seg_b.append((x, y))
            rad.append(r)
        self.seg_a = np.array(seg_a, dtype=float).reshape(-1, 2)
        self.seg_b = np.array(seg_b, dtype=float).reshape(-1, 2)
        self.seg_r = np.array(rad, dtype=float)
        self.bond_a = np.array(bond_a, dtype=float).reshape(-1, 2)
        self.bond_b = np.array(bond_b, dtype=float).reshape(-1, 2)
        self.atom_xy = np.array([a.pos for a in dep.atoms], dtype=float).reshape(-1, 2)
        #: The residue boxes of a chain ligand, ``(U, 4)`` as x0, y0, x1, y1; empty for a ligand drawn as atoms.
        self.unit_boxes = np.array([u.box for u in dep.units], dtype=float).reshape(-1, 4)
        #: The ink of the atoms that are drawn beside such a box, as small rectangles (every atom, every
        #: bond in two halves): a line that is another atom's does not run across them either. Residue by
        #: residue, with one rectangle round all of a residue's: most lines come near none of them, and are
        #: looked at against the small ones only where they come near.
        self._pieces = [np.array(u.rects, dtype=float).reshape(-1, 4) for u in dep.units if u.rects]
        self._piece_bounds = np.array([[g[:, 0].min(), g[:, 1].min(), g[:, 2].max(), g[:, 3].max()]
                                       for g in self._pieces], dtype=float).reshape(-1, 4)

        pts = np.vstack([self.seg_a, self.seg_b])
        # (Rounded before it is cut to a whole number: a bound that lies on one but for a last bit lies on it.)
        self.x0 = float(np.floor(np.round(pts[:, 0].min() - reach, 6)))
        self.y0 = float(np.floor(np.round(pts[:, 1].min() - reach, 6)))
        nx = int(np.ceil(np.round((pts[:, 0].max() + reach - self.x0) / self.step, 6))) + 1
        ny = int(np.ceil(np.round((pts[:, 1].max() + reach - self.y0) / self.step, 6))) + 1
        self.nx, self.ny = nx, ny
        gx = self.x0 + np.arange(nx) * self.step
        gy = self.y0 + np.arange(ny) * self.step
        self.gx, self.gy = gx, gy

    @cached_property
    def values(self) -> np.ndarray:
        """The distance at the nodes of the grid, ``(ny, nx)``. Worked out when it is first asked for."""
        grid = np.column_stack([np.tile(self.gx, self.ny), np.repeat(self.gy, self.nx)])
        best = np.full(len(grid), np.inf)
        # Chunk over segments to keep temporaries small.
        for s in range(0, len(self.seg_a), 24):
            d = points_segments_distance(grid, self.seg_a[s:s + 24], self.seg_b[s:s + 24]) - self.seg_r[None, s:s + 24]
            best = np.minimum(best, d.min(axis=1))
        return best.reshape(self.ny, self.nx)

    @cached_property
    def key(self) -> str:
        """A digest of all that is in the field: two with the same one are the same."""
        h = hashlib.blake2b(digest_size=20)
        for a in (self.seg_a, self.seg_b, self.seg_r, self.bond_a, self.bond_b, self.atom_xy, self.unit_boxes,
                  *self._pieces):
            h.update(f"{a.dtype.str}{a.shape}".encode())
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(repr((self.step, self.x0, self.y0, self.nx, self.ny)).encode())
        return h.hexdigest()

    # -- queries -----------------------------------------------------------
    def sdf(self, pts: np.ndarray) -> np.ndarray:
        """Clearance to the ligand at ``pts`` (``(..., 2)``), bilinearly interpolated.

        Points outside the grid are far from the ligand by construction; they
        get the edge value plus their distance to the grid, which keeps the
        field monotone.
        """
        p = np.asarray(pts, dtype=float)
        fx = (p[..., 0] - self.x0) / self.step
        fy = (p[..., 1] - self.y0) / self.step
        cx = np.clip(fx, 0.0, self.nx - 1.000001)
        cy = np.clip(fy, 0.0, self.ny - 1.000001)
        ix = cx.astype(int)
        iy = cy.astype(int)
        tx = cx - ix
        ty = cy - iy
        v = self.values
        out = (v[iy, ix] * (1 - tx) * (1 - ty) + v[iy, ix + 1] * tx * (1 - ty)
               + v[iy + 1, ix] * (1 - tx) * ty + v[iy + 1, ix + 1] * tx * ty)
        outside = np.hypot(fx - cx, fy - cy) * self.step
        return out + outside

    def outward(self, pts: np.ndarray) -> np.ndarray:
        """Unit vectors pointing away from the ligand at ``pts`` (the field gradient)."""
        p = np.asarray(pts, dtype=float)
        h = self.step
        ex = np.array([h, 0.0])
        ey = np.array([0.0, h])
        gx = self.sdf(p + ex) - self.sdf(p - ex)
        gy = self.sdf(p + ey) - self.sdf(p - ey)
        n = np.hypot(gx, gy)
        n = np.where(n < 1e-9, 1.0, n)
        return np.stack([gx / n, gy / n], axis=-1)

    def boxes_crossed(self, p: np.ndarray, q: np.ndarray, own: np.ndarray, margin: float = 1.0,
                      reach: float = 0.0) -> np.ndarray:
        """Does the segment ``p[i] -> q[i]`` pass through a residue box of a chain ligand?

        The box that holds ``own`` (the point the link aims at) does not count:
        the link ends on it. A contact is read off by the box its line reaches,
        so a line that runs across another box on its way says the wrong thing.
        Returns one boolean per segment; all ``False`` for a ligand drawn as atoms.

        Where atoms are drawn beside a box, their ink counts as well: a line
        does not run across an atom or a bond it does not end on. What lies
        within ``reach`` of ``own`` is its own target's ink (the atoms of a
        ring round its centre) and does not count.
        """
        n = len(p)
        if not len(self.unit_boxes) or not n:
            return np.zeros(n, dtype=bool)
        hit = self._crossed(p, q, own, self.unit_boxes, margin, 0.0)
        if len(self._pieces):
            near = self._through(p, q, self._piece_bounds, 0.0)
            for g in np.nonzero(near.any(axis=0))[0]:
                rows = np.nonzero(near[:, g] & ~hit)[0]
                if len(rows):
                    hit[rows] = self._crossed(p[rows], q[rows], own, self._pieces[g], 0.0, reach)
        return hit

    @classmethod
    def _crossed(cls, p: np.ndarray, q: np.ndarray, own: np.ndarray, boxes: np.ndarray, margin: float,
                 reach: float) -> np.ndarray:
        """One boolean per segment ``p[i] -> q[i]``: does it pass through one of ``boxes`` that is not
        within ``reach`` of ``own``?"""
        x0, y0 = boxes[:, 0] - margin, boxes[:, 1] - margin
        x1, y1 = boxes[:, 2] + margin, boxes[:, 3] + margin
        away = np.hypot(np.maximum(np.maximum(x0 - own[0], own[0] - x1), 0.0),
                        np.maximum(np.maximum(y0 - own[1], own[1] - y1), 0.0))
        foreign = away > reach
        return (cls._through(p, q, boxes, margin) & foreign[None, :]).any(axis=1)

    @staticmethod
    def _through(p: np.ndarray, q: np.ndarray, boxes: np.ndarray, margin: float) -> np.ndarray:
        """``(len(p), len(boxes))`` booleans: does the segment ``p[i] -> q[i]`` pass through ``boxes[j]``?"""
        n = len(p)
        x0, y0 = boxes[:, 0] - margin, boxes[:, 1] - margin
        x1, y1 = boxes[:, 2] + margin, boxes[:, 3] + margin
        lo = np.zeros((n, len(x0)))
        hi = np.ones((n, len(x0)))
        for axis, (b0, b1) in enumerate(((x0, x1), (y0, y1))):
            start = p[:, axis][:, None]
            d = (q[:, axis] - p[:, axis])[:, None]
            still = np.abs(d) < 1e-12
            safe = np.where(still, 1.0, d)
            t0, t1 = (b0[None, :] - start) / safe, (b1[None, :] - start) / safe
            inside = (start >= b0[None, :]) & (start <= b1[None, :])
            lo = np.maximum(lo, np.where(still, np.where(inside, -np.inf, np.inf), np.minimum(t0, t1)))
            hi = np.minimum(hi, np.where(still, np.where(inside, np.inf, -np.inf), np.maximum(t0, t1)))
        return hi > lo

    def bond_crossings(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Number of covalent bonds properly crossed by each segment ``p[i]->q[i]``."""
        p = np.asarray(p, dtype=float).reshape(-1, 2)
        q = np.asarray(q, dtype=float).reshape(-1, 2)
        if len(self.bond_a) == 0 or len(p) == 0:
            return np.zeros(len(p), dtype=int)
        return segment_cross_matrix(p, q, self.bond_a, self.bond_b).sum(axis=1)

    def contours(self, level: float) -> list[np.ndarray]:
        """Closed curves at constant clearance ``level``, longest first."""
        return marching_squares(self.values, level, self.x0, self.y0, self.step)

    def cells(self, lo: float, hi: float, stride: int = 2) -> np.ndarray:
        """Grid nodes whose clearance lies in ``[lo, hi]`` (every ``stride``-th node)."""
        v = self.values[::stride, ::stride]
        iy, ix = np.nonzero((v >= lo) & (v <= hi))
        return np.column_stack([self.gx[::stride][ix], self.gy[::stride][iy]])


def segment_cross_matrix(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(len(p), len(a))`` booleans: does segment ``p[i]->q[i]`` properly cross ``a[j]->b[j]``."""
    d1 = (q - p)[:, None, :]                       # (P, 1, 2)
    pa = a[None, :, :] - p[:, None, :]             # (P, A, 2)
    pb = b[None, :, :] - p[:, None, :]
    c1 = d1[..., 0] * pa[..., 1] - d1[..., 1] * pa[..., 0]
    c2 = d1[..., 0] * pb[..., 1] - d1[..., 1] * pb[..., 0]
    d2 = (b - a)[None, :, :]                       # (1, A, 2)
    ap = -pa
    aq = q[:, None, :] - a[None, :, :]
    c3 = d2[..., 0] * ap[..., 1] - d2[..., 1] * ap[..., 0]
    c4 = d2[..., 0] * aq[..., 1] - d2[..., 1] * aq[..., 0]
    eps = 1e-9
    return ((c1 > eps) & (c2 < -eps) | (c1 < -eps) & (c2 > eps)) & \
           ((c3 > eps) & (c4 < -eps) | (c3 < -eps) & (c4 > eps))
