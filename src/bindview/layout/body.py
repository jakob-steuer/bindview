"""Bodies: the rigid things the layout engine places.

A body is a set of collision shapes in its own coordinate frame, optionally a
text label that follows it without rotating, and one or more *ports* -- points
on the body tied to fixed points on the ligand. All shapes are capsules (a
segment swept by a disc); a disc is the degenerate case.

Every method that maps local geometry to the page is vectorised over a batch of
candidate poses, because the engine scores thousands of them at a time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..geometry import Point

__all__ = ["Port", "Body", "Pose", "LabelSpec", "march_entry", "SPRINGS"]

#: Spring constants by class of tie -- the ``weight`` a :class:`Port` gets. A tie pulls its body
#: towards the length ``ideal`` with a force constant of ``LayoutParams.w_length * weight`` (times
#: ``w_short`` while it is shorter than ideal); the ligand does not move at all.
#:
#: ``link``      an interaction that is drawn as a line.
#: ``sightline`` the line-less contacts of a residue, together: the constant is shared among them.
#: ``eyelash``   the same for a residue that makes line-less contacts only and is drawn as an
#:               eyelash: nothing else holds it, so its sightlines together pull like one link.
SPRINGS: dict[str, float] = {"link": 1.0, "sightline": 0.35, "eyelash": 1.0}


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    angle: float = 0.0
    mirror: bool = False

    def apply(self, p: Point) -> Point:
        lx, ly = p[0], (-p[1] if self.mirror else p[1])
        c, s = np.cos(self.angle), np.sin(self.angle)
        return (float(lx * c - ly * s + self.x), float(lx * s + ly * c + self.y))


@dataclass
class Port:
    """A tie between a point on the body and a point on the ligand.

    ``drawn`` ports become interaction lines and reserve room for themselves
    (and for a distance label, if ``label_size`` is set). Undrawn ports are
    sightlines: they pull the body toward the ligand atom without any ink.
    """

    local: Point
    target: Point
    ideal: float
    min_length: float = 0.0
    weight: float = 1.0
    drawn: bool = True
    port_radius: float = 0.0
    target_radius: float = 0.0
    #: How far from the point it aims at the ink is still that of its own target (the atoms of a ring
    #: round its centre, the label of an atom): a link may run across that, and across no other atom
    #: that is drawn beside a residue box of a chain ligand.
    target_reach: float = 0.0
    label_size: tuple[float, float] | None = None
    #: Ink around the target as world-space capsules ``(p0 (H, 2), p1 (H, 2), r (H,))``.
    #: When given, the link stops where it first meets them (an atom label, the
    #: halo of a charged group) instead of at ``target_radius``.
    target_caps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    #: Index into the body's upright boxes: the link stops at that box.
    port_upright: int | None = None
    #: Ink around the port in body-local coordinates (the halo of a charged
    #: group), plus the upright boxes of the atoms in it: the link stops at
    #: whichever it meets first.
    port_caps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    port_uprights: tuple[int, ...] = ()
    #: Preferred minimum angle (radians) between this link and other bodies'
    #: links into the same target, so that they fan out around it; 0 = no preference.
    fan: float = 0.0
    #: Indices into the body's upright boxes that the link may not run through: the names of the
    #: atoms of its own residue that it does not end on. Empty for every body that is not drawn at a
    #: level of detail: there a link across its own body costs, and is not forbidden.
    clear_of: tuple[int, ...] = ()
    #: ... and whether it may not run through the body's own name label either (a link that ends on
    #: an atom of a residue has no business at the residue's name).
    clear_of_name: bool = False


@dataclass(frozen=True)
class LabelSpec:
    """An upright text box attached to the body.

    The box is centred ``gap`` beyond ``anchor`` along ``direction`` (both in
    body-local coordinates), far enough that the box clears the anchor whatever
    the body's rotation.
    """

    anchor: Point
    direction: Point
    width: float
    height: float
    gap: float = 3.0
    pad: float = 1.0


@dataclass
class Body:
    key: Any
    kind: str                                    # "disc" | "fragment"
    shapes_p0: np.ndarray                        # (S, 2) local
    shapes_p1: np.ndarray                        # (S, 2) local
    shapes_r: np.ndarray                         # (S,)
    ports: list[Port]
    label: LabelSpec | None = None
    rotatable: bool = False
    mirrorable: bool = False
    ligand_clearance: float = 4.0
    axis: Point = (1.0, 0.0)                     # local direction that should point away from the ligand
    priority: float = 0.0
    payload: Any = None
    #: Text boxes that travel with the body but stay upright (atom-name labels on
    #: a fragment): local centre points ``(U, 2)`` and ``(width, height)`` each.
    upright_pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    upright_wh: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    upright_pad: float = 1.5
    _samples: np.ndarray = field(init=False, repr=False)
    _sample_r: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.shapes_p0 = np.asarray(self.shapes_p0, dtype=float).reshape(-1, 2)
        self.shapes_p1 = np.asarray(self.shapes_p1, dtype=float).reshape(-1, 2)
        self.shapes_r = np.asarray(self.shapes_r, dtype=float).reshape(-1)
        self.upright_pts = np.asarray(self.upright_pts, dtype=float).reshape(-1, 2)
        self.upright_wh = np.asarray(self.upright_wh, dtype=float).reshape(-1, 2)
        pts, rr = [], []
        for p0, p1, r in zip(self.shapes_p0, self.shapes_p1, self.shapes_r):
            # (The length by a function that every platform rounds alike, and rounded before it is cut to a whole
            # number of steps: a last bit decides neither.)
            n = max(1, math.ceil(round(math.hypot(float(p1[0] - p0[0]), float(p1[1] - p0[1])) / 7.0, 6)))
            for k in range(n + 1):
                t = k / n if n else 0.0
                pts.append(p0 + (p1 - p0) * t)
                rr.append(r)
                if np.allclose(p0, p1):
                    break
        self._samples = np.array(pts, dtype=float).reshape(-1, 2)
        self._sample_r = np.array(rr, dtype=float)
        self._port_local = np.array([p.local for p in self.ports], dtype=float).reshape(-1, 2)
        self._targets = np.array([p.target for p in self.ports], dtype=float).reshape(-1, 2)
        self._reach = np.array([p.target_reach for p in self.ports], dtype=float)
        self._drawn = np.array([p.drawn for p in self.ports], dtype=bool)
        self._ideal = np.array([p.ideal for p in self.ports], dtype=float)
        self._minlen = np.array([p.min_length for p in self.ports], dtype=float)
        self._weight = np.array([p.weight for p in self.ports], dtype=float)
        self._rport = np.array([p.port_radius for p in self.ports], dtype=float)
        self._rtarget = np.array([p.target_radius for p in self.ports], dtype=float)
        self._has_dlabel = np.array([p.label_size is not None for p in self.ports], dtype=bool)
        self._dlabel = np.array([p.label_size or (0.0, 0.0) for p in self.ports], dtype=float).reshape(-1, 2)

    # -- sizes -------------------------------------------------------------
    @property
    def radius(self) -> float:
        """Radius of a circle about the local origin containing every rotating shape."""
        best = 0.0
        if len(self.shapes_r):
            d0 = np.linalg.norm(self.shapes_p0, axis=1) + self.shapes_r
            d1 = np.linalg.norm(self.shapes_p1, axis=1) + self.shapes_r
            best = float(max(d0.max(), d1.max()))
        if len(self.upright_pts):
            reach = np.linalg.norm(self.upright_pts, axis=1) + np.hypot(self.upright_wh[:, 0], self.upright_wh[:, 1]) / 2.0
            best = max(best, float(reach.max()))
        return best

    @property
    def n_ports(self) -> int:
        return len(self.ports)

    @property
    def n_proper(self) -> int:
        """Number of solids that are the body itself (shapes, upright boxes, name
        label); the distance labels come after these in :meth:`solids`."""
        return len(self.shapes_r) + len(self.upright_pts) + (1 if self.label is not None else 0)

    @property
    def proper_radius(self) -> float:
        """Radius about the local origin that contains the body itself in any pose."""
        best = self.radius
        if self.label is not None:
            lab = self.label
            reach = (math.hypot(*lab.anchor) + lab.gap + max(lab.width, lab.height) / 2.0
                     + math.hypot(lab.width, lab.height) / 2.0 + lab.pad)
            best = max(best, reach)
        return best + self.upright_pad

    # -- transforms (batched) ----------------------------------------------
    @staticmethod
    def _xf(local: np.ndarray, pos: np.ndarray, ang: np.ndarray, mir: np.ndarray) -> np.ndarray:
        """Map local points ``(S, 2)`` through ``N`` poses: ``(N, S, 2)``."""
        ly = np.where(mir[:, None], -local[None, :, 1], local[None, :, 1])
        lx = np.broadcast_to(local[None, :, 0], ly.shape)
        c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
        return np.stack([lx * c - ly * s + pos[:, 0:1], lx * s + ly * c + pos[:, 1:2]], axis=-1)

    @staticmethod
    def _xd(vec: Sequence[float], ang: np.ndarray, mir: np.ndarray) -> np.ndarray:
        """Map one local direction through ``N`` poses: ``(N, 2)``."""
        ly = np.where(mir, -vec[1], vec[1])
        c, s = np.cos(ang), np.sin(ang)
        return np.stack([vec[0] * c - ly * s, vec[0] * s + ly * c], axis=-1)

    def ports_world(self, pos: np.ndarray, ang: np.ndarray, mir: np.ndarray) -> np.ndarray:
        return self._xf(self._port_local, pos, ang, mir)

    def label_centre(self, pos: np.ndarray, ang: np.ndarray, mir: np.ndarray) -> np.ndarray | None:
        if self.label is None:
            return None
        lab = self.label
        anchor = self._xf(np.array([lab.anchor], dtype=float), pos, ang, mir)[:, 0, :]
        d = self._xd(lab.direction, ang, mir)
        reach = lab.gap + np.abs(d[:, 0]) * lab.width / 2.0 + np.abs(d[:, 1]) * lab.height / 2.0
        return anchor + d * reach[:, None]

    def upright_capsules(self, pos: np.ndarray, ang: np.ndarray, mir: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World capsules of the upright boxes: ``(N, U, 2)``, ``(N, U, 2)``, ``(U,)``."""
        c = self._xf(self.upright_pts, pos, ang, mir)
        w, h = self.upright_wh[:, 0], self.upright_wh[:, 1]
        half = np.maximum(0.0, (w - h) / 2.0)
        off = np.stack([half, np.zeros_like(half)], axis=-1)[None, :, :]
        return c - off, c + off, h / 2.0 + self.upright_pad

    def link_segments(self, pos: np.ndarray, ang: np.ndarray, mir: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(port_end, target_end, length)`` for every port: the *visible* part of
        each link (stopped by the ink at both ends) and the full centre-to-centre
        length. Shapes ``(N, K, 2)``, ``(N, K, 2)``, ``(N, K)``."""
        p = self.ports_world(pos, ang, mir)
        t = np.broadcast_to(self._targets[None, :, :], p.shape)
        v = t - p
        length = np.linalg.norm(v, axis=2)
        u = v / np.maximum(length, 1e-9)[:, :, None]
        a = p + u * np.minimum(self._rport[None, :], length * 0.45)[:, :, None]
        b = t - u * np.minimum(self._rtarget[None, :], length * 0.45)[:, :, None]
        up = None
        for k, port in enumerate(self.ports):
            if port.target_caps is not None:
                c0, c1, cr = port.target_caps
                n = len(pos)
                b[:, k, :] = march_entry(p[:, k, :], t[:, k, :],
                                         np.broadcast_to(c0[None, :, :], (n,) + c0.shape),
                                         np.broadcast_to(c1[None, :, :], (n,) + c1.shape), cr)
            if port.port_upright is not None:
                if up is None:
                    up = self.upright_capsules(pos, ang, mir)
                j = port.port_upright
                a[:, k, :] = march_entry(t[:, k, :], p[:, k, :], up[0][:, j:j + 1, :], up[1][:, j:j + 1, :],
                                         up[2][j:j + 1])
            elif port.port_caps is not None or port.port_uprights:
                c0s, c1s, crs = [], [], []
                if port.port_caps is not None:
                    c0s.append(self._xf(port.port_caps[0], pos, ang, mir))
                    c1s.append(self._xf(port.port_caps[1], pos, ang, mir))
                    crs.append(port.port_caps[2])
                if port.port_uprights:
                    if up is None:
                        up = self.upright_capsules(pos, ang, mir)
                    sel = list(port.port_uprights)
                    c0s.append(up[0][:, sel, :])
                    c1s.append(up[1][:, sel, :])
                    crs.append(up[2][sel])
                a[:, k, :] = march_entry(t[:, k, :], p[:, k, :], np.concatenate(c0s, axis=1),
                                         np.concatenate(c1s, axis=1), np.concatenate(crs))
        return a, b, length

    def dlabel_capsules(self, a: np.ndarray, b: np.ndarray, label_t: np.ndarray | None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Capsules of the distance labels, given the visible link ends ``a`` / ``b``
        (``(N, K, 2)``) and each label's position along its link ``label_t`` (``(N, K)``,
        0 = protein end, 1 = ligand end; default the midpoint)."""
        idx = np.nonzero(self._has_dlabel)[0]
        if label_t is None:
            t = np.full((len(a), len(idx)), 0.5)
        else:
            t = label_t[:, idx]
        centre = a[:, idx, :] + (b[:, idx, :] - a[:, idx, :]) * t[:, :, None]
        w, h = self._dlabel[idx, 0], self._dlabel[idx, 1]
        half = np.maximum(0.0, (w - h) / 2.0)
        off = np.stack([half, np.zeros_like(half)], axis=-1)[None, :, :]
        return centre - off, centre + off, h / 2.0 + 1.0

    def solids(self, pos: np.ndarray, ang: np.ndarray, mir: np.ndarray, label_t: np.ndarray | None = None,
               links: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Everything of the body that must not be overlapped: its shapes, its
        upright boxes, its name label and the distance labels on its drawn links.

        Returns ``(p0, p1, r, is_dlabel)`` with shapes ``(N, S', 2)``,
        ``(N, S', 2)``, ``(S',)``, ``(S',)``.
        """
        n = len(pos)
        p0 = [self._xf(self.shapes_p0, pos, ang, mir)]
        p1 = [self._xf(self.shapes_p1, pos, ang, mir)]
        rr = [self.shapes_r]
        flag = [np.zeros(len(self.shapes_r), dtype=bool)]
        if len(self.upright_pts):
            u0, u1, ur = self.upright_capsules(pos, ang, mir)
            p0.append(u0)
            p1.append(u1)
            rr.append(ur)
            flag.append(np.zeros(len(ur), dtype=bool))
        if self.label is not None:
            c = self.label_centre(pos, ang, mir)
            assert c is not None
            half = max(0.0, (self.label.width - self.label.height) / 2.0)
            off = np.array([half, 0.0])
            p0.append((c - off)[:, None, :])
            p1.append((c + off)[:, None, :])
            rr.append(np.array([self.label.height / 2.0 + self.label.pad]))
            flag.append(np.zeros(1, dtype=bool))
        if self._has_dlabel.any():
            a, b, _ = links if links is not None else self.link_segments(pos, ang, mir)
            d0, d1, dr = self.dlabel_capsules(a, b, label_t)
            p0.append(d0)
            p1.append(d1)
            rr.append(dr)
            flag.append(np.ones(len(dr), dtype=bool))
        if n == 0:
            total = sum(len(r) for r in rr)
            return np.zeros((0, total, 2)), np.zeros((0, total, 2)), np.concatenate(rr), np.concatenate(flag)
        return np.concatenate(p0, axis=1), np.concatenate(p1, axis=1), np.concatenate(rr), np.concatenate(flag)

    def ligand_samples(self, pos: np.ndarray, ang: np.ndarray, mir: np.ndarray, some: slice = slice(0, None)
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Points (with radii) at which the body proper -- shapes, upright boxes
        and name label, not the distance labels -- is tested against the ligand
        field, or ``some`` of them. Returns ``(points (N, Q, 2), radius (Q,))``."""
        pts = [self._xf(self._samples[some], pos, ang, mir)]
        rr = [self._sample_r[some]]
        extra: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        if len(self.upright_pts):
            extra.append(self.upright_capsules(pos, ang, mir))
        if self.label is not None:
            c = self.label_centre(pos, ang, mir)
            assert c is not None
            half = max(0.0, (self.label.width - self.label.height) / 2.0)
            off = np.array([half, 0.0])
            extra.append(((c - off)[:, None, :], (c + off)[:, None, :],
                          np.array([self.label.height / 2.0 + self.label.pad])))
        for c0, c1, cr in extra:
            for t in (0.0, 0.5, 1.0) if some == slice(0, None) else (0.5,):
                pts.append(c0 + (c1 - c0) * t)
                rr.append(cr)
        return np.concatenate(pts, axis=1), np.concatenate(rr)

    def self_clear(self, pos: np.ndarray, ang: np.ndarray, mir: np.ndarray) -> np.ndarray:
        """``(N,)`` booleans: the name label does not sit on the body's own drawing.

        The label stays upright while the body turns, so whether it clears the
        body's atoms depends on the pose.
        """
        n = len(pos)
        if self.label is None or n == 0:
            return np.ones(n, dtype=bool)
        from ..geometry import segments_distance

        c = self.label_centre(pos, ang, mir)
        assert c is not None
        half = max(0.0, (self.label.width - self.label.height) / 2.0)
        off = np.array([half, 0.0])
        l0, l1 = (c - off)[:, None, :], (c + off)[:, None, :]
        lr = self.label.height / 2.0 + self.label.pad
        ok = np.ones(n, dtype=bool)
        if len(self.upright_pts):
            u0, u1, ur = self.upright_capsules(pos, ang, mir)
            ok &= ((segments_distance(l0, l1, u0, u1) - ur[None, :] - lr) >= 0.5).all(axis=1)
        if len(self.shapes_r):
            s0 = self._xf(self.shapes_p0, pos, ang, mir)
            s1 = self._xf(self.shapes_p1, pos, ang, mir)
            anchor = self._xf(np.array([self.label.anchor], dtype=float), pos, ang, mir)
            # Bonds that end at the anchor atom necessarily come close; judge the rest.
            touches = (segments_distance(anchor, anchor, s0, s1) - self.shapes_r[None, :]) <= 0.5
            gap = segments_distance(l0, l1, s0, s1) - self.shapes_r[None, :] - lr
            ok &= ((gap >= 0.5) | touches).all(axis=1)
        return ok

def march_entry(start: np.ndarray, end: np.ndarray, c0: np.ndarray, c1: np.ndarray, cr: np.ndarray,
                steps: int = 28) -> np.ndarray:
    """Where the segment ``start -> end`` first enters a union of capsules.

    ``start`` / ``end`` are ``(N, 2)``; the capsules are ``c0`` / ``c1`` of shape
    ``(N, H, 2)`` with radii ``cr`` ``(H,)``. The segment is sampled and the
    crossing refined by linear interpolation, which is exact enough for drawing
    and fully vectorised. Returns ``end`` where the segment never enters.
    """
    n = len(start)
    if n == 0:
        return np.zeros((0, 2))
    ts = np.linspace(0.0, 1.0, steps)
    sx, sy = start[:, 0][:, None], start[:, 1][:, None]                               # (N, 1)
    px, py = sx + (end[:, 0][:, None] - sx) * ts, sy + (end[:, 1][:, None] - sy) * ts  # (N, T)
    px, py = px[:, :, None], py[:, :, None]                                           # (N, T, 1)
    cx, cy = c0[:, :, 0][:, None, :], c0[:, :, 1][:, None, :]                         # (N, 1, H)
    abx, aby = c1[:, :, 0][:, None, :] - cx, c1[:, :, 1][:, None, :] - cy
    l2 = np.maximum(abx * abx + aby * aby, 1e-12)
    tt = ((px - cx) * abx + (py - cy) * aby) / l2                                     # (N, T, H)
    np.clip(tt, 0.0, 1.0, out=tt)
    dx, dy = px - (cx + abx * tt), py - (cy + aby * tt)
    d = (np.sqrt(dx * dx + dy * dy) - cr[None, None, :]).min(axis=2)                  # (N, T)
    inside = d <= 0.0
    has = inside.any(axis=1)
    first = np.argmax(inside, axis=1)
    prev = np.maximum(first - 1, 0)
    rows = np.arange(n)
    d0, d1 = d[rows, prev], d[rows, first]
    frac = np.where((d0 > 0.0) & (d0 - d1 > 1e-12), d0 / np.maximum(d0 - d1, 1e-12), 0.0)
    t_hit = np.where(has, (prev + frac * (first - prev)) / (steps - 1), 1.0)
    return start + (end - start) * t_hit[:, None]
