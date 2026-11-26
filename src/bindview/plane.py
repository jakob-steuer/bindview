"""Where the residues of one side of an interface lie: a plane through them, and how true its picture is.

The column of an interface figure (:mod:`bindview.interface`) orders a line. That
is the right picture of a peptide and the wrong one of a paratope: three loops
that are far apart in the sequence and side by side in space have no order in
which neighbours in the column are neighbours in space. For such an interface
the residues are drawn **where they lie**: projected onto the plane through
them, at one scale, seen from the partner's side.

A picture that looks like the arrangement in space is measured by its reader.
So the arrangement is not assumed to be faithful, it is measured
(:func:`fidelity`), the figure says what was found, and an interface that is
too curved for any plane is not drawn this way at all (:data:`KEPT_MIN`,
:data:`ERROR_MAX`).

Nothing here needs more than NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .geometry import Point

__all__ = ["NEAR", "APART", "KEPT_MIN", "ERROR_MAX", "STRETCH_GAP", "MIN_RESIDUES", "Fidelity", "Projection",
           "Arrangement", "Placed", "stretches", "project", "fidelity", "separate", "overlapping", "passes_through",
           "SCALE_BAR"]

#: Two residues are neighbours when their centres are this close (ångström): the customary distance
#: for a contact between two residues ...
NEAR = 8.0
#: ... and apart when they are further than this. Between the two a pair is neither: a pair at 8.1 Å
#: that is drawn at 7.9 Å is not an error of the picture.
APART = 10.0
#: A projection is drawn only if it keeps this share of the neighbour relations ...
KEPT_MIN = 0.90
#: ... and its distances are off by no more than this (stress; 0.05 is customarily "good", 0.10 "fair", 0.20 "poor").
ERROR_MAX = 0.10
#: Residues of the chain that are not at the interface and may lie between two that are, for the two
#: still to be one stretch: one turn of a helix, so that one face of a helix is one stretch.
STRETCH_GAP = 3
#: A plane needs three points.
MIN_RESIDUES = 3
#: The median distance between a box and its nearest neighbour, in units of the pitch of a column:
#: what makes room between the boxes for the residues that touch them.
ROOM = 2.2
#: ... and the least that distance is taken to be (ångström) when the scale is set.
MIN_SPACING = 2.5
#: Clear space kept between two boxes that had to be pushed apart.
BOX_GAP = 5.0
#: The length, in ångström, of the scale bar a figure with boxes on a plane carries.
SCALE_BAR = 5.0


def stretches(places: Sequence[int], gap: int = STRETCH_GAP) -> list[tuple[int, int]]:
    """Runs of residues that are close in the chain, as ``(first, last)`` indices into ``places``.

    ``places`` are the positions of residues in their chain, ascending. Two
    consecutive ones are in one stretch when no more than ``gap`` residues lie
    between them.
    """
    out: list[tuple[int, int]] = []
    start = 0
    for k in range(1, len(places) + 1):
        if k == len(places) or places[k] - places[k - 1] - 1 > gap:
            out.append((start, k - 1))
            start = k
    return out


@dataclass(frozen=True)
class Fidelity:
    """How faithfully positions in a figure reproduce positions in space.

    ``kept`` is the share of neighbour relations the figure has right, of the
    ``pairs`` that are neighbours (within :data:`NEAR`) in the figure or in
    space: a ``false`` neighbour is drawn within :data:`NEAR` and lies more
    than :data:`APART` away, a ``lost`` one the other way round. ``error`` is
    the root-mean-square deviation of all distances in the figure from those
    in space, relative to the root-mean-square distance (stress).
    """

    kept: float
    false: int
    lost: int
    pairs: int
    error: float

    @property
    def ok(self) -> bool:
        return self.kept >= KEPT_MIN and self.error <= ERROR_MAX

    def as_dict(self) -> dict[str, Any]:
        return {"neighbours_kept": round(self.kept, 3), "false_neighbours": self.false,
                "lost_neighbours": self.lost, "neighbour_pairs": self.pairs, "distance_error": round(self.error, 3)}


def _pairs(points: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    return d[np.triu_indices(len(points), 1)]


def fidelity(space: np.ndarray, figure: np.ndarray, near: float = NEAR, apart: float = APART) -> Fidelity:
    """Compare positions in a figure with positions in space.

    ``space`` is ``(n, 3)`` in ångström; ``figure`` is ``(n, 2)`` **in
    ångström too**: page coordinates divided by the scale of the figure. The
    scale is the figure's own and is not fitted: a reader has one scale.
    """
    space = np.asarray(space, dtype=float).reshape(-1, 3)
    figure = np.asarray(figure, dtype=float).reshape(-1, 2)
    if len(space) != len(figure):
        raise ValueError("as many positions in the figure as in space")
    if len(space) < 2:
        return Fidelity(1.0, 0, 0, 0, 0.0)
    a, b = _pairs(space), _pairs(figure)
    pairs = int(((a <= near) | (b <= near)).sum())
    false = int(((b <= near) & (a > apart)).sum())
    lost = int(((a <= near) & (b > apart)).sum())
    total = float((a * a).sum())
    error = float(np.sqrt(((b - a) ** 2).sum() / total)) if total > 0.0 else 0.0
    return Fidelity(1.0 - (false + lost) / pairs if pairs else 1.0, false, lost, pairs, error)


@dataclass(frozen=True)
class Projection:
    """Points projected onto the plane through them.

    ``xy`` are figure coordinates in ångström (x to the right, y **down**, as
    on a page), seen from the side the ``normal`` points to. ``rms`` is the
    root-mean-square distance of the points from the plane; ``tilt`` the angle
    (degrees) between the normal and the direction the plane was oriented by,
    ``None`` when there was none.
    """

    xy: np.ndarray
    normal: np.ndarray
    rms: float
    tilt: float | None


def project(points: np.ndarray, toward: "np.ndarray | None" = None) -> Projection:
    """Project ``points`` (``(n, 3)``, at least three) onto their least-squares plane.

    Of all planes this is the one whose projection keeps the most of the
    squared distances between the points. ``toward`` is a direction from the
    points to whoever looks at them (the partner at an interface): the plane
    is seen from that side, so the picture is not a mirror image. In the plane
    the long axis of the points is upright, with the first point nearer the
    top than the last.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < MIN_RESIDUES:
        raise ValueError(f"a plane needs {MIN_RESIDUES} points, not {len(pts)}")
    centre = pts.mean(axis=0)
    rel = pts - centre
    _, _, axes = np.linalg.svd(rel, full_matrices=True)
    long_axis, normal = axes[0], axes[2]
    tilt = None
    if toward is not None and float(np.linalg.norm(toward)) > 1e-9:
        look = np.asarray(toward, dtype=float) / float(np.linalg.norm(toward))
        if float(normal @ look) < 0.0:
            normal = -normal
        tilt = float(np.degrees(np.arccos(np.clip(float(normal @ look), -1.0, 1.0))))
    if float(rel[0] @ long_axis) < float(rel[-1] @ long_axis):
        long_axis = -long_axis                       # the first point nearer the top
    right = np.cross(long_axis, normal)              # right x up = normal: what is seen from the normal's side
    xy = np.column_stack([rel @ right, -(rel @ long_axis)])
    return Projection(xy, normal, float(np.sqrt(((rel @ normal) ** 2).mean())), tilt)


def overlapping(centres: np.ndarray, half_w: float, half_h: float, gap: float = 0.0) -> int:
    """How many pairs of equal boxes at ``centres`` overlap (or come closer than ``gap``)."""
    at = np.asarray(centres, dtype=float).reshape(-1, 2)
    if len(at) < 2:
        return 0
    d = np.abs(at[:, None, :] - at[None, :, :])
    close = (d[..., 0] < 2.0 * half_w + gap - 1e-6) & (d[..., 1] < 2.0 * half_h + gap - 1e-6)
    return int(np.triu(close, 1).sum())


def separate(centres: np.ndarray, half_w: float, half_h: float, gap: float = BOX_GAP,
             sweeps: int = 5000) -> np.ndarray:
    """Push equal boxes apart until no two overlap, each pair by the least it takes.

    Two boxes that overlap are moved apart along the axis on which that takes
    less, half the way each. Pairs are taken in a fixed order, so the result
    is the same every time. A handful of sweeps settles boxes at the spacing a
    figure has; thirty boxes thrown on one spot take a few thousand, which is
    what ``sweeps`` allows. Whether any still overlap is for the caller to
    ask (:func:`overlapping`): a figure never takes it on trust.
    """
    out = np.array(centres, dtype=float).reshape(-1, 2).copy()
    need = np.array([2.0 * half_w + gap, 2.0 * half_h + gap])
    n = len(out)
    for _ in range(sweeps):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = out[j] - out[i]
                inside = need - np.abs(d)
                if inside[0] <= 1e-9 or inside[1] <= 1e-9:
                    continue
                axis = 0 if inside[0] < inside[1] else 1
                way = 1.0 if d[axis] > 0.0 or (d[axis] == 0.0) else -1.0
                out[i, axis] -= way * inside[axis] / 2.0
                out[j, axis] += way * inside[axis] / 2.0
                moved = True
        if not moved:
            break
    return out


def passes_through(a: Point, b: Point, box: "tuple[float, float, float, float]", margin: float = 1.0) -> bool:
    """Does the segment from ``a`` to ``b`` run through ``box`` (grown by ``margin``)?"""
    lo, hi = 0.0, 1.0
    for start, end, b0, b1 in ((a[0], b[0], box[0] - margin, box[2] + margin),
                               (a[1], b[1], box[1] - margin, box[3] + margin)):
        d = end - start
        if abs(d) < 1e-12:
            if start < b0 or start > b1:
                return False
            continue
        t0, t1 = (b0 - start) / d, (b1 - start) / d
        lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
    return hi > lo


@dataclass(frozen=True)
class Placed:
    """The boxes of an arrangement on a page: what a figure says about its own truth."""

    scale: float                    # page units per ångström
    fidelity: Fidelity              # of the boxes as they are drawn
    bare: Fidelity                  # of the projection itself, before any box was moved
    rms: float                      # of the residue centres from the plane, ångström
    tilt: float | None
    moved: int                      # boxes that had to be pushed off their place
    covered: int = 0                # pairs of boxes that lie on top of each other as drawn

    @property
    def ok(self) -> bool:
        """Is this a figure to draw: faithful, and no box on top of another?"""
        return self.fidelity.ok and self.covered == 0

    def as_dict(self) -> dict[str, Any]:
        # The scale in full: a figure drawn again from its graph is measured with it.
        out = {**self.fidelity.as_dict(), "px_per_angstrom": round(self.scale, 6),
               "out_of_plane_rms": round(self.rms, 2), "boxes_moved": self.moved,
               **({"boxes_covering_each_other": self.covered} if self.covered else {}),
               "projection": self.bare.as_dict(),
               "thresholds": {"neighbours_kept_min": KEPT_MIN, "distance_error_max": ERROR_MAX,
                              "neighbour_within": NEAR, "apart_beyond": APART}}
        if self.tilt is not None:
            out["partner_off_normal_deg"] = round(self.tilt, 1)
        return out


@dataclass(frozen=True)
class Arrangement:
    """The residues of a chain where they lie on the plane through them.

    ``xy`` (figure coordinates in ångström) and ``space`` (their centres in
    space) are one per residue, in the order of the chain. ``joins`` are the
    positions (counted from 1) after which a dotted line joins two boxes: the
    two are in one stretch, with residues between them that are not drawn.
    """

    xy: tuple[Point, ...]
    space: tuple[tuple[float, float, float], ...]
    joins: frozenset[int]
    rms: float
    tilt: float | None
    bare: Fidelity

    @classmethod
    def of(cls, centres: np.ndarray, toward: "np.ndarray | None", joins: Sequence[int] = ()) -> "Arrangement":
        proj = project(centres, toward)
        space = np.asarray(centres, dtype=float).reshape(-1, 3)
        return cls(tuple((float(x), float(y)) for x, y in proj.xy),
                   tuple((float(p[0]), float(p[1]), float(p[2])) for p in space),
                   frozenset(joins), proj.rms, proj.tilt, fidelity(space, proj.xy))

    @property
    def spacing(self) -> float:
        """The median distance between a residue and its nearest neighbour on the plane, ångström."""
        xy = np.array(self.xy, dtype=float)
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2) + np.eye(len(xy)) * 1e9
        return float(np.median(d.min(axis=1)))

    def scale(self, pitch: float) -> float:
        """Page units per ångström at a column pitch of ``pitch`` page units.

        The median distance from a residue to its nearest neighbour on the plane becomes
        :data:`ROOM` pitches. Residues that lie one above the other fall on one spot; where that is
        the usual case the spacing is taken as :data:`MIN_SPACING`, or the figure would have no end.
        """
        return pitch * ROOM / max(self.spacing, MIN_SPACING)

    def place(self, pitch: float, half_w: float, half_h: float) -> tuple[list[Point], Placed]:
        """Box centres on the page, and what became of the arrangement on the way."""
        scale = self.scale(pitch)
        ideal = np.array(self.xy, dtype=float) * scale
        final = separate(ideal, half_w, half_h)
        return [(float(x), float(y)) for x, y in final], self.measure(final, scale, ideal, half_w, half_h)

    def measure(self, centres: "np.ndarray | Sequence[Point]", scale: float, ideal: "np.ndarray | None" = None,
                half_w: float = 0.0, half_h: float = 0.0) -> Placed:
        """What the boxes at ``centres`` (page units, any origin) say about the arrangement. With the
        half width and height of a box, whether two of them lie on top of each other is counted too."""
        final = np.array(centres, dtype=float).reshape(-1, 2)
        if ideal is None:
            ideal = np.array(self.xy, dtype=float) * scale
            ideal = ideal + (final.mean(axis=0) - ideal.mean(axis=0))
        moved = int((np.linalg.norm(final - ideal, axis=1) > 0.5).sum())
        return Placed(scale, fidelity(np.array(self.space, dtype=float), final / scale), self.bare, self.rms,
                      self.tilt, moved, overlapping(final, half_w, half_h) if half_w and half_h else 0)
