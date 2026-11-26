"""Seeded placement plus constrained refinement.

The problem: put every residue near the ligand atoms it contacts, with nothing
on top of anything else. A pocket's residues crowd one face of the ligand, so
"put each where it wants to be" piles them up, and plain force relaxation from
such a start gets stuck with labels trapped on the wrong side of the skeleton.

The engine therefore works in stages, all deterministic:

1. **Seeded placement.** Bodies are taken most-constrained first. For each one
   a few thousand candidate poses around its ligand atoms are scored with one
   cost function (link lengths, lines crossing ligand bonds or running over
   ligand ink, lines crossing other lines, distance from the ligand) and the
   cheapest pose that collides with nothing already placed is taken. Collision
   is a hard constraint, never a penalty, so an accepted pose is overlap-free
   by construction.
2. **Re-placement passes.** Each body in turn is lifted and re-seated against
   all the others, until a full pass changes nothing. This undoes the myopia of
   the greedy order.
3. **Ejection chains.** A body that is still badly placed (a long link, a line
   through the skeleton) usually is so because others took the room it needed
   first. It is tried at the poses it would like; whoever is in the way is
   lifted and re-seated; the exchange is kept if the total cost falls.
4. **Polish.** A pattern search nudges every body by shrinking steps (and
   rotates fragments slightly) under the same hard constraints plus a soft
   preference for even spacing, which is what makes the result look composed
   rather than merely legal.

A body that cannot be seated anywhere within reach is reported as unplaced; the
caller decides what to do with a diagram that could not be laid out.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

import numpy as np

from ..geometry import capsule_gaps, points_segments_distance, segments_distance
from .body import Body, Port, Pose
from .field import LigandField, segment_cross_matrix

__all__ = ["LayoutParams", "LayoutResult", "layout_bodies", "layout_key", "audit_layout"]

# Where along its link a distance label may sit, in order of preference.
_LABEL_T = np.array([0.5, 0.4, 0.6, 0.31, 0.69])


@dataclass(frozen=True)
class LayoutParams:
    unit: float = 30.0               # one bond length; makes the cost terms dimensionless
    solid_gap: float = 5.0           # minimum clearance between two bodies
    line_gap: float = 3.0            # minimum clearance between a link line and another body
    label_ligand_gap: float = 3.0    # minimum clearance between a distance label and ligand ink
    reach: float = 7.0               # candidate search radius, in units
    improve_passes: int = 5
    eject_threshold: float = 1.6     # bodies this much costlier than their ligand-only best get an ejection attempt
    eject_candidates: int = 5
    eject_rounds: int = 2
    max_candidates: int = 4000       # feasible candidate poses kept per body
    eject_max_blockers: int = 4
    give_up: int = 0                 # with this many unplaced after a round of ejections the layout is left as it is (0: never)
    w_length: float = 1.0
    w_short: float = 4.0             # a link shorter than ideal costs this much more than a long one
    w_bond_cross: float = 3.0
    w_sight_cross: float = 0.5
    w_line_cross: float = 1.4
    w_ink: float = 2.0
    ink_gap: float = 2.5
    w_hug: float = 0.12
    w_axis: float = 0.35
    w_soft: float = 0.6
    soft_gap: float = 10.0
    w_blocked_sight: float = 0.8
    w_near_line: float = 1.0
    w_self: float = 3.0              # a link running over its own residue's drawing
    w_assoc: float = 1.2             # a line-less contact sitting nearer to some other ligand atom
    w_fan: float = 1.5               # two links into one site at (nearly) the same angle


@dataclass
class LayoutResult:
    poses: dict[Any, Pose]
    unplaced: list[Any]
    cost: float
    overlaps: int
    line_crossings: int
    bond_crossings: int
    max_stretch: float
    label_t: dict[Any, tuple[float, ...]] = field(default_factory=dict)
    link_hits: int = 0              # links passing through a body they do not belong to
    left: int = 0                   # bodies without a place after the first round of ejections
    abandoned: bool = False         # left as it was then (``LayoutParams.give_up``)

    @property
    def ok(self) -> bool:
        return not self.unplaced and self.overlaps == 0 and self.link_hits == 0


@dataclass
class _World:
    """A placed body's geometry on the page."""

    sp0: np.ndarray
    sp1: np.ndarray
    sr: np.ndarray
    la: np.ndarray              # drawn link segments
    lb: np.ndarray
    lt: np.ndarray              # the ligand points they aim at


@dataclass
class _Geo:
    """Page geometry of ``N`` poses of one body."""

    pos: np.ndarray             # (N, 2) body origins
    p0: np.ndarray              # (N, S, 2) solids: the body proper first, then distance labels
    p1: np.ndarray
    r: np.ndarray               # (S,)
    a: np.ndarray               # (N, K, 2) visible link ends (protein side)
    b: np.ndarray               # (N, K, 2) visible link ends (ligand side)
    ports: np.ndarray           # (N, K, 2) port centres

    def __getitem__(self, sl: "slice | np.ndarray") -> "_Geo":
        return _Geo(self.pos[sl], self.p0[sl], self.p1[sl], self.r, self.a[sl], self.b[sl], self.ports[sl])


@dataclass
class _Batch:
    """The feasible candidate poses of one body, cheapest first, with their
    geometry. Computed once: the ligand does not move."""

    pos: np.ndarray
    ang: np.ndarray
    mir: np.ndarray
    base: np.ndarray
    label_t: np.ndarray
    geo: _Geo

    def __len__(self) -> int:
        return len(self.base)


class _Engine:
    def __init__(self, fld: LigandField, bodies: Sequence[Body], params: LayoutParams) -> None:
        self.field = fld
        self.bodies = list(bodies)
        self.p = params
        self.pos: dict[int, np.ndarray] = {}
        self.ang: dict[int, float] = {}
        self.mir: dict[int, bool] = {}
        self.lt: dict[int, np.ndarray] = {}
        self.world: dict[int, _World] = {}
        self._cand_cache: dict[tuple[int, bool], _Batch] = {}
        #: A number for every pose that a body has been placed in, the same one when it comes back to it.
        self._tags: list[dict[tuple, int]] = [{} for _ in self.bodies]
        self.tag: dict[int, int] = {}
        #: For the candidates of a body, what is known of each against each placed body (0: nothing, 1: they are
        #: clear of each other, 2: they are not), and the pose of the placed body that it is known for.
        self._known: dict[tuple[int, bool], tuple[np.ndarray, np.ndarray]] = {}
        #: What a body costs against the ligand alone in a pose it was placed in, whether it may stand there, and its geometry.
        self._priced: list[dict[int, tuple[float, bool, _Geo]]] = [{} for _ in self.bodies]

    # -- bookkeeping -------------------------------------------------------
    def _set(self, i: int, pos: np.ndarray, ang: float, mir: bool, label_t: np.ndarray) -> None:
        body = self.bodies[i]
        self.pos[i] = np.array(pos, dtype=float)
        self.ang[i] = float(ang)
        self.mir[i] = bool(mir)
        self.lt[i] = np.array(label_t, dtype=float)
        P, A, M = self.pos[i][None, :], np.array([ang]), np.array([mir])
        if body.n_ports:
            links = body.link_segments(P, A, M)
            d = body._drawn
            la, lb, lt = links[0][0][d], links[1][0][d], body._targets[d]
        else:
            links = None
            la = lb = lt = np.zeros((0, 2))
        p0, p1, r, _ = body.solids(P, A, M, self.lt[i][None, :], links)
        self.world[i] = _World(p0[0], p1[0], r, la, lb, lt)
        self.tag[i] = self._tags[i].setdefault((self.pos[i].tobytes(), self.ang[i], self.mir[i], self.lt[i].tobytes()),
                                               len(self._tags[i]) + 1)

    def _unset(self, i: int) -> None:
        for d in (self.pos, self.ang, self.mir, self.lt, self.world, self.tag):
            d.pop(i, None)

    def _snapshot(self, ids: Sequence[int]) -> dict[int, tuple]:
        # (Nothing here is ever changed in place: a pose that is set is a new one.)
        return {i: (self.pos[i], self.ang[i], self.mir[i], self.lt[i], self.world[i], self.tag[i])
                for i in ids if i in self.world}

    def _restore(self, snap: dict[int, tuple], ids: Sequence[int]) -> None:
        for i in ids:
            self._unset(i)
        for i, state in snap.items():
            self.pos[i], self.ang[i], self.mir[i], self.lt[i], self.world[i], self.tag[i] = state

    def _others(self, exclude: int, among: "Sequence[int] | None" = None
                ) -> tuple[_World, np.ndarray, np.ndarray, list[int]]:
        """Concatenated geometry of every placed body but ``exclude`` (or of those
        ``among`` them); the owner (as an index into the returned id list) of
        each solid and of each line; and the ids."""
        ids = [j for j in (sorted(self.world) if among is None else among) if j != exclude]
        if not ids:
            z = np.zeros((0, 2))
            return _World(z, z, np.zeros(0), z, z, z), np.zeros(0, dtype=int), np.zeros(0, dtype=int), ids
        ws = [self.world[j] for j in ids]
        sowner = np.repeat(np.arange(len(ids)), [len(w.sr) for w in ws])
        lowner = np.repeat(np.arange(len(ids)), [len(w.la) for w in ws])
        return (_World(np.concatenate([w.sp0 for w in ws]), np.concatenate([w.sp1 for w in ws]),
                       np.concatenate([w.sr for w in ws]), np.concatenate([w.la for w in ws]),
                       np.concatenate([w.lb for w in ws]), np.concatenate([w.lt for w in ws])), sowner, lowner, ids)

    # -- candidates --------------------------------------------------------
    def _candidates(self, body: Body, wide: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.p
        reach = p.reach * p.unit * (1.8 if wide else 1.0)
        if body.kind == "disc":
            lo = body.radius + body.ligand_clearance
            targets = body._targets if body.n_ports else self.field.atom_xy
            ideal = float(body._ideal.max()) if body.n_ports else lo
            cells = self.field.cells(lo, lo + reach, stride=2)
            if len(cells) == 0:
                return np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=bool)
            d = np.linalg.norm(cells[:, None, :] - targets[None, :, :], axis=2).min(axis=1)
            cells = cells[d <= ideal + reach]
            return cells, np.zeros(len(cells)), np.zeros(len(cells), dtype=bool)
        # Fragment: sweep the primary port around its target and try a fan of orientations.
        port = body.ports[0]
        mults = (1.0, 1.15, 1.32, 1.52, 1.78, 2.1, 2.5, 3.0) + ((3.8, 4.8, 6.0) if wide else ())
        phis = np.deg2rad(np.arange(0.0, 360.0, 7.5))
        deltas = np.deg2rad(np.array([0.0, 20.0, -20.0, 42.0, -42.0, 68.0, -68.0])) \
            if body.rotatable else np.array([0.0])
        mirrors = (False, True) if body.mirrorable else (False,)
        pos_l, ang_l, mir_l = [], [], []
        tx, ty = port.target
        for mlt in mults:
            length = port.ideal * mlt
            if length > reach + port.ideal:
                break
            for phi in phis:
                px, py = tx + length * math.cos(phi), ty + length * math.sin(phi)
                for mirror in mirrors:
                    ax, ay = body.axis[0], (-body.axis[1] if mirror else body.axis[1])
                    base = math.atan2(ay, ax)
                    lx, ly = port.local[0], (-port.local[1] if mirror else port.local[1])
                    for dl in deltas:
                        theta = (phi + dl) - base if body.rotatable else 0.0
                        c, s = math.cos(theta), math.sin(theta)
                        # Body origin such that the port lands on (px, py).
                        pos_l.append((px - (lx * c - ly * s), py - (lx * s + ly * c)))
                        ang_l.append(theta)
                        mir_l.append(mirror)
        return (np.array(pos_l, dtype=float).reshape(-1, 2), np.array(ang_l, dtype=float),
                np.array(mir_l, dtype=bool))

    def _geo(self, body: Body, pos: np.ndarray, ang: np.ndarray, mir: np.ndarray, label_t: np.ndarray,
             links: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> _Geo:
        n = len(pos)
        if body.n_ports:
            if links is None:
                links = body.link_segments(pos, ang, mir)
            a, b = links[0], links[1]
            ports = body.ports_world(pos, ang, mir)
        else:
            a = b = ports = np.zeros((n, 0, 2))
        p0, p1, r, _ = body.solids(pos, ang, mir, label_t, links)
        return _Geo(pos, p0, p1, r, a, b, ports)

    def _batch(self, i: int, wide: bool = False) -> _Batch:
        """Feasible candidate poses of body ``i`` scored against the ligand,
        cheapest first (cached: the ligand does not move, so this never changes)."""
        key = (i, wide)
        hit = self._cand_cache.get(key)
        if hit is None:
            body = self.bodies[i]
            pos, ang, mir = self._candidates(body, wide)
            # Throw away what the ligand alone rules out, cheaply, before the full scoring.
            for some in (slice(0, None, 6), slice(0, None)):
                # (A sixth of the points rules most of them out.)
                if len(pos):
                    pts, rr = body.ligand_samples(pos, ang, mir, some)
                    keep = ((self.field.sdf(pts) - rr[None, :]) >= body.ligand_clearance - 1e-9).all(axis=1)
                    pos, ang, mir = pos[keep], ang[keep], mir[keep]
            if len(pos):
                keep = body.self_clear(pos, ang, mir)
                pos, ang, mir = pos[keep], ang[keep], mir[keep]
            links = body.link_segments(pos, ang, mir) if body.n_ports and len(pos) else None
            if links is not None:
                keep = self._links_may(body, pos, ang, mir, links)
                pos, ang, mir = pos[keep], ang[keep], mir[keep]
                links = (links[0][keep], links[1][keep], links[2][keep])
            base, ok, lt = self._base(body, pos, ang, mir, checked=True, links=links)
            idx = np.nonzero(ok)[0]
            idx = idx[np.argsort(_classes(base[idx]), kind="stable")][: self.p.max_candidates]
            pos, ang, mir, base, lt = pos[idx], ang[idx], mir[idx], base[idx], lt[idx]
            if links is not None:
                links = (links[0][idx], links[1][idx], links[2][idx])
            hit = _Batch(pos, ang, mir, base, lt, self._geo(body, pos, ang, mir, lt, links))
            self._cand_cache[key] = hit
        return hit

    # -- costs -------------------------------------------------------------
    def _links_may(self, body: Body, pos: np.ndarray, ang: np.ndarray, mir: np.ndarray,
                   links: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """``(N,)`` booleans: are the links of a pose long enough, and do they run through nothing they may not?"""
        a, b, length = links
        ok = (length >= body._minlen[None, :] - 1e-9).all(axis=1)
        for k, port in enumerate(body.ports):
            if port.drawn:
                ok &= ~self.field.boxes_crossed(a[:, k], b[:, k], body._targets[k], reach=float(body._reach[k]))
                if port.clear_of or port.clear_of_name:
                    ok &= ~_through_names(body, port, a[:, k], b[:, k], pos, ang, mir)
        return ok

    def _base(self, body: Body, pos: np.ndarray, ang: np.ndarray, mir: np.ndarray, checked: bool = False,
              links: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cost, feasibility and distance-label positions of poses against the ligand alone. ``checked``: that the
        body and its links are where they may be is known already."""
        p, u = self.p, self.p.unit
        n = len(pos)
        cost = np.zeros(n)
        label_t = np.full((n, body.n_ports), 0.5)
        if n == 0:
            return cost, np.zeros(0, dtype=bool), label_t
        if checked:
            ok = np.ones(n, dtype=bool)
        else:
            pts, rr = body.ligand_samples(pos, ang, mir)
            ok = ((self.field.sdf(pts) - rr[None, :]) >= body.ligand_clearance - 1e-9).all(axis=1)
            ok &= body.self_clear(pos, ang, mir)
        if body.n_ports:
            a, b, length = links if links is not None else body.link_segments(pos, ang, mir)
            rel = (length - body._ideal[None, :]) / u
            pen = np.where(rel >= 0.0, rel * rel, p.w_short * rel * rel)
            cost += p.w_length * (pen * body._weight[None, :]).sum(axis=1)
            if not checked:
                ok &= self._links_may(body, pos, ang, mir, (a, b, length))
            ports = body.ports_world(pos, ang, mir)
            ts = np.linspace(0.0, 1.0, 7)[None, :, None]
            own_solids = None
            # Beside a chain ligand several links of one residue end on one box and run side by side:
            # there their labels have to keep clear of each other as well.
            chain = len(self.field.unit_boxes) > 0
            labels: list[tuple[np.ndarray, float, float]] = []
            for k, port in enumerate(body.ports):
                if port.drawn:
                    cost += p.w_bond_cross * self.field.bond_crossings(a[:, k], b[:, k])
                    samples = a[:, k][:, None, :] + (b[:, k] - a[:, k])[:, None, :] * ts
                    free = max(0.55 * u, port.target_radius + 6.0)
                    far = np.linalg.norm(samples - body._targets[k][None, None, :], axis=2) > free
                    cost += p.w_ink * ((self.field.sdf(samples) < p.ink_gap) & far).sum(axis=1)
                    if port.label_size is not None:
                        # Slide the distance label along the link to the first spot clear of ligand ink.
                        w, h = port.label_size
                        half = max(0.0, (w - h) / 2.0)
                        centre = a[:, k][:, None, :] + (b[:, k] - a[:, k])[:, None, :] * _LABEL_T[None, :, None]
                        clear = np.minimum(
                            np.minimum(self.field.sdf(centre), self.field.sdf(centre + np.array([half, 0.0]))),
                            self.field.sdf(centre - np.array([half, 0.0]))) - (h / 2.0 + 1.0)
                        good = clear >= p.label_ligand_gap
                        for other, ow, oh in labels:
                            good &= ~((np.abs(centre[:, :, 0] - other[:, None, 0]) < (w + ow) / 2.0 + 1.0)
                                      & (np.abs(centre[:, :, 1] - other[:, None, 1]) < (h + oh) / 2.0 + 1.0))
                        if port.clear_of or port.clear_of_name:
                            # At a level of detail a residue has more lines, and they run closer together:
                            # the label of one does not stand on another line of the same residue.
                            off = np.array([half, 0.0])
                            for j, sibling in enumerate(body.ports):
                                if j != k and sibling.drawn:
                                    for t_i in range(len(_LABEL_T)):
                                        c = centre[:, t_i, :]
                                        gap = segments_distance(c - off, c + off, a[:, j], b[:, j]) - (h / 2.0 + 1.0)
                                        good[:, t_i] &= gap >= 1.0
                        if body.kind == "fragment":
                            if own_solids is None:
                                own_solids = body.solids(pos, ang, mir, None, (a, b, length))
                            s0, s1, sr, sx = own_solids
                            keep = ~sx
                            off = np.array([half, 0.0])
                            seated = np.zeros(n, dtype=bool)
                            for t_i in range(len(_LABEL_T)):
                                # (A pose whose label has a place already is not asked about the next ones.)
                                at = np.nonzero(good[:, t_i] & ~seated)[0]
                                if len(at):
                                    c = centre[at, t_i, :]
                                    gap = segments_distance((c - off)[:, None, :], (c + off)[:, None, :], s0[at][:, keep, :],
                                                            s1[at][:, keep, :]) - sr[None, keep] - (h / 2.0 + 1.0)
                                    good[at, t_i] = (gap >= 1.5).all(axis=1)
                                    seated[at] |= good[at, t_i]
                        choice = np.argmax(good, axis=1)
                        ok &= good.any(axis=1)
                        label_t[:, k] = _LABEL_T[choice]
                        cost += 0.05 * choice
                        if chain:
                            labels.append((centre[np.arange(n), choice], w, h))
                    if body.kind == "fragment":
                        if own_solids is None:
                            own_solids = body.solids(pos, ang, mir, None, (a, b, length))
                        s0, s1, sr, sx = own_solids
                        gap = segments_distance(a[:, k][:, None, :], b[:, k][:, None, :], s0, s1) - sr[None, :]
                        over = (gap < 0.5) & ~sx[None, :]
                        at, m = np.nonzero(over)
                        if len(at):
                            # (What the link ends on does not count.)
                            pk = ports[at, k]
                            over[at, m] = (segments_distance(pk, pk, s0[at, m], s1[at, m]) - sr[m]) > port.port_radius + 2.0
                        cost += p.w_self * over.sum(axis=1)
                else:
                    tgt = np.broadcast_to(body._targets[k], (n, 2))
                    cost += p.w_sight_cross * body._weight[k] * self.field.bond_crossings(ports[:, k], tgt)
            if body.rotatable:
                out = ports[:, 0, :] - body._targets[0][None, :]
                out /= np.maximum(np.linalg.norm(out, axis=1), 1e-9)[:, None]
                axis = body._xd(body.axis, ang, mir)
                dev = np.arccos(np.clip((out * axis).sum(axis=1), -1.0, 1.0))
                cost += p.w_axis * dev * dev
        if body.kind == "disc":
            lo = body.radius + body.ligand_clearance
            cost += p.w_hug * np.maximum(0.0, self.field.sdf(pos) - lo) / u
        sight = ~body._drawn
        if sight.any() and len(self.field.atom_xy) > int(sight.sum()):
            # A contact without a line is read off by proximity: the body should be
            # closer to the atoms it contacts than to any other atom of the ligand.
            mine = body._targets[sight]
            atoms = self.field.atom_xy
            other = np.linalg.norm(atoms[:, None, :] - mine[None, :, :], axis=2).min(axis=1) > 1.0
            if other.any():
                at = ports[:, sight, :].mean(axis=1) if body.kind != "disc" else pos
                d_own = np.linalg.norm(at[:, None, :] - mine[None, :, :], axis=2).min(axis=1)
                d_other = np.linalg.norm(at[:, None, :] - atoms[other][None, :, :], axis=2).min(axis=1)
                cost += p.w_assoc * float(body._weight[sight].sum()) * np.maximum(0.0, (d_own - d_other) / u + 0.2)
        return cost, ok, label_t

    def _blocked(self, i: int, g: _Geo, others: "tuple[_World, np.ndarray, np.ndarray, list[int]]", soft: bool = False,
                 every: bool = True) -> tuple[np.ndarray, "np.ndarray | None"]:
        """Which of ``others`` (columns) each pose of body ``i`` (rows) is not clear of; with ``soft`` also how far
        they are apart (``inf`` where that is further than anyone asks).

        Unless ``every`` is set, a pose is not looked at further once something is found in its way: then one
        ``True`` is all that its row says, and a row without one is clear of all of them.

        Most poses are nowhere near most placed bodies, so exact capsule distances are only computed for the
        pairs whose bounding circles come within reach of each other.
        """
        body, p = self.bodies[i], self.p
        w, sowner, lowner, ids = others
        n, nb = len(g.pos), len(ids)
        bad = np.zeros((n, nb), dtype=bool)
        gaps = np.full((n, nb), np.inf) if soft else None
        if n == 0 or not ids:
            return bad, gaps
        rows = np.arange(n)                          # the poses that are still looked at, as rows of the answer

        def left() -> bool:
            nonlocal g, rows
            if not every:
                free = ~bad[rows].any(axis=1)
                if not free.all():
                    rows, g = rows[free], g[free]
            return len(rows) > 0

        sp = body.n_proper
        nd = g.p0.shape[1] - sp
        mids = (w.sp0 + w.sp1) / 2.0
        ext = np.linalg.norm(w.sp1 - w.sp0, axis=1) / 2.0 + w.sr
        least = p.solid_gap - 1e-9
        # Body proper against placed solids.
        reach = body.proper_radius + max(p.solid_gap, p.soft_gap if soft else 0.0)
        pi, mj = np.nonzero(_apart(g.pos, mids) <= reach + ext[None, :])
        if len(pi) and sp:
            d = self._gap(g.p0[pi, :sp, :], g.p1[pi, :sp, :], g.r[:sp], w.sp0[mj], w.sp1[mj], w.sr[mj],
                          max(p.solid_gap, p.soft_gap if soft else 0.0))
            if gaps is not None:
                np.minimum.at(gaps, (pi, sowner[mj]), d)
            hit = d < least
            bad[pi[hit], sowner[mj[hit]]] = True
        # Distance labels against placed solids (they sit out on the links, away from the body).
        for k in range(sp, sp + nd):
            if not left():
                return bad, gaps
            c = (g.p0[:, k, :] + g.p1[:, k, :]) / 2.0
            half = float(np.linalg.norm(g.p1[0, k, :] - g.p0[0, k, :])) / 2.0 + g.r[k]
            pi, mj = np.nonzero(_apart(c, mids) <= half + p.solid_gap + ext[None, :])
            if len(pi):
                d = segments_distance(g.p0[pi, k, :], g.p1[pi, k, :], w.sp0[mj], w.sp1[mj]) - g.r[k] - w.sr[mj]
                if gaps is not None:
                    np.minimum.at(gaps, (rows[pi], sowner[mj]), d)
                hit = d < least
                bad[rows[pi[hit]], sowner[mj[hit]]] = True
        if len(w.la) and left():
            # Body and its labels against placed link lines.
            lmid = (w.la + w.lb) / 2.0
            lext = np.linalg.norm(w.lb - w.la, axis=1) / 2.0
            li, lj = np.nonzero(_apart(g.pos, lmid) <= body.proper_radius + p.line_gap + lext[None, :])
            if len(li) and sp:
                d = self._gap(g.p0[li, :sp, :], g.p1[li, :sp, :], g.r[:sp], w.la[lj], w.lb[lj], 0.0, p.line_gap)
                hit = d < p.line_gap - 1e-9
                bad[rows[li[hit]], lowner[lj[hit]]] = True
            for k in range(sp, sp + nd):
                if not left():
                    return bad, gaps
                c = (g.p0[:, k, :] + g.p1[:, k, :]) / 2.0
                half = float(np.linalg.norm(g.p1[0, k, :] - g.p0[0, k, :])) / 2.0 + g.r[k]
                li, lj = np.nonzero(points_segments_distance(c, w.la, w.lb) <= half + p.line_gap)
                if len(li):
                    d = segments_distance(g.p0[li, k, :], g.p1[li, k, :], w.la[lj], w.lb[lj]) - g.r[k]
                    hit = d < p.line_gap - 1e-9
                    bad[rows[li[hit]], lowner[lj[hit]]] = True
        for k, port in enumerate(body.ports):
            if port.drawn and left():
                hit = self._under(g.a[:, k, :], g.b[:, k, :], w, mids, ext, sowner, nb, p.line_gap)
                bad[rows] |= hit
        return bad, gaps

    @staticmethod
    def _gap(p0: np.ndarray, p1: np.ndarray, r: np.ndarray, q0: np.ndarray, q1: np.ndarray, rq: "np.ndarray | float",
             within: float) -> np.ndarray:
        """How far a body is from one capsule, for pairs of a pose of it and a capsule: ``p0``, ``p1`` are the
        capsules of the body in each of ``N`` poses (``(N, S, 2)``, radii ``(S,)``), ``q0``, ``q1`` the other
        capsule in each (``(N, 2)``, radii ``(N,)`` or one for all). Exact where it is less than ``within``; some
        number that is no less where it is not."""
        half = np.sqrt((p1[:, :, 0] - p0[:, :, 0]) ** 2 + (p1[:, :, 1] - p0[:, :, 1]) ** 2) / 2.0
        span = np.sqrt((q1[:, 0] - q0[:, 0]) ** 2 + (q1[:, 1] - q0[:, 1]) ** 2) / 2.0
        dx = (p0[:, :, 0] + p1[:, :, 0]) / 2.0 - ((q0[:, 0] + q1[:, 0]) / 2.0)[:, None]
        dy = (p0[:, :, 1] + p1[:, :, 1]) / 2.0 - ((q0[:, 1] + q1[:, 1]) / 2.0)[:, None]
        # (Two capsules are at least as far apart as their middles are, less what each reaches from its own.)
        at_least = np.sqrt(dx * dx + dy * dy) - half - span[:, None] - r[None, :] - np.reshape(rq, (-1, 1))
        m, s = np.nonzero(at_least <= within + 1e-6)
        out = np.where(at_least <= within + 1e-6, np.inf, at_least)
        if len(m):
            d = segments_distance(p0[m, s], p1[m, s], q0[m], q1[m])
            out[m, s] = d - r[s] - (rq[m] if isinstance(rq, np.ndarray) else rq)
        return out.min(axis=1)

    @staticmethod
    def _under(a: np.ndarray, b: np.ndarray, w: _World, mids: np.ndarray, ext: np.ndarray, sowner: np.ndarray, nb: int,
               limit: float) -> np.ndarray:
        """``(len(a), nb)`` booleans: does the line ``a[i] -> b[i]`` come within ``limit`` of a solid of that placed body?"""
        hits = np.zeros((len(a), nb), dtype=bool)
        # Cull on the distance of the solids' centres to the segment.
        mj, pi = np.nonzero(points_segments_distance(mids, a, b) <= ext[:, None] + limit)
        if len(pi):
            d = segments_distance(a[pi], b[pi], w.sp0[mj], w.sp1[mj]) - w.sr[mj]
            sel = d < limit - 1e-9
            hits[pi[sel], sowner[mj[sel]]] = True
        return hits

    def _crowding(self, i: int, g: _Geo, others: "tuple[_World, np.ndarray, np.ndarray, list[int]]",
                  gaps: "np.ndarray | None" = None) -> np.ndarray:
        """What poses cost against the bodies already placed, beside what rules a pose out; ``gaps``: how far each
        is from each of them, where even spacing counts."""
        body, p, u = self.bodies[i], self.p, self.p.unit
        w, sowner, _, ids = others
        n, nb = len(g.pos), len(ids)
        cost = np.zeros(n)
        if n == 0 or not ids:
            return cost
        if gaps is not None:
            short = np.maximum(0.0, (p.soft_gap - np.minimum(gaps, p.soft_gap)) / u)
            cost += p.w_soft * (short * short).sum(axis=1)
        nl = len(w.la)
        sights = [k for k, port in enumerate(body.ports) if not port.drawn]
        if sights:
            mids = (w.sp0 + w.sp1) / 2.0
            ext = np.linalg.norm(w.sp1 - w.sp0, axis=1) / 2.0 + w.sr
        for k, port in enumerate(body.ports):
            if not port.drawn:
                hits = self._under(g.ports[:, k, :], np.broadcast_to(body._targets[k], (n, 2)), w, mids, ext, sowner, nb, 0.0)
                cost += p.w_blocked_sight * body._weight[k] * hits.sum(axis=1)
            elif nl:
                ak, bk = g.a[:, k, :], g.b[:, k, :]
                cost += p.w_line_cross * segment_cross_matrix(ak, bk, w.la, w.lb).sum(axis=1)
                # Two links that run side by side (typically into the same atom)
                # read as one thick line; keep their outer halves apart.
                mid = (ak + bk) / 2.0
                cost += p.w_near_line * (points_segments_distance(mid, w.la, w.lb) < 2.0 * p.line_gap).sum(axis=1)
                if port.fan > 0.0:
                    # Links into the same site should fan out around it, not queue on one side.
                    tgt = body._targets[k]
                    same = np.linalg.norm(w.lt - tgt[None, :], axis=1) < 1.0
                    if same.any():
                        u1 = ak - tgt[None, :]
                        u1 /= np.maximum(np.linalg.norm(u1, axis=1), 1e-9)[:, None]
                        u2 = w.la[same] - tgt[None, :]
                        u2 /= np.maximum(np.linalg.norm(u2, axis=1), 1e-9)[:, None]
                        ang = np.arccos(np.clip(u1 @ u2.T, -1.0, 1.0))
                        tight = np.maximum(0.0, 1.0 - ang / port.fan)
                        cost += p.w_fan * (tight * tight).sum(axis=1)
        return cost

    def _collide(self, i: int, g: _Geo, soft: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Cost and feasibility of poses against the bodies already placed."""
        others = self._others(i)
        bad, gaps = self._blocked(i, g, others, soft)
        return self._crowding(i, g, others, gaps), ~bad.any(axis=1)

    def _blockers(self, i: int, g: _Geo) -> set[int]:
        """The placed bodies that make the single pose given infeasible."""
        others = self._others(i)
        bad, _ = self._blocked(i, g, others)
        return {others[3][j] for j in np.nonzero(bad[0])[0]}

    def _score(self, i: int, pos: np.ndarray, ang: np.ndarray, mir: np.ndarray, soft: bool = False
               ) -> tuple[np.ndarray, np.ndarray]:
        """Total cost of poses (``inf`` where infeasible) and their label positions."""
        body = self.bodies[i]
        links = body.link_segments(pos, ang, mir) if body.n_ports else None
        base, ok, lt = self._base(body, pos, ang, mir, links=links)
        extra, ok2 = self._collide(i, self._geo(body, pos, ang, mir, lt, links), soft)
        total = base + extra
        total[~(ok & ok2)] = np.inf
        return total, lt

    def _current(self, i: int) -> float:
        """What body ``i`` costs where it stands (``inf`` where it may not)."""
        hit = self._priced[i].get(self.tag[i])
        if hit is None:
            # Against the ligand, which does not move: once for every pose the body is placed in.
            body = self.bodies[i]
            pos, ang, mir = self.pos[i][None, :], np.array([self.ang[i]]), np.array([self.mir[i]])
            links = body.link_segments(pos, ang, mir) if body.n_ports else None
            base, ok, lt = self._base(body, pos, ang, mir, links=links)
            hit = self._priced[i][self.tag[i]] = (float(base[0]), bool(ok[0]), self._geo(body, pos, ang, mir, lt, links))
        own, stands, geo = hit
        extra, clear = self._collide(i, geo)
        return own + float(extra[0]) if stands and clear[0] else math.inf

    # -- search ------------------------------------------------------------
    def _best(self, i: int, wide: bool = False, upper: float = np.inf
              ) -> tuple[float, np.ndarray, float, bool, np.ndarray] | None:
        """Cheapest feasible candidate pose for body ``i`` against everything placed.

        Only poses cheaper than ``upper`` are of interest to the caller, and a
        pose can never cost less than its ligand-only cost, so the search stops
        as soon as the (sorted) candidates reach that bound. The answer is the
        same as an unbounded search whenever one below ``upper`` exists.
        """
        batch = self._batch(i, wide)
        n = len(batch)
        if n == 0:
            return None
        ids = [j for j in sorted(self.world) if j != i]
        known = self._known_of(i, wide, n, ids)
        best_cost, best_j = float(upper), -1
        chunk = 128 if self.bodies[i].kind == "fragment" else 768
        start = 0
        while start < n:
            if batch.base[start] >= best_cost:
                break                      # sorted: every remaining candidate is already worse
            j, cost = self._cheapest(i, batch, start, min(n, start + chunk), best_cost, ids, known)
            if j >= 0 and cost < best_cost - 1e-9:
                best_cost, best_j = cost, j
            start += chunk
            chunk = min(chunk * 2, 1024)
        if best_j < 0:
            return None
        return (best_cost, batch.pos[best_j], float(batch.ang[best_j]), bool(batch.mir[best_j]),
                batch.label_t[best_j])

    def _known_of(self, i: int, wide: bool, n: int, ids: Sequence[int]) -> np.ndarray:
        """What is known of the candidates of body ``i`` against the placed bodies ``ids``, where they are now."""
        hit = self._known.get((i, wide))
        if hit is None:
            hit = self._known[(i, wide)] = (np.zeros((len(self.bodies), n), dtype=np.uint8),
                                            np.zeros(len(self.bodies), dtype=np.int64))
        known, tags = hit
        for j in ids:
            if tags[j] != self.tag[j]:
                known[j], tags[j] = 0, self.tag[j]
        return known

    def _cheapest(self, i: int, batch: _Batch, start: int, stop: int, bound: float, ids: Sequence[int],
                  known: np.ndarray) -> tuple[int, float]:
        """The cheapest feasible one of the candidates ``start:stop`` and its cost, or ``(-1, inf)``.

        A candidate that costs more by itself than one that was found costs with everything around it is not
        looked at; and one that something known is in the way of is not looked at again. Neither changes
        which one is the cheapest (the first of those that cost the same, :func:`_least`).
        """
        found: list[np.ndarray] = []
        costs: list[np.ndarray] = []
        others = None
        least, at, step = bound, start, 128
        while at < stop and batch.base[at] <= least + 1e-6:
            rows = np.arange(at, min(stop, at + step))
            rows = rows[batch.base[rows] <= least + 1e-6]
            rows = self._clear(i, batch, rows, ids, known)
            if len(rows):
                if others is None:
                    others = self._others(i)
                cost = batch.base[rows] + self._crowding(i, batch.geo[rows], others)
                found.append(rows)
                costs.append(cost)
                least = min(least, float(cost.min()))
            at += step
            step = min(step * 2, 512)
        if not found:
            return -1, math.inf
        total = np.concatenate(costs)
        j = _least(total)
        return int(np.concatenate(found)[j]), float(total[j])

    def _clear(self, i: int, batch: _Batch, rows: np.ndarray, ids: Sequence[int], known: np.ndarray) -> np.ndarray:
        """Those of the candidates ``rows`` that are clear of everything placed; what is found goes into ``known``."""
        if not ids:
            return rows
        at = np.array(ids)
        state = known[np.ix_(at, rows)]
        free = ~(state == 2).any(axis=0)
        rows, fresh = rows[free], at[(state[:, free] == 0).any(axis=1)]
        if len(rows) and len(fresh):
            bad, _ = self._blocked(i, batch.geo[rows], self._others(i, fresh.tolist()), every=False)
            clear = ~bad.any(axis=1)
            # (Of a candidate that something is in the way of, only that is known.)
            known[np.ix_(fresh, rows[clear])] = 1
            pose, body = np.nonzero(bad)
            known[fresh[body], rows[pose]] = 2
            rows = rows[clear]
        return rows

    def _seat(self, i: int) -> bool:
        hit = self._best(i) or self._best(i, wide=True)
        if hit is None:
            return False
        self._set(i, hit[1], hit[2], hit[3], hit[4])
        return True

    def _order(self) -> list[int]:
        """Most constrained first: fragments, then discs with the fewest places to go."""
        def room(i: int) -> int:
            if self.bodies[i].kind != "disc":
                return 0
            return int((np.round(self._batch(i).base, 6) < 1.5).sum())    # (rounded: a last bit decides nothing)
        keyed = [(-self.bodies[i].priority, room(i), i) for i in range(len(self.bodies))]
        return [i for _, _, i in sorted(keyed)]

    def run(self) -> LayoutResult:
        order = self._order()
        for i in order:
            self._seat(i)
        self._replace_passes(order)
        left, abandoned = len(self.bodies) - len(self.world), False
        for k in range(self.p.eject_rounds):
            more = self._eject_pass(order)
            if more:
                self._replace_passes(order)
            if k == 0:
                left = len(self.bodies) - len(self.world)
                abandoned = 0 < self.p.give_up <= left
            if not more or abandoned:
                break
        if not abandoned:
            self._polish(order)
        unplaced = [i for i in order if i not in self.world]
        result = self._result(unplaced)
        result.left, result.abandoned = left, abandoned
        return result

    def _replace_passes(self, order: Sequence[int]) -> None:
        for _ in range(self.p.improve_passes):
            moved = False
            for i in order:
                if i in self.world:
                    now = self._current(i)
                    if len(self._batch(i)) and now <= float(self._batch(i).base[0]) + 1e-6:
                        continue                    # already at its ligand-only optimum
                    saved = self._snapshot([i])
                    self._unset(i)
                    hit = self._best(i, upper=now)
                    if hit is not None and hit[0] < now - 1e-6:
                        self._set(i, hit[1], hit[2], hit[3], hit[4])
                        moved = True
                    else:
                        self._restore(saved, [i])
                elif self._seat(i):
                    moved = True
            if not moved:
                break

    def _eject_pass(self, order: Sequence[int]) -> bool:
        """Try to seat badly placed bodies where they want to be, moving whoever is in the way."""
        p = self.p
        improved = False
        costs = {i: self._current(i) for i in self.world}
        # Worth trying only where a body pays clearly more than the ligand alone would make it pay.
        gain = {i: costs[i] - (float(self._batch(i).base[0]) if len(self._batch(i)) else costs[i])
                for i in self.world}
        worst = sorted((i for i in order if i in self.world and gain[i] > p.eject_threshold),
                       key=lambda i: (-round(gain[i], 6), i))
        for i in worst + [i for i in order if i not in self.world]:
            batch = self._batch(i)
            now = self._current(i) if i in self.world else 1e6
            limit = int(np.searchsorted(batch.base, now - p.eject_threshold * 0.6))
            if limit == 0:
                continue
            # Spread the trials: the cheapest pose in each direction from the body's first target.
            tried: list[int] = []
            seen: set[int] = set()
            tgt = self.bodies[i]._targets[0] if self.bodies[i].n_ports else np.zeros(2)
            rel = (batch.geo.ports[:limit, 0, :] if self.bodies[i].n_ports else batch.pos[:limit]) - tgt
            sectors = _sectors(rel)
            for j in range(limit):
                if int(sectors[j]) in seen:
                    continue
                seen.add(int(sectors[j]))
                tried.append(j)
                if len(tried) >= p.eject_candidates or len(seen) == 8:
                    break
            all_ids = sorted(self.world)
            snap = self._snapshot(all_ids)
            for j in tried:
                self._unset(i)
                geo = batch.geo[j:j + 1]
                blockers = self._blockers(i, geo)
                if not blockers or len(blockers) > p.eject_max_blockers:
                    self._restore(snap, [i])
                    continue
                # An exchange is judged by the bodies that move; what they do to the
                # rest (crossed lines, blocked sightlines) is part of their own cost.
                before = now + sum(costs[bk] for bk in blockers)
                for bk in blockers:
                    self._unset(bk)
                _, ok = self._collide(i, geo)
                done = bool(ok[0])
                if done:
                    self._set(i, batch.pos[j], float(batch.ang[j]), bool(batch.mir[j]), batch.label_t[j])
                    after = self._current(i)
                    for bk in sorted(blockers, key=lambda b: order.index(b)):
                        hit = self._best(bk, upper=before - after) if after < before else None
                        if hit is None:
                            done = False
                            break
                        self._set(bk, hit[1], hit[2], hit[3], hit[4])
                        after += hit[0]
                if done and after < before - 1e-6:
                    improved = True
                    costs = {k: self._current(k) for k in self.world}
                    break
                self._restore(snap, sorted(set(all_ids) | {i} | blockers))
        return improved

    def _polish(self, order: Sequence[int]) -> None:
        dirs = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                (0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, 0.7071), (-0.7071, -0.7071)]
        for step in (6.0, 3.0, 1.5, 0.75):
            for _ in range(3):
                moved = False
                for i in order:
                    if i not in self.world:
                        continue
                    body = self.bodies[i]
                    cand_pos = [self.pos[i]] + [self.pos[i] + step * np.array(d) for d in dirs]
                    cand_ang = [self.ang[i]] * len(cand_pos)
                    if body.rotatable:
                        turn = step / max(body.radius, 12.0)
                        local = body.ports[0].local
                        ly = -local[1] if self.mir[i] else local[1]
                        pivot = np.array(Pose(float(self.pos[i][0]), float(self.pos[i][1]), self.ang[i],
                                              self.mir[i]).apply(local))
                        for sgn in (1.0, -1.0):
                            # Rotate about the primary port so the link end stays put.
                            th = self.ang[i] + sgn * turn
                            c, s = math.cos(th), math.sin(th)
                            cand_pos.append(pivot - np.array([local[0] * c - ly * s, local[0] * s + ly * c]))
                            cand_ang.append(th)
                    pos = np.array(cand_pos)
                    ang = np.array(cand_ang)
                    mir = np.full(len(pos), self.mir[i])
                    total, lt = self._score(i, pos, ang, mir, soft=True)
                    j = _least(total)
                    if j != 0 and total[j] < total[0] - 1e-7:
                        self._set(i, pos[j], float(ang[j]), self.mir[i], lt[j])
                        moved = True
                if not moved:
                    break

    # -- reporting ---------------------------------------------------------
    def _result(self, unplaced: Sequence[int]) -> LayoutResult:
        poses: dict[Any, Pose] = {}
        label_t: dict[Any, tuple[float, ...]] = {}
        total = 0.0
        stretch = 0.0
        bond_x = 0
        box_hits = 0
        for i in sorted(self.world):
            body = self.bodies[i]
            poses[body.key] = Pose(float(self.pos[i][0]), float(self.pos[i][1]), self.ang[i], self.mir[i])
            label_t[body.key] = tuple(float(v) for v in self.lt[i])
            c = self._current(i)
            total += c if math.isfinite(c) else 0.0
            if body.n_ports:
                P, A, M = self.pos[i][None, :], np.array([self.ang[i]]), np.array([self.mir[i]])
                a, b, length = body.link_segments(P, A, M)
                stretch = max(stretch, float((length[0] / np.maximum(body._ideal, 1e-9)).max()))
                d = body._drawn
                if d.any():
                    bond_x += int(self.field.bond_crossings(a[0][d], b[0][d]).sum())
                    box_hits += sum(int(self.field.boxes_crossed(a[0][k:k + 1], b[0][k:k + 1], body._targets[k],
                                                                 reach=float(body._reach[k]))[0])
                                    for k in np.nonzero(d)[0])
                    # ... and so, for a residue drawn at a level of detail, is a link through a name of its
                    # own residue that it may not run through, and a label that stands on a sibling link.
                    strict = [k for k in np.nonzero(d)[0] if body.ports[k].clear_of or body.ports[k].clear_of_name]
                    box_hits += sum(int(_through_names(body, body.ports[k], a[0][k:k + 1], b[0][k:k + 1], P, A, M)[0])
                                    for k in strict)
                    if strict:
                        box_hits += _labels_on_siblings(body, a, b, np.asarray(self.lt[i], dtype=float)[None, :])
        # Independent audit of the final state: nothing here trusts the search.
        ids = sorted(self.world)
        overlaps = 0
        line_x = 0
        link_hits = box_hits                 # ... a link across a residue box of a chain ligand is one as well
        for n_, i in enumerate(ids):
            wi = self.world[i]
            for j in ids[n_ + 1:]:
                wj = self.world[j]
                if len(wi.sr) and len(wj.sr) and capsule_gaps(wi.sp0, wi.sp1, wi.sr, wj.sp0, wj.sp1, wj.sr).min() < -0.5:
                    overlaps += 1
                if len(wi.la) and len(wj.la):
                    line_x += int(segment_cross_matrix(wi.la, wi.lb, wj.la, wj.lb).sum())
                for wa, wb in ((wi, wj), (wj, wi)):
                    if len(wa.la) and len(wb.sr):
                        gaps = capsule_gaps(wa.la, wa.lb, np.zeros(len(wa.la)), wb.sp0, wb.sp1, wb.sr)
                        link_hits += int((gaps.min(axis=1) < -0.5).sum())
        return LayoutResult(poses=poses, unplaced=[self.bodies[i].key for i in unplaced], cost=total,
                            overlaps=overlaps, line_crossings=line_x, bond_crossings=bond_x,
                            max_stretch=stretch, label_t=label_t, link_hits=link_hits)


def _classes(scores: np.ndarray, within: float = 1e-9) -> np.ndarray:
    """Scores as they are compared: each stands for the least of its class, and two scores are of one class where
    they lie within ``within`` of each other, or of a score between them.

    Two scores that differ by a last bit are so equal wherever they lie. Each rounded by itself they are not: a
    rounding has edges, and scores that are products of short decimals lie on them.
    """
    arr = np.asarray(scores, dtype=float)
    if len(arr) < 2:
        return arr
    order = np.argsort(arr, kind="stable")
    ordered = arr[order]
    with np.errstate(invalid="ignore"):
        first = np.concatenate([[True], np.diff(ordered) > within])
    first |= ~np.isfinite(ordered)
    least = ordered[np.maximum.accumulate(np.where(first, np.arange(len(ordered)), 0))]
    out = np.empty_like(arr)
    out[order] = least
    return out


def _least(scores: np.ndarray, within: float = 1e-9) -> int:
    """Where the least of ``scores`` is, and of those that are equal to it (:func:`_classes`) the first."""
    arr = np.asarray(scores, dtype=float)
    low = float(arr.min())
    if not math.isfinite(low):
        return int(np.argmin(arr))
    while True:
        reach = float(arr[arr <= low + within].max())
        if reach <= low:
            break
        low = reach
    return int(np.argmax(arr <= low))


def _sectors(rel: np.ndarray) -> np.ndarray:
    """Which of eight sectors of 45 degrees each direction lies in, the first about the x axis.

    The candidates of a fragment stand every 7.5 degrees, and so every sixth of them exactly on the edge
    between two sectors. The angle is rounded before it is divided: such a direction is in the sector
    above the edge, and not in the one that the last bit of its angle happens to say."""
    degrees = np.round(np.degrees(np.arctan2(rel[:, 1], rel[:, 0])), 6) % 360.0
    return ((degrees + 22.5) // 45.0).astype(int) % 8


def _apart(points: np.ndarray, others: np.ndarray) -> np.ndarray:
    """How far each of ``points`` is from each of ``others``: ``(len(points), len(others))``."""
    dx = points[:, 0][:, None] - others[:, 0]
    dy = points[:, 1][:, None] - others[:, 1]
    return np.sqrt(dx * dx + dy * dy)


def _through_names(body: Body, port: Port, a: np.ndarray, b: np.ndarray, pos: np.ndarray, ang: np.ndarray,
                   mir: np.ndarray) -> np.ndarray:
    """Does the visible part of a link, ``a[i] -> b[i]`` for the pose ``i``, run through a name of its
    own body that it may not run through: the name of an atom (``port.clear_of``), or the body's own
    name (``port.clear_of_name``)? One boolean for each pose."""
    hit = np.zeros(len(a), dtype=bool)
    if port.clear_of:
        u0, u1, ur = body.upright_capsules(pos, ang, mir)
        sel = list(port.clear_of)
        gap = segments_distance(a[:, None, :], b[:, None, :], u0[:, sel, :], u1[:, sel, :]) - ur[None, sel]
        hit |= (gap < 0.0).any(axis=1)
    if port.clear_of_name and body.label is not None:
        centre = body.label_centre(pos, ang, mir)
        assert centre is not None
        off = np.array([max(0.0, (body.label.width - body.label.height) / 2.0), 0.0])
        gap = segments_distance(a, b, centre - off, centre + off) - (body.label.height / 2.0 + body.label.pad)
        hit |= gap < 0.0
    return hit


def _labels_on_siblings(body: Body, a: np.ndarray, b: np.ndarray, label_t: np.ndarray) -> int:
    """How many labels of a body's drawn links stand on another drawn link of the same body, for the one
    pose whose visible link ends are ``a`` and ``b`` (``(1, K, 2)``) and whose labels are at ``label_t``."""
    idx = np.nonzero(body._has_dlabel)[0]
    if not len(idx):
        return 0
    d0, d1, dr = body.dlabel_capsules(a, b, label_t)
    count = 0
    for m, k in enumerate(idx):
        for j, sibling in enumerate(body.ports):
            if j != k and sibling.drawn and body.ports[k].drawn:
                gap = segments_distance(d0[:, m], d1[:, m], a[:, j], b[:, j]) - dr[m]
                count += int(gap[0] < 0.0)
    return count


def layout_bodies(fld: LigandField, bodies: Sequence[Body], params: LayoutParams | None = None) -> LayoutResult:
    """Place ``bodies`` around the ligand described by ``fld``."""
    if not bodies:
        return LayoutResult({}, [], 0.0, 0, 0, 0, 0.0)
    return _Engine(fld, bodies, params or LayoutParams()).run()


def layout_key(fld: LigandField, bodies: Sequence[Body], params: LayoutParams | None = None) -> str:
    """A digest of all that :func:`layout_bodies` depends on: where it is the same, so is the layout, and
    nothing has to be laid out again. What a figure is called, how it is coloured and what is written in it
    is not in it, unless it changes the size of something."""
    h = hashlib.blake2b(digest_size=20)

    def put(value: Any) -> None:
        if isinstance(value, np.ndarray):
            h.update(f"{value.dtype.str}{value.shape}".encode())
            h.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, (list, tuple)):
            h.update(f"[{len(value)}".encode())
            for item in value:
                put(item)
        else:
            h.update(repr(value).encode() + b"\0")

    put(fld.key)
    for body in bodies:
        for f in fields(body):
            if f.name == "ports":
                for port in body.ports:
                    put([getattr(port, p.name) for p in fields(port)])
            elif f.init and f.name != "payload":
                put(getattr(body, f.name))
    put(params or LayoutParams())
    return h.hexdigest()


def audit_layout(fld: LigandField, bodies: Sequence[Body], poses: "dict[Any, Pose]",
                 label_t: "Mapping[Any, Sequence[float]] | None" = None,
                 params: LayoutParams | None = None) -> LayoutResult:
    """Measure a placement that was made elsewhere, without searching.

    The poses may come from a stored layout or from a person dragging nodes;
    nothing is moved. What comes back is the same independent audit that closes
    every layout run -- overlaps, links through other bodies, crossings -- so a
    layout from outside is held to the standard of one from the engine. A body
    without a pose is reported as unplaced.
    """
    if not bodies:
        return LayoutResult({}, [], 0.0, 0, 0, 0, 0.0)
    engine = _Engine(fld, bodies, params or LayoutParams())
    unplaced = []
    for i, body in enumerate(engine.bodies):
        pose = poses.get(body.key)
        if pose is None:
            unplaced.append(i)
            continue
        at = list((label_t or {}).get(body.key, ()))
        if len(at) != body.n_ports:
            at = [0.5] * body.n_ports
        engine._set(i, np.array([pose.x, pose.y], dtype=float), pose.angle, pose.mirror, np.array(at, dtype=float))
    return engine._result(unplaced)
