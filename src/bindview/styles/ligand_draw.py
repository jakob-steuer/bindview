"""Drawing the ligand skeleton: bonds, wedges, atom labels. Shared by all styles."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..confidence import PLAIN, Look
from ..depict import AtomLabel, DepBond, Depiction, clip_to_labels
from ..geometry import Point, add, cross, dist, lerp, mul, perp, sub, unit
from ..scene import Circle, Group, Item, Line, Paint, Polygon, Text
from ..theme import Metrics, Palette

__all__ = ["draw_ligand", "bond_items", "coloured_bond_items", "label_items", "double_bond_side", "plddt_underlay"]


def _ring_for(dep: Depiction, bond: DepBond) -> tuple[int, ...] | None:
    """The ring a double bond should be drawn inside of."""
    rings = [r for r in dep.rings if bond.a in r and bond.b in r]
    if not rings:
        return None

    def n_double(ring: tuple[int, ...]) -> int:
        members = set(ring)
        return sum(1 for b in dep.bonds if b.order == 2 and b.a in members and b.b in members)

    # Aromatic first, then six-membered, then the ring with more double bonds.
    rings.sort(key=lambda r: (
        0 if all(dep.atoms[i].aromatic for i in r) else 1,
        0 if len(r) == 6 else 1,
        -n_double(r), len(r), r))
    return rings[0]


def double_bond_side(dep: Depiction, bond: DepBond) -> int:
    """``+1`` / ``-1``: the side of ``a->b`` the second line goes on; ``0`` = centred."""
    pa, pb = dep.pos(bond.a), dep.pos(bond.b)
    axis = sub(pb, pa)
    ring = _ring_for(dep, bond)
    if ring is not None:
        c = dep.centroid(ring)
        return 1 if cross(axis, sub(c, pa)) > 0 else -1
    na = [j for j in dep.neighbours(bond.a) if j != bond.b]
    nb = [j for j in dep.neighbours(bond.b) if j != bond.a]
    if not na or not nb:
        return 0                                   # terminal: centre the pair of lines
    score = 0
    for j in na + nb:
        score += 1 if cross(axis, sub(dep.pos(j), pa)) > 0 else -1
    return 1 if score >= 0 else -1


def _clipped(dep: Depiction, bond: DepBond, pa: Point, pb: Point) -> tuple[Point, Point] | None:
    return clip_to_labels(pa, pb, dep.atoms[bond.a].label, dep.atoms[bond.b].label, dep.metrics.label_gap)


def bond_items(dep: Depiction, bond: DepBond, stroke: str = "ligand_bond") -> list[Item]:
    m = dep.metrics
    pa, pb = dep.pos(bond.a), dep.pos(bond.b)
    if bond.kind == "coordination":
        seg = _clipped(dep, bond, pa, pb)
        if seg is None:
            return []
        return [Line(*seg[0], *seg[1], Paint(stroke="coordination", width=m.bond_width * 0.75,
                                             dash=(1.2, 3.2)))]
    seg = _clipped(dep, bond, pa, pb)
    if seg is None:
        return []
    a, b = seg
    paint = Paint(stroke=stroke, width=m.bond_width)
    n = perp(unit(sub(pb, pa)))

    if bond.stereo == "wedge":
        hw0, hw1 = m.bond_width * 0.35, m.wedge_width / 2.0
        pts = (add(a, mul(n, hw0)), add(b, mul(n, hw1)), add(b, mul(n, -hw1)), add(a, mul(n, -hw0)))
        return [Polygon(pts, Paint(fill=stroke, stroke=stroke, width=0.5, join="round"))]
    if bond.stereo == "hash":
        length = dist(a, b)
        count = max(3, int(round(length / m.hash_spacing)))
        thin = Paint(stroke=stroke, width=max(0.9, m.bond_width * 0.62), cap="butt")
        out: list[Item] = []
        for k in range(count):
            t = (k + 0.5) / count
            p = lerp(a, b, t)
            hw = 0.5 + (m.wedge_width / 2.0 - 0.5) * t
            out.append(Line(*add(p, mul(n, hw)), *add(p, mul(n, -hw)), thin))
        return out

    if bond.order == 1 and not bond.aromatic:
        return [Line(*a, *b, paint)]

    off = m.double_offset
    if bond.order == 3:
        out = [Line(*a, *b, paint)]
        for sgn in (1.0, -1.0):
            s2 = _clipped(dep, bond, add(pa, mul(n, sgn * off * 0.85)), add(pb, mul(n, sgn * off * 0.85)))
            if s2:
                out.append(Line(*s2[0], *s2[1], paint))
        return out

    side = double_bond_side(dep, bond)
    if side == 0 and not bond.aromatic:
        out = []
        for sgn in (0.5, -0.5):
            s2 = _clipped(dep, bond, add(pa, mul(n, sgn * off)), add(pb, mul(n, sgn * off)))
            if s2:
                out.append(Line(*s2[0], *s2[1], paint))
        return out

    side = side or 1
    out = [Line(*a, *b, paint)]
    length = dist(pa, pb)
    trim = min(off * 0.95, length * 0.2)
    ia = add(pa, mul(n, side * off))
    ib = add(pb, mul(n, side * off))
    ax = unit(sub(pb, pa))
    # Pull the inner line in at ends that are bare junctions, so it meets the
    # neighbouring bonds like a mitre instead of poking past them.
    if dep.atoms[bond.a].label is None and len(dep.neighbours(bond.a)) > 1:
        ia = add(ia, mul(ax, trim))
    if dep.atoms[bond.b].label is None and len(dep.neighbours(bond.b)) > 1:
        ib = add(ib, mul(ax, -trim))
    s2 = _clipped(dep, bond, ia, ib)
    if s2:
        inner = Paint(stroke=stroke, width=m.bond_width, dash=(2.6, 2.2)) if bond.aromatic else paint
        out.append(Line(*s2[0], *s2[1], inner))
    return out


def coloured_bond_items(dep: Depiction, bond: DepBond, bands: "Sequence[str | None]", stroke: str = "ligand_bond"
                        ) -> list[Item]:
    """A bond in the colours of its atoms' pLDDT bands, each half in that of the atom it leaves: ``bands`` has
    one entry for each depicted atom, and an atom without a value has the bond in its own colour (``stroke``)."""
    items = bond_items(dep, bond, stroke)
    if bond.kind == "coordination":
        return items
    of = [f"plddt_{bands[i]}" if bands[i] is not None else stroke for i in (bond.a, bond.b)]
    if of[0] == of[1]:
        return [_painted(it, of[0]) for it in items]
    pa, pb = dep.pos(bond.a), dep.pos(bond.b)
    axis, length = unit(sub(pb, pa)), dist(pa, pb) or 1.0

    def along(p: Point) -> float:
        return (sub(p, pa)[0] * axis[0] + sub(p, pa)[1] * axis[1]) / length

    out: list[Item] = []
    for it in items:
        if isinstance(it, Line):
            p, q = (it.x1, it.y1), (it.x2, it.y2)
            t0, t1 = along(p), along(q)
            if abs(t1 - t0) < 0.2:                        # a rung of a hashed bond lies across it
                out.append(_painted(it, of[0] if (t0 + t1) / 2.0 < 0.5 else of[1]))
                continue
            mid = lerp(p, q, min(max((0.5 - t0) / (t1 - t0), 0.0), 1.0))
            first, second = (of[0], of[1]) if t0 < t1 else (of[1], of[0])
            butt = replace(it.paint, cap="butt") if not it.paint.dash else it.paint
            out.append(Line(*p, *mid, replace(butt, stroke=first)))
            out.append(Line(*mid, *q, replace(butt, stroke=second)))
            # A round end where the bond meets others, as a bond of one colour has.
            out.extend(Circle(*end, it.paint.width / 2.0, Paint(fill=token)) for end, token in ((p, first), (q, second))
                       if not it.paint.dash)
        elif isinstance(it, Polygon) and len(it.points) == 4:
            a0, b0, b1, a1 = it.points                    # a wedge, from its narrow end to its wide one
            m0, m1 = lerp(a0, b0, 0.5), lerp(a1, b1, 0.5)
            out.append(Polygon((a0, m0, m1, a1), replace(it.paint, fill=of[0], stroke=of[0])))
            out.append(Polygon((m0, b0, b1, m1), replace(it.paint, fill=of[1], stroke=of[1])))
        else:
            out.append(_painted(it, of[0]))
    return out


def _painted(it: Item, token: str) -> Item:
    if isinstance(it, Polygon):
        return Polygon(it.points, replace(it.paint, fill=token, stroke=token))
    assert isinstance(it, Line)
    return Line(it.x1, it.y1, it.x2, it.y2, replace(it.paint, stroke=token))


def label_items(label: AtomLabel, size: float, fill: str) -> list[Item]:
    return [Text(run.x, run.y, run.spans, size, fill, run.anchor) for run in label.runs]


def atom_looks(dep: Depiction, plddt: "dict[str, float] | None", by_colour: bool = False) -> list[Look]:
    """The confidence look of every depicted atom.

    A depiction-only hydrogen takes the look of the atom it is drawn on; an atom
    without a value is drawn plainly. ``by_colour``: as in
    :func:`bindview.confidence.look_of`.
    """
    from ..confidence import look_of

    looks = [look_of(plddt.get(a.name), by_colour) if (plddt and a.name) else PLAIN for a in dep.atoms]
    if plddt:
        for a in dep.atoms:
            if a.name is None:
                nbrs = dep.neighbours(a.idx)
                if nbrs:
                    looks[a.idx] = looks[nbrs[0]]
    return looks


#: A figure that colours by pLDDT sets the atoms on a band of colour: how wide it is, in bond lengths,
#: and how strong. Wide enough to read as a colour, and pale enough for a bond to stand on it.
UNDERLAY_WIDTH = 0.5
UNDERLAY_OPACITY = 0.5


def plddt_underlay(dep: Depiction, bands: "Sequence[str | None]") -> "Group | None":
    """What stands behind atoms that are coloured by pLDDT.

    Under every atom that is drawn as an atom a disc in the colour of its band, and under every bond
    between two such atoms a stroke as wide as the disc, each half in the colour of the atom it leaves.
    ``bands`` has one entry for each depicted atom: its band, or ``None`` where the model gives it no
    value, and nothing stands behind that atom. The whole is one group and is made paler as one, so
    that where two shapes meet nothing adds up. ``None`` where there is nothing to draw.
    """
    width = UNDERLAY_WIDTH * dep.metrics.bond_length
    items: list[Item] = []
    for bond in dep.bonds:
        if bond.kind not in ("covalent", "coordination"):
            continue
        if dep.unit_of(bond.a) is not None or dep.unit_of(bond.b) is not None:
            continue                     # an atom inside a residue's box: the box is coloured, not the atom
        a, b = dep.atoms[bond.a], dep.atoms[bond.b]
        mx, my = (a.x + b.x) / 2.0, (a.y + b.y) / 2.0
        for atom in (a, b):
            band = bands[atom.idx]
            if band is not None:
                items.append(Line(atom.x, atom.y, mx, my, Paint(stroke=f"plddt_{band}", width=width, cap="butt")))
    for atom in dep.atoms:
        band = bands[atom.idx]
        if band is not None and dep.unit_of(atom.idx) is None:
            items.append(Circle(atom.x, atom.y, width / 2.0, Paint(fill=f"plddt_{band}")))
    return Group("plddt", items, opacity=UNDERLAY_OPACITY) if items else None


def draw_ligand(dep: Depiction, palette: Palette, metrics: Metrics,
                looks: "Sequence[Look] | None" = None, bands: "Sequence[str | None] | None" = None) -> Group:
    """The ligand as a group: bonds underneath, labels on top. ``bands``, one for each depicted atom: its bonds
    are drawn in the colours of these pLDDT bands.

    With ``looks`` (one per depicted atom) the drawing carries the model's
    confidence: a bond is drawn at the weaker of its two atoms, thinner and
    paler below pLDDT 70. Each confidence level fades as one group, so the
    strokes that meet at an atom do not add up to a darker joint.
    """
    g = Group("ligand")
    if looks is None or all(lk.level == 0 for lk in looks):
        bonds = Group("bonds")
        for bond in dep.bonds:
            bonds.extend(bond_items(dep, bond) if bands is None else coloured_bond_items(dep, bond, bands))
        labels = Group("atoms")
        for atom in dep.atoms:
            if atom.label is not None:
                labels.add(_atom_group(atom, palette, metrics))
        g.add(bonds, labels)
        return g

    bonds = Group("bonds")
    labels = Group("atoms")
    by_level: dict[int, Look] = {}
    for lk in looks:
        by_level.setdefault(lk.level, lk)
    for level in sorted(by_level, reverse=True):          # weakest underneath
        look = by_level[level]
        thin = replace(dep, metrics=dep.metrics.with_(bond_width=dep.metrics.bond_width * look.width))
        part = Group("bonds-" + (look.band or "plain"), opacity=look.opacity)
        for bond in dep.bonds:
            if max(looks[bond.a].level, looks[bond.b].level) == level:
                part.extend(bond_items(thin, bond))
        if part.items:
            bonds.add(part)
        text = Group("atoms-" + (look.band or "plain"), opacity=1.0 if level == 0 else max(0.6, look.opacity + 0.15))
        for atom in dep.atoms:
            if atom.label is not None and looks[atom.idx].level == level:
                text.add(_atom_group(atom, palette, metrics))
        if text.items:
            labels.add(text)
    g.add(bonds, labels)
    return g


def _atom_group(atom, palette: Palette, metrics: Metrics) -> Group:
    token = palette.element_token(atom.element)
    if atom.name is None:
        token = "ligand_carbon"                 # depiction-only stereo hydrogens
    ag = Group("atom", data={"atom": atom.name} if atom.name else {})
    ag.extend(label_items(atom.label, metrics.atom_font, token))
    return ag
