"""Drawing a chain ligand: its residues as boxes on a backbone, the rest of it as atoms.

The counterpart of :mod:`bindview.styles.ligand_draw` for a depiction that has
``units``. A residue is a rounded box with its three-letter code and, smaller
and paler, its position in the chain. The position is a count from the
ligand's own N-terminus and is set apart from the code on purpose: a protein
residue in the same figure reads ``Arg190``, and a chain residue must not look
as if it carried a number of that kind.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from ..confidence import PLAIN, Look, look_of
from ..depict import DepBond, Depiction, DepUnit
from ..geometry import Point, segment_box_entry
from ..polymer import BRACKET_OUT, BRACKET_STEP, INDEX_FONT, UNIT_FONT
from ..scene import Group, Line, Paint, Polyline, Rect, Span, Text
from ..textmetrics import CAP_HEIGHT
from ..theme import Metrics, Palette
from .ligand_draw import _atom_group, bond_items

__all__ = ["draw_polymer", "unit_looks", "bracket_points", "terminus_marks"]

_CORNER = 5.0
#: Outline of a box by confidence level: (dash, opacity), as for the residue bubbles.
_OUTLINE: dict[int, tuple[tuple[float, ...], float]] = {0: ((), 1.0), 1: ((3.6, 2.2), 0.85), 2: ((1.1, 2.3), 0.7)}
_TERMINUS_FONT = 8.5


def unit_looks(dep: Depiction, plddt: "dict[str, float] | None", by_colour: bool = False) -> list[Look]:
    """The confidence look of every residue of the chain: that of the mean pLDDT of its atoms.
    ``by_colour``: as in :func:`bindview.confidence.look_of`."""
    looks = []
    for unit in dep.units:
        names = [dep.atoms[a].name for a in unit.atoms]
        values = [plddt[n] for n in names if plddt and n is not None and n in plddt]
        looks.append(look_of(sum(values) / len(values), by_colour) if values else PLAIN)
    return looks


def bracket_points(dep: Depiction, bond: DepBond, lane: int) -> tuple[Point, ...]:
    """The path of a link between two residues that are not neighbours: out to the left, along, and back."""
    ua, ub = dep.unit_of(bond.a), dep.unit_of(bond.b)
    assert ua is not None and ub is not None
    x = min(ua.box[0], ub.box[0]) - BRACKET_OUT - lane * BRACKET_STEP
    return ((ua.box[0], ua.y), (x, ua.y), (x, ub.y), (ub.box[0], ub.y))


def brackets(dep: Depiction) -> list[tuple[DepBond, tuple[Point, ...]]]:
    """Every bracket of the depiction, each in a lane of its own where two would run along each other."""
    out: list[tuple[DepBond, tuple[Point, ...]]] = []
    spans: list[tuple[float, float, int]] = []
    if dep.arranged == "plane":
        return out                       # no column to stand beside: such a link is a straight line there
    for bond in dep.bonds:
        if bond.kind not in ("closure", "crosslink"):
            continue
        ua, ub = dep.unit_of(bond.a), dep.unit_of(bond.b)
        if ua is None or ub is None:
            continue
        lo, hi = sorted((ua.y, ub.y))
        lane = 0
        while any(l == lane and not (hi < a or lo > b) for a, b, l in spans):
            lane += 1
        spans.append((lo, hi, lane))
        out.append((bond, bracket_points(dep, bond, lane)))
    return out


def terminus_marks(dep: Depiction) -> list[tuple[Point, str]]:
    """Where "N" and "C" are printed: beyond the two ends of an open chain, unless something is attached there."""
    if not dep.units or not dep.sequential or any(b.kind == "closure" for b in dep.bonds):
        return []
    if dep.arranged == "plane":
        return []                        # the first box is not the top of anything
    first, last = dep.units[0], dep.units[-1]
    used_top = used_bottom = False
    for b in dep.bonds:
        if b.kind != "covalent":
            continue
        for inside, outside in ((b.a, b.b), (b.b, b.a)):
            unit = dep.unit_of(inside)
            if unit is None or dep.unit_of(outside) is not None:
                continue
            y = dep.atoms[outside].y
            used_top = used_top or (unit is first and y < first.box[1])
            used_bottom = used_bottom or (unit is last and y > last.box[3])
    cap = CAP_HEIGHT * _TERMINUS_FONT
    marks = []
    # A chain of nucleotides runs from its 5' end to its 3' end, as one of amino acids from N to C.
    top, bottom = ("5\u2032", "3\u2032") if first.residue_class == "nucleotide" else ("N", "C")
    if not used_top and not dep.open_ends[0]:
        marks.append(((first.x, first.box[1] - 4.0), top))
    if not used_bottom and not dep.open_ends[1]:
        marks.append(((last.x, last.box[3] + 4.0 + cap), bottom))
    return marks


def _unit_group(unit: DepUnit, look: Look, names: Sequence[str], deposited: bool = False,
                called: str = "the ligand", beside: Sequence[str] = (), by_colour: bool = False) -> Group:
    """One residue of the chain as its box. ``by_colour``: the box is filled with the colour of the
    residue's pLDDT band, where it has one, and its name stands on it in the colour that reads there."""
    dash, opacity = _OUTLINE[look.level]
    coloured = by_colour and look.band is not None
    fill = f"plddt_{look.band}" if coloured else "background"
    ink = f"plddt_text_{look.band}" if coloured else "text"
    counted = unit.number in ("", str(unit.position)) and not deposited
    title = (f"{unit.text} {unit.number or unit.position} of {called} "
             + ("(position counted from its N-terminus)" if counted else "(numbered as in the structure file)"))
    g = Group("unit", data={"position": str(unit.position), "residue": unit.name, "class": unit.residue_class,
                            "atoms": ",".join(names)}, title=title)
    if look.band is not None:
        g.data["confidence"] = look.band
    if beside:
        # Those of its atoms that stand beside the box as atoms of their own.
        g.data["beside"] = ",".join(beside)
    x0, y0, x1, y1 = unit.box
    g.add(Rect(x0, y0, x1 - x0, y1 - y0, Paint(fill=fill, stroke="ligand_bond", width=1.15, dash=dash,
                                               opacity=opacity), rx=_CORNER))
    baseline = unit.y + CAP_HEIGHT * UNIT_FONT / 2.0
    g.add(Text(unit.x, baseline, (Span(unit.text, bold=True),
                                  Span(" " + (unit.number or str(unit.position)), size=INDEX_FONT,
                                       fill=ink if coloured else "text_muted")),
               UNIT_FONT, ink, "middle", opacity=1.0 if look.level == 0 else 0.85))
    return g


def draw_polymer(dep: Depiction, palette: Palette, metrics: Metrics,
                 looks: "Sequence[Look] | None" = None, units: "Sequence[Look] | None" = None,
                 deposited: bool = False, own: "Mapping[int, Look] | None" = None,
                 by_colour: bool = False) -> Group:
    """The chain ligand as a group: backbone and bonds underneath, boxes and atom labels on top.

    ``looks`` is one look per depicted atom (it styles the atoms outside the
    chain), ``units`` one per residue of the chain. ``own`` is the look of every
    atom that stands beside its residue's box as an atom of its own; without it
    such atoms are drawn at full strength. ``by_colour``: the boxes are filled
    with the colours of their pLDDT bands (:func:`_unit_group`).
    """
    g = Group("ligand", data={"drawn-as": "chain"})
    line = Paint(stroke="ligand_bond", width=metrics.bond_width)
    under = Group("backbone")
    gap = Paint(stroke="ligand_bond", width=metrics.bond_width * 0.75, dash=(0.01, 3.2), cap="round", opacity=0.7)
    beside = {i for u in dep.units for i in u.beside}
    for bond in dep.bonds:
        if bond.kind not in ("chain", "gap"):
            continue
        ua, ub = dep.unit_of(bond.a), dep.unit_of(bond.b)
        if ua is None or ub is None:
            continue
        tail = ua.c if ua.c in ua.beside else None           # the carbonyl carbon below the upper box, drawn
        head = ub.n if ub.n in ub.beside else None           # the nitrogen above the lower one
        if tail is not None or head is not None:
            if bond.kind == "chain":
                continue                 # the bond between the two residues is drawn as what it is, at its atoms
            # Residues are missing here: the dotted line runs between what is drawn of the two ends.
            pa = dep.pos(tail) if tail is not None else segment_box_entry(ub.pos, ua.pos, ua.box)
            pb = dep.pos(head) if head is not None else segment_box_entry(ua.pos, ub.pos, ub.box)
            if tail is not None:
                pa = (pa[0], pa[1] + 3.0)
            if head is not None:
                label = dep.atoms[head].label
                pb = (pb[0], (label.box[1] if label is not None else pb[1]) - 2.0)
            if pb[1] > pa[1]:
                under.add(Line(pa[0], pa[1], pb[0], pb[1], gap))
            continue
        pa = segment_box_entry(ub.pos, ua.pos, ua.box)
        pb = segment_box_entry(ua.pos, ub.pos, ub.box)
        under.add(Line(pa[0], pa[1], pb[0], pb[1], line if bond.kind == "chain" else gap))
    if dep.units:
        # The chain goes on beyond what is drawn: a dotted stub where a terminus letter would stand.
        first, last = dep.units[0], dep.units[-1]
        if dep.open_ends[0]:
            under.add(Line(first.x, first.box[1] - 9.0, first.x, first.box[1], gap))
        if dep.open_ends[1]:
            under.add(Line(last.x, last.box[3], last.x, last.box[3] + 9.0, gap))
    for bond, points in brackets(dep):
        paint = line if bond.kind == "closure" else Paint(stroke="ligand_bond", width=metrics.bond_width * 0.8,
                                                          dash=(3.2, 2.4))
        under.add(Polyline(points, paint))
    if dep.arranged == "plane":
        # A link between two residues that are not neighbours in the chain (a disulfide): on a plane
        # the two boxes are where their residues are, and the link is the straight line between them.
        for bond in dep.bonds:
            ua, ub = dep.unit_of(bond.a), dep.unit_of(bond.b)
            if bond.kind not in ("closure", "crosslink") or ua is None or ub is None:
                continue
            pa = segment_box_entry(ub.pos, ua.pos, ua.box)
            pb = segment_box_entry(ua.pos, ub.pos, ub.box)
            under.add(Line(pa[0], pa[1], pb[0], pb[1],
                           Paint(stroke="ligand_bond", width=metrics.bond_width * 0.8, dash=(3.2, 2.4))))
    g.add(under)

    bonds = Group("bonds")
    # Atoms outside the chain (an appendage: a lipid on a lysine) are drawn at their own looks where one of
    # them is weaker than full strength; otherwise all of them at full strength, with the chain.
    appendage = {a.idx for a in dep.atoms if dep.unit_of(a.idx) is None and a.idx not in beside}
    faded = looks is not None and not by_colour and any(looks[i].level > 0 for i in appendage)
    later: list[DepBond] = []
    for bond in dep.bonds:
        if bond.kind not in ("covalent", "coordination"):
            continue
        if bond.a in beside or bond.b in beside:
            continue                     # of an atom beside its residue's box: drawn below, at its confidence
        if faded and (bond.a in appendage or bond.b in appendage):
            later.append(bond)           # drawn below, at its confidence
            continue
        ua, ub = dep.unit_of(bond.a), dep.unit_of(bond.b)
        if ua is None and ub is None:
            bonds.extend(bond_items(dep, bond))
            continue
        if ua is not None and ub is not None:
            continue
        # The bond that holds an atom to the chain: from the edge of the residue's box to the atom.
        unit, held = (ua, bond.b) if ua is not None else (ub, bond.a)
        assert unit is not None
        outside = dep.pos(held)
        start = segment_box_entry(outside, unit.pos, unit.box)
        end = outside
        label = dep.atoms[held].label
        if label is not None:
            end = segment_box_entry(start, outside, label.box)
        bonds.add(Line(start[0], start[1], end[0], end[1], line))
    labels = Group("atoms")
    if beside:
        _draw_beside(dep, palette, metrics, beside, own or {}, bonds, labels)
    if faded:
        assert looks is not None
        _draw_outside(dep, palette, metrics, appendage, looks, later, bonds, labels)
    g.add(bonds)

    boxes = Group("units")
    for k, unit in enumerate(dep.units):
        names = [dep.atoms[a].name or "" for a in unit.atoms]
        boxes.add(_unit_group(unit, units[k] if units else PLAIN, names, deposited, dep.called,
                              [dep.atoms[a].name or "" for a in unit.beside], by_colour))
    for point, letter in terminus_marks(dep):
        boxes.add(Text.plain(point[0], point[1], letter, _TERMINUS_FONT, "text_muted"))
    g.add(boxes)

    for atom in dep.atoms:
        if atom.label is not None and dep.unit_of(atom.idx) is None and atom.idx not in beside and not faded:
            labels.add(_atom_group(atom, palette, metrics))
    g.add(labels)
    return g


def _draw_outside(dep: Depiction, palette: Palette, metrics: Metrics, outside: "set[int]",
                  looks: "Sequence[Look]", strokes: "Sequence[DepBond]", bonds: Group, labels: Group) -> None:
    """The atoms outside the chain and their bonds, each at the look of its atom (a bond at the weaker of
    its ends outside the chain), every level of confidence fading as one group, the weakest underneath."""
    def weaker(bond: DepBond) -> Look:
        return max((looks[i] for i in (bond.a, bond.b) if i in outside), key=lambda lk: lk.level)

    levels = sorted({weaker(b).level for b in strokes} | {looks[i].level for i in outside}, reverse=True)
    for level in levels:
        mine = [b for b in strokes if weaker(b).level == level]
        seen = weaker(mine[0]) if mine else next(looks[i] for i in sorted(outside) if looks[i].level == level)
        fine = replace(dep, metrics=dep.metrics.with_(bond_width=metrics.bond_width * seen.width))
        part = Group("bonds-outside" + ("-" + seen.band if seen.band else ""), opacity=seen.opacity)
        paint = Paint(stroke="ligand_bond", width=metrics.bond_width * seen.width)
        for bond in mine:
            ua, ub = dep.unit_of(bond.a), dep.unit_of(bond.b)
            if ua is None and ub is None:
                part.extend(bond_items(fine, bond))
                continue
            # The bond that holds the appendage to the chain: from the edge of the residue's box to the atom.
            unit, held = (ua, bond.b) if ua is not None else (ub, bond.a)
            assert unit is not None
            start = segment_box_entry(dep.pos(held), unit.pos, unit.box)
            end = dep.pos(held)
            label = dep.atoms[held].label
            if label is not None:
                end = segment_box_entry(start, dep.pos(held), label.box)
            part.add(Line(start[0], start[1], end[0], end[1], paint))
        if part.items:
            bonds.add(part)
        names = Group("atoms-outside" + ("-" + seen.band if seen.band else ""),
                      opacity=1.0 if level == 0 else max(0.6, seen.opacity + 0.15))
        for i in sorted(outside):
            atom = dep.atoms[i]
            if atom.label is not None and looks[i].level == level:
                names.add(_atom_group(atom, palette, metrics))
        if names.items:
            labels.add(names)


def _draw_beside(dep: Depiction, palette: Palette, metrics: Metrics, beside: "set[int]",
                 own: "Mapping[int, Look]", bonds: Group, labels: Group) -> None:
    """The atoms that stand beside their residues' boxes: their bonds into ``bonds``, their names into
    ``labels``, each at the look of its atom (a bond at the weaker of its two ends). Drawn at the scale
    of the fragments around a chain, in the chain's colour; every level of confidence fades as one group."""
    def look(i: int) -> Look:
        return own.get(i, PLAIN)

    def weaker(bond: DepBond) -> Look:
        ends = [look(i) for i in (bond.a, bond.b) if i in beside]
        return max(ends, key=lambda lk: lk.level)

    mine = [b for b in dep.bonds if b.kind == "covalent" and (b.a in beside or b.b in beside)]
    levels = sorted({weaker(b).level for b in mine} | {look(i).level for i in beside}, reverse=True)
    for level in levels:                                  # the weakest underneath
        strokes = [b for b in mine if weaker(b).level == level]
        seen = weaker(strokes[0]) if strokes else next(look(i) for i in sorted(beside) if look(i).level == level)
        fine = replace(dep, metrics=dep.metrics.with_(bond_width=metrics.protein_bond_width * seen.width,
                                                      double_offset=metrics.double_offset * 0.8, label_gap=1.5))
        part = Group("bonds-beside" + ("-" + seen.band if seen.band else ""), opacity=seen.opacity)
        paint = Paint(stroke="ligand_bond", width=metrics.protein_bond_width * seen.width)
        for bond in strokes:
            ua, ub = dep.unit_of(bond.a), dep.unit_of(bond.b)
            if ua is None and ub is None:
                part.extend(bond_items(fine, bond))
                continue
            # From the edge of the residue's box (the alpha carbon, or a proline's nitrogen) to the atom.
            unit, held = (ua, bond.b) if ua is not None else (ub, bond.a)
            assert unit is not None
            if unit.residue_class == "nucleotide" and held in unit.beside \
                    and unit.anchor not in (bond.a, bond.b):
                # A nucleotide's box is its C4': a bond within it to an atom that is not drawn (O2', O3')
                # is no line from the box. (A bond to the next residue's box is, as it is for a peptide.)
                continue
            outside = dep.pos(held)
            start = segment_box_entry(outside, unit.pos, unit.box)
            end = outside
            label = dep.atoms[held].label
            if label is not None:
                end = segment_box_entry(start, outside, label.box)
            part.add(Line(start[0], start[1], end[0], end[1], paint))
        if part.items:
            bonds.add(part)
        names = Group("atoms-beside" + ("-" + seen.band if seen.band else ""),
                      opacity=1.0 if level == 0 else max(0.6, seen.opacity + 0.15))
        for i in sorted(beside):
            atom = dep.atoms[i]
            if atom.label is not None and look(i).level == level:
                names.add(_atom_group(atom, palette, metrics))
        if names.items:
            labels.add(names)


def chain_obstacles(dep: Depiction) -> list[tuple[float, float, float]]:
    """Discs over the ink of a chain that is not an atom or a bond: brackets and the two terminus letters."""
    out: list[tuple[float, float, float]] = []
    for _bond, points in brackets(dep):
        (_x0, y0), (x, _y1), (_x2, y2), _ = points
        lo, hi = sorted((y0, y2))
        steps = max(1, int((hi - lo) / 6.0))
        out.extend((x, lo + (hi - lo) * k / steps, 2.5) for k in range(steps + 1))
    for point, _letter in terminus_marks(dep):
        out.append((point[0], point[1] - CAP_HEIGHT * _TERMINUS_FONT / 2.0, 6.0))
    return out
