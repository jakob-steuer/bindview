"""A chain ligand as drawable geometry: residues on a path, the rest as atoms.

The result is an ordinary :class:`~bindview.depict.Depiction`, so that the layout
engine, the audit and the styles work on it unchanged. What makes it a chain:

* every residue is a :class:`~bindview.depict.DepUnit` -- a labelled box -- and
  all atoms of the residue sit at its centre, so a contact that names any of
  them lands on the residue;
* consecutive residues are joined by a ``chain`` bond, which the engine treats
  as ink it must not cross;
* atoms that are in no residue (a fatty acid on its linker, a terminal cap) keep
  positions of their own: they are laid out as the small molecule they are and
  set beside the residue they are bonded to.

**The path is a column.** Residue labels are horizontal text, so residues of a
protein read best to the left and right of a vertical chain, one above the
other like a list; along a horizontal chain their labels would collide at any
legible pitch. A chain is therefore drawn top to bottom from its N-terminus,
and it does not bend: a bend would put two stretches of chain side by side with
no room between them for what contacts either. The three-dimensional path of
the peptide is not shown, and nothing in the figure suggests that it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Collection, Hashable, Mapping, Sequence

import numpy as np
from rdkit import Chem

from .chem import quiet_rdkit
from .depict import AtomLabel, DepAtom, DepBond, Depiction, DepUnit, LabelRun, depict_ligand
from .geometry import Box, Point
from .ligand import Ligand
from .peptide import Appendage, Peptide, PeptideResidue
from .plane import Arrangement, passes_through
from .residues import PHOSPHATE, TAIL, canonical_atom, canonical_resname, residue_class, template_for, written
from .scene import Span
from .textmetrics import CAP_HEIGHT, text_width
from .theme import Metrics

__all__ = ["depict_polymer", "unit_label", "UNIT_FONT", "INDEX_FONT", "UNIT_HEIGHT", "PIECE_GAP", "Piece",
           "lay_piece", "piece_sides", "stack_column", "shown_atoms"]

UNIT_FONT = 10.5            # the residue's code
INDEX_FONT = 8.0            # its position in the chain
UNIT_HEIGHT = 17.0
_PAD = 6.0                  # between text and the box, left and right
_GAP = 2.5                  # between code and position
#: Centre-to-centre distance of two residues never falls below this (it is ``Metrics.chain_pitch`` bond lengths).
MIN_PITCH = UNIT_HEIGHT + 5.0
#: A crosslink is drawn as a bracket this far outside the boxes, and the next one this much further.
BRACKET_OUT = 9.0
BRACKET_STEP = 6.0
#: Where atoms of a residue are drawn beside its box: the room between the ink of one residue's atoms
#: and the ink of another residue, its atoms or its box.
PIECE_GAP = 6.0
#: What is bonded to nothing of the chain (an ion that the chain and the protein hold between them) stands this
#: many bond lengths from the box of its residue: as far as such an ion stands from its donor in a drawing of atoms.
_LOOSE = 1.35
#: How far every atom of a piece that hangs off a residue stands clear of the box's edge, in bond lengths.
_CLEAR = 0.3


def unit_label(res: PeptideResidue) -> tuple[str, str]:
    """What a residue's box says: its code, and its number (as deposited, or its position in the chain)."""
    return res.code3, res.shown


def unit_width(peptide: Peptide) -> float:
    """One width for all boxes of a chain: the column reads as a column."""
    widest = max(text_width(code, UNIT_FONT, bold=True) + _GAP + text_width(num, INDEX_FONT)
                 for code, num in (unit_label(r) for r in peptide.residues))
    return math.ceil(widest + 2.0 * _PAD)


def _class_of(res: PeptideResidue) -> str:
    return residue_class(res.name) if res.known else "other"


def _sub_ligand(lig: Ligand, keep: Sequence[int]) -> tuple[Ligand, list[int]]:
    """A ligand made of some atoms of another, and for each of its atoms the index it had there."""
    order = sorted(set(keep))
    rw = Chem.RWMol(lig.mol)
    wanted = set(order)
    for idx in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetIdx() not in wanted), reverse=True):
        rw.RemoveAtom(idx)
    mol = rw.GetMol()
    with quiet_rdkit():
        Chem.SanitizeMol(mol, catchErrors=True)
    return Ligand(mol=mol, names=[lig.names[i] for i in order], coords3d=np.asarray(lig.coords3d)[order],
                  bond_source=lig.bond_source), order


def _rotated(points: np.ndarray, about: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    rel = points - about
    return about + np.column_stack([rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c])


def _place_appendage(lig: Ligand, app: Appendage, metrics: Metrics, method: str, start: Point, direction: Point,
                     exact: "Mapping[str, Point] | None") -> tuple[Depiction, list[int]]:
    """The appendage drawn as atoms, growing from ``start`` in ``direction``.

    ``start`` is where the bond to the chain leaves the residue's box. The
    atoms are laid out as for any small molecule and then turned as one body,
    so that the piece runs away from the chain. A piece that no bond holds to
    the chain is not turned: it is set off from the box, level with ``start``.
    """
    holders = sorted({a for _, a, _ in app.attachments})
    sub, order = _sub_ligand(lig, list(app.atoms) + holders)
    back = {orig: k for k, orig in enumerate(order)}
    own = {lig.names[a] for a in app.atoms}
    if exact is not None and all(n in exact for n in own):
        # A stored layout: every atom where the graph has it; the holder where the bond leaves the box.
        final = {n: exact.get(n, start) for n in sub.names}
    else:
        first = depict_ligand(sub, metrics, method=method)
        xy = np.array([first.atoms[k].pos for k in range(len(order))], dtype=float)
        if app.attachments:
            holder = back[app.attachments[0][1]]
            mine = [back[a] for a in app.atoms]
            away = xy[mine].mean(axis=0) - xy[holder]
            if float(np.hypot(*away)) < 1e-6:
                away = xy[back[app.attachments[0][2]]] - xy[holder]
            turn = math.atan2(direction[1], direction[0]) - math.atan2(float(away[1]), float(away[0]))
            # A long piece turned by its middle can fold back over the column: of the turns 15 degrees apart,
            # the nearest one that keeps every atom of the piece clear of the box's edge, and where none does,
            # the clearest, set off from the edge.
            out = np.array(direction, dtype=float)
            margin = _CLEAR * metrics.bond_length
            best: "tuple[float, np.ndarray] | None" = None
            for step in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6):
                tried = _rotated(xy, xy[holder], turn + math.radians(15.0 * step))
                tried = tried + (np.array(start, dtype=float) - tried[holder])
                clear = float(((tried[mine] - np.array(start, dtype=float)) @ out).min())
                if best is None or clear > best[0]:
                    best = (clear, tried)
                if clear >= margin:
                    best = (clear, tried)
                    break
            assert best is not None
            xy = best[1] if best[0] >= margin else best[1] + out * (margin - best[0])
        else:
            # Its atom nearest to the box at that distance from it, and the piece level with where it starts.
            out = np.array(direction, dtype=float)
            across = np.array([-out[1], out[0]])
            xy = xy - out * float((xy @ out).min()) - across * float((xy @ across).mean())
            xy = xy + np.array(start, dtype=float) + out * _LOOSE * metrics.bond_length
        final = {n: (float(xy[k, 0]), float(xy[k, 1])) for k, n in enumerate(sub.names)}
    bl = metrics.bond_length
    coords = {n: (p[0] / bl, -p[1] / bl) for n, p in final.items()}
    dep = depict_ligand(sub, metrics, coords=coords, normalise=False, exact=dict(final))
    return dep, order


def _nearest_residue(lig: Ligand, peptide: Peptide, app: Appendage) -> int:
    """The position of the residue beside whose box a piece stands that no bond holds to the chain: of an ion
    that the chain holds (``lig.coordination``), the residue of the nearest atom that holds it; of anything
    else, the residue of the atom nearest to it in the model."""
    mine = list(app.atoms)
    donors = [d for m, d in lig.coordination if m in mine and peptide.residue_of(d) is not None]
    among = donors or [a for res in peptide.residues for a in res.atoms]
    xyz = np.asarray(lig.coords3d, dtype=float)
    apart = np.linalg.norm(xyz[among][:, None, :] - xyz[mine][None, :, :], axis=2).min(axis=1)
    res = peptide.residue_of(among[int(np.argmin(apart))])
    assert res is not None
    return res.position


def _column(n: int, pitch: float) -> list[Point]:
    return [(0.0, k * pitch) for k in range(n)]


# ---------------------------------------------------------------------------
# Atoms of a residue drawn beside its box
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Piece:
    """The atoms of one residue of the column that are drawn as atoms: where each stands, relative to
    the centre of the residue's box; the rectangles their ink fills there, atom by atom and half a bond
    at a time (for the room between residues: one rectangle round a whole side chain would claim the
    empty corners beside it); and the parts they come in -- the side chain, the nitrogen, the carbonyl
    -- each with the point of the box it starts from, if it starts from one."""

    xy: "dict[int, Point]"
    rects: "tuple[Box, ...]"
    parts: "tuple[tuple[tuple[int, ...], Point | None], ...]" = ()
    #: The bonds among these atoms, and the bonds that hold them to the box: from a point of the box
    #: (relative to its centre) to an atom.
    bonds: "tuple[tuple[int, int], ...]" = ()
    stems: "tuple[tuple[Point, int], ...]" = ()


def shown_atoms(peptide: Peptide, involved: "Mapping[int, Collection[int]]") -> dict[int, tuple[int, ...]]:
    """Which atoms of which residue are drawn beside its box, by position in the chain.

    ``involved`` are, for a position, the atoms of that residue that a directional contact names.
    Only such a residue is drawn with atoms, and only if ``peptide.detail`` asks for more than boxes:
    then the part that makes the contact and what the level adds (:func:`bindview.fragments.fragment_atoms`),
    without the alpha carbon, which is the box. A proline's nitrogen stays in the box too: its ring
    closes on it. A residue that is not one of the twenty, or a contact through an atom the template
    does not have, leaves the residue a box.
    """
    from .fragments import fragment_atoms

    if peptide.detail in (None, "box"):
        return {}
    out: dict[int, tuple[int, ...]] = {}
    for res in peptide.residues:
        atoms = involved.get(res.position)
        if not atoms or not res.known:
            continue
        local = {i: n for i, n in res.atom_names}
        if any(i not in local for i in atoms):
            continue
        keep = fragment_atoms(res.name, [local[i] for i in atoms], peptide.detail)
        if keep is None:
            continue
        if res.kind == "nucleotide":
            # Under the template's names (a file may say O1P for OP1); the box is the sugar's C4'.
            local = {i: canonical_atom(res.name, n) for i, n in local.items()}
            hidden = {"C4'"}
        else:
            hidden = {"CA", "N"} if canonical_resname(res.name) == "PRO" else {"CA"}
        chosen = tuple(i for i in res.atoms if local.get(i) in keep and local[i] not in hidden)
        if chosen:
            out[res.position] = chosen
    return out


def piece_sides(order: "Sequence[int]", partners: "Mapping[int, Collection[Hashable]]") -> dict[int, int]:
    """On which side of the column each residue's side chain stands: +1 right, -1 left.

    ``order`` are the positions of the residues that are drawn with atoms, from the top of the column;
    ``partners`` are, for a position, whatever makes directional contacts with its drawn atoms. Two
    residues that share a partner stand on the same side, so that the partner does not have to be on
    both sides of the column; the groups this makes alternate, the topmost to the right.
    """
    group = {p: p for p in order}

    def find(p: int) -> int:
        while group[p] != p:
            group[p] = group[group[p]]
            p = group[p]
        return p

    seen: dict[Hashable, int] = {}
    for p in order:
        for partner in sorted(partners.get(p, ()), key=repr):
            if partner in seen:
                group[find(p)] = find(seen[partner])
            else:
                seen[partner] = p
    sides: dict[int, int] = {}
    turn: dict[int, int] = {}
    for p in order:
        root = find(p)
        if root not in turn:
            turn[root] = 1 if len(turn) % 2 == 0 else -1
        sides[p] = turn[root]
    return sides


def lay_piece(res: PeptideResidue, shown: "Sequence[int]", side: int, half_w: float, half_h: float,
              metrics: Metrics) -> Piece:
    """Where the atoms of a residue stand that are drawn beside its box.

    The box is the alpha carbon. The side chain stands on ``side`` of it, laid out as its template is
    and turned so that its middle is on the level of the box. The backbone nitrogen stands on the line
    of the column above the box and the carbonyl carbon below it, as the chain runs; the oxygen beside
    that carbon, on the side away from the side chain.
    """
    from .fragments import _unit_layout

    if res.kind == "nucleotide":
        return _lay_nucleotide(res, shown, side, half_w, half_h, metrics)
    length = metrics.bond_length * metrics.protein_bond_scale
    font = metrics.protein_atom_font
    local = {i: n for i, n in res.atom_names}
    by_name = {local[i]: i for i in shown}
    xy: dict[int, Point] = {}
    template = template_for(res.name)
    assert template is not None
    chain = tuple(n for n in template.sidechain if n in by_name)
    if chain:
        unit, _rings = _unit_layout(canonical_resname(res.name), ("CA",) + chain)
        pts = np.array(unit, dtype=float)
        pts = pts - pts[0]
        middle = pts[1:].mean(axis=0)
        turn = -math.atan2(float(middle[1]), float(middle[0]))
        c, s = math.cos(turn), math.sin(turn)
        for name, (px, py) in zip(chain, pts[1:]):
            x, y = float(px * c - py * s), float(px * s + py * c)
            xy[by_name[name]] = (side * (half_w + x * length), -y * length)
    if "N" in by_name:
        xy[by_name["N"]] = (0.0, -half_h - length)
    if "C" in by_name:
        xy[by_name["C"]] = (0.0, half_h + length)
    if "O" in by_name:
        base = xy[by_name["C"]] if "C" in by_name else (0.0, half_h)
        xy[by_name["O"]] = (base[0] - side * length * math.sin(math.radians(60.0)),
                            base[1] + length * math.cos(math.radians(60.0)))

    parts: "list[tuple[tuple[int, ...], Point | None]]" = []
    if chain:
        parts.append((tuple(by_name[n] for n in chain), (side * half_w, 0.0)))
    if "N" in by_name:
        parts.append(((by_name["N"],), None))
    below = tuple(by_name[n] for n in ("C", "O") if n in by_name)
    if below:
        parts.append((below, None))
    names = {i: local[i] for i in shown}
    from .residues import template_bonds

    bonds = tuple((by_name[a], by_name[b]) for a, b, _order in template_bonds(canonical_resname(res.name))
                  if a in by_name and b in by_name)
    stems: "list[tuple[Point, int]]" = []
    if chain:
        stems.append(((side * half_w, 0.0), by_name[chain[0]]))
        if canonical_resname(res.name) == "PRO" and "CD" in by_name:
            stems.append(((side * half_w, 0.0), by_name["CD"]))           # the ring closes on the box
    if "N" in by_name:
        stems.append(((0.0, -half_h), by_name["N"]))
    if "C" in by_name:
        stems.append(((0.0, half_h), by_name["C"]))
    return Piece(xy, ink_rects(xy, names, font, bonds, stems), tuple(parts), bonds, tuple(stems))


def _lay_nucleotide(res: PeptideResidue, shown: "Sequence[int]", side: int, half_w: float, half_h: float,
                    metrics: Metrics) -> Piece:
    """Where the atoms of a nucleotide stand that are drawn beside its box (docs/reference.md has the rule).

    The box is the sugar's C4'. The base, and the atoms of the sugar that are drawn other than C4' and O3',
    stand on ``side`` of it, laid out as their template is and turned so that their middle is on the level
    of the box; where neither C3' nor O4' is drawn, the base hangs on the box by C1'. C5', O5' and P stand
    on the line of the column above the box, where the chain comes in, the phosphate's free oxygens beside
    P; O3' below the box, where the chain goes on.
    """
    from .fragments import _unit_layout
    from .residues import template_bonds

    length = metrics.bond_length * metrics.protein_bond_scale
    font = metrics.protein_atom_font
    resname = canonical_resname(res.name)
    local = {i: canonical_atom(res.name, n) for i, n in res.atom_names}
    by_name = {local[i]: i for i in shown}
    template = template_for(res.name)
    assert template is not None
    xy: dict[int, Point] = {}
    above = (*TAIL, "P")                        # C5', O5', P: one above the other, from the box up
    beside = tuple(n for n in template.atoms if n in by_name and n not in PHOSPHATE + TAIL + ("C4'", "O3'"))
    attach = tuple(n for n in ("C3'", "O4'") if n in beside)
    if beside:
        if attach:
            unit, _rings = _unit_layout(resname, ("C4'",) + beside)
            pts = np.array(unit, dtype=float)
            anchor, rest = pts[0], pts[1:]
        else:
            # Hung on the box by C1': a point one bond beyond C1', away from the base, stands for C4'.
            unit, _rings = _unit_layout(resname, beside)
            rest = np.array(unit, dtype=float)
            root = rest[beside.index("C1'")] if "C1'" in beside else rest[0]
            away = root - rest.mean(axis=0)
            norm = float(np.linalg.norm(away))
            anchor = root + (away / norm if norm > 1e-9 else np.array([-1.0, 0.0]))
        rest = rest - anchor
        middle = rest.mean(axis=0)
        turn = -math.atan2(float(middle[1]), float(middle[0]))
        c, s = math.cos(turn), math.sin(turn)
        for name, (px, py) in zip(beside, rest):
            x, y = float(px * c - py * s), float(px * s + py * c)
            xy[by_name[name]] = (side * (half_w + x * length), -y * length)
    y_up = -half_h
    for name in above:
        if name in by_name:
            y_up -= length
            xy[by_name[name]] = (0.0, y_up)
    if "P" in by_name:
        px, py = xy[by_name["P"]]
        for name, way in (("OP1", -side), ("OP2", side)):
            if name in by_name:
                xy[by_name[name]] = (px + way * length * math.sin(math.radians(60.0)),
                                     py - length * math.cos(math.radians(60.0)))
    if "O3'" in by_name:
        xy[by_name["O3'"]] = (0.0, half_h + length)

    parts: "list[tuple[tuple[int, ...], Point | None]]" = []
    if beside:
        parts.append((tuple(by_name[n] for n in beside), (side * half_w, 0.0)))
    up = tuple(by_name[n] for n in (*above, "OP1", "OP2") if n in by_name)
    if up:
        parts.append((up, (0.0, -half_h)))
    if "O3'" in by_name:
        parts.append(((by_name["O3'"],), (0.0, half_h)))
    names = {i: local[i] for i in shown}
    bonds = tuple((by_name[a], by_name[b]) for a, b, _order in template_bonds(resname)
                  if a in by_name and b in by_name)
    stems: "list[tuple[Point, int]]" = []
    for name in (attach or (("C1'",) if "C1'" in beside else beside[:1])):
        stems.append(((side * half_w, 0.0), by_name[name]))
    lowest = next((n for n in above if n in by_name), None)
    if lowest is not None:
        stems.append(((0.0, -half_h), by_name[lowest]))
    if "O3'" in by_name:
        stems.append(((0.0, half_h), by_name["O3'"]))
    return Piece(xy, ink_rects(xy, names, font, bonds, stems), tuple(parts), bonds, tuple(stems))


def ink_rects(xy: "Mapping[int, Point]", names: "Mapping[int, str]", font: float,
              bonds: "Sequence[tuple[int, int]]", stems: "Sequence[tuple[Point, int]]",
              at: Point = (0.0, 0.0)) -> "tuple[Box, ...]":
    """The ink of a piece as small rectangles: every atom (a carbon is a corner of the drawing, any
    other atom its name in ``font``), and every bond in two halves. One rectangle round a whole side
    chain would claim the empty corners beside it. ``stems`` start at points of the box, which are
    given relative to ``at``; ``xy`` are the atoms where they stand."""
    # (A carbon is where bonds meet: their halves are its ink. Only an atom with a name has more.)
    out = list(piece_rects([((i,), None) for i in xy if names[i][:1] != "C"], xy, names, font))
    lines = [(xy[a], xy[b]) for a, b in bonds] + [((at[0] + s[0], at[1] + s[1]), xy[i]) for s, i in stems]
    for (ax, ay), (bx, by) in lines:
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        for (px, py), (qx, qy) in (((ax, ay), (mx, my)), ((mx, my), (bx, by))):
            out.append((min(px, qx) - 1.0, min(py, qy) - 1.0, max(px, qx) + 1.0, max(py, qy) + 1.0))
    return tuple(out)


def piece_rects(parts: "Sequence[tuple[Sequence[int], Point | None]]", xy: "Mapping[int, Point]",
                names: "Mapping[int, str]", font: float, at: Point = (0.0, 0.0)) -> "tuple[Box, ...]":
    """The rectangle that the ink of each part of a piece fills: its atoms where ``xy`` has them (a
    carbon is a corner of the drawing, any other atom its name in ``font``), and the point of the box
    the part starts from, which is given relative to ``at``."""
    def ink(i: int) -> Box:
        x, y = xy[i]
        if names[i][:1] == "C":
            return (x - 2.0, y - 2.0, x + 2.0, y + 2.0)
        w, h = text_width(names[i], font) + 2.0, CAP_HEIGHT * font + 2.0
        return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)

    rects: list[Box] = []
    for atoms, start in parts:
        boxes = [ink(i) for i in atoms]
        if start is not None:
            boxes.append((at[0] + start[0], at[1] + start[1], at[0] + start[0], at[1] + start[1]))
        rects.append((min(b[0] for b in boxes), min(b[1] for b in boxes),
                      max(b[2] for b in boxes), max(b[3] for b in boxes)))
    return tuple(rects)


def stack_column(n: int, pitch: float, half_w: float, half_h: float, pieces: "Mapping[int, Piece]") -> list[Point]:
    """The centres of the boxes of a column in which some residues have atoms beside their boxes.

    Boxes are ``pitch`` apart, as in every column. A residue is moved further down by the least it takes
    for the ink of its atoms to keep :data:`PIECE_GAP` from the ink of every residue above it, atoms
    and box, and for its box to keep that from their atoms. ``pieces`` by index in the column.
    """
    box: Box = (-half_w, -half_h, half_w, half_h)
    ys = [0.0]
    for k in range(1, n):
        y = ys[k - 1] + pitch
        mine = pieces[k].rects if k in pieces else ()
        for j in range(k):
            theirs = pieces[j].rects if j in pieces else ()
            pairs = [(a, b) for a in theirs for b in mine] + [(a, box) for a in theirs] + [(box, b) for b in mine]
            for a, b in pairs:
                if a[0] < b[2] + PIECE_GAP and b[0] < a[2] + PIECE_GAP:
                    y = max(y, ys[j] + a[3] + PIECE_GAP - b[1])
        ys.append(y)
    return [(0.0, y) for y in ys]


def _name_label(pos: Point, text: str, font: float) -> AtomLabel:
    """An atom's name as its label, the way the fragments around a chain carry theirs."""
    x, y = pos
    w, cap = text_width(text, font), CAP_HEIGHT * font
    box: Box = (x - w / 2.0, y - cap / 2.0, x + w / 2.0, y + cap / 2.0)
    return AtomLabel((LabelRun(x, y + cap / 2.0, "middle", (Span(text, size=font),), box),), box)


def depict_polymer(lig: Ligand, peptide: Peptide, metrics: Metrics, *, method: str = "auto",
                   unit_xy: "Mapping[int, Point] | None" = None,
                   exact: "Mapping[str, Point] | None" = None,
                   plane: "Arrangement | None" = None, scale: "float | None" = None,
                   shown: "Mapping[int, Sequence[int]] | None" = None,
                   sides: "Mapping[int, int] | None" = None, labels: str = "names") -> Depiction:
    """The chain depiction of a peptide ligand.

    ``unit_xy`` (position -> centre) and ``exact`` (atom name -> point, for
    atoms outside the chain) redraw a stored layout: positions are then taken
    as they are.

    With ``plane`` the residues do not stand in a column: each box stands
    where its residue lies on the plane through them (:mod:`bindview.plane`),
    at one scale, and boxes that would cover each other are pushed apart by
    the least it takes. What that did to the arrangement is measured and goes
    with the depiction (``Depiction.plane``). ``scale`` (page units per
    ångström) is the scale of a stored layout.

    ``shown`` (position -> atoms, :func:`shown_atoms`) are the atoms of a
    residue that are drawn beside its box, ``sides`` (position -> +1 or -1,
    :func:`piece_sides`) the side its side chain stands on, ``labels`` what is
    written on them (:func:`bindview.residues.written`), which moves none of
    them: they are stacked by the room their names take. A column only:
    on a plane there is no side of a box that is nobody's.
    """
    n = len(peptide)
    width = unit_width(peptide)
    half_w, half_h = width / 2.0, UNIT_HEIGHT / 2.0
    pitch = max(MIN_PITCH, metrics.chain_pitch * metrics.bond_length)
    on_plane = None
    if plane is not None and len(plane.xy) != n:
        raise ValueError(f"{len(plane.xy)} places on the plane for {n} residues")
    pieces: dict[int, Piece] = {}                     # by index in the column
    if shown and plane is None:
        for k, res in enumerate(peptide.residues):
            if shown.get(res.position):
                pieces[k] = lay_piece(res, shown[res.position], (sides or {}).get(res.position, 1), half_w,
                                      half_h, metrics)
    if plane is not None and unit_xy is None:
        centres, on_plane = plane.place(pitch, half_w, half_h)
    elif pieces:
        centres = stack_column(n, pitch, half_w, half_h, pieces)
    else:
        centres = _column(n, pitch)
    if unit_xy is not None:
        given = [unit_xy.get(r.position, centres[k]) for k, r in enumerate(peptide.residues)]
        centres = [(float(p[0]), float(p[1])) for p in given]
        if plane is not None:
            # A stored layout: the boxes are where they were put, and are measured where they are.
            on_plane = plane.measure(centres, scale if scale else plane.scale(pitch), None, half_w, half_h)

    mol = lig.mol
    atoms: list[DepAtom | None] = [None] * lig.n_atoms
    units: list[DepUnit] = []
    for k, res in enumerate(peptide.residues):
        cx, cy = centres[k]
        box: Box = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        code, number = unit_label(res)
        piece = pieces.get(k)
        beside: dict[int, Point] = {}
        if piece is not None:
            # Atoms of their own beside the box: where the piece has them, or where a stored layout does.
            for i, (ox, oy) in piece.xy.items():
                beside[i] = (cx + ox, cy + oy)
                if exact is not None and lig.names[i] in exact:
                    beside[i] = (float(exact[lig.names[i]][0]), float(exact[lig.names[i]][1]))
        units.append(DepUnit(index=k, position=res.position, name=res.name, text=code, x=cx, y=cy, box=box,
                             anchor=res.ca, atoms=res.atoms,
                             backbone=tuple(a for a in res.atoms if res.is_backbone(a)),
                             residue_class=_class_of(res), number=number,
                             beside=tuple(sorted(beside)),
                             side=(sides or {}).get(res.position, 1) if piece is not None else 0,
                             n=res.n, c=res.c,
                             rects=ink_rects(beside, {i: written(res.name_of(i) or "C", labels) for i in beside},
                                             metrics.protein_atom_font, piece.bonds, piece.stems, (cx, cy))
                             if piece is not None else ()))
        for i in res.atoms:
            atom = mol.GetAtomWithIdx(i)
            if i in beside:
                px, py = beside[i]
                word = written(res.name_of(i) or atom.GetSymbol(), labels)
                own = None if atom.GetSymbol() == "C" else _name_label((px, py), word, metrics.protein_atom_font)
                atoms[i] = DepAtom(idx=i, x=px, y=py, element=atom.GetSymbol(), name=lig.names[i],
                                   hcount=atom.GetTotalNumHs(), charge=atom.GetFormalCharge(),
                                   isotope=atom.GetIsotope(), label=own, aromatic=atom.GetIsAromatic())
                continue
            # The box is the residue's ink: one label, on its alpha carbon, so that the field and the
            # audit see it once.
            label = AtomLabel(runs=(LabelRun(cx, cy, "middle", (), box),), box=box) if i == res.ca else None
            atoms[i] = DepAtom(idx=i, x=cx, y=cy, element=atom.GetSymbol(), name=lig.names[i],
                               hcount=atom.GetTotalNumHs(), charge=atom.GetFormalCharge(),
                               isotope=atom.GetIsotope(), label=label, aromatic=atom.GetIsAromatic())

    bonds: list[DepBond] = []
    for k in range(n - 1):
        if peptide.residues[k].position in peptide.breaks:
            # Residues are missing here (from the model, or left out because they are not at the
            # interface that is drawn): no backbone, but a dotted line that says the chain goes on.
            # On a plane only within a stretch: a dotted line from one loop to the next would cross
            # the figure and say nothing.
            if plane is None:
                dotted = peptide.sequential
            else:
                # ... and not where it would run under a third box: it would be read as that box's line.
                dotted = (peptide.residues[k].position in plane.joins
                          and not any(passes_through(centres[k], centres[k + 1], u.box)
                                      for u in units if u.index not in (k, k + 1)))
            if dotted:
                bonds.append(DepBond(peptide.residues[k].ca, peptide.residues[k + 1].ca, 1, kind="gap"))
            continue
        bonds.append(DepBond(peptide.residues[k].ca, peptide.residues[k + 1].ca, 1, kind="chain"))
    if peptide.cyclic and n > 2:
        bonds.append(DepBond(peptide.residues[-1].ca, peptide.residues[0].ca, 1, kind="closure"))
    for x in peptide.crosslinks:
        bonds.append(DepBond(peptide.residue(x.a[0]).ca, peptide.residue(x.b[0]).ca, 1, kind="crosslink"))

    notes: list[str] = []
    extra: list[DepAtom] = []
    rings: list[tuple[int, ...]] = []
    taken: list[tuple[int, int]] = []                 # (residue position, side) already holding an appendage
    explicit = {i for piece in pieces.values() for i in piece.xy}
    if explicit:
        # The bonds of the atoms that stand beside their boxes, with the orders the molecule has.
        kekulized = Chem.Mol(mol)
        flat = True
        with quiet_rdkit():
            try:
                Chem.Kekulize(kekulized, clearAromaticFlags=True)
            except Exception:                           # not kekulizable: aromatic bonds are drawn as such
                flat = False
        home = {i: k for k, res in enumerate(peptide.residues) for i in res.atoms}
        for mb in (kekulized if flat else mol).GetBonds():
            ia, ib = mb.GetBeginAtomIdx(), mb.GetEndAtomIdx()
            if ia not in explicit and ib not in explicit:
                continue
            ka, kb = home.get(ia), home.get(ib)
            if ka is not None and kb is not None and ka != kb:
                # A bond from one residue to another is drawn where the two stand one above the other
                # and it is the peptide bond between them, carbonyl carbon above, nitrogen below. A
                # column in the order of the interface has neighbours in the chain far apart: a bond
                # between those would be a line across the boxes in between.
                upper, lower = (ia, ib) if ka < kb else (ib, ia)
                up, low = peptide.residues[min(ka, kb)], peptide.residues[max(ka, kb)]
                if abs(ka - kb) != 1 or upper != up.c or lower != low.n:
                    continue
            ringed = mb.GetIsAromatic() and not flat
            mult = 1 if ringed else max(1, min(3, int(round(mb.GetBondTypeAsDouble()))))
            bonds.append(DepBond(ia, ib, mult, aromatic=ringed))
        rings.extend(tuple(ring) for ring in mol.GetRingInfo().AtomRings() if all(i in explicit for i in ring))
        # An appendage goes to the side of the box that the residue's own side chain does not stand on.
        taken.extend((peptide.residues[k].position, (sides or {}).get(peptide.residues[k].position, 1))
                     for k in sorted(pieces))
    for app in sorted(peptide.appendages, key=lambda a: (a.attachments, a.atoms)):
        if app.attachments:
            position, holder, _first = app.attachments[0]
            name = peptide.residue(position).name_of(holder)
        else:
            position, name = _nearest_residue(lig, peptide, app), None
        cx, cy = centres[position - 1]
        if name == "N" and position == 1 and not peptide.cyclic:
            start, direction = (cx, cy - half_h), (0.0, -1.0)           # a cap on the N-terminus: upward
        elif name in ("C", "OXT") and position == n and not peptide.cyclic:
            start, direction = (cx, cy + half_h), (0.0, 1.0)            # on the C-terminus: downward
        else:
            side = 1 if (position, 1) not in taken else -1              # to the right, then to the left
            taken.append((position, side))
            start, direction = (cx + side * half_w, cy), (float(side), 0.0)
        sub, order = _place_appendage(lig, app, metrics, method, start, direction, exact)
        local: dict[int, int] = {}                    # index in the piece -> index in this depiction
        for k, orig in enumerate(order):
            local[k] = orig
            if orig in app.atoms:
                d = sub.atoms[k]
                atoms[orig] = DepAtom(idx=orig, x=d.x, y=d.y, element=d.element, name=d.name, hcount=d.hcount,
                                      charge=d.charge, isotope=d.isotope, label=d.label, aromatic=d.aromatic)
        for d in sub.atoms[len(order):]:              # hydrogens the drawing adds at a stereocentre
            local[d.idx] = lig.n_atoms + len(extra)
            extra.append(DepAtom(idx=local[d.idx], x=d.x, y=d.y, element=d.element, name=None, hcount=d.hcount,
                                 charge=d.charge, isotope=d.isotope, label=d.label, aromatic=d.aromatic))
        for b in sub.bonds:
            ia, ib = local[b.a], local[b.b]
            if ia not in app.atoms and ib not in app.atoms and ia < lig.n_atoms and ib < lig.n_atoms:
                continue                              # a bond between two atoms of the chain: not this piece's
            bonds.append(DepBond(ia, ib, b.order, b.stereo, b.aromatic, b.kind))
        rings.extend(tuple(local[i] for i in ring) for ring in sub.rings
                     if all(local[i] in app.atoms for i in ring))
        notes.extend(sub.notes)

    placed = [d for d in atoms if d is not None]
    if len(placed) != lig.n_atoms:                    # cannot happen: every atom is in a residue or an appendage
        raise ValueError(f"{lig.n_atoms - len(placed)} ligand atom(s) are in neither a residue nor an appendage")
    return Depiction(atoms=placed + extra, bonds=bonds, rings=rings, n_heavy=lig.n_atoms, metrics=metrics,
                     coord_source="chain", notes=notes, units=tuple(units),
                     open_ends=peptide.open_ends if (peptide.sequential and plane is None) else (False, False),
                     pairwise=peptide.pairwise, sequential=peptide.sequential, called=peptide.called,
                     arranged="plane" if plane is not None else "column", plane=on_plane)
