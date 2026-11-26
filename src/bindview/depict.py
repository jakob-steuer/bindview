"""From a named-atom molecule to drawable geometry.

Produces a :class:`Depiction`: 2D positions for every ligand atom (in SVG user
units, y pointing down), the atom labels with their boxes, the bonds with
order and stereo style, and the ring membership needed to place double bonds.
Nothing here knows about residues or interactions.

Coordinates come from RDKit's bundled CoordGen (BSD-3), which draws ring
systems and macrocycles the way a chemist would, with RDKit's own depictor as a
fallback; the better of the two by a clash / crossing count is kept. The choice
and everything after it are deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdCoordGen, rdDepictor
from rdkit.Geometry import Point3D

from .chem import is_metal, quiet_rdkit
from .errors import BindviewError
from .geometry import Box, Point, box_expand, box_union, segment_box_entry, segments_intersect
from .ligand import Ligand
from .scene import Span
from .textmetrics import CAP_HEIGHT, DESCENT, text_width
from .theme import Metrics

__all__ = ["Depiction", "DepAtom", "DepBond", "AtomLabel", "LabelRun", "depict_ligand", "layout_score",
           "atom_keys"]

_MINUS = "−"


@dataclass(frozen=True)
class LabelRun:
    """One ``<text>`` element of an atom label: an anchor point plus spans."""

    x: float
    y: float                    # baseline
    anchor: str                 # start | middle | end
    spans: tuple[Span, ...]
    box: Box


@dataclass(frozen=True)
class AtomLabel:
    runs: tuple[LabelRun, ...]
    box: Box                    # union of all runs

    @property
    def boxes(self) -> tuple[Box, ...]:
        return tuple(r.box for r in self.runs)


@dataclass
class DepAtom:
    idx: int
    x: float
    y: float
    element: str
    name: str | None            # structure-file atom name; None for depiction-only hydrogens
    hcount: int = 0
    charge: int = 0
    isotope: int = 0
    label: AtomLabel | None = None
    aromatic: bool = False

    @property
    def pos(self) -> Point:
        return (self.x, self.y)


@dataclass
class DepBond:
    a: int
    b: int
    order: int                  # 1, 2, 3
    stereo: str = "none"        # none | wedge | hash  (narrow end at ``a``)
    aromatic: bool = False      # only set when the molecule could not be kekulized
    #: covalent | coordination, and in a chain ligand: chain (the peptide bond between two units),
    #: closure (the one that closes a ring of units) and crosslink (any other bond between two units)
    kind: str = "covalent"


@dataclass(frozen=True)
class DepUnit:
    """One residue of a chain ligand as it is drawn: a labelled box standing for all of its atoms.

    Every atom of the residue sits, in the depiction, at the centre of the box:
    a contact that names any of them lands on the residue. ``anchor`` is the
    atom that carries the box (the alpha carbon).

    ``beside`` are the atoms of the residue that are drawn beside the box as
    atoms of their own (a residue that makes a directional contact, where the
    chain is drawn with more than boxes: ``Peptide.detail``). They have places
    of their own, and a directional contact that names one of them lands on it.
    """

    index: int                  # into ``Depiction.units``
    position: int               # 1-based, from the N-terminus of the chain as perceived
    name: str                   # "LYS", "XAA"
    text: str                   # what is printed: "Lys", "D-Phe"
    x: float
    y: float
    box: Box
    anchor: int
    atoms: tuple[int, ...]
    backbone: tuple[int, ...]   # the atoms of it that are backbone
    residue_class: str = "other"
    number: str = ""            # what is printed beside the code: the deposited number, or the position
    beside: tuple[int, ...] = ()  # the atoms of it that stand beside the box as atoms of their own
    side: int = 0               # ... and on which side of the box its side chain stands: +1 right, -1 left
    n: int = -1                 # its backbone nitrogen and its carbonyl carbon (-1: not known), for the
    c: int = -1                 # link to the next residue where one of the two is drawn
    #: The rectangles that the ink of its drawn atoms fills, on the page: every atom, and every bond in
    #: two halves. A line that is another residue's does not run across one of them.
    rects: tuple[Box, ...] = ()

    @property
    def pos(self) -> Point:
        return (self.x, self.y)

    def part(self, atoms: Sequence[int]) -> str:
        """Which part of the residue these atoms are: ``backbone`` where all of them are, else the
        ``base`` of a nucleotide or the ``side chain`` of an amino acid."""
        if all(a in self.backbone for a in atoms):
            return "backbone"
        return "base" if self.residue_class == "nucleotide" else "side chain"


@dataclass
class Depiction:
    atoms: list[DepAtom]
    bonds: list[DepBond]
    rings: list[tuple[int, ...]]
    n_heavy: int                # atoms[:n_heavy] map 1:1 onto Ligand atoms
    metrics: Metrics
    coord_source: str = ""
    notes: list[str] = field(default_factory=list)
    #: The residues of a chain ligand, in chain order; empty for a ligand drawn as atoms.
    units: tuple[DepUnit, ...] = ()
    #: For a chain: does it go on beyond its first and its last unit (see ``Peptide.open_ends``)?
    open_ends: tuple[bool, bool] = (False, False)
    #: One line per pair of residues and kind of contact, instead of one per pair of atoms
    #: (the figure of an interface; see ``Peptide.pairwise``).
    pairwise: bool = False
    #: Are the units of a chain in the order of its sequence (see ``Peptide.sequential``)?
    sequential: bool = True
    #: What the chain is called in a tooltip (see ``Peptide.called``).
    called: str = "the ligand"
    #: How the units of a chain are set out: in a ``column``, or on a ``plane``, each where its residue
    #: lies in space (:mod:`bindview.plane`) ...
    arranged: str = "column"
    #: ... and then what became of that arrangement on the page (:class:`bindview.plane.Placed`).
    plane: Any = None
    _unit_of: dict[int, int] = field(default_factory=dict, repr=False, compare=False)
    _home_of: dict[int, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._home_of = {a: u.index for u in self.units for a in u.atoms}
        self._unit_of = {a: k for a, k in self._home_of.items() if a not in self.units[k].beside}

    # -- lookups -----------------------------------------------------------
    def pos(self, idx: int) -> Point:
        a = self.atoms[idx]
        return (a.x, a.y)

    def unit_of(self, idx: int) -> DepUnit | None:
        """The chain residue an atom is drawn as part of (its box stands for it), or ``None``: an atom
        outside the chain, or one that is drawn beside its residue's box as an atom of its own."""
        k = self._unit_of.get(idx)
        return None if k is None else self.units[k]

    def home_of(self, idx: int) -> DepUnit | None:
        """The chain residue an atom belongs to, whether its box stands for it or it is drawn beside it."""
        k = self._home_of.get(idx)
        return None if k is None else self.units[k]

    def unit_for(self, idxs: Sequence[int], whole: bool = False) -> DepUnit | None:
        """The chain residue whose box these atoms are all drawn as part of, or ``None``.

        ``whole``: the residue they all belong to, whether or not some of them are drawn as atoms
        of their own. A contact without a direction is one with the residue, whatever is drawn of it.
        """
        table = self._home_of if whole else self._unit_of
        found = {table.get(i) for i in idxs}
        if len(found) != 1:
            return None
        k = found.pop()
        return None if k is None else self.units[k]

    def landing(self, idxs: Sequence[int]) -> tuple[int, ...]:
        """The atoms a directional contact that names ``idxs`` lands on: those of them that are drawn
        as atoms of their own, where a residue has some drawn and some not; all of them otherwise."""
        own = tuple(i for i in idxs if i in self._home_of and i not in self._unit_of)
        return own if own and len(own) < len(idxs) and all(i in self._home_of for i in idxs) else tuple(idxs)

    def centroid(self, idxs: Sequence[int]) -> Point:
        xs = [self.atoms[i].x for i in idxs]
        ys = [self.atoms[i].y for i in idxs]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def neighbours(self, idx: int, covalent_only: bool = True) -> list[int]:
        out = []
        for b in self.bonds:
            if covalent_only and b.kind != "covalent":
                continue
            if b.a == idx:
                out.append(b.b)
            elif b.b == idx:
                out.append(b.a)
        return out

    def bounds(self) -> Box:
        boxes: list[Box] = []
        for a in self.atoms:
            boxes.append(a.label.box if a.label else (a.x, a.y, a.x, a.y))
        return box_union(boxes)


def clip_to_labels(pa: Point, pb: Point, la: AtomLabel | None, lb: AtomLabel | None,
                   gap: float) -> tuple[Point, Point] | None:
    """Shorten segment ``pa->pb`` so it stops short of the labels at either end."""
    a, b = pa, pb
    if la is not None:
        a = _enter(pb, pa, la, gap)
    if lb is not None:
        b = _enter(pa, pb, lb, gap)
    # Still pointing the same way and of positive length?
    if (b[0] - a[0]) * (pb[0] - pa[0]) + (b[1] - a[1]) * (pb[1] - pa[1]) <= 1e-6:
        return None
    return a, b


def _enter(outside: Point, centre: Point, label: AtomLabel, gap: float) -> Point:
    """First point where ``outside->centre`` meets any (padded) box of ``label``."""
    best: Point = centre
    best_d = float("inf")
    for box in label.boxes:
        hit = segment_box_entry(outside, centre, box_expand(box, gap))
        if hit == centre and not _contains(box_expand(box, gap), centre):
            continue                 # segment misses this box entirely
        d = math.hypot(hit[0] - outside[0], hit[1] - outside[1])
        if d < best_d:
            best_d, best = d, hit
    return best


def _contains(box: Box, p: Point) -> bool:
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

def layout_score(mol: Chem.Mol, xy: np.ndarray) -> tuple[int, int]:
    """``(atom clashes, bond crossings)`` of a 2D layout with unit bond length."""
    n = mol.GetNumAtoms()
    if n < 3:
        return (0, 0)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    clashes = int(((d < 0.45) & ~np.eye(n, dtype=bool)).sum() // 2)
    segs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    crossings = 0
    for i in range(len(segs)):
        a, b = segs[i]
        pa, pb = (xy[a, 0], xy[a, 1]), (xy[b, 0], xy[b, 1])
        for j in range(i + 1, len(segs)):
            c, e = segs[j]
            if a in (c, e) or b in (c, e):
                continue
            if segments_intersect(pa, pb, (xy[c, 0], xy[c, 1]), (xy[e, 0], xy[e, 1])):
                crossings += 1
    return (clashes, crossings)


def _conf_xy(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])


def _normalise(mol: Chem.Mol, xy: np.ndarray) -> np.ndarray:
    lengths = [float(np.linalg.norm(xy[b.GetBeginAtomIdx()] - xy[b.GetEndAtomIdx()])) for b in mol.GetBonds()]
    scale = float(np.median(lengths)) if lengths else 1.0
    return xy / scale if scale > 1e-9 else xy


def _coordgen(mol: Chem.Mol) -> np.ndarray:
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    rdCoordGen.AddCoords(m)
    return _normalise(m, _conf_xy(m))


def _rdkit_coords(mol: Chem.Mol) -> np.ndarray:
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    prefer = rdDepictor.GetPreferCoordGen()
    rdDepictor.SetPreferCoordGen(False)
    try:
        rdDepictor.Compute2DCoords(m, canonOrient=True)
    finally:
        rdDepictor.SetPreferCoordGen(prefer)
    return _normalise(m, _conf_xy(m))


def _choose_coords(mol: Chem.Mol, method: str, notes: list[str]) -> tuple[np.ndarray, str]:
    n = mol.GetNumAtoms()
    if n == 1:
        return np.zeros((1, 2)), "trivial"
    if method == "rdkit":
        return _rdkit_coords(mol), "rdkit"
    try:
        cg = _coordgen(mol)
        cg_score = layout_score(mol, cg)
    except Exception:  # pragma: no cover - CoordGen is robust; keep a net anyway
        cg, cg_score = None, (10**6, 10**6)
    if method == "coordgen" and cg is not None:
        return cg, "coordgen"
    if cg is not None and cg_score == (0, 0):
        return cg, "coordgen"
    rd = _rdkit_coords(mol)
    rd_score = layout_score(mol, rd)
    if cg is None or rd_score < cg_score:
        if rd_score != (0, 0):
            notes.append(f"ligand 2D layout has {rd_score[0]} atom clash(es) and {rd_score[1]} bond crossing(s)")
        return rd, "rdkit"
    notes.append(f"ligand 2D layout has {cg_score[0]} atom clash(es) and {cg_score[1]} bond crossing(s)")
    return cg, "coordgen"


def _orient_landscape(xy: np.ndarray) -> np.ndarray:
    """Rotate by a multiple of 30 degrees so the drawing is wider than tall.

    Multiples of 30 keep bonds on the usual drawing grid, and the orientation
    CoordGen chose (often a textbook one) is kept unless turning it shrinks the
    height noticeably.
    """
    if len(xy) < 3:
        return xy
    best_k, best_h = 0, None
    heights = []
    for k in range(6):
        ang = math.radians(30.0 * k)
        c, s = math.cos(ang), math.sin(ang)
        y = xy[:, 0] * s + xy[:, 1] * c
        heights.append(float(y.max() - y.min()))
    best_h = min(heights)
    if heights[0] <= best_h * 1.12 + 1e-9:
        best_k = 0
    else:
        best_k = min(range(6), key=lambda k: (round(heights[k], 6), min(k, 6 - k)))
    if best_k == 0:
        return xy
    # Prefer the smaller of the two equivalent turns (k*30 or k*30-180).
    ang = math.radians(30.0 * best_k if best_k <= 3 else 30.0 * best_k - 180.0)
    c, s = math.cos(ang), math.sin(ang)
    return np.column_stack([xy[:, 0] * c - xy[:, 1] * s, xy[:, 0] * s + xy[:, 1] * c])


def _place_fragments(mol: Chem.Mol, xy: np.ndarray, coordination: Sequence[tuple[int, int]],
                     names: Sequence[str], notes: list[str], bridging: Sequence[int] = ()
                     ) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Seat coordinated metals among their donors; move loose fragments clear. An ion between the ligand and the
    protein (``bridging``) is seated outside the ligand, beside the donors that hold it."""
    frags = [tuple(f) for f in Chem.GetMolFrags(mol)]
    drawn: list[tuple[int, int]] = []
    if len(frags) < 2:
        return xy, drawn
    xy = xy.copy()
    main = max(frags, key=lambda f: (len(f), -min(f)))
    others = sorted((f for f in frags if f != main), key=min)
    donors_of: dict[int, list[int]] = {}
    for metal, donor in coordination:
        donors_of.setdefault(metal, []).append(donor)

    main_idx = np.array(sorted(main))
    for m_idx in (b for b in bridging if b in donors_of):
        donors = sorted(donors_of[m_idx])
        placed = {k for k, _ in drawn}
        near_to = np.array([i for i in range(len(xy)) if i in main or (i in placed and i != m_idx)])
        best: "tuple[float, np.ndarray] | None" = None
        for d in donors:
            for k in range(24):
                ang = math.radians(15.0 * k)
                spot = xy[d] + 1.35 * np.array([math.cos(ang), math.sin(ang)])
                if float(np.linalg.norm(xy[near_to] - spot, axis=1).min()) < 1.0:
                    continue
                # Near to every donor that holds it, and away from everything else.
                far = sum(abs(float(np.linalg.norm(xy[e] - spot)) - 1.35) for e in donors)
                away = [int(i) for i in near_to if i not in donors]
                clear = float(np.linalg.norm(xy[away] - spot, axis=1).min()) if away else 9.0
                cost = round(far - 0.5 * min(clear, 2.5), 6)
                if best is None or cost < best[0]:
                    best = (cost, spot)
        if best is None:
            continue
        xy[m_idx] = best[1]
        others = [f for f in others if f != (m_idx,)]
        drawn.extend((m_idx, d) for d in donors if float(np.linalg.norm(xy[d] - best[1])) <= 2.6)
    seated: list[int] = []
    loose: list[tuple[int, ...]] = []
    for frag in others:
        done = False
        if len(frag) == 1 and is_metal(mol.GetAtomWithIdx(frag[0]).GetSymbol()) and frag[0] in donors_of:
            m_idx = frag[0]
            donors = sorted(donors_of[m_idx])
            pts = xy[donors]
            keep = list(range(len(donors)))
            for _ in range(4):
                centre = np.median(pts[keep], axis=0)
                near = [k for k in range(len(donors)) if np.linalg.norm(pts[k] - centre) <= 2.6]
                if len(near) < 2 or near == keep:
                    keep = near if len(near) >= 2 else keep
                    break
                keep = near
            if len(keep) >= 2:
                centre = pts[keep].mean(axis=0)
                # Only the covalent skeleton and metals already seated can block the seat;
                # loose fragments are moved out of the way afterwards.
                block = np.concatenate([main_idx, np.array(seated, dtype=int)])
                if float(np.linalg.norm(xy[block] - centre, axis=1).min()) >= 0.75:
                    xy[m_idx] = centre
                    seated.append(m_idx)
                    done = True
                    for k, d in enumerate(donors):
                        if k in keep:
                            drawn.append((m_idx, d))
                        else:
                            notes.append(
                                f"coordination {names[m_idx]}\u2013{names[d]} is not drawn: the two atoms are "
                                f"far apart in the 2D layout")
        if not done:
            loose.append(frag)

    for frag in loose:
        fa = np.array(frag)
        rest = np.array([i for i in range(len(xy)) if i not in frag])
        d = np.linalg.norm(xy[fa][:, None, :] - xy[rest][None, :, :], axis=2)
        if d.min() >= 1.6:
            continue
        # Park it to the right of everything else, level with the main fragment.
        right = float(xy[rest][:, 0].max())
        mid_y = float(xy[list(main)][:, 1].mean())
        cur = xy[fa]
        shift = np.array([right + 2.0 - float(cur[:, 0].min()), mid_y - float(cur[:, 1].mean())])
        xy[fa] = cur + shift
    return xy, drawn


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _needs_label(element: str, degree: int, charge: int, isotope: int) -> bool:
    if element != "C":
        return True
    return degree == 0 or charge != 0 or isotope != 0


def _charge_text(charge: int) -> str:
    sign = "+" if charge > 0 else _MINUS
    return (str(abs(charge)) if abs(charge) > 1 else "") + sign


def _h_side(pos: Point, nbrs: Sequence[Point]) -> str:
    """Where the hydrogens of a label go: E, W, N or S (N is up on the page)."""
    if not nbrs:
        return "E"
    vecs = []
    for q in nbrs:
        dx, dy = q[0] - pos[0], q[1] - pos[1]
        n = math.hypot(dx, dy)
        if n > 1e-9:
            vecs.append((dx / n, dy / n))
    if not vecs:
        return "E"
    if len(vecs) == 1:
        return "W" if vecs[0][0] > 0.2 else "E"
    best, best_score = "E", -2.0
    # y grows downward, so "N" (up) is the direction (0, -1).
    for side, d in (("E", (1.0, 0.0)), ("W", (-1.0, 0.0)), ("S", (0.0, 1.0)), ("N", (0.0, -1.0))):
        score = min(1.0 - (d[0] * v[0] + d[1] * v[1]) for v in vecs)   # 0 = on a bond, 2 = opposite
        if score > best_score + 0.05:
            best, best_score = side, score
    return best


def make_label(pos: Point, element: str, hcount: int, charge: int, isotope: int, nbrs: Sequence[Point],
               font: float, sub_font: float, charge_font: float) -> AtomLabel:
    """Lay out an atom label: symbol centred on the atom, hydrogens to the free side."""
    x, y = pos
    cap = CAP_HEIGHT * font
    base = y + cap / 2.0
    ws = text_width(element, font)
    desc = DESCENT * font * 0.8 if any(ch in "gjpqy" for ch in element) else 0.0
    runs: list[LabelRun] = [LabelRun(x, base, "middle", (Span(element),),
                                     (x - ws / 2.0, y - cap / 2.0, x + ws / 2.0, y + cap / 2.0 + desc))]
    side = _h_side(pos, nbrs) if hcount > 0 else "E"
    sub_drop = 0.26 * font
    sup_rise = 0.46 * font

    def h_spans() -> tuple[list[Span], float, float]:
        """Spans for 'H' plus count; returns (spans, width, extra depth below baseline)."""
        spans = [Span("H")]
        width = text_width("H", font)
        depth = 0.0
        if hcount > 1:
            cnt = str(hcount)
            spans.append(Span(cnt, size=sub_font, dy=sub_drop))
            spans.append(Span("", dy=-sub_drop))          # restore the baseline
            width += text_width(cnt, sub_font)
            depth = sub_drop
        return spans, width, depth

    charge_spans: list[Span] = []
    charge_w = 0.0
    if charge:
        txt = _charge_text(charge)
        charge_spans = [Span(txt, size=charge_font, dy=-sup_rise), Span("", dy=sup_rise)]
        charge_w = text_width(txt, charge_font)
    top_with_charge = y - cap / 2.0 - (sup_rise + CAP_HEIGHT * charge_font - cap if charge else 0.0)
    top_with_charge = min(top_with_charge, y - cap / 2.0)
    pad = 0.6

    if isotope:
        iso = str(isotope)
        wi = text_width(iso, charge_font)
        runs.append(LabelRun(x - ws / 2.0 - pad, base, "end",
                             (Span(iso, size=charge_font, dy=-sup_rise), Span("", dy=sup_rise)),
                             (x - ws / 2.0 - pad - wi, y - cap / 2.0 - sup_rise * 0.4, x - ws / 2.0 - pad, y)))

    right: list[Span] = []
    right_w, right_depth = 0.0, 0.0
    left: list[Span] = []
    left_w, left_depth = 0.0, 0.0
    if hcount > 0 and side == "E":
        hs, hw, hd = h_spans()
        right, right_w, right_depth = hs + charge_spans, hw + charge_w, hd
    elif hcount > 0 and side == "W":
        hs, hw, hd = h_spans()
        left, left_w, left_depth = hs, hw, hd
        right, right_w = charge_spans, charge_w
    elif hcount > 0:                       # N or S: hydrogens stacked above / below
        hs, hw, hd = h_spans()
        gap = 2.2
        if side == "S":
            hb = base + cap + gap
        else:
            hb = base - cap - gap - hd
        runs.append(LabelRun(x, hb, "middle", tuple(hs),
                             (x - hw / 2.0, hb - cap, x + hw / 2.0, hb + hd)))
        right, right_w = charge_spans, charge_w
    elif charge:
        only_right = [p for p in nbrs if p[0] - x > 0.2 * math.hypot(p[0] - x, p[1] - y)]
        if len(nbrs) == 1 and only_right:
            left, left_w = charge_spans, charge_w          # e.g. "-O" when the bond leaves to the right
        else:
            right, right_w = charge_spans, charge_w

    if right:
        x0 = x + ws / 2.0 + pad
        has_charge = bool(charge) and any(s.dy < 0 for s in right)
        top = top_with_charge if has_charge else y - cap / 2.0
        runs.append(LabelRun(x0, base, "start", tuple(right),
                             (x0, top, x0 + right_w, y + cap / 2.0 + right_depth)))
    if left:
        x1 = x - ws / 2.0 - pad - (text_width(str(isotope), charge_font) + pad if isotope else 0.0)
        has_charge = bool(charge) and any(s.dy < 0 for s in left)
        top = top_with_charge if has_charge else y - cap / 2.0
        runs.append(LabelRun(x1, base, "end", tuple(left),
                             (x1 - left_w, top, x1, y + cap / 2.0 + left_depth)))
    return AtomLabel(tuple(runs), box_union(r.box for r in runs))


# ---------------------------------------------------------------------------
# The depiction
# ---------------------------------------------------------------------------

def depict_ligand(lig: Ligand, metrics: Metrics, method: str = "auto",
                  coords: dict[str, tuple[float, float]] | None = None, normalise: bool = True,
                  exact: "dict[str, tuple[float, float]] | None" = None,
                  coordination: "Sequence[tuple[str, str]] | None" = None) -> Depiction:
    """Compute the drawable geometry of a ligand.

    ``method`` is ``"auto"`` (CoordGen unless it yields clashes), ``"coordgen"``
    or ``"rdkit"``. ``coords`` overrides it with caller-supplied 2D positions
    keyed by atom name (any scale and orientation; y up). With
    ``normalise=False`` those positions are taken as they are, in units of one
    bond length: this is how several ligands are drawn in one common frame,
    where rescaling each by its own median bond would move shared atoms apart.

    ``exact`` and ``coordination`` exist to redraw a stored layout to the last
    digit. ``exact`` gives final drawing positions (user units, y down) by atom
    name -- and ``"H@<name>"`` for a hydrogen the drawing adds on that atom --
    which replace the computed ones before labels are set. ``coordination``
    names the metal-donor pairs to draw, in order, instead of all of them.
    """
    with quiet_rdkit():
        return _depict(lig, metrics, method, coords, normalise, exact, coordination)


def atom_keys(dep: "Depiction") -> list[str]:
    """A stable key for every depicted atom: its structure-file name, or ``H@<name>`` for a
    hydrogen that exists only in the drawing (on the atom it is bonded to)."""
    keys: list[str] = []
    seen: dict[str, int] = {}
    for atom in dep.atoms:
        if atom.name:
            key = atom.name
        else:
            nbrs = dep.neighbours(atom.idx)
            key = "H@" + (dep.atoms[nbrs[0]].name or "?" if nbrs else "?")
        seen[key] = seen.get(key, 0) + 1
        keys.append(key if seen[key] == 1 else f"{key}#{seen[key]}")
    return keys


def _part_added_hydrogens(dep: Depiction, placed: "set[int]") -> None:
    """Part two hydrogens of stereocentres that the drawing puts on one another.

    RDKit sets a hydrogen it adds at a stereocentre on the bisector of its atom's free angle. At two
    neighbouring ring fusions (C8 and C14 of a steroid) the two free angles open into one corner and the
    two bisectors end in one point: H on H. Where two such hydrogens are closer than half a bond, each
    turns within its own free angle, keeping clear of its atom's bonds, to where the two, and each of them
    from every other atom, are farthest apart. A hydrogen stays between the same two neighbours, so its
    wedge says what it said. Hydrogens nowhere near another are left where RDKit put them."""
    bl = dep.metrics.bond_length
    added = [a.idx for a in dep.atoms[dep.n_heavy:] if a.element == "H" and a.idx not in placed
             and len(dep.neighbours(a.idx)) == 1]
    margin = math.radians(30.0)

    def choices(h: int) -> list[Point]:
        (p,) = dep.neighbours(h)
        px, py = dep.pos(p)
        hx, hy = dep.pos(h)
        r = math.hypot(hx - px, hy - py)
        here = math.atan2(hy - py, hx - px)
        others = sorted((math.atan2(dep.atoms[j].y - py, dep.atoms[j].x - px) - here) % math.tau
                        for j in dep.neighbours(p) if j != h)
        if not others:
            return [(hx, hy)]
        lo, hi = others[-1] - math.tau, others[0]            # the free angle the hydrogen stands in, around 0
        if hi - lo <= 2 * margin:
            return [(hx, hy)]
        steps = 24
        turns = [lo + margin + (hi - lo - 2 * margin) * k / steps for k in range(steps + 1)]
        return [(px + r * math.cos(here + t), py + r * math.sin(here + t)) for t in turns]

    for i, h in enumerate(added):
        for k in added[i + 1:]:
            if math.dist(dep.pos(h), dep.pos(k)) >= 0.5 * bl:
                continue
            fixed = {h, k, *dep.neighbours(h), *dep.neighbours(k)}
            rest = [a.pos for a in dep.atoms if a.idx not in fixed]

            def clear(q: Point) -> float:
                return min((math.dist(q, o) for o in rest), default=math.inf)

            # the nearest of: the two hydrogens to each other, either to any other atom; the least turn on a
            # tie; and never their two bonds across one another
            ph, pk = dep.pos(dep.neighbours(h)[0]), dep.pos(dep.neighbours(k)[0])
            ways = [(round(min(math.dist(a, b), clear(a), clear(b)), 6),
                     -(math.dist(a, dep.pos(h)) + math.dist(b, dep.pos(k))), a, b)
                    for a in choices(h) for b in choices(k) if not segments_intersect(ph, a, pk, b)]
            if ways:
                _, _, a, b = max(ways, key=lambda c: (c[0], c[1]))
                (dep.atoms[h].x, dep.atoms[h].y), (dep.atoms[k].x, dep.atoms[k].y) = a, b


def _load_rdkit_drawing() -> Any:
    from rdkit.Chem.Draw import rdMolDraw2D

    return rdMolDraw2D


def _rdkit_drawing() -> Any:
    """RDKit's drawing module, loaded when a ligand is drawn and not with the package. RDKit's wheel links that
    one module against the system libraries libXrender and libexpat, which a minimal container image does not
    have: loaded with the package, it kept ``import bindview`` itself from working there."""
    try:
        return _load_rdkit_drawing()
    except ImportError as exc:
        raise BindviewError(
            f"RDKit's drawing module, which sets the wedges and the double bonds of a ligand, cannot be loaded: "
            f"{exc}. On Linux it needs the system libraries libXrender and libexpat, which a minimal container "
            f"image leaves out (Debian and Ubuntu: apt-get install libxrender1 libexpat1)") from exc


def _depict(lig: Ligand, metrics: Metrics, method: str,
            coords: dict[str, tuple[float, float]] | None, normalise: bool = True,
            exact: "dict[str, tuple[float, float]] | None" = None,
            coordination: "Sequence[tuple[str, str]] | None" = None) -> Depiction:
    notes: list[str] = []
    mol = Chem.Mol(lig.mol)
    n = mol.GetNumAtoms()

    if coords is not None:
        missing = [nm for nm in lig.names if nm not in coords]
        if missing:
            raise ValueError(f"supplied 2D coordinates lack atoms: {', '.join(missing[:6])}")
        xy = np.array([coords[nm] for nm in lig.names], dtype=float).reshape(-1, 2)
        if normalise:
            xy = _normalise(mol, xy)
        source = "supplied"
        drawn_coord = [(m, d) for m, d in lig.coordination]
        if coordination is not None:
            index = {name: i for i, name in enumerate(lig.names)}
            drawn_coord = [(index[a], index[b]) for a, b in coordination if a in index and b in index]
    else:
        xy, source = _choose_coords(mol, method, notes)
        xy = _orient_landscape(xy)
        xy, drawn_coord = _place_fragments(mol, xy, lig.coordination, lig.names, notes, lig.bridging)

    # Hand RDKit the final (y-up) coordinates and let it kekulize, add the
    # hydrogens that stereo at ring fusions needs, and choose the wedges.
    mol.RemoveAllConformers()
    conf = Chem.Conformer(n)
    for i in range(n):
        conf.SetAtomPosition(i, Point3D(float(xy[i, 0]), float(xy[i, 1]), 0.0))
    conf.Set3D(False)
    mol.AddConformer(conf, assignId=True)
    kekulized = True
    draw = Chem.Mol(mol)
    has_stereo = lig.bond_source in ("template", "geometry")
    drawing = _rdkit_drawing()
    try:
        draw = drawing.PrepareMolForDrawing(mol, kekulize=True, addChiralHs=has_stereo,
                                            wedgeBonds=has_stereo, forceCoords=False)
    except Exception:
        kekulized = False
        try:
            draw = drawing.PrepareMolForDrawing(mol, kekulize=False, addChiralHs=has_stereo,
                                                wedgeBonds=has_stereo, forceCoords=False)
        except Exception:
            draw = Chem.Mol(mol)
            notes.append("stereo wedges could not be assigned")
        notes.append("the ligand could not be kekulized; aromatic bonds are drawn dashed")

    bl = metrics.bond_length
    dconf = draw.GetConformer()
    atoms: list[DepAtom] = []
    for a in draw.GetAtoms():
        i = a.GetIdx()
        p = dconf.GetAtomPosition(i)
        atoms.append(DepAtom(
            idx=i, x=float(p.x) * bl, y=-float(p.y) * bl, element=a.GetSymbol(),
            name=lig.names[i] if i < n else None,
            hcount=a.GetTotalNumHs(), charge=a.GetFormalCharge(), isotope=a.GetIsotope(),
            aromatic=a.GetIsAromatic(),
        ))

    bonds: list[DepBond] = []
    for b in draw.GetBonds():
        order = int(round(b.GetBondTypeAsDouble())) if not b.GetIsAromatic() or kekulized else 1
        order = max(1, min(3, order))
        stereo = "none"
        d = b.GetBondDir()
        if d == Chem.BondDir.BEGINWEDGE:
            stereo = "wedge"
        elif d == Chem.BondDir.BEGINDASH:
            stereo = "hash"
        bonds.append(DepBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), order, stereo,
                             aromatic=(b.GetIsAromatic() and not kekulized)))
    for metal, donor in drawn_coord:
        bonds.append(DepBond(metal, donor, 1, kind="coordination"))

    dep = Depiction(atoms=atoms, bonds=bonds, rings=[tuple(r) for r in draw.GetRingInfo().AtomRings()],
                    n_heavy=n, metrics=metrics, coord_source=source, notes=notes)
    placed: set[int] = set()
    if exact:
        for atom, key in zip(atoms, atom_keys(dep)):
            if key in exact:
                atom.x, atom.y = float(exact[key][0]), float(exact[key][1])
                placed.add(atom.idx)
    _part_added_hydrogens(dep, placed)

    for atom in atoms:
        degree = len(dep.neighbours(atom.idx))
        if _needs_label(atom.element, degree, atom.charge, atom.isotope):
            nbrs = [dep.pos(j) for j in dep.neighbours(atom.idx, covalent_only=False)]
            atom.label = make_label(atom.pos, atom.element, atom.hcount, atom.charge, atom.isotope, nbrs,
                                    metrics.atom_font, metrics.sub_font, metrics.charge_font)
    return dep
