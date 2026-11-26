#!/usr/bin/env python
"""A ligand and the residues it is in contact with, as sticks in three dimensions, in the pose of its diagram.

A SKETCH, not part of bindview: nothing here is tested or promised.

    python prototype/ppi_openbook/interact3d.py out.html      # a page of the figures below, where it is told to

``view(diagram)`` is the picture a structure viewer is opened for: the ligand as licorice, every residue
that makes a contact of the diagram as licorice, and the contacts as lines from atom to atom (to the
middle of a ring where the contact is made by a ring) with their distances. It is drawn as vectors, by
this file, and seen from where the library's flat diagram is drawn from: the ligand's own plane faces the
reader, turned so that its atoms lie as near as they can to where the diagram has them. A residue is
therefore at about the same place in both. What is nearer to the reader is drawn over what is farther,
and what is farther is thinner and paler.

``figure(diagram)`` is the library's diagram with that view under it, the two tied by a style sheet:
a residue pointed at in one is marked in the other.

``several(diagrams)`` is for the ligands of one pocket: the models superposed as the library's comparison
superposes them, the first ligand in full and the others thinner, each in a colour of its own, the
residues of the first model, and every contact of the first weighted by the share of the ligands that
make it.

The contacts are the diagram's own (``Diagram.interactions``, resolved to atoms by the library). A water
bridge runs through its water, which is drawn, where the table names the water and the structure has it.
A metal that the protein holds between itself and the ligand is not drawn: :func:`book.load_ligand`
leaves such a contact out. The figure does not turn; what it
hands back (``facts["camera"]``, ``facts["contacts"]``) is what a structure viewer needs to show the same
thing from the same place.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import book  # noqa: E402
import openbook  # noqa: E402
import pocket  # noqa: E402
import surface  # noqa: E402
from bindview import Diagram  # noqa: E402
from bindview.align import pocket_transforms  # noqa: E402
from bindview.confidence import BAND_LABELS, PLAIN, Look, band_of, look_of, occupancy_factors  # noqa: E402
from bindview.theme import LIGHT  # noqa: E402

__all__ = ["view", "figure", "several"]

WIDTH = 760.0            # the widest a view is drawn, in units of the figure
PX_PER_A = 30.0          # and the largest scale: units for an ångström
MARGIN = 2.7             # ångström of room around the atoms, for the names
LIGAND_W, RESIDUE_W, PALE_W, OTHER_W = 6.4, 4.4, 3.4, 3.6   # the width of a stick: ligand, residue, a residue with
#                                                            # hydrophobic contacts only, another ligand of a set
EDGE = 1.0               # the dark line around a stick
BALL = 1.22              # an atom that is not carbon is a little wider than its sticks
#: ... and a heavier one more so: it is then told from the carbon beside it by its size too (phosphorus
#: is orange, as the ligand's carbon is). Anything that is no non-metal of an organic molecule is a metal here.
BALLS = {"P": 1.75, "S": 1.55, "SE": 1.6, "CL": 1.5, "BR": 1.6, "I": 1.75}
ORGANIC = {"C", "N", "O", "S", "P", "SE", "F", "CL", "BR", "I", "B", "SI", "H"}
METAL_BALL = 2.6


def ball_of(symbol: str) -> float:
    """How much wider than its sticks an atom is drawn: 1 for carbon."""
    e = symbol.upper()
    if e == "C":
        return 1.0
    return BALLS.get(e, BALL) if e in ORGANIC else METAL_BALL
FOG = 0.34               # how far the farthest stick is mixed towards white
THIN = 0.78              # and how much thinner it is than the nearest
SLATE = (0.55, 0.60, 0.68)                       # the carbons of a residue
DARK = np.array([0.12, 0.16, 0.20])
LINE_W = 1.7
#: The kinds of contact that are a line here: those of the diagram that have a direction, and the
#: hydrophobic ones, fine and dotted, from carbon to carbon.
DOTTED = "hydrophobic"
#: The leans that are tried where the view is left to lean by itself: back or forward, and to a side, in degrees.
#: Never both by 35°: the ligand's plane then stands more than 45° off the page and is seen half from its edge.
LEANS = tuple((back, side) for back in (0.0, 20.0, -20.0, 35.0, -35.0) for side in (0.0, 20.0, -20.0, 35.0, -35.0)
              if abs(back) + abs(side) < 70.0)
LEAN_COSTS = 0.06        # a degree of lean is worth this much of what it uncovers: of two views, the flatter
WORTH_IT = 0.7           # and the view leans only if that leaves less than this share hidden of what was
SAME_FIT = 1.15          # two faces of a ligand fit its diagram alike when one fit is within this of the other


def _headline(x: float, y: float, letter: str, words: str) -> str:
    """The line over a view, with its letter where a figure has several."""
    mark = (f'<tspan font-weight="700" fill="{openbook.INK}">{letter}</tspan><tspan dx="7">{words}</tspan>'
            if letter else words)
    return openbook.text(x, y, mark, 11.5, openbook.QUIET)


def _lines(words: str, room: float, per: float = 5.0) -> "list[str]":
    """A sentence broken into lines that fit ``room`` units, at about ``per`` units a letter."""
    out, line = [], ""
    for word in words.split():
        if line and (len(line) + 1 + len(word)) * per > room:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return out + ([line] if line else [])


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def _mix(colour, fog: float) -> np.ndarray:
    return np.asarray(colour, dtype=float) * (1.0 - fog) + fog


_OTHER = {k.upper(): _rgb(v) for k, v in LIGHT.elements.items() if k != "*"}


def element_colour(symbol: str, carbon) -> np.ndarray:
    """Carbon in the colour of what it belongs to; nitrogen, oxygen, sulfur and phosphorus as the other
    sketches draw licorice; any other element in the colour the library's diagram gives it."""
    e = symbol.upper()
    if e == "C":
        return np.asarray(carbon, dtype=float)
    if e in surface.ELEMENT:
        return np.asarray(surface.ELEMENT[e], dtype=float)
    return _OTHER.get(e, _rgb(LIGHT.elements["*"]))


# ---------------------------------------------------------------------------------------- where it is seen from
@dataclass
class Pose:
    axes: np.ndarray                 # rows: across the page, up the page, towards the reader
    centre: np.ndarray
    by: str                          # "diagram": fitted to the flat diagram; "plane": the ligand's plane, no more
    fit: "float | None" = None       # ångström: how far the ligand's atoms lie from where the diagram has them
    scale: "float | None" = None     # units of the diagram for an ångström, by that fit
    lean: "tuple[float, float]" = (0.0, 0.0)     # degrees: back (what is in front comes down the page), and to a side

    def leaned(self, back: float, side: float) -> "Pose":
        """This pose with the ligand's plane leaned back by ``back`` degrees (what lies in front of the
        ligand comes down the page, what lies behind goes up) and then turned by ``side`` degrees about
        the upright of the page (what lies in front goes to the left)."""
        b, s = math.radians(back), math.radians(side)
        x, y, z = self.axes
        y, z = math.cos(b) * y - math.sin(b) * z, math.sin(b) * y + math.cos(b) * z
        x, z = math.cos(s) * x - math.sin(s) * z, math.sin(s) * x + math.cos(s) * z
        return Pose(np.stack([x, y, z]), self.centre, self.by, self.fit, self.scale, (back, side))


def pose_of(cx: book.Complex, flat: "dict[str, tuple[float, float]] | None", where=()) -> Pose:
    """The frame a ligand is seen in: its own plane faces the reader, and it is turned in that plane, and
    seen from that one of its two faces, so that its atoms lie nearest to where ``flat`` has them (the
    atoms of the library's diagram by name, in the units of the figure, y down). Without ``flat`` the
    longest axis of the ligand runs across the page.

    A long ligand that the diagram draws nearly straight fits about as well from either face. Where the
    two fits are within ``SAME_FIT`` of each other, ``where`` decides: the atoms of each residue with the
    place the diagram gives that residue, and the face is taken from which the residues lie nearer to
    those places."""
    ligand = cx.residues[-1]
    p = cx.xyz[ligand.atoms]
    centre = p.mean(axis=0)
    if len(p) >= 3:
        _, vec = np.linalg.eigh(np.cov((p - centre).T))
        normal, long_ = vec[:, 0], vec[:, 2]
    else:                                                   # an ion, two atoms: the plane the book has for it
        normal, long_ = cx.axes[0], cx.axes[1]
    best = None
    have = [(k, flat[cx.atoms[a]]) for k, a in enumerate(ligand.atoms) if flat and cx.atoms[a] in flat]
    if len(have) >= 3:
        q = np.array([[x, -y] for _, (x, y) in have], dtype=float)
        middle = q.mean(axis=0)
        q -= middle
        mine = p[[k for k, _ in have]]
        fits = []
        for face in (1.0, -1.0):
            z = face * normal
            u = long_
            v = np.cross(z, u)
            pr = (mine - centre) @ np.stack([u, v]).T
            pr -= pr.mean(axis=0)
            a = float((pr * q).sum())
            b = float((pr[:, 0] * q[:, 1] - pr[:, 1] * q[:, 0]).sum())
            c, s = math.cos(math.atan2(b, a)), math.sin(math.atan2(b, a))
            turned = pr @ np.array([[c, s], [-s, c]])
            k = float((turned * q).sum() / max(1e-9, (turned ** 2).sum()))      # units for an ångström
            rmsd = float(np.sqrt(((turned - q / k) ** 2).sum(axis=1).mean())) if k > 0 else float("inf")
            axes = np.stack([c * u - s * v, s * u + c * v, z])
            off = None
            if where and k > 0:
                off = float(np.mean([np.linalg.norm((cx.xyz[atoms].mean(axis=0) - mine.mean(axis=0)) @ axes[:2].T
                                                    - (np.array([xy[0], -xy[1]]) - middle) / k) for atoms, xy in where]))
            fits.append((rmsd, k, axes, off))
        fits.sort(key=lambda f: f[0])
        best = fits[0]
        if (fits[1][0] <= SAME_FIT * fits[0][0] and fits[0][3] is not None and fits[1][3] is not None
                and fits[1][3] < fits[0][3]):
            best = fits[1]
    if best is not None:
        return Pose(best[2], centre, "diagram", best[0], best[1])
    return Pose(np.stack([long_, np.cross(normal, long_), normal]), centre, "plane")


def hidden(at: np.ndarray, ligand, residues, lines) -> float:
    """How much of a view hides something else: the atoms of a residue that lie on the ligand, the atoms
    of two residues that lie on each other, and the contacts that are seen from their end (a line of
    less than 1.5 Å on the page, counted three times)."""
    flat = at[:, :2]
    score = 0.0
    lig = flat[ligand]
    for n, atoms in enumerate(residues):
        mine = flat[atoms]
        score += float((np.linalg.norm(mine[:, None, :] - lig[None, :, :], axis=2).min(axis=1) < 1.1).sum())
        for other in residues[n + 1:]:
            score += float((np.linalg.norm(mine[:, None, :] - flat[other][None, :, :], axis=2).min(axis=1) < 0.9).sum())
    for a, b in lines:
        if float(np.linalg.norm(flat[list(a)].mean(axis=0) - flat[list(b)].mean(axis=0))) < 1.5:
            score += 3.0
    return score


def best_lean(cx: book.Complex, pose: Pose, ligand, residues, lines) -> "tuple[Pose, float, float]":
    """The pose leaned to where the least is hidden, of the leans of ``LEANS``, a degree of lean counted
    as ``LEAN_COSTS`` of something hidden. The plane stays flat on the page, which is the pose of the
    diagram exactly, unless a lean leaves less than ``WORTH_IT`` of what is hidden then. Comes back
    with what is hidden in the view that is chosen, and what is hidden without a lean."""
    best, flat_score = None, 0.0
    for back, side in LEANS:
        leaned = pose.leaned(back, side)
        score = hidden((cx.xyz - pose.centre) @ leaned.axes.T, ligand, residues, lines)
        if not back and not side:
            flat_score = score
        rank = (score + LEAN_COSTS * (abs(back) + abs(side)), abs(back) + abs(side))
        if best is None or rank < best[0]:
            best = (rank, leaned, score)
    if best[2] > WORTH_IT * flat_score:
        return pose, flat_score, flat_score
    return best[1], best[2], flat_score


# ---------------------------------------------------------------------------------------- sticks
class Depth:
    """What is far from the reader is thinner and paler: by how much, for a height above the page."""

    def __init__(self, heights) -> None:
        self.far, self.near = float(np.min(heights)), float(np.max(heights))

    def at(self, z: float) -> float:
        return float(np.clip((z - self.far) / max(1e-6, self.near - self.far), 0.0, 1.0))

    def thin(self, z: float) -> float:
        return THIN + (1.0 - THIN) * self.at(z)

    def fog(self, z: float) -> float:
        return FOG * (1.0 - self.at(z))


def bonds_of(cx: book.Complex, atoms) -> "list[tuple[int, int]]":
    """Atoms that are close enough to be bonded, as the other sketches take them (no hydrogens here)."""
    out = []
    for n, a in enumerate(atoms):
        for b in atoms[n + 1:]:
            far = 2.25 if {cx.elements[a].upper(), cx.elements[b].upper()} & {"S", "P", "SE"} else 1.95
            if 0.4 < float(np.linalg.norm(cx.xyz[a] - cx.xyz[b])) < far:
                out.append((int(a), int(b)))
    return out


def licorice(at: np.ndarray, px, atoms, bonds, colours, wide, width: float, depth: Depth) -> str:
    """Atoms and their bonds as sticks, each half of a bond in the colour of its atom.

    Drawn atom by atom from the farthest to the nearest: an atom's dark outline first, then its colour,
    so that a stick that is nearer to the reader cuts across one that is farther. ``at`` is every atom
    in the frame of the view, ``colours`` the colour of each atom of ``atoms`` and ``wide`` says of an
    atom how much wider than its sticks it is drawn (:func:`ball_of`; an atom it does not name, not at all).
    """
    ends: "dict[int, list[tuple[np.ndarray, float]]]" = {int(a): [] for a in atoms}
    for a, b in bonds:
        if a not in ends or b not in ends:
            continue
        w = width * depth.thin(float(at[a][2] + at[b][2]) / 2.0)
        m = px((at[a] + at[b]) / 2.0)
        ends[a].append((m, w))
        ends[b].append((m, w))
    out = []
    for a in sorted(ends, key=lambda a: at[a][2]):
        p, fog = px(at[a]), depth.fog(float(at[a][2]))
        fill, dark = openbook.as_hex(_mix(colours[a], fog)), openbook.as_hex(_mix(DARK, fog))
        r = max([w for _, w in ends[a]] or [width * depth.thin(float(at[a][2]))]) / 2.0 * wide.get(a, 1.0)
        under = [f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="{r + EDGE:.2f}" fill="{dark}"/>']
        over = []
        for m, w in ends[a]:
            v = m - p
            n = float(np.hypot(*v))
            q = m + (0.4 * v / n if n > 1e-6 else 0.0)        # a hair past the middle: no seam between the halves
            xy = f'x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]:.1f}" y2="{q[1]:.1f}"'
            under.append(f'<line {xy} stroke="{dark}" stroke-width="{w + 2 * EDGE:.2f}"/>')
            over.append(f'<line {xy} stroke="{fill}" stroke-width="{w:.2f}"/>')
        over.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="{r:.2f}" fill="{fill}"/>')
        out += under + over
    return "".join(out)


def atoms_drawn(cx: book.Complex, i: int) -> "list[int]":
    """What is drawn of a residue: its side chain from the alpha carbon, and its backbone where the
    backbone makes one of its contacts (all of a glycine)."""
    r = cx.residues[i]
    used = {a for c in cx.contacts if c.a == i and c.kind in pocket.KINDS for a in c.atoms_a}
    backbone = {int(a) for a in r.atoms if cx.atoms[a] in ("N", "C", "O", "OXT")}
    atoms = [int(a) for a in r.atoms if int(a) not in backbone]
    if len(atoms) < 2 or used & backbone:
        atoms = [int(a) for a in r.atoms]
    return atoms


def ligand_bonds(diagram: Diagram, ligand: book.Residue) -> "list[tuple[int, int]]":
    """The bonds of the ligand as the library has them (from its SMILES), as indices of the complex."""
    return [(int(ligand.atoms[b.GetBeginAtomIdx()]), int(ligand.atoms[b.GetEndAtomIdx()]))
            for b in diagram.ligand.mol.GetBonds()]


WATERS = ("HOH", "WAT", "DOD", "H2O")
WATER = (0.56, 0.79, 0.91)                       # a water that a contact runs through


def bridging_waters(diagram: Diagram, cx: book.Complex) -> dict:
    """The water of every water bridge whose table names it and whose structure holds it: by the residue
    of the bridge and the ligand's atoms (as a contact of ``cx`` has them), the water's name
    (``chain:number``) and where its oxygen is, in the coordinates of the model."""
    st = diagram.structure
    if st is None:
        return {}
    where = {(a.chain, a.resnum): np.array(a.xyz, dtype=float) for a in st.atoms
             if a.resname in WATERS and a.element.upper() == "O"}
    first = int(cx.residues[-1].atoms[0])
    out = {}
    for r in diagram.interactions:
        name = getattr(r.record, "water", None)
        if r.type.value != "water_bridge" or not name or r.residue is None or r.ligand is None:
            continue
        chain, _, number = str(name).partition(":")
        if not number.lstrip("-").isdigit() or (chain, int(number)) not in where:
            continue
        ia = cx.find(r.residue.chain, r.residue.number, r.residue.icode)
        if ia is not None:
            out[(ia, tuple(first + i for i in r.ligand.atoms))] = (str(name), where[(chain, int(number))])
    return out


def atom_plddt(diagram: Diagram) -> "dict[str, float]":
    """The pLDDT of every atom of the ligand by its name, for a model that states one."""
    st = diagram.structure
    if st is None or st.plddt_scale is None:
        return {}
    return {x.name: float(x.b_iso) * st.plddt_scale for x in st.residue_atoms(diagram.ligand_key) if x.b_iso is not None}


# ---------------------------------------------------------------------------------------- the view
def residues_flat(cx: book.Complex, flat, drawn) -> list:
    """Where the library's diagram has each residue that is drawn here: the atoms of the residue, and its
    place in the units of the figure (y down), read off the diagram's graph. Empty where there is none."""
    graph = getattr(flat, "graph", None)
    if graph is None or not flat.atoms:
        return []
    nodes = list(graph.nodes.values()) if isinstance(graph.nodes, dict) else list(graph.nodes)
    shift = None                                            # the graph has its own origin: found from two atoms
    for n in nodes:
        if n.kind == "ligand_atom" and n.label in flat.atoms:
            d = np.array(flat.atoms[n.label]) - np.array([n.x, n.y])
            if shift is None:
                shift = d
            elif float(np.abs(d - shift).max()) > 0.5:
                return []                                   # not a shift alone: leave the residues out of it
    if shift is None:
        return []
    at = {(n.attrs.get("chain"), n.attrs.get("number")): np.array([n.x, n.y]) + shift for n in nodes if n.kind == "residue"}
    out = []
    for i, atoms in drawn.items():
        r = cx.residues[i]
        if (r.chain, r.number) in at:
            out.append((atoms, tuple(at[(r.chain, r.number)])))
    return out


def clear_spot(candidates, half_w: float, half_h: float, ink: np.ndarray, taken, box) -> np.ndarray:
    """The first of these places, each the middle of a label, where the label lies on the fewest sticks
    and on no other label: ``ink`` is points along everything that is drawn, ``taken`` the labels that
    are placed (middle, half width, half height), ``box`` the page. Earlier places are preferred."""
    best = None
    for n, c in enumerate(candidates):
        on_ink = int(((np.abs(ink[:, 0] - c[0]) < half_w + 2.0) & (np.abs(ink[:, 1] - c[1]) < half_h + 2.0)).sum()) if len(ink) else 0
        on_label = sum(1 for q, w, h in taken if abs(q[0] - c[0]) < w + half_w + 1.5 and abs(q[1] - c[1]) < h + half_h + 1.5)
        off_page = not (box[0] + half_w <= c[0] <= box[2] - half_w and box[1] + half_h <= c[1] <= box[3] - half_h)
        cost = 3.0 * on_ink + 40.0 * on_label + 500.0 * off_page + 1.0 * n
        if best is None or cost < best[0]:
            best = (cost, c)
    return np.asarray(best[1], dtype=float)


def _halo(x: float, y: float, words: str, size: float, fill: str, weight: str = "600", anchor: str = "middle") -> str:
    return openbook.text(x, y, words, size, fill, anchor, weight,
                         ' stroke="#ffffff" stroke-width="3.2" stroke-opacity="0.92" paint-order="stroke" '
                         'stroke-linejoin="round"')


def _contact_line(pa, pb, colour: str, dash: str, width: float, opacity: float, ring_a: bool, ring_b: bool,
                  fine: bool = False) -> str:
    """A contact from the protein's end to the ligand's: white under it, its colour and dashes, and a
    small ring where an end is the middle of a ring. Its distance is drawn by the caller, over all the lines."""
    xy = f'x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}"'
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out = [f'<line {xy} stroke="#ffffff" stroke-opacity="0.85" stroke-width="{width + 2.4:.2f}" stroke-linecap="round"/>',
           f'<line {xy} stroke="{colour}" stroke-width="{width:.2f}"{d} stroke-linecap="{"round" if fine else "butt"}"/>']
    for p, ring in ((pa, ring_a), (pb, ring_b)):
        if ring:
            out.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3.6" fill="#ffffff" stroke="{colour}" '
                       f'stroke-width="1.4"/>')
        elif not fine:
            out.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="2.1" fill="{colour}" stroke="#ffffff" '
                       f'stroke-width="0.8"/>')
    return f'<g opacity="{opacity:.2f}">{"".join(out)}</g>' if opacity < 0.999 else "".join(out)


def _lower(a: Look, b: Look) -> Look:
    return a if a.level >= b.level else b


@dataclass
class _Drawn:
    """What :func:`_draw` hands on: the picture, the complex it is of, and where its residues were put."""

    sketch: openbook.Sketch
    cx: book.Complex


def _draw(diagram: Diagram, style: str = "detailed", plddt_colors: bool = False, labels: bool = True,
          distances: bool = True, backdrop: bool = False, lean="auto", kinds=None, flat=None,
          others: "list[tuple[book.Complex, Diagram]] | None" = None,
          shares: "dict[tuple[str, int, str, str], int] | None" = None, letter: str = "",
          footnote: str = "") -> _Drawn:
    cx = book.load_ligand(diagram, ("the protein", diagram.name or "the ligand"))
    if flat is None:
        flat = diagram.render(style=style)
    ligand = cx.residues[-1]
    lig_atoms = [int(a) for a in ligand.atoms]
    if kinds is None:
        kinds = pocket.KINDS
    else:
        unknown = [k for k in kinds if k not in pocket.KINDS]
        if unknown or isinstance(kinds, str):
            raise ValueError(f"kinds={kinds!r}: a list of {', '.join(repr(k) for k in pocket.KINDS)}")
        kinds = {k: v for k, v in pocket.KINDS.items() if k in kinds}
    shown = sorted({c.a for c in cx.contacts if c.kind in kinds})
    lined = {c.a for c in cx.contacts if c.kind in pocket.LINED and c.kind in kinds}
    drawn = {i: atoms_drawn(cx, i) for i in shown}
    pose = pose_of(cx, dict(flat.atoms) if flat is not None and flat.atoms else None,
                   residues_flat(cx, flat, drawn) if flat is not None else ())
    ends = [(c.atoms_a, c.atoms_b) for c in cx.contacts
            if c.kind in pocket.LINED and c.kind in kinds and c.atoms_a and c.atoms_b]
    if isinstance(lean, str):
        if lean != "auto":
            raise ValueError(f"lean={lean!r} is not 'auto' or two angles in degrees (back, to the side)")
        pose, hides, hides_flat = best_lean(cx, pose, lig_atoms, [drawn[i] for i in shown], ends)
    else:
        back, side = (float(lean), 0.0) if isinstance(lean, (int, float)) else (float(lean[0]), float(lean[1]))
        hides_flat = hidden((cx.xyz - pose.centre) @ pose.axes.T, lig_atoms, [drawn[i] for i in shown], ends)
        pose = pose.leaned(back, side)
        hides = hidden((cx.xyz - pose.centre) @ pose.axes.T, lig_atoms, [drawn[i] for i in shown], ends)
    at = (cx.xyz - pose.centre) @ pose.axes.T
    # The waters that a water bridge runs through, where the table names them and the structure has them.
    waters = {key: (name, (xyz - pose.centre) @ pose.axes.T, xyz) for key, (name, xyz) in bridging_waters(diagram, cx).items()
              if "water_bridge" in kinds
              and any(c.kind == "water_bridge" and (c.a, tuple(c.atoms_b)) == key for c in cx.contacts)}

    # Other ligands of the same pocket, already in the frame of this model.
    more = []
    for k, (other, dia) in enumerate(others or (), start=1):
        o_lig = other.residues[-1]
        more.append(((other.xyz - pose.centre) @ pose.axes.T, [int(a) for a in o_lig.atoms], ligand_bonds(dia, o_lig),
                     other, pocket.CARBONS[k % len(pocket.CARBONS)]))

    # The page: every atom that is drawn, with room around it for the names.
    every = np.vstack([at[lig_atoms]] + [at[a] for a in drawn.values()] + [o[0][o[1]] for o in more]
                      + [w[1][None, :] for w in waters.values()])
    x0, x1 = float(every[:, 0].min()) - MARGIN, float(every[:, 0].max()) + MARGIN
    y0, y1 = float(every[:, 1].min()) - MARGIN, float(every[:, 1].max()) + MARGIN
    scale = min(PX_PER_A, WIDTH / (x1 - x0))
    head = 26.0
    w, h = (x1 - x0) * scale, (y1 - y0) * scale
    depth = Depth(every[:, 2])

    def px(q) -> np.ndarray:
        return np.array([(q[0] - x0) * scale, head + (y1 - q[1]) * scale])

    # How sure the model is: of a residue, of an atom of the ligand.
    of_atom = atom_plddt(diagram)
    stated = bool(cx.plddt) or bool(of_atom)

    def look_at(i: int) -> Look:
        return look_of(cx.plddt.get(i), by_colour=plddt_colors) if i in cx.plddt else PLAIN

    def look_ligand(atoms) -> Look:
        values = [of_atom[cx.atoms[a]] for a in atoms if cx.atoms[a] in of_atom]
        return look_of(min(values), by_colour=plddt_colors) if values else PLAIN

    parts = [_headline(0, 14, letter, _heading(cx, pose, bool(others)))]
    if backdrop:
        parts.append(_backdrop(cx, at, (x0, x1, y0, y1), scale, head))

    # Sticks: the residues behind the ligand, the ligand (and the other ligands of a set), the residues in front.
    def residue_sticks(i: int) -> str:
        r, look = cx.residues[i], look_at(i)
        atoms = drawn[i]
        if plddt_colors and look.band:                        # by colour: the whole residue in the colour of its band
            colours = {a: _rgb(openbook.BANDS[look.band]) for a in atoms}
        else:
            colours = {a: element_colour(cx.elements[a], np.array(SLATE)) for a in atoms}
        if i not in lined:                                     # hydrophobic contacts only: thinner, and paler
            colours = {a: _mix(c, 0.22) for a, c in colours.items()}
        width = (RESIDUE_W if i in lined else PALE_W) * max(0.6, look.width)
        made = sorted({kinds[c.kind][0] for c in cx.contacts if c.a == i and c.kind in kinds})
        sure = f"; pLDDT {cx.plddt[i]:.0f}" if i in cx.plddt else ""
        body = licorice(at, px, atoms, bonds_of(cx, atoms), colours,
                        {a: ball_of(cx.elements[a]) for a in atoms}, width, depth)
        fade = f' opacity="{look.opacity:.2f}"' if look.opacity < 0.999 else ""
        return (f'<g class="res" data-id="{r.key}"{fade}>{body}<title>{r.label} (chain {r.chain}): '
                f'{", ".join(made)}{sure}</title></g>')

    def height_of(i: int) -> float:
        """How far in front of the ligand a residue lies: against the atoms of the ligand it is in contact with."""
        theirs = [a for c in cx.contacts if c.a == i and c.kind in kinds for a in c.atoms_b] or lig_atoms
        return float(at[drawn[i]][:, 2].mean() - at[theirs][:, 2].mean())

    order = sorted(shown, key=lambda i: float(at[drawn[i]][:, 2].mean()))
    parts += [residue_sticks(i) for i in order if height_of(i) < 0.0]
    for o_at, o_atoms, o_bonds, other, carbon in more:
        colours = {a: _mix(element_colour(other.elements[a], carbon), 0.10) for a in o_atoms}
        parts.append('<g class="lig other" opacity="0.9">'
                     + licorice(o_at, px, o_atoms, o_bonds, colours,
                                {a: ball_of(other.elements[a]) for a in o_atoms}, OTHER_W, depth)
                     + f"<title>{html.escape(other.names[1])}</title></g>")
    lig_colours = {}
    for a in lig_atoms:
        if plddt_colors and cx.atoms[a] in of_atom:
            lig_colours[a] = _rgb(openbook.BANDS[band_of(of_atom[cx.atoms[a]])])
        else:
            lig_colours[a] = element_colour(cx.elements[a], np.array(openbook.CARBON))
    parts.append('<g class="lig">' + licorice(at, px, lig_atoms, ligand_bonds(diagram, ligand), lig_colours,
                                               {a: ball_of(cx.elements[a]) for a in lig_atoms}, LIGAND_W, depth)
                 + f"<title>{html.escape(cx.names[1])}</title></g>")
    parts += [residue_sticks(i) for i in order if height_of(i) >= 0.0]

    # Where there is ink: points along every stick, for the names and the distances to keep off.
    ink = [px(at[a]) for a in lig_atoms] + [px(at[a]) for atoms in drawn.values() for a in atoms]
    for a, b in ligand_bonds(diagram, ligand) + [bond for atoms in drawn.values() for bond in bonds_of(cx, atoms)]:
        ink += [px(at[a] + f * (at[b] - at[a])) for f in (0.25, 0.5, 0.75)]
    for o_at, o_atoms, o_bonds, _, _ in more:
        ink += [px(o_at[a]) for a in o_atoms] + [px((o_at[a] + o_at[b]) / 2.0) for a, b in o_bonds]
    ink = np.array(ink).reshape(-1, 2)
    page = (2.0, head + 2.0, w - 2.0, head + h - 2.0)
    taken: list = []

    # Names: beyond the end of a residue that is farthest from the ligand, turned aside where that is taken.
    names = []
    several_chains = len({cx.residues[i].chain for i in shown}) > 1      # then a name says its chain, as the diagram's does
    if labels and shown:
        size = 10.5
        middle = px(at[lig_atoms].mean(axis=0))
        for i in sorted(shown, key=lambda i: i not in lined):          # those with a line first: they choose first
            pts = np.array([px(at[a]) for a in drawn[i]])
            end = pts[int(np.argmax(np.linalg.norm(pts - middle, axis=1)))]
            d = end - middle
            d = d / max(1e-6, float(np.hypot(*d)))
            tag = cx.residues[i].label + (f"({cx.residues[i].chain})" if several_chains else "")
            hw, hh = len(tag) * size * 0.30 + 3.0, size * 0.5 + 2.0
            spots = []
            for reach in (1.0, 1.9, 3.0):
                for turn in (0.0, 30.0, -30.0, 60.0, -60.0, 95.0, -95.0):
                    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
                    e = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
                    spots.append(end + e * reach * (7.0 + abs(e[0]) * hw + abs(e[1]) * hh))
            q = clear_spot(spots, hw, hh, ink, taken, page)
            taken.append((q, hw, hh))
            r = cx.residues[i]
            lead = ""
            if float(np.hypot(*(q - spots[0]))) > 7.0:
                lead = (f'<line x1="{q[0]:.1f}" y1="{q[1]:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" '
                        f'stroke="{openbook.QUIET}" stroke-width="0.7"/>')
            ink_ = openbook.INK if i in lined else openbook.QUIET
            names.append(f'<g class="res" data-id="{r.key}">{lead}'
                         + _halo(q[0], q[1] + size * 0.36, tag, size, ink_, "700" if i in lined else "600")
                         + f"<title>{r.label} (chain {r.chain})</title></g>")

    # Contacts, over the sticks: from atom to atom, or to the middle of a ring.
    n_ligands = 1 + len(more)
    marks = rings = False
    numbers: "list[str]" = []
    balls: "dict[str, tuple[np.ndarray, float]]" = {}         # a water that is drawn: where, and how far from the reader
    legs: set = set()                                          # from a water to the ligand: once, whoever shares the water

    def line(r, c, pa, pb, width: float, opacity: float, label: str, said: str, ring_a: bool = False,
             ring_b: bool = False) -> None:
        """A contact, or one of the two halves of a water bridge, with its length where it has one."""
        fine = c.kind == DOTTED
        label_at = None
        if label:                                              # on its line, where no other label is; beside a short one
            hw, hh = len(label) * 8.8 * 0.29 + 2.0, 6.0
            v = pb - pa
            n = float(np.hypot(*v))
            side = np.array([-v[1], v[0]]) / max(n, 1e-6)
            spots = [pa + f * v for f in (0.5, 0.38, 0.62, 0.28, 0.72)]
            beside = [pa + f * v + k * side for f in (0.5, 0.35, 0.65) for k in (10.0, -10.0)]
            spots = beside + spots if n < 34.0 else spots + beside
            label_at = clear_spot(spots, hw, hh, np.zeros((0, 2)), taken, page)
            taken.append((label_at, hw, hh))
        parts.append(f'<g class="link" data-a="{r.key}" data-b="{ligand.key}">'
                     + _contact_line(pa, pb, pocket.colour_of(c.kind), kinds[c.kind][1], width, opacity, ring_a, ring_b,
                                     fine)
                     + f"<title>{said}</title></g>")
        if label:
            numbers.append(f'<g class="link" data-a="{r.key}" data-b="{ligand.key}"'
                           + (f' opacity="{max(opacity, 0.88):.2f}"' if opacity < 0.999 else "") + ">"
                           + _halo(label_at[0], label_at[1] + 3.0, label, 8.8, pocket.colour_of(c.kind))
                           + f"<title>{said}</title></g>")

    for c in sorted(cx.contacts, key=lambda c: c.kind in pocket.LINED):       # the fine dotted ones under the others
        if c.kind not in kinds or not c.atoms_a or not c.atoms_b:
            continue
        fine = c.kind == DOTTED
        r = cx.residues[c.a]
        pa, pb = px(at[list(c.atoms_a)].mean(axis=0)), px(at[list(c.atoms_b)].mean(axis=0))
        look = _lower(look_at(c.a), look_ligand(c.atoms_b))
        width, opacity = LINE_W * look.width, look.opacity
        mark = " ?" if look.mark else ""
        label = f"{c.distance:.1f}{mark}" if distances and not fine else ""
        marks = marks or (bool(label) and look.mark)
        said = f"{r.label}: {kinds[c.kind][0]}, {c.distance:.1f} Å"
        share = ""
        if shares is not None:
            made = shares.get((r.chain, r.number, r.icode, c.kind), 1)
            fw, fo = occupancy_factors(made / n_ligands)
            width, opacity = width * fw, opacity * fo
            share = f" · {made}/{n_ligands}"
            said += f"; made by {made} of the {n_ligands} ligands"
        if fine:
            width, opacity = 0.62 * width, 0.9 * opacity
        water = waters.get((c.a, tuple(c.atoms_b))) if c.kind == "water_bridge" else None
        if water is None:
            by_ring = c.kind in ("pi_stacking", "pi_cation")      # (a charged group has a middle too, and no ring)
            ring_a, ring_b = by_ring and len(c.atoms_a) >= 5, by_ring and len(c.atoms_b) >= 5
            rings = rings or ring_a or ring_b
            line(r, c, pa, pb, width, opacity, label + share if label else "", said, ring_a, ring_b)
            continue
        # Through its water: from the residue to the water, and (once for a water) from the water to the ligand.
        name, w_at, w_xyz = water
        pw = px(w_at)
        to_residue = float(np.linalg.norm(cx.xyz[list(c.atoms_a)].mean(axis=0) - w_xyz))
        to_ligand = float(np.linalg.norm(cx.xyz[list(c.atoms_b)].mean(axis=0) - w_xyz))
        said = (f"{r.label}: water bridge through water {name}, {to_residue:.1f} Å from the residue and "
                f"{to_ligand:.1f} Å from {cx.names[1]}" + said.partition(";")[1] + said.partition(";")[2])
        line(r, c, pa, pw, width, opacity, f"{to_residue:.1f}{mark}{share}" if distances else "", said)
        if (name, tuple(c.atoms_b)) not in legs:
            legs.add((name, tuple(c.atoms_b)))
            line(r, c, pw, pb, width, opacity, f"{to_ligand:.1f}{mark}" if distances else "", said)
        balls[name] = (pw, float(w_at[2]))
    for name, (pw, z) in sorted(balls.items(), key=lambda kv: kv[1][1]):
        fog = depth.fog(z)
        parts.append(f'<g class="water"><circle cx="{pw[0]:.1f}" cy="{pw[1]:.1f}" r="{5.0 + EDGE:.1f}" '
                     f'fill="{openbook.as_hex(_mix(DARK, fog))}"/><circle cx="{pw[0]:.1f}" cy="{pw[1]:.1f}" r="5.0" '
                     f'fill="{openbook.as_hex(_mix(WATER, 0.5 * fog))}"/><title>water {html.escape(name)}</title></g>')
    parts += numbers + names

    key, key_h = _key(cx, w, head + h + 12.0, plddt_colors, stated, marks, more, scale, shares is not None, backdrop,
                      rings=rings, pale=any(i not in lined for i in shown), water=bool(balls), numbered=bool(numbers),
                      kinds=kinds)
    parts.append(key)
    width_, height_ = max(w, 420.0), head + h + 12.0 + key_h
    for line in _lines(footnote, width_):
        height_ += 14.0
        parts.append(openbook.text(0, height_ - 4.0, html.escape(line), 9.5, openbook.QUIET))
    svg = (f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_:.0f} {height_:.0f}" '
           f'width="{width_:.0f}" font-family="{openbook.FONT}">' + "".join(parts) + "</svg>")
    present = [k for k in kinds if any(c.kind == k for c in cx.contacts)]
    facts = {"names": list(cx.names), "residues": len(shown), "with_a_line": len(lined & set(shown)),
             "kinds": {k: sum(1 for c in cx.contacts if c.kind == k) for k in present},
             "pose": pose.by, "fit": None if pose.fit is None else round(pose.fit, 2), "lean": list(pose.lean),
             "hidden": hides, "hidden_without_lean": hides_flat,
             "units_per_angstrom": round(scale, 1), "plddt": stated,
             "in_front": sum(1 for i in shown if height_of(i) >= 0.0), "waters": len(balls),
             # Where the view is seen from, in the coordinates of the model: for whatever is to show the
             # same thing and turn it (a script for a structure viewer, a page's own viewer).
             "camera": {"across": [round(float(v), 6) for v in pose.axes[0]],
                        "up": [round(float(v), 6) for v in pose.axes[1]],
                        "towards_the_reader": [round(float(v), 6) for v in pose.axes[2]],
                        "centre": [round(float(v), 3) for v in pose.centre]},
             "drawn": [f"{cx.residues[i].chain}:{cx.residues[i].name}{cx.residues[i].number}{cx.residues[i].icode}"
                       for i in shown],
             "contacts": [{"kind": c.kind,
                           "residue": f"{cx.residues[c.a].chain}:{cx.residues[c.a].name}{cx.residues[c.a].number}"
                                      f"{cx.residues[c.a].icode}",
                           "protein_atoms": [cx.atoms[a] for a in c.atoms_a],
                           "ligand_atoms": [cx.atoms[a] for a in c.atoms_b],
                           "distance": round(float(c.distance), 2)}
                          for c in cx.contacts if c.kind in kinds and c.atoms_a and c.atoms_b]}
    fit = (f"its atoms lie {pose.fit:.1f} Å (root mean square) from where the diagram has them" if pose.fit is not None
           else "the diagram gives no place for its atoms, so its longest axis runs across the page")
    leant = ("" if not any(pose.lean) else
             f" The view is leaned {_lean_words(pose.lean)}: {hides:.0f} atoms and contacts hide something else or "
             f"are seen from their end, against {hides_flat:.0f} with the ligand's plane flat on the page.")
    facts["said"] = (f"{len(shown)} residues make a contact of the diagram with {cx.names[1]} and are drawn, "
                     f"{facts['with_a_line']} of them a contact that has a direction. The ligand is turned as in "
                     f"the diagram: {fit}.{leant} {facts['in_front']} of the residues lie in front of the ligand "
                     f"and are drawn over it.")
    return _Drawn(openbook.Sketch(svg, "a ligand and its contacts in three dimensions", facts), cx)


def _lean_words(lean) -> str:
    back, side = lean
    said = []
    if back:
        said.append(f"{abs(back):.0f}° {'back' if back > 0 else 'forward'}")
    if side:
        said.append(f"{abs(side):.0f}° to the {'left' if side > 0 else 'right'}")
    return " and ".join(said)


def _heading(cx: book.Complex, pose: Pose, several_: bool) -> str:
    who = "the ligands" if several_ else html.escape(cx.names[1])
    how = "in the pose of the diagram" if pose.by == "diagram" else "seen across the ligand's own plane"
    lean = f", leaned {_lean_words(pose.lean)}" if any(pose.lean) else ""
    return f"{who} and the residues in contact, as sticks, {how}{lean}"


def _backdrop(cx: book.Complex, at: np.ndarray, bounds, scale: float, head: float) -> str:
    """The protein behind the middle of the ligand, as a pale shape under the sticks: cut on the plane
    through the ligand's middle that faces the reader, and seen from the cut. Where the body is cut
    through it is hatched; the hollow in it is the pocket. (With the ligand's plane flat on the page
    that is the cut of the second panel of :func:`pocket.figure`.)"""
    protein = cx.side == 0
    colours = np.tile(np.array([0.90, 0.91, 0.93]), (int(protein.sum()), 1))
    x0, x1, y0, y1 = bounds
    extra, k = 3.6, scale / 2.0                 # drawn a little beyond the page and trimmed: the surface is worn
    img, _ = surface.render(at[protein], [e for e, k_ in zip(cx.elements, protein) if k_], colours,      # back at an edge
                            (x0 - extra, x1 + extra, y0 - extra, y1 + extra), k, flat=0.55, cut=0.0)
    trim = int(round(extra * k))
    img = img[trim:img.shape[0] - trim, trim:img.shape[1] - trim]
    rows, cols = img.shape[:2]                                  # the body goes on beyond the page: it fades out
    n = max(1, int(round(2.5 * k)))
    edge = np.minimum(np.minimum(np.arange(cols)[None, :], np.arange(cols)[None, ::-1]),
                      np.minimum(np.arange(rows)[:, None], np.arange(rows)[::-1, None])) / n
    s = np.clip(edge, 0.0, 1.0)
    img = img.copy()
    img[..., 3] = (img[..., 3] * (s * s * (3.0 - 2.0 * s))).astype(np.uint8)
    src = "data:image/png;base64," + base64.b64encode(surface.png(img)).decode("ascii")
    return (f'<image href="{src}" x="0" y="{head:.1f}" width="{(x1 - x0) * scale:.1f}" height="{(y1 - y0) * scale:.1f}" '
            f'opacity="0.42"><title>the protein behind the middle of the ligand, cut on a plane that faces the reader'
            f'</title></image>')


_SAID = {"N": "nitrogen blue", "O": "oxygen red", "S": "sulfur yellow", "P": "phosphorus a larger ball",
         "F": "fluorine green", "CL": "chlorine green", "BR": "bromine brown", "I": "iodine violet"}


def _key(cx: book.Complex, width: float, top: float, plddt_colors: bool, stated: bool, marks: bool, more, scale: float,
         shared: bool, backdrop: bool = False, rings: bool = True, pale: bool = True, water: bool = False,
         numbered: bool = True, kinds=None) -> "tuple[str, float]":
    """What the lines, the sticks and their weights say, under the picture: of what occurs in it, and of
    nothing else. Comes back with its height."""
    out, x, y = [], 0.0, top + 10.0
    room = max(width, 420.0)

    def item(sample: str, sample_w: float, words: str) -> None:
        nonlocal x, y
        need = sample_w + 6.0 + 5.2 * len(words) + 16.0
        if x > 0.0 and x + need > room:
            x, y = 0.0, y + 17.0
        out.append(sample.replace("{x}", f"{x:.1f}").replace("{x2}", f"{x + sample_w:.1f}").replace("{y}", f"{y - 3.5:.1f}"))
        out.append(openbook.text(x + sample_w + 6.0, y, words, 9.5, openbook.QUIET))
        x += need

    def row() -> None:
        nonlocal x, y
        if x > 0.0:
            x, y = 0.0, y + 17.0

    for kind in [k for k in (kinds or pocket.KINDS) if any(c.kind == k for c in cx.contacts)]:
        label, dash = pocket.KINDS[kind]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        fine = kind == DOTTED
        item(f'<line x1="{{x}}" y1="{{y}}" x2="{{x2}}" y2="{{y}}" stroke="{pocket.colour_of(kind)}" '
             f'stroke-width="{LINE_W * (0.62 if fine else 1.0):.2f}"{d}/>', 24.0, label)
    if rings:
        item(f'<circle cx="{{x2}}" cy="{{y}}" r="3.6" fill="#ffffff" stroke="{openbook.QUIET}" stroke-width="1.4" '
             f'transform="translate(-6 0)"/>', 12.0, "the middle of a ring")
    if water:
        item(f'<circle cx="{{x2}}" cy="{{y}}" r="{5.0 + EDGE:.1f}" fill="{openbook.as_hex(DARK)}" transform="translate(-6 0)"/>'
             f'<circle cx="{{x2}}" cy="{{y}}" r="5.0" fill="{openbook.as_hex(np.array(WATER))}" transform="translate(-6 0)"/>',
             14.0, "a water that a bridge runs through")
    if numbered:
        item("", 0.0, "a number on a line: its length in Å")
    row()
    stick = ('<line x1="{x}" y1="{y}" x2="{x2}" y2="{y}" stroke="%s" stroke-width="%.1f" stroke-linecap="round"/>'
             '<line x1="{x}" y1="{y}" x2="{x2}" y2="{y}" stroke="%s" stroke-width="%.1f" stroke-linecap="round"/>')
    dark = openbook.as_hex(DARK)
    if stated and plddt_colors:
        item(stick % (dark, LIGAND_W + 2 * EDGE, "#c9ced6", LIGAND_W), 22.0,
             f"{html.escape(cx.names[1])}: the thicker sticks (its elements are in the diagram)")
    else:
        has = {cx.elements[a].upper() for a in cx.residues[-1].atoms}
        others = ", ".join(words for e, words in _SAID.items() if e in has)
        item(stick % (dark, LIGAND_W + 2 * EDGE, openbook.as_hex(np.array(openbook.CARBON)), LIGAND_W), 22.0,
             html.escape(cx.names[1]) + (" (carbon" + (f"; {others}" if others else "") + ")" if "C" in has else ""))
    for _, _, _, other, carbon in more:
        item(stick % (dark, OTHER_W + 2 * EDGE, openbook.as_hex(np.array(carbon)), OTHER_W), 22.0,
             html.escape(other.names[1]))
    row()
    # (A nucleotide has no alpha carbon to draw a side chain from: all of it is drawn.)
    amino = any(cx.residues[c.a].ca >= 0 for c in cx.contacts if c.kind in (kinds or pocket.KINDS))
    item(stick % (dark, RESIDUE_W + 2 * EDGE, openbook.as_hex(np.array(SLATE)), RESIDUE_W), 22.0,
         "a residue in contact: its side chain from Cα, its backbone where that makes the contact" if amino
         else "a nucleotide in contact: base, sugar and phosphate")
    if pale:
        item(stick % (dark, PALE_W + 2 * EDGE, openbook.as_hex(_mix(SLATE, 0.22)), PALE_W), 22.0, "hydrophobic contacts only")
    row()
    item("", 0.0, "what is nearer to the reader lies over what is farther, which is thinner and paler")
    row()
    if backdrop:
        item("", 0.0, "pale, under the sticks: the protein behind the middle of the ligand, cut on a plane that faces "
                      "the reader (hatched where it is cut through; the hollow is the pocket)")
        row()
    if stated and plddt_colors:
        out.append(openbook.text(x, y, "pLDDT, in colour: of a residue (all its atoms), of each atom of the ligand:", 9.5,
                                 openbook.QUIET))
        x += 372.0
        for band, colour in openbook.BANDS.items():
            item(f'<rect x="{{x}}" y="{{y}}" width="11" height="11" rx="2" fill="{colour}" transform="translate(0 -5.5)"/>',
                 11.0, html.escape(BAND_LABELS[band].replace("pLDDT ", "")))
        row()
    elif stated:
        grey = openbook.as_hex(np.array(SLATE))
        out.append(openbook.text(x, y, "pLDDT of the model, in weight:", 9.5, openbook.QUIET))
        x += 160.0
        for words, look in (("70 or more", look_of(95.0)), ("50 to 70", look_of(60.0)), ("below 50", look_of(30.0))):
            w = RESIDUE_W * max(0.6, look.width)
            item(f'<g opacity="{look.opacity:.2f}">' + stick % (dark, w + 2 * EDGE, grey, w) + "</g>", 22.0, words)
        item("", 0.0, "a contact at the lower of its two ends" + ("; ? below 50" if marks else ""))
        row()
    if shared:
        n = 1 + len(more)
        out.append(openbook.text(x, y, f"a contact is drawn for the first ligand, and weighted by how many of the "
                                       f"{n} ligands make it with that residue (k/{n})", 9.5, openbook.QUIET))
        y += 17.0
    bar_y = top - 12.0                                          # in the corner of the picture, over the key
    out.append(f'<line x1="{room - 5 * scale:.1f}" y1="{bar_y}" x2="{room:.1f}" y2="{bar_y}" stroke="{openbook.INK}" '
               f'stroke-width="1.2"/>' + openbook.text(room, bar_y - 4.0, "5 Å", 9.5, openbook.QUIET, "end"))
    return "".join(out), y - top + 4.0


def view(diagram: Diagram, *, style: str = "detailed", plddt_colors: bool = False, labels: bool = True,
         distances: bool = True, backdrop: bool = False, lean="auto", kinds=None) -> openbook.Sketch:
    """A ligand and the residues that make a contact of its diagram, as sticks, in the pose of the diagram.

    ``diagram`` is a ``bindview.Diagram`` made from a structure. ``style`` is that of the flat diagram
    whose pose is taken. ``plddt_colors`` shows a model's pLDDT by AlphaFold's four colours, a residue
    in the colour of its band and every atom of the ligand in the colour of its own, instead of by
    weight (in grey: a residue below 70 is thinner and paler, a contact is drawn at the lower of its
    two ends). The elements are then not told apart here: the diagram has them. ``labels`` names the residues, ``distances`` writes its length
    on every contact that has a direction. ``backdrop`` lays the protein behind the middle of the
    ligand under the sticks, pale: cut on a plane that faces the reader, the cut hatched.

    ``lean``: with the ligand's plane flat on the page, a residue that lies in front of the ligand or
    behind it (one that stacks on a ring) is drawn on top of it, and a contact between them is seen
    from its end. ``"auto"`` leans the plane, back or forward and to a side, by one of a few fixed
    angles up to 35°: the one that hides the least (:func:`hidden`, :func:`best_lean`), and only where
    that uncovers enough to be worth it. Two angles in degrees (back, to the left) lean it by those;
    ``0`` leaves the plane flat, which is the pose of the diagram exactly.

    ``kinds``: only these kinds of contact (``"hbond"``, ``"water_bridge"``, ``"saltbridge"``,
    ``"pi_stacking"``, ``"pi_cation"``, ``"halogen"``, ``"metal"``, ``"hydrophobic"``) and the residues
    that make them: for a pocket with more contacts than one picture holds. A water bridge runs through
    its water, which is drawn, where the table names the water and the structure has it.

    ``facts`` holds, beside what the figure says of itself, where it is seen from (``camera``: three axes
    and a centre in the coordinates of the model) and every contact it draws with its atoms
    (``contacts``): what a structure viewer needs to show the same thing from the same place.
    """
    return _draw(diagram, style, plddt_colors, labels, distances, backdrop, lean, kinds).sketch


def _over(flat: "str | None", under: str, names: "dict[str, str]", what) -> str:
    """The library's figure over the view, the two tied by a style sheet (as in :func:`pocket.figure`):
    what is pointed at, or kept, in one steps the rest of the other back. ``names`` says which residue
    of the one (``data-residue``) is which group of the other (``data-id``)."""
    w2, h2 = openbook.size_of(under)
    parts, top, wide = [], 0.0, w2
    if flat is not None:
        m = re.match(r'<svg ([^>]*?)width="([0-9.]+)" height="([0-9.]+)"', flat)
        assert m, "a figure of the library without a size"
        w1, h1 = float(m.group(2)), float(m.group(3))
        wide = max(w1, w2)
        parts.append(flat.replace(m.group(0), f'<svg x="{(wide - w1) / 2.0:.1f}" y="0" {m.group(1)}'
                                              f'width="{w1:.1f}" height="{h1:.1f}"', 1))
        top = h1 + 14.0
    parts.append(openbook.nest(under, (wide - w2) / 2.0, top))
    scope = "svg.i3-" + hashlib.sha1(json.dumps([list(what), sorted(names)]).encode()).hexdigest()[:8]
    return (f'<svg class="book interact {scope[4:]}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {wide:.0f} '
            f'{top + h2:.0f}" width="{wide:.0f}" font-family="{openbook.FONT}"><style>{openbook.tied(scope, names, False)}</style>'
            + "".join(parts) + "</svg>")


def figure(diagram: Diagram, style: str = "detailed", effort: str = "full", **options) -> openbook.Sketch:
    """The library's diagram of a ligand, and under it the same ligand in three dimensions, in the same pose.

    ``style`` and ``effort`` are those of the diagram. The other options are those of :func:`view`. A residue
    pointed at in the diagram is marked in the view and the other way round, by a style sheet in the figure, and
    so is one that the page around the figure asks for (``data-active`` on the root).
    """
    if style not in ("detailed", "schematic"):
        raise ValueError(f"style={style!r} is not one of 'detailed', 'schematic'")
    flat = diagram.render(style=style, interactive=True, effort=effort)
    got = _draw(diagram, style, flat=flat, **options)
    cx = got.cx
    held = {c.a for c in cx.contacts if c.kind in pocket.KINDS}      # (one that reaches through a water is not near)
    names = {f"{r.chain}:{r.name}{r.number}{r.icode}": r.key for r in cx.residues
             if r.side == 0 and (len(r.near) or r.index in held)}
    svg = _over(flat.svg, openbook.numbered(got.sketch.svg, cx.residues), names, [cx.names[1]])
    return openbook.Sketch(svg, "a diagram and its ligand in three dimensions",
                           dict(got.sketch.facts, style=style, diagram=flat.status), warnings=list(flat.warnings))


def several(diagrams, **options) -> openbook.Sketch:
    """Several ligands of one pocket in one view: the first in full, the others thinner.

    ``diagrams`` are bindview ``Diagram`` objects of ligands in the same pocket, as a ``Comparison`` takes
    them. Their models are superposed as the library's comparison superposes them (the backbone of the
    residues around the ligands). Drawn are the residues and the contacts of the first model, in the
    pose of the first ligand's diagram, and the other ligands where they lie in that frame, each in a
    colour of its own. A contact of the first is weighted by the share of the ligands that make a
    contact of that kind with that residue, and says so beside its length (``2/3``). What the others
    make and the first does not is counted in ``facts`` and said under the figure, not drawn. The
    options are those of :func:`view`.
    """
    diagrams = list(diagrams)
    n = len(diagrams)
    if n < 2:
        raise ValueError("several ligands are two or more")
    if n > len(pocket.CARBONS):
        raise ValueError(f"{n} ligands: the sketch has a colour for {len(pocket.CARBONS)}")
    fits: "list[float | None]" = []
    moves = pocket_transforms(diagrams, rmsds=fits)
    apart = [d.name for d, m in zip(diagrams, moves) if m is None]
    if apart:
        raise ValueError(f"no common frame for {', '.join(apart)}: the residues around the ligands do not match "
                         f"those of {diagrams[0].name}, or do not fit them")
    each = []
    for dia, (rot, shift) in zip(diagrams, moves):
        cx = book.load_ligand(dia, ("the protein", dia.name or "the ligand"))
        cx.xyz = cx.xyz @ rot.T + shift                       # into the frame of the first model
        each.append(cx)
    made: "dict[tuple[str, int, str, str], int]" = {}
    for cx in each:
        for key in {(cx.residues[c.a].chain, cx.residues[c.a].number, cx.residues[c.a].icode, c.kind)
                    for c in cx.contacts if c.kind in pocket.KINDS}:
            made[key] = made.get(key, 0) + 1
    first = {(each[0].residues[c.a].chain, each[0].residues[c.a].number, each[0].residues[c.a].icode, c.kind)
             for c in each[0].contacts if c.kind in pocket.KINDS}
    missing = sorted((k for k in made if k not in first), key=lambda k: (-made[k], k))
    every = sum(1 for k in first if made[k] == n)
    said = (f"{len(first)} contacts of {each[0].names[1]} with a residue, by kind: {every} of them are made by all "
            f"{n} ligands. {len(missing)} contacts that another ligand makes and {each[0].names[1]} does not are not "
            f"drawn" + (": " + ", ".join(f"{pocket.KINDS[k[3]][0]} with residue {k[1]} ({made[k]}/{n})"
                                         for k in missing[:3]) + (f" and {len(missing) - 3} more" if len(missing) > 3
                                                                   else "") if missing else "") + ". "
            f"The models are fitted on the first with {', '.join(f'{f:.1f}' for f in fits[1:])} Å over the backbone "
            f"around the ligands.")
    got = _draw(diagrams[0], others=list(zip(each[1:], diagrams[1:])), shares=made, footnote=said, **options)
    svg = got.sketch.svg
    facts = dict(got.sketch.facts, ligands=[cx.names[1] for cx in each],
                 fits=[None if f is None else round(f, 2) for f in fits], contacts_of_the_first=len(first),
                 made_by_all=every, not_drawn=len(missing), said=said)
    return openbook.Sketch(svg, "several ligands of one pocket, in three dimensions", facts)


# ---------------------------------------------------------------------------------------- a page of them
def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(__doc__.split("\n\n")[2].strip(), file=sys.stderr)
        return 2
    t = time.time()
    body = []
    for case, title, what in pocket.CASES:
        one = figure(pocket.diagram_of(case))
        body.append(f'<h2 id="{case}">{html.escape(title)}</h2><p class="quiet">{what} {html.escape(one.facts["said"])}</p>'
                    f"<figure>{one.svg}</figure>")
        print(f"{case:18s} {one.facts['residues']} residues, fit {one.facts['fit']} Å   [{time.time() - t:.0f} s]",
              flush=True)
    for slug, title, what, members in pocket.SETS:
        one = several([pocket.diagram_of(m) for m in members])
        body.append(f'<h2 id="{slug}">{html.escape(title)}</h2><p class="quiet">{what}</p><figure>{one.svg}</figure>')
        print(f"{slug:18s} {one.facts['made_by_all']} of {one.facts['contacts_of_the_first']} made by all   "
              f"[{time.time() - t:.0f} s]", flush=True)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sketches: a ligand and its contacts in three dimensions</title><style>{getattr(openbook, "CSS", "")}</style></head><body><main>
<h1>A ligand and its contacts in three dimensions, in the pose of its diagram</h1>
<p><span class="sketch">Sketches.</span> Not part of bindview, not tested, nothing here is promised.</p>
{"".join(body)}
</main><script>{getattr(openbook, "JS", "")}</script></body></html>
"""
    Path(args[0]).write_text(page, encoding="utf-8")
    print(f"wrote {args[0]}: {len(page) / 1e6:.2f} MB in {time.time() - t:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
