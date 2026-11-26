#!/usr/bin/env python
"""A small molecule in its pocket, opened like a book: the four panels of a peptide in its groove, for a ligand.

A SKETCH, not part of bindview: nothing here is tested or promised.

``panels(diagram)`` draws, for a bindview ``Diagram`` made from a structure, what :func:`openbook.panels`
draws for two chains: **a** the complex as it stands, from the side; **b** opened; **c** one face large,
with the ligand as licorice and every contact that has a direction from atom to atom; **d** the same
frame at the same scale as a flat map, with the ligand's skeleton, and with a box for every residue that
makes a contact. The contacts are the diagram's own (``Diagram.interactions``, resolved to atoms by the
library) and have its colours, as in ``pocket.py``.

**Cut, or not.** A protein that lies against one side of a ligand is opened as it is: its page shows the
face that the ligand covers, and the other page is the ligand. One that has the ligand inside shows
nothing that way, and is cut along the ligand's own plane: the two pages are then the two halves of the
pocket, each seen from the cut (``pocket.py`` does the same). Which of the two is decided by what can be
seen, with no threshold: the share of the area the ligand covers on the protein that has nothing of the
protein between it and a reader on the ligand's side (:func:`harder.page`), against the share that the
two halves of the cut show together (:func:`harder.seen_cut`). The protein is cut where the halves show
more. (0 against 41 % for imatinib in c-Kit, 5 against 50 % for osimertinib in EGFR: both are cut.)

**What is drawn of a model.** A prediction carries tails it is not sure of, and they would be most of the
picture. Where a model states its pLDDT, the panels show the piece that holds the ligand: the residues
of a pLDDT of 70 or more that hang together with the pocket, alpha carbon to alpha carbon within 7 Å,
and every residue of the pocket itself. The numbers (what is buried, what is seen) are of the whole model.
"""

from __future__ import annotations

import hashlib
import html
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import book  # noqa: E402
import harder  # noqa: E402
import interact3d  # noqa: E402
import openbook  # noqa: E402
import pocket  # noqa: E402
import route  # noqa: E402
import surface  # noqa: E402
from bindview import Diagram  # noqa: E402
from bindview.theme import LIGHT  # noqa: E402

__all__ = ["panels"]

INK, QUIET, FAINT = openbook.INK, openbook.QUIET, openbook.FAINT
ROOM, GAP = 774.0, 26.0      # the width of the figure (that of the four panels of a peptide), and between two panels
LARGEST = 14.0               # units for an ångström in panels c and d, at most
TALLEST = 620.0              # and how high those two may become
AROUND, BESIDE = 6.0, 3.0    # ångström of room in panels c and d: around the ligand's atoms, around a residue's
SURE, LINK = 70.0, 7.0       # what is drawn of a prediction (see above)
COLOURS = ("area", "class", "plddt")           # of the pocket
BAND = openbook.TRACE[1]     # under the skeleton of panel d: the ligand's colour, pale, as a peptide's chain has it
WIDTHS = {"hydrophobic": 1.3}                  # of a contact's line in panel d; 1.5 for any other


# ---------------------------------------------------------------------------------------- what is drawn of a model
def piece(cx: book.Complex) -> book.Complex:
    """The part of a model that the panels show, with the ligand: see the docstring of the file.

    The ligand is left out of the scale of the area (``buried`` is 0 for it): it is one piece, loses
    far more than any residue, and would leave the residues the pale end of the scale.
    """
    protein = [r for r in cx.residues if r.side == 0]
    keep = {r.index for r in protein if len(r.near)} | {c.a for c in cx.contacts}
    if cx.plddt:
        at = np.array([cx.xyz[r.ca] if r.ca >= 0 else cx.xyz[r.atoms].mean(axis=0) for r in protein])
        sure = np.array([cx.plddt.get(r.index, 100.0) >= SURE for r in protein])
        held = np.array([r.index in keep for r in protein]) & sure
        todo = list(np.where(held)[0])
        while todo:
            n = todo.pop()
            fresh = np.where(sure & ~held & (np.linalg.norm(at - at[n], axis=1) < LINK))[0]
            held[fresh] = True
            todo.extend(fresh)
        keep |= {r.index for r, k in zip(protein, held) if k}
    else:
        keep |= {r.index for r in protein}
    kept = sorted(keep | {r.index for r in cx.residues if r.side == 1})
    atoms = np.concatenate([cx.residues[i].atoms for i in kept])
    new = np.full(len(cx.xyz), -1)
    new[atoms] = np.arange(len(atoms))
    number = {i: n for n, i in enumerate(kept)}
    residues = [replace(r, atoms=new[r.atoms], near=new[r.near], index=number[r.index],
                        ca=int(new[r.ca]) if r.ca >= 0 else -1, buried=r.buried if r.side == 0 else 0.0)
                for r in (cx.residues[i] for i in kept)]
    contacts = [replace(c, a=number[c.a], b=number[c.b], atoms_a=tuple(int(new[a]) for a in c.atoms_a),
                        atoms_b=tuple(int(new[a]) for a in c.atoms_b)) for c in cx.contacts]
    return book.Complex(cx.title, cx.names, cx.xyz[atoms], [cx.elements[a] for a in atoms], cx.side[atoms],
                        np.concatenate([np.full(len(r.atoms), r.index) for r in residues]), residues, contacts,
                        cx.n_regions, cx.centre, cx.axes, {number[i]: v for i, v in cx.plddt.items() if i in number},
                        atoms=[cx.atoms[a] for a in atoms])


def turned(cx: book.Complex, low: bool) -> np.ndarray:
    """Every atom as it lies on one of the two pages, the reader at +z. ``low``: the page of what is below
    the plane, which is the protein's own page where nothing is cut; else the page of what is above it."""
    shut = cx.closed()
    return (np.stack([-shut[:, 2], shut[:, 1], shut[:, 0]], axis=1) if low
            else np.stack([shut[:, 2], shut[:, 1], -shut[:, 0]], axis=1))


def middle_of(r: book.Residue, xyz: np.ndarray) -> np.ndarray:
    """Where a residue is: the middle of its atoms at the pocket (of all of them, for one whose contact
    reaches farther than that)."""
    return xyz[r.near if len(r.near) else r.atoms].mean(axis=0)


# ---------------------------------------------------------------------------------------- names and boxes
@dataclass
class _Tag:
    """A name or a box: where it belongs, half its size, and where its lines go: to every one of ``ends``
    (a box and its contacts), or to the nearest of them alone (a name and its hairline). ``off`` are
    points of its own that it must not cover (the atoms of the residue it names)."""

    want: np.ndarray
    hw: float
    hh: float
    ends: np.ndarray
    nearest: bool = False
    off: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


def _crossings(p: np.ndarray, q: np.ndarray, r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """How many of the lines ``r`` to ``s`` each line ``p`` to ``q`` crosses (ends that meet do not count)."""
    if not len(r):
        return np.zeros(len(p))
    d1, d2, off = (q - p)[:, None, :], (s - r)[None, :, :], r[None, :, :] - p[:, None, :]
    down = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    down = np.where(np.abs(down) < 1e-9, np.inf, down)
    t = (off[..., 0] * d2[..., 1] - off[..., 1] * d2[..., 0]) / down
    u = (off[..., 0] * d1[..., 1] - off[..., 1] * d1[..., 0]) / down
    return ((t > 0.03) & (t < 0.97) & (u > 0.03) & (u < 0.97)).sum(axis=1).astype(float)


def _near_lines(p: np.ndarray, q: np.ndarray, points: np.ndarray, within: float, beyond: float) -> np.ndarray:
    """How many of ``points`` lie within ``within`` of each line ``p`` to ``q``, and farther than ``beyond``
    from its start: how much of what they stand for a line runs along."""
    out = np.zeros(len(p))
    v = q - p
    long2 = np.maximum((v * v).sum(axis=1), 1e-9)
    for k in range(0, len(points), 200):
        w = points[None, k:k + 200, :] - p[:, None, :]
        t = np.clip((w * v[:, None, :]).sum(axis=2) / long2[:, None], 0.0, 1.0)
        far = np.linalg.norm(w - t[..., None] * v[:, None, :], axis=2)
        out += ((far < within) & (np.linalg.norm(w, axis=2) > beyond)).sum(axis=1)
    return out


def _covered(c: np.ndarray, hw: float, hh: float, points: np.ndarray) -> np.ndarray:
    """How many of ``points`` a box of that half size covers at each of the places ``c``."""
    out = np.zeros(len(c))
    for k in range(0, len(points), 400):
        part = points[k:k + 400]
        out += ((np.abs(part[None, :, 0] - c[:, None, 0]) < hw)
                & (np.abs(part[None, :, 1] - c[:, None, 1]) < hh)).sum(axis=1)
    return out


def settle(tags: "list[_Tag]", solid: np.ndarray, bones: np.ndarray, page, clear: float = 4.0, soft: np.ndarray = None,
           letters: np.ndarray = None, fixed=(), reach: float = 150.0, step: float = 5.0, pad: float = 4.0,
           sweeps: int = 10) -> np.ndarray:
    """A place for every tag: near where it belongs, on no point of ``solid`` (with ``clear`` around it),
    on no other tag and off the lines of the others; its own lines across as few of ``bones`` (lines of
    the ligand), ``letters`` (boxes), other tags and other lines as can be. ``soft`` are points it rather
    does not cover, ``fixed`` tags already placed (middle, half width, half height, the line to it or
    ``None``).

    One tag after the other is put at the best place of a grid, the others staying, a few times over. Where
    that ends depends on who comes first, so it is done in several orders, and the best end is kept; then
    two tags change places wherever that is better.
    """
    if not tags:
        return np.zeros((0, 2))
    x0, y0, x1, y1 = page
    spots, walks = [], []
    for t in tags:
        gx, gy = np.meshgrid(np.arange(-reach, reach + 0.1, step), np.arange(-reach, reach + 0.1, step))
        c = np.vstack([t.want[None, :], t.want + np.stack([gx.ravel(), gy.ravel()], axis=1)])
        c = c[np.hypot(*(c - t.want).T) <= reach]
        c[:, 0] = np.clip(c[:, 0], x0 + t.hw, x1 - t.hw)
        c[:, 1] = np.clip(c[:, 1], y0 + t.hh, y1 - t.hh)
        free = np.ones(len(c), dtype=bool)
        for room in (clear, clear / 2.0, 0.5):
            free = _covered(c, t.hw + room, t.hh + room, solid) == 0
            free &= _covered(c, t.hw + 1.5, t.hh + 1.5, t.off) == 0
            if free.any():
                break
        c = c[free] if free.any() else c
        cost = np.hypot(*(c - t.want).T)
        if soft is not None and len(soft):
            cost = cost + 2.5 * _covered(c, t.hw, t.hh, soft)
        lines = ([t.ends[np.argmin(np.linalg.norm(t.ends[None, :, :] - c[:, None, :], axis=2), axis=1)]] if t.nearest
                 else [np.repeat(e[None, :], len(c), axis=0) for e in t.ends]) if len(t.ends) else []
        for ends in lines:
            cost = cost + 0.35 * np.hypot(*(c - ends).T)
            if len(bones):
                cost = cost + (30.0 if t.nearest else 14.0) * _crossings(ends, c, bones[:, :2], bones[:, 2:])
            if not t.nearest:                                 # (and a line long enough to be seen)
                gap = np.maximum(np.abs(ends - c)[:, 0] - t.hw, np.abs(ends - c)[:, 1] - t.hh)
                cost = cost + 2.0 * np.maximum(24.0 - gap, 0.0) + 2.0 * _near_lines(ends, c, solid, 3.5, 10.0)
                if letters is not None and len(letters):
                    others_own = np.abs(letters[None, :, :2] - ends[:, None, :]).max(axis=2) > 1.0
                    cost = cost + 30.0 * (route.crosses(ends, c, letters) & others_own).sum(axis=1)
        spots.append(c)
        walks.append((cost, lines))

    def others(n, at):
        rects = [(float(q[0]), float(q[1]), hw, hh) for q, hw, hh, _ in fixed]
        starts, ends = [], []
        for _, _, _, line in fixed:
            if line is not None:
                starts.append(line[0])
                ends.append(line[1])
        for m, k in enumerate(at):
            if m == n or k is None:
                continue
            rects.append((spots[m][k][0], spots[m][k][1], tags[m].hw, tags[m].hh))
            for line in walks[m][1]:
                starts.append(line[k])
                ends.append(spots[m][k])
        return np.array(rects).reshape(-1, 4), np.array(starts).reshape(-1, 2), np.array(ends).reshape(-1, 2)

    def costs(n, at, only=slice(None)) -> np.ndarray:
        t, c, total = tags[n], spots[n][only], walks[n][0][only].copy()
        lines = [line[only] for line in walks[n][1]]
        rects, starts, ends = others(n, at)
        if len(rects):
            over = ((np.abs(c[:, None, 0] - rects[None, :, 0]) < t.hw + rects[None, :, 2] + pad)
                    & (np.abs(c[:, None, 1] - rects[None, :, 1]) < t.hh + rects[None, :, 3] + pad))
            total += 5000.0 * over.sum(axis=1)
        for line in lines:
            if len(rects):
                total += 70.0 * route.crosses(line, c, rects + np.array([0.0, 0.0, 2.0, 2.0])).sum(axis=1)
            total += 35.0 * _crossings(line, c, starts, ends)
        if len(starts):
            mine = np.stack([c[:, 0], c[:, 1], np.full(len(c), t.hw + 2.0), np.full(len(c), t.hh + 2.0)], axis=1)
            total += 70.0 * route.crosses(starts, ends, mine).sum(axis=0)
        return total

    middle = np.mean([t.want for t in tags], axis=0)
    off = [t.want - middle for t in tags]
    best: "tuple[float, list] | None" = None
    for rank in (lambda n: float(np.hypot(*off[n])), lambda n: -float(np.hypot(*off[n])), lambda n: float(off[n][1]),
                 lambda n: -float(off[n][1])):
        order = sorted(range(len(tags)), key=lambda n: (round(rank(n), 3), n))
        at: "list[int | None]" = [None] * len(tags)
        for _ in range(sweeps):
            moved = False
            for n in order:
                k = int(np.argmin(costs(n, at)))
                if k != at[n]:
                    at[n], moved = k, True
            if not moved:
                break
        total = sum(float(costs(n, at, [at[n]])[0]) for n in range(len(tags)))
        if best is None or total < best[0] - 1e-6:
            best = (total, at)
    at = best[1]
    for _ in range(3):
        moved = False
        for m in range(len(tags)):
            for n in range(m + 1, len(tags)):
                other = list(at)
                other[m] = int(np.argmin(np.hypot(*(spots[m] - spots[n][at[n]]).T)))
                other[n] = int(np.argmin(np.hypot(*(spots[n] - spots[m][at[m]]).T)))
                now, then = (sum(float(costs(k, way, [way[k]])[0]) for k in (m, n)) for way in (at, other))
                if then < now - 1e-6:
                    at, moved = other, True
        for n in order if moved else ():
            at[n] = int(np.argmin(costs(n, at)))
        if not moved:
            break
    return np.array([spots[n][k] for n, k in enumerate(at)]).reshape(-1, 2)


# ---------------------------------------------------------------------------------------- panels a and b
def lost_by_atom(cx: book.Complex) -> "tuple[np.ndarray, float]":
    """The area the protein covers of every atom of the ligand, and where a scale for it ends (Å²)."""
    ligand, rad = cx.side == 1, surface.radii(cx.elements)
    which = np.where(ligand)[0]
    lost = (book.accessible(which, cx.xyz, rad, ligand)
            - book.accessible(which, cx.xyz, rad, np.ones(len(rad), dtype=bool)))
    return lost, max(5.0, float(5.0 * np.ceil(float(lost.max()) / 5.0)))


def whole(cx: book.Complex, colour: str, cut: bool, low: bool, who: str, frame) -> str:
    """The first two panels: the complex as it stands, and opened. ``low`` says which half panel c shows,
    and ``frame`` what of it (the bounds of panel c). ``who`` is the ligand, in a word or two."""
    metric = openbook.metric_of(cx, colour)
    colours = openbook.atom_colours(cx, metric)
    protein, ligand = cx.side == 0, cx.side == 1
    shut, pad, head, gap, foot = cx.closed(), 3.0, 34.0, 56.0, 34.0

    def box(v):
        return [v[:, 0].min() - pad, v[:, 0].max() + pad, v[:, 1].min() - pad, v[:, 1].max() + pad]

    b1 = box(shut)
    pages = []                                # of each: its atoms as they lie on it, its bounds, which are drawn
    if cut:
        for below in (True, False):
            v = turned(cx, below)
            kept = protein & (v[:, 2] <= 2.0)                 # (a ball that reaches the plane is cut through)
            pages.append((v, box(v[kept]), protein))
    else:
        if colour == "area":                                  # the ligand atom by atom, on a scale of its own
            lost, most = lost_by_atom(cx)
            for a, v in zip(np.where(ligand)[0], lost):
                colours[a] = openbook.blend(openbook.RAMP, v / most) if v > 0.5 else np.array(openbook.BODY[1])
        pages = [(turned(cx, below), box(turned(cx, below)[m]), m) for below, m in ((True, protein), (False, ligand))]
    y0, y1 = min(b[2] for _, b, _ in pages), max(b[3] for _, b, _ in pages)
    between = 8.0                                                         # ångström between the two pages
    wide = sum(b[1] - b[0] for _, b, _ in pages) + between
    k1, k2 = 3.3, 4.3
    shrink = min(1.0, (ROOM - gap - 12.0) / ((b1[1] - b1[0]) * k1 + wide * k2))
    k1, k2 = k1 * shrink, k2 * shrink
    balls = (shut[ligand], 0.92 * surface.radii([e for e, k in zip(cx.elements, ligand) if k]),
             np.tile(np.array(openbook.CARBON), (int(ligand.sum()), 1)))
    src1, w1, h1, _ = openbook.picture(shut, cx, colours, b1, k1, which=protein, licorice=balls)
    h2 = (y1 - y0) * k2
    top1, top2 = head + max(0.0, (h2 - h1) / 2.0), head + max(0.0, (h1 - h2) / 2.0)
    tall = max(h1, h2)
    x = w1 + gap
    title = (f"cut along the plane of {who} and opened, {who} taken out: a quarter turn each" if cut
             else "opened: a quarter turn each, both faces to the reader")
    parts = [openbook.heading(0, 14, "a", "as it stands, seen from the side"), openbook.heading(x, 14, "b", title),
             f'<image href="{src1}" x="0" y="{top1:.1f}" width="{w1}" height="{h1}"/>']
    sx = (0.0 - b1[0]) * k1
    parts.append(f'<line x1="{sx:.1f}" y1="{top1 - 4:.1f}" x2="{sx:.1f}" y2="{top1 + h1 + 4:.1f}" stroke="{INK}" '
                 f'stroke-width="0.9" stroke-dasharray="5 3"/>'
                 f'<path d="M {sx - 20:.1f} {top1 - 9:.1f} q 20 -13 40 0" fill="none" stroke="{INK}" '
                 f'stroke-width="0.9"/><path d="M {sx - 20:.1f} {top1 - 9:.1f} l 7.5 -1 l -4.6 -6.2 z" fill="{INK}"/>'
                 f'<path d="M {sx + 20:.1f} {top1 - 9:.1f} l -7.5 -1 l 4.6 -6.2 z" fill="{INK}"/>')
    a_name, b_name = html.escape(cx.names[0]), html.escape(cx.names[1])
    under = max(w1 / 2.0, 3.2 * max(len(a_name), len(b_name)))            # (a long name is not to leave the page)
    parts.append(openbook.text(under, top1 + h1 + 16, a_name, 11, INK, "middle", "600")
                 + openbook.text(under, top1 + h1 + 29, b_name, 11, openbook.EDGE[1], "middle", "600"))
    for n, (v, b, which) in enumerate(pages):
        bounds = (b[0], b[1], y0, y1)
        src, w, h, _ = openbook.picture(v, cx, colours, bounds, k2, which=which, cut=0.0 if cut else None)
        parts.append(f'<image href="{src}" x="{x:.1f}" y="{top2:.1f}" width="{w}" height="{h}"/>')
        for i in cx.at_interface(0) if (cut or n == 0) else ():
            r = cx.residues[i]
            if cut and (float(shut[r.near, 0].mean()) <= 0.0) != (n == 0):
                continue
            p = v[r.near].mean(axis=0)
            parts.append(f'<circle cx="{x + (p[0] - b[0]) * k2:.1f}" cy="{top2 + (y1 - p[1]) * k2:.1f}" r="6" '
                         f'fill="#000" fill-opacity="0"><title>{r.label} (chain {r.chain}): {r.buried:.0f} Å² '
                         'covered</title></circle>')
        says = (f'{"one half" if n == 0 else "the other half"} of {a_name}' if cut else (a_name, b_name)[n])
        ink = INK if cut or n == 0 else openbook.EDGE[1]
        parts.append(openbook.text(x + w / 2.0, top2 + h2 + 16, says, 11, ink, "middle", "600"))
        if (n == 0) == low:                                               # what of it panels c and d show
            fx0, fx1 = max(x, x + (frame[0] - b[0]) * k2), min(x + w, x + (frame[1] - b[0]) * k2)
            fy0, fy1 = max(top2, top2 + (y1 - frame[3]) * k2), min(top2 + h, top2 + (y1 - frame[2]) * k2)
            parts.append(f'<rect x="{fx0:.1f}" y="{fy0:.1f}" width="{fx1 - fx0:.1f}" height="{fy1 - fy0:.1f}" '
                         f'fill="none" stroke="{INK}" stroke-width="0.8"/>'
                         + openbook.text(fx1 - 4, fy0 + 12, "c, d", 9.5, INK, "end", "600",
                                         ' stroke="#ffffff" stroke-width="3" stroke-opacity="0.9" '
                                         'paint-order="stroke"'))
        if n == 0:
            spine = x + w + between * k2 / 2.0
            parts.append(f'<line x1="{spine:.1f}" y1="{top2 + 6:.1f}" x2="{spine:.1f}" y2="{top2 + h2 - 6:.1f}" '
                         f'stroke="{FAINT}" stroke-width="0.8" stroke-dasharray="5 3"/>')
        x += w + between * k2
    width, height = max(ROOM, x - between * k2), head + tall + foot
    return (f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
            f'width="{width:.0f}" font-family="{openbook.FONT}">' + "".join(parts) + "</svg>")


# ---------------------------------------------------------------------------------------- panels c and d
def says(cx: book.Complex, c: book.Contact) -> str:
    return (f"{cx.residues[c.a].label} – {html.escape(cx.residues[c.b].label)}: {pocket.KINDS[c.kind][0]}, "
            f"{c.distance:.1f} Å")


def tints(cx: book.Complex) -> np.ndarray:
    """A colour for every atom where it is drawn as licorice: by element, as the library has it."""
    return np.array([interact3d.element_colour(e, openbook.CARBON if s else openbook.CARBON_A)
                     for e, s in zip(cx.elements, cx.side)])


def thin_sticks(cx: book.Complex, i: int, view: np.ndarray, px, over) -> str:
    """A residue as thin sticks, faintly, as :func:`openbook.under` draws one with ``shown``, but behind
    what lies in front of it: where a stick of the ligand (``over``: its bonds, as the two atoms of
    each) is nearer to the reader, it is left out."""
    atoms = openbook.drawn_atoms(cx, i, pocket.LOOK)
    reach = 1.25 * surface.STICK                                          # the half width of a stick, and its edge
    runs, dots = [], []

    def hidden(q: np.ndarray) -> np.ndarray:
        out = np.zeros(len(q), dtype=bool)
        for a, b in over:
            v = view[b] - view[a]
            t = np.clip(((q - view[a])[:, :2] @ v[:2]) / max(1e-9, float(v[:2] @ v[:2])), 0.0, 1.0)
            near = view[a] + t[:, None] * v
            out |= (np.hypot(*(q - near)[:, :2].T) < reach) & (near[:, 2] > q[:, 2])
        return out

    for a, b in interact3d.bonds_of(cx, atoms):
        t = np.linspace(0.0, 1.0, 33)[:, None]
        q = view[a] + t * (view[b] - view[a])
        free = ~hidden(q)
        n = 0
        while n < len(q):
            if not free[n]:
                n += 1
                continue
            m = n
            while m + 1 < len(q) and free[m + 1]:
                m += 1
            if m > n:
                p0, p1 = px(q[n]), px(q[m])
                runs.append(f"M {p0[0]:.1f} {p0[1]:.1f} L {p1[0]:.1f} {p1[1]:.1f}")
            n = m + 1
    for a in atoms:
        if cx.elements[a].upper() in surface.ELEMENT and not hidden(view[a][None, :])[0]:
            p = px(view[a])
            dots.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="2.3" '
                        f'fill="{openbook.as_hex(np.array(surface.ELEMENT[cx.elements[a].upper()]))}"/>')
    d = " ".join(runs)
    return (f'<g opacity="0.45"><path d="{d}" fill="none" stroke="#ffffff" stroke-opacity="0.8" '
            f'stroke-width="4.4" stroke-linecap="round"/><path d="{d}" fill="none" stroke="{openbook.STICK}" '
            f'stroke-width="2.2" stroke-linecap="round"/>{"".join(dots)}</g>') if d else ""


def close_up(cx: book.Complex, view: np.ndarray, bounds, scale: float, colour: str, cut: bool, here, away,
             title: str) -> str:
    """Panel c. ``here`` are the residues that make a contact and are seen, ``away`` those that make one
    on the half that is cut away: these, with their contacts, come up while they are pointed at."""
    x0, x1, y0, y1 = bounds
    head = 30.0
    ligand, protein = cx.side == 1, cx.side == 0
    colours, by_element = openbook.atom_colours(cx, openbook.metric_of(cx, colour)), tints(cx)
    sticks = surface.sticks(view[ligand], [e for e, k in zip(cx.elements, ligand) if k], openbook.CARBON,
                            colours=by_element[ligand])
    src, w, h, _ = openbook.picture(view, cx, colours, bounds, scale, which=protein, fade=2.0, licorice=sticks,
                                    cut=0.0 if cut else None)

    def px(q) -> np.ndarray:
        return np.array([(q[0] - x0) * scale, head + (y1 - q[1]) * scale])

    made = [c for c in cx.contacts if c.kind in pocket.KINDS and c.atoms_a and c.atoms_b]
    far = {id(c): view[list(c.atoms_a)].mean(axis=0) - view[list(c.atoms_b)].mean(axis=0) for c in made}
    made.sort(key=lambda c: -float(np.hypot(*far[id(c)][:2])))            # (a short line over a long one)
    lines, traces = [], {}
    for c in made:
        pa, pb = px(view[list(c.atoms_a)].mean(axis=0)), px(view[list(c.atoms_b)].mean(axis=0))
        traces.setdefault(c.a, []).append((c, pa, pb))
        stroke, dash = pocket.LOOK[c.kind]
        at = f'x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}"'
        # Thin for the reader, where it is of this half and has a direction; bold for the pointer, whatever it is.
        for thick, ring, only in ((1.5, 2.4, ""), (2.8, 4.0, " only")):
            if not only and not (c.a in here and c.kind in pocket.LINED):
                continue
            ends = (f'<circle cx="{pa[0]:.1f}" cy="{pa[1]:.1f}" r="{ring}" fill="{stroke}" stroke="#ffffff" '
                    f'stroke-width="0.9"/>')
            if len(c.atoms_b) > 1 or only:
                ends += (f'<circle cx="{pb[0]:.1f}" cy="{pb[1]:.1f}" r="{ring}" fill="#ffffff" '
                         f'fill-opacity="{0.6 if only else 1}" stroke="{stroke}" '
                         f'stroke-width="{1.8 if only else 1.3}"/>')
            lines.append(f'<g class="link{only}" data-a="{cx.residues[c.a].key}" data-b="{cx.residues[c.b].key}">'
                         f'<line {at} stroke="#ffffff" stroke-opacity="0.85" stroke-width="{thick + 2.2:.1f}" '
                         f'stroke-linecap="round"/><line {at} stroke="{stroke}" stroke-width="{thick}" '
                         f'stroke-dasharray="{dash or "none"}"/>{ends}<title>{says(cx, c)}</title></g>')

    # The residues that are seen, faintly. And for whoever points at one: the ligand (the picture steps back,
    # and it must not go with it), and the residue as a whole, over the ligand.
    one = cx.residues[-1]
    bonds = interact3d.bonds_of(cx, np.where(ligand)[0])
    bodies = [f'<g class="res" data-id="{cx.residues[i].key}">{thin_sticks(cx, i, view, px, bonds)}</g>' for i in here]
    bodies.append(f'<g class="res" data-id="{one.key}">'
                  f'{openbook.under(cx, one.index, view, px, pocket.LOOK, thick=by_element, reach=False)}</g>')
    bodies += [f'<g class="res" data-id="{cx.residues[i].key}">'
               f'{openbook.under(cx, i, view, px, pocket.LOOK, reach=False)}</g>' for i in (*here, *away)]

    # The names: first those that stand there, then those that come up.
    size, half_h = 9.5, 7.2
    page = (2.0, head + 2.0, w - 2.0, head + h - 2.0)
    skeleton = openbook.licorice_points(cx, ligand, view, px)
    bones = [[*px(view[a]), *px(view[b])] for a, b in bonds]
    bones += [[*pa, *pb] for i in here for c, pa, pb in traces.get(i, ()) if c.kind in pocket.LINED]
    bones = np.array(bones).reshape(-1, 4)
    drawn_lines = [openbook.along(pa, pb) for i in here for c, pa, pb in traces.get(i, ()) if c.kind in pocket.LINED]
    atoms_of = {i: openbook.drawn_atoms(cx, i, pocket.LOOK) for i in (*here, *away)}
    own = {i: np.array([px(view[a]) for a in atoms]).reshape(-1, 2) for i, atoms in atoms_of.items()}
    body = {i: np.concatenate([own[i], *(openbook.along(px(view[a]), px(view[b]))
                                         for a, b in interact3d.bonds_of(cx, atoms))]) for i, atoms in atoms_of.items()}
    soft = np.concatenate([body[i] for i in here]) if here else np.zeros((0, 2))
    many = len({cx.residues[i].chain for i in (*here, *away)}) > 1
    words = {i: cx.residues[i].label + (f" {cx.residues[i].chain}" if many else "") for i in (*here, *away)}
    halo = 'stroke="#ffffff" stroke-width="3" stroke-opacity="0.9" paint-order="stroke" stroke-linejoin="round"'
    names, fixed = [], []
    for ids, hidden in ((here, False), (away, True)):
        more = [openbook.along(pa, pb) for i in ids for c, pa, pb in traces.get(i, ())] if hidden else []
        tags = [_Tag(own[i].mean(axis=0), len(words[i]) * size * 0.29 + 3.0, half_h, own[i], True, body[i])
                for i in ids]
        spots = settle(tags, np.concatenate([skeleton, *drawn_lines, *more]), bones, page, clear=4.0, soft=soft,
                       fixed=fixed, pad=3.0)
        for t, i, q in zip(tags, ids, spots):
            lead = openbook.hairline(q, t.hw, t.hh, own[i], edge=True)
            to = own[i][int(np.argmin(np.hypot(*(own[i] - q).T)))]
            fixed.append((q, t.hw, t.hh, (q, to)))
            name = (f'{lead}<text x="{q[0]:.1f}" y="{q[1] + size * 0.36:.1f}" font-size="{size}" font-weight="600" '
                    f'fill="{INK}" text-anchor="middle" {halo}>{words[i]}</text>')
            if hidden:
                name = f'<g class="reach">{name}</g>'
            r = cx.residues[i]
            kinds = ", ".join(sorted({pocket.KINDS[c.kind][0] for c in made if c.a == i}))
            names.append(f'<g class="res" data-id="{r.key}">{name}<title>{r.label} (chain {r.chain}): {kinds}; '
                         f'{r.buried:.0f} Å² covered</title></g>')
    return (f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {head + h:.0f}" width="{w}" '
            f'font-family="{openbook.FONT}">{openbook.heading(0, 14, "c", title)}<image class="ground" href="{src}" '
            f'x="0" y="{head}" width="{w}" height="{h}"/>' + "".join(bodies) + "".join(lines) + "".join(names)
            + "</svg>")


def skeleton_of(diagram: Diagram, at: np.ndarray) -> "tuple[str, str, np.ndarray, np.ndarray, np.ndarray]":
    """The ligand as a skeleton, its atoms at the places ``at``: every bond of the library's molecule, a
    double one as two lines, and a letter for every atom that is not carbon. Comes back as a pale band to
    lie under the contacts and the lines to lie over them, with the bonds as lines, with points along all
    of it, which nothing is to be put on, and with the boxes of the letters."""
    from rdkit import Chem                                   # the library's own dependency, for the double bonds

    mol = Chem.Mol(diagram.ligand.mol)
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:                                        # noqa: BLE001 - then the rings have single lines
        mol = diagram.ligand.mol
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    bones = np.array([[*at[b.GetBeginAtomIdx()], *at[b.GetEndAtomIdx()]] for b in mol.GetBonds()]).reshape(-1, 4)
    rings = [list(r) for r in mol.GetRingInfo().AtomRings()]
    band = " ".join(f"M {b[0]:.1f} {b[1]:.1f} L {b[2]:.1f} {b[3]:.1f}" for b in bones)
    under = (f'<path d="{band}" fill="none" stroke="{BAND}" stroke-width="9" stroke-opacity="0.55" '
             f'stroke-linecap="round" stroke-linejoin="round"/>')
    strokes = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        p, q = at[i], at[j]
        v = q - p
        long_ = float(np.hypot(*v))
        if long_ < 1e-6:
            continue
        u, off = v / long_, np.array([-v[1], v[0]]) / long_
        lo, hi = (6.2 if symbols[i] != "C" else 0.0), (6.2 if symbols[j] != "C" else 0.0)
        if long_ - lo - hi > 1.0:
            strokes.append((p + lo * u, q - hi * u))
        order = b.GetBondTypeAsDouble()
        if order >= 2.0 and long_ > 9.0:         # the second line: inside its ring, or on the side of more atoms
            ring = min((r for r in rings if i in r and j in r), key=len, default=None)
            near = [a.GetIdx() for a in (*mol.GetAtomWithIdx(i).GetNeighbors(), *mol.GetAtomWithIdx(j).GetNeighbors())
                    if a.GetIdx() not in (i, j)]
            others = ring if ring is not None else near
            mid = (p + q) / 2.0
            side = float(np.sign(sum(float((at[k] - mid) @ off) for k in others))) if others else 1.0
            side = side or 1.0
            trim = 2.6 if ring is not None or near else 0.0
            for way in ((side,) if order < 3.0 else (1.0, -1.0)):
                a, z = p + max(lo, trim) * u + 3.0 * way * off, q - max(hi, trim) * u + 3.0 * way * off
                if float((z - a) @ u) > 2.0:
                    strokes.append((a, z))
    d = " ".join(f"M {p[0]:.1f} {p[1]:.1f} L {q[0]:.1f} {q[1]:.1f}" for p, q in strokes)
    out = [f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="1.35" stroke-linecap="round" '
           f'stroke-linejoin="round"/>']
    points, letters = [openbook.along(b[:2], b[2:]) for b in bones], []
    for k, e in enumerate(symbols):
        if e != "C":
            colour = LIGHT.elements.get(e, LIGHT.elements["*"])
            out.append(openbook.text(at[k][0], at[k][1] + 3.6, e, 10.0, colour, "middle", "600",
                                     ' stroke="#ffffff" stroke-width="2.4" stroke-opacity="0.85" paint-order="stroke"'))
            points.append(at[k] + np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, -4.0], [0.0, 4.0]]))
            letters.append([at[k][0], at[k][1], 3.5 + 3.0 * len(e), 6.5])
    return under, "".join(out), bones, np.concatenate([at, *points]), np.array(letters).reshape(-1, 4)


def flat_map(diagram: Diagram, cx: book.Complex, view: np.ndarray, bounds, scale: float, colour: str, cut: bool, here,
             away, title: str) -> str:
    """Panel d. A box for every residue that makes a contact, at its place or, where the ligand or another
    box is, next to it; the ligand as its skeleton; a line for every contact from the ligand's atom (or
    the middle of its ring, or of its charged group) to the box, around every box but its own."""
    metric = openbook.metric_of(cx, colour)
    x0, x1, y0, y1 = bounds
    head = 30.0
    protein = cx.side == 0
    src, w, h, _ = openbook.picture(view, cx, openbook.atom_colours(cx, None), bounds, scale, which=protein, flat=1.0,
                                    fade=2.0, cut=0.0 if cut else None, hatched=False)

    def px(q) -> np.ndarray:
        return np.array([(q[0] - x0) * scale, head + (y1 - q[1]) * scale])

    one = cx.residues[-1]
    atoms = np.array([px(view[a]) for a in one.atoms])
    under, body, bones, solid, letters = skeleton_of(diagram, atoms)
    ids = [*here, *away]
    size = 10.0
    many = len({cx.residues[i].chain for i in ids}) > 1
    half = {i: openbook.mark_size(cx.residues[i], size, many, 5.0) for i in ids}
    # A line for each thing of the ligand that a residue is in contact with, and kind.
    lines: "dict[tuple[int, str, tuple[int, ...]], list[book.Contact]]" = {}
    for c in cx.contacts:
        if c.kind in pocket.KINDS and c.atoms_a and c.atoms_b:
            lines.setdefault((c.a, c.kind, tuple(sorted(c.atoms_b))), []).append(c)
    ends = {key: np.mean([px(view[a]) for a in key[2]], axis=0) for key in lines}
    tags = [_Tag(px(middle_of(cx.residues[i], view)), *half[i],
                 np.array([p for k, p in ends.items() if k[0] == i]).reshape(-1, 2)) for i in ids]
    spots = settle(tags, solid, bones, (3.0, head + 3.0, w - 3.0, head + h - 3.0), clear=11.0, letters=letters)
    at = {i: q for i, q in zip(ids, spots)}
    lanes = route.routers([(at[i][0], at[i][1], *half[i]) for i in ids])
    drawn, dots = [], []
    beside: "dict[tuple[int, int, int], int]" = {}             # lines from one place to one box lie beside each other
    order = sorted(lines, key=lambda key: (key[1] in pocket.LINED, ids.index(key[0]), key[1:]))
    for n, key in enumerate(order):
        made = lines[key]
        i, kind, group = key
        start, stroke, dash = ends[key], *pocket.LOOK[kind]
        same = (i, int(round(start[0])), int(round(start[1])))
        aim = at[i] + np.array([7.0 * beside.get(same, 0), 0.0])
        beside[same] = beside.get(same, 0) + 1
        way = None
        for lane in (lanes[n % len(lanes)], lanes[0]):
            way = lane.way(start, aim, own=(ids.index(i),))
            if way is not None:
                break
        way = [start, aim] if way is None else way
        way[-1] = route.leave(at[i], way[-2], *half[i], False) if not beside[same] - 1 else aim
        off = way[1] - way[0]
        far = float(np.hypot(*off))
        single = len(group) == 1
        letter = single and cx.elements[group[0]].upper() != "C"
        back = 6.4 if letter else (3.0 if single else 0.0)
        if far > back + 2.0:
            way[0] = way[0] + back / far * off
        d = route.path(way)
        said = "; ".join(says(cx, c) for c in made)
        link = f'class="link" data-a="{cx.residues[i].key}" data-b="{one.key}"'
        drawn.append(f'<path {link} d="{d}" fill="none" stroke="#ffffff" stroke-opacity="0.7" stroke-width="3.2"/>'
                     f'<path {link} d="{d}" fill="none" stroke="{stroke}" stroke-width="{WIDTHS.get(kind, 1.5)}"'
                     + (f' stroke-dasharray="{dash}"' if dash else "") + f'><title>{said}</title></path>')
        if not single:
            dots.append(f'<g {link}><circle cx="{start[0]:.1f}" cy="{start[1]:.1f}" r="2.6" fill="{stroke}" '
                        f'stroke="#ffffff" stroke-width="0.9"/></g>')
    marks = []
    for i in ids:
        r = cx.residues[i]
        fill = metric.of(r)
        kinds = ", ".join(sorted({pocket.KINDS[k[1]][0] for k in lines if k[0] == i}))
        where = "" if i in here else "; on the half that panel c does not show"
        marks.append(openbook.mark(r, at[i], *half[i], np.ones(3) if fill is None else fill, size,
                                   f"{r.label} (chain {r.chain}): {kinds}; {r.buried:.0f} Å² "
                                   f"covered{where}", chain=many,
                                   dash="" if i in here else "3.2 2.2"))
    ground = f'<image href="{src}" x="0" y="{head}" width="{w}" height="{h}" opacity="0.5"/>'
    return (f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {head + h:.0f}" width="{w}" '
            f'font-family="{openbook.FONT}">{openbook.heading(0, 14, "d", title)}{ground}'
            f'<g class="res" data-id="{one.key}">{under}<title>{html.escape(one.label)}</title></g>'
            f'<g class="links">{"".join(drawn)}</g><g class="res" data-id="{one.key}">{body}'
            f'<title>{html.escape(one.label)}</title></g>{"".join(dots)}{"".join(marks)}</svg>')


# ---------------------------------------------------------------------------------------- the figure
def key_of(cx: book.Complex, colour: str, cut: bool, away: bool, of: int, y: float, width: float
           ) -> "tuple[str, float]":
    """The key under the panels, from ``y`` on; comes back with where it ends. ``of`` is how many residues
    the model has."""
    metric = openbook.metric_of(cx, colour)
    out, x = metric.key(0, y)
    out = out.replace("the partner", "the ligand")
    left = max(x + 8.0, 176.0)
    x = left
    row = y + 9.0
    present = [k for k in pocket.KINDS if any(c.kind == k for c in cx.contacts)]
    for kind in present:
        label, dash = pocket.KINDS[kind]
        if kind not in pocket.LINED:
            label += " (d only)"
        need = 28.0 + 4.9 * len(label) + 14.0
        if x + need > width + 14.0:
            x, row = left, row + 17.0
        out += (f'<line x1="{x:.1f}" y1="{row - 3:.1f}" x2="{x + 22:.1f}" y2="{row - 3:.1f}" '
                f'stroke="{pocket.colour_of(kind)}" '
                f'stroke-width="1.5"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>"
                + openbook.text(x + 28, row, label, 9.5, QUIET))
        x += need
    row = max(y + 40.0, row + 24.0)
    names = {"N": "nitrogen blue", "O": "oxygen red", "S": "sulfur yellow", "P": "phosphorus orange",
             "F": "fluorine green", "CL": "chlorine green", "BR": "bromine brown", "I": "iodine purple"}
    held = {e.upper() for e, s in zip(cx.elements, cx.side) if s == 1}
    others = ", ".join(v for k, v in names.items() if k in held)
    b_name = html.escape(cx.names[1])
    said = f"atoms of {b_name} (carbon{'; ' + others if others else ''}); pale: behind the protein"
    out += (f'<line x1="3" y1="{row - 3:.1f}" x2="21" y2="{row - 3:.1f}" '
            f'stroke="{openbook.as_hex(np.array(openbook.CARBON))}" '
            f'stroke-width="4.5" stroke-linecap="round"/>' + openbook.text(28, row, said, 9.5, QUIET))
    x = 28.0 + 4.9 * len(said) + 22.0
    more: "list[tuple[str, str] | None]" = []
    if cut:
        more.append((f'<rect x="{{x}}" y="{{y}}" width="24" height="12" rx="1.5" fill="#dfe1e6" stroke="{INK}" '
                     f'stroke-width="0.6"/><path d="M {{x1}} {{y2}} l 9 -9 M {{x2}} {{y2}} l 9 -9" stroke="#a8b0ba" '
                     f'stroke-width="1.4"/>', "the protein, cut through"))
    if not cut and colour == "area":
        more.append(("", f"b: {b_name} by atom, 0 to {lost_by_atom(cx)[1]:.0f} Å²"))
    more.append(None)                                                     # a new row
    more.append((f'<rect x="{{x}}" y="{{y}}" width="24" height="12" rx="1.5" fill="#fff" stroke="{openbook.EDGE[0]}" '
                 f'stroke-width="0.9"/>', "a residue that makes a contact"))
    if away:
        more.append((f'<rect x="{{x}}" y="{{y}}" width="24" height="12" rx="1.5" fill="#fff" '
                     f'stroke="{openbook.EDGE[0]}" '
                     f'stroke-width="1.1" stroke-dasharray="3.2 2.2"/>', "one on the half that c does not show"))
    for entry in more:
        if entry is None:
            x, row = 0.8, row + 20.0
            continue
        drawing, label = entry
        need = (30.0 if drawing else 0.0) + 5.0 * len(label) + 20.0
        if x + need > width + 20.0:
            x, row = 0.8, row + 20.0
        out += (drawing.format(x=f"{x:.1f}", y=f"{row - 9:.1f}", x1=f"{x + 3:.1f}", x2=f"{x + 12:.1f}",
                               y2=f"{row + 1.5:.1f}")
                + openbook.text(x + (30 if drawing else 0), row, label, 9.5, QUIET))
        x += need
    row += 17.0
    out += openbook.text(0, row, "Pointing at a residue in c or d marks it in the other, with its atoms and every "
                                 "contact it makes.", 9.5, QUIET)
    drawn = sum(1 for r in cx.residues if r.side == 0)
    if drawn < of:
        row += 15.0
        out += openbook.text(0, row, f"Drawn are {drawn} of the model's {of} residues: those of a pLDDT of {SURE:g} or "
                                     "more that hang together with the pocket, and the pocket's own.", 9.5, QUIET)
    return out, row + 6.0


def panels(diagram: Diagram, colour: str = "area", cut: "bool | None" = None) -> openbook.Sketch:
    """Four panels for a ligand: **a** the complex as it stands, **b** opened, **c** the face that holds
    the ligand, large, with the ligand as licorice and its contacts from atom to atom, **d** the same
    frame as a flat map with every residue that makes a contact.

    ``diagram`` is a bindview ``Diagram`` made from a structure. ``colour`` is what the residues of the
    pocket are coloured by: ``"area"`` (what the ligand covers of a residue), ``"class"`` (the kind of
    residue), or ``"plddt"`` for a model that states it. ``cut`` overrides the rule by which the protein is cut
    along the ligand's plane or not (the docstring of the file says it). Where it is cut, panels c and d
    are of the half that holds more of the contacts (of the two, if they hold as many, the one with more
    residues that make one, then the first); a residue of the other half has a dashed outline in d.
    A residue pointed at in c or d is marked in both, by a style sheet in the figure, and so is one that
    the page around the figure asks for (``data-active="712"`` on its root).
    """
    openbook._one_of(colour, COLOURS, "colour")
    full = book.load_ligand(diagram, ("the protein", diagram.name or "the ligand"))
    _, exposed = pocket.open_side(full)
    across, halves = harder.page(full, 0).seen, harder.seen_cut(full, 0)
    by_rule = across < halves
    cut = by_rule if cut is None else bool(cut)
    cx = piece(full)
    shut = cx.closed()
    making = sorted({c.a for c in cx.contacts if c.kind in pocket.KINDS})
    below = {i: float(middle_of(cx.residues[i], shut)[0]) <= 0.0 for i in making}
    made = [c for c in cx.contacts if c.kind in pocket.KINDS]
    score = {side: (sum(1 for c in made if below[c.a] == side), sum(1 for i in making if below[i] == side))
             for side in (True, False)}
    low = not cut or score[True] >= score[False]
    here = [i for i in making if not cut or below[i] == low]
    away = [i for i in making if i not in here]

    # The frame of panels c and d: the pocket and the ligand, with room around them.
    view = turned(cx, low)
    ligand = cx.side == 1
    seen = [middle_of(cx.residues[i], view)[:2] for i in making]
    seen += [view[list(c.atoms_a)].mean(axis=0)[:2] for c in made if c.atoms_a]
    seen += [view[a][:2] for i in here for a in openbook.drawn_atoms(cx, i, pocket.LOOK)]
    spots = np.concatenate([view[ligand][:, :2] - AROUND, view[ligand][:, :2] + AROUND,
                            np.array(seen).reshape(-1, 2) - BESIDE, np.array(seen).reshape(-1, 2) + BESIDE])
    x0, y0 = spots.min(axis=0)
    x1, y1 = spots.max(axis=0)
    each = (ROOM - GAP) / 2.0
    scale = min(LARGEST, each / (x1 - x0), TALLEST / (y1 - y0))
    more = each / scale - (x1 - x0)                           # a narrow pocket gets the whole width of its panel
    x0, x1 = x0 - more / 2.0, x1 + more / 2.0
    bounds = (float(x0), float(x1), float(y0), float(y1))

    who = html.escape(cx.names[1]) if len(cx.names[1]) <= 14 else "the ligand"
    first = whole(cx, colour, cut, low, who, bounds)
    title = f"one half of the pocket, with {who} in it" if cut else f"the face of the protein, with {who} on it"
    third = close_up(cx, view, bounds, scale, colour, cut, here, away, title)
    fourth = flat_map(diagram, cx, view, bounds, scale, colour, cut, here, away,
                      "the same frame as a flat map: every residue in contact")
    (w1, h1), (w3, h3), (w4, h4) = (openbook.size_of(s) for s in (first, third, fourth))
    width = max(w1, w3 + GAP + w4)
    y2 = h1 + 12.0
    y3 = y2 + max(h3, h4) + 34.0
    n_drawn, n_all = (sum(1 for r in one.residues if r.side == 0) for one in (cx, full))
    key, bottom = key_of(cx, colour, cut, bool(away), n_all, y3, width)
    right = w3 + GAP + w4 - 2.0
    bar = (f'<line x1="{right - 10 * scale:.1f}" y1="{y3 - 20:.1f}" x2="{right:.1f}" y2="{y3 - 20:.1f}" stroke="{INK}" '
           f'stroke-width="1.2"/>' + openbook.text(right, y3 - 24, "10 Å", 9.5, QUIET, "end"))
    svg = (f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {bottom:.0f}" '
           f'width="{width:.0f}" font-family="{openbook.FONT}">' + openbook.nest(first, 0, 0)
           + openbook.nest(third, 0, y2) + openbook.nest(fourth, w3 + GAP, y2) + bar + key + "</svg>")
    one, those = cx.residues[-1], [cx.residues[i] for i in making]
    which = hashlib.sha1(json.dumps([cx.names[1], sorted(r.key for r in those)]).encode()).hexdigest()[:8]
    svg = openbook.answering(openbook.numbered(svg, those), [*those, one],
                             {one.key: {r.key for r in those}, **{r.key: {one.key} for r in those}}, f"lb-{which}")
    facts = {"names": list(cx.names), "pocket": len(cx.at_interface(0)), "named": len(making), "named_in_c": len(here),
             "contacts": len(made), "kinds": {k: sum(1 for c in made if c.kind == k) for k in pocket.KINDS
                                              if any(c.kind == k for c in made)},
             "free": round(exposed, 3), "seen_across": round(across, 2), "seen_cut": round(halves, 2), "cut": cut,
             "drawn": n_drawn, "of": n_all}
    where = (f", {len(here)} of them on the half that panel c shows" if cut else
             ", all of them on the face that panel c shows")
    why = (f"Cut, because a look from the ligand's side would show {100 * across:.0f} % of what the ligand covers on "
           f"the protein and the two halves show {100 * halves:.0f} %." if cut and by_rule else
           f"Not cut, because a look from the ligand's side shows {100 * across:.0f} % of what the ligand covers on "
           f"the protein, and two halves would show {100 * halves:.0f} %." if not cut and not by_rule else
           f"{'Cut' if cut else 'Not cut'} because it was asked for.")
    who_makes = (f"{len(making)} residue{'s make' if len(making) != 1 else ' makes'} the {len(made)} contact"
                 f"{'s' if len(made) != 1 else ''} of the diagram{where}" if making else
                 "No residue makes a contact of the diagram")
    facts["said"] = f"{who_makes}; {100 * (1 - exposed):.0f} % of the surface of {cx.names[1]} is buried. {why}"
    return openbook.Sketch(svg, "four panels of a ligand", facts)
