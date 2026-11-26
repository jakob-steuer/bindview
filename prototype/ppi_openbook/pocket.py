#!/usr/bin/env python
"""A small molecule in its pocket, in three dimensions beside its diagram.

A SKETCH, not part of bindview: nothing here is tested or promised.

    python prototype/ppi_openbook/pocket.py        # writes prototype/ppi_openbook/pocket.html

``several(diagrams)`` is for the ligands of one pocket: under the library's comparison of them, the pocket
with all of them in it, and a row for each ligand in its own model's pocket, every row cut on one common
plane and seen from one place.

``figure(diagram)`` is the library's own diagram of a ligand with three panels under it. **a** the pocket
from its most open side, the ligand as licorice: where the protein covers a stick, the stick shows
through, pale. **b**, **c** the pocket cut along the ligand's own plane and opened like a book: each half
is seen from the cut, which is drawn flat and hatched, and the ligand lies on both, on the second as seen
from its other face. A residue is named on the half it is in, and a contact that has a direction runs
from atom to atom in the colour the diagram gives it.

The contacts are the diagram's own (``Diagram.interactions``, resolved to atoms by the library), so the
panels and the diagram over them say the same. A residue pointed at in the diagram is marked in the
panels and the other way round: a style sheet in the figure does that, without a script.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import book  # noqa: E402
import harder  # noqa: E402
import openbook  # noqa: E402
from bindview import Comparison, Diagram, load_interactions  # noqa: E402
from bindview.align import pocket_transforms  # noqa: E402
from bindview.theme import LIGHT  # noqa: E402
from surface import PROBE, radii  # noqa: E402

#: The contacts of a diagram, strongest first, with what a key calls them and how their line is dashed.
KINDS = {"metal": ("metal", ""), "saltbridge": ("salt bridge", ""), "pi_cation": ("π–cation", "7 2.2 1.6 2.2"),
         "hbond": ("hydrogen bond", "4.2 2.6"), "water_bridge": ("water bridge", "2 2.2"),
         "halogen": ("halogen bond", "6.5 2.2 1.3 2.2"), "pi_stacking": ("π–π stacking", ""),
         "hydrophobic": ("hydrophobic contact", "1.5 2.5")}
LINED = tuple(k for k in KINDS if k != "hydrophobic")          # drawn as a line from atom to atom
STYLES = ("detailed", "schematic")                              # of the diagram over the panels
COLOURS = ("area", "class", "contacts", "plddt")               # of the pocket in the panels (pLDDT: a model's)
OPEN = 0.03          # a ligand with less than this share of its surface free has no open side to look from
#: The carbons of the ligands of a comparison, a colour each.
CARBONS = ((0.93, 0.62, 0.20), (0.16, 0.60, 0.62), (0.58, 0.40, 0.78), (0.38, 0.66, 0.30), (0.85, 0.42, 0.60),
           (0.50, 0.50, 0.55))


def colour_of(kind: str) -> str:
    return LIGHT.tokens()[kind]


#: A contact pointed at: the colour and the dashes of its line, by its kind.
LOOK = {kind: (colour_of(kind), dash) for kind, (_, dash) in KINDS.items()}


def open_side(cx: book.Complex) -> "tuple[np.ndarray | None, float]":
    """The direction the ligand is open to, and the share of its surface that is free in the pocket.

    The direction is the mean of the outward directions of what stays free of the ligand's surface (the
    points of the area calculation). ``None`` where next to nothing stays free: an enclosed ligand.
    """
    reach = radii(cx.elements) + PROBE
    alone, everything = cx.side == 1, np.ones(len(reach), dtype=bool)
    sphere = book._sphere(96)
    free_alone, free_here, towards = 0.0, 0.0, np.zeros(3)
    for i in np.where(alone)[0]:
        p = cx.xyz[i] + reach[i] * sphere
        own = harder._free(p, i, cx.xyz, reach, alone)
        here = own & harder._free(p, i, cx.xyz, reach, everything)
        share = 4.0 * np.pi * reach[i] ** 2 / len(sphere)
        free_alone += share * own.sum()
        free_here += share * here.sum()
        towards += share * sphere[here].sum(axis=0)
    exposed = free_here / max(free_alone, 1e-9)
    if exposed < OPEN or np.linalg.norm(towards) < 1e-6:
        return None, exposed
    return towards / np.linalg.norm(towards), exposed


def seen_from(cx: book.Complex, towards: np.ndarray) -> float:
    """The share of what the ligand covers on the protein that is seen from that side, uncut."""
    return harder.page(cx, 0, towards=towards).seen


def _names(cx: book.Complex, view: np.ndarray, mine, px, box, size: float = 9.5, fills=None, residues: bool = False):
    """Marks for these residues of the protein, each beside its residue and not on it: off the atoms
    of its residue and off its contacts, which appear under it when it is pointed at, off the ligand,
    and off the other marks (:func:`openbook.beside`); a hairline leads from a mark to the nearest atom
    of its residue. ``fills`` gives a residue (by its index) the colour of its mark; without it a mark
    is white. With every name go the atoms of its residue and all its contacts, drawn when it is
    pointed at (:func:`openbook.under`); with ``residues`` the atoms are there all the time, faintly."""
    half_w = np.array([len(cx.residues[i].label) * size * 0.29 + 4.0 for i in mine])
    half_h = size * 0.5 + 3.0
    want = np.array([px(view[cx.residues[i].near][:, :2].mean(axis=0)) for i in mine]).reshape(-1, 2)
    ligand = cx.side == 1
    lig_points = openbook.licorice_points(cx, ligand, view, px)
    lig_at = lig_points[:int(ligand.sum())]
    middle = lig_at.mean(axis=0) if len(lig_at) else want.mean(axis=0)
    drawn, own = zip(*[openbook.own_points(cx, i, view, px, LOOK) for i in mine]) if mine else ((), ())
    pos = openbook.beside(want, half_w, half_h, list(own), lig_points, box, middle)
    out = []
    for n, i in enumerate(mine):
        r, q = cx.residues[i], pos[n]
        lead = openbook.hairline(q, half_w[n], half_h, drawn[n] if len(drawn[n]) else want[n][None, :])
        made = sorted({KINDS[c.kind][0] for c in cx.contacts if c.a == i and c.kind in KINDS})
        out.append(openbook.mark(r, q, half_w[n], half_h, np.ones(3) if fills is None else fills.get(i, np.ones(3)),
                                 size, f"{r.label} (chain {r.chain}): {', '.join(made) or 'atoms within 4 Å'}; "
                                 f"{r.buried:.0f} Å² covered", opacity="0.86" if fills is None else "1", lead=lead,
                                 extra=openbook.under(cx, i, view, px, LOOK, residues)))
    return out


def _half(cx: book.Complex, low: bool, bounds, scale: float, colours: np.ndarray, sticks_in, origin: float, top: float,
          words: str, shown, fills=None, lines: bool = True, residues: bool = False) -> "tuple[list[str], int, float]":
    """One half of a pocket that is cut on the plane of ``cx``, at a place in a larger figure: the
    picture, the contacts that have a direction as lines from atom to atom, and the names. ``low``
    says which half; ``sticks_in`` gives the licorice in the frame it is handed. Comes back with its
    width and height."""
    shut = cx.closed()
    view = (np.stack([-shut[:, 2], shut[:, 1], shut[:, 0]], axis=1) if low
            else np.stack([shut[:, 2], shut[:, 1], -shut[:, 0]], axis=1))
    x0, x1, y0, y1 = bounds
    box = (x0, x1, y0, y1) if low else (-x1, -x0, y0, y1)
    head = 22.0
    src, w, h, _ = openbook.picture(view, cx, colours, box, scale, which=cx.side == 0, fade=2.0,
                                    licorice=sticks_in(view), cut=0.0)

    def px(q) -> np.ndarray:
        return np.array([origin + (q[0] - box[0]) * scale, top + head + (box[3] - q[1]) * scale])

    parts = [openbook.text(origin, top + 13, words, 10.5, openbook.QUIET),
             f'<image href="{src}" x="{origin:.1f}" y="{top + head:.1f}" width="{w}" height="{h}"/>']
    below = {i: float(shut[cx.residues[i].near, 0].mean()) <= 0.0 for i in cx.at_interface(0)}
    for c in cx.contacts if lines else ():
        if c.kind not in KINDS or below.get(c.a) != low:
            continue
        a_key, b_key = cx.residues[c.a].key, cx.residues[c.b].key
        if c.kind not in LINED or not c.atoms_a or not c.atoms_b:
            parts.append(f'<path class="link" data-a="{a_key}" data-b="{b_key}" d="M 0 0" stroke="none"/>')
            continue
        pa, pb = px(view[list(c.atoms_a)].mean(axis=0)), px(view[list(c.atoms_b)].mean(axis=0))
        stroke, dash = colour_of(c.kind), KINDS[c.kind][1]
        parts.append(
            f'<line x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" stroke="#ffffff" '
            f'stroke-opacity="0.8" stroke-width="3.4" stroke-linecap="round"/>'
            f'<line class="link" data-a="{a_key}" data-b="{b_key}" x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" '
            f'x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" stroke="{stroke}" stroke-width="1.5" '
            f'stroke-dasharray="{dash or "none"}"><title>{cx.residues[c.a].label}: {KINDS[c.kind][0]}, '
            f'{c.distance:.1f} Å</title></line>'
            f'<circle cx="{pa[0]:.1f}" cy="{pa[1]:.1f}" r="2.4" fill="{stroke}" stroke="#ffffff" stroke-width="0.9"/>')
    parts += _names(cx, view, [i for i in shown if below.get(i) == low], px,
                    (origin + 2, top + head + 2, origin + w - 2, top + head + h - 2), fills=fills, residues=residues)
    return parts, w, h + head


def panels(cx: book.Complex, colour: str = "area", residues: bool = False) -> openbook.Sketch:
    """The three panels for a ligand loaded with :func:`book.load_ligand`."""
    metric = openbook.metric_of(cx, openbook._one_of(colour, COLOURS, "colour"))
    colours = openbook.atom_colours(cx, metric)
    protein, ligand = cx.side == 0, cx.side == 1
    shut = cx.closed()
    lower = np.stack([-shut[:, 2], shut[:, 1], shut[:, 0]], axis=1)
    upper = np.stack([shut[:, 2], shut[:, 1], -shut[:, 0]], axis=1)
    shown = [i for i in cx.at_interface(0) if any(c.a == i and c.kind in KINDS for c in cx.contacts)]
    spots = np.array([lower[cx.residues[i].near][:, :2].mean(axis=0) for i in cx.at_interface(0)]
                     + [lower[ligand][:, :2].min(axis=0), lower[ligand][:, :2].max(axis=0)])
    margin, gap, head = 6.0, 22.0, 30.0
    x0, x1 = spots[:, 0].min() - margin, spots[:, 0].max() + margin
    y0, y1 = spots[:, 1].min() - margin, spots[:, 1].max() + margin
    scale = min(13.0, (940.0 - 2.0 * gap) / 3.0 / (x1 - x0))      # three panels side by side on the page
    elements = [e for e, k in zip(cx.elements, ligand) if k]
    below = {i: float(shut[cx.residues[i].near, 0].mean()) <= 0.0 for i in cx.at_interface(0)}
    a_name, b_name = html.escape(cx.names[0]), html.escape(cx.names[1])
    parts, origin = [], 0.0

    # a: from the open side, uncut. An enclosed ligand has none: it is then looked at along the cut's normal.
    towards, exposed = open_side(cx)
    said = "from its most open side" if towards is not None else "no side is open: seen across the plane of the cut"
    z = towards if towards is not None else cx.axes[0]
    up = cx.axes[1] - (cx.axes[1] @ z) * z
    if np.linalg.norm(up) < 0.3:
        up = cx.axes[2] - (cx.axes[2] @ z) * z
    up /= np.linalg.norm(up)
    front = (cx.xyz - cx.xyz[ligand].mean(axis=0)) @ np.stack([np.cross(up, z), up, z]).T
    half_w, half_h = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    src, w, h, _ = openbook.picture(front, cx, colours, (-half_w, half_w, -half_h, half_h), scale, which=protein, fade=2.0,
                                    licorice=openbook.surface.sticks(front[ligand], elements, openbook.CARBON))
    parts.append(openbook.heading(origin, 14, "a", said))
    parts.append(f'<image href="{src}" x="{origin:.1f}" y="{head}" width="{w}" height="{h}"><title>{b_name} in '
                 f'{a_name}: {100 * exposed:.0f} % of its surface is free</title></image>')
    origin += w + gap

    for view, bounds, low, words, letter in (
            (lower, (x0, x1, y0, y1), True, "cut along the ligand's plane: one half, from the cut", "b"),
            (upper, (-x1, -x0, y0, y1), False, "the other half (the page is turned over)", "c")):
        src, w, h, _ = openbook.picture(view, cx, colours, bounds, scale, which=protein, fade=2.0,
                                        licorice=openbook.surface.sticks(view[ligand], elements, openbook.CARBON), cut=0.0)

        def px(q, bounds=bounds, origin=origin) -> np.ndarray:
            return np.array([origin + (q[0] - bounds[0]) * scale, head + (bounds[3] - q[1]) * scale])

        parts.append(openbook.heading(origin, 14, letter, words))
        parts.append(f'<image href="{src}" x="{origin:.1f}" y="{head}" width="{w}" height="{h}"/>')
        for c in cx.contacts:
            if c.kind not in KINDS or below.get(c.a) != low:
                continue
            a_key, b_key = cx.residues[c.a].key, cx.residues[c.b].key
            if c.kind not in LINED or not c.atoms_a or not c.atoms_b:
                parts.append(f'<path class="link" data-a="{a_key}" data-b="{b_key}" d="M 0 0" stroke="none"/>')
                continue
            pa, pb = px(view[list(c.atoms_a)].mean(axis=0)), px(view[list(c.atoms_b)].mean(axis=0))
            stroke, dash = colour_of(c.kind), KINDS[c.kind][1]
            parts.append(
                f'<line x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" stroke="#ffffff" '
                f'stroke-opacity="0.8" stroke-width="3.4" stroke-linecap="round"/>'
                f'<line class="link" data-a="{a_key}" data-b="{b_key}" x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" '
                f'x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" stroke="{stroke}" stroke-width="1.5" '
                f'stroke-dasharray="{dash or "none"}"><title>{cx.residues[c.a].label}: {KINDS[c.kind][0]}, '
                f'{c.distance:.1f} Å</title></line>'
                f'<circle cx="{pa[0]:.1f}" cy="{pa[1]:.1f}" r="2.4" fill="{stroke}" stroke="#ffffff" stroke-width="0.9"/>')
        parts += _names(cx, view, [i for i in shown if below[i] == low], px,
                        (origin + 2, head + 2, origin + w - 2, head + h - 2), residues=residues)
        origin += w + gap

    width = origin - gap
    height = head + int(round((y1 - y0) * scale)) + 88.0
    key, x = metric.key(0, height - 62)
    present = [k for k in KINDS if any(c.kind == k for c in cx.contacts)]
    for kind in present:
        label, dash = KINDS[kind]
        x += 18
        d = f' stroke-dasharray="{dash}"' if dash else ""
        if kind in LINED:
            key += f'<line x1="{x}" y1="{height - 56}" x2="{x + 22}" y2="{height - 56}" stroke="{colour_of(kind)}" stroke-width="1.5"{d}/>'
            key += openbook.text(x + 28, height - 53, label, 9.5, openbook.QUIET)
            x += 28 + 5.3 * len(label)
        else:
            key += openbook.text(x, height - 53, f"{label}: a name, no line", 9.5, openbook.QUIET)
            x += 5.3 * (len(label) + 16)
    carbon = openbook.as_hex(np.array(openbook.CARBON))
    key += (f'<line x1="0" y1="{height - 24}" x2="22" y2="{height - 24}" stroke="{carbon}" stroke-width="4.5" '
            f'stroke-linecap="round"/>'
            + openbook.text(28, height - 21, f"atoms of {b_name} (carbon; nitrogen blue, oxygen red, sulfur yellow); pale: "
                                             f"behind the protein; hatched: where the protein is cut through", 9.5,
                            openbook.QUIET)
            + f'<line x1="0" y1="{height - 6}" x2="22" y2="{height - 6}" stroke="{openbook.STICK}" stroke-width="2.2" '
              f'stroke-linecap="round"/>'
            + openbook.text(28, height - 3, ("atoms of a residue that makes a contact, faint; " if residues else "")
                            + "pointing at a name shows its residue's atoms and every contact it makes, from atom "
                              "to atom", 9.5, openbook.QUIET))
    bar = (f'<line x1="{width - 10 * scale:.1f}" y1="{height - 70}" x2="{width:.1f}" y2="{height - 70}" '
           f'stroke="{openbook.INK}" stroke-width="1.2"/>'
           + openbook.text(width, height - 74, "10 Å", 9.5, openbook.QUIET, "end"))
    svg = (f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
           f'width="{width:.0f}" font-family="{openbook.FONT}">' + "".join(parts) + key + bar + "</svg>")
    across = harder.page(cx, 0)
    facts = {"names": list(cx.names), "pocket": len(cx.at_interface(0)), "named": len(shown),
             "free": round(exposed, 3), "open_side": towards is not None,
             "seen_open_side": round(seen_from(cx, towards), 2) if towards is not None else None,
             "seen_across": round(across.seen, 2), "seen_cut": round(harder.seen_cut(cx, 0), 2),
             "kinds": {k: sum(1 for c in cx.contacts if c.kind == k) for k in present}}
    where = (f"from its most open side {100 * facts['seen_open_side']:.0f} %" if towards is not None
             else "it has no open side to look from")
    facts["said"] = (f"{100 * (1 - exposed):.0f} % of the surface of {cx.names[1]} is covered. Of what it covers on "
                     f"the protein, a look across the plane shows {100 * facts['seen_across']:.0f} %, {where}, and "
                     f"the two halves of the cut together {100 * facts['seen_cut']:.0f} %. {len(shown)} of the "
                     f"{facts['pocket']} residues at the pocket make a contact of the diagram and are named.")
    return openbook.Sketch(svg, "a pocket, cut and opened", facts)


def figure(diagram: Diagram, style: str = "detailed", colour: str = "area", residues: bool = False,
           effort: str = "full") -> openbook.Sketch:
    """A ligand's diagram as the library draws it, and its pocket in three dimensions under it.

    ``diagram`` is a ``bindview.Diagram`` made from a structure. ``style`` is that of the diagram on top
    (``"detailed"``, ``"schematic"``). ``colour`` is what the pocket is coloured by in the panels:
    ``"area"`` (what the ligand covers of a residue), ``"class"`` (the kind of residue), ``"contacts"``,
    or ``"plddt"`` for a model that states it. ``residues`` draws the residues that make a contact
    into the two halves, faintly, as thin sticks: the side chain, and the backbone where it is the
    backbone that makes the contact. A residue pointed at in the diagram is marked in the panels and
    the other way round, by a style sheet in the figure, and whichever way it is pointed at, its
    atoms and every contact it makes are then drawn in full. The page around the figure can do the same,
    with ``data-active="712"`` on its root. ``effort`` is the diagram's (``"quick"``: one that is too
    crowded is given up early).
    """
    openbook._one_of(style, STYLES, "style")
    cx = book.load_ligand(diagram, ("the protein", diagram.name or "the ligand"))
    under = panels(cx, colour, residues)
    flat = diagram.render(style=style, interactive=True, effort=effort)
    w2, h2 = openbook.size_of(under.svg)
    parts, top = [], 0.0
    if flat.svg is not None:
        m = re.match(r'<svg ([^>]*?)width="([0-9.]+)" height="([0-9.]+)"', flat.svg)
        assert m, "a figure of the library without a size"
        w1, h1 = float(m.group(2)), float(m.group(3))
        k = min(1.0, w2 / w1)                                    # never wider than the panels under it
        parts.append(flat.svg.replace(m.group(0), f'<svg x="{(w2 - w1 * k) / 2.0:.1f}" y="0" {m.group(1)}'
                                                  f'width="{w1 * k:.1f}" height="{h1 * k:.1f}"', 1))
        top = h1 * k + 14.0
    parts.append(openbook.nest(openbook.numbered(under.svg, cx.residues), 0, top))
    # The diagram and the panels answer each other: what is pointed at (or kept) in one steps the rest of
    # the other back. A rule for each residue of the pocket, in the scope of this one figure.
    names = {f"{r.chain}:{r.name}{r.number}{r.icode}": r.key for r in cx.residues if r.side == 0 and len(r.near)}
    scope = "svg.pk-" + hashlib.sha1(json.dumps([cx.names[1], sorted(names)]).encode()).hexdigest()[:8]
    svg = (f'<svg class="book pocket {scope[4:]}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w2:.0f} '
           f'{top + h2:.0f}" width="{w2:.0f}" font-family="{openbook.FONT}"><style>{openbook.tied(scope, names, True)}</style>'
           + "".join(parts) + "</svg>")
    facts = dict(under.facts, style=style, diagram=flat.status)
    return openbook.Sketch(svg, "a pocket in three dimensions", facts, warnings=list(flat.warnings))




def _together(first: book.Complex, others) -> book.Complex:
    """The protein of the first model with the ligands of all of them as one piece on it."""
    protein = first.side == 0
    n = int(protein.sum())
    clouds = [first.xyz[first.side == 1]] + [cx.xyz[cx.side == 1] for cx in others]
    elements = list(np.array(first.elements)[protein]) + [e for cx in [first, *others]
                                                           for e, s in zip(cx.elements, cx.side) if s == 1]
    xyz = np.vstack([first.xyz[protein], *clouds])
    residues = [book.Residue(0, r.chain, r.number, r.icode, r.name, r.atoms.copy(), index=r.index, ca=r.ca)
                for r in first.residues if r.side == 0]
    piece = book.Whole(1, "", 0, "", "ALL", np.arange(n, len(xyz)), index=len(residues), shown="the ligands")
    side = np.array([0] * n + [1] * (len(xyz) - n))
    of = np.concatenate([first.residue_of[protein], np.full(len(xyz) - n, len(residues))])
    named = [a for a, s in zip(first.atoms, first.side) if s == 0] + [a for cx in [first, *others]
                                                                    for a, s in zip(cx.atoms, cx.side) if s == 1]
    cx = book.Complex("", ("the protein", "the ligands"), xyz, elements, side, of, residues + [piece], [], atoms=named)
    book._complete(cx, [], several=True, plane=side == 1)
    return cx


def several(diagrams, comparison: bool = True, residues: bool = False) -> openbook.Sketch:
    """Several ligands of one pocket, each in the pocket cut along one common plane.

    ``diagrams`` are bindview ``Diagram`` objects of ligands in the same pocket, as a ``Comparison`` takes
    them. Their models are superposed as the library's comparison superposes them (the backbone of the
    residues around the ligands), and one plane is laid through all the ligands together. The first row
    is the pocket of the first model with every ligand in it, each in a colour of its own, and the
    pocket coloured by how many of the ligands a residue is in contact with. Then a row for each
    ligand: its own model's pocket, cut on the same plane and seen from the same place, so the same
    spot on two rows is the same place in the pocket. A name's colour says with how many of the
    ligands the residue is in contact. With ``comparison`` the library's own figure of the set stands
    on top, and answers the rows as the diagram of one ligand answers its panels. ``residues`` draws
    the residues that make a contact into every half, faintly, as in :func:`figure`.
    """
    diagrams = list(diagrams)
    n = len(diagrams)
    if n < 2:
        raise ValueError("several ligands are two or more")
    if n > len(CARBONS):
        raise ValueError(f"{n} ligands: the sketch has a colour for {len(CARBONS)}")
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
    all_ = _together(each[0], each[1:])
    for cx in each:                                           # one plane, through all the ligands
        cx.centre, cx.axes = all_.centre, all_.axes

    def touched(cx: book.Complex) -> "set[tuple[str, int, str]]":
        return {(cx.residues[c.a].chain, cx.residues[c.a].number, cx.residues[c.a].icode)
                for c in cx.contacts if c.kind in KINDS}

    count: "dict[tuple[str, int, str], int]" = {}
    for cx in each:
        for key in touched(cx):
            count[key] = count.get(key, 0) + 1

    def share_of(cx: book.Complex) -> "dict[int, float]":
        return {r.index: count[(r.chain, r.number, r.icode)] / n for r in cx.residues
                if r.side == 0 and len(r.near) and (r.chain, r.number, r.icode) in count}

    all_.shared, all_.models = share_of(all_), n
    all_.shared_says = f"in contact with how many of the {n} ligands"
    metric = openbook.metric_of(all_, "ensemble")

    shut = all_.closed()
    lower = np.stack([-shut[:, 2], shut[:, 1], shut[:, 0]], axis=1)
    spots = [lower[all_.side == 1][:, :2].min(axis=0), lower[all_.side == 1][:, :2].max(axis=0)]
    for cx in each:
        low = cx.closed()
        spots += [np.array([-low[cx.residues[i].near, 2].mean(), low[cx.residues[i].near, 1].mean()])
                  for i in cx.at_interface(0)]
    spots = np.array(spots)
    margin, gap, row_gap = 6.0, 22.0, 20.0
    bounds = (spots[:, 0].min() - margin, spots[:, 0].max() + margin, spots[:, 1].min() - margin,
              spots[:, 1].max() + margin)
    scale = min(13.0, (700.0 - gap) / 2.0 / (bounds[1] - bounds[0]))    # two halves, and room for the key beside them
    parts, top = [], 0.0

    own = [[e for e, s in zip(cx.elements, cx.side) if s == 1] for cx in each]
    starts = np.cumsum([0] + [len(e) for e in own])           # where each ligand's atoms are among all of them

    def all_sticks(view: np.ndarray):
        atoms = view[all_.side == 1]
        got = [openbook.surface.sticks(atoms[starts[k]:starts[k + 1]], own[k], CARBONS[k]) for k in range(n)]
        return tuple(np.concatenate([g[j] for g in got]) for j in range(3))

    # Row 0: every ligand in the pocket of the first model.
    names_all = [r.index for r in all_.residues if r.side == 0 and all_.shared.get(r.index) == 1.0]
    fills_all = {i: openbook.blend(openbook.SURE, 1.0) for i in names_all}
    origin, tallest = 0.0, 0.0
    for low, words in ((True, "all the ligands in the pocket of the first model: one half, from the cut"),
                       (False, "the other half (the page is turned over)")):
        made, w, h = _half(all_, low, bounds, scale, openbook.atom_colours(all_, metric), all_sticks, origin, top,
                           words, names_all, fills=fills_all, lines=False, residues=residues)
        parts += made
        origin, tallest = origin + w + gap, max(tallest, h)
    # Beside it: whose colour is whose, and what the colours of the pocket and of the names say.
    x, y = origin, top + 34.0
    for k, cx in enumerate(each):
        parts.append(f'<line x1="{x}" y1="{y - 3}" x2="{x + 22}" y2="{y - 3}" stroke="{openbook.as_hex(np.array(CARBONS[k]))}" '
                     f'stroke-width="4.5" stroke-linecap="round"/>' + openbook.text(x + 30, y, html.escape(cx.names[1]), 10.5))
        y += 18.0
    parts.append(metric.key(x, y + 22.0)[0])
    parts.append(openbook.text(x, y + 62.0, "the pocket here, and every name below:", 9.5, openbook.QUIET)
                 + openbook.text(x, y + 75.0, "dark where a residue is in contact", 9.5, openbook.QUIET)
                 + openbook.text(x, y + 88.0, "with every ligand", 9.5, openbook.QUIET)
                 + openbook.text(x, y + 110.0, "hatched: the protein, cut through", 9.5, openbook.QUIET)
                 + openbook.text(x, y + 132.0, "a ligand over the hatching: its own", 9.5, openbook.QUIET)
                 + openbook.text(x, y + 145.0, "model's pocket is wider there", 9.5, openbook.QUIET))
    width = origin + 215.0
    top += tallest + row_gap

    # A row for each ligand, in its own model.
    for k, cx in enumerate(each):
        mine = sorted({c.a for c in cx.contacts if c.kind in KINDS})
        shares = share_of(cx)
        fills = {i: openbook.blend(openbook.SURE, (shares.get(i, 1.0 / n) * n - 1.0) / max(1.0, n - 1.0)) for i in mine}
        every = sum(1 for i in mine if shares.get(i) == 1.0)
        fit = "" if not k else f"; its model fitted on the first with {fits[k]:.1f} Å over the backbone around the ligands"
        parts.append(f'<line x1="0" y1="{top + 8}" x2="22" y2="{top + 8}" stroke="{openbook.as_hex(np.array(CARBONS[k]))}" '
                     f'stroke-width="4.5" stroke-linecap="round"/>'
                     + openbook.text(30, top + 12, f"{html.escape(cx.names[1])}", 11.5, openbook.INK, weight="600")
                     + openbook.text(30 + 7.2 * len(cx.names[1]) + 10, top + 12,
                                     f"{len(mine)} residues in contact, {every} of them with every ligand{fit}", 10,
                                     openbook.QUIET))
        origin, tallest = 0.0, 0.0
        for low, words in ((True, "one half"), (False, "the other half")):
            made, w, h = _half(cx, low, bounds, scale, openbook.atom_colours(cx, openbook.metric_of(cx, "area")),
                               lambda view, cx=cx, k=k: openbook.surface.sticks(view[cx.side == 1], own[k], CARBONS[k]),
                               origin, top + 18.0, words, mine, fills=fills, residues=residues)
            parts += made
            origin, tallest = origin + w + gap, max(tallest, h)
        top += 18.0 + tallest + row_gap
    under = (f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {top:.0f}" '
             f'width="{width:.0f}" font-family="{openbook.FONT}">' + "".join(parts) + "</svg>")
    consensus = sorted(k for k, c in count.items() if c == n)
    facts = {"ligands": [cx.names[1] for cx in each], "fits": [None if f is None else round(f, 2) for f in fits],
             "in_contact": len(count), "with_every_ligand": len(consensus),
             "said": (f"{len(count)} residues are in contact with one of the {n} ligands or more, {len(consensus)} "
                      f"of them with every one. The models are fitted on the first with "
                      f"{', '.join(f'{f:.1f}' for f in fits[1:])} Å over the backbone around the ligands.")}
    if not comparison:
        return openbook.Sketch(under, "several ligands of one pocket", facts)
    try:
        flat = Comparison(diagrams).render(interactive=True)
    except Exception as exc:                                    # noqa: BLE001 - said, and the rows are drawn
        # The rows need nothing of the library's comparison: where it cannot be made, they stand alone
        # and the figure says why.
        said = (f"the library's comparison of these ligands could not be drawn here ({type(exc).__name__}: {exc}); "
                f"the rows stand without it")
        return openbook.Sketch(under, "several ligands of one pocket", dict(facts, comparison=None), warnings=[said])
    names = {f"{r.chain}:{r.name}{r.number}{r.icode}": r.key for cx in each for r in cx.residues
             if r.side == 0 and len(r.near)}
    svg = _over(flat.svg, openbook.numbered(under, [r for cx in each for r in cx.residues]), names,
                [cx.names[1] for cx in each])
    return openbook.Sketch(svg, "several ligands of one pocket", dict(facts, comparison=flat.figure),
                           warnings=list(flat.warnings))


def _over(flat: "str | None", under: str, names: "dict[str, str]", what) -> str:
    """A figure of the library over a figure of the sketch, the two tied by a style sheet: what is
    pointed at (or kept) in one steps the rest of the other back. ``names`` says which residue of the
    one (``data-residue``) is which mark of the other (``data-id``)."""
    w2, h2 = openbook.size_of(under)
    parts, top = [], 0.0
    if flat is not None:
        m = re.match(r'<svg ([^>]*?)width="([0-9.]+)" height="([0-9.]+)"', flat)
        assert m, "a figure of the library without a size"
        w1, h1 = float(m.group(2)), float(m.group(3))
        k = min(1.0, w2 / w1)                                    # never wider than the figure under it
        parts.append(flat.replace(m.group(0), f'<svg x="{(w2 - w1 * k) / 2.0:.1f}" y="0" {m.group(1)}'
                                              f'width="{w1 * k:.1f}" height="{h1 * k:.1f}"', 1))
        top = h1 * k + 14.0
    parts.append(openbook.nest(under, 0, top))
    scope = "svg.pk-" + hashlib.sha1(json.dumps([list(what), sorted(names)]).encode()).hexdigest()[:8]
    return (f'<svg class="book pocket {scope[4:]}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w2:.0f} '
            f'{top + h2:.0f}" width="{w2:.0f}" font-family="{openbook.FONT}"><style>{openbook.tied(scope, names, True)}</style>'
            + "".join(parts) + "</svg>")


# ---------------------------------------------------------------------------------------- the page
CASES = (
    ("kit_imatinib", "Imatinib in the kinase domain of c-Kit (1T46)",
     "One drug with hydrogen bonds, ring stacking, a ring over a charged group and hydrophobic contacts."),
    ("ar_r1881", "R1881 in the androgen receptor (2AO6)", "A steroid packed in on all sides."),
    ("dhfr_methotrexate", "Methotrexate in dihydrofolate reductase (1U72)",
     "Two carboxylates and a pteridine; the cofactor NADPH lies beside the ligand."),
    ("egfr_osimertinib", "Osimertinib in the kinase domain of EGFR (6JXT)",
     "A drug that the file declares bonded to Cys797; the diagram here is drawn without that bond."),
)

#: Sets of ligands in one pocket: the folder of each member, under ``testdata``.
SETS = (
    ("ar_ligands", "Two unrelated ligands in the androgen receptor",
     "A steroid and a non-steroidal antagonist, each from its own crystal structure; the receptor of "
     "bicalutamide is a point mutant (W741L).",
     ("ar_r1881", "ar_bicalutamide")),
    ("gr_steroids", "Three corticosteroids in the glucocorticoid receptor",
     "A congeneric series, each from its own crystal structure: one core in one place, and what differs is at "
     "its ends.",
     ("gr_dexamethasone", "gr_mometasone", "gr_budesonide")),
)


def diagram_of(case: str) -> Diagram:
    """The diagram of a case of ``testdata``: the model its ``meta.json`` names, and the table beside it."""
    folder = book.ROOT / "testdata" / case
    meta = json.loads((folder / "meta.json").read_text())
    return Diagram.from_structure(folder / meta["source"]["file"], ligand=meta["ligand"]["residue_name_in_cif"],
                                  smiles=meta["ligand"]["smiles"], name=meta["ligand"]["name"],
                                  interactions=load_interactions(folder / "interactions.tsv"))


def main() -> int:
    t = time.time()
    body, rows = [], []
    for case, title, what in CASES:
        one = figure(diagram_of(case))
        n = one.facts
        rows.append(f"<tr><td><a href='#{case}'>{html.escape(title)}</a></td><td>{100 * (1 - n['free']):.0f}</td>"
                    f"<td>{100 * n['seen_across']:.0f}</td>"
                    f"<td>{'%.0f' % (100 * n['seen_open_side']) if n['open_side'] else '–'}</td>"
                    f"<td><b>{100 * n['seen_cut']:.0f}</b></td></tr>")
        body.append(f'<h2 id="{case}">{html.escape(title)}</h2><p class="quiet">{what} {html.escape(n["said"])}</p>'
                    f"<figure>{one.svg}<figcaption><b>The diagram</b>, as the library draws it "
                    f"(<code>render(interactive=True)</code>), and <b>the same pocket in three dimensions</b>. "
                    f"Pointing at a residue in the diagram marks it in the panels, and the other way round."
                    f"</figcaption></figure>")
        print(f"{case:18s} covered {100 * (1 - n['free']):3.0f} %  across {100 * n['seen_across']:3.0f} %  open side "
              f"{'%3.0f %%' % (100 * n['seen_open_side']) if n['open_side'] else '  none'}  cut {100 * n['seen_cut']:3.0f} %"
              f"   [{time.time() - t:.0f} s]", flush=True)
    body.append('<h2 id="several" style="margin-top:64px">Several ligands of one pocket</h2>'
                '<p class="quiet">Under the library\'s own comparison of the set: the pocket of the first model '
                'with every ligand in it, each in a colour of its own, and then a row for each ligand in its own '
                'model\'s pocket. The models are superposed as the comparison superposes them, one plane is laid '
                'through all the ligands together, and every row is cut on it and seen from the same place: the '
                'same spot on two rows is the same place in the pocket. A name is dark where its residue is in '
                'contact with every ligand. Pointing at a residue marks it in every row and in the comparison.</p>')
    for slug, title, what, members in SETS:
        one = several([diagram_of(m) for m in members])
        body.append(f'<h2 id="{slug}">{html.escape(title)}</h2><p class="quiet">{what} {html.escape(one.facts["said"])}</p>'
                    f"<figure>{one.svg}</figure>")
        print(f"{slug:18s} {one.facts['in_contact']} residues in contact, {one.facts['with_every_ligand']} with every "
              f"ligand; fits {one.facts['fits'][1:]}   [{time.time() - t:.0f} s]", flush=True)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sketches: a ligand's pocket in three dimensions</title><style>{openbook.CSS}
table {{ border-collapse: collapse; font-size: 13px; margin: 14px 0; }}
th, td {{ padding: 4px 10px; text-align: right; border-bottom: 1px solid #e4e7eb; }}
th:first-child, td:first-child {{ text-align: left; }} th {{ font-weight: 600; color: #52606d; }}
</style></head><body><main>
<h1>A ligand's pocket in three dimensions, beside its diagram</h1>
<p><span class="sketch">Sketches.</span> Not part of bindview, not tested, nothing here is promised:
<code>python prototype/ppi_openbook/pocket.py</code> writes this page. Each figure is
<code>pocket.figure(diagram)</code>: the diagram of the ligand is the library's own, the three panels under
it are drawn by the sketch code from the same contacts.</p>
<p class="quiet"><b>a</b> the pocket from its most open side. <b>b</b>, <b>c</b> the pocket cut along the
ligand's own plane and opened: each half is seen from the cut, and the ligand lies on both. The numbers
say how much of what the ligand covers on the protein each way of looking shows.</p>
<table><tr><th>ligand</th><th>its surface covered, %</th><th>seen across the plane, uncut, %</th>
<th>seen from the open side, %</th><th>seen in the two halves, %</th></tr>{"".join(rows)}</table>
{"".join(body)}
</main><script>{openbook.JS}</script></body></html>
"""
    out = HERE / "pocket.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out.relative_to(book.ROOT)}: {len(page) / 1e6:.2f} MB in {time.time() - t:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
