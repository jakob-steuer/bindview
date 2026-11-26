#!/usr/bin/env python
"""Sketches of an interface between two proteins, opened like a book.

A SKETCH, not part of bindview: nothing here is tested or promised by the package, and it will change. It is
drawn so that it can be looked at and argued with.

Three figures, each for a complex read by :func:`book.load`:

``panels``  four panels: the complex as it stands, opened like a book, one face with the partner's atoms on
            it as licorice, and the same face as a flat map of residues
``faces``   the two opened faces close up, with names, contacts and regions
``table``   who touches whom, in full

    python prototype/ppi_openbook/openbook.py        # writes prototype/ppi_openbook/index.html
"""

from __future__ import annotations

import base64
import html
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import book  # noqa: E402
import route  # noqa: E402
import surface  # noqa: E402
import unroll  # noqa: E402
from bindview.confidence import BAND_LABELS, band_of  # noqa: E402
from bindview.residues import residue_class  # noqa: E402
from bindview.theme import LIGHT  # noqa: E402

__all__ = ["Sketch", "panels", "faces", "table", "unrolled", "colours_of", "COLOURS", "MODEL_COLOURS", "NAMES",
           "LINES"]

FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
INK, QUIET, FAINT = "#1f2933", "#52606d", "#9aa5b1"
RAMP = ("#dcebf2", "#9fd0e3", "#4a9cc5", "#14608f")      # area a residue loses to its partner, little to much
CAP = 120.0          # Å²; a residue that loses more is drawn as one that loses this much
STROKES = {"saltbridge": ("#c026d3", "", 1.6), "hbond": ("#1d4ed8", "4 2.5", 1.3),
           "hydrophobic": ("#3f8f5a", "1.5 2.5", 1.2), "near": ("#9aa5b1", "", 0.6)}
KIND = {"saltbridge": "salt bridge", "hbond": "hydrogen bond", "hydrophobic": "hydrophobic contact",
        "near": "atoms within 4 Å, no kind given"}
CLASSES = ("hydrophobic", "polar", "positive", "negative", "glycine")
BODY = ((0.925, 0.925, 0.932), (0.945, 0.936, 0.908))
CHAINS = ((0.945, 0.928, 0.900), (0.890, 0.920, 0.948))   # two chains on one side, told apart by a tint
ROMAN = ("I", "II", "III", "IV", "V", "VI")
TRACE = ("#c3ccd6", "#efc3a4")                            # the chain from one residue to the next, per partner
PATCHES = ("#7fb8d6", "#f0b56b", "#8fcf9f", "#e59ab8", "#c2a5e0", "#d9cf6a")   # a part of the interface each
BANDS = {"very_high": "#0053d6", "confident": "#65cbf3", "low": "#ffdb13", "very_low": "#ff7d45"}   # AlphaFold's own
SURE = ("#f3ecf8", "#cdb4e3", "#9565c0", "#5b2a86")       # how sure a model is of something, little to much
COUNT = ("#f8f0df", "#f1cf8f", "#e19a35", "#a85d00")      # how many, few to many
PAE_TOP = 15.0       # ångström; an error of this much or more is drawn as this much
EDGE = ("#3e4c59", "#9a3412")                             # the outline of a box, per partner, where both share a sheet
CARBON = (0.93, 0.62, 0.20)                               # the carbons of a partner drawn as licorice
CARBON_A = (0.44, 0.52, 0.62)                             # those of a residue of the first partner, where it is
SECOND = (0.975, 0.880, 0.760)                            # the body of the second partner where both stand together
NO_VALUE = (0.66, 0.69, 0.73)                             # a residue its colouring says nothing of, as licorice
WHOLE = 30                                                # a partner of up to this many residues is drawn whole as licorice

#: What a figure may be coloured by, which residues get a name, and which contacts are lines.
COLOURS = ("area", "class", "region", "contacts")          # for any complex
MODEL_COLOURS = ("plddt", "pae", "ipsae", "ensemble")      # for a model that says so much of itself
NAMES = ("contacts", "touching", "all")
LINES = ("all", "polar")
STICKS = ("element",)                                      # the licorice by element; or by anything of the above
_KINDS = {"all": ("saltbridge", "hbond", "hydrophobic"), "polar": ("saltbridge", "hbond")}


#: What every sketch carries for the pointer, so that no page has to: what is pointed at stands out far
#: more than the rest, and under its name its atoms and every contact it makes appear (see :func:`under`).
#: A picture that the marked residues are drawn over (``.ground``) steps back with the rest.
STYLE = ("/*book*/"
         "svg.book .res,svg.book .link,svg.book .ground{transition:opacity .12s}"
         "svg.book.dim .res:not(.hot){opacity:.16}"
         "svg.book.dim .link:not(.hot){opacity:.04}"
         "svg.book.dim .ground{opacity:.42}"
         "svg.book .link.hot{stroke-width:3.2px;stroke-opacity:1}"
         "svg.book .link.only:not(.hot){visibility:hidden}"
         "svg.book .res.hot>rect,svg.book .res.hot>ellipse{stroke-width:2.2px;stroke:#0b1220}"
         "svg.book .sticks,svg.book .reach{opacity:0;pointer-events:none;transition:opacity .12s}"
         "svg.book .sticks.shown{opacity:.45}"
         "svg.book .res:hover .sticks,svg.book .res.hot .sticks,svg.book .res:hover .reach,svg.book .res.hot .reach"
         "{opacity:1}")
STICK = "#334155"            # the carbons of a residue of the protein that is drawn as thin sticks


@dataclass
class Sketch:
    """What a sketch hands back: the picture, and its name, facts, status and warnings, as a result of the library
    has them."""

    svg: str
    figure: str
    facts: dict = field(default_factory=dict)
    status: str = "sketch"
    warnings: list = field(default_factory=list)
    quality: None = None

    def __post_init__(self) -> None:
        if "/*book*/" not in self.svg:                       # a figure made of figures has it already
            at = self.svg.index(">") + 1
            self.svg = self.svg[:at] + f"<style>{STYLE}</style>" + self.svg[at:]


def numbered(svg: str, residues) -> str:
    """The figure with its residues named for a page that joins on them: the marks and the contacts of
    ``residues`` say ``data-residue`` and ``data-residue-number``, as those of the library do."""
    known = {r.key: f'data-residue="{r.chain}:{r.name}{r.number}{r.icode}" data-residue-number="{r.number}{r.icode}"'
             for r in residues if r.side == 0}

    def told(m: "re.Match[str]") -> str:
        return m.group(0) + (f" {known[m.group(1)]}" if m.group(1) in known else "")

    return re.sub(r'data-(?:id|a)="([^"]*)"', told, svg)


def asked_by_page(chain: str, number: str, pointed: str) -> str:
    """While the page around a figure asks for a residue (``data-active`` on the root: its number, or its chain and
    its number), and nothing in the figure is ``pointed`` at: what stands behind the root's own selector."""
    return f':is([data-active="{number}"],[data-active="{chain}:{number}"]):not(:has({pointed}))'


def answering(svg: str, residues, partners: "dict[str, set[str]]", mark: str) -> str:
    """The figure with a style sheet that does what :data:`JS` does, for a page that has no script.

    Pointing at a residue keeps it and its partners (``partners``, by their keys) and steps the rest back, in
    every panel; and so does the page around the figure, with ``data-active="741"`` on the root, while nothing
    in the figure is pointed at. ``residues`` are those that can be asked for. ``mark`` tells this figure from
    the others of a page. Where the page has the script, the script is what speaks.
    """
    scope, rules = f"svg.{mark}:not(.dim)", []
    for r in residues:
        key = r.key
        states = [f':has(.res[data-id="{key}"]:hover)']
        if r.side == 0:
            states.append(asked_by_page(r.chain, f"{r.number}{r.icode}", ".res:hover"))
        asked = f"{scope}:is({','.join(states)})"
        kept = ",".join(f'[data-id="{k}"]' for k in sorted({key} | partners.get(key, set())))
        own = f':is([data-a="{key}"],[data-b="{key}"])'
        rules += [f"{asked} .res:not({kept}){{opacity:.16}}",
                  f"{asked} .link:not({own}){{opacity:.04}}",
                  f"{asked} .ground{{opacity:.42}}",
                  f"{asked} .link{own}{{stroke-width:3.2px;stroke-opacity:1;visibility:visible}}",
                  f"{asked} .res:is({kept})>:is(rect,ellipse){{stroke-width:2.2px;stroke:#0b1220}}",
                  f"{asked} .res:is({kept}) :is(.sticks,.reach){{opacity:1}}"]
    at = svg.index(">") + 1
    return svg[:at].replace('class="book', f'class="book {mark}', 1) + f"<style>{''.join(rules)}</style>" + svg[at:]


def tied(scope: str, names: "dict[str, str]", atoms: bool) -> str:
    """The style sheet of a figure of the library over a sketch: what is pointed at, or kept, in one steps the rest
    of the other back, and the page around the two can ask for a residue of both (``data-active`` on the root).
    ``names`` says which residue of the one (``data-residue``) is which mark of the other (``data-id``); ``atoms``:
    the sketch has the atoms of a residue to bring up."""
    rules = []
    for ident, key in names.items():
        chain = ident.split(":", 1)[0]
        mine = f':is(.res,.link):not([data-id="{key}"],[data-a="{key}"])'
        here = f':is(svg.bindview :is(g.residue,g.interaction):not([data-residue="{ident}"]),{mine})'
        there = f'{scope}:has(svg.bindview [data-residue="{ident}"]:is(:hover,:focus))'
        page = scope + asked_by_page(chain, key[1 + len(chain):], ".res:hover,svg.bindview [data-residue]:is(:hover,:focus)")
        rules.append(f"{there} {mine}{{opacity:.16}}")
        if atoms:
            rules.append(f':is({there},{page}) .res[data-id="{key}"] :is(.sticks,.reach){{opacity:1}}')
        rules.append(f':is({scope}:has(.res[data-id="{key}"]:hover),{page}) {here}{{opacity:.16}}')
    return "".join(rules)


def _one_of(value: str, allowed, what: str) -> str:
    if value not in allowed:
        raise ValueError(f"{what}={value!r} is not one of {', '.join(repr(a) for a in allowed)}")
    return value


def named(cx: book.Complex, names: str, kinds) -> "set[int]":
    """The residues that get a name.

    ``"contacts"``: those that make a contact which the figure draws, so a name is always the end of a
    line. ``"touching"``: those with an atom within 4 Å of the partner, whether the contact has a kind or
    not. ``"all"``: every residue at the interface (an atom within 4.5 Å). The others are colour on the
    surface, and are named to whoever points at them.
    """
    if names == "all":
        return set(cx.at_interface(0) + cx.at_interface(1))
    return {i for c in cx.contacts if names == "touching" or c.kind in kinds for i in (c.a, c.b)}


def rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def ramp(f: float) -> np.ndarray:
    f = min(1.0, max(0.0, f)) * (len(RAMP) - 1)
    i = min(int(f), len(RAMP) - 2)
    return rgb(RAMP[i]) * (1.0 - (f - i)) + rgb(RAMP[i + 1]) * (f - i)


def as_hex(c: np.ndarray) -> str:
    return "#" + "".join(f"{int(round(255 * float(v))):02x}" for v in np.clip(c, 0.0, 1.0))


def top_of(cx: book.Complex) -> float:
    most = max(r.buried for r in cx.residues)
    return min(CAP, float(20.0 * np.ceil(most / 20.0)))


@dataclass
class Metric:
    """What a figure is coloured by: the colour of a residue (``None``: none), and the key that says so."""

    name: str
    of: "Callable[[book.Residue], np.ndarray | None]"
    key: "Callable[[float, float], tuple[str, float]]"       # at x, y: the drawing, and where it ends


def blend(stops, f: float) -> np.ndarray:
    f = min(1.0, max(0.0, f)) * (len(stops) - 1)
    i = min(int(f), len(stops) - 2)
    return rgb(stops[i]) * (1.0 - (f - i)) + rgb(stops[i + 1]) * (f - i)


def colours_of(cx: book.Complex) -> "tuple[str, ...]":
    """What this complex can be coloured by."""
    have = (("plddt", cx.plddt), ("pae", cx.pae), ("ipsae", cx.ipsae), ("ensemble", cx.shared))
    return COLOURS + tuple(n for n, values in have if values) + tuple(cx.given)


def metric_of(cx: book.Complex, colour: str) -> Metric:
    """The colouring of that name for this complex.

    ``"area"``: what the partner covers of a residue. ``"class"``: the kind of residue. ``"region"``:
    the part of the interface a residue belongs to, the same colour on both partners, so that what
    lies on what is seen. ``"contacts"``: how many residues of the partner it touches. For a model
    that says so much of itself: ``"plddt"`` (AlphaFold's four bands in their usual colours),
    ``"pae"`` (the mean predicted aligned error to the residues it touches), ``"ipsae"`` (its term of
    ipSAE) and ``"ensemble"`` (in how many of several models it is at the interface). And any name of
    ``cx.given``: numbers from outside.
    """
    top = top_of(cx)
    if colour == "area":
        return Metric(colour, lambda r: blend(RAMP, r.buried / top) if r.buried > 1.0 else None,
                      lambda x, y: ramp_key(x, y, RAMP, "area", "area the partner covers, per residue", "0",
                                            f"{top:.0f} Å²" + (" or more" if top >= CAP else "")))
    if colour == "class":
        return Metric(colour, lambda r: rgb(LIGHT.residue_class(residue_class(r.name))),
                      lambda x, y: swatch_key(x, y, "kind of residue",
                                              [(name, LIGHT.residue_classes[name]) for name in CLASSES]))
    if colour == "region":
        return Metric(colour, lambda r: rgb(PATCHES[r.region % len(PATCHES)]) if r.region >= 0 else None,
                      lambda x, y: swatch_key(x, y, "part of the interface: the same colour lies on the same colour",
                                              [(ROMAN[g], PATCHES[g % len(PATCHES)]) for g in range(cx.n_regions)]))
    if colour == "contacts":
        most = max(2, max(r.partners for r in cx.residues))
        return Metric(colour, lambda r: blend(COUNT, r.partners / most) if r.partners else None,
                      lambda x, y: ramp_key(x, y, COUNT, "count", "residues of the partner it touches", "0",
                                            str(most)))
    missing = {"plddt": (cx.plddt, "a model that states its pLDDT, and this structure states none"),
               "pae": (cx.pae, "a model loaded with its matrix of predicted aligned errors (pae=...)"),
               "ipsae": (cx.ipsae, "a model loaded with its matrix of predicted aligned errors (pae=...)"),
               "ensemble": (cx.shared, "a complex that was given other models of itself (book.ensemble)")}
    if colour in missing and not missing[colour][0]:
        raise ValueError(f"colour={colour!r} is for {missing[colour][1]}")
    if colour == "plddt":
        return Metric(colour, lambda r: rgb(BANDS[band_of(cx.plddt[r.index])]) if r.index in cx.plddt else None,
                      lambda x, y: swatch_key(x, y, "pLDDT of the residue (AlphaFold's bands and colours)",
                                              [(BAND_LABELS[b].replace("pLDDT ", ""), c) for b, c in BANDS.items()]))
    if colour == "pae":
        return Metric(colour, lambda r: (blend(SURE, 1.0 - cx.pae[r.index] / PAE_TOP) if r.index in cx.pae else None),
                      lambda x, y: ramp_key(x, y, SURE[::-1], "pae", "predicted aligned error to the residues it "
                                            "touches (dark: sure)", "0", f"{PAE_TOP:.0f} Å or more"))
    if colour == "ipsae":
        return Metric(colour, lambda r: blend(SURE, cx.ipsae[r.index]) if r.index in cx.ipsae else None,
                      lambda x, y: ramp_key(x, y, SURE, "sure", "its term of ipSAE: the partner scored, the model "
                                            "aligned on this residue", "0", "1"))
    if colour == "ensemble":
        n = cx.models
        return Metric(colour, lambda r: blend(SURE, (cx.shared[r.index] * n - 1.0) / max(1.0, n - 1.0))
                      if r.index in cx.shared else None,
                      lambda x, y: swatch_key(x, y, cx.shared_says or f"at the interface in how many of the {n} models",
                                              [(str(k), as_hex(blend(SURE, (k - 1.0) / max(1.0, n - 1.0))))
                                               for k in range(1, n + 1)]))
    if colour in cx.given:
        says, values = cx.given[colour]
        lo, hi = min(values.values()), max(values.values())
        return Metric(colour, lambda r: blend(RAMP, (values[r.index] - lo) / max(1e-9, hi - lo))
                      if r.index in values else None,
                      lambda x, y: ramp_key(x, y, RAMP, "given", html.escape(says), f"{lo:g}", f"{hi:g}"))
    raise ValueError(f"colour={colour!r} is not one of {', '.join(repr(c) for c in colours_of(cx))}")


def atom_colours(cx: book.Complex, metric: "Metric | None") -> np.ndarray:
    """One colour per atom: a residue of the interface in the colour its metric gives it, the rest in the
    tint of its body. Without a metric the interface is a shade apart from the body, and no more."""
    col = np.empty((len(cx.xyz), 3))
    for s in (0, 1):
        chains = sorted({r.chain for r in cx.residues if r.side == s})
        for r in cx.residues:
            if r.side != s:
                continue
            base = np.array(CHAINS[chains.index(r.chain) % 2] if len(chains) > 1 else BODY[s])
            col[r.atoms] = base
            if not len(r.near):
                continue
            if metric is None:
                col[r.atoms] = base * np.array([0.86, 0.93, 0.97])
            else:
                own = metric.of(r)
                if own is not None:
                    col[r.atoms] = own
    return col


def licorice(cx: book.Complex, atoms, view: np.ndarray, px, colours: np.ndarray, width: float = 4.6) -> str:
    """These atoms as licorice in the drawing itself, not in the picture: under every bond a white and a
    dark casing, over them a half in the colour of each of its atoms (``colours``: one for every atom of
    the complex). The far bonds first, so that a near one lies over them. Empty where no two are bonded."""
    atoms = [int(a) for a in atoms]
    at = {a: px(view[a]) for a in atoms}
    bonds = []
    for n, a in enumerate(atoms):
        for b in atoms[n + 1:]:
            far = 2.25 if {cx.elements[a].upper(), cx.elements[b].upper()} & {"S", "P", "SE"} else 1.95
            if 0.4 < float(np.linalg.norm(cx.xyz[a] - cx.xyz[b])) < far:
                bonds.append((float(view[a][2] + view[b][2]), a, b))
    if not bonds:
        return ""
    bonds.sort()
    d = " ".join(f"M {at[a][0]:.1f} {at[a][1]:.1f} L {at[b][0]:.1f} {at[b][1]:.1f}" for _, a, b in bonds)
    halves = []
    for _, a, b in bonds:
        whole = f"M {at[a][0]:.1f} {at[a][1]:.1f} L {at[b][0]:.1f} {at[b][1]:.1f}"
        one, other = as_hex(colours[a]), as_hex(colours[b])
        halves.append(f'<path d="{whole}" stroke="{other}"/>')
        if one != other:
            m = (at[a] + at[b]) / 2.0
            halves.append(f'<path d="M {at[a][0]:.1f} {at[a][1]:.1f} L {m[0]:.1f} {m[1]:.1f}" stroke="{one}"/>')
    return (f'<path d="{d}" fill="none" stroke="#ffffff" stroke-opacity="0.9" stroke-width="{width + 3.6:.1f}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
            f'<path d="{d}" fill="none" stroke="#1f2933" stroke-width="{width + 1.5:.1f}" stroke-linecap="round"/>'
            f'<g fill="none" stroke-width="{width:.1f}" stroke-linecap="round">{"".join(halves)}</g>')


def drawn_atoms(cx: book.Complex, i: int, look) -> "list[int]":
    """The atoms of a residue that :func:`under` draws: its side chain from the alpha carbon, and its
    backbone where the backbone makes a contact of a kind ``look`` names (all of a glycine)."""
    r = cx.residues[i]
    mine = [c for c in cx.contacts if i in (c.a, c.b) and c.kind in look and c.atoms_a and c.atoms_b]
    used = {a for c in mine for a in (c.atoms_a if c.a == i else c.atoms_b)}
    backbone = {int(a) for a in r.atoms if cx.atoms[a] in ("N", "C", "O", "OXT")}
    atoms = [int(a) for a in r.atoms if int(a) not in backbone]
    if len(atoms) < 2 or used & backbone:
        atoms = [int(a) for a in r.atoms]
    return atoms


def under(cx: book.Complex, i: int, view: np.ndarray, px, look, shown: bool = False,
          thick: "np.ndarray | None" = None, reach: bool = True) -> str:
    """What stands under a residue's name in a picture of atoms, for :func:`mark` to take in.

    The residue itself as thin sticks: its side chain from the alpha carbon, and its backbone where the
    backbone makes a contact (all of a glycine). And every contact it makes, as a bold line from atom
    to atom with a ring on the partner's atom; ``look`` says of each kind of contact its colour and how
    its line is dashed, and a kind it does not name is left out. All of it is hidden until the residue
    is pointed at; with ``shown`` the sticks are there all the time, faintly. ``view`` is every atom in
    the frame of the picture and ``px`` takes a place of that frame to the drawing. ``thick`` (a colour
    for every atom of the complex) draws the residue as licorice and not as thin sticks. Without
    ``reach`` the contacts are left out: for a picture that draws them itself (:func:`reaching`).
    """
    mine = [c for c in cx.contacts if i in (c.a, c.b) and c.kind in look and c.atoms_a and c.atoms_b]
    atoms = drawn_atoms(cx, i, look)
    at = {a: px(view[a]) for a in atoms}
    bonds = []
    for n, a in enumerate(atoms):
        for b in atoms[n + 1:]:
            far = 2.25 if {cx.elements[a].upper(), cx.elements[b].upper()} & {"S", "P", "SE"} else 1.95
            if 0.4 < float(np.linalg.norm(cx.xyz[a] - cx.xyz[b])) < far:
                bonds.append(f"M {at[a][0]:.1f} {at[a][1]:.1f} L {at[b][0]:.1f} {at[b][1]:.1f}")
    d = " ".join(bonds)
    dots = "".join(f'<circle cx="{at[a][0]:.1f}" cy="{at[a][1]:.1f}" r="2.3" '
                   f'fill="{as_hex(np.array(surface.ELEMENT[cx.elements[a].upper()]))}"/>'
                   for a in atoms if cx.elements[a].upper() in surface.ELEMENT)
    sticks = (f'<g class="sticks{" shown" if shown else ""}"><path d="{d}" fill="none" stroke="#ffffff" '
              f'stroke-opacity="0.8" stroke-width="4.4" stroke-linecap="round"/><path d="{d}" fill="none" '
              f'stroke="{STICK}" stroke-width="2.2" stroke-linecap="round"/>{dots}</g>') if d else ""
    if thick is not None:
        body = licorice(cx, atoms, view, px, thick)
        sticks = f'<g class="sticks">{body}</g>' if body else ""
    if not reach:
        return sticks
    lines = []
    for c in mine:
        own, other = (c.atoms_a, c.atoms_b) if c.a == i else (c.atoms_b, c.atoms_a)
        pa, pb = px(view[list(own)].mean(axis=0)), px(view[list(other)].mean(axis=0))
        colour, dash = look[c.kind]
        lines.append(
            f'<line x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" stroke="#ffffff" '
            f'stroke-opacity="0.85" stroke-width="5.6" stroke-linecap="round"/>'
            f'<line x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" stroke="{colour}" '
            f'stroke-width="2.8" stroke-linecap="round"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>"
            f'<circle cx="{pb[0]:.1f}" cy="{pb[1]:.1f}" r="4.6" fill="none" stroke="#ffffff" stroke-width="3.4"/>'
            f'<circle cx="{pb[0]:.1f}" cy="{pb[1]:.1f}" r="4.6" fill="none" stroke="{colour}" stroke-width="1.8"/>'
            f'<circle cx="{pa[0]:.1f}" cy="{pa[1]:.1f}" r="2.6" fill="{colour}" stroke="#ffffff" stroke-width="0.9"/>')
    return sticks + (f'<g class="reach">{"".join(lines)}</g>' if lines else "")


def reaching(cx: book.Complex, c: book.Contact, view: np.ndarray, px, look) -> str:
    """A contact from atom to atom, bold, with a ring on each of its two atoms: a line that is there only
    while one of its two residues is pointed at (``link only``), so that a residue shows the contacts it
    makes and not those of its partners."""
    pa, pb = px(view[list(c.atoms_a)].mean(axis=0)), px(view[list(c.atoms_b)].mean(axis=0))
    colour, dash = look[c.kind]
    line = f'x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" stroke-linecap="round"'
    rings = "".join(f'<circle cx="{q[0]:.1f}" cy="{q[1]:.1f}" r="4.4" fill="#ffffff" fill-opacity="0.55" '
                    f'stroke="#ffffff" stroke-width="3.4"/><circle cx="{q[0]:.1f}" cy="{q[1]:.1f}" r="4.4" '
                    f'fill="none" stroke="{colour}" stroke-width="1.8"/>' for q in (pa, pb))
    return (f'<g class="link only" data-a="{cx.residues[c.a].key}" data-b="{cx.residues[c.b].key}">'
            f'<line {line} stroke="#ffffff" stroke-opacity="0.85" stroke-width="5.6"/>'
            f'<line {line} stroke="{colour}" stroke-width="2.8"' + (f' stroke-dasharray="{dash}"' if dash else "")
            + f'/>{rings}<title>{cx.residues[c.a].label} – {cx.residues[c.b].label}: {KIND[c.kind]}, '
              f'{c.distance:.1f} Å</title></g>')


def mark(r: book.Residue, p, hw: float, hh: float, fill: np.ndarray, size: float, title: str, opacity: str = "1",
         chain: bool = False, lead: str = "", extra: str = "", dash: str = "") -> str:
    """A residue's mark: a box for the first partner and an ellipse for the second, so that the two are
    told apart by their shape wherever they share a sheet. The name is dark or white, by its ground.
    ``extra`` is drawn under the name as part of the mark (see :func:`under`); ``dash`` dashes the outline of a
    box."""
    dark = float(fill @ np.array([0.299, 0.587, 0.114])) < 0.55 and opacity == "1"
    ink = "#ffffff" if dark else INK
    if r.side == 0:
        shape = (f'<rect x="{p[0] - hw:.1f}" y="{p[1] - hh:.1f}" width="{2 * hw:.1f}" height="{2 * hh:.1f}" rx="1.5" '
                 f'fill="{as_hex(fill)}" fill-opacity="{opacity}" stroke="{EDGE[0]}" stroke-width="0.9"'
                 + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
    else:
        shape = (f'<ellipse cx="{p[0]:.1f}" cy="{p[1]:.1f}" rx="{hw:.1f}" ry="{hh:.1f}" fill="{as_hex(fill)}" '
                 f'fill-opacity="{opacity}" stroke="{EDGE[1]}" stroke-width="1.3"/>')
    letter = f'<tspan font-size="{size - 2.5}" fill="{ink}" fill-opacity="0.7" dx="2">{r.chain}</tspan>' if chain else ""
    return (f'<g class="res" data-id="{r.key}">{extra}{lead}{shape}<text x="{p[0]:.1f}" y="{p[1] + size * 0.36:.1f}" '
            f'font-size="{size}" fill="{ink}" text-anchor="middle">{r.label}{letter}</text><title>{title}</title></g>')


def mark_size(r: book.Residue, size: float, chain: bool, pad: float) -> "tuple[float, float]":
    """Half the width and half the height of a residue's mark; an ellipse needs more room for the same name."""
    round_ = 1.0 if r.side else 0.0
    return ((len(r.label) * size * 0.58 + (9.0 if chain else 0.0)) / 2.0 + pad + 3.0 * round_,
            size * 0.5 + 3.2 + 1.5 * round_)


def _picture(xyz: np.ndarray, elements, colours: np.ndarray, bounds, scale: float, flat: float = 0.0,
             fade: float = 0.0, groups=None, licorice=None, cut: "float | None" = None, hatched: bool = True
             ) -> "tuple[str, int, int, np.ndarray | None]":
    """The surface of these atoms inside ``bounds`` as a PNG in a data address, with its size in pixels."""
    extra = 2.2
    x0, x1, y0, y1 = bounds
    img, front = surface.render(xyz, elements, colours, (x0 - extra, x1 + extra, y0 - extra, y1 + extra), scale,
                                flat=flat, groups=groups, licorice=licorice, cut=cut, hatched=hatched)
    trim = int(round(extra * scale))
    img = img[trim:img.shape[0] - trim, trim:img.shape[1] - trim].copy()
    if front is not None:
        front = front[trim:front.shape[0] - trim, trim:front.shape[1] - trim]
    if fade > 0.0:                                        # a body that goes on beyond the page fades out
        h, w = img.shape[:2]
        n = max(1, int(round(fade * scale)))
        edge = np.minimum(np.minimum(np.arange(w)[None, :], np.arange(w)[None, ::-1]),
                          np.minimum(np.arange(h)[:, None], np.arange(h)[::-1, None])) / n
        s = np.clip(edge, 0.0, 1.0)
        img[..., 3] = (img[..., 3] * (s * s * (3.0 - 2.0 * s))).astype(np.uint8)
    data = base64.b64encode(surface.png(img)).decode("ascii")
    return "data:image/png;base64," + data, img.shape[1], img.shape[0], front


def picture(view: np.ndarray, cx: book.Complex, colours: np.ndarray, bounds, scale: float, which=None,
            flat: float = 0.0, fade: float = 0.0, regions: bool = False, licorice=None, cut: "float | None" = None,
            hatched: bool = True) -> "tuple[str, int, int, np.ndarray | None]":
    """The surface inside ``bounds`` as a PNG in a data address, with its size in pixels. With ``regions``
    the borders between the regions of the interface are drawn on it, and the region in front comes back
    per pixel. ``licorice`` is drawn into the same picture (see :func:`surface.sticks`); ``cut`` and
    ``hatched`` are those of :func:`surface.render`."""
    m = np.ones(len(view), dtype=bool) if which is None else which
    groups = None
    if regions:
        groups = np.full(len(view), -1)
        for r in cx.residues:
            if len(r.near):
                groups[r.atoms] = r.region
        groups = groups[m]
    return _picture(view[m], [e for e, k in zip(cx.elements, m) if k], colours[m], bounds, scale, flat, fade, groups,
                    licorice, cut, hatched)


def text(x: float, y: float, words: str, size: float = 10.5, fill: str = INK, anchor: str = "start",
         weight: str = "400", extra: str = "") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{extra}>{words}</text>')


def heading(x: float, y: float, letter: str, words: str) -> str:
    """The line over a panel, with its letter where the figure has several."""
    mark = f'<tspan font-weight="700" fill="{INK}">{letter}</tspan><tspan dx="7">{words}</tspan>' if letter else words
    return text(x, y, mark, 11.5, QUIET)


def ramp_key(x: float, y: float, stops, ident: str, says: str, left: str, right: str, w: float = 120.0
             ) -> "tuple[str, float]":
    """A scale from one colour to another, with what it measures over it and its ends under it."""
    marks = "".join(f'<stop offset="{i / (len(stops) - 1):.3f}" stop-color="{c}"/>' for i, c in enumerate(stops))
    return (f'<defs><linearGradient id="ramp-{ident}">{marks}</linearGradient></defs>'
            f'<rect x="{x}" y="{y}" width="{w}" height="9" rx="2" fill="url(#ramp-{ident})" stroke="{FAINT}" '
            f'stroke-width="0.5"/>' + text(x, y + 21, left, 9.5, QUIET) + text(x + w, y + 21, right, 9.5, QUIET, "end")
            + text(x, y - 5, says, 9.5, QUIET)), x + max(w, 12.0 + 4.9 * len(says))


def swatch_key(x: float, y: float, says: str, items) -> "tuple[str, float]":
    """A square of colour for each of a few things, named beside it."""
    out, start = [text(x, y - 5, says, 9.5, QUIET)], x
    for name, colour in items:
        out.append(f'<rect x="{x}" y="{y}" width="11" height="11" rx="2" fill="{colour}" '
                   f'stroke="{FAINT}" stroke-width="0.5"/>' + text(x + 15, y + 9.5, html.escape(name), 9.5, QUIET))
        x += 22 + 5.4 * len(name)
    return "".join(out), max(x, start + 12.0 + 4.9 * len(says))


def shape_key(x: float, y: float, cx: book.Complex) -> "tuple[str, float]":
    """Whose residue a mark is: a box or an ellipse."""
    a, b = html.escape(cx.names[0]), html.escape(cx.names[1])
    second = x + 34 + 5.3 * len(cx.names[0]) + 16
    return (f'<rect x="{x}" y="{y - 9}" width="24" height="12" rx="1.5" fill="#fff" stroke="{EDGE[0]}" '
            f'stroke-width="0.9"/>' + text(x + 30, y, a, 9.5, QUIET)
            + f'<ellipse cx="{second + 13}" cy="{y - 3}" rx="13" ry="7" fill="#fff" stroke="{EDGE[1]}" '
              f'stroke-width="1.3"/>' + text(second + 32, y, b, 9.5, QUIET)), second + 36 + 5.3 * len(cx.names[1])


def ribbon_key(x: float, y: float) -> "tuple[str, float]":
    """What the pale bands under the marks are."""
    says = "next residue in the chain (dotted: one residue between)"
    return (f'<line x1="{x + 3}" y1="{y - 3}" x2="{x + 25}" y2="{y - 3}" stroke="{TRACE[0]}" stroke-width="6" '
            f'stroke-linecap="round"/>' + text(x + 34, y, says, 9.5, QUIET)), x + 34 + 5.0 * len(says)


def line_key(x: float, y: float, kinds) -> "tuple[str, float]":
    out = []
    for kind in kinds:
        stroke, dash, width = STROKES[kind]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{x}" y1="{y - 3}" x2="{x + 22}" y2="{y - 3}" stroke="{stroke}" '
                   f'stroke-width="{max(width, 1.0)}"{d}/>' + text(x + 28, y, KIND[kind], 9.5, QUIET))
        x += 34 + 5.3 * len(KIND[kind]) + 14
    return "".join(out), x


# ---------------------------------------------------------------------------------------- a numeral for a region
def numeral(x: float, y: float, g: int, r: float = 9.0) -> str:
    size = (8.6 if len(ROMAN[g]) < 3 else 7.6) * r / 9.0
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#ffffff" stroke="{INK}" stroke-width="1"/>'
            + text(x, y + 0.36 * size, ROMAN[g], round(size, 1), INK, "middle", "700"))


def free_spot(target, boxes, taken, bounds, allowed=None, radius: float = 9.0, step: float = 3.0, points=None):
    """The place nearest to ``target`` where a numeral covers no box and no other numeral.

    ``boxes`` are (centre x, centre y, half width, half height); ``allowed`` says of a grid of places which
    are in the region the numeral names; ``points`` (an array of places) are not to be covered either.
    ``None`` where there is no such place.
    """
    x0, y0, x1, y1 = bounds
    gx, gy = np.meshgrid(np.arange(x0 + radius + 2.0, x1 - radius - 2.0, step),
                         np.arange(y0 + radius + 2.0, y1 - radius - 2.0, step))
    if not gx.size:
        return None
    ok = np.ones(gx.shape, dtype=bool)
    for bx, by, hw, hh in boxes:
        ok &= np.hypot(np.maximum(np.abs(gx - bx) - hw, 0.0), np.maximum(np.abs(gy - by) - hh, 0.0)) >= radius + 1.5
    if points is not None:
        xs, ys, reach = gx[0, :], gy[:, 0], radius + 1.5
        for qx, qy in points:
            c0, c1 = np.searchsorted(xs, qx - reach), np.searchsorted(xs, qx + reach, side="right")
            r0, r1 = np.searchsorted(ys, qy - reach), np.searchsorted(ys, qy + reach, side="right")
            if c0 < c1 and r0 < r1:
                ok[r0:r1, c0:c1] &= np.hypot(gx[r0:r1, c0:c1] - qx, gy[r0:r1, c0:c1] - qy) >= reach
    for ux, uy in taken:
        ok &= np.hypot(gx - ux, gy - uy) >= 2.0 * radius + 3.0
    if allowed is not None:
        ok &= allowed(gx, gy)
    if not ok.any():
        return None
    far = np.where(ok, (gx - target[0]) ** 2 + (gy - target[1]) ** 2, np.inf)
    k = np.unravel_index(int(np.argmin(far)), far.shape)
    return float(gx[k]), float(gy[k])


def numerals(cx: book.Complex, side, middles, boxes, bounds, front, radius: float = 9.0, points=None) -> "list[str]":
    """A numeral in each region of one picture: in the free place nearest to the middle of the region.
    ``front`` is the region per pixel of the picture whose top left corner is ``bounds[:2]``; ``points``
    are places a numeral is not to cover (a partner drawn as licorice), where there is room."""
    out, taken = [], []
    for g in range(cx.n_regions):
        if g not in middles:
            continue

        def inside(gx, gy, g=g):
            rows = np.clip((gy - bounds[1]).astype(int), 0, front.shape[0] - 1)
            cols = np.clip((gx - bounds[0]).astype(int), 0, front.shape[1] - 1)
            return front[rows, cols] == g

        middle = middles[g]
        region = inside if front is not None else None
        spot = ((free_spot(middle, boxes, taken, bounds, region, radius, points=points) if points is not None else None)
                or free_spot(middle, boxes, taken, bounds, region, radius)
                or free_spot(middle, boxes, taken, bounds, None, radius) or (float(middle[0]), float(middle[1])))
        taken.append(spot)
        out.append(numeral(spot[0], spot[1], g, radius))
    return out


def lands(boxes_of, w: float, h: float, top: float, reach: float = 21.0, fine: int = 2):
    """The sheet divided among the regions: every place within ``reach`` of a box belongs to the region of
    the nearest box. Comes back as a transparent picture of the borders, and as the region per pixel."""
    cols, rows = int(w * fine), int(h * fine)
    gx, gy = np.meshgrid((np.arange(cols) + 0.5) / fine, top + (np.arange(rows) + 0.5) / fine)
    best, who = np.full(gx.shape, np.inf), np.full(gx.shape, -1)
    for g, boxes in boxes_of.items():
        for bx, by, hw, hh in boxes:
            d = np.hypot(np.maximum(np.abs(gx - bx) - hw, 0.0), np.maximum(np.abs(gy - by) - hh, 0.0))
            closer = d < best
            best[closer] = d[closer]
            who[closer] = g
    who[best > reach] = -1
    edge = np.zeros(who.shape, dtype=bool)
    for axis in (0, 1):
        differs = who != np.roll(who, -1, axis=axis)
        edge |= differs | np.roll(differs, 1, axis=axis)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = False
    alpha = np.clip(surface.blur(edge.astype(float), 1, 1) * 1.6, 0.0, 1.0)
    rgba = np.zeros(who.shape + (4,))
    rgba[..., :3] = rgb(QUIET)
    rgba[..., 3] = 0.75 * alpha
    small = rgba[:(rows // fine) * fine, :(cols // fine) * fine].reshape(rows // fine, fine, cols // fine, fine, 4)
    img = (np.clip(small.mean(axis=(1, 3)), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    src = "data:image/png;base64," + base64.b64encode(surface.png(img)).decode("ascii")
    return src, img.shape[1], img.shape[0], who[::fine, ::fine]


# ---------------------------------------------------------------------------------------- the whole, closed and opened
def overview(cx: book.Complex, colour: str = "area", regions: bool = False, letters=("", ""), key: bool = True) -> str:
    metric = metric_of(cx, colour)
    colours = atom_colours(cx, metric)
    shut, opened = cx.closed(), cx.opened()
    pad = 3.0

    def box(v):
        return (v[:, 0].min() - pad, v[:, 0].max() + pad, v[:, 1].min() - pad, v[:, 1].max() + pad)

    k1, k2 = 2.5, 4.3
    b1, b2 = box(shut), box(opened)
    # As it stands, the second partner has to be told from the first. One of a few residues is lost in a
    # surface seen from the side: every atom of it is a ball in the colour of its licorice, which shows
    # through, pale, where the first partner stands in front of it, and the panel is drawn a little
    # larger. A larger one keeps its surface and gets a tint of its own.
    second = cx.side == 1
    if sum(1 for r in cx.residues if r.side == 1) <= WHOLE:
        k1 = 3.3
        balls = (shut[second], 0.92 * surface.radii([e for e, k in zip(cx.elements, second) if k]),
                 np.tile(np.array(CARBON), (int(second.sum()), 1)))
        src1, w1, h1, _ = picture(shut, cx, colours, b1, k1, which=~second, licorice=balls)
    else:
        stands = colours.copy()
        for r in cx.residues:
            if r.side == 1 and not len(r.near):
                stands[r.atoms] = np.array(SECOND)
        src1, w1, h1, _ = picture(shut, cx, stands, b1, k1)
    src2, w2, h2, front = picture(opened, cx, colours, b2, k2, regions=regions)
    gap, head, foot = 56, 34, (58 if key else 26)
    width, height = max(w1 + gap + w2 + 70, w1 + gap + 330), head + max(h1, h2) + foot
    y1, y2 = head + (max(h1, h2) - h1) / 2.0, head + (max(h1, h2) - h2) / 2.0
    x2 = w1 + gap
    parts = [f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
             f'width="{width}" font-family="{FONT}">',
             heading(0, 14, letters[0], "as it stands, seen from the side"),
             heading(x2, 14, letters[1], "opened: a quarter turn each, both faces to the reader"),
             f'<image href="{src1}" x="0" y="{y1:.1f}" width="{w1}" height="{h1}"/>',
             f'<image href="{src2}" x="{x2}" y="{y2:.1f}" width="{w2}" height="{h2}"/>']
    # The spine in the closed view, and where it went.
    sx = (0.0 - b1[0]) * k1
    parts.append(f'<line x1="{sx:.1f}" y1="{y1 - 4:.1f}" x2="{sx:.1f}" y2="{y1 + h1 + 4:.1f}" stroke="{INK}" '
                 f'stroke-width="0.9" stroke-dasharray="5 3"/>')
    parts.append(f'<path d="M {sx - 20:.1f} {y1 - 9:.1f} q 20 -13 40 0" fill="none" stroke="{INK}" stroke-width="0.9"/>'
                 f'<path d="M {sx - 20:.1f} {y1 - 9:.1f} l 7.5 -1 l -4.6 -6.2 z" fill="{INK}"/>'
                 f'<path d="M {sx + 20:.1f} {y1 - 9:.1f} l -7.5 -1 l 4.6 -6.2 z" fill="{INK}"/>')
    mid = x2 + (0.0 - b2[0]) * k2
    parts.append(f'<line x1="{mid:.1f}" y1="{y2 + 6:.1f}" x2="{mid:.1f}" y2="{y2 + h2 - 6:.1f}" stroke="{FAINT}" '
                 f'stroke-width="0.8" stroke-dasharray="5 3"/>')
    under = [(shut[cx.side == s, 0].mean() - b1[0]) * k1 for s in (0, 1)]
    close = abs(under[1] - under[0]) < 3.3 * (len(cx.names[0]) + len(cx.names[1]))
    for s in (0, 1):
        m = cx.side == s
        parts.append(text(min(max(under[s], 40.0), w1 - 20.0), y1 + h1 + 16 + (13 if close and s else 0),
                          html.escape(cx.names[s]), 11, EDGE[1] if s else INK, "middle", "600"))
        cx2 = x2 + (opened[m, 0].mean() - b2[0]) * k2
        parts.append(text(cx2, y2 + h2 + 16, html.escape(cx.names[s]), 11, INK, "middle", "600"))
    # A name on the surface for whoever points at a residue.
    for i in cx.at_interface(0) + cx.at_interface(1):
        r = cx.residues[i]
        p = opened[r.near].mean(axis=0)
        parts.append(f'<circle cx="{x2 + (p[0] - b2[0]) * k2:.1f}" cy="{y2 + (b2[3] - p[1]) * k2:.1f}" r="7" '
                     f'fill="#000" fill-opacity="0"><title>{r.label} (chain {r.chain}): {r.buried:.0f} Å² covered'
                     f'</title></circle>')
    if regions and front is not None:
        for s in (0, 1):
            middles = {}
            for g in range(cx.n_regions):
                pts = [opened[cx.residues[i].near].mean(axis=0) for i in cx.at_interface(s) if cx.residues[i].region == g]
                if pts:
                    p = np.mean(pts, axis=0)
                    middles[g] = np.array([x2 + (p[0] - b2[0]) * k2, y2 + (b2[3] - p[1]) * k2])
            parts += numerals(cx, s, middles, [], (x2, y2, x2 + w2, y2 + h2), front, radius=7.0)
    if key:
        parts.append(metric.key(0, height - 30)[0])
        chains = sorted({r.chain for r in cx.residues if r.side == 0})
        if len(chains) > 1 and colour == "area":
            x = 190.0
            for n, chain in enumerate(chains):
                parts.append(f'<rect x="{x}" y="{height - 31}" width="11" height="11" rx="2" '
                             f'fill="{as_hex(np.array(CHAINS[n % 2]))}" stroke="{FAINT}" stroke-width="0.6"/>'
                             + text(x + 16, height - 22, f"chain {chain}", 9.5, QUIET))
                x += 66
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------------------- the two pages, close up
def spread(want: np.ndarray, half_w: np.ndarray, half_h, box, gap: float = 3.0, rounds: int = 600) -> np.ndarray:
    """Marks pushed off each other and kept on the page; each stays as near to where it belongs as that
    allows. ``half_h`` is one number for all of them, or one for each."""
    pos = want.copy()
    hh = np.broadcast_to(np.asarray(half_h, dtype=float), (len(pos),))
    x0, y0, x1, y1 = box
    for _ in range(rounds):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx, dy = pos[j] - pos[i]
                ox, oy = half_w[i] + half_w[j] + gap - abs(dx), hh[i] + hh[j] + gap - abs(dy)
                if ox > 0 and oy > 0:
                    moved = True
                    if ox < 2.2 * oy:
                        push = 0.5 * ox * (1.0 if dx >= 0 else -1.0)
                        pos[i, 0] -= push
                        pos[j, 0] += push
                    else:
                        push = 0.5 * oy * (1.0 if dy >= 0 else -1.0)
                        pos[i, 1] -= push
                        pos[j, 1] += push
        pos += 0.03 * (want - pos)
        pos[:, 0] = np.clip(pos[:, 0], x0 + half_w, x1 - half_w)
        pos[:, 1] = np.clip(pos[:, 1], y0 + hh, y1 - hh)
        if not moved:
            break
    return pos


#: Room around a name that nothing it must not cover may come into, and the farthest a name is put from
#: its residue (pixels of the page); a name rather goes outward from the middle than inward.
CLEAR, REACH, INWARD = 3.0, 120.0, 0.5


def along(a: np.ndarray, b: np.ndarray, step: float = 3.0) -> np.ndarray:
    """Points along a line on the page, ``step`` apart, both ends among them."""
    n = max(2, int(np.ceil(float(np.hypot(*(b - a))) / step)) + 1)
    return a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)


def place(at: np.ndarray, hw: float, hh: float, keep_clear: "list[np.ndarray]", taken, box, outward: np.ndarray
          ) -> np.ndarray:
    """Where a name goes: the place nearest its residue (``at``) at which its box covers no point of
    ``keep_clear`` and overlaps no name already placed (``taken``: centre and half width; one half
    height for all), inside ``box``. Outward counts as nearer than inward. Where no place within REACH
    is free of every point, the points of the last array are let go first, then the others; a name is
    never put on another."""
    x0, y0, x1, y1 = box
    turn = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    ways = np.stack([np.cos(turn), np.sin(turn)], axis=1)
    rings = np.arange(0.0, REACH + 0.1, 3.0)
    centres = (at + rings[:, None, None] * ways[None, :, :]).reshape(-1, 2)
    out = outward / (np.linalg.norm(outward) or 1.0)
    cost = (rings[:, None] * (1.0 + INWARD * (1.0 - ways @ out) / 2.0)[None, :]).reshape(-1)
    ok = ((centres[:, 0] >= x0 + hw) & (centres[:, 0] <= x1 - hw) & (centres[:, 1] >= y0 + hh)
          & (centres[:, 1] <= y1 - hh))
    for q, w in taken:
        ok &= ~((np.abs(centres[:, 0] - q[0]) < hw + w + 2.0) & (np.abs(centres[:, 1] - q[1]) < 2.0 * hh + 2.0))
    for kept in range(len(keep_clear), -1, -1):
        free = ok.copy()
        for points in keep_clear[:kept]:
            if len(points):
                inside = ((np.abs(points[None, :, 0] - centres[:, None, 0]) < hw + CLEAR)
                          & (np.abs(points[None, :, 1] - centres[:, None, 1]) < hh + CLEAR))
                free &= ~inside.any(axis=1)
        if free.any():
            return centres[np.where(free)[0][np.argmin(cost[free])]]
    return at


def beside(want: np.ndarray, half_w: np.ndarray, half_h: float, own: "list[np.ndarray]", common: np.ndarray, box,
           middle: np.ndarray, taken: "list | None" = None, also: "list[np.ndarray] | None" = None) -> np.ndarray:
    """Places for names, each beside its residue and not on it (:func:`place`): off its own points
    (``own``, an array for each name: its residue as pointing at it brings it up, see
    :func:`own_points`), off what comes up with it (``also``, an array for each name: its partners and
    theirs, see :func:`coming_up`), off the ``common`` points (a partner drawn as licorice,
    :func:`licorice_points`), and off every name placed before it; ``taken`` holds those of an earlier
    call and gets these. Where there is no room, the common points are let go first, then what comes
    up with it. The crowded middle first, and outward from ``middle`` rather than inward. (A name used
    to stand on its residue, and covered what pointing at it brought up.)"""
    pos, taken = want.copy(), ([] if taken is None else taken)
    for n in sorted(range(len(want)), key=lambda n: float(np.hypot(*(want[n] - middle)))):
        keep = [own[n], common] if also is None else [own[n], also[n], common]
        pos[n] = place(want[n], half_w[n], half_h, keep, taken, box, want[n] - middle)
        taken.append((pos[n], half_w[n]))
    return pos


def partners_of(cx: book.Complex, ids, kinds) -> "dict[int, set[int]]":
    """The residues among ``ids`` that a residue makes a contact of one of ``kinds`` with: those the
    script of a page marks with it while it is pointed at."""
    out: "dict[int, set[int]]" = {i: set() for i in ids}
    for c in cx.contacts:
        if c.kind in kinds and c.a in out and c.b in out:
            out[c.a].add(c.b)
            out[c.b].add(c.a)
    return out


def coming_up(i: int, partners: "dict[int, set[int]]", points: "dict[int, np.ndarray]") -> np.ndarray:
    """What comes up with a residue's name besides the residue itself, as points of the page: its
    partners, which are marked while it is pointed at, and their partners, which are marked with it
    while one of its partners is; each with its atoms and contacts (``points``, by residue: the second
    part of :func:`own_points`; a residue that has none is not drawn on this page)."""
    mine = partners.get(i, set())
    near = (mine | {s for p in mine for s in partners.get(p, set())}) - {i}
    got = [points[x] for x in sorted(near) if x in points]
    return np.concatenate(got) if got else np.zeros((0, 2))


def own_points(cx: book.Complex, i: int, view: np.ndarray, px, look) -> "tuple[np.ndarray, np.ndarray]":
    """What pointing at a residue brings up, as points of the page: its atoms that :func:`under` draws,
    and, with them, its contacts of a kind ``look`` names, from atom to atom. Comes back as the atoms,
    and the atoms with the contacts."""
    atoms = np.array([px(view[a]) for a in drawn_atoms(cx, i, look)], dtype=float).reshape(-1, 2)
    lines = [along(px(view[list(c.atoms_a if c.a == i else c.atoms_b)].mean(axis=0)),
                   px(view[list(c.atoms_b if c.a == i else c.atoms_a)].mean(axis=0)))
             for c in cx.contacts if i in (c.a, c.b) and c.kind in look and c.atoms_a and c.atoms_b]
    return atoms, (np.concatenate([atoms, *lines]) if lines else atoms)


def licorice_points(cx: book.Complex, which: np.ndarray, view: np.ndarray, px) -> np.ndarray:
    """The atoms of a partner drawn as licorice (``which``: a mask of the atoms) and its bonds, as points
    of the page."""
    atoms = np.where(which)[0]
    at = np.array([px(view[a]) for a in atoms], dtype=float).reshape(-1, 2)
    bonds = [along(at[m], at[n]) for m in range(len(atoms)) for n in range(m + 1, len(atoms))
             if 0.4 < float(np.linalg.norm(cx.xyz[atoms[m]] - cx.xyz[atoms[n]])) < 1.95]
    return np.concatenate([at, *bonds]) if bonds else at


def hairline(q: np.ndarray, hw: float, hh: float, atoms: np.ndarray, edge: bool = False) -> str:
    """A hairline from a name (middle ``q``, half its width and height) to the nearest of ``atoms``,
    where that one is not under the name. From the middle, for a name with a box that covers the start;
    with ``edge`` from the edge, for a name of letters only, whose halo would not hide the line."""
    if not len(atoms):
        return ""
    to = atoms[int(np.argmin(np.hypot(*(atoms - q).T)))]
    if abs(float(to[0] - q[0])) <= hw + 2.0 and abs(float(to[1] - q[1])) <= hh + 2.0:
        return ""
    start = q
    if edge:
        d = to - q
        start = q + min(hw / abs(float(d[0])) if d[0] else np.inf, hh / abs(float(d[1])) if d[1] else np.inf, 1.0) * d
    return (f'<line x1="{start[0]:.1f}" y1="{start[1]:.1f}" x2="{to[0]:.1f}" y2="{to[1]:.1f}" stroke="{INK}" '
            f'stroke-width="0.6"/>')


def routed(cx: book.Complex, contacts, marks, casing: bool = False, wrap: float = 0.0, width: float = 0.0
           ) -> "list[str]":
    """The lines of these contacts. ``marks`` says of a residue (by its index) where the middle of its
    mark is and half its width and height. A line is led around every mark but the two it joins
    (:mod:`route`): one that ran under a third mark would read as a contact of the third. On a sheet
    that is ``wrap`` wide once around and ``width`` wide as drawn, a pair that is nearer across the slit
    gets a stub from each mark to its edge of the sheet.
    """
    ids = list(marks)
    number = {i: n for n, i in enumerate(ids)}
    lanes = route.routers([(marks[i][0][0], marks[i][0][1], marks[i][1], marks[i][2]) for i in ids])
    out = []
    for n, c in enumerate(contacts):
        (pa, wa, ha), (pb, wb, hb) = marks[c.a], marks[c.b]
        pa, pb = np.asarray(pa, dtype=float), np.asarray(pb, dtype=float)
        colour, dash, stroke = STROKES[c.kind]
        look = (f'class="link" data-a="{cx.residues[c.a].key}" data-b="{cx.residues[c.b].key}" fill="none" '
                f'stroke="{colour}" stroke-width="{max(stroke, 0.8)}"' + (f' stroke-dasharray="{dash}"' if dash else ""))
        said = (f'<title>{cx.residues[c.a].label} – {cx.residues[c.b].label}: {KIND[c.kind]}, {c.distance:.1f} Å'
                f'</title>')
        if wrap and abs(pa[0] - pb[0]) > wrap / 2.0:
            for here, there, hw, hh, round_ in ((pa, pb, wa, ha, False), (pb, pa, wb, hb, True)):
                turn = wrap if there[0] < here[0] else -wrap             # the partner, once around
                ghost = there + np.array([turn, 0.0])
                edge = width if turn > 0 else 0.0
                reach = float(np.clip((edge - here[0]) / (ghost[0] - here[0]), 0.0, 1.0))
                start, end = route.leave(here, ghost, hw, hh, round_), here + reach * (ghost - here)
                out.append(f'<path {look} d="M {start[0]:.1f} {start[1]:.1f} L {end[0]:.1f} {end[1]:.1f}">{said}</path>'
                           f'<circle cx="{end[0]:.1f}" cy="{end[1]:.1f}" r="2.2" fill="{colour}"/>')
            continue
        way = None
        for lane in (lanes[n % len(lanes)], lanes[0]):
            way = lane.way(pa, pb, own=(number[c.a], number[c.b]))
            if way is not None:
                break
        way = [pa, pb] if way is None else way
        way[0], way[-1] = route.leave(pa, way[1], wa, ha, False), route.leave(pb, way[-2], wb, hb, True)
        d = route.path(way)
        if casing:
            out.append(f'<path d="{d}" fill="none" stroke="#ffffff" stroke-opacity="0.75" stroke-width="{stroke + 1.8}"/>')
        out.append(f'<path {look} d="{d}">{said}</path>')
    return out


def pages(cx: book.Complex, mode: str = "surface", kinds=_KINDS["all"], scale: float = 11.0, margin: float = 6.5,
          colour: str = "area", regions: bool = True, names: str = "contacts") -> str:
    """The two faces close up, side by side. ``mode``: ``"surface"`` (the shaded surface carries the
    colour, the marks are names) or ``"boxes"`` (a flat outline, marks that carry the colour). ``regions``
    draws the borders between the parts of the interface and the same numeral in a part on both pages."""
    metric = metric_of(cx, colour)
    opened = cx.opened()
    ids = (cx.at_interface(0), cx.at_interface(1))
    at = {i: opened[cx.residues[i].near].mean(axis=0) for i in ids[0] + ids[1]}
    ys = [at[i][1] for i in ids[0] + ids[1]]
    y0, y1 = min(ys) - margin, max(ys) + margin
    gutter, head, foot = 64.0, 30.0, 78.0
    colours = atom_colours(cx, metric if mode == "surface" else None)
    shown = named(cx, names, kinds)
    size = 10.0 if mode == "boxes" else 9.5
    parts, placed, origin_x = [], {}, 0.0
    spans, badges = [], []
    for s in (0, 1):
        xs = [at[i][0] for i in ids[s]]
        x0, x1 = min(xs) - margin, max(xs) + margin
        src, w, h, front = picture(opened, cx, colours, (x0, x1, y0, y1), scale, which=cx.side == s,
                                   flat=0.0 if mode == "surface" else 1.0, fade=2.0, regions=regions)
        parts.append(f'<image href="{src}" x="{origin_x:.1f}" y="{head}" width="{w}" height="{h}"'
                     + (' opacity="0.55"' if mode == "boxes" else "") + "/>")
        mine = [i for i in ids[s] if i in shown]
        many = len({cx.residues[i].chain for i in ids[s]}) > 1
        sizes = np.array([mark_size(cx.residues[i], size, many, 4.0) for i in mine]).reshape(-1, 2)
        want = np.array([[origin_x + (at[i][0] - x0) * scale, head + (y1 - at[i][1]) * scale] for i in mine]
                        ).reshape(-1, 2)
        pos = spread(want, sizes[:, 0], sizes[:, 1], (origin_x + 2, head + 2, origin_x + w - 2, head + h - 2))
        for n, i in enumerate(mine):
            placed[i] = (pos[n], want[n], sizes[n, 0], sizes[n, 1], many)
        # A place to point at for a residue that has no name here.
        for i in ids[s]:
            if i not in shown:
                r = cx.residues[i]
                parts.append(f'<circle cx="{origin_x + (at[i][0] - x0) * scale:.1f}" '
                             f'cy="{head + (y1 - at[i][1]) * scale:.1f}" r="9" fill="#000" fill-opacity="0">'
                             f'<title>{r.label} (chain {r.chain}): {r.buried:.0f} Å² covered, no contact that is '
                             f'drawn</title></circle>')
        if regions and front is not None:
            middles = {}
            for g in range(cx.n_regions):
                pts = [[origin_x + (at[i][0] - x0) * scale, head + (y1 - at[i][1]) * scale]
                       for i in ids[s] if cx.residues[i].region == g]
                if pts:
                    middles[g] = np.mean(pts, axis=0)
            boxes = [(pos[n][0], pos[n][1], sizes[n, 0], sizes[n, 1]) for n in range(len(pos))]
            badges += numerals(cx, s, middles, boxes, (origin_x, head, origin_x + w, head + h), front)
        spans.append((origin_x, w, h))
        origin_x += w + gutter
    width, height = max(origin_x - gutter, 600.0), head + spans[0][2] + foot
    spine_x = spans[0][0] + spans[0][1] + gutter / 2.0
    spine = (f'<line x1="{spine_x:.1f}" y1="{head + 4}" x2="{spine_x:.1f}" y2="{head + spans[0][2] - 4:.1f}" '
             f'stroke="{FAINT}" stroke-width="0.8" stroke-dasharray="5 3"/>')

    order = {"near": 0, "hydrophobic": 1, "hbond": 2, "saltbridge": 3}
    drawn = [c for c in sorted(cx.contacts, key=lambda c: order[c.kind])
             if c.kind in kinds and c.a in placed and c.b in placed]
    lines = routed(cx, drawn, {i: (v[0], v[2], v[3]) for i, v in placed.items()}, casing=mode == "surface")
    marks = []
    for i, (q, want_i, hw, hh, many) in placed.items():
        r = cx.residues[i]
        own = metric.of(r) if mode == "boxes" else None
        fill = np.ones(3) if own is None else own
        lead = ""
        if float(np.hypot(*(q - want_i))) > hh + 4.0:      # a mark that had to give way points to its place
            lead = (f'<line x1="{q[0]:.1f}" y1="{q[1]:.1f}" x2="{want_i[0]:.1f}" y2="{want_i[1]:.1f}" stroke="{INK}" '
                    f'stroke-width="0.6"/><circle cx="{want_i[0]:.1f}" cy="{want_i[1]:.1f}" r="1.7" fill="{INK}"/>')
        where = f", region {ROMAN[r.region]}" if regions and r.region >= 0 else ""
        marks.append(mark(r, q, hw, hh, fill, size,
                          f'{r.label} (chain {r.chain}): {r.buried:.0f} Å² covered, {r.partners} partner'
                          f'{"s" if r.partners != 1 else ""}{where}',
                          opacity="1" if mode == "boxes" else "0.86", chain=many, lead=lead))
    heads = []
    for s, (ox, w, h) in enumerate(spans):
        heads.append(text(ox + w / 2.0, 16, html.escape(cx.names[s]), 11.5, INK, "middle", "600"))
    key, x = metric.key(0, height - 46)
    key += line_key(x + 16, height - 37, [k for k in ("saltbridge", "hbond", "hydrophobic", "near") if k in kinds])[0]
    key += shape_key(0, height - 4, cx)[0]
    scale_bar = (f'<line x1="{width - 10 * scale:.1f}" y1="{height - 56}" x2="{width:.1f}" y2="{height - 56}" '
                 f'stroke="{INK}" stroke-width="1.2"/>' + text(width, height - 60, "10 Å", 9.5, QUIET, "end"))
    return (f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
            f'width="{width:.0f}" font-family="{FONT}">' + "".join(parts) + spine + "".join(heads)
            + '<g class="links">' + "".join(lines) + "</g>" + "".join(marks) + "".join(badges) + key + scale_bar
            + "</svg>")


# ---------------------------------------------------------------------------------------- one face, with the partner on it
def frame_of(cx: book.Complex, margin: float = 6.0, room: float = 980.0, gap: float = 26.0):
    """The frame the third and fourth panel share: A's face, with B where it sits. Comes back as every
    atom in that frame, where each residue of the interface lies in it, its bounds, and a scale at which
    two panels fit side by side."""
    f = book.faced(cx)
    at = {i: f[cx.residues[i].near][:, :2].mean(axis=0) for i in cx.at_interface(0) + cx.at_interface(1)}
    xs, ys = [p[0] for p in at.values()], [p[1] for p in at.values()]
    bounds = (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)
    return f, at, bounds, min(13.0, (room - gap) / 2.0 / (bounds[1] - bounds[0]))


def stick_colours(cx: book.Complex, sticks: str = "element") -> np.ndarray:
    """A colour for every atom where it is drawn as licorice. ``"element"``: nitrogen blue, oxygen red,
    sulfur yellow, and carbon in the colour of its partner. Anything else is a name :func:`metric_of`
    takes: every atom of a residue of the second partner in the colour that gives the residue (grey
    where it says nothing of it); the first partner's stay by element, so the two are still told apart."""
    col = np.array([surface.ELEMENT.get(e.upper(), CARBON if s else CARBON_A) for e, s in zip(cx.elements, cx.side)],
                   dtype=float)
    if sticks != "element":
        by = metric_of(cx, sticks)
        for r in cx.residues:
            if r.side == 1:
                own = by.of(r)
                col[r.atoms] = np.array(NO_VALUE) if own is None else own
    return col


def face(cx: book.Complex, f, at, bounds, scale: float, colour: str, names: str, kinds, regions: bool,
         letter: str = "", sticks: str = "element") -> str:
    """The face of A with the atoms of B on it as licorice: the whole of B where B is small, the residues
    of B at the interface where it is not. Hydrogen bonds and salt bridges run from atom to atom.
    ``sticks`` says what that licorice is coloured by (:func:`stick_colours`).

    Every residue that has a name on the flat map is here too, as licorice in the drawing, and is not
    seen until it is pointed at, there or here, or one of its partners is: then it stands on the
    picture with its name and with every contact it makes from atom to atom, and the picture steps back.
    """
    x0, x1, y0, y1 = bounds
    head = 30.0
    b_residues = [r for r in cx.residues if r.side == 1]
    whole = len(b_residues) <= WHOLE
    drawn = np.zeros(len(f), dtype=bool)
    for r in b_residues:
        if whole or len(r.near):
            drawn[r.atoms] = True
    tints = stick_colours(cx, sticks)
    licorice = surface.sticks(f[drawn], [e for e, k in zip(cx.elements, drawn) if k], CARBON, colours=tints[drawn])
    src, w, h, front = picture(f, cx, atom_colours(cx, metric_of(cx, colour)), bounds, scale, which=cx.side == 0,
                               fade=2.0, regions=regions, licorice=licorice)

    def px(p) -> np.ndarray:
        return np.array([(p[0] - x0) * scale, head + (y1 - p[1]) * scale])

    # Hydrogen bonds and salt bridges, from the atom of B to the atom of A, which is a dot on the surface.
    bonds = []
    for c in cx.contacts:
        if c.kind not in kinds or c.kind not in _KINDS["polar"] or not c.atoms_a or not c.atoms_b:
            continue
        pa, pb = px(f[list(c.atoms_a)].mean(axis=0)), px(f[list(c.atoms_b)].mean(axis=0))
        stroke, dash, width = STROKES[c.kind]
        element = cx.elements[c.atoms_a[0]].upper()
        dot = as_hex(np.array(surface.ELEMENT.get(element, (0.35, 0.38, 0.42))))
        bonds.append(
            f'<g class="link" data-a="{cx.residues[c.a].key}" data-b="{cx.residues[c.b].key}">'
            f'<line x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" stroke="#ffffff" '
            f'stroke-opacity="0.8" stroke-width="{width + 2.0}" stroke-linecap="round"/>'
            f'<line x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" stroke="{stroke}" '
            f'stroke-width="{width + 0.2}" stroke-dasharray="{dash or "none"}"/>'
            f'<circle cx="{pa[0]:.1f}" cy="{pa[1]:.1f}" r="2.6" fill="{dot}" stroke="#ffffff" stroke-width="1"/>'
            f'<title>{cx.residues[c.a].label} – {cx.residues[c.b].label}: {KIND[c.kind]}, {c.distance:.1f} Å</title>'
            f'</g>')

    # The names of B's residues, beside their atoms and not on them (:func:`beside`): pointing at one
    # brings up its atoms and contacts, and they are not to be under its name. A's residues are named on
    # the flat map next to this picture, and here only while they are pointed at.
    shown = [i for i in sorted(named(cx, names, kinds)) if cx.residues[i].side == 1]
    size, half_h = 9.5, 7.2
    several = len({r.chain for r in b_residues}) > 1
    labels = [cx.residues[i].label + (f" {cx.residues[i].chain}" if several else "") for i in shown]
    half_w = np.array([len(t) * size * 0.29 + 3.0 for t in labels])
    want = np.array([px(f[cx.residues[i].atoms][:, :2].mean(axis=0)) for i in shown]).reshape(-1, 2)
    look = {k: STROKES[k][:2] for k in kinds}
    page = (2, head + 2, w - 2, head + h - 2)
    sticks_at = licorice_points(cx, drawn, f, px)
    middle = sticks_at[:int(drawn.sum())].mean(axis=0) if drawn.any() else np.array([w / 2.0, head + h / 2.0])
    ids = sorted(named(cx, names, kinds))
    full = {i: own_points(cx, i, f, px, look) for i in ids}
    partners, with_lines = partners_of(cx, ids, kinds), {i: v[1] for i, v in full.items()}
    taken: list = []
    pos = beside(want, half_w, half_h, [full[i][1] for i in shown], sticks_at, page, middle, taken,
                 also=[coming_up(i, partners, with_lines) for i in shown])
    halo = 'stroke="#ffffff" stroke-width="3" stroke-opacity="0.9" paint-order="stroke" stroke-linejoin="round"'
    tags = []
    for n, i in enumerate(shown):
        r, p = cx.residues[i], pos[n]
        tags.append(f'<g class="res" data-id="{r.key}">{hairline(p, half_w[n], half_h, full[i][0], edge=True)}'
                    f'<text x="{p[0]:.1f}" y="{p[1] + size * 0.36:.1f}" '
                    f'font-size="{size}" font-weight="600" fill="{INK}" text-anchor="middle" {halo}>{labels[n]}'
                    f'</text><title>{r.label} (chain {r.chain}): {r.buried:.0f} Å² covered</title></g>')
    # What is there for the pointer alone. Every residue the flat map names, as licorice; every contact
    # that is a line there, bold, from atom to atom; and a name for A's residues, which have none here,
    # placed after B's and off them too.
    # A residue's own atoms and contacts, and its partners' atoms: not its partners' other contacts.
    bodies = [f'<g class="res" data-id="{cx.residues[i].key}">{under(cx, i, f, px, look, thick=tints, reach=False)}</g>'
              for i in ids]
    reach = [reaching(cx, c, f, px, look) for c in cx.contacts
             if c.kind in kinds and c.atoms_a and c.atoms_b and c.a in ids and c.b in ids]
    theirs = [i for i in ids if cx.residues[i].side == 0]
    wide = np.array([len(cx.residues[i].label) * size * 0.29 + 3.0 for i in theirs])
    near = np.array([px(f[cx.residues[i].near][:, :2].mean(axis=0)) for i in theirs]).reshape(-1, 2)
    spots = beside(near, wide, half_h, [full[i][1] for i in theirs], sticks_at, page, middle, taken,
                   also=[coming_up(i, partners, with_lines) for i in theirs])
    ghosts = []
    for n, i in enumerate(theirs):
        r, q = cx.residues[i], spots[n]
        ghosts.append(f'<g class="res" data-id="{r.key}"><g class="reach">'
                      f'{hairline(q, wide[n], half_h, full[i][0], edge=True)}<text x="{q[0]:.1f}" '
                      f'y="{q[1] + size * 0.36:.1f}" font-size="{size}" font-weight="600" fill="{EDGE[0]}" '
                      f'text-anchor="middle" {halo}>{r.label}</text></g></g>')
    badges = []
    if regions and front is not None:
        middles = {}
        for g in range(cx.n_regions):
            pts = [px(at[i]) for i in cx.at_interface(0) if cx.residues[i].region == g]
            if pts:
                middles[g] = np.mean(pts, axis=0)
        boxes = [(pos[n][0], pos[n][1], half_w[n], half_h) for n in range(len(pos))]
        badges = numerals(cx, 0, middles, boxes, (0, head, w, head + h), front, radius=8.0, points=sticks_at)
    what = "all of" if whole else "the residues at the interface of"
    title = heading(0, 14, letter, f"the face of {html.escape(cx.names[0])}, with the atoms of "
                                   f"{html.escape(cx.names[1])} on it")
    return (f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {head + h:.0f}" width="{w}" '
            f'font-family="{FONT}">{title}<image class="ground" href="{src}" x="0" y="{head}" width="{w}" '
            f'height="{h}"><title>{what} {html.escape(cx.names[1])} as licorice</title></image>' + "".join(badges)
            + "".join(bonds) + "".join(bodies) + "".join(reach) + "".join(ghosts) + "".join(tags) + "</svg>")


def sheet(cx: book.Complex, f, at, bounds, scale: float, colour: str, names: str, kinds, regions: bool,
          letter: str = "", key: bool = True, background=None, title: str = "the same face as a flat map of residues",
          wrap: float = 0.0) -> str:
    """One sheet with both partners on it: a box for a residue of the first, an ellipse for a residue of
    the second, laid where it sits. Partners are neighbours here, so a contact is a short line. A pale
    band runs from a residue to the next one of its chain, so a loop or a strand is seen as one.

    ``background`` is a picture to lie under it (address, width, height); without one it is the outline
    of the first partner in the frame ``f``. ``wrap`` is the width of a sheet that is slit open and
    closes on itself (see :func:`unrolled`).
    """
    metric = metric_of(cx, colour)
    x0, x1, y0, y1 = bounds
    head, foot = 30.0, (84.0 if key else 0.0)
    if background is None:
        background = picture(f, cx, atom_colours(cx, None), bounds, scale, which=cx.side == 0, flat=1.0, fade=2.0)[:3]
    src, w, h = background
    ids = sorted(named(cx, names, kinds))
    size = 10.0
    many = [len({cx.residues[i].chain for i in ids if cx.residues[i].side == s}) > 1 for s in (0, 1)]
    sizes = np.array([mark_size(cx.residues[i], size, many[cx.residues[i].side], 5.0) for i in ids]).reshape(-1, 2)
    half_w, half_h = sizes[:, 0], sizes[:, 1]
    # A a little up, B a little down: two that lie on top of each other become a pair to read.
    want = np.array([[(at[i][0] - x0) * scale, head + (y1 - at[i][1]) * scale
                      + (-0.62 if cx.residues[i].side == 0 else 0.62) * (half_h[n] + 1.5)]
                     for n, i in enumerate(ids)]).reshape(-1, 2)
    pos = spread(want, half_w, half_h, (2, head + 2, w - 2, head + h - 2), gap=2.5)
    where = {i: n for n, i in enumerate(ids)}

    # The chain, from a residue to the next (a band) or the next but one (dotted): pale, and under everything.
    traces = []
    for s in (0, 1):
        mine = sorted((i for i in ids if cx.residues[i].side == s),
                      key=lambda i: (cx.residues[i].chain, cx.residues[i].number))
        for i, j in zip(mine, mine[1:]):
            ri, rj = cx.residues[i], cx.residues[j]
            step = rj.number - ri.number
            if ri.chain != rj.chain or step > 2:
                continue
            pa, pb = pos[where[i]], pos[where[j]]
            if wrap and (abs(pa[0] - pb[0]) > wrap / 2.0 or (s == 0 and float(np.hypot(*(pa - pb))) > 12.0 * scale)):
                continue
            dash = ' stroke-dasharray="1 9"' if step == 2 else ""
            traces.append(f'<line x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" '
                          f'stroke="{TRACE[s]}" stroke-width="{4 if wrap else 6}" stroke-linecap="round"{dash}><title>'
                          f'{ri.label} and {rj.label}: {"next to each other" if step == 1 else "one residue between"} '
                          f'in the chain</title></line>')
    rings, badges = [], []
    if regions and ids:
        boxes_of: dict = {}
        for i in ids:
            n = where[i]
            boxes_of.setdefault(cx.residues[i].region, []).append((pos[n][0], pos[n][1], half_w[n], half_h[n]))
        land, lw, lh, who = lands(boxes_of, w, h, head)
        rings = [f'<image href="{land}" x="0" y="{head}" width="{lw}" height="{lh}"/>']
        middles = {g: np.mean([[b[0], b[1]] for b in boxes], axis=0) for g, boxes in boxes_of.items()}
        badges = numerals(cx, 0, middles, [b for boxes in boxes_of.values() for b in boxes], (0, head, w, head + h),
                          who, radius=8.0)

    order = {"near": 0, "hydrophobic": 1, "hbond": 2, "saltbridge": 3}
    drawn = [c for c in sorted(cx.contacts, key=lambda c: order[c.kind])
             if c.kind in kinds and c.a in where and c.b in where]
    lines = routed(cx, drawn, {i: (pos[n], half_w[n], half_h[n]) for i, n in where.items()}, wrap=wrap, width=w)
    marks = []
    for n, i in enumerate(ids):
        r = cx.residues[i]
        own = metric.of(r)
        marks.append(mark(r, pos[n], half_w[n], half_h[n], np.ones(3) if own is None else own, size,
                          f'{r.label} (chain {r.chain}): {r.buried:.0f} Å² covered, region {ROMAN[r.region]}',
                          chain=many[r.side]))
    width, height = (max(w, 660.0) if key else w), head + h + foot
    legend = ""
    if key:
        legend, x = metric.key(0, height - 52)
        legend += line_key(x + 16, height - 43, kinds)[0]
        shapes, x = shape_key(0, height - 6, cx)
        legend += shapes + ribbon_key(x + 20, height - 6)[0]
    return (f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
            f'width="{width:.0f}" font-family="{FONT}">{heading(0, 14, letter, title)}<image href="{src}" x="0" '
            f'y="{head}" width="{w}" height="{h}" opacity="0.5"/>' + "".join(rings) + "".join(traces)
            + '<g class="links">' + "".join(lines) + "</g>" + "".join(marks) + "".join(badges) + legend + "</svg>")


def size_of(svg: str) -> "tuple[float, float]":
    m = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg)
    assert m, "a figure without a size"
    return float(m.group(1)), float(m.group(2))


def nest(svg: str, x: float, y: float) -> str:
    """A whole figure as one panel of a larger one."""
    w, h = size_of(svg)
    return re.sub(r'^<svg class="book" xmlns="[^"]+" viewBox="[^"]+" width="[^"]+"',
                  f'<svg x="{x:.1f}" y="{y:.1f}" width="{w:g}" height="{h:g}" viewBox="0 0 {w:g} {h:g}"', svg, count=1)


def matrix(cx: book.Complex, cell: float = 14.0) -> str:
    """Who touches whom, in full: a row per residue of A, a column per residue of B, region by region and,
    inside a region, in the order along the spine. A region of the interface is a block on the diagonal."""
    top = top_of(cx)
    shut = cx.closed()
    height_of = {i: float(shut[cx.residues[i].near, 1].mean()) for i in cx.at_interface(0) + cx.at_interface(1)}
    rows = sorted(cx.at_interface(0), key=lambda i: (cx.residues[i].region, -height_of[i]))
    cols = sorted(cx.at_interface(1), key=lambda i: (cx.residues[i].region, -height_of[i]))
    left, head = 96.0, 78.0
    grid_w, grid_h = len(cols) * cell, len(rows) * cell
    w, h = left + grid_w + 34, head + grid_h + 78
    out = [f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {max(w, 600):.0f} {h:.0f}" '
           f'width="{max(w, 600):.0f}" font-family="{FONT}">']
    # The regions: a block on the diagonal each, with its numeral at the right and below.
    for g in range(cx.n_regions):
        rr = [n for n, i in enumerate(rows) if cx.residues[i].region == g]
        cc = [n for n, i in enumerate(cols) if cx.residues[i].region == g]
        if rr and cc:
            out.append(f'<rect x="{left + cc[0] * cell:.1f}" y="{head + rr[0] * cell:.1f}" '
                       f'width="{len(cc) * cell:.1f}" height="{len(rr) * cell:.1f}" fill="#eef2f6" stroke="{QUIET}" '
                       f'stroke-width="0.8" stroke-dasharray="4 3"/>')
        if rr:
            out.append(numeral(left + grid_w + 16, head + (rr[0] + len(rr) / 2.0) * cell, g))
        if cc:
            out.append(numeral(left + (cc[0] + len(cc) / 2.0) * cell, head + grid_h + 16, g))
    many = len({cx.residues[i].chain for i in rows}) > 1
    for n, i in enumerate(rows):
        r, y = cx.residues[i], head + n * cell
        out.append(f'<g class="res" data-id="{r.key}"><rect x="{left - 13}" y="{y + 2:.1f}" width="9" '
                   f'height="{cell - 4}" rx="2" fill="{as_hex(ramp(r.buried / top))}" stroke="{FAINT}" '
                   f'stroke-width="0.4"/>' + text(left - 18, y + cell * 0.72, r.label + (f" {r.chain}" if many else ""),
                                                 9.5, INK, "end") + "</g>")
    for n, i in enumerate(cols):
        r, x = cx.residues[i], left + n * cell
        out.append(f'<g class="res" data-id="{r.key}"><rect x="{x + 2:.1f}" y="{head - 13}" width="{cell - 4}" '
                   f'height="9" rx="2" fill="{as_hex(ramp(r.buried / top))}" stroke="{FAINT}" stroke-width="0.4"/>'
                   + text(0, 0, r.label, 9.5, INK, "start",
                          extra=f' transform="translate({x + cell * 0.72:.1f} {head - 18}) rotate(-55)"') + "</g>")
    for n in range(len(rows) + 1):
        out.append(f'<line x1="{left}" y1="{head + n * cell:.1f}" x2="{left + grid_w}" y2="{head + n * cell:.1f}" '
                   f'stroke="#e4e7eb" stroke-width="0.6"/>')
    for n in range(len(cols) + 1):
        out.append(f'<line x1="{left + n * cell:.1f}" y1="{head}" x2="{left + n * cell:.1f}" y2="{head + grid_h}" '
                   f'stroke="#e4e7eb" stroke-width="0.6"/>')
    row_of, col_of = {i: n for n, i in enumerate(rows)}, {i: n for n, i in enumerate(cols)}
    for c in cx.contacts:
        stroke = STROKES[c.kind][0]
        x, y = left + col_of[c.b] * cell, head + row_of[c.a] * cell
        strong = c.kind in ("saltbridge", "hbond")
        out.append(f'<rect class="link" data-a="{cx.residues[c.a].key}" data-b="{cx.residues[c.b].key}" '
                   f'x="{x + 1.5:.1f}" y="{y + 1.5:.1f}" width="{cell - 3}" height="{cell - 3}" rx="2.5" '
                   f'fill="{stroke}" fill-opacity="{1.0 if strong else 0.55}"><title>{cx.residues[c.a].label} – '
                   f'{cx.residues[c.b].label}: {KIND[c.kind]}, {c.distance:.1f} Å</title></rect>')
    out.append(text(left - 18, head - 26, html.escape(cx.names[0]) + " ↓", 10.5, INK, "end", "600"))
    out.append(text(left, 12, html.escape(cx.names[1]) + " →", 10.5, INK, "start", "600"))
    x = 0.0
    for kind in ("saltbridge", "hbond", "hydrophobic", "near"):
        out.append(f'<rect x="{x}" y="{h - 26}" width="11" height="11" rx="2.5" fill="{STROKES[kind][0]}" '
                   f'fill-opacity="{1.0 if kind in ("saltbridge", "hbond") else 0.55}"/>'
                   + text(x + 16, h - 17, KIND[kind], 9.5, QUIET))
        x += 30 + 5.3 * len(KIND[kind])
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------------------- the three figures
def facts_of(cx: book.Complex, names: str, kinds) -> dict:
    from collections import Counter

    n = Counter(c.kind for c in cx.contacts)
    shown = named(cx, names, kinds)
    at = (cx.at_interface(0), cx.at_interface(1))
    return {
        "names": list(cx.names),
        "at_the_interface": [len(at[0]), len(at[1])],
        "named": [sum(1 for i in at[0] if i in shown), sum(1 for i in at[1] if i in shown)],
        "area": round(sum(cx.residues[i].buried for i in at[0] + at[1])),
        "pairs": len(cx.contacts),
        "kinds": {k: n[k] for k in ("saltbridge", "hbond", "hydrophobic", "near")},
        "lines": sum(1 for c in cx.contacts if c.kind in kinds),
        "regions": cx.n_regions,
        "between_regions": sum(1 for c in cx.contacts if cx.residues[c.a].region != cx.residues[c.b].region),
        "colours": list(colours_of(cx)),
        "ipsae": [round(v, 3) for v in cx.ipsae_both] if cx.ipsae_both else None,
        "models": cx.models,
    }


def panels(cx: book.Complex, colour: str = "area", names: str = "contacts", lines: str = "all",
           regions: bool = True, sticks: str = "element") -> Sketch:
    """Four panels. **a** the complex as it stands, **b** opened like a book, **c** the face of the first
    partner with the atoms of the second on it as licorice, **d** the same face as a flat map of residues.
    Panels c and d are the same frame at the same scale, so a residue is at the same place in both, and
    a residue pointed at in d stands in c as licorice, with its partners and its contacts.

    ``colour``: what a residue is coloured by, see :func:`metric_of` (``"area"``, ``"class"``,
    ``"region"``, ``"contacts"``, and for a model what it says of itself). ``names``: which residues
    are named, see :func:`named`. ``lines``: ``"all"`` contacts of a kind, or the ``"polar"`` ones only.
    ``regions``: the parts of the interface, with the same numeral in all panels. ``sticks``: what the
    licorice of the second partner in panel c is coloured by: ``"element"``, or anything ``colour``
    takes (``"plddt"`` and ``"pae"`` for a model, ``"class"``, ``"area"``...), a residue in one colour.
    """
    kinds = _KINDS[_one_of(lines, LINES, "lines")]
    _one_of(names, NAMES, "names")
    _one_of(sticks, STICKS + colours_of(cx), "sticks")
    metric = metric_of(cx, colour)
    first = overview(cx, colour, regions, letters=("a", "b"), key=False)
    f, at, bounds, scale = frame_of(cx)
    third = face(cx, f, at, bounds, scale, colour, names, kinds, regions, letter="c", sticks=sticks)
    fourth = sheet(cx, f, at, bounds, scale, colour, names, kinds, regions, letter="d", key=False)
    (w1, h1), (w3, h3), (w4, h4) = size_of(first), size_of(third), size_of(fourth)
    gap = 26.0
    width = max(w1, w3 + gap + w4, 760.0)
    y2 = h1 + 10.0
    y3 = y2 + max(h3, h4) + 34.0
    key, x = metric.key(0, y3)
    more, x = line_key(x + 16, y3 + 9, kinds)
    # What the licorice of panel c is: by element, or a residue in the colour of a key (its own, where
    # that is not the key of the surface).
    b_name, row, own_key = html.escape(cx.names[1]), y3 + 40.0, ""
    if sticks == "element":
        said = f"atoms of {b_name} (carbon; nitrogen blue, oxygen red, sulfur yellow)"
        swatch = as_hex(np.array(CARBON))
    elif sticks == colour:
        said = f"atoms of {b_name}: a residue in the colour the key above gives it"
        swatch = as_hex(np.array(NO_VALUE))
    else:
        said = f"atoms of {b_name}: a residue in the colour the key under this line gives it (grey: no value)"
        swatch = as_hex(np.array(NO_VALUE))
        own_key = metric_of(cx, sticks).key(28.0, row + 24.0)[0]
    marks = (f'<line x1="0" y1="{row - 3}" x2="22" y2="{row - 3}" stroke="{swatch}" stroke-width="4.5" '
             f'stroke-linecap="round"/>' + text(28, row, said, 9.5, QUIET) + own_key)
    if own_key:
        row += 66.0
        marks += ribbon_key(0, row)[0]
    else:
        marks += ribbon_key(52 + 4.9 * len(said), row)[0]
    marks += shape_key(0, row + 22, cx)[0]
    bar = (f'<line x1="{w3 + gap + w4 - 10 * scale:.1f}" y1="{y3 - 14}" x2="{w3 + gap + w4:.1f}" y2="{y3 - 14}" '
           f'stroke="{INK}" stroke-width="1.2"/>' + text(w3 + gap + w4, y3 - 18, "10 Å", 9.5, QUIET, "end"))
    height = row + 32.0
    svg = (f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
           f'width="{width:.0f}" font-family="{FONT}">' + nest(first, 0, 0) + nest(third, 0, y2)
           + nest(fourth, w3 + gap, y2) + bar + key + more + marks + "</svg>")
    return Sketch(svg, "four panels", facts_of(cx, names, kinds))


def faces(cx: book.Complex, colour: str = "area", names: str = "contacts", lines: str = "all",
          regions: bool = True, surface_: bool = True) -> Sketch:
    """The two opened faces close up. With ``surface_`` the shaded face carries the colour and the marks
    are names; without it the face is an outline and the marks carry the colour. The other options are
    those of :func:`panels`."""
    kinds = _KINDS[_one_of(lines, LINES, "lines")]
    _one_of(names, NAMES, "names")
    svg = pages(cx, "surface" if surface_ else "boxes", kinds, colour=colour, regions=regions, names=names)
    return Sketch(svg, "the opened faces", facts_of(cx, names, kinds))


def table(cx: book.Complex) -> Sketch:
    """Who touches whom, in full: every pair of residues with two atoms within 4 Å, with or without a kind."""
    return Sketch(matrix(cx), "who touches whom", facts_of(cx, "touching", _KINDS["all"]))


def unrolled(cx: book.Complex, colour: str = "area", names: str = "contacts", lines: str = "all",
             regions: bool = True) -> Sketch:
    """The wall around the second partner, slit open along that partner's own path and laid flat.

    For a peptide that lies inside its partner, or whose path bends, so that no plane shows the
    interface. Down the sheet runs the peptide from its first residue to its last, across the sheet the
    angle around it; the wall is seen from the peptide. A residue of the peptide lies where it touches
    the wall, so its partners are its neighbours on the sheet. The left and the right edge are one
    line, the slit, which is put where the wall has its widest gap; a pair that is nearer across the
    slit gets a stub from each mark to its edge. The other options are those of :func:`panels`.
    """
    kinds = _KINDS[_one_of(lines, LINES, "lines")]
    _one_of(names, NAMES, "names")
    fr = unroll.frame(cx)
    scale = min(11.0, 640.0 / (fr.bounds[1] - fr.bounds[0]))
    m = fr.wall
    xyz, elements, colours = fr.xyz[m], [e for e, k in zip(cx.elements, m) if k], atom_colours(cx, None)[m]
    for turn in (fr.width, -fr.width):               # what lies at the slit is drawn at both edges of the sheet
        at_edge = fr.xyz[m][:, 0] * np.sign(turn) < -fr.width / 2.0 + 5.0
        xyz = np.vstack([xyz, fr.xyz[m][at_edge] + np.array([turn, 0.0, 0.0])])
        elements = elements + [e for e, k in zip([e for e, k in zip(cx.elements, m) if k], at_edge) if k]
        colours = np.vstack([colours, atom_colours(cx, None)[m][at_edge]])
    src, w, h, _ = _picture(xyz, elements, colours, fr.bounds, scale, flat=1.0, fade=2.0)
    a, b = html.escape(cx.names[0]), html.escape(cx.names[1])
    svg = sheet(cx, None, fr.at, fr.bounds, scale, colour, names, kinds, regions, background=(src, w, h),
                title=f"the wall of {a} around {b}, slit open along {b} and laid flat", wrap=fr.width * scale)
    facts = facts_of(cx, names, kinds)
    facts.update(radius=round(fr.radius, 1), around=round(fr.width, 1), along=round(fr.length, 1))
    return Sketch(svg, "unrolled", facts)


# ---------------------------------------------------------------------------------------- the page of sketches
CSS = """
body { font: 15px/1.5 -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; color: #1f2933; background: #fff;
       margin: 0; }
main { max-width: 1060px; margin: 0 auto; padding: 28px 24px 80px; }
h1 { font-size: 24px; margin: 0 0 6px; } h2 { font-size: 18px; margin: 44px 0 4px; }
p { margin: 8px 0; max-width: 78ch; } .quiet { color: #52606d; } .sketch { color: #9a3412; font-weight: 600; }
figure { margin: 14px 0 6px; padding: 16px; border: 1px solid #e4e7eb; border-radius: 8px; overflow-x: auto; }
figure svg { display: block; max-width: none; }
figcaption { font-size: 13px; color: #52606d; margin-top: 10px; max-width: 86ch; }
svg.book .res, svg.book .link { transition: opacity .12s; } svg.book .res { cursor: pointer; }
svg.book.dim .res:not(.hot) { opacity: .22; } svg.book.dim .link:not(.hot) { opacity: .05; }
svg.book .link.hot { stroke-width: 2.4; }
code { font-size: 13px; background: #f5f7fa; padding: 1px 4px; border-radius: 3px; }
"""

#: Pointing at a residue keeps it and its partners and steps the rest back, in every panel of the figure.
#: A click makes that stay; a click on the same residue, or beside the marks, lets it go.
JS = """
function linkFigures(root) {
  (root || document).querySelectorAll('svg.book').forEach(function (svg) {
    if (svg.parentNode.closest('svg')) { return; }
    var kept = null;
    function show(id) {
      svg.classList.add('dim');
      svg.querySelectorAll('.hot').forEach(function (e) { e.classList.remove('hot'); });
      svg.querySelectorAll('.res[data-id="' + id + '"]').forEach(function (e) { e.classList.add('hot'); });
      svg.querySelectorAll('.link').forEach(function (l) {
        if (l.dataset.a !== id && l.dataset.b !== id) { return; }
        l.classList.add('hot');
        var other = l.dataset.a === id ? l.dataset.b : l.dataset.a;
        svg.querySelectorAll('.res[data-id="' + other + '"]').forEach(function (e) { e.classList.add('hot'); });
      });
    }
    function clear() {
      svg.classList.remove('dim');
      svg.querySelectorAll('.hot').forEach(function (e) { e.classList.remove('hot'); });
    }
    svg.querySelectorAll('.res').forEach(function (g) {
      g.addEventListener('mouseenter', function () { if (!kept) { show(g.dataset.id); } });
      g.addEventListener('mouseleave', function () { if (!kept) { clear(); } });
      g.addEventListener('click', function (event) {
        event.stopPropagation();
        if (kept === g.dataset.id) { kept = null; clear(); return; }
        kept = g.dataset.id;
        show(kept);
      });
    });
    svg.addEventListener('click', function (event) {
      if (kept) { kept = null; clear(); event.stopPropagation(); }
    });
  });
}
linkFigures();
"""


def section(cx: book.Complex, what: str) -> str:
    four = panels(cx)
    n = four.facts
    a_name, b_name = html.escape(cx.names[0]), html.escape(cx.names[1])
    facts = (f"{n['at_the_interface'][0]} residues of {a_name} and {n['at_the_interface'][1]} of {b_name} at the "
             f"interface, {n['area']} Å² covered in all. {n['named'][0]} and {n['named'][1]} of them make a contact "
             f"of a kind and are named ({n['kinds']['saltbridge']} salt bridges, {n['kinds']['hbond']} hydrogen bonds, "
             f"{n['kinds']['hydrophobic']} hydrophobic contacts, by the plain distance table of "
             f"<code>bindview.distance_table</code>); {n['kinds']['near']} more pairs have atoms within 4 Å and no "
             f"kind, and are in the table only. {n['regions']} regions; {n['between_regions']} of the {n['pairs']} "
             f"pairs reach from one region into another.")
    rolled = ""
    if n["at_the_interface"][1] and sum(1 for r in cx.residues if r.side == 1) <= unroll.LONGEST:
        rolled = (f"<figure>{unrolled(cx).svg}<figcaption><b>6. Unrolled.</b> The wall of {a_name} around "
                  f"{b_name}, slit open along {b_name} and laid flat, seen from {b_name}: down the sheet runs the "
                  f"peptide, across it the angle around the peptide. The left and the right edge are one line."
                  f"</figcaption></figure>\n")
    return f"""
<h2>{html.escape(cx.title)}</h2>
<p class="quiet">{what} {facts}</p>
<figure>{four.svg}
<figcaption><b>1. Four panels.</b> <b>a</b> as it stands, <b>b</b> opened, <b>c</b> the face of {a_name} with the
atoms of {b_name} on it as licorice, hydrogen bonds and salt bridges from atom to atom, <b>d</b> the same face as a
flat map. Panels c and d are one frame at one scale: a residue is at the same place in both, and pointing at it in
one marks it in the other.</figcaption></figure>
<figure>{faces(cx).svg}
<figcaption><b>2. The two opened faces, close up.</b> A name only where a line ends: a residue that makes no
contact of a kind is colour on the surface, and is named to whoever points at it. A box is a residue of
{a_name}, an ellipse one of {b_name}; a line goes around every mark but the two it joins.</figcaption></figure>
<figure>{faces(cx, colour="class", lines="polar").svg}
<figcaption><b>3. The same by kind of residue, hydrogen bonds and salt bridges only.</b> Fewer lines, and so
fewer names.</figcaption></figure>
<figure>{faces(cx, colour="region").svg}
<figcaption><b>4. What lies on what.</b> A part of the interface has one colour on both faces.</figcaption></figure>
<figure>{faces(cx, surface_=False).svg}
<figcaption><b>5. The faces as outlines, the marks carry the colour.</b> No shaded surface.</figcaption></figure>
{rolled}<figure>{table(cx).svg}
<figcaption><b>Who touches whom, in full.</b> Every pair with atoms within 4 Å, region by region.</figcaption></figure>
"""


def main() -> int:
    t = time.time()
    fab = book.load(book.ROOT / "testdata/fab_lysozyme/model.cif.gz", ["H", "L"], ["C"],
                    ("HyHEL-10 Fab", "lysozyme"), "An antibody on its antigen (3D9A)")
    pmhc = book.load(book.ROOT / "testdata/mhc_peptide/model.cif.gz", ["A"], ["P"],
                     ("HLA-A2", "peptide GLCTLVAML"), "A peptide in a groove (3MRE)")
    body = section(pmhc, "Nine residues lying in a groove: one partner is a strand, and the interface is a "
                         "trough and not a plane.")
    body += section(fab, "Two globular partners and a flat interface: the case an open book is made for.")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open-book sketches</title><style>{CSS}</style></head><body><main>
<h1>An interface opened like a book</h1>
<p><span class="sketch">Sketches.</span> Not part of bindview, not tested, nothing here is promised. Made to be
looked at and argued with: <code>python prototype/ppi_openbook/openbook.py</code> writes this page.</p>
<p class="quiet">A <b>region</b> is a part of the interface: the touching pairs of residues are grouped by where
they lie on the plane of the interface (k-means on the middles of the pairs, one region for about
{book.PER_REGION:.0f} residues of each partner), and a residue belongs to the region that holds most of its
pairs. Regions are counted down the spine, I at the top.</p>
{body}
</main><script>{JS}</script></body></html>
"""
    out = HERE / "index.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out.relative_to(book.ROOT)}: {len(page) / 1e6:.2f} MB in {time.time() - t:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
