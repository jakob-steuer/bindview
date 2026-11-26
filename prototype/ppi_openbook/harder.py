#!/usr/bin/env python
"""The opened book on cases it was not made for, and how much of an interface each page shows.

A SKETCH, not part of bindview: nothing here is tested or promised.

    python prototype/ppi_openbook/harder.py        # writes prototype/ppi_openbook/harder.html

What is measured:

*covered points*      the points of the area calculation that are free in a partner alone and not free in
                      the complex
*seen*                the share of the covered area with nothing of its own partner between it and the
                      reader of the opened page; the rest is *turned away* (it looks away from the reader)
                      or *behind* (it looks at the reader, and its own partner stands in front)
*off the plane*       root mean square distance of the interface atoms from the plane the book opens on
*from how many sides* the length of the mean outward direction of the covered points: 1 where a partner
                      is covered from one side, 0 where it is covered from all around
"""

from __future__ import annotations

import html
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import book  # noqa: E402
import openbook  # noqa: E402
import unroll  # noqa: E402
from bindview import Structure, perceive_peptide, read_structure  # noqa: E402
from bindview.ligand import build_ligand  # noqa: E402
from surface import PROBE, radii  # noqa: E402


# ---------------------------------------------------------------------------------------- a peptide stored as one lump
def as_chain(st: Structure, component: str = "LIG_B", chain: str = "P") -> Structure:
    """A peptide that the file holds as one component, rewritten as a chain of residues.

    A co-folding model writes a peptide ligand as one component with atoms C1, C2, ...; bindview finds the
    residues in it (``perceive_peptide``). Here every atom gets the residue it was found in, the number
    of that residue in the chain as perceived, and its usual name, so that what reads chains can read it.
    What is attached to a residue and is no residue itself (a fatty acid on a lysine) is kept as part of
    the residue that carries it.
    """
    key = next(k for k in st.residues() if st.residue_name(k) == component)
    atoms = [a for a in st.residue_atoms(key) if a.element not in ("H", "D")]
    ligand = build_ligand(atoms, st.chem_comp_smiles.get(component))
    peptide = perceive_peptide(ligand.mol)
    if peptide is None:
        raise ValueError(f"{component} is not a peptide")
    new: "dict[str, tuple[str, int, str]]" = {}
    for r in peptide.residues:
        known = dict(r.atom_names)
        for n, i in enumerate(r.atoms):
            new[ligand.names[i]] = (r.name, r.position, known.get(i) or f"X{n}")
    for app in peptide.appendages:
        carrier = peptide.residue(app.positions[0])
        for n, i in enumerate(app.atoms):
            new[ligand.names[i]] = (carrier.name, carrier.position, f"Y{n}")
    out = []
    for a in st.atoms:
        if a.residue != key:
            out.append(replace(a, index=len(out)))
        elif a.name in new:
            name, number, atom = new[a.name]
            out.append(replace(a, index=len(out), name=atom, resname=name, chain=chain, resnum=number, icode="",
                               hetero=False))
    return Structure(out, st.source, dict(st.chem_comp_smiles), st.plddt_scale, st.qa_metrics)


# ---------------------------------------------------------------------------------------- how much a page shows
@dataclass
class Page:
    """One partner's page of the opened book, in numbers (shares of the area its partner covers)."""

    area: float           # Å² that the partner covers
    seen: float
    behind: float
    away: float
    sides: float          # 1: covered from one side, 0: from all around


def _free(p: np.ndarray, i: int, xyz: np.ndarray, reach: np.ndarray, among: np.ndarray) -> np.ndarray:
    env = np.where(among)[0]
    d = np.linalg.norm(xyz[env] - xyz[i], axis=1)
    nb = env[(d < reach[i] + reach[env]) & (env != i)]
    if not len(nb):
        return np.ones(len(p), dtype=bool)
    return (np.linalg.norm(p[:, None, :] - xyz[nb][None, :, :], axis=2) > reach[nb][None, :]).all(axis=1)


def covered(cx: book.Complex, side: int, n: int = 96):
    """The covered points of one partner: where they are, which way they look, the area each stands for,
    and the atom each belongs to."""
    reach = radii(cx.elements) + PROBE
    own, everything = cx.side == side, np.ones(len(reach), dtype=bool)
    sphere = book._sphere(n)
    where, look, share, atom = [], [], [], []
    for i in np.concatenate([cx.residues[r].atoms for r in cx.at_interface(side)] or [np.zeros(0, dtype=int)]):
        p = cx.xyz[i] + reach[i] * sphere
        lost = _free(p, i, cx.xyz, reach, own) & ~_free(p, i, cx.xyz, reach, everything)
        if lost.any():
            where.append(p[lost])
            look.append(sphere[lost])
            share.append(np.full(int(lost.sum()), 4.0 * np.pi * reach[i] ** 2 / n))
            atom.append(np.full(int(lost.sum()), i))
    if not where:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int)
    return np.concatenate(where), np.concatenate(look), np.concatenate(share), np.concatenate(atom)


def in_sight(p: np.ndarray, towards: np.ndarray, xyz: np.ndarray, reach: np.ndarray, chunk: int = 400) -> np.ndarray:
    """Per point: does a ray from it in the direction ``towards`` miss every sphere (centres ``xyz``)?"""
    free = np.ones(len(p), dtype=bool)
    for k in range(0, len(p), chunk):
        w = xyz[None, :, :] - p[k:k + chunk, None, :]
        along = w @ towards
        beside = (w * w).sum(axis=2) - along * along
        free[k:k + chunk] = ~((along > 1e-6) & (beside < reach[None, :] ** 2 - 1e-6)).any(axis=1)
    return free


def page(cx: book.Complex, side: int, towards: "np.ndarray | None" = None) -> Page:
    """One partner's page in numbers. The reader of A's page stands on B's side and the other way round;
    ``towards`` is another direction to look from."""
    p, look, share, _ = covered(cx, side)
    if not len(p):
        return Page(0.0, 0.0, 0.0, 0.0, 1.0)
    if towards is None:
        towards = cx.axes[0] if side == 0 else -cx.axes[0]
    own = cx.side == side
    reach = radii(cx.elements) + PROBE
    facing = look @ towards > 0.0
    clear = in_sight(p, towards, cx.xyz[own], reach[own])
    total = float(share.sum())
    return Page(total, float(share[clear].sum()) / total, float(share[facing & ~clear].sum()) / total,
                float(share[~facing].sum()) / total, float(np.linalg.norm((share[:, None] * look).sum(axis=0))) / total)


def seen_cut(cx: book.Complex, side: int = 0) -> float:
    """The share of what is covered on one partner that is seen when that partner is cut along the plane
    and each half is looked at from the cut.

    The covered points lie a probe's radius outside the surface that is drawn, so the body they bound is
    cut that far beyond the plane (as the drawing does it: grown, cut, worn back). A point in the slab
    between the two cuts is on both halves, and counts where it is seen.
    """
    p, _, share, _ = covered(cx, side)
    if not len(p):
        return 0.0
    own = cx.side == side
    xyz, reach = cx.xyz[own], (radii(cx.elements) + PROBE)[own]
    h = (p - cx.centre) @ cx.axes[0]
    seen = np.zeros(len(p), dtype=bool)
    for sign in (-1.0, 1.0):                      # the half below the plane is seen from above, the other from below
        m = np.where(h * sign > -PROBE)[0]
        towards, limit = -sign * cx.axes[0], PROBE + sign * h[m]
        for k in range(0, len(m), 400):
            w = xyz[None, :, :] - p[m[k:k + 400], None, :]
            along = w @ towards
            beside = (w * w).sum(axis=2) - along * along
            inside = beside < reach[None, :] ** 2 - 1e-6
            root = np.sqrt(np.where(inside, reach[None, :] ** 2 - beside, 0.0))
            hit = inside & (along + root > 1e-6) & (along - root < limit[k:k + 400, None] - 1e-6)
            seen[m[k:k + 400][~hit.any(axis=1)]] = True
    return float(share[seen].sum() / share.sum())


def seen_unrolled(cx: book.Complex) -> float:
    """The share of what the second partner covers on the first that is seen from the second partner's
    own path: nothing of the first partner lies between the covered point and the nearest point of the
    path. This is what the unrolled sheet shows, as far as a sheet of residues shows a surface."""
    p, _, share, _ = covered(cx, 0)
    if not len(p):
        return 0.0
    _, _, path = unroll.chain_of(cx)
    nearest = np.empty(len(p), dtype=int)
    for k in range(0, len(p), 2000):
        nearest[k:k + 2000] = np.argmin(np.linalg.norm(p[k:k + 2000, None, :] - path[0][None, :, :], axis=2), axis=1)
    to = path[0][nearest] - p
    far = np.linalg.norm(to, axis=1)
    to /= np.maximum(far, 1e-9)[:, None]
    own = cx.side == 0
    xyz, reach = cx.xyz[own], (radii(cx.elements) + PROBE)[own]
    seen = np.zeros(len(p), dtype=bool)
    for k in range(0, len(p), 400):
        w = xyz[None, :, :] - p[k:k + 400, None, :]
        along = np.einsum("pak,pk->pa", w, to[k:k + 400])
        beside = (w * w).sum(axis=2) - along * along
        inside = beside < reach[None, :] ** 2 - 1e-6
        root = np.sqrt(np.where(inside, reach[None, :] ** 2 - beside, 0.0))
        hit = inside & (along + root > 1e-6) & (along - root < far[k:k + 400, None] - 1e-6)
        seen[k:k + 400] = ~hit.any(axis=1)
    return float(share[seen].sum() / share.sum())


def off_plane(cx: book.Complex) -> float:
    near = np.concatenate([r.near for r in cx.residues if len(r.near)])
    return float(np.sqrt((cx.closed()[near, 0] ** 2).mean()))


# ---------------------------------------------------------------------------------------- cut, then opened
def halves(cx: book.Complex, colour: str = "area", names: str = "contacts", lines: str = "all",
           residues: bool = False) -> openbook.Sketch:
    """The first partner cut along the plane of the interface, and its two halves opened like a book.

    For a partner that wraps the other: nothing of it faces a reader until it is cut. Each page is one
    half seen from the cut, which is drawn flat and hatched; the second partner lies on both pages as
    licorice, on the right page as seen from the other side. A residue of the first partner is named on
    the page of the half it is in. Hydrogen bonds and salt bridges run from atom to atom. A residue
    that is pointed at shows its atoms and every contact it makes; with ``residues`` the atoms of the
    named residues are there all the time, faintly.
    """
    look = {k: openbook.STROKES[k][:2] for k in ("saltbridge", "hbond", "hydrophobic")}
    kinds = openbook._KINDS[openbook._one_of(lines, openbook.LINES, "lines")]
    metric = openbook.metric_of(cx, colour)
    shut = cx.closed()
    lower = np.stack([-shut[:, 2], shut[:, 1], shut[:, 0]], axis=1)       # the reader on B's side: book.faced
    upper = np.stack([shut[:, 2], shut[:, 1], -shut[:, 0]], axis=1)       # the reader on the other side
    ids = cx.at_interface(0) + cx.at_interface(1)
    spots = np.array([lower[cx.residues[i].near][:, :2].mean(axis=0) for i in ids])
    margin, gap, head, foot = 6.0, 30.0, 30.0, 66.0
    x0, x1 = spots[:, 0].min() - margin, spots[:, 0].max() + margin
    y0, y1 = spots[:, 1].min() - margin, spots[:, 1].max() + margin
    scale = min(13.0, (980.0 - gap) / 2.0 / (x1 - x0))
    colours = openbook.atom_colours(cx, metric)
    shown = openbook.named(cx, names, kinds)
    b_residues = [r for r in cx.residues if r.side == 1]
    whole = len(b_residues) <= openbook.WHOLE
    drawn = np.zeros(len(cx.xyz), dtype=bool)
    for r in b_residues:
        if whole or len(r.near):
            drawn[r.atoms] = True
    elements = [e for e, k in zip(cx.elements, drawn) if k]
    below = {i: float(shut[cx.residues[i].near, 0].mean()) <= 0.0 for i in cx.at_interface(0)}
    parts, origin = [], 0.0
    width_of = 0
    for view, bounds, low, words in (
            (lower, (x0, x1, y0, y1), True, "one half, seen from the cut"),
            (upper, (-x1, -x0, y0, y1), False, "the other half, seen from the cut (the page is turned over)")):
        src, w, h, _ = openbook.picture(view, cx, colours, bounds, scale, which=cx.side == 0, fade=2.0,
                                        licorice=openbook.surface.sticks(view[drawn], elements, openbook.CARBON), cut=0.0)
        width_of = w

        def px(q, bounds=bounds, origin=origin) -> np.ndarray:
            return np.array([origin + (q[0] - bounds[0]) * scale, head + (bounds[3] - q[1]) * scale])

        parts.append(openbook.heading(origin, 14, "ab"[0 if low else 1], words))
        parts.append(f'<image href="{src}" x="{origin:.1f}" y="{head}" width="{w}" height="{h}"/>')
        # The first partner's residues of this half get a box, the second partner's a name, on both pages:
        # each beside its residue and not on it, off what comes up with it when pointed at, off the
        # licorice and off one another (:func:`openbook.beside`), with a hairline to the nearest of its atoms.
        mine = [i for i in cx.at_interface(0) if i in shown and below[i] == low]
        theirs = [i for i in sorted(shown) if cx.residues[i].side == 1]
        size = 9.5
        labels = [cx.residues[i].label for i in mine + theirs]
        half_w = np.array([len(s) * size * 0.29 + (4.0 if n < len(mine) else 3.0) for n, s in enumerate(labels)])
        half_h = size * 0.5 + 3.0
        want = np.array([px(view[cx.residues[i].near][:, :2].mean(axis=0)) for i in mine]
                        + [px(view[cx.residues[i].atoms][:, :2].mean(axis=0)) for i in theirs]).reshape(-1, 2)
        sticks_at = openbook.licorice_points(cx, drawn, view, px)
        middle = sticks_at[:int(drawn.sum())].mean(axis=0) if drawn.any() else want.mean(axis=0)
        both = mine + theirs
        full = {i: openbook.own_points(cx, i, view, px, look) for i in both}
        partners, with_lines = openbook.partners_of(cx, both, kinds), {i: v[1] for i, v in full.items()}
        pos = openbook.beside(want, half_w, half_h, [full[i][1] for i in both], sticks_at,
                              (origin + 2, head + 2, origin + w - 2, head + h - 2), middle,
                              also=[openbook.coming_up(i, partners, with_lines) for i in both])
        boxes = []                                   # drawn after the lines: a line that passes a box goes under it
        for n, i in enumerate(mine):
            r, q = cx.residues[i], pos[n]
            boxes.append(openbook.mark(r, q, half_w[n], half_h, np.ones(3), size,
                                       f"{r.label} (chain {r.chain}): {r.buried:.0f} Å² covered", opacity="0.86",
                                       lead=openbook.hairline(q, half_w[n], half_h, full[i][0]),
                                       extra=openbook.under(cx, i, view, px, look, residues)))
        # Hydrogen bonds and salt bridges of this half, from atom to atom; the others only for the pointing.
        for c in cx.contacts:
            if c.kind not in kinds or below.get(c.a) != low:
                continue
            a_key, b_key = cx.residues[c.a].key, cx.residues[c.b].key
            said = (f'<title>{cx.residues[c.a].label} – {cx.residues[c.b].label}: {openbook.KIND[c.kind]}, '
                    f'{c.distance:.1f} Å</title>')
            if c.kind not in openbook._KINDS["polar"] or not c.atoms_a or not c.atoms_b:
                parts.append(f'<path class="link" data-a="{a_key}" data-b="{b_key}" d="M 0 0" stroke="none"/>')
                continue
            pa, pb = px(view[list(c.atoms_a)].mean(axis=0)), px(view[list(c.atoms_b)].mean(axis=0))
            stroke, dash, width = openbook.STROKES[c.kind]
            parts.append(
                f'<line x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" stroke="#ffffff" '
                f'stroke-opacity="0.8" stroke-width="{width + 2.0}" stroke-linecap="round"/>'
                f'<line class="link" data-a="{a_key}" data-b="{b_key}" x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" '
                f'x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" stroke="{stroke}" stroke-width="{width + 0.2}" '
                f'stroke-dasharray="{dash or "none"}">{said}</line>')
        parts += boxes
        for n, i in enumerate(theirs):
            r, q = cx.residues[i], pos[len(mine) + n]
            lead = openbook.hairline(q, half_w[len(mine) + n], half_h, full[i][0], edge=True)
            parts.append(f'<g class="res" data-id="{r.key}">{lead}<text x="{q[0]:.1f}" y="{q[1] + size * 0.36:.1f}" '
                         f'font-size="{size}" font-weight="600" fill="#7c2d12" text-anchor="middle" stroke="#ffffff" '
                         f'stroke-width="3" stroke-opacity="0.9" paint-order="stroke" stroke-linejoin="round">'
                         f'{r.label}</text><title>{r.label} (chain {r.chain}): {r.buried:.0f} Å² covered</title></g>')
        origin += w + gap
    width, height = origin - gap, head + int(round((y1 - y0) * scale)) + foot
    key, x = metric.key(0, height - 32)
    key += openbook.line_key(x + 16, height - 23, [k for k in kinds if k in openbook._KINDS["polar"]])[0]
    carbon = openbook.as_hex(np.array(openbook.CARBON))
    key += (f'<line x1="0" y1="{height - 4}" x2="22" y2="{height - 4}" stroke="{carbon}" stroke-width="4.5" '
            f'stroke-linecap="round"/>'
            + openbook.text(28, height - 1, f"atoms of {html.escape(cx.names[1])}, the same on both pages; hatched: "
                                            f"where {html.escape(cx.names[0])} is cut through", 9.5, openbook.QUIET))
    bar = (f'<line x1="{width - 10 * scale:.1f}" y1="{height - 40}" x2="{width:.1f}" y2="{height - 40}" '
           f'stroke="{openbook.INK}" stroke-width="1.2"/>'
           + openbook.text(width, height - 44, "10 Å", 9.5, openbook.QUIET, "end"))
    svg = (f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
           f'width="{width:.0f}" font-family="{openbook.FONT}">' + "".join(parts) + key + bar + "</svg>")
    facts = openbook.facts_of(cx, names, kinds)
    facts["seen"], facts["seen_cut"] = round(page(cx, 0).seen, 2), round(seen_cut(cx, 0), 2)
    return openbook.Sketch(svg, "cut and opened", facts)


# ---------------------------------------------------------------------------------------- the cases
@dataclass
class Case:
    slug: str
    title: str
    what: str
    cx: book.Complex
    wraps: bool = False          # also drawn cut and opened, and unrolled
    colours: "tuple[str, ...]" = ()     # also drawn as two faces in each of these colourings


def cases() -> "list[Case]":
    root = book.ROOT / "testdata"
    trimer, tectb = root / "interface_multimer/tnfa_trimer", root / "interface_pae/afdb_tectb"
    return [
        Case("groove", "A peptide in a groove (3MRE)", "The first of the two easy cases, as the reference.",
             book.load(root / "mhc_peptide/model.cif.gz", ["A"], ["P"], ("HLA-A2", "peptide GLCTLVAML"))),
        Case("antibody", "An antibody on its antigen (3D9A)", "The second easy case, as the reference.",
             book.load(root / "fab_lysozyme/model.cif.gz", ["H", "L"], ["C"], ("HyHEL-10 Fab", "lysozyme"))),
        Case("trimer", "One protomer of the TNF trimer against the other two (a prediction)",
             "Three participants around an axis: one protomer is a wedge between the other two, so its "
             "interface has two faces at an angle. A prediction of Boltz, of which two draws are in the "
             "repository: below the four panels it is coloured by what the model says of itself.",
             book.ensemble(book.load(trimer / "hTNFA_canonical_trimer_model_0.cif", ["A"], ["B", "C"],
                                     ("TNF protomer A", "protomers B and C"),
                                     pae=trimer / "pae_hTNFA_canonical_trimer_model_0.npz"),
                           sorted(trimer.glob("hTNFA_canonical_trimer_model_[1-9].cif")), ["A"], ["B", "C"]),
             colours=("ensemble", "ipsae")),
        Case("hirudin", "Thrombin and the tail of hirudin (2UUF)",
             "Ten residues along a groove on the surface, away from the active site.",
             book.load(root / "thrombin_hirudin/model.cif.gz", ["A", "B"], ["H"], ("thrombin", "hirudin peptide"))),
        Case("domains", "Two domains of beta-tectorin (the AlphaFold database, one chain cut in two)",
             "A model that is sure of one part of an interface and not of another: the predicted aligned error of "
             "the pairs of residues that touch runs from 2 to 14 Å. Below the four panels it is coloured by that, "
             "and by pLDDT.",
             book.load(read_structure(tectb / "model.cif.gz").split("A", {"N": (28, 124), "Z": (126, 269)}),
                       ["N"], ["Z"], ("N-terminal domain", "ZP domain"), pae=tectb / "predicted_aligned_error.json.gz"),
             colours=("pae", "plddt")),
        Case("semaglutide", "GLP-1 receptor and semaglutide (7KI0)",
             "The partner wraps: the first residues of the peptide are inside the bundle of helices, the rest "
             "lies against the outer domain.",
             book.load(root / "glp1r_semaglutide/model.cif.gz", ["R"], ["P"], ("GLP-1 receptor", "semaglutide")),
             wraps=True),
    ]


def numbers(cx: book.Complex) -> dict:
    a, b = page(cx, 0), page(cx, 1)
    rolled = None
    if sum(1 for r in cx.residues if r.side == 1) <= unroll.LONGEST and len({r.chain for r in cx.residues if r.side == 1}) == 1:
        rolled = seen_unrolled(cx)
    return {"residues": [len(cx.at_interface(0)), len(cx.at_interface(1))], "off_plane": off_plane(cx),
            "pages": [a, b], "cut": seen_cut(cx, 0), "rolled": rolled}


# ---------------------------------------------------------------------------------------- the page
def percent(x: float) -> str:
    return f"{100.0 * x:.0f}"


def row(case: Case, n: dict) -> str:
    cells = [f'<td><a href="#{case.slug}">{html.escape(case.title)}</a></td>',
             f'<td>{n["residues"][0]} + {n["residues"][1]}</td>', f'<td>{n["off_plane"]:.1f}</td>']
    for p in n["pages"]:
        cells.append(f'<td>{p.area:.0f}</td><td><b>{percent(p.seen)}</b></td><td>{percent(p.behind)} / '
                     f'{percent(p.away)}</td><td>{p.sides:.2f}</td>')
    cells.append(f'<td>{percent(n["cut"])}</td><td>{percent(n["rolled"]) if n["rolled"] is not None else "–"}</td>')
    return "<tr>" + "".join(cells) + "</tr>"


COLOUR_SAYS = {"ensemble": "how many of the models have the residue at the interface",
               "ipsae": "the residue's term of ipSAE", "pae": "predicted aligned error to the residues it touches",
               "plddt": "pLDDT, in AlphaFold's bands and colours"}

CSS = openbook.CSS + """
table { border-collapse: collapse; font-size: 13px; margin: 14px 0; }
th, td { padding: 4px 7px; text-align: right; border-bottom: 1px solid #e4e7eb; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; white-space: normal; } th { font-weight: 600; color: #52606d; }
tr.group th { border-bottom: none; text-align: center; color: #1f2933; }
"""


def main() -> int:
    t = time.time()
    rows, body = [], []
    for case in cases():
        n = numbers(case.cx)
        rows.append(row(case, n))
        a, b = n["pages"]
        names = [html.escape(x) for x in case.cx.names]
        said = (f"The page of {names[0]} shows {percent(a.seen)} % of what {names[1]} covers on it "
                f"({percent(a.behind)} % is behind something of its own, {percent(a.away)} % looks away); the page "
                f"of {names[1]} shows {percent(b.seen)} % ({percent(b.behind)} %, {percent(b.away)} %). The "
                f"interface atoms lie {n['off_plane']:.1f} Å off the plane (root mean square).")
        four = openbook.panels(case.cx)
        body.append(f'<h2 id="{case.slug}">{html.escape(case.title)}</h2><p class="quiet">{case.what} {said}</p>'
                    f"<figure>{four.svg}<figcaption><b>As the sketch draws it.</b></figcaption></figure>")
        if case.wraps:
            first = names[0][0].upper() + names[0][1:]
            body.append(f"<figure>{halves(case.cx).svg}<figcaption><b>Cut, then opened.</b> {first} is cut along "
                        f"the plane and each half is a page, seen from the cut; {names[1]} lies on both as licorice. "
                        f"The two pages together show {percent(n['cut'])} % of what {names[1]} covers on "
                        f"{names[0]}, where the uncut page shows {percent(a.seen)} %.</figcaption></figure>")
            body.append(f"<figure>{openbook.unrolled(case.cx).svg}<figcaption><b>Unrolled.</b> The wall of {names[0]} "
                        f"around {names[1]}, slit open along {names[1]} and laid flat, seen from {names[1]}. From "
                        f"that path {percent(n['rolled'])} % of what {names[1]} covers is in sight.</figcaption></figure>")
        for colour in case.colours:
            body.append(f"<figure>{openbook.faces(case.cx, colour=colour).svg}<figcaption><b>The two "
                        f"faces by {COLOUR_SAYS[colour]}.</b></figcaption></figure>")
        print(f"{case.slug:12s} {n['residues'][0]:3d}+{n['residues'][1]:<3d} off {n['off_plane']:4.1f}  "
              + "  ".join(f"{p.area:5.0f} Å² seen {percent(p.seen):>3s} behind {percent(p.behind):>3s} "
                          f"away {percent(p.away):>3s} sides {p.sides:.2f}" for p in n["pages"])
              + f"   cut {percent(n['cut']):>3s}   unrolled {percent(n['rolled']) if n['rolled'] is not None else '-':>3s}"
              + f"   [{time.time() - t:.0f} s]", flush=True)
    page_ = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open-book sketches: harder cases</title><style>{CSS}</style></head><body><main>
<h1>The opened book on harder cases</h1>
<p><span class="sketch">Sketches.</span> Not part of bindview, not tested, nothing here is promised:
<code>python prototype/ppi_openbook/harder.py</code> writes this page. Every figure is drawn as the sketch
draws the two easy cases; nothing is adjusted for a case.</p>
<p class="quiet"><b>seen</b>: of the area a partner covers on a page, the share that has nothing of the page's
own protein between it and the reader. The rest is <b>behind</b> (it looks at the reader, and its own protein
stands in front) or looks <b>away</b>. <b>sides</b>: 1 where a partner is covered from one side, 0 where it
is covered from all around (a surface of atoms is rough: a flat interface comes to about 0.6). <b>off the plane</b>: root mean square distance of the interface atoms from the
plane the book opens on. No number decides anything here. <b>if cut</b>: the share of the first partner's
covered area that is seen when that partner is cut along the plane and each half is looked at from the cut.
<b>unrolled</b>: the share that is in sight from the second partner's own path, where that partner is a
peptide.</p>
<table><tr class="group"><th></th><th></th><th></th><th colspan="4">first partner's page</th>
<th colspan="4">second partner's page</th><th>first, if cut</th><th>first, unrolled</th></tr>
<tr><th>case</th><th>residues</th><th>off plane, Å</th>
<th>Å²</th><th>seen %</th><th>behind / away %</th><th>sides</th>
<th>Å²</th><th>seen %</th><th>behind / away %</th><th>sides</th><th>seen %</th><th>seen %</th></tr>
{"".join(rows)}</table>
{"".join(body)}
</main><script>{openbook.JS}</script></body></html>
"""
    out = HERE / "harder.html"
    out.write_text(page_, encoding="utf-8")
    print(f"wrote {out.relative_to(book.ROOT)}: {len(page_) / 1e6:.2f} MB in {time.time() - t:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
