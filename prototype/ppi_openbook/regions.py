#!/usr/bin/env python
"""How the regions of a protein–protein interface are calculated, step by step, for one complex.

A SKETCH, not part of bindview: nothing here is tested or promised.

``figure(cx)`` draws, for a complex read by :func:`book.load`, the five steps of :func:`book._complete`
and :func:`book.group`: the atoms at the interface, the plane through them and its long axis, the pairs of
residues that touch as points on that plane, the points in groups, and the regions on the two opened
faces. Every number in it is of the complex that is drawn: what the steps found is what
:func:`book.grouping` hands out, and not a second calculation.

The panels 3, 4 and the left face of 5 are one frame: the face of the first partner, the spine upright,
as it lies on its page of the opened book. The other face is mirrored, as a page is.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import book  # noqa: E402
import openbook  # noqa: E402
import surface  # noqa: E402

__all__ = ["figure"]

INK, QUIET, FAINT, EDGE = openbook.INK, openbook.QUIET, openbook.FAINT, openbook.EDGE
WIDTH, GAP = 780.0, 28.0                          # the figure, and between two panels that stand side by side
HALF = (WIDTH - GAP) / 2.0
WORDS = 10.0                                      # the size of the words under a heading
FIRST = 48.0                                      # where a panel's drawing begins under its heading and two lines
NEAR_COLOURS = ((0.22, 0.45, 0.70), (0.86, 0.45, 0.16))      # an atom at the interface, per partner
CALLED = ("blue", "orange")                                  # and what the words call these colours
HALO = ' stroke="#ffffff" stroke-width="3" stroke-opacity="0.9" paint-order="stroke" stroke-linejoin="round"'


def _head(x: float, y: float, number: int, words: str, lines) -> str:
    """The heading of a panel with its number, and at most two lines of words under it."""
    out = [openbook.text(x, y + 13, f'<tspan font-weight="700">{number}</tspan><tspan dx="7" '
                                    f'font-weight="600">{words}</tspan>',
                         12.0, INK)]
    out += [openbook.text(x, y + 29 + 13.5 * n, line, WORDS, QUIET) for n, line in enumerate(lines)]
    return "".join(out)


def _near(cx: book.Complex, side: int) -> np.ndarray:
    """The atoms of a partner that are at the interface."""
    return np.concatenate([cx.residues[i].near for i in cx.at_interface(side)] or [np.zeros(0, dtype=int)])


# ---------------------------------------------------------------------------------------- 1: the atoms
def _atoms(cx: book.Complex, x: float, y: float) -> "tuple[str, float]":
    shut, pad = cx.closed(), 3.0
    colours = np.empty((len(cx.xyz), 3))
    for s in (0, 1):
        colours[cx.side == s] = np.array(openbook.BODY[s])
        colours[_near(cx, s)] = np.array(NEAR_COLOURS[s])
    # As balls: most of them are inside, where the two lie on each other, and show through the surface.
    which = np.concatenate([_near(cx, s) for s in (0, 1)])
    balls = (shut[which], 0.92 * surface.radii([cx.elements[a] for a in which]), colours[which])
    bounds = (shut[:, 0].min() - pad, shut[:, 0].max() + pad, shut[:, 1].min() - pad, shut[:, 1].max() + pad)
    scale = min(HALF / (bounds[1] - bounds[0]), 230.0 / (bounds[3] - bounds[2]))
    src, w, h, _ = openbook.picture(shut, cx, colours, bounds, scale, licorice=balls)
    n = [len(_near(cx, s)) for s in (0, 1)]
    r = [len(cx.at_interface(s)) for s in (0, 1)]
    names = [html.escape(v) for v in cx.names]
    out = [_head(x, y, 1, f"The atoms within {book.NEAR:g} Å of the partner are the interface",
                 (f"{n[0]} atoms of {r[0]} residues of {names[0]} ({CALLED[0]}) and {n[1]} atoms of {r[1]} residues",
                  f"of {names[1]} ({CALLED[1]}); pale: seen through the surface.")),
           f'<image href="{src}" x="{x:.1f}" y="{y + FIRST + 4:.1f}" width="{w}" height="{h}"/>']
    for s in (0, 1):
        at = x + (shut[cx.side == s, 0].mean() - bounds[0]) * scale
        at = min(max(at, x + 3.2 * len(cx.names[s])), x + w - 3.2 * len(cx.names[s]))
        out.append(openbook.text(at, y + FIRST + h + 18, names[s], 11, EDGE[1] if s else INK, "middle", "600"))
    return "".join(out), FIRST + h + 24.0


# ---------------------------------------------------------------------------------------- 2: the plane and the spine
def _arrow(x: float, y: float, dx: float, dy: float, size: float = 6.0) -> str:
    """The head of an arrow that points along (dx, dy), its tip at (x, y)."""
    u = np.array([dx, dy]) / float(np.hypot(dx, dy))
    v = np.array([-u[1], u[0]])
    a, b = np.array([x, y]) - size * u + 0.45 * size * v, np.array([x, y]) - size * u - 0.45 * size * v
    return f'<path d="M {x:.1f} {y:.1f} L {a[0]:.1f} {a[1]:.1f} L {b[0]:.1f} {b[1]:.1f} Z" fill="{INK}"/>'


def _plane(cx: book.Complex, x: float, y: float) -> "tuple[str, float]":
    shut = cx.closed()
    near = [_near(cx, s) for s in (0, 1)]
    every = shut[np.concatenate(near)]
    off, spine, short = np.sqrt((every ** 2).mean(axis=0))
    reach = float(np.abs(every[:, 1]).max()) + 3.0
    scale = 205.0 / (2.0 * reach)
    top = y + FIRST + 30.0
    middle = top + reach * scale
    out = [_head(x, y, 2, "A plane is laid through them; its long axis is the spine",
                 (f"The atoms lie {off:.1f} Å off the plane, {spine:.1f} Å along the spine and {short:.1f} Å across it",
                  "(root mean squares). The book opens about the spine, a quarter turn each."))]
    # From the side: the plane is a line.
    wide = 15.0
    ox = x + 14.0 + wide * scale
    bounds = (-wide, wide, -reach, reach)
    colours = openbook.atom_colours(cx, None)
    src, w, h, _ = openbook.picture(shut, cx, colours, bounds, scale, flat=1.0, fade=3.0)
    out.append(f'<image href="{src}" x="{ox - wide * scale:.1f}" y="{top:.1f}" width="{w}" height="{h}" '
               'opacity="0.6"/>')

    def dots(cx_, ax, sign):
        """The atoms in front of each other as they come: by what lies nearer to the reader."""
        order = sorted(((int(a), s) for s in (0, 1) for a in near[s]), key=lambda t: (float(shut[t[0], ax[2]]), t[0]))
        return "".join(f'<circle cx="{cx_ + sign * shut[a, ax[0]] * scale:.1f}" '
                       f'cy="{middle - shut[a, ax[1]] * scale:.1f}" '
                       f'r="2.6" fill="{openbook.as_hex(np.array(NEAR_COLOURS[s]))}" fill-opacity="0.85" '
                       f'stroke="#ffffff" stroke-width="0.6"/>' for a, s in order)

    out.append(dots(ox, (0, 1, 2), 1.0))
    out.append(f'<line x1="{ox:.1f}" y1="{top - 6:.1f}" x2="{ox:.1f}" y2="{top + 2 * reach * scale + 4:.1f}" '
               f'stroke="{INK}" '
               f'stroke-width="1.3" stroke-dasharray="5 3"/>'
               f'<path d="M {ox - 34:.1f} {top + 2:.1f} Q {ox - 20:.1f} {top - 17:.1f} {ox:.1f} {top - 17:.1f} '
               f'Q {ox + 20:.1f} {top - 17:.1f} {ox + 34:.1f} {top + 2:.1f}" fill="none" stroke="{INK}" '
               'stroke-width="1"/>'
               + _arrow(ox - 34, top + 2, -14, 20) + _arrow(ox + 34, top + 2, 14, 20)
               + openbook.text(ox - 30, top - 14, "quarter turn", 9.5, QUIET, "end")
               + openbook.text(ox + 30, top - 14, "quarter turn", 9.5, QUIET))
    low = top + 2 * reach * scale + 17.0
    out.append(openbook.text(ox, low, "from the side: the plane is a line", 9.5, QUIET, "middle"))
    # Face on: the spine is the long axis.
    broad = float(np.abs(every[:, 2]).max()) + 3.0
    fx = max(ox + wide * scale + 46.0 + broad * scale, x + HALF - 18.0 - broad * scale)
    out.append(f'<rect x="{fx - broad * scale:.1f}" y="{top:.1f}" width="{2 * broad * scale:.1f}" '
               f'height="{2 * reach * scale:.1f}" '
               f'fill="#f6f7f9" stroke="{FAINT}" stroke-width="0.8"/>' + dots(fx, (2, 1, 0), -1.0))
    up, side = 1.0 * reach * scale, 1.0 * broad * scale
    out.append(f'<line x1="{fx - side + 4:.1f}" y1="{middle:.1f}" x2="{fx + side - 4:.1f}" y2="{middle:.1f}" '
               f'stroke="{INK}" '
               f'stroke-width="0.8" stroke-dasharray="2 2.5"/>'
               f'<line x1="{fx:.1f}" y1="{middle - up + 5:.1f}" x2="{fx:.1f}" y2="{middle + up - 5:.1f}" '
               f'stroke="#ffffff" stroke-width="4.4" stroke-opacity="0.85"/>'
               f'<line x1="{fx:.1f}" y1="{middle - up + 5:.1f}" x2="{fx:.1f}" y2="{middle + up - 5:.1f}" '
               f'stroke="{INK}" '
               f'stroke-width="2"/>' + _arrow(fx, middle - up + 1, 0, -1, 8) + _arrow(fx, middle + up - 1, 0, 1, 8)
               + openbook.text(fx + 6, middle - up + 16, "spine", 10, INK, "start", "700",
                               ' stroke="#ffffff" stroke-width="3" paint-order="stroke"')
               + openbook.text(fx, low, "face on: the plane is the page", 9.5, QUIET, "middle"))
    return "".join(out), low - y + 8.0


# ---------------------------------------------------------------------------------------- 3 and 4: points and groups
def _frame(g: book.Grouping, x: float, y: float, scale: float, margin: float):
    """The flat view that panels 3 and 4 share, its top left corner at (x, y): its size, and what takes a place
    on the plane (along the spine, across the page) to the drawing."""
    at = np.array([p for p in g.place.values()])
    lo, hi = at.min(axis=0) - margin, at.max(axis=0) + margin
    w, h = (hi[1] - lo[1]) * scale, (hi[0] - lo[0]) * scale

    def px(p) -> np.ndarray:
        return np.array([x + (hi[1] - p[1]) * scale, y + (hi[0] - p[0]) * scale])          # across as on A's page

    return w, h, px


def _spine(px, at: np.ndarray, top: float, bottom: float) -> str:
    x = float(px(at)[0])
    return (f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" stroke="{FAINT}" stroke-width="0.8" '
            f'stroke-dasharray="5 3"/>')


def _beside(at: np.ndarray, hw: float, hh: float, marks: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Where a name goes that belongs at ``at``: near it, on no mark and as little on lines as can be (both
    given as points), its hairline over no mark."""
    best = None
    for far in np.arange(12.0, 90.0, 3.0):
        for turn in np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False):
            q = at + far * np.array([np.cos(turn), np.sin(turn)])
            on = int(((np.abs(marks - q)[:, 0] < hw + 6.0) & (np.abs(marks - q)[:, 1] < hh + 6.0)).sum())
            under = int(((np.abs(lines - q)[:, 0] < hw) & (np.abs(lines - q)[:, 1] < hh)).sum())
            way = openbook.along(at + 7.0 * (q - at) / far, q, 2.0)
            near = np.linalg.norm(marks[None, :, :] - way[:, None, :], axis=2).min(axis=0)
            other = np.linalg.norm(marks - at, axis=1) > 1.0
            cost = far + 200.0 * on + 4.0 * under + 60.0 * int(((near < 6.0) & other).sum())
            if best is None or cost < best[0] - 1e-9:
                best = (cost, q)
    return best[1]


def _pairs(cx: book.Complex, g: book.Grouping, x: float, y: float, scale: float, example: "int | None"
           ) -> "tuple[str, float]":
    w, h, px = _frame(g, x + 6.0, y + FIRST + 6.0, scale, 1.6)
    names = [html.escape(v) for v in cx.names]
    out = [_head(x, y, 3, "Two residues that touch make a point",
                 (f"{len(g.mids)} pairs of residues have atoms within {book.TOUCH:g} Å of each other. The point of "
                  "a pair", "is the middle of its two residues on the plane (face on, the spine upright)."))]
    out.append(_spine(px, np.zeros(2), y + FIRST + 4, y + FIRST + 8 + h))
    out.append(openbook.text(float(px(np.zeros(2))[0]) + 5, y + FIRST + 13, "spine", 9.5, FAINT))
    lines, marks, points = [], [], []
    for c, m, own in zip(cx.contacts, g.mids, g.own):
        a, b, p = px(g.place[c.a]), px(g.place[c.b]), px(m)
        tie = f'class="link" data-a="{cx.residues[c.a].key}" data-b="{cx.residues[c.b].key}"'
        said = f"<title>{cx.residues[c.a].label} – {cx.residues[c.b].label}: {c.distance:.1f} Å</title>"
        told = example in (c.a, c.b)
        lines.append(f'<line {tie} x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                     f'stroke="{INK if told else FAINT}" stroke-width="{1.5 if told else 0.8}">{said}</line>')
        points.append(f'<g {tie}><circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="3.3" '
                      f'fill="{openbook.PATCHES[own]}" stroke="{INK}" stroke-width="0.8"/>{said}</g>')
    for i in g.ids:
        r, p = cx.residues[i], px(g.place[i])
        shape = (f'<rect x="{p[0] - 4:.1f}" y="{p[1] - 4:.1f}" width="8" height="8" rx="1" fill="#ffffff" '
                 f'stroke="{EDGE[0]}" '
                 f'stroke-width="1.1"/>' if r.side == 0 else
                 f'<ellipse cx="{p[0]:.1f}" cy="{p[1]:.1f}" rx="4.6" ry="4.6" fill="#ffffff" stroke="{EDGE[1]}" '
                 f'stroke-width="1.3"/>')
        marks.append(f'<g class="res" data-id="{r.key}">{shape}<title>{r.label} (chain {r.chain}): in '
                     f'{int(g.votes[i].sum())} '
                     f'pairs</title></g>')
    out += lines + marks + points
    if example is not None:                                   # the residue that panel 5 tells of, named
        r, p = cx.residues[example], px(g.place[example])
        half = len(r.label) * 2.9 + 3.0
        q = _beside(p, half, 7.0, np.array([px(v) for v in g.place.values()] + [px(m) for m in g.mids]),
                    np.concatenate([openbook.along(px(g.place[c.a]), px(g.place[c.b])) for c in cx.contacts]))
        out.append(f'<g class="res" data-id="{r.key}">{openbook.hairline(q, half, 7.0, p[None, :], edge=True)}'
                   + openbook.text(q[0], q[1] + 3.5, r.label, 9.5, INK, "middle", "600", HALO) + "</g>")
    # The key, beside the drawing.
    kx, ky = x + w + 22.0, y + FIRST + 24.0
    rows = ((f'<rect x="{{x}}" y="{{y0}}" width="8" height="8" rx="1" fill="#fff" stroke="{EDGE[0]}" '
             'stroke-width="1.1"/>',
             "a residue of", names[0]),
            (f'<ellipse cx="{{xm}}" cy="{{ym}}" rx="4.6" ry="4.6" fill="#fff" stroke="{EDGE[1]}" stroke-width="1.3"/>',
             "a residue of", names[1]),
            (f'<line x1="{{xl}}" y1="{{ym}}" x2="{{xr}}" y2="{{ym}}" stroke="{FAINT}" stroke-width="0.9"/>'
             f'<circle cx="{{xm}}" cy="{{ym}}" r="3.3" fill="#e4e7eb" stroke="{INK}" stroke-width="0.8"/>',
             "two that touch,", "and their point"),
            ("", "(a point in the", "colour of its group,"), ("", "see 4)", ""))
    for drawing, one, two in rows:
        out.append(drawing.format(x=f"{kx + 2:.1f}", y0=f"{ky - 7:.1f}", xm=f"{kx + 6:.1f}", ym=f"{ky - 3:.1f}",
                                  xl=f"{kx - 5:.1f}", xr=f"{kx + 17:.1f}"))
        out.append(openbook.text(kx + 22, ky, one, 9.5, QUIET))
        if two:
            out.append(openbook.text(kx + 22, ky + 12.5, two, 9.5, QUIET))
        ky += 34.0 if drawing else 25.0
    return "".join(out), FIRST + h + 14.0


def _borders(g: book.Grouping, px, lo: np.ndarray, hi: np.ndarray) -> "list[tuple[np.ndarray, np.ndarray]]":
    """The lines between the groups: the places as far from one middle as from another and nearer to none,
    inside a rectangle of the plane."""
    out = []
    size = float(np.linalg.norm(hi - lo))
    for i in range(g.k):
        for j in range(i + 1, g.k):
            mid, along = (g.centre[i] + g.centre[j]) / 2.0, g.centre[j] - g.centre[i]
            way = np.array([-along[1], along[0]]) / np.linalg.norm(along)
            t = np.linspace(-size, size, 4001)
            p = mid + t[:, None] * way
            d = np.linalg.norm(p[:, None, :] - g.centre[None, :, :], axis=2)
            ok = (d[:, i] <= d.min(axis=1) + 1e-9) & (p >= lo).all(axis=1) & (p <= hi).all(axis=1)
            if ok.any():
                n = np.where(ok)[0]
                out.append((px(p[n[0]]), px(p[n[-1]])))
    return out


def _groups(cx: book.Complex, g: book.Grouping, x: float, y: float, scale: float) -> "tuple[str, float]":
    w, h, px = _frame(g, x + 6.0, y + FIRST + 6.0, scale, 1.6)
    exact = len(g.ids) / 2.0 / book.PER_REGION
    out = [_head(x, y, 4, "The points are put into groups",
                 (f"{len(g.ids)} residues / 2 partners / {book.PER_REGION:g} to a group = {exact:.1f}: {g.k} groups "
                  "(2 at least, 6 "
                  f"at most).", "A point goes to the nearest middle, a middle to the mean of its points (k-means)."))]
    at = np.array(list(g.place.values()))
    lo, hi = at.min(axis=0) - 1.6, at.max(axis=0) + 1.6
    for p, q in _borders(g, px, lo, hi):
        out.append(f'<line x1="{p[0]:.1f}" y1="{p[1]:.1f}" x2="{q[0]:.1f}" y2="{q[1]:.1f}" stroke="{QUIET}" '
                   f'stroke-width="0.9"/>')
    for c, m, own in zip(cx.contacts, g.mids, g.own):
        p = px(m)
        out.append(f'<g class="link" data-a="{cx.residues[c.a].key}" data-b="{cx.residues[c.b].key}"><circle '
                   f'cx="{p[0]:.1f}" '
                   f'cy="{p[1]:.1f}" r="5" fill="{openbook.PATCHES[own]}" stroke="{INK}" stroke-width="0.8"/><title>'
                   f'{cx.residues[c.a].label} – {cx.residues[c.b].label}: group {openbook.ROMAN[own]}</title></g>')
    for n in g.starts:
        p = px(g.mids[n])
        out.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="8.6" fill="none" stroke="{INK}" stroke-width="1.3"/>')
    for c in g.centre:
        p = px(c)
        for colour, wide in (("#ffffff", 4.2), (INK, 2.2)):
            out.append(f'<path d="M {p[0] - 6:.1f} {p[1] - 6:.1f} l 12 12 M {p[0] - 6:.1f} {p[1] + 6:.1f} l 12 -12" '
                       f'stroke="{colour}" stroke-width="{wide}" stroke-linecap="round"/>')
    # How many points each group holds, and what the other marks are, beside the drawing.
    kx, ky = x + w + 22.0, y + FIRST + 24.0
    for n in range(g.k):
        held = int((g.own == n).sum())
        out.append(f'<circle cx="{kx + 6:.1f}" cy="{ky - 3.5:.1f}" r="5" fill="{openbook.PATCHES[n]}" stroke="{INK}" '
                   f'stroke-width="0.8"/>' + openbook.text(kx + 22, ky, f"group {openbook.ROMAN[n]}: {held} "
                                                                        "points", 9.5, QUIET))
        ky += 20.0
    ky += 12.0
    out.append(f'<circle cx="{kx + 6:.1f}" cy="{ky - 3.5:.1f}" r="8" fill="none" stroke="{INK}" stroke-width="1.3"/>'
               + openbook.text(kx + 22, ky, "a first middle: the", 9.5, QUIET)
               + openbook.text(kx + 22, ky + 12.5, "points farthest apart", 9.5, QUIET))
    ky += 36.0
    out.append(f'<path d="M {kx + 1:.1f} {ky - 8.5:.1f} l 10 10 M {kx + 1:.1f} {ky + 1.5:.1f} l 10 -10" stroke="{INK}" '
               f'stroke-width="2.2" stroke-linecap="round"/>'
               + openbook.text(kx + 22, ky, "the middle in the end", 9.5, QUIET))
    ky += 22.0
    out.append(f'<line x1="{kx - 3:.1f}" y1="{ky - 3.5:.1f}" x2="{kx + 15:.1f}" y2="{ky - 3.5:.1f}" stroke="{QUIET}" '
               f'stroke-width="0.9"/>' + openbook.text(kx + 22, ky, "between two groups", 9.5, QUIET))
    return "".join(out), FIRST + h + 14.0


# ---------------------------------------------------------------------------------------- 5: the regions
def _example(g: book.Grouping) -> "int | None":
    """A residue whose pairs are in more than one group: the one with the most pairs (and the clearest vote)."""
    split = [i for i in g.ids if (g.votes[i] > 0).sum() > 1 and (g.votes[i] == g.votes[i].max()).sum() == 1]
    return max(split, key=lambda i: (g.votes[i].sum(), g.votes[i].max(), -i)) if split else None


def _told(cx: book.Complex, g: book.Grouping, i: "int | None") -> str:
    """That residue's vote, in words."""
    if i is None:
        return ""
    order = sorted((n for n in range(g.k) if g.votes[i][n] > 0), key=lambda n: -g.votes[i][n])
    told = " and ".join(f"{int(g.votes[i][n])} in {openbook.ROMAN[n]}" for n in order)
    return (f"{cx.residues[i].label} of {html.escape(cx.names[cx.residues[i].side])} has {int(g.votes[i].sum())} "
            f"pairs, {told}: it is of {openbook.ROMAN[order[0]]}.")


def _faces(cx: book.Complex, g: book.Grouping, x: float, y: float, example: "int | None") -> "tuple[str, float]":
    """The two opened faces, the regions outlined and numbered as on every figure of the open book."""
    opened, margin, gutter = cx.opened(), 6.5, 60.0
    metric = openbook.metric_of(cx, "region")
    colours = openbook.atom_colours(cx, metric)
    ids = (cx.at_interface(0), cx.at_interface(1))
    at = {i: opened[cx.residues[i].near].mean(axis=0) for i in ids[0] + ids[1]}
    ys = [p[1] for p in at.values()]
    y0, y1 = min(ys) - margin, max(ys) + margin
    spans = []
    for s in (0, 1):
        xs = [at[i][0] for i in ids[s]]
        spans.append((min(xs) - margin, max(xs) + margin))
    scale = min(10.0, (WIDTH - gutter) / sum(b - a for a, b in spans))
    alone = sum(1 for i in g.ids if g.votes[i].sum() == 0)
    between = sum(1 for c in cx.contacts if cx.residues[c.a].region != cx.residues[c.b].region)
    lines = [f"{_told(cx, g, example)} {alone} of the {len(g.ids)} residues have no pair and go to the nearest "
             "middle.",
             f"The regions are numbered down the spine; {len(g.mids) - between} of the {len(g.mids)} pairs lie inside "
             f"one region, {between} reach into another."]
    out = [_head(x, y, 5, "A residue goes to the group that holds most of its pairs: the regions of the two faces",
                 lines)]
    total = sum(b - a for a, b in spans) * scale + gutter
    ox, top = x + max(0.0, (WIDTH - total) / 2.0), y + FIRST + 24.0
    tall = 0.0
    for s, (x0, x1) in enumerate(spans):
        src, w, h, front = openbook.picture(opened, cx, colours, (x0, x1, y0, y1), scale, which=cx.side == s, fade=2.0,
                                            regions=True)
        tall = h
        out.append(openbook.text(ox + w / 2.0, top - 7, f"the face of {html.escape(cx.names[s])}", 11,
                                 EDGE[1] if s else INK, "middle", "600"))
        out.append(f'<image class="ground" href="{src}" x="{ox:.1f}" y="{top:.1f}" width="{w}" height="{h}"/>')
        middles: "dict[int, np.ndarray]" = {}
        for n in range(cx.n_regions):
            pts = [[ox + (at[i][0] - x0) * scale, top + (y1 - at[i][1]) * scale] for i in ids[s]
                   if cx.residues[i].region == n]
            if pts:
                middles[n] = np.mean(pts, axis=0)
        boxes = []
        if example in ids[s]:
            q = np.array([ox + (at[example][0] - x0) * scale, top + (y1 - at[example][1]) * scale])
            boxes = [(q[0], q[1] - 7.0, 24.0, 16.0)]
            r = cx.residues[example]
            out.append(f'<g class="res" data-id="{r.key}"><circle cx="{q[0]:.1f}" cy="{q[1]:.1f}" r="3" fill="{INK}" '
                       f'stroke="#ffffff" stroke-width="1.2"/>'
                       + openbook.text(q[0], q[1] - 9, r.label, 9.5, INK, "middle", "600", HALO) + "</g>")
        out += openbook.numerals(cx, s, middles, boxes, (ox, top, ox + w, top + h), front, radius=10.0)
        for i in ids[s]:
            r = cx.residues[i]
            px, py = ox + (at[i][0] - x0) * scale, top + (y1 - at[i][1]) * scale
            out.append(f'<g class="res" data-id="{r.key}"><ellipse cx="{px:.1f}" cy="{py:.1f}" rx="8" ry="8" '
                       f'fill="#000000" fill-opacity="0"/><g class="reach">'
                       + openbook.text(px, py - 12, r.label if i != example else "", 9.5, INK, "middle", "600", HALO)
                       + f'</g><title>{r.label} (chain {r.chain}): region {openbook.ROMAN[r.region]}</title></g>')
        if s == 0:
            sx = ox + w + gutter / 2.0
            out.append(f'<line x1="{sx:.1f}" y1="{top + 4:.1f}" x2="{sx:.1f}" y2="{top + h - 4:.1f}" stroke="{FAINT}" '
                       f'stroke-width="0.8" stroke-dasharray="5 3"/>')
        ox += w + gutter
    right = x + WIDTH - 2.0
    out.append(f'<line x1="{right - 10 * scale:.1f}" y1="{top + tall + 6:.1f}" x2="{right:.1f}" '
               f'y2="{top + tall + 6:.1f}" '
               f'stroke="{INK}" stroke-width="1.2"/>' + openbook.text(right, top + tall + 2, "10 Å", 9.5, QUIET, "end"))
    return "".join(out), top + tall - y + 12.0


def figure(cx: book.Complex) -> openbook.Sketch:
    """How the regions of this interface come about, in five panels: **1** the atoms at the interface, **2** the
    plane through them and its long axis, **3** the pairs of residues that touch, as points on the plane,
    **4** the points in groups, **5** the regions on the two opened faces. ``cx`` is a complex as
    :func:`book.load` reads one. A group has one colour in 3, 4 and 5.
    """
    g = book.grouping(cx)
    if g is None:
        raise ValueError("no two residues of the two partners touch: there is nothing to group")
    if cx.n_regions != g.k:
        raise ValueError("the regions of this complex are not those of its grouping (book.group)")
    one, h1 = _atoms(cx, 0.0, 0.0)
    two, h2 = _plane(cx, HALF + GAP, 0.0)
    y = max(h1, h2) + 26.0
    at = np.array(list(g.place.values()))
    reach = np.ptp(at, axis=0) + 2 * 1.6
    scale = min(11.0, (HALF - 140.0) / reach[1], 360.0 / reach[0])
    example = _example(g)
    three, h3 = _pairs(cx, g, 0.0, y, scale, example)
    four, h4 = _groups(cx, g, HALF + GAP, y, scale)
    y5 = y + max(h3, h4) + 26.0
    five, h5 = _faces(cx, g, 0.0, y5, example)
    height = y5 + h5
    svg = (f'<svg class="book" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH:.0f} {height:.0f}" '
           f'width="{WIDTH:.0f}" font-family="{openbook.FONT}">' + one + two + three + four + five + "</svg>")
    between = sum(1 for c in cx.contacts if cx.residues[c.a].region != cx.residues[c.b].region)
    near = [len(_near(cx, s)) for s in (0, 1)]
    facts = {"names": list(cx.names), "atoms": near, "at_the_interface": [len(cx.at_interface(s)) for s in (0, 1)],
             "pairs": len(g.mids), "regions": g.k, "points_per_group": [int((g.own == n).sum()) for n in range(g.k)],
             "rounds": g.rounds, "between_regions": between, "named": int(example is not None),
             "without_a_pair": sum(1 for i in g.ids if g.votes[i].sum() == 0)}
    facts["said"] = (f"{near[0]} atoms of {cx.names[0]} and {near[1]} of {cx.names[1]} are within {book.NEAR:g} Å of "
                     f"the partner; the {len(g.mids)} pairs of residues with atoms within {book.TOUCH:g} Å fall into "
                     f"{g.k} groups of {', '.join(str(v) for v in facts['points_per_group'][:-1])} and "
                     f"{facts['points_per_group'][-1]}. {len(g.mids) - between} of the pairs lie inside one region, {between} reach into another.")
    return openbook.Sketch(svg, "how the regions are calculated", facts)
