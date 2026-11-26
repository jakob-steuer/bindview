"""Figures for the cases where an atom-level diagram is not the right answer.

Two situations, kept strictly apart because they mean different things:

* **empty** -- the complex was analysed and nothing was found. The figure shows
  the ligand and says so in words, so nobody mistakes it for a diagram whose
  annotations failed to render.
* **summary** -- there are interactions, but too many, on too large a ligand,
  for an atom-level drawing to be legible at the requested figure size. Rather
  than a hairball, the contacts are tabulated per residue, and the figure says
  why it is a table.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..confidence import MARK, look_of
from ..interactions import InteractionType
from ..scene import Circle, Group, Item, Line, Paint, Rect, Text
from ..textmetrics import CAP_HEIGHT, text_width
from ..theme import Metrics
from . import absence as other_protein
from .base import LINK_STYLES, ResidueGroup

__all__ = ["wrap_text", "text_block", "summary_table"]

T = InteractionType
_ORDER = [T.HBOND, T.SALTBRIDGE, T.PI_STACKING, T.PI_CATION, T.HALOGEN, T.METAL, T.WATER_BRIDGE, T.HYDROPHOBIC,
          T.COVALENT]
_SHORT = {T.HBOND: "H-bond", T.SALTBRIDGE: "Salt bridge", T.PI_STACKING: "π–π",
          T.PI_CATION: "π–cation", T.HALOGEN: "Halogen", T.METAL: "Metal",
          T.WATER_BRIDGE: "Water", T.HYDROPHOBIC: "Hydrophobic", T.COVALENT: "Covalent"}


def wrap_text(text: str, size: float, max_width: float, bold: bool = False) -> list[str]:
    """Greedy word wrap using the metrics table."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = word if not cur else f"{cur} {word}"
        if cur and text_width(trial, size, bold) > max_width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def text_block(lines: Sequence[str], x: float, y: float, size: float, fill: str, bold: bool = False
               ) -> tuple[list[Item], float]:
    """Stacked lines of text starting at top ``y``, set from the left. Returns ``(items, height)``."""
    items: list[Item] = []
    step = size * 1.45
    for k, line in enumerate(lines):
        items.append(Text.plain(x, y + k * step + CAP_HEIGHT * size, line, size, fill, anchor="start", bold=bold))
    return items, (len(lines) * step if lines else 0.0)


_ABBREVIATED = {T.SALTBRIDGE: "Salt br.", T.PI_STACKING: "\u03c0\u2013\u03c0", T.PI_CATION: "\u03c0\u2013cat.",
                T.HYDROPHOBIC: "Hydroph.", T.WATER_BRIDGE: "Water", T.HALOGEN: "Halog."}


def summary_table(groups: Sequence[ResidueGroup], metrics: Metrics, max_width: float,
                  plddt: "Mapping[tuple[str, int, str], float] | None" = None) -> tuple[Group, float, float]:
    """Residue-by-type contact table. Returns ``(group, width, height)``; origin top-left.

    Long tables are set in as many side-by-side blocks as fit ``max_width``;
    column headings are abbreviated when that is what it takes to fit a second
    block, because a tall single column wastes the figure. With ``plddt``
    (residue key -> value) a confidence column is added and a residue below 70
    is set in regular rather than bold type, below 50 with the mark.
    """
    kinds = [k for k in _ORDER if any(k in g.types for g in groups)]
    font = metrics.legend_font
    head_font = font * 0.9
    row_h = font * 1.7
    name_w = max((text_width(g.label + "\u2020", font, bold=True) for g in groups), default=40.0) + 8.0
    dist_w = text_width("0.00 \u00c5", font) + 8.0
    conf_w = text_width("pLDDT", head_font) + 12.0 if plddt else 0.0
    gutter = 26.0

    def plan(headers: dict) -> tuple[list[float], float, int]:
        cols = [max(text_width(headers[k], head_font) + 8.0, 30.0) for k in kinds]
        width = name_w + sum(cols) + conf_w + dist_w
        return cols, width, max(1, int((max_width + gutter) // (width + gutter)))

    headers = dict(_SHORT)
    cols, block_w, fit = plan(headers)
    if len(groups) > 16:
        short = {**_SHORT, **_ABBREVIATED}
        cols2, block_w2, fit2 = plan(short)
        if fit2 > fit:
            headers, cols, block_w, fit = short, cols2, block_w2, fit2
    n_blocks = max(1, min(fit, -(-len(groups) // 10)))      # no block shorter than about ten rows
    rows = -(-len(groups) // n_blocks)
    offs = [name_w]
    for c in cols:
        offs.append(offs[-1] + c)

    g = Group("summary-table")
    head_h = row_h * 1.15
    for blk in range(n_blocks):
        x0 = blk * (block_w + gutter)
        part = list(groups[blk * rows:(blk + 1) * rows])
        if not part:
            continue
        g.add(Text.plain(x0, CAP_HEIGHT * font, "Residue", head_font, "text_muted", anchor="start"))
        for c, k in enumerate(kinds):
            g.add(Text.plain(x0 + offs[c] + cols[c] / 2.0, CAP_HEIGHT * font, headers[k], head_font,
                             LINK_STYLES[k].token))
        g.add(Text.plain(x0 + block_w, CAP_HEIGHT * font, "closest", head_font, "text_muted", anchor="end"))
        if plddt:
            g.add(Text.plain(x0 + offs[-1] + conf_w / 2.0, CAP_HEIGHT * font, "pLDDT", head_font, "text_muted"))
        g.add(Line(x0, head_h - 3.0, x0 + block_w, head_h - 3.0, Paint(stroke="panel_border", width=0.8, cap="butt")))
        for r, grp in enumerate(part):
            y = head_h + r * row_h
            if r % 2 == 1:
                g.add(Rect(x0 - 3.0, y, block_w + 6.0, row_h, Paint(fill="panel"), rx=2.0))
            base = y + row_h / 2.0 + CAP_HEIGHT * font / 2.0
            if grp.lost and not grp.faint:
                name = text_width(grp.label, font, bold=True)
                g.add(Group("absence", [other_protein.plate(x0 + name / 2.0, y + row_h / 2.0, name, CAP_HEIGHT * font)],
                            data={"residue": grp.ident}, opacity=other_protein.OPACITY))
            value = plddt.get(grp.key) if plddt else None
            look = look_of(value)
            g.add(Text.plain(x0, base, grp.label + ("\u2020" if grp.beyond else ""), font,
                             "beyond" if grp.beyond or grp.faint else "text", anchor="start",
                             bold=look.level == 0 and not grp.faint, italic=grp.faint))
            if value is not None:
                g.add(Text.plain(x0 + offs[-1] + conf_w / 2.0, base, f"{value:.0f}" + (MARK if look.mark else ""),
                                 font, "text" if look.level == 0 else "text_muted", bold=look.level == 0))
            for c, k in enumerate(kinds):
                n = sum(1 for i in grp.interactions if i.type is k)
                if not n:
                    continue
                cx = x0 + offs[c] + cols[c] / 2.0
                g.add(Circle(cx - (5.5 if n > 1 else 0.0), y + row_h / 2.0, 3.4, Paint(fill=LINK_STYLES[k].token)))
                if n > 1:
                    g.add(Text.plain(cx + 1.0, base, f"\u00d7{n}", head_font, LINK_STYLES[k].token, anchor="start"))
            ds = [i.record.distance for i in grp.interactions if i.record.distance is not None]
            if ds:
                g.add(Text.plain(x0 + block_w, base, f"{min(ds):.2f} \u00c5", font, "text_muted", anchor="end"))
    width = n_blocks * block_w + (n_blocks - 1) * gutter
    return g, width, head_h + rows * row_h
