"""Residues that another protein does not have: ``render(absent=[702, 705, ...], absent_label="AR-V7")``.

One ligand, one pocket and two proteins. The figure is of the protein that was modelled; the mask names the residues
that the other one (a splice isoform, a construct, an orthologue) lacks, and the figure marks those of them that
touch the ligand, with their contacts. Nothing is computed for the other protein and nothing moves: the figure is
the one without the mask, with a stroke of a marker pen under what is lost (``absent_style="marked"``), or with
what is lost set back (``"faint"``). The sentence under the key counts them.

This is not what :meth:`bindview.Ensemble.diagram` calls absent, which is a contact that other models make and the
one shown does not. In the figure's data the two are told apart: ``data-absent`` is that, ``data-absent-from`` this.
"""

from __future__ import annotations

import math
import numbers
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ..depict import Depiction
from ..geometry import Point
from ..scene import Circle, Group, Item, Line, Paint, Polyline, Rect

if TYPE_CHECKING:
    from .base import RenderOptions, ResidueGroup

__all__ = ["ABSENT_STYLES", "MARKER", "OPACITY", "FROM", "absent_option", "mark_groups", "Absence", "absence",
           "from_whom", "said_of", "legend_text", "marker_group", "stroke", "under", "skeleton", "plate", "disc", "UNDER"]

ABSENT_STYLES = ("marked", "faint")
#: The colour of the marker pen (a token, :data:`bindview.theme.EXTRAS`), how strong it is, and how wide a stroke is
#: in widths of a contact's line.
MARKER = "absent"
OPACITY = 0.55
_WIDE = 6.6
#: ... and under the atoms of a residue, in bond lengths of its drawing.
UNDER = 0.5
#: What the groups of such a residue and of its contacts say.
FROM = "absent-from"


def _refused(name: Any) -> ValueError:
    return ValueError(f"absent names residues by their number or name, not {name!r}")


def _named(name: Any) -> "str | None":
    """One entry of ``absent`` as a name; None for a gap in a column of numbers (NaN)."""
    if isinstance(name, (bool, np.bool_, bytes, bytearray)):
        raise _refused(name)
    if isinstance(name, numbers.Integral):
        return str(int(name))
    if isinstance(name, numbers.Real):                   # (of Python or of NumPy: a column of numbers is of floats)
        number = float(name)
        if math.isnan(number):
            return None
        if not number.is_integer():
            raise _refused(name)
        return str(int(number))
    return str(name).strip()


def absent_option(value: Any) -> "tuple[str, ...] | None":
    """``absent`` as it is kept: None, or the names in a fixed order (none of them: nothing is absent)."""
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray, numbers.Real, np.bool_)):          # one residue
        name = _named(value)
        if name is None:
            raise _refused(value)
        return (name,)
    return tuple(sorted({name for name in map(_named, value) if name is not None}))


def mark_groups(groups: "Sequence[ResidueGroup]", options: "RenderOptions") -> None:
    """Say of each residue whether the other protein lacks it."""
    if options.absent is None:
        return
    from .base import residue_names

    names = {n.lower() for n in options.absent}
    for grp in groups:
        grp.lost = any(n.lower() in names for n in residue_names(grp))
        grp.faint = grp.lost and options.absent_style == "faint"


class Absence:
    """Which of a figure's contact residues the other protein lacks."""

    def __init__(self, groups: "Sequence[ResidueGroup]", label: "str | None") -> None:
        seen: "dict[tuple, ResidueGroup]" = {}
        for grp in groups:
            # (A predicted water is no contact, nor is one that only other models of an ensemble make.)
            if any(not r.record.predicted and not r.absent for r in grp.interactions):
                seen.setdefault(grp.key, grp)
        self.label = label
        self.absent = [g.ident for g in seen.values() if g.lost]
        self.present = [g.ident for g in seen.values() if not g.lost]

    @property
    def sentence(self) -> str:
        n, k = len(self.absent) + len(self.present), len(self.absent)
        whose = f"absent from {from_whom(self.label)}."
        if n == 1:
            return f"The only contact residue is {'' if k else 'not '}{whose}"
        if not k:
            return f"None of the {n} contact residues is {whose}"
        if k == n:
            return f"All {n} contact residues are {whose}"
        return f"{k} of {n} contact residues {'is' if k == 1 else 'are'} {whose}"

    def as_dict(self) -> "dict[str, Any]":
        return {"from": self.label, "contact_residues": len(self.absent) + len(self.present),
                "absent": list(self.absent), "present": list(self.present)}


def absence(groups: "Sequence[ResidueGroup]", options: "RenderOptions") -> "Absence | None":
    """The mask as it falls on the residues of a figure, where one was asked for."""
    return None if options.absent is None else Absence(groups, options.absent_label)


def from_whom(label: "str | None") -> str:
    return label or "the other protein"


def said_of(options: "RenderOptions") -> str:
    """The value of ``data-absent-from``."""
    return options.absent_label or "true"


def legend_text(options: "RenderOptions") -> str:
    """What the key calls them."""
    whose = f"bsent from {from_whom(options.absent_label)}"
    return f"Faint: a{whose}" if options.absent_style == "faint" else f"A{whose}"


# ---------------------------------------------------------------------------
# The marker pen
# ---------------------------------------------------------------------------

def marker_group(grp: "ResidueGroup", options: "RenderOptions") -> Group:
    """What is marked of one residue, in one group behind everything: as strong as one stroke wherever two meet. It is
    no part of what the page is sized by (a mark at the edge lies in the margin), so that the figure with the mask
    lies on its page where the one without it does."""
    return Group("absence", opacity=OPACITY, data={"residue": grp.ident, FROM: said_of(options)}, bounded=False)


def stroke(points: "Sequence[Point]", line_width: float, wide: float = _WIDE) -> Item:
    """Along a line, or along a run of points."""
    paint = Paint(stroke=MARKER, width=wide * line_width, cap="round", join="round")
    if len(points) == 2:
        (x1, y1), (x2, y2) = points
        return Line(x1, y1, x2, y2, paint)
    return Polyline(tuple(points), paint)


def under(items: "Sequence[Item]", line_width: float) -> "list[Item]":
    """Under the ticks of a hydrophobic contact."""
    return [stroke([(it.x1, it.y1), (it.x2, it.y2)], line_width, 0.55 * _WIDE) for it in items if isinstance(it, Line)]


def skeleton(dep: Depiction, width: float) -> "list[Item]":
    """Under the atoms and bonds that are drawn of a residue."""
    out: list[Item] = []
    for bond in dep.bonds:
        a, b = dep.atoms[bond.a], dep.atoms[bond.b]
        out.append(Line(a.x, a.y, b.x, b.y, Paint(stroke=MARKER, width=width, cap="round")))
    if not dep.bonds:
        out.extend(Circle(a.x, a.y, width / 2.0, Paint(fill=MARKER)) for a in dep.atoms)
    return out


def plate(x: float, y: float, width: float, height: float) -> Rect:
    """Under a name that is centred at ``x``, ``y``."""
    return Rect(x - width / 2.0 - 4.0, y - height / 2.0 - 3.5, width + 8.0, height + 7.0, Paint(fill=MARKER), rx=4.5)


def disc(x: float, y: float, radius: float) -> Circle:
    """Around a bubble or a ring."""
    return Circle(x, y, radius + 5.0, Paint(fill=MARKER))
