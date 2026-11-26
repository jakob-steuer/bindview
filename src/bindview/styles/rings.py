"""A residue in its ring: what share of the models of an ensemble contacts it (:mod:`bindview.consensus`).

A disc in the colour of the residue's class, with its name and the number of models, and around it a ring as long
as their share, clockwise from the top. Both styles draw a residue so where a figure counts by residue: in the
place of its bubble, or where the name of a residue stands that is drawn as atoms.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..scene import Circle, Item, Paint, Polyline, Text
from ..textmetrics import CAP_HEIGHT, text_width

if TYPE_CHECKING:
    from ..consensus import ResidueShare

__all__ = ["ring_items", "ring_reach", "ring_title", "ring_sample", "BOLD_FROM"]

#: The letters of the name and, below it, of the number of models.
_NAME, _COUNT = 8.8, 7.6
#: From the edge of the disc to the middle of the ring, the width of the ring, and the room beyond it.
_GAP, _WIDTH, _PAD = 4.4, 3.4, 0.9
_LEAST_RADIUS = 15.0
#: From this share on the name is bold.
BOLD_FROM = 0.5


def _count(share: "ResidueShare") -> str:
    return f"{share.n_models}/{share.n_total}"


def _radius(label: str, share: "ResidueShare") -> float:
    wide = max(text_width(label, _NAME, bold=True), text_width(_count(share), _COUNT))
    return max(_LEAST_RADIUS, math.hypot(wide / 2.0, CAP_HEIGHT * _NAME + 1.0) + 1.6)


def ring_reach(label: str, share: "ResidueShare") -> float:
    """How far a residue in its ring reaches from its middle."""
    return _radius(label, share) + _GAP + _WIDTH / 2.0 + _PAD


def ring_title(label: str, share: "ResidueShare") -> str:
    return f"{label}: {share.said}"


def _arc(x: float, y: float, r: float, share: float) -> Item:
    if share >= 1.0 - 1e-9:
        return Circle(x, y, r, Paint(stroke="text", width=_WIDTH))
    n = max(2, int(math.ceil(2.0 * math.pi * share * r / 2.0)))
    turn = [-math.pi / 2.0 + 2.0 * math.pi * share * k / n for k in range(n + 1)]
    return Polyline(tuple((x + r * math.cos(a), y + r * math.sin(a)) for a in turn),
                    Paint(stroke="text", width=_WIDTH, cap="butt"))


def ring_items(x: float, y: float, label: str, share: "ResidueShare", fill: str, ink: str = "bubble_text") -> list[Item]:
    """A residue in its ring, at (x, y)."""
    radius = _radius(label, share)
    return [Circle(x, y, radius, Paint(fill=fill, stroke="bubble_line", width=0.8, opacity=0.55 + 0.45 * share.share)),
            Circle(x, y, radius + _GAP, Paint(stroke="panel_border", width=0.8)),
            _arc(x, y, radius + _GAP, share.share),
            Text.plain(x, y - 0.6, label, _NAME, ink, bold=share.share >= BOLD_FROM),
            Text.plain(x, y + CAP_HEIGHT * _COUNT + 2.0, _count(share), _COUNT, ink)]


def ring_sample(x: float, y: float, radius: float) -> list[Item]:
    """The ring of the key: that of a residue that three of four models contact."""
    return [Circle(x, y, radius - 1.2, Paint(stroke="panel_border", width=0.8)),
            *_thin(_arc(x, y, radius - 1.2, 0.75))]


def _thin(arc: Item) -> list[Item]:
    assert isinstance(arc, Polyline)
    return [Polyline(arc.points, Paint(stroke="text", width=2.2, cap="butt"))]
