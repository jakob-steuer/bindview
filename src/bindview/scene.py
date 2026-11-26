"""A tiny retained scene graph.

Renderers do not write SVG. They build a :class:`Scene` out of primitives whose
colours are palette *tokens*, and :mod:`bindview.svg` serialises it. Two things
follow: styles share all the plumbing (a second style is a different way of
filling a scene, nothing more), and the geometry that was drawn can be inspected
by tests -- "nothing overlaps" is checked on the scene, not on pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

__all__ = ["Paint", "Line", "Polygon", "Polyline", "Circle", "Rect", "Path", "Text", "Span",
           "Group", "Scene", "Item", "item_bounds", "items_bounds", "text_box"]


@dataclass(frozen=True)
class Paint:
    """How a primitive is painted. ``stroke`` / ``fill`` are palette tokens
    (``"hbond"``, ``"element_N"``) or ``None``; a token starting with ``#`` is a
    literal colour and should only come from a caller-supplied palette."""

    stroke: str | None = None
    fill: str | None = None
    width: float = 1.0
    dash: tuple[float, ...] = ()
    opacity: float = 1.0
    cap: str = "round"
    join: str = "round"


@dataclass(frozen=True)
class Line:
    x1: float
    y1: float
    x2: float
    y2: float
    paint: Paint


@dataclass(frozen=True)
class Polyline:
    points: tuple[tuple[float, float], ...]
    paint: Paint
    closed: bool = False


@dataclass(frozen=True)
class Polygon:
    points: tuple[tuple[float, float], ...]
    paint: Paint


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float
    paint: Paint


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float
    paint: Paint
    rx: float = 0.0


@dataclass(frozen=True)
class Path:
    """Raw path data, already in final coordinates: a list of ``(cmd, numbers...)``."""

    commands: tuple[tuple, ...]
    paint: Paint
    bbox: tuple[float, float, float, float] | None = None      # ink box, when the commands include arcs


@dataclass(frozen=True)
class Span:
    """A run of text. ``dy`` shifts the baseline (sub/superscripts)."""

    text: str
    size: float | None = None
    dy: float = 0.0
    fill: str | None = None
    bold: bool | None = None
    italic: bool = False


@dataclass(frozen=True)
class Text:
    """Text anchored at ``(x, y)`` where ``y`` is the *baseline*."""

    x: float
    y: float
    spans: tuple[Span, ...]
    size: float
    fill: str
    anchor: str = "middle"          # start | middle | end
    bold: bool = False
    italic: bool = False
    opacity: float = 1.0

    @classmethod
    def plain(cls, x: float, y: float, text: str, size: float, fill: str, anchor: str = "middle",
              bold: bool = False, italic: bool = False, opacity: float = 1.0) -> "Text":
        return cls(x, y, (Span(text),), size, fill, anchor, bold, italic, opacity)

    @property
    def content(self) -> str:
        return "".join(s.text for s in self.spans)


@dataclass
class Group:
    """A named group. ``cls`` becomes the SVG class, ``data`` become ``data-*``
    attributes, ``title`` a tooltip. ``opacity`` fades the group *as a whole*:
    strokes that overlap inside it (bonds meeting at an atom) do not add up. What is
    not ``bounded`` takes no room: the page is not made larger for it."""

    cls: str
    items: list["Item"] = field(default_factory=list)
    data: dict[str, str] = field(default_factory=dict)
    title: str | None = None
    translate: tuple[float, float] | None = None
    opacity: float = 1.0
    bounded: bool = True

    def add(self, *items: "Item") -> "Group":
        self.items.extend(items)
        return self

    def extend(self, items: Sequence["Item"]) -> "Group":
        self.items.extend(items)
        return self

    def copy(self) -> "Group":
        """A group of its own with the same content: the groups in it are copied too, and what else it holds
        cannot be changed, so the two share that."""
        return replace(self, items=_copies(self.items), data=dict(self.data))


Item = Line | Polyline | Polygon | Circle | Rect | Path | Text | Group


@dataclass
class Scene:
    width: float
    height: float
    items: list[Item] = field(default_factory=list)
    title: str = ""
    description: str = ""
    #: The words around the drawing, drawn or left off (:attr:`bindview.diagram.RenderResult.notes`).
    #: Not written into the file.
    notes: list[str] = field(default_factory=list)

    def add(self, *items: Item) -> None:
        self.items.extend(items)

    def copy(self) -> "Scene":
        """A scene of its own with the same content (:meth:`Group.copy`): changing it changes no other."""
        return replace(self, items=_copies(self.items), notes=list(self.notes))


def _copies(items: Sequence[Item]) -> list[Item]:
    return [it.copy() if isinstance(it, Group) else it for it in items]


# -- bounds ------------------------------------------------------------------

def text_box(t: Text) -> tuple[float, float, float, float]:
    """Estimated ink box of a text item ``(x0, y0, x1, y1)`` from the metrics table."""
    from .textmetrics import CAP_HEIGHT, DESCENT, text_width

    width = 0.0
    up, down = CAP_HEIGHT * t.size, 0.0
    shift = 0.0
    for s in t.spans:
        size = s.size if s.size is not None else t.size
        bold = s.bold if s.bold is not None else t.bold
        shift += s.dy
        if s.text:
            width += text_width(s.text, size, bold)
            up = max(up, CAP_HEIGHT * size - shift)
            down = max(down, shift + (DESCENT * size if any(c in "gjpqy()," for c in s.text) else 0.0))
    if t.anchor == "middle":
        x0 = t.x - width / 2.0
    elif t.anchor == "end":
        x0 = t.x - width
    else:
        x0 = t.x
    return (x0, t.y - up, x0 + width, t.y + down)


def item_bounds(it: "Item") -> tuple[float, float, float, float] | None:
    """Bounding box of an item in its parent's coordinates, stroke width included."""
    if isinstance(it, Group):
        b = items_bounds(it.items) if it.bounded else None
        if b is None or it.translate is None:
            return b
        dx, dy = it.translate
        return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)
    if isinstance(it, Text):
        return text_box(it)
    pad = it.paint.width / 2.0 if it.paint.stroke is not None else 0.0
    if isinstance(it, Line):
        return (min(it.x1, it.x2) - pad, min(it.y1, it.y2) - pad, max(it.x1, it.x2) + pad, max(it.y1, it.y2) + pad)
    if isinstance(it, (Polygon, Polyline)):
        xs = [p[0] for p in it.points]
        ys = [p[1] for p in it.points]
        return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    if isinstance(it, Circle):
        return (it.cx - it.r - pad, it.cy - it.r - pad, it.cx + it.r + pad, it.cy + it.r + pad)
    if isinstance(it, Rect):
        return (it.x - pad, it.y - pad, it.x + it.w + pad, it.y + it.h + pad)
    if isinstance(it, Path):
        if it.bbox is not None:
            return (it.bbox[0] - pad, it.bbox[1] - pad, it.bbox[2] + pad, it.bbox[3] + pad)
        xs, ys = [], []
        for c in it.commands:
            nums = [v for v in c[1:] if isinstance(v, float)]
            xs.extend(nums[0::2])
            ys.extend(nums[1::2])
        if not xs:
            return None
        return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    return None


def items_bounds(items: Sequence["Item"]) -> tuple[float, float, float, float] | None:
    boxes = [b for b in (item_bounds(i) for i in items) if b is not None]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
