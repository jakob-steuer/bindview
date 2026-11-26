"""A contact's line, kept off words that are not its own.

The layout seats residues so that their lines miss other residues and, where it can, bonds. A line can
still run over a word: the letter of a ligand atom that lies on its way to a ring centre, the name of a
residue, the distance written on another line. Where it does, the line is broken around the word, the way
:func:`bindview.styles.base.link_items` breaks it around its own distance, so the word stays readable on any
backdrop and nothing needs a patch behind it.

A line that ends in a word is pointing at it and is left alone: the line to a residue drawn as a bubble
runs to the middle of the bubble, under its name. A figure in which no line runs over a word is left
exactly as it is.
"""

from __future__ import annotations

from .scene import Group, Item, Line, Text, text_box

__all__ = ["keep_lines_off_words", "INSIDE", "ROOM", "KEPT"]

#: A line runs over a word when it passes this far inside the word's box; touching the edge is not it.
INSIDE = 0.75
#: What is left free beside and above a word: what a line leaves around its own distance.
ROOM = (1.5, 1.7)
#: A line is broken only while this share of it is left; a short line that would all but vanish stays whole.
#: The share is of the piece, or of all that is drawn of its contact: a line split by its own distance or
#: tag is two pieces, and the shorter of them can lose most of itself to a word while the line keeps most.
KEPT = 0.5

Box = tuple[float, float, float, float]
Point = tuple[float, float]


def _span(a: Point, b: Point, box: Box) -> "tuple[float, float] | None":
    """The stretch of ``a``-``b`` inside ``box``, as two shares of its length."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - box[0]), (dx, box[2] - a[0]), (-dy, a[1] - box[1]), (dy, box[3] - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
    return (t0, t1) if t0 < t1 else None


def _is_contact(group: Group) -> bool:
    return group.cls.split(" ", 1)[0] == "interaction"


def keep_lines_off_words(content: Group) -> int:
    """Break the lines of the contacts in ``content`` around the words they run over; how many were broken."""
    words: list[tuple[Box, int]] = []                       # box in the coordinates of ``content``, whose it is
    lines: list[tuple[Group, int, Line, float, float, int]] = []

    def walk(group: Group, dx: float, dy: float, mine: int) -> None:
        for k, it in enumerate(group.items):
            if isinstance(it, Group):
                tx, ty = it.translate or (0.0, 0.0)
                walk(it, dx + tx, dy + ty, id(it) if _is_contact(it) else mine)
            elif isinstance(it, Text):
                if it.content.strip():
                    b = text_box(it)
                    words.append(((b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy), mine))
            elif isinstance(it, Line) and mine and it.paint.stroke is not None:
                lines.append((group, k, it, dx, dy, mine))

    walk(content, 0.0, 0.0, 0)
    broken = 0
    replace: dict[int, dict[int, list[Line]]] = {}
    groups: dict[int, Group] = {}
    plans: list[tuple[Group, int, Line, list[tuple[float, float]], float, int]] = []
    drawn: dict[int, float] = {}                           # length of every contact's line, all its pieces
    lost: dict[int, float] = {}                            # ... and what breaking them would take away
    for group, k, line, dx, dy, mine in lines:
        a, b = (line.x1 + dx, line.y1 + dy), (line.x2 + dx, line.y2 + dy)
        length = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        drawn[mine] = drawn.get(mine, 0.0) + length
        if length < 1e-6:
            continue
        gaps: list[tuple[float, float]] = []
        for box, whose in words:
            if whose == mine:
                continue
            if _span(a, b, (box[0] + INSIDE, box[1] + INSIDE, box[2] - INSIDE, box[3] - INSIDE)) is None:
                continue
            gap = _span(a, b, (box[0] - ROOM[0], box[1] - ROOM[1], box[2] + ROOM[0], box[3] + ROOM[1]))
            if gap is None or gap[0] <= 1e-9 or gap[1] >= 1.0 - 1e-9:
                continue                                    # it ends at the word: that is pointing at it
            gaps.append(gap)
        if not gaps:
            continue
        gaps.sort()
        left: list[tuple[float, float]] = []
        at = 0.0
        for t0, t1 in gaps:
            if t0 > at:
                left.append((at, t0))
            at = max(at, t1)
        if at < 1.0:
            left.append((at, 1.0))
        left = [(t0, t1) for t0, t1 in left if (t1 - t0) * length > 0.4]
        plans.append((group, k, line, left, length, mine))
        lost[mine] = lost.get(mine, 0.0) + length * (1.0 - sum(t1 - t0 for t0, t1 in left))
    for group, k, line, left, length, mine in plans:
        if sum(t1 - t0 for t0, t1 in left) < KEPT and (not left or 1.0 - lost[mine] / drawn[mine] < KEPT):
            continue                                        # (a piece that would vanish is never broken away)
        pieces = [Line(line.x1 + (line.x2 - line.x1) * t0, line.y1 + (line.y2 - line.y1) * t0,
                       line.x1 + (line.x2 - line.x1) * t1, line.y1 + (line.y2 - line.y1) * t1, line.paint)
                  for t0, t1 in left]
        replace.setdefault(id(group), {})[k] = pieces
        groups[id(group)] = group
        broken += 1
    for key, by_index in replace.items():
        group = groups[key]
        items: list[Item] = []
        for k, it in enumerate(group.items):
            items.extend(by_index.get(k, [it]))
        group.items = items
    return broken
