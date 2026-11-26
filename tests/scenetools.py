"""Reading geometry back out of a rendered scene, in page coordinates."""

from __future__ import annotations

from dataclasses import dataclass

from bindview.scene import Circle, Group, Line, Path, Polygon, Polyline, Scene, Text, text_box


@dataclass
class Found:
    item: object
    path: tuple[Group, ...]
    dx: float
    dy: float

    def classes(self) -> list[str]:
        return [g.cls for g in self.path]

    def in_class(self, name: str) -> Group | None:
        """Innermost enclosing group whose class is ``name`` or ``name <something>``."""
        for g in reversed(self.path):
            if g.cls == name or g.cls.startswith(name + " "):
                return g
        return None

    def box(self) -> tuple[float, float, float, float]:
        it = self.item
        if isinstance(it, Text):
            b = text_box(it)
        elif isinstance(it, Line):
            b = (min(it.x1, it.x2), min(it.y1, it.y2), max(it.x1, it.x2), max(it.y1, it.y2))
        elif isinstance(it, (Polygon, Polyline)):
            xs, ys = [p[0] for p in it.points], [p[1] for p in it.points]
            b = (min(xs), min(ys), max(xs), max(ys))
        elif isinstance(it, Circle):
            b = (it.cx - it.r, it.cy - it.r, it.cx + it.r, it.cy + it.r)
        elif isinstance(it, Path) and it.bbox:
            b = it.bbox
        else:
            raise TypeError(type(it).__name__)
        return (b[0] + self.dx, b[1] + self.dy, b[2] + self.dx, b[3] + self.dy)

    def segment(self) -> tuple[tuple[float, float], tuple[float, float]]:
        it = self.item
        assert isinstance(it, Line)
        return (it.x1 + self.dx, it.y1 + self.dy), (it.x2 + self.dx, it.y2 + self.dy)


def walk(scene: Scene) -> list[Found]:
    out: list[Found] = []

    def rec(items, path, dx, dy):
        for it in items:
            if isinstance(it, Group):
                tx, ty = it.translate or (0.0, 0.0)
                rec(it.items, path + (it,), dx + tx, dy + ty)
            else:
                out.append(Found(it, path, dx, dy))
    rec(scene.items, (), 0.0, 0.0)
    return out


def overlap(a, b, slack: float = 0.75) -> bool:
    """Do two boxes overlap by more than ``slack`` in both directions?"""
    return (min(a[2], b[2]) - max(a[0], b[0]) > slack) and (min(a[3], b[3]) - max(a[1], b[1]) > slack)


def segment_hits_box(p, q, box, pad: float = 0.0) -> bool:
    """Does segment ``p-q`` pass through ``box`` shrunk by ``pad``?"""
    x0, y0, x1, y1 = box[0] + pad, box[1] + pad, box[2] - pad, box[3] - pad
    if x0 >= x1 or y0 >= y1:
        return False
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for d, lo, hi, s in ((dx, x0, x1, p[0]), (dy, y0, y1, p[1])):
        if abs(d) < 1e-12:
            if s < lo or s > hi:
                return False
            continue
        a, b = (lo - s) / d, (hi - s) / d
        if a > b:
            a, b = b, a
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return False
    return True


# -- what a figure holds, by the classes of its groups ---------------------------------------------

def groups(result, name):
    """Groups whose class is ``name`` or ``name <something>`` (not the layer group ``names``)."""
    seen, out = set(), []
    for f in walk(result.scene):
        for g in f.path:
            if (g.cls == name or g.cls.startswith(name + " ")) and id(g) not in seen:
                seen.add(id(g))
                out.append(g)
    return out


def inside(result, group, kind=None):
    """What the scene of ``result`` holds inside ``group``. Every result has a scene of its own: a group that was
    taken from another result of the same figure is found by what it holds."""
    found = [f for f in walk(result.scene) if any(g is group for g in f.path)]
    if not found:
        found = [f for f in walk(result.scene) if any(g == group for g in f.path)]
    return [f for f in found if kind is None or isinstance(f.item, kind)]


def interactions(result, type_=None, residue=None):
    out = []
    for g in groups(result, "interaction"):
        if type_ and g.data.get("type") != type_:
            continue
        if residue and not g.data.get("residue", "").endswith(residue.upper()):
            continue
        out.append(g)
    return out


def texts_with_owner(result):
    """Text items of the diagram proper with an owner id (runs of one atom label share an owner)."""
    out = []
    for f in walk(result.scene):
        if not isinstance(f.item, Text) or f.in_class("diagram") is None:
            continue
        atom = f.in_class("atom")
        out.append((f.box(), id(atom) if atom is not None else id(f.item), f))
    return out


def words(res) -> str:
    return " ".join(" ".join(f.item.content for f in walk(res.scene) if isinstance(f.item, Text)).split())


def tags(res) -> list[Text]:
    """The PAE tags written on contact lines."""
    return [f.item for f in walk(res.scene)
            if isinstance(f.item, Text) and f.item.content.startswith("PAE ") and f.item.content[4:].isdigit()
            and f.in_class("legend") is None]


def key(res) -> list[Text]:
    """The legend's samples of a PAE tag: they read like tags and are none."""
    return [f.item for f in walk(res.scene)
            if isinstance(f.item, Text) and f.item.content.startswith("PAE ") and f.item.content[4:].isdigit()
            and f.in_class("legend") is not None]
