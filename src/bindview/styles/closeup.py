"""What a figure that answers the pointer has beside what it draws: the atoms of a residue that is drawn
without them, and a place to point at on every atom of the ligand.

A bubble (or a name, or an eyelash) says which residue makes a contact and not how. Pointed at, it gives way to
its residue as the detailed style draws one, in the figure's own lines and with its distances: the part that
makes the contact and the whole side chain. Each is placed alone, as the engine places a residue beside the
ligand, as near as may be to where the residue stands in the figure and within the figure's frame. The others
have stepped far back by then, so it may lie on them; it never lies on the ligand.

None of it is drawn: the groups are kept out of the scene (so nothing that is measured of a figure knows them),
are written last, and are hidden until the style sheet shows one (:mod:`bindview.interactive`).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from ..layout import Body, LayoutParams, LigandField, Pose
from ..layout.engine import _Engine
from ..scene import Circle, Group, Paint
from .base import Context, Drawn, ResidueGroup, Settings
from .detailed import DetailedStyle

__all__ = ["closeups", "atom_hits", "MORE", "HIT"]

#: The classes of the groups: the atoms of a residue, and a place to point at on an atom of the ligand.
MORE = "more"
HIT = "hit"
#: How much a place costs for each bond length that it is away from where the residue stands in the figure
#: (a contact's line that is a bond length longer than it is best costs 1).
_HOME = 1.4
#: What is drawn of a residue pointed at, where the figure has no level of its own.
_LEVEL = "sidechain"
#: How far into the drawing around an atom the pointer reaches it, in bond lengths.
_HIT_RADIUS = 0.34


def _settings_for(settings: Settings) -> Settings:
    """The detailed style in the lines of the figure, with its distances."""
    return Settings("atoms", settings.lines, False, "none", True)


def _rename(group: Group) -> Group:
    """Its parts under names of their own: what reads the residues and the contacts of a figure by their
    class finds each of them once."""
    for item in group.items:
        if isinstance(item, Group):
            if item.cls.split(" ", 1)[0] in ("residue", "interaction", "contact", "absence"):
                item.cls = f"{MORE}-{item.cls}"
            _rename(item)
    return group


def closeups(ctx: Context, bodies: Sequence[Body], poses: "dict", field: LigandField, params: LayoutParams,
             settings: Settings, frame: "tuple[float, float, float, float]") -> list[Group]:
    """One hidden group for each residue that is drawn without its atoms and has atoms to show. ``frame`` is
    what the figure's drawing covers."""
    if ctx.dep.units:
        return []                              # beside a chain a residue has its place, and no other
    # (The page has a margin on either side of its drawing, and a little room above and below it.)
    m = ctx.metrics
    frame = (frame[0] - 0.7 * m.margin, frame[1] - 6.0, frame[2] + 0.7 * m.margin, frame[3] + 6.0)
    style = DetailedStyle(_settings_for(settings))
    level: "dict[str, str | None]" = {"*": _LEVEL}
    near = replace(ctx, notes=[], detail=level)
    out: list[Group] = []
    for body in bodies:
        pose = poses.get(body.key)
        grp: ResidueGroup = body.payload.group
        if pose is None or body.payload.form == "fragment":
            continue
        drawn = style._fragment_body(near, grp)
        if drawn.kind != "fragment":
            continue                           # a residue there is no template for: its name is all there is
        place = _seat(field, drawn, params, (pose.x, pose.y), frame)
        if place is None:
            continue
        where, label_t = place
        d = Drawn({"backdrop": [], "halos": [], "interactions": [], "contacts": [], "residues": []})
        style.draw_bodies(near, [drawn], {drawn.key: where}, {drawn.key: label_t}, d)
        group = Group(MORE, data={"residue": grp.ident})
        for layer in ("backdrop", "halos", "interactions", "contacts", "residues"):
            group.extend(d.layers.get(layer, []))
        out.append(_rename(group))
    return out


def _seat(field: LigandField, body: Body, params: LayoutParams, home: "tuple[float, float]",
          frame: "tuple[float, float, float, float]") -> "tuple[Pose, tuple[float, ...]] | None":
    engine = _Engine(field, [body], params)
    for wide in (False, True):
        batch = engine._batch(0, wide)
        if not len(batch):
            continue
        geo = batch.geo
        # How far each place reaches beyond the frame.
        r = geo.r[None, :]
        lo = np.minimum(geo.p0, geo.p1)
        hi = np.maximum(geo.p0, geo.p1)
        over = np.maximum.reduce([frame[0] - (lo[..., 0] - r), (hi[..., 0] + r) - frame[2],
                                  frame[1] - (lo[..., 1] - r), (hi[..., 1] + r) - frame[3]]).max(axis=1)
        inside = over <= 0.5
        if not inside.any() and not wide:
            continue
        away = np.linalg.norm(batch.pos - np.array(home), axis=1) / params.unit
        cost = np.round(batch.base + _HOME * away + np.where(inside, 0.0, 10.0 + np.maximum(over, 0.0)), 9)
        k = int(np.argmin(cost))
        pose = Pose(float(batch.pos[k, 0]), float(batch.pos[k, 1]), float(batch.ang[k]), bool(batch.mir[k]))
        return pose, tuple(float(t) for t in batch.label_t[k])
    return None


def atom_hits(ctx: Context) -> list[Group]:
    """A place to point at on every atom of a ligand that is drawn as atoms."""
    if ctx.dep.units:
        return []
    made: "dict[int, list[str]]" = {}
    whose: "dict[int, list[str]]" = {}             # an atom: the residues it makes a contact with
    for grp in ctx.groups:
        for r in grp.interactions:
            if r.ligand is not None:
                for i in r.ligand.atoms:
                    said = f"{r.type.label.lower()} with {grp.label}"
                    if said not in made.setdefault(i, []):
                        made[i].append(said)
                    if grp.ident not in whose.setdefault(i, []):
                        whose[i].append(grp.ident)
    out = []
    for atom in ctx.dep.atoms:
        if atom.idx >= ctx.ligand.n_atoms:
            continue
        name = ctx.ligand.names[atom.idx]
        title = f"{name}: " + (", ".join(made[atom.idx]) if atom.idx in made else "no contact")
        if ctx.confidence is not None and ctx.options.confidence:
            value = ctx.confidence.ligand([name])
            if value is not None:
                title += f"; pLDDT {value:.0f}"
        data = {"atom": name}
        if atom.idx in whose:
            data["residues"] = ",".join(whose[atom.idx])
        out.append(Group(HIT, [Circle(atom.x, atom.y, _HIT_RADIUS * ctx.metrics.bond_length, Paint())],
                         data=data, title=title))
    return out
