"""The schematic style, after Maestro's Ligand Interaction Diagram.

The ligand is drawn exactly as in the detailed style; what changes is how the
protein is shown. Every residue is a bubble coloured by residue class, pointing
at the ligand atoms it contacts. Only directional interactions get a line
(H-bonds as arrows from donor to acceptor, solid for backbone and dashed for
side chain; pi and charge interactions to ring centroids and charged groups);
a hydrophobic contact is a stretch of the lining, and beside a ligand that is
drawn as a chain of residues a fine line as well. The lining is the wall of the
pocket around the ligand: a stretch for each contact, in the colour of the part
of the residue that makes it, and open where nothing does (docs/reference.md). Grey
halos mark ligand atoms that are exposed to solvent.

What is drawn here is how this style draws unless it is told otherwise: the
lines, the lining and the halos are settings (:class:`~.base.Settings`).

This module supplies the bodies and the drawing. The chemistry, the anchor
resolution and the layout engine are those of the detailed style.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..confidence import PLAIN, Look, occupancy_factors
from ..interactions import InteractionType
from ..interactive import NUMBER
from ..layout import Body
from ..scene import Circle, Group, Paint, Path, Text
from ..textmetrics import CAP_HEIGHT, text_width
from . import absence as other_protein
from .base import (_CLASS_LABEL, _CLASS_ORDER, SCHEMATIC_LINKS, SCHEMATIC_SIDECHAIN, STYLE_SETTINGS, Context,
                   Drawn, PortInfo, ResidueGroup, Settings, StyleDrawing, finish_drawing, group_skeleton, halo_items, halo_radius,
                   hbond_direction, interaction_data, interaction_title, is_distance, lining_claims, link_items, make_ports,
                   marked_label, port_infos, predicted_items, predicted_style, residue_occupancy, residue_plddt,
                   set_lanes)
from .rings import ring_items, ring_reach, ring_title

if TYPE_CHECKING:
    from ..consensus import ResidueShare

__all__ = ["SchematicStyle"]

T = InteractionType

_STYLES = SCHEMATIC_LINKS
_HBOND_SIDECHAIN = SCHEMATIC_SIDECHAIN

_FONT = 9.5
_TIP = 1.3                      # teardrop tip distance from the centre, in radii


@dataclass
class _Bubble:
    group: ResidueGroup
    ports: list[PortInfo]
    radius: float
    name: str
    number: str
    plddt: float | None = None             # the residue's own pLDDT
    look: Look = PLAIN
    occupancy: float | None = None         # that of the residue's most persistent contact
    share: "ResidueShare | None" = None    # it stands in a ring: the models that contact it
    form = "bubble"

    @property
    def ident(self) -> str:
        return self.group.node_ident


def _bubble_text(grp: ResidueGroup, look: Look = PLAIN) -> tuple[str, str]:
    chain, number, icode = grp.key
    num = f"{number}{icode}"
    if "(" in grp.label and chain:             # the caller asked for chains to be shown
        num = f"{chain}:{num}"
    return grp.resname.upper(), marked_label(num + ("†" if grp.beyond else ""), look)


#: Outline of a bubble by confidence level: (dash, width factor). Solid, dashed, dotted --
#: a difference in kind, so it survives greyscale where a paler fill alone would not.
_OUTLINE: dict[int, tuple[tuple[float, ...], float]] = {0: ((), 1.0), 1: ((3.6, 2.2), 1.0), 2: ((1.1, 2.3), 1.15)}


class SchematicStyle:
    name = "schematic"

    def __init__(self, settings: "Settings | None" = None) -> None:
        self.settings = settings or STYLE_SETTINGS["schematic"]

    def ligand_obstacles(self, ctx: Context) -> list[tuple[float, float, float]]:
        return []

    # ------------------------------------------------------------------ bodies
    def bodies(self, ctx: Context) -> list[Body]:
        return [self.body(ctx, grp) for grp in ctx.groups]

    def body(self, ctx: Context, grp: ResidueGroup) -> Body:
        m = ctx.metrics
        plddt = residue_plddt(ctx, grp)
        look = ctx.look(plddt)
        name, number = _bubble_text(grp, look)
        width = max(text_width(name, _FONT, bold=True), text_width(number, _FONT))
        radius = max(15.0, math.hypot(width / 2.0, CAP_HEIGHT * _FONT + 1.1) + 2.2)
        infos = _merge_by_target(port_infos(ctx, grp, self.settings.labelled(ctx.options)))
        share = ctx.share_of(grp)
        if share is not None:
            return self._ring_body(ctx, grp, share, infos, look, plddt)
        set_lanes([i for i in infos if i.drawn and i.target.unit is not None])   # merging may have left fewer
        # A bubble is one point; two links from it to one residue of a chain ligand leave it side by side.
        ports = make_ports(ctx, infos, lambda info: ((0.0, info.lane), radius + 1.5, None, radius, None))
        n_drawn = sum(1 for i in infos if i.drawn)
        return Body(
            key=grp.uid, kind="disc", shapes_p0=np.zeros((1, 2)), shapes_p1=np.zeros((1, 2)),
            shapes_r=np.array([radius + 1.5]),
            ports=ports, ligand_clearance=0.95 * m.bond_length, priority=float(n_drawn),
            payload=_Bubble(grp, infos, radius, name, number, plddt, look, residue_occupancy(grp)))

    def _ring_body(self, ctx: Context, grp: ResidueGroup, share: "ResidueShare", infos: list[PortInfo], look: Look,
                   plddt: "float | None") -> Body:
        """A residue that is its ring alone: beside what most of the models have it on, without a line."""
        bl = ctx.metrics.bond_length
        radius = ring_reach(grp.label, share)
        for info in infos:
            info.drawn, info.label = False, None
        ports = make_ports(ctx, infos, lambda info: ((0.0, 0.0), radius, None, radius, None))
        for port, info in zip(ports, infos):
            port.ideal, port.weight = radius + 0.6 * bl, info.occupancy or 0.1
        return Body(
            key=grp.uid, kind="disc", shapes_p0=np.zeros((1, 2)), shapes_p1=np.zeros((1, 2)),
            shapes_r=np.array([radius]), ports=ports, ligand_clearance=0.6 * bl, priority=0.0,
            payload=_Bubble(grp, infos, radius, grp.label, "", plddt, look, share.share, share))

    # ------------------------------------------------------------------ drawing
    def draw(self, ctx: Context, bodies: list[Body], poses: dict, label_ts: dict | None = None) -> StyleDrawing:
        d = Drawn({"backdrop": [], "halos": [], "interactions": [], "residues": []})
        self.draw_bodies(ctx, bodies, poses, label_ts or {}, d)
        return finish_drawing(ctx, self.settings, d)

    def draw_bodies(self, ctx: Context, bodies: Sequence[Body], poses: dict, label_ts: dict, d: Drawn) -> None:
        m = ctx.metrics
        layers, kinds, notes = d.layers, d.kinds, d.notes
        links = self.settings.lines
        asked = sorted({level for level in (ctx.detail_of(b.payload.group) for b in bodies)
                        if level in ("backbone", "sidechain", "atoms")})
        if asked:
            # A level that was asked for and is not drawn is said, in the result: bubbles have no atoms.
            notes.append(f"the residues around {ctx.dep.called if ctx.dep.units else 'the ligand'} are drawn as "
                         f"bubbles, without the atoms that detail {' and '.join(repr(a) for a in asked)} asks "
                         f"for: the schematic style has none for them (the detailed style draws them)")
        for body in bodies:
            pose = poses.get(body.key)
            if pose is None:
                continue
            bub: _Bubble = body.payload
            grp = bub.group
            cls = grp.residue_class
            # What the bubble is filled with, and its stretch of the pocket's outline with it: the colour
            # of its class, or, where the figure colours by pLDDT and the residue has one, that of its band.
            fill = ctx.fill_of(bub.look, f"class_{cls}")
            ink = ctx.text_on(bub.look, "bubble_text")
            if fill == f"class_{cls}" and cls not in d.classes:
                d.classes.append(cls)
            d.beyond = d.beyond or grp.beyond
            d.placed.append(((pose.x, pose.y), bub.radius, fill))
            d.claims.extend(lining_claims(ctx, grp, bub.look, bub.ports, [(pose.x, pose.y)] * len(bub.ports)))
            P = np.array([[pose.x, pose.y]])
            A = np.array([0.0])
            M = np.array([False])
            a_all, b_all, _ = body.link_segments(P, A, M)
            label_t = label_ts.get(body.key, ())
            tx = sum(i.target.point[0] for i in bub.ports) / len(bub.ports)
            ty = sum(i.target.point[1] for i in bub.ports) / len(bub.ports)
            theta = math.atan2(ty - pose.y, tx - pose.x)
            look = bub.look
            if ctx.confidence is not None:
                d.levels.add(look.level)
            # A residue that only other models contact is set back like one outside the contact set.
            away = grp.absent or grp.faint
            muted = grp.beyond or away
            opacity = min(look.opacity * occupancy_factors(bub.occupancy, ctx.options.occupancy_style == "steps")[1], 0.55 if muted else 1.0)
            title = f"{grp.label}: " + ", ".join(t.label.lower() for t in grp.types)
            g = Group("residue", data={"residue": grp.ident, NUMBER: grp.numbered, "class": cls, "form": "bubble"},
                      title=title)
            if grp.parts > 1:
                g.data["drawing"] = f"{grp.part + 1} of {grp.parts}"
            if bub.plddt is not None and look.band is not None:
                g.data["plddt"] = f"{bub.plddt:.1f}"
                g.data["confidence"] = look.band
                g.title = f"{title}; residue pLDDT {bub.plddt:.0f}"
            dash, wide = _OUTLINE[look.level]
            mark: Group | None = None
            if grp.lost:
                d.lost = True
                g.data[other_protein.FROM] = other_protein.said_of(ctx.options)
                if not grp.faint:
                    mark = other_protein.marker_group(grp, ctx.options)
                    mark.add(other_protein.disc(pose.x, pose.y, bub.radius))
                    layers.setdefault("backdrop", []).append(mark)
            if bub.share is not None:
                d.rings = True
                g.title = ring_title(grp.label, bub.share)
                g.data["n-models"] = f"{bub.share.n_models}/{bub.share.n_total}"
                g.extend(ring_items(pose.x, pose.y, grp.label, bub.share, fill, ink))
                layers["residues"].append(g)
                continue
            if look.level == 0:
                g.add(_teardrop((pose.x, pose.y), bub.radius, theta,
                                Paint(fill=fill, stroke="bubble_line", width=0.9, opacity=opacity)))
            else:
                # Fill and outline are separate items: the fill fades, the outline changes kind.
                g.add(_teardrop((pose.x, pose.y), bub.radius, theta, Paint(fill=fill, opacity=opacity)))
                g.add(_teardrop((pose.x, pose.y), bub.radius, theta,
                                Paint(stroke="bubble_line", width=0.9 * wide, dash=dash,
                                      opacity=0.55 if muted else 1.0)))
            cap = CAP_HEIGHT * _FONT
            text_opacity = (0.55 if muted else 1.0) * (1.0 if look.level == 0 else 0.85)
            g.add(Text.plain(pose.x, pose.y - 1.1, bub.name, _FONT, ink, bold=look.level == 0 and not away,
                             opacity=text_opacity))
            g.add(Text.plain(pose.x, pose.y + cap + 1.1, bub.number, _FONT, ink, opacity=text_opacity))
            layers["residues"].append(g)

            for k, info in enumerate(bub.ports):
                d.occupancy = d.occupancy or info.occupancy is not None
                if not info.drawn:
                    continue
                if info.predicted:
                    d.predicted = True
                elif info.kind is T.HYDROPHOBIC:
                    d.whole = True
                elif info.kind not in kinds:
                    kinds.append(info.kind)
                if info.plddt is not None:
                    d.levels.add(info.look.level)
                d.absent = d.absent or info.absent
                first = info.interactions[0]
                style = predicted_style(links.styles[info.kind]) if info.predicted else links.styles[info.kind]
                arrow = None
                if info.kind in (T.HBOND, T.WATER_BRIDGE) and not info.predicted:
                    arrow, note = hbond_direction(first, ctx.dep)
                    if note:
                        notes.append(note)
                if info.kind is T.HBOND:
                    style, which = links.hbond(info, grp)
                    d.hbonds.add(which)
                a = (float(a_all[0, k, 0]), float(a_all[0, k, 1]))
                b = (float(b_all[0, k, 0]), float(b_all[0, k, 1]))
                lg = Group(f"interaction {info.kind.value}", data=interaction_data(ctx, grp, info),
                           title="; ".join(interaction_title(r, ctx.ligand, info.plddt, ctx.dep)
                                           for r in info.interactions))
                at = label_t[k] if k < len(label_t) else 0.5
                if info.predicted:
                    lg.extend(predicted_items(a, b, info, at, m, style, info.opacity, ctx.options.solvent_mark))
                else:
                    lg.extend(link_items(a, b, info.kind, m, label=info.label, arrow=arrow, tag=info.tag,
                                         opacity=info.opacity, style=style, weight=info.weight, label_at=at,
                                         glow=info.glow))
                    d.glows = d.glows or info.glow is not None
                if info.tag is not None:
                    d.tags.append(info.tag[1])
                d.has_distances = d.has_distances or is_distance(info.label)
                if info.target.kind == "ring":
                    key = (info.target.atoms, style.token)
                    if key not in d.dots_done:
                        d.dots_done.add(key)
                        lg.add(Circle(info.target.point[0], info.target.point[1], 2.5,
                                      Paint(fill=style.token, opacity=info.opacity)))
                if style.end_dots:
                    lg.add(Circle(a[0], a[1], 2.3, Paint(fill=style.token, opacity=info.opacity)))
                layers["interactions"].append(lg)
                if mark is not None:
                    mark.add(other_protein.stroke([a, b], m.interaction_width))
                if info.target.kind == "group" and ctx.options.charge_halos:
                    key = (info.target.atoms, style.token)
                    if key not in d.halo_done:
                        d.halo_done.add(key)
                        layers["halos"].extend(halo_items(group_skeleton(ctx.dep, info.target.atoms),
                                                          halo_radius(m), style.token))


def _merge_by_target(infos: list[PortInfo]) -> list[PortInfo]:
    """One link per (interaction type, ligand site).

    A bubble stands for the whole residue, so two contacts that differ only in
    which protein atom or ring makes them (tryptophan's two rings stacking on
    one ligand ring) would be the same line drawn twice.
    """
    merged: dict[tuple, PortInfo] = {}
    for info in infos:
        key = (info.kind, *info.target.key)
        seen = merged.get(key)
        if seen is None:
            merged[key] = info
        else:
            seen.absorb(info)
    return list(merged.values())


def _teardrop(centre: tuple[float, float], radius: float, theta: float, paint: Paint) -> Path:
    """A disc drawn out to a point in direction ``theta``."""
    cx, cy = centre
    phi = math.acos(1.0 / _TIP)
    tip = (cx + _TIP * radius * math.cos(theta), cy + _TIP * radius * math.sin(theta))
    p1 = (cx + radius * math.cos(theta + phi), cy + radius * math.sin(theta + phi))
    p2 = (cx + radius * math.cos(theta - phi), cy + radius * math.sin(theta - phi))
    reach = _TIP * radius
    return Path((("M", tip[0], tip[1]), ("L", p1[0], p1[1]), ("A", radius, radius, 0, 1, 1, p2[0], p2[1]), ("Z",)),
                paint, bbox=(cx - reach, cy - reach, cx + reach, cy + reach))
