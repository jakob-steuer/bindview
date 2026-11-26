"""The detailed style, after LigPlot.

Every ligand atom and bond is drawn. A residue that makes a *directional*
contact (H-bond, salt bridge, halogen bond, metal coordination, pi interaction)
has the part of it that makes the contact drawn explicitly, in a second colour,
with the interaction line running atom to atom (or to a ring centroid, or to
a charged group, which can be shaded). A residue that only makes hydrophobic contacts is
an "eyelash": its name inside an arc of lashes that faces the ligand atoms it
touches, which carry matching spokes. No line is drawn for a hydrophobic
contact, because a line would claim a direction the contact does not have.
Beside a ligand that is drawn as a chain of residues one is, a fine one without
a distance: there a contact is not read off by what lies beside what.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Sequence

import numpy as np

from ..confidence import PLAIN, Look, occupancy_factors
from ..depict import AtomLabel, DepAtom, DepBond, Depiction, LabelRun
from ..fragments import Fragment, build_fragment, conformations
from ..geometry import Point, free_direction, sub, unit
from ..interactions import InteractionType
from ..interactive import NUMBER
from ..layout import SPRINGS, Body, Port, Pose
from ..layout.body import LabelSpec
from ..residues import aromatic_rings, canonical_resname, is_nucleotide, written
from ..scene import Circle, Group, Line, Paint, Rect, Span, Text
from ..textmetrics import CAP_HEIGHT, text_width
from . import absence as other_protein
from .base import (CHAIN_CONTACT_LENGTH, STYLE_SETTINGS, Context, Drawn, PortInfo, ResidueGroup, Settings,
                   StyleDrawing, eyelash_items, eyelash_mark, finish_drawing, group_skeleton, halo_items, halo_radius,
                   hbond_direction, interaction_data, interaction_title, is_distance, label_size, lining_claims, link_items,
                   make_ports, marked_label, port_infos, predicted_items, predicted_style, residue_occupancy,
                   residue_plddt, share_label, spoke_items, spoke_reach, tag_slack)
from .ligand_draw import bond_items, coloured_bond_items, plddt_underlay
from .rings import ring_items, ring_reach, ring_title

if TYPE_CHECKING:
    from ..consensus import ResidueShare

__all__ = ["DetailedStyle"]

#: Between the name of a residue and the share that is written under it.
_SHARE_GAP = 3.2

#: The plate a residue's name stands on where the figure colours by pLDDT: how far it reaches beyond
#: the letters to either side and above and below, and how round its corners are.
_PLATE = (3.2, 2.6, 3.0)

T = InteractionType

# Atoms a charged-group halo is drawn along, per residue.
_HALO_ATOMS = {
    "ARG": ("CZ", "NE", "NH1", "NH2"), "ASP": ("CG", "OD1", "OD2"), "GLU": ("CD", "OE1", "OE2"),
    "LYS": ("NZ",), "HIS": ("CG", "ND1", "CE1", "NE2", "CD2"),
}


@dataclass
class _Payload:
    group: ResidueGroup
    form: str                              # fragment | eyelash | chip
    ports: list[PortInfo]
    label_text: str
    frag: Fragment | None = None
    local: dict[str, Point] = field(default_factory=dict)
    arc_radius: float = 0.0
    plddt: float | None = None             # the residue's own pLDDT
    look: Look = PLAIN
    occupancy: float | None = None         # that of the residue's most persistent contact
    #: Where a level of detail was asked for and the model has confidence: the look of every atom of
    #: the fragment, by name (its own, or its residue's where the model gives one value for a residue).
    own: "dict[str, Look] | None" = None
    detail: "str | None" = None            # the level that was asked for, if one was
    share: "ResidueShare | None" = None    # its name stands in a ring: the models that contact it
    below: "str | None" = None             # written under its name: the share of an ensemble, where no line has it

    @property
    def ident(self) -> str:
        return self.group.node_ident


def _bands(pay: _Payload, dep: Depiction) -> "list[str | None]":
    """The pLDDT band of every atom that is drawn of a residue: its own where it is drawn at its own, and the
    residue's otherwise."""
    return [(pay.own[a.name].band if pay.own is not None and a.name in pay.own else pay.look.band) for a in dep.atoms]


def _label_text(grp: ResidueGroup, look: Look = PLAIN) -> str:
    return marked_label(grp.label + ("†" if grp.beyond else ""), look)


#: Beside a chain ligand a residue that makes hydrophobic contacts only is seated before the others:
#: its line is the shortest there is, so it has the fewest places to go.
_CHAIN_EYELASH_FIRST = 30.0


class DetailedStyle:
    name = "detailed"

    def __init__(self, settings: "Settings | None" = None) -> None:
        self.settings = settings or STYLE_SETTINGS["detailed"]

    def ligand_obstacles(self, ctx: Context, groups: "Sequence[ResidueGroup] | None" = None
                         ) -> list[tuple[float, float, float]]:
        """Discs covering the contact spokes that will be drawn on ligand atoms."""
        atoms: set[int] = set()
        for grp in ctx.groups if groups is None else groups:
            for r in grp.interactions:
                if r.type is T.HYDROPHOBIC and r.ligand is not None and r.ligand.kind == "atom" \
                        and ctx.dep.home_of(r.ligand.atom) is None:      # on a chain residue it is a line
                    atoms.add(r.ligand.atom)
        return [(*ctx.dep.pos(a), spoke_reach(ctx.dep, a, ctx.metrics) + 0.5) for a in sorted(atoms)]

    # ------------------------------------------------------------------ bodies
    def bodies(self, ctx: Context) -> list[Body]:
        return [self.body(ctx, grp) for grp in ctx.groups]

    def body(self, ctx: Context, grp: ResidueGroup) -> Body:
        return self._eyelash_body(ctx, grp) if grp.only_hydrophobic else self._fragment_body(ctx, grp)

    def _eyelash_body(self, ctx: Context, grp: ResidueGroup) -> Body:
        m = ctx.metrics
        plddt = residue_plddt(ctx, grp)
        look = ctx.look(plddt)
        text = _label_text(grp, look)
        share = ctx.share_of(grp)
        if share is not None:
            text += f" {share.n_models}/{share.n_total}"       # an eyelash has no room for a ring
        w = text_width(text, m.residue_font, bold=True)
        h = CAP_HEIGHT * m.residue_font
        occupancy = residue_occupancy(grp)
        below = share_label(occupancy) if ctx.shares_written and share is None else None
        if below:
            # It has no line to write its share on: the share stands under its name.
            w = max(w, text_width(below, m.distance_font))
            h += _SHARE_GAP + CAP_HEIGHT * m.distance_font
        arc = math.hypot(w / 2.0, h / 2.0) + 4.5
        radius = arc + m.lash_length
        infos = port_infos(ctx, grp, self.settings.labelled(ctx.options))
        clearance = 4.5
        ports = []
        for info in infos:
            if info.drawn:
                # A residue of a chain ligand: the contact is a line from the lashes to the residue's box.
                visible, least = (v * m.bond_length for v in CHAIN_CONTACT_LENGTH)
                # Only a tag asks for more room here: a line without one keeps these two lengths.
                written = label_size(info.shown, m.distance_font) if (info.tag and info.shown) else None
                if written is not None:
                    # Its predicted aligned error is written on the line: it needs the room.
                    visible, least = max(visible, written[0] + 24.0), max(least, written[0] + 13.0)
                ends = radius + info.target.extent + tag_slack(ctx.dep, info)
                ports.append(Port((0.0, 0.0), info.target.point, ideal=ends + visible, min_length=ends + least,
                                  weight=SPRINGS["eyelash"] / len(infos), drawn=True, port_radius=radius + 1.0,
                                  target_radius=info.target.radius, target_caps=info.target.caps,
                                  label_size=written))
                continue
            reach = max(spoke_reach(ctx.dep, a, m) for a in info.target.atoms)
            ports.append(Port((0.0, 0.0), info.target.point, ideal=radius + clearance + reach + 3.0,
                              weight=SPRINGS["eyelash"] / len(infos), drawn=False))
        return Body(key=grp.uid, kind="disc", shapes_p0=np.zeros((1, 2)), shapes_p1=np.zeros((1, 2)),
                    shapes_r=np.array([radius]),
                    ports=ports, ligand_clearance=clearance,
                    priority=_CHAIN_EYELASH_FIRST if any(i.drawn for i in infos) else 0.0,
                    payload=_Payload(grp, "eyelash", infos, text, arc_radius=arc, plddt=plddt, look=look,
                                     occupancy=occupancy, below=below))

    def _fragment_body(self, ctx: Context, grp: ResidueGroup) -> Body:
        m = ctx.metrics
        infos = port_infos(ctx, grp, self.settings.labelled(ctx.options))
        involved: set[str] = set()
        for info in infos:
            if info.kind is T.HYDROPHOBIC and info.drawn:
                continue        # beside a chain ligand: a line from the residue, not from one of its atoms
            if info.site_kind in ("atom", "ring", "group"):
                involved |= set(info.site_atoms)
        # The level that was asked for adds to the part that makes the contact; "box" draws no atoms at all.
        level = "box" if self.settings.form_of(grp) == "names" else ctx.detail_of(grp)
        like = None
        if is_nucleotide(grp.resname):
            # How this figure's contacts name the atoms of a nucleotide: every nucleotide is named so.
            like = [n for g in ctx.groups for r in g.interactions
                    for n in (r.record.protein_atom or "", *r.record.protein_atoms) if n]
        frag = (build_fragment(grp.resname, involved, m.bond_length * m.protein_bond_scale, level, like)
                if involved and level != "box" else None)
        if frag is None or any(i.site_kind == "residue" for i in infos):
            return self._chip_body(ctx, grp, infos)

        shape = self._conformation(ctx, frag, infos)
        cx = sum(q[0] for q in shape.values()) / len(shape)
        cy = sum(q[1] for q in shape.values()) / len(shape)
        local = {a: (x - cx, y - cy) for a, (x, y) in shape.items()}
        upright_pts: list[Point] = []
        upright_wh: list[tuple[float, float]] = []
        upright_of: dict[str, int] = {}
        for a in frag.atoms:
            if frag.elements[a] != "C":
                upright_of[a] = len(upright_pts)
                upright_pts.append(local[a])
                upright_wh.append((text_width(written(a, ctx.options.atom_labels), m.protein_atom_font) + 1.0,
                                   CAP_HEIGHT * m.protein_atom_font + 1.0))
        p0: list[Point] = []
        p1: list[Point] = []
        rr: list[float] = []
        for a, b, order in frag.bonds:
            p0.append(local[a])
            p1.append(local[b])
            rr.append(m.protein_bond_width / 2.0 + 1.0 + (m.double_offset * 0.5 if order > 1 else 0.0))

        hr = halo_radius(m) * m.protein_bond_scale
        spoked: set[str] = set()

        def locate(info: PortInfo):
            if info.kind is T.HYDROPHOBIC and info.drawn:
                # It leaves where the residue's name stands: at the alpha carbon of what is drawn of it.
                return local[frag.root], 2.6, None, 3.0, None
            if info.site_kind == "atom" or len(info.site_atoms) == 1:
                name = info.site_atoms[0]
                if not info.drawn:
                    spoked.add(name)
                return local[name], 2.6, upright_of.get(name), (9.0 if name in upright_of else 3.0), None
            pts = [local[n] for n in info.site_atoms]
            centre = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
            spread = max(math.hypot(q[0] - centre[0], q[1] - centre[1]) for q in pts)
            if info.site_kind == "ring":
                return centre, 3.6, None, spread, None
            # A charged group: the link stops at its halo or at the labels of its atoms.
            names = [n for n in _HALO_ATOMS.get(canonical_resname(grp.resname), info.site_atoms) if n in local]
            chosen = set(names)
            segs = [(local[a], local[b]) for a, b, _ in frag.bonds if a in chosen and b in chosen] \
                or [(local[n], local[n]) for n in names]
            caps = (np.array([q[0] for q in segs], dtype=float), np.array([q[1] for q in segs], dtype=float),
                    np.full(len(segs), hr + 1.0))
            ups = tuple(upright_of[n] for n in names if n in upright_of)
            return centre, 6.0, None, 0.62 * (spread + hr), (caps, ups)

        ports = make_ports(ctx, infos, locate)
        if level is not None:
            # At a level a fragment has atoms that none of its lines ends on, and its lines come from
            # the atoms beside a column, from more than one side. None of them runs through the name
            # of an atom it does not end on, and one that ends on an atom does not run through the
            # residue's name: forbidden in the layout and counted in the audit, as a line across the
            # atoms beside a box is. (A line that leaves where the name stands ends on the name.)
            for port, info in zip(ports, infos):
                if info.drawn:
                    whole = info.kind is T.HYDROPHOBIC
                    ends_on = set() if whole else set(info.site_atoms)
                    port.clear_of = tuple(sorted(j for name, j in upright_of.items() if name not in ends_on))
                    port.clear_of_name = not whole
        for name in sorted(spoked):
            # Room for the contact spokes drawn on this atom.
            p0.append(local[name])
            p1.append(local[name])
            rr.append(9.0)
        root = local[frag.root]
        bonded = [local[b if a == frag.root else a] for a, b, _ in frag.bonds if frag.root in (a, b)]
        # The name goes into the widest gap between the bonds at the alpha carbon.
        away = free_direction(root, bonded) if bonded else (-1.0, 0.0)
        plddt = residue_plddt(ctx, grp)
        look = ctx.look(plddt)
        text = _label_text(grp, look)
        lw, lh, gap = text_width(text, m.residue_font, bold=True) + 2.0, CAP_HEIGHT * m.residue_font + 3.0, 4.0
        share = ctx.share_of(grp)
        if share is not None:
            # Its name stands in its ring, which touches what is drawn of it.
            lw = lh = 2.0 * ring_reach(grp.label, share)
            gap = 1.5
        axis = unit(sub(root, ports[0].local), away)
        if math.hypot(root[0] - ports[0].local[0], root[1] - ports[0].local[1]) < 1e-6:
            axis = away
        n_drawn = sum(1 for i in infos if i.drawn)
        own = None
        conf = ctx.confidence
        if conf is not None and ctx.options.confidence and (
                level is not None or (ctx.options.plddt_by_atom and conf.per_atom)):
            # Every atom at its own pLDDT: at a level that was asked for, and without one where the model
            # gives the atoms of a residue values of their own (``plddt_by_atom``).
            own = {a: ctx.look(conf.atom(grp.key, a)) for a in frag.atoms}
        return Body(
            key=grp.uid, kind="fragment", shapes_p0=np.array(p0, dtype=float).reshape(-1, 2),
            shapes_p1=np.array(p1, dtype=float).reshape(-1, 2), shapes_r=np.array(rr, dtype=float), ports=ports,
            label=LabelSpec(anchor=root, direction=away, width=lw, height=lh, gap=gap),
            rotatable=True, mirrorable=True, ligand_clearance=0.3 * m.bond_length, axis=axis,
            priority=10.0 + n_drawn + 0.01 * len(frag.atoms),
            upright_pts=np.array(upright_pts, dtype=float).reshape(-1, 2),
            upright_wh=np.array(upright_wh, dtype=float).reshape(-1, 2),
            payload=_Payload(grp, "fragment", infos, text, frag=frag, local=local, plddt=plddt, look=look,
                             # (With a ring the share is the ring's, and the atoms are drawn in full.)
                             occupancy=residue_occupancy(grp) if share is None else None, own=own, detail=level,
                             share=share))

    @staticmethod
    def _conformation(ctx: Context, frag: Fragment, infos: list[PortInfo]) -> dict[str, Point]:
        """Pick the flat layout of a fragment whose contact atoms are spaced like their targets.

        With one contact point the template layout is used. With several, the
        distance between two contact atoms should resemble the distance between
        the ligand sites they go to, so that both links can be short; a residue
        reaching the ligand with backbone and side-chain tip is drawn kinked
        rather than as a rod with one end stranded.
        """
        pts: list[tuple[tuple[str, ...], Point]] = []
        for info in infos:
            if info.kind is T.HYDROPHOBIC and info.drawn:
                continue                     # beside a chain ligand it leaves from the residue, not from an atom
            if info.site_kind in ("atom", "ring", "group") and info.site_atoms:
                pts.append((info.site_atoms, info.target.point))
        sites = {p[0] for p in pts}
        if len(sites) < 2:
            return dict(frag.xy)

        def centre(xy: dict[str, Point], names: tuple[str, ...]) -> Point:
            return (sum(xy[n][0] for n in names) / len(names), sum(xy[n][1] for n in names) / len(names))

        slack = 1.2 * ctx.metrics.bond_length
        best, best_score = dict(frag.xy), None
        for n_flips, xy in enumerate(conformations(frag)):
            score = 0.0
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    if pts[i][0] == pts[j][0]:
                        continue
                    a, b = centre(xy, pts[i][0]), centre(xy, pts[j][0])
                    have = math.hypot(a[0] - b[0], a[1] - b[1])
                    want = math.hypot(pts[i][1][0] - pts[j][1][0], pts[i][1][1] - pts[j][1][1])
                    # Being a little further apart than the targets is free: links fan out.
                    score += max(0.0, abs(have - want) - slack) ** 2
            score = round(score, 6)
            if best_score is None or score < best_score - 1e-9:
                best, best_score = xy, score
        return best

    def _chip_body(self, ctx: Context, grp: ResidueGroup, infos: list[PortInfo]) -> Body:
        """A residue we have no template for: its name, with links ending on it."""
        m = ctx.metrics
        plddt = residue_plddt(ctx, grp)
        look = ctx.look(plddt)
        text = _label_text(grp, look)
        wh = (text_width(text, m.residue_font, bold=True) + 4.0, CAP_HEIGHT * m.residue_font + 5.0)
        share = ctx.share_of(grp)
        if share is not None:
            wh = (2.0 * ring_reach(grp.label, share),) * 2
        ports = make_ports(ctx, infos, lambda info: ((0.0, 0.0), 0.0, 0, wh[0] / 2.0, None))
        return Body(key=grp.uid, kind="disc", shapes_p0=np.zeros((0, 2)), shapes_p1=np.zeros((0, 2)),
                    shapes_r=np.zeros(0), ports=ports, ligand_clearance=6.0, priority=5.0,
                    upright_pts=np.array([(0.0, 0.0)]), upright_wh=np.array([wh]),
                    payload=_Payload(grp, "chip", infos, text, plddt=plddt, look=look,
                                     occupancy=residue_occupancy(grp), detail=ctx.detail_of(grp), share=share))

    # ------------------------------------------------------------------ drawing
    def draw(self, ctx: Context, bodies: list[Body], poses: dict, label_ts: dict | None = None) -> StyleDrawing:
        d = Drawn({"halos": [], "interactions": [], "contacts": [], "residues": []})
        self.draw_bodies(ctx, bodies, poses, label_ts or {}, d)
        return finish_drawing(ctx, self.settings, d)

    def draw_bodies(self, ctx: Context, bodies: Sequence[Body], poses: dict, label_ts: dict, d: Drawn) -> None:
        m = ctx.metrics
        layers, kinds, notes = d.layers, d.kinds, d.notes
        links = self.settings.lines

        for body in bodies:
            pose = poses.get(body.key)
            if pose is None:
                continue
            pay: _Payload = body.payload
            grp = pay.group
            d.beyond = d.beyond or grp.beyond
            P = np.array([[pose.x, pose.y]])
            A = np.array([pose.angle])
            M = np.array([pose.mirror])
            a_all, b_all, _ = body.link_segments(P, A, M)
            ports_w = body.ports_world(P, A, M)[0]
            label_t = label_ts.get(body.key, ())
            res_group = Group("residue", data={"residue": grp.ident, NUMBER: grp.numbered, "class": grp.residue_class,
                                                "form": pay.form})
            if grp.parts > 1:
                res_group.data["drawing"] = f"{grp.part + 1} of {grp.parts}"
            # A residue that only other models contact is set back like one outside the contact set.
            away = grp.absent or grp.faint
            muted = grp.beyond or away
            label_fill = "beyond" if muted else "protein_label"
            look = pay.look
            occ_width, occ_opacity = occupancy_factors(pay.occupancy, ctx.options.occupancy_style == "steps")
            if ctx.confidence is not None:
                d.levels.add(look.level)
            if pay.plddt is not None and look.band is not None:
                res_group.data["plddt"] = f"{pay.plddt:.1f}"
                res_group.data["confidence"] = look.band
                res_group.title = f"{grp.label}: residue pLDDT {pay.plddt:.0f}"

            if pay.detail is not None:
                res_group.data["detail"] = pay.detail
            mark: Group | None = None
            if grp.lost:
                d.lost = True
                res_group.data[other_protein.FROM] = other_protein.said_of(ctx.options)
                if not grp.faint:
                    mark = other_protein.marker_group(grp, ctx.options)
                    layers.setdefault("backdrop", []).append(mark)
            frag_dep: Depiction | None = None
            frag_index: dict[str, int] = {}
            if pay.form == "fragment":
                assert pay.frag is not None
                frag_dep, frag_index = self._fragment_depiction(ctx, pay, pose)
                d.atoms_at_level = d.atoms_at_level or pay.detail is not None
                d.atoms_by_own = d.atoms_by_own or (pay.own is not None and pay.detail is None)
                if pay.own is None:
                    # The fragment fades as one object, so bonds meeting at an atom do not add up.
                    drawn = Group("fragment", opacity=min(look.opacity * occ_opacity, 0.55 if away else 1.0))
                    self._draw_fragment(ctx, pay, frag_dep, drawn)
                    res_group.add(drawn)
                else:
                    # Every atom at its own confidence: each level fades as one object.
                    d.levels.update(lk.level for lk in pay.own.values())
                    res_group.extend(self._fragment_by_atom(ctx, pay, frag_dep, occ_width,
                                                            0.55 * occ_opacity if grp.faint else occ_opacity))
                centre = body.label_centre(P, A, M)
                assert centre is not None
                lx, ly = float(centre[0, 0]), float(centre[0, 1])
                if ctx.by_colour and not ctx.by_bond:
                    # The atoms that are drawn of the residue stand on colour, like those of the ligand:
                    # each at its own band where it is drawn at its own pLDDT, at the residue's otherwise.
                    under = plddt_underlay(frag_dep, _bands(pay, frag_dep))
                    if under is not None:
                        layers.setdefault("backdrop", []).append(under)
                if mark is not None:
                    mark.extend(other_protein.skeleton(frag_dep, other_protein.UNDER * frag_dep.metrics.bond_length))
            else:
                lx, ly = pose.x, pose.y
            # (A residue whose bonds have the colour says it with them.)
            plate = "" if ctx.by_bond and pay.form == "fragment" else ctx.fill_of(look, "")
            if plate:
                # ... and its name stands on the colour of the residue's own band.
                half_w = text_width(pay.label_text, m.residue_font, bold=look.level == 0) / 2.0 + _PLATE[0]
                half_h = CAP_HEIGHT * m.residue_font / 2.0 + _PLATE[1]
                res_group.add(Rect(lx - half_w, ly - half_h, 2.0 * half_w, 2.0 * half_h, Paint(fill=plate),
                                   rx=_PLATE[2]))
                label_fill = ctx.text_on(look, label_fill)
            name_box = None
            if pay.form == "fragment" and pay.detail is not None:
                # Where the name is printed, for the lines that leave from where it stands (below).
                half_w = text_width(pay.label_text, m.residue_font, bold=look.level == 0) / 2.0
                half_h = CAP_HEIGHT * m.residue_font / 2.0
                name_box = (lx - half_w, ly - half_h, lx + half_w, ly + half_h)
            if pay.form == "eyelash":
                targets = [info.target.point for info in pay.ports]
                res_group.extend(eyelash_items((pose.x, pose.y), pay.arc_radius, targets, m,
                                               opacity=min(look.opacity * occ_opacity,
                                                           0.55 if muted else 1.0),
                                               weight=look.width * occ_width))
                if mark is not None:
                    mark.extend(eyelash_mark((pose.x, pose.y), pay.arc_radius, targets, m))
            if mark is not None:
                mark.add(other_protein.disc(lx, ly, ring_reach(grp.label, pay.share)) if pay.share is not None else
                         other_protein.plate(lx, ly, text_width(pay.label_text, m.residue_font, bold=look.level == 0),
                                             CAP_HEIGHT * m.residue_font))
            if pay.share is not None:
                self._ring(ctx, body, pose, (lx, ly), res_group, d)
            else:
                if pay.below:
                    res_group.add(Text.plain(lx, ly + (CAP_HEIGHT * m.residue_font + _SHARE_GAP) / 2.0
                                             + CAP_HEIGHT * m.distance_font / 2.0, pay.below, m.distance_font, "hydrophobic",
                                             opacity=0.55 if muted else max(0.7, occ_opacity)))
                    ly -= (_SHARE_GAP + CAP_HEIGHT * m.distance_font) / 2.0
                # The name stays legible at any confidence; its weight is the greyscale channel.
                res_group.add(Text.plain(lx, ly + CAP_HEIGHT * m.residue_font / 2.0, pay.label_text, m.residue_font,
                                         label_fill, bold=look.level == 0 and not away, italic=muted,
                                         opacity=1.0 if look.level == 0 else max(0.72, look.opacity + 0.2)))
            layers["residues"].append(res_group)
            d.placed.append(((pose.x, pose.y), float(body.proper_radius),
                             ctx.fill_of(look, f"class_{grp.residue_class}")))
            d.claims.extend(lining_claims(ctx, grp, look, pay.ports,
                                          [(float(q[0]), float(q[1])) for q in ports_w]))

            for k, info in enumerate(pay.ports):
                if info.predicted:
                    d.predicted = True
                elif info.kind is T.HYDROPHOBIC:
                    d.lashes = True
                elif info.kind not in kinds:
                    kinds.append(info.kind)
                d.absent = d.absent or info.absent
                opacity = info.opacity
                weight = info.weight
                if info.plddt is not None:
                    d.levels.add(info.look.level)
                d.occupancy = d.occupancy or info.occupancy is not None
                first = info.interactions[0]
                if not info.drawn:
                    # Hydrophobic: spokes on the ligand atom, and on the fragment atom if one is drawn.
                    port_pt = (float(ports_w[k][0]), float(ports_w[k][1]))
                    g = Group("contact hydrophobic", data=interaction_data(ctx, grp, info),
                              title=interaction_title(first, ctx.ligand, info.plddt, ctx.dep))
                    for atom in info.target.atoms:
                        g.extend(spoke_items(ctx.dep, atom, port_pt, m, opacity, weight))
                    if frag_dep is not None and info.site_kind == "atom" and info.site_atoms[0] in frag_index:
                        fm = frag_dep.metrics.with_(spoke_inner=4.5, spoke_outer=8.5)
                        g.extend(spoke_items(frag_dep, frag_index[info.site_atoms[0]], info.target.point, fm,
                                             opacity, weight))
                    if g.items:
                        layers["contacts"].append(g)
                    if mark is not None:
                        mark.extend(other_protein.under(g.items, m.interaction_width))
                    continue
                a = (float(a_all[0, k, 0]), float(a_all[0, k, 1]))
                b = (float(b_all[0, k, 0]), float(b_all[0, k, 1]))
                arrow = None
                if info.kind in (T.HBOND, T.WATER_BRIDGE) and not info.predicted:
                    arrow, note = hbond_direction(first, ctx.dep)
                    if note:
                        notes.append(note)
                g = Group(f"interaction {info.kind.value}", data=interaction_data(ctx, grp, info),
                          title="; ".join(interaction_title(r, ctx.ligand, info.plddt, ctx.dep)
                                          for r in info.interactions))
                at = label_t[k] if k < len(label_t) else 0.5
                if name_box is not None and info.kind is T.HYDROPHOBIC:
                    # A contact with the residue as a whole leaves where the residue's name stands. At a
                    # level of detail a fragment stands where its atom's partner is, which is beside the
                    # column and not on it, and such a line can come from the side the name is on: it
                    # then ends on the name, as the lines of a residue that is a name alone do.
                    a, at = _from_the_name(a, b, at, name_box)
                style = links.styles[info.kind]
                if info.predicted:
                    # A guess is drawn as a different kind of line, with its label set off as a node.
                    g.extend(predicted_items(a, b, info, at, m, predicted_style(style), opacity,
                                             ctx.options.solvent_mark))
                else:
                    if info.kind is T.HBOND:
                        style, which = links.hbond(info, grp)
                        d.hbonds.add(which)
                    g.extend(link_items(a, b, info.kind, m, label=info.label, arrow=arrow, opacity=opacity,
                                        label_at=at, weight=weight, tag=info.tag, style=style, glow=info.glow))
                    d.glows = d.glows or info.glow is not None
                    if info.tag is not None:
                        d.tags.append(info.tag[1])
                d.has_distances = d.has_distances or is_distance(info.label)
                if info.target.kind == "ring":
                    key = (info.target.atoms, style.token)
                    if key not in d.dots_done:
                        d.dots_done.add(key)
                        tp = info.target.point
                        g.add(Circle(tp[0], tp[1], 2.5, _fill(style.token, opacity)))
                if info.site_kind == "ring" and pay.form == "fragment":
                    pp = ports_w[k]
                    g.add(Circle(float(pp[0]), float(pp[1]), 2.3, _fill(style.token, opacity)))
                layers["interactions"].append(g)
                if mark is not None:
                    mark.add(other_protein.stroke([a, b], m.interaction_width))
                if info.target.kind == "group" and ctx.options.charge_halos:
                    key = (info.target.atoms, style.token)
                    if key not in d.halo_done:
                        d.halo_done.add(key)
                        layers["halos"].extend(halo_items(group_skeleton(ctx.dep, info.target.atoms),
                                                          halo_radius(m), style.token))
                if info.site_kind == "group" and frag_dep is not None and ctx.options.charge_halos:
                    names = [n for n in _HALO_ATOMS.get(canonical_resname(grp.resname), info.site_atoms)
                             if n in frag_index]
                    res_key = (grp.uid, style.token)
                    if names and res_key not in d.halo_done:
                        d.halo_done.add(res_key)
                        layers["halos"].extend(halo_items(
                            group_skeleton(frag_dep, [frag_index[n] for n in names]),
                            halo_radius(m) * m.protein_bond_scale, style.token))

    @staticmethod
    def _ring(ctx: Context, body: Body, pose: Pose, at: Point, group: Group, d: Drawn) -> None:
        """The name of a residue in its ring, and a bond to the ring from the atom that the name belongs to."""
        grp, share = body.payload.group, body.payload.share
        d.rings = True
        if grp.residue_class not in d.classes:
            d.classes.append(grp.residue_class)
        group.title = ring_title(grp.label, share)
        group.data["n-models"] = f"{share.n_models}/{share.n_total}"
        if body.label is not None:
            x, y = pose.apply(body.label.anchor)
            far, inside = math.hypot(at[0] - x, at[1] - y), ring_reach(grp.label, share) - 1.0
            if far > inside:
                group.add(Line(x, y, at[0] - (at[0] - x) / far * inside, at[1] - (at[1] - y) / far * inside,
                               Paint(stroke="protein_bond", width=ctx.metrics.protein_bond_width)))
        group.extend(ring_items(at[0], at[1], grp.label, share, f"class_{grp.residue_class}"))

    def _fragment_depiction(self, ctx: Context, pay: _Payload, pose: Pose) -> tuple[Depiction, dict[str, int]]:
        """The fragment on the page, as a small depiction the bond drawer understands."""
        frag = pay.frag
        assert frag is not None
        m = ctx.metrics
        fm = m.with_(bond_width=m.protein_bond_width * pay.look.width * occupancy_factors(pay.occupancy, ctx.options.occupancy_style == "steps")[0],
                     double_offset=m.double_offset * 0.8,
                     label_gap=1.5,
                     atom_font=m.protein_atom_font, bond_length=m.bond_length * m.protein_bond_scale)
        index = {a: i for i, a in enumerate(frag.atoms)}
        ring_atoms = {a for ring in aromatic_rings(frag.resname) for a in ring}
        atoms = []
        for a in frag.atoms:
            x, y = pose.apply(pay.local[a])
            atom = DepAtom(idx=index[a], x=x, y=y, element=frag.elements[a], name=a, aromatic=a in ring_atoms)
            if frag.elements[a] != "C":
                word = written(a, ctx.options.atom_labels)
                w = text_width(word, fm.atom_font)
                cap = CAP_HEIGHT * fm.atom_font
                box = (x - w / 2.0, y - cap / 2.0, x + w / 2.0, y + cap / 2.0)
                atom.label = AtomLabel((LabelRun(x, y + cap / 2.0, "middle", (Span(word),), box),), box)
            atoms.append(atom)
        bonds = [DepBond(index[a], index[b], order) for a, b, order in frag.bonds]
        rings = [tuple(index[a] for a in ring) for ring in frag.rings]
        return Depiction(atoms=atoms, bonds=bonds, rings=rings, n_heavy=len(atoms), metrics=fm), index

    def _fragment_by_atom(self, ctx: Context, pay: _Payload, dep: Depiction, occ_width: float,
                          occ_opacity: float) -> list[Group]:
        """The fragment with every atom at its own look: a bond at the weaker of its two ends, and one
        group for each level of confidence, the weakest underneath."""
        assert pay.frag is not None and pay.own is not None
        m = ctx.metrics
        looks = [pay.own[a] for a in pay.frag.atoms]
        out: list[Group] = []
        for level in sorted({lk.level for lk in looks}, reverse=True):
            seen = next(lk for lk in looks if lk.level == level)
            fine = replace(dep, metrics=dep.metrics.with_(bond_width=m.protein_bond_width * seen.width * occ_width))
            part = Group("fragment", opacity=seen.opacity * occ_opacity)
            if seen.band is not None:
                part.data["confidence"] = seen.band
            for bond in fine.bonds:
                if max(looks[bond.a].level, looks[bond.b].level) == level:
                    part.extend(coloured_bond_items(fine, bond, [lk.band for lk in looks], "protein_bond") if ctx.by_bond
                                else bond_items(fine, bond, stroke="protein_bond"))
            for atom in fine.atoms:
                if atom.label is None or looks[atom.idx].level != level:
                    continue
                token = ctx.palette.element_token(atom.element, on_protein=True)
                for run in atom.label.runs:
                    part.add(Text(run.x, run.y, run.spans, fine.metrics.atom_font, token, run.anchor))
            if part.items:
                out.append(part)
        return out

    def _draw_fragment(self, ctx: Context, pay: _Payload, dep: Depiction, out: Group) -> None:
        for bond in dep.bonds:
            out.extend(coloured_bond_items(dep, bond, _bands(pay, dep), "protein_bond") if ctx.by_bond
                       else bond_items(dep, bond, stroke="protein_bond"))
        for atom in dep.atoms:
            if atom.label is None:
                continue
            token = ctx.palette.element_token(atom.element, on_protein=True)
            for run in atom.label.runs:
                out.add(Text(run.x, run.y, run.spans, dep.metrics.atom_font, token, run.anchor))


#: How far from the name of its residue a line stops that ends on the name.
_NAME_GAP = 2.0


def _from_the_name(a: Point, b: Point, at: float, name: "tuple[float, float, float, float]") -> tuple[Point, float]:
    """A line from ``a`` (at its residue) to ``b`` that runs through the residue's name (the box of
    its text): the point it starts from instead, :data:`_NAME_GAP` beyond the name on the way to
    ``b``, and where its label is along the shortened line (``at`` is its place along the whole one;
    the label stays where it was, or goes to the new start if it stood on the part that is gone).
    Unchanged if the line does not meet the name, or ends in it."""
    dx, dy = b[0] - a[0], b[1] - a[1]

    def within(box: "tuple[float, float, float, float]") -> "tuple[float, float] | None":
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, a[0] - box[0]), (dx, box[2] - a[0]), (-dy, a[1] - box[1]), (dy, box[3] - a[1])):
            if abs(p) < 1e-9:
                if q < 0.0:
                    return None
                continue
            r = q / p
            if p < 0.0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
        return (t0, t1) if t0 < t1 else None

    if within(name) is None:
        return a, at
    around = within((name[0] - _NAME_GAP, name[1] - _NAME_GAP, name[2] + _NAME_GAP, name[3] + _NAME_GAP))
    assert around is not None
    leaves = around[1]
    if leaves >= 1.0:
        return a, at
    return (a[0] + dx * leaves, a[1] + dy * leaves), max(0.0, (at - leaves) / (1.0 - leaves))


def _fill(token: str, opacity: float):
    return Paint(fill=token, opacity=opacity)
