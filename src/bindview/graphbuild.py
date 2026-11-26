"""Between the layout engine and the renderers: building a :class:`~bindview.graph.Graph`
from a laid-out diagram, and reading poses back out of one.

The engine places bodies; a renderer draws them. In between sits the graph:
:func:`diagram_graph` turns the engine's result into nodes and contacts, and
:func:`poses_from` gives a renderer the poses the graph holds -- whether the
graph was made a moment ago, loaded from JSON, or edited by someone dragging a
node. A renderer therefore never sees the engine's own result.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .depict import Depiction, atom_keys, depict_ligand
from .errors import GraphError
from .graph import Contact, Graph, Node, RigidGroup
from .layout import SPRINGS, Body, LayoutParams, LayoutResult, Pose
from .layout.body import Port
from .ligand import Ligand
from .styles.base import Context, PortInfo, Target
from .theme import Metrics

__all__ = ["LigandPart", "diagram_graph", "residue_contacts", "poses_from", "depiction_from", "engine_springs",
           "unique_ids"]


def engine_springs(params: LayoutParams) -> dict[str, float]:
    """The layout engine's spring constants, as the graph carries them.

    ``link`` / ``sightline`` / ``eyelash`` are the weights by class of tie
    (:data:`bindview.layout.SPRINGS`); ``length`` and ``short`` are the engine's
    ``w_length`` and ``w_short``; ``unit`` is the length that makes them
    dimensionless (one bond length). A tie of weight ``w`` and rest length
    ``ideal`` costs ``length * w * ((d - ideal) / unit) ** 2`` at length ``d``,
    times ``short`` while ``d < ideal``.
    """
    return {**SPRINGS, "length": params.w_length, "short": params.w_short, "unit": params.unit}


class LigandPart:
    """The nodes of one drawn ligand: its atoms, and the ring centroids and groups contacts land on."""

    def __init__(self, dep: Depiction, *, prefix: str = "L", group: str = "ligand", label: str = "",
                 plddt: "Mapping[str, float] | None" = None, exposure: Any = None) -> None:
        self.dep = dep
        self.prefix = prefix
        self.group_id = group
        self.label = label
        self._atoms: list[Node] = []
        self._extra: dict[str, Node] = {}
        self._units: list[Node] = []
        self._outside: list[Node] = []               # of a chain ligand: the nodes of its atoms outside the chain
        self._pieces: dict[str, list[str]] = {}
        if dep.units:
            self._chain(plddt, exposure)
            return
        for atom, key in zip(dep.atoms, atom_keys(dep)):
            node = Node(id=f"{prefix}:{key}", kind="ligand_atom", label=atom.name or atom.element, x=atom.x,
                        y=atom.y, group=group, attrs={"element": atom.element})
            if atom.name is None:
                node.attrs["drawing_only"] = True          # a hydrogen the depiction adds at a stereocentre
            elif plddt is not None and atom.name in plddt:
                node.confidence = float(plddt[atom.name])
            if exposure is not None and atom.name is not None and atom.idx < len(exposure):
                node.attrs["exposure"] = round(float(exposure[atom.idx]), 4)
            self._atoms.append(node)
        names = [a.name for a in dep.atoms]
        self.coordination = [[names[b.a], names[b.b]] for b in dep.bonds
                             if b.kind == "coordination" and names[b.a] and names[b.b]]

    def _chain(self, plddt: "Mapping[str, float] | None", exposure: Any) -> None:
        """The nodes of a ligand drawn as a chain: one per residue, and one per atom outside the chain.

        A residue is a node of its own and belongs to no rigid group unless
        something outside the chain is bonded to it: then the two form one, so
        that a lipid follows the lysine it sits on and nothing else does.
        """
        dep, prefix = self.dep, self.prefix
        holder: dict[int, int] = {}                  # atom outside the chain -> index of the unit that holds its piece
        outside = [a.idx for a in dep.atoms if dep.unit_of(a.idx) is None]
        nbrs: dict[int, list[int]] = {i: [] for i in outside}
        touch: dict[int, int] = {}
        for b in dep.bonds:
            if b.kind != "covalent":
                continue
            ua, ub = dep.unit_of(b.a), dep.unit_of(b.b)
            if ua is None and ub is None:
                nbrs[b.a].append(b.b)
                nbrs[b.b].append(b.a)
            elif ua is None and ub is not None:
                touch.setdefault(b.a, ub.index)
            elif ub is None and ua is not None:
                touch.setdefault(b.b, ua.index)
        for start in sorted(touch):                  # flood each piece from where it is held
            stack = [start]
            while stack:
                i = stack.pop()
                if i in holder:
                    continue
                holder[i] = touch[start]
                stack.extend(nbrs[i])
        group_of = {u: f"{self.group_id}:at{dep.units[u].position}" for u in set(holder.values())}
        for unit in dep.units:
            names = [dep.atoms[a].name or "" for a in unit.atoms]
            # A residue of the ligand is a group of its atoms that a contact attaches to as a whole: the
            # kind schema 1.0 already has for that. Its atoms are not nodes of their own in this drawing,
            # so they are named in ``atoms`` and ``members`` stays empty.
            node = Node(id=f"{prefix}:res:{unit.position}", kind="ligand_group",
                        label=f"{unit.text}{unit.number or unit.position}", x=unit.x, y=unit.y,
                        group=group_of.get(unit.index),
                        attrs={"form": "chain_residue", "resname": unit.name, "position": unit.position,
                               **({"number": unit.number} if unit.number not in ("", str(unit.position)) else {}),
                               "class": unit.residue_class, "atoms": names,
                               "backbone": [dep.atoms[a].name or "" for a in unit.backbone],
                               "width": round(unit.box[2] - unit.box[0], 3),
                               "height": round(unit.box[3] - unit.box[1], 3)})
            values = [plddt[n] for n in names if plddt is not None and n in plddt]
            if values:
                node.confidence = float(sum(values) / len(values))
            if unit.beside:
                # Those of its atoms that stand beside the box as atoms of their own, and are nodes.
                node.attrs["beside"] = [dep.atoms[a].name or "" for a in unit.beside]
            self._units.append(node)
            if unit.index in group_of:
                self._pieces.setdefault(group_of[unit.index], []).append(node.id)
        keys = atom_keys(dep)
        by_idx: dict[int, Node] = {}
        for atom in dep.atoms:
            if dep.unit_of(atom.idx) is not None:
                continue
            piece = group_of.get(holder.get(atom.idx, -1))
            node = Node(id=f"{prefix}:{keys[atom.idx]}", kind="ligand_atom", label=atom.name or atom.element,
                        x=atom.x, y=atom.y, group=piece, attrs={"element": atom.element})
            if atom.name is None:
                node.attrs["drawing_only"] = True
                nb = dep.neighbours(atom.idx)
                piece = group_of.get(holder.get(nb[0], -1)) if nb else None
                node.group = piece
            elif plddt is not None and atom.name in plddt:
                node.confidence = float(plddt[atom.name])
            if exposure is not None and atom.name is not None and atom.idx < len(exposure):
                node.attrs["exposure"] = round(float(exposure[atom.idx]), 4)
            by_idx[atom.idx] = node
            if piece is not None:
                self._pieces.setdefault(piece, []).append(node.id)
        # ``_atoms`` is indexed by depiction atom: an atom of the chain stands for its residue's node.
        self._atoms = [by_idx[a.idx] if a.idx in by_idx else self._units[dep.unit_of(a.idx).index]  # type: ignore[union-attr]
                       for a in dep.atoms]
        self._outside = [by_idx[i] for i in sorted(by_idx)]
        self.coordination = []

    def anchor(self, target: Target, label: str = "") -> Node:
        """The node a link target stands for: an atom, or the centroid / group node made for it."""
        if target.unit is not None:
            return self._units[target.unit]
        if target.kind == "atom":
            return self._atoms[target.atoms[0]]
        ring = target.kind == "ring"
        members = [self._atoms[i] for i in target.atoms]
        ident = f"{self.prefix}:{'ring' if ring else 'group'}:{'-'.join(m.label for m in members)}"
        node = self._extra.get(ident)
        if node is None:
            node = Node(id=ident, kind="ring_centroid" if ring else "ligand_group",
                        label=label or ("ring" if ring else "group"), x=float(target.point[0]),
                        y=float(target.point[1]), group=self.group_id, members=tuple(m.id for m in members))
            self._extra[ident] = node
        return node

    def nodes(self) -> list[Node]:
        if self._units:
            return self._units + self._outside + list(self._extra.values())
        return self._atoms + list(self._extra.values())

    def rigid_group(self, **attrs: Any) -> RigidGroup:
        """The ligand as one rigid body: its depiction keeps its internal geometry whatever moves."""
        extra = dict(attrs)
        if self.coordination:
            extra["coordination"] = self.coordination
        return RigidGroup(id=self.group_id, label=self.label, members=tuple(n.id for n in self.nodes()),
                          movable=True, attrs=extra)

    def rigid_groups(self, **attrs: Any) -> list[RigidGroup]:
        """The rigid bodies of the ligand: the whole of it when it is drawn as atoms; for a chain, only
        each piece outside the chain together with the residue that holds it."""
        if not self._units:
            return [self.rigid_group(**attrs)]
        for node in self._extra.values():            # a ring or group of atoms outside the chain follows its piece
            first = next((n for n in self._atoms if n.id in node.members), None)
            if first is not None and first.group is not None:
                node.group = first.group
                self._pieces.setdefault(first.group, []).append(node.id)
            else:
                node.group = None
        return [RigidGroup(id=gid, label=self.label, members=tuple(members), movable=True, attrs=dict(attrs))
                for gid, members in sorted(self._pieces.items())]


def residue_contacts(part: LigandPart, ident: str, infos: Sequence[PortInfo], ports: Sequence[Port],
                     label_t: Sequence[float], offset: int = 0, **attrs: Any) -> list[Contact]:
    """One contact per tether of a residue, in tether order.

    ``infos`` and ``ports`` run in parallel (``ports[offset + k]`` is the tie of
    ``infos[k]``); ``label_t`` is the engine's label position for every port.
    """
    out = []
    for k, info in enumerate(infos):
        port = ports[offset + k]
        first = info.interactions[0]
        group_label = first.ligand.group_kind if first.ligand is not None else ""
        anchor = part.anchor(info.target, group_label)
        cid = "|".join((ident, info.kind.value, anchor.id, f"{info.site_kind}:{','.join(info.site_atoms)}"))
        # Where along the line its text sits: kept for a line that carries only a tag (a predicted
        # aligned error and no distance) as for one with a label, or a redraw would not find it.
        at = label_t[offset + k] if (info.shown and offset + k < len(label_t)) else None
        contact = Contact(
            id=cid, type=info.kind.value, residue="R:" + ident, ligand=anchor.id, drawn=info.drawn,
            distance=first.record.distance, occupancy=info.occupancy, confidence=info.plddt, label=info.label,
            label_t=None if at is None else float(at), ideal=float(port.ideal), min_length=float(port.min_length),
            weight=float(port.weight), site=info.site_kind, protein_atoms=tuple(info.site_atoms),
            records=tuple(r.index for r in info.interactions), attrs=dict(attrs))
        if info.beyond:
            contact.attrs["beyond"] = True
        errors = [r.record.pae for r in info.interactions if r.record.pae is not None]
        if errors:
            contact.attrs["pae"] = round(min(errors), 2)        # carried for the reader of the graph; not drawn
            draws = next(r.record.pae_draws for r in info.interactions if r.record.pae == min(errors))
            if len(draws) > 1:
                contact.attrs["pae_draws"] = [round(v, 2) for v in draws]
        if info.target.unit is not None:
            # A residue of a chain ligand makes it: say which, with which atoms, and whether by its backbone.
            unit = part.dep.units[info.target.unit]
            made_by = [a for r in info.interactions if r.ligand is not None for a in r.ligand.atoms]
            contact.ligand_residue = anchor.label
            contact.attrs["ligand_atoms"] = list(dict.fromkeys(part.dep.atoms[a].name or "" for a in made_by))
            contact.attrs["ligand_part"] = unit.part(made_by)
        out.append(contact)
    return out


def unique_ids(contacts: Sequence[Contact]) -> None:
    """Make contact ids unique in place (two tethers can differ only in what a style merged away)."""
    seen: dict[str, int] = {}
    for c in contacts:
        seen[c.id] = seen.get(c.id, 0) + 1
        if seen[c.id] > 1:
            c.id = f"{c.id}#{seen[c.id]}"


def link_references(graph: Graph) -> None:
    """Fill in, on every ligand node, the contacts that land on it."""
    by_id = {n.id: n for n in graph.nodes}
    for c in graph.contacts:
        if c.ligand is not None and c.ligand in by_id:
            node = by_id[c.ligand]
            if c.id not in node.contacts:
                node.contacts = node.contacts + (c.id,)


def diagram_graph(view: str, ctx: Context, bodies: Sequence[Body], layout: LayoutResult, params: LayoutParams,
                  *, name: str = "", exposure: Any = None) -> Graph:
    """The graph of a single-ligand diagram, from what the layout engine returned.

    Each body's payload gives its residue (``group``), its tethers (``ports``,
    parallel to the body's own ports) and how it is drawn (``form``).
    """
    conf = ctx.confidence
    part = LigandPart(ctx.dep, label=name, plddt=conf.ligand_atoms if conf is not None else None,
                      exposure=exposure)
    graph = Graph(view=view, bond_length=float(ctx.metrics.bond_length), ensemble=ctx.ensemble,
                  springs=engine_springs(params))
    residues = []
    waters: list[tuple[Node, Contact]] = []
    for body in bodies:
        pay = body.payload
        grp = pay.group
        pose = layout.poses.get(body.key)
        contacts = residue_contacts(part, grp.node_ident, pay.ports, body.ports, layout.label_t.get(body.key, ()))
        graph.contacts.extend(contacts)
        attrs: dict[str, Any] = {"resname": grp.resname, "chain": grp.key[0], "number": grp.key[1],
                                 "class": grp.residue_class, "form": pay.form,
                                 "radius": round(float(body.proper_radius), 3)}
        if grp.key[2]:
            attrs["icode"] = grp.key[2]
        if grp.parts > 1:
            # One of several drawings of this residue, each beside the stretch of the chain it touches there.
            attrs["drawing"] = grp.part + 1
            attrs["drawings"] = grp.parts
        if grp.beyond:
            attrs["beyond"] = True
        share = ctx.share_of(grp)
        if share is not None:
            # A figure of all the models of an ensemble: how many of them contact it.
            attrs["n_models"], attrs["n_total"] = share.n_models, share.n_total
        if pose is not None and any(info.predicted for info in pay.ports):
            # A predicted solvent site is a node of its own, where its label is drawn on the line.
            a_all, b_all, _ = body.link_segments(np.array([[pose.x, pose.y]]), np.array([pose.angle]),
                                                 np.array([pose.mirror]))
            for k, (info, contact) in enumerate(zip(pay.ports, contacts)):
                if not info.predicted:
                    continue
                at = 0.5 if contact.label_t is None else contact.label_t
                x = float(a_all[0, k, 0] + (b_all[0, k, 0] - a_all[0, k, 0]) * at)
                y = float(a_all[0, k, 1] + (b_all[0, k, 1] - a_all[0, k, 1]) * at)
                tag = (info.interactions[0].record.water or "").partition(":")[2]
                extra: dict[str, Any] = {"predicted": True}
                if tag.isdigit() and int(tag) < len(ctx.solvent_sites):
                    found = ctx.solvent_sites[int(tag)].as_dict()
                    extra.update({"score": found["score"], "enclosure": found["enclosure"], "xyz": found["xyz"],
                                  "partners": {"protein": found["protein"], "ligand": found["ligand"]}})
                site = Node(id=f"W:{tag or len(waters)}", kind="solvent_site", label="predicted water", x=x, y=y,
                            movable=False, anchors=tuple(a for a in (contact.ligand, "R:" + grp.node_ident) if a),
                            contacts=(contact.id,), attrs=extra)
                contact.attrs["predicted"] = True
                contact.attrs["via"] = site.id
                waters.append((site, contact))
        node = Node(id="R:" + grp.node_ident, kind="residue", label=grp.label,
                    x=None if pose is None else float(pose.x), y=None if pose is None else float(pose.y),
                    angle=0.0 if pose is None else float(pose.angle), mirror=bool(pose.mirror) if pose else False,
                    anchors=tuple(dict.fromkeys(c.ligand for c in contacts if c.ligand)),
                    spring=float(sum(p.weight for p in body.ports)), contacts=(), confidence=pay.plddt,
                    occupancy=pay.occupancy, attrs=attrs)
        residues.append((node, contacts))
    unique_ids(graph.contacts)
    for node, contacts in residues:
        node.contacts = tuple(c.id for c in contacts)
    for site, contact in waters:                     # ids were made unique above: follow them
        site.contacts = (contact.id,)
    graph.nodes = part.nodes() + [node for node, _ in residues] + [site for site, _ in waters]
    graph.groups = part.rigid_groups()
    link_references(graph)
    return graph


def polymer_positions(graph: Graph, *, prefix: str = "L") -> tuple[dict[int, tuple[float, float]],
                                                                    dict[str, tuple[float, float]]]:
    """Where a graph has a chain ligand: its residues by position, and its atoms outside the chain by name
    (``H@<name>`` for a hydrogen the drawing adds). Positions are taken as they stand."""
    units: dict[int, tuple[float, float]] = {}
    atoms: dict[str, tuple[float, float]] = {}
    for node in graph.nodes:
        if not node.id.startswith(prefix + ":"):
            continue
        if node.kind == "ligand_group" and "position" in node.attrs:
            if node.x is None or node.y is None:
                raise GraphError(f"ligand residue {node.id} has no position in the graph")
            try:
                units[int(node.attrs["position"])] = (node.x, node.y)
            except (KeyError, TypeError, ValueError) as exc:
                raise GraphError(f"ligand residue {node.id} has no usable 'position': {exc}") from exc
        elif node.kind == "ligand_atom":
            if node.x is None or node.y is None:
                raise GraphError(f"ligand atom {node.id} has no position in the graph")
            atoms[node.id[len(prefix) + 1:]] = (node.x, node.y)
    if not units:
        raise GraphError("the graph does not hold a ligand drawn as a chain: none of its nodes is a residue "
                         "of the ligand")
    return units, atoms


def poses_from(graph: Graph, bodies: Sequence[Body], tethered: bool = True
               ) -> tuple[dict[Any, Pose], dict[Any, tuple[float, ...]]]:
    """The poses a graph holds for ``bodies``, and where the label of each tether sits.

    A body is matched to its node by the residue it stands for; a node without
    a position leaves its body unplaced. A graph that was not made for these
    bodies -- another complex, or other render options -- is refused. With
    ``tethered=False`` the node's contacts are not ties (a consensus node's
    contacts stand for a set of ligands) and only positions are read.
    """
    by_id = {c.id: c for c in graph.contacts}
    poses: dict[Any, Pose] = {}
    label_t: dict[Any, tuple[float, ...]] = {}
    for body in bodies:
        ident = body.payload.ident
        node = graph.find("R:" + ident)
        if node is None:
            raise GraphError(f"the graph has no node for residue {ident}: it was not made for this diagram")
        if not tethered:
            if node.x is not None and node.y is not None:
                poses[body.key] = Pose(node.x, node.y, node.angle, node.mirror)
            continue
        drawn, form = node.attrs.get("form"), getattr(body.payload, "form", None)
        if drawn is not None and form is not None and drawn != form:
            raise GraphError(f"residue {ident} is drawn as {form!r} in this diagram and as {drawn!r} in the graph: "
                             "the graph was made with other options (residues=, detail=)")
        if len(node.contacts) != body.n_ports:
            raise GraphError(f"residue {ident} has {body.n_ports} tether(s) in this diagram and "
                             f"{len(node.contacts)} in the graph: the graph was made with other data or options")
        if node.x is None or node.y is None:
            continue
        poses[body.key] = Pose(node.x, node.y, node.angle, node.mirror)
        at = []
        for cid in node.contacts:
            if cid not in by_id:
                raise GraphError(f"residue {ident} refers to an unknown contact {cid!r}")
            value = by_id[cid].label_t
            at.append(0.5 if value is None else value)
        label_t[body.key] = tuple(at)
    return poses, label_t


def depiction_from(lig: Ligand, metrics: Metrics, graph: Graph, *, prefix: str = "L",
                   group: str = "ligand") -> Depiction:
    """The ligand drawn where a graph has its atoms -- to the last digit.

    The atoms' positions are taken as they stand, so a graph that went through
    JSON redraws the identical figure and one whose ligand was moved redraws it
    moved. The ligand is a rigid body in the graph, and nothing here checks
    that it stayed one: positions are simply used.
    """
    bl = metrics.bond_length
    exact: dict[str, tuple[float, float]] = {}
    for node in graph.nodes:
        if node.kind != "ligand_atom" or node.group != group or not node.id.startswith(prefix + ":"):
            continue
        if node.x is None or node.y is None:
            raise GraphError(f"ligand atom {node.id} has no position in the graph")
        exact[node.id[len(prefix) + 1:]] = (node.x, node.y)
    missing = [n for n in lig.names if n not in exact]
    if missing:
        raise GraphError("the graph does not hold this ligand: it has no node for atom(s) "
                         + ", ".join(missing[:6]))
    coords = {n: (exact[n][0] / bl, -exact[n][1] / bl) for n in lig.names}
    try:
        pairs = [(str(a), str(b)) for a, b in graph.group(group).attrs.get("coordination", [])]
    except (TypeError, ValueError) as exc:
        raise GraphError(f"malformed 'coordination' of group {group!r}: {exc}") from exc
    return depict_ligand(lig, metrics, coords=coords, normalise=False, exact=exact, coordination=pairs)
