"""The semantic layer: what a diagram is made of, before it is drawn.

:mod:`bindview.scene` is a tree of drawing primitives. That is the right thing to
serialise to SVG and the wrong thing to interact with: a circle has no identity,
no tether and no weight. A :class:`Graph` says what the picture *is*:

* **nodes** -- ligand atoms, ring centroids and charged groups of the ligand,
  residues, the legend -- each with a stable id, a position, what it is anchored
  to, whether it may be moved and how strongly it is held;
* **contacts** -- the interactions between a residue node and a ligand node,
  each carrying its own distance, occupancy and confidence;
* **rigid groups** -- sets of nodes that move as one body. The 2D depiction of
  a ligand is one: dragging any atom of it moves all of them.

The layout engine's result is turned into a graph, and the renderers draw from
the graph. Serialised, the same object is three more things: the model a GUI
drags, the response a web service returns, and a regression target that does
not move when the SVG's cosmetics do.

The JSON form is versioned (:data:`SCHEMA`, :data:`VERSION`) and documented in
``docs/graph-schema.md``. This module is data only: it imports neither RDKit
nor the layout engine.
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import GraphError

__all__ = ["Graph", "Node", "Contact", "RigidGroup", "SCHEMA", "VERSION", "NODE_KINDS"]

#: Name of the JSON schema.
SCHEMA = "bindview.graph"
#: Version of the schema this module writes: ``major.minor``. A reader accepts any
#: document with its own major version; keys it does not know are kept, not dropped.
VERSION = "1.0"
#: What a document says when it has a ``ligand_group`` node that is a residue of a ligand drawn as a
#: chain (``position`` and its other attributes) or the contact key ``ligand_residue``, all of them
#: optional keys. Such a residue is no node kind of its own: it is a group of ligand atoms that a contact
#: attaches to as a whole, which is what a ``ligand_group`` is. A graph that has none of this is
#: written as 1.0.
VERSION_CHAIN = "1.1"
#: ... and "1.2" when it is the graph of an interface (``meta.interface``) or a contact carries a
#: predicted aligned error (``pae``).
VERSION_INTERFACE = "1.2"
#: ... and "1.3" when the residues of an interface's spine stand on a plane, each where it lies in space
#: (``meta.interface.layout`` is ``"plane"``): the positions of those nodes then mean something, and
#: ``meta.interface.plane`` says how much.
VERSION_PLANE = "1.3"
#: ... and "1.4" when it is the graph of two strands and their base pairs (:class:`bindview.BasePairs`,
#: ``meta.basepairs``): a ``residue`` node for each nucleotide, a ``ligand_group`` for a ligand beside them,
#: and contacts between two residue nodes (``base_pair``, ``base_stack``, ``hbond``), the second in ``ligand``.
VERSION_BASEPAIRS = "1.4"

#: The node kinds of schema 1. A ``solvent_site`` is a predicted water on a bridge, in a figure that was
#: asked for them (``render(solvent_sites=True)``).
NODE_KINDS = ("ligand_atom", "ligand_group", "ring_centroid", "residue", "solvent_site", "legend")


def _clean(value: Any) -> Any:
    """Plain JSON data: tuples become lists, nothing else changes."""
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    return value


def _num(value: Any, what: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraphError(f"{what} must be a number, not {value!r}")
    if not math.isfinite(value):
        raise GraphError(f"{what} must be a finite number, not {value!r}")
    return float(value)


def _finite(value: Any) -> bool:
    """Is it a number with a value: no word, no truth value, not NaN and not infinite?"""
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _pair(value: Any, what: str) -> tuple[float, float]:
    """Two numbers, as a place or a step on the page is given."""
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_finite(v) for v in value):
        return (float(value[0]), float(value[1]))
    raise GraphError(f"{what} must be two numbers, not {value!r}")


def _names_of(node: "Node") -> set[str]:
    """What a residue of a figure is called, in small letters: by its label, with its chain, by its number."""
    number = f"{node.attrs.get('number', '')}{node.attrs.get('icode', '')}"
    chain = str(node.attrs.get("chain") or "")
    plain = f"{node.attrs.get('resname', '')}{number}"
    out = {node.label, node.id.removeprefix("R:"), plain, number}
    if chain:
        out |= {f"{chain}:{plain}", f"{chain}:{number}"}
    return {t.lower() for t in out if t}


def _turns(node: "Node") -> bool:
    """Is it a residue that is drawn with atoms? Anything else has no direction of its own."""
    return node.kind == "residue" and node.attrs.get("form") == "fragment"


def _strs(value: Any, what: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise GraphError(f"{what} must be a list of ids, not {value!r}")
    return tuple(str(v) for v in value)


@dataclass
class Node:
    """One thing in the diagram that has an identity and a place.

    ``x`` / ``y`` are in the diagram's own frame (add :attr:`Graph.origin` for
    the page coordinates of the SVG); ``None`` means the layout could not place
    the node. ``anchors`` are the nodes this one is tethered to and ``spring``
    the summed weight of those tethers -- how strongly it is held where it is.
    ``contacts`` lists the contacts that reference the node; for a residue they
    are in tether order. Everything specific to a kind lives in ``attrs``.
    """

    id: str
    kind: str
    label: str
    x: float | None = None
    y: float | None = None
    angle: float = 0.0                  # rotation of the node's own drawing, radians
    mirror: bool = False                # ... and whether it is drawn mirrored
    group: str | None = None            # rigid group the node belongs to
    movable: bool = True
    anchors: tuple[str, ...] = ()
    spring: float | None = None         # None: not tethered to anything
    contacts: tuple[str, ...] = ()
    members: tuple[str, ...] = ()       # ring_centroid / ligand_group: the atoms it stands for
    confidence: float | None = None     # pLDDT scale, 0-100
    occupancy: float | None = None      # share of the ensemble, 0-1
    attrs: dict[str, Any] = field(default_factory=dict)

    _CORE = ("id", "kind", "label", "x", "y", "angle", "mirror", "group", "movable", "anchors", "spring",
             "contacts", "members", "confidence", "occupancy")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "kind": self.kind, "label": self.label, "x": self.x, "y": self.y}
        if self.angle:
            out["angle"] = self.angle
        if self.mirror:
            out["mirror"] = True
        if self.group is not None:
            out["group"] = self.group
        out["movable"] = self.movable
        if self.anchors:
            out["anchors"] = list(self.anchors)
        if self.spring is not None:
            out["spring"] = self.spring
        if self.contacts:
            out["contacts"] = list(self.contacts)
        if self.members:
            out["members"] = list(self.members)
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.occupancy is not None:
            out["occupancy"] = self.occupancy
        for key in sorted(self.attrs):
            out[key] = _clean(self.attrs[key])
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Node":
        try:
            ident, kind = str(data["id"]), str(data["kind"])
        except KeyError as exc:
            raise GraphError(f"a node lacks {exc.args[0]!r}: {dict(data)!r}") from None
        return cls(
            id=ident, kind=kind, label=str(data.get("label", "")),
            x=_num(data.get("x"), f"{ident}.x"), y=_num(data.get("y"), f"{ident}.y"),
            angle=_num(data.get("angle", 0.0), f"{ident}.angle") or 0.0, mirror=bool(data.get("mirror", False)),
            group=None if data.get("group") is None else str(data["group"]),
            movable=bool(data.get("movable", True)), anchors=_strs(data.get("anchors"), f"{ident}.anchors"),
            spring=_num(data.get("spring"), f"{ident}.spring"),
            contacts=_strs(data.get("contacts"), f"{ident}.contacts"),
            members=_strs(data.get("members"), f"{ident}.members"),
            confidence=_num(data.get("confidence"), f"{ident}.confidence"),
            occupancy=_num(data.get("occupancy"), f"{ident}.occupancy"),
            attrs={k: v for k, v in data.items() if k not in cls._CORE})


@dataclass
class Contact:
    """One interaction, between a residue node and a ligand node.

    ``ligand`` is the node the contact attaches to on the ligand -- an atom, a
    ring centroid or a charged group -- and is ``None`` in a consensus view,
    where a contact stands for a whole set of ligands. ``occupancy`` and
    ``confidence`` belong to the contact, not to the residue. ``ideal``,
    ``min_length`` and ``weight`` describe the tether the layout engine used
    for it: a spring of rest length ``ideal``.
    """

    id: str
    type: str
    residue: str
    ligand: str | None = None
    drawn: bool = True                  # False: a contact without direction, which gets no line
    distance: float | None = None       # angstrom, as reported
    occupancy: float | None = None      # share of the ensemble in which the contact is made
    confidence: float | None = None     # pLDDT scale; the contact's own
    label: str | None = None            # text printed on the line
    label_t: float | None = None        # where along the line it sits, 0 = residue end
    ideal: float | None = None
    min_length: float | None = None
    weight: float | None = None
    site: str = ""                      # what makes it on the residue: atom | ring | group | residue
    protein_atoms: tuple[str, ...] = ()
    records: tuple[int, ...] = ()       # indices of the interaction records it stands for
    #: For a ligand drawn as a chain: the residue of the ligand that makes the contact, as ``Lys20`` --
    #: the code and the position in the ligand's own chain. ``None`` for a ligand drawn as atoms.
    ligand_residue: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    _CORE = ("id", "type", "residue", "ligand", "drawn", "distance", "occupancy", "confidence", "label", "label_t",
             "ideal", "min_length", "weight", "site", "protein_atoms", "records", "ligand_residue")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "type": self.type, "residue": self.residue, "ligand": self.ligand,
                               "drawn": self.drawn}
        for key in ("distance", "occupancy", "confidence", "label", "label_t", "ideal", "min_length", "weight"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.site:
            out["site"] = self.site
        if self.protein_atoms:
            out["protein_atoms"] = list(self.protein_atoms)
        if self.records:
            out["records"] = list(self.records)
        if self.ligand_residue is not None:
            out["ligand_residue"] = self.ligand_residue
        for key in sorted(self.attrs):
            out[key] = _clean(self.attrs[key])
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Contact":
        try:
            ident, kind, residue = str(data["id"]), str(data["type"]), str(data["residue"])
        except KeyError as exc:
            raise GraphError(f"a contact lacks {exc.args[0]!r}: {dict(data)!r}") from None
        return cls(
            id=ident, type=kind, residue=residue,
            ligand=None if data.get("ligand") is None else str(data["ligand"]), drawn=bool(data.get("drawn", True)),
            distance=_num(data.get("distance"), f"{ident}.distance"),
            occupancy=_num(data.get("occupancy"), f"{ident}.occupancy"),
            confidence=_num(data.get("confidence"), f"{ident}.confidence"),
            label=None if data.get("label") is None else str(data["label"]),
            label_t=_num(data.get("label_t"), f"{ident}.label_t"), ideal=_num(data.get("ideal"), f"{ident}.ideal"),
            min_length=_num(data.get("min_length"), f"{ident}.min_length"),
            weight=_num(data.get("weight"), f"{ident}.weight"), site=str(data.get("site", "")),
            protein_atoms=_strs(data.get("protein_atoms"), f"{ident}.protein_atoms"),
            records=tuple(int(v) for v in data.get("records", ())),
            ligand_residue=None if data.get("ligand_residue") is None else str(data["ligand_residue"]),
            attrs={k: v for k, v in data.items() if k not in cls._CORE})


@dataclass
class RigidGroup:
    """Nodes that move as one body: their positions relative to each other are fixed."""

    id: str
    label: str = ""
    members: tuple[str, ...] = ()
    movable: bool = True
    attrs: dict[str, Any] = field(default_factory=dict)

    _CORE = ("id", "kind", "label", "members", "movable")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "kind": "rigid", "label": self.label, "members": list(self.members),
                               "movable": self.movable}
        for key in sorted(self.attrs):
            out[key] = _clean(self.attrs[key])
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RigidGroup":
        if "id" not in data:
            raise GraphError(f"a group lacks 'id': {dict(data)!r}")
        if data.get("kind", "rigid") != "rigid":
            raise GraphError(f"group {data['id']!r}: unknown kind {data.get('kind')!r} (schema 1 knows 'rigid')")
        return cls(id=str(data["id"]), label=str(data.get("label", "")),
                   members=_strs(data.get("members"), f"{data['id']}.members"),
                   movable=bool(data.get("movable", True)),
                   attrs={k: v for k, v in data.items() if k not in cls._CORE})


@dataclass
class Graph:
    """A diagram as nodes, contacts and rigid groups.

    ``view`` names the renderer the graph was laid out for (``detailed``,
    ``schematic``, ``difference``, ``consensus``). Positions are in the
    diagram's own frame, in SVG user units; ``origin`` is the offset of that
    frame on the page and ``size`` the page. ``springs`` holds the layout
    engine's constants, so that whoever moves a node can know what held it.
    """

    view: str
    bond_length: float
    nodes: list[Node] = field(default_factory=list)
    contacts: list[Contact] = field(default_factory=list)
    groups: list[RigidGroup] = field(default_factory=list)
    origin: tuple[float, float] = (0.0, 0.0)
    size: tuple[float, float] | None = None
    ensemble: str | None = None         # what the contacts' occupancy is a share of
    springs: dict[str, float] = field(default_factory=dict)
    panels: list[dict[str, Any]] = field(default_factory=list)   # difference view: one per ligand
    meta: dict[str, Any] = field(default_factory=dict)

    # -- lookups -----------------------------------------------------------
    def find(self, ident: str) -> Node | None:
        """The node with exactly this id, or None."""
        for node in self.nodes:
            if node.id == ident:
                return node
        return None

    def node(self, ident: str) -> Node:
        """The node with this id, or the residue that the figure calls so: ``"R:A:MET743"``, ``"Met743"``,
        ``"A:Met743"``, ``"743"``; and ``"ligand"`` for any atom of it (it is one body)."""
        hit = self.find(ident)
        if hit is not None:
            return hit
        wanted = ident.strip().lower()
        named = [n for n in self.nodes if n.kind == "residue" and wanted in _names_of(n)]
        if len(named) == 1:
            return named[0]
        if len(named) > 1:
            raise GraphError(f"{ident!r} is {len(named)} nodes of this graph ({', '.join(n.id for n in named)}): "
                             "name one by its id")
        if wanted == "ligand":
            atoms = [n for n in self.nodes if n.kind == "ligand_atom" and n.group is not None]
            if atoms:
                return atoms[0]
        raise GraphError(f"no node {ident!r} in the graph")

    def contact(self, ident: str) -> Contact:
        """The contact with this id."""
        for c in self.contacts:
            if c.id == ident:
                return c
        raise GraphError(f"no contact {ident!r} in the graph")

    def group(self, ident: str) -> RigidGroup:
        """The rigid group with this id."""
        for g in self.groups:
            if g.id == ident:
                return g
        raise GraphError(f"no group {ident!r} in the graph")

    def of_kind(self, kind: str) -> list[Node]:
        """All nodes of a kind: ``"residue"``, ``"ligand_atom"``, ..."""
        return [n for n in self.nodes if n.kind == kind]

    def page(self, node: "Node | str") -> tuple[float, float] | None:
        """Where a node is on the page of the SVG (``None`` if it was not placed)."""
        n = self.node(node) if isinstance(node, str) else node
        if n.x is None or n.y is None:
            return None
        return (n.x + self.origin[0], n.y + self.origin[1])

    # -- editing -----------------------------------------------------------
    def move(self, ident: str, dx: float, dy: float) -> list[str]:
        """Move a node the way a drag does, and return the ids of what moved.

        A node of a rigid group takes the whole group with it: the depiction
        of a ligand drawn as atoms is one body, so every atom, ring centroid
        and group node of it shifts by the same vector. A ligand drawn as a
        chain is not one body: each of its residues moves alone, and takes
        with it only what is bonded to it outside the chain. Any other node
        moves alone. Nothing else follows -- tethers are springs, not rods,
        and whether to relax the layout afterwards is the caller's decision.
        """
        if not (_finite(dx) and _finite(dy)):
            raise GraphError(f"move takes a step of two finite numbers, not dx={dx!r} and dy={dy!r}")
        dx, dy = float(dx), float(dy)
        node = self.node(ident)
        if not node.movable:
            raise GraphError(f"node {ident!r} is not movable")
        targets = [node]
        if node.group is not None:
            group = self.group(node.group)
            if not group.movable:
                raise GraphError(f"group {group.id!r} is not movable")
            members = set(group.members)
            targets = [n for n in self.nodes if n.id in members]
        if node.kind == "legend":
            offset = self.legend_offset()
            node.attrs["offset"] = [offset[0] + dx, offset[1] + dy]
        for n in targets:
            if n.x is not None and n.y is not None:
                n.x, n.y = n.x + dx, n.y + dy
        return [n.id for n in targets]

    def rotate(self, ident: str, degrees: float) -> list[str]:
        """Turn a node about itself, clockwise on the page, and return the ids of what turned.

        A residue that is drawn with atoms turns on the spot; its name and its lines follow when the figure is
        drawn again. One that is drawn without them (an eyelash, a bubble, a name) faces the ligand wherever it
        stands, and there is nothing to turn. ``"ligand"`` turns a ligand that is drawn as atoms about its
        middle and leaves the residues where they are; ``"figure"`` turns everything, the ligand with its
        residues, about the ligand's middle.
        """
        if not _finite(degrees):
            raise GraphError(f"rotate takes an angle in degrees, a finite number, not {degrees!r}")
        angle = math.radians(float(degrees))
        if ident in ("ligand", "figure") and self.find(ident) is None:
            group = self._body()
            placed = [n for n in self.nodes if n.id in set(group.members) and n.x is not None and n.y is not None]
            atoms = [n for n in placed if n.kind == "ligand_atom"] or placed
            cx = sum(n.x for n in atoms if n.x is not None) / len(atoms)
            cy = sum(n.y for n in atoms if n.y is not None) / len(atoms)
            if ident == "figure":
                placed = [n for n in self.nodes if n.kind != "legend" and n.x is not None and n.y is not None]
            c, s = math.cos(angle), math.sin(angle)
            for n in placed:
                dx, dy = float(n.x or 0.0) - cx, float(n.y or 0.0) - cy
                n.x, n.y = cx + dx * c - dy * s, cy + dx * s + dy * c
                if n.kind == "residue" and _turns(n):
                    n.angle += angle
            return [n.id for n in placed]
        node = self.node(ident)
        if not _turns(node):
            raise GraphError(f"{node.label} is drawn as {node.attrs.get('form', node.kind)!r}: it has no atoms to turn, and "
                             "faces the ligand wherever it stands")
        node.angle += angle
        return [node.id]

    def flip(self, ident: str) -> list[str]:
        """Mirror a residue that is drawn with atoms in its own axis, as a page is turned over. (A ligand is not
        flipped: its drawing shows its stereochemistry.)"""
        node = self.node(ident)
        if not _turns(node):
            raise GraphError(f"{node.label} is drawn as {node.attrs.get('form', node.kind)!r}: there is nothing to flip")
        node.mirror = not node.mirror
        return [node.id]

    def legend_offset(self) -> tuple[float, float]:
        """How far the legend was moved from where the page puts it; ``(0, 0)`` for a graph without one."""
        legend = self.find("legend")
        if legend is None or "offset" not in legend.attrs:
            return (0.0, 0.0)
        return _pair(legend.attrs["offset"], "the offset of the legend")

    def _body(self) -> RigidGroup:
        bodies = [g for g in self.groups if g.movable and g.id == "ligand"]
        if not bodies:
            raise GraphError("this graph has no ligand that is one body: a ligand drawn as a chain is moved residue by "
                             "residue")
        return bodies[0]

    # -- checking ----------------------------------------------------------
    def validate(self) -> "Graph":
        """Raise :class:`GraphError` unless ids are unique, every reference resolves, and the scale and every
        place are numbers that a figure can be drawn from."""
        if not _finite(self.bond_length) or self.bond_length <= 0:
            raise GraphError(f"bond_length must be a positive number, not {self.bond_length!r}")
        self.legend_offset()
        ids: set[str] = set()
        for n in self.nodes:
            if n.id in ids:
                raise GraphError(f"node id {n.id!r} occurs twice")
            ids.add(n.id)
            for name in ("x", "y", "angle"):
                value = getattr(n, name)
                if value is not None and not _finite(value):
                    raise GraphError(f"{n.id}.{name} must be a finite number, not {value!r}")
            if n.kind not in NODE_KINDS:
                raise GraphError(f"node {n.id!r}: unknown kind {n.kind!r}; schema {VERSION} knows "
                                 f"{', '.join(NODE_KINDS)}")
        cids: set[str] = set()
        for c in self.contacts:
            if c.id in cids:
                raise GraphError(f"contact id {c.id!r} occurs twice")
            cids.add(c.id)
            if c.residue not in ids:
                raise GraphError(f"contact {c.id!r} refers to an unknown node {c.residue!r}")
            if c.ligand is not None and c.ligand not in ids:
                raise GraphError(f"contact {c.id!r} refers to an unknown node {c.ligand!r}")
            if c.occupancy is not None and not 0.0 <= c.occupancy <= 1.0:
                raise GraphError(f"contact {c.id!r}: occupancy {c.occupancy!r} is not between 0 and 1")
        gids = {g.id for g in self.groups}
        if len(gids) != len(self.groups):
            raise GraphError("a group id occurs twice")
        for g in self.groups:
            for m in g.members:
                if m not in ids:
                    raise GraphError(f"group {g.id!r} lists an unknown node {m!r}")
        for n in self.nodes:
            if n.group is not None and n.group not in gids:
                raise GraphError(f"node {n.id!r} belongs to an unknown group {n.group!r}")
            for ref in n.anchors + n.members:
                if ref not in ids:
                    raise GraphError(f"node {n.id!r} refers to an unknown node {ref!r}")
            for ref in n.contacts:
                if ref not in cids:
                    raise GraphError(f"node {n.id!r} refers to an unknown contact {ref!r}")
        return self

    # -- JSON --------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """The graph as plain data (``docs/graph-schema.md``)."""
        chain = (any(n.kind == "ligand_group" and "position" in n.attrs for n in self.nodes)
                 or any(c.ligand_residue is not None for c in self.contacts))
        interface = "interface" in self.meta or any("pae" in c.attrs for c in self.contacts)
        version = VERSION_INTERFACE if interface else VERSION_CHAIN if chain else VERSION
        if interface and (self.meta.get("interface") or {}).get("layout") == "plane":
            version = VERSION_PLANE
        if "basepairs" in self.meta:
            version = VERSION_BASEPAIRS
        out: dict[str, Any] = {"schema": SCHEMA, "version": version, "view": self.view,
                               "bond_length": self.bond_length, "origin": list(self.origin)}
        if self.size is not None:
            out["size"] = list(self.size)
        if self.ensemble is not None:
            out["ensemble"] = self.ensemble
        if self.springs:
            out["springs"] = {k: self.springs[k] for k in sorted(self.springs)}
        if self.panels:
            out["panels"] = _clean(self.panels)
        if self.meta:
            out["meta"] = {k: _clean(self.meta[k]) for k in sorted(self.meta)}
        out["groups"] = [g.to_dict() for g in self.groups]
        out["nodes"] = [n.to_dict() for n in self.nodes]
        out["contacts"] = [c.to_dict() for c in self.contacts]
        return out

    def to_json(self, path: "str | Path | None" = None, indent: int | None = 1) -> str:
        """The graph as JSON text (key order fixed, floats in full precision, so the same graph
        always gives the same bytes). Written to ``path`` when given; always returned."""
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False) + "\n"
        if path is not None:
            try:
                Path(path).write_text(text, encoding="utf-8")
            except OSError as exc:
                raise GraphError(f"cannot write {path}: {exc}") from exc
        return text

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Graph":
        """A graph from plain data; refused where it is not one."""
        if not isinstance(data, Mapping) or data.get("schema") != SCHEMA:
            raise GraphError(f"not a {SCHEMA} document (its 'schema' is {data.get('schema')!r})"
                             if isinstance(data, Mapping) else "a graph document must be a JSON object")
        version = str(data.get("version", ""))
        if version.split(".")[0] != VERSION.split(".")[0]:
            raise GraphError(f"graph schema version {version!r} is not supported: this is bindview.graph {VERSION}, "
                             f"which reads version {VERSION.split('.')[0]}.x")
        try:
            size = data.get("size")
            graph = cls(
                view=str(data["view"]), bond_length=_num(data["bond_length"], "bond_length") or 0.0,
                nodes=[Node.from_dict(n) for n in data.get("nodes", ())],
                contacts=[Contact.from_dict(c) for c in data.get("contacts", ())],
                groups=[RigidGroup.from_dict(g) for g in data.get("groups", ())],
                origin=_pair(data.get("origin", (0.0, 0.0)), "origin"),
                size=None if size is None else _pair(size, "size"),
                ensemble=None if data.get("ensemble") is None else str(data["ensemble"]),
                springs={str(k): float(v) for k, v in dict(data.get("springs", {})).items()},
                panels=[dict(p) for p in data.get("panels", ())], meta=dict(data.get("meta", {})))
        except KeyError as exc:
            raise GraphError(f"the graph document lacks {exc.args[0]!r}") from None
        except (TypeError, ValueError, IndexError) as exc:
            raise GraphError(f"malformed graph document: {exc}") from exc
        return graph.validate()

    @classmethod
    def from_json(cls, source: "str | Path") -> "Graph":
        """Read a graph from a file or from JSON text. A ``Path`` is a file; a string is one where it is the name
        of a file that is there, and JSON text where it is not."""
        text, file = str(source), isinstance(source, Path) or _names_a_file(str(source))
        if file:
            try:
                text = Path(source).read_text(encoding="utf-8")
            except (OSError, ValueError) as exc:         # (ValueError: it is not text)
                raise GraphError(f"cannot read {source}: {exc}") from exc
        try:
            data = json.loads(text.removeprefix("\ufeff"))
        except ValueError as exc:
            if file:
                raise GraphError(f"{source} is not valid JSON: {exc}") from exc
            shown = repr(text if len(text) <= 60 else text[:60] + "...")
            raise GraphError(f"not valid JSON, and not the name of a file that is there: {shown} ({exc})") from exc
        return cls.from_dict(data)


def _names_a_file(text: str) -> bool:
    """Is it the name of a file that is there? No JSON document is: a name has no line break in it."""
    try:
        return "\n" not in text and Path(text).is_file()
    except (OSError, ValueError):                        # too long for a name, or with a character no name has
        return False
