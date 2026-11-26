"""Comparing ligands of one pocket: the difference view (two) and the consensus view (n).

One design rule carries both views: **the residues are the invariant, the
ligand is the variable.** The pocket is laid out once, as a frame of residue
nodes, and what changes from ligand to ligand is drawn against that frame. A
residue never moves between panels, so a difference between two panels is a
difference between two ligands and nothing else.

* **Difference view** -- two panels on one frame. Each panel shows one ligand
  and its contacts; a node says whether its residue is contacted by both
  ligands, by this one only, or by the other one only. That classification,
  taken per (residue, interaction type), is the figure, and it is also returned
  as data.
* **Consensus view** -- one frame, one node per residue, each annotated with
  how many ligands of the set contact it and through which interaction types.
  No per-ligand lines are drawn at any set size: past two or three ligands they
  are what makes such figures unreadable, and conservation is the signal.

Whether the ligands are drawn on a common scaffold is decided by
:mod:`bindview.align` and stated in the figure either way. Each contact keeps the
confidence of the model it comes from (:mod:`bindview.confidence`); in the
consensus view the ring around a node is drawn heavy only for the ligands whose
contact is confidently predicted.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .align import (MIN_ATOMS, MIN_FRACTION, CommonCore, core_coords, find_common_core, pocket_transforms,
                    pose_coords, pose_frame)
from .confidence import LOOKS, MARK, PLAIN, Look, look_of
from .depict import Depiction, depict_ligand
from .diagram import (_PT_PER_PX, _PX_PER_MM, Diagram, Quality, RenderResult, _better, _confidence_strip, _dedupe,
                      _scaled, _set_fans)
from .errors import RenderError
from .fingerprint import TYPE_ORDER, Bit, ConsensusRow, Fingerprint, FingerprintTable
from .graph import Contact, Graph, Node
from .graphbuild import LigandPart, engine_springs, link_references, poses_from, residue_contacts, unique_ids
from .geometry import polygon_area, resample_closed
from .interactions import Interaction, InteractionType
from .interactive import NOTHING_TO_POINT_AT, NUMBER, ROOT_CLASS
from .layout import Body, LayoutParams, LayoutResult, LigandField, Port, Pose, layout_bodies
from .residues import format_residue_label, is_backbone_atom, residue_class
from .resolve import ResolvedInteraction
from .scene import Circle, Group, Item, Line, Paint, Polyline, Rect, Scene, Span, Text, items_bounds
from .styles.base import (Context, LegendEntry, RenderOptions, beyond_any_page, charge_legend, confidence_legend,
                          group_residues, group_skeleton, too_small_on_page,
                          halo_items, halo_radius, hbond_direction, interaction_data, interaction_title,
                          legend_group, link_items, link_legend, make_ports, port_infos, residue_plddt)
from .styles.ligand_draw import atom_looks, draw_ligand
from .styles.schematic import _CLASS_LABEL, _CLASS_ORDER, _HBOND_SIDECHAIN, _STYLES, _merge_by_target
from .styles.summary import text_block, wrap_text
from .svg import svg_and_css
from .textmetrics import CAP_HEIGHT, text_width
from .structure import ResidueKey
from .theme import Metrics, Palette, get_palette, scheme

__all__ = ["Comparison", "Difference", "VIEWS", "TYPE_CODES"]

T = InteractionType
VIEWS = ("difference", "consensus")

#: Short codes for interaction types, printed on nodes where a line would not do.
TYPE_CODES: dict[InteractionType, str] = {
    T.HYDROPHOBIC: "hyd", T.HBOND: "HB", T.SALTBRIDGE: "SB", T.PI_STACKING: "π–π",
    T.PI_CATION: "π–cat", T.HALOGEN: "XB", T.METAL: "M", T.WATER_BRIDGE: "wat",
    T.COVALENT: "cov",
}
_CODE_TOKEN = {T.HYDROPHOBIC: "hydrophobic", T.HBOND: "sch_hbond", T.SALTBRIDGE: "sch_saltbridge",
               T.PI_STACKING: "sch_pi_stacking", T.PI_CATION: "sch_pi_cation", T.HALOGEN: "sch_halogen",
               T.METAL: "sch_metal", T.WATER_BRIDGE: "sch_water_bridge", T.COVALENT: "ligand_bond"}

_FONT = 9.5                      # residue name and number in a node
_NOTE = 8.5                      # annotations under a node
_THUMB_FONT = 7.5                # smallest text of a ligand thumbnail
_BOND_LENGTHS = (26.0, 22.0, 19.0)
_SUBSTITUENT = "‡"          # double dagger: contacted only through atoms outside the common core
_FADED = Look(None, width=0.8, opacity=0.45, level=1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Difference:
    """Contacts of two ligands, classified. A contact is a (residue, interaction type) pair, and a residue is
    a chain, a number and an insertion code, whatever the two structures call it: a shared contact carries the
    name that the first ligand's structure gives its residue."""

    names: tuple[str, str]
    shared: tuple[Bit, ...]
    only_a: tuple[Bit, ...]
    only_b: tuple[Bit, ...]

    @property
    def tanimoto(self) -> float:
        """How alike the two are: shared bits over all bits."""
        total = len(self.shared) + len(self.only_a) + len(self.only_b)
        return len(self.shared) / total if total else 1.0

    def residues(self) -> dict[tuple[str, int, str], str]:
        """Each contacted residue as ``"shared"``, ``"a"`` or ``"b"`` (contacted by both ligands
        in any way, or by one of them only)."""
        in_a = {b.residue for b in self.shared + self.only_a}
        in_b = {b.residue for b in self.shared + self.only_b}
        return {res: "shared" if (res in in_a and res in in_b) else ("a" if res in in_a else "b")
                for res in sorted(in_a | in_b)}

    def as_dict(self) -> dict[str, Any]:
        """The classification as plain data."""
        return {"ligands": list(self.names), "tanimoto": round(self.tanimoto, 3),
                "shared": [b.column for b in self.shared], "only_a": [b.column for b in self.only_a],
                "only_b": [b.column for b in self.only_b],
                "residues": {f"{k[0]}:{k[1]}{k[2]}": v for k, v in self.residues().items()}}


@dataclass
class _Node:
    """What a frame body stands for."""

    key: tuple[str, int, str]
    resname: str
    name: str                       # text in the node, first line
    number: str                     # second line
    radius: float
    infos: list[list] = field(default_factory=list)              # per ligand: PortInfo list
    spans: list[tuple[int, int]] = field(default_factory=list)   # per ligand: its slice of the body's ports
    note_at: float = 0.0            # y offset of the annotation below the centre
    ring: float = 0.0               # radius of the conservation ring (consensus view)
    row: ConsensusRow | None = None

    @property
    def ident(self) -> str:
        chain, number, icode = self.key
        return f"{chain}:{self.resname}{number}{icode}" if chain else f"{self.resname}{number}{icode}"

    @property
    def numbered(self) -> str:
        return f"{self.key[1]}{self.key[2]}"


@dataclass
class _Frame:
    """A laid-out residue frame and what is drawn against it."""

    m: Metrics
    bodies: list[Body]
    layout: LayoutResult
    contexts: list[Context]
    mode: str                                   # core | pose | own | envelope | sequence
    fld: LigandField | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    #: The frame as a graph, and the poses read back out of it: what the panels are drawn from.
    graph: Graph | None = None
    poses: dict[Any, Pose] = field(default_factory=dict)
    label_t: dict[Any, tuple[float, ...]] = field(default_factory=dict)


def _origins(scene: Scene) -> tuple[list[tuple[float, float]], tuple[float, float] | None, Group | None]:
    """Where the diagram frames and the legend ended up on the page: the page offset of every
    ``diagram`` group in drawing order, and of the legend with the legend group itself."""
    frames: list[tuple[float, float]] = []
    found: list[tuple[tuple[float, float], Group]] = []

    def walk(items: Sequence[Item], dx: float, dy: float) -> None:
        for it in items:
            if not isinstance(it, Group):
                continue
            tx, ty = it.translate or (0.0, 0.0)
            if it.cls == "diagram":
                frames.append((dx + tx, dy + ty))
            elif it.cls == "legend":
                found.append(((dx + tx, dy + ty), it))
            else:
                walk(it.items, dx + tx, dy + ty)
    walk(scene.items, 0.0, 0.0)
    return frames, (found[0][0] if found else None), (found[0][1] if found else None)


def _on_page(graph: Graph, scene: Scene) -> Graph:
    """A copy of a frame's graph with what only the finished page knows: origin, size, legend."""
    out = copy.deepcopy(graph)
    frames, legend_at, legend = _origins(scene)
    if frames:
        out.origin = frames[0]
    out.size = (scene.width, scene.height)
    for panel, at in zip(out.panels, frames):
        panel["origin"] = [at[0], at[1]]
    if legend is not None and legend_at is not None:
        box = items_bounds(legend.items) or (0.0, 0.0, 0.0, 0.0)
        out.nodes.append(Node(id="legend", kind="legend", label="Legend", x=legend_at[0] - out.origin[0],
                              y=legend_at[1] - out.origin[1],
                              attrs={"width": round(box[2] - box[0], 3), "height": round(box[3] - box[1], 3),
                                     "offset": [0.0, 0.0]}))
    return out


def _residue_of(hit: ResolvedInteraction) -> tuple[str, int, str]:
    """The residue of a record as a comparison keys it: on the chain the structure resolved it to, which the
    record's own table may leave empty."""
    rec = hit.record
    return (hit.residue.chain if hit.residue is not None else rec.chain, rec.residue_number, rec.insertion_code)


def _arc(cx: float, cy: float, r: float, a0: float, a1: float) -> tuple[tuple[float, float], ...]:
    n = max(2, int(math.ceil(abs(a1 - a0) * r / 2.5)))
    return tuple((cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n))
                 for k in range(n + 1))


def _merge(deps: Sequence[Depiction], m: Metrics) -> Depiction:
    """Several drawings in one frame, as one obstacle for the layout."""
    atoms: list = []
    bonds: list = []
    rings: list = []
    off = 0
    for dep in deps:
        atoms.extend(replace(a, idx=a.idx + off) for a in dep.atoms)
        bonds.extend(replace(b, a=b.a + off, b=b.b + off) for b in dep.bonds)
        rings.extend(tuple(i + off for i in ring) for ring in dep.rings)
        off += len(dep.atoms)
    return Depiction(atoms=atoms, bonds=bonds, rings=rings, n_heavy=len(atoms), metrics=m)


def _frame_depiction(lig: Any, m: Metrics, xy: dict[str, tuple[float, float]]) -> Depiction:
    """A ligand drawn on frame coordinates.

    Coordination lines are kept only where the two atoms ended up next to each
    other: a counter-ion that was set down beside the drawing would otherwise
    be tied to its donors by lines running across the whole molecule.
    """
    dep = depict_ligand(lig, m, coords=xy, normalise=False)
    limit = 2.2 * m.bond_length
    dep.bonds = [b for b in dep.bonds if b.kind != "coordination"
                 or math.dist(dep.pos(b.a), dep.pos(b.b)) <= limit]
    return dep


def _node_body(node: _Node, body_radius: float, ports: list[Port], note: tuple[float, float] | None,
               priority: float, clearance: float) -> Body:
    pts = np.zeros((0, 2))
    wh = np.zeros((0, 2))
    if note is not None:
        node.note_at = body_radius + 2.0
        pts = np.array([(0.0, body_radius + 2.0 + note[1] / 2.0)])
        wh = np.array([note])
    return Body(key=node.key, kind="disc", shapes_p0=np.zeros((1, 2)), shapes_p1=np.zeros((1, 2)),
                shapes_r=np.array([body_radius]), ports=ports, ligand_clearance=clearance, priority=priority,
                upright_pts=pts, upright_wh=wh, payload=node)


def _node_text(resname: str, key: tuple[str, int, str], with_chain: bool) -> tuple[str, str]:
    num = f"{key[1]}{key[2]}"
    return resname.upper(), (f"{key[0]}:{num}" if with_chain and key[0] else num)


def _node_radius(name: str, number: str) -> float:
    width = max(text_width(name, _FONT, bold=True), text_width(number, _FONT))
    return max(15.0, math.hypot(width / 2.0, CAP_HEIGHT * _FONT + 1.1) + 2.2)


def _code_spans(kinds: Sequence[InteractionType], prefix: str = "", counts: "dict | None" = None,
                sep: str = " ") -> tuple[tuple[Span, ...], float]:
    """Type codes as coloured spans, with the width they take at the annotation size."""
    spans: list[Span] = []
    width = 0.0
    for i, kind in enumerate(kinds):
        lead = (sep if i else "") + prefix
        text = TYPE_CODES[kind] + (f" {counts[kind]}" if counts else "")
        if lead:
            spans.append(Span(lead))
            width += text_width(lead, _NOTE)
        spans.append(Span(text, fill=_CODE_TOKEN[kind]))
        width += text_width(text, _NOTE)
    return tuple(spans), width


def _range(values: Sequence[float], digits: int) -> str:
    lo, hi = min(values), max(values)
    return f"{lo:.{digits}f}" if round(lo, digits) == round(hi, digits) else f"{lo:.{digits}f}–{hi:.{digits}f}"


class _Page:
    """Blocks stacked top to bottom on a page sized to fit them."""

    def __init__(self, m: Metrics) -> None:
        self.m = m
        self.blocks: list[tuple[Group, float, float, float]] = []

    def add(self, group: Group, width: float, height: float, gap: float = 10.0) -> None:
        if height > 0:
            self.blocks.append((group, width, height, gap))

    def text(self, text: str, wrap: float, fill: str = "text_muted", cls: str = "statement", gap: float = 6.0,
             size: float | None = None, bold: bool = False) -> None:
        size = size or self.m.caption_font
        lines = wrap_text(text, size, wrap, bold)
        items, h = text_block(lines, 0.0, 0.0, size, fill, bold=bold)
        self.add(Group(cls, items), max((text_width(ln, size, bold) for ln in lines), default=0.0), h, gap)

    def build(self, title: str | None, subtitle: str | None, banner: str | None, scene_title: str,
              description: str = "") -> Scene:
        m = self.m
        pad = m.margin
        inner = max([150.0] + [w for _, w, _, _ in self.blocks])
        if title:
            inner = max(inner, text_width(title, m.title_font, bold=True))
        if subtitle:
            inner = max(inner, text_width(subtitle, m.caption_font))
        items: list[Item] = []
        y = pad
        lines: list[str] = []
        if banner:
            lines = wrap_text(banner, m.title_font, max(inner, 330.0), bold=True)
            inner = max(inner, max(text_width(ln, m.title_font, bold=True) for ln in lines))
            y += len(lines) * m.title_font * 1.45 + pad * 0.9
        if title:
            items.append(Text.plain(pad, y + CAP_HEIGHT * m.title_font, title, m.title_font, "text", anchor="start",
                                    bold=True))
            y += m.title_font * 1.5
        if subtitle:
            items.append(Text.plain(pad, y + CAP_HEIGHT * m.caption_font, subtitle, m.caption_font, "text_muted",
                                    anchor="start"))
            y += m.caption_font * 1.6
        for i, (group, _w, h, gap) in enumerate(self.blocks):
            y += gap if (i or items) else 0.0
            outer = Group("block", [group])
            outer.translate = (pad, y)
            items.append(outer)
            y += h
        width, height = inner + 2.0 * pad, y + pad
        scene = Scene(width, height, title=scene_title, description=description)
        scene.items = items
        if lines:
            bh = len(lines) * m.title_font * 1.45 + pad * 0.5
            text_items, _ = text_block(lines, pad, pad * 0.35, m.title_font, "background", bold=True)
            flag = Group("model-flag", [Rect(0.0, 0.0, width, bh, Paint(fill="warning"))] + text_items,
                         data={"trusted": "false"})
            frame = Rect(1.5, 1.5, width - 3.0, height - 3.0, Paint(stroke="warning", width=3.0, join="miter"))
            scene.items = [flag] + scene.items + [Group("model-flag-frame", [frame])]
        return scene


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

class Comparison:
    """Two or more ligands of one pocket, each built as a :class:`~bindview.diagram.Diagram`.

    ``names`` label the ligands in figures and fingerprints (default: each
    diagram's ``name``). The residue numbering is taken to be the same in all
    structures -- that is what makes the residues an invariant. A residue is
    its chain, number and insertion code: one that two structures give two
    names (a point mutant among them) is one residue, in the figures, in
    :meth:`difference` and in :meth:`consensus`, and a warning names it.
    """

    def __init__(self, diagrams: Sequence[Diagram], names: Sequence[str] | None = None, *,
                 min_fraction: float = MIN_FRACTION, min_atoms: int = MIN_ATOMS) -> None:
        if len(diagrams) < 2:
            raise ValueError("a comparison needs at least two ligands")
        self.diagrams = list(diagrams)
        raw = list(names) if names is not None else [d.name or f"ligand {i + 1}" for i, d in enumerate(diagrams)]
        if len(raw) != len(self.diagrams):
            raise ValueError("one name per ligand is needed")
        seen: dict[str, int] = {}
        self.names: list[str] = []
        for nm in raw:
            seen[nm] = seen.get(nm, 0) + 1
            self.names.append(nm if seen[nm] == 1 else f"{nm} ({seen[nm]})")
        self.fingerprints = [Fingerprint.from_diagram(d, nm) for d, nm in zip(self.diagrams, self.names)]
        self.table = FingerprintTable(self.fingerprints)
        #: The common substructure and the verdict on aligning on it.
        self.core: CommonCore = find_common_core([d.ligand for d in self.diagrams], min_fraction, min_atoms)
        #: Backbone RMSD of each structure's pocket on the first one's (``None``: nothing to fit).
        self.pocket_rmsd: list[float | None] = []
        #: Rigid transforms into the first structure's frame (``None``: the pockets are not one pocket).
        self.transforms = pocket_transforms(self.diagrams, rmsds=self.pocket_rmsd)
        self._coords_cache: dict[int, tuple[list[dict[str, tuple[float, float]]], str]] = {}
        self._cache: dict[tuple, tuple] = {}

    # ------------------------------------------------------------------ data
    @property
    def n(self) -> int:
        """How many ligands are compared."""
        return len(self.diagrams)

    def consensus(self) -> list[ConsensusRow]:
        """Per residue, how many ligands contact it and how (see :meth:`FingerprintTable.consensus`)."""
        return self.table.consensus()

    def consensus_contacts(self) -> list[Interaction]:
        """The set as contact records, one per (residue, interaction type): ``occupancy`` is the
        share of the set's ligands that make the contact (see :meth:`FingerprintTable.contacts`)."""
        return self.table.contacts()

    def diagram(self, k: int = 0) -> Diagram:
        """Ligand ``k`` as a single diagram whose contacts carry the series as their occupancy.

        Every interaction record of the ligand gets, as ``occupancy``, the share
        of the set's ligands that make the same contact (same residue, same
        interaction type), and the diagram names the set as its ensemble. It
        then renders like any diagram, in either style: a contact every ligand
        makes is drawn at full strength, one that few make is thin and pale.
        This is the channel a trajectory uses, fed from a different ensemble.
        """
        share = {(c.chain, c.residue_number, c.insertion_code, c.type): c.occupancy
                 for c in self.consensus_contacts()}
        base = self.diagrams[k]
        resolved = []
        for r in base.interactions:
            rec = r.record
            occupancy = share.get((*_residue_of(r), rec.type))
            resolved.append(replace(r, record=rec.with_(occupancy=occupancy)))
        return Diagram(base.ligand, resolved, structure=base.structure, ligand_key=base.ligand_key,
                       name=self.names[k], source=base.source, warnings=base.warnings,
                       confidence=base.confidence, ensemble=f"the {self.n} ligands of the set")

    def difference(self, a: int = 0, b: int = 1) -> Difference:
        """Shared / only-A / only-B contacts of two ligands of the set.

        Two contacts are one where the residue (chain, number, insertion code) and the interaction type
        are the same; the name of the residue is not compared (see the class).
        """
        first, second = self.fingerprints[a], self.fingerprints[b]
        in_a = {(bit.residue, bit.type) for bit in first.bits}
        in_b = {(bit.residue, bit.type) for bit in second.bits}
        return Difference((self.names[a], self.names[b]),
                          tuple(bit for bit in first.columns if (bit.residue, bit.type) in in_b),
                          tuple(bit for bit in first.columns if (bit.residue, bit.type) not in in_b),
                          tuple(bit for bit in second.columns if (bit.residue, bit.type) not in in_a))

    def _two_names(self) -> list[str]:
        """A note for every residue that the structures of the set give more than one name."""
        named: dict[tuple[str, int, str], dict[str, list[str]]] = {}
        for fp in self.fingerprints:
            for bit in fp.bits:
                ligands = named.setdefault(bit.residue, {}).setdefault(bit.resname.upper(), [])
                if fp.name not in ligands:
                    ligands.append(fp.name)
        out = []
        for (chain, number, icode), names in sorted(named.items()):
            if len(names) > 1:
                said = ", ".join(f"{name} with {' and '.join(ligands)}" for name, ligands in names.items())
                out.append(f"{chain}:{number}{icode} has two names in the set ({said}): it is counted as one residue, "
                           "as a comparison goes by chain and number")
        return out

    @property
    def representative(self) -> int | None:
        """The ligand closest to the common substructure (fewest atoms outside it), when there is one."""
        if not self.core.substantial:
            return None
        return min(range(self.n), key=lambda k: (self.core.sizes[k] - self.core.n_atoms, k))

    def core_atoms(self, k: int) -> frozenset[str]:
        """Names of ligand ``k``'s atoms in the common substructure (empty when it is not aligned on)."""
        return frozenset(self.core.atoms[k]) if self.core.substantial else frozenset()

    def contact_via(self) -> dict[tuple[str, int, str], tuple[int, int]]:
        """Per residue: ``(ligands that contact it through an atom of the common core, ligands
        that contact it only through atoms outside the core)``. Empty unless the set is aligned."""
        if not self.core.substantial:
            return {}
        out: dict[tuple[str, int, str], list[int]] = {}
        for k, fp in enumerate(self.fingerprints):
            core = self.core_atoms(k)
            per: dict[tuple[str, int, str], bool] = {}
            for bit, contact in fp.bits.items():
                via_core = any(a in core for a in contact.ligand_atoms)
                per[bit.residue] = per.get(bit.residue, False) or via_core
            for res, via_core in per.items():
                out.setdefault(res, [0, 0])[0 if via_core else 1] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    @property
    def without_bond_orders(self) -> list[str]:
        """The ligands of which only the connectivity is known (no SMILES, and no hydrogens to infer the bond
        orders from). The substructure search takes each of their bonds for a single one, so a double bond of
        another ligand has no match in them: the common substructure can be smaller for it, or missing."""
        return [name for name, dia in zip(self.names, self.diagrams) if dia.ligand.bond_source == "connectivity"]

    def _core_sentence(self) -> str:
        """What was found to align on, and which ligands went into the search without bond orders."""
        bare = self.without_bond_orders
        if not bare:
            return self.core.describe()
        who = bare[0] if len(bare) == 1 else ", ".join(bare[:-1]) + " and " + bare[-1]
        return (f"{self.core.describe()} {who} {'has' if len(bare) == 1 else 'have'} no bond orders (built from "
                "the connectivity alone, without a SMILES), and a double bond of another ligand has no match there: "
                "the common substructure can be smaller than what the ligands share, or missing.")

    def alignment(self) -> dict[str, Any]:
        """The alignment decision as plain data."""
        c = self.core
        return {"common_atoms": c.n_atoms, "smallest_ligand_atoms": min(c.sizes), "fraction": round(c.fraction, 3),
                "min_fraction": c.min_fraction, "aligned": c.substantial, "search_complete": c.complete,
                "without_bond_orders": self.without_bond_orders,
                "superposed": all(t is not None for t in self.transforms),
                "pocket_rmsd": [None if r is None else round(r, 2) for r in self.pocket_rmsd],
                "statement": self._core_sentence()}

    def _numbered_otherwise(self) -> list[str]:
        """The ligands whose pocket has too few residues in common with the first one's to be laid on it: another
        residue numbering, other chain names or another protein (see :func:`bindview.align.pocket_transforms`).
        Residues are compared by chain and number, so what such a ligand shares with the others is not found."""
        if self.diagrams[0].structure is None:
            return []
        return [name for name, dia, fit in zip(self.names, self.diagrams, self.pocket_rmsd)
                if fit is None and dia.structure is not None]

    def _numbering_note(self, apart: Sequence[str]) -> str:
        return (f"{', '.join(apart)}: the pocket has too few residues in common with that of {self.names[0]} to be "
                "laid on it (another residue numbering, other chain names or another protein). Residues are compared "
                "by chain and number: what is shared with the other ligands is counted as different residues")

    def _superposition_sentence(self) -> str:
        """Why there is no common spatial frame, in the figure's words."""
        fitted = [r for r in self.pocket_rmsd if r is not None]
        if len(fitted) < self.n:
            return ("The structures do not share a residue numbering to superpose on, so no spatial frame "
                    "could be made.")
        return (f"The models do not share one pocket conformation either: after superposition on the residues "
                f"around the ligands the backbone RMSD reaches {max(fitted):.0f} \u00c5, so no spatial frame "
                "is claimed.")

    # ------------------------------------------------------------------ frames
    def _coords(self, ref: int) -> tuple[list[dict[str, tuple[float, float]]], str]:
        """2D coordinates of every ligand in one frame (bond-length units, y up), and how they
        were brought there: ``core`` / ``pose`` / ``own``."""
        if ref in self._coords_cache:
            return self._coords_cache[ref]
        base = Metrics()

        def own(k: int) -> dict[str, tuple[float, float]]:
            dep = depict_ligand(self.diagrams[k].ligand, base)
            return {a.name: (a.x / base.bond_length, -a.y / base.bond_length) for a in dep.atoms if a.name}

        ligs = [d.ligand for d in self.diagrams]
        if self.core.substantial:
            ref_xy = own(ref)
            ref_core = [ref_xy[n] for n in self.core.atoms[ref]]
            out = [ref_xy if k == ref else core_coords(ligs[k], self.core.atoms[k], ref_core) for k in range(self.n)]
            mode = "core"
        elif all(t is not None for t in self.transforms):
            clouds = self._clouds()
            frame = pose_frame(clouds)
            owns = [own(k) for k in range(self.n)]
            arrays = [np.array([owns[k][n] for n in ligs[k].names]) for k in range(self.n)]
            proj = [frame.project(c) for c in clouds]

            def spread(a: np.ndarray) -> float:
                return float(np.sqrt(((a - a.mean(axis=0)) ** 2).sum(axis=1).mean())) if len(a) else 0.0
            # One magnification for drawings and poses: bond lengths per angstrom.
            scale = sum(spread(a) for a in arrays) / max(sum(spread(p) for p in proj), 1e-6)
            out = [pose_coords(ligs[k].names, arrays[k], proj[k], scale) for k in range(self.n)]
            mode = "pose"
        else:
            out = []
            for k in range(self.n):
                xy = own(k)
                cx = sum(p[0] for p in xy.values()) / len(xy)
                cy = sum(p[1] for p in xy.values()) / len(xy)
                out.append({n: (x - cx, y - cy) for n, (x, y) in xy.items()})
            mode = "own"
        self._coords_cache[ref] = (out, mode)
        return out, mode

    def _clouds(self) -> list[np.ndarray]:
        """Every ligand's atoms in the first structure's frame."""
        out = []
        for dia, t in zip(self.diagrams, self.transforms):
            assert t is not None
            out.append(dia.ligand.coords3d @ t[0].T + t[1])
        return out

    def _alignment_sentence(self, mode: str, more: str = "") -> str:
        text = self._core_sentence()
        if mode == "core":
            text += " That substructure sits on identical coordinates in every drawing."
            apart = self._numbered_otherwise()
            if apart:
                who = apart[0] if len(apart) == 1 else ", ".join(apart[:-1]) + " and " + apart[-1]
                text += (f" {who} {'does' if len(apart) == 1 else 'do'} not number the residues of the pocket as "
                         f"{self.names[0]} does: residues are compared by chain and number, so what is shared may "
                         "be counted as different residues.")
        elif mode == "pose":
            text += " Each ligand is drawn turned to its own pose in the superposed pocket."
        elif mode == "own":
            text += " " + self._superposition_sentence() + " Each ligand is drawn in its own orientation."
        return text + more

    def _contexts(self, m: Metrics, opts: RenderOptions, palette: Palette, coords: list[dict]) -> list[Context]:
        out = []
        for dia, xy in zip(self.diagrams, coords):
            dep = _frame_depiction(dia.ligand, m, xy)
            drawable = [r for r in dia.interactions if r.ligand is not None]
            out.append(Context(dia.ligand, dep, group_residues(drawable, opts.show_chain), m, palette, opts,
                               confidence=dia.confidence if opts.confidence else None))
        return out

    def _lay_out(self, m: Metrics, bodies: list[Body], obstacle: Depiction,
                 extra: Sequence[tuple[float, float, float]] = ()) -> tuple[LayoutResult, LigandField]:
        _set_fans(bodies)
        params = LayoutParams(unit=m.bond_length)
        fld = LigandField(obstacle, reach=(params.reach + 3.0) * m.bond_length, step=max(2.0, m.bond_length / 10.0),
                          extra=extra)
        return layout_bodies(fld, bodies, params), fld

    # ------------------------------------------------------------------ render
    def render(self, view: str = "auto", theme: "str | Palette" = "light", *, metrics: Metrics | None = None,
               pair: tuple[int, int] = (0, 1), min_ligands: int = 1, thumbnails: bool = True,
               **options: Any) -> RenderResult:
        """Draw the comparison.

        ``view`` is ``"difference"`` (two ligands; ``pair`` picks which two of a
        larger set), ``"consensus"`` (the whole set) or ``"auto"`` (difference
        for two ligands, consensus for more). ``min_ligands`` drops residues
        contacted by fewer ligands from the consensus view, and the figure
        says so. ``thumbnails`` adds the ligands of an aligned set, each drawn
        on the common core. Remaining options are those of
        :class:`~bindview.styles.base.RenderOptions`.
        """
        if view == "auto":
            view = "difference" if self.n == 2 else "consensus"
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}; choose from {', '.join(VIEWS)}")
        if view == "difference" and (self.n != 2 or tuple(pair) != (0, 1)):
            a, b = pair
            if a == b or not (0 <= a < self.n and 0 <= b < self.n):
                raise ValueError(f"pair {pair!r} does not name two ligands of this set")
            sub = Comparison([self.diagrams[a], self.diagrams[b]], [self.names[a], self.names[b]],
                             min_fraction=self.core.min_fraction, min_atoms=self.core.min_atoms)
            return sub.render("difference", theme, metrics=metrics, **options)
        opts = RenderOptions(**options)
        coloured = opts.plddt_colors                    # asked for, and not drawn: a comparison draws pLDDT by weight
        opts = replace(opts, plddt_colors=False)
        css_vars = isinstance(theme, str) and theme.lower() == "auto"
        light, dark = scheme(opts.colors)
        palette = light if css_vars else get_palette(theme, opts.colors)
        base = metrics or Metrics()
        key = (view, base, opts, min_ligands, thumbnails, frozenset(palette.elements))
        if key not in self._cache:
            if view == "difference":
                self._cache[key] = self._render_difference(palette, base, opts)
            else:
                self._cache[key] = self._render_consensus(palette, base, opts, min_ligands, thumbnails)
        scene, status, figure, quality, notes, data = self._cache[key]
        if coloured:
            # Said, and not left to be found out: a comparison draws pLDDT by weight, whatever was asked.
            notes = list(notes) + ["plddt_colors is not drawn in a comparison: its figures show pLDDT by stroke "
                                   "weight and opacity"]
        if options.get("break_lines"):
            # The same, where it was asked for in so many words: asked for and not done has to be said.
            # Left to its default a comparison says nothing; its lines are drawn as they are laid out.
            notes = list(notes) + ["break_lines is not done in a comparison: the lines of its figures are drawn "
                                   "as they are laid out"]
        unused = [k for k in ("residues", "lines", "exposure", "lining", "solvent_mark", "absent") if options.get(k) is not None]
        if unused:
            notes = list(notes) + [f"{', '.join(unused)}: a setting of a single diagram, which a comparison has not"]
        graph = data.get("_graph")
        # A copy: the views are kept, and what a caller does to the data of one result is not in the next.
        data = copy.deepcopy({k: v for k, v in data.items() if k != "_graph"})
        svg = css = None
        if scene is not None:
            svg, css = svg_and_css(scene, palette, background=opts.background, css_vars=css_vars,
                                   dark_palette=dark if css_vars else None, interactive=opts.interactive,
                                   stylesheet=opts.stylesheet)
            if opts.interactive and ROOT_CLASS not in svg.split(">", 1)[0]:
                notes = list(notes) + [NOTHING_TO_POINT_AT]
        conf = {}
        for k in range(self.n):
            c = self.diagrams[k].confidence
            if opts.confidence and c is not None:
                conf[self.names[k]] = self.diagrams[k]._confidence_report(c)
        burial = {self.names[k]: self.diagrams[k]._burial_report() for k in range(self.n)
                  if self.diagrams[k].burial() is not None}
        return RenderResult(status=status, svg=svg, figure=figure, warnings=_dedupe(notes), quality=quality,
                            style=view, scene=scene,
                            # (The theme, whatever the colour scheme, as a ligand's figure reports it.)
                            theme="auto" if css_vars else palette.name if isinstance(theme, Palette) else str(theme).lower(),
                            confidence=conf or None, data=data, burial=burial or None, css=css,
                            graph=None if graph is None else copy.deepcopy(graph))

    def render_svg(self, path: "str | Path | None" = None, view: str = "auto", theme: "str | Palette" = "light",
                   **options: Any) -> RenderResult:
        """Render and, when ``path`` is given and there is a figure, write it."""
        result = self.render(view=view, theme=theme, **options)
        if path is not None and result.svg is not None:
            p = Path(path)
            try:
                p.write_text(result.svg, encoding="utf-8")
            except OSError as exc:
                raise RenderError(f"cannot write {p}: {exc}") from exc
            result.path = p
        return result

    # -- shared pieces -------------------------------------------------------
    def _banner(self, opts: RenderOptions) -> str | None:
        flagged = []
        for k in range(self.n):
            conf = self.diagrams[k].confidence
            if opts.confidence and conf is not None and conf.marked:
                verdict = f" ({conf.model.flag})" if conf.model is not None and conf.model.flag else ""
                flagged.append(self.names[k] + verdict)
        if not flagged:
            return None
        noun = "Model" if len(flagged) == 1 else "Models"
        return (f"{noun} flagged as not trustworthy: {', '.join(flagged)} — shown for inspection, "
                "not as a result.")

    def _member_notes(self, opts: RenderOptions) -> tuple[list[str], bool]:
        """Warnings carried over from the single diagrams, and whether any of them is degraded."""
        notes: list[str] = []
        degraded = False
        for name, dia in zip(self.names, self.diagrams):
            notes.extend(f"{name}: {w}" for w in dia.warnings)
            if dia.unresolved:
                n = len(dia.unresolved)
                notes.append(f"{name}: {n} reported interaction{'s' if n != 1 else ''} could not be attached to "
                             "a ligand atom; counted in the fingerprint, not drawn")
                degraded = True
            if dia.ligand.bond_source == "connectivity":
                degraded = True
                notes.append(f"{name}: no bond orders (built from the connectivity alone, without a SMILES). The "
                             "common substructure of the set is searched with them as they are and can be smaller "
                             "for it, or missing: give the ligand's SMILES")
            conf = dia.confidence
            if opts.confidence and conf is not None and conf.marked:
                verdict = f" ({conf.model.flag})" if conf.model is not None and conf.model.flag else ""
                notes.append(f"{name}: the model is flagged as not trustworthy{verdict}; the figure is marked")
            exposed = dia._burial_warning()        # none for a chain of residues: the cut is not its own
            if exposed is not None:
                notes.append(f"{name}: {exposed}")
        return notes, degraded

    def _quality(self, scene: Scene, m: Metrics, opts: RenderOptions, layout: LayoutResult | None,
                 n_residues: int, min_font_px: float, estimated: bool = False) -> Quality:
        fit = min(opts.figure_width_mm * _PX_PER_MM / scene.width, opts.figure_height_mm * _PX_PER_MM / scene.height,
                  1.0)
        font_pt = min_font_px * _PT_PER_PX * fit
        required = (math.inf if beyond_any_page(min_font_px * _PT_PER_PX, opts, fixed=True) else
                    scene.width * opts.min_font_pt / (min_font_px * _PT_PER_PX) / _PX_PER_MM)
        n_inter = sum(len(d.interactions) for d in self.diagrams)
        return Quality(
            width=scene.width, height=scene.height, bond_length=m.bond_length, n_residues=n_residues,
            n_interactions=n_inter, overlaps=layout.overlaps if layout else 0,
            unplaced=len(layout.unplaced) if layout else 0, line_crossings=layout.line_crossings if layout else 0,
            bond_crossings=layout.bond_crossings if layout else 0, max_stretch=layout.max_stretch if layout else 0.0,
            min_font_pt=font_pt, required_width_mm=required, readable=font_pt >= opts.min_font_pt - 1e-9,
            estimated=estimated, link_hits=layout.link_hits if layout else 0)

    @staticmethod
    def _text_width(m: Metrics, opts: RenderOptions) -> float:
        """Width that running text and thumbnails may take: the figure's, so they do not pile up."""
        return max(360.0, opts.figure_width_mm * _PX_PER_MM - 2.0 * m.margin)

    def _models_line(self, opts: RenderOptions, size: float) -> tuple[Group, float, float] | None:
        """One line with the range of the set's model-level evidence."""
        confs = [d.confidence for d in self.diagrams if opts.confidence and d.confidence is not None]
        if not confs:
            return None
        parts: list[tuple[str, str]] = []
        iptm = [c.model.iptm for c in confs if c.model is not None and c.model.iptm is not None]
        if iptm:
            parts.append(("ipTM", _range(iptm, 2)))
        pocket = [c.pocket_plddt for c in confs if c.pocket_plddt is not None]
        if pocket:
            parts.append(("pocket pLDDT", _range(pocket, 0)))
        lig = [c.ligand_plddt for c in confs if c.ligand_plddt is not None]
        if lig:
            parts.append(("ligand pLDDT", _range(lig, 0)))
        if opts.burial:
            buried = [100.0 * b.buried_fraction for b in (d.burial() for d in self.diagrams) if b is not None]
            if buried and parts:
                parts.append(("ligands buried", _range(buried, 0) + " %"))
        if not parts:
            return None
        lead = f"Model confidence over {len(confs)} models  "
        spans: list[Span] = [Span(lead)]
        width = text_width(lead, size)
        data = {}
        for i, (label, value) in enumerate(parts):
            head = ("  ·  " if i else "") + label + " "
            spans.extend([Span(head), Span(value, fill="text", bold=True)])
            width += text_width(head, size) + text_width(value, size, bold=True)
            data[label.replace(" ", "-").lower()] = value
        text = Text(0.0, CAP_HEIGHT * size, tuple(spans), size, "text_muted", anchor="start")
        return Group("confidence", [text], data=data), width, size * 1.5

    def _summary_page(self, m: Metrics, opts: RenderOptions, headline: str, why: str, table: Group, tw: float,
                      th: float, foot: Sequence[str], scene_title: str) -> Scene:
        page = _Page(m)
        wrap = max(tw, 360.0)
        models = self._models_line(opts, m.caption_font + 1.0)
        if models is not None:
            page.add(*models, gap=4.0)
        page.text(headline, wrap, fill="text", cls="headline", size=m.title_font, bold=True, gap=8.0)
        page.text(why, wrap, fill="warning", cls="reason", gap=3.0)
        page.add(table, tw, th, gap=12.0)
        for line in foot:
            page.text(line, wrap, cls="footnotes", gap=6.0)
        return page.build(opts.title, opts.subtitle, self._banner(opts), scene_title)

    # ------------------------------------------------------------------ difference view
    def _render_difference(self, palette: Palette, base: Metrics, opts: RenderOptions) -> tuple:
        diff = self.difference()
        notes, degraded = self._member_notes(opts)
        notes.extend(self._two_names())
        data: dict[str, Any] = {"view": "difference", "alignment": self.alignment(), "difference": diff.as_dict()}
        title = opts.title or f"{self.names[0]} vs {self.names[1]}"
        if not diff.shared and not diff.only_a and not diff.only_b:
            page = _Page(base)
            page.text("No interactions in either supplied set", 330.0, fill="text", cls="empty-state",
                      size=base.title_font, bold=True)
            page.text(f"Both ligands ({self.names[0]}, {self.names[1]}) were analysed and no contact was found; "
                      "there is nothing to compare.", 330.0, cls="empty-state")
            return page.build(opts.title, opts.subtitle, self._banner(opts), title), "empty", "empty", None, notes, data

        best: tuple[Scene, Quality, _Frame, str] | None = None
        lengths = [opts.bond_length] if opts.bond_length else list(_BOND_LENGTHS)
        done = False
        for bl in lengths:
            frame = self._difference_frame(_scaled(base, bl), opts, palette)
            for arrangement in ("row", "column"):
                scene, panel_notes = self._difference_scene(frame, diff, opts, palette, arrangement)
                q = self._quality(scene, frame.m, opts, frame.layout, len(frame.bodies), _NOTE)
                if best is None or _better(q, best[1]):
                    best = (scene, q, frame, arrangement)
                    best_notes = panel_notes
                if q.valid and q.readable:
                    best = (scene, q, frame, arrangement)
                    best_notes = panel_notes
                    done = True
                    break
            if done:
                break
        assert best is not None
        scene, q, frame, arrangement = best
        notes = notes + best_notes
        data["arrangement"] = arrangement
        data["frame"] = frame.mode
        if frame.mode == "own":
            notes.append("the structures could not be superposed (no common residue numbering); each ligand is "
                         "drawn in its own orientation")
            degraded = True
        elif self._numbered_otherwise():                # on a common scaffold: no frame is missing, the residues are
            notes.append(self._numbering_note(self._numbered_otherwise()))
            degraded = True
        if frame.graph is not None:
            data["_graph"] = _on_page(frame.graph, scene)
        if q.valid and q.readable:
            return scene, "degraded" if degraded else "ok", "difference", q, notes, data
        if not q.valid:
            reason = (f"{q.unplaced} residue(s) could not be placed, {q.overlaps} pair(s) overlap and "
                      f"{q.link_hits} interaction line(s) cross another residue")
        else:
            reason = beyond_any_page(_NOTE * _PT_PER_PX, opts, fixed=True) or too_small_on_page(
                q.min_font_pt, q.required_width_mm, q.width, q.height, opts)
        message = f"a two-panel difference view of these ligands is not readable: {reason}"
        if opts.on_unreadable == "render":
            return scene, "unreadable", "difference", q, notes + [message + "; rendered anyway on request"], data
        data.pop("_graph", None)                       # no node-level figure is handed out
        if opts.on_unreadable == "none":
            return None, "unreadable", "none", q, notes + [message], data
        table, tw, th = self._difference_table(diff, base, opts.figure_width_mm * _PX_PER_MM - 2.0 * base.margin)
        why = (f"A two-panel difference view is not readable at {opts.figure_width_mm:g} mm: {reason}. "
               "Contacts are listed per residue instead.")
        foot = [f"Tanimoto similarity of the two fingerprints: {diff.tanimoto:.2f}.", self._core_sentence()]
        scene = self._summary_page(base, opts, "Contact differences by residue", why, table, tw, th, foot, title)
        return (scene, "unreadable", "summary", q, notes + [message + "; a per-residue summary was drawn instead"],
                data)

    def _difference_frame(self, m: Metrics, opts: RenderOptions, palette: Palette) -> _Frame:
        coords, mode = self._coords(0)
        ctxs = self._contexts(m, opts, palette, coords)
        by_key = [{g.key: g for g in c.groups} for c in ctxs]
        keys = sorted(set(by_key[0]) | set(by_key[1]))
        with_chain = len({k[0] for k in keys}) > 1
        bits = [{(g.key, r.type) for g in c.groups for r in g.interactions} for c in ctxs]
        bodies = []
        for key in keys:
            grp = by_key[0].get(key) or by_key[1][key]
            name, number = _node_text(grp.resname, key, with_chain)
            radius = max(_node_radius(by_key[k][key].resname.upper(), number + MARK)       # (room for either name)
                         for k in (0, 1) if key in by_key[k])
            node = _Node(key, grp.resname, name, number, radius)
            ports: list[Port] = []
            for k in (0, 1):
                infos = (_merge_by_target(port_infos(ctxs[k], by_key[k][key], lambda kind: False))
                         if key in by_key[k] else [])
                start = len(ports)
                if infos:
                    ports.extend(make_ports(ctxs[k], infos,
                                            lambda info, r=radius: ((0.0, 0.0), r + 1.5, None, r, None)))
                node.infos.append(infos)
                node.spans.append((start, len(ports)))
            note = None
            if key in by_key[0] and key in by_key[1]:
                widths = []
                for k in (0, 1):
                    plus = [t for t in TYPE_ORDER if (key, t) in bits[k] and (key, t) not in bits[1 - k]]
                    if plus:
                        widths.append(_code_spans(plus, "+")[1])
                if widths:
                    note = (max(widths) + 2.0, CAP_HEIGHT * _NOTE + 3.0)
            n_drawn = sum(1 for infos in node.infos for i in infos if i.drawn)
            bodies.append(_node_body(node, radius + 1.5, ports, note, float(n_drawn), 0.95 * m.bond_length))
        layout, fld = self._lay_out(m, bodies, _merge([c.dep for c in ctxs], m))
        frame = _Frame(m, bodies, layout, ctxs, mode, fld, {"bits": bits, "by_key": by_key})
        # The engine's result becomes the graph; the panels are drawn from the graph.
        frame.graph = self._difference_graph(frame)
        frame.poses, frame.label_t = poses_from(frame.graph, bodies)
        return frame

    def _difference_graph(self, frame: _Frame) -> Graph:
        """The frame of a difference view as a graph: two rigid ligands, one set of residue nodes."""
        m = frame.m
        by_key, bits = frame.extra["by_key"], frame.extra["bits"]
        graph = Graph(view="difference", bond_length=float(m.bond_length),
                      springs=engine_springs(LayoutParams(unit=m.bond_length)))
        parts = []
        for k in (0, 1):
            conf = frame.contexts[k].confidence
            parts.append(LigandPart(frame.contexts[k].dep, prefix=f"L{k + 1}", group=f"ligand:{k + 1}",
                                    label=self.names[k], plddt=conf.ligand_atoms if conf is not None else None))
        residues = []
        for body in frame.bodies:
            node: _Node = body.payload
            pose = frame.layout.poses.get(body.key)
            at = frame.layout.label_t.get(body.key, ())
            contacts: list[Contact] = []
            state: dict[str, str] = {}
            own: dict[str, float] = {}
            for k in (0, 1):
                other = 1 - k
                grp = by_key[k].get(node.key)
                state[f"ligand:{k + 1}"] = ("absent" if grp is None else
                                            ("shared" if node.key in by_key[other] else "only"))
                if grp is not None:
                    value = residue_plddt(frame.contexts[k], grp)
                    if value is not None:
                        own[f"ligand:{k + 1}"] = value
                for c in residue_contacts(parts[k], node.ident, node.infos[k], body.ports, at,
                                          offset=node.spans[k][0], ligand_group=f"ligand:{k + 1}"):
                    c.attrs["shared"] = (node.key, InteractionType(c.type)) in bits[other]
                    contacts.append(c)
            graph.contacts.extend(contacts)
            attrs: dict[str, Any] = {"resname": node.resname, "chain": node.key[0], "number": node.key[1],
                                     "class": residue_class(node.resname), "form": "node",
                                     "radius": round(node.radius, 3), "contact": state}
            if node.key[2]:
                attrs["icode"] = node.key[2]
            if own:
                attrs["confidence_by_ligand"] = own          # one value per model: the residue's pLDDT in each
            gnode = Node(id="R:" + node.ident, kind="residue",
                         label=format_residue_label(node.resname, node.key[1], node.key[2], None),
                         x=None if pose is None else float(pose.x), y=None if pose is None else float(pose.y),
                         anchors=tuple(dict.fromkeys(c.ligand for c in contacts if c.ligand)),
                         spring=float(sum(p.weight for p in body.ports)), attrs=attrs)
            residues.append((gnode, contacts))
        unique_ids(graph.contacts)
        for gnode, contacts in residues:
            gnode.contacts = tuple(c.id for c in contacts)
        graph.nodes = parts[0].nodes() + parts[1].nodes() + [n for n, _ in residues]
        graph.groups = [parts[k].rigid_group(panel="AB"[k]) for k in (0, 1)]
        graph.panels = [{"id": "AB"[k], "group": f"ligand:{k + 1}", "label": self.names[k]} for k in (0, 1)]
        graph.meta = {"ligands": list(self.names), "frame": frame.mode}
        link_references(graph)
        return graph

    def _resname_in(self, k: int, key: tuple[str, int, str]) -> "str | None":
        """The name that the structure of the ``k``-th ligand gives a residue, where it has one."""
        structure = self.diagrams[k].structure
        if structure is None:
            return None
        try:
            return structure.residue_name(ResidueKey(*key))
        except KeyError:
            return None

    def _difference_panel(self, frame: _Frame, k: int, palette: Palette) -> tuple[Group, dict[str, Any]]:
        m = frame.m
        ctx = frame.contexts[k]
        other = 1 - k
        by_key, bits = frame.extra["by_key"], frame.extra["bits"]
        layers: dict[str, list[Item]] = {"marks": [], "halos": [], "interactions": [], "residues": []}
        seen: dict[str, Any] = {"kinds": [], "classes": [], "levels": set(), "backbone": False, "sidechain": False,
                                "plus": [], "notes": [], "states": set()}
        halo_done: set[tuple] = set()
        dots_done: set[tuple] = set()
        cap = CAP_HEIGHT * _FONT
        for body in frame.bodies:
            pose = frame.poses.get(body.key)
            if pose is None:
                continue
            node: _Node = body.payload
            grp = by_key[k].get(node.key)
            # (A residue that the two structures name differently has, in each panel, the name of its own,
            # also where only the other ligand touches it.)
            resname = grp.resname if grp is not None else self._resname_in(k, node.key) or node.resname
            own = resname.upper()
            cls = residue_class(resname)
            if cls not in seen["classes"]:
                seen["classes"].append(cls)
            state = "absent" if grp is None else ("shared" if node.key in by_key[other] else "only")
            seen["states"].add(state)
            g = Group("residue", data={"residue": node.ident, NUMBER: node.numbered, "class": cls, "form": "node",
                                       "contact": state})
            if grp is None:
                # Contacted by the other ligand only: the node keeps its place and says so.
                g.title = f"{own.capitalize()}{node.number}: no contact with {self.names[k]}"
                g.add(Circle(pose.x, pose.y, node.radius, Paint(stroke="bubble_line", width=0.8, dash=(1.2, 2.6))))
                g.add(Text.plain(pose.x, pose.y - 1.1, own, _FONT, "text_muted"))
                g.add(Text.plain(pose.x, pose.y + cap + 1.1, node.number, _FONT, "text_muted"))
                layers["residues"].append(g)
                continue
            plddt = residue_plddt(ctx, grp)
            look = look_of(plddt)
            if ctx.confidence is not None:
                seen["levels"].add(look.level)
            g.title = f"{grp.label}: " + ", ".join(t.label.lower() for t in grp.types)
            if plddt is not None and look.band is not None:
                g.data["plddt"] = f"{plddt:.1f}"
                g.data["confidence"] = look.band
                g.title += f"; residue pLDDT {plddt:.0f}"
            heavy = state == "only"
            g.add(Circle(pose.x, pose.y, node.radius, Paint(fill=f"class_{cls}", opacity=look.opacity)))
            g.add(Circle(pose.x, pose.y, node.radius - (0.8 if heavy else 0.0),
                         Paint(stroke="bubble_line", width=2.5 if heavy else 0.9)))
            g.add(Text.plain(pose.x, pose.y - 1.1, own, _FONT, "bubble_text", bold=look.level == 0))
            g.add(Text.plain(pose.x, pose.y + cap + 1.1, node.number + (MARK if look.mark else ""), _FONT,
                             "bubble_text"))
            if state == "shared":
                plus = [t for t in TYPE_ORDER if (node.key, t) in bits[k] and (node.key, t) not in bits[other]]
                if plus:
                    spans, _ = _code_spans(plus, "+")
                    g.add(Text(pose.x, pose.y + node.note_at + CAP_HEIGHT * _NOTE + 1.5, spans, _NOTE, "text_muted"))
                    seen["plus"].extend(t for t in plus if t not in seen["plus"])
            layers["residues"].append(g)

            P = np.array([[pose.x, pose.y]])
            a_all, b_all, _ = body.link_segments(P, np.array([0.0]), np.array([False]))
            label_t = frame.label_t.get(body.key, ())
            start = node.spans[k][0]
            for j, info in enumerate(node.infos[k]):
                idx = start + j
                if info.kind not in seen["kinds"]:
                    seen["kinds"].append(info.kind)
                if ctx.confidence is not None:
                    seen["levels"].add(info.look.level)
                first = info.interactions[0]
                data = interaction_data(ctx, grp, info)
                data["contact"] = "shared" if (node.key, info.kind) in bits[other] else "only"
                title = "; ".join(interaction_title(r, ctx.ligand, info.plddt) for r in info.interactions)
                if not info.drawn:
                    # No line for a contact without direction; it stays addressable as data.
                    layers["interactions"].append(Group(f"contact {info.kind.value}", data=data, title=title))
                    continue
                style = _STYLES[info.kind]
                arrow = None
                if info.kind in (T.HBOND, T.WATER_BRIDGE):
                    arrow, note = hbond_direction(first, ctx.dep)
                    if note:
                        seen["notes"].append(f"{self.names[k]}: {note}")
                if info.kind is T.HBOND:
                    if info.site_kind == "atom" and not is_backbone_atom(grp.resname, info.site_atoms[0]):
                        style = _HBOND_SIDECHAIN
                        seen["sidechain"] = True
                    else:
                        seen["backbone"] = True
                a = (float(a_all[0, idx, 0]), float(a_all[0, idx, 1]))
                b = (float(b_all[0, idx, 0]), float(b_all[0, idx, 1]))
                lg = Group(f"interaction {info.kind.value}", data=data, title=title)
                lg.extend(link_items(a, b, info.kind, m, label=info.label, arrow=arrow, opacity=info.opacity,
                                     style=style, weight=info.weight,
                                     label_at=label_t[idx] if idx < len(label_t) else 0.5))
                if info.target.kind == "ring":
                    dot = (info.target.atoms, style.token)
                    if dot not in dots_done:
                        dots_done.add(dot)
                        lg.add(Circle(info.target.point[0], info.target.point[1], 2.5,
                                      Paint(fill=style.token, opacity=info.opacity)))
                if style.end_dots:
                    lg.add(Circle(a[0], a[1], 2.3, Paint(fill=style.token, opacity=info.opacity)))
                layers["interactions"].append(lg)
                if info.target.kind == "group" and ctx.options.charge_halos:
                    halo = (info.target.atoms, style.token)
                    if halo not in halo_done:
                        halo_done.add(halo)
                        layers["halos"].extend(halo_items(group_skeleton(ctx.dep, info.target.atoms),
                                                          halo_radius(m), style.token))
        seen["shaded"] = {token for _, token in halo_done}

        dep = ctx.dep
        if frame.mode == "core":
            # What is not part of the common substructure is what differs: mark it.
            core = self.core_atoms(k)
            outside = {a.idx for a in dep.atoms if a.name and a.name not in core}
            paint = Paint(stroke="exposure", width=0.62 * m.bond_length, opacity=0.6, cap="round")
            bonded: set[int] = set()
            for bond in dep.bonds:
                if bond.kind == "covalent" and (bond.a in outside or bond.b in outside) \
                        and dep.atoms[bond.a].name and dep.atoms[bond.b].name:
                    pa, pb = dep.pos(bond.a), dep.pos(bond.b)
                    layers["marks"].append(Line(pa[0], pa[1], pb[0], pb[1], paint))
                    bonded |= {bond.a, bond.b}
            for i in sorted(outside - bonded):
                p = dep.pos(i)
                layers["marks"].append(Circle(p[0], p[1], 0.31 * m.bond_length, Paint(fill="exposure", opacity=0.6)))
            seen["marked"] = bool(layers["marks"])
        conf = ctx.confidence
        looks = atom_looks(dep, conf.ligand_atoms) if conf is not None else None
        if looks:
            seen["levels"] |= {lk.level for lk in looks}
        ligand = draw_ligand(dep, palette, m, looks)
        ligand.data["ligand"] = self.names[k]
        content = Group("diagram")
        for name in ("marks", "halos", "interactions"):
            if layers[name]:
                content.add(Group(name, layers[name]))
        content.add(ligand)
        content.add(Group("residues", layers["residues"]))
        return content, seen

    def _panel_header(self, k: int, diff: Difference, m: Metrics, opts: RenderOptions) -> tuple[list[Item], float, float]:
        only = diff.only_a if k == 0 else diff.only_b
        n = len(diff.shared) + len(only)
        size = m.title_font - 0.5
        items: list[Item] = [Text.plain(0.0, CAP_HEIGHT * size, self.names[k], size, "text", anchor="start",
                                        bold=True)]
        width = text_width(self.names[k], size, bold=True)
        y = size * 1.45
        line = (f"{n} contact{'s' if n != 1 else ''} (residue × type): {len(diff.shared)} shared, "
                f"{len(only)} only this ligand")
        items.append(Text.plain(0.0, y + CAP_HEIGHT * m.caption_font, line, m.caption_font, "text_muted",
                                anchor="start"))
        width = max(width, text_width(line, m.caption_font))
        y += m.caption_font * 1.5
        conf = self.diagrams[k].confidence if opts.confidence else None
        shown = self.diagrams[k].burial() if opts.burial else None
        strip = _confidence_strip(conf, m.caption_font, shown) if (conf is not None or shown is not None) else None
        if strip is not None:
            items.append(Group("confidence", [Text(0.0, y + CAP_HEIGHT * strip[1], strip[0], strip[1], "text_muted",
                                                   anchor="start")],
                               data={key: str(v) for key, v in strip[3].items()}))
            width = max(width, strip[2])
            y += strip[1] * 1.5
        return items, width, y

    def _difference_scene(self, frame: _Frame, diff: Difference, opts: RenderOptions, palette: Palette,
                          arrangement: str) -> tuple[Scene, list[str]]:
        m = frame.m
        drawn = [self._difference_panel(frame, k, palette) for k in (0, 1)]
        box = items_bounds([d[0] for d in drawn]) or (0.0, 0.0, 1.0, 1.0)
        cw, ch = box[2] - box[0], box[3] - box[1]
        heads = [self._panel_header(k, diff, m, opts) for k in (0, 1)]
        # One width and one header height for both panels: the frame sits at the same place in each.
        pw = max(cw, heads[0][1], heads[1][1])
        hh = max(heads[0][2], heads[1][2]) + 6.0
        ph = hh + ch
        gap = 28.0 if arrangement == "row" else 20.0
        panels = Group("panels")
        for k in (0, 1):
            content = drawn[k][0]
            content.translate = ((pw - cw) / 2.0 - box[0], hh - box[1])
            panel = Group("panel", [Group("panel-header", heads[k][0]), content],
                          data={"ligand": self.names[k], "panel": "AB"[k]})
            panel.translate = (k * (pw + gap), 0.0) if arrangement == "row" else (0.0, k * (ph + gap))
            panels.add(panel)
        total_w = 2 * pw + gap if arrangement == "row" else pw
        total_h = ph if arrangement == "row" else 2 * ph + gap

        seen = [d[1] for d in drawn]
        classes = [c for c in _CLASS_ORDER if any(c in s["classes"] for s in seen)]
        kinds = [t for t in TYPE_ORDER if any(t in s["kinds"] for s in seen)]
        levels: set[int] = set().union(*(s["levels"] for s in seen))
        states: set[str] = set().union(*(s["states"] for s in seen))
        legend = [LegendEntry(_CLASS_LABEL[c], "swatch", token=f"class_{c}", stroke="bubble_line") for c in classes]
        if "shared" in states:
            legend.append(LegendEntry("contact with both ligands", "node", token="panel", weight=0.9))
        if "only" in states:
            legend.append(LegendEntry("with this ligand only", "node", token="panel", weight=2.5))
        if "absent" in states:
            legend.append(LegendEntry("with the other ligand only", "node", weight=0.8, dash=(1.2, 2.6)))
        if any(s["backbone"] for s in seen):
            legend.append(LegendEntry("H-bond (backbone)", "link", kind=T.HBOND, style=_STYLES[T.HBOND], arrow=True))
        if any(s["sidechain"] for s in seen):
            legend.append(LegendEntry("H-bond (side chain)", "link", kind=T.HBOND, style=_HBOND_SIDECHAIN,
                                      arrow=True))
        legend.extend(link_legend([t for t in kinds if t is not T.HBOND], _STYLES))
        if opts.charge_key:
            legend.extend(charge_legend(set().union(*(s.get("shaded", set()) for s in seen)), _STYLES))
        plus = [t for t in TYPE_ORDER if any(t in s["plus"] for s in seen)]
        if plus:
            legend.append(LegendEntry("+ a contact type only this ligand makes at a shared residue: "
                                      + ", ".join(f"{TYPE_CODES[t]} = {t.label.lower()}" for t in plus)))
        if any(s.get("marked") for s in seen):
            legend.append(LegendEntry("outside the common substructure", "halo", token="exposure"))
        legend.extend(confidence_legend(levels))

        page = _Page(m)
        wrap = max(total_w, self._text_width(m, opts))
        page.text(self._alignment_sentence(frame.mode), wrap, gap=4.0)
        page.add(panels, total_w, total_h, gap=12.0)
        if opts.legend and legend:
            lg, lw, lh = legend_group(legend, m, wrap)
            page.add(lg, lw, lh, gap=10.0)
        if opts.caption:
            foot = (f"Residue positions are identical in both panels. Fingerprint similarity (Tanimoto) "
                    f"{diff.tanimoto:.2f}: {len(diff.shared)} shared contacts, {len(diff.only_a)} only "
                    f"{self.names[0]}, {len(diff.only_b)} only {self.names[1]}. Hydrophobic contacts carry no line.")
            page.text(foot, wrap, cls="footnotes")
            if any(lv > 0 for lv in levels):
                page.text("Stroke weight and opacity follow each model's pLDDT: a contact is drawn at the lower of "
                          f"its ligand-atom and residue values, a residue at its own. {MARK} marks pLDDT below 50.",
                          wrap, cls="footnotes", gap=3.0)
        scene = page.build(opts.title, opts.subtitle, self._banner(opts),
                           opts.title or f"{self.names[0]} vs {self.names[1]}",
                           f"Difference view: {self.names[0]} and {self.names[1]}. " + self._core_sentence())
        return scene, [n for s in seen for n in s["notes"]]

    def _difference_table(self, diff: Difference, m: Metrics, max_w: float) -> tuple[Group, float, float]:
        """Per-residue listing: what each ligand does at every contacted residue."""
        rows = []
        fps = self.fingerprints
        for res, cls in diff.residues().items():
            kinds = [[t for t in TYPE_ORDER if any(b.residue == res and b.type is t for b in fp.bits)] for fp in fps]
            resname = next(b.resname for fp in fps for b in fp.bits if b.residue == res)
            rows.append((f"{resname.capitalize()}{res[1]}{res[2]}", cls, kinds))
        font = m.legend_font
        row_h = font * 1.7
        name_w = max(text_width(r[0], font, bold=True) for r in rows) + 10.0
        col_w = [max([text_width(nm, font * 0.9)] + [_code_spans(r[2][k], sep=" · ")[1] for r in rows]) + 14.0
                 for k, nm in enumerate(diff.names)]
        tag_w = text_width("both", font * 0.9) + 8.0
        block_w = name_w + sum(col_w) + tag_w
        gutter = 26.0
        fit = max(1, int((max_w + gutter) // (block_w + gutter)))
        n_blocks = max(1, min(fit, -(-len(rows) // 12)))
        per = -(-len(rows) // n_blocks)
        g = Group("summary-table")
        head_h = row_h * 1.15
        for blk in range(n_blocks):
            x0 = blk * (block_w + gutter)
            part = rows[blk * per:(blk + 1) * per]
            if not part:
                continue
            g.add(Text.plain(x0, CAP_HEIGHT * font, "Residue", font * 0.9, "text_muted", anchor="start"))
            x = x0 + name_w
            for k, nm in enumerate(diff.names):
                g.add(Text.plain(x, CAP_HEIGHT * font, nm, font * 0.9, "text_muted", anchor="start"))
                x += col_w[k]
            g.add(Line(x0, head_h - 3.0, x0 + block_w, head_h - 3.0, Paint(stroke="panel_border", width=0.8,
                                                                         cap="butt")))
            for r, (label, cls, kinds) in enumerate(part):
                y = head_h + r * row_h
                if r % 2 == 1:
                    g.add(Rect(x0 - 3.0, y, block_w + 6.0, row_h, Paint(fill="panel"), rx=2.0))
                basel = y + row_h / 2.0 + CAP_HEIGHT * font / 2.0
                g.add(Text.plain(x0, basel, label, font, "text", anchor="start", bold=cls == "shared"))
                x = x0 + name_w
                for k in (0, 1):
                    if kinds[k]:
                        spans, _ = _code_spans(kinds[k], sep=" · ")
                        g.add(Text(x, basel, spans, _NOTE, "text_muted", anchor="start"))
                    else:
                        g.add(Text.plain(x, basel, "–", font, "text_muted", anchor="start"))
                    x += col_w[k]
                tag = {"shared": "both", "a": "A", "b": "B"}[cls]
                g.add(Text.plain(x0 + block_w, basel, tag, font * 0.9, "text_muted", anchor="end"))
        return g, n_blocks * block_w + (n_blocks - 1) * gutter, head_h + per * row_h

    # ------------------------------------------------------------------ consensus view
    @staticmethod
    def _levels(rows: Sequence[ConsensusRow], opts: RenderOptions) -> dict[tuple[str, int, str], list[int]] | None:
        """Per residue, how many ligands contact it at confidence level 0 / 1 / 2, as the rows carry
        it (``None``: no confidence, or it is switched off)."""
        if not opts.confidence:
            return None
        out = {r.residue: list(r.levels) for r in rows if r.levels is not None}
        return out or None

    def _render_consensus(self, palette: Palette, base: Metrics, opts: RenderOptions, min_ligands: int,
                          thumbnails: bool) -> tuple:
        all_rows = self.consensus()
        rows = [r for r in all_rows if r.n_ligands >= min_ligands]
        via = self.contact_via()
        levels = self._levels(all_rows, opts)
        notes, degraded = self._member_notes(opts)
        notes.extend(self._two_names())
        apart = self._numbered_otherwise()
        if apart:
            notes.append(self._numbering_note(apart))
            degraded = True
        rep = self.representative
        data: dict[str, Any] = {
            "view": "consensus", "alignment": self.alignment(), "n_ligands": self.n,
            "drawn_ligand": self.names[rep] if rep is not None else None,
            "omitted_residues": len(all_rows) - len(rows), "min_ligands": min_ligands,
            "consensus": [dict(r.as_dict(), via_core=via.get(r.residue, (None, None))[0],
                               via_substituent_only=via.get(r.residue, (None, None))[1],
                               by_confidence=levels.get(r.residue) if levels else None) for r in all_rows],
        }
        title = opts.title or f"Consensus contacts of {self.n} ligands"
        if not rows:
            page = _Page(base)
            head = "No interactions in the supplied sets" if not all_rows else "No residue passes the filter"
            body = (f"All {self.n} ligands were analysed and no contact was found." if not all_rows else
                    f"No residue is contacted by at least {min_ligands} of the {self.n} ligands.")
            page.text(head, 330.0, fill="text", cls="empty-state", size=base.title_font, bold=True)
            page.text(body, 330.0, cls="empty-state")
            return page.build(opts.title, opts.subtitle, self._banner(opts), title), "empty", "empty", None, notes, data

        spatial = rep is not None or all(t is not None for t in self.transforms)
        q: Quality | None = None
        best: tuple[Scene, Quality, _Frame, bool] | None = None
        if not (opts.on_unreadable != "render" and self._hopeless(opts, rows)):
            # Without a scaffold and without a common pocket there is no geometry to draw; the
            # frame is then the sequence, which needs no bond length to try.
            lengths = ([opts.bond_length] if opts.bond_length else list(_BOND_LENGTHS)) if spatial else [0.0]
            frames: dict[float, _Frame] = {}
            done = False
            for thumbs in ((True, False) if (thumbnails and rep is not None) else (False,)):
                for bl in lengths:
                    if bl not in frames:
                        frames[bl] = (self._consensus_frame(_scaled(base, bl), opts, palette, rows, via, levels)
                                      if spatial else self._sequence_frame(base, opts, rows, via, levels))
                    frame = frames[bl]
                    scene = self._consensus_scene(frame, rows, all_rows, opts, palette, thumbs, min_ligands, levels)
                    q = self._quality(scene, frame.m, opts, frame.layout, len(rows),
                                      _THUMB_FONT if thumbs else _NOTE)
                    if best is None or _better(q, best[1]):
                        best = (scene, q, frame, thumbs)
                    if q.valid and q.readable:
                        best = (scene, q, frame, thumbs)
                        done = True
                        break
                if done:
                    break
        if best is not None:
            scene, q, frame, thumbs = best
            data["frame"] = frame.mode
            data["thumbnails"] = thumbs
            if thumbnails and rep is not None and not thumbs and q.valid and q.readable:
                notes.append("the ligand thumbnails were left out to keep the text readable at the figure size")
            if frame.graph is not None:
                data["_graph"] = _on_page(frame.graph, scene)
            if q.valid and q.readable:
                return scene, "degraded" if degraded else "ok", "consensus", q, notes, data
            if not q.valid:
                reason = (f"{q.unplaced} residue(s) could not be placed, {q.overlaps} pair(s) overlap and "
                          f"{q.link_hits} line(s) cross a residue")
            else:
                # (The smallest text of the view as it was drawn: with the ligands small beside it, or without.)
                reason = (beyond_any_page((_THUMB_FONT if thumbs else _NOTE) * _PT_PER_PX, opts, fixed=True)
                          or too_small_on_page(q.min_font_pt, q.required_width_mm, q.width, q.height, opts))
            if opts.on_unreadable == "render":
                message = f"a consensus view of this set is not readable: {reason}; rendered anyway on request"
                return scene, "unreadable", "consensus", q, notes + [message], data
        else:
            reason = beyond_any_page(_NOTE * _PT_PER_PX, opts, fixed=True) or (
                f"{len(rows)} residue nodes cannot be set in one frame at {opts.figure_width_mm:g} mm "
                f"with {opts.min_font_pt:g} pt text")
        message = f"a consensus view of this set is not readable: {reason}"
        data.pop("_graph", None)                       # no node-level figure is handed out
        if opts.on_unreadable == "none":
            return None, "unreadable", "none", q, notes + [message], data
        table, tw, th = self._consensus_table(rows, base, opts.figure_width_mm * _PX_PER_MM - 2.0 * base.margin,
                                              levels, via)
        why = (f"A consensus diagram of this set is not readable at {opts.figure_width_mm:g} mm: {reason}. "
               "Contacts are listed per residue instead.")
        foot = [self._core_sentence(),
                "Bar: share of the ligands that contact the residue. Codes give the interaction types and how "
                "many ligands make each."]
        scene = self._summary_page(base, opts, "Consensus contacts by residue", why, table, tw, th, foot, title)
        return (scene, "unreadable", "summary", q, notes + [message + "; a per-residue summary was drawn instead"],
                data)

    def _hopeless(self, opts: RenderOptions, rows: Sequence[ConsensusRow]) -> bool:
        """Can this many nodes not possibly be read around one frame? A lower bound, no layout run."""
        # A node with its annotation takes about 46 x 60 units; packed without any gap they cover this much.
        area = len(rows) * 46.0 * 60.0
        side = math.sqrt(area * 1.6)
        fit = min(opts.figure_width_mm * _PX_PER_MM / side, 1.0)
        return _NOTE * _PT_PER_PX * fit < opts.min_font_pt - 1e-9

    def _rep_atoms(self, rep: int, key: tuple[str, int, str], ctx: Context) -> list[int]:
        """Where on the drawn ligand a residue belongs: the atoms it contacts, or -- for a residue
        that only other ligands contact -- the drawn ligand's atoms nearest to it in its own structure.
        Empty when the residue cannot be found at all."""
        for grp in ctx.groups:
            if grp.key == key:
                atoms = sorted({i for r in grp.interactions if r.ligand is not None for i in r.ligand.atoms})
                if atoms:
                    return atoms[:4]
        dia = self.diagrams[rep]
        st = dia.structure
        res = st.find_residue(key[0] or None, key[1], key[2]) if st is not None else None
        if st is None or res is None:
            return []
        pts = np.array([a.xyz for a in st.residue_atoms(res) if a.element not in ("H", "D")], dtype=float)
        d = np.linalg.norm(dia.ligand.coords3d[:, None, :] - pts[None, :, :], axis=2).min(axis=1)
        order = sorted(range(len(d)), key=lambda i: (round(float(d[i]), 3), i))
        return [i for i in order if d[i] <= d[order[0]] + 0.75][:3]

    def _consensus_frame(self, m: Metrics, opts: RenderOptions, palette: Palette, rows: Sequence[ConsensusRow],
                         via: dict, levels: dict | None) -> _Frame:
        bl = m.bond_length
        rep = self.representative
        extra: list[tuple[float, float, float]] = []
        info: dict[str, Any] = {}
        if rep is not None:
            coords, mode = self._coords(rep)
            ctx = self._contexts(m, opts, palette, coords)[rep]
            obstacle = ctx.dep
            anchors = {r.residue: self._rep_atoms(rep, r.residue, ctx) for r in rows}
            middle = obstacle.centroid(range(obstacle.n_heavy))
            targets = {res: ([obstacle.pos(i) for i in idx] or [middle]) for res, idx in anchors.items()}
            contexts = [ctx]
            info["coords"] = coords
            info["anchors"] = anchors
        else:
            # No common scaffold: no ligand is drawn. The frame is the pocket itself -- the
            # superposed poses seen in their best-fit plane -- and residues go where they are contacted.
            mode = "envelope"
            scale = bl / 1.5 * 0.85
            clouds = self._clouds()
            plane = pose_frame(clouds)
            pts = [plane.project(c) * scale for c in clouds]
            where: dict[tuple[str, int, str], list[np.ndarray]] = {}
            for k, dia in enumerate(self.diagrams):
                for hit in dia.interactions:
                    if hit.ligand is not None:
                        where.setdefault(_residue_of(hit), []).extend(pts[k][i] for i in hit.ligand.atoms)
            cloud = np.vstack(pts)
            centre = cloud.mean(axis=0)
            targets = {}
            for r in rows:
                hits = where.get(r.residue)
                p = np.mean(hits, axis=0) if hits else centre
                targets[r.residue] = [(float(p[0]), float(p[1]))]
            extra = [(float(x), float(y), 0.62 * bl) for x, y in cloud]
            obstacle = Depiction(atoms=[], bonds=[], rings=[], n_heavy=0, metrics=m)
            contexts = []
            info["points"] = cloud
        with_chain = len({r.chain for r in rows}) > 1
        bodies = []
        for row in rows:
            node, body_r, note = self._consensus_node(row, with_chain, via, levels)
            ts = targets[row.residue]
            ports = [Port((0.0, 0.0), t, ideal=body_r + 0.95 * bl, weight=1.0 / len(ts), drawn=False) for t in ts]
            bodies.append(_node_body(node, body_r, ports, note, float(row.n_ligands), 0.5 * bl))
        layout, fld = self._lay_out(m, bodies, obstacle, extra)
        info["targets"] = targets
        frame = _Frame(m, bodies, layout, contexts, mode, fld, info)
        return self._with_consensus_graph(frame, via)

    def _with_consensus_graph(self, frame: _Frame, via: dict) -> _Frame:
        """Give a consensus frame its graph, and the poses its nodes are then drawn from.

        A consensus node stands for a set of ligands. Its ``occupancy`` is the
        share of them that contact the residue at all; each of its contacts is
        one interaction type, with the share of ligands making it as its
        ``occupancy`` -- the records of :meth:`consensus_contacts`. No contact
        has a ligand end: no ligand's atoms are being claimed.
        """
        m = frame.m
        graph = Graph(view="consensus", bond_length=float(m.bond_length),
                      ensemble=f"the {self.n} ligands of the set",
                      springs=engine_springs(LayoutParams(unit=m.bond_length)))
        part = None
        if frame.mode == "core":
            ctx = frame.contexts[0]
            conf = ctx.confidence
            part = LigandPart(ctx.dep, label=self.names[self.representative or 0],
                              plddt=conf.ligand_atoms if conf is not None else None)
        residues = []
        for body in frame.bodies:
            node: _Node = body.payload
            row = node.row
            assert row is not None
            pose = frame.layout.poses.get(body.key)
            contacts = [Contact(id=f"{node.ident}|{c.type.value}", type=c.type.value, residue="R:" + node.ident,
                                ligand=None, drawn=False, distance=c.distance, occupancy=c.occupancy,
                                confidence=c.confidence) for c in row.contacts]
            graph.contacts.extend(contacts)
            attrs: dict[str, Any] = {"resname": node.resname, "chain": node.key[0], "number": node.key[1],
                                     "class": residue_class(node.resname), "form": "consensus",
                                     "radius": round(node.radius, 3), "n_ligands": row.n_ligands,
                                     "n_total": row.n_total, "ligands": list(row.ligands)}
            if node.key[2]:
                attrs["icode"] = node.key[2]
            if row.levels is not None:
                attrs["by_confidence"] = list(row.levels)
            if via:
                attrs["via_core"], attrs["via_substituent_only"] = via.get(row.residue, (0, 0))
            anchors: tuple[str, ...] = ()
            if part is not None:
                names = [part.dep.atoms[i].name for i in frame.extra["anchors"].get(row.residue, [])]
                anchors = tuple(f"L:{n}" for n in names if n)
            elif frame.mode == "envelope":
                attrs["anchor_point"] = [float(v) for v in frame.extra["targets"][row.residue][0]]
            residues.append(Node(id="R:" + node.ident, kind="residue", label=row.label,
                                 x=None if pose is None else float(pose.x), y=None if pose is None else float(pose.y),
                                 anchors=anchors, spring=float(sum(p.weight for p in body.ports)) or None,
                                 contacts=tuple(c.id for c in contacts), occupancy=row.occupancy, attrs=attrs))
        graph.nodes = (part.nodes() if part is not None else []) + residues
        graph.groups = [part.rigid_group()] if part is not None else []
        graph.meta = {"ligands": list(self.names), "frame": frame.mode,
                      "drawn_ligand": self.names[self.representative] if self.representative is not None else None}
        frame.graph = graph
        frame.poses, _ = poses_from(graph, frame.bodies, tethered=False)
        return frame

    def _consensus_node(self, row: ConsensusRow, with_chain: bool, via: dict, levels: dict | None
                        ) -> tuple[_Node, float, tuple[float, float]]:
        """A consensus node, the radius it occupies with its ring, and the size of its annotation."""
        name, number = _node_text(row.resname, row.residue, with_chain)
        if via and via.get(row.residue, (1, 0))[0] == 0:
            number += _SUBSTITUENT
        radius = _node_radius(name, number) + 3.0 * row.fraction
        node = _Node(row.residue, row.resname, name, number, radius, ring=radius + 4.2, row=row)
        lv = levels.get(row.residue) if levels else None
        count = f"{row.n_ligands}/{self.n}" + (MARK if (lv is not None and lv[0] + lv[1] == 0) else "")
        _, tw = _code_spans([t for t, _ in row.types], counts=dict(row.types), sep=" \u00b7 ")
        note = (max(text_width(count, _FONT, bold=True), tw) + 2.0, CAP_HEIGHT * _FONT + CAP_HEIGHT * _NOTE + 8.0)
        return node, node.ring + 2.4, note

    def _sequence_frame(self, m: Metrics, opts: RenderOptions, rows: Sequence[ConsensusRow], via: dict,
                        levels: dict | None) -> _Frame:
        """The frame of last resort: residues in sequence order.

        Used when the set has neither a common scaffold nor a common pocket to
        look at. Sequence order is a property of the residues themselves, so the
        arrangement is the same for every ligand and claims nothing about where
        in space a residue is. Runs of neighbouring residues are kept together
        and labelled with their range.
        """
        ordered = sorted(rows, key=lambda r: (r.chain, r.number, r.icode))
        with_chain = len({r.chain for r in rows}) > 1
        built = [self._consensus_node(row, with_chain, via, levels) for row in ordered]
        widths = [max(2.0 * body_r, note[0]) + 10.0 for _, body_r, note in built]
        top = max(body_r for _, body_r, _ in built)
        cell_h = top + max(body_r + 2.0 + note[1] for _, body_r, note in built) + 8.0
        max_w = self._text_width(m, opts)
        # Runs: consecutive residues no more than a few positions apart.
        runs: list[list[int]] = []
        for i, row in enumerate(ordered):
            prev = ordered[i - 1] if i else None
            if prev is not None and prev.chain == row.chain and row.number - prev.number <= 6:
                runs[-1].append(i)
            else:
                runs.append([i])
        label_h = 15.0
        run_gap = 18.0
        line_h = label_h + cell_h + 10.0
        poses: dict[Any, Pose] = {}
        marks: list[Item] = []
        x, y = 0.0, 0.0

        def close(part: list[int], x0: float, x1: float, y0: float) -> None:
            """Label a stretch of one run with its residue range."""
            if not part:
                return
            first, last = ordered[part[0]], ordered[part[-1]]
            label = (f"{first.number}{first.icode}" if len(part) == 1 else
                     f"{first.number}{first.icode}\u2013{last.number}{last.icode}")
            if with_chain and first.chain:
                label = f"{first.chain}:{label}"
            marks.append(Line(x0 + 3.0, y0 + label_h - 3.0, x1 - 3.0, y0 + label_h - 3.0,
                              Paint(stroke="panel_border", width=1.0, cap="butt")))
            marks.append(Text.plain(x0 + 3.0, y0 + CAP_HEIGHT * _NOTE, label, _NOTE, "text_muted", anchor="start"))

        for run in runs:
            total = sum(widths[i] for i in run)
            if x > 0.0 and x + total > max_w and total <= max_w:
                x, y = 0.0, y + line_h                       # keep a run on one line when it fits on one
            part: list[int] = []
            x0 = x
            for i in run:
                if part and x + widths[i] > max_w:
                    close(part, x0, x, y)                    # a run longer than the line continues below
                    part, x, y = [], 0.0, y + line_h
                    x0 = 0.0
                poses[built[i][0].key] = Pose(x + widths[i] / 2.0, y + label_h + top + 3.0)
                part.append(i)
                x += widths[i]
            close(part, x0, x, y)
            x += run_gap
        bodies = [_node_body(node, body_r, [], note, float(node.row.n_ligands if node.row else 0), 0.0)
                  for node, body_r, note in built]
        layout = LayoutResult(poses=poses, unplaced=[], cost=0.0, overlaps=0, line_crossings=0, bond_crossings=0,
                              max_stretch=0.0)
        return self._with_consensus_graph(_Frame(m, bodies, layout, [], "sequence", None, {"marks": marks}), via)

    def _consensus_nodes(self, frame: _Frame, levels: dict | None) -> tuple[list[Item], set[int], list[str]]:
        out: list[Item] = []
        seen_levels: set[int] = set()
        classes: list[str] = []
        cap = CAP_HEIGHT * _FONT
        by_level = {lk.level: lk for lk in (LOOKS["confident"], LOOKS["low"], LOOKS["very_low"])}
        for body in frame.bodies:
            pose = frame.poses.get(body.key)
            if pose is None:
                continue
            node: _Node = body.payload
            row = node.row
            assert row is not None
            cls = residue_class(node.resname)
            if cls not in classes:
                classes.append(cls)
            lv = levels.get(row.residue) if levels else None
            counts = lv if lv is not None else [row.n_ligands, 0, 0]
            types = ", ".join(f"{t.label.lower()} ({n})" for t, n in row.types)
            g = Group("residue", data={"residue": node.ident, NUMBER: node.numbered, "class": cls, "form": "consensus",
                                       "n-ligands": str(row.n_ligands), "n-total": str(self.n),
                                       "fraction": f"{row.fraction:.3f}",
                                       "types": ",".join(t.value for t, _ in row.types)},
                      title=f"{row.label}: contacted by {row.n_ligands} of {self.n} ligands; {types}")
            if lv is not None:
                g.data["confident"] = str(lv[0])
            x, y = pose.x, pose.y
            # The ring: a faint full track, and an arc as long as the share of ligands that
            # contact the residue -- heavy where that contact is confidently predicted.
            g.add(Circle(x, y, node.ring, Paint(stroke="panel_border", width=0.8)))
            a0 = -math.pi / 2.0
            for level in (0, 1, 2):
                if not counts[level]:
                    continue
                if lv is not None:
                    seen_levels.add(level)
                look = by_level[level]
                sweep = 2.0 * math.pi * counts[level] / self.n
                g.add(Polyline(_arc(x, y, node.ring, a0, a0 + sweep),
                               Paint(stroke="text", width=3.2 * look.width, opacity=look.opacity, cap="butt")))
                a0 += sweep
            g.add(Circle(x, y, node.radius, Paint(fill=f"class_{cls}", stroke="bubble_line", width=0.9,
                                                 opacity=0.5 + 0.5 * row.fraction)))
            major = row.fraction >= 0.5
            g.add(Text.plain(x, y - 1.1, node.name, _FONT, "bubble_text", bold=major))
            g.add(Text.plain(x, y + cap + 1.1, node.number, _FONT, "bubble_text"))
            count = f"{row.n_ligands}/{self.n}" + (MARK if (lv is not None and lv[0] + lv[1] == 0) else "")
            top = y + node.note_at
            g.add(Text.plain(x, top + cap + 1.5, count, _FONT, "text", bold=major))
            spans, _ = _code_spans([t for t, _ in row.types], counts=dict(row.types), sep=" · ")
            g.add(Text(x, top + cap + CAP_HEIGHT * _NOTE + 6.0, spans, _NOTE, "text_muted"))
            out.append(g)
        return out, seen_levels, classes

    def _envelope(self, frame: _Frame) -> list[Item]:
        """The footprint of the superposed poses: where the ligands are, without drawing any one of them."""
        fld = frame.fld
        assert fld is not None
        loops = fld.contours(1.0)
        out: list[Item] = []
        if loops:
            sign = polygon_area(loops[0]) >= 0
            for loop in loops:
                if (polygon_area(loop) >= 0) != sign or len(loop) < 12:
                    continue
                pts = resample_closed(loop, 4.0)
                for _ in range(3):
                    pts = (np.roll(pts, 1, axis=0) + pts + np.roll(pts, -1, axis=0)) / 3.0
                out.append(Polyline(tuple((float(px), float(py)) for px, py in pts),
                                    Paint(fill="exposure", stroke="panel_border", width=1.0, opacity=0.45,
                                          dash=(3.0, 2.4)), closed=True))
        for px, py in frame.extra["points"]:
            out.append(Circle(float(px), float(py), 1.25, Paint(fill="text_muted", opacity=0.4)))
        return out

    def _thumbnails(self, coords: list[dict], palette: Palette, max_w: float) -> tuple[Group, float, float]:
        """Every ligand of an aligned set, small, on the common core: the core faint, what differs in ink."""
        ms = Metrics().with_(bond_length=11.5, bond_width=1.0, double_offset=2.2, wedge_width=3.0, hash_spacing=1.6,
                             label_gap=1.1, atom_font=8.0, sub_font=_THUMB_FONT, charge_font=_THUMB_FONT)
        deps = [_frame_depiction(d.ligand, ms, xy) for d, xy in zip(self.diagrams, coords)]
        boxes = [dep.bounds() for dep in deps]
        box = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        font = 9.0
        cell_w = max(box[2] - box[0], 96.0)
        captions = [wrap_text(nm, font, cell_w, bold=True)[:2] for nm in self.names]
        cap_h = max(len(c) for c in captions) * font * 1.3 + font * 1.3
        cell_h = box[3] - box[1] + 5.0 + cap_h
        gap = 14.0
        per_row = max(1, int((max_w + gap) // (cell_w + gap)))
        per_row = min(per_row, -(-self.n // max(1, -(-self.n // per_row))))        # balance the rows
        g = Group("thumbnails")
        for k, dep in enumerate(deps):
            core = self.core_atoms(k)
            looks = [PLAIN if (a.name and a.name not in core) else _FADED for a in dep.atoms]
            lig = draw_ligand(dep, palette, ms, looks)
            lig.translate = ((cell_w - (box[2] - box[0])) / 2.0 - box[0], -box[1])
            y = box[3] - box[1] + 5.0
            items: list[Item] = [lig]
            for line in captions[k]:
                items.append(Text.plain(cell_w / 2.0, y + CAP_HEIGHT * font, line, font, "text", bold=True))
                y += font * 1.3
            n_bits = len(self.fingerprints[k])
            items.append(Text.plain(cell_w / 2.0, y + CAP_HEIGHT * font, f"{n_bits} contact{'s' if n_bits != 1 else ''}",
                                    font, "text_muted"))
            cell = Group("thumbnail", items, data={"ligand": self.names[k]})
            cell.translate = ((k % per_row) * (cell_w + gap), (k // per_row) * (cell_h + 10.0))
            g.add(cell)
        n_rows = -(-self.n // per_row)
        return g, per_row * cell_w + (per_row - 1) * gap, n_rows * cell_h + (n_rows - 1) * 10.0

    def _consensus_scene(self, frame: _Frame, rows: Sequence[ConsensusRow], all_rows: Sequence[ConsensusRow],
                         opts: RenderOptions, palette: Palette, thumbs: bool, min_ligands: int,
                         levels: dict | None) -> Scene:
        m = frame.m
        rep = self.representative
        nodes, seen_levels, classes = self._consensus_nodes(frame, levels)
        content = Group("diagram")
        if frame.mode == "envelope":
            content.add(Group("envelope", self._envelope(frame)))
        elif frame.mode == "sequence":
            content.add(Group("sequence-frame", list(frame.extra["marks"])))
        else:
            assert rep is not None
            ctx = frame.contexts[0]
            conf = ctx.confidence
            looks = atom_looks(ctx.dep, conf.ligand_atoms) if conf is not None else None
            if looks:
                seen_levels |= {lk.level for lk in looks}
            ligand = draw_ligand(ctx.dep, palette, m, looks)
            ligand.data["ligand"] = self.names[rep]
            content.add(ligand)
        content.add(Group("residues", nodes))
        box = items_bounds([content]) or (0.0, 0.0, 1.0, 1.0)
        cw, ch = box[2] - box[0], box[3] - box[1]
        content.translate = (-box[0], -box[1])

        page = _Page(m)
        wrap = max(cw, self._text_width(m, opts))
        if frame.mode == "envelope":
            more = (f" No ligand is drawn: the shaded area is the footprint of the {self.n} superposed poses, and "
                    "each residue stands where it is contacted.")
        elif frame.mode == "sequence":
            more = (" " + self._superposition_sentence() + " No ligand is drawn, and the residues are set out in "
                    "sequence order: their place says where they are in the chain, not in the pocket.")
        else:
            assert rep is not None
            more = (f" Drawn: {self.names[rep]}, the member closest to that substructure; each residue stands by "
                    "the atoms that are nearest to it.")
        page.text(self._alignment_sentence(frame.mode, more), wrap, gap=4.0)
        models = self._models_line(opts, m.caption_font + 1.0)
        if models is not None:
            page.add(*models, gap=6.0)
        page.add(Group("panel", [content], data={"view": "consensus"}), cw, ch, gap=10.0)
        if thumbs:
            # As wide as the page may get before the thumbnails' smallest text drops under the minimum.
            room = (opts.figure_width_mm * _PX_PER_MM * _THUMB_FONT * _PT_PER_PX / opts.min_font_pt
                    - 2.0 * m.margin)
            tg, tw, th = self._thumbnails(frame.extra["coords"], palette, max(wrap, room))
            page.text(f"The {self.n} ligands, each on the common substructure (drawn faint); what differs is in ink.",
                      wrap, gap=12.0)
            page.add(tg, tw, th, gap=6.0)
        else:
            page.text(f"Ligands ({self.n}): " + ", ".join(self.names) + ".", wrap, gap=10.0)

        if opts.legend:
            legend = [LegendEntry(_CLASS_LABEL[c], "swatch", token=f"class_{c}", stroke="bubble_line")
                      for c in _CLASS_ORDER if c in classes]
            kinds = [t for t in TYPE_ORDER if any(t is kt for r in rows for kt, _ in r.types)]
            legend.extend(LegendEntry(f"{TYPE_CODES[t]} = {t.label.lower()}") for t in kinds)
            if any(n.payload.number.endswith(_SUBSTITUENT) for n in frame.bodies):
                legend.append(LegendEntry(f"{_SUBSTITUENT} contacted only through atoms outside the common "
                                          "substructure"))
            legend.extend(confidence_legend(seen_levels))
            lg, lw, lh = legend_group(legend, m, wrap)
            page.add(lg, lw, lh, gap=10.0)
        if opts.caption:
            foot = (f"Ring: share of the {self.n} ligands that contact the residue; n/{self.n} below it, then the "
                    "interaction types with the number of ligands making each. No per-ligand lines are drawn.")
            page.text(foot, wrap, cls="footnotes")
            if levels is not None:
                page.text("A ring segment is heavy where the contact is confidently predicted (pLDDT ≥ 70, the "
                          f"lower of ligand-atom and residue value) and lighter below; {MARK} after a count means "
                          "no ligand's contact with that residue reaches pLDDT 50.", wrap, cls="footnotes", gap=3.0)
            omitted = len(all_rows) - len(rows)
            if omitted:
                page.text(f"{omitted} residue{'s' if omitted != 1 else ''} contacted by fewer than {min_ligands} "
                          f"ligands {'are' if omitted != 1 else 'is'} not shown.", wrap, cls="footnotes", gap=3.0)
        return page.build(opts.title, opts.subtitle, self._banner(opts),
                          opts.title or f"Consensus contacts of {self.n} ligands",
                          f"Consensus view of {self.n} ligands. " + self._core_sentence())

    def _consensus_table(self, rows: Sequence[ConsensusRow], m: Metrics, max_w: float, levels: dict | None,
                         via: dict) -> tuple[Group, float, float]:
        font = m.legend_font
        row_h = font * 1.7
        bar_w = 64.0
        name_w = max(text_width(r.label + _SUBSTITUENT, font, bold=True) for r in rows) + 10.0
        count_w = text_width(f"{self.n}/{self.n}{MARK}", font, bold=True) + 10.0
        type_w = max(_code_spans([t for t, _ in r.types], counts=dict(r.types), sep=" · ")[1] for r in rows) + 6.0
        block_w = name_w + count_w + bar_w + 12.0 + type_w
        gutter = 28.0
        fit = max(1, int((max_w + gutter) // (block_w + gutter)))
        n_blocks = max(1, min(fit, -(-len(rows) // 12)))
        per = -(-len(rows) // n_blocks)
        by_level = {lk.level: lk for lk in (LOOKS["confident"], LOOKS["low"], LOOKS["very_low"])}
        g = Group("summary-table")
        head_h = row_h * 1.15
        for blk in range(n_blocks):
            x0 = blk * (block_w + gutter)
            part = rows[blk * per:(blk + 1) * per]
            if not part:
                continue
            g.add(Text.plain(x0, CAP_HEIGHT * font, "Residue", font * 0.9, "text_muted", anchor="start"))
            g.add(Text.plain(x0 + name_w, CAP_HEIGHT * font, "ligands", font * 0.9, "text_muted", anchor="start"))
            g.add(Text.plain(x0 + name_w + count_w + bar_w + 12.0, CAP_HEIGHT * font, "types", font * 0.9,
                             "text_muted", anchor="start"))
            g.add(Line(x0, head_h - 3.0, x0 + block_w, head_h - 3.0, Paint(stroke="panel_border", width=0.8,
                                                                         cap="butt")))
            for r, row in enumerate(part):
                y = head_h + r * row_h
                if r % 2 == 1:
                    g.add(Rect(x0 - 3.0, y, block_w + 6.0, row_h, Paint(fill="panel"), rx=2.0))
                basel = y + row_h / 2.0 + CAP_HEIGHT * font / 2.0
                lv = levels.get(row.residue) if levels else None
                only_sub = via.get(row.residue, (1, 0))[0] == 0 if via else False
                rg = Group("residue", data={"residue": f"{row.chain}:{row.resname}{row.number}{row.icode}",
                                            NUMBER: f"{row.number}{row.icode}",
                                            "n-ligands": str(row.n_ligands), "n-total": str(self.n),
                                            "form": "row"})
                rg.add(Text.plain(x0, basel, row.label + (_SUBSTITUENT if only_sub else ""), font, "text",
                                  anchor="start", bold=row.fraction >= 0.5))
                count = f"{row.n_ligands}/{self.n}" + (MARK if (lv is not None and lv[0] + lv[1] == 0) else "")
                rg.add(Text.plain(x0 + name_w, basel, count, font, "text", anchor="start", bold=row.fraction >= 0.5))
                bx = x0 + name_w + count_w
                cy = y + row_h / 2.0
                rg.add(Line(bx, cy, bx + bar_w, cy, Paint(stroke="panel_border", width=0.8, cap="butt")))
                counts = lv if lv is not None else [row.n_ligands, 0, 0]
                x = bx
                for level in (0, 1, 2):
                    if counts[level]:
                        w = bar_w * counts[level] / self.n
                        look = by_level[level]
                        rg.add(Line(x, cy, x + w, cy, Paint(stroke="text", width=5.0 * look.width,
                                                            opacity=look.opacity, cap="butt")))
                        x += w
                spans, _ = _code_spans([t for t, _ in row.types], counts=dict(row.types), sep=" · ")
                rg.add(Text(bx + bar_w + 12.0, basel, spans, _NOTE, "text_muted", anchor="start"))
                g.add(rg)
        return g, n_blocks * block_w + (n_blocks - 1) * gutter, head_h + per * row_h
