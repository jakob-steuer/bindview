"""An interface: the contacts between two polymer participants, drawn with a focus.

A ligand diagram has a centre, the ligand, and the residues stand around it.
An interface has none. What this module draws instead has a **spine**: the
residues of one participant that touch the other, as a column of residue boxes
in the order of their sequence, with a dotted line wherever residues in
between are left out. The residues of the other participant are drawn around
that column with their contacts, at the resolution of the style: with the
atoms that make the contacts, or as bubbles. Resolution is therefore mixed,
and the figure says which participant is drawn how.

A column orders a line, which is the right picture of a binder that is one
stretch of its chain and the wrong one of three loops that lie side by side.
Where the residues of the spine come in several stretches they can therefore
be set out on a **plane**, each box where its residue lies in space
(:mod:`bindview.plane`), provided that picture is faithful, which is measured
and written in the figure. That is asked for (``layout="auto"`` lets the rule
of :meth:`Interface._verdict` choose, ``layout="plane"`` names it); left alone,
an interface is drawn as a column.

An interface is an ordered list of participants with contacts between
neighbours in that order. Two participants are the plain case. Three are the
same object: in ``MHC - peptide - receptor`` the middle one touches both and is
the spine.

A whole interface is rarely drawn. A figure holds about thirty residues of the
spine; with more, or when the whole does not lay out, it narrows to a **site**,
chosen by a stated rule (:func:`choose_site`), and says which site and how
much of the interface that is.

Nothing here detects contacts. They are supplied
(:class:`InterfaceContact`, :func:`load_interface_contacts`) or come from PLIP's
peptide mode behind :func:`bindview.detect.detect_interface`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from operator import index as whole_number
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .errors import BindviewError, InteractionError
from .exposure import WATER_NAMES
from .fragments import check_detail
from .interactions import Interaction, InteractionType, check_confidence_scale
from .pae import (IPSAE_CUTOFF, PAE_SHOWN, PAE_STRONG, Ipsae, Pae, PaeDraws, ipsae_of, pae_shown, pae_tag,
                  read_pae)
from .plane import (APART, ERROR_MAX, KEPT_MIN, MIN_RESIDUES, NEAR, STRETCH_GAP, Arrangement, Fidelity, Placed,
                    stretches)
from .structure import ResidueKey, Structure, read_structure

__all__ = ["Participant", "InterfaceContact", "Site", "Interface", "InterfaceIpsae", "choose_site",
           "load_interface_contacts", "dump_interface_contacts", "MAX_SPINE", "LAYOUTS"]

#: How the residues of the spine are set out: in a ``column`` (the default), on a ``plane`` (each where
#: it lies in space), or as the rule decides (``auto``; :meth:`Interface._verdict`, in words in
#: ``Interface.layout_rule``).
LAYOUTS = ("auto", "column", "plane")

#: Residues of the spine a figure shows at most. The default page takes 54 boxes and nothing around
#: them.
MAX_SPINE = 30
#: The carbon and the nitrogen of a peptide bond, or an oxygen and the phosphorus after it, are this near at most (ångström).
_BOND = 2.0

COLUMNS = ("interaction_type", "chain_a", "residue_number_a", "insertion_code_a", "residue_name_a", "atoms_a",
           "chain_b", "residue_number_b", "insertion_code_b", "residue_name_b", "atoms_b", "distance", "angle",
           "a_is_donor", "confidence", "pae", "water")


@dataclass(frozen=True)
class Participant:
    """One side of an interface: a name, and the chains it consists of.

    ``detail`` is how much of its residues is drawn where they are drawn with
    atoms (:data:`bindview.fragments.DETAILS`): ``"box"``, ``"backbone"``,
    ``"sidechain"`` or ``"atoms"``. ``None`` leaves it to the figure
    (``Interface(detail=...)``). Where that names no level either, the column
    is drawn as boxes and the residues around it as the style has them.
    """

    name: str
    chains: tuple[str, ...]
    detail: "str | None" = None

    def __post_init__(self) -> None:
        try:
            check_detail(self.detail, f"the detail of {self.name or 'a participant'}")
        except ValueError as exc:
            raise BindviewError(str(exc)) from None

    @classmethod
    def parse(cls, spec: "str | Participant | Sequence[str]") -> "Participant":
        """``"A"``, ``"D+E"`` or ``"receptor=D+E"``; or a sequence of chain names."""
        if isinstance(spec, Participant):
            return spec
        if isinstance(spec, str):
            name, _, chains = spec.rpartition("=")
            parts = tuple(c.strip() for c in chains.replace(",", "+").split("+") if c.strip())
        else:
            name, parts = "", tuple(str(c) for c in spec)
        if not parts:
            raise BindviewError(f"a participant needs at least one chain: {spec!r}")
        return cls(name.strip() or ("chain " + parts[0] if len(parts) == 1 else "chains " + ", ".join(parts)), parts)


@dataclass(frozen=True)
class InterfaceContact:
    """One contact across an interface, with the residue and the atoms on both sides.

    ``confidence`` (pLDDT scale, 0 to 100) and ``pae`` (predicted aligned error
    between the two residues, ångström) ride on the contact into the graph.
    ``pae`` is what the reliability of a contact between two chains means in a
    predicted complex, and is written on the contact's line. Where it is the
    median over several draws of the model, ``pae_draws`` are the values of
    the draws.
    """

    type: InteractionType
    chain_a: str
    number_a: int
    resname_a: str
    atoms_a: tuple[str, ...]
    chain_b: str
    number_b: int
    resname_b: str
    atoms_b: tuple[str, ...]
    icode_a: str = ""
    icode_b: str = ""
    distance: float | None = None
    angle: float | None = None
    a_is_donor: bool | None = None
    confidence: float | None = None
    pae: float | None = None
    water: str | None = None
    pae_draws: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", InteractionType(self.type) if not isinstance(self.type, InteractionType)
                           else self.type)
        object.__setattr__(self, "atoms_a", tuple(self.atoms_a))
        object.__setattr__(self, "atoms_b", tuple(self.atoms_b))
        object.__setattr__(self, "pae_draws", tuple(float(v) for v in self.pae_draws))
        for value in (self.pae, *self.pae_draws):
            if value is not None and not 0.0 <= float(value) < float("inf"):
                raise InteractionError(f"the contact of {self.residue_a} with {self.residue_b} has a PAE of "
                                       f"{float(value):g}: a PAE is a number of ångström from 0")

    @property
    def residue_a(self) -> ResidueKey:
        """``(chain, number, insertion code)`` on the first side."""
        return ResidueKey(self.chain_a, int(self.number_a), self.icode_a)

    @property
    def residue_b(self) -> ResidueKey:
        """``(chain, number, insertion code)`` on the second side."""
        return ResidueKey(self.chain_b, int(self.number_b), self.icode_b)

    @property
    def sort_key(self) -> tuple:
        """The order of contacts in a table."""
        return (self.type.value, self.residue_a, self.atoms_a, self.residue_b, self.atoms_b, self.distance or 0.0)

    def flipped(self) -> "InterfaceContact":
        """The same contact seen from the other side."""
        return replace(self, chain_a=self.chain_b, number_a=self.number_b, icode_a=self.icode_b,
                       resname_a=self.resname_b, atoms_a=self.atoms_b, chain_b=self.chain_a, number_b=self.number_a,
                       icode_b=self.icode_a, resname_b=self.resname_a, atoms_b=self.atoms_a,
                       a_is_donor=None if self.a_is_donor is None else not self.a_is_donor)

    def as_interaction(self) -> Interaction:
        """The contact as a record of a ligand diagram whose ligand is side ``a`` (the spine): its atoms
        are named as in a chain ligand, ``63/OE1``."""
        mine = tuple(f"{self.number_a}{self.icode_a}/{n}" for n in self.atoms_a)
        one = len(mine) == 1 and len(self.atoms_b) == 1
        return Interaction(
            type=self.type, residue_number=self.number_b, residue_name=self.resname_b, chain=self.chain_b,
            insertion_code=self.icode_b, protein_atom=self.atoms_b[0] if one else None,
            ligand_atom=mine[0] if one else None, ligand_atoms=() if one else mine,
            protein_atoms=() if one else self.atoms_b, distance=self.distance, angle=self.angle,
            ligand_is_donor=self.a_is_donor, water=self.water, confidence=self.confidence, pae=self.pae,
            pae_draws=self.pae_draws)


def dump_interface_contacts(contacts: Iterable[InterfaceContact]) -> str:
    """The contacts as a tab-separated table (:data:`COLUMNS`; and ``pae_draws``, the value of each
    draw behind a median, where any contact has them)."""
    def cell(value: Any) -> str:
        return "" if value is None else ",".join(str(v) for v in value) if isinstance(value, tuple) else str(value)

    contacts = list(contacts)
    draws = any(c.pae_draws for c in contacts)
    lines = ["\t".join(COLUMNS + (("pae_draws",) if draws else ()))]
    for c in contacts:
        lines.append("\t".join(cell(v) for v in (
            c.type.value, c.chain_a, c.number_a, c.icode_a, c.resname_a, c.atoms_a, c.chain_b, c.number_b, c.icode_b,
            c.resname_b, c.atoms_b, c.distance, c.angle, c.a_is_donor, c.confidence, c.pae, c.water)
            + ((c.pae_draws,) if draws else ())))
    return "\n".join(lines) + "\n"


def load_interface_contacts(path: "str | Path") -> list[InterfaceContact]:
    """Read a table written by :func:`dump_interface_contacts` (or by hand, with those columns). Empty
    lines and lines that begin with ``#`` are passed over; an error names the line of the file."""
    text = Path(path).read_text(encoding="utf-8-sig")            # (-sig: without a byte order mark, if it has one)
    rows = [(n, line.split("\t")) for n, line in enumerate(text.splitlines(), start=1)
            if line.strip() and not line.startswith("#")]
    if not rows:
        return []
    head = [h.strip() for h in rows[0][1]]
    need = {"interaction_type", "chain_a", "residue_number_a", "residue_name_a", "chain_b", "residue_number_b",
            "residue_name_b"}
    if not need <= set(head):
        raise BindviewError(f"{path}: an interface table needs the columns {', '.join(sorted(need))}")
    out = []
    for n, row in rows[1:]:
        d = dict(zip(head, (cell.strip() for cell in row)))

        def number(key: str) -> "float | None":
            return float(d[key]) if d.get(key) else None

        try:
            out.append(InterfaceContact(
                type=InteractionType(d["interaction_type"]), chain_a=d["chain_a"], number_a=int(d["residue_number_a"]),
                icode_a=d.get("insertion_code_a", ""), resname_a=d["residue_name_a"],
                atoms_a=tuple(a for a in d.get("atoms_a", "").split(",") if a), chain_b=d["chain_b"],
                number_b=int(d["residue_number_b"]), icode_b=d.get("insertion_code_b", ""),
                resname_b=d["residue_name_b"], atoms_b=tuple(a for a in d.get("atoms_b", "").split(",") if a),
                distance=number("distance"), angle=number("angle"),
                a_is_donor={"true": True, "false": False}.get(d.get("a_is_donor", "").lower()),
                confidence=number("confidence"), pae=number("pae"), water=d.get("water") or None,
                pae_draws=tuple(float(v) for v in d.get("pae_draws", "").split(",") if v.strip())))
        except KeyError as exc:
            raise BindviewError(f"{path}, line {n}: the row ends before the column {exc}") from exc
        except (ValueError, BindviewError) as exc:
            raise BindviewError(f"{path}, line {n}: {exc}") from exc
    return out


@dataclass(frozen=True)
class Site:
    """The part of an interface a figure shows, and how it was chosen."""

    kind: str                           # whole | seed | cluster | window
    residues: tuple[ResidueKey, ...]    # of the spine, in the order of its sequence
    of_residues: int                    # residues of the spine at the interface in all
    contacts: int
    of_contacts: int
    rule: str                           # in words, for the figure

    @property
    def whole(self) -> bool:
        return self.kind == "whole"


def _span(residues: Sequence[ResidueKey]) -> str:
    first, last = residues[0], residues[-1]
    return (f"{first.number}{first.icode}" if len(residues) == 1
            else f"{first.number}{first.icode} to {last.number}{last.icode}")


def _joined(structure: Structure, a: ResidueKey, b: ResidueKey) -> bool:
    """Is residue ``b`` bonded to ``a`` before it: by a peptide bond, or by the phosphate of a nucleotide?"""
    for before, after in (("C", "N"), ("O3'", "P"), ("O3*", "P")):
        one, other = structure.atom_by_name(a, before), structure.atom_by_name(b, after)
        if one is not None and other is not None:
            return float(np.linalg.norm(np.array(one.xyz) - np.array(other.xyz))) <= _BOND
    return False


def _places(structure: Structure, keys: Sequence[ResidueKey]) -> dict[ResidueKey, int]:
    """Where each residue of a chain stands in it, for what is said and drawn of stretches and neighbours.

    One after the other in the order of the file, and as far apart as their numbers where the file leaves
    residues out between two: a loop that was not modelled lies between its two ends in the figure as it
    does in the chain. Numbers that jump between two residues bonded to each other (the numbering of
    another protein, or one without a zero) leave nothing out."""
    places: dict[ResidueKey, int] = {}
    at = 0
    for k, key in enumerate(keys):
        if k:
            step = abs(key.number - keys[k - 1].number)
            at += 1 if step <= 1 or _joined(structure, keys[k - 1], key) else step
        places[key] = at
    return places


def choose_site(contacts: Sequence[InterfaceContact], order: Sequence[ResidueKey], index: "dict[ResidueKey, int]",
                *, limit: int = MAX_SPINE, seed: "ResidueKey | None" = None, radius: float = 8.0,
                structure: "Structure | None" = None) -> Site:
    """Which residues of the spine a figure shows. Deterministic, and in words in the result.

    ``contacts`` have the spine as side ``a``; ``order`` are the residues of the
    spine at the interface in the order of its sequence, ``index`` the place of
    every residue of the spine in its chain.

    * With a ``seed`` residue (of either side): the residues of the spine at
      the interface that have an atom within ``radius`` ångström of it; the
      nearest ``limit`` if there are more.
    * Otherwise the whole interface, if it has no more than ``limit`` residues
      of the spine.
    * Otherwise whole contact clusters, those with the most contacts first, as
      many as fit: two residues of the spine are in one cluster if they are
      neighbours in the sequence or touch the same residue of the partner.
    * If that is still too large: its ``limit`` consecutive residues (in the
      order of the spine) that carry the most contacts.
    """
    if limit < 1:
        raise BindviewError(f"a site shows 1 residue or more, not {limit!r}")
    per: dict[ResidueKey, int] = {k: 0 for k in order}
    for c in contacts:
        per[c.residue_a] += 1
    total = len(contacts)

    def made(kind: str, residues: Sequence[ResidueKey], rule: str) -> Site:
        chosen = tuple(k for k in order if k in set(residues))
        return Site(kind, chosen, len(order), sum(per[k] for k in chosen), total, rule)

    if seed is not None:
        if structure is None:
            raise BindviewError("a site around a residue needs the structure")
        centre = np.array([a.xyz for a in structure.residue_atoms(seed) if a.element not in ("H", "D")], dtype=float)
        if not len(centre):
            raise BindviewError(f"no residue {seed} to centre the site on")
        near: list[tuple[float, int, ResidueKey]] = []
        for n, k in enumerate(order):
            pts = np.array([a.xyz for a in structure.residue_atoms(k) if a.element not in ("H", "D")], dtype=float)
            d = float(np.linalg.norm(pts[:, None, :] - centre[None, :, :], axis=2).min())
            if d <= radius:
                near.append((d, n, k))
        if not near:
            raise BindviewError(f"no residue of the spine at the interface lies within {radius:g} Å of {seed}")
        picked = [k for _, _, k in sorted(near)[:limit]]
        return made("seed", picked, f"within {radius:g} Å of {structure.residue_name(seed).title()}"
                                    f"{seed.number}{seed.icode} of chain {seed.chain}")
    if len(order) <= limit:
        return made("whole", order, "the whole interface")
    parent = {k: k for k in order}

    def find(k: ResidueKey) -> ResidueKey:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for a, b in zip(order, order[1:]):
        if a.chain == b.chain and index[b] - index[a] == 1:
            parent[find(b)] = find(a)
    by_partner: dict[ResidueKey, ResidueKey] = {}
    for c in contacts:
        first = by_partner.setdefault(c.residue_b, c.residue_a)
        if find(first) != find(c.residue_a):
            parent[find(c.residue_a)] = find(first)
    clusters: dict[ResidueKey, list[ResidueKey]] = {}
    for k in order:
        clusters.setdefault(find(k), []).append(k)
    ranked = sorted(clusters.values(), key=lambda ks: (-sum(per[k] for k in ks), order.index(ks[0])))
    best = ranked[0]
    if len(best) <= limit:
        # The clusters with the most contacts, as many as fit. Not the largest alone: of an interface
        # in many small clusters that is a small part (8 of 33 residues where they come in 13 clusters).
        taken: list[ResidueKey] = []
        n = 0
        for cluster in ranked:
            if len(taken) + len(cluster) <= limit:
                taken.extend(cluster)
                n += 1
        return made("cluster", taken, f"the {n} of its {len(ranked)} contact clusters with the most contacts "
                                      f"that fit a figure")
    sums = [sum(per[k] for k in best[i:i + limit]) for i in range(len(best) - limit + 1)]
    start = sums.index(max(sums))
    window = best[start:start + limit]
    return made("window", window, f"the {limit} consecutive residues of the interface with the most contacts, "
                                  f"residues {_span(window)}")


def _site_meta(site: Site) -> dict[str, Any]:
    """A site as the graph of its figure holds it (``meta.interface.site``)."""
    return {"kind": site.kind, "rule": site.rule, "residues": len(site.residues),
            "of_residues": site.of_residues, "contacts": site.contacts, "of_contacts": site.of_contacts}


@dataclass
class _Info:
    """What the figure says about the interface: handed to the chain backend."""

    legend: str
    notes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    #: Further lines of the legend, and further numbers for the line under the title (label, value).
    legend_extra: list[str] = field(default_factory=list)
    #: Which of the notes are of the PAE as it is written on the lines: not those of a figure that draws it as a glow.
    tag_notes: list[str] = field(default_factory=list)
    strip: list[tuple[str, str]] = field(default_factory=list)
    #: ... and the same numbers as ``data-`` attributes of that line, for a page that reads the figure.
    strip_data: dict[str, str] = field(default_factory=dict)
    #: Said in the result as well as in the figure.
    warnings: list[str] = field(default_factory=list)
    #: What still has to be said when the figure could not be drawn and a table stands in its place.
    summary_notes: list[str] = field(default_factory=list)
    #: What the boxes are called as a whole, for the line under the title: a column, or boxes on a plane.
    shape: str = "column"

    def said(self, dep: Any = None) -> list[str]:
        """The footnotes of the figure as it was drawn: with boxes on a plane, what was measured of
        them where they stand goes in after the sentence that says they are on a plane."""
        placed = getattr(dep, "plane", None)
        if placed is None:
            return list(self.notes)
        return self.notes[:1] + [_fidelity_sentence(placed)] + self.notes[1:]

    def described(self, dep: Any = None) -> str:
        """One sentence on boxes that stand on a plane, for the description of the figure; empty otherwise."""
        placed = getattr(dep, "plane", None)
        if placed is None:
            return ""
        f = placed.fidelity
        return (f"The boxes stand where their residues lie on the plane through them: {_percent(f.kept)} of "
                f"neighbour relations kept, distances off by {_percent(f.error)}.")

    def meta_for(self, dep: Any = None) -> dict[str, Any]:
        """The graph's ``meta.interface`` for the figure as it was drawn."""
        out = dict(self.meta)
        placed = getattr(dep, "plane", None)
        if placed is not None:
            out["plane"] = {"drawn": True, **placed.as_dict()}
        return out


def _percent(value: float) -> str:
    return f"{100.0 * value:.0f} %"


def _shortfall(bare: Fidelity) -> str:
    """Why a projection is not drawn: its two numbers against what is required."""
    return (f"a projection would keep {_percent(bare.kept)} of neighbour relations and be off by "
            f"{_percent(bare.error)} in distances; at least {_percent(KEPT_MIN)} and at most "
            f"{_percent(ERROR_MAX)} are required")


def _fidelity_sentence(placed: Placed) -> str:
    """What was measured of the boxes on the plane, in words for the figure."""
    f = placed.fidelity
    if f.pairs:
        wrong = []
        if f.false:
            wrong.append(f"{f.false} {'is' if f.false == 1 else 'are'} drawn as neighbours and "
                         f"{'lies' if f.false == 1 else 'lie'} more than {APART:g} \u00c5 apart in space")
        if f.lost:
            wrong.append(f"{f.lost} {'is' if f.lost == 1 else 'are'} neighbours in space and drawn more than "
                         f"{APART:g} \u00c5 apart")
        pairs = (f"{f.pairs} pairs of boxes are neighbours (centres within {NEAR:g} \u00c5) in the figure or in "
                 f"space" if f.pairs != 1 else
                 f"One pair of boxes are neighbours (centres within {NEAR:g} \u00c5) in the figure or in space")
        said = (f"{pairs}; " + ("; ".join(wrong) if wrong else "none is a neighbour in the one and more than "
                                                                f"{APART:g} \u00c5 apart in the other")
                + f": {_percent(f.kept)} of neighbour relations are kept.")
    else:
        said = f"No two boxes are neighbours (centres within {NEAR:g} \u00c5), in the figure or in space."
    said += (f" Distances between the centres of boxes differ from those in space by {_percent(f.error)} "
             f"(root mean square). The residues' centres lie {placed.rms:.1f} \u00c5 (root mean square) off the plane")
    if placed.moved:
        said += (f"; {placed.moved} box{'es were' if placed.moved != 1 else ' was'} moved off "
                 f"{'their places' if placed.moved != 1 else 'its place'} to keep boxes apart")
    return "Measured on the boxes as drawn: " + said[0].lower() + said[1:] + "."


@dataclass(frozen=True)
class InterfaceIpsae:
    """ipSAE between the participant of the column and one of the others (:func:`bindview.pae.ipsae`).

    ``first`` is the participant of the column. ``draws`` holds the score of one matrix, or of every
    draw of the model in the order the draws were given; :attr:`forward` and :attr:`backward` are then
    the medians over the draws, as the PAE of a contact is.
    """

    first: str
    second: str
    draws: tuple[Ipsae, ...]

    @property
    def forward(self) -> float:
        """Aligned on a residue of ``first``, scored over ``second``."""
        return float(np.median([d.forward.value for d in self.draws]))

    @property
    def backward(self) -> float:
        """Aligned on a residue of ``second``, scored over ``first``."""
        return float(np.median([d.backward.value for d in self.draws]))

    @property
    def cutoff(self) -> float:
        return self.draws[0].cutoff


@dataclass
class _Verdict:
    """What a site is drawn as, and why: the outcome of :meth:`Interface._verdict`."""

    layout: str
    rule: str
    runs: "list[tuple[int, int]]"
    arrangement: "Arrangement | None" = None
    considered: "Arrangement | None" = None
    #: A column that stands in for a plane: in the order of the sequence, whatever spreads contacts least.
    sequence_only: bool = False


class Interface:
    """The contacts between two (or three) polymer participants as one figure.

    ``participants`` in order; contacts are between neighbours. ``contacts``
    are :class:`InterfaceContact` records in either orientation, or ``None``
    to detect them with PLIP (:func:`bindview.detect.detect_interface`).
    ``spine`` names the participant drawn as the column of boxes (default: of
    three the middle one, which touches both others; of two the one that is a
    single chain where the other is several, else the one with fewer residues
    at the interface, and the first where they have as many). ``site`` is a
    residue to centre a site on, as ``"A:84"``.
    ``layout`` is how the boxes are set out: ``"column"`` (the default),
    ``"plane"`` (each where its residue lies in space), or ``"auto"``, which
    lets the rule of :meth:`_verdict` choose between the two. What a figure
    is drawn as is :attr:`layout`, and why, in words, :attr:`layout_rule`,
    which for the default says what the rule would have chosen.
    ``pae`` is the model's predicted aligned error (:mod:`bindview.pae`): one
    file for the model that is drawn, or several for the draws of one model,
    whose median a contact then carries.
    ``ipsae`` asks for a score of the interface as a whole, computed from that
    matrix (:func:`bindview.pae.ipsae`): ``True`` for the usual cut-off of 10 Å,
    or the cut-off in ångström as a number with a point (``15.0``; a whole
    number is refused, since ``1`` could mean either). The figure then prints
    it under its title, in its two directions, and says what it is;
    :attr:`ipsae` holds it. Left out, nothing is computed.
    ``model_confidence`` is what a model says of itself as a whole (ipTM and
    the like), as for a ligand diagram: a mapping, a
    :class:`~bindview.confidence.ModelConfidence`, the path of a JSON file that
    has those keys, or ``"auto"`` (the default), which reads AlphaFold 3's
    summary file where it lies beside the model.
    ``detail`` is how much of a residue is drawn where it makes a directional
    contact: ``"box"``, ``"backbone"``, ``"sidechain"`` or ``"atoms"``, for
    every participant that names no level of its own, or a mapping from the
    name of a participant to its level. Left alone, the column is drawn as
    boxes and the residues around it as the style has them. Atoms stand beside
    the boxes of a column only, not of a plane.
    """

    def __init__(self, structure: "Structure | str | Path", participants: Sequence["str | Participant | Sequence[str]"],
                 contacts: "Iterable[InterfaceContact] | None" = None, *, spine: "str | None" = None,
                 site: "str | ResidueKey | None" = None, radius: float = 8.0, max_residues: int = MAX_SPINE,
                 name: "str | None" = None, order: str = "auto",
                 pae: "str | Path | Pae | PaeDraws | Sequence[str | Path | Pae] | None" = None,
                 layout: str = "column", detail: "str | Mapping[str, str | None] | None" = None,
                 ipsae: "bool | float" = False, model_confidence: Any = "auto") -> None:
        if order not in ("auto", "sequence", "interface"):
            raise BindviewError(f"order is 'auto', 'sequence' or 'interface', not {order!r}")
        if layout not in LAYOUTS:
            raise BindviewError(f"layout is {', '.join(repr(v) for v in LAYOUTS[:-1])} or {LAYOUTS[-1]!r}, "
                              f"not {layout!r}")
        try:
            most = 0 if isinstance(max_residues, bool) else whole_number(max_residues)
        except TypeError:
            most = 0
        if most < 1:
            raise BindviewError(f"max_residues is a whole number of residues, 1 or more, not {max_residues!r}")
        self.order = order
        #: What was asked for; :attr:`layout` is what a figure of the site is drawn as.
        self.layout_asked = layout
        self.layout = "column"
        self.layout_rule = ""
        self._arrangement: "Arrangement | None" = None
        self._considered: "Arrangement | None" = None
        self._runs: list[tuple[int, int]] = []
        self._sequence_only = False
        self._no_plane: dict[tuple[ResidueKey, ...], str] = {}
        self._columns: dict[tuple[ResidueKey, ...], Any] = {}
        self.structure = structure if isinstance(structure, Structure) else read_structure(structure)
        self.participants = tuple(Participant.parse(p) for p in participants)
        if not 2 <= len(self.participants) <= 3:
            raise BindviewError("an interface is drawn between two participants, or along three in a row; "
                              f"{len(self.participants)} were named")
        chains = [c for p in self.participants for c in p.chains]
        if len(set(chains)) != len(chains):
            raise BindviewError("a chain can be in one participant only")
        #: How much of a residue of each participant is drawn where it makes a directional contact,
        #: by the participant's name: its own level, else the figure's, else ``None`` (the column as
        #: boxes, the residues around it as the style has them).
        self.detail = self._details(detail)
        present = {a.chain for a in self.structure.atoms}
        missing = [c for c in chains if c not in present]
        if missing:
            raise BindviewError(f"no chain {', '.join(missing)} in {self.structure.source or 'the structure'}")
        self.source = "supplied"
        if contacts is None:
            from .detect import detect_interface

            contacts = [c for a, b in zip(self.participants, self.participants[1:])
                        for c in detect_interface(self.structure, a.chains, b.chains)]
            self.source = "detected"
        self.notes: list[str] = []
        contacts = list(contacts)
        # Before anything is drawn, and of the whole interface: a site would show the mistake in part.
        check_confidence_scale((c.confidence for c in contacts), "The contacts of this interface")
        #: The model's predicted aligned error, where there is one: AlphaFold 3's confidences file, the
        #: ``pae_*.npz`` of a Boltz model (its rows are those of the structure), or a
        #: :class:`~bindview.pae.Pae`; several of them are the draws of one model, and a pair's value is
        #: then their median (:class:`~bindview.pae.PaeDraws`). A contact that brings no value of its own
        #: gets the one of its two residues.
        self.pae = read_pae(pae, self.structure)
        #: The pairs of residues with a contact whose PAE was read from the matrix and did not come with it.
        self._from_matrix: set[frozenset[ResidueKey]] = set()
        if self.pae is not None:
            given = contacts
            contacts = [c if c.pae is not None else self._with_pae(c) for c in given]
            self._from_matrix = {frozenset((c.residue_a, c.residue_b)) for c, was in zip(contacts, given)
                                 if was.pae is None and c.pae is not None}
            if contacts and all(c.pae is None for c in contacts):
                # A figure without a word about PAE would read as a confident one, not as an unmeasured one.
                names = sorted(set(self.pae.chains))
                raise BindviewError(
                    f"{self.pae.source or 'the PAE matrix'} has no value for any contact of this interface: the "
                    f"residues in contact are not in it (it has chain{'s' if len(names) != 1 else ''} "
                    f"{', '.join(repr(n) for n in names)}, residue numbers {min(self.pae.numbers)} to "
                    f"{max(self.pae.numbers)})")
        self.contacts = self._between_neighbours(list(contacts))
        if not self.contacts:
            raise BindviewError("no contact between the participants: there is no interface to draw")
        self.spine = self._spine(spine)
        if len(self.spine.chains) != 1:
            raise BindviewError(f"the spine has to be one chain; {self.spine.name} has {len(self.spine.chains)}. "
                              "Name the other participant as the spine")
        if layout == "plane" and self._beside:
            raise BindviewError(
                f"{self.spine.name} is asked for with the atoms of its residues (detail "
                f"{self.detail[self.spine.name]!r}) and with its boxes where they lie on a plane (layout 'plane'). "
                f"Atoms are drawn beside the boxes of a column only: a column has a side of every box that is "
                f"nobody's, and boxes that stand where their residues lie have none. Ask for one of the two")
        self.name = name
        self.radius = radius
        self.max_residues = max_residues
        self.seed = self._seed(site)
        self._oriented = [c if c.chain_a in self.spine.chains else c.flipped() for c in self.contacts]
        chain = self.spine.chains[0]
        keys = [k for k in self.structure.residues() if k.chain == chain
                and self.structure.residue_name(k).upper() not in WATER_NAMES]
        self._index = _places(self.structure, keys)
        touched = {c.residue_a for c in self._oriented}
        unknown = sorted(touched - set(keys))
        if unknown:
            raise BindviewError(f"a contact names residue {unknown[0]}, which is not in the structure")
        self._order = [k for k in keys if k in touched]
        #: What the model says of itself as a whole, as it was handed in (see the class).
        self.model_confidence = model_confidence
        #: ipSAE between the participant of the column and each of the others, where it was asked for.
        self.ipsae: list[InterfaceIpsae] = self._scores(ipsae)
        #: The site and the diagram of the figure that was drawn last; before any, those the interface
        #: was built with, which every figure starts from (:meth:`render`).
        self.site = choose_site(self._oriented, self._order, self._index, limit=max_residues, seed=self.seed,
                                radius=radius, structure=self.structure)
        self.diagram = self._diagram(self.site)
        self._built = self.site
        self._diagrams: dict[Site, Any] = {self.site: self.diagram}

    def _scores(self, asked: "bool | float") -> "list[InterfaceIpsae]":
        """ipSAE of the participant of the column with each of the others, from the PAE that was given."""
        if asked is False or asked is None:
            return []
        if isinstance(asked, int) and not isinstance(asked, bool):
            # 1 is True almost everywhere, and would be a cut-off of 1 Å here: neither is guessed.
            raise BindviewError(f"ipsae={asked!r}: ipsae is True, for the usual cut-off of {IPSAE_CUTOFF:g} Å, or the "
                                f"cut-off in ångström as a number with a point ({float(asked)!r})")
        if self.pae is None:
            raise BindviewError("ipSAE is computed from the model's PAE matrix, and none was given: hand it in "
                              "with pae=")
        cutoff = IPSAE_CUTOFF if asked is True else float(asked)

        def residues(p: Participant) -> list[ResidueKey]:
            return [k for k in self.structure.residues()
                    if k.chain in p.chains and self.structure.residue_name(k).upper() not in WATER_NAMES]

        return [InterfaceIpsae(self.spine.name, other.name,
                               ipsae_of(self.pae, residues(self.spine), residues(other), cutoff,
                                        structure=self.structure))
                for other in self.others]

    def _details(self, detail: "str | Mapping[str, str | None] | None") -> dict[str, "str | None"]:
        """The level of every participant: its own, or the one the figure was given for it."""
        names = [p.name for p in self.participants]
        try:
            if detail is None or isinstance(detail, str):
                given = dict.fromkeys(names, check_detail(detail))
            else:
                unknown = [k for k in detail if k not in names]
                if unknown:
                    raise ValueError(f"detail names {unknown[0]!r}, which is not a participant of this interface "
                                     f"({', '.join(repr(n) for n in names)})")
                given = {n: check_detail(detail.get(n), f"the detail of {n}") for n in names}
        except ValueError as exc:
            raise BindviewError(str(exc)) from None
        return {p.name: p.detail if p.detail is not None else given[p.name] for p in self.participants}

    @property
    def _beside(self) -> bool:
        """Are atoms of the column's residues to be drawn beside their boxes?"""
        return self.detail[self.spine.name] not in (None, "box")

    def _with_pae(self, c: InterfaceContact) -> InterfaceContact:
        """A contact with the PAE of its two residues: of one matrix, or the median over the draws
        with the value of every draw beside it."""
        assert self.pae is not None
        if isinstance(self.pae, PaeDraws):
            each = self.pae.each(c.residue_a, c.residue_b)
            known = tuple(v for v in each if v is not None)
            # A value for every draw, or none: a median beside four of five values would not be theirs.
            return replace(c, pae=self.pae.between(c.residue_a, c.residue_b),
                           pae_draws=known if len(known) == len(each) else ())
        return replace(c, pae=self.pae.between(c.residue_a, c.residue_b))

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_structure(cls, structure: "Structure | str | Path",
                       participants: Sequence["str | Participant | Sequence[str]"], **options: Any) -> "Interface":
        """The same as the class itself, under the name the other figures are built by
        (``Diagram.from_structure``, ``Ensemble.from_structures``)."""
        return cls(structure, participants, **options)

    def _between_neighbours(self, contacts: list[InterfaceContact]) -> list[InterfaceContact]:
        owner = {c: i for i, p in enumerate(self.participants) for c in p.chains}
        out, elsewhere = [], 0
        for c in contacts:
            i, j = owner.get(c.chain_a), owner.get(c.chain_b)
            if i is None or j is None or abs(i - j) != 1:
                elsewhere += 1
                continue
            out.append(c if i < j else c.flipped())
        if elsewhere:
            self.notes.append(f"{elsewhere} contact{'s' if elsewhere != 1 else ''} not between neighbouring "
                              f"participants {'were' if elsewhere != 1 else 'was'} left out")
        return sorted(out, key=lambda c: c.sort_key)

    def _spine(self, named: "str | None") -> Participant:
        if named is not None:
            for n, p in enumerate(self.participants):
                if named == p.name or named in p.chains or named == "+".join(p.chains):
                    if len(self.participants) == 3 and n != 1:
                        # The contacts are between neighbours: one at an end has none with the other end.
                        middle, far = self.participants[1], self.participants[2 - n]
                        raise BindviewError(
                            f"the spine of three participants in a row is the middle one, which touches both "
                            f"others: {p.name} is at an end and does not touch {far.name}. Name {p.name} in the "
                            f"middle to draw the other two around it, or draw it with {middle.name} alone")
                    return p
            raise BindviewError(f"no participant {named!r} to be the spine")
        if len(self.participants) == 3:
            return self.participants[1]
        a, b = self.participants
        n_a = len({c.residue_a for c in self.contacts})
        n_b = len({c.residue_b for c in self.contacts})
        single = [p for p in (a, b) if len(p.chains) == 1]
        if len(single) == 1:
            return single[0]
        return a if n_a <= n_b else b

    def _seed(self, site: "str | ResidueKey | None") -> "ResidueKey | None":
        if site is None or isinstance(site, ResidueKey):
            return site
        chain, _, rest = str(site).partition(":")
        digits = rest.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        try:
            key = self.structure.find_residue(chain, int(digits), rest[len(digits):])
        except ValueError:
            key = None
        if key is None:
            raise BindviewError(f"no residue {site!r} to centre the site on (write it as chain:number, 'A:84')")
        return key

    def _diagram(self, site: Site) -> Any:
        from .backends import Polymer
        from .diagram import Diagram

        self._decide(site)
        shown = set(site.residues)
        records = [c.as_interaction() for c in self._oriented if c.residue_a in shown]
        # The column's level is the chain's; the levels of the participants around it go by their chains.
        level = {"ligand": self.detail[self.spine.name], "protein": None}
        around = {c: self.detail[p.name] for p in self.others for c in p.chains if self.detail[p.name] is not None}
        if self.layout == "plane":
            # On the plane the boxes are not in any order of their own: the residues as the chain has them.
            st, key, given, notes = self.structure.with_residues_as_ligand(list(site.residues))
            dia = Diagram._assemble(st, key, given, list(notes) + list(self.notes), None, records,
                                    name=self.name or self.title, ligand_as="chain", source=self.source,
                                    detail=level, model_confidence=self.model_confidence)
            dia.detail_by_chain = dict(around)
            assert isinstance(dia.backend, Polymer)
            drawn = [r.number for r in dia.backend.peptide.residues]
            if drawn == [f"{k.number}{k.icode}" for k in site.residues]:
                dia.backend.interface = self._info(site)
                dia.backend.arrangement = self._arrangement
                dia.backend.peptide = replace(dia.backend.peptide, pairwise=True, sequential=True,
                                              called=self.spine.name, open_ends=(False, False))
                return dia
            # A residue without a backbone is no box of its own: there is then no place for every box.
            self._no_plane[tuple(site.residues)] = ("not every residue of them has a backbone of its own, "
                                                    "so not every one can be given its place")
            self._decide(site)
        st, key, given, notes = self.structure.with_residues_as_ligand(self._arranged(site.residues))
        dia = Diagram._assemble(st, key, given, list(notes) + list(self.notes), None, records,
                                name=self.name or self.title, ligand_as="chain", source=self.source,
                                detail=level, model_confidence=self.model_confidence)
        dia.detail_by_chain = dict(around)
        assert isinstance(dia.backend, Polymer)
        dia.backend.interface = self._info(site)
        # The column starts and ends where the interface does, which is seldom where the chain does.
        chain_keys = sorted(self._index, key=lambda k: self._index[k])
        dia.backend.peptide = replace(dia.backend.peptide, pairwise=True, sequential=not self._along(site.residues),
                                      called=self.spine.name,
                                      open_ends=(site.residues[0] != chain_keys[0],
                                                 site.residues[-1] != chain_keys[-1]))
        return dia

    # ------------------------------------------------------------------ column or plane
    def _centre(self, key: ResidueKey) -> np.ndarray:
        """The centre of a residue: the centroid of its heavy atoms."""
        return np.array([a.xyz for a in self.structure.residue_atoms(key) if a.element not in ("H", "D")],
                        dtype=float).mean(axis=0)

    def _viewer(self, residues: Sequence[ResidueKey]) -> tuple[Participant, np.ndarray]:
        """From whose side the plane is seen, and the direction to them: the participant with the
        most pairs of residues in contact with those that are drawn (the first named, on a tie)."""
        shown = set(residues)
        owner = {c: p for p in self.others for c in p.chains}
        pairs: dict[str, dict[tuple[ResidueKey, ResidueKey], None]] = {p.name: {} for p in self.others}
        for c in self._oriented:
            if c.residue_a in shown:
                pairs[owner[c.chain_b].name][(c.residue_a, c.residue_b)] = None
        who = max(self.others, key=lambda p: len(pairs[p.name]))
        ways = []
        for a, b in pairs[who.name]:
            v = self._centre(b) - self._centre(a)
            length = float(np.linalg.norm(v))
            if length > 1e-6:
                ways.append(v / length)
        return who, (np.mean(ways, axis=0) if ways else np.zeros(3))

    def _project(self, residues: Sequence[ResidueKey], runs: Sequence[tuple[int, int]]) -> Arrangement:
        """The residues on the plane through their centres, seen from the partner's side."""
        centres = np.array([self._centre(k) for k in residues], dtype=float)
        _, toward = self._viewer(residues)
        joins = [k + 1 for first, last in runs for k in range(first, last)
                 if self._index[residues[k + 1]] - self._index[residues[k]] > 1]
        return Arrangement.of(centres, toward, joins)

    def _spans(self, residues: Sequence[ResidueKey], runs: Sequence[tuple[int, int]]) -> str:
        return ", ".join(_span(residues[first:last + 1]) for first, last in runs)

    def _decide(self, site: Site) -> None:
        """Column or plane for this site: sets :attr:`layout` and, in words, :attr:`layout_rule`."""
        said = self._verdict(site, self.layout_asked)
        self._column_for_atoms = False
        if said.layout == "plane" and self._beside:
            # Left to the rule, these residues would stand on a plane. Their atoms were asked for, and
            # atoms stand beside the boxes of a column: the column it is, and the figure says why.
            said = _Verdict("column", f"a column, because the atoms of its residues are drawn beside their boxes "
                                      f"(detail {self.detail[self.spine.name]!r}) and boxes on a plane have no "
                                      f"side for them; the rule alone would draw these residues {said.rule}",
                            said.runs, considered=said.considered)
            self._column_for_atoms = True
        self.layout, self.layout_rule, self._runs = said.layout, said.rule, said.runs
        self._arrangement, self._considered, self._sequence_only = said.arrangement, said.considered, said.sequence_only

    def _verdict(self, site: Site, asked: str) -> "_Verdict":
        """What a site is drawn as when ``asked`` for a column, a plane, or the rule (``auto``).

        **The column is the default**, and nothing about it depends on the rule: an interface is
        drawn as a column unless the caller asks otherwise. What the rule would have chosen is said
        in the result all the same.

        The rule, where it is asked for:

        * A **column** when fewer than three residues are shown, and when the residues that are
          shown are one stretch of the chain (no more than :data:`~bindview.plane.STRETCH_GAP` residues
          that are not at the interface between one and the next): a contiguous binder reads best as
          the list it is.
        * Otherwise the **plane**: each box where its residue lies on the least-squares plane through
          their centres -- if the projection is faithful (:class:`~bindview.plane.Fidelity`).
        * If it is not (the interface is too curved for one plane) the column again, in the order of
          the sequence, and the figure says that the arrangement is not shown.

        Asked for outright, the plane is drawn for one stretch as well, and still not for a
        projection that is not faithful.
        """
        residues = list(site.residues)
        runs = stretches([self._index[k] for k in residues])
        spine, n = self.spine.name, len(residues)
        some = f"{len(runs)} stretches" if len(runs) != 1 else "one stretch"
        if asked == "column":
            rule = "a column, the default"
            try:
                would = self._verdict(site, "auto").layout
            except (ValueError, np.linalg.LinAlgError):
                would = "column"         # the column is drawn whatever the rule trips over
            if would == "plane":
                rule += (f'; asked to choose (layout "auto"), the rule would draw these residues on the plane '
                         f"through them: they come in {some} of the chain")
            return _Verdict("column", rule, runs)
        if n < MIN_RESIDUES:
            return _Verdict("column", f"a column: {'one residue does' if n == 1 else f'{n} residues do'} not make "
                                      f"a plane", runs)
        if asked == "auto" and len(runs) == 1:
            return _Verdict("column", f"a column: the residues of {spine} at the interface that are shown are one "
                                      f"stretch of its chain ({self._spans(residues, runs)})", runs)
        why, they = ((f"the residues of {spine} at the interface that are shown come in {some} of its chain", "they")
                     if asked == "auto" else ("the plane was asked for", "the residues"))
        failed = self._no_plane.get(tuple(residues))
        if failed is not None:
            return _Verdict("column", f"a column in the order of the sequence: {why}, but {failed}", runs,
                            sequence_only=True)
        arrangement = self._project(residues, runs)
        if not arrangement.bare.ok:
            return _Verdict("column", f"a column in the order of the sequence: {why}, but {they} do not lie near "
                                      f"enough to one plane to be drawn where they lie "
                                      f"({_shortfall(arrangement.bare)})", runs, considered=arrangement,
                            sequence_only=True)
        return _Verdict("plane", f"on the plane through them: {why}", runs, arrangement=arrangement,
                        considered=arrangement)

    def reach(self, column: Sequence[ResidueKey]) -> int:
        """How far the contacts of the residues around a column are spread along it, summed over those
        residues (in boxes). It is what makes lines long, makes them cross, and makes a residue be
        drawn twice; of two orders of the column the one with the smaller reach is the better drawing."""
        place = {k: i for i, k in enumerate(column)}
        spread: dict[ResidueKey, tuple[int, int]] = {}
        for c in self._oriented:
            if c.residue_a in place:
                lo, hi = spread.get(c.residue_b, (place[c.residue_a], place[c.residue_a]))
                spread[c.residue_b] = (min(lo, place[c.residue_a]), max(hi, place[c.residue_a]))
        return sum(hi - lo for lo, hi in spread.values())

    def _along(self, residues: Sequence[ResidueKey]) -> bool:
        """Is the column ordered along the interface (and not by the sequence)? By default where that
        shortens the reach of the residues around it; where it does not, the sequence is kept."""
        if len(residues) < 3 or self.order == "sequence":
            return False
        if self.order == "interface":
            return True
        if self._sequence_only:
            # Several stretches that could not be drawn where they lie: no order that promises
            # neighbours in space either.
            return False
        return self.reach(self._by_position(residues)) < self.reach(residues)

    def _arranged(self, residues: Sequence[ResidueKey]) -> list[ResidueKey]:
        """The residues of the spine in the order of the column."""
        return self._by_position(residues) if self._along(residues) else list(residues)

    def _by_position(self, residues: Sequence[ResidueKey]) -> list[ResidueKey]:
        """The residues along the long axis of the interface: the first principal axis of their alpha carbons."""
        centres = []
        for k in residues:
            atoms = self.structure.residue_atoms(k)
            ca = [a.xyz for a in atoms if a.name == "CA"] or [a.xyz for a in atoms]
            centres.append(ca[0])
        xyz = np.array(centres, dtype=float)
        _, _, axes = np.linalg.svd(xyz - xyz.mean(axis=0), full_matrices=False)
        along = (xyz - xyz.mean(axis=0)) @ axes[0]
        if along[0] > along[-1]:
            along = -along                 # the residue that comes first in the sequence nearer the top
        return [k for _, _, k in sorted(zip(along.tolist(), range(len(residues)), residues))]

    # ------------------------------------------------------------------ what the figure says
    @property
    def others(self) -> tuple[Participant, ...]:
        """The participants that stand around the spine."""
        return tuple(p for p in self.participants if p is not self.spine)

    @property
    def title(self) -> str:
        """What the figure is called unless it is given a title."""
        return " – ".join(p.name for p in self.participants)

    def _info(self, site: Site) -> _Info:
        spine, others = self.spine, " and ".join(p.name for p in self.others)
        plane = self.layout == "plane"
        if plane:
            assert self._arrangement is not None
            viewer, _ = self._viewer(site.residues)
            tilt = self._arrangement.tilt
            seen = (f"seen from the side of {viewer.name}" if tilt is None or tilt <= 45.0 else
                    f"seen at right angles to that plane ({viewer.name} lies {tilt:.0f}\u00b0 off that line of sight)")
            some = f"{len(self._runs)} stretches" if len(self._runs) != 1 else "one stretch"
            notes = [f"An interface has no centre. The residues of {spine.name} that touch {others} come in {some} "
                     f"of its chain: each box stands where its residue's centre lies on the plane through those "
                     f"centres, at one scale, {seen}; a line joins neighbours in the chain, a dotted line two with up "
                     f"to {'three' if STRETCH_GAP == 3 else STRETCH_GAP} residues between them. Those of {others} "
                     f"are not drawn where they lie but beside the boxes they touch. A line to one of them is a "
                     f"pair of residues, at the shortest of its contacts of one kind."]
        else:
            if self._along(site.residues):
                how = ("as boxes, ordered along the interface and not by their sequence: neighbours in the column "
                       "are neighbours in space")
            else:
                how = ("as boxes, in the order of the sequence; a dotted line stands for residues in between that "
                       "do not")
            notes = [f"An interface has no centre. The residues of {spine.name} that touch {others} stand in a "
                     f"column, {how}; those of {others} around it. A line is a pair of residues, at the shortest of "
                     f"its contacts of one kind."]
        unshown: "str | None" = None
        if not plane and self._sequence_only:
            # Several stretches, and no plane for them: the column is a list, and has to say so.
            some = f"{len(self._runs)} stretches" if len(self._runs) != 1 else "one stretch"
            failed = self._no_plane.get(tuple(site.residues))
            reason = (failed if failed is not None else
                      f"they do not lie near enough to one plane ({_shortfall(self._considered.bare)})"
                      if self._considered is not None else "that was not possible")
            if len(self._runs) > 1:
                unshown = (f"These residues come in {some} of the chain, which no column can set out as they lie in "
                           f"space. They are not drawn where they lie either: {reason}. How the stretches lie to "
                           f"each other is not shown.")
            else:
                unshown = (f"These residues were to be drawn where they lie, and are not: {reason}. The column is "
                           f"their order in the chain and nothing more.")
            notes.append(unshown)
        directional = any(c.type is not InteractionType.HYDROPHOBIC for c in self._oriented
                          if c.residue_a in set(site.residues))
        if self._beside and not plane and directional:
            # Boxes with atoms beside them: what the atoms are, and what a line from one of them is.
            more = {"backbone": ", and with them its backbone nitrogen and carbonyl",
                    "sidechain": ", and with them the whole side chain",
                    "atoms": ", and with them every other atom of the residue"}[self.detail[spine.name] or "atoms"]
            notes.append(f"Where a residue of {spine.name} makes a contact that has a direction (an H-bond, a salt "
                         f"bridge and the like), the atoms that make it are drawn beside its box{more}. The box "
                         f"is the residue's alpha carbon. A line that ends on such an atom is the contact of that "
                         f"atom and gives its distance; a line that ends on a box stands for the pair of residues.")
        if site.whole:
            notes.append(f"All {site.of_residues} residues of {spine.name} at the interface are shown, with "
                         f"{site.of_contacts} contacts.")
        else:
            notes.append(f"A site, not the whole interface: {len(site.residues)} of the {site.of_residues} residues "
                         f"of {spine.name} at the interface ({site.rule}), with {site.contacts} of "
                         f"{site.of_contacts} contacts.")
        meta = {"participants": [{"name": p.name, "chains": list(p.chains)} for p in self.participants],
                "spine": spine.name,
                "spine_order": None if plane else "interface" if self._along(site.residues) else "sequence",
                "contacts_from": self.source,
                "site": _site_meta(site),
                "layout": self.layout, "layout_rule": self.layout_rule, "stretches": len(self._runs)}
        if any(v is not None for v in self.detail.values()):
            # Said only where a level was asked for: the graph of a figure without one has no such key.
            meta["detail"] = {p.name: self.detail[p.name] for p in self.participants}
        if self._considered is not None and not plane:
            # A plane that was looked at and not drawn: what it would have been, and against what.
            considered = self._considered
            meta["plane"] = {"drawn": False, **considered.bare.as_dict(),
                             "out_of_plane_rms": round(considered.rms, 2),
                             "thresholds": {"neighbours_kept_min": KEPT_MIN, "distance_error_max": ERROR_MAX,
                                            "neighbour_within": NEAR, "apart_beyond": APART}}
        info = _Info(f"Residue of {spine.name}, numbered as in the structure file", notes, meta,
                     shape="plane" if plane else "column")
        if unshown is not None:
            info.summary_notes.append(unshown)
        shown = [c for c in self._oriented if c.residue_a in set(site.residues)]
        errors = [c.pae for c in shown if c.pae is not None]
        if errors:
            # A PAE is a number of two residues, and so is a line of the figure: counted by pair.
            pairs: dict[tuple[ResidueKey, ResidueKey], float | None] = {}
            for c in shown:
                key = (c.residue_a, c.residue_b)
                known = pairs.get(key)
                pairs[key] = c.pae if known is None else known if c.pae is None else min(known, c.pae)
            # Draws are said of a figure all of whose values are medians of the same number of draws.
            counts = {len(c.pae_draws) for c in shown if c.pae is not None}
            draws = counts.pop() if len(counts) == 1 else 0
            by_draw = [[c.pae_draws[k] for c in shown if c.pae is not None] for k in range(draws if draws > 1 else 0)]
            read = any(frozenset((c.residue_a, c.residue_b)) in self._from_matrix for c in shown)
            self._say_pae(info, errors, len(shown), [v for v in pairs.values() if v is not None], len(pairs),
                          from_matrix=read, by_draw=by_draw,
                          draws=self.pae if isinstance(self.pae, PaeDraws) else None)
        if self.ipsae:
            self._say_ipsae(info, self.ipsae)
        return info

    @staticmethod
    def _say_ipsae(info: _Info, scores: "Sequence[InterfaceIpsae]") -> None:
        """What a figure says of the ipSAE that was asked for: the numbers under the title, in both
        directions, and in a footnote what they are. No verdict: there is no threshold of bindview's."""
        n_draws = len(scores[0].draws)
        cutoff = scores[0].cutoff
        # Short, because the line under the title does not wrap: which value is which is in the footnote.
        info.strip.append(("ipSAE", " \u00b7 ".join(f"{s.forward:.2f} / {s.backward:.2f}" for s in scores)))
        info.strip_data["ipsae"] = " ".join(f"{s.forward:.2f} {s.backward:.2f}" for s in scores)
        info.strip_data["ipsae-cutoff"] = f"{cutoff:g}"
        first = scores[0].first
        said = (f"ipSAE (Dunbrack 2025) is a score from 0 to 1 of how well the model places two chains against "
                f"each other. It is worked out from the PAE of every pair of their residues that is below "
                f"{cutoff:g} \u00c5, not from the contacts shown here. It has a direction: ")
        if len(scores) == 1:
            said += (f"the first value is with the model aligned on a residue of {first} and scored over "
                     f"{scores[0].second}, the second the reverse.")
        else:
            pairs = " and ".join(f"{first} with {s.second}" for s in scores)
            said += (f"of each pair of values the first is with the model aligned on a residue of {first} and "
                     f"scored over the other participant, the second the reverse. The pairs are {pairs}, in "
                     f"that order.")
        if n_draws > 1:
            each = "; ".join(", ".join(f"{d.forward.value:.2f} / {d.backward.value:.2f}" for d in s.draws)
                             for s in scores)
            said += f" Each value is the median of {n_draws} draws of the model (draw by draw: {each})."
        if any(d.same_chain for s in scores for d in s.draws):
            said += (" Both sides are parts of one chain of this model. The score was defined for two chains, and "
                     "residues that follow each other in a chain have a low PAE whatever its domains do.")
        left_out = max(d.left_out for s in scores for d in s.draws)
        if left_out:
            said += (f" {left_out} residue{'s' if left_out != 1 else ''} of the two sides "
                     f"{'have' if left_out != 1 else 'has'} no alpha carbon and "
                     f"{'are' if left_out != 1 else 'is'} not in it.")
        info.notes.append(said)
        info.summary_notes.append(said)
        info.meta["ipsae"] = {
            "cutoff": cutoff, "d0": "from the residues below the cut-off, for each aligned residue",
            "scores": [{"aligned_on": a, "scored_over": b, "value": round(v, 3),
                        **({"by_draw": [round(x, 3) for x in each_draw]} if n_draws > 1 else {})}
                       for s in scores
                       for a, b, v, each_draw in (
                           (s.first, s.second, s.forward, [d.forward.value for d in s.draws]),
                           (s.second, s.first, s.backward, [d.backward.value for d in s.draws]))]}
        if n_draws > 1:
            info.meta["ipsae"].update({"draws": n_draws, "over_draws": "median"})

    @staticmethod
    def _say_pae(info: _Info, errors: Sequence[float], n_contacts: int, pairs: Sequence[float],
                 n_pairs: int, from_matrix: bool = False, by_draw: "Sequence[Sequence[float]]" = (),
                 draws: "PaeDraws | None" = None) -> None:
        """What a figure says about the predicted aligned error of its contacts, beside the tags on the lines.

        ``errors`` are the values of the contacts, ``pairs`` those of the residue pairs they are between
        (``n_pairs`` with the pairs that have none). ``from_matrix``: values among them were read from a
        matrix, so it is known which of its two directions they are; values that came with the contacts
        are taken as given, whether a matrix was handed in beside them or not.
        ``by_draw``: where every value is the median over the draws of a model, the contacts' values in
        each draw. The figure then says that it shows a median, of how many, and what each draw says;
        and, from ``draws``, where a draw names the copies of a chain in another order than the
        structure that is drawn.
        """
        lo, hi = pae_shown(min(errors)), pae_shown(max(errors))
        tagged = sum(1 for v in pairs if pae_tag(v) is not None)
        n_draws = len(by_draw)
        info.strip.append((f"contact PAE, median of {n_draws} draws" if n_draws else "contact PAE",
                           f"{min(errors):.1f} \u00c5" if round(min(errors), 1) == round(max(errors), 1)
                           else f"{min(errors):.1f}\u2013{max(errors):.1f} \u00c5"))
        info.strip_data.update({"contact-pae-min": f"{min(errors):.1f}", "contact-pae-max": f"{max(errors):.1f}"})
        if n_draws:
            info.strip_data["contact-pae-draws"] = str(n_draws)
        what = ("PAE (predicted aligned error) is the model's expected error, in \u00e5ngstr\u00f6m, in placing one "
                "residue of a contact relative to the other: low is confident, about 32 means no idea. It is not "
                "pLDDT")
        info.notes.append(what + ", which line weight, dashes and ? show.")
        info.tag_notes.append(info.notes[-1])
        info.summary_notes.append(what + ". The range under the title is over all contacts; this table does not "
                                         "give it contact by contact.")
        if n_draws:
            # A median of draws is not what one draw says: how many, why the median, and each draw's own range.
            spans = [f"{min(v):.1f}\u2013{max(v):.1f}" for v in by_draw]
            each = ", ".join(spans[:-1]) + " and " + spans[-1]
            # Said of the draws that were handed in: how many times the model was sampled in all is not known here.
            said = (f"The PAE comes from {n_draws} samples of the model ({n_draws} draws). The PAE of a contact is "
                    f"the median of its {n_draws} values, not the largest, which would rise with the number of "
                    f"draws whatever the model knows. Draw by draw the contacts shown here range over {each} \u00c5.")
            renamed = draws.renamed if draws is not None else []
            if renamed:
                # Copies of a chain are named as the model pleases, draw by draw: said where it happened.
                groups = " and ".join(", ".join(g[:-1]) + " and " + g[-1] for g in draws.copies)  # type: ignore[union-attr]
                said += (f" Chains {groups} are copies of one another, and the model names them in another order "
                         f"in {len(renamed)} of these {n_draws} draws: the chains of every draw were matched to "
                         f"those of the structure shown here, by superposition, before its values were read.")
            info.notes.append(said)
            info.summary_notes.append(said)
        if tagged:
            info.legend_extra.append(f"Predicted aligned error (PAE) of a contact in \u00c5"
                                     + (f", median of {n_draws} draws" if n_draws else "") + f": written from "
                                     f"{PAE_SHOWN} \u00c5, bold from {PAE_STRONG} \u00c5. Not pLDDT")
            if tagged == len(pairs):
                lead = ("The one residue pair in contact has" if tagged == 1
                        else f"All {tagged} residue pairs in contact have")
            else:
                lead = f"{tagged} of the {len(pairs)} residue pairs in contact {'has' if tagged == 1 else 'have'}"
            note = (f"{lead} a PAE of {PAE_SHOWN} \u00c5 or more, written on "
                    f"{'its line' if tagged == 1 else 'their lines'}.")
        else:
            note = f"No contact here reaches {PAE_SHOWN} \u00c5, so none carries a PAE on its line."
        if len(pairs) < n_pairs:
            none = n_pairs - len(pairs)
            note += f" For {none} more pair{'s' if none != 1 else ''} there is no PAE."
        info.notes.append(note)
        info.tag_notes.append(note)
        assert lo is not None and hi is not None
        if lo >= PAE_STRONG:
            # Said of every contact only where every contact has a value.
            whom = ("Every contact in this figure has a PAE" if len(errors) == n_contacts
                    else "Every contact in this figure that has a PAE has one")
            info.warnings.append(
                f"{whom} of {PAE_STRONG} \u00c5 or more: the model does not place "
                f"the two sides against each other, whatever the pLDDT of their residues.")
        info.meta["pae"] = {"min": round(min(errors), 2), "max": round(max(errors), 2),
                            "median": round(float(np.median(list(errors))), 2), "with_a_value": len(errors),
                            "of_contacts": n_contacts, "pairs": len(pairs), "of_pairs": n_pairs,
                            "pairs_written": tagged, "written_from": PAE_SHOWN, "bold_from": PAE_STRONG}
        if from_matrix:
            info.meta["pae"]["direction"] = "the smaller of the two directions of the matrix"
        if n_draws:
            info.meta["pae"].update({
                "draws": n_draws, "over_draws": "median",
                "by_draw": [{"min": round(min(v), 2), "median": round(float(np.median(list(v))), 2),
                             "max": round(max(v), 2)} for v in by_draw]})
            if draws is not None and draws.copies:
                info.meta["pae"].update({"copies": draws.copies, "chains_matched_by": "superposition",
                                         "draws_named_otherwise": draws.renamed})

    # ------------------------------------------------------------------ rendering
    def render(self, style: str = "detailed", **options: Any) -> Any:
        """Render the figure. When the site that was chosen cannot be drawn legibly and nobody asked
        for it by name, the figure narrows to a smaller one, twice at most, and says so.

        Every figure starts from the site the interface was built with, whatever an earlier figure
        had to narrow to: what one style or one page cannot lay out does not decide what the next is
        given. After a render, :attr:`site`, :attr:`diagram` and :attr:`layout` are those of the
        figure it drew. A ``graph`` is a figure of the site it was made for: it is drawn at that
        site or it is not drawn, never narrowed.
        """
        given = options.get("graph")
        sites = self._sites()
        site = next(sites)
        if given is not None:
            held = (given.meta.get("interface") or {}).get("site")
            site = next((s for s in self._sites() if _site_meta(s) == held), site)
        self._show(site)
        result = self._draw(style, options)
        while result.status == "unreadable" and given is None:
            narrower = next(sites, None)
            if narrower is None:
                break
            self._show(narrower)
            result = self._draw(style, options)
            result.warnings.append(f"the interface could not be drawn legibly with more residues: narrowed to "
                                   f"{len(self.site.residues)} residues of {self.spine.name} ({self.site.rule})")
        return result

    def _sites(self) -> "Iterator[Site]":
        """The site the interface was built with, and after it the sites a figure narrows to where it
        cannot be drawn legibly: twice at most, each time to three fifths of the residues and to no
        fewer than eight. A site that was asked for by name is not narrowed."""
        site = self._built
        yield site
        for _ in range(2):
            if self.seed is not None or len(site.residues) <= 8:
                return
            site = choose_site(self._oriented, self._order, self._index, limit=max(8, (len(site.residues) * 3) // 5))
            yield site

    def _show(self, site: Site) -> None:
        """Make ``site`` the one the next figure is of: its diagram, which is built once, and its layout."""
        if site == self.site:
            return
        if site in self._diagrams:
            self._decide(site)
        else:
            self._diagrams[site] = self._diagram(site)
        self.site, self.diagram = site, self._diagrams[site]

    def _draw(self, style: str, options: "dict[str, Any]") -> Any:
        """One figure of the site as it stands. Where the residues were to stand on a plane and that
        cannot be laid out legibly, the column is drawn in its place, and the figure and the result say so."""
        given = options.get("graph")
        held = (given.meta.get("interface") or {}) if given is not None else {}
        if given is not None and held.get("layout", "column") != self.layout:
            # A graph is drawn as what it holds. The one case this object can serve from where it
            # stands: the column it drew itself in place of a plane that could not be laid out.
            column = self._column_instead(self.site) if self.layout == "plane" else None
            if column is None or column.backend.interface.meta["layout_rule"] != held.get("layout_rule"):
                raise BindviewError(
                    f"the graph holds {'boxes on a plane' if held.get('layout') == 'plane' else 'a column'}, and "
                    f"this interface is drawn {'on a plane' if self.layout == 'plane' else 'as a column'}: build it "
                    f"with layout={held.get('layout', 'column')!r} to draw that graph")
            result = column.render(style=style, **options)
            result.warnings.extend(column.backend.interface.warnings)
            return result
        mine = self.diagram.backend.interface.meta.get("spine_order")
        if given is not None and "spine_order" in held and held["spine_order"] != mine:
            # The boxes of a graph are known by their place in the column: another order, other residues.
            raise BindviewError(f"the graph holds a column in the order of the {held['spine_order']}, and this "
                              f"interface orders its column by the {mine}: build it with "
                              f"order={held['spine_order']!r} to draw that graph")
        result = self.diagram.render(style=style, **options)
        result.warnings.extend(self.diagram.backend.interface.warnings)
        if self.layout == "column" and (self._sequence_only or self.layout_asked == "plane"
                                        or self._column_for_atoms):
            # A plane that the rule would have drawn, or that was asked for, and is not: said in the result.
            result.warnings.append(f"{self.spine.name} is drawn as {self.layout_rule}")
        if self.layout != "plane" or result.status != "unreadable" or given is not None:
            return result
        column = self._column_instead(self.site)
        result = column.render(style=style, **options)
        info = column.backend.interface
        result.warnings.extend(info.warnings)
        result.warnings.append(f"{self.spine.name} is drawn as {info.meta['layout_rule']}")
        return result

    def _column_instead(self, site: Site) -> Any:
        """The column of a site whose plane cannot be laid out legibly: for the figure that was asked
        for and no other. The object stays what it is, so that what one style cannot lay out does not
        decide what the other is given."""
        key = tuple(site.residues)
        if key not in self._columns:
            self._no_plane[key] = "a figure with the boxes where they lie could not be laid out legibly"
            try:
                self._columns[key] = self._diagram(site)
            finally:
                del self._no_plane[key]
                self._decide(site)
        return self._columns[key]

    def render_svg(self, path: "str | Path | None" = None, style: str = "detailed", **options: Any) -> Any:
        """Draw the interface and write it to ``path``, where one is given."""
        result = self.render(style=style, **options)
        if path is not None and result.svg is not None:
            Path(path).write_text(result.svg, encoding="utf-8")
            result.path = Path(path)
        return result
