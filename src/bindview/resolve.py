"""Resolving each interaction to the thing that makes it.

Hard requirement of the library: an interaction attaches to a named atom, or to
a ring centroid / charged group -- never to "somewhere on the ligand".

Records come in two flavours:

* **named** -- the detector gave atom names (H-bonds, hydrophobic contacts,
  halogen bonds). The ligand atom is looked up by name; nothing is inferred.
* **unnamed** -- pi-stacking, pi-cation, salt bridges and (in PLIP's tables)
  metal coordination carry only a residue and a distance. The partners are
  then identified from the structure: every chemically possible pair of
  (ligand ring / charged group / metal, protein ring / charged group / atom) is
  enumerated and the one whose centre-to-centre distance reproduces the
  reported distance is taken. Reported distances are rounded to 0.01 A, which
  in practice pins the pair uniquely; where two candidates fall inside the
  tolerance the chemically more canonical one wins, and several records from
  the same residue are assigned jointly so no pair is used twice.

Each result says how it was obtained (``method``) and how well the geometry
agrees (``delta``), so a caller can audit the attachment. A record that cannot
be resolved is returned as unresolved and is *not drawn*.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .chem import is_metal
from .features import LigandGroup, perceive_charged_groups, perceive_rings
from .interactions import NOT_GIVEN_OR_A_NAME, Interaction, InteractionType
from .ligand import Ligand
from .residues import aromatic_rings, charged_groups
from .structure import ResidueKey, Structure

__all__ = ["Anchor", "ProteinSite", "ResolvedInteraction", "resolve_interactions", "DISTANCE_TOLERANCE"]

#: Agreement (angstrom) required between a computed and a reported distance.
#: Reported values are rounded to 0.01, so 0.02 leaves room for that and for
#: coordinate rounding in format conversions.
DISTANCE_TOLERANCE = 0.02

#: Beyond this disagreement a geometric match is not even called approximate.
_LOOSE_TOLERANCE = 0.6

T = InteractionType


@dataclass(frozen=True)
class Anchor:
    """Where an interaction attaches on the ligand.

    ``kind`` is ``"atom"`` (one atom), ``"ring"`` (attach to the centroid of
    ``atoms``) or ``"group"`` (a charged group; ``atoms`` are its members).
    """

    kind: str
    atoms: tuple[int, ...]
    group_kind: str = ""

    @property
    def atom(self) -> int:
        return self.atoms[0]


@dataclass(frozen=True)
class ProteinSite:
    """Where an interaction attaches on the protein residue.

    ``kind`` is ``"atom"``, ``"ring"``, ``"group"`` or ``"residue"`` (unknown --
    the residue as a whole). ``atoms`` holds atom names.
    """

    kind: str
    atoms: tuple[str, ...] = ()
    charge: int = 0


@dataclass
class ResolvedInteraction:
    record: Interaction
    index: int
    residue: ResidueKey | None
    ligand: Anchor | None
    protein: ProteinSite
    method: str                      # named | supplied | distance-match | approximate | nearest | unresolved
    delta: float | None = None       # |computed - reported| in angstrom, when geometry was used
    computed_distance: float | None = None
    notes: list[str] = field(default_factory=list)
    #: A contact that the structure the figure shows does not make, and other models of its ensemble
    #: do (:meth:`bindview.ensemble.Ensemble.diagram`). It is drawn faint, with its share where a
    #: distance would stand, and is never counted as a contact of the diagram.
    absent: bool = False

    @property
    def resolved(self) -> bool:
        return self.ligand is not None

    @property
    def type(self) -> InteractionType:
        return self.record.type


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    ligand: Anchor
    protein: ProteinSite
    distance: float
    rank: int


def _centroid(points: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.asarray(points, dtype=float), axis=0)


def _protein_points(structure: Structure, key: ResidueKey) -> dict[str, np.ndarray]:
    return {a.name: np.array(a.xyz) for a in structure.residue_atoms(key) if a.element not in ("H", "D")}


def _protein_metals(structure: Structure, key: ResidueKey) -> frozenset[str]:
    """The names of the atoms of a residue that are metals, by their element: the ion of a residue of one atom,
    and the iron of a haem."""
    return frozenset(a.name for a in structure.residue_atoms(key) if is_metal(a.element.capitalize()))


def _protein_ring_sites(resname: str, pts: dict[str, np.ndarray]) -> list[tuple[ProteinSite, np.ndarray]]:
    out = []
    for ring in aromatic_rings(resname):
        if all(n in pts for n in ring):
            out.append((ProteinSite("ring", ring), _centroid([pts[n] for n in ring])))
    return out


def _protein_charge_sites(resname: str, pts: dict[str, np.ndarray], sign: int | None = None
                          ) -> list[tuple[ProteinSite, np.ndarray]]:
    out = []
    for charge, names in charged_groups(resname):
        if sign is not None and charge != sign:
            continue
        present = tuple(n for n in names if n in pts)
        if present:
            out.append((ProteinSite("group", present, charge), _centroid([pts[n] for n in present])))
    return out


def _anchor_for(group: LigandGroup) -> Anchor:
    if group.kind == "ring":
        return Anchor("ring", group.atoms)
    if len(group.atoms) == 1:
        return Anchor("atom", group.atoms, group.kind)
    return Anchor("group", group.atoms, group.kind)


def _candidates(rec: Interaction, lig: Ligand, rings: list[LigandGroup], charged: list[LigandGroup],
                pts: dict[str, np.ndarray], protein_metals: frozenset[str]) -> list[_Candidate]:
    """Every chemically possible (ligand site, protein site) pair for an unnamed record. ``pts`` are the heavy
    atoms of the residue and ``protein_metals`` the names of those that are metals."""
    out: list[_Candidate] = []
    xyz = lig.coords3d
    resname = rec.residue_name

    def add(group: LigandGroup, site: ProteinSite, centre: np.ndarray, extra_rank: int = 0) -> None:
        d = float(np.linalg.norm(group.center3d(xyz) - centre))
        out.append(_Candidate(_anchor_for(group), site, d, group.rank + extra_rank))

    if rec.type is T.PI_STACKING:
        for site, centre in _protein_ring_sites(resname, pts):
            for g in rings:
                add(g, site, centre)
    elif rec.type is T.PI_CATION:
        # Either partner may supply the ring.
        for site, centre in _protein_ring_sites(resname, pts):
            for g in charged:
                if g.charge > 0:
                    add(g, site, centre)
        for site, centre in _protein_charge_sites(resname, pts, sign=+1):
            for g in rings:
                add(g, site, centre)
    elif rec.type is T.SALTBRIDGE:
        sites = _protein_charge_sites(resname, pts)
        for site, centre in sites:
            for g in charged:
                if g.charge == -site.charge:
                    add(g, site, centre)
        if not sites:
            # Non-standard residue: fall back to its individual hetero atoms.
            for name, p in pts.items():
                if name[:1] in ("N", "O", "S"):
                    for g in charged:
                        add(g, ProteinSite("atom", (name,)), p, extra_rank=3)
    elif rec.type is T.METAL:
        metals = [g for g in charged if g.kind == "metal"]
        if metals:
            for name, p in pts.items():
                rank = 0 if name[:1] in ("N", "O", "S") else 3
                for g in metals:
                    add(g, ProteinSite("atom", (name,)), p, extra_rank=rank)
        else:
            # The metal sits on the protein side; the ligand donates.
            for name, p in pts.items():
                if name not in protein_metals:
                    continue
                for i in range(lig.n_atoms):
                    el = lig.element(i)
                    if el in ("N", "O", "S"):
                        d = float(np.linalg.norm(xyz[i] - p))
                        out.append(_Candidate(Anchor("atom", (i,)), ProteinSite("atom", (name,)), d, 0))
    else:
        # A normally named type that arrived without names: match atom pairs.
        for name, p in pts.items():
            for i in range(lig.n_atoms):
                if rec.ligand_atom and lig.names[i] != rec.ligand_atom:
                    continue
                if rec.protein_atom and name != rec.protein_atom:
                    continue
                d = float(np.linalg.norm(xyz[i] - p))
                out.append(_Candidate(Anchor("atom", (i,)), ProteinSite("atom", (name,)), d, 0))
    return out


def _cost(cand: _Candidate, reported: float | None) -> tuple[float, float | None]:
    """Lower is better. Exact matches sort by chemical rank, then by agreement."""
    if reported is None:
        return 500.0 + cand.rank * 50.0 + cand.distance, None
    delta = abs(cand.distance - reported)
    if delta <= DISTANCE_TOLERANCE:
        return cand.rank + 0.5 * delta / DISTANCE_TOLERANCE, delta
    return 100.0 + delta * 10.0 + cand.rank, delta


def _least_total(cost: Sequence[Sequence[float]]) -> list[int]:
    """For every row a column of its own, such that the costs that are chosen add up to the least there is:
    the assignment problem, by the Hungarian method, in a time of rows x rows x columns. There are not more
    rows than columns."""
    n, m = len(cost), len(cost[0])
    inf = float("inf")
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    row_of = [0] * (m + 1)                   # the row each column has (rows and columns count from 1; 0 is none)
    for i in range(1, n + 1):
        row_of[0] = i
        least, came_from, used = [inf] * (m + 1), [0] * (m + 1), [False] * (m + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0, step, j1 = row_of[j0], inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    reduced = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if reduced < least[j]:
                        least[j], came_from[j] = reduced, j0
                    if least[j] < step:
                        step, j1 = least[j], j
            for j in range(m + 1):
                if used[j]:
                    u[row_of[j]] += step
                    v[j] -= step
                else:
                    least[j] -= step
            j0 = j1
            if row_of[j0] == 0:
                break
        while j0:
            row_of[j0] = row_of[came_from[j0]]
            j0 = came_from[j0]
    chosen = [0] * n
    for j in range(1, m + 1):
        if row_of[j]:
            chosen[row_of[j] - 1] = j - 1
    return chosen


#: Up to this many records of one residue and type are assigned by trying every combination of their best
#: candidates, which takes a time that grows with (records + 1) to the power of records: 7 to the 6th, 117,649
#: combinations, at most. More records are assigned by :func:`_least_total`.
_TRY_EVERY_COMBINATION = 6


def _assign(records: list[Interaction], cands: list[_Candidate]) -> list[tuple[_Candidate, float | None] | None]:
    """Jointly assign distinct candidates to the records of one residue and type."""
    if not cands:
        return [None] * len(records)
    costs = [[_cost(c, rec.distance) for c in cands] for rec in records]
    best: list[tuple[int, float | None]] | None = None
    if len(records) <= _TRY_EVERY_COMBINATION:
        per_record = [sorted((cost, ci, delta) for ci, (cost, delta) in enumerate(row))[: max(4, len(records) + 1)]
                      for row in costs]
        best_total = float("inf")
        for combo in itertools.product(*per_record):
            idx = [c[1] for c in combo]
            if len(set(idx)) != len(idx):
                continue
            total = sum(c[0] for c in combo)
            if total < best_total - 1e-12:
                best_total = total
                best = [(c[1], c[2]) for c in combo]
    elif len(cands) >= len(records):
        chosen = _least_total([[cost for cost, _ in row] for row in costs])
        best = [(ci, row[ci][1]) for ci, row in zip(chosen, costs)]
    if best is None:
        # More records than distinct candidates (duplicated rows): allow reuse.
        cheapest = [min(range(len(cands)), key=lambda ci: (row[ci][0], ci)) for row in costs]
        best = [(ci, row[ci][1]) for ci, row in zip(cheapest, costs)]
    return [(cands[ci], delta) for ci, delta in best]


def _named_anchor(rec: Interaction, lig: Ligand) -> tuple[Anchor | None, str | None]:
    idx = lig.index_of(rec.ligand_atom or "")
    if idx is None:
        return None, f"ligand atom {rec.ligand_atom!r} is not in the structure"
    return Anchor("atom", (idx,)), None


def _supplied_group(rec: Interaction, lig: Ligand) -> tuple[Anchor | None, str | None]:
    idx = [lig.index_of(n) for n in rec.ligand_atoms]
    missing = [n for n, i in zip(rec.ligand_atoms, idx) if i is None]
    if missing:
        return None, f"ligand atoms not in the structure: {', '.join(missing)}"
    members = tuple(int(i) for i in idx if i is not None)
    if len(members) == 1:
        return Anchor("atom", members), None
    kind = "ring" if rec.type in (T.PI_STACKING, T.PI_CATION) and _is_ring(lig, members) else "group"
    return Anchor(kind, members), None


def _is_ring(lig: Ligand, members: tuple[int, ...]) -> bool:
    want = set(members)
    return any(set(r) == want for r in lig.mol.GetRingInfo().AtomRings())


def _protein_site_from_record(rec: Interaction) -> ProteinSite:
    if rec.protein_atoms:
        if len(rec.protein_atoms) == 1:
            return ProteinSite("atom", rec.protein_atoms)
        rings = aromatic_rings(rec.residue_name)
        if any(set(r) == set(rec.protein_atoms) for r in rings):
            return ProteinSite("ring", rec.protein_atoms)
        charge = 0
        for sign, names in charged_groups(rec.residue_name):
            if set(rec.protein_atoms) <= set(names):
                charge = sign
        return ProteinSite("group", rec.protein_atoms, charge)
    if rec.protein_atom:
        return ProteinSite("atom", (rec.protein_atom,))
    return ProteinSite("residue")


def _locate(structure: Structure | None, rec: Interaction) -> ResidueKey | None:
    if structure is None:
        return None
    return structure.find_residue(rec.chain or None, rec.residue_number, rec.insertion_code)


def _why_not_located(structure: Structure, rec: Interaction) -> str:
    """Why the residue of a record was not found: the structure has none, or the record names no chain and
    several chains have a residue of that number."""
    chains = [] if rec.chain else [k.chain for k in structure.residues()
                                   if k.number == rec.residue_number and k.icode == rec.insertion_code]
    if len(chains) > 1:
        listed = ", ".join(chains[:8]) + (", ..." if len(chains) > 8 else "")
        return (f"residue {rec.residue_number}{rec.insertion_code} is in {len(chains)} chains ({listed}) "
                f"and the record names no chain")
    return f"residue {rec.chain or '?'}:{rec.residue_number} is not in the structure"


def _chain_that_is_there(rec: Interaction, structure: Structure | None) -> Interaction:
    """The record with ``NA`` in the place of the chain read as what it is: the name of a chain, where the
    structure has one called so, and otherwise a chain that is not given."""
    if rec.chain == NOT_GIVEN_OR_A_NAME and (structure is None
                                             or all(k.chain != NOT_GIVEN_OR_A_NAME for k in structure.residues())):
        return rec.with_(chain="")
    return rec


def _names_that_are_there(rec: Interaction, lig: Ligand, structure: Structure | None,
                          residue: ResidueKey | None) -> Interaction:
    """The record with ``NA`` in the place of an atom's name read as what it is: the name of an atom, where
    the ligand (the residue) has one called so, and otherwise a name that is not given. The same where it is
    all that stands for the atoms of a group."""
    alone = (NOT_GIVEN_OR_A_NAME,)
    if lig.index_of(NOT_GIVEN_OR_A_NAME) is None:
        if rec.ligand_atom == NOT_GIVEN_OR_A_NAME:
            rec = rec.with_(ligand_atom=None)
        if rec.ligand_atoms == alone:
            rec = rec.with_(ligand_atoms=())
    if rec.protein_atom == NOT_GIVEN_OR_A_NAME or rec.protein_atoms == alone:
        there = (NOT_GIVEN_OR_A_NAME in _protein_points(structure, residue) if structure is not None and residue is not None
                 else rec.residue_name == NOT_GIVEN_OR_A_NAME)
        if not there:
            rec = rec.with_(protein_atom=None if rec.protein_atom == NOT_GIVEN_OR_A_NAME else rec.protein_atom,
                            protein_atoms=() if rec.protein_atoms == alone else rec.protein_atoms)
    return rec


def resolve_interactions(interactions: Sequence[Interaction], lig: Ligand,
                         structure: Structure | None = None) -> list[ResolvedInteraction]:
    """Resolve every record to its ligand anchor and protein site.

    ``structure`` is needed only for records that do not name their atoms.
    Results come back in input order.
    """
    rings = perceive_rings(lig)
    charged = perceive_charged_groups(lig)
    results: list[ResolvedInteraction | None] = [None] * len(interactions)
    pending: dict[tuple[tuple[str, int, str], InteractionType, str | None], list[int]] = {}

    for n, given in enumerate(interactions):
        given = _chain_that_is_there(given, structure)
        residue = _locate(structure, given)
        rec = _names_that_are_there(given, lig, structure, residue)
        site = _protein_site_from_record(rec)
        notes: list[str] = []
        if residue is not None and structure is not None:
            actual = structure.residue_name(residue)
            if actual.upper() != rec.residue_name.upper():
                notes.append(f"record says {rec.residue_name}{rec.residue_number}, "
                             f"the structure has {actual} at that position")
        if rec.ligand_atoms:
            anchor, problem = _supplied_group(rec, lig)
            if problem:
                notes.append(problem)
            results[n] = ResolvedInteraction(rec, n, residue, anchor, site,
                                             "supplied" if anchor else "unresolved", notes=notes)
            continue
        group_type = rec.type in (T.PI_STACKING, T.PI_CATION, T.SALTBRIDGE)
        if rec.ligand_atom and not group_type:
            anchor, problem = _named_anchor(rec, lig)
            if problem:
                notes.append(problem)
            if anchor is not None and site.kind == "residue" and residue is not None and structure is not None:
                # Ligand atom named, protein atom not: find it by distance.
                pts = _protein_points(structure, residue)
                cands = [c for c in _candidates(rec.with_(protein_atom=None), lig, rings, charged, pts,
                                                _protein_metals(structure, residue))
                         if c.ligand.atoms == anchor.atoms]
                picked = _assign([rec], cands)[0]
                if picked is not None and picked[1] is not None and picked[1] <= DISTANCE_TOLERANCE:
                    site = picked[0].protein
            results[n] = ResolvedInteraction(rec, n, residue, anchor, site,
                                             "named" if anchor else "unresolved", notes=notes)
            continue
        if residue is None:
            why = "no structure was supplied" if structure is None else _why_not_located(structure, rec)
            notes.append(f"cannot resolve the {rec.type.value} partner: {why}")
            results[n] = ResolvedInteraction(rec, n, None, None, site, "unresolved", notes=notes)
            continue
        results[n] = ResolvedInteraction(rec, n, residue, None, site, "unresolved", notes=notes)
        # Records are assigned together with those that have the same candidates: of one residue and type and,
        # where the candidates are pairs of atoms, with the same atom of the residue named (or none).
        named = None if group_type or rec.type is T.METAL else rec.protein_atom
        pending.setdefault((rec.residue_id, rec.type, named), []).append(n)

    for idxs in pending.values():
        first = results[idxs[0]]
        assert first is not None and first.residue is not None and structure is not None
        recs = [r.record for r in (results[i] for i in idxs) if r is not None]
        pts = _protein_points(structure, first.residue)
        cands = _candidates(recs[0], lig, rings, charged, pts, _protein_metals(structure, first.residue))
        for i, picked in zip(idxs, _assign(recs, cands)):
            res = results[i]
            assert res is not None
            if picked is None:
                res.notes.append(f"no {res.record.type.value} partner could be identified on the ligand "
                                 f"or on {res.record.residue_label}")
                continue
            cand, delta = picked
            res.computed_distance = cand.distance
            res.delta = delta
            if delta is None:
                res.method = "nearest"
                res.notes.append("no distance reported; attached to the chemically most plausible nearest partner")
            elif delta <= DISTANCE_TOLERANCE:
                res.method = "distance-match"
            elif delta <= _LOOSE_TOLERANCE:
                res.method = "approximate"
                res.notes.append(
                    f"closest geometric match is {cand.distance:.2f} A against a reported {res.record.distance:.2f} A")
            else:
                res.notes.append(
                    f"no partner reproduces the reported {res.record.distance:.2f} A "
                    f"(closest candidate {cand.distance:.2f} A); not drawn")
                continue
            res.ligand = cand.ligand
            if res.protein.kind == "residue" or not res.record.protein_atoms:
                res.protein = cand.protein

    return [r for r in results if r is not None]
