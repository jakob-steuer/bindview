"""What the models of an ensemble agree on, counted by residue.

:meth:`bindview.ensemble.Ensemble.diagram` draws one model and weighs each of its contacts by the share of the
models that make it. Where the models differ much, that hides what they agree on. A residue that nearly all of
them hold the ligand with, each by another atom, has several faint lines; and one that the drawn model happens
to miss has none, however many of the others contact it. Here a *residue* is counted, and all the models are
drawn (:meth:`bindview.ensemble.Ensemble.consensus`).

The rules:

**A residue is contacted in a model** that has any record with it which is attached to the ligand. Its share is a
count over the tables and depends on nothing else.

**The parts of the ligand** (:func:`ligand_parts`). A line goes to a part, not to an atom: whether a lysine is on
one oxygen of a phosphate or on the other is no difference, and a model often calls the contact a hydrogen bond in
one sample and a salt bridge in the next. A part is (1) the rings that share atoms; (2) an end group: a phosphorus
or sulphur with its terminal oxygens and nitrogens, or another atom with two or more such ends (a carboxylate, an
amidine, a nitro group); (3) a piece of :data:`PIECE` atoms or more of what is left. A smaller piece goes to the
part that is nearest along the bonds, atom by atom.

**A line** stands for the models that have any contact between the residue and that part, and has the colour of the
kind that most of them make there (of two that equally many make, a covalent bond or a coordination before the others),
on the atoms that most of them make it with. A residue that is drawn as atoms has
one line for each of its sites (an atom, a ring, a charged group): to the part that most of the models have it on,
where at least half of the models that contact the residue do. A residue with no such line has the one that is made
most. (Half of the residue's own models, and no share of all the models: a share of its own for lines would be one
more number to explain, and ``least`` would draw lines that a third of a residue's models make.)

**Which end is which.** Two ends of a molecule can be the same (the two adenosines of AP5A), and which of them a
model calls A is a convention. Every model's atoms are named by the symmetry of the molecule that makes its contacts
agree best with those of the others, starting from the model that is drawn (:func:`matched`). A molecule without
symmetry has nothing to choose, and the shares of the residues do not depend on it. Those of the lines can only
rise by it, which the figure says where a model was renamed.

**What is drawn how.** A residue that fewer than ``least`` of the models contact is left out, and named. From the
share ``atoms`` on a residue is drawn as atoms, if any of its lines has a direction. Any other is a ring alone,
beside the part of the ligand that most of the models have it on, without a line.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from .fingerprint import TYPE_ORDER
from .interactions import Interaction, InteractionType
from .ligand import Ligand
from .residues import format_residue_label
from .resolve import ResolvedInteraction

if TYPE_CHECKING:
    from .ensemble import Ensemble

__all__ = ["ResidueShare", "ligand_parts", "residue_shares", "PIECE"]

#: A piece of the ligand between its rings and end groups is a part of its own from this many atoms on.
PIECE = 4
#: The search for the naming that agrees best ends by itself after a few turns; at most this many.
_TURNS = 10
#: Of two kinds that equally many models make, the one that says more: a bond and a coordination before all else.
_RANK = {t: i for i, t in enumerate(sorted(TYPE_ORDER, key=lambda t: t not in (InteractionType.COVALENT,
                                                                             InteractionType.METAL)))}


@dataclass(frozen=True)
class _Made:
    """One contact of one model, on the atoms of the first model's ligand."""

    model: int
    atoms: tuple[int, ...]
    resolved: ResolvedInteraction

    @property
    def kind(self) -> InteractionType:
        return self.resolved.type

    @property
    def site(self) -> tuple[str, tuple[str, ...]]:
        return (self.resolved.protein.kind, self.resolved.protein.atoms)


@dataclass(frozen=True)
class ResidueShare:
    """A residue, and the models of an ensemble that contact it."""

    chain: str
    residue_number: int
    insertion_code: str
    residue_name: str
    models: tuple[int, ...]                 # indices of the models that contact it, ascending
    n_total: int                            # models in the ensemble
    #: For each kind of contact, how many of the models make one with the residue; the most frequent first.
    kinds: tuple[tuple[InteractionType, int], ...]
    #: For each part of the ligand, the contacts with it.
    places: "dict[int, list[_Made]]" = field(repr=False, compare=False, default_factory=dict)

    @property
    def n_models(self) -> int:
        """How many models contact the residue."""
        return len(self.models)

    @property
    def share(self) -> float:
        """The share of the models that do."""
        return len(self.models) / self.n_total

    @property
    def residue_id(self) -> tuple[str, int, str]:
        """``(chain, number, insertion code)`` of the residue."""
        return (self.chain, self.residue_number, self.insertion_code)

    @property
    def residue_label(self) -> str:
        """The residue as a figure names it."""
        return format_residue_label(self.residue_name, self.residue_number, self.insertion_code, None)

    @property
    def said(self) -> str:
        """In words, for a tooltip."""
        kinds = ", ".join(f"{kind.label.lower()} {n}" for kind, n in self.kinds)
        return f"contacted in {self.n_models} of {self.n_total} models ({kinds})"

    def models_at(self, part: int) -> set[int]:
        """The models that have it on one part of the ligand."""
        return {m.model for m in self.places.get(part, ())}

    def kind_at(self, part: int) -> InteractionType:
        """The kind of contact that most models make with this part of the ligand."""
        models: "dict[InteractionType, set[int]]" = defaultdict(set)
        for m in self.places[part]:
            models[m.kind].add(m.model)
        return max(models, key=lambda kind: (len(models[kind]), -_RANK[kind]))


def ligand_parts(ligand: Ligand) -> list[tuple[int, ...]]:
    """The parts of a ligand, as indices of its atoms (see the module)."""
    mol = ligand.mol
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    seeds: list[set[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        atoms = set(ring)
        for other in [s for s in seeds if s & atoms]:
            seeds.remove(other)
            atoms |= other
        seeds.append(atoms)
    taken: set[int] = set().union(*seeds) if seeds else set()
    for a in mol.GetAtoms():
        if a.GetIdx() in taken:
            continue
        ends = [b.GetIdx() for b in a.GetNeighbors()
                if b.GetAtomicNum() in (7, 8) and sum(c.GetAtomicNum() > 1 for c in b.GetNeighbors()) == 1]
        if len(ends) >= 2 or (ends and a.GetAtomicNum() in (15, 16)):
            seeds.append({a.GetIdx(), *ends})
            taken |= seeds[-1]
    seeds.sort(key=min)
    left = set(heavy) - taken
    pieces: list[set[int]] = []
    while left:
        piece, front = set(), [min(left)]
        while front:
            i = front.pop()
            if i in piece:
                continue
            piece.add(i)
            front.extend(b.GetIdx() for b in mol.GetAtomWithIdx(i).GetNeighbors() if b.GetIdx() in left)
        left -= piece
        pieces.append(piece)
    for piece in pieces:
        held = any(b.GetIdx() in taken for i in piece for b in mol.GetAtomWithIdx(i).GetNeighbors())
        if len(piece) >= PIECE or not held:
            seeds.append(piece)
    owner = {i: k for k, atoms in enumerate(seeds) for i in atoms}
    front = sorted(owner, key=lambda i: (owner[i], i))
    while front:                                          # breadth first: an atom goes to the part nearest to it
        fresh: list[int] = []
        for i in front:
            for b in mol.GetAtomWithIdx(i).GetNeighbors():
                if b.GetAtomicNum() > 1 and b.GetIdx() not in owner:
                    owner[b.GetIdx()] = owner[i]
                    fresh.append(b.GetIdx())
        front = fresh
    out: "dict[int, list[int]]" = defaultdict(list)
    for i in sorted(owner):
        out[owner[i]].append(i)
    return [tuple(out[k]) for k in sorted(out)]


def matched(ens: "Ensemble", symmetries: np.ndarray) -> tuple["dict[tuple, list[_Made]]", list[int]]:
    """Every attached contact of every model by its residue, on the atoms of the first model's ligand; and for
    each model which of the symmetries its atoms are named by (the first is the names as they are)."""
    first = ens.diagrams[0].ligand
    place = {name: i for i, name in enumerate(first.names)}
    raw: "list[list[tuple[tuple, tuple[int, ...], ResolvedInteraction]]]" = []
    for d in ens.diagrams:
        rows = []
        for r in d.interactions:
            if r.ligand is None:
                continue
            chain = r.residue.chain if r.residue is not None else r.record.chain
            key = (chain, r.record.residue_number, r.record.insertion_code, r.record.residue_name.upper())
            rows.append((key, tuple(place[d.ligand.names[i]] for i in r.ligand.atoms), r))
        raw.append(rows)
    keys = {key: i for i, key in enumerate(sorted({key for rows in raw for key, _, _ in rows}))}
    pairs = [np.array(sorted({(keys[key], a) for key, atoms, _ in rows for a in atoms}), dtype=int).reshape(-1, 2)
             for rows in raw]

    def seen(k: int, p: int) -> np.ndarray:
        out = np.zeros((len(keys), len(place)), dtype=int)
        out[pairs[k][:, 0], symmetries[p][pairs[k][:, 1]]] = 1
        return out

    def best(k: int, others: np.ndarray) -> int:
        # The first of equal ones: the names as they are, if they do as well as any.
        return int(others[pairs[k][:, 0][None, :], symmetries[:, pairs[k][:, 1]]].sum(axis=1).argmax())

    choice = [0] * ens.n
    if len(symmetries) > 1 and keys:
        known = seen(ens.member, 0)
        choice = [0 if k == ens.member else best(k, known) for k in range(ens.n)]
        for _ in range(_TURNS):
            # Every model against all the others as they are named now; all at once, so that the order of
            # the models decides nothing. The drawn one keeps its names: it is the one the others are named by.
            total = sum((seen(k, choice[k]) for k in range(ens.n)), np.zeros_like(known))
            new = [0 if k == ens.member else best(k, total - seen(k, choice[k])) for k in range(ens.n)]
            if new == choice:
                break
            choice = new
    made: "dict[tuple, list[_Made]]" = {}
    for k, rows in enumerate(raw):
        for key, atoms, r in rows:
            made.setdefault(key, []).append(_Made(k, tuple(int(symmetries[choice[k]][a]) for a in atoms), r))
    return made, choice


def residue_shares(ens: "Ensemble", symmetries: np.ndarray) -> tuple[list[ResidueShare], list[int]]:
    """Every residue that any model contacts, the most frequent first; and the models whose ends were named the
    other way round."""
    made, choice = matched(ens, symmetries)
    owner = {i: k for k, atoms in enumerate(ligand_parts(ens.diagrams[0].ligand)) for i in atoms}
    out = []
    for (chain, number, icode, name), contacts in made.items():
        kinds: "dict[InteractionType, set[int]]" = defaultdict(set)
        places: "dict[int, list[_Made]]" = defaultdict(list)
        for m in contacts:
            kinds[m.kind].add(m.model)
            on = Counter(owner[a] for a in m.atoms if a in owner)
            if on:
                places[min(on, key=lambda p: (-on[p], p))].append(m)
        out.append(ResidueShare(
            chain, number, icode, name, tuple(sorted({m.model for m in contacts})), ens.n,
            tuple((t, len(kinds[t])) for t in sorted(kinds, key=lambda t: (-len(kinds[t]), _RANK[t]))), dict(places)))
    out.sort(key=lambda r: (-r.n_models, r.chain, r.residue_number, r.insertion_code))
    turned = [k for k, p in enumerate(choice)
              if any(owner.get(int(symmetries[p][a])) != owner.get(a) for a in range(len(symmetries[p])))]
    return out, turned


def lines_of(ens: "Ensemble", residue: ResidueShare, atoms: bool) -> list[ResolvedInteraction]:
    """The lines of a residue, on the ligand of the drawn model: see the module."""
    base = ens.diagrams[ens.member]
    names = ens.diagrams[0].ligand.names
    here = {name: i for i, name in enumerate(base.ligand.names)}
    order = sorted(residue.places, key=lambda p: (-len(residue.models_at(p)), p))
    wanted = [p for p in order if 2 * len(residue.models_at(p)) >= residue.n_models] if atoms else []
    sites: set = set()
    out = []
    for p in wanted or order[:1]:
        kind = residue.kind_at(p)
        there = [m for m in residue.places[p] if m.kind is kind]
        on = _most(there, lambda m: m.atoms)
        there = [m for m in there if m.atoms == on]
        site = _most(there, lambda m: m.site)
        if site in sites:
            continue
        sites.add(site)
        # A line is attached as the models' own contacts are, to the same atoms: nothing is looked for.
        of = next(m for m in there if m.site == site).resolved
        assert of.ligand is not None
        one = len(on) == 1
        out.append(replace(
            of, ligand=replace(of.ligand, atoms=tuple(here[names[a]] for a in on)), method="named", delta=None,
            computed_distance=None, notes=[], absent=False,
            record=Interaction(
                type=kind, residue_number=residue.residue_number, residue_name=residue.residue_name,
                chain=residue.chain, insertion_code=residue.insertion_code,
                occupancy=len(residue.models_at(p)) / ens.n,
                ligand_is_donor=_most(there, lambda m: m.resolved.record.ligand_is_donor),
                ligand_atom=names[on[0]] if one else None, ligand_atoms=() if one else tuple(names[a] for a in on),
                protein_atom=site[1][0] if site[0] == "atom" and site[1] else None,
                protein_atoms=() if site[0] == "atom" else tuple(site[1]))))
    return out


def _most(made: "Sequence[_Made]", of: "Callable[[_Made], Any]") -> Any:
    """What most of the models have: a model counts once, however many records it has. Of what equally
    many models have, the first."""
    models: "dict[Any, set[int]]" = {}
    for m in made:
        models.setdefault(of(m), set()).add(m.model)
    return max(models, key=lambda value: len(models[value]))
