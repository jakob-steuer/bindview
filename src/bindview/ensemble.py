"""One ligand in its pocket, from several models of the same complex.

A :class:`~bindview.compare.Comparison` holds several *ligands* of one pocket. An
:class:`Ensemble` holds the other thing that comes in several: *models* of one
complex -- the samples of a co-fold, the conformations a generative model
draws, the members of an NMR bundle or the frames of a trajectory written out
as files, several crystal structures. It answers one question: in how many of
the models is each contact made, and at what distance?

The answer is data (:attr:`Ensemble.contacts`) and one figure
(:meth:`Ensemble.diagram`): an ordinary :class:`~bindview.diagram.Diagram` of one
member in which every contact carries, as its ``occupancy``, the share of the
models that make it. Occupancy is drawn as it always is (weight and opacity of
the line, :func:`bindview.confidence.occupancy_factors`); nothing here draws.

Three rules decide the result, and none of them is a threshold:

**Which member is drawn.** The medoid: the model whose ligand lies nearest to
the others'. Every two models are superposed on their pocket
(:func:`bindview.align.pocket_transforms_between`), and the distance between them is the
RMSD of the ligand's heavy atoms after that, without fitting the ligand itself
and taking the smallest value over the symmetries of the ligand's graph, so
that a flipped phenyl ring or a turned CF3 is not a difference. ``member=``
overrules the choice. The order of the models decides ties and nothing else.

**When two contacts are one contact.** Same residue, same interaction type,
same ligand atoms -- with three things said about "same":

* ligand atoms that the symmetry of the molecule exchanges count as one atom
  (the fluorines of a CF3, the oxygens of a carboxylate, the two sides of a
  para-substituted ring): a halogen bond made through ``F1`` in two samples
  and through ``F3`` in three is one contact that is always there. Which of a
  phosphate's free oxygens is called ``O1A`` is a convention, and deposited
  structures of one cofactor do not always agree on it;
* a hydrophobic contact is matched on the residue alone: which carbon a
  detector pairs with which is its bookkeeping, and the figure draws such a
  contact as an arc, not as a line to an atom;
* the protein atom, the direction of a hydrogen bond and the water of a water
  bridge are not part of it.

Occupancy is therefore a count over the contact tables and does not depend on
the superposition.

**A contact the drawn member does not make** is not drawn. A line of a bindview
figure is a contact of the model it shows, with that model's distance on it.
Such a contact is a row of :attr:`Ensemble.contacts` with ``in_member`` false,
and the figure says in a sentence how many there are and names the most
frequent (:attr:`bindview.diagram.Diagram.remarks`). On request it is drawn
after all (``Ensemble.diagram(absent="faint")``): faint, with the share of the
models that make it where a distance would stand, and still not counted as a
contact of the diagram.

Models are read as every structure is: one file a model
(:func:`bindview.structure.read_structure`), or the models of one file
(:func:`bindview.structure.read_models`: an NMR bundle, frames written into one
PDB file). No trajectory format is read. Residues are matched by chain and
number, as a comparison matches them, and chains are taken as they are named:
models whose pockets do not carry the same residues are refused, and models of
a homo-oligomer that name its chains in another order are not put right.
"""

from __future__ import annotations

import copy
import math
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from rdkit import Chem

from .align import pocket_transforms_between
from .backends import Polymer
from .consensus import ResidueShare, lines_of, residue_shares
from .diagram import DISTANCES, Diagram
from .errors import EnsembleError, BindviewError
from .fingerprint import TYPE_ORDER, Fingerprint, FingerprintTable, _write_rows
from .interactions import Interaction, InteractionType
from .ligand import Ligand
from .residues import format_residue_label
from .resolve import ResolvedInteraction, resolve_interactions
from .structure import Structure

__all__ = ["Ensemble", "EnsembleContact", "symmetry_classes", "contact_key", "MAX_SYMMETRIES", "CONTACT_COLUMNS",
           "ABSENT"]

#: What :meth:`Ensemble.diagram` can do with a contact that the drawn member does not make.
ABSENT = ("named", "faint")

#: At most this many symmetries of the ligand's graph are tried for the RMSD of two poses. The
#: pairing of the atoms by name is always among them; an ensemble says when the search stopped here.
MAX_SYMMETRIES = 10_000
#: Two mean distances that differ by less than this (angstrom) are equal: the model given first is drawn.
_TIE = 1e-3
#: Contacts the drawn member does not make that the figure names; beyond that it gives their number.
_NAMED = 3
_TYPE_RANK = {t: i for i, t in enumerate(TYPE_ORDER)}

#: Columns of :meth:`Ensemble.to_csv`, in order.
CONTACT_COLUMNS = ("chain", "residue_number", "insertion_code", "residue_name", "interaction_type", "ligand_atoms",
                   "n_models", "n_total", "occupancy", "in_member", "distance_median", "distance_min",
                   "distance_max", "models")

_Key = tuple[str, int, str, str, InteractionType, tuple[int, ...]]


#: An end of a conjugated group: an oxygen or nitrogen with one neighbour, on an atom that holds another
#: such end by the other kind of bond. The two oxygens of a carboxylate, the free oxygens of a phosphate
#: or a sulfonate, those of a nitro group, the nitrogens of an amidine. (RDKit's own pattern for this, in
#: the symmetric RMSD of ``rdMolAlign``.)
_TERMINAL = Chem.MolFromSmarts("[O,N;D1;$([O,N;D1]-[*]=[O,N;D1]),$([O,N;D1]=[*]-[O,N;D1])]")


def _symmetric_form(ligand: Ligand) -> Chem.Mol:
    """The ligand's graph with the ends of every conjugated group made alike.

    A molecule is written with one of a phosphate's free oxygens double-bonded and the other
    charged, and a graph of that takes the two for different atoms. They are not: which of them is
    called ``O1A`` is a convention, and two deposited structures of one cofactor can disagree on
    it. Here every such end loses its charge, its hydrogens and its bond order, so that the
    symmetry of the graph is that of the molecule. Only symmetry is read off this form; it is not
    a molecule to draw.
    """
    mol = Chem.RWMol(ligand.mol)
    for (index,) in mol.GetSubstructMatches(_TERMINAL):
        atom = mol.GetAtomWithIdx(index)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        for bond in atom.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
    mol.UpdatePropertyCache(strict=False)
    return mol


def symmetry_classes(ligand: Ligand) -> dict[str, int]:
    """For every atom of a ligand, by name, the class of atoms it can be exchanged with.

    Two atoms share a class when the symmetry of the molecule exchanges them: the fluorines of a
    CF3, the two sides of a para-substituted ring, the two oxygens of a carboxylate. These are
    RDKit's canonical ranks without tie-breaking, on the graph with the ends of conjugated groups
    made alike (:func:`_symmetric_form`) and stereochemistry left aside: stereo is read off each
    model's coordinates, and the classes must not depend on which model they were made from.
    """
    ranks = list(Chem.CanonicalRankAtoms(_symmetric_form(ligand), breakTies=False, includeChirality=False))
    return {name: int(ranks[i]) for i, name in enumerate(ligand.names)}


def contact_key(r: ResolvedInteraction, ligand: Ligand, classes: Mapping[str, int], *, symmetry: bool = True,
                hydrophobic_by_residue: bool = True) -> "_Key | None":
    """What makes two records of two models the same contact; ``None`` for a record that is attached nowhere.

    The residue (chain, number, insertion code, name), the interaction type, and the ligand atoms
    the record was resolved onto, each by its class in ``classes``
    (:func:`symmetry_classes`). The two switches exist so that what each rule merges can be
    counted on a set, which is how the rules are judged; an :class:`Ensemble` uses both rules.
    """
    if r.ligand is None:
        return None
    rec = r.record
    chain = r.residue.chain if r.residue is not None else rec.chain
    if hydrophobic_by_residue and rec.type is InteractionType.HYDROPHOBIC:
        site: tuple[int, ...] = ()
    elif symmetry:
        site = tuple(sorted({classes[ligand.names[i]] for i in r.ligand.atoms}))
    else:
        # Without the symmetry rule an atom is itself: its own number, below every class.
        order = {name: -1 - k for k, name in enumerate(sorted(classes))}
        site = tuple(sorted({order[ligand.names[i]] for i in r.ligand.atoms}))
    return (chain, rec.residue_number, rec.insertion_code, rec.residue_name.upper(), rec.type, site)


@dataclass(frozen=True)
class EnsembleContact:
    """One contact of an ensemble: which models make it, and at what distance.

    ``ligand_atoms`` are the atoms of the ligand through which the models make it: one atom, the
    atoms of a ring or of a charged group, several atoms where the models use atoms that the
    molecule's symmetry exchanges, and for a hydrophobic contact every carbon a model pairs with
    the residue. ``distances`` holds, for each model of ``models``, the distance its table
    reports (the shortest, where it has several records of the contact); nothing is measured in
    a model that does not make the contact.
    """

    chain: str
    residue_number: int
    insertion_code: str
    residue_name: str
    type: InteractionType
    ligand_atoms: tuple[str, ...]
    models: tuple[int, ...]                 # indices of the models that make it, ascending
    n_total: int                            # models in the ensemble
    in_member: bool                         # the drawn member makes it
    distances: tuple["float | None", ...]   # one for each of ``models``

    @property
    def n_models(self) -> int:
        """How many models make it."""
        return len(self.models)

    @property
    def occupancy(self) -> float:
        """The share of the models that make the contact."""
        return self.n_models / self.n_total

    def _measured(self) -> list[float]:
        return [d for d in self.distances if d is not None]

    @property
    def distance_median(self) -> "float | None":
        """The median of its distance over the models that make it."""
        values = self._measured()
        return float(statistics.median(values)) if values else None

    @property
    def distance_min(self) -> "float | None":
        """The shortest of its distances."""
        values = self._measured()
        return min(values) if values else None

    @property
    def distance_max(self) -> "float | None":
        """The longest of its distances."""
        values = self._measured()
        return max(values) if values else None

    @property
    def residue_id(self) -> tuple[str, int, str]:
        """``(chain, number, insertion code)`` of the residue."""
        return (self.chain, self.residue_number, self.insertion_code)

    @property
    def residue_label(self) -> str:
        """The residue as a figure names it."""
        return format_residue_label(self.residue_name, self.residue_number, self.insertion_code, None)

    def as_dict(self) -> dict[str, Any]:
        """The contact as plain data."""
        return {"chain": self.chain, "residue_number": self.residue_number, "insertion_code": self.insertion_code,
                "residue_name": self.residue_name, "type": self.type.value, "ligand_atoms": list(self.ligand_atoms),
                "models": list(self.models), "n_models": self.n_models, "n_total": self.n_total,
                "occupancy": self.occupancy, "in_member": self.in_member, "distances": list(self.distances),
                "distance_median": self.distance_median, "distance_min": self.distance_min,
                "distance_max": self.distance_max}


def _symmetries(ligand: Ligand) -> tuple[np.ndarray, bool]:
    """The symmetries of a ligand's graph as rows of atom indices, the identity first, and whether
    the search ended by itself (``False``: it was stopped at :data:`MAX_SYMMETRIES`)."""
    n = ligand.n_atoms
    identity = tuple(range(n))
    graph = _symmetric_form(ligand)
    found = graph.GetSubstructMatches(graph, uniquify=False, useChirality=False, maxMatches=MAX_SYMMETRIES)
    rows = [identity] + [tuple(m) for m in found if len(m) == n and tuple(m) != identity]
    return np.array(rows, dtype=int).reshape(-1, n), len(found) < MAX_SYMMETRIES


def _default_names(models: Sequence[Any]) -> list[str]:
    """A name for every model: what tells the files apart, else ``model 1``, ``model 2``, ..."""
    numbered = [f"model {i + 1}" for i in range(len(models))]
    if all(isinstance(m, Structure) and m.model is not None for m in models):
        # The models of one file (read_models): each by the number the file gives it.
        of_file = [f"model {m.model}" for m in models]
        if len(set(of_file)) == len(of_file):
            return of_file
    for pick in (lambda p: p.stem, lambda p: f"{p.parent.name}/{p.stem}"):
        names = []
        for m in models:
            source = m.source if isinstance(m, Structure) else m
            names.append(pick(Path(source)) if source else "")
        if all(names) and len(set(names)) == len(names):
            return names
    return numbered


class Ensemble:
    """Several models of one complex, each built as a :class:`~bindview.diagram.Diagram`.

    ``names`` label the models in the result and in the figure's sentences (default ``model 1``,
    ``model 2``, ...). ``member`` names the model to draw, by index or by name; left out, the
    medoid is drawn (see the module docstring). ``ensemble`` is what the legend calls the set:
    occupancy is a share, and the figure has to say of what (default ``"the N models"``).

    All models must hold the same ligand, with the same atom names, and number the residues of
    its pocket alike; an :class:`~bindview.errors.EnsembleError` says where they do not.
    """

    def __init__(self, diagrams: Sequence[Diagram], names: "Sequence[str] | None" = None, *,
                 member: "int | str | None" = None, ensemble: "str | None" = None) -> None:
        if len(diagrams) < 2:
            raise ValueError("an ensemble needs at least two models")
        self.diagrams = list(diagrams)
        raw = list(names) if names is not None else [f"model {i + 1}" for i in range(len(self.diagrams))]
        if len(raw) != len(self.diagrams):
            raise ValueError("one name per model is needed")
        seen: dict[str, int] = {}
        self.names: list[str] = []
        for nm in raw:
            seen[nm] = seen.get(nm, 0) + 1
            self.names.append(nm if seen[nm] == 1 else f"{nm} ({seen[nm]})")
        #: What the contacts' occupancy is a share of, for the legend.
        self.ensemble = ensemble or f"the {self.n} models"
        #: What deviated from the clean path, in words.
        self.warnings: list[str] = []

        first = self.diagrams[0].ligand
        for dia, nm in zip(self.diagrams[1:], self.names[1:]):
            if set(dia.ligand.names) != set(first.names):
                only_first = sorted(set(first.names) - set(dia.ligand.names))
                only_this = sorted(set(dia.ligand.names) - set(first.names))
                raise EnsembleError(
                    f"{nm} does not hold the ligand of {self.names[0]}: its heavy atoms are named otherwise "
                    f"(only in {self.names[0]}: {', '.join(only_first[:6]) or 'none'}; only in {nm}: "
                    f"{', '.join(only_this[:6]) or 'none'}). Contacts are matched on the names of the ligand's "
                    "atoms, so every model has to name them alike")
        self._classes = symmetry_classes(first)
        #: For two models that superpose: which atom of the second lies where each atom of the first
        #: lies, as indices into the first model's atoms. The identity unless the ligand has a symmetry.
        self._same_place: dict[tuple[int, int], np.ndarray] = {}
        self._made: dict[_Key, dict[int, list[ResolvedInteraction]]] = {}

        #: RMSD of the ligand's heavy atoms between every two models, pockets superposed (angstrom;
        #: ``nan`` where two pockets do not superpose within 3 angstrom, or a model has no structure).
        self.rmsd: np.ndarray
        #: Backbone RMSD of every two pockets after superposition (angstrom; ``nan`` without structures).
        self.pocket_rmsd: np.ndarray
        self.rmsd, self.pocket_rmsd = self._pose_distances()
        #: Per model, its mean ligand RMSD to the others; ``None`` for a model that is no candidate.
        self.mean_rmsd: "list[float | None]" = []
        #: The model that is drawn, and how it came to be: ``"medoid"``, ``"given"`` or ``"first"``.
        self.member: int
        self.chosen_by: str
        self.member, self.chosen_by = self._choose(member)

        self._by_key: dict[_Key, EnsembleContact] = {}
        #: Every contact of the ensemble, by residue, interaction type and ligand atom.
        self.contacts: list[EnsembleContact] = self._match()
        self._table: "FingerprintTable | None" = None
        self._counted: "tuple[list[ResidueShare], list[int]] | None" = None

    # ------------------------------------------------------------------ build
    @classmethod
    def from_structures(cls, models: "Sequence[str | Path | Structure]", ligand: Any, smiles: "str | None" = None,
                        interactions: "Sequence[Any] | str | None" = None, *, names: "Sequence[str] | None" = None,
                        member: "int | str | None" = None, ensemble: "str | None" = None,
                        **options: Any) -> "Ensemble":
        """Build an ensemble from model files and one contact table for each.

        Parameters
        ----------
        models:
            The models, one file each (mmCIF or PDB), or parsed :class:`Structure` objects: those
            of one file with several models are what ``bindview.read_models(path)`` gives. Every
            model is built by :meth:`Diagram.from_structure <bindview.diagram.Diagram.from_structure>`.
        ligand:
            Which residue is the ligand, as ``Diagram.from_structure`` takes it. One selection for
            all models, or a list with one for each (the same ligand can carry another residue
            number in another file).
        smiles:
            The ligand's SMILES, the same for every model.
        interactions:
            One contact table for each model, in the order of the models: a list of
            :class:`~bindview.interactions.Interaction` or dict rows, or the path of a table. An
            empty table is a model that makes no contact, and counts. ``None`` in place of the
            list, or of one table, detects (the optional PLIP extra); ``"distances"`` makes it from the
            positions of the heavy atoms.
        names, member, ensemble:
            As for :class:`Ensemble`. Left out, the models are named after their files where
            that tells them apart.
        options:
            Passed on to ``Diagram.from_structure`` for every model: ``name`` (the ligand's),
            ``plddt``, ``model_confidence``, ``ionize_acids``, ``ligand_as``, ``detail``, ...
        """
        models = list(models)
        if len(models) < 2:
            raise ValueError("an ensemble needs at least two models")
        ligands = list(ligand) if isinstance(ligand, list) else [ligand] * len(models)
        if len(ligands) != len(models):
            raise ValueError(f"{len(ligands)} ligand selections for {len(models)} models: give one for all, "
                             "or a list with one for each")
        if isinstance(interactions, (str, Path)) and interactions != DISTANCES:
            # (A string is a sequence too: its characters would be taken for the tables.)
            raise ValueError(f"interactions takes one contact table for each of the {len(models)} models, in their "
                             f"order, and {str(interactions)!r} is one path: give a list with a table for each model")
        tables = ([interactions] * len(models) if interactions is None or interactions == DISTANCES
                  else list(interactions))
        if len(tables) != len(models) or any(isinstance(t, (Interaction, Mapping)) for t in tables):
            raise ValueError(f"interactions takes one contact table for each of the {len(models)} models, "
                             "in their order (a list of tables, not the rows of one)")
        labels = list(names) if names is not None else _default_names(models)
        if len(labels) != len(models):
            raise ValueError("one name per model is needed")
        diagrams = []
        for model, selection, table, label in zip(models, ligands, tables, labels):
            try:
                diagrams.append(Diagram.from_structure(model, selection, smiles, table, **options))
            except BindviewError as exc:
                raise type(exc)(f"{label}: {exc}") from None
        return cls(diagrams, labels, member=member, ensemble=ensemble)

    # ------------------------------------------------------------------ data
    @property
    def n(self) -> int:
        """How many models the ensemble has."""
        return len(self.diagrams)

    @property
    def table(self) -> FingerprintTable:
        """The models as fingerprints, one row a model: per residue and interaction type, as
        ``Comparison.table`` is for ligands. ``table.consensus()`` gives, per residue, the share of
        the models that contact it at all."""
        if self._table is None:
            self._table = FingerprintTable([Fingerprint.from_diagram(d, nm)
                                            for d, nm in zip(self.diagrams, self.names)])
        return self._table

    def to_csv(self, path: "str | Path | None" = None) -> str:
        """The contacts as a table, one row each (:data:`CONTACT_COLUMNS`); written to ``path`` if given."""
        def num(value: "float | None", digits: int) -> str:
            return "" if value is None else f"{value:.{digits}f}"
        rows = [{"chain": c.chain, "residue_number": str(c.residue_number), "insertion_code": c.insertion_code,
                 "residue_name": c.residue_name, "interaction_type": c.type.value,
                 "ligand_atoms": " ".join(c.ligand_atoms), "n_models": str(c.n_models), "n_total": str(c.n_total),
                 "occupancy": num(c.occupancy, 4), "in_member": "true" if c.in_member else "false",
                 "distance_median": num(c.distance_median, 3), "distance_min": num(c.distance_min, 2),
                 "distance_max": num(c.distance_max, 2), "models": ";".join(self.names[k] for k in c.models)}
                for c in self.contacts]
        return _write_rows(CONTACT_COLUMNS, rows, path)

    def diagram(self, absent: str = "named") -> Diagram:
        """The drawn member as a single diagram whose contacts carry the ensemble as their occupancy.

        Every interaction record of that model gets, as ``occupancy``, the share of the models
        that make the same contact; its distances stay its own. The diagram names the ensemble
        for the legend and says under itself which model it is
        (:attr:`~bindview.diagram.Diagram.remarks`). It renders like any diagram, in either style.
        The member itself is left untouched.

        ``absent`` says what becomes of a contact that the drawn member does not make and other
        models do. ``"named"`` (default): it is not drawn, and the figure says in a sentence how
        many there are and names the most frequent. ``"faint"``: it is drawn, faint, from its
        residue to the atom of the ligand that lies where the other model's atom lies, with the
        share of the models that make it where a distance would stand; the figure has no distance
        for it, and it is not counted among the diagram's contacts. Either way it is a row of
        :attr:`contacts` with ``in_member`` false.
        """
        if absent not in ABSENT:
            raise ValueError(f"absent is one of {', '.join(repr(a) for a in ABSENT)}, not {absent!r}")
        base = self.diagrams[self.member]
        resolved = []
        for r in base.interactions:
            key = contact_key(r, base.ligand, self._classes)
            occupancy = self._by_key[key].occupancy if key is not None else None
            resolved.append(replace(r, record=r.record.with_(occupancy=occupancy)))
        dia = copy.copy(base)
        dia.interactions = resolved
        dia.warnings = list(self.warnings) + list(base.warnings)
        dia.ensemble = self.ensemble
        dia._composed = {}
        which, not_drawn = self._remarks()
        if absent == "faint" and not_drawn is not None:
            dia.elsewhere, lost = self._elsewhere(len(resolved))
            k = len(dia.elsewhere)
            name = self.names[self.member]
            drawn = (f"Faint, with the share of the {self.n} models that make it: {k} contact{'s' if k != 1 else ''} "
                     f"that other models make and {name} does not.")
            if lost:
                drawn += f" {lost} more could not be placed on {name} and {'are' if lost != 1 else 'is'} not drawn."
            dia.remarks = [which]
            dia.elsewhere_said = (drawn, not_drawn) if k else ("", not_drawn)
            if not k:
                dia.remarks.append(not_drawn)
        else:
            dia.remarks = [which, not_drawn or f"No other model makes a contact that {self.names[self.member]} does not."]
        return dia

    @property
    def residues(self) -> list[ResidueShare]:
        """Every residue that any model contacts, with the models that do: the most frequent first
        (:class:`bindview.consensus.ResidueShare`). What :meth:`consensus` draws."""
        return self._by_residue()[0]

    def _by_residue(self) -> tuple[list[ResidueShare], list[int]]:
        if self._counted is None:
            self._counted = residue_shares(self, _symmetries(self.diagrams[0].ligand)[0])
        return self._counted

    def consensus(self, least: float = 0.25, atoms: float = 0.5) -> Diagram:
        """All the models in one diagram, counted by residue (:mod:`bindview.consensus`).

        Every residue stands in a ring as long as the share of the models that contact it, with their
        number in it. A residue that ``atoms`` of them or more contact is drawn as atoms if one of its
        lines has a direction (any contact but a hydrophobic one), and each of its sites (an atom, a
        ring, a charged group) has one line, to the part of the ligand that most of the models have it
        on, with the share of the models that do; a line that fewer than half of the residue's models
        make is not drawn. Any other residue is its ring alone, and the figure says how many of those
        that ``atoms`` of the models contact are.
        Residues that fewer than ``least`` of the models contact are left out, and the figure names them.
        Where the models give one residue number two names (a point mutant among them), the figure has the
        residue of the model whose ligand it draws, counted over the models that have that residue, and
        names the other.

        No single model is drawn, so there are no distances, no buried fraction and no confidence, and
        the pocket is not lined: the lines are what most of the models agree on, and the ligand is
        drawn as in the member :meth:`diagram` draws. It renders like any diagram, in either style;
        ``residues="bubbles"`` draws every residue as its ring alone.
        """
        if not 0.0 <= least <= 1.0:
            raise ValueError(f"least is a share of the models, from 0 to 1, not {least!r}")
        if not 0.0 <= atoms <= 1.0:
            raise ValueError(f"atoms is a share of the models, from 0 to 1, not {atoms!r}")
        base = self.diagrams[self.member]
        if isinstance(base.backend, Polymer):
            raise EnsembleError("the consensus of an ensemble is drawn for a ligand that is drawn as atoms, and this one is "
                                "a chain: diagram() draws one model of it, with the share of every contact")
        counted, turned = self._by_residue()
        every, elsewhere = self._at_one_place(counted)
        shown = [r for r in every if r.share >= least - 1e-9]
        if every and not shown:
            top = every[0]
            raise EnsembleError(f"no residue is contacted by {100 * least:.0f} % of the models; the most frequent is "
                                f"{top.residue_label}, by {top.n_models} of {self.n}: pass a smaller least")
        rows = [x for r in shown for x in lines_of(self, r, r.share >= atoms - 1e-9)]
        for k, row in enumerate(rows):
            row.index = k
        directed = {r.record.residue_id for r in rows if r.type is not InteractionType.HYDROPHOBIC}
        dia = copy.copy(base)
        dia.interactions, dia.elsewhere, dia.elsewhere_said = rows, [], ("", "")
        dia.warnings = list(self.warnings) + list(base.warnings)
        dia.ensemble = self.ensemble
        dia.confidence, dia.pae = None, None
        dia._composed = {}
        dia.shares = {r.residue_id: r for r in shown}
        dia.forms = {f"{r.residue_id[0]}:{r.residue_name.upper()}{r.residue_number}{r.insertion_code}".lstrip(":"):
                     "atoms" if r.share >= atoms - 1e-9 and r.residue_id in directed else "bubbles" for r in shown}
        if any(d.confidence is not None for d in self.diagrams):
            dia.warnings.append("the models have a confidence of their own, and it is not drawn: this is no single model")
        rings = [r for r in shown if r.share >= atoms - 1e-9 and r.residue_id not in directed]
        dia.remarks = self._consensus_remarks(every, shown, turned, atoms, rings)
        if elsewhere:
            name = self.names[self.member]
            named = "; ".join(f"{r.residue_label} ({r.n_models} of {self.n})" for r in elsewhere)
            dia.remarks.append(f"Not drawn: {named}, which other models have where {name} has another residue.")
            dia.warnings.append(f"{named}: at one place with another residue of {name}, whose ligand is drawn (the "
                                "models do not name that residue alike). Counted in Ensemble.residues, not drawn")
        return dia

    def _at_one_place(self, counted: "list[ResidueShare]") -> "tuple[list[ResidueShare], list[ResidueShare]]":
        """The residues that the figure can show, and those it cannot: where the models give one residue number
        two names (a point mutant among them), there are two residues and one place. It goes to the residue
        that the model whose ligand is drawn has there, with the models that residue is counted in; without
        a structure, to the one that more models contact."""
        places: "dict[tuple[str, int, str], list[ResidueShare]]" = {}
        for r in counted:
            places.setdefault(r.residue_id, []).append(r)
        if all(len(group) == 1 for group in places.values()):
            return counted, []
        st = self.diagrams[self.member].structure
        elsewhere: "list[ResidueShare]" = []
        for (chain, number, icode), group in places.items():
            if len(group) > 1:
                key = st.find_residue(chain or None, number, icode) if st is not None else None
                own = st.residue_name(key).upper() if st is not None and key is not None else ""
                kept = next((r for r in group if r.residue_name.upper() == own), group[0])
                elsewhere.extend(r for r in group if r is not kept)
        return [r for r in counted if not any(r is x for x in elsewhere)], elsewhere

    def _consensus_remarks(self, every: "list[ResidueShare]", shown: "list[ResidueShare]", turned: "list[int]",
                           atoms: float, rings: "Sequence[ResidueShare]" = ()) -> list[str]:
        """``rings``: the residues that ``atoms`` of the models contact and that are a ring alone all the same."""
        many = "half" if atoms == 0.5 else f"{100 * atoms:.0f} %"
        unless = ""
        if rings:
            k = len(rings)
            unless = (", unless its lines are hydrophobic contacts, which have no direction "
                      f"({k} here: {'; '.join(r.residue_label for r in rings[:_NAMED])}"
                      + (f"; and {k - _NAMED} more" if k > _NAMED else "") + ")")
        said = [f"Counted by residue, over all {self.n} models: no single one is drawn, and the ligand is drawn as in "
                f"{self.names[self.member]}. A residue that {many} of them or more contact is drawn as atoms{unless}; "
                "a line goes to the part of the ligand that most of them have it on."]
        rest = every[len(shown):]
        if rest:
            k, chains = len(rest), {r.chain for r in every}
            named = "; ".join(format_residue_label(r.residue_name, r.residue_number, r.insertion_code,
                                                   r.chain if len(chains) > 1 else None)
                              + f" ({r.n_models} of {self.n})" for r in rest[:_NAMED])
            more = f"; and {k - _NAMED} more" if k > _NAMED else ""
            said.append(f"Not drawn: {k} residue{'s' if k != 1 else ''} that fewer of the models contact ({named}{more}).")
        if turned:
            k = len(turned)
            said.append(f"The ligand has ends that are alike: in {k} model{'s' if k != 1 else ''} they are named the other "
                        "way round, so that the contacts agree best.")
        return said

    def _elsewhere(self, first_index: int) -> tuple[list[ResolvedInteraction], int]:
        """The contacts the drawn member does not make, as records on that member: each on the atoms
        of its ligand that lie where the atoms of the model that makes the contact lie, and on the
        protein atoms that model names. No distance: the member has none for it. With them, how
        many could not be attached to the member at all."""
        base = self.diagrams[self.member]
        names = self.diagrams[0].ligand.names
        place = {name: k for k, name in enumerate(names)}
        records = []
        for c in self.contacts:
            if c.in_member:
                continue
            key = next(k for k, v in self._by_key.items() if v is c)
            source = c.models[0]                                    # the first model that makes it speaks for it
            made = sorted(self._made[key][source],
                          key=lambda r: r.record.distance if r.record.distance is not None else math.inf)[0]
            lig = self.diagrams[source].ligand
            there = self._same_place.get((source, self.member))     # no common frame: the atoms of the same name
            atoms = [lig.names[i] for i in made.ligand.atoms]       # type: ignore[union-attr]
            if there is not None:
                atoms = [names[int(there[place[a]])] for a in atoms]
            fields: dict[str, Any] = {"type": c.type, "residue_number": c.residue_number,
                                      "residue_name": c.residue_name, "chain": c.chain,
                                      "insertion_code": c.insertion_code, "occupancy": c.occupancy,
                                      "ligand_is_donor": made.record.ligand_is_donor}
            if made.ligand.kind == "atom":                          # type: ignore[union-attr]
                fields["ligand_atom"] = atoms[0]
            else:
                fields["ligand_atoms"] = tuple(atoms)
            if made.protein.kind == "atom" and made.protein.atoms:
                fields["protein_atom"] = made.protein.atoms[0]
            elif made.protein.atoms:
                fields["protein_atoms"] = made.protein.atoms
            records.append(Interaction(**fields))
        if base.structure is None:
            return [], len(records)
        placed = [r for r in resolve_interactions(records, base.ligand, base.structure) if r.ligand is not None]
        for k, r in enumerate(placed):
            r.index, r.absent = first_index + k, True
        return placed, len(records) - len(placed)

    # ------------------------------------------------------------------ the member
    def _pose_distances(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        dist, pocket = np.full((n, n), np.nan), np.full((n, n), np.nan)
        np.fill_diagonal(dist, 0.0)
        np.fill_diagonal(pocket, 0.0)
        if any(d.structure is None for d in self.diagrams):
            self.warnings.append("a model has no structure: the models cannot be superposed, and no member "
                                 "can be chosen by pose")
            return dist, pocket
        first = self.diagrams[0].ligand
        symmetries, complete = _symmetries(first)
        if not complete:
            self.warnings.append(f"the ligand has more than {MAX_SYMMETRIES:,} symmetries: the RMSD between two "
                                 "poses is the smallest over those that were tried, and may be too large")
        # Every model's ligand in the atom order of the first: names are what the models share.
        xyz = []
        for d in self.diagrams:
            place = {name: k for k, name in enumerate(d.ligand.names)}
            xyz.append(np.asarray(d.ligand.coords3d, dtype=float)[[place[name] for name in first.names]])
        fitted = pocket_transforms_between(self.diagrams)       # every model is read once
        for i in range(n - 1):
            for j in range(i + 1, n):
                frame, fit = fitted[(i, j)]
                if fit is None:
                    raise EnsembleError(
                        f"{self.names[j]} and {self.names[i]} have no pocket in common to superpose: fewer than "
                        "four residues around the ligand are in both, or under 70 % of them carry the same name. "
                        "That is another protein, another numbering or other chain names, not another model of "
                        "this complex. Residues are matched by chain and number: name the chains alike first")
                pocket[i, j] = pocket[j, i] = float(fit)
                if frame is None:
                    continue                             # the pockets differ by more than the fit may leave
                rot, shift = frame
                moved = xyz[j] @ rot.T + shift
                delta = xyz[i][None, :, :] - moved[symmetries]
                each = np.sqrt((delta ** 2).sum(axis=2).mean(axis=1))
                best = int(each.argmin())
                dist[i, j] = dist[j, i] = float(each[best])
                there = symmetries[best]                 # atom q of model i lies where atom there[q] of model j lies
                back = np.empty_like(there)
                back[there] = np.arange(len(there))
                self._same_place[(i, j)], self._same_place[(j, i)] = there, back
        return dist, pocket

    def _choose(self, member: "int | str | None") -> tuple[int, str]:
        n = self.n
        need = math.ceil((n - 1) / 2)                    # a distance to at least half of the others
        means: "list[float | None]" = []
        for i in range(n):
            have = [float(self.rmsd[i, j]) for j in range(n) if j != i and not math.isnan(self.rmsd[i, j])]
            means.append(sum(have) / len(have) if have and len(have) >= need else None)
        self.mean_rmsd = means
        apart = [self.names[i] for i in range(n) if means[i] is None]
        candidates = [i for i in range(n) if means[i] is not None]
        if apart and candidates:
            self.warnings.append(f"{', '.join(apart)}: the pocket does not superpose on that of at least half of "
                                 "the other models within 3 Å. Counted for occupancy; not a candidate for the "
                                 "member that is drawn")
        if member is not None:
            if isinstance(member, str):
                if member not in self.names:
                    raise ValueError(f"no model is named {member!r}; the models are {', '.join(self.names)}")
                return self.names.index(member), "given"
            if not -n <= int(member) < n:
                raise ValueError(f"member {member} is not one of the {n} models")
            return int(member) % n, "given"
        if n == 2 or not candidates:
            if not candidates and not any("no structure" in w for w in self.warnings):
                self.warnings.append("no member could be chosen by pose: the pockets of the models do not "
                                     "superpose. The first model is drawn")
            return 0, "first"
        best = min(means[i] for i in candidates)                               # type: ignore[type-var]
        return next(i for i in candidates if means[i] <= best + _TIE), "medoid"  # type: ignore[operator]

    # ------------------------------------------------------------------ the contacts
    def _match(self) -> list[EnsembleContact]:
        first = self.diagrams[0].ligand
        place = {name: k for k, name in enumerate(first.names)}
        made: dict[_Key, dict[int, list[ResolvedInteraction]]] = {}
        for k, dia in enumerate(self.diagrams):
            loose = 0
            for r in dia.interactions:
                key = contact_key(r, dia.ligand, self._classes)
                if key is None:
                    loose += 1
                    continue
                made.setdefault(key, {}).setdefault(k, []).append(r)
            if loose:
                self.warnings.append(f"{self.names[k]}: {loose} record{'s' if loose != 1 else ''} could not be "
                                     "attached to the ligand and "
                                     f"{'are' if loose != 1 else 'is'} not counted")
        self._made = made
        out = []
        for key, by_model in made.items():
            chain, number, icode, resname, kind, _site = key
            models = tuple(sorted(by_model))
            distances = []
            atoms: set[str] = set()
            for k in models:
                reported = [r.record.distance for r in by_model[k] if r.record.distance is not None]
                distances.append(min(reported) if reported else None)
                lig = self.diagrams[k].ligand
                atoms |= {lig.names[i] for r in by_model[k] for i in r.ligand.atoms}       # type: ignore[union-attr]
            contact = EnsembleContact(chain, number, icode, resname, kind, tuple(sorted(atoms, key=place.__getitem__)),
                                      models, self.n, self.member in by_model, tuple(distances))
            self._by_key[key] = contact
            out.append(contact)
        out.sort(key=lambda c: (c.chain, c.residue_number, c.insertion_code, _TYPE_RANK[c.type],
                                tuple(place[a] for a in c.ligand_atoms)))
        return out

    # ------------------------------------------------------------------ what the figure says
    def _remarks(self) -> "tuple[str, str | None]":
        """Which model the figure shows, and the sentence on the contacts of other models that it
        does not draw (``None`` where there is none)."""
        name, n = self.names[self.member], self.n
        mean = self.mean_rmsd[self.member]
        if self.chosen_by == "medoid":
            # Of the candidates, and over the models its pocket superposes on: all of them, unless a warning says not.
            candidates = sum(m is not None for m in self.mean_rmsd)
            over = sum(not math.isnan(self.rmsd[self.member, k]) for k in range(n)) - 1
            among = (f"of the {n} models" if candidates == n else
                     f"of the {candidates} of the {n} models whose pocket superposes on that of at least half of the "
                     "others")
            how = ("pockets superposed" if over == n - 1 else
                   f"over the {over} models whose pocket superposes on its own")
            which = (f"Shown is {name}: {among}, the one whose ligand lies nearest to the others' "
                     f"(RMSD {mean:.2f} Å on average, {how}).")
        elif self.chosen_by == "given":
            which = f"Shown is {name} of the {n} models, as asked."
        else:
            which = f"Shown is {name}, the first of the {n} models."
        which += " Distances are that model's."
        missing = sorted((c for c in self.contacts if not c.in_member), key=lambda c: -c.n_models)   # stable
        if not missing:
            return which, None
        chains = {c.chain for c in self.contacts}

        def said(c: EnsembleContact) -> str:
            label = format_residue_label(c.residue_name, c.residue_number, c.insertion_code,
                                         c.chain if len(chains) > 1 else None)
            to = (f" to {c.ligand_atoms[0]}" if len(c.ligand_atoms) == 1
                  and c.type is not InteractionType.HYDROPHOBIC else "")
            return f"{label} {c.type.label.lower()}{to} ({c.n_models} of {n})"
        k = len(missing)
        named = "; ".join(said(c) for c in missing[:_NAMED])
        more = f"; and {k - _NAMED} more" if k > _NAMED else ""
        return which, (f"Not drawn: {k} contact{'s' if k != 1 else ''} that other models make and {name} does not "
                       f"({named}{more}).")
