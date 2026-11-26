"""How a ligand is drawn: as atoms, or as a chain of residues.

A :class:`~bindview.ligand.Ligand` is the molecule as it was built from the
structure file: atoms with their names, bonds, coordinates. A *backend* decides
what a reader is shown of it and what a contact attaches to:

* :class:`SmallMolecule` -- every atom and bond, as a chemist draws a
  structure.
* :class:`Polymer` -- the ligand as a chain of its residues, each a box on a
  backbone; whatever is not a residue (a fatty acid, a cap) stays atoms and is
  shown attached. A contact lands on the residue whose atom makes it.

Which one a ligand gets is decided by **what the molecule is, not by how big it
is** (:func:`bindview.peptide.decide`): a cobalamin of ninety atoms has no chain
of amino acids and is drawn as atoms; a peptide of twenty residues is drawn as
a chain. A caller can overrule the decision either way.

The layout engine, the audit, the graph and the two styles see a
:class:`~bindview.depict.Depiction` and do not know which backend made it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

from rdkit import Chem

from .chem import quiet_rdkit
from .confidence import Confidence, Look
from .depict import Depiction, depict_ligand
from .ligand import Ligand
from .peptide import Decision, Peptide, decide, decide_given, perceive_peptide
from .polymer import depict_polymer
from .scene import Group
from .theme import Metrics, Palette

if TYPE_CHECKING:  # pragma: no cover
    from .graph import Graph

__all__ = ["LigandBackend", "SmallMolecule", "Polymer", "choose_backend", "BACKENDS"]

#: What ``ligand_as`` accepts.
BACKENDS = ("auto", "atoms", "chain")

NUMBERING_NOTE = ("Residues of the ligand are numbered by their position in its own chain, counted from its "
                  "N-terminus as read from its bonds; these are not positions in any sequence database.")


@runtime_checkable
class LigandBackend(Protocol):
    """What the diagram needs from the way a ligand is drawn."""

    kind: str                       # "atoms" | "chain"
    ligand: Ligand
    decision: Decision              # why this backend, in a sentence a figure's notes can quote

    def depict(self, metrics: Metrics, method: str = "auto") -> Depiction:
        """Geometry of the ligand, laid out afresh."""

    def depict_from(self, metrics: Metrics, graph: "Graph") -> Depiction:
        """Geometry of the ligand where a graph has it."""

    def draw(self, dep: Depiction, palette: Palette, metrics: Metrics,
             confidence: "Confidence | None" = None, by_colour: bool = False, by_bond: bool = False
             ) -> tuple[Group, set[int]]:
        """The ligand as a scene group, and the confidence levels that occur in it. ``by_colour``: the
        figure shows pLDDT by colour, so nothing of the ligand is drawn weaker for it; ``by_bond``: the
        bonds of a ligand that is drawn as atoms have the colour."""

    def bands(self, dep: Depiction, confidence: "Confidence") -> "list[str | None]":
        """The pLDDT band of every depicted atom, for a figure that colours by it."""

    def obstacles(self, dep: Depiction) -> list[tuple[float, float, float]]:
        """Discs over ink of the ligand that is neither an atom label nor a bond."""

    def footnotes(self, dep: "Depiction | None" = None) -> list[str]:
        """Sentences the figure must carry because of how the ligand is drawn (``dep``: as it was drawn)."""


class SmallMolecule:
    """The ligand as atoms and bonds, as a chemist draws a structure."""

    kind = "atoms"

    def __init__(self, ligand: Ligand, decision: Decision | None = None) -> None:
        self.ligand = ligand
        self.decision = decision or Decision(False, "it is drawn as atoms")

    def depict(self, metrics: Metrics, method: str = "auto") -> Depiction:
        return depict_ligand(self.ligand, metrics, method=method)

    def depict_from(self, metrics: Metrics, graph: "Graph") -> Depiction:
        from .graphbuild import depiction_from

        return depiction_from(self.ligand, metrics, graph)

    def draw(self, dep: Depiction, palette: Palette, metrics: Metrics,
             confidence: "Confidence | None" = None, by_colour: bool = False, by_bond: bool = False
             ) -> tuple[Group, set[int]]:
        from .styles.ligand_draw import atom_looks, draw_ligand

        looks: "Sequence[Look] | None" = (atom_looks(dep, confidence.ligand_atoms, by_colour)
                                          if confidence is not None else None)
        bands = [lk.band for lk in looks] if by_bond and looks else None
        return draw_ligand(dep, palette, metrics, looks, bands), ({lk.level for lk in looks} if looks else set())

    def bands(self, dep: Depiction, confidence: "Confidence") -> "list[str | None]":
        """The pLDDT band of every depicted atom, for a figure that colours by it; ``None`` where the
        model gives the atom no value."""
        from .styles.ligand_draw import atom_looks

        return [lk.band for lk in atom_looks(dep, confidence.ligand_atoms, True)]

    def obstacles(self, dep: Depiction) -> list[tuple[float, float, float]]:
        return []

    def footnotes(self, dep: "Depiction | None" = None) -> list[str]:
        return []


class Polymer:
    """The ligand as a chain of residues, with what is not a residue drawn as atoms."""

    kind = "chain"

    def __init__(self, ligand: Ligand, peptide: Peptide, decision: Decision | None = None) -> None:
        self.ligand = ligand
        self.peptide = peptide
        self.decision = decision or decide(peptide)
        #: Set when the chain is the spine of an interface figure (:mod:`bindview.interface`): what the
        #: legend and the footnotes say then (``legend``, ``notes``) and what goes into the graph (``meta``).
        self.interface: Any = None
        #: Set when the residues are not drawn in a column but where they lie on the plane through
        #: them (:class:`bindview.plane.Arrangement`).
        self.arrangement: Any = None
        #: Which atoms of which residue stand beside its box as atoms (position -> atoms), and on
        #: which side its side chain does (position -> +1 or -1). The diagram sets both from the
        #: contacts it draws and from ``peptide.detail`` before the chain is depicted
        #: (:func:`bindview.polymer.shown_atoms`, :func:`bindview.polymer.piece_sides`).
        self.shown: dict[int, tuple[int, ...]] = {}
        self.sides: dict[int, int] = {}
        #: What is written on them (``RenderOptions.atom_labels``).
        self.atom_labels = "names"

    def depict(self, metrics: Metrics, method: str = "auto") -> Depiction:
        return depict_polymer(self.ligand, self.peptide, metrics, method=method, plane=self.arrangement,
                              shown=self.shown, sides=self.sides, labels=self.atom_labels)

    def depict_from(self, metrics: Metrics, graph: "Graph") -> Depiction:
        from .graphbuild import polymer_positions

        units, atoms = polymer_positions(graph)
        scale = None
        if self.arrangement is not None:
            stored = (graph.meta.get("interface") or {}).get("plane") or {}
            scale = stored.get("px_per_angstrom")
        return depict_polymer(self.ligand, self.peptide, metrics, unit_xy=units, exact=atoms,
                              plane=self.arrangement, scale=scale, shown=self.shown, sides=self.sides, labels=self.atom_labels)

    def draw(self, dep: Depiction, palette: Palette, metrics: Metrics,
             confidence: "Confidence | None" = None, by_colour: bool = False, by_bond: bool = False
             ) -> tuple[Group, set[int]]:
        from .styles.ligand_draw import atom_looks
        from .styles.polymer_draw import draw_polymer, unit_looks

        if confidence is None:
            return draw_polymer(dep, palette, metrics, deposited=self.deposited), set()
        looks = atom_looks(dep, confidence.ligand_atoms, by_colour)
        units = unit_looks(dep, confidence.ligand_atoms, by_colour)
        if by_colour:
            # Nothing is drawn weaker for its pLDDT: the boxes are filled, and the atoms stand on colour.
            return draw_polymer(dep, palette, metrics, looks, units, deposited=self.deposited,
                                by_colour=True), {0}
        outside = {looks[a.idx].level for a in dep.atoms if dep.unit_of(a.idx) is None}
        # An atom that stands beside its residue's box: at its own value where the model gives the atoms
        # of a residue values of their own, at its residue's where it gives one for the residue.
        own = {i: (looks[i] if confidence.per_atom else units[u.index]) for u in dep.units for i in u.beside}
        return (draw_polymer(dep, palette, metrics, looks, units, deposited=self.deposited, own=own),
                outside | {u.level for u in units} | {lk.level for lk in own.values()})

    def bands(self, dep: Depiction, confidence: "Confidence") -> "list[str | None]":
        """The pLDDT band of every depicted atom, for a figure that colours by it: an atom's own, and
        for an atom beside its residue's box the residue's where the model gives one value for a residue."""
        from .styles.ligand_draw import atom_looks
        from .styles.polymer_draw import unit_looks

        bands = [lk.band for lk in atom_looks(dep, confidence.ligand_atoms, True)]
        if not confidence.per_atom:
            units = unit_looks(dep, confidence.ligand_atoms, True)
            for u in dep.units:
                for i in u.beside:
                    bands[i] = units[u.index].band
        return bands

    def obstacles(self, dep: Depiction) -> list[tuple[float, float, float]]:
        from .styles.polymer_draw import chain_obstacles

        return chain_obstacles(dep)

    @property
    def deposited(self) -> bool:
        """Are the residues' numbers those of the structure file (and not a count made here)?"""
        return self.peptide.numbering == "deposited"

    def footnotes(self, dep: "Depiction | None" = None) -> list[str]:
        if self.interface is not None:
            return list(self.interface.said(dep))
        if self.deposited:
            chain = f" (chain {self.ligand.chain})" if self.ligand.chain else ""
            return [f"Residues of the ligand carry the names and numbers of the structure file{chain}."]
        return [NUMBERING_NOTE]

    def legend_text(self) -> str:
        """What the legend says a box is, and whose its number is."""
        if self.interface is not None:
            return str(self.interface.legend)
        return ("Residue of the ligand, numbered as in the structure file" if self.deposited
                else "Residue of the ligand, numbered from its own N-terminus")

    def residue_label(self, atom: int) -> str | None:
        """``Lys20`` for an atom of the chain, ``None`` for one outside it."""
        res = self.peptide.residue_of(atom)
        return None if res is None else f"{res.code3}{res.shown}"

    def part_of(self, atoms: Sequence[int]) -> str:
        """``backbone``, ``side chain`` or ``outside the chain``: where in the ligand these atoms are.
        Of a nucleotide the base is what a side chain is of an amino acid, and is called ``base``."""
        residues = [self.peptide.residue_of(a) for a in atoms]
        if any(r is None for r in residues):
            return "outside the chain"
        if all(r.is_backbone(a) for r, a in zip(residues, atoms) if r is not None):
            return "backbone"
        return "base" if all(r is not None and r.kind == "nucleotide" for r in residues) else "side chain"


def _reference(ligand: Ligand) -> Chem.Mol | None:
    """The molecule the ligand was specified as, for its stereochemistry: a model can invert a centre."""
    if not ligand.smiles or ligand.bond_source != "template":
        return None
    with quiet_rdkit():
        return Chem.MolFromSmiles(ligand.smiles)


def choose_backend(ligand: Ligand, ligand_as: str = "auto") -> "SmallMolecule | Polymer":
    """The backend for a ligand: by what it is (``auto``), or as the caller says.

    ``chain`` for a ligand that has no chain of residues at all is an error:
    there is nothing to draw as one.
    """
    if ligand_as not in BACKENDS:
        raise ValueError(f"ligand_as must be one of {', '.join(BACKENDS)}, not {ligand_as!r}")
    if ligand_as == "atoms" and ligand.n_atoms < 2:
        return SmallMolecule(ligand)
    peptide: Peptide | None = None
    if ligand.given:
        # A chain of its own in the structure file: its residues are given, not perceived.
        from .chains import given_peptide

        with quiet_rdkit():
            peptide = given_peptide(ligand.mol, ligand.names, ligand.given)
        decision = decide_given(peptide)
    else:
        if ligand.bond_source != "single-atom":
            with quiet_rdkit():
                peptide = perceive_peptide(ligand.mol, stereo_from=_reference(ligand))
        decision = decide(peptide)
    if ligand_as == "atoms":
        return SmallMolecule(ligand, Decision(False, "it was asked to be drawn as atoms", decision.longest_run,
                                              decision.recognised, found=decision.polymer))
    if ligand_as == "chain":
        if peptide is None:
            raise ValueError("this ligand cannot be drawn as a chain: " + decision.reason)
        return Polymer(ligand, peptide, Decision(True, "it was asked to be drawn as a chain", decision.longest_run,
                                                 decision.recognised, found=decision.polymer))
    if decision.polymer and peptide is not None:
        return Polymer(ligand, peptide, decision)
    return SmallMolecule(ligand, decision)


def describe(backend: Any) -> dict[str, Any]:
    """The backend as plain data, for a result or a graph."""
    out: dict[str, Any] = {"drawn_as": backend.kind, "why": backend.decision.reason}
    if isinstance(backend, Polymer):
        pep = backend.peptide
        out.update({"residues": len(pep), "sequence": pep.sequence, "cyclic": pep.cyclic,
                    "numbering": ("as deposited in the structure file" if backend.deposited
                                  else "position in the ligand's own chain, from its N-terminus"),
                    "numbering_origin": pep.numbering,
                    "atoms_outside_the_chain": pep.n_atoms - pep.in_chain})
    return out
