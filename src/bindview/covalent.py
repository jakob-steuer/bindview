"""A ligand that is bound to its target by a covalent bond.

A contact table has no such row: what finds contacts looks for pairs of atoms that are near each other and not
bonded. A covalent inhibitor is bonded, and its figure has to say so. ``Diagram.from_structure(covalent=True)``
looks for the bond in the model:

* **declared**: a ``_struct_conn`` row of a covalent kind (mmCIF), a ``LINK`` record (PDB);
* **by distance**: an atom of the ligand within :data:`TOLERANCE` ångström of the sum of the covalent radii of a
  nitrogen, oxygen, sulphur or selenium of a residue. A model that was made (a co-fold, an ensemble) declares
  nothing.

The bond becomes a row of the kind ``covalent``, drawn as a bond on a highlight from atom to atom. A table can
bring such rows itself. Two things follow from the bond, for a row that was found and for one that was brought:

* a contact between atoms that the bond joins or brings within three bonds of each other (the cysteine's carbon and
  the carbon it is bound through) is the bond's and no contact, and is left out;
* the ligand is drawn as it is bound. A SMILES is as a rule that of the free ligand: where the atom that is bound
  has a multiple bond there, and the model has that bond at the length of a lower order (the double bond of an
  acrylamide at 1.5 Å), it is drawn at the lower order (:func:`as_bound`).

Off unless it is asked for: what is drawn of a model does not change by itself.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
from rdkit import Chem

from .chem import covalent_radius, is_metal, quiet_rdkit
from .exposure import WATER_NAMES
from .interactions import Interaction, InteractionType
from .ligand import Ligand
from .structure import Atom, ResidueKey, Structure

if TYPE_CHECKING:  # pragma: no cover
    from .resolve import ResolvedInteraction

__all__ = ["CovalentBond", "covalent_bonds", "as_bound", "drop_bonded", "TOLERANCE"]

T = InteractionType
#: How much longer than the sum of the covalent radii a bond that the file does not declare may be, in ångström.
TOLERANCE = 0.25
#: ... and how long one is at most that it declares.
DECLARED_AT_MOST = 2.6
#: A contact between atoms that are at most this many bonds apart is no contact.
BONDED = 3
_NUCLEOPHILES = frozenset({"N", "O", "S", "Se"})
#: The two atoms of the bond that joins a residue to the next of its chain: of a protein, of a nucleic acid.
_CHAIN_BONDS = ({"N", "C"}, {"P", "O3'"})
#: The length above which a bond of the first order (or bond) is taken for one of the second: halfway between the two.
_LENGTHS: "dict[frozenset[str], dict[int, float]]" = {
    frozenset({"C"}): {3: 1.27, 2: 1.43},
    frozenset({"C", "N"}): {3: 1.22, 2: 1.37},
    frozenset({"C", "O"}): {2: 1.32},
    frozenset({"C", "S"}): {2: 1.72},
}


@dataclass(frozen=True)
class CovalentBond:
    """A bond between an atom of the ligand and an atom of a residue."""

    ligand_atom: str
    residue: ResidueKey
    resname: str
    atom: str
    distance: float
    declared: bool

    def as_interaction(self) -> Interaction:
        return Interaction(type=T.COVALENT, residue_number=self.residue.number, residue_name=self.resname,
                           chain=self.residue.chain, insertion_code=self.residue.icode, protein_atom=self.atom,
                           ligand_atom=self.ligand_atom, distance=round(self.distance, 2))

    def __str__(self) -> str:
        how = "declared by the file" if self.declared else "found by distance"
        return f"{self.ligand_atom} is bound to {self.resname.capitalize()}{self.residue.number} {self.atom} " \
               f"({self.distance:.2f} Å, {how})"


def covalent_bonds(structure: Structure, ligand: ResidueKey) -> list[CovalentBond]:
    """The covalent bonds between a residue of the structure and everything else in it, a water and a metal left
    out, and the bonds of a chain the residue is part of (the peptide bond or the phosphodiester to the residue
    before it and to the one after it): the declared ones first, then those by distance."""
    own = {a.name: a for a in structure.residue_atoms(ligand) if a.element not in ("H", "D")}
    out: list[CovalentBond] = []
    chain = [k for k in structure.residues() if k.chain == ligand.chain]
    beside: "set[ResidueKey]" = set()                                  # the residues before and after it in the file
    if ligand in chain:
        at = chain.index(ligand)
        beside = set(chain[max(at - 1, 0):at + 2]) - {ligand}

    def add(mine: Atom, other: Atom, declared: bool) -> None:
        if is_metal(mine.element) or is_metal(other.element) or other.resname.upper() in WATER_NAMES:
            return
        if other.residue in beside and {mine.name.replace("*", "'"), other.name.replace("*", "'")} in _CHAIN_BONDS:
            return                                                     # the next residue of its own chain
        if any(b.ligand_atom == mine.name and b.residue == other.residue and b.atom == other.name for b in out):
            return
        d = float(np.linalg.norm(np.array(mine.xyz) - np.array(other.xyz)))
        if declared and d > DECLARED_AT_MOST:
            return
        out.append(CovalentBond(mine.name, other.residue, other.resname, other.name, d, declared))

    for link in structure.links:
        for (me, my_atom), (it, its_atom) in (((link.a, link.a_atom), (link.b, link.b_atom)),
                                              ((link.b, link.b_atom), (link.a, link.a_atom))):
            if me == ligand and it != ligand and my_atom in own:
                other = structure.atom_by_name(it, its_atom)
                if other is not None:
                    add(own[my_atom], other, True)
    others = [a for a in structure.atoms if a.residue != ligand and a.element in _NUCLEOPHILES
              and a.resname.upper() not in WATER_NAMES]
    if own and others:
        there = np.array([a.xyz for a in others], dtype=float)
        bound = {(b.ligand_atom) for b in out}
        for mine in own.values():
            if mine.name in bound:
                continue
            d = np.linalg.norm(there - np.array(mine.xyz, dtype=float), axis=1)
            k = int(np.argmin(d))
            if d[k] <= covalent_radius(mine.element) + covalent_radius(others[k].element) + TOLERANCE:
                add(mine, others[k], False)
    return out


def as_bound(lig: Ligand, atom: int) -> "str | None":
    """Draw the ligand as it is bound through ``atom``: the multiple bond there at the order its length in the
    model says. Changes ``lig.mol``; returns what was done in words, or ``None``."""
    if lig.bond_source != "template":
        return None
    mol = Chem.RWMol(lig.mol)
    for bond in mol.GetAtomWithIdx(atom).GetBonds():
        order = {Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}.get(bond.GetBondType())
        other = bond.GetOtherAtomIdx(atom)
        if order is None:
            continue
        longest = _LENGTHS.get(frozenset({lig.element(atom), lig.element(other)}), {}).get(order)
        length = float(np.linalg.norm(lig.coords3d[atom] - lig.coords3d[other]))
        if longest is None or length <= longest:
            continue
        bond.SetBondType(Chem.BondType.SINGLE if order == 2 else Chem.BondType.DOUBLE)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        for end in (atom, other):
            for b in mol.GetAtomWithIdx(end).GetBonds():
                b.SetBondDir(Chem.BondDir.NONE)
        # The atom that is bound has the new bond in place of the old one's; the other end takes a hydrogen.
        far = mol.GetAtomWithIdx(other)
        far.SetNumExplicitHs(far.GetNumExplicitHs() + 1)
        out = mol.GetMol()
        with quiet_rdkit():
            Chem.SanitizeMol(out)
            Chem.AssignStereochemistry(out, cleanIt=True, force=True)
        lig.mol = out
        was, now = ("double", "single") if order == 2 else ("triple", "double")
        return (f"drawn as bound: the {was} bond {lig.names[atom]}–{lig.names[other]} of the SMILES is {length:.2f} "
                f"Å long in the model, and is drawn {now}")
    bound = mol.GetAtomWithIdx(atom)
    if bound.GetSymbol() in _NUCLEOPHILES and bound.GetNumExplicitHs() > 0 and not bound.GetIsAromatic():
        bound.SetNumExplicitHs(bound.GetNumExplicitHs() - 1)           # its hydrogen is where the bond is now
        out = mol.GetMol()
        with quiet_rdkit():
            out.UpdatePropertyCache(strict=False)
        lig.mol = out
        return f"drawn as bound: {lig.names[atom]} without the hydrogen that its bond to the target took the place of"
    return None


def _steps(pairs: "Iterable[tuple[int, int]]", start: int) -> "dict[int, int]":
    """How many bonds away from ``start`` every atom is that bonds lead to."""
    near: "dict[int, list[int]]" = {}
    for a, b in pairs:
        near.setdefault(a, []).append(b)
        near.setdefault(b, []).append(a)
    out, queue = {start: 0}, deque([start])
    while queue:
        a = queue.popleft()
        for b in near.get(a, ()):
            if b not in out:
                out[b] = out[a] + 1
                queue.append(b)
    return out


def drop_bonded(resolved: "Sequence[ResolvedInteraction]", lig: Ligand, structure: "Structure | None"
                ) -> "tuple[list[ResolvedInteraction], list[ResolvedInteraction]]":
    """The contacts without those between atoms that a covalent bond of the list joins or brings within
    :data:`BONDED` bonds of each other; and those."""
    bonds = [r for r in resolved if r.type is T.COVALENT and r.ligand is not None and r.residue is not None]
    if not bonds or structure is None:
        return list(resolved), []
    mine = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in lig.mol.GetBonds()]
    kept: "list[ResolvedInteraction]" = []
    dropped: "list[ResolvedInteraction]" = []
    for r in resolved:
        apart = None
        if r.type is not T.COVALENT and r.ligand is not None and r.residue is not None:
            for b in bonds:
                if b.residue != r.residue or not b.protein.atoms or not r.protein.atoms:
                    continue
                atoms = [a for a in structure.residue_atoms(r.residue) if a.element not in ("H", "D")]
                index = {a.name: k for k, a in enumerate(atoms)}
                xyz = np.array([a.xyz for a in atoms], dtype=float)
                d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
                theirs = [(i, j) for i in range(len(atoms)) for j in range(i + 1, len(atoms))
                          if d[i, j] <= 1.3 * (covalent_radius(atoms[i].element) + covalent_radius(atoms[j].element))]
                if b.protein.atoms[0] not in index:
                    continue
                here = _steps(mine, b.ligand.atom)                     # type: ignore[union-attr]
                there = _steps(theirs, index[b.protein.atoms[0]])
                steps = [here[i] + 1 + there[index[n]] for i in r.ligand.atoms if i in here
                         for n in r.protein.atoms if n in index and index[n] in there]
                if steps and (apart is None or min(steps) < apart):
                    apart = min(steps)
        (dropped if apart is not None and apart <= BONDED else kept).append(r)
    return kept, dropped
