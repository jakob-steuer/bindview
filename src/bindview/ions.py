"""An ion between a ligand and its target, drawn as the bridge it is.

A zinc that a sulphonamide binds and three histidines hold, a magnesium between two phosphates and an aspartate: a
table of contacts has the ion's contact with the ligand and nothing of what holds it, and a figure of that table
shows an ion in the air. ``Diagram.from_structure(ions=True)`` takes an ion that the model has and that touches both
(:func:`bridges`) into the drawing of the ligand, on the atoms of it that hold it, and gives each residue that holds
it a contact with it, with its length. A hydrogen bond of the table between two atoms that hold the same ion is not
one (they are 3 Å apart because the ion is between them) and is left out, and so is one of a histidine's nitrogen
that holds an ion, be it one between the two or a metal of the ligand itself (a heme's iron): it has neither a
hydrogen nor a lone pair left.

Where the model has no ion, none is made up. Off unless it is asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from rdkit import Chem

from .chem import is_metal
from .exposure import WATER_NAMES
from .interactions import Interaction, InteractionType
from .ligand import Ligand
from .structure import Atom, ResidueKey, Structure

__all__ = ["IonBridge", "bridges", "holders", "with_ions", "REACH"]

T = InteractionType
#: How far a donor is from an ion that it holds, at most, in ångström.
REACH = {"Mg": 2.6, "Mn": 2.7, "Zn": 2.7, "Fe": 2.7, "Co": 2.7, "Ni": 2.7, "Cu": 2.7, "Ca": 2.9, "Na": 2.9, "K": 3.3,
         "Cd": 2.8}
_REACH = 2.8
#: The charge of an ion by the name its file gives it (in capitals), and by its element: the alkali metals, silver,
#: thallium and gold have one, aluminium and its group, chromium and the rare earths three. Any other metal has two.
_CHARGE = {"FE": 3, "FE2": 2, "CU1": 1, "MN3": 3, "3CO": 3, "AU3": 3, "EU": 2, "YB2": 2,
           "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Ag": 1, "Tl": 1, "Au": 1,
           "Al": 3, "Ga": 3, "In": 3, "Bi": 3, "Cr": 3, "Sc": 3, "Y": 3, "La": 3, "Ce": 3, "Pr": 3, "Nd": 3, "Sm": 3,
           "Eu": 3, "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3}
_DONORS = ("N", "O", "S")


@dataclass(frozen=True)
class IonBridge:
    """An ion of the model, and what holds it."""

    ion: Atom
    #: Its name in the drawing of the ligand.
    name: str
    ligand: "tuple[tuple[str, float], ...]"           # (atom name, distance), nearest first
    protein: "tuple[tuple[Atom, float], ...]"

    @property
    def charge(self) -> int:
        return _CHARGE.get(self.ion.resname.upper(), _CHARGE.get(self.ion.element, 2))

    def records(self) -> list[Interaction]:
        return [Interaction(type=T.METAL, residue_number=a.resnum, residue_name=a.resname, chain=a.chain,
                            insertion_code=a.icode, protein_atom=a.name, ligand_atom=self.name, distance=round(d, 2))
                for a, d in self.protein]

    def __str__(self) -> str:
        held = ", ".join(f"{a.resname.capitalize()}{a.resnum} {a.name}" for a, _ in self.protein)
        return f"{self.ion.element} is between {', '.join(n for n, _ in self.ligand)} of the ligand and {held}"


def bridges(structure: Structure, ligand: ResidueKey) -> list[IonBridge]:
    """The ions of the structure that a donor of the ligand and a donor of another residue hold."""
    own = [a for a in structure.residue_atoms(ligand) if a.element in _DONORS]
    others = [a for a in structure.atoms if a.element in _DONORS and a.residue != ligand
              and a.resname.upper() not in WATER_NAMES]
    if not own or not others:
        return []
    taken = {a.name for a in structure.residue_atoms(ligand)}
    mine = np.array([a.xyz for a in own], dtype=float)
    theirs = np.array([a.xyz for a in others], dtype=float)
    out = []
    for ion in structure.atoms:
        if not is_metal(ion.element) or ion.residue == ligand or len(structure.residue_atoms(ion.residue)) != 1:
            continue
        at, reach = np.array(ion.xyz, dtype=float), REACH.get(ion.element, _REACH)
        d, e = np.linalg.norm(mine - at, axis=1), np.linalg.norm(theirs - at, axis=1)
        held = sorted(((own[i].name, float(d[i])) for i in np.nonzero(d <= reach)[0]), key=lambda t: t[1])
        by = sorted(((others[i], float(e[i])) for i in np.nonzero(e <= reach)[0]), key=lambda t: t[1])
        if not held or not by:
            continue
        name, k = ion.element.upper(), 1
        while name in taken:
            k += 1
            name = f"{ion.element.upper()}{k}"
        taken.add(name)
        out.append(IonBridge(ion, name, tuple(held), tuple(by)))
    return out


def with_ions(structure: Structure, ligand: ResidueKey, smiles: "str | None", found: "Sequence[IonBridge]"
              ) -> "tuple[Structure, str | None]":
    """The structure with the ions of ``found`` as atoms of the ligand, and the SMILES with them."""
    names = {(b.ion.residue, b.ion.name): b.name for b in found}
    resname = structure.residue_name(ligand)
    atoms = [replace(a, name=names[(a.residue, a.name)], resname=resname, chain=ligand.chain, resnum=ligand.number,
                     icode=ligand.icode) if (a.residue, a.name) in names else a for a in structure.atoms]
    if smiles:
        smiles = smiles + "".join(f".[{b.ion.element}+{b.charge if b.charge > 1 else ''}]" for b in found)
    return replace(structure, atoms=atoms, _by_residue={}, _coords=None), smiles


def seat(lig: Ligand, found: "Sequence[IonBridge]") -> None:
    """Tell the ligand which of its atoms are ions between it and the protein, and what of it holds them."""
    index = {n: i for i, n in enumerate(lig.names)}
    ions = {index[b.name]: b for b in found}
    lig.bridging = sorted(ions)
    lig.coordination = sorted([(m, d) for m, d in lig.coordination if m not in ions]
                              + [(m, index[n]) for m, b in ions.items() for n, _ in b.ligand if n in index])
    loose = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in lig.mol.GetBonds()
             if b.GetBeginAtomIdx() in ions or b.GetEndAtomIdx() in ions]
    if loose:                                  # without a template the ion's neighbours were taken for bonds
        mol = Chem.RWMol(lig.mol)
        for i, j in loose:
            mol.RemoveBond(i, j)
        lig.mol = mol.GetMol()
        lig.mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(lig.mol)


def holders(structure: Structure, ligand: ResidueKey) -> "list[Atom]":
    """The atoms of other residues that hold a metal of the ligand itself."""
    metals = [a for a in structure.residue_atoms(ligand) if is_metal(a.element)]
    return [a for a in structure.atoms if a.element in _DONORS and a.residue != ligand
            and a.resname.upper() not in WATER_NAMES
            and any(float(np.linalg.norm(np.array(a.xyz) - np.array(m.xyz))) <= REACH.get(m.element, _REACH)
                    for m in metals)]


def without_the_bridge(records: "Sequence[Interaction]", found: "Sequence[IonBridge]", holding: "Sequence[Atom]" = ()
                       ) -> "tuple[list[Interaction], int]":
    """The records of a table without the ions' own (they are drawn with the ligand) and without hydrogen bonds
    between two atoms that hold the same ion, or of a histidine's nitrogen that holds one (``holding``: what holds
    a metal of the ligand itself); and how many of those there were."""
    ions = {(b.ion.chain, b.ion.resnum, b.ion.icode) for b in found}
    pairs = {(n, a.chain, a.resnum, a.icode, a.name) for b in found for n, _ in b.ligand for a, _ in b.protein}
    spent = {(a.chain, a.resnum, a.icode, a.name) for a in [a for b in found for a, _ in b.protein] + list(holding)
             if a.element == "N" and a.resname.upper() in ("HIS", "HID", "HIE", "HIP")}
    out, dropped = [], 0
    for r in records:
        if r.residue_id in ions:
            continue
        end = (*r.residue_id, r.protein_atom)
        if r.type is T.HBOND and (end in spent or (r.ligand_atom, *end) in pairs):
            dropped += 1
            continue
        out.append(r)
    return out, dropped
