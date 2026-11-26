"""Perception of the ligand features interactions attach to: rings and charged groups.

Pi interactions attach to a ring centroid and salt bridges to a charged group,
and detectors usually report those without atom names. This module enumerates
the candidates on the ligand; :mod:`bindview.resolve` then picks the one whose
geometry reproduces the reported distance.

The definitions are written from the chemistry, deliberately a little broader
than any one detector's, and each candidate carries a ``rank`` saying how
canonical it is (0 = textbook) so that ties are broken toward chemical sense.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from .chem import is_metal
from .ligand import Ligand

__all__ = ["LigandGroup", "perceive_rings", "perceive_charged_groups", "perceive_metals"]


@dataclass(frozen=True)
class LigandGroup:
    """A ring or functional group on the ligand.

    ``atoms`` are the members (what gets highlighted, and whose 2D centroid is
    the drawing anchor). ``center_atoms`` are the atoms whose 3D centroid is
    the interaction centre a detector measures to -- e.g. the phosphorus of a
    phosphate, the two oxygens of a carboxylate.
    """

    kind: str
    atoms: tuple[int, ...]
    center_atoms: tuple[int, ...]
    charge: int
    rank: int

    def center3d(self, coords: np.ndarray) -> np.ndarray:
        return coords[list(self.center_atoms)].mean(axis=0)


def _ring_planarity(coords: np.ndarray, ring: tuple[int, ...]) -> float:
    """RMS distance (angstrom) of ring atoms from their best-fit plane."""
    pts = coords[list(ring)]
    centred = pts - pts.mean(axis=0)
    # Smallest singular value direction is the plane normal.
    _, s, _ = np.linalg.svd(centred, full_matrices=False)
    return float(s[-1] / np.sqrt(len(ring)))


def perceive_rings(lig: Ligand) -> list[LigandGroup]:
    """Five- and six-membered rings, aromatic ones first.

    Non-aromatic rings are kept as lower-ranked candidates when they are planar
    in the model, because detectors working from geometry accept those too.
    """
    out: list[LigandGroup] = []
    mol = lig.mol
    for ring in mol.GetRingInfo().AtomRings():
        if not 5 <= len(ring) <= 6:
            continue
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        if aromatic:
            rank = 0
        elif _ring_planarity(lig.coords3d, ring) < 0.12:
            rank = 1
        else:
            rank = 3
        members = tuple(ring)
        out.append(LigandGroup("ring", members, members, 0, rank))
    out.sort(key=lambda g: (g.rank, g.atoms))
    return out


def _in_aromatic_six_ring(atom: Chem.Atom) -> bool:
    return atom.GetIsAromatic() and atom.IsInRingSize(6)


def _second_amino(atom: Chem.Atom) -> bool:
    """Another carbon of the atom's aromatic six-ring carries an amino group (2,4-diaminopyrimidines, pteridines)."""
    mol = atom.GetOwningMol()
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 6 or atom.GetIdx() not in ring:
            continue
        for j in ring:
            other = mol.GetAtomWithIdx(j)
            if j == atom.GetIdx() or other.GetSymbol() != "C":
                continue
            if any(n.GetSymbol() == "N" and not n.IsInRing() and n.GetTotalNumHs() > 0 for n in other.GetNeighbors()):
                return True
    return False


def _heavy_neighbors(atom: Chem.Atom) -> list[Chem.Atom]:
    return [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]


def _is_terminal_oxygen(atom: Chem.Atom) -> bool:
    return atom.GetSymbol() == "O" and len(_heavy_neighbors(atom)) == 1


def _is_amide_like(n_atom: Chem.Atom) -> bool:
    """Nitrogen whose lone pair is tied up: amide, sulfonamide, aniline, enamine, N-oxide..."""
    for nb in n_atom.GetNeighbors():
        if nb.GetIsAromatic():
            return True
        if nb.GetSymbol() in ("S", "P"):
            return True
        for b in nb.GetBonds():
            if b.GetBondTypeAsDouble() >= 2 and b.GetOtherAtomIdx(nb.GetIdx()) != n_atom.GetIdx():
                return True
    return False


def perceive_charged_groups(lig: Ligand) -> list[LigandGroup]:
    """Groups that carry (or readily carry) a charge at physiological pH."""
    mol = lig.mol
    out: list[LigandGroup] = []
    seen_single: set[int] = set()
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        sym = atom.GetSymbol()
        nbrs = _heavy_neighbors(atom)
        if sym == "C":
            term_o = [n.GetIdx() for n in nbrs if _is_terminal_oxygen(n)]
            if len(term_o) == 2 and len(nbrs) == 3:
                out.append(LigandGroup("carboxylate", (i, *term_o), tuple(term_o), -1, 0))
            n_nb = [n for n in nbrs if n.GetSymbol() == "N"]
            if len(n_nb) == 3 and len(nbrs) == 3 and (not _in_aromatic_six_ring(atom) or _second_amino(atom)):
                out.append(LigandGroup("guanidinium", (i, *(n.GetIdx() for n in n_nb)), (i,), +1, 0))
            elif len(n_nb) == 2 and len(nbrs) == 3 and not atom.GetIsAromatic() \
                    and all(not n.GetIsAromatic() for n in n_nb):
                out.append(LigandGroup("amidinium", (i, *(n.GetIdx() for n in n_nb)), (i,), +1, 1))
        elif sym == "P":
            o_nb = [n.GetIdx() for n in nbrs if n.GetSymbol() == "O"]
            if len(o_nb) >= 2:
                term = [j for j in o_nb if _is_terminal_oxygen(mol.GetAtomWithIdx(j))]
                out.append(LigandGroup("phosphate", (i, *(term or o_nb)), (i,), -1, 0 if len(o_nb) >= 3 else 1))
        elif sym == "S":
            o_nb = [n.GetIdx() for n in nbrs if n.GetSymbol() == "O"]
            term = [j for j in o_nb if _is_terminal_oxygen(mol.GetAtomWithIdx(j))]
            if len(o_nb) >= 3:
                out.append(LigandGroup("sulfonate", (i, *term), (i,), -1, 0))
            elif len(nbrs) == 3 and all(n.GetSymbol() == "C" for n in nbrs) and atom.GetFormalCharge() > 0:
                out.append(LigandGroup("sulfonium", (i,), (i,), +1, 0))
        elif sym == "N":
            if atom.GetFormalCharge() > 0 and not any(n.GetFormalCharge() < 0 for n in nbrs):
                out.append(LigandGroup("ammonium", (i,), (i,), +1, 0))
                seen_single.add(i)
            elif not atom.GetIsAromatic() and atom.GetHybridization() == Chem.HybridizationType.SP3 \
                    and all(n.GetSymbol() == "C" for n in nbrs) and not _is_amide_like(atom):
                # Basic aliphatic amine. Tertiary first: that is the class detectors treat as cationic.
                rank = 0 if len(nbrs) == 3 else 1
                out.append(LigandGroup("amine", (i,), (i,), +1, rank))
                seen_single.add(i)
        elif is_metal(sym):
            out.append(LigandGroup("metal", (i,), (i,), +1, 0))
            seen_single.add(i)
        if atom.GetFormalCharge() < 0 and sym in ("O", "N", "S") and i not in seen_single:
            covered = any(i in g.atoms for g in out if g.charge < 0)
            if not covered:
                out.append(LigandGroup("anion", (i,), (i,), -1, 2))
                seen_single.add(i)
    # Any remaining nitrogen is a last-resort cation candidate: a detector with
    # different typing rules may have called it charged, and the reported
    # distance will say so unambiguously.
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "N" and atom.GetIdx() not in seen_single:
            out.append(LigandGroup("nitrogen", (atom.GetIdx(),), (atom.GetIdx(),), +1, 4))
    out.sort(key=lambda g: (g.rank, g.kind, g.atoms))
    return out


def perceive_metals(lig: Ligand) -> list[int]:
    return [a.GetIdx() for a in lig.mol.GetAtoms() if is_metal(a.GetSymbol())]
