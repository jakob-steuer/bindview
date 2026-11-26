"""Small chemistry tables shared across the package."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

__all__ = ["METALS", "HALOGENS", "is_metal", "covalent_radius", "quiet_rdkit"]

# Elements treated as metals for bond pruning and coordination drawing.
METALS: frozenset[str] = frozenset({
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
    "Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "U",
})

HALOGENS: frozenset[str] = frozenset({"F", "Cl", "Br", "I"})

# Covalent radii in angstrom (Cordero et al. 2008, rounded) for the elements a
# ligand realistically contains: two atoms that are not farther apart than their
# radii allow are bound (:mod:`bindview.covalent`).
_RCOV = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "As": 1.19, "Se": 1.20,
    "Br": 1.20, "I": 1.39,
    "Li": 1.28, "Na": 1.66, "K": 2.03, "Mg": 1.41, "Ca": 1.76,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Mo": 1.54, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Cd": 1.44,
}


def is_metal(element: str) -> bool:
    return element in METALS


def covalent_radius(element: str) -> float:
    return _RCOV.get(element, 1.20)


@contextmanager
def quiet_rdkit() -> Iterator[None]:
    """Silence RDKit's log output for the duration, then restore the caller's setting.

    Template matching and kekulization probe molecules that RDKit complains
    about on stderr; those complaints are handled here and are noise to a caller.
    """
    from rdkit import rdBase

    block = rdBase.BlockLogs()
    try:
        yield
    finally:
        del block
