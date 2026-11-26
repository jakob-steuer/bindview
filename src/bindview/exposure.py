"""Solvent exposure of the ligand: per atom for drawing, and as a reported number.

Both are plain Shrake-Rupley estimates on heavy atoms and need nothing but
NumPy.

* :func:`relative_exposure` -- each ligand atom's accessible surface in the
  complex relative to the same atom in the free ligand. A drawing aid: which
  atoms stick out of the pocket (the schematic style's grey halos).
* :func:`ligand_burial` -- the ligand's solvent-accessible surface in square
  angstrom, free and in the complex, and from the two the **buried fraction**.
  This one is reported. It measures the model as given: a ligand docked against
  half a site (one chain of an oligomer) stays exposed, and a pocket pLDDT
  read against its usual bands does not say so
  (``docs/reference.md``). The reverse does not hold: a ligand that
  has drifted into a deep enough groove elsewhere is buried, so a high value
  is no evidence that a site is complete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["relative_exposure", "ligand_burial", "structure_burial", "Burial", "PROBE", "SASA_POINTS",
           "WATER_NAMES", "BURIAL_WARNING_BELOW", "burial_warning"]

#: A ligand less buried than this draws a warning. The value is calibrated: ``docs/reference.md`` says on
#: what, and what the result does not show.
BURIAL_WARNING_BELOW = 0.70

#: The names a file gives a water molecule, in capitals. Water is solvent: it buries no ligand, holds no ion as
#: a residue does and is no partner of a bond or of a contact. One set for every module that asks.
WATER_NAMES = frozenset({"HOH", "WAT", "DOD", "H2O", "D2O", "TIP", "TIP3", "SOL"})

#: Radius of the solvent probe, angstrom.
PROBE = 1.4
#: Points per atom for the reported surface. 240 keeps the buried fraction of a
#: drug-sized ligand stable to about one part in a hundred.
SASA_POINTS = 240
#: Points per atom for the drawing aid, which needs no more than which atoms stick out.
_DRAWING_POINTS = 96

_VDW = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8, "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98,
        "Se": 1.9, "B": 1.92, "H": 1.1}
_DEFAULT_VDW = 1.8


def _radius(element: str) -> float:
    return _VDW.get(element, _DEFAULT_VDW)


def _sphere(n: int) -> np.ndarray:
    """``n`` points spread evenly on the unit sphere (golden-angle spiral)."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * k
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


def relative_exposure(lig_xyz: np.ndarray, lig_elements: Sequence[str], env_xyz: np.ndarray,
                      env_elements: Sequence[str]) -> np.ndarray:
    """Per ligand atom: accessible surface in the complex / accessible surface when free (0..1)."""
    n = len(lig_xyz)
    if n == 0:
        return np.zeros(0)
    sphere = _sphere(_DRAWING_POINTS)
    lig_r = np.array([_radius(e) for e in lig_elements]) + PROBE
    if len(env_xyz):
        lo, hi = lig_xyz.min(axis=0) - 9.0, lig_xyz.max(axis=0) + 9.0
        near = ((env_xyz >= lo) & (env_xyz <= hi)).all(axis=1)
        env_xyz = env_xyz[near]
        env_r = np.array([_radius(e) for e, keep in zip(env_elements, near) if keep]) + PROBE
    else:
        env_r = np.zeros(0)
    out = np.zeros(n)
    for i in range(n):
        pts = lig_xyz[i] + lig_r[i] * sphere                                     # (P, 3)
        others = np.arange(n) != i
        d_lig = np.linalg.norm(pts[:, None, :] - lig_xyz[others][None, :, :], axis=2)
        free = ~(d_lig < lig_r[others][None, :]).any(axis=1)
        n_free = int(free.sum())
        if n_free == 0:
            continue
        if len(env_xyz):
            d_env = np.linalg.norm(pts[free][:, None, :] - env_xyz[None, :, :], axis=2)
            open_ = ~(d_env < env_r[None, :]).any(axis=1)
            out[i] = float(open_.sum()) / n_free
        else:
            out[i] = 1.0
    return out


@dataclass(frozen=True)
class Burial:
    """How much of the ligand's surface the rest of the model covers."""

    sasa_free: float                    # A^2: the ligand alone
    sasa_complex: float                 # A^2: the ligand with everything else in the model around it
    atom_free: tuple[float, ...]        # the same per ligand atom, in the order of ``atoms``
    atom_complex: tuple[float, ...]
    probe: float = PROBE
    n_points: int = SASA_POINTS
    atoms: tuple[str, ...] = ()         # names of the ligand's heavy atoms, when known

    @property
    def buried_fraction(self) -> float:
        """``1 - SASA_complex / SASA_free``: 0 for a ligand in open solvent, 1 for one fully enclosed.

        A low value is a warning. A high one is not reassurance: a ligand buried in the wrong groove is
        buried all the same.
        """
        return 1.0 - self.sasa_complex / self.sasa_free if self.sasa_free > 0.0 else 0.0

    @property
    def exposed_fraction(self) -> float:
        return 1.0 - self.buried_fraction

    def as_dict(self) -> dict[str, Any]:
        return {"sasa_free": round(self.sasa_free, 1), "sasa_complex": round(self.sasa_complex, 1),
                "buried_fraction": round(self.buried_fraction, 3), "exposed_fraction": round(self.exposed_fraction, 3),
                "probe": self.probe, "n_points": self.n_points}


def ligand_burial(lig_xyz: np.ndarray, lig_elements: Sequence[str], env_xyz: np.ndarray,
                  env_elements: Sequence[str], probe: float = PROBE, n_points: int = SASA_POINTS) -> Burial:
    """Solvent-accessible surface of the ligand, free and in the complex.

    Heavy atoms only, on both sides; van der Waals radii as for the drawing
    aid. ``env_xyz`` / ``env_elements`` are whatever surrounds the ligand in
    the model: the caller decides what that is (the diagram leaves waters
    out, since they are solvent, and keeps everything else).
    """
    lig_xyz = np.asarray(lig_xyz, dtype=float).reshape(-1, 3)
    n = len(lig_xyz)
    if n == 0:
        return Burial(0.0, 0.0, (), (), probe, n_points)
    sphere = _sphere(n_points)
    lig_r = np.array([_radius(e) for e in lig_elements]) + probe
    env_xyz = np.asarray(env_xyz, dtype=float).reshape(-1, 3)
    env_r = np.array([_radius(e) for e in env_elements]) + probe
    if len(env_xyz):
        reach = float(lig_r.max() + env_r.max())
        lo, hi = lig_xyz.min(axis=0) - reach, lig_xyz.max(axis=0) + reach
        near = ((env_xyz >= lo) & (env_xyz <= hi)).all(axis=1)
        env_xyz, env_r = env_xyz[near], env_r[near]
    free = np.zeros(n)
    bound = np.zeros(n)
    for i in range(n):
        pts = lig_xyz[i] + lig_r[i] * sphere                                     # (P, 3)
        area = 4.0 * math.pi * lig_r[i] ** 2 / n_points
        others = np.arange(n) != i
        open_ = np.ones(n_points, dtype=bool)
        if others.any():
            d_lig = np.linalg.norm(pts[:, None, :] - lig_xyz[others][None, :, :], axis=2)
            open_ = ~(d_lig < lig_r[others][None, :]).any(axis=1)
        free[i] = area * float(open_.sum())
        if open_.any() and len(env_xyz):
            # Only atoms whose probe-expanded sphere can reach this one's matter.
            close = np.linalg.norm(env_xyz - lig_xyz[i], axis=1) < (lig_r[i] + env_r)
            if close.any():
                d_env = np.linalg.norm(pts[open_][:, None, :] - env_xyz[close][None, :, :], axis=2)
                bound[i] = area * float((~(d_env < env_r[close][None, :]).any(axis=1)).sum())
            else:
                bound[i] = free[i]
        else:
            bound[i] = free[i] if open_.any() else 0.0
    return Burial(float(free.sum()), float(bound.sum()), tuple(float(v) for v in free),
                  tuple(float(v) for v in bound), probe, n_points)


def structure_burial(structure: Any, ligand: Any, probe: float = PROBE, n_points: int = SASA_POINTS) -> Burial:
    """Burial of one residue of a parsed structure by everything else in the model.

    "Everything else" is every heavy atom that is not the ligand and not water:
    protein, nucleic acid, cofactors, ions, other ligands. Water is solvent --
    counting a crystallographic water shell as burial would make a ligand on
    the surface of a well-resolved structure look enclosed.
    """
    lig = [a for a in structure.residue_atoms(ligand) if a.element not in ("H", "D")]
    env = [a for a in structure.atoms
           if a.residue != ligand and a.element not in ("H", "D") and a.resname.upper() not in WATER_NAMES]
    out = ligand_burial(np.array([a.xyz for a in lig], dtype=float).reshape(-1, 3), [a.element for a in lig],
                        np.array([a.xyz for a in env], dtype=float).reshape(-1, 3), [a.element for a in env],
                        probe, n_points)
    return Burial(out.sasa_free, out.sasa_complex, out.atom_free, out.atom_complex, probe, n_points,
                  tuple(a.name for a in lig))


def burial_warning(burial: "Burial | None") -> str | None:
    """The note a poorly buried ligand gets, or ``None``.

    It states the observation -- most of the ligand's surface is open to
    solvent in this model -- and offers the two readings the calibration set
    contains: a site this model cannot form (one chain of an oligomer, a
    missing cofactor), or a ligand that is not in a pocket at all. It does not
    say which. A well buried ligand proves nothing either: one that has
    drifted into another cavity of an incomplete model is buried too.
    """
    if burial is None or burial.sasa_free <= 0.0 or burial.buried_fraction >= BURIAL_WARNING_BELOW:
        return None
    return (f"the ligand is only {100.0 * burial.buried_fraction:.0f} % buried: "
            f"{100.0 * burial.exposed_fraction:.0f} % of its surface stays solvent-exposed in this model "
            f"(below the calibrated {100.0 * BURIAL_WARNING_BELOW:.0f} %). The site may be incomplete here "
            "(one chain of an oligomer, a missing cofactor), or the ligand may not sit in a pocket at all")
