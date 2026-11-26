"""Predicted solvent sites: where a water could bridge ligand and protein.

A predicted model has no solvent. Where the ligand and the protein leave a
small enclosed gap lined by hydrogen-bond partners on both sides, a water
molecule is a reasonable guess -- and it is a guess. This module makes it, by
geometry alone and with NumPy alone:

1. a lattice is laid over the space around the ligand's polar atoms;
2. points too close to any heavy atom to hold a water oxygen are dropped;
3. points that are open to bulk solvent are dropped (the *enclosure* filter:
   of a set of directions from the point, most must run into the model);
4. each remaining point is scored by the polar atoms of protein and ligand that
   lie at hydrogen-bond distance **and point at it**;
5. sites are picked greedily, best first, no two closer than two waters can be.

Only *bridging* sites are returned: at least one partner on the protein and
one on the ligand, because that is what a diagram can draw.

What this is not: it is not a hydration model, it knows nothing of energies or
occupancy, and it cannot tell a water from an ion. What it should be judged by
is its hit rate against crystallographic waters, not this description:
``docs/reference.md`` has that measurement. In short, with the waters
held out of crystal structures it finds about seven bridging waters in ten,
and nearly half of the sites it proposes have no water in the crystal. That
was measured where the gap a water left is still in the model; a predicted
model need not leave one, and there it has not been tested. It is off unless
asked for, never runs on a structure that contains waters of its own, and
everything it produces is marked as predicted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .exposure import PROBE, WATER_NAMES, _radius, _sphere

__all__ = ["SolventSite", "SolventParams", "predict_solvent_sites", "candidate_points", "water_oxygens",
           "has_waters", "VALIDATION"]

#: What is known about how good these predictions are, in one sentence, printed under every figure
#: that shows one. It is the sentence ``scripts/validate_solvent.py`` writes from the hold-out numbers
#: (``tests/data/solvent-validation.json``); a test keeps the two equal.
VALIDATION = ("In a hold-out test on 42 crystal structures they recovered 71 % of the bridging waters, and 46 % of "
              "the predicted sites had no observed water within 1.5 Å.")


@dataclass(frozen=True)
class SolventParams:
    """The knobs of the predictor. The defaults are set from geometry; none is fitted to the validation."""

    spacing: float = 0.5            # lattice spacing, angstrom
    reach: float = 3.5              # a site lies within this of a ligand N or O
    polar_contact: float = 2.6      # closest a water oxygen comes to an N or O (a short hydrogen bond)
    hbond_max: float = 3.3          # farthest an N or O can be and still count as a partner
    min_enclosure: float = 0.6      # share of directions that must run into the model
    ray_length: float = 8.0         # how far a direction is followed
    n_rays: int = 30
    separation: float = 2.6         # two sites are at least this far apart
    cone_one: float = 100.0         # partner with one bonded heavy atom: site within this angle of its open side
    cone_many: float = 65.0         # ... with two or more
    max_sites: int = 6


@dataclass(frozen=True)
class SolventSite:
    """One predicted site, with the atoms that would hold a water there."""

    xyz: tuple[float, float, float]
    score: int                                                       # partners pointing at the site
    enclosure: float                                                 # 0 = open solvent, 1 = fully enclosed
    protein: tuple[tuple[str, int, str, str, str, float], ...]       # (chain, number, icode, resname, atom, distance)
    ligand: tuple[tuple[str, float], ...]                            # (atom name, distance)

    @property
    def bridging(self) -> bool:
        return bool(self.protein) and bool(self.ligand)

    def as_dict(self) -> dict[str, Any]:
        return {"xyz": [round(v, 2) for v in self.xyz], "score": self.score, "enclosure": round(self.enclosure, 2),
                "protein": [{"chain": c, "number": n, "icode": i, "resname": r, "atom": a, "distance": round(d, 2)}
                            for c, n, i, r, a, d in self.protein],
                "ligand": [{"atom": a, "distance": round(d, 2)} for a, d in self.ligand], "predicted": True}


def has_waters(structure: Any) -> bool:
    """Does the structure contain water molecules of its own?"""
    return any(a.resname.upper() in WATER_NAMES for a in structure.atoms)


def water_oxygens(structure: Any) -> np.ndarray:
    """Coordinates of every water oxygen in the structure, ``(n, 3)``."""
    return np.array([a.xyz for a in structure.atoms if a.resname.upper() in WATER_NAMES and a.element == "O"],
                    dtype=float).reshape(-1, 3)


def _open_sides(xyz: np.ndarray, polar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every polar atom: the unit vector away from its bonded heavy atoms, and how many it has.

    Bonds are taken from distances (under 1.95 angstrom between heavy atoms),
    which is all a structure without hydrogens offers. The open side is where
    a hydrogen or a lone pair is.
    """
    out = np.zeros((len(polar), 3))
    count = np.zeros(len(polar), dtype=int)
    for k, i in enumerate(polar):
        d = np.linalg.norm(xyz - xyz[i], axis=1)
        bonded = np.nonzero((d < 1.95) & (d > 0.1))[0]
        count[k] = len(bonded)
        if len(bonded):
            away = -(xyz[bonded] - xyz[i]).sum(axis=0)
            norm = float(np.linalg.norm(away))
            if norm > 1e-6:
                out[k] = away / norm
    return out, count


@dataclass(frozen=True)
class _Scene:
    """The surroundings of one ligand, as the predictor sees them."""

    env: list[Any]            # heavy atoms that can exclude, enclose or hold a water near the ligand; no waters
    xyz: np.ndarray
    is_lig: np.ndarray
    polar: np.ndarray
    vdw: np.ndarray
    grid: np.ndarray          # lattice points within reach of a ligand N or O that have room for a water oxygen


def _scene(structure: Any, ligand: Any, p: SolventParams) -> _Scene | None:
    """Steps 1 and 2: the lattice around the ligand's polar atoms, and the points with room for a water."""
    atoms = [a for a in structure.atoms if a.element not in ("H", "D") and a.resname.upper() not in WATER_NAMES]
    lig = [a for a in atoms if a.residue == ligand]
    lig_polar = [a for a in lig if a.element in ("N", "O")]
    if not lig_polar:
        return None
    centre_xyz = np.array([a.xyz for a in lig_polar], dtype=float)
    lo, hi = centre_xyz.min(axis=0) - p.reach, centre_xyz.max(axis=0) + p.reach
    # Everything that can exclude, enclose or hold a water near the ligand.
    all_xyz = np.array([a.xyz for a in atoms], dtype=float)
    margin = p.ray_length + 3.0
    near = ((all_xyz >= lo - margin) & (all_xyz <= hi + margin)).all(axis=1)
    env = [a for a, keep in zip(atoms, near) if keep]
    xyz = all_xyz[near]
    is_lig = np.array([a.residue == ligand for a in env])
    polar = np.array([a.element in ("N", "O") for a in env])
    contact = np.array([p.polar_contact if a.element in ("N", "O") else _radius(a.element) + PROBE for a in env])
    vdw = np.array([_radius(a.element) for a in env])

    # 1. the lattice, on a grid fixed in space, within reach of a ligand N or O
    s = p.spacing
    axes = [np.arange(math.floor(lo[k] / s) * s, hi[k] + s / 2.0, s) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    keep = np.zeros(len(grid), dtype=bool)
    for c in centre_xyz:
        keep |= ((grid - c) ** 2).sum(axis=1) <= p.reach ** 2
    grid = grid[keep]
    # 2. room for a water oxygen
    free = np.ones(len(grid), dtype=bool)
    for start in range(0, len(xyz), 64):
        d = np.linalg.norm(grid[:, None, :] - xyz[None, start:start + 64, :], axis=2)
        free &= (d >= contact[None, start:start + 64] - 1e-9).all(axis=1)
    return _Scene(env, xyz, is_lig, polar, vdw, grid[free])


def candidate_points(structure: Any, ligand: Any, params: SolventParams | None = None) -> np.ndarray:
    """Every lattice point near the ligand's polar atoms that merely has room for a water oxygen, ``(n, 3)``.

    This is where the predictor starts, before enclosure and hydrogen-bond
    partners are looked at. The validation draws its chance baseline from it.
    """
    scene = _scene(structure, ligand, params or SolventParams())
    return scene.grid if scene is not None else np.zeros((0, 3))


def predict_solvent_sites(structure: Any, ligand: Any, params: SolventParams | None = None) -> list[SolventSite]:
    """Bridging solvent sites around one ligand of a structure, best first.

    Waters in the structure are ignored as atoms (they are what is being
    predicted), so the function can be run on an experimental structure for
    validation; :meth:`Diagram.from_structure` does not call it for one.
    """
    p = params or SolventParams()
    scene = _scene(structure, ligand, p)
    if scene is None or not len(scene.grid):
        return []
    env, xyz, is_lig, polar, vdw, grid = scene.env, scene.xyz, scene.is_lig, scene.polar, scene.vdw, scene.grid
    s = p.spacing

    # 3. enclosure and 4. partners, point by point
    rays = _sphere(p.n_rays)
    pol_idx = np.nonzero(polar)[0]
    side, n_bonded = _open_sides(xyz, pol_idx)
    cos_one, cos_many = math.cos(math.radians(p.cone_one)), math.cos(math.radians(p.cone_many))
    pol_xyz = xyz[pol_idx]
    enclosure = np.zeros(len(grid))
    aim = np.zeros(len(grid))               # how squarely the partners point at the point: sum of cosines
    partners: list[np.ndarray] = []
    for g, point in enumerate(grid):
        rel = xyz - point
        dist2 = (rel ** 2).sum(axis=1)
        close = dist2 < (p.ray_length + 2.0) ** 2
        along = rel[close] @ rays.T                                             # (atoms, rays)
        perp2 = dist2[close][:, None] - along ** 2
        hit = (along > 0.0) & (along < p.ray_length) & (perp2 < (vdw[close] ** 2)[:, None])
        enclosure[g] = float(hit.any(axis=0).mean())
        to_point = point - pol_xyz
        dp = np.linalg.norm(to_point, axis=1)
        ok = dp <= p.hbond_max
        cosang = np.einsum("ij,ij->i", to_point, side) / np.maximum(dp, 1e-9)
        pointing = np.where(n_bonded == 0, True, np.where(n_bonded == 1, cosang >= cos_one, cosang >= cos_many))
        chosen = np.nonzero(ok & pointing)[0]
        partners.append(chosen)
        aim[g] = float(np.where(n_bonded[chosen] == 0, 1.0, cosang[chosen]).sum())
    n_lig = np.array([int(is_lig[pol_idx[idx]].sum()) for idx in partners])
    n_pro = np.array([len(idx) for idx in partners]) - n_lig
    score = n_lig + n_pro
    good = (enclosure >= p.min_enclosure) & (n_lig >= 1) & (n_pro >= 1)
    if not good.any():
        return []

    # 5. greedy pick: most partners, then the point they point at most squarely (a straight
    #    hydrogen bond beats a bent one), then the most enclosed, then a fixed order in space
    order = sorted(np.nonzero(good)[0], key=lambda i: (-int(score[i]), -round(float(aim[i]), 3),
                                                       -round(float(enclosure[i]), 6),
                                                       round(float(grid[i, 0]), 3), round(float(grid[i, 1]), 3),
                                                       round(float(grid[i, 2]), 3)))
    taken: list[np.ndarray] = []
    sites: list[SolventSite] = []
    for i in order:
        if any(np.linalg.norm(grid[i] - t) < p.separation for t in taken):
            continue
        # Settle on the middle of the equally good points around the pick, off the lattice.
        same = (good & (score == score[i]) & (aim >= aim[i] - 0.05)
                & (np.linalg.norm(grid - grid[i], axis=1) <= 1.5 * s))
        centre = grid[same].mean(axis=0)
        taken.append(centre)
        pro, ligs = [], []
        for j in partners[i]:
            atom = env[pol_idx[j]]
            d = float(np.linalg.norm(centre - pol_xyz[j]))
            if is_lig[pol_idx[j]]:
                ligs.append((atom.name, d))
            else:
                pro.append((atom.chain, atom.resnum, atom.icode, atom.resname, atom.name, d))
        sites.append(SolventSite((float(centre[0]), float(centre[1]), float(centre[2])), int(score[i]),
                                 float(enclosure[i]), tuple(sorted(pro, key=lambda t: (t[5], t[0], t[1], t[4]))),
                                 tuple(sorted(ligs, key=lambda t: (t[1], t[0])))))
        if len(sites) >= p.max_sites:
            break
    return sites
