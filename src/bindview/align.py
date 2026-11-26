"""Putting several ligands of one pocket into one drawing frame.

A comparison is only readable if the thing that is the same looks the same.
Two ways of getting there, and one rule about when to use neither:

* **Common substructure.** When the ligands share a substantial substructure
  (a congeneric series), every ligand is drawn with that substructure on the
  same 2D coordinates, so the shared part is identical in every drawing and
  only the differences move.
* **Pose.** When they do not, a forced maximum-common-substructure alignment
  would look confident and mean nothing. The alignment is refused, and each
  ligand is instead turned to match its own pose in the superposed pocket --
  an orientation the data supports whatever the chemistry.
* The refusal is a number, not a judgement call: the common substructure must
  cover ``min_fraction`` of the smallest ligand's heavy atoms (default 60 %)
  and at least ``min_atoms`` atoms. Both numbers are reported with the result,
  so a figure can say why it did or did not align.

The substructure search is RDKit's FMCS, seeded with the common substructure of
the Murcko scaffolds, which is what makes it fast. Every search has a time limit
of 30 s and is back at it whatever it is given (:class:`_Clock`). A seeded search
that is stopped there is made again without the seed; a result that the limit cut
short is marked incomplete rather than silently used as if it were the maximum.

The scaffold of a ligand is RDKit's, and where RDKit gives none it is made here
(:func:`_framework`), because a comparison must not fall with it: a build of RDKit
can raise ``bad any cast`` for every molecule, and any RDKit refuses a ligand of which
only the connectivity is known, where a clash in the model has given an atom more
bonds than its element can have.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Geometry import Point2D

from .chem import quiet_rdkit
from .ligand import Ligand
from .structure import Structure

if TYPE_CHECKING:  # pragma: no cover
    from .diagram import Diagram

__all__ = ["CommonCore", "find_common_core", "core_coords", "pocket_transforms", "pocket_transforms_between",
           "PoseFrame", "pose_frame", "pose_coords", "MIN_FRACTION", "MIN_ATOMS"]

#: A common substructure is worth aligning on when it covers this fraction of
#: the smallest ligand's heavy atoms ...
MIN_FRACTION = 0.6
#: ... and has at least this many atoms.
MIN_ATOMS = 6
_TIMEOUT = 30                      # seconds for one search; a safety net, see the module docstring
_RDKIT_BOND = 1.5                  # bond length of RDKit's depictor
_SNAP = math.radians(30.0)         # a drawing is turned to its pose in steps of this: bonds stay on the usual grid


@dataclass(frozen=True)
class CommonCore:
    """The common substructure of a set of ligands, and whether it is worth aligning on."""

    n_atoms: int
    atoms: tuple[tuple[str, ...], ...]      # per ligand: names of its core atoms, in the order of the core
    sizes: tuple[int, ...]                  # heavy atoms per ligand
    min_fraction: float
    min_atoms: int
    complete: bool = True                   # False if the search hit its timeout

    @property
    def fraction(self) -> float:
        """Share of the smallest ligand's heavy atoms that the core covers."""
        smallest = min(self.sizes) if self.sizes else 0
        return self.n_atoms / smallest if smallest else 0.0

    @property
    def substantial(self) -> bool:
        return self.n_atoms >= self.min_atoms and self.fraction >= self.min_fraction - 1e-9

    def describe(self) -> str:
        """One sentence for the figure: what was found and what was done with it."""
        smallest = min(self.sizes) if self.sizes else 0
        what = (f"the largest common substructure has {self.n_atoms} of {smallest} heavy atoms of the smallest "
                f"ligand ({100.0 * self.fraction:.0f} %)")
        if self.substantial:
            text = f"Aligned on a common substructure: {what}."
        else:
            # The share is the reason where it is too small; else the core has too few atoms, whatever its share.
            short = (f"below the {100.0 * self.min_fraction:.0f} % needed"
                     if self.fraction < self.min_fraction - 1e-9 else f"fewer than the {self.min_atoms} atoms needed")
            text = f"No common scaffold: {what}, {short} to align on. No substructure alignment was made."
        if not self.complete:
            text += " The substructure search stopped at its time limit."
        return text


class _OutOfTime(Exception):
    """A search is at its time limit."""


class _Clock(rdFMCS.MCSFinalMatchCheck):
    """Ends a search at its time limit, from inside it.

    RDKit looks at its own clock between two candidates only. One candidate that many bonds leave, and
    that does not match with all of them, has every subset of those bonds tried before the clock is read
    again: a seed of a macrolide's rings with its 19 substituents is half a million subsets. This check is
    asked at every match, accepts each, and raises once the limit is over.
    """

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.end = time.monotonic() + seconds

    def __call__(self, *args: object) -> bool:
        if time.monotonic() > self.end:
            raise _OutOfTime
        return True


@dataclass(frozen=True)
class _Stopped:
    """What a search that was ended at its limit found: nothing that can be used."""

    numAtoms: int = 0
    smartsString: str = ""
    canceled: bool = True


def _fmcs(mols: Sequence[Chem.Mol], seed: str = "", orders: bool = True) -> "rdFMCS.MCSResult | _Stopped":
    """RDKit's search, which is back within :data:`_TIMEOUT` seconds whatever it is given. With ``orders``
    an aromatic bond matches an aromatic one only; without, it matches a single bond as well."""
    params = rdFMCS.MCSParameters()
    params.Timeout = _TIMEOUT                           # type: ignore[assignment]  # (RDKit's stubs)
    params.AtomTyper = rdFMCS.AtomCompare.CompareElements
    params.BondTyper = rdFMCS.BondCompare.CompareOrderExact if orders else rdFMCS.BondCompare.CompareOrder
    params.AtomCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.CompleteRingsOnly = True
    params.InitialSeed = seed                           # type: ignore[assignment]
    params.FinalMatchChecker = _Clock(_TIMEOUT)
    try:
        return rdFMCS.FindMCS(list(mols), params)
    except _OutOfTime:
        return _Stopped()


def _framework(mol: Chem.Mol) -> Chem.Mol:
    """The Murcko scaffold of a molecule, made without RDKit's ``MurckoDecompose``.

    The rings, what joins them, and what is double-bonded to either. An atom at the end of a chain is
    taken away, then the atom that has become an end, until none is left: what remains are the rings
    and the chains between them (a molecule without a ring leaves nothing). An atom that was taken
    away and is double-bonded to one that remains is put back. An atom that lost a neighbour is left
    as RDKit leaves it: an aromatic heteroatom gets a hydrogen, any other atom loses its chirality and
    takes the hydrogens its valence asks for.

    The atoms keep their order, so the molecule is the one RDKit's ``GetScaffoldForMol`` gives, atom
    for atom, and a search that is seeded with it finds the same.
    """
    n = mol.GetNumAtoms()
    nbrs = [[b.GetOtherAtomIdx(a.GetIdx()) for b in a.GetBonds()] for a in mol.GetAtoms()]
    keep, degree = [True] * n, [len(x) for x in nbrs]
    ends = [i for i in range(n) if degree[i] <= 1]
    while ends:
        i = ends.pop()
        if not keep[i]:
            continue
        keep[i] = False
        for j in nbrs[i]:
            if keep[j]:
                degree[j] -= 1
                if degree[j] <= 1:
                    ends.append(j)
    core = list(keep)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if b.GetBondType() == Chem.BondType.DOUBLE and core[i] != core[j]:
            keep[i] = keep[j] = True
    rw = Chem.RWMol(mol)
    for i in range(n):
        if keep[i]:
            continue
        for j in nbrs[i]:
            if keep[j]:
                atom = rw.GetAtomWithIdx(j)
                if atom.GetIsAromatic() and atom.GetAtomicNum() != 6:
                    atom.SetNumExplicitHs(1)
                else:
                    atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                    atom.SetNoImplicit(False)
                    atom.SetNumExplicitHs(0)
    for i in range(n - 1, -1, -1):                      # from the last, so that no index moves under the loop
        if not keep[i]:
            rw.RemoveAtom(i)
    out = rw.GetMol()
    out.ClearComputedProps()
    out.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(out)
    return out


def _scaffold(mol: Chem.Mol) -> Chem.Mol:
    """The Murcko scaffold of a molecule: RDKit's, and the one made here where RDKit raises."""
    try:
        return MurckoScaffold.GetScaffoldForMol(mol)
    except (RuntimeError, ValueError):                   # (RDKit's valence error is a ValueError)
        return _framework(mol)


def find_common_core(ligands: Sequence[Ligand], min_fraction: float = MIN_FRACTION,
                     min_atoms: int = MIN_ATOMS) -> CommonCore:
    """The maximum common substructure of ``ligands``, with the verdict on whether to align on it.

    Elements must match, and bond orders: a single, a double, a triple and an aromatic bond match their
    own kind (a phenyl ring is no cyclohexyl ring). Where the bond orders of one of the ligands are not
    known, only its connectivity, all its bonds are single ones to the search, and an aromatic bond of
    another ligand matches a single one: a double bond of another ligand still does not. An atom of a
    ring matches an atom of a ring only, and a ring is in the substructure whole or not at all.
    """
    sizes = tuple(lig.n_atoms for lig in ligands)
    empty = CommonCore(0, tuple(() for _ in ligands), sizes, min_fraction, min_atoms)
    if len(ligands) < 2 or min(sizes) < 2:
        return empty
    mols = [lig.mol for lig in ligands]
    orders = all(lig.bond_source != "connectivity" for lig in ligands)
    with quiet_rdkit():
        scaffolds = [_scaffold(m) for m in mols]
        seed, complete = "", True
        if all(s.GetNumAtoms() >= 3 for s in scaffolds):
            first = _fmcs(scaffolds, orders=orders)
            complete = not first.canceled
            if first.numAtoms >= 3:
                seed = first.smartsString
        result = _fmcs(mols, seed, orders)
        if result.numAtoms == 0 and seed:
            # The scaffold core did not extend, or the seeded search was stopped at its limit: search unseeded.
            result = _fmcs(mols, orders=orders)
        complete = complete and not result.canceled
        if result.numAtoms == 0:
            return CommonCore(0, empty.atoms, sizes, min_fraction, min_atoms, complete)
        query = Chem.MolFromSmarts(result.smartsString)
        matches = []
        for lig, mol in zip(ligands, mols):
            match = mol.GetSubstructMatch(query)
            if not match:                              # cannot happen for a true common substructure
                return CommonCore(0, empty.atoms, sizes, min_fraction, min_atoms, False)
            matches.append(tuple(lig.names[i] for i in match))
    return CommonCore(int(result.numAtoms), tuple(matches), sizes, min_fraction, min_atoms, complete)


def core_coords(lig: Ligand, core_names: Sequence[str], reference: Sequence[tuple[float, float]]
                ) -> dict[str, tuple[float, float]]:
    """2D coordinates of ``lig`` (bond-length units, y up) with its core atoms **on** ``reference``.

    The core atoms are locked to the reference positions and RDKit's depictor
    lays out the rest around them, so the shared part of two ligands is
    identical to the last digit and only what differs is drawn differently.
    """
    index = {name: i for i, name in enumerate(lig.names)}
    locked = {index[name]: Point2D(float(x) * _RDKIT_BOND, float(y) * _RDKIT_BOND)
              for name, (x, y) in zip(core_names, reference)}
    mol = Chem.Mol(lig.mol)
    mol.RemoveAllConformers()
    with quiet_rdkit():
        prefer = rdDepictor.GetPreferCoordGen()
        rdDepictor.SetPreferCoordGen(False)
        try:
            rdDepictor.Compute2DCoords(mol, coordMap=locked, canonOrient=False, clearConfs=True)
        finally:
            rdDepictor.SetPreferCoordGen(prefer)
    conf = mol.GetConformer()
    out = {}
    for i, name in enumerate(lig.names):
        p = conf.GetAtomPosition(i)
        out[name] = (float(p.x) / _RDKIT_BOND, float(p.y) / _RDKIT_BOND)
    # Counter-ions and other pieces that hold no core atom are set down in a row
    # under the drawing instead of wherever the depictor left them.
    anchored = [frag for frag in Chem.GetMolFrags(mol) if any(i in locked for i in frag)]
    loose = [frag for frag in Chem.GetMolFrags(mol) if not any(i in locked for i in frag)]
    if loose and anchored:
        main = [out[lig.names[i]] for frag in anchored for i in frag]
        x0, y0 = min(p[0] for p in main), min(p[1] for p in main)
        for j, frag in enumerate(loose):
            fx = sum(out[lig.names[i]][0] for i in frag) / len(frag)
            fy = sum(out[lig.names[i]][1] for i in frag) / len(frag)
            for i in frag:
                x, y = out[lig.names[i]]
                out[lig.names[i]] = (x - fx + x0 + 2.4 * j, y - fy + y0 - 1.6)
    for name, (x, y) in zip(core_names, reference):
        out[name] = (float(x), float(y))               # exactly, not to depictor precision
    return out


# ---------------------------------------------------------------------------
# Pocket superposition and pose-based orientation
# ---------------------------------------------------------------------------

_BACKBONE = ("N", "CA", "C")


def _kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` with ``target ~ mobile @ R.T + t`` (least squares, proper rotation)."""
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    h = (mobile - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, tc - mc @ rot.T


def _backbone(st: Structure) -> dict[tuple[str, int, str], tuple[str, dict[str, np.ndarray]]]:
    """Backbone atoms of every polymer residue: ``(chain, number, icode) -> (name, {atom: xyz})``."""
    coords = st.coords
    out: dict[tuple[str, int, str], tuple[str, dict[str, np.ndarray]]] = {}
    for a in st.atoms:
        if not a.hetero and a.name in _BACKBONE:
            out.setdefault((a.chain, a.resnum, a.icode), (a.resname, {}))[1][a.name] = coords[a.index]
    return out


def _near(backbone: dict, lig_xyz: np.ndarray, radius: float) -> set[tuple[str, int, str]]:
    out = set()
    for key, (_name, atoms) in backbone.items():
        pts = np.array(list(atoms.values()))
        if float(np.linalg.norm(pts[:, None, :] - lig_xyz[None, :, :], axis=2).min()) <= radius:
            out.add(key)
    return out


#: A structure is superposed by its residues with a backbone atom within this many angstrom of its ligand.
_RADIUS = 10.0
#: Share of the residues around the ligands that must carry the same name in both structures ...
_MIN_IDENTITY = 0.7
#: ... and the backbone RMSD (angstrom) the fit may leave, for the two to count as one pocket.
_MAX_RMSD = 3.0


_Pocket = tuple[dict[tuple[str, int, str], tuple[str, dict[str, np.ndarray]]], set[tuple[str, int, str]]]
_Fit = tuple[tuple[np.ndarray, np.ndarray] | None, float | None]


def _pocket(dia: "Diagram") -> "_Pocket | None":
    """What a structure is superposed by: its backbone, and the residues of it around its ligand."""
    if dia.structure is None:
        return None
    backbone = _backbone(dia.structure)
    return backbone, _near(backbone, dia.ligand.coords3d, _RADIUS)


def _fit(ref: _Pocket, own: _Pocket) -> _Fit:
    """The transform that puts the structure of ``own`` into the frame of ``ref``, and the backbone RMSD that
    the fit leaves: no transform where that is more than 3 angstrom, and neither where no fit can be made."""
    (ref_bb, ref_near), (own_bb, own_near) = ref, own
    both = sorted((ref_near | own_near) & set(ref_bb) & set(own_bb))
    same = [key for key in both if ref_bb[key][0] == own_bb[key][0]]
    if len(same) < 4 or len(same) < _MIN_IDENTITY * len(both):
        return None, None
    mobile, target = [], []
    for key in same:
        for atom in _BACKBONE:
            if atom in ref_bb[key][1] and atom in own_bb[key][1]:
                mobile.append(own_bb[key][1][atom])
                target.append(ref_bb[key][1][atom])
    rot, shift = _kabsch(np.array(mobile), np.array(target))
    rmsd = float(np.sqrt((((np.array(mobile) @ rot.T + shift) - np.array(target)) ** 2).sum(axis=1).mean()))
    return ((rot, shift) if rmsd <= _MAX_RMSD else None), rmsd


def pocket_transforms(diagrams: "Sequence[Diagram]", rmsds: "list[float | None] | None" = None
                      ) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Rigid transforms that put every structure into the frame of the first.

    Each structure is superposed on the first by the backbone atoms (N, CA,
    C) of the residues around either ligand. The residue numbering is what a
    comparison takes as given, and it is checked rather than trusted: an entry
    is ``None`` -- no common frame -- unless at least four residues match, 70 %
    of the residues compared carry the same name in both structures, and the
    fit leaves a backbone RMSD of at most 3 angstrom. Two different proteins
    share a few (number, name) pairs by chance and do not pass this; neither do
    two models of one protein whose pockets are folded differently. ``rmsds``,
    if given, receives the RMSD of every fit that was made (``None`` where no
    fit was possible).
    """
    out: list[tuple[np.ndarray, np.ndarray] | None] = []
    if rmsds is not None:
        rmsds[:] = [None] * len(diagrams)
    base = _pocket(diagrams[0])
    if base is None:
        return [None] * len(diagrams)
    for k, dia in enumerate(diagrams):
        if k == 0:
            out.append((np.eye(3), np.zeros(3)))
            if rmsds is not None:
                rmsds[k] = 0.0
            continue
        own = _pocket(dia)
        if own is None:
            out.append(None)
            continue
        frame, rmsd = _fit(base, own)
        if rmsds is not None:
            rmsds[k] = rmsd
        out.append(frame)
    return out


def pocket_transforms_between(diagrams: "Sequence[Diagram]") -> dict[tuple[int, int], _Fit]:
    """For every two structures ``i < j``: the transform that puts ``j`` into the frame of ``i`` and the backbone
    RMSD of the fit, as :func:`pocket_transforms` gives them for the two (``None`` for either as there, and
    for both where one has no structure). Every structure is read once, however many it is compared with."""
    pockets = [_pocket(dia) for dia in diagrams]
    out: dict[tuple[int, int], _Fit] = {}
    for i, ref in enumerate(pockets):
        for j in range(i + 1, len(pockets)):
            own = pockets[j]
            out[(i, j)] = _fit(ref, own) if ref is not None and own is not None else (None, None)
    return out


@dataclass(frozen=True)
class PoseFrame:
    """A plane through the superposed ligands: where 3D poses are looked at from."""

    origin: np.ndarray              # (3,) centroid of all ligand atoms
    axes: np.ndarray                # (2, 3) orthonormal in-plane axes, widest first

    def project(self, xyz: np.ndarray) -> np.ndarray:
        return (np.asarray(xyz, dtype=float) - self.origin) @ self.axes.T


def pose_frame(clouds: Sequence[np.ndarray]) -> PoseFrame:
    """The best-fit plane of all ligand atoms (already in one 3D frame).

    The two in-plane axes are the directions of largest spread, so the view is
    the one in which the poses overlap least. Their signs are fixed by a rule
    (positive third moment, then first atom), which makes the frame a function
    of the coordinates alone.
    """
    pts = np.vstack([np.asarray(c, dtype=float).reshape(-1, 3) for c in clouds])
    origin = pts.mean(axis=0)
    centred = pts - origin
    if len(pts) < 3:
        return PoseFrame(origin, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axes = vt[:2].copy()
    for i in range(2):
        proj = centred @ axes[i]
        skew = float((proj ** 3).sum())
        if abs(skew) < 1e-6:
            skew = float(proj[0])
        if skew < 0:
            axes[i] = -axes[i]
    return PoseFrame(origin, axes)


def pose_coords(names: Sequence[str], own_xy: np.ndarray, projected: np.ndarray, scale: float
                ) -> dict[str, tuple[float, float]]:
    """Turn a ligand's own 2D drawing so that it matches its projected 3D pose.

    ``own_xy`` are the drawing's coordinates (bond-length units, y up),
    ``projected`` the same atoms in the pose plane (angstrom). The drawing is
    rotated -- and mirrored if that fits better -- by the least-squares fit,
    with the angle snapped to steps of 30 degrees so bonds stay on the usual
    drawing grid, and moved so that its centre sits where the pose's centre is
    (``scale`` bond lengths per angstrom). The drawing itself is not distorted.
    """
    u = np.asarray(own_xy, dtype=float).reshape(-1, 2)
    p = np.asarray(projected, dtype=float).reshape(-1, 2)
    uc, pc = u.mean(axis=0), p.mean(axis=0)
    best: tuple[float, np.ndarray] | None = None
    for mirror in (False, True):
        v = (u - uc) * (np.array([1.0, -1.0]) if mirror else 1.0)
        # Optimal rotation angle of v onto p - pc (2D Procrustes in closed form).
        num = float((v[:, 0] * (p[:, 1] - pc[1]) - v[:, 1] * (p[:, 0] - pc[0])).sum())
        den = float((v[:, 0] * (p[:, 0] - pc[0]) + v[:, 1] * (p[:, 1] - pc[1])).sum())
        theta = round(math.atan2(num, den) / _SNAP) * _SNAP
        c, s = math.cos(theta), math.sin(theta)
        turned = np.column_stack([v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c])
        # Compare shapes at the pose's own size, so the choice does not depend on the units.
        size_t = math.sqrt(float((turned ** 2).sum())) or 1.0
        size_p = math.sqrt(float(((p - pc) ** 2).sum())) or 1.0
        err = round(float(((turned / size_t - (p - pc) / size_p) ** 2).sum()), 9)
        if best is None or err < best[0] - 1e-9:
            best = (err, turned)
    assert best is not None
    placed = best[1] + pc * scale
    return {name: (float(placed[i, 0]), float(placed[i, 1])) for i, name in enumerate(names)}
