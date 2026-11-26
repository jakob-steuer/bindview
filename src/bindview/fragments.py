"""2D geometry for the part of a residue that is drawn explicitly.

In the detailed style a residue that makes a directional contact is not a
bubble: the piece of it that makes the contact is drawn as a small skeletal
formula in a second colour, the way LigPlot shows H-bonded residues. This
module decides which atoms that piece consists of and lays them out. Rotation
and placement on the page are the layout engine's business.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor

from .chem import quiet_rdkit
from .geometry import Point
from .residues import (NUCLEOTIDE_BACKBONE, PHOSPHATE, SUGAR, TAIL, ResidueTemplate, canonical_atom,
                       canonical_resname, element_of, given_atom, template_bonds, template_for)

__all__ = ["Fragment", "fragment_atoms", "build_fragment", "conformations", "DETAILS", "check_detail"]

#: How much of a residue is drawn where it is drawn with atoms: nothing of it (it stays one labelled
#: thing), its backbone, its side chain, or every heavy atom. ``None`` is not among them: it asks for
#: no level, and a residue is then what its style makes of it.
DETAILS = ("box", "backbone", "sidechain", "atoms")


def check_detail(detail: "str | None", what: str = "detail") -> "str | None":
    """``detail`` if it is ``None`` or one of :data:`DETAILS`; a :class:`ValueError` that names them otherwise."""
    if detail is None or detail in DETAILS:
        return detail
    raise ValueError(f"{what} is {', '.join(repr(d) for d in DETAILS[:-1])} or {DETAILS[-1]!r} "
                     f"(or None, for what the style makes of a residue by itself), not {detail!r}")


@dataclass(frozen=True)
class Fragment:
    resname: str
    atoms: tuple[str, ...]
    elements: dict[str, str]
    xy: dict[str, Point]                       # local coordinates, y down, in user units
    bonds: tuple[tuple[str, str, int], ...]
    rings: tuple[tuple[str, ...], ...]
    root: str                                  # the residue label sits beyond this atom

    def centroid(self, names: Iterable[str]) -> Point:
        pts = [self.xy[n] for n in names]
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def fragment_atoms(resname: str, involved: Iterable[str], detail: "str | None" = None) -> tuple[str, ...] | None:
    """Which atoms of a residue to draw, given the atoms that make contacts.

    The rule keeps fragments recognisable and small: the alpha carbon always;
    the *whole* side chain if any side-chain atom is involved (half an arginine
    is not an arginine); the backbone nitrogen and/or carbonyl only when they
    are themselves involved. Returns ``None`` for residues without a template
    or contacts through atoms the template does not have -- those are drawn as
    a bare labelled atom instead.

    ``detail`` (one of :data:`DETAILS`) adds to that and takes nothing away:
    ``"backbone"`` the nitrogen and the carbonyl, ``"sidechain"`` the side
    chain, ``"atoms"`` both. The part that makes the contact is drawn at
    every level, so a line of the residue always ends on an atom.

    A nucleotide has three parts and the same rule (:func:`_nucleotide_atoms`):
    its base is its side chain, its sugar and its phosphate its backbone.
    """
    t = template_for(resname)
    if t is None:
        return None
    wanted = {canonical_atom(resname, a) for a in involved}
    if not wanted <= set(t.atoms):
        return None
    if t.kind == "nucleotide":
        return _nucleotide_atoms(t, wanted, detail)
    side = set(t.sidechain)
    proline = canonical_resname(resname) == "PRO"
    keep = {"CA"}
    if wanted & side:
        keep |= side
        if proline:
            keep.add("N")                       # the ring closes on the backbone nitrogen
    if "N" in wanted:
        keep.add("N")
    if wanted & {"C", "O"}:
        keep |= {"C", "O"}
    if len(keep) == 1:
        keep |= {"N", "C", "O"}                 # a lone vertex would be invisible
    # What the level adds to the part that makes the contact.
    if detail in ("sidechain", "atoms"):
        keep |= side
        if proline:
            keep.add("N")
    if detail in ("backbone", "atoms"):
        keep |= {"N", "C", "O"}
    return tuple(a for a in t.atoms if a in keep)


def _nucleotide_atoms(t: ResidueTemplate, wanted: "set[str]", detail: "str | None") -> tuple[str, ...]:
    """Which atoms of a nucleotide to draw, given those that make contacts (under the template's names).

    The part that makes the contact, whole: the base with C1' if any atom of the base is involved; the
    ring of the sugar, with the hydroxyl that is involved; the phosphate with the O5' and C5' that hang
    it on the sugar. A phosphate alone is drawn without the sugar, as five atoms: most contacts of a
    charged ligand are with phosphates, and a sugar on each would crowd the figure. Where both ends of a
    nucleotide are involved the sugar between them is drawn, so that the fragment is one piece.
    ``detail`` adds the base (``"sidechain"``), sugar and phosphate (``"backbone"``) or all (``"atoms"``).
    """
    base = set(t.atoms) - set(NUCLEOTIDE_BACKBONE)
    ring = {"C1'", "C2'", "C3'", "C4'", "O4'"}
    keep: set[str] = set()
    if wanted & base:
        keep |= base | {"C1'"}
    if wanted & set(SUGAR):
        keep |= ring | (wanted & {"O2'", "O3'"})
    if wanted & set(PHOSPHATE):
        keep |= set(PHOSPHATE) | set(TAIL)
    if wanted & set(TAIL):
        keep |= set(TAIL)
    if detail in ("sidechain", "atoms"):
        keep |= base | {"C1'"}
    if detail in ("backbone", "atoms"):
        keep |= set(NUCLEOTIDE_BACKBONE)
    if keep & (base | ring) and keep & (set(PHOSPHATE) | set(TAIL)):
        keep |= ring | set(TAIL)                # the sugar, between a base or a hydroxyl and the phosphate
    return tuple(a for a in t.atoms if a in keep)


@lru_cache(maxsize=None)
def _unit_layout(resname: str, atoms: tuple[str, ...]) -> tuple[tuple[tuple[float, float], ...],
                                                              tuple[tuple[str, ...], ...]]:
    """Unit-bond-length 2D coordinates (y up) and rings for a fragment."""
    t = template_for(resname)
    assert t is not None
    with quiet_rdkit():
        full = Chem.MolFromSmiles(t.smiles)
        keep = [t.atoms.index(a) for a in atoms]
        new_index = {old: new for new, old in enumerate(keep)}
        rw = Chem.RWMol()
        for old in keep:
            atom = Chem.Atom(full.GetAtomWithIdx(old).GetSymbol())
            atom.SetNoImplicit(True)
            rw.AddAtom(atom)
        for b in full.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in new_index and j in new_index:
                rw.AddBond(new_index[i], new_index[j], Chem.BondType.SINGLE)
        mol = rw.GetMol()
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
        prefer = rdDepictor.GetPreferCoordGen()
        rdDepictor.SetPreferCoordGen(False)
        try:
            rdDepictor.Compute2DCoords(mol, canonOrient=True)
        finally:
            rdDepictor.SetPreferCoordGen(prefer)
        conf = mol.GetConformer()
        xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
        lengths = [float(np.linalg.norm(xy[b.GetBeginAtomIdx()] - xy[b.GetEndAtomIdx()])) for b in mol.GetBonds()]
        scale = float(np.median(lengths)) if lengths else 1.0
        xy = xy / scale if scale > 1e-9 else xy
        rings = tuple(tuple(atoms[i] for i in ring) for ring in mol.GetRingInfo().AtomRings())
        return tuple((float(x), float(y)) for x, y in xy), rings


def build_fragment(resname: str, involved: Iterable[str], bond_length: float,
                   detail: "str | None" = None, like: "Iterable[str] | None" = None) -> Fragment | None:
    """The drawable fragment of ``resname`` for the given contact atoms, or ``None``; with ``detail``,
    at least that much of the residue (:func:`fragment_atoms`).

    The atoms of a nucleotide are named as a file names them, which is one of two ways (``OP1`` or
    ``O1P``, ``C1'`` or ``C1*``): ``like`` is the names the contacts of the whole figure use, so that
    every nucleotide of a figure is named in one way; left out, the names of ``involved`` decide."""
    involved = tuple(involved)
    atoms = fragment_atoms(resname, involved, detail)
    if atoms is None:
        return None
    unit, rings = _unit_layout(canonical_resname(resname), atoms)
    xy = {a: (x * bond_length, -y * bond_length) for a, (x, y) in zip(atoms, unit)}
    chosen = set(atoms)
    bonds = tuple((a, b, order) for a, b, order in template_bonds(canonical_resname(resname))
                  if a in chosen and b in chosen)
    elements = {a: element_of(a) for a in atoms}
    t = template_for(resname)
    if t is None or t.kind != "nucleotide":
        return Fragment(resname=canonical_resname(resname), atoms=atoms, elements=elements, xy=xy,
                        bonds=bonds, rings=rings, root="CA")
    # A nucleotide: its name stands beyond C1', or beyond C5' where the phosphate is drawn alone. And its
    # atoms are named as the figure's contacts name them, so that a contact finds its atom under its name.
    root = "C1'" if "C1'" in chosen else "C5'"
    like = involved if like is None else tuple(like)
    own = {canonical_atom(resname, a): a for a in involved}

    def as_given(name: str) -> str:
        return own.get(name) or given_atom(name, like)

    return Fragment(resname=canonical_resname(resname), atoms=tuple(as_given(a) for a in atoms),
                    elements={as_given(a): e for a, e in elements.items()},
                    xy={as_given(a): p for a, p in xy.items()},
                    bonds=tuple((as_given(a), as_given(b), order) for a, b, order in bonds),
                    rings=tuple(tuple(as_given(a) for a in ring) for ring in rings), root=as_given(root))


#: A fragment has at most this many layouts (:func:`conformations`).
_MOST_LAYOUTS = 24


def _reflect(p: Point, a: Point, b: Point) -> Point:
    """Mirror ``p`` in the line through ``a`` and ``b``."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n2 = dx * dx + dy * dy
    if n2 < 1e-12:
        return p
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / n2
    fx, fy = a[0] + t * dx, a[1] + t * dy
    return (2.0 * fx - p[0], 2.0 * fy - p[1])


def conformations(frag: Fragment) -> list[dict[str, Point]]:
    """Alternative flat layouts of a fragment, the template one first.

    A long side chain drawn fully extended cannot bring its tip and its
    backbone near the same part of a ligand. LigPlot's answer, used here too, is
    to flip the far part of the chain about a bond: bond lengths and angles are
    untouched, the zig-zag gains a kink. Every layout with up to two such flips
    that keeps non-bonded atoms apart is returned, :data:`_MOST_LAYOUTS` at most.
    """
    nbrs: dict[str, list[str]] = {a: [] for a in frag.atoms}
    for a, b, _ in frag.bonds:
        nbrs[a].append(b)
        nbrs[b].append(a)
    ring_bond = {frozenset((r[i], r[(i + 1) % len(r)])) for r in frag.rings for i in range(len(r))}
    # Depth-first from the root gives every atom its parent and its subtree.
    parent: dict[str, str | None] = {frag.root: None}
    order = [frag.root]
    for a in order:
        for b in nbrs[a]:
            if b not in parent:
                parent[b] = a
                order.append(b)
    children: dict[str, list[str]] = {a: [] for a in frag.atoms}
    for a, par in parent.items():
        if par is not None:
            children[par].append(a)

    def subtree(a: str) -> list[str]:
        out, stack = [], list(children[a])
        while stack:
            x = stack.pop()
            out.append(x)
            stack.extend(children[x])
        return out

    flippable = []
    for far in order[1:]:
        near = parent[far]
        assert near is not None
        if frozenset((near, far)) in ring_bond or not children[far]:
            continue                      # ring bonds do not flip; a terminal atom has nothing to move
        flippable.append((near, far))

    bonded = {frozenset((a, b)) for a, b, _ in frag.bonds}
    lengths = [((frag.xy[a][0] - frag.xy[b][0]) ** 2 + (frag.xy[a][1] - frag.xy[b][1]) ** 2) ** 0.5
               for a, b, _ in frag.bonds]
    unit_len = sorted(lengths)[len(lengths) // 2] if lengths else 1.0

    def clean(xy: dict[str, Point]) -> bool:
        names = list(frag.atoms)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if frozenset((a, b)) in bonded:
                    continue
                if ((xy[a][0] - xy[b][0]) ** 2 + (xy[a][1] - xy[b][1]) ** 2) ** 0.5 < 0.8 * unit_len:
                    return False
        return True

    def apply(flips: tuple[int, ...]) -> dict[str, Point]:
        xy = dict(frag.xy)
        for k in flips:                       # ``flippable`` is ordered root outward
            a, b = flippable[k]
            for x in subtree(b):
                xy[x] = _reflect(xy[x], xy[a], xy[b])
        return xy

    out = [dict(frag.xy)]
    seen = {tuple(round(v, 3) for a in frag.atoms for v in frag.xy[a])}
    combos: list[tuple[int, ...]] = [(k,) for k in range(len(flippable))]
    combos += [(i, j) for i in range(len(flippable)) for j in range(i + 1, len(flippable))]
    for flips in combos:
        xy = apply(flips)
        sig = tuple(round(v, 3) for a in frag.atoms for v in xy[a])
        if sig in seen or not clean(xy):
            continue
        seen.add(sig)
        out.append(xy)
        if len(out) >= _MOST_LAYOUTS:
            break
    return out
