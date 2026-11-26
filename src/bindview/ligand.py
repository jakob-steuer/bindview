"""Building the ligand as an RDKit molecule *from the structure file*.

The whole point of this module is the atom-name map. A SMILES laid out on its
own has an atom order unrelated to the structure file, so an interaction
reported against ``F2`` could not be attached to anything. Building the
molecule from the ``_atom_site`` records keeps each name on its atom through
bond perception and 2D layout.

The route:

1. atoms + 3D conformer from the structure, ``atomName`` set on every atom;
2. connectivity from geometry (``rdDetermineBonds.DetermineConnectivity``);
3. bond orders, charges and hydrogen counts from a SMILES template;
4. 2D coordinates.

Two hardening steps sit between 2 and 3, both needed by real AlphaFold 3 output:

* Geometry produces bonds the template does not have -- metal coordination
  (cobalamin's Co-N) and outright clashes (a peptide ligand with two atoms
  1.9 A apart). Those are identified by the template match and removed, so the
  drawn molecule is exactly the template's graph. Coordination bonds are kept
  aside for display as intra-ligand contacts.
* After transfer the result is checked against the template by canonical
  SMILES. A mismatch is an error, never a silently different molecule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from .chem import is_metal, quiet_rdkit
from .errors import LigandBuildError
from .structure import Atom

__all__ = ["Ligand", "build_ligand"]

_COV_FACTORS = (1.3, 1.4, 1.5, 1.6)
#: Of the ways a template fits the atoms of the model, how many are looked at. A drug has a handful (a benzene
#: ring turned over, the two oxygens of an acid), a peptide more than can be counted.
_WAYS = 1000
#: What the model has to say for one way against another, in ångström for each bond order that the two put on
#: another bond: half of what a single and a double bond differ by in length.
_TELLS_APART = 0.1
#: What a hydrogen of the file counts, in ångström, on an atom to which a way gives none (or the reverse).
_HYDROGEN = 0.1
#: A hydrogen of the file is on the heavy atom next to it, if that is no further than this (ångström).
_HYDROGEN_BOND = 1.6


@dataclass
class Ligand:
    """A ligand with named atoms.

    Attributes
    ----------
    mol:
        Sanitized heavy-atom RDKit molecule. Atom ``i`` carries the property
        ``atomName`` and corresponds to ``names[i]`` / ``coords3d[i]``.
    names:
        Structure-file atom name of every atom.
    coords3d:
        ``(n, 3)`` model coordinates.
    bond_source:
        ``"template"`` (orders from SMILES), ``"geometry"`` (orders inferred
        from a 3D structure that has hydrogens), ``"connectivity"`` (orders
        unknown, drawn single) or ``"single-atom"``.
    coordination:
        Intra-ligand metal coordination pairs ``(metal_idx, donor_idx)`` that
        geometry reported and the template does not bond.
    artefact_bonds:
        Geometric bonds dropped because the template has no such bond and no
        metal is involved -- atoms the model placed too close together.
    diagnostics:
        Human-readable notes on anything that deviated from the clean path.
    """

    mol: Chem.Mol
    names: list[str]
    coords3d: np.ndarray
    bond_source: str
    smiles: str | None = None
    coordination: list[tuple[int, int]] = field(default_factory=list)
    artefact_bonds: list[tuple[int, int]] = field(default_factory=list)
    stereo_mismatches: list[str] = field(default_factory=list)
    ionized: list[int] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    #: For a ligand that is a polymer chain of its own in the structure file: its residues as the
    #: file gives them (:class:`bindview.structure.ChainResidue`), and the chain's name. Empty otherwise.
    given: tuple = ()
    chain: str = ""
    #: The ions between the ligand and the protein that are drawn with it (:mod:`bindview.ions`).
    bridging: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._index = {n: i for i, n in enumerate(self.names)}

    @property
    def n_atoms(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int | None:
        """Atom index for a structure-file atom name, or ``None``."""
        return self._index.get(name)

    def element(self, idx: int) -> str:
        return self.mol.GetAtomWithIdx(idx).GetSymbol()

    def apart(self) -> "list[tuple[tuple[int, ...], tuple[int, float] | None, tuple[int, float]]]":
        """The parts of the ligand that the model binds to nothing else of it, by a bond or by coordination
        (a hydroxide the model has not put on its cobalt). For each: its atoms; the nearest metal of the
        rest and how far it is, where the rest has one; and the nearest atom of the rest and how far.
        Distances in the model, in ångström. Empty for a ligand in one piece."""
        parent = list(range(self.n_atoms))

        def root(i: int) -> int:
            while parent[i] != i:
                i = parent[i]
            return i

        pairs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in self.mol.GetBonds()] + list(self.coordination)
        for i, j in pairs:
            if root(i) != root(j):
                parent[root(i)] = root(j)
        parts: dict[int, list[int]] = {}
        for i in range(self.n_atoms):
            parts.setdefault(root(i), []).append(i)
        if len(parts) < 2:
            return []
        main = max(parts.values(), key=lambda p: (len(p), -min(p)))
        out = []
        for part in sorted((p for p in parts.values() if p is not main), key=min):
            rest = [i for i in range(self.n_atoms) if i not in part]
            d = np.linalg.norm(self.coords3d[part][:, None, :] - self.coords3d[rest][None, :, :], axis=2)
            near = rest[int(d.min(axis=0).argmin())]
            metals = [k for k, i in enumerate(rest) if is_metal(self.element(i))]
            metal = None
            if metals:
                k = min(metals, key=lambda k: float(d[:, k].min()))
                metal = (rest[k], float(d[:, k].min()))
            out.append((tuple(part), metal, (near, float(d.min()))))
        return out


# ---------------------------------------------------------------------------


def _seed_mol(atoms: Sequence[Atom]) -> Chem.RWMol:
    m = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i, a in enumerate(atoms):
        try:
            atom = Chem.Atom(a.element)
        except Exception as exc:  # RDKit raises a generic error for unknown symbols
            raise LigandBuildError(f"unknown element {a.element!r} on ligand atom {a.name}") from exc
        atom.SetProp("atomName", a.name)
        atom.SetNoImplicit(True)
        m.AddAtom(atom)
        conf.SetAtomPosition(i, Point3D(a.x, a.y, a.z))
    conf.Set3D(True)
    m.AddConformer(conf, assignId=True)
    return m


def _stripped_template(template: Chem.Mol) -> Chem.Mol:
    """Template reduced to its bare graph: single bonds, no charges, no H counts."""
    q = Chem.RWMol(template)
    for b in q.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in q.GetAtoms():
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetNumRadicalElectrons(0)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        a.SetIsotope(0)
    return q.GetMol()


def _bare_graph(m: Chem.RWMol) -> Chem.Mol:
    g = Chem.RWMol(m)
    for b in g.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in g.GetAtoms():
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetNumRadicalElectrons(0)
    return g.GetMol()


def _connect(atoms: Sequence[Atom], cov_factor: float) -> Chem.RWMol:
    m = _seed_mol(atoms)
    try:
        rdDetermineBonds.DetermineConnectivity(m, covFactor=cov_factor)
    except Exception as exc:
        raise LigandBuildError(f"could not perceive ligand connectivity: {exc}") from exc
    return m


def _fit_template(template: Chem.Mol, n_heavy: int, diagnostics: list[str]) -> Chem.Mol:
    """Drop template fragments the structure cannot contain (e.g. counter-ions)."""
    if template.GetNumAtoms() <= n_heavy:
        return template
    frags = list(Chem.GetMolFrags(template, asMols=True))
    if len(frags) == 1:
        return template
    frags.sort(key=lambda f: -f.GetNumAtoms())
    kept: list[Chem.Mol] = []
    total = 0
    for f in frags:
        if total + f.GetNumAtoms() <= n_heavy:
            kept.append(f)
            total += f.GetNumAtoms()
    if not kept or total != n_heavy:
        return template
    out = kept[0]
    for f in kept[1:]:
        out = Chem.CombineMols(out, f)
    diagnostics.append(
        f"SMILES has {template.GetNumAtoms()} heavy atoms, the structure {n_heavy}: "
        f"{len(frags) - len(kept)} template fragment(s) not present in the structure were ignored")
    return out


def _hydrogens_on(heavy: Sequence[Atom], hydrogens: Sequence[Atom]) -> np.ndarray:
    """How many hydrogens the file has on each heavy atom: every hydrogen on the atom next to it."""
    xyz = np.array([a.xyz for a in heavy], dtype=float)
    count = np.zeros(len(heavy), dtype=int)
    for h in hydrogens:
        apart = np.linalg.norm(xyz - np.array(h.xyz, dtype=float), axis=1)
        if apart.min() <= _HYDROGEN_BOND:
            count[int(apart.argmin())] += 1
    return count


def _best_way(ways: Sequence[Sequence[int]], template: Chem.Mol, atoms: Sequence[Atom],
              hydrogens: Sequence[Atom], diagnostics: list[str]) -> Sequence[int]:
    """Of the ways the template fits the atoms of the model, the one the model's geometry agrees with best.

    The graph without bond orders has symmetries that the molecule does not have: 4-hydroxycyclohexanone fits
    with the ketone on either oxygen. A benzene ring turned over is no other way: the same molecule on the
    same atoms. Of the others there are two sorts.

    Ways that differ inside the molecule, in the bond orders, charge or hydrogens of an atom with more than
    one neighbour. A way is the better the shorter the bonds of the model on which it puts its higher orders
    (the sum of order times length) and, in a file with hydrogens, the fewer hydrogens it gives an atom more
    or less than the file has on it. Another than the first is taken only where it is better by more than
    ``_TELLS_APART`` for each bond order that it puts on another bond. Where the model does not tell another
    from the one taken, ``diagnostics`` says so.

    Ways that differ in the ends of one group alone: the oxygens of a carboxylate or a phosphate, the
    nitrogens of a guanidinium. They are one group either way, and which of its bonds a model has the
    shorter says nothing: the ends stay as the first way has them, unless the hydrogens of the file say
    which of them has one.
    """
    if len(ways) == 1:
        return ways[0]
    on = np.array(ways)                                     # (ways, atoms of the template): the atom of the model
    rows, n = np.arange(len(ways))[:, None], len(atoms)
    xyz = np.array([a.xyz for a in atoms], dtype=float)
    moves = (on != on[0]).any(axis=0)                       # the atoms of the template that not all ways put alike
    bonds = [b for b in template.GetBonds() if moves[b.GetBeginAtomIdx()] or moves[b.GetEndAtomIdx()]]
    i = on[:, np.array([b.GetBeginAtomIdx() for b in bonds], dtype=int)]
    j = on[:, np.array([b.GetEndAtomIdx() for b in bonds], dtype=int)]
    pairs, where = np.unique(np.minimum(i, j) * n + np.maximum(i, j), return_inverse=True)
    orders = np.zeros((len(ways), len(pairs)))              # (ways, pairs of atoms): the bond order put there
    orders[rows, where.reshape(i.shape)] = [b.GetBondTypeAsDouble() for b in bonds]
    misfit = orders @ np.linalg.norm(xyz[pairs // n] - xyz[pairs % n], axis=1)
    off = np.zeros(len(ways))                               # the hydrogens a way gives that the file has not there
    if hydrogens:
        wanted = np.array([a.GetTotalNumHs() for a in template.GetAtoms()])[moves]
        off = np.abs(wanted - _hydrogens_on(atoms, hydrogens)[on[:, moves]]).sum(axis=1)
        misfit = misfit + _HYDROGEN * off

    # What a way gives an atom besides its bonds; the ways that are alike inside the molecule are of one sort.
    kinds: dict[tuple, int] = {}
    kind = np.zeros((len(ways), n), dtype=int)
    kind[rows, on[:, moves]] = np.array([kinds.setdefault(
        (a.GetFormalCharge(), a.GetTotalNumHs(), a.GetIsAromatic(), a.GetNumRadicalElectrons(), a.GetIsotope()),
        len(kinds) + 1) for a in template.GetAtoms()])[moves]
    inside = np.zeros(n, dtype=bool)
    inside[on[0]] = [a.GetDegree() > 1 for a in template.GetAtoms()]
    mark = np.hstack([kind[:, inside], orders[:, inside[pairs // n] & inside[pairs % n]]])
    differs = (mark != mark[0]).any(axis=0)
    alike = np.zeros(len(ways), dtype=int)                  # the same number for ways of one sort
    if differs.any():
        alike = np.unique(mark[:, differs], axis=0, return_inverse=True)[1].reshape(-1)

    def beside(way: int) -> tuple[np.ndarray, np.ndarray]:
        """Of each sort, the way to hold against this one: the fewest hydrogens off, then the fewest bond
        orders on another bond than here, then the first. And that number of bond orders, of every way."""
        apart = np.abs(orders - orders[way]).sum(axis=1) / 2
        order = np.lexsort((apart, off, alike))
        return np.sort(order[np.r_[True, alike[order][1:] != alike[order][:-1]]]), apart

    stand, apart = beside(0)
    cost = misfit[stand] + _TELLS_APART * apart[stand]
    best = int(stand[np.flatnonzero(cost <= cost.min() + 1e-9)[0]])
    stand, apart = beside(best)
    as_good = (alike[stand] != alike[best]) & (misfit[stand] - misfit[best] <= _TELLS_APART * apart[stand] + 1e-9)
    for way in stand[as_good][:1]:
        differ = set(np.flatnonzero(kind[way] != kind[best]).tolist())
        differ.update(int(end) for pair in pairs[orders[way] != orders[best]] for end in (pair // n, pair % n))
        named = [atoms[k].name for k in sorted(differ)]
        diagnostics.append(
            "the SMILES fits the model's atoms in more than one way, and the model's geometry does not say "
            f"which: another way gives {', '.join(named[:8])}{' ...' if len(named) > 8 else ''} other bond "
            "orders, charges or hydrogens")
    return ways[best]


def _apply_template(atoms: Sequence[Atom], template: Chem.Mol, diagnostics: list[str],
                    hydrogens: Sequence[Atom] = ()
                    ) -> tuple[Chem.Mol, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Template route. Returns ``None`` when the template cannot be matched. ``hydrogens``: those of the file,
    which help to say how the template lies where it fits in more than one way (:func:`_best_way`)."""
    query = _stripped_template(template)
    coords = np.array([a.xyz for a in atoms], dtype=float)
    for factor in _COV_FACTORS:
        m = _connect(atoms, factor)
        ways = _bare_graph(m).GetSubstructMatches(query, uniquify=False, maxMatches=_WAYS)
        if not ways or len(ways[0]) != template.GetNumAtoms():
            continue
        if len(set(ways[0])) != m.GetNumAtoms():
            # Template smaller than the ligand in the file: a different molecule.
            return None
        if factor != _COV_FACTORS[0]:
            diagnostics.append(
                f"template matched only after relaxing the bond-length tolerance to {factor:.2f}x covalent radii "
                f"(stretched bonds in the model)")
        match = _best_way(ways, template, atoms, hydrogens, diagnostics)
        keep = {frozenset((match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()])) for b in template.GetBonds()}
        coordination: list[tuple[int, int]] = []
        artefacts: list[tuple[int, int]] = []
        for b in list(m.GetBonds()):
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if frozenset((i, j)) in keep:
                continue
            ei, ej = m.GetAtomWithIdx(i).GetSymbol(), m.GetAtomWithIdx(j).GetSymbol()
            if is_metal(ei) or is_metal(ej):
                coordination.append((i, j) if is_metal(ei) else (j, i))
            else:
                artefacts.append((min(i, j), max(i, j)))
            m.RemoveBond(i, j)
        for b in template.GetBonds():
            b2 = m.GetBondBetweenAtoms(match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()])
            b2.SetBondType(b.GetBondType())
            b2.SetIsAromatic(b.GetIsAromatic())
        for a in template.GetAtoms():
            a2 = m.GetAtomWithIdx(match[a.GetIdx()])
            a2.SetFormalCharge(a.GetFormalCharge())
            a2.SetIsAromatic(a.GetIsAromatic())
            a2.SetNoImplicit(True)
            a2.SetNumExplicitHs(a.GetTotalNumHs())
            a2.SetNumRadicalElectrons(a.GetNumRadicalElectrons())
            a2.SetIsotope(a.GetIsotope())
        mol = m.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            metals = sorted({a.GetSymbol() for a in mol.GetAtoms() if is_metal(a.GetSymbol())})
            hint = (f". The ligand holds a metal ({', '.join(metals)}): a SMILES that writes the bonds to a metal "
                    f"like those between any two atoms gives the metal's neighbours a valence they cannot have. "
                    f"Without a SMILES the ligand is drawn from its connectivity alone") if metals else ""
            raise LigandBuildError(f"template bond orders do not sanitize on the model geometry: {exc}{hint}") from exc
        got = Chem.MolToSmiles(mol, isomericSmiles=False)
        want = Chem.MolToSmiles(template, isomericSmiles=False)
        if got != want:
            raise LigandBuildError(
                "ligand built from the structure is not the template molecule "
                f"(built {got!r}, template {want!r})")
        for i, j in artefacts:
            d = float(np.linalg.norm(coords[i] - coords[j]))
            diagnostics.append(
                f"model geometry puts {atoms[i].name} and {atoms[j].name} {d:.2f} A apart without a bond in the "
                f"template; treated as a clash, not drawn as a bond")
        return mol, coordination, artefacts
    return None


def _from_geometry(heavy: Sequence[Atom], hydrogens: Sequence[Atom], diagnostics: list[str]
                   ) -> tuple[Chem.Mol, str]:
    """No usable template: infer what can honestly be inferred from coordinates.

    Bond orders can be derived from geometry only when the structure carries
    hydrogens, because the method works by satisfying valences. On heavy atoms
    alone (all AlphaFold 3 output, most crystal structures) it "satisfies" them
    with multiple bonds that are not there -- benzene comes out with three
    triple bonds. In that situation only the connectivity is trustworthy, so
    that is all that is drawn, and the result says so.
    """
    if hydrogens:
        everything = list(heavy) + list(hydrogens)
        try:
            trial = _seed_mol(everything)
            rdDetermineBonds.DetermineBonds(trial, charge=0)
            mol = Chem.RemoveHs(trial.GetMol())
            Chem.SanitizeMol(mol)
            if mol.GetNumAtoms() == len(heavy):
                diagnostics.append("no SMILES template: bond orders were inferred from the 3D geometry "
                                   "(the structure has hydrogens)")
                return mol, "geometry"
        except Exception:
            pass
    m = _connect(heavy, _COV_FACTORS[0])
    for a in m.GetAtoms():
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetNumRadicalElectrons(0)
    mol = m.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    why = ("bond orders could not be inferred from the geometry" if hydrogens else
           "the structure has no hydrogens, so bond orders cannot be inferred from geometry")
    diagnostics.append(f"no SMILES template and {why}: only the connectivity is known, "
                       "all bonds are drawn single and hydrogens are not shown")
    return mol, "connectivity"


def _compare_stereo(mol: Chem.Mol, template: Chem.Mol, names: list[str]) -> list[str]:
    """Stereocentres whose model (3D) configuration differs from the SMILES."""
    query = Chem.Mol(template)
    Chem.AssignStereochemistry(query, cleanIt=True, force=True)
    match = mol.GetSubstructMatch(query, useChirality=False)
    if not match:
        return []
    out: list[str] = []
    for a in query.GetAtoms():
        want = a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else None
        b = mol.GetAtomWithIdx(match[a.GetIdx()])
        got = b.GetProp("_CIPCode") if b.HasProp("_CIPCode") else None
        if want and got and want != got:
            out.append(f"{names[b.GetIdx()]} is {got} in the model, {want} in the SMILES")
        elif want and not got:
            out.append(f"{names[b.GetIdx()]} is {want} in the SMILES, undefined in the model")
    return out


# Acids that are deprotonated at physiological pH. Drawn ionised so that a salt
# bridge lands on a visibly charged group; bases are left as the SMILES has
# them, because their protonation is not safe to assert from structure alone.
_ACID_SMARTS = (
    "[CX3](=[OX1])[OX2H1]",            # carboxylic acid
    "[PX4](=[OX1])[OX2H1]",            # phosphoric / phosphonic acid
    "[SX4](=[OX1])(=[OX1])[OX2H1]",    # sulfonic / sulfuric acid
)


def _ionize_acids(mol: Chem.Mol) -> tuple[Chem.Mol, list[int]]:
    """Deprotonate carboxylic, phosphoric and sulfonic acid OH groups."""
    targets: set[int] = set()
    for smarts in _ACID_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        for match in mol.GetSubstructMatches(patt):
            targets.add(match[-1])
    if not targets:
        return mol, []
    rw = Chem.RWMol(mol)
    for i in sorted(targets):
        atom = rw.GetAtomWithIdx(i)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        atom.SetFormalCharge(-1)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out, sorted(targets)


def _clear_pseudo_stereo(mol: Chem.Mol) -> None:
    """Drop stereo on P / S centres that only look chiral.

    A phosphate drawn ``P(=O)(O)`` distinguishes two oxygens that are
    equivalent by resonance; the 3D coordinates then yield an R/S label, and a
    wedge, that means nothing chemically.
    """
    changed = False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ("P", "S") or atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        terminal_o = [n for n in atom.GetNeighbors() if n.GetSymbol() in ("O", "S") and n.GetDegree() == 1]
        if len(terminal_o) >= 2:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            changed = True
    if changed:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


def build_ligand(atoms: Sequence[Atom], smiles: str | None = None, ionize_acids: bool = True) -> Ligand:
    """Build a :class:`Ligand` from structure atoms and an optional SMILES template.

    Hydrogens present in the structure are not drawn; the diagram is drawn on the
    heavy-atom skeleton with hydrogen counts taken from the template.

    A SMILES can fit the atoms in more than one way: the bonds of the structure have
    no orders, and without them the ketone and the hydroxyl of 4-hydroxycyclohexanone
    are alike. The way is taken whose bond orders agree best with the bond lengths of
    the structure and, where the file has hydrogens, with the hydrogens on each atom;
    where the structure does not tell two ways apart, the first, and ``diagnostics``
    says so. The ends of one group (the two oxygens of a carboxylate) are one group
    either way: they lie as the first way has them, whatever their bond lengths,
    unless the hydrogens of the file say which of them has one.

    With ``ionize_acids`` (default) carboxylic, phosphoric and sulfonic acid
    groups are deprotonated for display, as they are at physiological pH.
    """
    heavy = [a for a in atoms if a.element not in ("H", "D")]
    hydrogens = [a for a in atoms if a.element in ("H", "D")]
    if not heavy:
        raise LigandBuildError("ligand has no heavy atoms")
    names = [a.name for a in heavy]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise LigandBuildError(f"ligand atom names are not unique: {', '.join(dup[:6])}")
    coords = np.array([a.xyz for a in heavy], dtype=float)
    diagnostics: list[str] = []

    with quiet_rdkit():
        template: Chem.Mol | None = None
        if smiles:
            template = Chem.MolFromSmiles(smiles)
            if template is None:
                raise LigandBuildError(f"SMILES does not parse: {smiles!r}")
            for a in template.GetAtoms():
                a.SetAtomMapNum(0)           # ([CH3:1]: which atom is which in a reaction, nothing of the molecule)
            template = _fit_template(template, len(heavy), diagnostics)

        if len(heavy) == 1:
            m = _seed_mol(heavy)
            atom = m.GetAtomWithIdx(0)
            if template is not None and template.GetNumAtoms() == 1 \
                    and template.GetAtomWithIdx(0).GetSymbol() == atom.GetSymbol():
                t = template.GetAtomWithIdx(0)
                atom.SetFormalCharge(t.GetFormalCharge())
                atom.SetNumExplicitHs(t.GetTotalNumHs())
            elif template is not None:
                diagnostics.append("SMILES does not describe the single ligand atom in the structure; ignored")
            mol = m.GetMol()
            mol.UpdatePropertyCache(strict=False)
            return Ligand(mol=mol, names=names, coords3d=coords, bond_source="single-atom",
                          smiles=smiles, diagnostics=diagnostics)

        coordination: list[tuple[int, int]] = []
        artefacts: list[tuple[int, int]] = []
        source = "template"
        built = _apply_template(heavy, template, diagnostics, hydrogens) if template is not None else None
        if built is not None:
            mol, coordination, artefacts = built
        else:
            if template is not None:
                diagnostics.append(
                    "the SMILES template does not match the ligand in the structure "
                    f"({template.GetNumAtoms()} template atoms, {len(heavy)} structure atoms); it was not used")
            mol, source = _from_geometry(heavy, hydrogens, diagnostics)

        ionized: list[int] = []
        if ionize_acids and source in ("template", "geometry"):
            mol, ionized = _ionize_acids(mol)

        stereo: list[str] = []
        if source != "connectivity":
            try:
                Chem.AssignStereochemistryFrom3D(mol)
                _clear_pseudo_stereo(mol)
            except Exception:  # pragma: no cover - defensive, RDKit rarely fails here
                diagnostics.append("stereochemistry could not be perceived from the 3D coordinates")
            if template is not None and source == "template":
                stereo = _compare_stereo(mol, template, names)
                if stereo:
                    diagnostics.append(
                        f"{len(stereo)} stereocentre(s) differ between the model and the SMILES: " + "; ".join(stereo))
        return Ligand(mol=mol, names=names, coords3d=coords, bond_source=source, smiles=smiles,
                      coordination=sorted(coordination), artefact_bonds=sorted(artefacts),
                      stereo_mismatches=stereo, ionized=ionized, diagnostics=diagnostics)
