"""Seeing a peptide in a ligand that is stored as one lump of atoms.

A co-folded peptide ligand is not a chain of named residues in the structure
file: it was handed to the predictor as a ligand, so liraglutide is one hetero
component of 266 atoms with a single residue record. Nothing can be read; the
decomposition has to be *perceived* from the bonds.

1. **Backbone units.** Every ``N-CA-C(=O)`` with an sp3 alpha carbon.
2. **Runs.** Two units are consecutive when the carbonyl carbon of one is
   bonded to the nitrogen of the next. The longest run is the chain; it is
   walked from its N-terminus (a ring of units from a canonical start).
3. **Residues.** Each unit's side chain is compared with the 20 proteinogenic
   side chains and a short list of common others. A side chain is recognised
   when its atoms match exactly; more may hang off one of its heteroatoms (an
   acyl group on a lysine nitrogen, a phosphate on a serine oxygen), never off
   a carbon. A unit that matches nothing is ``Xaa`` and keeps its whole side
   chain.
4. **The rest.** Atoms that belong to no residue are *appendages* (a fatty acid
   on its linker, a terminal cap), each with the bonds that attach it; bonds
   between two residues that are not peptide bonds are *crosslinks*.

Whether a molecule with such a chain is *drawn* as one is a separate question,
answered by :func:`decide`: the rule and the reason for its two numbers are in
``docs/reference.md``.

Positions count from the N-terminus of the chain **as perceived here**. They
are not sequence-database numbers, and nothing in this module pretends so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from rdkit import Chem

from .residues import _NUCLEOTIDES, _TEMPLATES, NUCLEOTIDE_BACKBONE, canonical_resname

__all__ = ["PeptideResidue", "Appendage", "Crosslink", "Peptide", "Decision", "perceive_peptide", "decide",
           "decide_given", "MIN_RESIDUES", "MIN_RECOGNISED"]

#: A chain is drawn as a chain from this many residues on (see :func:`decide`).
MIN_RESIDUES = 8
#: A chain of nucleotides that a file gives is drawn as a chain from this many on: two nucleotides are
#: about 45 heavy atoms, which the drawing of atoms holds, and three are not.
MIN_NUCLEOTIDES = 3
#: ... and only if at least this share of them carries a proteinogenic side chain.
MIN_RECOGNISED = 0.5

_UNIT = Chem.MolFromSmarts("[#7;X3,X4+][#6;X4][#6;X3]=[#8;X1]")

_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
        "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V"}

#: Common residues outside the twenty: named in the drawing, not counted as proteinogenic.
#: SMILES atom order is N, CA, C, O, then the side chain (as in :mod:`bindview.residues`).
_OTHERS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("AIB", "NC(C=O)(C)C", ("N", "CA", "C", "O", "CB1", "CB2")),
    ("ABU", "NC(C=O)CC", ("N", "CA", "C", "O", "CB", "CG")),
    ("NVA", "NC(C=O)CCC", ("N", "CA", "C", "O", "CB", "CG", "CD")),
    ("NLE", "NC(C=O)CCCC", ("N", "CA", "C", "O", "CB", "CG", "CD", "CE")),
    ("ORN", "NC(C=O)CCCN", ("N", "CA", "C", "O", "CB", "CG", "CD", "NE")),
    ("DAB", "NC(C=O)CCN", ("N", "CA", "C", "O", "CB", "CG", "ND")),
    ("CIT", "NC(C=O)CCCNC(N)=O", ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH", "OH")),
    ("HYP", "N1C(C=O)CC(O)C1", ("N", "CA", "C", "O", "CB", "CG", "OD", "CD")),
    ("PCA", "N1C(C=O)CCC1=O", ("N", "CA", "C", "O", "CB", "CG", "CD", "OE")),
    ("HSE", "NC(C=O)CCO", ("N", "CA", "C", "O", "CB", "CG", "OD")),
    ("PEN", "NC(C=O)C(C)(C)S", ("N", "CA", "C", "O", "CB", "CG1", "CG2", "SG")),
    ("TLE", "NC(C=O)C(C)(C)C", ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CG3")),
    ("CHA", "NC(C=O)CC1CCCCC1", ("N", "CA", "C", "O", "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2")),
    ("PHG", "NC(C=O)c1ccccc1", ("N", "CA", "C", "O", "CB", "CG1", "CD1", "CE", "CD2", "CG2")),
    ("HPH", "NC(C=O)CCc1ccccc1", ("N", "CA", "C", "O", "CB", "CG", "CD", "CE1", "CZ1", "CH", "CZ2", "CE2")),
    ("NAL", "NC(C=O)Cc1ccc2ccccc2c1",
     ("N", "CA", "C", "O", "CB", "CG", "CD1", "CE1", "CZ1", "CH1", "CT1", "CT2", "CH2", "CZ2", "CE2", "CD2")),
    ("SEC", "NC(C=O)C[Se]", ("N", "CA", "C", "O", "CB", "SE")),
    ("MSE", "NC(C=O)CC[Se]C", ("N", "CA", "C", "O", "CB", "CG", "SE", "CE")),
)


def _title(name: str) -> str:
    return name[0] + name[1:].lower()


@dataclass(frozen=True)
class _Template:
    name: str
    atoms: tuple[str, ...]
    query: Chem.Mol
    degrees: tuple[int, ...]            # heavy-atom degree of every template atom
    carbon: tuple[bool, ...]
    proteinogenic: bool


def _query(smiles: str) -> tuple[Chem.Mol, tuple[int, ...], tuple[bool, ...]]:
    """A substructure query that fixes elements, aromaticity and the geometry of carbons, and no bond order.

    Bond orders are left open because a guanidine, a carboxylate or an
    imidazole is written with its double bond in different places by different
    sources; that a carbon is sp3, sp2 or aromatic is what tells cyclohexyl-
    alanine from phenylalanine.
    """
    t = Chem.MolFromSmiles(smiles)
    q = Chem.RWMol()
    for a in t.GetAtoms():
        if a.GetAtomicNum() == 6 and not a.GetIsAromatic():
            smarts = f"[#6;A;X{a.GetTotalDegree()}]"
        elif a.GetIsAromatic():
            smarts = f"[#{a.GetAtomicNum()};a]"
        else:
            smarts = f"[#{a.GetAtomicNum()};A]"
        q.AddAtom(Chem.AtomFromSmarts(smarts))
    for b in t.GetBonds():
        q.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), Chem.BondType.SINGLE)
        q.ReplaceBond(q.GetNumBonds() - 1, Chem.BondFromSmarts("~"))
    return q.GetMol(), tuple(a.GetDegree() for a in t.GetAtoms()), tuple(a.GetAtomicNum() == 6 for a in t.GetAtoms())


def _templates() -> tuple[_Template, ...]:
    out = []
    for t in _TEMPLATES.values():
        q, deg, carbon = _query(t.smiles)
        out.append(_Template(t.name, t.atoms, q, deg, carbon, True))
    for name, smiles, atoms in _OTHERS:
        q, deg, carbon = _query(smiles)
        out.append(_Template(name, atoms, q, deg, carbon, False))
    # Largest first: where two templates fit a unit exactly, the larger says more.
    return tuple(sorted(out, key=lambda t: (-len(t.atoms), t.name)))


_ALL = _templates()


@dataclass
class _Found:
    """A unit of the chain while its atoms are being collected."""

    n: int
    ca: int
    c: int
    o: int
    template: _Template | None = None
    atoms: list[int] = field(default_factory=list)
    names: dict[int, str] = field(default_factory=dict)
    n_methyl: bool = False
    n_substituted: bool = False

    @property
    def name(self) -> str:
        return self.template.name if self.template is not None else "XAA"


@dataclass(frozen=True)
class PeptideResidue:
    """One residue of the chain, as perceived."""

    position: int                       # 1-based, from the N-terminus of the chain as perceived
    name: str                           # "LYS", "AIB", "XAA"
    atoms: tuple[int, ...]              # ligand atom indices that belong to it
    atom_names: tuple[tuple[int, str], ...]   # (atom index, name within the residue) where it is known
    n: int
    ca: int
    c: int
    o: int
    proteinogenic: bool
    stereo: str | None = None           # "L" | "D" | None (glycine, or not determinable)
    n_methyl: bool = False
    #: The residue's number in the structure file, where the chain was given by the file ("63", "63A").
    #: Empty for a perceived residue, which has only its position.
    number: str = ""
    #: ``"amino"``, or ``"nucleotide"`` for a residue of RNA or DNA. Of a nucleotide ``n`` is the atom its
    #: chain comes in by (the phosphorus), ``c`` the one it goes on by (``O3'``), and ``ca`` its ``C4'``.
    kind: str = "amino"

    @property
    def known(self) -> bool:
        return self.name != "XAA"

    @property
    def shown(self) -> str:
        """The number printed with the residue: the deposited one, or else its position in the chain."""
        return self.number or str(self.position)

    @property
    def code3(self) -> str:
        """Three-letter code as it is printed: ``Lys``, ``Aib``, ``Xaa``; a D-residue is ``D-Phe``.
        A nucleotide keeps its capitals and is named as the templates name it: ``G``, ``DA``."""
        if self.kind == "nucleotide":
            return canonical_resname(self.name)
        base = _title(self.name)
        if self.n_methyl:
            base = "Me" + base
        return ("D-" + base) if self.stereo == "D" else base

    @property
    def code1(self) -> str:
        """One-letter code: lower case for a D-residue, ``X`` outside the twenty. Of a nucleotide its
        base, ``N`` for one that is not among the eight."""
        if self.kind == "nucleotide":
            return _NUCLEOTIDES.get(canonical_resname(self.name), "N")
        letter = _ONE.get(self.name, "X")
        return letter.lower() if self.stereo == "D" and letter != "X" else letter

    def name_of(self, atom: int) -> str | None:
        for idx, name in self.atom_names:
            if idx == atom:
                return name
        return None

    def is_backbone(self, atom: int) -> bool:
        if self.kind == "nucleotide":
            return self.name_of(atom) in NUCLEOTIDE_BACKBONE + ("OP3",)
        return self.name_of(atom) in ("N", "CA", "C", "O", "OXT")


@dataclass(frozen=True)
class Appendage:
    """Atoms that are in no residue, and the bonds that attach them to the chain."""

    atoms: tuple[int, ...]
    #: ``(position of the residue, atom of the residue, atom of the appendage)`` per attaching bond
    attachments: tuple[tuple[int, int, int], ...]

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(sorted({a[0] for a in self.attachments}))


@dataclass(frozen=True)
class Crosslink:
    """A bond between two residues of the chain that is not a peptide bond."""

    a: tuple[int, int]                  # (position, atom)
    b: tuple[int, int]
    kind: str                           # "disulfide" | "bond"


@dataclass(frozen=True)
class Peptide:
    residues: tuple[PeptideResidue, ...]
    appendages: tuple[Appendage, ...] = ()
    crosslinks: tuple[Crosslink, ...] = ()
    cyclic: bool = False                # the chain closes on itself head to tail
    runs: tuple[int, ...] = ()          # the length of every run of units in the molecule, longest first
    n_atoms: int = 0
    #: Whose the numbers of the residues are: ``"derived"`` -- a count from the N-terminus of a chain
    #: that was perceived -- or ``"deposited"``, read from the structure file with the chain itself.
    numbering: str = "derived"
    #: Positions after which a chain given by a file is not continuous (residues missing from the model).
    breaks: tuple[int, ...] = ()
    #: Does the chain go on before the first and after the last residue that is drawn? True for the
    #: spine of an interface that starts in the middle of a chain: no "N" or "C" is printed there.
    open_ends: tuple[bool, bool] = (False, False)
    #: Draw one line per pair of residues and kind of contact (the spine of an interface).
    pairwise: bool = False
    #: What the chain is called where a tooltip names it: a ligand, or one side of an interface.
    called: str = "the ligand"
    #: Do the residues stand in the order of the sequence? False for the spine of an interface that is
    #: ordered along the interface: then nothing joins two boxes unless they are neighbours in the chain.
    sequential: bool = True
    #: How much of a residue that makes a directional contact is drawn beside its box: ``None`` (nothing),
    #: or one of :data:`bindview.fragments.DETAILS`.
    detail: "str | None" = None
    _owner: dict[int, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        owner = {a: r.position for r in self.residues for a in r.atoms}
        object.__setattr__(self, "_owner", owner)

    def __len__(self) -> int:
        return len(self.residues)

    def residue(self, position: int) -> PeptideResidue:
        """The residue at a position, counted from 1."""
        return self.residues[position - 1]

    def residue_of(self, atom: int) -> PeptideResidue | None:
        """The residue an atom belongs to, or ``None`` for an atom of an appendage."""
        pos = self._owner.get(atom)
        return None if pos is None else self.residues[pos - 1]

    def appendage_of(self, atom: int) -> Appendage | None:
        """What is bonded to the chain and is no residue of it, at this atom."""
        for app in self.appendages:
            if atom in app.atoms:
                return app
        return None

    @property
    def sequence(self) -> str:
        """One-letter codes from the N-terminus; ``X`` for what is not one of the twenty."""
        return "".join(r.code1 for r in self.residues)

    @property
    def recognised(self) -> int:
        """The residues whose side chain is known."""
        return sum(1 for r in self.residues if r.proteinogenic)

    @property
    def in_chain(self) -> int:
        """Heavy atoms that belong to a residue of the chain."""
        return len(self._owner)


@dataclass(frozen=True)
class Decision:
    """Whether a ligand is drawn as a chain, and why -- the why is printed, so it is a sentence."""

    polymer: bool
    reason: str
    longest_run: int = 0
    recognised: int = 0
    #: What the ligand is by the rules of :func:`decide` and :func:`decide_given`, when the caller
    #: asked for it to be drawn otherwise (``ligand_as``). ``None``: the same as ``polymer``.
    found: bool | None = None

    @property
    def is_chain(self) -> bool:
        """Is the ligand a chain of residues by what it is, however it is drawn?"""
        return self.polymer if self.found is None else self.found


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

def _units(mol: Chem.Mol) -> list[tuple[int, int, int, int]]:
    """Every backbone unit as ``(N, CA, C, O)``, one per alpha carbon."""
    seen: dict[int, tuple[int, int, int, int]] = {}
    for n, ca, c, o in sorted(mol.GetSubstructMatches(_UNIT, uniquify=True, maxMatches=1000000)):
        seen.setdefault(ca, (n, ca, c, o))
    return sorted(seen.values())


def _runs(mol: Chem.Mol, units: Sequence[tuple[int, int, int, int]]) -> list[tuple[list[int], bool]]:
    """Runs of consecutive units, as lists of indices into ``units``, each with whether it is a ring."""
    by_n: dict[int, int] = {}
    for i, u in enumerate(units):
        by_n.setdefault(u[0], i)
    nxt: dict[int, int] = {}
    prev: dict[int, int] = {}
    for i, (_n, _ca, c, _o) in enumerate(units):
        for nb in mol.GetAtomWithIdx(c).GetNeighbors():
            j = by_n.get(nb.GetIdx())
            if j is not None and j != i and j not in prev and i not in nxt:
                nxt[i], prev[j] = j, i
    runs: list[tuple[list[int], bool]] = []
    used: set[int] = set()
    for i in range(len(units)):                       # open runs, from their first unit
        if i in prev or i in used:
            continue
        run = [i]
        while run[-1] in nxt:
            run.append(nxt[run[-1]])
        used.update(run)
        runs.append((run, False))
    for i in range(len(units)):                       # what is left closes on itself
        if i in used:
            continue
        run = [i]
        while nxt[run[-1]] != i:
            run.append(nxt[run[-1]])
        used.update(run)
        runs.append((run, True))
    runs.sort(key=lambda r: (-len(r[0]), min(r[0])))
    return runs


def _matches(mol: Chem.Mol) -> dict[int, list[tuple[_Template, tuple[int, ...]]]]:
    """For every alpha carbon: the templates whose atoms are there exactly, with the atoms they map to."""
    out: dict[int, list[tuple[_Template, tuple[int, ...]]]] = {}
    for t in _ALL:
        best: dict[int, tuple[int, ...]] = {}
        for m in mol.GetSubstructMatches(t.query, uniquify=False, maxMatches=1000000):
            ok = True
            for k, a in enumerate(m):
                name = t.atoms[k]
                if name in ("N", "C", "O"):
                    continue
                atom = mol.GetAtomWithIdx(a)
                if t.carbon[k] and atom.GetDegree() != t.degrees[k]:
                    ok = False                       # something hangs off a carbon: not this residue
                    break
                if not t.carbon[k] and atom.GetDegree() < t.degrees[k]:
                    ok = False
                    break
            if ok and (m[1] not in best or m < best[m[1]]):
                best[m[1]] = m
        for ca, m in best.items():
            out.setdefault(ca, []).append((t, m))
    return out


def _stereo(mol: Chem.Mol, n: int, ca: int, c: int) -> str | None:
    """L or D at an alpha carbon, from its CIP label; ``None`` where there is none."""
    atom = mol.GetAtomWithIdx(ca)
    if not atom.HasProp("_CIPCode"):
        return None
    label = atom.GetProp("_CIPCode")
    side = [nb for nb in atom.GetNeighbors() if nb.GetIdx() not in (n, c)]
    if len(side) != 1:
        return None                                   # glycine, or two side chains (Aib): no L or D
    # L is S unless the side chain outranks the carboxyl carbon, which it does only through an
    # atom heavier than oxygen on the beta carbon (cysteine and its like).
    heavy = max((x.GetAtomicNum() for x in side[0].GetNeighbors() if x.GetIdx() != ca), default=0)
    l_label = "R" if heavy > 8 else "S"
    return "L" if label == l_label else "D"


def perceive_peptide(mol: Chem.Mol, stereo_from: Chem.Mol | None = None) -> Peptide | None:
    """The peptide chain in a molecule, or ``None`` if it has no two consecutive backbone units.

    ``stereo_from`` is a molecule with trustworthy stereochemistry (the SMILES
    the ligand was specified with): L and D are then read from it, residue by
    residue, and not from a model that may have inverted a centre.
    """
    units = _units(mol)
    if not units:
        return None
    runs = _runs(mol, units)
    lengths = tuple(len(r) for r, _ in runs)
    order, cyclic = runs[0]
    if len(order) < 2:
        return None
    exact = _matches(mol)
    chain = [units[i] for i in order]
    backbone = {a for u in chain for a in u}

    def assign(start: int) -> tuple[list[_Found], set[int]]:
        seq = chain[start:] + chain[:start]
        taken: set[int] = set(backbone)
        found: list[_Found] = []
        for n, ca, c, o in seq:
            fits = [(t, m) for t, m in exact.get(ca, []) if m[0] == n and m[2] == c]
            entry = _Found(n, ca, c, o, atoms=[n, ca, c, o], names={n: "N", ca: "CA", c: "C", o: "O"})
            if fits:
                entry.template, match = fits[0]
                for atom_name, a in zip(entry.template.atoms, match):
                    entry.names[a] = atom_name
                    if a not in entry.atoms:
                        entry.atoms.append(a)
            found.append(entry)
            taken.update(entry.atoms)
        return found, taken

    start = 0
    if cyclic:
        # No terminus to start from: take the rotation whose sequence reads smallest, so that the
        # numbering does not depend on the order of the atoms in the file.
        probe, _ = assign(0)
        names = [e.name for e in probe]
        start = min(range(len(names)), key=lambda s: (names[s:] + names[:s], s))
    found, taken = assign(start)
    n_res = len(found)

    # Terminal atoms, N-methyls, and the side chains of units that match nothing.
    for k, e in enumerate(found):
        for nb in mol.GetAtomWithIdx(e.c).GetNeighbors():
            i = nb.GetIdx()
            if i in taken or nb.GetAtomicNum() != 8:
                continue
            e.atoms.append(i)                          # the second oxygen of a C-terminal acid (or ester)
            e.names[i] = "OXT"
            taken.add(i)
        extra = [nb for nb in mol.GetAtomWithIdx(e.n).GetNeighbors() if nb.GetIdx() not in taken]
        if len(extra) == 1 and extra[0].GetAtomicNum() == 6 and extra[0].GetDegree() == 1 and (k > 0 or cyclic):
            e.atoms.append(extra[0].GetIdx())
            e.names[extra[0].GetIdx()] = "CN"
            e.n_methyl = True
            taken.add(extra[0].GetIdx())
        elif extra and (k > 0 or cyclic):
            e.n_substituted = True                     # a side chain on the nitrogen: a peptoid unit
    for e in found:
        if e.template is not None:
            continue
        frontier = [nb.GetIdx() for nb in mol.GetAtomWithIdx(e.ca).GetNeighbors() if nb.GetIdx() not in taken]
        while frontier:
            i = frontier.pop()
            if i in taken:
                continue
            taken.add(i)
            e.atoms.append(i)
            frontier.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() not in taken)

    # Stereo, read where it can be trusted.
    reference: list[str | None] | None = None
    if stereo_from is not None:
        ref = perceive_peptide(stereo_from)
        if ref is not None and len(ref) == n_res and [r.name for r in ref.residues] == [
                "XAA" if e.n_substituted else e.name for e in found]:
            reference = [r.stereo for r in ref.residues]
    if reference is None:
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)

    owner: dict[int, int] = {}
    residues: list[PeptideResidue] = []
    for k, e in enumerate(found):
        recognised = e.template is not None and not e.n_substituted
        stereo = reference[k] if reference is not None else _stereo(mol, e.n, e.ca, e.c)
        residues.append(PeptideResidue(
            position=k + 1, name=e.name if recognised else "XAA", atoms=tuple(sorted(e.atoms)),
            atom_names=tuple(sorted(e.names.items())), n=e.n, ca=e.ca, c=e.c, o=e.o,
            proteinogenic=bool(recognised and e.template is not None and e.template.proteinogenic),
            stereo=stereo, n_methyl=e.n_methyl))
        for a in e.atoms:
            owner[a] = k + 1

    # Appendages: what is left, in connected pieces, with the bonds that hold each.
    rest = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in owner]
    appendages: list[Appendage] = []
    seen: set[int] = set()
    for a0 in rest:
        if a0 in seen:
            continue
        piece, frontier = [], [a0]
        attach: list[tuple[int, int, int]] = []
        while frontier:
            i = frontier.pop()
            if i in seen:
                continue
            seen.add(i)
            piece.append(i)
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                j = nb.GetIdx()
                if j in owner:
                    attach.append((owner[j], j, i))
                elif j not in seen:
                    frontier.append(j)
        appendages.append(Appendage(tuple(sorted(piece)), tuple(sorted(attach))))

    # Crosslinks: bonds between two residues other than the peptide bonds along the chain.
    crosslinks: list[Crosslink] = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        pi, pj = owner.get(i), owner.get(j)
        if pi is None or pj is None or pi == pj:
            continue
        ri, rj = residues[pi - 1], residues[pj - 1]
        peptide_bond = (i == ri.c and j == rj.n) or (j == rj.c and i == ri.n) or (i == ri.n and j == rj.c) or (
            j == rj.n and i == ri.c)
        if peptide_bond and (abs(pi - pj) == 1 or (cyclic and {pi, pj} == {1, n_res})):
            continue
        both_s = mol.GetAtomWithIdx(i).GetAtomicNum() == 16 and mol.GetAtomWithIdx(j).GetAtomicNum() == 16
        lo, hi = sorted(((pi, i), (pj, j)))
        crosslinks.append(Crosslink(lo, hi, "disulfide" if both_s else "bond"))

    return Peptide(residues=tuple(residues), appendages=tuple(appendages),
                   crosslinks=tuple(sorted(crosslinks, key=lambda x: (x.a, x.b))), cyclic=cyclic, runs=lengths,
                   n_atoms=mol.GetNumAtoms())


def decide(peptide: Peptide | None) -> Decision:
    """Is this ligand drawn as a chain of residues? By what it is, not by how big it is.

    It is, on two conditions:

    * a run of at least :data:`MIN_RESIDUES` backbone units. The atom-level
      drawing is known to work up to about 55 heavy atoms, which is seven
      residues; below eight a peptide loses nothing by being drawn as atoms,
      and a peptidomimetic of four units drawn as beads would lose the atoms
      that matter;
    * at least :data:`MIN_RECOGNISED` of them with a proteinogenic side chain.
      A row of ``Xaa`` says nothing.
    """
    if peptide is None:
        return Decision(False, "it has no chain of amino-acid residues")
    n, known = len(peptide), peptide.recognised
    if n < MIN_RESIDUES:
        return Decision(False, f"its longest chain of amino-acid residues has {n}, fewer than {MIN_RESIDUES}", n, known)
    if known < MIN_RECOGNISED * n:
        return Decision(False, f"only {known} of the {n} residues in its chain carry a proteinogenic side chain", n,
                        known)
    return Decision(True, f"it has a chain of {n} amino-acid residues, {known} of them proteinogenic", n, known)


def decide_given(peptide: Peptide | None) -> Decision:
    """The same question for a chain that the structure file gives as a chain of its own.

    Only the first condition of :func:`decide` is asked. The second guards
    against a row of ``Xaa``, and a residue given by a file has a name.
    """
    if peptide is None:
        return Decision(False, "its chain has no residue with a backbone")
    n, known = len(peptide), peptide.recognised
    if all(r.kind == "nucleotide" for r in peptide.residues):
        if n < MIN_NUCLEOTIDES:
            return Decision(False, f"its chain has {n} nucleotides in the structure file, fewer than "
                                   f"{MIN_NUCLEOTIDES}", n, known)
        return Decision(True, f"it is a chain of {n} nucleotides in the structure file", n, known)
    if n < MIN_RESIDUES:
        return Decision(False, f"its chain has {n} residues in the structure file, fewer than {MIN_RESIDUES}", n,
                        known)
    return Decision(True, f"it is a chain of {n} residues in the structure file, {known} of them proteinogenic", n,
                    known)


def describe(peptide: Peptide, names: Iterable[str] | None = None) -> str:
    """A few lines of plain text: sequence, appendages, crosslinks. For logs and for reading a result."""
    label = list(names) if names is not None else None
    lines = [f"{len(peptide)} residues{' (cyclic)' if peptide.cyclic else ''}: "
             + "-".join(r.code3 for r in peptide.residues)]
    for app in peptide.appendages:
        where = ", ".join(f"{peptide.residue(p).code3}{peptide.residue(p).shown} "
                          f"{peptide.residue(p).name_of(a) or (label[a] if label else a)}"
                          for p, a, _ in app.attachments)
        lines.append(f"appendage of {len(app.atoms)} atoms on {where}")
    for x in peptide.crosslinks:
        lines.append(f"{x.kind} between {peptide.residue(x.a[0]).code3}{peptide.residue(x.a[0]).shown} and "
                     f"{peptide.residue(x.b[0]).code3}{peptide.residue(x.b[0]).shown}")
    return "\n".join(lines)
