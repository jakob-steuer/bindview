"""Reading atoms out of structure files.

Only what a 2D diagram needs is read: atom names, elements, residue identity,
coordinates and the B-factor column. mmCIF is parsed with a small tokenizer of
our own so the core has no dependency beyond RDKit and NumPy; legacy PDB files
are supported too.

The B-factor column is kept as the number it is. Whether it may be read as a
*confidence* (pLDDT) is a separate question that the file answers itself: a
predicted model declares a local pLDDT metric in ``_ma_qa_metric``, a crystal
structure does not, and in a crystal structure the same column is a
temperature factor where high means worse. :attr:`Structure.plddt_scale` records
the declaration; nothing in this package guesses from the value range.

The file is opened read-only and never written to.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from .chem import is_metal
from .errors import LigandNotFoundError, StructureError
from .exposure import WATER_NAMES
from .near import pairs_within

__all__ = ["Atom", "ResidueKey", "ChainResidue", "Link", "Structure", "read_structure", "read_models", "parse_mmcif",
           "parse_pdb", "MAX_LIGAND_RESIDUES"]

#: A chain with more residues than this is not drawn as a ligand, whole: no page holds it at a
#: legible size (54 boxes is what the default page takes). It is then one side of an interface,
#: of which :mod:`bindview.interface` draws the residues that touch the other side.
MAX_LIGAND_RESIDUES = 54
#: Two heavy atoms closer than this (ångström) are not two atoms.
_TOO_CLOSE = 1.05


@dataclass(frozen=True, order=True)
class ResidueKey:
    """Identity of one residue: chain, residue number and insertion code."""

    chain: str
    number: int
    icode: str = ""

    def __str__(self) -> str:
        return f"{self.chain}:{self.number}{self.icode}"


@dataclass(frozen=True)
class ChainResidue:
    """One residue of a polymer chain that is taken as the ligand, as the file has it.

    When a whole chain is the ligand its atoms become one lump, like the single
    component a co-folding model writes for a peptide, and every atom gets a
    name that is unique in it: ``63/OE1`` is atom ``OE1`` of residue 63. This
    record keeps what the file said about the residue those atoms came from.
    """

    name: str                           # component name: "GLU", "TYS", "ACE"
    number: str                         # residue number as deposited, with its insertion code: "63", "63A"
    atoms: tuple[str, ...]              # the names its heavy atoms have in the ligand


@dataclass(frozen=True)
class Link:
    """A bond between two residues that the file declares: a ``_struct_conn`` row of a covalent kind, a
    ``LINK`` record."""

    a: ResidueKey
    a_atom: str
    b: ResidueKey
    b_atom: str


@dataclass(frozen=True)
class Atom:
    """One atom record."""

    index: int
    name: str
    element: str
    resname: str
    chain: str
    resnum: int
    icode: str
    x: float
    y: float
    z: float
    hetero: bool
    #: ``_atom_site.B_iso_or_equiv`` / the PDB B-factor column, as written.
    b_iso: float | None = None
    #: ``_atom_site.label_asym_id`` where the file has it beside the author's chain name (``chain``):
    #: the two differ in many deposited structures, and a caller may know a chain by either.
    label_chain: str = ""

    @property
    def residue(self) -> ResidueKey:
        return ResidueKey(self.chain, self.resnum, self.icode)

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Structure:
    """A parsed structure: a flat atom list plus lookup tables.

    ``chem_comp_smiles`` carries ``_chem_comp.pdbx_smiles`` when the file has it
    (AlphaFold 3 writes the input SMILES there), keyed by component id.

    ``plddt_scale`` is ``None`` unless the file *declares* that its B-factor
    column holds a per-atom pLDDT (a ``_ma_qa_metric`` row with mode ``local``
    and a pLDDT type); it is then the factor that brings the column to the
    usual 0-100 scale (1, or 100 for the ``in [0,1]`` variants).
    ``qa_metrics`` lists every declared metric as ``(mode, name, type)``.
    """

    atoms: list[Atom]
    source: str = ""
    chem_comp_smiles: dict[str, str] = field(default_factory=dict)
    plddt_scale: float | None = None
    qa_metrics: tuple[tuple[str, str, str], ...] = ()
    #: The bonds between residues that the file declares (:class:`Link`), metal coordination left out.
    links: tuple[Link, ...] = ()
    _by_residue: dict[ResidueKey, list[int]] = field(default_factory=dict, repr=False)
    _coords: np.ndarray | None = field(default=None, repr=False)
    #: The number of this model in its file, where it was read as one of the file's models
    #: (``read_structure(path, model=n)``, :func:`read_models`). ``None`` for a structure that was read
    #: without naming a model: the first model of its file, whatever number that has.
    model: int | None = None

    def __post_init__(self) -> None:
        if not self._by_residue:
            table: dict[ResidueKey, list[int]] = {}
            for a in self.atoms:
                table.setdefault(a.residue, []).append(a.index)
            self._by_residue = table

    # -- lookups -----------------------------------------------------------
    @property
    def coords(self) -> np.ndarray:
        """``(n_atoms, 3)`` array of coordinates."""
        if self._coords is None:
            self._coords = np.array([a.xyz for a in self.atoms], dtype=float).reshape(-1, 3)
        return self._coords

    def residues(self) -> list[ResidueKey]:
        """All residues, in file order."""
        return list(self._by_residue)

    def residue_atoms(self, key: ResidueKey) -> list[Atom]:
        """The atoms of a residue."""
        return [self.atoms[i] for i in self._by_residue.get(key, [])]

    def residue_name(self, key: ResidueKey) -> str:
        """The name of a residue, as the file gives it."""
        idx = self._by_residue.get(key)
        return self.atoms[idx[0]].resname if idx else ""

    def find_residue(self, chain: str | None, number: int, icode: str = "") -> ResidueKey | None:
        """Locate a residue by number; ``chain=None`` matches any chain if unique."""
        if chain:
            key = ResidueKey(chain, number, icode)
            return key if key in self._by_residue else None
        hits = [k for k in self._by_residue if k.number == number and k.icode == icode]
        return hits[0] if len(hits) == 1 else None

    @property
    def declares_plddt(self) -> bool:
        """Does the file itself say that its B-factor column is a per-atom pLDDT?"""
        return self.plddt_scale is not None

    # -- a polymer chain as the ligand ---------------------------------------
    def polymer_residues(self, chain: str) -> list[ResidueKey]:
        """The residues of ``chain`` that make up its polymer, in file order.

        Those with a backbone of their own (atoms ``N``, ``CA`` and ``C``) and
        whatever is covalently bound to one of them: a cap such as ``ACE``, a
        modified residue. Waters are never part of it, and neither is anything
        else that merely carries the chain's name -- a deposited structure
        files an ion or a solvent molecule under the chain it lies next to.
        """
        keys = [k for k in self._by_residue if k.chain == chain and self.residue_name(k).upper() not in WATER_NAMES]
        heavy = {k: [a for a in self.residue_atoms(k) if a.element not in ("H", "D")] for k in keys}
        inside = [k for k in keys if {"N", "CA", "C"} <= {a.name for a in heavy[k]}]
        if not inside:
            inside = self._nucleotides(chain)       # a chain of RNA or DNA has no such residue
        if not inside:
            return []
        rest = [k for k in keys if k not in inside and heavy[k]]
        grew = True
        while grew and rest:
            grew = False
            have = np.array([a.xyz for k in inside for a in heavy[k]], dtype=float)
            big = np.array([a.element in ("S", "P", "SE", "Se") for k in inside for a in heavy[k]])
            for k in list(rest):
                pts = np.array([a.xyz for a in heavy[k]], dtype=float)
                own = np.array([a.element in ("S", "P", "SE", "Se") for a in heavy[k]])
                d = np.linalg.norm(pts[:, None, :] - have[None, :, :], axis=2)
                limit = np.where(own[:, None] | big[None, :], 2.3, 1.9)
                if (d < limit).any():
                    inside.append(k)
                    rest.remove(k)
                    grew = True
        order = {k: i for i, k in enumerate(keys)}
        return sorted(inside, key=lambda k: order[k])

    def split(self, chain: str, parts: "Mapping[str, tuple[int, int]]") -> "Structure":
        """The structure with stretches of ``chain`` under chain names of their own.

        ``parts`` maps a new chain name to the first and the last residue number of a stretch: the
        domains of one chain, which an :class:`~bindview.interface.Interface` then takes as two
        participants (a participant is whole chains). Residues of the chain outside every stretch
        keep its name, and so does everything else. A name that a chain already has, a stretch that
        holds no residue, and two stretches that overlap are errors.
        """
        taken = {a.chain for a in self.atoms}
        spans = {name: (int(lo), int(hi)) for name, (lo, hi) in parts.items()}
        for name, (lo, hi) in spans.items():
            if name in taken:
                raise StructureError(f"a part cannot be called {name!r}: the structure has a chain of that name")
            if lo > hi:
                raise StructureError(f"part {name!r} runs from {lo} to {hi}: the first number comes first")
            if any(other != name and lo <= b and a <= hi for other, (a, b) in spans.items()):
                raise StructureError(f"part {name!r} ({lo} to {hi}) overlaps another part of chain {chain!r}")
            if not any(a.chain == chain and lo <= a.resnum <= hi for a in self.atoms):
                raise StructureError(f"chain {chain!r} has no residue from {lo} to {hi} (part {name!r})")

        def named(atom: Atom) -> str:
            if atom.chain != chain:
                return atom.chain
            return next((name for name, (lo, hi) in spans.items() if lo <= atom.resnum <= hi), chain)

        return replace(self, atoms=[replace(a, chain=named(a)) for a in self.atoms], _by_residue={}, _coords=None)

    def with_chain_as_ligand(self, chain: str
                             ) -> tuple["Structure", ResidueKey, tuple[ChainResidue, ...], list[str]]:
        """The structure with the polymer of ``chain`` turned into one ligand residue.

        Returns the new structure, the key of that residue, what the file said
        about the residues it was made of, and notes on anything that was not
        taken as it stood. Nothing else in the structure changes, so every
        other chain is environment as before.

        One thing is not taken as it stands: a heavy atom less than
        :data:`_TOO_CLOSE` ångström from another one. No two heavy atoms are
        that close; in a deposited structure it is a mislabelled hydrogen or a
        duplicate. Of such a pair the atom that is bonded to nothing else is
        left out of the structure altogether, and a note says so.
        """
        keys = self.polymer_residues(chain)
        if not keys:
            raise LigandNotFoundError(f"chain {chain!r} of {self.source or 'the structure'} has no polymer residues")
        if len(keys) > MAX_LIGAND_RESIDUES:
            raise LigandNotFoundError(
                f"chain {chain!r} has {len(keys)} residues. A chain of more than {MAX_LIGAND_RESIDUES} is not "
                f"drawn as a ligand: it is one side of an interface. Draw the interface between it and its "
                f"partner instead (bindview interface, or bindview.Interface), which shows the residues of each "
                f"side that touch the other")
        return self.with_residues_as_ligand(keys)

    def with_residues_as_ligand(self, keys: Sequence[ResidueKey]
                                ) -> tuple["Structure", ResidueKey, tuple[ChainResidue, ...], list[str]]:
        """The structure with these residues of one chain turned into one ligand residue.

        What :meth:`with_chain_as_ligand` does for a whole chain, for any
        residues of a chain: the interface residues of one participant, which
        are the *spine* of an interface figure (:mod:`bindview.interface`). The
        residues that are not named stay what they were.
        """
        keys = list(keys)
        if not keys or len({k.chain for k in keys}) != 1:
            raise LigandNotFoundError("the residues that are drawn as one chain have to be of one chain")
        chain = keys[0].chain
        notes: list[str] = []
        heavy = [a for k in keys for a in self.residue_atoms(k) if a.element not in ("H", "D")]
        xyz = np.array([a.xyz for a in heavy], dtype=float)
        # The pairs within bonding distance (each twice, and each atom with itself), never the array of all pairs.
        first, second, apart = pairs_within(xyz, xyz, 1.9)
        bonded = np.bincount(first[(apart < 1.9) & (apart > _TOO_CLOSE)], minlength=len(heavy))
        clash = (apart < _TOO_CLOSE) & (first < second)
        dropped: set[int] = set()
        for i, j, d in zip(first[clash].tolist(), second[clash].tolist(), apart[clash].tolist()):
            if heavy[i].index in dropped or heavy[j].index in dropped:
                continue                     # (an atom too close to two others goes once, with one note)
            # Of the two the one that is bonded to fewer other atoms goes, and of two that are bonded alike
            # the one that comes later in the file.
            out = i if bonded[i] < bonded[j] else j
            keep = j if out == i else i
            a, b = heavy[out], heavy[keep]
            dropped.add(a.index)
            notes.append(f"atom {a.name} of {a.resname} {a.resnum}{a.icode} lies {d:.2f} \u00c5 from "
                         f"{b.name} of {b.resname} {b.resnum}{b.icode} and was left out: no two heavy atoms "
                         f"are that close")
        merged = ResidueKey(chain, keys[0].number, keys[0].icode)
        wanted = set(keys)
        atoms: list[Atom] = []
        residues: dict[ResidueKey, list[str]] = {k: [] for k in keys}
        for a in self.atoms:
            if a.index in dropped:
                continue
            if a.residue in wanted:
                name = f"{a.resnum}{a.icode}/{a.name}"
                if a.element not in ("H", "D"):
                    residues[a.residue].append(name)
                a = replace(a, name=name, resname=f"CHAIN_{chain}", resnum=merged.number, icode=merged.icode,
                            hetero=True)
            atoms.append(replace(a, index=len(atoms)))
        given = tuple(ChainResidue(self.residue_name(k), f"{k.number}{k.icode}", tuple(residues[k]))
                      for k in keys if residues[k])
        return replace(self, atoms=atoms, _by_residue={}, _coords=None), merged, given, notes

    def resolve_ligand(self, spec: "str | ResidueKey | tuple[str, int | str]"
                       ) -> tuple["Structure", ResidueKey, tuple[ChainResidue, ...], list[str]]:
        """Resolve a ligand selection that may also name a whole polymer chain.

        Everything :meth:`select_ligand` accepts, and in addition::

            "C"            chain C, when no residue is named ``C`` and the chain is a polymer
            "chain:C"      chain C, said outright

        A chain is looked up by the name its authors gave it
        (``auth_asym_id``, what every other part of this package and a PDB
        file call the chain). Only if there is no such chain, the name is tried
        as the mmCIF ``label_asym_id``, and a note says which chain that is.

        A chain that is the ligand comes back as one residue of a new
        structure, together with the residues the file gave it (see
        :meth:`with_chain_as_ligand`); for any other selection the structure is
        returned as it is and the third element is empty. Which chain is the
        ligand is never guessed: the caller names it.
        """
        if isinstance(spec, str):
            text = spec.strip()
            head, _, tail = text.partition(":")
            said = head.lower() == "chain" and bool(tail) and ":" not in tail
            name = tail if said else text
            if said or (":" not in text and not any(self.residue_name(k) == text for k in self._by_residue)):
                if said or len(self._backbone(name)) >= 2:
                    if self._backbone(name) or not self._by_label(name):
                        return self.with_chain_as_ligand(name)
                by_label = self._by_label(name)
                if len(by_label) == 1 and len(self._backbone(by_label[0])) >= (1 if said else 2):
                    st, key, given, notes = self.with_chain_as_ligand(by_label[0])
                    return st, key, given, [f"chain {name!r} is the file's label_asym_id; its authors call the "
                                            f"chain {by_label[0]!r}, and so does the figure"] + notes
        return self, self.select_ligand(spec), (), []

    def _backbone(self, chain: str) -> list[ResidueKey]:
        return [k for k in self._by_residue if k.chain == chain
                and {"N", "CA", "C"} <= {a.name for a in self.residue_atoms(k)}] or self._nucleotides(chain)

    def _nucleotides(self, chain: str) -> list[ResidueKey]:
        """The nucleotides of ``chain`` that are joined to one another, in file order: residues with a
        sugar (``C1'``, ``C3'``, ``C4'``, ``O4'``) whose ``O3'`` is bound to the phosphorus of another.
        By their atoms and not by their names, so a modified nucleotide is one. Two nucleotides that
        merely lie in one chain, two copies of a cofactor, are no chain."""
        found: dict[ResidueKey, dict[str, Atom]] = {}
        for k in self._by_residue:
            if k.chain != chain or self.residue_name(k).upper() in WATER_NAMES:
                continue
            atoms = {a.name.replace("*", "'"): a for a in self.residue_atoms(k)}
            if {"C1'", "C3'", "C4'", "O4'"} <= set(atoms):
                found[k] = atoms
        ends = [(k, atoms["O3'"]) for k, atoms in found.items() if "O3'" in atoms]
        heads = [(k, atoms["P"]) for k, atoms in found.items() if "P" in atoms]
        joined: set[ResidueKey] = set()
        for k, o3 in ends:
            for other, p in heads:
                if other != k and (o3.x - p.x) ** 2 + (o3.y - p.y) ** 2 + (o3.z - p.z) ** 2 < 2.0 ** 2:
                    joined.update((k, other))
        return [k for k in found if k in joined]

    def _by_label(self, label: str) -> list[str]:
        """The author chain names of the polymer atoms whose ``label_asym_id`` is ``label``."""
        return sorted({a.chain for a in self.atoms if a.label_chain == label and a.resname.upper() not in WATER_NAMES
                       and a.name in ("N", "CA", "C", "C4'", "C4*")})

    def atom_by_name(self, key: ResidueKey, name: str) -> Atom | None:
        """An atom of a residue by its name, or None."""
        for i in self._by_residue.get(key, []):
            if self.atoms[i].name == name:
                return self.atoms[i]
        return None

    # -- ligand selection --------------------------------------------------
    def select_ligand(self, spec: "str | ResidueKey | tuple[str, int | str]") -> ResidueKey:
        """Resolve a ligand selection to exactly one residue.

        Accepted forms::

            "LIG_B"        residue (component) name
            "B:1"          chain and residue number
            "B:LIG_B:1"    chain, residue name and number
            ("B", 1)       chain and residue number
            ("B", "1A")    chain and residue number with its insertion code
            ResidueKey     used as is

        A bare name that matches several residues is an error rather than a
        silent pick of the first: the caller has to say which copy is meant.
        """
        if isinstance(spec, ResidueKey):
            if spec not in self._by_residue:
                raise LigandNotFoundError(f"no residue {spec} in {self.source or 'structure'}")
            return spec
        if isinstance(spec, tuple):
            try:
                chain, given = spec
                m = re.fullmatch(r"(-?\d+)([A-Za-z]?)", given.strip() if isinstance(given, str) else str(int(given)))
                if m is None:
                    raise ValueError(given)
            except (TypeError, ValueError):
                raise LigandNotFoundError(
                    f"cannot parse ligand selection {spec!r}; a pair is (chain, number), the number with its "
                    f"insertion code where it has one: ('A', 698) or ('A', '698A')") from None
            number, icode = int(m.group(1)), m.group(2)
            key = self.find_residue(str(chain), number, icode)
            if key is None:
                raise LigandNotFoundError(f"no residue {chain}:{number}{icode} in {self.source or 'structure'}")
            return key
        text = str(spec).strip()
        if not text:
            raise LigandNotFoundError("empty ligand selection")
        parts = text.split(":")
        if len(parts) == 1:
            hits = [k for k in self._by_residue if self.residue_name(k) == text]
            if not hits:
                # Convenience: a bare chain id whose chain holds a single hetero residue.
                chain_hits = [k for k in self._by_residue if k.chain == text]
                het = [k for k in chain_hits if all(a.hetero for a in self.residue_atoms(k))]
                if len(het) == 1:
                    return het[0]
                names = sorted({self.residue_name(k) for k in self._by_residue
                                if any(a.hetero for a in self.residue_atoms(k))})
                raise LigandNotFoundError(
                    f"no residue named {text!r}; hetero residues present: {', '.join(names) or 'none'}")
            if len(hits) > 1:
                listed = ", ".join(str(k) for k in hits[:8])
                raise LigandNotFoundError(
                    f"{len(hits)} residues are named {text!r} ({listed}); select one as 'chain:number'")
            return hits[0]
        try:
            if len(parts) == 2:
                chain, numtxt, name = parts[0], parts[1], None
            elif len(parts) == 3:
                chain, name, numtxt = parts
            else:
                raise ValueError(text)
            m = re.fullmatch(r"(-?\d+)([A-Za-z]?)", numtxt)
            if not m:
                raise ValueError(text)
            number, icode = int(m.group(1)), m.group(2)
        except ValueError:
            raise LigandNotFoundError(
                f"cannot parse ligand selection {text!r}; use 'NAME', 'chain:number' or 'chain:NAME:number'"
            ) from None
        key = self.find_residue(chain, number, icode)
        if key is None or (name is not None and self.residue_name(key) != name):
            raise LigandNotFoundError(f"no residue matching {text!r} in {self.source or 'structure'}")
        return key


# ---------------------------------------------------------------------------
# mmCIF
# ---------------------------------------------------------------------------

_TOKEN = re.compile(
    r"""
    '((?:[^']|'(?=\S))*)'(?=\s|$)      # single-quoted (a quote only closes before whitespace)
  | "((?:[^"]|"(?=\S))*)"(?=\s|$)      # double-quoted
  | (\#.*)                             # comment to end of line
  | (\S+)                              # bare word
    """,
    re.VERBOSE,
)


def _tokens(lines: Sequence[str]) -> Iterator[tuple[str, bool]]:
    """Yield ``(token, quoted)`` pairs from mmCIF text.

    ``quoted`` distinguishes the literal string ``"loop_"`` from the keyword.
    """
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.startswith(";"):
            # Multi-line text field: runs until a line that starts with ';'.
            buf = [line[1:].rstrip("\n")]
            i += 1
            while i < n and not lines[i].startswith(";"):
                buf.append(lines[i].rstrip("\n"))
                i += 1
            yield "\n".join(buf).strip(), True
            if i >= n:
                break
            line = lines[i][1:]              # what follows the closing ';' on its line are values like any other
        for m in _TOKEN.finditer(line):
            if m.group(3) is not None:
                break
            if m.group(1) is not None:
                yield m.group(1), True
            elif m.group(2) is not None:
                yield m.group(2), True
            else:
                yield m.group(4), False
        i += 1


def _read_categories(text: str, wanted: set[str]) -> dict[str, dict[str, list[str]]]:
    """Read the requested categories of the first data block as column tables."""
    out: dict[str, dict[str, list[str]]] = {}
    toks = _tokens(text.splitlines())
    pending: tuple[str, bool] | None = None
    seen_block = False

    def nxt() -> tuple[str, bool] | None:
        nonlocal pending
        if pending is not None:
            t, pending = pending, None
            return t
        return next(toks, None)

    while True:
        tok = nxt()
        if tok is None:
            break
        word, quoted = tok
        if quoted:
            continue
        low = word.lower()
        if low.startswith("data_"):
            if seen_block:
                break                    # only the first data block
            seen_block = True
            continue
        if low == "loop_":
            names: list[str] = []
            while True:
                t = nxt()
                if t is None:
                    break
                if not t[1] and t[0].startswith("_"):
                    names.append(t[0])
                else:
                    pending = t
                    break
            if not names:
                continue
            cat = names[0].split(".", 1)[0].lower()
            keep = cat in wanted
            cols: list[list[str]] = [[] for _ in names]
            k = 0
            while True:
                t = nxt()
                if t is None:
                    break
                w, q = t
                if not q and (w.startswith("_") or w.lower() == "loop_" or w.lower().startswith("data_")):
                    pending = t
                    break
                if keep:
                    cols[k].append(w)
                k = (k + 1) % len(names)
            if keep:
                table = out.setdefault(cat, {})
                for nm, col in zip(names, cols):
                    table[nm.split(".", 1)[1].lower() if "." in nm else nm.lower()] = col
            continue
        if word.startswith("_"):
            val = nxt()
            cat, _, item = word.partition(".")
            if cat.lower() in wanted and val is not None:
                out.setdefault(cat.lower(), {})[item.lower()] = [val[0]]
    return out


def _int(text: str, default: int = 0) -> int:
    try:
        return int(text)
    except ValueError:
        return default


def _norm_element(symbol: str) -> str:
    s = symbol.strip()
    return s[:1].upper() + s[1:].lower() if s else ""


def _of_another_type(records: "Sequence[tuple[tuple[str, int, str], str, str]]") -> set[int]:
    """The atom records that are not read because their position holds residues of two types.

    ``records`` are the atom records of one model in the order of the file, each as its position (chain,
    number, insertion code), its residue name and its alternate-location id (empty where it has none). A
    file writes two residue types at one position as alternate locations of each other: serine as ``A``,
    threonine as ``B``. Such a position is read as one residue, of the type of its first record and of that
    type's first alternate location; a record of that type without an alternate-location id belongs to it
    too. What is returned are the places in ``records`` of the others. A position with one residue type has
    none among them, however many conformations it has. Nor has one where a second residue name stands
    without an alternate-location id: that is no alternate location, but a number given twice (a ligand
    that a docking program numbers like a residue of the protein), and nothing of it is left out.
    """
    types: dict[tuple[str, int, str], str] = {}
    mixed: set[tuple[str, int, str]] = set()
    for place, resname, alt in records:
        if types.setdefault(place, resname) != resname and alt:
            mixed.add(place)
    out: set[int] = set()
    if not mixed:
        return out
    location: dict[tuple[str, int, str], str] = {}
    for k, (place, resname, alt) in enumerate(records):
        if place in mixed and alt and (resname != types[place] or location.setdefault(place, alt) != alt):
            out.add(k)
    return out


def parse_mmcif(text: str, source: str = "", model: "int | None" = None) -> Structure:
    """Parse mmCIF text into a :class:`Structure`.

    Of the atoms that have one name in one residue, as the alternate locations of an atom have, the
    first in the file is read. Where one position (chain, number, insertion code) holds residues of two
    types, as alternate locations of each other, the residue is the one that comes first in the file, in
    its first alternate location: the atoms of the other type are not read.

    ``model``: the model to read, by the number the file gives it
    (``_atom_site.pdbx_PDB_model_num``). ``None`` is the first model of the file.

    A residue is numbered by ``auth_seq_id`` and, where the file has none, by ``label_seq_id``. A ligand
    or a water has no ``label_seq_id`` (it is a dot): in a file without the author's numbers such a
    residue is numbered after the last numbered residue of its chain, in the order of the file. The rows
    of one such residue are those that follow one another with one chain and component name, until an
    atom name comes a second time.
    """
    return _mmcif_models(text, source, _model_asked(model, source or "mmCIF"))[0]


def _mmcif_models(text: str, source: str, model: "int | None | str") -> list[Structure]:
    """The models of mmCIF text: the first (``None``), the one numbered ``model``, or all of them
    (``"all"``), each a structure of its own. The text is read once."""
    cats = _read_categories(text, {"_atom_site", "_chem_comp", "_ma_qa_metric", "_struct_conn"})
    site = cats.get("_atom_site")
    if not site or "cartn_x" not in site:
        raise StructureError(f"{source or 'mmCIF'}: no _atom_site coordinates found")
    lacking = [written for written in ("Cartn_y", "Cartn_z") if written.lower() not in site]
    if lacking:
        raise StructureError(f"{source or 'mmCIF'}: _atom_site has no {', '.join(lacking)}")
    lengths = [len(column) for column in site.values()]
    if min(lengths) != max(lengths):
        # The values of a loop are dealt to its columns in turn: columns of unequal length are a last row
        # that is not whole. Read on, its atom would have no place, and the columns after the cut none at all.
        raise StructureError(f"{source or 'mmCIF'}: the _atom_site loop ends inside a row: its last row has "
                             f"{lengths.count(max(lengths))} of {len(lengths)} values (a file that was cut short?)")
    n = len(site["cartn_x"])

    def col(*names: str) -> list[str] | None:
        for nm in names:
            c = site.get(nm)
            if c is not None and len(c) == n:
                return c
        return None

    xs, ys, zs = site["cartn_x"], site["cartn_y"], site["cartn_z"]
    group = col("group_pdb")
    element = col("type_symbol")
    name = col("auth_atom_id", "label_atom_id")
    comp = col("auth_comp_id", "label_comp_id")
    chain = col("auth_asym_id", "label_asym_id")
    label_chain = col("label_asym_id")
    seq = col("auth_seq_id", "label_seq_id")
    seq_alt = col("label_seq_id")
    icode = col("pdbx_pdb_ins_code")
    alt = col("label_alt_id")
    numbers = col("pdbx_pdb_model_num")
    b_iso = col("b_iso_or_equiv")
    if name is None or comp is None or chain is None or seq is None:
        raise StructureError(f"{source or 'mmCIF'}: _atom_site lacks atom/residue identifiers")

    smiles: dict[str, str] = {}
    cc = cats.get("_chem_comp")
    if cc and "id" in cc and "pdbx_smiles" in cc:
        for cid, smi in zip(cc["id"], cc["pdbx_smiles"]):
            if smi not in ("?", "."):
                smiles[cid] = smi
    metrics, scale = _qa_metrics(cats.get("_ma_qa_metric"))
    links = _mmcif_links(cats.get("_struct_conn"))

    def number_of(i: int) -> "int | None":
        """The number the file gives the residue of row ``i``: the author's, else the label's, else none."""
        for column in (seq, seq_alt):
            if column is not None:
                try:
                    return int(column[i])
                except ValueError:
                    pass
        return None

    def build(rows: Sequence[int], number: "int | None") -> Structure:
        atoms: list[Atom] = []
        seen: set[tuple[str, int, str, str]] = set()
        # A residue without a number (see parse_mmcif): the highest number of each chain, counted up for
        # each such residue, and the one the rows are in, with the atoms it has so far and its number.
        highest: "dict[str, int] | None" = None
        within: "tuple[tuple[str, str, str], set[tuple[str, str]], int] | None" = None
        places: list[tuple[tuple[str, int, str], str, str]] = []
        for i in rows:
            given = number_of(i)
            if given is None:
                if highest is None:
                    highest = {}
                    for j in rows:
                        other = number_of(j)
                        if other is not None and other > highest.get(chain[j], 0):
                            highest[chain[j]] = other
                part = (chain[i], label_chain[i] if label_chain is not None else "", comp[i])
                atom = (name[i], alt[i] if alt is not None else "")
                if within is None or within[0] != part or atom in within[1]:
                    highest[chain[i]] = highest.get(chain[i], 0) + 1
                    within = (part, set(), highest[chain[i]])
                within[1].add(atom)
                num = within[2]
            else:
                num = given
            ic = icode[i] if icode is not None and icode[i] not in (".", "?") else ""
            location = alt[i] if alt is not None and alt[i] not in (".", "?") else ""
            places.append(((chain[i], num, ic), comp[i], location))
        left_out = _of_another_type(places)
        for k, i in enumerate(rows):
            if k in left_out:
                continue                     # of another residue type than the one its position is read as
            (_, num, ic), _, _ = places[k]
            ident = (chain[i], num, ic, name[i])
            if ident in seen:
                continue                     # a second atom of that name in the residue: the first one is kept
            seen.add(ident)
            try:
                x, y, z = float(xs[i]), float(ys[i]), float(zs[i])
            except ValueError as exc:
                raise StructureError(f"{source or 'mmCIF'}: bad coordinate in _atom_site row {i + 1}") from exc
            el = _norm_element(element[i]) if element is not None else _guess_element(name[i])
            atoms.append(Atom(
                index=len(atoms), name=name[i], element=el, resname=comp[i], chain=chain[i],
                resnum=num, icode=ic, x=x, y=y, z=z,
                hetero=(group[i].upper() == "HETATM") if group is not None else False,
                b_iso=_float_or_none(b_iso[i]) if b_iso is not None else None,
                label_chain=label_chain[i] if label_chain is not None else "",
            ))
        if not atoms:
            raise StructureError(f"{source or 'mmCIF'}: no atoms"
                                 + (f" in model {number}" if number is not None else ""))
        return Structure(atoms=atoms, source=source, chem_comp_smiles=dict(smiles), plddt_scale=scale,
                         qa_metrics=metrics, links=links, model=number)

    if model is None:
        # The first model: the rows down to the first that belongs to another one.
        first = numbers[0] if numbers else None
        stop = next((i for i in range(n) if numbers is not None and numbers[i] != first), n)
        return [build(range(stop), None)]
    by_model: dict[str, list[int]] = {}
    for i in range(n):
        by_model.setdefault(numbers[i] if numbers is not None else "1", []).append(i)
    numbered = _model_numbers(list(by_model))
    if model == "all":
        return [build(rows, number) for rows, number in zip(by_model.values(), numbered)]
    if model not in numbered:
        raise StructureError(_no_such_model(source or "mmCIF", model, numbered))
    return [build(list(by_model.values())[numbered.index(model)], int(model))]


def _model_asked(model: "int | float | str | None", source: str) -> "int | None":
    """The model that is asked for, as the number it is: also where it comes as text (``"2"``) or as a
    whole number of another kind (``2.0``)."""
    if model is None:
        return None
    try:
        if isinstance(model, bool) or (isinstance(model, str) and not re.fullmatch(r"\s*[+-]?\d+\s*", model)):
            raise ValueError(model)
        number = int(model)
        if not isinstance(model, str) and number != model:
            raise ValueError(model)
    except (TypeError, ValueError):
        raise StructureError(f"{source}: model is the number of one model of the file, not {model!r} "
                             f"(read_models gives every model of a file)") from None
    return number


def _model_numbers(written: "Sequence[str | None]") -> list[int]:
    """The numbers of a file's models: those it writes, where they are numbers and no two are alike;
    otherwise their order, from 1 (some programs write every frame as model 1, or with no number)."""
    try:
        numbers = [int(str(w)) for w in written]
    except (TypeError, ValueError):
        numbers = []
    if len(numbers) != len(written) or len(set(numbers)) != len(numbers):
        return list(range(1, len(written) + 1))
    return numbers


def _no_such_model(source: str, model: object, numbered: "Sequence[int]") -> str:
    if len(numbered) == 1:
        return f"{source}: there is no model {model}: the file has one model, number {numbered[0]}"
    listed = (", ".join(str(k) for k in numbered) if len(numbered) <= 12
              else f"{numbered[0]} to {numbered[-1]}" if numbered == list(range(numbered[0], numbered[-1] + 1))
              else ", ".join(str(k) for k in numbered[:12]) + ", ...")
    return f"{source}: there is no model {model}: the file has {len(numbered)} models, numbered {listed}"


def _mmcif_links(table: "dict[str, list[str]] | None") -> tuple[Link, ...]:
    """The covalent bonds of ``_struct_conn`` within one copy of the molecules."""
    if not table or "conn_type_id" not in table:
        return ()
    n = len(table["conn_type_id"])

    def column(*names: str) -> "list[str] | None":
        return next((table[c] for c in names if len(table.get(c, ())) == n), None)

    ends = []
    for k in "12":
        ends.append((column(f"ptnr{k}_auth_asym_id", f"ptnr{k}_label_asym_id"),
                     column(f"ptnr{k}_auth_seq_id", f"ptnr{k}_label_seq_id"),
                     column(f"pdbx_ptnr{k}_pdb_ins_code"), column(f"ptnr{k}_label_atom_id"),
                     column(f"ptnr{k}_symmetry")))
    if any(e[0] is None or e[1] is None or e[3] is None for e in ends):
        return ()
    out = []
    for i, kind in enumerate(table["conn_type_id"]):
        if not kind.lower().startswith("covale"):
            continue
        if len({e[4][i] for e in ends if e[4] is not None}) > 1:
            continue                         # to another copy of the molecule
        keys = []
        for chain, number, icode, _, _ in ends:
            assert chain is not None and number is not None
            ic = icode[i] if icode is not None and icode[i] not in (".", "?") else ""
            keys.append(ResidueKey(chain[i], _int(number[i]), ic))
        out.append(Link(keys[0], ends[0][3][i], keys[1], ends[1][3][i]))  # type: ignore[index]
    return tuple(out)


def _pdb_links(lines: Iterable[str]) -> tuple[Link, ...]:
    """The ``LINK`` records of a PDB file between atoms of one copy of the molecules, metals left out."""
    out = []
    for line in lines:
        if line[:6] != "LINK  " or len(line) < 57:
            continue
        if line[59:65].strip() != line[66:72].strip():
            continue                         # to another copy of the molecule
        try:
            ends = [(ResidueKey(line[c].strip() or "A", int(line[c + 1:c + 5]), line[c + 5:c + 6].strip()),
                     line[n:n + 4]) for c, n in ((21, 12), (51, 42))]
        except (ValueError, IndexError):
            continue
        if any(is_metal(_guess_element(name)) for _, name in ends):
            continue
        out.append(Link(ends[0][0], ends[0][1].strip(), ends[1][0], ends[1][1].strip()))
    return tuple(out)


def _float_or_none(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _qa_metrics(table: dict[str, list[str]] | None) -> tuple[tuple[tuple[str, str, str], ...], float | None]:
    """The ``_ma_qa_metric`` declarations, and the pLDDT scale if a *local* pLDDT is among them.

    ModelCIF names the metric in ``type`` (``pLDDT``, ``pLDDT all-atom``, and
    the same ``in [0,1]``); writers also repeat it in ``name``. A global-only
    declaration is not enough: it says nothing about the per-atom column.
    """
    if not table:
        return (), None
    n = max((len(v) for v in table.values()), default=0)

    def column(key: str) -> list[str]:
        col = table.get(key)
        return col if col is not None and len(col) == n else [""] * n

    rows = tuple(zip(column("mode"), column("name"), column("type")))
    scale: float | None = None
    for mode, name, kind in rows:
        if mode.strip().lower() != "local":
            continue
        described = f"{kind} {name}".lower()
        if "plddt" in described:
            scale = 100.0 if "[0,1]" in described.replace(" ", "") else 1.0
            break
    return rows, scale


# ---------------------------------------------------------------------------
# PDB
# ---------------------------------------------------------------------------

_TWO_LETTER = {"CL", "BR", "FE", "ZN", "MG", "MN", "CA", "NA", "CU", "CO", "NI", "SE", "HG", "CD", "LI", "SI"}


def _guess_element(atom_name: str) -> str:
    """Element from an atom name when the element column is blank.

    Given the raw four-character PDB field, alignment decides: a two-letter
    element starts in the first column (``"CA  "`` is calcium, ``"FE  "`` iron)
    while a one-letter element is preceded by a space or digit (``" CA "`` is an
    alpha carbon). Without alignment only the first letter is trusted.

    A hydrogen whose name fills the four columns begins in the first one too: ``HG11`` to ``HG23`` are the
    hydrogens on the gamma carbons of valine, isoleucine and threonine, and no mercury. An ``H`` that two
    digits follow after its second letter is therefore a hydrogen.
    """
    if len(atom_name) == 4 and atom_name[:1].isalpha() and atom_name[:2].upper() in _TWO_LETTER \
            and not atom_name[2:].strip(" 0123456789'\"*") \
            and not (atom_name[:1].upper() == "H" and atom_name[2:].isdigit()):
        return atom_name[:2].capitalize()
    letters = re.sub(r"[^A-Za-z]", "", atom_name)
    return letters[0].upper() if letters else ""


def parse_pdb(text: str, source: str = "", model: "int | None" = None) -> Structure:
    """Parse legacy PDB text into a :class:`Structure`.

    Of the atoms that have one name in one residue, as the alternate locations of an atom have, the
    first in the file is read. Where one position (chain, number, insertion code) holds residues of two
    types, as alternate locations of each other, the residue is the one that comes first in the file, in
    its first alternate location: the atoms of the other type are not read.

    ``model``: the model to read, by the number of its ``MODEL`` record. ``None`` is the first model
    of the file.
    """
    return _pdb_models(text, source, _model_asked(model, source or "PDB"))[0]


def _pdb_models(text: str, source: str, model: "int | None | str") -> list[Structure]:
    """The models of PDB text: the first (``None``), the one numbered ``model``, or all (``"all"``)."""
    links = _pdb_links(text.splitlines())

    def build(lines: Sequence[str], number: "int | None") -> Structure:
        atoms: list[Atom] = []
        seen: set[tuple[str, int, str, str]] = set()
        places: list[tuple[tuple[str, int, str], str, str]] = []
        placed: list[tuple[str, float, float, float]] = []
        for line in lines:
            try:
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21].strip() or "A"
                num = int(line[22:26])
                ic = line[26].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except (ValueError, IndexError) as exc:
                raise StructureError(f"{source or 'PDB'}: malformed atom record: {line[:30]!r}") from exc
            places.append(((chain, num, ic), resname, line[16].strip()))
            placed.append((name, x, y, z))
        left_out = _of_another_type(places)
        for k, line in enumerate(lines):
            if k in left_out:
                continue                     # of another residue type than the one its position is read as
            rec = line[:6]
            (chain, num, ic), resname, _ = places[k]
            name, x, y, z = placed[k]
            b_iso = _float_or_none(line[60:66]) if len(line) >= 66 else None
            ident = (chain, num, ic, name)
            if ident in seen:
                continue
            seen.add(ident)
            el = _norm_element(line[76:78]) if len(line) >= 78 and line[76:78].strip() else _guess_element(line[12:16])
            atoms.append(Atom(index=len(atoms), name=name, element=el, resname=resname, chain=chain,
                              resnum=num, icode=ic, x=x, y=y, z=z, hetero=(rec == "HETATM"), b_iso=b_iso))
        if not atoms:
            raise StructureError(f"{source or 'PDB'}: no ATOM/HETATM records"
                                 + (f" in model {number}" if number is not None else ""))
        return Structure(atoms=atoms, source=source, links=links, model=number)

    if model is None:
        # The first model: every atom record down to the first ENDMDL.
        first: list[str] = []
        for line in text.splitlines():
            if line[:6].startswith("ENDMDL"):
                break
            if line[:6] in ("ATOM  ", "HETATM"):
                first.append(line)
        return [build(first, None)]
    # The atom records of every model, with the number its MODEL record writes (none without one).
    blocks: list[tuple[str | None, list[str]]] = []
    written: str | None = None
    current: list[str] = []
    opened = False
    for line in text.splitlines():
        rec = line[:6]
        if rec == "MODEL ":
            if current:                              # atoms before a MODEL record: a model of their own
                blocks.append((written, current))
            fields = line[6:].split()
            written, current, opened = (fields[0] if fields else None), [], True
        elif rec.startswith("ENDMDL"):
            blocks.append((written, current))
            written, current, opened = None, [], False
        elif rec in ("ATOM  ", "HETATM"):
            current.append(line)
    if current or opened or not blocks:
        blocks.append((written, current))
    numbered = _model_numbers([w for w, _ in blocks]) if len(blocks) > 1 or blocks[0][0] is not None else [1]
    if model == "all":
        return [build(lines, number) for (_, lines), number in zip(blocks, numbered)]
    if model not in numbered:
        raise StructureError(_no_such_model(source or "PDB", model, numbered))
    return [build(blocks[numbered.index(model)][1], int(model))]


def _read(path: "str | Path", model: "int | None | str") -> list[Structure]:
    p = Path(path)
    try:
        if p.suffix.lower() == ".gz":
            with gzip.open(p, "rt", encoding="utf-8") as handle:
                text = handle.read()
        else:
            text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise StructureError(f"cannot read {p}: it is not UTF-8 text (it has the byte 0x{exc.object[exc.start]:02x}); "
                             f"save it as UTF-8") from exc
    except (OSError, EOFError) as exc:
        raise StructureError(f"cannot read {p}: {exc}") from exc
    suffixes = [x.lower() for x in p.suffixes]
    suffix = suffixes[-2] if suffixes and suffixes[-1] == ".gz" and len(suffixes) > 1 else p.suffix.lower()
    if suffix in (".pdb", ".ent"):
        return _pdb_models(text, str(p), model)
    if suffix in (".cif", ".mmcif"):
        return _mmcif_models(text, str(p), model)
    # Unknown extension: sniff.
    if re.search(r"^data_", text, re.MULTILINE) and "_atom_site." in text:
        return _mmcif_models(text, str(p), model)
    return _pdb_models(text, str(p), model)


def read_structure(path: "str | Path", model: "int | None" = None) -> Structure:
    """Read an mmCIF (``.cif``/``.mmcif``) or PDB (``.pdb``/``.ent``) file, gzipped (``.gz``) or not.

    ``model``: which model of the file, by the number the file gives it
    (``_atom_site.pdbx_PDB_model_num``, the serial of ``MODEL``). ``None`` is the first model; a
    number the file does not have is refused. :func:`read_models` gives them all.
    """
    return _read(path, _model_asked(model, str(path)))[0]


def read_models(path: "str | Path") -> list[Structure]:
    """Every model of a structure file as a structure of its own, in the order of the file.

    The models of an NMR bundle, or frames that were written into one PDB file. Each structure
    knows its number in the file (:attr:`Structure.model`); a file that numbers no model is one
    model, numbered 1, and models that a PDB file does not tell apart by number are numbered by
    their order. This is the structure file :func:`read_structure` reads, all of it: no
    trajectory format is read.
    """
    return _read(path, "all")
