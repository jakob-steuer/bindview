"""Standard amino-acid and nucleotide knowledge: topology, residue classes, rings, charged groups.

The templates are written out by hand (atom names in SMILES order) rather than
taken from a component dictionary so that the core needs no data files and no
network. Only the 20 standard residues and the eight standard nucleotides of RNA
and DNA are covered; anything else is drawn as a bare labelled atom, which is
always correct if less informative.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .exposure import WATER_NAMES

__all__ = [
    "ResidueTemplate", "template_for", "residue_class", "BACKBONE",
    "aromatic_rings", "charged_groups", "can_donate", "can_accept", "canonical_resname", "format_residue_label",
    "PHOSPHATE", "TAIL", "SUGAR", "NUCLEOTIDE_BACKBONE", "is_nucleotide", "canonical_atom", "given_atom",
    "is_backbone_atom", "element_of", "written", "ATOM_LABELS",
]

BACKBONE = ("N", "CA", "C", "O")


@dataclass(frozen=True)
class ResidueTemplate:
    name: str
    smiles: str
    atoms: tuple[str, ...]
    kind: str = "amino"                        # or "nucleotide"

    @property
    def sidechain(self) -> tuple[str, ...]:
        return tuple(a for a in self.atoms if a not in BACKBONE)


# SMILES atom order == ``atoms`` order. Backbone is written N, CA, C, O first.
_TEMPLATES: dict[str, ResidueTemplate] = {
    t.name: t for t in (
        ResidueTemplate("ALA", "NC(C=O)C", ("N", "CA", "C", "O", "CB")),
        ResidueTemplate("ARG", "NC(C=O)CCCNC(N)=N", ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2")),
        ResidueTemplate("ASN", "NC(C=O)CC(=O)N", ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2")),
        ResidueTemplate("ASP", "NC(C=O)CC(=O)O", ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2")),
        ResidueTemplate("CYS", "NC(C=O)CS", ("N", "CA", "C", "O", "CB", "SG")),
        ResidueTemplate("GLN", "NC(C=O)CCC(=O)N", ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2")),
        ResidueTemplate("GLU", "NC(C=O)CCC(=O)O", ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2")),
        ResidueTemplate("GLY", "NCC=O", ("N", "CA", "C", "O")),
        ResidueTemplate("HIS", "NC(C=O)Cc1[nH]cnc1", ("N", "CA", "C", "O", "CB", "CG", "ND1", "CE1", "NE2", "CD2")),
        ResidueTemplate("ILE", "NC(C=O)C(CC)C", ("N", "CA", "C", "O", "CB", "CG1", "CD1", "CG2")),
        ResidueTemplate("LEU", "NC(C=O)CC(C)C", ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2")),
        ResidueTemplate("LYS", "NC(C=O)CCCCN", ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ")),
        ResidueTemplate("MET", "NC(C=O)CCSC", ("N", "CA", "C", "O", "CB", "CG", "SD", "CE")),
        ResidueTemplate("PHE", "NC(C=O)Cc1ccccc1", ("N", "CA", "C", "O", "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2")),
        ResidueTemplate("PRO", "N1C(C=O)CCC1", ("N", "CA", "C", "O", "CB", "CG", "CD")),
        ResidueTemplate("SER", "NC(C=O)CO", ("N", "CA", "C", "O", "CB", "OG")),
        ResidueTemplate("THR", "NC(C=O)C(O)C", ("N", "CA", "C", "O", "CB", "OG1", "CG2")),
        ResidueTemplate("TRP", "NC(C=O)Cc1c[nH]c2ccccc12",
                        ("N", "CA", "C", "O", "CB", "CG", "CD1", "NE1", "CE2", "CZ2", "CH2", "CZ3", "CE3", "CD2")),
        ResidueTemplate("TYR", "NC(C=O)Cc1ccc(O)cc1",
                        ("N", "CA", "C", "O", "CB", "CG", "CD1", "CE1", "CZ", "OH", "CE2", "CD2")),
        ResidueTemplate("VAL", "NC(C=O)C(C)C", ("N", "CA", "C", "O", "CB", "CG1", "CG2")),
    )
}

# -- nucleotides ---------------------------------------------------------------
# A nucleotide in a chain: its phosphate, the C5' and O5' that lead from it to the sugar, the sugar, and
# the base. The glycosidic carbon C1' is to it what the alpha carbon is to an amino acid: where the base,
# its "side chain", leaves the backbone of sugar and phosphate.
PHOSPHATE = ("P", "OP1", "OP2")
TAIL = ("C5'", "O5'")
SUGAR = ("C1'", "C2'", "O2'", "C3'", "O3'", "C4'", "O4'")
NUCLEOTIDE_BACKBONE = PHOSPHATE + TAIL + SUGAR

# A base as it hangs on C1': its SMILES (ring closures 2 and 3; the sugar has 1) and its atoms in that order.
_BASES = {
    "A": ("n2cnc3c(N)ncnc32", ("N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4")),
    "G": ("n2cnc3c(=O)[nH]c(N)nc32", ("N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4")),
    "C": ("n2c(=O)nc(N)cc2", ("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6")),
    "U": ("n2c(=O)[nH]c(=O)cc2", ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6")),
    "T": ("n2c(=O)[nH]c(=O)c(C)c2", ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6")),
}
_PURINE = (("N9", "C8", "N7", "C5", "C4"), ("C4", "C5", "C6", "N1", "C2", "N3"))
_PYRIMIDINE = (("N1", "C2", "N3", "C4", "C5", "C6"),)
#: The standard nucleotides by residue name, with the base each carries: RNA, then DNA.
_NUCLEOTIDES = {"A": "A", "C": "C", "G": "G", "U": "U", "DA": "A", "DC": "C", "DG": "G", "DT": "T"}


def _nucleotide(name: str) -> ResidueTemplate:
    """The template of a nucleotide: phosphate, C5', sugar with the base on it (a ribose, or without O2')."""
    smiles, atoms = _BASES[_NUCLEOTIDES[name]]
    head = ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C1'")
    if name.startswith("D"):
        return ResidueTemplate(name, f"P(=O)(O)OCC1OC({smiles})CC1O", head + atoms + ("C2'", "C3'", "O3'"), "nucleotide")
    return ResidueTemplate(name, f"P(=O)(O)OCC1OC({smiles})C(O)C1O", head + atoms + ("C2'", "O2'", "C3'", "O3'"),
                           "nucleotide")


# The nucleotides have a table of their own: ``_TEMPLATES`` holds the twenty amino acids and nothing else.
_NUCLEOTIDE_TEMPLATES = {name: _nucleotide(name) for name in _NUCLEOTIDES}

# Protonation / naming variants written by modelling packages.
_ALIASES = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "ARN": "ARG",
}
# A nucleotide as force fields name it: with R for RNA, and with 5 or 3 where it ends a chain.
_ALIASES.update({alias: base for base in "ACGU"
                 for alias in (f"R{base}", f"{base}5", f"{base}3", f"R{base}5", f"R{base}3")})
_ALIASES.update({alias: f"D{base}" for base in "ACGT" for alias in (f"D{base}5", f"D{base}3")})
# An atom of a nucleotide as older files name it.
_ATOM_ALIASES = {"O1P": "OP1", "O2P": "OP2", "C5M": "C7"}

_CLASS = {
    "ALA": "hydrophobic", "VAL": "hydrophobic", "LEU": "hydrophobic", "ILE": "hydrophobic",
    "MET": "hydrophobic", "PHE": "hydrophobic", "TRP": "hydrophobic", "TYR": "hydrophobic",
    "PRO": "hydrophobic", "CYS": "hydrophobic",
    "SER": "polar", "THR": "polar", "ASN": "polar", "GLN": "polar", "HIS": "polar",
    "ARG": "positive", "LYS": "positive",
    "ASP": "negative", "GLU": "negative",
    "GLY": "glycine",
}
_CLASS.update({name: "water" for name in WATER_NAMES})
_CLASS.update({name: "nucleotide" for name in _NUCLEOTIDES})

_RINGS = {
    "PHE": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
    "TYR": (("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),),
    "TRP": (("CG", "CD1", "NE1", "CE2", "CD2"), ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3")),
    "HIS": (("CG", "ND1", "CE1", "NE2", "CD2"),),
}
_RINGS.update({name: _PURINE if base in "AG" else _PYRIMIDINE for name, base in _NUCLEOTIDES.items()})

# Charge centres follow the common detector convention (PLIP): the centroid of
# the side-chain nitrogens / oxygens.
_POSITIVE = {"ARG": ("NE", "NH1", "NH2"), "LYS": ("NZ",), "HIS": ("ND1", "NE2")}
_NEGATIVE: dict[str, tuple[str, ...]] = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
# The phosphate of a nucleotide, under the names a file may give its two free oxygens.
_NEGATIVE.update({name: ("P", "OP1", "OP2", "O1P", "O2P") for name in _NUCLEOTIDES})

# Atoms that carry a hydrogen in the standard protonation state.
_DONORS = {
    "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"}, "GLN": {"NE2"}, "HIS": {"ND1", "NE2"},
    "LYS": {"NZ"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}, "TRP": {"NE1"}, "CYS": {"SG"},
}

_DONORS.update({name: {"A": {"N6"}, "G": {"N1", "N2"}, "C": {"N4"}, "U": {"N3"}, "T": {"N3"}}[base]
                | (set() if name.startswith("D") else {"O2'"}) for name, base in _NUCLEOTIDES.items()})

# Nitrogens that take a hydrogen bond: a ring nitrogen of histidine (which of the two carries the hydrogen
# a structure does not say), and the ring nitrogens of a base that carry none and are not bound to the sugar.
_ACCEPTOR_N: dict[str, set[str]] = {"HIS": {"ND1", "NE2"}}
_ACCEPTOR_N.update({name: {"A": {"N1", "N3", "N7"}, "G": {"N3", "N7"}, "C": {"N3"}, "U": set(), "T": set()}[base]
                    for name, base in _NUCLEOTIDES.items()})

_METAL_RESIDUES = {
    "ZN", "MG", "MN", "CA", "FE", "FE2", "CU", "CU1", "NI", "CO", "NA", "K", "CD", "HG",
}


def canonical_resname(name: str) -> str:
    up = name.upper()
    return _ALIASES.get(up, up)


def template_for(resname: str) -> ResidueTemplate | None:
    name = canonical_resname(resname)
    return _TEMPLATES.get(name) or _NUCLEOTIDE_TEMPLATES.get(name)


def is_nucleotide(resname: str) -> bool:
    t = template_for(resname)
    return t is not None and t.kind == "nucleotide"


def canonical_atom(resname: str, atom: str) -> str:
    """The name the templates have for an atom. Of a nucleotide the current one (``OP1`` for ``O1P``,
    ``C1'`` for ``C1*``, ``C7`` for ``C5M``); any other atom as it is."""
    if not is_nucleotide(resname):
        return atom
    name = atom.replace("*", "'")
    return _ATOM_ALIASES.get(name, name)


def given_atom(name: str, like: "Iterable[str]") -> str:
    """A nucleotide's atom, named by the templates, under the name a file would give it that names its
    atoms as ``like``: ``O1P`` where those say ``O1P`` or ``O2P`` (or ``C5M``), a star where they write
    the prime as a star. The names of a figure's contacts say which file it is."""
    like = tuple(like)
    if any(n.replace("*", "'") in _ATOM_ALIASES for n in like):
        name = {new: old for old, new in _ATOM_ALIASES.items()}.get(name, name)
    return name.replace("'", "*") if any("*" in n for n in like) else name


#: What is written on an atom that is drawn of a residue (``RenderOptions.atom_labels``).
ATOM_LABELS = ("elements", "names")


def element_of(name: str) -> str:
    """The element of an atom of a standard residue, by its name: N of ``NZ``, O of ``OG1`` and ``O5'``, S of ``SD``."""
    return "".join(ch for ch in name if ch.isalpha())[:1]


def written(name: str, labels: str = "names") -> str:
    """What a figure writes on an atom of a standard residue: its name in the structure file, or its element."""
    return name if labels == "names" else element_of(name)


def is_backbone_atom(resname: str, atom: str) -> bool:
    """Is this an atom of the backbone and not of the side chain: N, CA, C, O and OXT of an amino acid,
    the sugar and the phosphate of a nucleotide (whose base is its side chain)."""
    if is_nucleotide(resname):
        return canonical_atom(resname, atom) in NUCLEOTIDE_BACKBONE
    return atom in BACKBONE + ("OXT",)


def residue_class(resname: str) -> str:
    """One of ``hydrophobic``, ``polar``, ``positive``, ``negative``, ``glycine``,
    ``nucleotide``, ``water``, ``metal`` or ``other``."""
    up = canonical_resname(resname)
    if up in _CLASS:
        return _CLASS[up]
    if up in _METAL_RESIDUES:
        return "metal"
    return "other"


def aromatic_rings(resname: str) -> tuple[tuple[str, ...], ...]:
    return _RINGS.get(canonical_resname(resname), ())


def charged_groups(resname: str) -> list[tuple[int, tuple[str, ...]]]:
    """``[(sign, atom_names)]`` for the residue's side-chain charge centres."""
    up = canonical_resname(resname)
    out: list[tuple[int, tuple[str, ...]]] = []
    if up in _POSITIVE:
        out.append((+1, _POSITIVE[up]))
    if up in _NEGATIVE:
        out.append((-1, _NEGATIVE[up]))
    return out


def can_donate(resname: str, atom: str) -> bool | None:
    """Whether a protein atom can donate an H-bond in the standard protonation state.

    ``None`` when the residue is not a standard amino acid or nucleotide (unknown), and for
    the two oxygens of a nucleotide that carry a hydrogen only where a chain ends.
    """
    up = canonical_resname(resname)
    if up in _NUCLEOTIDE_TEMPLATES:
        name = canonical_atom(up, atom)
        return None if name in ("O3'", "O5'") else name in _DONORS.get(up, set())
    if up not in _TEMPLATES:
        return None
    if atom == "N":
        return up != "PRO"
    return atom in _DONORS.get(up, set())


def can_accept(resname: str, atom: str, element: str) -> bool | None:
    """Whether a protein atom can take an H-bond: an oxygen, or a nitrogen with a free lone pair.

    ``None`` when the residue is not a standard amino acid or nucleotide (unknown).
    """
    up = canonical_resname(resname)
    if up not in _TEMPLATES and up not in _NUCLEOTIDE_TEMPLATES:
        return None
    if element.upper() == "O":
        return True
    return canonical_atom(up, atom) in _ACCEPTOR_N.get(up, set())


def format_residue_label(resname: str, number: int, icode: str = "", chain: str | None = None) -> str:
    """``Gln712`` or, with a chain, ``Gln712(A)`` -- the LigPlot convention. A nucleotide keeps its
    capitals and is named as the templates name it: ``G7``, ``DA12``."""
    name = canonical_resname(resname) if is_nucleotide(resname) else resname.capitalize()
    base = f"{name}{number}{icode}"
    return f"{base}({chain})" if chain else base


@lru_cache(maxsize=None)
def template_bonds(resname: str) -> tuple[tuple[str, str, int], ...]:
    """Bonds of a residue template as ``(atom, atom, order)`` (aromatic kekulized)."""
    from rdkit import Chem

    t = template_for(resname)
    if t is None:
        return ()
    mol = Chem.MolFromSmiles(t.smiles)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    out = []
    for b in mol.GetBonds():
        out.append((t.atoms[b.GetBeginAtomIdx()], t.atoms[b.GetEndAtomIdx()], int(b.GetBondTypeAsDouble())))
    return tuple(out)
