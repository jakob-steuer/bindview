"""Two chains of RNA or DNA and the base pairs between them, drawn as their secondary structures facing
each other.

An :class:`~bindview.Interface` of two nucleic-acid chains is a column of one strand with the other around
it, and a base pair is one of its hydrogen bonds among the others. What a reader of such an interface asks
is another thing: which nucleotides pair across, in which register, by which edges (the Leontis–Westhof
family of the pair), where on each strand's own fold, and what stacks across. :class:`BasePairs` draws
that.

The pairs are **read, not detected**, as PLIP's contacts are read:

- :func:`read_mmcif_pairs`: the PDB's own annotation of a deposited entry (``_ndb_struct_na_base_pair`` of
  its mmCIF file). Its family is a number without the order of the edges for the asymmetric families, and
  it covers the asymmetric unit: a pair with a crystal mate is marked and not drawn.
- :func:`read_fr3d_pairs`, :func:`read_fr3d_stacks`: the FR3D tables RNA 3D Hub serves for an entry
  (``https://rna.bgsu.edu/rna3dhub/pdb/<ID>/interactions/fr3d/basepairs/csv``, and ``stacking``; licensed
  CC BY 4.0, which the figure's source line credits). The family names the first nucleotide's edge first.
- :func:`read_rnapolis_pairs`, :func:`read_rnapolis_stacks`: the JSON that RNApolis's ``annotator`` writes
  (``annotator -j pairs.json model.pdb``; RNApolis is MIT, run by the user in an environment of its own, and
  never imported here). It annotates any model, a frame of a simulation or a prediction among them; the
  family names the first nucleotide's edge first.
- :func:`pairs_from_dot_bracket`: a strand's secondary structure as the caller has it, with no family
  unless the caller names one.
- :func:`pairs_from_contacts`: where there is no annotation (a frame of a simulation, a prediction), a
  stand-in from bindview's plain distance table: two nucleotides with two or more hydrogen bonds from base
  to base, and not stacked, are a pair *without a family*, drawn dashed and said so in the key.

The figure: each strand's own pairs of the
Watson–Crick family make stems and loops. Where those of both strands and the pairs across nest when the
second strand is written after the first (a duplex, a ribozyme on its substrate), the two are laid out as
one secondary structure with a nick between them; where they do not (a kissing complex is a pseudoknot of
the joined strands), each strand is laid out alone, the nucleotides of a loop that pair across on one
straight side of it, and the second strand is turned onto the first by its pairs across. A pair that does
not shape the layout (another family, a tertiary contact) is drawn pale. Under the drawing, a key of what
it draws and a line saying where its pairs came from.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .diagram import Quality, RenderResult
from .errors import BindviewError, RenderError
from .graph import Graph
from .interactive import hooks_of_links
from .near import pairs_within
from .scene import Circle, Group, Item, Line, Paint, Polygon, Polyline, Rect, Scene, Text, text_box
from .structure import Structure, _read_categories, read_structure
from .svg import to_svg
from .theme import Metrics, Palette, get_palette, scheme

__all__ = ["BasePair", "BaseStack", "BasePairs", "read_mmcif_pairs", "read_fr3d_pairs", "read_fr3d_stacks",
           "read_rnapolis_pairs", "read_rnapolis_stacks", "pairs_from_dot_bracket", "pairs_from_contacts",
           "pairs_within_from_table", "FAMILIES", "SOURCES"]

#: The twelve Leontis–Westhof families, in the numbering of the mmCIF item ``hbond_type_12``.
FAMILIES = ("cWW", "tWW", "cWH", "tWH", "cWS", "tWS", "cHH", "tHH", "cHS", "tHS", "cSS", "tSS")

#: What the source line of a figure says of a source tag (:attr:`BasePair.source`); another tag is
#: written as it is.
SOURCES = {
    "pdb": "the PDB's annotation in the entry (_ndb_struct_na_base_pair)",
    "fr3d": "FR3D, from RNA 3D Hub (rna.bgsu.edu/rna3dhub, CC BY 4.0)",
    "rnapolis": "RNApolis's annotator (MIT), run on the model",
    "dot-bracket": "a secondary structure given as a dot-bracket",
    "plain-table": "bindview's plain distance table, without a family",
}

_ONE = {"A": "A", "C": "C", "G": "G", "U": "U", "DA": "A", "DC": "C", "DG": "G", "DT": "T", "RA": "A", "RC": "C",
        "RG": "G", "RU": "U", "A5": "A", "A3": "A", "C5": "C", "C3": "C", "G5": "G", "G3": "G", "U5": "U", "U3": "U",
        "RA5": "A", "RA3": "A", "RC5": "C", "RC3": "C", "RG5": "G", "RG3": "G", "RU5": "U", "RU3": "U",
        "DA5": "A", "DA3": "A", "DC5": "C", "DC3": "C", "DG5": "G", "DG3": "G", "DT5": "T", "DT3": "T"}
#: Modified nucleotides, drawn by their parent's letter and marked as modified.
_PARENT = {"CBV": "C", "5BU": "U", "OMC": "C", "OMG": "G", "OMU": "U", "PSU": "U", "5MC": "C", "5MU": "U",
           "1MA": "A", "1MG": "G", "2MG": "G", "M2G": "G", "7MG": "G", "H2U": "U", "4SU": "U", "GDP": "G",
           "GTP": "G", "A2M": "A", "I": "G"}
_BACKBONE = {"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'",
             "O2'", "C1'"}


def _letter(resname: str) -> tuple[str, bool]:
    """One letter for a nucleotide, and whether it is modified."""
    if resname in _ONE:
        return _ONE[resname], False
    return _PARENT.get(resname, "N"), True


# ---------------------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BasePair:
    """Two nucleotides whose bases pair, as a source says.

    ``family`` is the Leontis–Westhof family (one of :data:`FAMILIES`, or the same with the edges the other
    way round, ``"tHW"``), the edge of nucleotide ``a`` first; ``None`` where the source gives none.
    ``ordered`` is false where the source names the family without saying which base uses which edge.
    ``symmetry`` is the crystal operator of a partner that is a symmetry mate (``"6_555"``), empty for one
    in the file. ``source`` is a tag of :data:`SOURCES` or the caller's own words.
    """

    chain_a: str
    number_a: int
    chain_b: str
    number_b: int
    family: str | None = None
    icode_a: str = ""
    icode_b: str = ""
    ordered: bool = True
    symmetry: str = ""
    source: str = ""

    @property
    def key_a(self) -> tuple[str, int, str]:
        """``(chain, number, insertion code)`` of the first nucleotide."""
        return (self.chain_a, self.number_a, self.icode_a)

    @property
    def key_b(self) -> tuple[str, int, str]:
        """``(chain, number, insertion code)`` of the second."""
        return (self.chain_b, self.number_b, self.icode_b)


@dataclass(frozen=True)
class BaseStack:
    """Two bases stacked on each other, as a source says."""

    chain_a: str
    number_a: int
    chain_b: str
    number_b: int
    icode_a: str = ""
    icode_b: str = ""
    source: str = ""

    @property
    def key_a(self) -> tuple[str, int, str]:
        """``(chain, number, insertion code)`` of the first nucleotide."""
        return (self.chain_a, self.number_a, self.icode_a)

    @property
    def key_b(self) -> tuple[str, int, str]:
        """``(chain, number, insertion code)`` of the second."""
        return (self.chain_b, self.number_b, self.icode_b)


def _family(text: str) -> str | None:
    t = text.strip()
    if len(t) == 3 and t[0] in "ct" and t[1].upper() in "WHS" and t[2].upper() in "WHS":
        return t[0] + t[1].upper() + t[2].upper()
    return None


# ---------------------------------------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------------------------------------

def _text_of(path_or_text: "str | Path", is_text: "Callable[[str], bool]" = lambda text: False) -> str:
    """What a reader was handed: the text of the file at a path, gzipped or not, or the text itself.
    A ``str`` is the text where it has a line break, or where ``is_text`` says that it reads as what the
    reader reads (a table of one row has no line break); any other names a file, which has to be there."""
    if isinstance(path_or_text, str) and ("\n" in path_or_text or is_text(path_or_text)):
        return path_or_text
    p = Path(path_or_text)
    try:
        if p.suffix.lower() == ".gz":
            with gzip.open(p, "rt", encoding="utf-8") as handle:
                return handle.read()
        return p.read_text(encoding="utf-8")
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise BindviewError(f"cannot read {p}: {exc}") from exc


def read_mmcif_pairs(path_or_text: "str | Path") -> list[BasePair]:
    """The base pairs the PDB annotated in a deposited entry: ``_ndb_struct_na_base_pair`` of its mmCIF
    file (a path, gzipped or not, or the text). Chains and numbers are the authors', and the label ids
    where the table has no others, as the atoms of such a file are named. Of an entry of several models,
    the pairs of the first that the table lists. A pair whose partner is
    a crystal mate keeps its operator in :attr:`BasePair.symmetry`. ``hbond_type_12`` gives the family;
    for the asymmetric ones (cWH, tWH, cWS, tWS, cHS, tHS) the file does not say which base uses which
    edge, and the pair says so (``ordered=False``). A file without the category gives no pairs."""
    table = _read_categories(_text_of(path_or_text), {"_ndb_struct_na_base_pair"}).get("_ndb_struct_na_base_pair")
    if not table:
        return []
    if len({len(values) for values in table.values()}) > 1:
        raise BindviewError("_ndb_struct_na_base_pair: its columns do not have the same number of values (a row is "
                            "short)")
    named = {end: tuple(next((f"{end}_{ids}_{what}" for ids in ("auth", "label") if f"{end}_{ids}_{what}" in table),
                             "") for what in ("asym_id", "seq_id")) for end in "ij"}
    if not all(name for names in named.values() for name in names):
        raise BindviewError("_ndb_struct_na_base_pair names no nucleotide: it has neither the authors' chain and "
                            "number of each (i_auth_asym_id, i_auth_seq_id, j_auth_asym_id, j_auth_seq_id) nor the "
                            "label ids")
    (chain_i, seq_i), (chain_j, seq_j) = named["i"], named["j"]
    models = table.get("model_number", [])
    out = []
    for k in range(len(table[seq_i])):
        if models and models[k] != models[0]:
            continue                                  # the first model only, as the structure is read

        def col(name: str) -> str:
            v = table.get(name, ["?"] * (k + 1))[k]
            return "" if v in ("?", ".") else v

        def number(name: str) -> int:
            try:
                return int(col(name))
            except ValueError:
                raise BindviewError(f"_ndb_struct_na_base_pair, row {k + 1}: {name} is {table[name][k]!r}, which is "
                                    f"no residue number") from None
        code = col("hbond_type_12")
        fam = FAMILIES[int(code) - 1] if code.isdigit() and 1 <= int(code) <= 12 else None
        sym_i, sym_j = col("i_symmetry") or "1_555", col("j_symmetry") or "1_555"
        sym = "" if sym_i == sym_j else (sym_j if sym_i == "1_555" else sym_i)
        out.append(BasePair(col(chain_i), number(seq_i), col(chain_j),
                            number(seq_j), fam, col("i_pdb_ins_code"), col("j_pdb_ins_code"),
                            ordered=fam is None or fam[1] == fam[2], symmetry=sym, source="pdb"))
    return out


def _unit(text: str) -> tuple[str, int, str, str]:
    """An FR3D unit id, ``PDB|model|chain|component|number|atom|alt|icode|symmetry``: chain, number,
    insertion code, symmetry operator."""
    f = text.strip().split("|")
    if len(f) < 5 or not f[4].lstrip("-").isdigit():
        raise BindviewError(f"not an FR3D unit id: {text.strip()!r}")
    icode = f[7] if len(f) > 7 else ""
    sym = f[8] if len(f) > 8 else ""
    return f[2], int(f[4]), icode, sym


def _fr3d_rows(path_or_text: "str | Path") -> Iterable[tuple[tuple[str, int, str, str], str, tuple[str, int, str, str]]]:
    # A unit id has its fields between bars, and no file name has one.
    table = csv.reader(io.StringIO(_text_of(path_or_text, lambda text: "|" in text)))
    for row in table:
        if len(row) < 3 or not row[0].strip():
            continue
        if row[0].split("|")[1:2] not in (["1"], []):      # the first model only
            continue
        try:
            units = _unit(row[0]), _unit(row[2])
        except BindviewError as exc:
            raise BindviewError(f"line {table.line_num} of the FR3D table: {exc}") from None
        yield units[0], row[1].strip(), units[1]


def read_fr3d_pairs(path_or_text: "str | Path", source: str = "fr3d") -> list[BasePair]:
    """The base pairs of an FR3D table as RNA 3D Hub serves it (CSV: unit id, family, unit id; a path or
    the text). Each pair is listed once though the table lists it from both sides; near pairs (``ncWW``)
    are left out. ``source`` is what the figure credits: the default credits RNA 3D Hub; give your own
    words for a table made elsewhere."""
    out: dict[frozenset, BasePair] = {}
    for (c1, n1, i1, s1), fam, (c2, n2, i2, s2) in _fr3d_rows(path_or_text):
        f = _family(fam)
        if f is None:
            continue
        key = frozenset(((c1, n1, i1, s1), (c2, n2, i2, s2)))
        if key not in out:
            sym = s2 if s2 and s2 != "1_555" else (s1 if s1 and s1 != "1_555" else "")
            out[key] = BasePair(c1, n1, c2, n2, f, i1, i2, symmetry=sym, source=source)
    return list(out.values())


def read_fr3d_stacks(path_or_text: "str | Path", source: str = "fr3d") -> list[BaseStack]:
    """The stacks of RNA 3D Hub's FR3D stacking table (``s33``, ``s35``, ``s53``, ``s55``), each once. A
    stack with a crystal mate (a unit id with a symmetry operator other than the identity, ``1_555``) is
    left out: a figure has the chains of the file."""
    out: dict[frozenset, BaseStack] = {}
    for (c1, n1, i1, s1), kind, (c2, n2, i2, s2) in _fr3d_rows(path_or_text):
        if not kind.startswith("s") or s1 not in ("", "1_555") or s2 not in ("", "1_555"):
            continue
        key = frozenset(((c1, n1, i1), (c2, n2, i2)))
        out.setdefault(key, BaseStack(c1, n1, c2, n2, i1, i2, source=source))
    return list(out.values())


_RNAPOLIS_LISTS = ("base_pairs", "stackings")


def _rnapolis(path_or_json: "str | Path | dict") -> dict:
    """What RNApolis's annotator wrote, as the object it is: one with a list of ``base_pairs``, of
    ``stackings``, or of both. An object with neither is another program's, and is refused."""
    data: Any = path_or_json
    if not isinstance(data, dict):
        try:
            data = json.loads(_text_of(data, lambda text: text.lstrip().startswith(("{", "["))))
        except ValueError as exc:
            raise BindviewError(f"not the JSON of RNApolis's annotator: it is no JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise BindviewError(f"not the JSON of RNApolis's annotator: an object is expected, not a JSON "
                            f"{type(data).__name__}")
    if not any(key in data for key in _RNAPOLIS_LISTS):
        raise BindviewError(f"not the JSON of RNApolis's annotator: it has neither 'base_pairs' nor 'stackings' "
                            f"(its keys: {', '.join(repr(k) for k in data) or 'none'})")
    for key in _RNAPOLIS_LISTS:
        if not isinstance(data.get(key, []), list):
            raise BindviewError(f"not the JSON of RNApolis's annotator: its {key!r} is no list")
    return data


def _rnapolis_entries(path_or_json: "str | Path | dict", key: str
                      ) -> Iterable[tuple[tuple[str, int, str], tuple[str, int, str], dict]]:
    """The entries of one list of RNApolis's JSON, each with its two residues."""
    for n, entry in enumerate(_rnapolis(path_or_json).get(key, []), start=1):
        where = f"{key}, entry {n}, of RNApolis's JSON"
        if not isinstance(entry, dict):
            raise BindviewError(f"{where} is no object: {entry!r}")
        for end in ("nt1", "nt2"):
            if end not in entry:
                raise BindviewError(f"{where} has no {end!r}")
        yield _rnapolis_nt(entry["nt1"]), _rnapolis_nt(entry["nt2"]), entry


def _rnapolis_nt(entry: Any) -> tuple[str, int, str]:
    """A residue of RNApolis's JSON: its author fields where it has them, else its label fields."""
    try:
        ids = entry.get("auth") or entry.get("label") or {}
        return str(ids["chain"]), int(ids["number"]), str(ids.get("icode") or "")
    except (AttributeError, KeyError, TypeError, ValueError):
        raise BindviewError(f"a residue of RNApolis's JSON without a chain and a number: {entry!r}") from None


def read_rnapolis_pairs(path_or_json: "str | Path | dict", source: str = "rnapolis") -> list[BasePair]:
    """The base pairs of the JSON that RNApolis's ``annotator`` writes (``annotator -j pairs.json model.pdb``),
    a path, its text or the parsed object. The family (``lw``) names the first nucleotide's edge first; a pair
    without one is read without a family. Each pair once."""
    out: dict[frozenset, BasePair] = {}
    for (c1, n1, i1), (c2, n2, i2), bp in _rnapolis_entries(path_or_json, "base_pairs"):
        key = frozenset(((c1, n1, i1), (c2, n2, i2)))
        if key not in out:
            out[key] = BasePair(c1, n1, c2, n2, _family(str(bp.get("lw") or "")), i1, i2, source=source)
    return list(out.values())


def read_rnapolis_stacks(path_or_json: "str | Path | dict", source: str = "rnapolis") -> list[BaseStack]:
    """The stacks of RNApolis's JSON (``stackings``), each once; a figure draws those across its strands."""
    out: dict[frozenset, BaseStack] = {}
    for (c1, n1, i1), (c2, n2, i2), _ in _rnapolis_entries(path_or_json, "stackings"):
        out.setdefault(frozenset(((c1, n1, i1), (c2, n2, i2))), BaseStack(c1, n1, c2, n2, i1, i2, source=source))
    return list(out.values())


def pairs_from_dot_bracket(structure: "Structure | str | Path", chain: str, dot_bracket: str,
                           family: str | None = None, source: str = "dot-bracket") -> list[BasePair]:
    """The pairs of one strand's secondary structure written as a dot-bracket, one character a nucleotide
    of ``chain`` in the order of the chain: ``()``, ``[]``, ``{}``, ``<>`` for pairs, a dot (or any other
    sign) for an unpaired nucleotide. A letter is refused: some notations write a further level of
    pseudoknots with letters (``A`` closing with ``a``), which are not read here and would be pairs lost
    without a word. No family unless the caller names one (``family="cWW"`` for a secondary structure of
    Watson–Crick pairs)."""
    st = structure if isinstance(structure, Structure) else read_structure(structure)
    residues = st.polymer_residues(chain)
    if len(residues) != len(dot_bracket):
        raise BindviewError(f"the dot-bracket has {len(dot_bracket)} characters and chain {chain} has "
                            f"{len(residues)} nucleotides")
    if family is not None and _family(family) is None:
        raise BindviewError(f"{family!r} is not a Leontis–Westhof family (such as 'cWW')")
    opening = {"(": ")", "[": "]", "{": "}", "<": ">"}
    stacks: dict[str, list[int]] = {k: [] for k in opening.values()}
    out = []
    for i, ch in enumerate(dot_bracket):
        if ch in opening:
            stacks[opening[ch]].append(i)
        elif ch in stacks:
            if not stacks[ch]:
                raise BindviewError(f"the dot-bracket closes a pair at {i + 1} that it did not open")
            j = stacks[ch].pop()
            a, b = residues[j], residues[i]
            out.append(BasePair(chain, a.number, chain, b.number, _family(family) if family else None,
                                a.icode, b.icode, source=source))
        elif ch.isalpha():
            raise BindviewError(f"the dot-bracket has the letter {ch!r} at {i + 1}: a pair is written with one of "
                                f"the brackets () [] {{}} <>, and a letter, which some notations use for a further "
                                f"level of pseudoknots, is not read")
    if any(stacks.values()):
        raise BindviewError("the dot-bracket opens a pair that it does not close")
    return out


def _bond_atoms(c: Any) -> tuple[str, str]:
    """The atom at each end of a hydrogen bond of the plain table, an atom of the sugar under the name it
    has today (``O2'`` for the ``O2*`` of old files). A row without them (a table written by hand may
    leave them out) cannot be told to be from base to base or not, and is refused."""
    if not c.atoms_a or not c.atoms_b:
        raise BindviewError(f"the hydrogen bond of {c.chain_a}:{c.number_a}{c.icode_a} with "
                            f"{c.chain_b}:{c.number_b}{c.icode_b} names no atom on "
                            f"{'either side' if not c.atoms_a and not c.atoms_b else 'one side'}: whether it is from "
                            f"base to base cannot be told without its atoms")
    return c.atoms_a[0].replace("*", "'"), c.atoms_b[0].replace("*", "'")


def pairs_from_contacts(contacts: Sequence[Any]) -> tuple[list[BasePair], list[BaseStack]]:
    """A stand-in where no annotation is at hand: from bindview's plain distance table between two chains
    (:func:`bindview.distance_table.interface_contacts`, with ``stacking=True`` for the stacks), two
    nucleotides with two or more hydrogen bonds from base to base are a pair *without a family*, unless the
    table also finds them stacked (a pair and a stack exclude each other); the stacks of the table are
    stacks. A row may be written from either side: a bond is counted once, and a pair or a stack is named
    from the side its first row is written from. Nothing here classifies a pair."""
    bonds: dict[frozenset, set[frozenset]] = {}
    ends: dict[frozenset, tuple] = {}
    stacks: dict[frozenset, BaseStack] = {}
    for c in contacts:
        a, b = (c.chain_a, c.number_a, c.icode_a), (c.chain_b, c.number_b, c.icode_b)
        two = frozenset((a, b))
        if c.type.value == "pi_stacking":
            stacks.setdefault(two, BaseStack(a[0], a[1], b[0], b[1], a[2], b[2], source="plain-table"))
        elif c.type.value == "hbond":
            atom_a, atom_b = _bond_atoms(c)
            if atom_a not in _BACKBONE and atom_b not in _BACKBONE:
                ends.setdefault(two, (a, b))
                bonds.setdefault(two, set()).add(frozenset(((a, atom_a), (b, atom_b))))
    pairs = [BasePair(a[0], a[1], b[0], b[1], None, a[2], b[2], source="plain-table")
             for two, (a, b) in ends.items() if len(bonds[two]) >= 2 and two not in stacks]
    return pairs, list(stacks.values())


def pairs_within_from_table(structure: "Structure | str | Path", chain: str) -> list[BasePair]:
    """The stand-in of :func:`pairs_from_contacts` within one strand, where no annotation and no dot-bracket
    is at hand: two nucleotides of the strand that are not neighbours, with two or more hydrogen bonds from
    base to base by the plain distance table's rule (an N or O within
    :data:`~bindview.distance_table.HBOND` of an N or O, one able to donate and the other to accept), are a
    pair *without a family*. A nucleotide is in one pair at most: pairs are taken most bonds first, then
    the shorter mean bond, and one whose nucleotide is taken is left out -- so a base triple shows as one
    pair. Stacks within a strand are not looked for; nothing here classifies a pair.

    The one-pair rule is needed because in a helix a base also makes two longer bonds (3.25 to 3.5 Å) to
    the base diagonal to it, which would count as further pairs. It is fitted to one frame, of two
    hairpins of a riboswitch, for which it gives the seven stem pairs of each that their dot-bracket
    gives."""
    from .distance_table import HBOND, _accepts, _donates

    st = structure if isinstance(structure, Structure) else read_structure(structure)
    keys = st.polymer_residues(chain)
    atoms = []                                   # (position in the strand, key, resname, name, element, xyz)
    for n, k in enumerate(keys):
        name = st.residue_name(k)
        for a in st.residue_atoms(k):
            label = a.name.replace("*", "'")
            if a.element in ("N", "O") and label not in _BACKBONE:
                atoms.append((n, k, name, label, a.element, np.array(a.xyz, dtype=float)))
    if not atoms:
        return []
    xyz = np.array([a[5] for a in atoms])
    first, second, apart = pairs_within(xyz, xyz, HBOND)
    bonds: dict[tuple[int, int], list[float]] = {}
    for x, y, d in zip(first.tolist(), second.tolist(), apart.tolist()):
        if y <= x:                                   # (each pair once, in the order of the strand)
            continue
        (n1, _, r1, a1, e1, _), (n2, _, r2, a2, e2, _) = atoms[x], atoms[y]
        if abs(n1 - n2) < 2:
            continue
        if (_donates(r1, a1) and _accepts(r2, a2, e2)) or (_donates(r2, a2) and _accepts(r1, a1, e1)):
            bonds.setdefault((min(n1, n2), max(n1, n2)), []).append(d)
    taken: set[int] = set()
    kept = []
    for (i, j), d in sorted(bonds.items(), key=lambda kv: (-len(kv[1]), sum(kv[1]) / len(kv[1]), kv[0])):
        if len(d) >= 2 and i not in taken and j not in taken:
            taken |= {i, j}
            kept.append((i, j))
    return [BasePair(chain, keys[i].number, chain, keys[j].number, None, keys[i].icode, keys[j].icode,
                     source="plain-table") for i, j in sorted(kept)]


# ---------------------------------------------------------------------------------------------------------
# Layout (in steps: one step between neighbours of a strand)
# ---------------------------------------------------------------------------------------------------------

B = 1.0                  # backbone step
P = 1.7                  # pair width, in steps


def _rot(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]])


def _unit_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _circle_radius(lengths: list[float]) -> float:
    lo, hi = max(lengths) / 2 + 1e-9, sum(lengths)

    def total(r: float) -> float:
        return sum(2 * math.asin(min(1.0, e / (2 * r))) for e in lengths)
    if total(lo) < 2 * math.pi:
        return lo
    for _ in range(80):
        mid = (lo + hi) / 2
        if total(mid) > 2 * math.pi:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _radial(n: int, pairs: Sequence[tuple[int, int]], nick: int | None = None,
            runs: "dict[int, int] | None" = None) -> np.ndarray:
    """Coordinates for ``n`` nucleotides with nested ``pairs``: the exterior along the x axis, stems up,
    each loop a polygon on a circle; the link ``nick -> nick + 1`` is no backbone; consecutive members of
    one run (``runs``: nucleotide -> run) in a loop lie on one chord of its circle."""
    run_of = runs or {}
    pt = [-1] * n
    for i, j in pairs:
        pt[i], pt[j] = j, i
    X = np.full((n, 2), np.nan)

    def stem(i: int, j: int) -> None:
        u = _rot(_unit_vec(X[j] - X[i]))
        while i + 1 < j - 1 and pt[i + 1] == j - 1:
            X[i + 1], X[j - 1] = X[i] + B * u, X[j] + B * u
            i, j = i + 1, j - 1
        loop(i, j, u)

    def loop(k: int, l: int, u: np.ndarray) -> None:
        V, t = [k], k + 1
        while t < l:
            if pt[t] == -1:
                V.append(t)
                t += 1
            else:
                V += [t, pt[t]]
                t = pt[t] + 1
        V.append(l)
        children = [(V[a], V[a + 1]) for a in range(1, len(V) - 2) if pt[V[a]] == V[a + 1]]
        e1 = _unit_vec(X[l] - X[k])
        if nick is not None and k <= nick < l and not children:
            for s, t in enumerate(range(k + 1, nick + 1), 1):       # open ends: the strands run on
                X[t] = X[k] + s * B * u
            for s, t in enumerate(range(l - 1, nick, -1), 1):
                X[t] = X[l] + s * B * u
            return
        corners: list[int] = []
        lengths: list[float] = []
        inner: dict[int, list[int]] = {}
        a = 0
        while True:
            corners.append(V[a])
            if a == len(V) - 1:
                break
            b = a
            while (a >= 1 and b + 1 < len(V) - 1 and V[b + 1] == V[b] + 1 and pt[V[b]] != V[b + 1]
                   and V[b] in run_of and run_of.get(V[b + 1]) == run_of[V[b]]):
                b += 1
            if b > a + 1:
                inner[V[a]] = V[a + 1:b]
                lengths.append((b - a) * B)
                a = b
            else:
                x, y = V[a], V[a + 1]
                lengths.append(P if pt[x] == y else (2.5 * B if x == nick else B))
                a += 1
        lengths.append(P)
        r = _circle_radius(lengths)
        h = math.sqrt(max(r * r - P * P / 4, 0.0))
        c = (X[k] + X[l]) / 2 + h * u
        ang = math.atan2(-h, -P / 2)
        for a in range(1, len(corners) - 1):
            ang -= 2 * math.asin(min(1.0, lengths[a - 1] / (2 * r)))
            X[corners[a]] = c + r * (math.cos(ang) * e1 + math.sin(ang) * u)
        for a in range(len(corners) - 1):
            mids = inner.get(corners[a], [])
            s0, s1 = X[corners[a]], X[corners[a + 1]]
            for q, m in enumerate(mids, 1):
                X[m] = s0 + (s1 - s0) * q / (len(mids) + 1)
        for x, y in children:
            stem(x, y)

    x, t = 0.0, 0
    while t < n:
        if pt[t] == -1 or pt[t] < t:
            X[t] = (x, 0.0)
            x += 2.5 * B if t == nick else B
            t += 1
        else:
            j = pt[t]
            X[t], X[j] = (x, 0.0), (x + P, 0.0)
            stem(t, j)
            x += P + (2.5 * B if j == nick else B)
            t = j + 1
    return X


def _crossing(p: tuple[int, int], q: tuple[int, int]) -> bool:
    (i, j), (k, l) = sorted(p), sorted(q)
    return i < k < j < l or k < i < l < j


def _nested(pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """The pairs that make the layout: runs of stacked pairs, longest first, each kept where it crosses
    nothing kept before it and uses no nucleotide twice."""
    ps = sorted({(min(p), max(p)) for p in pairs})
    have = set(ps)
    runs, seen = [], set()
    for p in ps:
        if p in seen:
            continue
        run, (i, j) = [p], p
        while (i + 1, j - 1) in have and i + 1 < j - 1:
            i, j = i + 1, j - 1
            run.append((i, j))
        seen.update(run)
        runs.append(run)
    runs.sort(key=lambda r: (-len(r), r[0]))
    kept: list[tuple[int, int]] = []
    used: set[int] = set()
    for run in runs:
        if any(i in used or j in used for i, j in run):
            continue
        if any(_crossing(p, q) for p in run for q in kept):
            continue
        kept += run
        used.update(x for p in run for x in p)
    return kept


def _runs(across: Sequence[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    have = set(across)
    out, seen = [], set()
    for p in sorted(across):
        if p in seen:
            continue
        run, (i, j) = [p], p
        while (i + 1, j - 1) in have:
            i, j = i + 1, j - 1
            run.append((i, j))
        seen.update(run)
        out.append(run)
    return out


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    r = vt.T @ np.diag([1.0, d]) @ u.T
    return r, cd - r @ cs


def _relax(X: np.ndarray, springs: list[tuple[int, int, float, float]], iters: int = 2500,
           d_min: float = 1.3) -> np.ndarray:
    """Springs and a short-range push, each step scaled by a node's own stiffness."""
    X = X.copy()
    si = np.array([s[0] for s in springs], dtype=int)
    sj = np.array([s[1] for s in springs], dtype=int)
    rest = np.array([s[2] for s in springs])
    w = np.array([s[3] for s in springs])
    n = len(X)
    stiff = np.full(n, 2.0)
    np.add.at(stiff, si, w)
    np.add.at(stiff, sj, w)
    iu = np.triu_indices(n, 1)
    for it in range(iters):
        g = np.zeros_like(X)
        d = X[si] - X[sj]
        L = np.linalg.norm(d, axis=1) + 1e-9
        f = (w * (L - rest) / L)[:, None] * d
        np.add.at(g, si, f)
        np.add.at(g, sj, -f)
        D = X[iu[0]] - X[iu[1]]
        Ld = np.linalg.norm(D, axis=1) + 1e-9
        close = Ld < d_min
        if close.any():
            fr = (-(d_min - Ld[close]) / Ld[close])[:, None] * D[close] * 2.0
            np.add.at(g, iu[0][close], fr)
            np.add.at(g, iu[1][close], -fr)
        X -= 0.45 * (1 - 0.8 * it / iters) * g / stiff[:, None]
    return X


def _tidy(X: np.ndarray, side: np.ndarray, kept: Sequence[tuple[int, int]],
          across: Sequence[tuple[int, int]]) -> np.ndarray:
    """Push apart what overlaps, keeping each stretch's shape: backbone and pairs as they are, every
    nucleotide at its distance from the next but one, and runs across as straight ladders."""
    springs: list[tuple[int, int, float, float]] = []
    n = len(X)
    for i in range(n - 1):
        if side[i] == side[i + 1]:
            springs.append((i, i + 1, B, 8.0))
            if i + 2 < n and side[i + 2] == side[i]:
                springs.append((i, i + 2, float(np.linalg.norm(X[i] - X[i + 2])), 2.0))
    have = set(kept)
    for i, j in kept:
        springs.append((i, j, P, 8.0))
        if (i + 1, j - 1) in have:
            springs += [(i, j - 1, math.hypot(B, P), 4.0), (i + 1, j, math.hypot(B, P), 4.0)]
    for run in _runs(across):
        for q, (i, j) in enumerate(run):
            springs.append((i, j, P, 6.0))
            if q + 1 < len(run):
                i2, j2 = run[q + 1]
                springs += [(i, j2, math.hypot(B, P), 4.0), (i2, j, math.hypot(B, P), 4.0)]
            if q + 2 < len(run):
                i3, j3 = run[q + 2]
                springs += [(i, i3, 2 * B, 3.0), (j, j3, 2 * B, 3.0)]
    return _relax(X, springs)


def _upright(X: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Turned so that the long axis is horizontal, the first strand left of or above the second; never
    mirrored (a strand runs 5' to 3' the same way round in every figure)."""
    c = X - X.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    a = vt[0]
    Y = c @ np.array([[a[0], a[1]], [-a[1], a[0]]]).T
    m0, m1 = Y[side == 0].mean(0), Y[side == 1].mean(0)
    dx, dy = m1 - m0
    if (abs(dx) >= abs(dy) and dx < 0) or (abs(dx) < abs(dy) and dy > 0):
        Y = -Y
    return Y


def _shapes(p: BasePair) -> bool:
    """May a pair shape the layout: one of the Watson–Crick family, or one of no known family."""
    return p.family is None or p.family == "cWW"


@dataclass
class _Laid:
    X: np.ndarray                      # steps, y up
    side: np.ndarray
    route: str
    shaping: set[frozenset]            # index pairs that shaped the layout


def _lay_out(n1: int, n2: int, pairs: Sequence[tuple[int, int, BasePair]]) -> _Laid:
    side = np.array([0] * n1 + [1] * n2)
    elig = [(i, j) for i, j, p in pairs if _shapes(p)]
    across = [(i, j) if i < n1 else (j, i) for i, j, p in pairs if _shapes(p) and (i < n1) != (j < n1)]
    kept = _nested(elig)
    kept_set = {frozenset(k) for k in kept}
    if across and all(frozenset(x) in kept_set for x in across):
        X = _radial(n1 + n2, kept, nick=n1 - 1)
        X = _tidy(X, side, kept, across)
        return _Laid(_upright(X, side), side, "one structure of the joined strands", kept_set)
    runs = _runs(across)
    run_of: dict[int, int] = {}
    for rn, run in enumerate(runs):
        if len(run) >= 2:
            for i, j in run:
                run_of[i], run_of[j] = rn, rn
    parts, own_all = [], []
    for s, (base, size) in enumerate(((0, n1), (n1, n2))):
        own = [(i - base, j - base) for i, j in elig if side[i] == s and side[j] == s]
        k = _nested(own)
        own_all += [(i + base, j + base) for i, j in k]
        local = {i - base: r for i, r in run_of.items() if side[i] == s}
        parts.append(_radial(size, k, runs=local))
    X1, X2 = parts
    if across:
        n_out = _normals_of(X1)
        src = np.array([X2[j - n1] for _, j in across])
        dst = np.array([X1[i] + P * n_out[i] for i, _ in across])
        if len(across) >= 2:
            r, t = _kabsch(src, dst)
        else:
            r, t = np.eye(2), dst.mean(0) - src.mean(0)
    else:
        r, t = np.eye(2), np.array([X1[:, 0].max() + 4 * B - X2[:, 0].min(), 0.0])
    X = np.vstack([X1, X2 @ r.T + t])
    X = _tidy(X, side, own_all, across)
    return _Laid(_upright(X, side), side, "each strand alone, placed by its pairs across",
                 {frozenset(k) for k in own_all} | {frozenset(k) for k in across})


def _normals_of(X: np.ndarray) -> np.ndarray:
    """Outward directions: away from the neighbours within reach."""
    N = np.zeros_like(X)
    for i in range(len(X)):
        d = np.linalg.norm(X - X[i], axis=1)
        near = (d < 2.6) & (d > 0)
        N[i] = _unit_vec(X[i] - X[near].mean(0)) if near.any() else np.array([0.0, 1.0])
    return N


# ---------------------------------------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------------------------------------

#: The colours of the two strands and their fills, by palette name; a palette without them (one a caller
#: made) is given these of the light or the dark palette, by its background.
_STRAND_LIGHT = {"strand_1": "#2f63b0", "strand_1_fill": "#e2ebf8", "strand_2": "#c0561b", "strand_2_fill": "#f9e5d8"}
_STRAND_DARK = {"strand_1": "#7aa7e6", "strand_1_fill": "#1f3350", "strand_2": "#ea9a63", "strand_2_fill": "#4a2c19"}

_EDGE = {"W": "Watson–Crick", "H": "Hoogsteen", "S": "sugar"}

#: What each kind of mark means, in the key's words.
_KEY = {
    "strands": "strand {a} and strand {b}; filled: paired across",
    "across": "pair across the strands, cis Watson–Crick (G–C, A–U) where it has no symbol",
    "across-none": "pair across, its family not annotated",
    "within": "pair within a strand",
    "within-none": "pair within a strand, its family not annotated",
    "tertiary": "pale: a pair that does not shape the layout (another family, or a tertiary contact)",
    "wobble": "G·U wobble",
    "stack": "bases stacked across the strands",
    "hbond": "other hydrogen bond across, with a sugar or phosphate atom",
    "modified": "modified nucleotide, by its parent's letter",
}


def _family_name(f: str) -> str:
    """One name for a family whichever base is named first: tHW is tWH (edges in the order W, H, S)."""
    return f[0] + "".join(sorted(f[1:], key="WHS".index))


@dataclass(frozen=True)
class _Nt:
    chain: str
    number: int
    icode: str
    resname: str
    letter: str
    modified: bool

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.number, self.icode)

    @property
    def name(self) -> str:
        return f"{self.resname} {self.number}{self.icode}, chain {self.chain}"


class BasePairs:
    """Two chains of RNA or DNA and the base pairs between them, as their secondary structures facing each
    other (see the module's words).

    ``structure`` is a :class:`~bindview.Structure` or a path; ``chains`` names the two strands. ``pairs``
    are :class:`BasePair` records from any of the readers (or several, added together); the pairs of each
    strand with itself shape its stems and loops, those between the two strands are the interface. Pairs
    naming another chain, or a nucleotide the chains do not have, are left out and said in the result.
    ``stacks``: :class:`BaseStack` records; left out, they are taken from ``contacts``. ``contacts``: the
    plain distance table between the two chains (:func:`bindview.distance_table.interface_contacts`); its
    hydrogen bonds with a sugar or phosphate atom between two nucleotides that are no pair are drawn as
    other hydrogen bonds across, and with ``stacking=True`` its stacks are the stacks. ``source``: the line
    under the key; left out, it is made from the sources the records name.

    ``ligand``: a residue of the structure that is no part of the two chains (``"BGD"``, ``"C:1"``), drawn as a
    box with its name between the nucleotides it touches, with a line for each kind of contact to each of
    them. Its contacts are ``ligand_contacts`` -- interaction records, rows, or the path of a table, as a
    :class:`~bindview.Diagram` takes them (PLIP's, say) -- and left out, those of bindview's plain distance
    table (:func:`bindview.distance_table.ligand_contacts`, with ``ligand_smiles`` for the bond orders).
    Contacts to anything but the nucleotides of the two chains are left out and said. ``ligand_name``: the
    word in the box (the residue name, left out).
    """

    def __init__(self, structure: "Structure | str | Path", chains: Sequence[str], pairs: Iterable[BasePair],
                 *, stacks: "Iterable[BaseStack] | None" = None, contacts: Iterable[Any] = (),
                 source: str | None = None, ligand: "str | tuple[str, int] | Any | None" = None,
                 ligand_contacts: "Iterable[Any] | str | Path | None" = None, ligand_name: str | None = None,
                 ligand_smiles: str | None = None) -> None:
        self.structure = structure if isinstance(structure, Structure) else read_structure(structure)
        chains = list(chains)
        if len(chains) != 2 or chains[0] == chains[1]:
            raise BindviewError(f"BasePairs draws two chains; {chains!r} are not two")
        self.chains = (chains[0], chains[1])
        self.strands: list[list[_Nt]] = []
        for chain in self.chains:
            residues = self.structure.polymer_residues(chain)
            if not residues:
                where = self.structure.source or "the structure"
                # A chain is nucleotides that are joined to one another: one alone, or several apart, is none.
                apart = [k for k in self.structure.residues() if k.chain == chain and self._is_nucleotide(k)]
                if apart:
                    named = ", ".join(f"{self.structure.residue_name(k)} {k.number}{k.icode}" for k in apart[:4])
                    raise BindviewError(
                        f"chain {chain!r} has {'one nucleotide' if len(apart) == 1 else f'{len(apart)} nucleotides'} "
                        f"in {where} ({named}{', ...' if len(apart) > 4 else ''}) and no two that are joined to each "
                        f"other: a strand is two or more nucleotides joined through a phosphate, and BasePairs "
                        f"draws two strands")
                have = sorted({a.chain for a in self.structure.atoms})
                raise BindviewError(f"chain {chain!r} has no nucleotides in {where}"
                                    f" (its chains: {', '.join(have)})")
            strand = []
            for k in residues:
                name = self.structure.residue_name(k)
                letter, mod = _letter(name)
                if letter == "N" and mod and not self._is_nucleotide(k):
                    raise BindviewError(f"chain {chain!r} is not a chain of nucleotides ({name} {k.number})")
                strand.append(_Nt(chain, k.number, k.icode, name, letter, mod))
            self.strands.append(strand)
        self.warnings: list[str] = []
        index = {n.key: i for i, n in enumerate(self.strands[0] + self.strands[1])}
        self._index = index
        self.pairs: list[BasePair] = []
        seen: set[frozenset] = set()
        foreign = missing = mates = 0
        for p in pairs:
            if p.chain_a not in self.chains or p.chain_b not in self.chains:
                foreign += 1
                continue
            if p.symmetry:
                mates += 1
                continue
            if p.key_a not in index or p.key_b not in index:
                missing += 1
                continue
            pk = frozenset((p.key_a, p.key_b))
            if pk in seen or p.key_a == p.key_b:
                continue
            seen.add(pk)
            self.pairs.append(p)
        if mates:
            self.warnings.append(f"{mates} pair(s) with a crystal mate left out: the figure has the chains of the "
                                 f"structure, not their symmetry mates")
        if missing:
            self.warnings.append(f"{missing} pair(s) naming a nucleotide that chains {self.chains[0]} and "
                                 f"{self.chains[1]} do not have left out")
        if foreign:
            self.warnings.append(f"{foreign} pair(s) of other chains left out")
        contacts = list(contacts)
        if stacks is None:
            stacks = pairs_from_contacts(contacts)[1] if contacts else []
        self.stacks: list[BaseStack] = []
        sseen: set[frozenset] = set()
        for s in stacks:
            if {s.chain_a, s.chain_b} != set(self.chains) or s.key_a not in index or s.key_b not in index:
                continue
            sk = frozenset((s.key_a, s.key_b))
            if sk not in sseen:
                sseen.add(sk)
                self.stacks.append(s)
        self.hbonds: list[tuple[tuple[str, int, str], tuple[str, int, str], int]] = []
        bonds: dict[tuple, set[tuple[str, str]]] = {}
        for c in contacts:
            if c.type.value != "hbond":
                continue
            a, b = (c.chain_a, c.number_a, c.icode_a), (c.chain_b, c.number_b, c.icode_b)
            if a not in index or b not in index or frozenset((a, b)) in seen or a[0] == b[0]:
                continue
            atom_a, atom_b = _bond_atoms(c)
            if atom_a in _BACKBONE or atom_b in _BACKBONE:
                if index[a] > index[b]:                 # a row written from the second strand: the same bond
                    a, b, atom_a, atom_b = b, a, atom_b, atom_a
                bonds.setdefault((a, b), set()).add((atom_a, atom_b))
        self.hbonds = [(a, b, len(atoms)) for (a, b), atoms in bonds.items()]
        self.source = source
        self._laid: tuple[tuple[BasePair, ...], _Laid] | None = None
        #: The ligand beside the pairs, when one is named: its residue, its word, where its contacts came
        #: from, and its contacts to each nucleotide (nucleotide -> the records).
        self.ligand_key: Any = None
        self.ligand_name: str | None = None
        self.ligand_source: str | None = None
        self.ligand_contacts: dict[tuple[str, int, str], list[Any]] = {}
        if ligand is not None:
            self._take_ligand(ligand, ligand_contacts, ligand_name, ligand_smiles)

    def _take_ligand(self, ligand: Any, contacts: Any, name: str | None, smiles: str | None) -> None:
        from .interactions import Interaction, load_interactions

        key = self.structure.select_ligand(ligand)
        if (key.chain, key.number, key.icode) in self._index:
            raise BindviewError(f"the ligand {key} is a nucleotide of chain {key.chain}, one of the two strands")
        self.ligand_key = key
        self.ligand_name = name or self.structure.residue_name(key)
        if contacts is None:
            from .distance_table import ligand_contacts as plain_table

            records = [Interaction.from_mapping(r) for r in plain_table(self.structure, key, smiles)]
            self.ligand_source = "bindview's plain distance table"
        elif isinstance(contacts, (str, Path)):
            records = load_interactions(contacts)
            self.ligand_source = Path(contacts).name
        else:
            records = [r if isinstance(r, Interaction) else Interaction.from_mapping(r) for r in contacts]
            self.ligand_source = "as given"
        outside = 0
        for r in records:
            k = (r.chain, int(r.residue_number), r.insertion_code or "")
            if k in self._index:
                self.ligand_contacts.setdefault(k, []).append(r)
            else:
                outside += 1
        if outside:
            self.warnings.append(f"{outside} contact(s) of the ligand {self.ligand_name} with residues outside "
                                 f"chains {self.chains[0]} and {self.chains[1]} left out")
        if not self.ligand_contacts:
            self.warnings.append(f"the ligand {self.ligand_name} touches no nucleotide of chains {self.chains[0]} "
                                 f"and {self.chains[1]}: it is not drawn")

    def _is_nucleotide(self, key: Any) -> bool:
        return any(a.name in ("C1'", "C1*") for a in self.structure.residue_atoms(key))

    # -- what the figure says of itself ---------------------------------------------------------------
    @property
    def across(self) -> list[BasePair]:
        """The pairs from one strand to the other."""
        return [p for p in self.pairs if p.chain_a != p.chain_b]

    @property
    def within(self) -> list[BasePair]:
        """The pairs within a strand."""
        return [p for p in self.pairs if p.chain_a == p.chain_b]

    def source_line(self) -> str:
        """Where the pairs are from, in a sentence."""
        if self.source is not None:
            return self.source
        tags: list[str] = []
        for p in self.pairs:
            if p.source not in tags:
                tags.append(p.source)
        st: list[str] = []
        for s in self.stacks:
            if s.source not in st:
                st.append(s.source)
        both = bool(st) and st == tags                  # one source for both: said once
        head = "Pairs and stacks" if both else "Pairs"

        def said(of: list[str]) -> str:
            """The sources in words; a record that names none (one made by hand) is said to have none."""
            named = [SOURCES.get(t, t) for t in of if t]
            if "" in of:
                named.append("some given without a source" if named else "given without a source")
            return "; ".join(named)

        parts = [f"{head}: {said(tags)}." if tags else "No pairs given."]
        if st and not both:
            parts.append(f"Stacks: {said(st).replace(', without a family', '')}.")
        if self.hbonds:
            parts.append("Other hydrogen bonds across: bindview's plain distance table.")
        if self.ligand_contacts:
            parts.append(f"Contacts of {self.ligand_name}: {self.ligand_source}.")
        return " ".join(parts)

    def _layout(self) -> _Laid:
        """The layout of the pairs the figure has now: laid out once, and again when they have changed."""
        pairs = tuple(self.pairs)
        if self._laid is None or self._laid[0] != pairs:
            ij = [(self._index[p.key_a], self._index[p.key_b], p) for p in pairs]
            self._laid = (pairs, _lay_out(len(self.strands[0]), len(self.strands[1]), ij))
        return self._laid[1]

    # -- drawing ------------------------------------------------------------------------------------
    def render(self, theme: "str | Palette" = "light", *, metrics: Metrics | None = None, title: str | None = None,
               legend: bool = True, caption: bool = True, background: bool = True, colors: str = "standard",
               step: float = 24.0, figure_width_mm: float = 180.0, figure_height_mm: float = 240.0,
               min_font_pt: float = 5.0, interactive: bool = False, graph: "Graph | None" = None) -> RenderResult:
        """Draw the figure. ``theme``: ``"light"``, ``"dark"``, ``"auto"`` (both, following the reader) or a
        :class:`~bindview.Palette`; ``colors``: the colour scheme. ``step``: the distance of two neighbours
        of a strand, in px. ``legend``: the key under the drawing; ``caption``: the line of the source,
        which the result keeps in its ``notes`` also where it is left off the figure.

        The figure is drawn whatever its size. Where its smallest text would be under ``min_font_pt`` on
        a page of ``figure_width_mm`` by ``figure_height_mm``, the result's status is ``"unreadable"``
        and its warnings say how wide the figure is at the least. A figure with no pair, stack or
        hydrogen bond between the two strands is ``"empty"``.

        ``interactive``: the figure answers the pointer (:mod:`bindview.interactive`): pointing at a
        nucleotide, or at the ligand, keeps it, its pairs, stacks and contacts and the nucleotides at their
        other ends, and steps the rest back; a click keeps that. ``RenderResult.graph`` is the figure as
        nodes (a ``residue`` for each nucleotide, a ``ligand_group`` for the ligand) and contacts (each pair,
        stack, other hydrogen bond and contact of the ligand); ``graph``: such a graph, its nodes moved,
        to draw from instead of the layout."""
        css_vars = isinstance(theme, str) and theme.lower() == "auto"
        light, dark = scheme(colors)
        palette = light if css_vars else get_palette(theme, colors)
        palette = _with_strands(palette)
        dark_p = _with_strands(dark) if css_vars else None
        m = metrics or Metrics()
        laid = self._layout()
        fixed = None
        if graph is not None:
            laid, fixed = self._from_graph(laid, graph, step)
        scene, quality, data, live, drawn = self._scene(laid, m, step, title, legend, caption, figure_width_mm,
                                                        figure_height_mm, min_font_pt, fixed)
        notes = list(self.warnings)
        if not quality.readable:
            notes.append(f"at {figure_width_mm:.0f} × {figure_height_mm:.0f} mm its smallest text would be "
                         f"under {min_font_pt:g} pt: the figure is {quality.required_width_mm:.0f} mm wide at the "
                         f"least")
        status = "ok" if self.across or self.stacks or self.hbonds else "empty"
        if status == "empty":
            notes.append(f"no pair, stack or hydrogen bond between chains {self.chains[0]} and {self.chains[1]}")
        elif not quality.readable:
            status = "unreadable"               # drawn all the same: there is no smaller figure of two strands
        hooks = hooks_of_links(*live) if interactive else None
        svg = to_svg(scene, palette, background=background, css_vars=css_vars, dark_palette=dark_p,
                     interactive=hooks if hooks is not None else False)
        return RenderResult(status=status, svg=svg, figure="diagram", warnings=notes, quality=quality,
                            style="basepairs", scene=scene,
                            theme="auto" if css_vars else palette.name if isinstance(theme, Palette) else str(theme).lower(),
                            data=data, notes=[self.source_line()], graph=drawn)

    def _from_graph(self, laid: _Laid, graph: "Graph", step: float) -> "tuple[_Laid, np.ndarray | None]":
        """The layout as a graph of this figure has it: each nucleotide where its node is, and the ligand."""
        X = laid.X.copy()
        for i, nt in enumerate(self.strands[0] + self.strands[1]):
            node = graph.find(_node_id(nt.key))
            if node is None or node.x is None or node.y is None:
                raise BindviewError(f"the graph is not one of this figure: it has no place for {nt.name}")
            X[i] = (node.x / step, -node.y / step)
        fixed = None
        if self.ligand_key is not None:
            node = graph.find(f"ligand:{self.ligand_key}")
            if node is not None and node.x is not None and node.y is not None:
                fixed = np.array([node.x / step, -node.y / step])
        return _Laid(X, laid.side, laid.route, laid.shaping), fixed

    def render_svg(self, path: "str | Path | None" = None, theme: "str | Palette" = "light",
                   **options: Any) -> RenderResult:
        """Render and, when ``path`` is given, write the figure there."""
        result = self.render(theme=theme, **options)
        if path is not None and result.svg is not None:
            p = Path(path)
            try:
                p.write_text(result.svg, encoding="utf-8")
            except OSError as exc:
                raise RenderError(f"cannot write {p}: {exc}") from exc
            result.path = p
        return result

    def _scene(self, laid: _Laid, m: Metrics, step: float, title: str | None, legend: bool, caption: bool,
               width_mm: float, height_mm: float, min_pt: float, fixed: "np.ndarray | None" = None
               ) -> "tuple[Scene, Quality, dict[str, Any], tuple[dict[str, list[Group]], list], Graph]":
        nts = self.strands[0] + self.strands[1]
        side = laid.side
        r_nt = step * 0.42
        letter_font, number_font = m.residue_font, m.distance_font
        pad = step * 1.6
        lo, hi = laid.X.min(0), laid.X.max(0)
        lig = self._ligand_box(laid, nts, step, r_nt, letter_font, fixed)
        if lig is not None:
            lo = np.minimum(lo, lig[0] - lig[1])
            hi = np.maximum(hi, lig[0] + lig[1])
        top = pad + (m.title_font * 1.8 if title else 0.0)
        ox, oy = pad - lo[0] * step, top + hi[1] * step

        def at(i: int) -> np.ndarray:
            return np.array([ox + laid.X[i][0] * step, oy - laid.X[i][1] * step])
        pos = [at(i) for i in range(len(nts))]
        draw_w = (hi[0] - lo[0]) * step + 2 * pad
        draw_h = top + (hi[1] - lo[1]) * step + pad
        pale = {frozenset((self._index[p.key_a], self._index[p.key_b])) for p in self.pairs} - laid.shaping
        across_ids = {self._index[k] for p in self.across for k in (p.key_a, p.key_b)}
        items: list[Item] = []
        lines: list[tuple[np.ndarray, np.ndarray, str, tuple[int, int]]] = []
        parts: dict[str, list[Group]] = {}
        links: list[tuple[Group, tuple[str, ...]]] = []
        if title:
            items.append(Text.plain(pad, pad + m.title_font, title, m.title_font, "text", anchor="start", bold=True))

        def segment(i: int, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            d = _unit_vec(pos[j] - pos[i])
            return pos[i] + d * r_nt, pos[j] - d * r_nt, d

        for s in (0, 1):
            ids = [i for i in range(len(nts)) if side[i] == s]
            items.append(Group("strand", [Polyline(tuple((float(pos[i][0]), float(pos[i][1])) for i in ids),
                                                    Paint(stroke=f"strand_{s + 1}", width=2.2, opacity=0.55))],
                               data={"chain": self.chains[s]}))
        for a, b, n in self.hbonds:
            i, j = self._index[a], self._index[b]
            u, v, _ = segment(i, j)
            hb = Group("hbond", [_line(u, v, Paint(stroke="hbond", width=1.2, dash=(1.5, 2.5)))],
                       title=f"{nts[i].name} – {nts[j].name}: {n} hydrogen bond(s) with a sugar or phosphate atom")
            items.append(hb)
            links.append((hb, (f"n{i}", f"n{j}")))
            lines.append((pos[i], pos[j], "hbond", (i, j)))
        for st in self.stacks:
            i, j = self._index[st.key_a], self._index[st.key_b]
            u, v, _ = segment(i, j)
            sg = Group("stack", [_line(u, v, Paint(stroke="pi_stacking", width=1.8, dash=(0.1, 3.4), opacity=0.7))],
                       title=f"{nts[i].name} on {nts[j].name}: stacked")
            items.append(sg)
            links.append((sg, (f"n{i}", f"n{j}")))
            lines.append((pos[i], pos[j], "stack", (i, j)))
        kinds: set[str] = {"strands"}
        rungs = []
        for p in sorted(self.pairs, key=lambda q: q.chain_a != q.chain_b):          # within first, across on top
            i, j = self._index[p.key_a], self._index[p.key_b]
            is_across = p.chain_a != p.chain_b
            is_pale = frozenset((i, j)) in pale
            colour = "text" if is_across else "protein_bond"
            width = (1.3 if is_pale else 2.4) if is_across else (1.1 if is_pale else 1.6)
            u, v, d = segment(i, j)
            dash = (4.0, 3.0) if p.family is None else ()
            paint = Paint(stroke=colour, width=width, dash=dash)
            mid = (u + v) / 2
            arc = abs(i - j) == 1 and side[i] == side[j]
            if arc:
                # Neighbours of a strand: no room for a symbol between them. A small arc out of the strand,
                # away from what lies around them, with the symbols at its top.
                e = _rot(_unit_vec(pos[j] - pos[i]))
                centre = (pos[i] + pos[j]) / 2
                near = [q for q in pos if 0 < float(np.linalg.norm(q - centre)) < 2.5 * step]
                if near and float(np.dot(np.mean(near, axis=0) - centre, e)) > 0:
                    e = -e
                a0, a1 = pos[i] + e * r_nt, pos[j] + e * r_nt
                top = centre + e * (r_nt + step * 0.75)
                ctrl = 2 * top - (a0 + a1) / 2
                curve = [(1 - s) ** 2 * a0 + 2 * (1 - s) * s * ctrl + s ** 2 * a1 for s in np.linspace(0, 1, 13)]
                first: Item = Polyline(tuple((float(q[0]), float(q[1])) for q in curve), paint)
                mid = top
                lines.extend((curve[q], curve[q + 1], "within", (i, j)) for q in range(len(curve) - 1))
            else:
                first = _line(u, v, paint)
            g = Group("pair", [first],
                      data={"family": p.family or "none", "across": "yes" if is_across else "no"},
                      title=f"{nts[i].name} – {nts[j].name}: {p.family or 'family not annotated'}",
                      opacity=0.55 if is_pale else 1.0)
            canonical = p.family == "cWW" and {nts[i].letter, nts[j].letter} in ({"G", "C"}, {"A", "U"}, {"A", "T"})
            wobble = p.family == "cWW" and {nts[i].letter, nts[j].letter} == {"G", "U"}
            drawn_as = p.family if p.ordered or p.family is None else _family_name(p.family)
            g.extend(_symbols(drawn_as, canonical, wobble, mid, d, colour, step * 0.42))
            items.append(g)
            links.append((g, (f"n{i}", f"n{j}")))
            if is_pale:
                kinds.add("tertiary")
            elif is_across:
                kinds.add("across-none" if p.family is None else "across")
            else:
                kinds.add("within-none" if p.family is None else "within")
            if p.family is not None and not canonical:
                kinds.add("wobble" if wobble else f"{'sym' if p.ordered else 'symu'}:{_family_name(p.family)}")
            if not arc:
                lines.append((pos[i], pos[j], "across" if is_across else "within", (i, j)))
            if is_across:
                rungs.append((pos[i], pos[j], (i, j)))
        if self.stacks:
            kinds.add("stack")
        if self.hbonds:
            kinds.add("hbond")
        obstacles: list[tuple[float, float, float, float]] = []
        lig_at: np.ndarray | None = None
        if lig is not None:
            lig_at = np.array([ox + lig[0][0] * step, oy - lig[0][1] * step])
            half = lig[1] * step
            obstacles.append((float(lig_at[0] - half[0]), float(lig_at[1] - half[1]), float(lig_at[0] + half[0]),
                              float(lig_at[1] + half[1])))
            items.extend(self._ligand_items(nts, pos, lig_at, half, r_nt, letter_font, m, kinds, lines, parts,
                                            links))
        for i, nt in enumerate(nts):
            s = int(side[i])
            fill = f"strand_{s + 1}_fill" if i in across_ids else "background"
            dash = (2.0, 1.5) if nt.modified else ()
            if nt.modified:
                kinds.add("modified")
            mark = Group("nucleotide", [
                Circle(float(pos[i][0]), float(pos[i][1]), r_nt, Paint(stroke=f"strand_{s + 1}", fill=fill, width=1.5,
                                                                      dash=dash)),
                Text.plain(float(pos[i][0]), float(pos[i][1]) + letter_font * 0.36, nt.letter, letter_font, "text",
                           bold=True)],
                data={"chain": nt.chain, "number": f"{nt.number}{nt.icode}", "resname": nt.resname},
                title=nt.name)
            items.append(mark)
            parts[f"n{i}"] = [mark]
        labels, label_boxes = self._labels(nts, pos, side, across_ids, lines, r_nt, number_font, obstacles)
        items.extend(labels)

        # the counts the figure is judged by
        def seg_point(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
            ab = b - a
            t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0, 1))
            return float(np.linalg.norm(p - (a + t * ab)))
        crossings = sum(1 for x in range(len(rungs)) for y in range(x + 1, len(rungs))
                        if not set(rungs[x][2]) & set(rungs[y][2]) and _cross(rungs[x][0], rungs[x][1], rungs[y][0],
                                                                               rungs[y][1]))
        over = sum(1 for a, b, kind, ends in lines if kind in ("across", "within")
                   for k in range(len(pos)) if k not in ends and seg_point(pos[k], a, b) < r_nt * 0.95)
        marks_on_marks = sum(1 for x in range(len(pos)) for y in range(x + 1, len(pos))
                             if float(np.linalg.norm(pos[x] - pos[y])) < 2 * r_nt * 0.9)
        names_on = sum(1 for x in range(len(label_boxes)) for y in range(x + 1, len(label_boxes))
                       if _boxes_overlap(label_boxes[x], label_boxes[y]))
        names_on += sum(1 for bx in label_boxes for q in pos if _box_circle(bx, q, r_nt))
        backbone = [float(np.linalg.norm(pos[i + 1] - pos[i])) / step for i in range(len(pos) - 1)
                    if side[i] == side[i + 1]]

        # key and source under the drawing
        y = draw_h
        key_w = 0.0
        if legend:
            group, kh, kw = self._key(kinds, m, y, step)
            items.append(group)
            y += kh
            key_w = kw
        source = self.source_line()
        if caption:
            src = source if "hbond" in kinds or self.source is not None else source.replace(
                " Other hydrogen bonds across: bindview's plain distance table.", "")
            width_for_text = max(draw_w, key_w, 360.0) - 2 * 14
            for line in _wrap(src, m.caption_font, width_for_text):
                y += m.caption_font * 1.35
                items.append(Text.plain(14, y, line, m.caption_font, "text_muted", anchor="start"))
            y += m.caption_font * 0.9
        width = max(draw_w, key_w + 28, 360.0 if caption else 0.0)
        height = y + 10
        scene = Scene(width, height, [Group("basepairs", items, data={"chains": " ".join(self.chains)})],
                      title=title or f"Base pairs of chains {self.chains[0]} and {self.chains[1]}",
                      description=source)
        smallest = min(number_font, letter_font, m.legend_font if legend else 99.0, m.caption_font if caption else 99.0)
        px_per_mm = 96 / 25.4
        fit = min(1.0, width_mm * px_per_mm / width, height_mm * px_per_mm / height)
        min_font = smallest * 0.75
        q = Quality(width=width, height=height, bond_length=step, n_residues=len(nts),
                    n_interactions=len(self.across) + len(self.stacks) + len(self.hbonds), overlaps=marks_on_marks,
                    unplaced=0, line_crossings=crossings, bond_crossings=0,
                    max_stretch=max(backbone, default=1.0), min_font_pt=min_font,
                    required_width_mm=width / px_per_mm * (min_pt / min_font),
                    readable=min_font * fit >= min_pt - 1e-9, link_hits=over)
        data = {"layout": laid.route, "pairs_across": len(self.across), "pairs_within": len(self.within),
                "stacks": len(self.stacks), "hbonds": len(self.hbonds), "pale": len(pale),
                "names_on_names_or_marks": names_on,
                "kinds": sorted(kinds), "families": sorted({p.family or "none" for p in self.pairs})}
        if lig_at is not None:
            counts: dict[str, int] = {}
            for recs in self.ligand_contacts.values():
                for kind in dict.fromkeys(str(r.type) for r in recs):
                    counts[kind] = counts.get(kind, 0) + 1
            data["ligand"] = {"residue": str(self.ligand_key), "name": self.ligand_name,
                              "nucleotides": len(self.ligand_contacts), "lines": counts,
                              "lines_over_marks": sum(1 for a, b, kind, ends in lines if kind == "ligand"
                                                      for k in range(len(pos)) if k not in ends
                                                      and seg_point(pos[k], a, b) < r_nt * 0.95),
                              "at": [round(float(lig_at[0]), 2), round(float(lig_at[1]), 2)]}
        drawn = self._graph(nts, pos, lig_at, pale, step, scene, data)
        return scene, q, data, (parts, links), drawn

    def _graph(self, nts: list[_Nt], pos: list[np.ndarray], lig_at: "np.ndarray | None", pale: set[frozenset],
               step: float, scene: Scene, data: dict[str, Any]) -> "Graph":
        """The figure as nodes and contacts (:mod:`bindview.graph`), in the page's frame."""
        from .graph import Contact, Graph, Node

        across = {self._index[k] for p in self.across for k in (p.key_a, p.key_b)}
        found: dict[str, list[str]] = {}
        contacts: list[Contact] = []

        def add(c: Contact) -> None:
            contacts.append(c)
            found.setdefault(c.residue, []).append(c.id)
            if c.ligand is not None:
                found.setdefault(c.ligand, []).append(c.id)

        for n, p in enumerate(self.pairs):
            i, j = self._index[p.key_a], self._index[p.key_b]
            add(Contact(id=f"pair:{n}", type="base_pair", residue=_node_id(p.key_a), ligand=_node_id(p.key_b),
                        label=p.family, attrs={"family": p.family, "ordered": p.ordered,
                                               "across": p.chain_a != p.chain_b, "source": p.source,
                                               "pale": frozenset((i, j)) in pale}))
        for n, st in enumerate(self.stacks):
            add(Contact(id=f"stack:{n}", type="base_stack", residue=_node_id(st.key_a), ligand=_node_id(st.key_b),
                        attrs={"source": st.source}))
        for n, (a, b, count) in enumerate(self.hbonds):
            add(Contact(id=f"hbond:{n}", type="hbond", residue=_node_id(a), ligand=_node_id(b),
                        attrs={"bonds": count, "with": "a sugar or phosphate atom"}))
        lig_id = f"ligand:{self.ligand_key}"
        if lig_at is not None:
            for key, recs in self.ligand_contacts.items():
                for n, r in enumerate(recs):
                    add(Contact(id=f"{lig_id}:{_node_id(key)}:{n}", type=str(r.type), residue=_node_id(key),
                                ligand=lig_id, distance=None if r.distance is None else float(r.distance),
                                protein_atoms=tuple(a for a in (r.protein_atom,) if a),
                                attrs={"ligand_atom": r.ligand_atom} if r.ligand_atom else {}))
        nodes = [Node(id=_node_id(nt.key), kind="residue", label=f"{nt.letter}{nt.number}{nt.icode}",
                      x=float(pos[i][0]), y=float(pos[i][1]), contacts=tuple(found.get(_node_id(nt.key), ())),
                      attrs={"chain": nt.chain, "number": f"{nt.number}{nt.icode}", "resname": nt.resname,
                             "letter": nt.letter, "modified": nt.modified, "strand": int(nt.chain == self.chains[1]) + 1,
                             "paired_across": i in across})
                 for i, nt in enumerate(nts)]
        if lig_at is not None:
            nodes.append(Node(id=lig_id, kind="ligand_group", label=self.ligand_name or "", x=float(lig_at[0]),
                              y=float(lig_at[1]), contacts=tuple(found.get(lig_id, ())),
                              attrs={"residue": str(self.ligand_key), "contacts_from": self.ligand_source}))
        meta = {"basepairs": {"chains": list(self.chains), "layout": data["layout"], "source": self.source_line(),
                              "pairs_across": data["pairs_across"], "pairs_within": data["pairs_within"],
                              "stacks": data["stacks"], "hbonds": data["hbonds"]}}
        return Graph(view="basepairs", bond_length=step, nodes=nodes, contacts=contacts,
                     size=(scene.width, scene.height), meta=meta).validate()

    # -- the ligand beside the pairs ------------------------------------------------------------------
    def _ligand_box(self, laid: _Laid, nts: list[_Nt], step: float, r_nt: float, font: float,
                    fixed: "np.ndarray | None" = None) -> "tuple[np.ndarray, np.ndarray] | None":
        """Where the ligand's box goes and its half size, in steps; ``None`` without a ligand to draw.

        Near the nucleotides it touches and on none of them, its lines crossing as few lines of the figure
        and passing over as few nucleotides as there is room for: the first best of rings of places around
        the middle of those nucleotides, a quarter step apart, so that the same input gives the same place.
        """
        if not self.ligand_contacts:
            return None
        from .textmetrics import text_width

        X = laid.X
        half = np.array([(text_width(self.ligand_name or "", font, bold=True) + font * 1.4) / 2,
                         font * 0.95]) / step
        if fixed is not None:
            return fixed, half
        targets = [self._index[k] for k in self.ligand_contacts]
        segs = [(self._index[p.key_a], self._index[p.key_b]) for p in self.pairs]
        segs += [(self._index[st.key_a], self._index[st.key_b]) for st in self.stacks]
        segs += [(self._index[a], self._index[b]) for a, b, _ in self.hbonds]
        segs += [(i, i + 1) for i in range(len(nts) - 1) if laid.side[i] == laid.side[i + 1]]
        r = r_nt / step
        centre = X[targets].mean(0)
        best: tuple[float, np.ndarray] | None = None
        for ring in range(0, 25):
            n = 1 if ring == 0 else 8 * ring
            for k in range(n):
                ang = 2 * math.pi * k / n
                c = centre + ring * 0.25 * np.array([math.cos(ang), math.sin(ang)])
                gap = np.maximum(np.abs(X - c) - half, 0.0)
                if (np.hypot(gap[:, 0], gap[:, 1]) < r + 0.2).any():
                    continue                                    # on a nucleotide, or too close to one
                score = 0.0
                for t in targets:
                    score += float(np.linalg.norm(X[t] - c))
                    score += 2.0 * sum(1 for a, b in segs if t not in (a, b) and _cross(c, X[t], X[a], X[b]))
                    ab = X[t] - c
                    tt = np.clip(((X - c) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
                    d = np.linalg.norm(X - (c + tt[:, None] * ab), axis=1)
                    score += 2.0 * sum(1 for q in range(len(X)) if q != t and d[q] < r)
                if best is None or score < best[0] - 1e-9:
                    best = (score, c)
        if best is None:
            best = (0.0, np.array([centre[0], X[:, 1].max() + 2.0]))
        return best[1], half

    def _ligand_items(self, nts: list[_Nt], pos: list[np.ndarray], at: np.ndarray, half: np.ndarray, r_nt: float,
                      font: float, m: Metrics, kinds: set[str],
                      lines: list[tuple[np.ndarray, np.ndarray, str, tuple[int, int]]],
                      parts: dict[str, list[Group]], links: list[tuple[Group, tuple[str, ...]]]) -> list[Item]:
        """The ligand's box and its lines: one for each kind of contact to each nucleotide, side by side, in
        the line style of that kind in the library's other figures; the records of each in its title."""
        from .interactions import InteractionType as T
        from .styles.base import LINK_STYLES

        order = [T.SALTBRIDGE, T.HBOND, T.PI_CATION, T.PI_STACKING, T.METAL, T.HALOGEN, T.WATER_BRIDGE,
                 T.HYDROPHOBIC, T.COVALENT]
        out: list[Item] = []
        for key, recs in self.ligand_contacts.items():
            i = self._index[key]
            d = _unit_vec(pos[i] - at)
            # where the line leaves the box
            tx = half[0] / abs(d[0]) if abs(d[0]) > 1e-9 else math.inf
            ty = half[1] / abs(d[1]) if abs(d[1]) > 1e-9 else math.inf
            start, end = at + d * (min(tx, ty) + 2.0), pos[i] - d * r_nt
            present = [k for k in order if any(r.type is k for r in recs)]
            e = _rot(d)
            strokes: list[Item] = []
            words = []
            for n, kind in enumerate(present):
                style = LINK_STYLES[kind]
                off = e * (n - (len(present) - 1) / 2) * 3.4
                strokes.append(_line(start + off, end + off, Paint(stroke=style.token, width=m.interaction_width
                                                                 * style.width, dash=style.dash, cap=style.cap)))
                kinds.add(f"lig:{kind.value}")
                dists = sorted(float(r.distance) for r in recs if r.type is kind and r.distance is not None)
                words.append(kind.label.lower() + (f" ({', '.join(f'{x:.2f}' for x in dists)} Å)" if dists else ""))
            lines.append((at, pos[i], "ligand", (i, i)))
            line = Group("ligand-contact", strokes, data={"chain": key[0], "number": f"{key[1]}{key[2]}",
                                                          "kinds": " ".join(k.value for k in present)},
                         title=f"{self.ligand_name} – {nts[i].name}: {'; '.join(words)}")
            out.append(line)
            links.append((line, (f"n{i}", "g0")))
        kinds.add("ligand")
        w, h = 2 * half
        box = Group("ligand", [
            Rect(float(at[0] - half[0]), float(at[1] - half[1]), float(w), float(h),
                 Paint(stroke="ligand_bond", fill="background", width=1.6), rx=float(h / 2)),
            Text.plain(float(at[0]), float(at[1] + font * 0.36), self.ligand_name or "", font, "text", bold=True)],
            data={"residue": str(self.ligand_key)},
            title=f"{self.ligand_name} ({self.ligand_key}): touches {len(self.ligand_contacts)} nucleotide(s)")
        out.append(box)
        parts["g0"] = [box]
        return out

    def _labels(self, nts: list[_Nt], pos: list[np.ndarray], side: np.ndarray, across_ids: set[int],
                lines: list[tuple[np.ndarray, np.ndarray, str, tuple[int, int]]], r_nt: float, font: float,
                obstacles: Sequence[tuple[float, float, float, float]] = ()
                ) -> tuple[list[Item], list[tuple[float, float, float, float]]]:
        """The ends (5', 3') and the numbers: every fifth, the ends of a strand, the ends of each stretch that
        pairs across; each where it covers no mark, no number and as few lines as there is room for."""
        items: list[Item] = []
        boxes: list[tuple[float, float, float, float]] = []
        X = np.array(pos)
        normals = np.zeros_like(X)
        for i in range(len(X)):
            d = np.linalg.norm(X - X[i], axis=1)
            near = (d < 2.6 * r_nt / 0.42) & (d > 0)
            normals[i] = _unit_vec(X[i] - X[near].mean(0)) if near.any() else np.array([0.0, -1.0])
        for s in (0, 1):
            ids = [i for i in range(len(nts)) if side[i] == s]
            for end, other, word in ((ids[0], ids[1] if len(ids) > 1 else None, "5′"),
                                     (ids[-1], ids[-2] if len(ids) > 1 else None, "3′")):
                away = _unit_vec(X[end] - X[other]) if other is not None else np.array([1.0, 0.0])
                c = X[end] + away * (r_nt + font * 0.9)
                t = Text.plain(float(c[0]), float(c[1] + font * 0.35), word, font, f"strand_{s + 1}", bold=True)
                items.append(t)
                boxes.append(text_box(t))
        run_ends = set()
        for s in (0, 1):
            ids = [i for i in range(len(nts)) if side[i] == s]
            for a, b in zip([-1] + ids[:-1], ids):
                if b in across_ids and a not in across_ids:
                    run_ends.add(b)
            for a, b in zip(ids, ids[1:] + [-1]):
                if a in across_ids and b not in across_ids:
                    run_ends.add(a)

        def badness(box: tuple[float, float, float, float]) -> float:
            bad = 0.0
            centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            half = max(box[2] - box[0], box[3] - box[1]) / 2
            for q in X:
                if _box_circle(box, q, r_nt + 1.5):
                    bad += 100
            for bx in boxes:
                if _boxes_overlap(bx, box):
                    bad += 100
            for bx in obstacles:
                if _boxes_overlap(bx, box):
                    bad += 100
            for a, b, _, _ in lines:
                ab = b - a
                t = float(np.clip(np.dot(centre - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0, 1))
                if float(np.linalg.norm(centre - (a + t * ab))) < half * 0.8:
                    bad += 15
            return bad

        for i, nt in enumerate(nts):
            ids = [k for k in range(len(nts)) if side[k] == side[i]]
            if not (nt.number % 5 == 0 or i == ids[0] or i == ids[-1] or i in run_ends):
                continue
            word = f"{nt.number}{nt.icode}"
            probe = Text.plain(0.0, 0.0, word, font, "text_muted")
            b0 = text_box(probe)
            w, h = b0[2] - b0[0], b0[3] - b0[1]
            best: tuple[float, np.ndarray] | None = None
            for extra in (0.0, 0.35, 0.8, 1.3):
                for k in range(24):
                    ang = 2 * math.pi * k / 24
                    d = np.array([math.cos(ang), math.sin(ang)])
                    c = X[i] + d * (r_nt + 2 + extra * r_nt / 0.42 + abs(d[0]) * w / 2 + abs(d[1]) * h / 2)
                    box = (c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2)
                    score = badness(box) - 3 * float(np.dot(d, normals[i])) + 4 * extra
                    if best is None or score < best[0]:
                        best = (score, c)
                if best is not None and best[0] < 0:
                    break
            assert best is not None
            c = best[1]
            t = Text.plain(float(c[0]), float(c[1] + h / 2 - (b0[3])), word, font, "text_muted")
            items.append(t)
            boxes.append(text_box(t))
        return items, boxes

    def _key(self, kinds: set[str], m: Metrics, y0: float, step: float = 24.0) -> tuple[Group, float, float]:
        """The key under the drawing: a row for each kind of mark the figure has, its sample and its words.
        The symbol of a family is drawn at the size it has on a pair (``step``), and where that is more
        than a row and a sample hold at their usual size, the rows stand further apart and the samples
        are wider."""
        font = m.legend_font
        size = step * 0.42
        rows: list[str] = [k for k in ("strands", "across", "across-none", "within", "within-none", "tertiary",
                                       "wobble") if k in kinds]
        rows += sorted((k for k in kinds if k.startswith(("sym:", "symu:"))), key=lambda k: (k.split(":")[1], k))
        rows += [k for k in ("stack", "hbond", "modified") if k in kinds]
        rows += [k for k in ("ligand",) if k in kinds]
        rows += [f"lig:{t}" for t in ("saltbridge", "hbond", "pi_cation", "pi_stacking", "metal", "halogen",
                                      "water_bridge", "hydrophobic") if f"lig:{t}" in kinds]
        items: list[Item] = []
        y = y0 + font * 1.2
        x0, sample = 14.0, max(48.0, 2.4 * size)
        widest = 0.0
        for k in rows:
            mid = y - font * 0.35
            if k == "strands":
                items += [Circle(x0 + 10, mid, 8, Paint(stroke="strand_1", fill="strand_1_fill", width=1.5)),
                          Circle(x0 + 34, mid, 8, Paint(stroke="strand_2", fill="strand_2_fill", width=1.5))]
            elif k in ("across", "across-none"):
                items.append(Line(x0, mid, x0 + sample, mid, Paint(stroke="text", width=2.4,
                                                                   dash=(4.0, 3.0) if k == "across-none" else ())))
            elif k in ("within", "within-none"):
                items.append(Line(x0, mid, x0 + sample, mid, Paint(stroke="protein_bond", width=1.6,
                                                                   dash=(4.0, 3.0) if k == "within-none" else ())))
            elif k == "tertiary":
                items.append(Line(x0, mid, x0 + sample, mid, Paint(stroke="text", width=1.3, opacity=0.55)))
            elif k == "wobble" or k.startswith(("sym:", "symu:")):
                items.append(Line(x0, mid, x0 + sample, mid, Paint(stroke="text", width=1.6)))
                fam = "cWW" if k == "wobble" else k.split(":")[1]
                items += _symbols(fam, False, k == "wobble", np.array([x0 + sample / 2, mid]), np.array([1.0, 0.0]),
                                  "text", size)
            elif k == "stack":
                items.append(Line(x0, mid, x0 + sample, mid, Paint(stroke="pi_stacking", width=1.8, dash=(0.1, 3.4),
                                                                   opacity=0.7)))
            elif k == "hbond":
                items.append(Line(x0, mid, x0 + sample, mid, Paint(stroke="hbond", width=1.2, dash=(1.5, 2.5))))
            elif k == "modified":
                items.append(Circle(x0 + sample / 2, mid, 8, Paint(stroke="text_muted", fill="background", width=1.5,
                                                                   dash=(2.0, 1.5))))
            elif k == "ligand":
                items.append(Rect(x0 + 4, mid - 8, sample - 8, 16, Paint(stroke="ligand_bond", fill="background",
                                                                         width=1.6), rx=8))
            elif k.startswith("lig:"):
                from .interactions import InteractionType
                from .styles.base import LINK_STYLES

                style = LINK_STYLES[InteractionType(k[4:])]
                items.append(Line(x0, mid, x0 + sample, mid, Paint(stroke=style.token, width=m.interaction_width
                                                                   * style.width, dash=style.dash, cap=style.cap)))
            if k.startswith(("sym:", "symu:")):
                f = k.split(":")[1]
                text = (f"{f}: {'cis' if f[0] == 'c' else 'trans'}, {_EDGE[f[1]]} edge with {_EDGE[f[2]]} edge"
                        + (" (not G–C, A–U or G·U)" if f == "cWW" else ""))
                if f[1] != f[2]:
                    text += ("; each symbol at its own base" if k.startswith("sym:")
                             else "; which base uses which edge, the source does not say")
            elif k == "ligand":
                text = (f"the ligand, {self.ligand_name}: a line for each kind of contact with a nucleotide "
                        f"({self.ligand_source})")
            elif k.startswith("lig:"):
                from .interactions import InteractionType

                text = f"{InteractionType(k[4:]).label.lower()} with the ligand"
            else:
                text = _KEY[k].format(a=self.chains[0], b=self.chains[1])
            t = Text.plain(x0 + sample + 14, y, text, font, "text", anchor="start")
            items.append(t)
            widest = max(widest, text_box(t)[2])
            y += max(font * 2.0, size + 4.0)
        return Group("key", items), y - y0, widest


def _node_id(key: tuple[str, int, str]) -> str:
    """A nucleotide's id in the graph: ``A:8``, ``B:12A``."""
    return f"{key[0]}:{key[1]}{key[2]}"


def _with_strands(p: Palette) -> Palette:
    """A palette with the strand colours: its own where it has them, else those of the light or the dark
    palette, by how dark its background is."""
    if all(k in p.extra for k in _STRAND_LIGHT):
        return p
    bg = p.background.lstrip("#")
    dark = len(bg) == 6 and sum(int(bg[i:i + 2], 16) for i in (0, 2, 4)) < 3 * 128
    return p.with_(extra={**(_STRAND_DARK if dark else _STRAND_LIGHT), **p.extra})


def _symbols(family: str | None, canonical: bool, wobble: bool, m: np.ndarray, d: np.ndarray, colour: str,
             size: float) -> list[Item]:
    """The Leontis–Westhof symbol of a pair at ``m`` on its line (direction ``d``, from base a to base b):
    none for a canonical cis Watson–Crick pair or one without a family, a small dot for G·U, else the symbol
    of each base's edge (circle Watson–Crick, square Hoogsteen, triangle sugar edge), filled for cis, open
    for trans, one where both edges are the same."""
    if family is None or canonical:
        return []
    if wobble:
        return [Circle(float(m[0]), float(m[1]), 3.0, Paint(fill=colour))]
    cis, e1, e2 = family[0] == "c", family[1], family[2]
    fill = colour if cis else "background"
    if e1 == e2:
        return [_edge(e1, m, d, size, colour, fill)]
    return [_edge(e1, m - d * size * 0.55, d, size, colour, fill), _edge(e2, m + d * size * 0.55, d, size, colour, fill)]


def _edge(edge: str, c: np.ndarray, d: np.ndarray, s: float, colour: str, fill: str) -> Item:
    paint = Paint(stroke=colour, fill=fill, width=1.3, join="miter", cap="butt")
    if edge == "W":
        return Circle(float(c[0]), float(c[1]), s * 0.5, paint)
    e = np.array([-d[1], d[0]])
    if edge == "H":
        h = s * 0.45
        pts = [c + h * (a * d + b * e) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    else:
        pts = [c + s * 0.55 * v for v in (d, -0.5 * d + 0.87 * e, -0.5 * d - 0.87 * e)]
    return Polygon(tuple((float(p[0]), float(p[1])) for p in pts), paint)


def _line(u: np.ndarray, v: np.ndarray, paint: Paint) -> Line:
    return Line(float(u[0]), float(u[1]), float(v[0]), float(v[1]), paint)


def _cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _box_circle(bx: tuple[float, float, float, float], c: np.ndarray, r: float) -> bool:
    x = min(max(float(c[0]), bx[0]), bx[2])
    y = min(max(float(c[1]), bx[1]), bx[3])
    return math.hypot(float(c[0]) - x, float(c[1]) - y) < r


def _wrap(text: str, size: float, width: float) -> list[str]:
    from .textmetrics import text_width

    out, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and text_width(trial, size) > width:
            out.append(line)
            line = word
        else:
            line = trial
    return out + ([line] if line else [])
