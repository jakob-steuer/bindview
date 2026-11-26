"""What an input holds, and the figures that go with it: ``bindview report``.

Nobody says whether the input is a small molecule in a pocket, a peptide, an interface or an ensemble:
the report looks, draws the figure for what it finds, and prints the call that drew it. That call is the
manual mode: copy it and change it.

    bindview report INPUT [-o report.html] [--smiles NAME=SMILES ...] [--domain NAME=CHAIN:FROM-TO ...]
                          [--each] [--one-view] [--look]

    from bindview.report import survey

    found = survey("model.cif")          # what the input holds, and the figures that follow: nothing drawn
    found.draw()                         # runs the call of every figure
    page = found.html()                  # one page, each figure under its call

INPUT is a model (mmCIF or PDB, gzipped or not) or a folder that holds one. What lies beside the model is
used: a table of contacts, a file of predicted aligned errors, a ``meta.json``. A small molecule is drawn
from a table beside the model, else with PLIP where it is installed, else from the table of heavy atoms
(:func:`bindview.distance_table.ligand_contacts`), which is cruder than PLIP and says so. A peptide chain
and an interface are drawn from the plain distance table of the same module. A call names its files from
the folder of the input and is run there, so a page is the same wherever the report is run.

The rules:

R1  A folder's models are the mmCIF and PDB files directly in it. One: that is the complex. Several: they
    are a set (R1b, R1c).
R1b A folder that holds no model of its own and subfolders that hold one model each: the subfolders are the
    members of a set. A subfolder with several models is no member, and the page names it.
R1c A set of one composition is an ensemble: the first member by name is read as the complex, and all of
    them are counted (R9). A composition is the chains with their names, kinds and residue counts, and every
    other residue with its name, its number of copies, its heavy-atom formula and the file's own SMILES. Else,
    where the first protein chain of every member has the same sequence but for a tenth and every member
    has exactly one small molecule, the set is a series of ligands on one protein and is drawn as a
    comparison. Else it is refused, and the refusal names ``--each`` (R16).
R2  A residue is of a protein chain when it has N, CA and C and is bonded through them to a neighbour
    (N and C within 1.6 A), or when it is written as ATOM and named as an amino acid, however much of it
    is modelled. A peptide is a protein chain of 2 to 50 residues.
R2b A protein chain of 2 to 50 residues with a heavy atom within bonding distance of another protein
    chain (a disulfide, say) is a part of that protein and no peptide: listed, not drawn as a ligand.
R2c A nucleic chain is what the library takes for one: a chain without a residue of R2 and with two or
    more nucleotides joined to one another. A residue of such a chain that is written as ATOM and named as
    a nucleotide belongs to it, however much of it is modelled. From 3 to 50 nucleotides it is an
    oligonucleotide.
R3  Every other residue that is not water is a candidate ligand: a cap (ACE, NH2, NME, FOR within bonding
    distance of a chain), an ion (one heavy atom), an additive (a fixed list of what crystals and buffers
    bring), a sugar (a fixed list, bonded to the chain), or a small molecule (the rest). Residues of one
    name are copies of one ligand; the copy with the most heavy atoms stands for them.
R4  Drawn: every small molecule, the largest first, six at most; ions only where there is no small
    molecule and no chain that is drawn as a ligand (R7, R7b), three at most. The others are listed with
    the selection that would draw them.
R5  A table beside the model belongs to a candidate when every ligand atom it names is an atom of the
    candidate; of several copies it is the one within 6 A of the first residue the table names. The table
    that is named after the model (``<model>.interactions.tsv``) comes first, and one that is named after
    another model of the folder is not read. A table that names no ligand atom comes last and is taken
    where the model has one ligand only. Without a table, PLIP where it can be imported, with the nucleic
    acid as the receptor when no protein atom and a nucleotide lie within 6 A. Without either, the table
    of heavy atoms. A file beside the model that cannot be read as a table is named on the page.
R6  The SMILES: as handed in, else the file's own, else a ``meta.json`` beside the model that names the
    candidate, else none (the library then draws the connectivity and says so).
R7  A peptide is drawn as the ligand with the plain distance table, unless a table beside the model
    names atoms the way a chain ligand's are named (``63/OE1``): the first such table by name is taken,
    for every chain that is drawn as a ligand. Three at most, the shortest first.
R7b An oligonucleotide is drawn as the ligand, like a peptide, where the complex has a protein chain of
    more than 50 residues. Peptides and oligonucleotides together: three at most, the shortest first.
    Its partners are the residues of the protein chains: what it pairs with in another nucleic chain
    is in the interface of the two (R8b).
R8  An interface is drawn for every pair of protein chains of more than 50 residues for which the plain
    distance table has a row; three at most, those with the most rows first. A file of predicted
    aligned errors beside the model goes in: ``*confidences.json`` (not the summary), the AlphaFold database's
    ``*predicted_aligned_error*.json``, each gzipped or not, or ``pae_<model>.npz``.
R8b Interfaces are between chains of more than 50 residues, protein or nucleic, and between any two
    nucleic chains that touch, whatever their length. Three at most.
R9  An ensemble is the models of one file, or the members of a set of one composition (R1c). It is drawn
    for the largest small molecule from a table for every model (``model_NN.interactions.tsv`` by the
    model's number; for files of one folder ``<file>.interactions.tsv``; for members in folders of their
    own the table beside each, R5), else with PLIP for 25 models at most, else from the table of heavy
    atoms, model by model.
R10 Confidence is the library's business. The report says what it saw (a declared pLDDT, a file of
    predicted aligned errors, a summary of the model's scores) and hands on two things: the file of
    predicted aligned errors to an interface (R8, R12), and ``plddt_colors=True`` to one view (R15).
R11 The page has what the input holds first, then the figures in the order of these rules (small
    molecules or ions, chains as ligands, interfaces, each pair of nucleic chains followed by its base
    pairs, parts, the ensemble), then what was found and has no figure, with the reason.
R12 Where a ``meta.json`` beside the model names the parts of one chain (``chain``, and
    ``participants`` with ``name`` and ``residues``), or the caller does (``--domain
    NAME=CHAIN:FROM-TO``, at least two), each part gets a chain name of its own and the interface of
    every pair of parts that touches is drawn, with the PAE file where there is one. Three at most.
    A part that cannot be cut out of its chain is refused where the caller named it, and is a line of
    the page where a ``meta.json`` did.
R13 What is drawn from the plain distance table (a peptide, an oligonucleotide, an interface, two
    parts) asks for ring stacking (``stacking=True``), and the rows a pair is ranked by are counted
    with it.
R14 A figure whose call asks for the detailed style and comes back unreadable is drawn again by the
    same call in the schematic style. Where that one is drawn (ok or degraded), it is shown, its call is
    printed, and the page says what the detailed style said; else the detailed figure stays, with the
    library's summary in its place. A call that names no style (an ensemble, a series, base pairs) is
    left as it is.
R15 Views. Where a figure's call names a style and its figure is ok or degraded, the same object is drawn
    again, each time by a last line of its own that the page prints above it: in the other style (not
    where R14 left the detailed style for being unreadable); with pLDDT in AlphaFold's colours
    (``plddt_colors=True``), where the file declares a pLDDT; and as a figure that answers the pointer
    (``interactive=True``). The first figure is what it was. A view that comes back unreadable, empty
    or refused is not shown: the page names it and what the library said. ``views=False``
    (``--one-view``) draws the first figure alone. A call that names no style (an ensemble, a series)
    has no views.
R16 Each, for a set to screen. With ``--each`` (:func:`write_each`) the members of a folder (R1, R1b) are
    not read as one input: each is surveyed and drawn as an input of its own (R2 to R15, R17) and gets a
    page of its own, and an index lists them, one row a member, in the order of their names: its name and
    a link to its page; its figures, each with its status; for its first figure that is drawn, the
    contacts by kind, ipTM, pocket and ligand pLDDT where the model gives them, the buried fraction, and
    how many warnings the library gave. A member that cannot be read, that the library refuses, or that
    holds nothing to draw, is a row that says so, and so is a subfolder that is no member (R1b). The index
    is written as a page and as a CSV of the same rows. Without ``--each`` a folder whose members are
    neither one complex nor one series is refused, and the refusal names ``--each``.
R17 Base pairs. After each interface of two nucleic chains that R8b draws, the two chains are drawn again
    as their base pairs (:class:`~bindview.BasePairs`). The pairs: the PDB's own annotation where the model
    is an mmCIF that has pairs of these chains; else RNApolis's JSON beside the model
    (``<model>.rnapolis.json``, or ``basepairs.rnapolis.json``), with its stacks; else the stand-in of the
    plain distance table, across and within each strand, without a family, which the page says. The other
    hydrogen bonds across, and the stacks where the annotation has none, come from the plain table. The
    first small molecule of R4 with a heavy atom within 4.0 A of either chain stands beside the pairs, its
    contacts from the table R5 gives it, else from the plain table. Its one view (R15) is the figure that
    answers the pointer: it has no style to change and no pLDDT to colour.

Limits: a short subunit that is not bonded to its protein is taken for a peptide; two ligands of one
formula without a SMILES in their files are taken for one; the copies of one complex in a crystal touch
one another, and those contacts are drawn as interfaces like any other; a model named ``.mmcif.gz`` is
not asked for the PDB's own base pairs (R17).
"""

from __future__ import annotations

import contextlib
import csv
import difflib
import html
import importlib.util
import io
import json
import os
import urllib.parse
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from . import __version__
from .distance_table import interface_contacts
from .errors import BindviewError, StructureError
from .exposure import WATER_NAMES
from .interactions import NOT_GIVEN_OR_A_NAME, Interaction, load_interactions
from .near import pairs_within
from .residues import is_nucleotide, template_for
from .structure import Structure, read_models, read_structure

__all__ = ["survey", "write_each", "Survey", "Track", "View", "Row", "Found", "Chain", "Candidate", "Beside",
           "Member"]

MODEL_SUFFIXES = (".cif", ".mmcif", ".pdb", ".ent")
ADDITIVES = frozenset("GOL EDO PEG PGE PG4 1PE P6G 2PE MPD DMS ACT ACY FMT SO4 PO4 NO3 CIT TRS MES EPE BME DTT "
                      "IPA EOH IMD NH4 SCN BU3 TLA MLI".split())
SUGARS = frozenset("NAG NDG MAN BMA FUC FUL GAL GLA GLC BGC XYP SIA".split())
CAPS = frozenset({"ACE", "NH2", "NME", "FOR"})
PEPTIDE = (2, 50)                    # R2: residues of a chain that is a peptide
OLIGO = (3, 50)                      # R2c: nucleotides of a chain that is an oligonucleotide
SAME_PROTEIN = 0.9                   # R1c: the share of two sequences that has to agree
NEAR = 6.0                           # R5: angstrom
BESIDE_PAIRS = 4.0                   # R17: angstrom from a strand for a small molecule to stand beside the pairs
PEPTIDE_BOND = 1.6                   # R2: angstrom between N and C of two residues that are one chain
BOND, BOND_HEAVY = 1.9, 2.3          # R3: bonded to the chain; the longer where S, P or Se is one of the two
#: R16: the kinds of contact an index counts, in its order, and what it calls them.
KINDS = ("hbond", "saltbridge", "pi_stacking", "pi_cation", "halogen", "metal", "water_bridge", "hydrophobic")
KIND_NAMES = {"hbond": "H-bond", "saltbridge": "salt bridge", "pi_stacking": "\u03c0-stacking",
              "pi_cation": "\u03c0-cation", "halogen": "halogen bond", "metal": "metal", "water_bridge": "water bridge",
              "hydrophobic": "hydrophobic"}
MOST: Mapping[str, int] = {"small molecule": 6, "ion": 3, "peptide": 3, "interface": 3, "models": 25}
DETAILED, SCHEMATIC = 'style="detailed"', 'style="schematic"'                # R14: in the last line of a call
SUMMARY = "; a per-residue summary was drawn instead"         # R14: the library's words for what it replaces


# ---------------------------------------------------------------------------
# What the input holds
# ---------------------------------------------------------------------------

@dataclass
class Chain:
    name: str
    kind: str                        # protein | nucleic | "" (no polymer: it only carries other residues)
    residues: int
    sequence: tuple[str, ...] = ()   # the names of its residues, in the order of the file
    joined: str = ""                 # R2b: the protein chain a short protein chain is bonded to

    @property
    def peptide(self) -> bool:
        return self.kind == "protein" and PEPTIDE[0] <= self.residues <= PEPTIDE[1] and not self.joined

    @property
    def oligo(self) -> bool:
        return self.kind == "nucleic" and OLIGO[0] <= self.residues <= OLIGO[1]

    @property
    def what(self) -> str:
        """The chain in words, for the page."""
        if self.peptide:
            return "a peptide"
        if self.joined:
            return f"protein, joined to chain {self.joined}"
        if self.oligo:
            return "an oligonucleotide"
        return self.kind or "no polymer: it carries other residues"


@dataclass
class Candidate:
    """A residue outside the chains that could be the ligand."""

    name: str
    chain: str
    number: int
    heavy: int
    kind: str                        # ion | cap | additive | sugar | small molecule
    bonded: bool                     # an atom of it lies within bonding distance of a chain
    atoms: frozenset[str]
    xyz: np.ndarray
    copies: int = 1
    formula: str = ""                # its heavy atoms by element: C25F3N7O2

    @property
    def spec(self) -> str:
        """The residue as ``Diagram.from_structure(ligand=...)`` takes it."""
        return f"{self.chain}:{self.name}:{self.number}"


@dataclass
class Found:
    """What one model holds."""

    chains: list[Chain]
    candidates: list[Candidate]                  # one for each name, the largest first
    copies: dict[str, list[Candidate]]           # name -> every copy
    waters: int
    plddt: bool
    smiles_in_file: dict[str, str]


@dataclass
class Beside:
    """What lies beside a model."""

    #: ``(path, rows, the ligand atoms it names, whether those are a chain ligand's names)``
    tables: list[tuple[Path, list[Interaction], set[str], bool]] = field(default_factory=list)
    #: The ``.tsv`` and ``.csv`` files that are no table of contacts, each with the reason: the page names them.
    not_tables: list[tuple[Path, str]] = field(default_factory=list)
    pae: Path | None = None
    summary: Path | None = None
    meta: dict[str, Any] | None = None


@dataclass
class Member:
    """One complex of a set: a ligand of a series."""

    name: str
    model: Path
    ligand: Candidate
    table: Path | None               # the table of contacts beside the model that is the ligand's
    smiles: str | None               # to hand in; None where the file has its own, or there is none
    smiles_from: str


@dataclass
class View:
    """R15: the object of a figure drawn once more, by a last line of its own."""

    title: str
    line: str                        # the last line that draws it, run where the figure's call ran
    svg: str = ""
    status: str = ""
    said: list[str] = field(default_factory=list)


@dataclass
class Track:
    """One figure of the report: what was seen, and the call that draws it."""

    kind: str                        # ligand | ion | peptide | oligonucleotide | interface | basepairs | ensemble | series | parts
    title: str
    why: str                         # what was seen that makes it this figure
    code: str                        # the call, as source: it is run, and printed
    svg: str = ""
    status: str = ""
    said: list[str] = field(default_factory=list)
    schematic: bool = False          # R14: drawn in the schematic style because the detailed one is unreadable
    views: list[View] = field(default_factory=list)       # R15
    #: R16: what an index says of the figure: contacts by kind, confidence, buried fraction, warnings.
    facts: dict[str, Any] = field(default_factory=dict)


def _stem(model: Path) -> str:
    name = model.name[:-3] if model.name.endswith(".gz") else model.name
    return name.rsplit(".", 1)[0]


def look(st: Structure) -> Found:
    """R2 and R3: the chains of a structure, and what else it holds."""
    keys = st.residues()
    names_of = {k: {a.name for a in st.residue_atoms(k)} for k in keys}
    whole = [k for k in keys if {"N", "CA", "C"} <= names_of[k]]
    backbone = set()
    if len(whole) > 1:
        def at(k: Any, name: str) -> tuple[float, float, float]:
            atom = st.atom_by_name(k, name)
            assert atom is not None                      # a residue of ``whole`` has N, CA and C
            return atom.xyz

        n_at = np.array([at(k, "N") for k in whole], dtype=float)
        c_at = np.array([at(k, "C") for k in whole], dtype=float)
        of_n, of_c, _ = pairs_within(n_at, c_at, PEPTIDE_BOND)
        joined = of_n != of_c                            # (a residue's own N and C are no peptide bond)
        backbone = {whole[i] for i in set(of_n[joined].tolist()) | set(of_c[joined].tolist())}
    for k in keys:
        named = template_for(st.residue_name(k))
        if named is not None and named.kind == "amino" and not any(a.hetero for a in st.residue_atoms(k)):
            backbone.add(k)
    order = list(dict.fromkeys(k.chain for k in keys))
    chains: list[Chain] = []
    in_chain: set[Any] = set()
    of_chain: dict[str, list[Any]] = {}
    for c in order:
        own = [k for k in keys if k.chain == c]
        protein = [k for k in own if k in backbone]
        nucleic = [] if protein else st._nucleotides(c)          # R2c: nucleotides joined to one another
        if len(nucleic) >= 2:                                    # and those of which little is modelled
            joined = set(nucleic)
            nucleic = [k for k in own if k in joined or (is_nucleotide(st.residue_name(k))
                                                         and not any(a.hetero for a in st.residue_atoms(k)))]
        if protein:
            chains.append(Chain(c, "protein", len(protein), tuple(st.residue_name(k) for k in protein)))
            in_chain.update(protein)
            of_chain[c] = protein
        elif len(nucleic) >= 2:
            chains.append(Chain(c, "nucleic", len(nucleic), tuple(st.residue_name(k) for k in nucleic)))
            in_chain.update(nucleic)
        else:
            chains.append(Chain(c, "", 0))

    def heavy_of(k: Any) -> list[Any]:
        return [a for a in st.residue_atoms(k) if a.element not in ("H", "D")]

    # R2b: a short protein chain that is bonded to another protein chain is a part of that protein.
    for chain in chains:
        if chain.kind != "protein" or not PEPTIDE[0] <= chain.residues <= PEPTIDE[1]:
            continue
        mine = [a for k in of_chain[chain.name] for a in heavy_of(k)]
        here = np.array([a.xyz for a in mine], dtype=float)
        big = np.array([a.element.upper() in ("S", "P", "SE") for a in mine], dtype=bool)
        low, high = here.min(axis=0) - BOND_HEAVY, here.max(axis=0) + BOND_HEAVY
        for other in chains:
            if other.kind != "protein" or other.name == chain.name:
                continue
            theirs = [a for k in of_chain[other.name] for a in heavy_of(k)
                      if low[0] <= a.x <= high[0] and low[1] <= a.y <= high[1] and low[2] <= a.z <= high[2]]
            if not theirs:
                continue
            d = np.linalg.norm(here[:, None, :] - np.array([a.xyz for a in theirs], dtype=float)[None, :, :], axis=2)
            big2 = np.array([a.element.upper() in ("S", "P", "SE") for a in theirs], dtype=bool)
            if (d < np.where(big[:, None] | big2[None, :], BOND_HEAVY, BOND)).any():
                chain.joined = other.name
                break

    chain_atoms = [a for k in keys if k in in_chain for a in heavy_of(k)]
    chain_xyz = np.array([a.xyz for a in chain_atoms], dtype=float).reshape(-1, 3)
    chain_big = np.array([a.element.upper() in ("S", "P", "SE") for a in chain_atoms], dtype=bool)

    def is_bonded(atoms: list[Any]) -> bool:
        if not len(chain_xyz):
            return False
        d = np.linalg.norm(np.array([a.xyz for a in atoms], dtype=float)[:, None, :] - chain_xyz[None, :, :], axis=2)
        big = np.array([a.element.upper() in ("S", "P", "SE") for a in atoms], dtype=bool)
        return bool((d < np.where(big[:, None] | chain_big[None, :], BOND_HEAVY, BOND)).any())

    copies: dict[str, list[Candidate]] = {}
    waters = 0
    for k in keys:
        name = st.residue_name(k)
        if name.upper() in WATER_NAMES:
            waters += 1
            continue
        if k in in_chain:
            continue
        heavy = heavy_of(k)
        if not heavy:
            continue
        bonded = is_bonded(heavy)
        kind = ("cap" if name in CAPS and bonded else "ion" if len(heavy) == 1       # (the cap NH2 is one atom too)
                else "additive" if name in ADDITIVES else "sugar" if name in SUGARS and bonded
                else "small molecule")
        elements = Counter(a.element.capitalize() for a in heavy)
        copies.setdefault(name, []).append(Candidate(
            name, k.chain, k.number, len(heavy), kind, bonded, frozenset(a.name for a in heavy),
            np.array([a.xyz for a in heavy], dtype=float),
            formula="".join(f"{e}{n if n > 1 else ''}" for e, n in sorted(elements.items()))))
    candidates = []
    for group in copies.values():
        first = max(group, key=lambda c: c.heavy)            # the first in the file among equals
        first.copies = len(group)
        candidates.append(first)
    candidates.sort(key=lambda c: -c.heavy)
    return Found(chains, candidates, copies, waters, bool(st.declares_plddt), dict(st.chem_comp_smiles))


def _composition(found: Found) -> tuple[Any, ...]:
    """R1c: what a model is made of. Two co-folds name every ligand alike, so a residue outside the chains
    counts with its formula and with the SMILES its file gives it."""
    return (tuple((c.name, c.kind, c.residues) for c in found.chains),
            tuple(sorted((name, len(group), group[0].formula, found.smiles_in_file.get(name, ""))
                         for name, group in found.copies.items())))


def _is_model(p: Path) -> bool:
    name = p.name[:-3] if p.name.endswith(".gz") else p.name
    return p.is_file() and name.lower().endswith(MODEL_SUFFIXES)


def _members(path: Path) -> list[tuple[str, Path]]:
    """R1 and R1b: the models of an input, each with the name it has in a set."""
    if not path.is_dir():
        return [(_stem(path), path)] if _is_model(path) else []
    own = sorted(p for p in path.iterdir() if _is_model(p))
    if own:
        return [(_stem(p), p) for p in own]
    out = []
    for sub in sorted(p for p in path.iterdir() if p.is_dir()):
        models = sorted(p for p in sub.iterdir() if _is_model(p))
        if len(models) == 1:
            out.append((sub.name, models[0]))
    return out


def _passed_over(path: Path) -> list[tuple[Path, list[str]]]:
    """R1b: the subfolders of a set that are no member of it, each with the models it holds: more than one."""
    if not path.is_dir() or any(_is_model(p) for p in path.iterdir()):
        return []
    out = []
    for sub in sorted(p for p in path.iterdir() if p.is_dir()):
        models = sorted(p.name for p in sub.iterdir() if _is_model(p))
        if len(models) > 1:
            out.append((sub, models))
    return out


_ONE_MODEL = "a member of a set is a folder with one model: name this folder by itself to have it read"


def _named(models: Sequence[str]) -> str:
    return ", ".join(models[:4]) + (" and more" if len(models) > 4 else "")


def _one_protein(looks: Sequence[Found]) -> bool:
    """R1c: do the first protein chains of all members have one sequence, but for a tenth?"""
    firsts = [next((c for c in f.chains if c.kind == "protein"), None) for f in looks]
    if any(c is None for c in firsts):
        return False
    ref = firsts[0].sequence if firsts[0] is not None else ()
    return all(c is not None and difflib.SequenceMatcher(None, ref, c.sequence, autojunk=False).ratio() >= SAME_PROTEIN
               for c in firsts[1:])


def _beside(model: Path) -> Beside:
    out = Beside()
    # R5: a table that is named after a model is that model's: its own comes first, another model's is not read.
    own = f"{_stem(model)}.interactions.tsv"
    others = {f"{_stem(p)}.interactions.tsv" for p in model.parent.iterdir() if p != model and _is_model(p)} - {own}
    for p in sorted(model.parent.iterdir(), key=lambda p: (p.name != own, p)):
        low = p.name.lower()
        if low.endswith((".tsv", ".csv")):
            if p.name in others:
                continue
            try:
                rows = load_interactions(p)
            except BindviewError as exc:                    # (a spreadsheet's export in another encoding, say)
                why = str(exc)
                for lead in (f"cannot read {p}: ", f"{p}: "):     # the reader names the file: so does the page
                    why = why.removeprefix(lead)
                out.not_tables.append((p, why.replace(str(p), p.name)))
                continue
            # (NA can stand for a name that is not given: it does not tell whose table this is.)
            named = {n for r in rows for n in ([r.ligand_atom] if r.ligand_atom not in (None, NOT_GIVEN_OR_A_NAME) else [])
                     + list(r.ligand_atoms or ())}
            out.tables.append((p, rows, named, any("/" in n for n in named)))
        elif low.endswith(("summary_confidences.json", "summary_confidences.json.gz")):
            out.summary = out.summary or p
        elif (low.endswith(("confidences.json", "confidences.json.gz")) or low == f"pae_{_stem(model).lower()}.npz"
              or ("predicted_aligned_error" in low and low.endswith((".json", ".json.gz")))):
            out.pae = out.pae or p                          # R8
        elif low == "meta.json":
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(meta, dict):
                out.meta = meta
    return out


def plip_here() -> bool:
    """Can PLIP be imported where this runs? It is asked, and not imported."""
    return importlib.util.find_spec("plip") is not None


# ---------------------------------------------------------------------------
# The figures that follow
# ---------------------------------------------------------------------------

#: The folder of the input that is being surveyed: the printed calls name their files from it, and are run in it.
_FOLDER: "ContextVar[Path | None]" = ContextVar("the folder of the input", default=None)


def _shown(path: Path) -> str:
    """A path as a printed call names it: from the folder of the input, in which the call is run, and never
    from above it (a file that does not lie in that folder has its bare name)."""
    full, folder = Path(os.path.abspath(path)), _FOLDER.get()
    if folder is not None and (full == folder or folder in full.parents):
        return full.relative_to(folder).as_posix()
    return full.name


@contextlib.contextmanager
def _run_in(folder: Path) -> "Iterator[None]":
    """Inside, the working directory is ``folder``; after, it is what it was."""
    before = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(before)


def _q(text: object) -> str:
    """Words as a Python string in a printed call. Whatever a file calls a chain or a residue, and
    whatever a path or a SMILES holds, it stays inside the quotes."""
    return json.dumps(str(text), ensure_ascii=False)


def _table_of(cand: Candidate, found: Found, st: Structure, by: Beside) -> tuple[Candidate, Path | None]:
    """R5: the table beside the model that is this candidate's, and the copy it is about. A table that names no
    atom of a ligand says nothing of whose it is: it comes after those that do, and is taken where the model has
    one ligand only."""
    alone = sum(c.kind in ("small molecule", "ion") for c in found.candidates) == 1
    for path, rows, named, chain_names in sorted(by.tables, key=lambda table: not table[2]):
        if chain_names or not named <= cand.atoms or not (named or alone):
            continue
        if rows and len(found.copies[cand.name]) > 1:
            first = rows[0]
            # (NA for the chain is a chain of that name where the model has one, and no chain given where not.)
            chain = first.chain if any(k.chain == first.chain for k in st.residues()) else None
            key = st.find_residue(chain, first.residue_number, first.insertion_code)
            if key is not None:
                at = np.array([a.xyz for a in st.residue_atoms(key)], dtype=float)
                for copy in found.copies[cand.name]:
                    if np.linalg.norm(copy.xyz[:, None, :] - at[None, :, :], axis=2).min() <= NEAR:
                        copy.copies = cand.copies
                        return copy, path
        return cand, path
    return cand, None


def _receptor(cand: Candidate, st: Structure, found: Found) -> str:
    """R5: what PLIP is to take for the receptor of a candidate."""
    kinds = {c.name: c.kind for c in found.chains}
    near = set()
    for a in st.atoms:
        if a.element in ("H", "D") or a.resname.upper() in WATER_NAMES or kinds.get(a.chain) not in ("protein", "nucleic"):
            continue
        if (a.chain, a.resnum) == (cand.chain, cand.number):
            continue
        if np.linalg.norm(cand.xyz - np.array(a.xyz), axis=1).min() <= NEAR:
            near.add(kinds[a.chain])
    return "nucleic" if near == {"nucleic"} else "protein"


def _smiles(cand: Candidate, found: Found, by: Beside, given: Mapping[str, str]) -> tuple[str | None, str]:
    """R6: the SMILES to hand in (None: nothing to hand in), and where it is from, in words."""
    if cand.name in given:
        return given[cand.name], "handed in"
    if cand.name in found.smiles_in_file:
        return None, "in the file itself"
    ligand = (by.meta or {}).get("ligand")
    named = {cand.name, f"{cand.chain}:{cand.number}", f"{cand.chain}:{cand.name}:{cand.number}"}
    if isinstance(ligand, dict) and ligand.get("residue_name_in_cif") in named and ligand.get("smiles"):
        return str(ligand["smiles"]), "from meta.json beside the model"
    return None, "none: the library draws the connectivity and says so"


def _ligand_track(cand: Candidate, model: Path, st: Structure, found: Found, by: Beside,
                  given: Mapping[str, str]) -> Track:
    cand, table = _table_of(cand, found, st, by)
    smiles, where = _smiles(cand, found, by, given)
    kind = "ion" if cand.kind == "ion" else "ligand"
    what = "an ion" if cand.kind == "ion" else f"a {cand.kind}"
    seen = (f"{cand.name} on chain {cand.chain}: {what} of {cand.heavy} heavy atom{'s' if cand.heavy != 1 else ''}"
            + (f", {cand.copies} copies in the file" if cand.copies > 1 else "")
            + (", within bonding distance of the chain" if cand.bonded else "") + f". SMILES: {where}.")
    extra = f",\n                             smiles={_q(smiles)}" if smiles else ""
    if table is not None:
        names_atoms = any(named for path, _, named, _ in by.tables if path == table)
        seen += (f" Contacts: the table {table.name} beside the model, which names its atoms." if names_atoms else
                 f" Contacts: the table {table.name} beside the model, which names no atom of a ligand: it is taken "
                 "for this one's, the only ligand of the model.")
        code = ("from bindview import Diagram, load_interactions\n\n"
                f"dia = Diagram.from_structure({_q(_shown(model))}, ligand={_q(cand.spec)}{extra},\n"
                f"                             interactions=load_interactions({_q(_shown(table))}))\n"
                'figure = dia.render(style="detailed")')
        return Track(kind, f"{cand.name} in its pocket", seen, code)
    receptor = _receptor(cand, st, found)
    said = "" if receptor == "protein" else ', receptor="nucleic"'
    code = ("from bindview import Diagram\n"
            "from bindview.detect import detect_interactions        # PLIP, which is optional and GPL\n\n"
            f"model = {_q(_shown(model))}\n"
            f"dia = Diagram.from_structure(model, ligand={_q(cand.spec)}{extra},\n"
            f"                             interactions=detect_interactions(model, {_q(cand.spec)}{said}))\n"
            'figure = dia.render(style="detailed")')
    if plip_here():
        seen += (" Contacts: no table beside the model, so PLIP finds them"
                 + (", with the nucleic acid as the receptor." if said else "."))
        return Track(kind, f"{cand.name} in its pocket", seen, code)
    seen += (" Contacts: no table beside the model and no PLIP where this ran, so they are the table of heavy "
             "atoms: no hydrogen is placed, a hydrogen bond has no angle, and there is no water bridge.")
    first = f"smiles = {_q(smiles)}\n" if smiles else ""
    hand = ", smiles=smiles" if smiles else ""
    code = ("from bindview import Diagram, read_structure\n"
            "from bindview.distance_table import ligand_contacts      # heavy atoms only: cruder than PLIP\n\n"
            f"model = read_structure({_q(_shown(model))})\n" + first
            + f"dia = Diagram.from_structure(model, ligand={_q(cand.spec)}{hand},\n"
            f"                             interactions=ligand_contacts(model, {_q(cand.spec)}{', smiles' if smiles else ''}))\n"
            'figure = dia.render(style="detailed")')
    return Track(kind, f"{cand.name} in its pocket", seen, code)


def _peptide_track(chain: Chain, model: Path, by: Beside, proteins: Sequence[str] = ()) -> Track:
    table = next((path for path, _, _, chain_names in by.tables if chain_names), None)
    if chain.kind == "nucleic":
        seen = (f"Chain {chain.name} is a nucleic chain of {chain.residues} nucleotides beside a protein: an "
                "oligonucleotide, drawn as the ligand.")
    else:
        seen = f"Chain {chain.name} is a protein chain of {chain.residues} residues: a peptide, drawn as the ligand."
    if table is not None:
        seen += f" Contacts: the table {table.name} beside the model, which names atoms residue by residue."
        code = ("from bindview import Diagram, load_interactions\n\n"
                f"dia = Diagram.from_structure({_q(_shown(model))}, {_q('chain:' + chain.name)},\n"
                f"                             interactions=load_interactions({_q(_shown(table))}))\n"
                'figure = dia.render(style="detailed")')
    else:
        seen += " Contacts: the plain distance table with ring stacking, which is no interaction analysis."
        partners = ""
        if chain.kind == "nucleic" and proteins:                 # R7b
            seen += (" Its partners are the residues of the protein: what the chain pairs with in another "
                     "nucleic chain is in their interface.")
            partners = f", partners=[{', '.join(_q(p) for p in proteins)}]"
        code = ("from bindview import Diagram, read_structure\n"
                "from bindview.distance_table import contact_table      # distances only: no interaction analysis\n\n"
                f"model = read_structure({_q(_shown(model))})\n"
                f"dia = Diagram.from_structure(model, {_q('chain:' + chain.name)},\n"
                f"                             interactions=contact_table(model, {_q(chain.name)}{partners}, stacking=True))\n"
                'figure = dia.render(style="detailed")')
    if chain.kind == "nucleic":
        return Track("oligonucleotide", f"The oligonucleotide of chain {chain.name}", seen, code)
    return Track("peptide", f"The peptide of chain {chain.name}", seen, code)


def _interface_track(a: Chain, b: Chain, rows: int, model: Path, by: Beside) -> Track:
    unit = {"protein": "residues", "nucleic": "nucleotides"}
    seen = (f"Chains {a.name} ({a.kind}, {a.residues} {unit[a.kind]}) and {b.name} ({b.kind}, {b.residues} "
            f"{unit[b.kind]}) touch: the plain distance table, with ring stacking, has {rows} rows for the pair.")
    pae = ""
    if by.pae is not None:
        seen += f" Predicted aligned errors: {by.pae.name} beside the model."
        pae = f",\n                 pae={_q(_shown(by.pae))}"
    code = ("from bindview import Interface, read_structure\n"
            "from bindview.distance_table import interface_contacts      # distances only: no interaction analysis\n\n"
            f"model = read_structure({_q(_shown(model))})\n"
            f"face = Interface(model, [{_q(a.name)}, {_q(b.name)}], "
            f"interface_contacts(model, [{_q(a.name)}], [{_q(b.name)}], stacking=True){pae})\n"
            'figure = face.render(style="detailed")')
    return Track("interface", f"The interface of chains {a.name} and {b.name}", seen, code)


def _basepairs_track(a: Chain, b: Chain, model: Path, st: Structure, found: Found, by: Beside,
                     small: Sequence[Candidate], given: Mapping[str, str]) -> Track:
    """R17: two nucleic chains as their base pairs, with a small molecule beside them where one is."""
    from .basepairs import read_mmcif_pairs

    pair = f"[{_q(a.name)}, {_q(b.name)}]"
    head = ["BasePairs", "read_structure"]
    lines = [f"model = read_structure({_q(_shown(model))})",
             f"contacts = interface_contacts(model, [{_q(a.name)}], [{_q(b.name)}], stacking=True)"]
    own = ([p for p in read_mmcif_pairs(model) if {p.chain_a, p.chain_b} <= {a.name, b.name}]
           if model.name.lower().endswith((".cif", ".cif.gz", ".mmcif")) else [])
    beside = [f for f in (model.parent / f"{_stem(model)}.rnapolis.json", model.parent / "basepairs.rnapolis.json")
              if f.exists()]
    stacks = ""
    if own:
        head.append("read_mmcif_pairs")
        lines.append(f"found = read_mmcif_pairs({_q(_shown(model))})")
        seen = f"Pairs: the PDB's own annotation in {model.name} ({len(own)} pairs of these chains)."
    elif beside:
        head += ["read_rnapolis_pairs", "read_rnapolis_stacks"]
        lines += [f"annotation = {_q(_shown(beside[0]))}", "found = read_rnapolis_pairs(annotation)"]
        stacks = ", stacks=read_rnapolis_stacks(annotation)"
        seen = f"Pairs and stacks: RNApolis's annotation {beside[0].name} beside the model."
    else:
        head += ["pairs_from_contacts", "pairs_within_from_table"]
        lines.append(f"found = (pairs_from_contacts(contacts)[0] + pairs_within_from_table(model, {_q(a.name)})\n"
                     f"         + pairs_within_from_table(model, {_q(b.name)}))")
        seen = ("No annotation of the pairs beside the model (an mmCIF with the PDB's, or RNApolis's JSON): the "
                "pairs are the plain distance table's stand-in, two or more hydrogen bonds from base to base, "
                "without a family.")
    lig = ""
    strands = np.array([x.xyz for x in st.atoms if x.chain in (a.name, b.name) and x.element not in ("H", "D")],
                       dtype=float).reshape(-1, 3)
    for cand in small:
        if len(strands) and np.linalg.norm(cand.xyz[:, None, :] - strands[None, :, :], axis=2).min() <= BESIDE_PAIRS:
            cand, table = _table_of(cand, found, st, by)
            if table is not None:
                lig = f", ligand={_q(cand.spec)},\n                  ligand_contacts={_q(_shown(table))}"
                seen += f" {cand.name} lies beside them, with its contacts from {table.name}."
            else:
                smiles, _ = _smiles(cand, found, by, given)
                lig = f", ligand={_q(cand.spec)}" + (f",\n                  ligand_smiles={_q(smiles)}" if smiles else "")
                seen += f" {cand.name} lies beside them, with its contacts from the plain distance table."
            break
    code = (f"from bindview import {', '.join(head)}\n"
            "from bindview.distance_table import interface_contacts      # distances only: no interaction analysis\n\n"
            + "\n".join(lines) + "\n"
            + f"pairs = BasePairs(model, {pair}, found, contacts=contacts{stacks}{lig})\n"
            "figure = pairs.render()")
    return Track("basepairs", f"The base pairs of chains {a.name} and {b.name}", seen, code)


def _ensemble_track(cand: Candidate, models: Sequence[Path], numbers: Sequence[int | None], by: Beside, found: Found,
                    given: Mapping[str, str], own: "Sequence[Path | None] | None" = None) -> Track:
    """``own``: for the members of a set that lie in folders of their own, the table beside each."""
    smiles, where = _smiles(cand, found, by, given)
    many = len(numbers) if len(models) == 1 else len(models)
    of = f"the {many} models of {models[0].name}" if len(models) == 1 else f"{many} models of one complex"
    seen = f"{of[:1].upper() + of[1:]}: an ensemble, counted for {cand.name}. SMILES: {where}."
    folder = models[0].parent
    wanted: list[Path]
    if len(models) == 1:
        wanted = [folder / f"model_{n:02d}.interactions.tsv" for n in numbers if n is not None]
        reads = (f"models = read_models({_q(_shown(models[0]))})\n"
                 f'tables = [{_q(_shown(folder))} + f"/model_{{m.model:02d}}.interactions.tsv" for m in models]\n')
    else:
        wanted = ([t for t in own if t is not None] if own is not None
                  else [m.parent / f"{_stem(m)}.interactions.tsv" for m in models])
        reads = ("models = [" + ", ".join(_q(_shown(m)) for m in models) + "]\n"
                 "tables = [" + ", ".join(_q(_shown(t)) for t in wanted) + "]\n")
    have = len(wanted) == many and all(t.exists() for t in wanted)
    smi = _q(smiles) if smiles else "None"
    head = "from bindview import Ensemble" + (", read_models" if len(models) == 1 else "") + "\n\n"
    if have:
        seen += " Contacts: a table for every model beside it."
        code = (head + reads + f"ens = Ensemble.from_structures(models, {_q(cand.spec)}, {smi}, tables,\n"
                f"                               ensemble={_q(of)})\nfigure = ens.diagram().render()")
        return Track("ensemble", f"{cand.name} over {many} models", seen, code)
    code = (head + reads.split("\n")[0] + "\n"
            + f"ens = Ensemble.from_structures(models, {_q(cand.spec)}, {smi},        # no tables: PLIP finds the contacts\n"
              f"                               ensemble={_q(of)})\nfigure = ens.diagram().render()")
    if plip_here() and many <= MOST["models"]:
        seen += " Contacts: no table for every model, so PLIP finds them model by model."
        return Track("ensemble", f"{cand.name} over {many} models", seen, code)
    seen += (" Contacts: no table for every model"
             + (f" and more models than the {MOST['models']} this report has PLIP go through"
                if plip_here() else " and no PLIP where this ran")
             + ", so they are the table of heavy atoms, model by model.")
    loads = (f"models = read_models({_q(_shown(models[0]))})\n" if len(models) == 1 else
             "models = [read_structure(p) for p in (" + ", ".join(_q(_shown(m)) for m in models) + ")]\n")
    code = ("from bindview import Ensemble, " + ("read_models" if len(models) == 1 else "read_structure") + "\n"
            "from bindview.distance_table import ligand_contacts      # heavy atoms only: cruder than PLIP\n\n"
            + loads + f"tables = [ligand_contacts(m, {_q(cand.spec)}, {smi}) for m in models]\n"
            f"ens = Ensemble.from_structures(models, {_q(cand.spec)}, {smi}, tables,\n"
            f"                               ensemble={_q(of)})\nfigure = ens.diagram().render()")
    return Track("ensemble", f"{cand.name} over {many} models", seen, code)


# ---------------------------------------------------------------------------
# The survey, and its page
# ---------------------------------------------------------------------------

@dataclass
class Survey:
    """What an input holds and the figures that follow from it. Nothing is drawn until :meth:`draw`."""

    path: Path
    models: list[Path]                       # R1: one, or the models of a set
    found: Found                             # what the first model holds
    models_in_file: int
    beside: Beside
    tracks: list[Track]
    left: list[tuple[str, str]]              # what was found and has no figure, and why
    plip: bool
    #: ``"complex"``, ``"ensemble"`` (one complex in several models) or ``"series"`` (R1c).
    kind: str = "complex"
    #: The complexes of a series, one ligand each.
    members: list[Member] = field(default_factory=list)
    #: The folder of the input: the calls of the figures name their files from it, and are run in it.
    folder: Path = field(default_factory=Path.cwd)

    @property
    def model(self) -> Path:
        return self.models[0]

    def draw(self, views: bool = True) -> None:
        """Run the call of every figure, and draw its views (R15) unless ``views=False``. What the library
        refuses is said on the page, in its words."""
        for track in self.tracks:
            if track.status:
                continue
            track.svg, track.status, track.said, names = _run(track.code, track.title, self.folder)
            head, _, last = track.code.rpartition("\n")
            if track.status == "unreadable" and DETAILED in last:
                code = f"{head}\n{last.replace(DETAILED, SCHEMATIC)}"                         # R14
                svg, status, said, again = _run(code, track.title, self.folder)
                if status not in ("ok", "degraded"):         # not drawn: the detailed figure stays
                    track.said.append("the schematic style was tried and is not readable either"
                                      if status == "unreadable" else
                                      f"the schematic style was tried and came back {status}")
                else:
                    told = [s.removesuffix(SUMMARY) for s in track.said if "not readable" in s] or ["it is not readable"]
                    track.code, track.svg, track.status, track.schematic = code, svg, status, True
                    track.said = [f"drawn in the schematic style: in the detailed style, {s}" for s in told] + said
                    names = again
            if track.status in ("ok", "degraded"):
                track.facts = _facts(names.get("figure"))
            if views and track.status in ("ok", "degraded"):
                with _run_in(self.folder):
                    track.views = _views(track, names, self.found.plddt)

    def html(self) -> str:
        """The report as one page. It holds no date, no timing and no path from above the input's folder: the
        same input gives the same page, wherever the report is run."""
        return _page(self)


def _views(track: Track, names: dict[str, Any], plddt: bool) -> list[View]:
    """R15: the figure's object drawn again, each by a last line of its own, where the figure's call ran."""
    last = track.code.rpartition("\n")[2]
    style = DETAILED if DETAILED in last else SCHEMATIC if SCHEMATIC in last else None
    out = []
    if track.kind == "basepairs" and last.endswith("()"):                       # R17: its one view
        out.append(View("a figure that answers the pointer: point at a nucleotide, or click it",
                        last[:-1] + "interactive=True)"))
    elif style is None or not last.endswith(")"):
        return []
    else:
        if not track.schematic:
            other = SCHEMATIC if style == DETAILED else DETAILED
            out.append(View(f"in the {'schematic' if other == SCHEMATIC else 'detailed'} style",
                            last.replace(style, other)))
        if plddt:
            out.append(View("pLDDT in AlphaFold's colours", last[:-1] + ", plddt_colors=True)"))
        out.append(View("a figure that answers the pointer: point at a residue, or click it",
                        last[:-1] + ", interactive=True)"))
    for view in out:
        try:
            exec(compile(view.line, f"<{track.title}: {view.title}>", "exec"), names)   # noqa: S102 - as printed
        except BindviewError as exc:
            view.status, view.said = "refused", [str(exc)]
            continue
        figure = names["figure"]
        view.status, view.said = figure.status, list(figure.warnings)
        if figure.status in ("ok", "degraded"):
            view.svg = figure.svg
    return out


def _facts(figure: Any) -> dict[str, Any]:
    """R16: what an index says of a figure, read off the result that drew it."""
    graph = getattr(figure, "graph", None)
    kinds = Counter(c.type for c in graph.contacts) if graph is not None else Counter()
    conf = getattr(figure, "confidence", None) or {}
    burial = getattr(figure, "burial", None) or {}
    return {"contacts": {k: kinds[k] for k in KINDS if kinds[k]}, "iptm": conf.get("iptm"),
            "pocket_plddt": conf.get("pocket_plddt"), "ligand_plddt": conf.get("ligand_plddt"),
            "buried_fraction": burial.get("buried_fraction"), "warnings": len(getattr(figure, "warnings", []))}


def _run(code: str, title: str, folder: Path) -> tuple[str, str, list[str], dict[str, Any]]:
    """Run the call of a figure in the folder it is written for: its picture, its status, what the library
    said, and the names the call made (its diagram, for the views)."""
    names: dict[str, Any] = {}
    try:
        with _run_in(folder):
            exec(compile(code, f"<{title}>", "exec"), names)      # noqa: S102 - the call the page prints
    except BindviewError as exc:
        return "", "refused", [str(exc)], names
    figure = names["figure"]
    said = list(figure.warnings)
    peptide = getattr(getattr(names.get("dia"), "backend", None), "peptide", None)
    if peptide is not None:
        said.insert(0, f"read as a chain of {len(peptide.residues)} residues")
    return figure.svg, figure.status, said, names


def _member_table(model: Path, cand: Candidate, found: Found, st: Structure, alone: bool) -> "Path | None":
    """The table of contacts of a member of a set: the one named after its model, or, where the model
    lies in a folder of its own, the table beside it that names the ligand's atoms (R5)."""
    named = model.parent / f"{_stem(model)}.interactions.tsv"
    if named.exists():
        return named
    return _table_of(cand, found, st, _beside(model))[1] if alone else None


def _series_track(members: Sequence[Member]) -> Track:
    """R1c: several ligands on one protein, as the library's comparison of them."""
    n = len(members)
    tables = sum(1 for m in members if m.table is not None)
    seen = (f"{n} complexes of one protein, each with one small molecule: a series of ligands, drawn as a "
            f"comparison ({'what the two do differently' if n == 2 else 'what they have in common'}). Contacts: "
            + ("the table beside each model." if tables == n else
               f"the table beside the model for {tables} of them, the table of heavy atoms for the others."))

    def line(m: Member) -> str:
        smiles = f", smiles={_q(m.smiles)}" if m.smiles else ""
        if tables == n:
            return (f"    Diagram.from_structure({_q(_shown(m.model))}, ligand={_q(m.ligand.spec)}{smiles}, "
                    f"name={_q(m.name)},\n                           "
                    f"interactions=load_interactions({_q(_shown(m.table or m.model))})),\n")
        table = f", table={_q(_shown(m.table))}" if m.table else ""
        return f"    member({_q(_shown(m.model))}, {_q(m.ligand.spec)}, {_q(m.name)}{table}{smiles}),\n"

    if tables == n:
        head = "from bindview import Comparison, Diagram, load_interactions\n\n"
    else:
        head = ("from bindview import Comparison, Diagram, load_interactions, read_structure\n"
                "from bindview.distance_table import ligand_contacts      # heavy atoms only: where a member has no table\n\n\n"
                "def member(path, ligand, name, table=None, smiles=None):\n"
                "    model = read_structure(path)\n"
                "    contacts = load_interactions(table) if table else ligand_contacts(model, ligand, smiles)\n"
                "    return Diagram.from_structure(model, ligand=ligand, smiles=smiles, name=name, interactions=contacts)\n\n\n")
    code = head + "diagrams = [\n" + "".join(line(m) for m in members) + "]\nfigure = Comparison(diagrams).render()"
    return Track("series", f"{n} ligands of one pocket", seen, code)


def _parts(by: Beside, given: "Sequence[tuple[str, str, int, int]] | None"
           ) -> tuple[list[tuple[str, str, int, int]], list[tuple[str, str]]]:
    """R12: the parts of a chain that somebody named, as ``(name, chain, first, last)``; and what a
    ``meta.json`` names as a part without saying where it is, each with the reason it has no figure."""
    if given:
        if len(given) == 1:
            raise BindviewError(f"one part was named ({given[0][0]!r}): an interface is between two, so name both")
        return list(given), []
    meta = by.meta or {}
    chain, parts = meta.get("chain"), meta.get("participants")
    out, unread = [], []
    if isinstance(chain, str) and isinstance(parts, list):
        for part in parts:
            span = part.get("residues") if isinstance(part, dict) else None
            if isinstance(span, list) and len(span) == 2 and isinstance(part.get("name"), str):
                try:
                    out.append((str(part["name"]), chain, int(span[0]), int(span[1])))
                except (TypeError, ValueError):
                    unread.append((f"part {part['name']!r} of meta.json",
                                   f"its residues are {span!r}, not two residue numbers"))
    return (out if len(out) >= 2 else []), unread


def _part_tracks(parts: Sequence[tuple[str, str, int, int]], model: Path, st: Structure, by: Beside,
                 named_by: str) -> tuple[list[Track], list[tuple[str, str]]]:
    """R12: the interface of every pair of named parts that touches. A part that cannot be cut out of its
    chain is refused here, under the name it was given: for the cut it has a chain name of the report's."""
    numbers: dict[str, list[int]] = {}
    for key in st.residues():
        numbers.setdefault(key.chain, []).append(key.number)
    for name, chain, first, last in parts:
        if first > last:
            raise BindviewError(f"part {name!r} runs from {first} to {last}: the first number comes first")
        if not any(first <= number <= last for number in numbers.get(chain, ())):
            raise BindviewError(f"part {name!r}: chain {chain!r} has no residue from {first} to {last}")
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            if a[1] == b[1] and a[2] <= b[3] and b[2] <= a[3]:
                raise BindviewError(f"parts {a[0]!r} ({a[2]} to {a[3]}) and {b[0]!r} ({b[2]} to {b[3]}) overlap on "
                                    f"chain {a[1]!r}")
    taken = {a.chain for a in st.atoms}
    free = [c for c in "XYZWVUTSRQPONMLKJIHGFEDCBA" if c not in taken]
    if len(free) < len(parts):
        raise BindviewError(f"{len(parts)} parts were named and only {len(free)} chain names are free")
    letter = {part: free[k] for k, part in enumerate(parts)}
    cut = st
    for chain in dict.fromkeys(p[1] for p in parts):
        cut = cut.split(chain, {letter[p]: (p[2], p[3]) for p in parts if p[1] == chain})
    pairs = []
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            rows = len(interface_contacts(cut, [letter[a]], [letter[b]], stacking=True))
            if rows:
                pairs.append((rows, a, b))
    pairs.sort(key=lambda x: -x[0])
    tracks: list[Track] = []
    left: list[tuple[str, str]] = []

    def said(name: str) -> str:
        return name.replace("=", " ").replace("+", " ")

    for rows, a, b in pairs:
        if len(tracks) == MOST["interface"]:
            left.append((f"the interface of {a[0]} and {b[0]} ({rows} rows)", "more interfaces than the report draws"))
            continue
        seen = (f"{a[0]} (chain {a[1]}, {a[2]} to {a[3]}) and {b[0]} (chain {b[1]}, {b[2]} to {b[3]}), named "
                f"{named_by}, touch: the plain distance table, with ring stacking, has {rows} rows for the pair.")
        pae = ""
        if by.pae is not None:
            seen += f" Predicted aligned errors: {by.pae.name} beside the model."
            pae = f",\n                 pae={_q(_shown(by.pae))}"
        splits = "".join(f".split({_q(chain)}, {{" + ", ".join(f"{_q(letter[p])}: ({p[2]}, {p[3]})" for p in (a, b) if p[1] == chain)
                         + "})" for chain in dict.fromkeys(p[1] for p in (a, b)))
        code = ("from bindview import Interface, read_structure\n"
                "from bindview.distance_table import interface_contacts      # distances only: no interaction analysis\n\n"
                f"model = read_structure({_q(_shown(model))}){splits}      # each part a chain of its own\n"
                f"face = Interface(model, [{_q(said(a[0]) + '=' + letter[a])}, {_q(said(b[0]) + '=' + letter[b])}],\n"
                f"                 interface_contacts(model, [{_q(letter[a])}], [{_q(letter[b])}], stacking=True){pae})\n"
                'figure = face.render(style="detailed")')
        tracks.append(Track("parts", f"The interface of {a[0]} and {b[0]}", seen, code))
    if not pairs:
        left.append((", ".join(p[0] for p in parts), "the parts that were named do not touch"))
    return tracks, left


def survey(path: "str | Path", smiles: "Mapping[str, str] | None" = None,
           domains: "Sequence[tuple[str, str, int, int]] | None" = None) -> Survey:
    """Look at a model, or at a folder that holds one or a set of them: what it holds, and the figures
    that follow.

    ``smiles`` hands in the SMILES of a residue by its name, for the bond orders that a deposited
    structure does not carry. ``domains`` names parts of a chain as ``(name, chain, first, last)``: the
    interface between them is drawn (R12). Nothing is drawn here: :meth:`Survey.draw` does that.
    """
    path = Path(path)
    if not path.exists():
        raise BindviewError(f"{path}: no such file or folder")
    folder = Path(os.path.abspath(path if path.is_dir() else path.parent))
    naming = _FOLDER.set(folder)
    try:
        found = _survey(path, dict(smiles or {}), domains)
    finally:
        _FOLDER.reset(naming)
    found.folder = folder
    return found


def _survey(path: Path, given: "dict[str, str]", domains: "Sequence[tuple[str, str, int, int]] | None") -> Survey:
    members = _members(path)
    passed = [(f"folder {sub.name} ({len(names)} models: {_named(names)})", _ONE_MODEL)      # R1b
              for sub, names in _passed_over(path)]
    if not members:
        raise BindviewError(f"{path}: no model here (an mmCIF or PDB file, or a folder that holds one)"
                            + (f"; {'; '.join(what for what, _ in passed)}: {_ONE_MODEL}" if passed else ""))
    models = [m for _, m in members]
    alone = len({m.parent for m in models}) == len(models) and len(models) > 1     # each in a folder of its own
    structures = [read_structure(m) for m in models]
    looks = [look(s) for s in structures]
    if len({_composition(f) for f in looks}) > 1:                                 # R1c
        ligands = [[c for c in f.candidates if c.kind == "small molecule"] for f in looks]
        if not (_one_protein(looks) and all(len(found) == 1 for found in ligands)):
            raise BindviewError(f"{path} holds {len(models)} models that are neither one complex nor a series of "
                              f"ligands on one protein ({', '.join(m.name if not alone else m.parent.name for m in models)}): "
                              "name one of them, or ask for a page of each (--each)")
        series = []
        for (name, model), st_m, found_m, (cand,) in zip(members, structures, looks, ligands):
            by_m = _beside(model)
            handed, where = _smiles(cand, found_m, by_m if alone or len(models) == 1 else Beside(), given)
            series.append(Member(name, model, cand, _member_table(model, cand, found_m, st_m, alone), handed, where))
        return Survey(path, models, looks[0], 1, _beside(models[0]), [_series_track(series)], passed, plip_here(),
                      kind="series", members=series)
    model, found, st = models[0], looks[0], structures[0]
    numbers = [m.model for m in read_models(model)]
    by = _beside(model)
    tracks: list[Track] = []
    left: list[tuple[str, str]] = list(passed)
    small = [c for c in found.candidates if c.kind == "small molecule"]
    ions = [c for c in found.candidates if c.kind == "ion"]
    long_protein = any(c.kind == "protein" and c.residues > PEPTIDE[1] for c in found.chains)
    short = sorted((c for c in found.chains if c.peptide or (c.oligo and long_protein)), key=lambda c: c.residues)   # R7, R7b
    for cand in small[:MOST["small molecule"]]:
        tracks.append(_ligand_track(cand, model, st, found, by, given))
    if not small and not short:
        for cand in ions[:MOST["ion"]]:
            tracks.append(_ligand_track(cand, model, st, found, by, given))
    for chain in short[:MOST["peptide"]]:
        tracks.append(_peptide_track(chain, model, by, [c.name for c in found.chains if c.kind == "protein"]))
    # R8, R8b: two chains of more than 50, or two nucleic chains of any length
    polymers = [c for c in found.chains if c.kind in ("protein", "nucleic")]
    pairs = []
    for i, a in enumerate(polymers):
        for b in polymers[i + 1:]:
            both_long = a.residues > PEPTIDE[1] and b.residues > PEPTIDE[1]
            if not (both_long or (a.kind == "nucleic" and b.kind == "nucleic")):
                continue
            rows = len(interface_contacts(st, [a.name], [b.name], stacking=True))
            if rows:
                pairs.append((rows, a, b))
    pairs.sort(key=lambda x: -x[0])
    for rows, a, b in pairs[:MOST["interface"]]:
        tracks.append(_interface_track(a, b, rows, model, by))
        if a.kind == "nucleic" and b.kind == "nucleic":                           # R17
            tracks.append(_basepairs_track(a, b, model, st, found, by, small[:MOST["small molecule"]], given))
    parts, unread = _parts(by, domains)                                           # R12
    left.extend(unread)
    if parts:
        try:
            more, missing = _part_tracks(parts, model, st, by, "by the caller" if domains else "in meta.json")
        except BindviewError as exc:
            if domains:
                raise
            # A file beside the model was found, not handed in: what is wrong in it does not end the report.
            more, missing = [], [(", ".join(p[0] for p in parts) + " (parts that meta.json names)",
                                  f"no interface of them is drawn: {exc}")]
        tracks.extend(more)
        left.extend(missing)
    many = len(numbers) if len(models) == 1 else len(models)
    if many >= 2 and small:
        own = None
        if alone:
            own = [_member_table(m, small[0], f, s, True) for m, f, s in zip(models, looks, structures)]
        tracks.append(_ensemble_track(small[0], models, numbers, by, found, given, own))
    # What was found and has no figure, with the reason (R4, R7, R8, R9)
    for cand in found.candidates:
        if cand.kind in ("cap", "additive", "sugar") or (cand.kind == "ion" and (small or short)):
            why = {"cap": "the end of a chain", "additive": "on the list of what crystals and buffers bring",
                   "sugar": "a sugar bonded to the chain", "ion": "an ion beside a ligand"}[cand.kind]
            left.append((f"{cand.name} ({cand.kind}, {cand.heavy} heavy atom{'s' if cand.heavy != 1 else ''}, "
                         f"{cand.copies} cop{'ies' if cand.copies != 1 else 'y'})",
                         f"{why}; ligand={_q(cand.spec)} draws it"))
    for cand in small[MOST["small molecule"]:] + ([] if small or short else ions[MOST["ion"]:]):
        left.append((f"{cand.name} ({cand.kind})", f"more than the report draws; ligand={_q(cand.spec)} draws it"))
    for chain in short[MOST["peptide"]:]:
        left.append((f"chain {chain.name} ({chain.what} of {chain.residues})", "more chains as ligands than the report draws"))
    for rows, a, b in pairs[MOST["interface"]:]:
        left.append((f"the interface of chains {a.name} and {b.name} ({rows} rows)", "more interfaces than the report draws"))
    for chain in found.chains:
        if chain.joined:                                                          # R2b
            left.append((f"chain {chain.name} (protein, {chain.residues} residues)",
                         f"bonded to chain {chain.joined}: a part of that protein, and no peptide; "
                         f"ligand={_q('chain:' + chain.name)} draws it as one all the same"))
        elif chain.kind == "nucleic" and not any(chain.name in (a.name, b.name) for _, a, b in pairs) \
                and chain not in short:
            left.append((f"chain {chain.name} (nucleic, {chain.residues} nucleotides)",
                         "it touches no other nucleic chain, and no protein chain of more than 50 residues is "
                         "here for it to be the ligand of"))
    if many >= 2 and not small:
        left.append((f"{many} models", "an ensemble is counted for a small molecule, and there is none"))
    proteins = [c for c in found.chains if c.kind == "protein"]
    if not tracks and len(proteins) == 1 and not found.candidates:
        left.append((f"chain {proteins[0].name} alone",
                     "one protein chain and nothing beside it. An interface between two domains of one chain "
                     "can be drawn, but somebody has to say which residues the domains are (--domain)"))
    return Survey(path, models, found, len(numbers), by, tracks, left, plip_here(),
                  kind="ensemble" if many >= 2 else "complex")


@dataclass
class Row:
    """R16: a member of a set in its index."""

    name: str
    model: Path
    page: str = ""                   # the member's page, from the index
    tracks: list[Track] = field(default_factory=list)
    refused: str = ""                # what the library said, where the member has no page
    why: str = "refused"             # and what that was: it refused the member, or could not read its model
    left: list[tuple[str, str]] = field(default_factory=list)

    @property
    def note(self) -> str:
        """The library's words where the member has no page, or why it has no figure."""
        if self.refused:
            return f"{self.why}: {self.refused}"
        if not self.tracks:
            return "nothing to draw" + (f": {self.left[0][0]}, {self.left[0][1]}" if self.left else "")
        return ""

    @property
    def first(self) -> "Track | None":
        """The member's first figure that is drawn."""
        return next((t for t in self.tracks if t.status in ("ok", "degraded")), None)


def write_each(path: "str | Path", out: "str | Path | None" = None, smiles: "Mapping[str, str] | None" = None,
               views: bool = True, say: Any = None) -> tuple[list[Row], Path]:
    """R16: a page for each member of a folder, and an index of them as a page and a CSV.

    The index is ``<folder>.report.html`` beside the folder unless ``out`` says where; the members' pages
    are in the folder of the same name without ``.html``, and the CSV beside the index. ``say`` is told of
    each member as it is done."""
    source = Path(path)
    if not source.is_dir():
        raise BindviewError(f"{source}: --each takes a folder of models, or of folders that hold one")
    members = _members(source)
    passed = _passed_over(source)
    if not members and not passed:
        raise BindviewError(f"{source}: no model here (an mmCIF or PDB file, or a folder that holds one)")
    index = Path(out) if out else source.parent / f"{source.name}.report.html"
    pages = index.with_name(index.name.removesuffix(".html"))
    pages.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    # A page is named after its member, and a name is one page: the second member of a name gets a number that
    # no member has (letters in either case are one name where a file system takes them for one).
    taken = {name.casefold() for name, _ in members}
    seen: Counter[str] = Counter()
    for k, (name, model) in enumerate(members, 1):
        seen[name.casefold()] += 1
        if seen[name.casefold()] > 1:
            number = seen[name.casefold()]
            while f"{name}-{number}".casefold() in taken:
                number += 1
            name = f"{name}-{number}"
            taken.add(name.casefold())
        row = Row(name, model)
        try:
            found = survey(model, smiles)
            found.draw(views=views)
        except BindviewError as exc:
            # Named within the folder: the index is read beside it, not on the machine that wrote it.
            row.refused = str(exc).replace(str(model), str(model.relative_to(source)))
            if isinstance(exc, StructureError):
                row.why = "could not be read"
        else:
            row.tracks, row.left = found.tracks, found.left
            target = pages / f"{name}.html"
            target.write_text(found.html(), encoding="utf-8")
            row.page = f"{pages.name}/{target.name}"
        rows.append(row)
        if say is not None:
            done = ", ".join(f"{t.title}: {t.status}" for t in row.tracks) or row.refused or "nothing drawn"
            say(f"  [{k}/{len(members)}] {name}: {done}")
    for sub, names in passed:                            # R1b: no member, and a row that says so
        rows.append(Row(sub.name, sub, why="not read",
                        refused=f"{len(names)} models in one folder ({_named(names)}), and {_ONE_MODEL}"))
        if say is not None:
            say(f"  {sub.name}: {rows[-1].note}")
    if passed:
        rows.sort(key=lambda row: row.name)
    index.write_text(_index_page(source, rows), encoding="utf-8")
    index.with_suffix(".csv").write_text(_index_csv(rows), encoding="utf-8")
    return rows, index


def _number(value: Any, digits: int = 0, percent: bool = False) -> str:
    if value is None:
        return ""
    return f"{100 * value:.{digits}f} %" if percent else f"{value:.{digits}f}"


def _index_csv(rows: Sequence[Row]) -> str:
    """R16: the rows of the index, one a member, with a column for each kind of contact."""
    buf = io.StringIO()
    out = csv.writer(buf, lineterminator="\n")
    out.writerow(["member", "page", "figures", "status", *KINDS, "iptm", "pocket_plddt", "ligand_plddt",
                  "buried_fraction", "warnings", "note"])
    for row in rows:
        first = row.first
        facts = first.facts if first is not None else {}
        kinds = facts.get("contacts", {})
        out.writerow([row.name, row.page, " | ".join(t.title for t in row.tracks),
                      " | ".join(t.status for t in row.tracks),
                      *[kinds.get(k, 0) if first is not None else "" for k in KINDS],
                      _number(facts.get("iptm"), 2), _number(facts.get("pocket_plddt"), 1),
                      _number(facts.get("ligand_plddt"), 1), _number(facts.get("buried_fraction"), 3),
                      facts["warnings"] if first is not None else "", row.note])
    return buf.getvalue()


def _index_page(source: Path, rows: Sequence[Row]) -> str:
    """R16: the index of a set's pages. It holds no date and no timing, as a report does not."""
    esc = html.escape
    drawn = sum(1 for r in rows if r.first is not None)
    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<title>bindview report: {esc(source.name)}, each</title><style>{_CSS}</style></head><body><main>",
           f"<h1>{esc(source.name)}: {len(rows)} inputs, a page each</h1>",
           f"<p>Each input of the folder was looked at and drawn as an input of its own; its page has its "
           f"figures, each under the call that drew it. {drawn} of {len(rows)} have a figure. The numbers are "
           f"those of each input's first figure. bindview {esc(__version__)}; PLIP is "
           f"{'installed' if plip_here() else 'not installed'} where this ran. The same rows are in the CSV "
           "beside this page.</p>",
           "<table><tr><th>input</th><th>figures</th><th>contacts of the first</th><th>ipTM</th>"
           "<th>pocket pLDDT</th><th>ligand pLDDT</th><th>buried</th><th>warnings</th></tr>"]
    for row in rows:
        # (A link is an address: a name with #, ? or % in it is quoted, or a browser asks for another page.)
        name = f"<a href='{esc(urllib.parse.quote(row.page))}'>{esc(row.name)}</a>" if row.page else esc(row.name)
        if row.refused:
            out.append(f"<tr><td>{name}</td><td colspan='7'><span class='badge refused'>{esc(row.why)}</span> "
                       f"{esc(row.refused)}</td></tr>")
            continue
        figures = "<br>".join(f"{esc(t.title)} <span class='badge {esc(t.status.split()[0] if t.status else 'not')}'>"
                              f"{esc(t.status or 'not run')}</span>" for t in row.tracks)
        if not figures:
            figures = esc(row.note)
        first = row.first
        facts = first.facts if first is not None else {}
        kinds = ", ".join(f"<span class='k'>{n} {esc(KIND_NAMES[k])}</span>" for k, n in facts.get("contacts", {}).items())
        out.append(f"<tr><td>{name}</td><td>{figures}</td><td>{kinds}</td>"
                   f"<td>{_number(facts.get('iptm'), 2)}</td><td>{_number(facts.get('pocket_plddt'))}</td>"
                   f"<td>{_number(facts.get('ligand_plddt'))}</td>"
                   f"<td>{_number(facts.get('buried_fraction'), percent=True)}</td>"
                   f"<td>{facts.get('warnings', '') if first is not None else ''}</td></tr>")
    out.append("</table></main></body></html>\n")
    return "".join(out)

_CSS = ("body{font:15px/1.5 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;color:#1f2933;margin:0;"
        "background:#f8fafc}main{max-width:1080px;margin:0 auto;padding:28px 24px 80px}"
        "h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:40px 0 6px}h3{font-size:15px;margin:0 0 4px}"
        "p{max-width:84ch;margin:6px 0}.quiet{color:#52606d}"
        "table{border-collapse:collapse;margin:8px 0;font-size:14px}td,th{border:1px solid #d9e2ec;"
        "padding:4px 10px;text-align:left;vertical-align:top}th{background:#eef2f6;font-weight:600}"
        "article{background:#fff;border:1px solid #d9e2ec;border-radius:8px;padding:16px 18px;margin:14px 0}"
        "pre{background:#f1f5f9;border-radius:6px;padding:10px 12px;overflow-x:auto;font-size:12.5px;"
        "line-height:1.45}.fig{overflow-x:auto;margin:10px 0}.fig svg{display:block;max-width:100%;height:auto}"
        ".badge{display:inline-block;border-radius:10px;padding:1px 9px;font-size:12px;font-weight:600;"
        "color:#fff;background:#2f855a;margin-right:8px}.badge.degraded{background:#b7791f}"
        ".badge.refused,.badge.unreadable{background:#c53030}.badge.empty,.badge.not{background:#718096}"
        "ul{margin:6px 0;padding-left:22px}li{margin:2px 0}code{font-size:13px}"
        "h4{font-size:14px;margin:18px 0 4px}.views{display:grid;grid-template-columns:repeat(auto-fit,"
        "minmax(320px,1fr));gap:14px}.views figure{margin:0;min-width:0}.views pre{white-space:pre-wrap}"
        "figcaption{font-size:13px}.k{white-space:nowrap}")


def _page(result: Survey) -> str:
    esc = html.escape
    found, by, models = result.found, result.beside, result.models
    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<title>bindview report: {esc(result.path.name)}</title><style>{_CSS}</style></head><body><main>",
           f"<h1>What {esc(result.path.name)} holds, and its figures</h1>",
           "<p>Nobody said what kind of input this is: the report looked. Each figure is drawn by the call "
           "printed above it; copy a call and change it. A call names its files from the folder of the input: "
           "run it there. Under a figure, its views: the same object drawn again by the line printed above each. "
           f"bindview {esc(__version__)}; PLIP is {'installed' if result.plip else 'not installed'} "
           "where this ran.</p>",
           "<h2>What the input holds</h2>"]
    if result.kind == "series":
        out.append(f"<p>{len(result.members)} complexes of one protein, each with one small molecule: a series of "
                   "ligands. The chains below are those of the first.</p>")
        out.append("<table><tr><th>member</th><th>ligand</th><th>heavy atoms</th><th>SMILES</th><th>contacts from</th></tr>"
                   + "".join(f"<tr><td>{esc(m.name)}</td><td>{esc(m.ligand.name)} ({esc(m.ligand.formula)})</td>"
                             f"<td>{m.ligand.heavy}</td><td>{esc(m.smiles_from)}</td>"
                             f"<td>{esc(m.table.name) if m.table else 'the table of heavy atoms'}</td></tr>"
                             for m in result.members) + "</table>")
    elif len(models) > 1:
        out.append(f"<p>{len(models)} models of one composition: {esc(', '.join(m.name for m in models[:8]))}"
                   f"{' and more' if len(models) > 8 else ''}. What follows is read from the first.</p>")
    elif result.models_in_file > 1:
        out.append(f"<p>{esc(models[0].name)} holds {result.models_in_file} models. "
                   "What follows is read from the first.</p>")
    else:
        out.append(f"<p>One model: {esc(models[0].name)}.</p>")
    out.append("<table><tr><th>chain</th><th>what it is</th><th>residues</th></tr>" + "".join(
        f"<tr><td>{esc(c.name)}</td><td>{esc(c.what)}</td><td>{c.residues or ''}</td></tr>"
        for c in found.chains) + "</table>")
    if result.kind == "series":
        pass                                         # the ligands of a series stand in the table above
    elif found.candidates:
        out.append("<table><tr><th>other residue</th><th>copies</th><th>heavy atoms</th><th>taken for</th>"
                   "<th>within bonding distance of a chain</th></tr>" + "".join(
                       f"<tr><td>{esc(c.name)}</td><td>{c.copies}</td><td>{c.heavy}</td><td>{c.kind}</td>"
                       f"<td>{'yes' if c.bonded else ''}</td></tr>" for c in found.candidates) + "</table>")
    else:
        out.append("<p>No residue outside the chains but water.</p>")
    seen = [f"{found.waters} water{'s' if found.waters != 1 else ''}",
            "the file declares a per-atom pLDDT, which the figures draw" if found.plddt
            else "the file declares no pLDDT",
            f"predicted aligned errors beside the model ({by.pae.name})" if by.pae
            else "no file of predicted aligned errors beside the model"]
    if by.summary:
        seen.append(f"a summary of the model's own scores beside it ({by.summary.name})")
    seen.append("tables of contacts beside the model: " + ", ".join(t[0].name for t in by.tables[:6])
                + (f" and {len(by.tables) - 6} more" if len(by.tables) > 6 else "") if by.tables
                else "no table of contacts beside the model")
    if by.not_tables:
        seen.append("not read as a table of contacts: " + "; ".join(f"{p.name} ({why})" for p, why in by.not_tables[:6])
                    + (f"; and {len(by.not_tables) - 6} more" if len(by.not_tables) > 6 else ""))
    out.append("<ul>" + "".join(f"<li>{esc(s)}</li>" for s in seen) + "</ul>")
    drawn = result.tracks
    out.append(f"<h2>The figures ({len(drawn)})</h2>")
    if not drawn:
        out.append("<p>None. What was found is listed below, with what is missing.</p>")
    for t in drawn:
        notes = "".join(f"<li>{esc(s)}</li>" for s in t.said)
        status = t.status or "not run"
        shown = [v for v in t.views if v.svg]
        views = "".join(f"<figure><pre><code>{esc(v.line)}</code></pre><div class='fig'>{v.svg}</div>"
                        f"<figcaption class='quiet'>{esc(v.title)}</figcaption></figure>" for v in shown)
        unshown = "".join(f"<li>{esc(v.title)} (<code>{esc(v.line)}</code>): {esc(v.status)}"
                          + (f": {esc(v.said[0])}" if v.said else "") + "</li>" for v in t.views if not v.svg)
        out.append(f"<article><h3>{esc(t.title)}</h3><p class='quiet'>{esc(t.why)}</p><pre><code>{esc(t.code)}"
                   f"</code></pre><div class='fig'>{t.svg}</div><p><span class='badge {esc(status.split()[0])}'>"
                   f"{esc(status)}</span></p>" + (f"<ul class='quiet'>{notes}</ul>" if notes else "")
                   + (f"<h4>Views ({len(shown)})</h4><div class='views'>{views}</div>" if shown else "")
                   + (f"<ul class='quiet'>{unshown}</ul>" if unshown else "") + "</article>")
    if result.left:
        out.append("<h2>Found, and not drawn</h2>")
        out.append("<table><tr><th>found</th><th>why it has no figure</th></tr>" + "".join(
            f"<tr><td>{esc(what)}</td><td>{esc(why)}</td></tr>" for what, why in result.left) + "</table>")
    out.append("</main></body></html>\n")
    return "".join(out)
