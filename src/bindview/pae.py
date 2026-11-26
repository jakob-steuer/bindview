"""Predicted aligned error: what a model says about the placement of one residue against another.

pLDDT is a claim about a residue's own surroundings. The predicted aligned
error (PAE) is a claim about a *pair*: the expected error, in ångström, in the
position of residue ``j`` when the model is aligned on residue ``i``. For a
contact across an interface it is the number that says whether the two sides
are placed against each other at all; a model can have confident residues on
both sides and no idea how they sit together. Low is confident. AlphaFold 3
caps the scale near 31.75 Å, which means "no idea".

This module reads the matrix (AlphaFold 3's ``confidences.json``, the PAE file
of the AlphaFold database, each plain or gzipped, or the ``pae_*.npz`` Boltz
writes) and answers for two residues. The
value of a pair is **the smaller of the two directions** of the matrix.
Nothing here draws; a contact carries the value
(:attr:`bindview.interactions.Interaction.pae`) and the styles write it on the
contact's line as a tag that says what it is, ``PAE 11`` -- never by stroke
weight, opacity or dash, which belong to pLDDT. Or, where it is asked for
(``render(pae_colors=True)``) and in the diagram of a ligand, as a glow behind
the line in the colour of its band (:data:`PAE_BANDS`).

A model that is run several times gives several matrices
(:class:`PaeDraws`). They become one number per pair by their **median**: the
largest of n draws rises with n and the smallest falls with it, so either
would make a figure of five draws say something else than a figure of one for
no reason in the model. The value of each draw is kept beside it.

Draws of a model with several copies of one chain need one more step. The
copies are interchangeable to the model, and it is free to name them in another
order in every draw: "chain A, residue 94 and chain B, residue 151" is a
contact in one draw and two residues on opposite sides of the assembly in the
next. Before a pair is looked up in a draw, the chains of that draw are
therefore matched to those of the structure that is drawn, by superposition
(:func:`match_chains`), which needs the model file of every draw. Without them
such draws are refused, not read by name.

Two neighbours of PAE are **not** read here, and are refused where a PAE is
expected rather than taken for one. The predicted distance error (PDE, Boltz's
``pde_*.npz``) is the expected error of the *distance* between two residues:
it has no frame of reference, and is not a second opinion on PAE. pLDDT
(``plddt_*.npz``) is about one residue.

One number is worked out from the matrix here and is not a model's own: **ipSAE**
(:func:`ipsae`), a score of how well two sides of a complex are placed against
each other. It is computed only where a caller asks for it.
"""

from __future__ import annotations

import gzip
import json
import zipfile
from dataclasses import dataclass, field, replace
from itertools import permutations, product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

from .errors import BindviewError
from .structure import ResidueKey

if TYPE_CHECKING:  # pragma: no cover
    from .structure import Structure

__all__ = ["Pae", "PaeDraws", "PAE_SHOWN", "PAE_STRONG", "PAE_BANDS", "pae_band", "pae_tag", "pae_shown", "tokens_of",
           "read_pae",
           "match_chains", "model_beside", "SAME_FIT", "Ipsae", "OneWay", "IPSAE_CUTOFF", "ipsae", "ipsae_d0",
           "ipsae_of"]

#: A contact whose PAE, rounded to a whole ångström, is below this carries no tag: below it the
#: relative placement of two residues is customarily read as confident.
PAE_SHOWN = 5
#: ... and from here on the tag is bold, and a figure all of whose contacts are here says so in words.
PAE_STRONG = 10


#: The bands of a PAE that is drawn as a glow, best first: the name of the colour (those that pLDDT is drawn in,
#: with the same meaning: blue is sure, orange is not) and the PAE, in ångström, from which it is this one.
PAE_BANDS: "tuple[tuple[str, float], ...]" = (("very_high", 0.0), ("confident", 5.0), ("low", 10.0), ("very_low", 15.0))


def _a_pae(value: float) -> float:
    """A PAE as the number it has to be: of ångström, from 0. Not a number, below 0 and without end are refused."""
    if not 0.0 <= float(value) < float("inf"):
        raise BindviewError(f"a PAE is a number of ångström from 0, not {value!r}")
    return float(value)


def pae_band(value: float) -> str:
    """The band of a PAE (:data:`PAE_BANDS`)."""
    value = _a_pae(value)
    return next(name for name, least in reversed(PAE_BANDS) if value >= least)


#: The residues a co-folding model of the AlphaFold 3 kind (Boltz among them) takes as one token
#: each: the standard amino acids and nucleotides, and the unknown ones. Any other residue (a ligand,
#: a modified residue) is one token for each of its heavy atoms.
ONE_TOKEN = frozenset("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL UNK "
                      "A C G U N DA DC DG DT DN".split())
#: What an ``.npz`` that is not a PAE says of itself, by the name of its array.
_NOT_PAE = {
    "pde": ("a predicted distance error (PDE)",
            "PDE is another quantity: the expected error of the distance between two residues, with no frame "
            "of reference. bindview writes PAE on a contact and does not draw PDE; give the pae_*.npz of the "
            "same model"),
    "plddt": ("per-residue pLDDT",
              "pLDDT is about one residue and PAE about two. bindview reads pLDDT from the model file, where "
              "Boltz writes it on the 0-100 scale and declares it; give the pae_*.npz of the same model"),
}


#: Two assignments of the copies of a chain whose fits differ by no more than this (CA RMSD, ångström)
#: fit equally well: a symmetric oligomer does in each of its rotations.
SAME_FIT = 0.5
#: Assignments of copies that are tried one by one; beyond this many a model is not matched.
MAX_ASSIGNMENTS = 5040


def tokens_of(structure: "Structure") -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Chain and residue number of every token of a model, in the order of its file.

    A matrix that comes without the names of its rows (Boltz's ``pae_*.npz``)
    has one row per token, in the order in which the model's file lists its
    residues: one for a standard residue of a polymer, one per heavy atom for
    anything else (:data:`ONE_TOKEN`), as the published tokenisation has it.
    A matrix is only ever taken if its size is exactly the number of tokens
    counted this way.
    """
    chains, numbers, _, _, _ = _tokens(structure)
    return chains, numbers


def _tokens(structure: "Structure"
            ) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...], np.ndarray, tuple[str, ...]]:
    """:func:`tokens_of`, and with every token the name of its residue, a place for it (its CA, or
    the centre of its heavy atoms where it has none) and the insertion code of its residue."""
    chains: list[str] = []
    numbers: list[int] = []
    names: list[str] = []
    places: list[tuple[float, float, float]] = []
    icodes: list[str] = []
    for key in structure.residues():
        atoms = [a for a in structure.residue_atoms(key) if a.element not in ("H", "D")]
        name = structure.residue_name(key)
        n = 1 if name in ONE_TOKEN else len(atoms)
        ca = [a.xyz for a in atoms if a.name == "CA"]
        place = ca[0] if ca else tuple(np.mean([a.xyz for a in atoms], axis=0)) if atoms else (0.0, 0.0, 0.0)
        chains.extend([key.chain] * n)
        numbers.extend([key.number] * n)
        names.extend([name] * n)
        places.extend([place] * n)
        icodes.extend([key.icode] * n)
    return tuple(chains), tuple(numbers), tuple(names), np.array(places, dtype=float).reshape(-1, 3), tuple(icodes)


def _per_chain(chains: Sequence[str]) -> str:
    """How many tokens each chain has, in words: ``chain A: 157, chain B: 157``."""
    per: dict[str, int] = {}
    for c in chains:
        per[c] = per.get(c, 0) + 1
    return ", ".join(f"chain {c}: {n}" for c, n in per.items())


def _afdb_matrix(data: Any, source: str = "") -> "np.ndarray | None":
    """The matrix of the AlphaFold database's PAE file, or ``None`` where ``data`` is not what such a file holds.

    The database writes a list with one object, ``[{"predicted_aligned_error": [[...], ...],
    "max_predicted_aligned_error": 31.75}]``. Some older files are that object without the list, and
    the first ones hold three flat lists instead of the matrix: ``residue1``, ``residue2`` and
    ``distance``, the two residues counted from 1. In every form the matrix is n × n, in ångström,
    for the n residues of the one chain of the model, in the model's order, and ``[i][j]`` (in the
    flat lists: ``residue1`` i + 1 with ``residue2`` j + 1) is the error of residue j when the model
    is aligned on residue i: rows are aligned residues, as in :class:`Pae`.
    """
    held = data[0] if isinstance(data, list) and len(data) == 1 else data
    if not isinstance(held, dict) or "pae" in held:
        return None
    flat = ("residue1", "residue2", "distance")
    if "predicted_aligned_error" not in held and not all(k in held for k in flat):
        return None
    what = source or "the PAE file"
    try:
        if "predicted_aligned_error" in held:
            matrix = np.array(held["predicted_aligned_error"], dtype=float)
        else:
            first, second = (np.array(held[k], dtype=int) for k in flat[:2])
            values = np.array(held["distance"], dtype=float)
            if not (first.ndim == second.ndim == values.ndim == 1 and len(first) == len(second) == len(values)):
                raise ValueError(f"'residue1', 'residue2' and 'distance' are lists of {first.size}, {second.size} "
                                 f"and {values.size} values")
            if not len(first) or min(first.min(), second.min()) < 1:
                raise ValueError("'residue1' and 'residue2' count the residues from 1")
            n = int(max(first.max(), second.max()))
            matrix = np.full((n, n), np.nan)
            matrix[first - 1, second - 1] = values
    except (TypeError, ValueError) as exc:
        raise BindviewError(f"{what} is written as the PAE file of the AlphaFold database, and its predicted aligned "
                            f"errors are not a matrix of numbers: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        raise BindviewError(f"{what} is written as the PAE file of the AlphaFold database, and its "
                            f"'predicted_aligned_error' is not a square matrix "
                            f"({' x '.join(str(n) for n in matrix.shape) or 'no rows'})")
    return matrix


def model_beside(path: "str | Path") -> "Path | None":
    """The model file a matrix belongs to, where it sits beside it under the name Boltz gives the
    two: ``pae_<name>.npz`` with ``<name>.cif`` (or ``.pdb``). By name only, never "whatever
    structure is in the folder"."""
    p = Path(path)
    if p.suffix != ".npz" or not p.name.startswith("pae_"):
        return None
    for suffix in (".cif", ".pdb"):
        cand = p.with_name(p.name[len("pae_"):-len(".npz")] + suffix)
        if cand.is_file():
            return cand
    return None


def _fit(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean square distance between two sets of points after the best superposition of one on the other."""
    a0, b0 = a - a.mean(axis=0), b - b.mean(axis=0)
    u, sv, vt = np.linalg.svd(a0.T @ b0)
    sign = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
    rest = float((a0 ** 2).sum() + (b0 ** 2).sum() - 2.0 * (sv[0] + sv[1] + sign * sv[2]))
    return float(np.sqrt(max(rest, 0.0) / len(a)))


def _copies(chains: Sequence[str], numbers: Sequence[int], names: Sequence[str]) -> list[list[str]]:
    """The chains that are copies of one another: the same residues, by number and by name, in the
    same order. Each group in the order of the file. Without ``names`` (a matrix that came without
    its model) these are the chains with the same residue numbers, which may or may not be copies."""
    made: dict[str, list[tuple[int, str]]] = {}
    for i, c in enumerate(chains):
        made.setdefault(c, []).append((numbers[i], names[i] if names else ""))
    groups: dict[tuple, list[str]] = {}
    for c, residues in made.items():
        groups.setdefault(tuple(residues), []).append(c)
    return [g for g in groups.values() if len(g) > 1]


def match_chains(reference: "Pae", draw: "Pae") -> tuple[dict[str, str], float, float]:
    """Which chain of ``draw`` is which chain of ``reference``, where a model has copies of a chain.

    Returns the assignment (chain of the reference: chain of the draw), its CA RMSD after
    superposition, and the RMSD with the chains as they are named. Chains that are not copies of
    another keep their names. Every assignment of the copies is tried; the one that fits best wins,
    and among those within :data:`SAME_FIT` of the best (a symmetric oligomer fits equally well in
    each of its rotations) the one that renames the fewest chains, then the first in alphabetical
    order. A model with copies and no coordinates for one of the two is refused: a matrix cannot say
    which of two identical chains is which. So are chains with the same residue numbers where the
    names of their residues are not known either: they may be copies.
    """
    if draw.chains != reference.chains or draw.numbers != reference.numbers:
        raise BindviewError(f"{draw.source or 'a draw'} and {reference.source or 'the reference'} do not have the "
                          f"same residues in the same order: they are not of one model")
    order = list(dict.fromkeys(reference.chains))
    same = {c: c for c in order}
    copies = _copies(reference.chains, reference.numbers, reference.resnames)
    if not copies:
        return same, 0.0, 0.0
    if reference.places is None or draw.places is None:
        named = " and ".join(", ".join(g) for g in copies)
        # What their residues are is known from the model alone: without it, equal numbers are all there is.
        what = ("are copies of one another, and a model" if reference.resnames else
                "have the same residue numbers and may be copies of one another (which residues they are is not "
                "known without the model). A model")
        raise BindviewError(
            f"chains {named} {what} is free to name its copies in another order "
            f"in every draw: the values of {draw.source or 'a draw'} cannot be read by chain name. Give the model "
            f"file of every draw with its matrix (a pair of matrix and structure; Boltz's pae_<name>.npz is "
            f"paired with <name>.cif beside it by itself)")
    count = 1
    for g in copies:
        for k in range(2, len(g) + 1):
            count *= k
    if count > MAX_ASSIGNMENTS:
        raise BindviewError(f"{count} ways to assign the copies of this model's chains to each other: more than "
                          f"are tried ({MAX_ASSIGNMENTS})")
    rows = {c: [i for i, x in enumerate(draw.chains) if x == c] for c in order}
    mine = np.concatenate([[i for i, x in enumerate(reference.chains) if x == c] for c in order])
    tried = []
    for choice in product(*(permutations(g) for g in copies)):
        assignment = dict(same)
        for g, chosen in zip(copies, choice):
            assignment.update(zip(g, chosen))
        theirs = np.concatenate([rows[assignment[c]] for c in order])
        changed = sum(1 for c in order if assignment[c] != c)
        tried.append((_fit(reference.places[mine], draw.places[theirs]), changed, tuple(assignment[c] for c in order)))
    best = min(t[0] for t in tried)
    fit, _, names = min((t for t in tried if t[0] <= best + SAME_FIT), key=lambda t: (t[1], t[2]))
    as_named = next(t[0] for t in tried if t[1] == 0)
    return dict(zip(order, names)), fit, as_named


def pae_shown(value: "float | None") -> "int | None":
    """The PAE of a contact as it is printed: whole ångström, half rounded up; ``None`` when there is none."""
    return None if value is None else int(_a_pae(value) + 0.5)


def pae_tag(value: "float | None") -> "tuple[str, bool] | None":
    """What is written on a contact's line for its PAE: ``("PAE 11", bold)``, or ``None`` below the threshold."""
    shown = pae_shown(value)
    if shown is None or shown < PAE_SHOWN:
        return None
    return f"PAE {shown}", shown >= PAE_STRONG


def pae_key(bold: "Iterable[bool]") -> "tuple[tuple[str, bool], ...]":
    """The legend's sample of a PAE tag, for a figure whose tags are written as ``bold`` says (one entry
    for each tag: is it bold?).

    A key shows a thing the way the figure draws it. Where every tag is bold the key is a bold tag, where
    none is a plain one, and where both occur one of each, the plain one first. The samples are tags as
    :func:`pae_tag` writes them (of 7 and of 11 Å), not values of the figure: a key that is bold in a
    figure whose tags are all plain reads as a value, the largest one, that no contact has.
    """
    written = set(bold)
    plain, strong = pae_tag(PAE_SHOWN + 2.0), pae_tag(PAE_STRONG + 1.0)
    assert plain is not None and strong is not None
    return tuple(key for key in (plain, strong) if key[1] in written) or (strong,)


@dataclass(frozen=True)
class Pae:
    """A PAE matrix with the residue each row stands for."""

    matrix: np.ndarray                  # (n, n); [i, j]: expected error of token j when aligned on token i
    chains: tuple[str, ...]
    numbers: tuple[int, ...]
    source: str = ""
    #: The name of every row's residue and a place for it (its CA), where the model the matrix belongs
    #: to was read with it. Needed to tell the copies of a chain apart from one draw to the next
    #: (:func:`match_chains`), and for nothing else.
    resnames: tuple[str, ...] = ()
    places: "np.ndarray | None" = field(default=None, repr=False, compare=False)
    #: The insertion code of every row's residue, where the rows are those of a model file (Boltz's
    #: ``pae_*.npz``): ``H:100`` and ``H:100A`` are then two residues. Empty where the matrix names its
    #: rows itself, by chain and number.
    icodes: tuple[str, ...] = ()
    #: Is a residue that the matrix does not have under its chain name looked up by its number alone?
    #: Settled where the matrix was read with its model: ``True`` for a matrix of one chain whose model
    #: has that chain cut into participants, under other names; ``False`` where the rows are named as
    #: the model names its residues. ``None``, read without a model: where the matrix has one chain.
    by_number: "bool | None" = None

    @classmethod
    def read(cls, path: "str | Path", structure: "Structure | None" = None, placed: bool = True) -> "Pae":
        """Read a PAE matrix from a model's own file.

        AlphaFold 3's ``confidences.json`` (or ``.json.gz``) names its rows
        itself (keys ``pae``, ``token_res_ids``, ``token_chain_ids``). Boltz's
        ``pae_*.npz`` is the matrix and nothing else: the rows are then those of
        ``structure``, the model the matrix belongs to (:func:`tokens_of`), and
        without it the file cannot be read. Either file is refused where it is
        not the matrix of ``structure``: where its rows are not the tokens of
        that model (:meth:`at`).

        The PAE file of the AlphaFold database
        (``AF-<accession>-F1-predicted_aligned_error_v4.json``, or ``.json.gz``;
        a list with one object that holds ``predicted_aligned_error``, and the
        two older forms of it) is a matrix without the names of its rows too:
        n × n for the n residues of the one chain of its model, in the model's
        order. Its rows are the residues of the polymer chain of ``structure``,
        by their chain name and numbers there, and it is refused, with both
        sizes, where that chain has another number of residues or the model has
        several chains. One chain that was cut into participants
        (:meth:`bindview.structure.Structure.split`) is still one chain: its
        parts keep their residue numbers, which run on from one part to the
        next.

        In every file, and in :attr:`matrix`, row i and column j hold the error
        of residue j when the model is aligned on residue i.

        ``placed``: the structure is the matrix's own model (and not another
        draw's, lent for the names of the rows), so its coordinates may be kept
        with the rows.
        """
        p = Path(path)
        if p.suffix == ".npz":
            return cls._read_npz(p, structure, placed)
        try:
            raw = gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, encoding="utf-8")
            with raw as fh:
                data: Any = json.load(fh)
        except (OSError, ValueError) as exc:
            raise BindviewError(f"cannot read a PAE matrix from {p}: {exc}") from exc
        return cls.from_mapping(data, p.name, structure, placed)

    def at(self, structure: "Structure", placed: bool = True) -> "Pae":
        """The same matrix with the residue names and places of its rows from the model it belongs to.

        The rows have to be the tokens of that model (:func:`tokens_of`), chain by chain and number
        by number, in whatever order. A matrix of one chain is also taken for a model that has its
        residues, number for number, under other chain names: a chain that was cut into participants
        keeps its numbers and not its name. Its rows are then found by number alone
        (:attr:`by_number`), and it is given no names and places. Any other matrix is not of this
        model and is refused, with the tokens of both. ``placed=False`` checks, and gives no names
        and places either.
        """
        chains, numbers, _, _, _ = _tokens(structure)
        if sorted(zip(self.chains, self.numbers)) != sorted(zip(chains, numbers)):
            if len(set(self.chains)) == 1 and sorted(self.numbers) == sorted(numbers):
                return replace(self, by_number=True)
            raise BindviewError(
                f"{self.source or 'the PAE matrix'} has {len(self.numbers)} tokens ({_per_chain(self.chains)}), and "
                f"{structure.source or 'the structure'} has {len(numbers)} ({_per_chain(chains)}): the two are not "
                f"of the same model, or its residues are not tokens the way bindview counts them")
        if not placed:
            return replace(self, by_number=False)
        where = {}
        for key in structure.residues():
            atoms = [a for a in structure.residue_atoms(key) if a.element not in ("H", "D")]
            ca = [a.xyz for a in atoms if a.name == "CA"]
            if atoms:
                where[(key.chain, key.number)] = (structure.residue_name(key),
                                                  ca[0] if ca else tuple(np.mean([a.xyz for a in atoms], axis=0)))
        try:
            found = [where[(c, n)] for c, n in zip(self.chains, self.numbers)]
        except KeyError:
            return replace(self, by_number=False)
        return replace(self, resnames=tuple(f[0] for f in found), by_number=False,
                       places=np.array([f[1] for f in found], dtype=float).reshape(-1, 3))

    @classmethod
    def from_mapping(cls, data: Any, source: str = "", structure: "Structure | None" = None,
                     placed: bool = True) -> "Pae":
        """A matrix from what a model's JSON file holds, as :meth:`read` reads it from the file.

        AlphaFold 3's mapping (``pae``, ``token_res_ids`` and ``token_chain_ids``) names its rows
        itself, and is checked against ``structure`` where one is given (:meth:`at`). What the PAE
        file of the AlphaFold database holds does not: its rows are the residues of ``structure``,
        which it cannot be read without. ``source`` is what an error calls the file.
        """
        held = _afdb_matrix(data, source)
        if held is not None:
            return cls._of_one_chain(held, source, structure, placed)
        if not isinstance(data, dict):
            raise BindviewError(
                f"{source or 'the PAE file'} is no PAE file that is read here: neither AlphaFold 3's (an object "
                f"with 'pae' and 'token_res_ids') nor the AlphaFold database's (a list with one object that has "
                f"'predicted_aligned_error'), but a JSON {type(data).__name__}")
        try:
            matrix = np.array(data["pae"], dtype=float)
            numbers = tuple(int(n) for n in data["token_res_ids"])
            chains = tuple(str(c) for c in data.get("token_chain_ids") or [""] * len(numbers))
        except (KeyError, TypeError, ValueError) as exc:
            raise BindviewError(f"{source or 'the PAE file'} has no 'pae' with 'token_res_ids': {exc}") from exc
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(numbers) \
                or len(chains) != len(numbers):
            raise BindviewError(f"{source or 'the PAE file'}: the matrix and its residue numbers do not fit "
                              f"({matrix.shape}, {len(numbers)} residues)")
        made = cls(matrix, chains, numbers, source)
        return made.at(structure, placed) if structure is not None else made

    @classmethod
    def _of_one_chain(cls, matrix: np.ndarray, source: str, structure: "Structure | None",
                      placed: bool = True) -> "Pae":
        """The matrix of the AlphaFold database's file with the residues of its model as its rows.

        The rows are the residues of the polymer chain of ``structure``
        (:meth:`bindview.structure.Structure.polymer_residues`), in the order of the file; what else
        the file holds (water, a ligand that was put into the model) has no row. A model of several
        polymer chains is not the model of such a matrix, unless they are one chain that was cut into
        participants: their residue numbers then run on from one to the next and none comes twice.
        The rows of that chain go under no chain name and are found by number alone.
        """
        what, size = source or "the PAE file", " x ".join(str(n) for n in matrix.shape)
        if structure is None:
            raise BindviewError(f"{what} is the PAE file of the AlphaFold database, a matrix without the names of "
                                f"its rows: the model it belongs to has to be given with it, to say which row is "
                                f"which residue")
        model = structure.source or "the structure"
        parts = {c: structure.polymer_residues(c) for c in dict.fromkeys(a.chain for a in structure.atoms)}
        parts = {c: keys for c, keys in parts.items() if keys}
        inside = {k for keys in parts.values() for k in keys}
        keys = [k for k in structure.residues() if k in inside]
        counted = ", ".join(f"chain {c}: {len(ks)}" for c, ks in parts.items()) or "no polymer chain"
        cut = len(parts) > 1 and all(a.number < b.number for a, b in zip(keys, keys[1:]))
        if len(parts) > 1 and not cut:
            raise BindviewError(
                f"{what} is a matrix of {size}, for the residues of one chain as the AlphaFold database predicts "
                f"it, and {model} has {len(parts)} polymer chains ({counted}), which are not one chain cut into "
                f"parts either (their residue numbers do not run on from one to the next): the two are not of the "
                f"same model")
        if len(keys) != matrix.shape[0]:
            raise BindviewError(f"{what} is a matrix of {size}, and {model} has {len(keys)} residues ({counted}): "
                                f"the two are not of the same model")
        names, places = [], []
        for k in keys:
            atoms = [a for a in structure.residue_atoms(k) if a.element not in ("H", "D")]
            ca = [a.xyz for a in atoms if a.name == "CA"]
            names.append(structure.residue_name(k))
            places.append(ca[0] if ca else tuple(np.mean([a.xyz for a in atoms], axis=0)))
        return cls(matrix, tuple("" if cut else k.chain for k in keys), tuple(k.number for k in keys), source,
                   tuple(names), np.array(places, dtype=float).reshape(-1, 3) if placed else None,
                   tuple(k.icode for k in keys), by_number=cut)

    @classmethod
    def _read_npz(cls, p: Path, structure: "Structure | None", placed: bool = True) -> "Pae":
        """Boltz's ``pae_*.npz``: one array, named ``pae``. An array of another name is another quantity."""
        try:
            with np.load(p, allow_pickle=False) as held:
                arrays = list(held.files)
                matrix = np.array(held["pae"], dtype=float) if "pae" in arrays else None
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as exc:
            raise BindviewError(f"cannot read a PAE matrix from {p}: {exc}") from exc
        if matrix is None:
            for name, (what, why) in _NOT_PAE.items():
                if name in arrays:
                    raise BindviewError(f"{p.name} holds {what}, not a predicted aligned error (its array is named "
                                      f"{name!r}). {why}")
            raise BindviewError(f"{p.name} has no array named 'pae' (it has {', '.join(map(repr, arrays)) or 'none'})")
        if structure is None:
            raise BindviewError(f"{p.name} is a matrix without the names of its rows: the model it belongs to has to "
                              f"be given with it, to say which row is which residue")
        chains, numbers, names, places, icodes = _tokens(structure)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(numbers):
            raise BindviewError(
                f"{p.name} is a matrix of {' x '.join(str(n) for n in matrix.shape)}, and "
                f"{structure.source or 'the structure'} has {len(numbers)} tokens "
                f"({_per_chain(chains)}): the two are not of the same model, or "
                f"its residues are not tokens the way bindview counts them")
        return cls(matrix, chains, numbers, p.name, names, places if placed else None, icodes, by_number=False)

    def _rows(self, key: ResidueKey) -> list[int]:
        """The rows of a residue: by chain and number, and by number alone where the matrix has one chain
        and its model has that chain under other names (a chain that was cut into participants keeps
        its numbers and not its name; :attr:`by_number`). Where the insertion codes of the rows are
        known (:attr:`icodes`), a row is the residue's only with its own."""
        mine = [i for i, n in enumerate(self.numbers)
                if n == key.number and (not self.icodes or self.icodes[i] == key.icode)]
        rows = [i for i in mine if self.chains[i] == key.chain]
        if not rows and self._number_alone:
            rows = mine
        return rows

    @property
    def _number_alone(self) -> bool:
        return len(set(self.chains)) == 1 if self.by_number is None else self.by_number

    def between(self, a: ResidueKey, b: ResidueKey, atoms: "tuple[Sequence[int], int] | None" = None
                ) -> "float | None":
        """The PAE of a pair of residues: the smaller of the two directions; ``None`` if either is not in the file.

        ``atoms``: which of the heavy atoms of ``b`` are meant, by their order in the file, and how many it has.
        A residue that is one token for each of its atoms (a ligand) is then read at those, and the smallest of
        their values is the pair's."""
        ra, rb = self._rows(a), self._rows(b)
        if atoms is not None and len(rb) == atoms[1]:
            rb = [rb[i] for i in atoms[0]]
        if not ra or not rb:
            return None
        # The smaller of the two directions, and of a value and a missing one the value.
        block = np.fmin(self.matrix[np.ix_(ra, rb)], self.matrix[np.ix_(rb, ra)].T)
        if np.isnan(block).all():
            return None
        return float(np.nanmin(block))

    def summary(self, values: Sequence[float]) -> dict[str, float]:
        """Smallest, median and largest of some contact values, for a figure-level statement."""
        ordered = sorted(values)
        return {"min": round(ordered[0], 1), "median": round(float(np.median(ordered)), 1), "max": round(ordered[-1], 1)}


@dataclass(frozen=True)
class PaeDraws:
    """The PAE matrices of several draws of one model of the same complex, read as one.

    The value of a pair of residues is the **median** of its values in the
    draws (each the smaller of the two directions of that draw's matrix).
    Not the largest: the largest of n draws rises with n whatever the model
    knows, and a figure of five draws would look worse than a figure of one.
    Not the smallest, which falls with n. :meth:`each` gives the values draw
    by draw; they travel with the contact (``pae_draws``). A draw is given
    once: the same matrix twice is refused.

    A pair is named by the chains of ``reference``: the structure that is
    drawn (the first draw, if none is given). Where the model has copies of a
    chain, every draw's chains are matched to the reference's first
    (:func:`match_chains`); :attr:`renamed` lists the draws in which a chain is
    another's by name.
    """

    draws: tuple[Pae, ...]
    reference: "Pae | None" = None
    #: For every draw, which of its chains is which chain of the reference; and how well it fits.
    assignments: tuple[dict[str, str], ...] = field(default=(), init=False)
    fits: tuple[tuple[float, float], ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "draws", tuple(self.draws))
        if not self.draws:
            raise BindviewError("no PAE matrix was given")
        first = self.draws[0]
        for other in self.draws[1:]:
            if other.chains != first.chains or other.numbers != first.numbers:
                raise BindviewError(f"{other.source or 'a PAE matrix'} and {first.source or 'the first'} do not have "
                                  f"the same residues in the same order: they are not draws of one model")
        for k, later in enumerate(self.draws):
            for j, earlier in enumerate(self.draws[:k]):
                # (The first row first: two draws of a model differ there, and whole matrices are large.)
                if later.matrix.size and np.array_equal(later.matrix[0], earlier.matrix[0], equal_nan=True) \
                        and np.array_equal(later.matrix, earlier.matrix, equal_nan=True):
                    raise BindviewError(
                        f"draws {j + 1} and {k + 1} ({earlier.source or 'unnamed'} and {later.source or 'unnamed'}) "
                        f"are the same matrix: a draw is given once, or the median over the draws is not that of "
                        f"as many samples of the model as there are draws")
        reference = self.reference if self.reference is not None else first
        if reference.chains != first.chains or reference.numbers != first.numbers:
            raise BindviewError(f"the structure that is drawn and {first.source or 'the draws'} do not have the same "
                              f"residues in the same order: the draws are not of this model")
        matched = [match_chains(reference, d) for d in self.draws]
        object.__setattr__(self, "assignments", tuple(m[0] for m in matched))
        object.__setattr__(self, "fits", tuple((m[1], m[2]) for m in matched))

    @property
    def chains(self) -> tuple[str, ...]:
        return self.draws[0].chains

    @property
    def numbers(self) -> tuple[int, ...]:
        return self.draws[0].numbers

    @property
    def source(self) -> str:
        return f"{len(self.draws)} draws ({self.draws[0].source} to {self.draws[-1].source})" \
            if self.draws[0].source else f"{len(self.draws)} draws"

    @property
    def copies(self) -> list[list[str]]:
        """The chains of the model that are copies of one another, group by group: the same residues,
        by number and by name."""
        ref = self.reference if self.reference is not None else self.draws[0]
        return _copies(ref.chains, ref.numbers, ref.resnames)

    @property
    def renamed(self) -> list[dict[str, Any]]:
        """The draws whose chains were matched under other names than their own: which draw (counted
        from 0, in the order given), which chain of it stands for which of the reference, and the CA
        RMSD of the draw on the reference with that assignment and with the chains as they are named."""
        return [{"draw": k, "source": d.source, "chains": dict(a), "rmsd": round(fit, 2),
                 "rmsd_as_named": round(named, 2)}
                for k, (d, a, (fit, named)) in enumerate(zip(self.draws, self.assignments, self.fits))
                if any(mine != theirs for mine, theirs in a.items())]

    def each(self, a: ResidueKey, b: ResidueKey, atoms: "tuple[Sequence[int], int] | None" = None
             ) -> tuple["float | None", ...]:
        """The PAE of a pair of residues in every draw, in the order the draws were given: of the
        residues that are ``a`` and ``b`` once the draw's chains are matched to the reference's."""
        return tuple(d.between(ResidueKey(names.get(a.chain, a.chain), a.number, a.icode),
                               ResidueKey(names.get(b.chain, b.chain), b.number, b.icode), atoms)
                     for d, names in zip(self.draws, self.assignments))

    def between(self, a: ResidueKey, b: ResidueKey, atoms: "tuple[Sequence[int], int] | None" = None
                ) -> "float | None":
        """The PAE of a pair of residues: the median over the draws; ``None`` if no draw has a value."""
        values = [v for v in self.each(a, b, atoms) if v is not None]
        return float(np.median(values)) if values else None


def _for_model(pae: Pae, structure: "Structure | None") -> Pae:
    """A matrix that was made without a model, for the model it is used with: whether a residue under
    another chain name is found by its number alone (:attr:`Pae.by_number`) is settled for that model.
    Only where the matrix has one chain and the model has its residues, number for number."""
    if structure is None or pae.by_number is not None:
        return pae
    numbers = _tokens(structure)[1]
    return replace(pae, by_number=len(set(pae.chains)) == 1 and sorted(pae.numbers) == sorted(numbers))


def read_pae(given: Any, structure: "Structure | None" = None) -> "Pae | PaeDraws | None":
    """A PAE as a caller hands it in.

    A :class:`Pae` or :class:`PaeDraws`; the path of one file, which is then the matrix of
    ``structure``; or several, which are the draws of one model. A draw is a :class:`Pae`, a pair
    ``(matrix file, its model: a path or a Structure)``, or the path of a matrix alone, whose model
    is then looked for beside it (:func:`model_beside`). A draw that comes without its model lends
    the names of its rows from ``structure`` and has no coordinates of its own, which is enough
    unless the model has copies of a chain (:func:`match_chains`). One of several is not a set of
    draws: a figure of one draw says nothing of a median. A :class:`Pae` that was made without a
    model is not checked against ``structure``; a residue is found in it by its number alone only
    where it has one chain and ``structure`` has its residues, number for number.
    """
    from .structure import Structure, read_structure

    if isinstance(given, Pae):
        return _for_model(given, structure)
    if given is None or isinstance(given, PaeDraws):
        return given
    if isinstance(given, (str, Path)):
        return Pae.read(given, structure)
    if not isinstance(given, Iterable):
        raise BindviewError(f"a PAE is a file, a Pae, or several of them, not {type(given).__name__}")
    draws = []
    for item in given:
        if isinstance(item, Pae):
            draws.append(_for_model(item, structure))
        elif isinstance(item, (str, Path)):
            beside = model_beside(item)
            draws.append(Pae.read(item, read_structure(beside)) if beside is not None
                         else Pae.read(item, structure, placed=False))
        elif isinstance(item, Sequence) and len(item) == 2:
            matrix, model = item
            draws.append(Pae.read(matrix, model if isinstance(model, Structure) else read_structure(model)))
        else:
            raise BindviewError(f"a draw is a matrix file, a Pae, or a pair of matrix file and model, not {item!r}")
    if not draws:
        return None
    if len(draws) == 1:
        return draws[0]
    reference = None
    if structure is not None:
        chains, numbers, names, places, icodes = _tokens(structure)
        if chains == draws[0].chains and numbers == draws[0].numbers:
            reference = Pae(np.zeros((0, 0)), chains, numbers, structure.source or "the structure", names, places,
                            icodes)
    return PaeDraws(tuple(draws), reference)


# ---------------------------------------------------------------------------
# ipSAE: a score of an interface as a whole, from the matrix
# ---------------------------------------------------------------------------

#: The PAE, in ångström, below which a pair of residues counts for ipSAE: the value its authors use for
#: models of AlphaFold 3 and of Boltz, which are the ones whose matrices are read here. (They name 10
#: or 15 Å; a caller can give another.)
IPSAE_CUTOFF = 10.0


def _d0(n: np.ndarray) -> np.ndarray:
    return np.maximum(1.0, 1.24 * np.cbrt(n - 15.0) - 1.8)


def ipsae_d0(n: float) -> float:
    """The length scale of a TM-score over ``n`` residues, in ångström (Yang and Skolnick's d0), as
    ipSAE takes it for one aligned residue: never below 1, which it is for fewer than 27 residues."""
    return float(_d0(np.asarray(float(n))))


@dataclass(frozen=True)
class OneWay:
    """ipSAE in one direction: the model aligned on a residue of one side, scored over the other side."""

    value: float
    #: The aligned residue that gives the value, as chain and number in the matrix; ``None`` where no
    #: pair of residues is below the cut-off, and the value is 0.
    aligned: "tuple[str, int] | None" = None
    #: How many residues of the scored side have a PAE below the cut-off to that residue, and the d0
    #: of that many.
    n0: int = 0
    d0: float = 1.0


@dataclass(frozen=True)
class Ipsae:
    """ipSAE between two sides of a predicted complex, in its two directions (:func:`ipsae`).

    The score has a direction and the two values differ. The reference script also gives the larger
    of the two, and work on designed binders usually quotes the smaller; neither is chosen here.
    """

    forward: OneWay                         # aligned on the first side, scored over the second
    backward: OneWay                        # aligned on the second, scored over the first
    cutoff: float = IPSAE_CUTOFF
    residues: tuple[int, int] = (0, 0)      # of each side: those that entered
    #: Residues of the two sides that did not enter: those without an alpha carbon (a ligand, an ion, a
    #: nucleotide) and, where no structure was given, every residue the model takes atom by atom.
    left_out: int = 0
    same_chain: bool = False                # the matrix has both sides in one chain of the model
    #: Of the residues that entered, those the model takes atom by atom (modified residues): each
    #: entered by the row of its alpha carbon.
    by_alpha_carbon: int = 0

    @property
    def values(self) -> tuple[float, float]:
        return self.forward.value, self.backward.value


def _residue_rows(pae: Pae, residues: "Iterable[ResidueKey]", structure: "Structure | None" = None,
                  names: "dict[str, str] | None" = None) -> tuple[list[int], int, int]:
    """The row of every residue that enters ipSAE: the one that stands for its alpha carbon.

    A standard residue is one row of the matrix. A residue that the model takes atom by atom (a
    modified residue) is one row for each of its heavy atoms, in the order of the model file, and
    enters by the row at the place of its CA among them; that needs ``structure``, the model with the
    residues as they are named here. What has no alpha carbon does not enter: a ligand, and, where the
    structure says so, a residue of one row without one (an ion, a nucleotide). Rows are found as
    :meth:`Pae._rows` finds them; ``names`` says under which chain a chain goes in this matrix (a draw
    that names the copies of a chain otherwise).

    Returns the rows, how many of them are of residues taken atom by atom, and how many residues did
    not enter.
    """
    index: dict[tuple[str, int, str], list[int]] = {}
    for i, (chain, number) in enumerate(zip(pae.chains, pae.numbers)):
        index.setdefault((chain, number, pae.icodes[i] if pae.icodes else ""), []).append(i)
    by_number: dict[tuple[int, str], list[int]] = {}
    if pae._number_alone:
        # A chain that was cut into participants keeps its numbers and not its name.
        by_number = {(number, icode): rows for (_, number, icode), rows in index.items()}
    rows: list[int] = []
    by_alpha_carbon = left_out = 0
    for key in residues:
        chain = names.get(key.chain, key.chain) if names else key.chain
        icode = key.icode if pae.icodes else ""
        found = index.get((chain, key.number, icode)) or by_number.get((key.number, icode)) or []
        if not found:
            continue
        heavy = ([a.name for a in structure.residue_atoms(key) if a.element not in ("H", "D")]
                 if structure is not None else [])
        if len(found) == 1:
            if heavy and "CA" not in heavy:
                left_out += 1
            else:
                rows.append(found[0])
        elif "CA" in heavy and len(heavy) == len(found):
            rows.append(found[heavy.index("CA")])
            by_alpha_carbon += 1
        else:
            left_out += 1
    return rows, by_alpha_carbon, left_out


def _one_way(pae: Pae, aligned: list[int], scored: list[int], cutoff: float,
             chain_of: "dict[int, str] | None" = None) -> OneWay:
    """One direction of ipSAE. ``chain_of`` names the chain of a row where the matrix names none (a chain
    cut into parts is read by number): the chain of the residue that row was asked for."""
    block = pae.matrix[np.ix_(aligned, scored)]
    with np.errstate(invalid="ignore"):
        below = block < cutoff                      # a value that is missing is below nothing
    n = below.sum(axis=1)
    d0 = _d0(n.astype(float))
    with np.errstate(invalid="ignore"):
        terms = np.where(below, 1.0 / (1.0 + (block / d0[:, None]) ** 2), 0.0)
    values = np.divide(terms.sum(axis=1), n, out=np.zeros(len(aligned)), where=n > 0)
    if not (n > 0).any():
        return OneWay(0.0)
    best = int(np.argmax(values))
    row = aligned[best]
    chain = pae.chains[row] or (chain_of or {}).get(row, "")
    return OneWay(float(values[best]), (chain, pae.numbers[row]), int(n[best]), float(d0[best]))


def _ipsae(pae: Pae, first: "list[ResidueKey]", second: "list[ResidueKey]", cutoff: float,
           structure: "Structure | None", names: "dict[str, str] | None" = None) -> Ipsae:
    if not cutoff > 0.0:
        raise BindviewError(f"the PAE cut-off of ipSAE is a positive number of ångström, not {cutoff!r}")
    a, ca_a, out_a = _residue_rows(pae, first, structure, names)
    b, ca_b, out_b = _residue_rows(pae, second, structure, names)
    if not a or not b:
        raise BindviewError(f"{pae.source or 'the PAE matrix'} has no row for a residue with an alpha carbon on "
                          f"{'either side' if not a and not b else 'one side'} of the interface: ipSAE is "
                          f"computed between protein chains and cannot be computed here")
    if set(a) & set(b):
        raise BindviewError("the two sides of an ipSAE share residues: it is a score between two sides")
    same = bool({pae.chains[i] for i in a} & {pae.chains[i] for i in b})
    chain_of = {row: key.chain for key in first + second for row in _residue_rows(pae, [key], structure, names)[0]}
    return Ipsae(_one_way(pae, a, b, cutoff, chain_of), _one_way(pae, b, a, cutoff, chain_of), float(cutoff),
                 (len(a), len(b)),
                 out_a + out_b, same, ca_a + ca_b)


def ipsae(pae: Pae, first: "Iterable[ResidueKey]", second: "Iterable[ResidueKey]",
          cutoff: float = IPSAE_CUTOFF, structure: "Structure | None" = None) -> Ipsae:
    """ipSAE between the residues ``first`` and the residues ``second``, from one PAE matrix.

    The score is Dunbrack's (2025, "Rēs ipSAE loquunt: What's wrong with AlphaFold's ipTM score and how
    to fix it"), in the variant his script reports as plain ipSAE. For one direction, ``first`` to
    ``second``: the model is aligned on a residue i of the first side, and the residues j of the second
    whose PAE to it is below ``cutoff`` (rows of the matrix are aligned residues, columns scored ones)
    are averaged as ``1 / (1 + (PAE[i][j] / d0)**2)``, with d0 from how many they are
    (:func:`ipsae_d0`). The score is the largest such average over the residues i, and 0 where no pair
    is below the cut-off. It runs from 0 to 1. The other direction is another number.

    What enters is every pair of residues of the two sides, not the contacts of a figure, and a residue
    enters by its alpha carbon. A standard residue is one row of the matrix. A modified residue is one
    row for each of its atoms and enters by that of its CA, as in the reference script; ``structure``,
    the model the matrix belongs to, is needed to tell which row that is, and without it such a residue
    is left out. What has no alpha carbon (a ligand, an ion, a nucleotide) does not enter. Residues
    that did not enter are counted in ``left_out``. A side may be several chains, which are then scored
    as one; the published score is of two chains of protein or nucleic acid, and only protein is scored
    here.
    """
    return _ipsae(pae, list(first), list(second), cutoff, structure)


def ipsae_of(pae: "Pae | PaeDraws", first: "Iterable[ResidueKey]", second: "Iterable[ResidueKey]",
             cutoff: float = IPSAE_CUTOFF, structure: "Structure | None" = None) -> tuple[Ipsae, ...]:
    """:func:`ipsae` of one matrix, or of every draw of a model in the order the draws were given:
    the residues are named by the chains of the structure that is drawn, and looked up in each draw
    under the chains that are theirs there (:class:`PaeDraws`)."""
    first, second = list(first), list(second)
    if isinstance(pae, Pae):
        return (_ipsae(pae, first, second, cutoff, structure),)
    return tuple(_ipsae(draw, first, second, cutoff, structure, names)
                 for draw, names in zip(pae.draws, pae.assignments))
