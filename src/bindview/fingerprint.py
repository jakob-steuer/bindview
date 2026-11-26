"""Interaction fingerprints: which residue makes which kind of contact.

A fingerprint is the per-(residue, interaction type) vector of one complex. It
is what a comparison of ligands is computed from, it is the form in which the
content of a figure becomes a number, and it is useful on its own to anyone who
wants a table instead of a drawing.

Three things are fixed so that fingerprints from different runs, machines and
ligands can be compared:

* a **bit** is ``(chain, residue number, insertion code, residue name, type)``
  -- the residue numbering of the structure file, the same one the interaction
  records use;
* bits are **ordered** by chain, residue number, insertion code and then by
  interaction type in the order of :class:`~bindview.interactions.InteractionType`
  -- never by the order records happened to arrive in;
* both the long form (one row per set bit, lossless) and the wide form (one row
  per ligand, one column per bit) **round-trip through CSV**.

A bit carries how many records set it, the shortest reported distance, which
ligand atoms make the contact and, for predicted models, the confidence of its
best-supported record (see :mod:`bindview.confidence`): presence and evidence are
kept as separate numbers and never multiplied into one score.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Sequence

from .confidence import look_of
from .errors import InteractionError
from .interactions import Interaction, InteractionType
from .residues import format_residue_label

if TYPE_CHECKING:  # pragma: no cover
    from .diagram import Diagram

__all__ = ["Bit", "BitValue", "Fingerprint", "FingerprintTable", "ConsensusRow", "TYPE_ORDER", "LONG_COLUMNS"]

#: Interaction types in column order.
TYPE_ORDER: tuple[InteractionType, ...] = tuple(InteractionType)
_TYPE_RANK = {t: i for i, t in enumerate(TYPE_ORDER)}

#: Columns of the long CSV form, in order.
LONG_COLUMNS = ("ligand", "chain", "residue_number", "insertion_code", "residue_name", "interaction_type",
                "count", "min_distance", "plddt", "occupancy", "ligand_atoms")


@dataclass(frozen=True)
class Bit:
    """One fingerprint column: a residue and an interaction type."""

    chain: str
    number: int
    icode: str
    resname: str
    type: InteractionType

    @property
    def residue(self) -> tuple[str, int, str]:
        return (self.chain, self.number, self.icode)

    @property
    def sort_key(self) -> tuple[str, int, str, int]:
        return (self.chain, self.number, self.icode, _TYPE_RANK[self.type])

    def __lt__(self, other: "Bit") -> bool:
        return self.sort_key < other.sort_key

    @property
    def column(self) -> str:
        """The column name of this bit: ``A:GLN570:hbond``. Where the name of the residue is not letters alone
        (``CA2``, ``H2U``, ``5MC``), or its insertion code is not, the name, the number and the code stand
        apart, so that one cannot be read into the other: ``A:CA2:7::metal``."""
        name = self.resname.upper()
        if _letters(name) and (not self.icode or _letters(self.icode)):
            return f"{self.chain}:{name}{self.number}{self.icode}:{self.type.value}"
        return f"{self.chain}:{name}:{self.number}:{self.icode}:{self.type.value}"

    @property
    def residue_label(self) -> str:
        return format_residue_label(self.resname, self.number, self.icode, None)

    @classmethod
    def from_column(cls, text: str) -> "Bit":
        """The bit that a column name stands for (:attr:`column`, in either of its two forms)."""
        parts = text.split(":")
        try:
            if len(parts) == 5:
                chain, resname, number, icode, kind = parts
                return cls(chain, int(number), icode, resname, InteractionType(kind))
            chain, residue, kind = parts
            split = next(i for i, ch in enumerate(residue) if ch.isdigit() or ch == "-")
            digits = residue[split:].rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
            return cls(chain, int(digits), residue[split + len(digits):], residue[:split], InteractionType(kind))
        except (ValueError, StopIteration):
            raise InteractionError(f"cannot read fingerprint column {text!r}; expected 'A:GLN570:hbond', or "
                                   "'A:CA2:7::metal' for a residue whose name is not letters alone") from None


@dataclass(frozen=True)
class BitValue:
    """What is known about one set bit."""

    count: int = 1                              # interaction records behind the bit
    distance: float | None = None               # shortest reported distance, angstrom
    plddt: float | None = None                  # confidence of the best-supported record
    ligand_atoms: tuple[str, ...] = ()          # names of the ligand atoms that make the contact, sorted
    occupancy: float | None = None              # share of an ensemble in which the contact is made

    def merged(self, other: "BitValue") -> "BitValue":
        ds = [d for d in (self.distance, other.distance) if d is not None]
        ps = [p for p in (self.plddt, other.plddt) if p is not None]
        occ = [o for o in (self.occupancy, other.occupancy) if o is not None]
        return BitValue(self.count + other.count, min(ds) if ds else None, max(ps) if ps else None,
                        tuple(sorted(set(self.ligand_atoms) | set(other.ligand_atoms))), max(occ) if occ else None)


def _letters(text: str) -> bool:
    """Is it letters of the alphabet and nothing else?"""
    return text.isascii() and text.isalpha()


def _num(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _opt_float(text: str) -> float | None:
    text = text.strip()
    return float(text) if text else None


@dataclass
class Fingerprint:
    """The set bits of one ligand, with what is known about each."""

    name: str
    bits: dict[Bit, BitValue] = field(default_factory=dict)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_diagram(cls, diagram: "Diagram", name: str | None = None) -> "Fingerprint":
        """The fingerprint of a built :class:`~bindview.diagram.Diagram`.

        Every supplied record counts, including one that could not be attached
        to a ligand atom (it is still a reported contact with that residue).
        """
        conf = diagram.confidence
        bits: dict[Bit, BitValue] = {}
        for r in diagram.interactions:
            rec = r.record
            chain = r.residue.chain if r.residue is not None else rec.chain
            bit = Bit(chain, rec.residue_number, rec.insertion_code, rec.residue_name.upper(), rec.type)
            atoms = tuple(sorted(diagram.ligand.names[i] for i in r.ligand.atoms)) if r.ligand is not None else ()
            plddt = rec.confidence                      # a record may carry its own number
            if plddt is None and conf is not None and atoms:
                plddt = conf.contact(r.residue if r.residue is not None else rec.residue_id, atoms)
            # Stored at the precision the CSV carries, so that a fingerprint equals its round trip.
            one = BitValue(1, None if rec.distance is None else round(rec.distance, 2),
                           None if plddt is None else round(plddt, 1), atoms,
                           None if rec.occupancy is None else round(rec.occupancy, 4))
            bits[bit] = bits[bit].merged(one) if bit in bits else one
        return cls(name or diagram.name or "ligand", dict(sorted(bits.items(), key=lambda kv: kv[0].sort_key)))

    # -- reading -----------------------------------------------------------
    @property
    def columns(self) -> list[Bit]:
        """The set bits, in the fixed order."""
        return sorted(self.bits, key=lambda b: b.sort_key)

    def __len__(self) -> int:
        return len(self.bits)

    def __iter__(self) -> Iterator[Bit]:
        return iter(self.columns)

    def __contains__(self, bit: object) -> bool:
        return bit in self.bits

    def vector(self, columns: Sequence[Bit] | None = None, counts: bool = False) -> list[int]:
        """The fingerprint over ``columns`` (default: its own bits): 1/0, or record counts."""
        cols = self.columns if columns is None else list(columns)
        return [(self.bits[b].count if counts else 1) if b in self.bits else 0 for b in cols]

    def residues(self) -> dict[tuple[str, int, str], list[InteractionType]]:
        """Interaction types per contacted residue, both in the fixed order."""
        out: dict[tuple[str, int, str], list[InteractionType]] = {}
        for b in self.columns:
            out.setdefault(b.residue, []).append(b.type)
        return out

    def tanimoto(self, other: "Fingerprint") -> float:
        """Shared bits over all bits set in either fingerprint (1.0 for two empty ones)."""
        a, b = set(self.bits), set(other.bits)
        return len(a & b) / len(a | b) if (a or b) else 1.0

    def difference(self, other: "Fingerprint") -> tuple[list[Bit], list[Bit], list[Bit]]:
        """``(shared, only in self, only in other)``, each in the fixed order."""
        a, b = set(self.bits), set(other.bits)

        def ordered(bits: Iterable[Bit]) -> list[Bit]:
            return sorted(bits, key=lambda x: x.sort_key)
        return ordered(a & b), ordered(a - b), ordered(b - a)

    # -- CSV (long form) ---------------------------------------------------
    def rows(self) -> list[dict[str, str]]:
        """One row for each bit, as plain data."""
        out = []
        for b in self.columns:
            c = self.bits[b]
            out.append({
                "ligand": self.name, "chain": b.chain, "residue_number": str(b.number), "insertion_code": b.icode,
                "residue_name": b.resname, "interaction_type": b.type.value, "count": str(c.count),
                "min_distance": _num(c.distance, 2), "plddt": _num(c.plddt, 1),
                "occupancy": _num(c.occupancy, 4), "ligand_atoms": " ".join(c.ligand_atoms),
            })
        return out

    def to_csv(self, path: "str | Path | None" = None) -> str:
        """Long-form CSV: one row per set bit. Written to ``path`` when given; always returned."""
        return _write_rows(LONG_COLUMNS, self.rows(), path)

    @classmethod
    def from_csv(cls, source: "str | Path") -> "Fingerprint":
        """Read one fingerprint back from :meth:`to_csv` output (text or a path)."""
        table = FingerprintTable.from_csv(source)
        if len(table.fingerprints) != 1:
            raise InteractionError(f"expected one fingerprint, found {len(table.fingerprints)}; "
                                   "use FingerprintTable.from_csv")
        return table.fingerprints[0]


@dataclass(frozen=True)
class ConsensusRow:
    """One residue of a set: how many ligands contact it, and how.

    The numbers live on contact records. ``contacts`` holds one
    :class:`~bindview.interactions.Interaction` per interaction type made with
    the residue, and its ``occupancy`` is the share of the set's ligands that
    make it: the same field, drawn through the same channel, that carries the
    share of frames when the ensemble is a trajectory. ``occupancy`` of the row
    itself is the share of ligands that contact the residue in any way, which
    the per-type numbers cannot give (one ligand can make several types).
    """

    chain: str
    number: int
    icode: str
    resname: str
    n_ligands: int                                  # ligands with any contact to the residue
    n_total: int                                    # ligands in the set
    contacts: tuple[Interaction, ...]               # one per interaction type, in the fixed order
    ligands: tuple[str, ...]                        # names of the contacting ligands, in set order
    #: Contacting ligands by the confidence of their contact with the residue:
    #: unqualified or pLDDT >= 70, 50-70, below 50. ``None``: the set carries no confidence.
    levels: tuple[int, int, int] | None = None

    @property
    def residue(self) -> tuple[str, int, str]:
        return (self.chain, self.number, self.icode)

    @property
    def label(self) -> str:
        return format_residue_label(self.resname, self.number, self.icode, None)

    @property
    def occupancy(self) -> float:
        """Share of the set's ligands that contact the residue in any way."""
        return self.n_ligands / self.n_total if self.n_total else 0.0

    @property
    def fraction(self) -> float:
        return self.occupancy

    @property
    def types(self) -> tuple[tuple[InteractionType, int], ...]:
        """``(type, ligands making it)`` in the fixed order -- read off the contact records."""
        return tuple((c.type, int(round((c.occupancy or 0.0) * self.n_total))) for c in self.contacts)

    @property
    def n_confident(self) -> int | None:
        return self.levels[0] if self.levels is not None else None

    def as_dict(self) -> dict[str, Any]:
        return {"residue": self.label, "chain": self.chain, "residue_number": self.number,
                "n_ligands": self.n_ligands, "frac": round(self.fraction, 3),
                "types": sorted(t.value for t, _ in self.types),
                "type_counts": {t.value: n for t, n in self.types}, "ligands": list(self.ligands),
                "n_confident": self.n_confident,
                "contacts": [{"type": c.type.value, "occupancy": round(c.occupancy or 0.0, 4),
                              "confidence": c.confidence, "distance": c.distance} for c in self.contacts]}


@dataclass
class FingerprintTable:
    """The fingerprints of a set of ligands over one residue numbering."""

    fingerprints: list[Fingerprint]

    def __post_init__(self) -> None:
        names = [f.name for f in self.fingerprints]
        if len(set(names)) != len(names):
            raise InteractionError(f"ligand names must be unique in a fingerprint table: {names}")

    @property
    def names(self) -> list[str]:
        """The names of the ligands, in order."""
        return [f.name for f in self.fingerprints]

    @property
    def columns(self) -> list[Bit]:
        """The union of all set bits, in the fixed order."""
        union: set[Bit] = set()
        for f in self.fingerprints:
            union |= set(f.bits)
        return sorted(union, key=lambda b: b.sort_key)

    def matrix(self, counts: bool = False) -> list[list[int]]:
        """The bits as a matrix, one row a ligand; with ``counts`` the number of records."""
        cols = self.columns
        return [f.vector(cols, counts) for f in self.fingerprints]

    def consensus(self) -> list[ConsensusRow]:
        """Per residue: how many ligands contact it and with which types.

        Rows are ordered by the number of ligands (most first), then by chain
        and residue number. Each row's contact records carry, per interaction
        type, the share of ligands that make it (``occupancy``), the shortest
        distance over the set, and the confidence of the best-supported ligand.
        """
        n = len(self.fingerprints)
        has_conf = any(v.plddt is not None for f in self.fingerprints for v in f.bits.values())
        slots: dict[tuple[str, int, str], dict[str, Any]] = {}
        for f in self.fingerprints:
            best: dict[tuple[str, int, str], float | None] = {}
            for b in f.columns:
                value = f.bits[b]
                slot = slots.setdefault(b.residue, {"resname": b.resname, "types": {}, "ligands": [],
                                                    "levels": [0, 0, 0]})
                kind = slot["types"].setdefault(b.type, {"n": 0, "distance": None, "plddt": None})
                kind["n"] += 1
                if value.distance is not None:
                    kind["distance"] = value.distance if kind["distance"] is None else min(kind["distance"],
                                                                                           value.distance)
                if value.plddt is not None:
                    kind["plddt"] = value.plddt if kind["plddt"] is None else max(kind["plddt"], value.plddt)
                if f.name not in slot["ligands"]:
                    slot["ligands"].append(f.name)
                if b.residue not in best:
                    best[b.residue] = value.plddt
                elif value.plddt is not None:
                    current = best[b.residue]
                    best[b.residue] = value.plddt if current is None else max(current, value.plddt)
            for res, top in best.items():
                slots[res]["levels"][look_of(top).level] += 1
        rows = []
        for res, slot in slots.items():
            contacts = tuple(
                Interaction(type=t, residue_number=res[1], residue_name=slot["resname"], chain=res[0],
                            insertion_code=res[2], distance=slot["types"][t]["distance"],
                            occupancy=slot["types"][t]["n"] / n, confidence=slot["types"][t]["plddt"])
                for t in TYPE_ORDER if t in slot["types"])
            levels = (slot["levels"][0], slot["levels"][1], slot["levels"][2]) if has_conf else None
            rows.append(ConsensusRow(res[0], res[1], res[2], slot["resname"], len(slot["ligands"]), n, contacts,
                                     tuple(slot["ligands"]), levels))
        rows.sort(key=lambda r: (-r.n_ligands, r.chain, r.number, r.icode))
        return rows

    def contacts(self) -> list[Interaction]:
        """The set as contact records: one per (residue, interaction type), in the fixed column
        order, with ``occupancy`` the share of ligands that make the contact."""
        rows = {r.residue: r for r in self.consensus()}
        return [c for res in sorted(rows) for c in rows[res].contacts]

    # -- CSV ---------------------------------------------------------------
    def to_csv(self, path: "str | Path | None" = None, form: str = "long", counts: bool = False) -> str:
        """``form="long"``: one row per (ligand, set bit), lossless. ``form="wide"``: one row per
        ligand and one column per bit of the union, holding 1/0 (or record counts)."""
        if form == "long":
            rows = [row for f in self.fingerprints for row in f.rows()]
            # A ligand with no contacts still belongs to the set: keep it as a row with no bit.
            for f in self.fingerprints:
                if not f.bits:
                    rows.append({**{k: "" for k in LONG_COLUMNS}, "ligand": f.name})
            return _write_rows(LONG_COLUMNS, rows, path)
        if form != "wide":
            raise ValueError("form must be 'long' or 'wide'")
        cols = self.columns
        header = ["ligand"] + [b.column for b in cols]
        rows = [dict(zip(header, [f.name] + [str(v) for v in f.vector(cols, counts)])) for f in self.fingerprints]
        return _write_rows(header, rows, path)

    @classmethod
    def from_csv(cls, source: "str | Path") -> "FingerprintTable":
        """Read either CSV form back (text or a path). The wide form carries presence
        (or counts) only; distances, confidence and atoms are in the long form."""
        text = _read_text(source)
        reader = csv.DictReader(io.StringIO(text))
        header = reader.fieldnames or []
        if not header or header[0] != "ligand":
            one_line = isinstance(source, str) and "\n" not in source.strip() and len(source) <= 200
            raise InteractionError("not a fingerprint CSV: the first column must be 'ligand'"
                                   + (f" (and {source!r} is not the name of a file that is there)" if one_line else ""))
        fps: dict[str, Fingerprint] = {}
        if "interaction_type" in header:
            for row in reader:
                fp = fps.setdefault(row["ligand"], Fingerprint(row["ligand"]))
                if not row.get("interaction_type"):
                    continue
                try:
                    bit = Bit(row["chain"], int(row["residue_number"]), row["insertion_code"], row["residue_name"],
                              InteractionType(row["interaction_type"]))
                    fp.bits[bit] = BitValue(int(row["count"] or 1), _opt_float(row.get("min_distance", "")),
                                            _opt_float(row.get("plddt", "")),
                                            tuple((row.get("ligand_atoms") or "").split()),
                                            _opt_float(row.get("occupancy") or ""))
                except (KeyError, ValueError) as exc:
                    raise InteractionError(f"bad fingerprint row {row!r}: {exc}") from exc
        else:
            bits = [Bit.from_column(c) for c in header[1:]]
            for row in reader:
                fp = fps.setdefault(row["ligand"], Fingerprint(row["ligand"]))
                for name, b in zip(header[1:], bits):
                    cell = (row[name] or "").strip()         # (a row that ends early has none)
                    if not cell.isdecimal():
                        if cell:
                            raise InteractionError(f"{cell!r} in column {name!r} of ligand {row['ligand']!r} is not a "
                                                   "count: a cell of the wide form holds 0, 1 or how many records")
                        continue
                    if int(cell):
                        fp.bits[b] = BitValue(int(cell))
        return cls(list(fps.values()))


def _read_text(source: "str | Path") -> str:
    """The text of a table. A ``Path`` is a file; a string is one where it is the name of a file that is there,
    whatever it is called, and the table itself where it is not."""
    if isinstance(source, Path) or _names_a_file(str(source)):
        try:
            return Path(source).read_text(encoding="utf-8").removeprefix("\ufeff")
        except (OSError, ValueError) as exc:             # (ValueError: it is not text)
            raise InteractionError(f"cannot read {source}: {exc}") from exc
    return str(source).removeprefix("\ufeff")


def _names_a_file(text: str) -> bool:
    """Is it the name of a file that is there? No table is: a name has no line break in it."""
    try:
        return "\n" not in text and Path(text).is_file()
    except (OSError, ValueError):                        # too long for a name, or with a character no name has
        return False


def _write_rows(header: Sequence[str], rows: Sequence[Mapping[str, str]], path: "str | Path | None") -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(header), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    text = buf.getvalue()
    if path is not None:
        try:
            Path(path).write_text(text, encoding="utf-8")
        except OSError as exc:
            raise InteractionError(f"cannot write {path}: {exc}") from exc
    return text
