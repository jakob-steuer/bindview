"""The interaction record: the one data structure callers hand to this library.

A caller that already has interactions (from PLIP, from a database, from any
other detector) passes a list of :class:`Interaction` and never touches
detection. The record is deliberately small and flat so it maps one-to-one onto
a table row; ``testdata/*/interactions.tsv`` is exactly this shape.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import InteractionError

__all__ = ["InteractionType", "Interaction", "load_interactions", "parse_interactions", "coerce_interactions",
           "check_confidence_scale"]


class InteractionType(str, Enum):
    """The interaction classes the diagram knows how to draw."""

    HYDROPHOBIC = "hydrophobic"
    HBOND = "hbond"
    SALTBRIDGE = "saltbridge"
    PI_STACKING = "pi_stacking"
    PI_CATION = "pi_cation"
    HALOGEN = "halogen"
    METAL = "metal"
    WATER_BRIDGE = "water_bridge"
    #: A bond, and no contact: the ligand is bound to the residue (:mod:`bindview.covalent`).
    COVALENT = "covalent"

    def __str__(self) -> str:
        return self.value

    @property
    def label(self) -> str:
        """The kind in words, as the key of a figure has it."""
        return _LABELS[self]


_LABELS = {
    InteractionType.HYDROPHOBIC: "Hydrophobic contact",
    InteractionType.HBOND: "Hydrogen bond",
    InteractionType.SALTBRIDGE: "Salt bridge",
    InteractionType.PI_STACKING: "π–π stacking",
    InteractionType.PI_CATION: "π–cation",
    InteractionType.HALOGEN: "Halogen bond",
    InteractionType.METAL: "Metal coordination",
    InteractionType.WATER_BRIDGE: "Water bridge",
    InteractionType.COVALENT: "Covalent bond",
}

# Spellings seen in the wild, normalised to the canonical value.
_ALIASES = {
    "hydrophobic": "hydrophobic", "hydrophobic_contact": "hydrophobic", "hydroph": "hydrophobic",
    "hbond": "hbond", "h_bond": "hbond", "hydrogen_bond": "hbond", "hydrogenbond": "hbond",
    "saltbridge": "saltbridge", "salt_bridge": "saltbridge", "ionic": "saltbridge",
    "pi_stacking": "pi_stacking", "pistacking": "pi_stacking", "pi_stack": "pi_stacking", "pistack": "pi_stacking",
    "pi_pi": "pi_stacking",
    "pi_cation": "pi_cation", "pication": "pi_cation", "cation_pi": "pi_cation",
    "halogen": "halogen", "halogen_bond": "halogen", "halogenbond": "halogen", "xbond": "halogen",
    "metal": "metal", "metal_complex": "metal", "metal_coordination": "metal",
    "water_bridge": "water_bridge", "waterbridge": "water_bridge", "water": "water_bridge",
    "covalent": "covalent", "covalent_bond": "covalent", "covale": "covalent",
}


def _as_type(value: Any) -> InteractionType:
    if isinstance(value, InteractionType):
        return value
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    canon = _ALIASES.get(key)
    if canon is None:
        raise InteractionError(
            f"unknown interaction type {value!r}; expected one of {', '.join(t.value for t in InteractionType)}")
    return InteractionType(canon)


#: The one word a table writes for a value that is not given that is also a name: of an atom (a heme has
#: a nitrogen called so), of a residue (sodium) and of a chain.
NOT_GIVEN_OR_A_NAME = "NA"


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s not in ("?", ".", "None", "none", "NA", "nan", "NaN") else None


def _opt_name(value: Any) -> str | None:
    """The name of an atom, a residue or a chain, or None where none is given. ``NA`` is kept: a heme has a
    nitrogen of that name, sodium is called so, and a large structure has a chain of that name. A table can also
    write it for a name that is not given; whether the molecule has an atom, or the structure a chain, called so
    says which it is (:func:`bindview.resolve.resolve_interactions`)."""
    s = None if value is None else str(value).strip()
    return s if s == NOT_GIVEN_OR_A_NAME else _opt_str(value)


def _opt_float(value: Any, field_name: str) -> float | None:
    s = _opt_str(value)
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        raise InteractionError(f"{field_name} is not a number: {value!r}") from None


def _opt_bool(value: Any, field_name: str) -> bool | None:
    if isinstance(value, bool):
        return value
    s = _opt_str(value)
    if s is None:
        return None
    low = s.lower()
    if low in ("true", "t", "1", "yes", "y"):
        return True
    if low in ("false", "f", "0", "no", "n"):
        return False
    raise InteractionError(f"{field_name} is not a boolean: {value!r}")


def _names(value: Any) -> tuple[str, ...]:
    """Several names: a sequence of them, or text with commas or semicolons between them. What stands for a
    value that is not given in any other column does so here: an empty cell, ``?``, ``.``, the ``nan`` of a
    data frame. ``NA`` alone is kept, as it is for one name (:func:`_opt_name`)."""
    if isinstance(value, (str, float)) or value is None:
        text = _opt_name(value)
        return tuple(p for p in (part.strip() for part in text.replace(";", ",").split(",")) if p) if text else ()
    return tuple(str(v) for v in value)


def _floats(value: Any, field_name: str) -> tuple[float, ...]:
    """Several numbers: a sequence, or text with commas, semicolons or blanks between them."""
    if value is None:
        return ()
    parts = value.replace(";", ",").replace(" ", ",").split(",") if isinstance(value, str) else list(value)
    try:
        return tuple(float(v) for v in parts if str(v).strip())
    except (TypeError, ValueError):
        raise InteractionError(f"{field_name} are not numbers: {value!r}") from None


@dataclass(frozen=True)
class Interaction:
    """One protein-ligand interaction.

    In every field an empty value, ``?``, ``.``, ``None`` and ``nan`` say that
    the value is not given, and so does ``NA``, which is what R writes. Where
    ``NA`` can also be a name (of an atom, of a chain) it is that name if the
    structure has an atom or a chain called so, and a value that is not given
    otherwise.

    Parameters
    ----------
    type:
        One of :class:`InteractionType` (a plain string such as ``"hbond"`` is accepted).
    residue_number, residue_name, chain:
        The protein residue. ``chain`` may be empty when the structure has a
        single chain carrying that residue number.
    protein_atom, ligand_atom:
        Atom names as they appear in the structure file. Leave ``None`` for
        interactions made by a *group* -- an aromatic ring (pi-stacking,
        pi-cation), a charged group (salt bridge) -- or when the detector did
        not report one (PLIP's metal rows). The group is then resolved from the
        structure geometry using ``distance``; see :mod:`bindview.resolve`.
    ligand_atoms, protein_atoms:
        Optional explicit group membership (ring atoms, the atoms of a charged
        group). When given, nothing is inferred for that side.
    distance:
        In angstrom. Atom-atom for hydrophobic, H-bond, halogen and metal
        contacts; centre-centre for ring and charge interactions.
    angle:
        In degrees, as reported by the detector (donor angle, ring-plane angle, ...).
    ligand_is_donor:
        Direction of an H-bond or water bridge: ``True`` when the ligand atom
        donates. ``None`` when not applicable or unknown.
    beyond_contact_set:
        ``True`` when the residue lies outside the caller's contact definition
        (e.g. a pi interaction measured centroid-to-centroid to a residue with
        no heavy atom within the pocket cut-off). Such contacts are drawn in a
        visibly different, muted way.
    water:
        For water bridges, ``"chain:number"`` of the bridging water, if known.
    occupancy:
        The share of an *ensemble* in which this contact is made, from 0 to 1:
        the fraction of frames of a trajectory, of the ligands of a series, of
        the models of an NMR bundle, of the poses of a docking run. The number
        does not say which; the caller names the ensemble (``ensemble=`` on
        the diagram) and the legend prints that name. ``None`` for a contact
        observed in one structure. bindview reads no trajectories: occupancy
        arrives as data, like every other field here.
    confidence:
        How much the model is to be believed about *this contact*, on the scale
        of pLDDT (0 to 100, higher is better). ``None`` leaves it to be derived
        where the model allows: the lower of the ligand atom's pLDDT and the
        residue's (see :mod:`bindview.confidence`). A value given here is used
        as it stands, so a caller can let a contact carry a number of its own.
    predicted:
        ``True`` for a contact that was neither observed nor detected in the
        structure but inferred: a water bridge through a *predicted*
        solvent site. Such a record is drawn in a visibly different way and
        labelled as predicted wherever it appears.
    """

    type: InteractionType
    residue_number: int
    residue_name: str
    chain: str = ""
    protein_atom: str | None = None
    ligand_atom: str | None = None
    distance: float | None = None
    angle: float | None = None
    ligand_is_donor: bool | None = None
    beyond_contact_set: bool = False
    ligand_atoms: tuple[str, ...] = ()
    protein_atoms: tuple[str, ...] = ()
    insertion_code: str = ""
    water: str | None = None
    occupancy: float | None = None
    confidence: float | None = None
    predicted: bool = False
    #: Predicted aligned error between the two residues of the contact, in ångström: what a
    #: structure-prediction model says about how well it places one relative to the other, which
    #: is what the reliability of a contact across an interface means. Carried into the graph, and
    #: drawn with the contact: as a glow behind its line or as a number on it (``pae_colors``).
    pae: float | None = None
    #: Where ``pae`` is the median over several draws of a model: the value of each draw, in the
    #: order the draws were given. Empty for a contact with one value and no draws behind it.
    pae_draws: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _as_type(self.type))
        try:
            object.__setattr__(self, "residue_number", int(self.residue_number))
        except (TypeError, ValueError):
            raise InteractionError(f"residue_number is not an integer: {self.residue_number!r}") from None
        name = _opt_name(self.residue_name)          # (it has to be given: sodium's is NA)
        if name is None:
            raise InteractionError("residue_name is required")
        object.__setattr__(self, "residue_name", name.upper())
        object.__setattr__(self, "chain", _opt_name(self.chain) or "")
        object.__setattr__(self, "protein_atom", _opt_name(self.protein_atom))
        object.__setattr__(self, "ligand_atom", _opt_name(self.ligand_atom))
        object.__setattr__(self, "distance", _opt_float(self.distance, "distance"))
        object.__setattr__(self, "angle", _opt_float(self.angle, "angle"))
        object.__setattr__(self, "ligand_is_donor", _opt_bool(self.ligand_is_donor, "ligand_is_donor"))
        object.__setattr__(self, "beyond_contact_set", bool(_opt_bool(self.beyond_contact_set, "beyond_contact_set")))
        object.__setattr__(self, "ligand_atoms", _names(self.ligand_atoms))
        object.__setattr__(self, "protein_atoms", _names(self.protein_atoms))
        object.__setattr__(self, "insertion_code", _opt_str(self.insertion_code) or "")
        object.__setattr__(self, "water", _opt_str(self.water))
        if self.distance is not None and self.distance < 0:
            raise InteractionError(f"distance must not be negative: {self.distance}")
        object.__setattr__(self, "predicted", bool(_opt_bool(self.predicted, "predicted")))
        object.__setattr__(self, "occupancy", _opt_float(self.occupancy, "occupancy"))
        object.__setattr__(self, "confidence", _opt_float(self.confidence, "confidence"))
        object.__setattr__(self, "pae", _opt_float(self.pae, "pae"))
        if self.pae is not None and self.pae < 0:
            raise InteractionError(f"pae is an error in ångström and cannot be negative: {self.pae:g}")
        object.__setattr__(self, "pae_draws", _floats(self.pae_draws, "pae_draws"))
        if any(v < 0 for v in self.pae_draws):
            raise InteractionError("pae_draws are errors in ångström and cannot be negative")
        if self.occupancy is not None and not 0.0 <= self.occupancy <= 1.0:
            raise InteractionError(f"occupancy is a fraction between 0 and 1, not {self.occupancy:g} "
                                   "(divide a percentage by 100)")
        if self.confidence is not None and not 0.0 <= self.confidence <= 100.0:
            raise InteractionError(f"confidence is on the pLDDT scale, 0 to 100, not {self.confidence:g}")

    # -- construction ------------------------------------------------------
    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "Interaction":
        """Build a record from a dict-like row (a TSV line, a database row)."""
        def pick(*keys: str) -> Any:
            for k in keys:
                if k in row and row[k] is not None:
                    return row[k]
            return None

        typ = pick("type", "interaction_type")
        if typ is None:
            raise InteractionError("row has no 'type' / 'interaction_type'")
        number = pick("residue_number", "resnr", "resnum", "residue_id")
        if number is None or str(number).strip() == "":
            raise InteractionError("row has no 'residue_number'")
        return cls(
            type=typ,
            residue_number=number,
            residue_name=pick("residue_name", "restype", "resname") or "",
            chain=pick("chain", "reschain", "chain_id") or "",
            protein_atom=pick("protein_atom", "protein_atom_name"),
            ligand_atom=pick("ligand_atom", "ligand_atom_name"),
            distance=pick("distance", "dist"),
            angle=pick("angle"),
            ligand_is_donor=pick("ligand_is_donor"),
            beyond_contact_set=pick("beyond_contact_set") or False,
            ligand_atoms=pick("ligand_atoms") or (),
            protein_atoms=pick("protein_atoms") or (),
            insertion_code=pick("insertion_code", "icode") or "",
            water=pick("water"),
            occupancy=pick("occupancy", "frequency", "persistence"),
            confidence=pick("confidence", "contact_confidence"),
            predicted=pick("predicted") or False,
            pae=pick("pae", "predicted_aligned_error"),
            pae_draws=pick("pae_draws") or (),
        )

    def with_(self, **changes: Any) -> "Interaction":
        """The same record with some of its fields changed."""
        return replace(self, **changes)

    # -- convenience -------------------------------------------------------
    @property
    def residue_label(self) -> str:
        """``Gln712`` -- three-letter name in title case plus number."""
        return f"{self.residue_name.capitalize()}{self.residue_number}{self.insertion_code}"

    @property
    def residue_id(self) -> tuple[str, int, str]:
        """``(chain, number, insertion code)`` of the residue."""
        return (self.chain, self.residue_number, self.insertion_code)


def check_confidence_scale(values: Iterable["float | None"], what: str = "The contacts") -> None:
    """Refuse a set of contact confidences that cannot be on the 0-100 scale they are documented on.

    A record's ``confidence`` is on the scale of pLDDT, 0 to 100, and is used as it stands. One
    value says nothing about its scale; a whole set none of which exceeds 1 does: on 0-100 a model
    has no value below 1, so these are on 0-1 (what Boltz writes into ``plddt_*.npz``), and drawn
    as they stand every contact would look unreliable, in a figure that still looks like a result.
    Nothing is converted: the scale is the caller's to put right.
    """
    vals = [float(v) for v in values if v is not None]
    if not vals or max(vals) > 1.0 or max(vals) <= 0.0:
        return
    lo, hi = min(vals), max(vals)
    raise InteractionError(
        f"{what} carry a confidence of {lo:.3g}{'' if lo == hi else f' to {hi:.3g}'}: nothing above 1. "
        f"'confidence' is on the scale of pLDDT, 0 to 100, on which a model has no value below 1, so these "
        f"look like the 0-1 scale that Boltz writes into plddt_*.npz. The scale is not guessed and nothing is "
        f"multiplied for you: give them times 100, or leave them out and let the model file speak (bindview reads "
        f"pLDDT from a file that declares it, and Boltz's does, on 0-100)")


def coerce_interactions(items: Iterable[Interaction | Mapping[str, Any]]) -> list[Interaction]:
    """Accept records or dict-like rows and return records."""
    out: list[Interaction] = []
    for n, item in enumerate(items):
        if isinstance(item, Interaction):
            out.append(item)
        elif isinstance(item, Mapping):
            try:
                out.append(Interaction.from_mapping(item))
            except InteractionError as exc:
                raise InteractionError(f"interaction #{n + 1}: {exc}") from None
        else:
            raise InteractionError(f"interaction #{n + 1}: expected Interaction or mapping, got {type(item).__name__}")
    return out


#: What may stand between two fields of a table. Where none is named, it is the one the header row has most of,
#: and the earlier of two that are as frequent. A spreadsheet program writes semicolons where its list separator is one.
_DELIMITERS = ("\t", ",", ";")


def parse_interactions(text: str, delimiter: str | None = None) -> list[Interaction]:
    """Parse delimited text with a header row.

    ``delimiter`` is the character between two fields. Left out, it is found in the header row: a tab, a comma
    or a semicolon. Numbers are written with a decimal point.

    A file with a header and no rows is a valid, *empty* result: it means the
    complex was analysed and nothing was found.
    """
    text = text.removeprefix("\ufeff")       # the byte-order mark a spreadsheet program puts before the header
    if not text.strip():
        raise InteractionError("interaction table is empty (no header row)")
    if delimiter is None:
        head = text.splitlines()[0]
        delimiter = max(_DELIMITERS, key=head.count)        # max keeps the first of equals
    elif len(delimiter) != 1:
        raise InteractionError(f"delimiter is one character, not {delimiter!r}")
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames or not ({"type", "interaction_type"} & {f.strip() for f in reader.fieldnames}):
        raise InteractionError("interaction table needs a 'type' or 'interaction_type' column")
    out: list[Interaction] = []
    for n, row in enumerate(reader, start=2):
        clean = {(k or "").strip(): v for k, v in row.items()}
        if not any((v or "").strip() for v in clean.values()):
            continue
        try:
            out.append(Interaction.from_mapping(clean))
        except InteractionError as exc:
            raise InteractionError(f"line {n}: {exc}") from None
    return out


def load_interactions(path: str | Path) -> list[Interaction]:
    """Read an interaction table (``.tsv`` / ``.csv``) from disk."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InteractionError(f"cannot read {p}: it is not UTF-8 text (it has the byte "
                               f"0x{exc.object[exc.start]:02x}); save it as UTF-8") from exc
    except OSError as exc:
        raise InteractionError(f"cannot read {p}: {exc}") from exc
    try:
        return parse_interactions(text)
    except InteractionError as exc:
        raise InteractionError(f"{p}: {exc}") from None
