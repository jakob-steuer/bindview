"""How much of a predicted model to believe.

Every 2D interaction diagram tool was written for crystal structures and draws
each contact as equally real. A predicted complex is not that kind of evidence:
AlphaFold-style models say, atom by atom, how confident they are (pLDDT), and a
contact between a poorly placed ligand atom and a poorly placed side chain is a
different claim from one between two well placed ones.

What this module does, and where it stops:

* It **reads** confidence; it never computes one. Per-atom pLDDT comes from the
  B-factor column of the model file, and only when the file *declares* that the
  column is a pLDDT (``_ma_qa_metric``), or when the caller asserts it or hands
  values in. The value range is never used as a hint: in a crystal structure
  the same column is a temperature factor, where high means worse.
* A residue's confidence is the mean over its atom records -- the same atoms
  the interactions are keyed on, so no second residue numbering is involved.
* A **contact** is as believable as its weaker end: the lower of the ligand
  atom's pLDDT and the residue's. Not their mean.
* Values are binned with AlphaFold's own thresholds (90 / 70 / 50). The bins
  decide how strongly something is drawn; they do not change which interactions
  exist, how they are typed, or how long their lines are. Confidence qualifies
  the evidence; it is not folded into the chemistry.
* Model-level numbers (ipTM, pTM, ranking score) and the verdict "do not trust
  this model" belong to the caller. They are displayed, not derived.
* pLDDT is written on **two scales**: 0 to 100 (AlphaFold; the B-factor column
  of a model file, Boltz's included) and 0 to 1 (Boltz's ``plddt_*.npz``, among
  others). Everything here is on 0-100. The scale of what is handed in is never
  guessed and never converted on the quiet, because the wrong guess is silent
  both ways: 0-1 values read as 0-100 put every residue below 50, and a
  hundredfold applied to values already on 0-100 puts every residue above 90.
  Values that cannot be on the scale they are said to be on are refused
  (:func:`plddt_factor`), and a caller can state the scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .errors import BindviewError, StructureError
from .structure import ResidueKey, Structure

__all__ = ["Confidence", "ModelConfidence", "Look", "band_of", "look_of", "BANDS", "BAND_LABELS", "LOOKS",
           "PLAIN", "POCKET_CUTOFF", "occupancy_factors", "OCCUPANCY_SAMPLES", "OCCUPANCY_STEPS",
           "SHARE_WRITTEN_BELOW", "SCALES", "plddt_factor"]

#: Bands, best first, with their lower pLDDT bounds (AlphaFold's convention).
BANDS: tuple[tuple[str, float], ...] = (("very_high", 90.0), ("confident", 70.0), ("low", 50.0),
                                        ("very_low", float("-inf")))
BAND_LABELS = {"very_high": "pLDDT ≥ 90", "confident": "pLDDT 70–90", "low": "pLDDT 50–70",
               "very_low": "pLDDT < 50"}
#: Residues with a heavy atom this close (angstrom) to the ligand form the pocket.
POCKET_CUTOFF = 4.5


#: The scales a pLDDT can be stated to be on. bindview draws on the first.
SCALES = ("0-100", "0-1")
#: The atoms of a residue have values of their own when two of them differ by more than this (on 0-100).
#: A model that writes one value per residue writes it on every atom, to the last digit.
ATOM_SPREAD = 0.01


def gives_atoms(groups: "Iterable[Iterable[float]]") -> bool:
    """Does a model give the atoms of a residue values of their own? ``groups`` are the values of the
    atoms of each residue. Yes if in at least one residue two of them differ by more than
    :data:`ATOM_SPREAD`; a residue of one atom says nothing either way."""
    for group in groups:
        values = list(group)
        if len(values) > 1 and max(values) - min(values) > ATOM_SPREAD:
            return True
    return False


def plddt_factor(values: Iterable[float], what: str = "the pLDDT values handed in are", scale: "str | None" = None,
                 how: str = "scale='0-1' reads them as {lo} to {hi}; scale='0-100' takes them as they stand") -> float:
    """What brings a set of pLDDT values to the 0-100 scale: 1 or 100. Or a refusal.

    * ``scale=None``, not stated: the values are on 0-100, as documented. If none of them exceeds 1
      (and they are not all zero) they cannot be -- on 0-100 a model has no residue below 1 -- and
      are refused, with both scales named and how to state one.
    * ``scale="0-100"``: stated, taken as they stand.
    * ``scale="0-1"``: stated; the factor is 100. A value above 1 contradicts it and is refused.

    On either scale a value below 0, or above the top of the scale, is refused. Nothing is
    converted that the caller did not state.

    ``what`` begins the error and ends with its verb (``"the B-factor column of model.cif holds"``);
    ``how`` says how the scale is stated where the values came in (``{lo}`` and ``{hi}``: the values
    times 100).
    """
    if scale is not None and scale not in SCALES:
        raise ValueError(f"the scale of a pLDDT is {' or '.join(repr(s) for s in SCALES)}, not {scale!r}")
    vals = [float(v) for v in values if v is not None and v == v]          # (v == v: not NaN)
    if not vals:
        return 100.0 if scale == "0-1" else 1.0
    lo, hi = min(vals), max(vals)
    span = f"{lo:.3g}" if lo == hi else f"{lo:.3g} to {hi:.3g}"
    if lo < 0.0:
        raise BindviewError(f"{what} {span}: a pLDDT is not negative")
    if scale == "0-1":
        if hi > 1.0:
            raise BindviewError(f"{what} {span}, and the scale stated is 0-1, on which nothing exceeds 1. Nothing is "
                              f"divided for you: if the values are on 0-100, state that, or state no scale")
        return 100.0
    if hi > 100.0:
        raise BindviewError(f"{what} {span}: on the 0-100 scale of pLDDT nothing exceeds 100")
    if scale is None and 0.0 < hi <= 1.0:
        raise BindviewError(
            f"{what} {span}: nothing above 1. bindview reads pLDDT on the 0-100 scale, where a model has no residue "
            f"below 1, so these look like the 0-1 scale that Boltz writes into plddt_*.npz. The scale is not "
            f"guessed and nothing is multiplied for you. State it: "
            + how.format(lo=f"{100 * lo:.0f}", hi=f"{100 * hi:.0f}"))
    return 1.0


def band_of(value: float) -> str:
    """The pLDDT band of ``value`` (0-100 scale)."""
    for name, lower in BANDS:
        if value >= lower:
            return name
    return "very_low"       # NaN


@dataclass(frozen=True)
class Look:
    """How strongly to draw something.

    Three channels, so that the message survives greyscale printing and colour
    vision deficiency: stroke ``width`` (a factor on the normal width),
    ``opacity``, and -- for the lowest band only -- a visible ``mark``.
    ``level`` orders the looks: 0 is unqualified.
    """

    band: str | None
    width: float = 1.0
    opacity: float = 1.0
    mark: bool = False
    level: int = 0


#: Drawn as if nothing were known: used when there is no confidence information.
PLAIN = Look(None)
#: At or above 70 a contact is drawn at full strength: a confident prediction
#: should look like the unqualified figure, so that most figures end up barely
#: touched and the qualified ones stand out.
LOOKS: dict[str, Look] = {
    "very_high": Look("very_high"),
    "confident": Look("confident"),
    "low": Look("low", width=0.7, opacity=0.62, level=1),
    "very_low": Look("very_low", width=0.5, opacity=0.42, mark=True, level=2),
}
#: The mark that goes with the lowest band.
MARK = "?"


def look_of(value: float | None, by_colour: bool = False) -> Look:
    """The look for a pLDDT value; :data:`PLAIN` when there is none.

    ``by_colour``: the figure shows the bands by colour and not by weight. The
    look is then the full-strength one and says only which band it is: nothing
    is drawn thinner, paler or with a mark for its pLDDT.
    """
    if value is None:
        return PLAIN
    band = band_of(value)
    return Look(band) if by_colour else LOOKS[band]


#: Occupancies shown as samples in the legend.
OCCUPANCY_SAMPLES = (1.0, 0.5, 0.1)


#: Occupancy in three steps (``RenderOptions.occupancy_style="steps"``, the default): from a share of
#: the first number on, a contact is drawn at the stroke-width and opacity factors after it. The top
#: step is a full line, so that a contact made in every model is drawn exactly like one that carries
#: no occupancy at all; the share written on the line tells 95 % from 100 %.
OCCUPANCY_STEPS = ((0.9, 1.0, 1.0), (0.5, 0.7, 0.8), (0.0, 0.38, 0.5))
#: A contact with a share below this is written on its line (rounded, it would not read 100 %).
SHARE_WRITTEN_BELOW = 0.995


def occupancy_factors(occupancy: float | None, steps: bool = False) -> tuple[float, float]:
    """Stroke-width and opacity factors for a contact made in ``occupancy`` of an ensemble.

    Occupancy uses the same visual channel as confidence -- weight and opacity
    -- because it answers the same kind of question: how much of the evidence
    stands behind this line. The mapping is continuous, or with ``steps`` in the
    three steps of :data:`OCCUPANCY_STEPS`, and a contact that is always made (or
    whose occupancy is unknown) is drawn exactly like one that carries no
    occupancy at all, so a single structure is the limiting case of an ensemble.
    The two factors scale the stroke; they are never combined with pLDDT into a
    number.
    """
    if occupancy is None or occupancy >= 1.0:
        return (1.0, 1.0)
    o = max(0.0, occupancy)
    if steps:
        return next((width, opacity) for least, width, opacity in OCCUPANCY_STEPS if o >= least)
    return (0.4 + 0.6 * o, 0.35 + 0.65 * o)


@dataclass(frozen=True)
class ModelConfidence:
    """Model-level evidence, supplied by the caller and displayed as given.

    ``trusted=False`` is the caller's verdict that the model should not be read
    as a result; the figure is then *marked* (a banner and a frame), not merely
    drawn faint. ``flag`` is that verdict in the caller's words, e.g.
    ``"grade C"``. bindview applies no threshold of its own to decide this.
    """

    iptm: float | None = None
    ptm: float | None = None
    ranking_score: float | None = None
    trusted: bool = True
    flag: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], **overrides: Any) -> "ModelConfidence":
        """From a mapping, such as AlphaFold 3's summary: ``iptm``, ``ptm`` and ``ranking_score`` as
        numbers, or as text that is one (a table gives ``"0.83"``); ``trusted`` as true or false, or as
        one of those words; ``flag``. A key that is not there, or holds nothing, is left out. A value
        that is neither is refused by its key, not read as something else."""
        def number(key: str) -> float | None:
            value = data.get(key)
            if value is None:
                return None
            try:
                if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                    raise ValueError
                read = float(value)
                if read != read:                        # nan: no number
                    raise ValueError
            except ValueError:
                raise BindviewError(f"model confidence: {key!r} is {value!r}, which is no number") from None
            return read

        def verdict(key: str) -> bool:
            value = data.get(key)
            if value is None:
                return True
            if isinstance(value, bool) or (isinstance(value, int) and value in (0, 1)):
                return bool(value)
            said = value.strip().lower() if isinstance(value, str) else None
            if said in ("true", "yes"):
                return True
            if said in ("false", "no"):
                return False
            raise BindviewError(f"model confidence: {key!r} is {value!r}; it is true or false")

        base = cls(iptm=number("iptm"), ptm=number("ptm"), ranking_score=number("ranking_score"),
                   trusted=verdict("trusted"), flag=data.get("flag") or None)
        return replace(base, **overrides) if overrides else base

    @classmethod
    def from_json(cls, path: "str | Path", **overrides: Any) -> "ModelConfidence":
        """Read an AlphaFold 3 ``summary_confidences.json`` (or any JSON with those keys)."""
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            raise StructureError(f"cannot read model confidence from {path}: {exc}") from exc
        if not isinstance(data, Mapping):
            raise StructureError(f"{path}: expected a JSON object with 'iptm' / 'ptm' / 'ranking_score'")
        return cls.from_mapping(data, **overrides)

    @staticmethod
    def beside(structure_path: "str | Path") -> Path | None:
        """The summary file AlphaFold 3 writes next to a model, if it is there.

        The pairing is by name only -- ``model.cif`` with
        ``summary_confidences.json``, ``<job>_model.cif`` with
        ``<job>_summary_confidences.json`` -- never "whatever JSON is in the
        directory", so a neighbouring model's numbers cannot be picked up.
        """
        p = Path(structure_path)
        stem = p.stem
        if stem == "model":
            cand = p.with_name("summary_confidences.json")
        elif stem.endswith("_model"):
            cand = p.with_name(stem[: -len("model")] + "summary_confidences.json")
        else:
            return None
        return cand if cand.is_file() else None

    @classmethod
    def coerce(cls, value: Any) -> "ModelConfidence | None":
        """From whatever ``model_confidence`` was given."""
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_mapping(value)
        if isinstance(value, (str, Path)):
            return cls.from_json(value)
        raise TypeError("model confidence must be a ModelConfidence, a mapping or a path to a JSON file")


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else None


@dataclass
class Confidence:
    """Per-residue and per-ligand-atom pLDDT of one model, on the 0-100 scale.

    ``origin`` records why the numbers may be read as confidence:
    ``"declared"`` (the file's ``_ma_qa_metric`` says so), ``"asserted"`` (the
    caller said the B-factor column is pLDDT) or ``"supplied"`` (the caller
    passed the values).
    """

    residues: dict[tuple[str, int, str], float]
    ligand_atoms: dict[str, float]
    pocket_plddt: float | None = None
    pocket_residues: int = 0
    origin: str = "supplied"
    model: ModelConfidence | None = None
    #: The scale the values were *given* on, where the caller stated one that is not 0-100: they are
    #: held on 0-100 like all others, and the result says that they were brought there.
    given_on: str | None = None
    #: pLDDT of the single atoms of every residue (not of the ligand), where the model was read from a
    #: structure: ``(chain, number, icode) -> atom name -> value``. Empty where values were handed in.
    atoms: dict[tuple[str, int, str], dict[str, float]] = field(default_factory=dict)
    #: Does the model give the atoms of a residue values of their own (:func:`gives_atoms`)? What a
    #: residue's atoms are drawn at where they are drawn, and what the figure then says they are drawn at.
    per_atom: bool = False
    _by_number: dict[tuple[int, str], list[tuple[str, int, str]]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        table: dict[tuple[int, str], list[tuple[str, int, str]]] = {}
        for key in self.residues:
            table.setdefault((key[1], key[2]), []).append(key)
        self._by_number = table

    # -- construction ------------------------------------------------------
    @classmethod
    def from_structure(cls, structure: Structure, ligand: ResidueKey, *, scale: float = 1.0,
                       origin: str = "declared", cutoff: float = POCKET_CUTOFF,
                       model: ModelConfidence | None = None) -> "Confidence | None":
        """Read pLDDT from the B-factor column of ``structure``.

        This does **not** check that the column is a pLDDT -- the caller of
        this method has established that (see :meth:`Diagram.from_structure`).
        Returns ``None`` when the column is absent. ``scale`` is what brings
        the column to 0-100: 1, or 100 for a column on 0-1. A column that
        cannot be on the scale it is said to be on is refused
        (:func:`plddt_factor`): one read as 0-100 in which nothing exceeds 1,
        one read as 0-1 in which something does.
        """
        said = {"declared": "declared a pLDDT", "asserted": "asserted to be pLDDT"}.get(origin, "read as pLDDT")
        on_unit = scale == 100.0
        plddt_factor((a.b_iso for a in structure.atoms if a.b_iso is not None),
                     f"the B-factor column of {structure.source or 'the structure'}, {said} on the "
                     f"{'0-1' if on_unit else '0-100'} scale, holds",
                     "0-1" if on_unit else None if scale == 1.0 else "0-100",
                     how="plddt='bfactor01' (--plddt bfactor01 on the command line) is the assertion that the "
                         "column is pLDDT on 0-1, and reads it as {lo} to {hi}")
        residues: dict[tuple[str, int, str], float] = {}
        single: dict[tuple[str, int, str], dict[str, float]] = {}
        for key in structure.residues():
            if key == ligand:
                continue
            vals = [a.b_iso * scale for a in structure.residue_atoms(key) if a.b_iso is not None]
            if vals:
                residues[(key.chain, key.number, key.icode)] = sum(vals) / len(vals)
                single[(key.chain, key.number, key.icode)] = {
                    a.name: a.b_iso * scale for a in structure.residue_atoms(key)
                    if a.b_iso is not None and a.element not in ("H", "D")}
        lig_atoms = {a.name: a.b_iso * scale for a in structure.residue_atoms(ligand) if a.b_iso is not None}
        if not residues and not lig_atoms:
            return None
        pocket = pocket_atoms(structure, ligand, cutoff)
        vals = [a.b_iso * scale for atoms in pocket.values() for a in atoms if a.b_iso is not None]
        return cls(residues, lig_atoms, pocket_plddt=_mean(vals), pocket_residues=len(pocket), origin=origin,
                   model=model, atoms=single,
                   per_atom=gives_atoms([list(v.values()) for v in single.values()] + _chain_groups(lig_atoms)))

    @classmethod
    def from_values(cls, residues: Mapping[Any, float], ligand_atoms: Mapping[str, float] | None = None, *,
                    pocket_plddt: float | None = None, model: ModelConfidence | None = None,
                    scale: str | None = None) -> "Confidence":
        """Confidence from values the caller already has.

        ``residues`` is keyed by :class:`ResidueKey`, ``(chain, number)``,
        ``(chain, number, icode)`` or ``"A:570"``; ``ligand_atoms`` by atom name.

        The values are on the 0-100 scale unless ``scale="0-1"`` says
        otherwise (Boltz's ``plddt_*.npz`` is on 0-1); they are then brought to
        0-100, and the result says so (``given_on``). Values that cannot be on
        the scale they are said to be on are refused, and so are values without
        a stated scale none of which exceeds 1 (:func:`plddt_factor`):
        ``scale="0-100"`` states that they are meant as they stand.
        """
        every = [float(v) for v in residues.values()] + [float(v) for v in (ligand_atoms or {}).values()]
        if pocket_plddt is not None:
            every.append(float(pocket_plddt))
        factor = plddt_factor(every, "the pLDDT values handed in are", scale)
        table: dict[tuple[str, int, str], float] = {}
        for key, value in residues.items():
            table[_residue_key(key)] = float(value) * factor
        given = {str(k): float(v) * factor for k, v in (ligand_atoms or {}).items()}
        return cls(table, given,
                   pocket_plddt=None if pocket_plddt is None else float(pocket_plddt) * factor,
                   pocket_residues=0, origin="supplied", model=model,
                   given_on=scale if factor != 1.0 else None, per_atom=gives_atoms(_chain_groups(given)))

    # -- lookups -----------------------------------------------------------
    def residue(self, key: "ResidueKey | tuple[str, int, str] | None") -> float | None:
        """pLDDT of a residue: the mean over its atom records."""
        if key is None:
            return None
        k = (key.chain, key.number, key.icode) if isinstance(key, ResidueKey) else (key[0], key[1], key[2])
        hit = self.residues.get(k)
        if hit is None and not k[0]:
            # A record without a chain id: unambiguous only if one chain has that number.
            cands = self._by_number.get((k[1], k[2]), [])
            if len(cands) == 1:
                hit = self.residues[cands[0]]
        return hit

    def atom(self, key: "ResidueKey | tuple[str, int, str] | None", name: str) -> float | None:
        """pLDDT of one atom of a residue: its own where the model gives the atoms of a residue values
        of their own and this one has one, the residue's otherwise."""
        if key is None:
            return None
        k = (key.chain, key.number, key.icode) if isinstance(key, ResidueKey) else (key[0], key[1], key[2])
        if self.per_atom and name in self.atoms.get(k, {}):
            return self.atoms[k][name]
        return self.residue(key)

    def ligand(self, names: Sequence[str]) -> float | None:
        """pLDDT of a ligand site: the atom's own value, or the mean over the
        atoms of a ring or charged group (whose centre is what the contact attaches to)."""
        return _mean(self.ligand_atoms[n] for n in names if n in self.ligand_atoms)

    def contact(self, residue: "ResidueKey | tuple[str, int, str] | None", names: Sequence[str]) -> float | None:
        """Confidence of a contact: the **lower** of its two ends.

        If only one end is known (values supplied for residues only), that end
        is all that can be said and is returned.
        """
        ends = [v for v in (self.residue(residue), self.ligand(names)) if v is not None]
        return min(ends) if ends else None

    @property
    def ligand_plddt(self) -> float | None:
        """Mean pLDDT over the ligand's atoms."""
        return _mean(self.ligand_atoms.values())

    @property
    def marked(self) -> bool:
        """Has the caller flagged this model as not to be trusted?"""
        return self.model is not None and not self.model.trusted

    def with_model(self, model: ModelConfidence | None) -> "Confidence":
        """The same with the numbers of the model as a whole."""
        return replace(self, model=model)

    def summary(self) -> dict[str, Any]:
        """The figure-level numbers, as plain data."""
        m = self.model
        out = {
            "origin": self.origin,
            "iptm": m.iptm if m else None, "ptm": m.ptm if m else None,
            "ranking_score": m.ranking_score if m else None,
            "pocket_plddt": None if self.pocket_plddt is None else round(self.pocket_plddt, 2),
            "pocket_residues": self.pocket_residues,
            "ligand_plddt": None if self.ligand_plddt is None else round(self.ligand_plddt, 2),
            "trusted": m.trusted if m else True, "flag": m.flag if m else None,
        }
        if self.given_on is not None:
            # Said only where it happened: the summary of a model given on 0-100 has no such key.
            out["given_on"] = f"the {self.given_on} scale, as stated; held and drawn on 0-100"
        return out


def _chain_groups(ligand_atoms: Mapping[str, float]) -> list[list[float]]:
    """The values of a chain taken as the ligand, residue by residue: its atoms are named
    ``<residue>/<atom>`` (:meth:`bindview.structure.Structure.with_chain_as_ligand`). A ligand that is
    one molecule is no residue, and says nothing of what a model gives the atoms of one."""
    groups: dict[str, list[float]] = {}
    for name, value in ligand_atoms.items():
        if "/" in name:
            groups.setdefault(name.split("/", 1)[0], []).append(value)
    return list(groups.values())


def _residue_key(key: Any) -> tuple[str, int, str]:
    if isinstance(key, ResidueKey):
        return (key.chain, key.number, key.icode)
    if isinstance(key, str):
        chain, _, rest = key.rpartition(":")
        digits = rest.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        try:
            return (chain, int(digits), rest[len(digits):])
        except ValueError:
            raise ValueError(f"cannot read residue key {key!r}; use 'A:570' or ('A', 570)") from None
    if isinstance(key, int):
        return ("", key, "")
    parts = tuple(key)
    if len(parts) == 2:
        return (str(parts[0]), int(parts[1]), "")
    if len(parts) == 3:
        return (str(parts[0]), int(parts[1]), str(parts[2]))
    raise ValueError(f"cannot read residue key {key!r}")


def pocket_atoms(structure: Structure, ligand: ResidueKey, cutoff: float = POCKET_CUTOFF
                 ) -> dict[ResidueKey, list]:
    """Residues with a heavy atom within ``cutoff`` of a ligand heavy atom, with all their atoms."""
    lig_xyz = np.array([a.xyz for a in structure.residue_atoms(ligand) if a.element not in ("H", "D")],
                       dtype=float).reshape(-1, 3)
    if not len(lig_xyz):
        return {}
    coords = structure.coords
    lo, hi = lig_xyz.min(axis=0) - cutoff, lig_xyz.max(axis=0) + cutoff
    near = np.nonzero(((coords >= lo) & (coords <= hi)).all(axis=1))[0]
    hits: dict[ResidueKey, list] = {}
    for i in near:
        atom = structure.atoms[int(i)]
        key = atom.residue
        if key == ligand or atom.element in ("H", "D") or key in hits:
            continue
        if float(np.linalg.norm(lig_xyz - coords[i], axis=1).min()) <= cutoff:
            hits[key] = structure.residue_atoms(key)
    return dict(sorted(hits.items()))
