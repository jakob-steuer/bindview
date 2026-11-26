#!/usr/bin/env python
"""Hold-out validation of the predicted solvent sites against crystallographic waters.

    python scripts/validate_solvent.py                       # the Astex Diverse Set in testdata/astex
    python scripts/validate_solvent.py --data testdata/waters --set testdata/waters/manifest.json --name targets

``--data`` holds experimental structures that contain waters, as ``.cif`` or
``.pdb`` files, compressed (``.gz``) or not, in one directory or one each.
Both sets are fetched by ``scripts/fetch_structures.py`` and are not kept in
the repository. Which ligand is evaluated is settled, in this order, by a
``meta.json`` beside the file (``{"ligand": {"residue_name_in_cif": "STI"}}``
or ``{"ligand": "A:STI:201"}``), by the set (``--set``: a table of structure
and ligand such as ``scripts/solvent_validation_set.tsv``, or a
``manifest.json`` whose ``entries`` name ``file``, ``ligand`` and optionally
``gene``), or, for a structure in neither, by a rule: every distinct free
ligand once (:func:`ligands_of`). Two sets are never pooled: each run writes
its own file, ``solvent-validation.json`` or, with ``--name``,
``solvent-validation.<name>.json``, into ``--out`` (``tests/data``), and
prints the result in words and tables.

The protocol (its result in words: :func:`headline`, :func:`footnote`):

* all waters are removed before the predictor runs (it ignores them as atoms);
* truth for **recall** is the set of *bridging* waters: a water oxygen within
  3.5 A of a ligand N/O and within 3.5 A of an N/O of the rest of the model;
* a predicted site **matches** a water within 1.5 A (1.0 and 2.0 A alongside),
  and one site accounts for one water at most;
* **invented** = predicted sites with no crystallographic water of any kind
  within 1.5 A. An unmodelled water is not proof of an empty site, so this is
  an upper bound on false positives;
* **chance** = the same number of sites placed at random where there is merely
  room for a water near a polar ligand atom, counted the same way;
* **reachable** = the bridging waters that have such a place within the match
  distance at all: the ceiling of what the predictor could recover;
* structures are sorted by name; parameters may be tuned on the odd positions
  only, and the reported numbers come from the even positions. The parameters
  are the library's (:class:`bindview.solvent.SolventParams`) and are not
  tuned unless ``--tuned`` says so.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import re
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bindview.exposure import BURIAL_WARNING_BELOW, WATER_NAMES, burial_warning, structure_burial  # noqa: E402
from bindview.solvent import SolventParams, candidate_points, predict_solvent_sites, water_oxygens  # noqa: E402
from bindview.structure import _read_categories, parse_mmcif, parse_pdb  # noqa: E402

ADDITIVES = {"GOL", "EDO", "PEG", "PGE", "PG4", "1PE", "P6G", "MPD", "DMS", "ACT", "ACY", "FMT", "SO4", "PO4", "NO3",
             "CL", "NA", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "NI", "CO", "CD", "IOD", "BR", "TRS", "MES", "EPE",
             "BME", "DTT", "IMD", "CIT", "TLA", "NH4", "UNX", "UNL"}
METALS = {"LI", "NA", "K", "RB", "CS", "BE", "MG", "CA", "SR", "BA", "AL", "GA", "IN", "TL", "SN", "PB", "BI", "SC",
          "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "Y", "ZR", "NB", "MO", "RU", "RH", "PD", "AG", "CD", "HF",
          "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG", "LA", "CE", "PR", "ND", "SM", "EU", "GD", "TB", "DY", "HO",
          "ER", "TM", "YB", "LU", "U"}
LARGE = {"S", "SE", "BR", "I"}             # atoms whose bonds are longer than 2.0 A
SUFFIXES = (".cif", ".mmcif", ".pdb", ".ent")
SET = ROOT / "scripts" / "solvent_validation_set.tsv"
CUTS = (1.0, 1.5, 2.0)
BRIDGE = 3.5
MIN_HEAVY = 6
DRAWS = 200                                 # random placements per ligand for the chance figures
HIGH_RESOLUTION = 2.0
SEED = 20261112                             # for the intervals: structures are resampled
N_BOOT = 2000
#: Bands of resolution for the table by resolution: above the first number, up to and including the second.
RESOLUTION_BANDS = (("up to 1.5 Å", 0.0, 1.5), ("above 1.5, up to 2.0 Å", 1.5, 2.0), ("above 2.0 Å", 2.0, 99.0))
#: What a row carries besides its counts; left out before rows are added up.
NOT_COUNTS = ("half", "resolution", "attached", "complete", "cofactor", "buried_fraction", "burial_warning")


def matches(predicted: np.ndarray, truth: np.ndarray, cut: float) -> tuple[int, int]:
    """``(truth points recovered, predictions supported)`` at a distance cut.

    A prediction accounts for one truth point at most, so *recovered* is the
    size of the largest one-to-one pairing within the cut. *Supported* counts
    the predictions that have any truth point within it.
    """
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 3)
    truth = np.asarray(truth, dtype=float).reshape(-1, 3)
    if not len(predicted) or not len(truth):
        return 0, 0
    near = np.linalg.norm(truth[:, None, :] - predicted[None, :, :], axis=2) <= cut       # (truth, predicted)
    owner = [-1] * near.shape[1]                                                          # truth point of each prediction

    def place(t: int, seen: list[bool]) -> bool:
        for j in np.nonzero(near[t])[0]:
            if seen[j]:
                continue
            seen[j] = True
            if owner[j] < 0 or place(owner[j], seen):
                owner[j] = t
                return True
        return False

    recovered = sum(place(int(t), [False] * near.shape[1]) for t in np.nonzero(near.any(axis=1))[0])
    return int(recovered), int(near.any(axis=0).sum())


def within(points: np.ndarray, others: np.ndarray, cut: float) -> np.ndarray:
    """For every point: is any of ``others`` within ``cut``?"""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    others = np.asarray(others, dtype=float).reshape(-1, 3)
    out = np.zeros(len(points), dtype=bool)
    if not len(points) or not len(others):
        return out
    for start in range(0, len(others), 2000):
        d = np.linalg.norm(points[:, None, :] - others[None, start:start + 2000, :], axis=2)
        out |= (d <= cut).any(axis=1)
    return out


def covalently_attached(st: Any, key: Any) -> bool:
    """Is the residue bonded to another one: a modified amino acid, a glycan, a covalent inhibitor?

    Bonds are read from distances between non-metal heavy atoms: under 2.0 A,
    or under 2.3 A where sulfur, selenium, bromine or iodine takes part.
    """
    def counts(a: Any) -> bool:
        return a.element not in ("H", "D") and a.element.upper() not in METALS and a.resname.upper() not in WATER_NAMES

    own = [a for a in st.residue_atoms(key) if counts(a)]
    rest = [a for a in st.atoms if a.residue != key and counts(a)]
    if not own or not rest:
        return False
    a_xyz = np.array([a.xyz for a in own], dtype=float)
    b_xyz = np.array([a.xyz for a in rest], dtype=float)
    close = ((b_xyz >= a_xyz.min(axis=0) - 2.3) & (b_xyz <= a_xyz.max(axis=0) + 2.3)).all(axis=1)
    if not close.any():
        return False
    b_xyz = b_xyz[close]
    a_big = np.array([a.element.upper() in LARGE for a in own])
    b_big = np.array([a.element.upper() in LARGE for a, keep in zip(rest, close) if keep])
    limit = np.where(a_big[:, None] | b_big[None, :], 2.3, 2.0)
    return bool((np.linalg.norm(a_xyz[:, None, :] - b_xyz[None, :, :], axis=2) < limit).any())


def _entries(path: Path) -> list[dict[str, Any]]:
    """The entries of a manifest: a JSON file with a list ``entries`` of ``{"file", "ligand", "gene"}``."""
    listed = json.loads(path.read_text(encoding="utf-8")).get("entries", [])
    return [e for e in listed if isinstance(e, dict) and e.get("file") and e.get("ligand")]


def heavy_atoms_by_formula(text: str) -> dict[str, int]:
    """For every compound an mmCIF file describes: the heavy atoms its formula has (``_chem_comp.formula``)."""
    table = _read_categories(text, {"_chem_comp"}).get("_chem_comp", {}) if "_chem_comp." in text else {}
    out: dict[str, int] = {}
    for name, formula in zip(table.get("id", []), table.get("formula", [])):
        counts = [(m.group(1), int(m.group(2) or 1)) for m in re.finditer(r"([A-Z][a-z]?)(\d*)", formula)]
        if counts:
            out[name] = sum(n for element, n in counts if element not in ("H", "D"))
    return out


def named_ligands(path: Path) -> dict[str, str]:
    """A validation set: structure identifier (lower case) and the name of its ligand."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    if path.suffix.lower() == ".json":
        return {name_of(Path(str(e["file"]))).lower(): str(e["ligand"]) for e in _entries(path)}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = line.split("\t")
        if line.startswith("#") or len(cells) < 2 or cells[0] == "pdb_id":
            continue
        out[cells[0].strip().lower()] = cells[1].strip()
    return out


def named_cofactors(path: Path) -> set[str]:
    """For a set that says so: the structures (lower case) whose named ligand is a cofactor and not a drug."""
    if path.is_file() and path.suffix.lower() == ".json":
        return {name_of(Path(str(e["file"]))).lower() for e in _entries(path) if e.get("ligand_is_cofactor")}
    return set()


def named_groups(path: Path) -> dict[str, str]:
    """For a set that sorts its structures by target: structure identifier (lower case) and target."""
    if path.is_file() and path.suffix.lower() == ".json":
        return {name_of(Path(str(e["file"]))).lower(): str(e["gene"]) for e in _entries(path) if e.get("gene")}
    return {}


def is_structure(path: Path) -> bool:
    """Is this a structure file, compressed or not?"""
    name = path.name.lower()
    return name.endswith(SUFFIXES) or (name.endswith(".gz") and name[:-3].endswith(SUFFIXES))


def read_file(path: Path) -> tuple[Any, str, str]:
    """The structure in a file, its text, and the SHA-256 of that text. A ``.gz`` file is unpacked in memory."""
    data, name = path.read_bytes(), path.name.lower()
    if name.endswith(".gz"):
        data, name = gzip.decompress(data), name[:-3]
    text = data.decode("utf-8", errors="replace")
    parse = parse_pdb if name.endswith((".pdb", ".ent")) else parse_mmcif
    return parse(text, source=str(path)), text, hashlib.sha256(data).hexdigest()


def ligands_of(st: Any, folder: Path, named: str | None = None) -> tuple[list[Any], list[dict[str, str]], str]:
    """The ligands to evaluate in one structure, the hetero compounds passed over, and what settled it.

    A ``meta.json`` beside the structure names one ligand; failing that,
    ``named`` does (the ligand the validation set lists for the structure).
    Of several copies of a named compound the first in the file is taken, and
    a named ligand that is not in the file is an error, not an occasion to
    pick another. A copy is a hetero compound of that name: a residue of a
    protein chain that happens to share the name is not one.

    Only a structure with neither falls under the rule: every hetero compound
    qualifies that is not water, not a common additive or ion, has at least
    six heavy atoms and is not covalently attached to another residue. Of
    several copies of one compound, the first in the file that qualifies is
    taken.
    """
    spec, source = named, "list of the set"
    meta = folder / "meta.json"
    if meta.exists():
        told = json.loads(meta.read_text(encoding="utf-8")).get("ligand")
        if isinstance(told, dict):
            told = told.get("selection") or told.get("residue_name_in_cif")
        if told:
            spec, source = told, "meta.json"
    if spec:
        # A compound is named, not a residue of a chain: where the name is also an amino acid's (free
        # tyrosine in 1x8x, whose protein has tyrosines of its own), only hetero records are the ligand.
        same = [k for k in st.residues() if st.residue_name(k) == spec]
        copies = [k for k in same if all(a.hetero for a in st.residue_atoms(k))]
        if same and not copies:
            raise ValueError(f"{spec} is in the file only as a residue of a chain, not as a ligand")
        return [copies[0] if copies else st.select_ligand(spec)], [], source
    chosen: dict[str, Any] = {}
    passed: dict[str, str] = {}
    for key in st.residues():
        atoms = st.residue_atoms(key)
        name = atoms[0].resname.upper()
        if not all(a.hetero for a in atoms) or name in WATER_NAMES or name in chosen:
            continue
        heavy = sum(1 for a in atoms if a.element not in ("H", "D"))
        if name in ADDITIVES:
            reason = "additive or ion"
        elif heavy < MIN_HEAVY:
            reason = f"fewer than {MIN_HEAVY} heavy atoms"
        elif covalently_attached(st, key):
            reason = "covalently attached to another residue"
        else:
            chosen[name] = key
            passed.pop(name, None)
            continue
        passed.setdefault(name, reason)
    return list(chosen.values()), [{"compound": n, "reason": r} for n, r in passed.items()], "rule"


def resolution_of(text: str) -> float | None:
    """The resolution a structure file states, if it states one where a single line can be read."""
    for pattern in (r"^_refine\.ls_d_res_high\s+(\S+)", r"^_reflns\.d_resolution_high\s+(\S+)",
                    r"^REMARK   2 RESOLUTION\.\s+(\S+)"):
        found = re.search(pattern, text, re.MULTILINE)
        if found:
            try:
                return float(found.group(1))
            except ValueError:
                continue
    return None


def draw(points: np.ndarray, k: int, separation: float, rng: np.random.Generator) -> np.ndarray:
    """``k`` of the points, taken in random order, no two closer than ``separation`` (fewer if they run out)."""
    chosen: list[np.ndarray] = []
    if k > 0:
        for i in rng.permutation(len(points)):
            if all(float(np.linalg.norm(points[i] - c)) >= separation for c in chosen):
                chosen.append(points[i])
                if len(chosen) == k:
                    break
    return np.array(chosen, dtype=float).reshape(-1, 3)


def evaluate_ligand(st: Any, key: Any, waters: np.ndarray, params: SolventParams, structure: str) -> dict[str, Any]:
    """One row of the report: what the predictor did for one ligand, and what chance would have done."""
    polar = [a for a in st.atoms if a.element in ("N", "O") and a.resname.upper() not in WATER_NAMES]
    lig = np.array([a.xyz for a in polar if a.residue == key], dtype=float).reshape(-1, 3)
    rest = np.array([a.xyz for a in polar if a.residue != key], dtype=float).reshape(-1, 3)
    by_ligand = waters[within(waters, lig, BRIDGE)]
    bridging = by_ligand[within(by_ligand, rest, BRIDGE)]
    found = predict_solvent_sites(st, key, params)
    sites = np.array([s.xyz for s in found], dtype=float).reshape(-1, 3)
    allowed = candidate_points(st, key, params)
    row: dict[str, Any] = {"structure": structure, "ligand": f"{key.chain}:{st.residue_name(key)}:{key.number}",
                           "bridging_waters": int(len(bridging)), "predicted": int(len(sites))}
    rng = np.random.default_rng(zlib.crc32(f"{structure}|{row['ligand']}".encode("utf-8")))
    draws = [draw(allowed, len(sites), params.separation, rng) for _ in range(DRAWS)] if len(sites) else []
    row["chance_sites"] = round(float(np.mean([len(d) for d in draws])), 3) if draws else 0.0
    for cut in CUTS:
        row[f"recovered_{cut}"] = matches(sites, bridging, cut)[0]
        row[f"invented_{cut}"] = int(len(sites) - matches(sites, waters, cut)[1])
        row[f"reachable_{cut}"] = int(within(bridging, allowed, cut).sum())
        row[f"chance_recovered_{cut}"] = (round(float(np.mean([matches(d, bridging, cut)[0] for d in draws])), 3)
                                          if draws else 0.0)
        row[f"chance_invented_{cut}"] = (round(float(np.mean([len(d) - matches(d, waters, cut)[1] for d in draws])), 3)
                                         if draws else 0.0)

    def nearest(point: np.ndarray, others: np.ndarray) -> float | None:
        return float(np.linalg.norm(others - point, axis=1).min()) if len(others) else None

    row["sites"] = []                       # in the order the predictor returns them: best first
    for rank, (site, xyz) in enumerate(zip(found, sites), start=1):
        to_any, to_bridging = nearest(xyz, waters), nearest(xyz, bridging)
        row["sites"].append({"rank": rank, "partners": site.score, "enclosure": round(site.enclosure, 3),
                             "nearest_water": None if to_any is None else round(to_any, 2),
                             "water": to_any is not None and to_any <= 1.5,
                             "bridging_water": to_bridging is not None and to_bridging <= 1.5})
    return row


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals over ligands, and the shares the protocol names."""
    out: dict[str, Any] = {"structures": len({r["structure"] for r in rows}), "ligands": len(rows)}
    if not rows:
        return out
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    total = {k: sum(r[k] for r in rows) for k in keys}
    out.update({k: round(v, 3) if isinstance(v, float) else v for k, v in total.items()})

    def share(part: str, whole: str) -> float | None:
        return round(total[part] / total[whole], 3) if total.get(whole) else None

    for cut in CUTS:
        out[f"recall_{cut}"] = share(f"recovered_{cut}", "bridging_waters")
        out[f"invented_share_{cut}"] = share(f"invented_{cut}", "predicted")
        if "chance_sites" in total:
            out[f"reachable_share_{cut}"] = share(f"reachable_{cut}", "bridging_waters")
            out[f"chance_recall_{cut}"] = share(f"chance_recovered_{cut}", "bridging_waters")
            out[f"chance_invented_share_{cut}"] = share(f"chance_invented_{cut}", "chance_sites")
    return out


# ---------------------------------------------------------------------------
# Beyond the protocol: how sure the numbers are, and where the unsupported sites come from
# ---------------------------------------------------------------------------

def intervals(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    """95 % intervals of the reported shares and of their distance from chance: structures are resampled."""
    names = sorted({r["structure"] for r in rows})
    keys = ("bridging_waters", "predicted", "recovered_1.5", "invented_1.5", "chance_recovered_1.5",
            "chance_invented_1.5", "chance_sites")
    table = {n: np.zeros(len(keys)) for n in names}
    for r in rows:
        table[r["structure"]] += np.array([r[k] for k in keys], dtype=float)
    per = np.array([table[n] for n in names]).reshape(-1, len(keys))
    drawn: dict[str, list[float]] = {"recall_1.5": [], "invented_share_1.5": [], "recall_above_chance_1.5": [],
                                     "invented_share_below_chance_1.5": []}
    rng = np.random.default_rng(SEED)
    for _ in range(N_BOOT if len(per) > 1 else 0):
        waters, sites, recovered, invented, by_chance, invented_by_chance, placed = per[
            rng.integers(0, len(per), len(per))].sum(axis=0)
        if waters:
            drawn["recall_1.5"].append(recovered / waters)
            drawn["recall_above_chance_1.5"].append((recovered - by_chance) / waters)
        if sites:
            drawn["invented_share_1.5"].append(invented / sites)
        if sites and placed:
            drawn["invented_share_below_chance_1.5"].append(invented_by_chance / placed - invented / sites)
    return {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
            for k, v in drawn.items() if v}


def site_breakdown(rows: list[dict[str, Any]], rays: int) -> dict[str, dict[str, dict[str, int]]]:
    """The predicted sites of some ligands, sorted three ways: how many of each kind have a water within 1.5 A."""
    sites = [s for r in rows for s in r.get("sites", [])]

    def count(chosen: list[dict[str, Any]]) -> dict[str, int]:
        return {"sites": len(chosen), "water": sum(bool(s["water"]) for s in chosen),
                "bridging_water": sum(bool(s["bridging_water"]) for s in chosen)}

    def tenth(site: dict[str, Any]) -> int:                      # 6 for an enclosure from 0.6 up to, not including, 0.7
        return min(9, int(round(site["enclosure"] * rays) * 10 // rays))

    return {"by_enclosure": {f"{t / 10:.1f} to {(t + 1) / 10:.1f}": count([s for s in sites if tenth(s) == t])
                             for t in sorted({tenth(s) for s in sites})},
            "by_partners": {str(n): count([s for s in sites if s["partners"] == n])
                            for n in sorted({s["partners"] for s in sites})},
            "by_rank": {str(n): count([s for s in sites if s["rank"] == n]) for n in sorted({s["rank"] for s in sites})}}


def enclosure_contrast(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Weakly enclosed sites (under 0.7) against well enclosed ones (0.8 and above): the share with a water.

    Sites of one structure are not independent, so the 95 % intervals resample
    structures, as everywhere else.
    """
    names = sorted({r["structure"] for r in rows})
    table = {n: np.zeros(4) for n in names}                      # weak sites, of them with a water; well; with a water
    for r in rows:
        weak = [s for s in r.get("sites", []) if s["enclosure"] < 0.7 - 1e-9]
        well = [s for s in r.get("sites", []) if s["enclosure"] >= 0.8 - 1e-9]
        table[r["structure"]] += np.array([len(weak), sum(bool(s["water"]) for s in weak),
                                           len(well), sum(bool(s["water"]) for s in well)], dtype=float)
    per = np.array([table[n] for n in names]).reshape(-1, 4)
    weak_n, weak_w, well_n, well_w = per.sum(axis=0) if len(per) else np.zeros(4)
    if not weak_n or not well_n:
        return {}
    drawn: dict[str, list[float]] = {"weak": [], "well": [], "difference": []}
    rng = np.random.default_rng(SEED)
    for _ in range(N_BOOT if len(per) > 1 else 0):
        a, b, c, d = per[rng.integers(0, len(per), len(per))].sum(axis=0)
        if a and c:
            drawn["weak"].append(b / a)
            drawn["well"].append(d / c)
            drawn["difference"].append(d / c - b / a)
    span = {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if v else None
            for k, v in drawn.items()}
    return {"weak": {"sites": int(weak_n), "water": int(weak_w), "share": round(weak_w / weak_n, 3),
                     "interval": span["weak"], "structures": int((per[:, 0] > 0).sum())},
            "well": {"sites": int(well_n), "water": int(well_w), "share": round(well_w / well_n, 3),
                     "interval": span["well"], "structures": int((per[:, 2] > 0).sum())},
            "difference": {"value": round(well_w / well_n - weak_w / weak_n, 3), "interval": span["difference"]}}


def contrast_sentence(result: dict[str, Any]) -> str:
    """The enclosure contrast in words, with its intervals."""
    c = result.get("enclosure_contrast") or {}
    if not c:
        return ""

    def span(pair: list[float] | None, unit: str = " %") -> str:
        return "n/a" if not pair else f"{100 * pair[0]:.0f} to {100 * pair[1]:.0f}{unit}"

    weak, well, diff = c["weak"], c["well"], c["difference"]
    return (f"Weakly enclosed sites (under 0.7) with a water: {weak['water']} of {weak['sites']} "
            f"({pct(weak['share'])}; {span(weak['interval'])}), in {weak['structures']} structures. Well enclosed "
            f"(0.8 and above): {well['water']} of {well['sites']} ({pct(well['share'])}; {span(well['interval'])}). "
            f"Difference: {100 * diff['value']:.0f} points ({span(diff['interval'], ' points')}).")


def burial_of(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The buried fraction of the evaluated ligands in their crystal structures, and whom the warning would reach.

    As everywhere here, the asymmetric unit as deposited: a site completed by a
    symmetry mate is half a site, which is the case the warning is for.
    """
    values = sorted(r["buried_fraction"] for r in rows)
    if not values:
        return {}
    quartiles = [round(float(np.percentile(values, q)), 3) for q in (25, 50, 75)]
    return {"cut": BURIAL_WARNING_BELOW, "ligands": len(values),
            "five_numbers": [values[0], *quartiles, values[-1]],
            "warned": [{"structure": r["structure"], "ligand": r["ligand"], "buried_fraction": r["buried_fraction"]}
                       for r in sorted(rows, key=lambda r: (r["buried_fraction"], r["structure"])) if r["burial_warning"]]}


def burial_sentence(result: dict[str, Any]) -> str:
    """The by-product in words."""
    b = result.get("burial") or {}
    if not b:
        return ""
    low, q1, median, q3, high = b["five_numbers"]
    warned = "; ".join(f"{w['structure']} {w['buried_fraction']:.3f}" for w in b["warned"])
    return (f"Buried fraction of the {b['ligands']} ligands in their crystal structures (asymmetric unit, water not "
            f"counted as cover): lowest {low:.3f}, quartiles {q1:.2f} / {median:.2f} / {q3:.2f}, highest {high:.2f}. "
            f"Below the warning's cut of {b['cut']:.2f}: {len(b['warned'])} of {b['ligands']}"
            + (f" ({warned})." if warned else "."))


def after_the_run(result: dict[str, Any]) -> str:
    """The counts by resolution, intervals, and the sites sorted three ways, as Markdown. Not part of the protocol.

    By resolution, all structures of the set are counted. The sites are those
    of the development half; for a set on which nothing is or will be tuned
    (``all_sites`` in the result), those of all its structures.
    """
    def span(pair: list[float] | None, unit: str = " %") -> str:
        return "n/a" if not pair else f"{100 * pair[0]:.0f} to {100 * pair[1]:.0f}{unit}"

    lines = [resolution_table(result), "",
             "| set | recall at 1.5 Å | above chance by | no water within 1.5 Å | below chance by |", "|" + "---|" * 5]
    for label, key in (("held out (even positions)", "held_out"), ("all", "all")):
        i, part = result["intervals"].get(key, {}), result[key]
        if part.get("ligands"):
            lines.append(f"| {label} | {pct(part.get('recall_1.5'))} ({span(i.get('recall_1.5'))}) | "
                         f"{span(i.get('recall_above_chance_1.5'), ' points')} | "
                         f"{pct(part.get('invented_share_1.5'))} ({span(i.get('invented_share_1.5'))}) | "
                         f"{span(i.get('invented_share_below_chance_1.5'), ' points')} |")
    text = "\n".join(lines)
    for title, key in (("enclosure", "by_enclosure"), ("partners", "by_partners"), ("rank", "by_rank")):
        table = result.get("all_sites", result["development_sites"]).get(key, {})
        if not table:
            continue
        text += f"\n\n| {title} | sites | water within 1.5 Å | bridging water within 1.5 Å |\n" + "|" + "---|" * 4
        for name, c in table.items():
            text += (f"\n| {name} | {c['sites']} | {c['water']} ({pct(c['water'] / c['sites'])}) | "
                     f"{c['bridging_water']} ({pct(c['bridging_water'] / c['sites'])}) |")
        if key == "by_enclosure" and contrast_sentence(result):
            text += "\n\n" + contrast_sentence(result)
    return text


# ---------------------------------------------------------------------------
# The result in words
# ---------------------------------------------------------------------------

def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.0f} %"


def footnote(result: dict[str, Any]) -> str:
    """The sentence printed under every figure that shows a predicted site (``bindview.solvent.VALIDATION``)."""
    h = result["held_out"]
    return (f"In a hold-out test on {h['structures']} crystal structures they recovered {pct(h.get('recall_1.5'))} of "
            f"the bridging waters, and {pct(h.get('invented_share_1.5'))} of the predicted sites had no observed water "
            f"within 1.5 Å.")


def _two_sentences(result: dict[str, Any], key: str, what: str) -> str:
    h = result[key]
    how = "tuned on the other half" if result.get("tuned") else "not tuned"
    return (f"{what} ({h['structures']} structures, {h['ligands']} ligands), parameters {how}: "
            f"{h.get('recovered_1.5', 0)} of {h.get('bridging_waters', 0)} bridging waters recovered within 1.5 Å "
            f"({pct(h.get('recall_1.5'))}; {pct(h.get('chance_recall_1.5'))} for the same number of sites placed at "
            f"random where there is room for a water; {pct(h.get('reachable_share_1.5'))} is the ceiling). "
            f"{h.get('invented_1.5', 0)} of {h.get('predicted', 0)} predicted sites "
            f"({pct(h.get('invented_share_1.5'))}) have no crystallographic water within 1.5 Å "
            f"({pct(h.get('chance_invented_share_1.5'))} at random).")


def headline(result: dict[str, Any]) -> str:
    """The result in two sentences, for the documentation."""
    return _two_sentences(result, "held_out", "Held-out half")


def whole(result: dict[str, Any]) -> str:
    """The same two sentences for all of a set: for a set on which nothing was or will be tuned."""
    return _two_sentences(result, "all", "Whole set")


def table(result: dict[str, Any]) -> str:
    """The halves side by side, as Markdown."""
    head = ("| set | structures | ligands | bridging waters | recovered at 1.0 / 1.5 / 2.0 Å | recall at 1.5 Å | "
            "by chance | ceiling | predicted sites | no water within 1.5 Å | by chance |")
    lines = [head, "|" + "---|" * 11]
    parts = [("development (odd positions)", "development"), ("**held out (even positions)**", "held_out"),
             (f"held out, resolution ≤ {HIGH_RESOLUTION:.1f} Å", "held_out_high_resolution"), ("all", "all")]
    if result.get("free_and_complete", result["all"]).get("ligands") != result["all"].get("ligands"):
        parts.append(("all, ligand free and complete", "free_and_complete"))
    if "not_a_cofactor" in result:
        parts.append(("all, ligand not a cofactor", "not_a_cofactor"))
    for label, key in parts:
        s = result[key]
        if not s.get("ligands"):
            lines.append(f"| {label} | 0 | 0 | | | | | | | | |")
            continue
        lines.append(
            f"| {label} | {s['structures']} | {s['ligands']} | {s['bridging_waters']} | "
            f"{s['recovered_1.0']} / {s['recovered_1.5']} / {s['recovered_2.0']} | {pct(s['recall_1.5'])} | "
            f"{pct(s['chance_recall_1.5'])} | {pct(s['reachable_share_1.5'])} | {s['predicted']} | "
            f"{s['invented_1.5']} ({pct(s['invented_share_1.5'])}) | {pct(s['chance_invented_share_1.5'])} |")
    return "\n".join(lines)


def _rows_by(what: str, parts: dict[str, dict[str, Any]]) -> str:
    head = (f"| {what} | structures | bridging waters | recovered within 1.5 Å | by chance | predicted sites | "
            "no water within 1.5 Å | by chance |")
    lines = [head, "|" + "---|" * 8]
    for name, s in parts.items():
        if not s.get("ligands"):
            lines.append(f"| {name} | 0 | | | | | | |")
            continue
        lines.append(f"| {name} | {s['structures']} | {s['bridging_waters']} | {s['recovered_1.5']} "
                     f"({pct(s['recall_1.5'])}) | {pct(s['chance_recall_1.5'])} | {s['predicted']} | "
                     f"{s['invented_1.5']} ({pct(s['invented_share_1.5'])}) | {pct(s['chance_invented_share_1.5'])} |")
    return "\n".join(lines)


def group_table(result: dict[str, Any]) -> str:
    """One row per target, for a set that sorts its structures by target."""
    return _rows_by("target", result.get("by_group", {}))


def resolution_table(result: dict[str, Any]) -> str:
    """One row per band of resolution, all structures of the set. Not part of the protocol."""
    return _rows_by("resolution", result.get("by_resolution", {}))


# ---------------------------------------------------------------------------

def name_of(path: Path) -> str:
    name = path.name[:-3] if path.name.lower().endswith(".gz") else path.name
    stem = Path(name).stem
    return path.parent.name if stem.lower() in ("model", "structure") else stem


def run(data: Path, params: SolventParams, tuned: bool = False, named: dict[str, str] | None = None,
        groups: dict[str, str] | None = None, nothing_tuned_here: bool = False,
        cofactors: set[str] | None = None) -> dict[str, Any]:
    """Evaluate every structure under ``data``.

    ``named`` is the validation set: identifier and ligand. Which half a
    structure belongs to does not depend on what is in it: it is its place in
    the sorted identifiers of the set, so that a missing file moves no other
    structure to the other half; for a structure outside the set, by its place
    in the sorted list of files. ``groups`` sorts the structures by target,
    for a table with one row each. ``nothing_tuned_here`` is for a set that
    takes no part in any choice of parameters: only then are the sites of all
    its structures sorted by enclosure, partners and rank, and not just those
    of its development half. ``cofactors`` names the structures whose ligand
    the set flags as a cofactor, for a row without them.
    """
    named, groups = named or {}, groups or {}
    listed = {ident: place for place, ident in enumerate(sorted(named))}
    files = sorted((p for p in data.rglob("*") if p.is_file() and is_structure(p)),
                   key=lambda p: (name_of(p).lower(), str(p))) if data.is_dir() else []
    rows: list[dict[str, Any]] = []
    structures: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for position, path in enumerate(files):
        name = name_of(path)
        half = "development" if listed.get(name.lower(), position) % 2 == 0 else "held_out"
        try:
            st, text, digest = read_file(path)
            waters = water_oxygens(st)
            if not len(waters):
                raise ValueError("no waters in the structure")
            ligands, passed, settled = ligands_of(st, path.parent, named.get(name.lower()))
            if not ligands:
                raise ValueError("no ligand found; name one in meta.json")
            mine = [evaluate_ligand(st, key, waters, params, name) for key in ligands]
            in_formula = heavy_atoms_by_formula(text)
            for key, row in zip(ligands, mine):
                heavy = sum(1 for a in st.residue_atoms(key) if a.element not in ("H", "D"))
                row["attached"] = covalently_attached(st, key)
                row["complete"] = heavy >= in_formula.get(st.residue_name(key), 0)
                # a by-product, not part of the hold-out: what the burial warning makes of a crystal structure
                burial = structure_burial(st, key)
                row["buried_fraction"] = round(burial.buried_fraction, 3)
                row["burial_warning"] = burial_warning(burial) is not None
                if cofactors:
                    row["cofactor"] = name.lower() in cofactors
        except Exception as exc:  # a structure that cannot be evaluated is reported, not hidden
            failed.append({"structure": name, "half": half, "error": f"{type(exc).__name__}: {exc}"})
            continue
        resolution = resolution_of(text)
        for row in mine:
            row["half"], row["resolution"] = half, resolution
        rows.extend(mine)
        structures.append({"structure": name, "half": half, "resolution": resolution, "waters": int(len(waters)),
                           "ligands": [r["ligand"] for r in mine], "ligand_from": settled, "passed_over": passed,
                           "sha256": digest, **({"group": groups[name.lower()]} if name.lower() in groups else {})})

    def part(half: str | None, sharp: bool = False, group: str | None = None, clean: bool = False,
             drugs: bool = False, band: tuple[float, float] | None = None) -> dict[str, Any]:
        chosen = [{k: v for k, v in r.items() if k not in NOT_COUNTS} for r in rows
                  if (half is None or r["half"] == half)
                  and (group is None or groups.get(r["structure"].lower()) == group)
                  and (not clean or (r["complete"] and not r["attached"]))
                  and (not drugs or not r.get("cofactor"))
                  and (band is None or (r["resolution"] is not None and band[0] < r["resolution"] <= band[1]))
                  and (not sharp or (r["resolution"] is not None and r["resolution"] <= HIGH_RESOLUTION))]
        return summarise(chosen)

    result = {"protocol": {"cuts": list(CUTS), "bridge": BRIDGE, "min_heavy_atoms": MIN_HEAVY, "random_draws": DRAWS,
                           "high_resolution": HIGH_RESOLUTION},
              "numpy": np.__version__, "tuned": bool(tuned), "parameters": dataclasses.asdict(params),
              "files": len(files), "development": part("development"), "held_out": part("held_out"),
              "held_out_high_resolution": part("held_out", sharp=True), "all": part(None),
              "free_and_complete": part(None, clean=True),
              "intervals": {"held_out": intervals([r for r in rows if r["half"] == "held_out"]),
                            "all": intervals(rows)},
              "development_sites": site_breakdown([r for r in rows if r["half"] == "development"], params.n_rays),
              "by_resolution": {label: part(None, band=(lo, hi)) for label, lo, hi in RESOLUTION_BANDS},
              "per_ligand": rows, "structures": structures, "not_evaluated": failed}
    if groups:
        result["by_group"] = {g: part(None, group=g) for g in sorted(set(groups.values()))}
    result["enclosure_contrast"] = enclosure_contrast(rows if nothing_tuned_here
                                                      else [r for r in rows if r["half"] == "development"])
    result["burial"] = burial_of(rows)
    if nothing_tuned_here:
        result["all_sites"] = site_breakdown(rows, params.n_rays)
    if cofactors:
        result["not_a_cofactor"] = part(None, drugs=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--data", default=str(ROOT / "testdata" / "astex"),
                    help="the folder of structures with waters (testdata/astex)")
    ap.add_argument("--out", default=str(ROOT / "tests" / "data"), help="the folder the JSON is written to (tests/data)")
    ap.add_argument("--set", default=str(SET),
                    help="the validation set: a table of structure and ligand, or a manifest.json")
    ap.add_argument("--name", default="", help="a second set writes solvent-validation.<name>.json, not over the first")
    ap.add_argument("--tuned", action="store_true",
                    help="say in the result that the parameters were tuned on the development half")
    args = ap.parse_args()
    result = run(Path(args.data), SolventParams(), tuned=args.tuned, named=named_ligands(Path(args.set)),
                 groups=named_groups(Path(args.set)), nothing_tuned_here=bool(args.name),
                 cofactors=named_cofactors(Path(args.set)))
    if not result["files"]:
        print(f"no structures under {args.data}: the validation has not been run", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("solvent-validation" + (f".{args.name}" if args.name else "") + ".json")
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    text = (whole(result) if args.name else headline(result)) + "\n\n" + table(result) + "\n"
    if result.get("by_group"):
        text += "\n" + group_table(result) + "\n"
    if not args.name:
        text += "\nUnder a figure: “" + footnote(result) + "”\n"
    text += "\nNot part of the protocol:\n\n" + after_the_run(result) + "\n"
    text += "\nA by-product, not part of the hold-out:\n\n" + burial_sentence(result) + "\n"
    print(text)
    print(f"wrote {path}")
    if result["not_evaluated"]:
        print(f"{len(result['not_evaluated'])} structure(s) not evaluated", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
