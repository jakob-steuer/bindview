#!/usr/bin/env python
"""Write the contact tables of a set of models of one ligand complex (``testdata/ensemble/<set>``).

A set is a ``manifest.json`` that names its models: the file of each (and, where a file holds several
models, the number of the model in it), which residue of it is the ligand, and the names its chains are
given where the file calls them otherwise than the other models do. This
script reads every model as the manifest says, lets PLIP find its contacts and writes them as one table a
model, beside the manifest. That is what ``bindview.Ensemble`` is handed: the models and a table for each.

It needs PLIP (optional, GPL-2.0), so it runs in an environment that has it:

    python scripts/ensemble_tables.py testdata/ensemble/maob_fad            # write the tables
    python scripts/ensemble_tables.py testdata/ensemble/maob_fad --check    # exit 1 where one differs

Every row PLIP gives is kept; nothing is filtered, merged or repaired. A structure that can be fetched
from the archive is not kept in the set's folder: the manifest says where it is from, and
``scripts/fetch_structures.py`` fetches it.

:func:`structure_of` is how a model of a set is read, here and in the tests: the file as it is, and the
chains under the names the manifest gives them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bindview.interactions import Interaction  # noqa: E402
from bindview.structure import Structure, parse_mmcif, read_models, read_structure  # noqa: E402

#: The columns of a table, in this order. They are the fields of an ``Interaction`` that a detector fills.
COLUMNS = ("interaction_type", "residue_number", "residue_name", "chain", "insertion_code", "protein_atom",
           "ligand_atom", "distance", "angle", "ligand_is_donor", "protein_atoms", "ligand_atoms", "water")


def read_set(folder: "str | Path") -> dict:
    """The manifest of a set."""
    return json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _models_of(path: str) -> "dict[int | None, Structure]":
    """Every model of a file that holds several, by its number: the file is read once for all of them."""
    return {st.model: st for st in read_models(path)}


def structure_of(model: dict, root: Path = ROOT) -> Structure:
    """One model of a set: its file as it is, with the chains named as the manifest says.

    A model's ``chains`` maps the names of its file onto the names the set uses. Residues are matched
    across models by chain and number, so two entries of one protein that call their chains ``A`` and
    ``AAA`` have to be told that these are one chain; nothing else of a file is changed.
    """
    path = root / model["file"]
    number = model.get("model")                      # a model of a file that holds several: its number there
    if number is not None:
        models = _models_of(str(path))
        if number not in models:
            raise KeyError(f"{model['file']} has no model {number}; it has {sorted(k for k in models if k is not None)}")
        st = models[number]
    elif path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            st = parse_mmcif(handle.read(), source=model["id"])
    else:
        st = read_structure(path)
    names = model.get("chains") or {}
    if not names:
        return st
    return Structure(atoms=[replace(a, chain=names.get(a.chain, a.chain)) for a in st.atoms], source=st.source,
                     chem_comp_smiles=st.chem_comp_smiles, plddt_scale=st.plddt_scale, qa_metrics=st.qa_metrics)


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (tuple, list)):
        return ",".join(str(v) for v in value)
    return str(value)


def table_text(records: "list[Interaction]") -> str:
    """Records as the text of a table that ``bindview.load_interactions`` reads back to the same records."""
    lines = ["\t".join(COLUMNS)]
    for r in records:
        row = {"interaction_type": r.type.value, "residue_number": r.residue_number, "residue_name": r.residue_name,
               "chain": r.chain, "insertion_code": r.insertion_code, "protein_atom": r.protein_atom,
               "ligand_atom": r.ligand_atom, "distance": r.distance, "angle": r.angle,
               "ligand_is_donor": r.ligand_is_donor, "protein_atoms": r.protein_atoms,
               "ligand_atoms": r.ligand_atoms, "water": r.water}
        lines.append("\t".join(_cell(row[c]) for c in COLUMNS))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", help="the folder of a set: it holds manifest.json")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 where a table on disk differs")
    args = parser.parse_args()

    from bindview.detect import detect_interactions       # PLIP: only here

    folder = Path(args.folder)
    spec = read_set(folder)
    different = 0
    for model in spec["models"]:
        st = structure_of(model)
        records = detect_interactions(st, model["ligand"])
        text = table_text(records)
        path = folder / model["table"]
        kinds: dict[str, int] = {}
        for r in records:
            kinds[r.type.value] = kinds.get(r.type.value, 0) + 1
        summary = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
        if args.check:
            same = path.exists() and path.read_text(encoding="utf-8") == text
            different += not same
            print(f"{model['id']}: {len(records)} rows ({summary}): {'as on disk' if same else 'DIFFERS from ' + str(path)}")
        else:
            path.write_text(text, encoding="utf-8")
            print(f"{model['id']}: {len(records)} rows ({summary}) -> {path}")
    return 1 if different else 0


if __name__ == "__main__":
    sys.exit(main())
