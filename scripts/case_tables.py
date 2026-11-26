#!/usr/bin/env python
"""Write the contact table of a test case (``testdata/<case>/interactions.tsv``) with PLIP.

A case is a folder with a model and a ``meta.json`` that names the ligand in it. This script reads the
model, lets PLIP find the ligand's contacts and writes them beside the model, one row a contact. That is
the table a caller of bindview brings: the tests and the pictures of the documentation read it.

It needs PLIP (optional, GPL-2.0), so it runs in an environment that has it:

    python scripts/case_tables.py testdata/kit_imatinib            # write the table
    python scripts/case_tables.py testdata/*/ --check              # exit 1 where one differs

Every row PLIP gives is kept; nothing is filtered, merged or repaired. A folder without a ``meta.json``
that names a ligand is passed over.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ensemble_tables import table_text  # noqa: E402

from bindview.structure import read_structure  # noqa: E402


def model_of(folder: Path, meta: dict) -> Path:
    """The model of a case: the file its ``meta.json`` names, in the folder."""
    return folder / meta.get("source", {}).get("file", "model.cif")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folders", nargs="+", help="the folder of a case: it holds meta.json and the model")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 where a table on disk differs")
    args = parser.parse_args(argv)

    from bindview.detect import detect_interactions       # PLIP: only here

    different = 0
    for name in args.folders:
        folder = Path(name)
        meta_file = folder / "meta.json"
        if not meta_file.is_file():
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        ligand = (meta.get("ligand") or {}).get("residue_name_in_cif")
        if not ligand or meta.get("contacts_by") not in (None, "plip"):
            continue
        records = detect_interactions(read_structure(model_of(folder, meta)), ligand)
        text = table_text(records)
        path = folder / "interactions.tsv"
        kinds: dict[str, int] = {}
        for r in records:
            kinds[r.type.value] = kinds.get(r.type.value, 0) + 1
        summary = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())) or "none"
        if args.check:
            same = path.exists() and path.read_text(encoding="utf-8") == text
            different += not same
            print(f"{folder.name}: {len(records)} rows ({summary}): {'as on disk' if same else 'DIFFERS from ' + str(path)}")
        else:
            path.write_text(text, encoding="utf-8")
            print(f"{folder.name}: {len(records)} rows ({summary})")
    return 1 if different else 0


if __name__ == "__main__":
    sys.exit(main())
