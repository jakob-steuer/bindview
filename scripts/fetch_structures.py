#!/usr/bin/env python
"""The deposited structures that are listed in the repository and not kept in it: what is on disk, and fetching the rest.

    python scripts/fetch_structures.py                 # compare what is on disk with the lists and the checksums
    python scripts/fetch_structures.py --download      # fetch what is missing from the RCSB PDB
    python scripts/fetch_structures.py --record        # write the checksums of what is on disk

Three sets are read from the archive: the structures of the solvent-site validation
(``scripts/solvent_validation_set.tsv``, kept in ``testdata/astex``), the entries of its second set
(``testdata/waters/manifest.json``), and what the ``manifest.json`` of a set of models under
``testdata/ensemble`` names under ``fetch``. Nothing else needs them: the tests that read one are skipped
where it is absent.

``testdata/structures.sha256`` holds the SHA-256 of every file (in the format ``shasum -a 256 -c`` reads). A file
is put in its place only when it is whole and has the recorded checksum. The archive revises its files from time
to time without changing a coordinate; a file that is whole and has another checksum is left out, and taken
with ``--revised``.

Nothing is fetched unless ``--download`` is given.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "testdata"
SUMS = TESTDATA / "structures.sha256"
URL = "https://files.rcsb.org/download/{id}.cif.gz"
SETS = ("astex", "waters", "ensemble")


def listed() -> list[tuple[str, str, Path]]:
    """Every structure a list names: ``(set, PDB id, where it is kept)``, each as the archive sends it."""
    out: list[tuple[str, str, Path]] = []
    table = ROOT / "scripts" / "solvent_validation_set.tsv"
    rows = [line for line in table.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    for row in csv.DictReader(rows, delimiter="\t"):
        pdb = row["pdb_id"].lower()
        out.append(("astex", pdb.upper(), TESTDATA / "astex" / pdb / f"{pdb}.cif.gz"))
    for entry in json.loads((TESTDATA / "waters" / "manifest.json").read_text(encoding="utf-8")).get("entries", []):
        out.append(("waters", entry["pdb_id"].upper(), TESTDATA / "waters" / (entry["file"] + ".gz")))
    for manifest in sorted((TESTDATA / "ensemble").glob("*/manifest.json")):
        for entry in json.loads(manifest.read_text(encoding="utf-8")).get("fetch", []):
            out.append(("ensemble", entry["pdb_id"].upper(), ROOT / entry["file"]))
    return out


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def recorded() -> dict[str, str]:
    if not SUMS.exists():
        return {}
    pairs = (line.split(None, 1) for line in SUMS.read_text(encoding="utf-8").splitlines() if line.strip())
    return {name.strip().lstrip("*"): digest for digest, name in pairs}


def whole(data: bytes) -> str:
    """Why ``data`` is no entry of the archive; empty where it is one."""
    try:
        text = gzip.decompress(data)
    except (OSError, EOFError) as exc:
        return f"it is not a whole gzip file ({exc})"
    return "" if text.lstrip()[:5] == b"data_" else "it is no mmCIF file"


def fetch(pdb: str) -> bytes:
    request = urllib.request.Request(URL.format(id=pdb), headers={"User-Agent": "fetch_structures.py"})
    with urllib.request.urlopen(request, timeout=60) as response:      # noqa: S310 (a fixed https address)
        return response.read()


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--download", action="store_true", help="fetch what is listed and not on disk")
    ap.add_argument("--revised", action="store_true", help="also take a fetched file whose checksum is not the recorded one")
    ap.add_argument("--record", action="store_true", help=f"write {SUMS.relative_to(ROOT)} from what is on disk")
    ap.add_argument("--only", metavar="SET", choices=SETS, help="one set: " + ", ".join(SETS))
    args = ap.parse_args(argv)
    entries = [e for e in listed() if args.only in (None, e[0])]
    known = recorded()
    missing = differing = unrecorded = refused = 0
    lines = []
    for name, pdb, path in entries:
        rel = str(path.relative_to(TESTDATA))
        if not path.exists() and args.download:
            data = fetch(pdb)
            broken = whole(data)
            changed = rel in known and known[rel] != sha256(data)
            if broken or (changed and not args.revised):
                refused += 1
                print(f"NOT TAKEN {pdb}: " + (broken or "its checksum is not the recorded one (take it with --revised)"))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                part = path.with_name(path.name + ".part")
                part.write_bytes(data)
                part.replace(path)
                print(f"fetched {pdb} -> testdata/{rel}: {len(data)} bytes" + (", REVISED" if changed else ""))
        if not path.exists():
            missing += 1
            print(f"missing   {name:9s} {pdb}  testdata/{rel}")
            continue
        digest = sha256(path.read_bytes())
        lines.append(f"{digest}  {rel}")
        if rel not in known:
            unrecorded += 1
            if not args.record:
                print(f"no record {name:9s} {pdb}  testdata/{rel}")
        elif known[rel] != digest:
            differing += 1
            print(f"differs   {name:9s} {pdb}  testdata/{rel}: not the file whose checksum is recorded")
    if args.record:
        SUMS.write_text("\n".join(sorted(lines, key=lambda line: line.split("  ", 1)[1])) + "\n", encoding="utf-8")
        print(f"recorded {len(lines)} checksums in {SUMS.relative_to(ROOT)}")
    print(f"{len(entries)} listed, {len(entries) - missing} on disk, {missing} missing, {differing} differing, "
          f"{refused} not taken" + ("" if args.record else f", {unrecorded} without a recorded checksum"))
    return 1 if (missing and not args.download) or differing or refused else 0


if __name__ == "__main__":
    sys.exit(main())
