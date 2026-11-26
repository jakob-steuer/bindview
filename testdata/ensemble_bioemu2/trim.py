"""Write ``ake_ap5a.pdb.gz`` from the samples BioEmu 2 made (testdata/bioemu2/ake_1ake/models, not in the repository):
the residues with an atom within 10 Å of the ligand in any of the 20 samples, and the ligand.

    python testdata/ensemble_bioemu2/trim.py
"""
import gzip
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from bindview.structure import read_structure  # noqa: E402

models = [read_structure(p) for p in sorted((HERE.parent / "bioemu2" / "ake_1ake" / "models").glob("*.cif"))]
keep = set()
for st in models:
    lig = np.array([a.xyz for a in st.atoms if a.resname == "AP5"])
    for a in st.atoms:
        if a.resname != "AP5" and np.linalg.norm(lig - np.array(a.xyz), axis=1).min() <= 10.0:
            keep.add(a.residue)
lines = ["REMARK  20 samples of adenylate kinase with AP5A by BioEmu 2 (README.md); predictions, not measurements",
         "REMARK  trimmed to the residues within 10 A of the ligand in any sample"]
for k, st in enumerate(models, 1):
    lines.append(f"MODEL     {k:>4}")
    n = 0
    for a in st.atoms:
        if a.resname != "AP5" and a.residue not in keep:
            continue
        n += 1
        rec = "HETATM" if a.resname == "AP5" else "ATOM"
        name = a.name if len(a.name) == 4 else f" {a.name:<3}"
        x, y, z = a.xyz
        lines.append(f"{rec:<6}{n:>5} {name:<4} {a.resname:>3} {a.chain}{a.resnum:>4}    {x:8.3f}{y:8.3f}{z:8.3f}"
                     f"  1.00  0.00          {a.element.upper():>2}")
    lines.append("ENDMDL")
lines.append("END")
with open(HERE / "ake_ap5a.pdb.gz", "wb") as fh:
    fh.write(gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0))
print(len(models), "models,", len(keep), "residues")
