"""The architectural promise: the core carries no GPL dependency.

PLIP (GPL-2.0) may only ever be imported by ``bindview.detect``, and only when a caller actually asks for
detection. No script and no sketch imports it. MDAnalysis (GPL) and ProLIF are for the sketch of contacts
over a simulation alone.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT, meta_of

SRC = ROOT / "src" / "bindview"
FORBIDDEN = ("plip", "openbabel", "pybel")
#: The simulation tools, by the names they are imported under (in lower case).
SIMULATION = ("mdanalysis", "prolif")


def _imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module.split(".")[0])
    return out


def test_only_the_detector_module_names_plip():
    offenders = {str(p.relative_to(SRC)): sorted(_imports(p) & set(FORBIDDEN))
                 for p in SRC.rglob("*.py") if _imports(p) & set(FORBIDDEN)}
    assert offenders == {"detect.py": ["plip"]}


def test_third_party_imports_are_rdkit_and_numpy_only():
    stdlib = set(sys.stdlib_module_names)
    third = set()
    for p in SRC.rglob("*.py"):
        if p.name == "detect.py":
            continue
        third |= {m for m in _imports(p) if m not in stdlib and m != "bindview"}
    assert third == {"numpy", "rdkit"}


def test_no_script_or_sketch_imports_plip():
    offenders = {str(p.relative_to(ROOT)): sorted(_imports(p) & set(FORBIDDEN))
                 for folder in ("scripts", "prototype") for p in (ROOT / folder).rglob("*.py")
                 if _imports(p) & set(FORBIDDEN)}
    assert offenders == {}


def test_the_simulation_tools_stay_in_the_sketch_of_contacts_over_a_simulation():
    """MDAnalysis and ProLIF may be imported by ``prototype/md_occupancy``, and by no file of the package, its
    tests or its scripts."""
    users = {str(p.relative_to(ROOT)) for folder in ("src", "tests", "scripts", "prototype")
             for p in (ROOT / folder).rglob("*.py") if {m.lower() for m in _imports(p)} & set(SIMULATION)}
    assert all(u.startswith("prototype/md_occupancy/") for u in users), sorted(users)


def test_drawing_from_a_table_never_loads_the_detector():
    """In a fresh interpreter: importing bindview, drawing a diagram from a table in both styles, making the plain
    distance table and the report of a folder with its table load neither PLIP nor ``bindview.detect``."""
    smiles = meta_of("kit_imatinib")["ligand"]["smiles"]
    code = (
        "import sys\n"
        "import bindview\n"
        "from bindview import Diagram, load_interactions, read_structure\n"
        "from bindview.distance_table import ligand_contacts\n"
        "from bindview.report import survey\n"
        "d = 'testdata/kit_imatinib'\n"
        f"smiles = {smiles!r}\n"
        "dia = Diagram.from_structure(d + '/model.cif.gz', 'STI', smiles, load_interactions(d + '/interactions.tsv'))\n"
        "assert dia.render().status == 'ok' and dia.render(style='schematic').status == 'ok'\n"
        "assert ligand_contacts(read_structure(d + '/model.cif.gz'), 'STI', smiles)\n"
        "found = survey(d)\n"
        "found.draw(views=False)\n"
        "assert [t.status for t in found.tracks] == ['ok']\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('plip', 'openbabel', 'pybel') or m == 'bindview.detect']\n"
        "print('LOADED', bad)\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "LOADED []"


def test_declared_dependencies_keep_plip_optional():
    try:
        import tomllib
    except ImportError:  # Python 3.10
        pytest.skip("tomllib needs Python 3.11")
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    core = " ".join(meta["dependencies"]).lower()
    assert "plip" not in core and "openbabel" not in core
    assert any("plip" in dep for dep in meta["optional-dependencies"]["plip"])
