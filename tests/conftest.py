"""What the tests are built on: the test systems of ``testdata``, each read and built once for a session.

Every system is open data: entries of the PDB as the archive gives them (``testdata/<slug>``), three models
of the AlphaFold database with the database's own PAE files (``testdata/interface_pae``), one prediction of
three chains (``testdata/interface_multimer``), a frame of a simulation of an RNA with its ligand
(``testdata/nucleic``) and twenty sampled conformations of one protein (``testdata/ensemble_bioemu2``). Each
folder has a ``meta.json`` that says what it is and where it comes from.

A crystal structure has no confidence. Where a test needs a pLDDT on a ligand's pocket, it gets one that is
made up and says so (:func:`invented_plddt`).
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pytest

from bindview import Diagram, load_interactions

ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "testdata"
GOLDEN = Path(__file__).resolve().parent / "golden"
#: The single ligands: one folder each, with ``model.cif.gz``, ``meta.json`` and ``interactions.tsv`` (by PLIP).
CASE_SPECS = {c["slug"]: c for c in json.loads((TESTDATA / "cases.json").read_text())}
CASES = list(CASE_SPECS)
STYLES = ("detailed", "schematic")

_meta: dict[str, dict] = {}
_diagrams: dict[str, Diagram] = {}
_renders: dict[tuple, object] = {}
_structures: dict[str, Any] = {}


def meta_of(case: str) -> dict:
    if case not in _meta:
        _meta[case] = json.loads((TESTDATA / case / "meta.json").read_text())
    return _meta[case]


def model_of(case: str) -> Path:
    """The model of a case: the file its ``meta.json`` names (``model.cif.gz``, as the archive packs an entry)."""
    return TESTDATA / case / meta_of(case).get("source", {}).get("file", "model.cif.gz")


def structure_of(slug: str):
    """The model of a case, read once for the session: do not change it."""
    from bindview.structure import read_structure

    if slug not in _structures:
        _structures[slug] = read_structure(model_of(slug))
    return _structures[slug]


def fresh_diagram(case: str, **kwargs) -> Diagram:
    """A case's diagram as an object of its own: for a test that changes it, or that needs another way of
    building it (``Diagram.from_structure(**kwargs)``)."""
    m = meta_of(case)
    kwargs.setdefault("name", m["ligand"]["name"])
    return Diagram.from_structure(
        model_of(case), ligand=m["ligand"]["residue_name_in_cif"], smiles=m["ligand"]["smiles"],
        interactions=load_interactions(TESTDATA / case / "interactions.tsv"), **kwargs)


def diagram_at(case: str, detail) -> Diagram:
    """A case's diagram with a level of detail for the residues that are drawn with atoms
    (``Diagram.from_structure(detail=...)``); an object of its own each time."""
    return fresh_diagram(case, detail=detail)


def diagram_of(case: str) -> Diagram:
    """A case's diagram, built once for the session: do not change it (:func:`fresh_diagram` gives one to change)."""
    if case not in _diagrams:
        _diagrams[case] = fresh_diagram(case)
    return _diagrams[case]


def render_of(case: str, style: str = "detailed", **options):
    key = (case, style, tuple(sorted(options.items())))
    if key not in _renders:
        _renders[key] = diagram_of(case).render(style=style, **options)
    return _renders[key]


# -- confidence on a crystal: made up, and said to be ------------------------------------------------

def invented_plddt(structure, low: float = 25.0, high: float = 99.0):
    """The same atoms as a predicted model would give them: every atom with a pLDDT, declared as a model
    declares it. **The values are made up for the tests**: 100 minus the atom's temperature factor, held
    between ``low`` and ``high``. A crystal structure has no confidence; this stands in for one, so that what
    is drawn of a confidence is tested on open data. Atoms of one residue get values of their own, as a model
    with a pLDDT for every atom gives them."""
    from dataclasses import replace

    from bindview.structure import Structure

    atoms = [replace(a, b_iso=round(min(high, max(low, 100.0 - (a.b_iso or 0.0))), 2)) for a in structure.atoms]
    return Structure(atoms=atoms, source=f"{structure.source or 'model'}, with an invented pLDDT",
                     chem_comp_smiles=dict(structure.chem_comp_smiles), plddt_scale=1.0,
                     qa_metrics=(("local", "pLDDT", "pLDDT"),), links=structure.links)


def predicted_diagram(slug: str, **kwargs) -> Diagram:
    """A case's diagram on its model with an invented pLDDT (:func:`invented_plddt`); an object of its own."""
    m = meta_of(slug)
    kwargs.setdefault("name", m["ligand"]["name"])
    return Diagram.from_structure(
        invented_plddt(structure_of(slug)), ligand=m["ligand"]["residue_name_in_cif"], smiles=m["ligand"]["smiles"],
        interactions=load_interactions(TESTDATA / slug / "interactions.tsv"), **kwargs)


def without_waters(structure):
    """The same structure without its water molecules: what a predicted model holds, for what is done only
    where a structure has no water of its own (the solvent sites)."""
    from bindview.exposure import WATER_NAMES
    from bindview.structure import Structure

    return Structure(atoms=[a for a in structure.atoms if a.resname.upper() not in WATER_NAMES],
                     source=f"{structure.source or 'model'}, without its waters",
                     chem_comp_smiles=dict(structure.chem_comp_smiles), plddt_scale=structure.plddt_scale,
                     qa_metrics=structure.qa_metrics, links=structure.links)


# -- chains: a peptide or a strand as the ligand, and the interface between chains --------------------

ONE_LETTER = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
              "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
              "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def chain_diagram(slug: str, chain: "str | None" = None, structure=None, **kwargs) -> Diagram:
    """A chain of a case as the ligand (a peptide, a strand of RNA or DNA), with the plain distance table;
    the chain its ``meta.json`` names as the peptide unless another is asked for. An object of its own."""
    from bindview.distance_table import contact_table

    chain = chain or meta_of(slug)["peptide_chain"]
    model = structure if structure is not None else structure_of(slug)
    return Diagram.from_structure(model, f"chain:{chain}", interactions=contact_table(model, chain), **kwargs)


def one_residue_peptide(slug: str, chain: "str | None" = None, name: str = "LIG"):
    """A peptide chain of a case rewritten as ONE hetero residue with atom names of its own, and its SMILES:
    how a co-folding model writes a peptide that it was given as a SMILES. Returns ``(structure, smiles)``;
    the residue is ``name`` in a chain called ``Z``. Only for a chain of the twenty amino acids."""
    from collections import Counter
    from dataclasses import replace

    from rdkit import Chem

    from bindview.exposure import WATER_NAMES
    from bindview.structure import Structure

    chain = chain or meta_of(slug)["peptide_chain"]
    st = structure_of(slug)
    seen: Counter = Counter()
    atoms, sequence, last = [], [], None
    for a in st.atoms:
        if a.chain != chain or a.resname.upper() in WATER_NAMES:
            atoms.append(a)
            continue
        if (a.resnum, a.icode) != last:
            sequence.append(ONE_LETTER[a.resname])
            last = (a.resnum, a.icode)
        if a.element in ("H", "D"):
            continue
        seen[a.element] += 1
        atoms.append(replace(a, chain="Z", resname=name, resnum=1, icode="", name=f"{a.element.upper()}{seen[a.element]}",
                             hetero=True))
    smiles = Chem.MolToSmiles(Chem.MolFromSequence("".join(sequence)))
    return Structure(atoms=atoms, source=f"{slug}, chain {chain} as one residue", plddt_scale=st.plddt_scale,
                     qa_metrics=st.qa_metrics), smiles


def chains_interface(slug: str, a, b, layout: str = "column", structure=None, **options):
    """An :class:`bindview.Interface` between chains of a case, with contacts from the plain distance table.
    ``a`` and ``b`` are a chain name or a list of them. As a column unless asked otherwise, which is the
    library's own default."""
    from bindview.distance_table import interface_contacts
    from bindview.interface import Interface

    model = structure if structure is not None else structure_of(slug)
    one, other = ([x] if isinstance(x, str) else list(x) for x in (a, b))
    names = ["+".join(one), "+".join(other)] if max(len(one), len(other)) > 1 else one + other
    return Interface(model, names, interface_contacts(model, one, other), layout=layout, **options)


# -- a ligand in RNA, a strand as the ligand, two strands (testdata/nucleic) --------------------------

def nucleic_diagram(slug: str = "riboswitch_bisguanidine") -> Diagram:
    """A ligand whose partner is RNA: the model as a PDB file, its SMILES and its contact table, which PLIP
    made with the nucleic acid as the receptor."""
    key = f"nucleic/{slug}"
    if key not in _diagrams:
        folder = TESTDATA / "nucleic" / slug
        m = json.loads((folder / "meta.json").read_text())
        _diagrams[key] = Diagram.from_structure(
            folder / "model.pdb", ligand=m["ligand"]["residue_name_in_cif"], smiles=m["ligand"]["smiles"],
            interactions=load_interactions(folder / "interactions.tsv"), name=m["ligand"]["name"])
    return _diagrams[key]


def nucleic_chain(chain: str = "B", slug: str = "riboswitch_bisguanidine") -> Diagram:
    """A chain of RNA as the ligand: one hairpin of the riboswitch, with the plain distance table."""
    from bindview import read_structure
    from bindview.distance_table import contact_table

    key = f"nucleic/{slug}/chain {chain}"
    if key not in _diagrams:
        model = read_structure(TESTDATA / "nucleic" / slug / "model.pdb")
        _diagrams[key] = Diagram.from_structure(model, f"chain:{chain}", interactions=contact_table(model, chain))
    return _diagrams[key]


def nucleic_interface(a: str = "A", b: str = "B", slug: str = "riboswitch_bisguanidine", **options):
    """The interface between the two hairpins of the riboswitch, with the plain distance table."""
    from bindview import Interface, read_structure
    from bindview.distance_table import interface_contacts

    model = read_structure(TESTDATA / "nucleic" / slug / "model.pdb")
    return Interface(model, [a, b], interface_contacts(model, [a], [b]), **options)


#: The secondary structure of each hairpin of the riboswitch: a stem of seven Watson-Crick pairs and a loop of four.
RIBOSWITCH_HAIRPIN = "(((((((....)))))))"


def nucleic_basepairs(slug: str = "riboswitch_bisguanidine"):
    """The base pairs of the two hairpins of the riboswitch (:class:`bindview.BasePairs`): the stems given as
    a dot-bracket, the pairs and stacks across from the plain distance table, without a family."""
    from bindview import BasePairs, pairs_from_contacts, pairs_from_dot_bracket, read_structure
    from bindview.distance_table import interface_contacts

    model = read_structure(TESTDATA / "nucleic" / slug / "model.pdb")
    stems = [p for c in "AB" for p in pairs_from_dot_bracket(model, c, RIBOSWITCH_HAIRPIN, family="cWW")]
    contacts = interface_contacts(model, ["A"], ["B"], stacking=True)
    across, _ = pairs_from_contacts(contacts)
    return BasePairs(model, ["A", "B"], stems + across, contacts=contacts)


def nucleic_basepairs_rnapolis(slug: str = "riboswitch_bisguanidine"):
    """The same two hairpins with their pairs and stacks as RNApolis's annotator found them (the trimmed
    output committed beside the model), and the other hydrogen bonds across from the plain table."""
    from bindview import BasePairs, read_rnapolis_pairs, read_rnapolis_stacks, read_structure
    from bindview.distance_table import interface_contacts

    folder = TESTDATA / "nucleic" / slug
    model = read_structure(folder / "model.pdb")
    annotation = folder / "basepairs.rnapolis.json"
    return BasePairs(model, ["A", "B"], read_rnapolis_pairs(annotation), stacks=read_rnapolis_stacks(annotation),
                     contacts=interface_contacts(model, ["A"], ["B"], stacking=True))


# -- sets of ligands of one pocket (testdata/overlay) -------------------------------------------------

OVERLAY = TESTDATA / "overlay"
#: ``gr_steroids``: four steroids of the glucocorticoid receptor, one scaffold. ``gr_diverse``: three ligands
#: of that pocket without a common scaffold. ``ar_ligands``: two of the androgen receptor.
SETS = ("gr_steroids", "gr_diverse", "ar_ligands")
_sets: dict[str, dict] = {}
_overlay: dict[tuple, Diagram] = {}


def overlay_set(key: str) -> dict:
    """The ``set.json`` of one set."""
    if key not in _sets:
        _sets[key] = json.loads((OVERLAY / key / "set.json").read_text())
    return _sets[key]


def overlay_members(key: str) -> list[tuple[str, Path]]:
    """``(slug, directory)`` of every member of a set, in the set's own order: the set names single cases
    of ``testdata`` by paths relative to the folder of the sets."""
    out = []
    for rel in overlay_set(key)["members_are_existing_cases_at"]:
        path = next(p for p in ((OVERLAY / rel).resolve(), (OVERLAY / key / rel).resolve()) if p.is_dir())
        out.append((path.name, path))
    return out


def overlay_diagram(key: str, slug: str, **kwargs) -> Diagram:
    ident = (key, slug, tuple(sorted(kwargs.items())))
    if ident not in _overlay:
        folder = dict(overlay_members(key))[slug]
        m = json.loads((folder / "meta.json").read_text())
        _overlay[ident] = Diagram.from_structure(
            folder / m.get("source", {}).get("file", "model.cif.gz"), ligand=m["ligand"]["residue_name_in_cif"],
            smiles=m["ligand"]["smiles"],
            interactions=load_interactions(folder / "interactions.tsv"), name=m["ligand"]["name"], **kwargs)
    return _overlay[ident]


def overlay_meta(key: str, slug: str) -> dict:
    return json.loads((dict(overlay_members(key))[slug] / "meta.json").read_text())


# -- an interface with a predicted aligned error: three models of the AlphaFold database, each cut in two ----

#: A confident interface (PAE about 1 on every contact), an uncertain one (13 to 19) and a mixed one (3 to 13).
PAE_CASES = ("afdb_fah", "afdb_apobec3d", "afdb_tectb")
_pae: dict[tuple, Any] = {}


def pae_case(slug: str, with_pae: bool = True, fresh: bool = False, layout: str = "column", **options):
    """An :class:`bindview.Interface` between the two domains of one of the models in ``testdata/interface_pae``.

    The model is one chain; its two domains (``meta.json``) are given chain names of their own, X and Y.
    Contacts come from the plain distance table, the predicted aligned error from the database's own file for
    the entry. This is PAE between two parts of one predicted chain, not between two separately predicted
    chains. ``fresh`` gives an object of its own, which is not kept. As a column unless asked otherwise.
    """
    from dataclasses import replace

    from bindview.distance_table import interface_contacts
    from bindview.interface import Interface
    from bindview.structure import Structure, read_structure

    key = (slug, with_pae, layout, repr(sorted(options.items())))
    if fresh:
        _pae.pop(key, None)
    if key not in _pae:
        folder = TESTDATA / "interface_pae" / slug
        meta = json.loads((folder / "meta.json").read_text())
        (a0, a1), (b0, b1) = (p["residues"] for p in meta["participants"])
        st = read_structure(folder / "model.cif.gz")
        cut = Structure(atoms=[replace(a, chain="X" if a0 <= a.resnum <= a1 else "Y" if b0 <= a.resnum <= b1 else a.chain)
                               for a in st.atoms], source=slug, plddt_scale=st.plddt_scale, qa_metrics=st.qa_metrics)
        names = [f"the {p['name']}={c}" for p, c in zip(meta["participants"], "XY")]
        contacts = interface_contacts(cut, ["X"], ["Y"])
        _pae[key] = Interface(cut, names, contacts,
                              pae=folder / "predicted_aligned_error.json.gz" if with_pae else None,
                              layout=layout, **options)
    return _pae.pop(key) if fresh else _pae[key]


# -- a predicted multimer with its predicted aligned error: three chains predicted together -----------

MULTIMER = TESTDATA / "interface_multimer" / "tnfa_trimer"
MULTIMER_NAME = "hTNFA_canonical_trimer"
#: The draws of the prediction that are here, 0 and 1: the pair in which the three chains are named in
#: another order.
MULTIMER_KEPT = 2
_trimer: dict[tuple, Any] = {}


def trimer_file(kind: str, draw: int = 0) -> Path:
    """A file of the prediction as Boltz names it: ``model`` (the CIF), ``pae``, ``plddt``, ``pde``, ``confidence``."""
    if kind == "model":
        return MULTIMER / f"{MULTIMER_NAME}_model_{draw}.cif"
    return MULTIMER / f"{kind}_{MULTIMER_NAME}_model_{draw}.{'json' if kind == 'confidence' else 'npz'}"


def trimer_structure(draw: int = 0):
    from bindview.structure import read_structure

    key = ("structure", draw)
    if key not in _trimer:
        _trimer[key] = read_structure(trimer_file("model", draw))
    return _trimer[key]


def trimer_contacts(a: str = "A", b: str = "B", draw: int = 0) -> list:
    """The contacts between two chains of the trimer, from the plain distance table."""
    from bindview.distance_table import interface_contacts

    key = ("contacts", a, b, draw)
    if key not in _trimer:
        _trimer[key] = interface_contacts(trimer_structure(draw), [a], [b])
    return list(_trimer[key])


def trimer(a: str = "A", b: str = "B", draws: int = MULTIMER_KEPT, layout: str = "column", **options):
    """An :class:`bindview.Interface` between two chains of a prediction of human TNF-alpha as a homotrimer
    (``testdata/interface_multimer``), drawn from the draw the model ranks first.

    Contacts from the plain distance table. The predicted aligned error is the model's own: with
    ``draws=2``, as it is left alone, the median over the two draws; with ``draws=1`` that of the draw that
    is drawn; with ``draws=0`` none. This is PAE between chains that were predicted together.
    """
    from bindview.interface import Interface

    key = ("interface", a, b, draws, layout, tuple(sorted(options.items())))
    if key not in _trimer:
        pae = None if not draws else trimer_file("pae") if draws == 1 else [trimer_file("pae", k) for k in range(draws)]
        _trimer[key] = Interface(trimer_structure(), [a, b], trimer_contacts(a, b), pae=pae, layout=layout, **options)
    return _trimer[key]


# -- an ensemble: twenty sampled conformations of adenylate kinase with AP5A (testdata/ensemble_bioemu2) ---

BIOEMU = TESTDATA / "ensemble_bioemu2" / "ake_ap5a.pdb.gz"
#: The ligand of those models: P1,P5-di(adenosine) pentaphosphate.
AP5A = ("Nc1ncnc2n(cnc12)[C@@H]3O[C@H](COP(O)(=O)OP(O)(=O)OP(O)(=O)OP(O)(=O)OP(O)(=O)OC[C@H]4O[C@H]([C@H](O)"
        "[C@@H]4O)n5cnc6c(N)ncnc56)[C@@H](O)[C@H]3O")
_samples: list[Any] = []


def bioemu_samples():
    """The twenty models as a :class:`bindview.Ensemble`, with the plain distance table for each; built once for
    the session: do not change it."""
    from bindview import Ensemble, read_models
    from bindview.distance_table import ligand_contacts

    if not _samples:
        models = read_models(BIOEMU)
        _samples.append(Ensemble.from_structures(models, "AP5", AP5A, [ligand_contacts(m, "AP5", AP5A) for m in models],
                                                 ensemble="20 samples of BioEmu 2"))
    return _samples[0]


# -- what a test may need that is not always there ----------------------------------------------------

def fetched(*relative: str):
    """The mark of a test that reads structures which are fetched and not kept in the repository
    (``scripts/fetch_structures.py``): each argument a path under ``testdata``. The test is skipped where
    one of them is absent, as in a fresh clone."""
    def there(rel: str) -> bool:
        path = TESTDATA / rel
        if path.is_dir():                    # (a folder whose list is kept is there in a clone, and empty)
            return any(p.name != "manifest.json" for p in path.iterdir())
        return path.exists()

    missing = [r for r in relative if not there(r)]
    return pytest.mark.skipif(bool(missing), reason=f"fetched, not kept: testdata/{missing[0] if missing else ''} "
                                                    f"(scripts/fetch_structures.py)")


#: The mark of a test that runs PLIP, which is optional and under another licence: skipped where it is absent.
needs_plip = pytest.mark.skipif(importlib.util.find_spec("plip") is None,
                                reason="PLIP is optional and is not installed here")


def golden_versions() -> dict[str, str]:
    """What the stored figures depend on, as this environment has it: RDKit lays out the ligand, and its
    coordinates differ in the last digits between platforms and between its releases."""
    import numpy
    import rdkit

    return {"rdkit": rdkit.__version__, "numpy": numpy.__version__.split(".")[0],
            "platform": f"{sys.platform}-{platform.machine()}"}


def on_golden_platform() -> bool:
    """Is this the environment the stored figures and numbers were made in (``tests/golden/VERSIONS.json``)?
    Only there are they compared byte for byte, or digit for digit."""
    stored = GOLDEN / "VERSIONS.json"
    return stored.exists() and json.loads(stored.read_text()) == golden_versions()


@pytest.fixture(scope="session")
def testdata() -> Path:
    return TESTDATA


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run tests marked slow")
    parser.addoption("--update-golden", action="store_true", default=False,
                     help="rewrite tests/golden from the current output instead of comparing")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes tens of seconds")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
