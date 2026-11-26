"""The stored figures: on the platform they were made on, a figure must not drift by a byte.

``tests/golden/VERSIONS.json`` records what the bytes depend on: the release of RDKit, whose CoordGen lays
out the ligand, the major version of NumPy, and the platform (RDKit's 2D coordinates differ in the last
digits between macOS on arm64 and Linux on x86-64, and the layout follows them). In that environment every
figure is compared with its stored file byte for byte. In any other, the same figures are drawn and held
to what does not depend on those digits: the status, and what the figure holds (its residues, its contacts
by kind, its ligand's atoms). To store the figures again, on purpose and after looking at them:

    pytest tests/test_golden.py --update-golden
"""

from __future__ import annotations

import json
import re

import pytest

import conftest
from conftest import CASES, GOLDEN, STYLES, golden_versions, on_golden_platform

VERSIONS = GOLDEN / "VERSIONS.json"
CONTENTS = GOLDEN / "CONTENTS.json"


def _case(case, **options):
    return lambda: conftest.diagram_of(case).render(**options)


def _comparison(key, slugs=None, **options):
    def build():
        from bindview import Comparison

        members = slugs or [slug for slug, _ in conftest.overlay_members(key)]
        return Comparison([conftest.overlay_diagram(key, slug) for slug in members]).render(**options)
    return build


def _series_member(key, slug):
    def build():
        from bindview import Comparison

        members = [s for s, _ in conftest.overlay_members(key)]
        return Comparison([conftest.overlay_diagram(key, s) for s in members]).diagram(members.index(slug)).render()
    return build


def _one_residue_peptide(style):
    def build():
        from bindview import Diagram
        from bindview.distance_table import ligand_contacts

        structure, smiles = conftest.one_residue_peptide("mhc_peptide")
        return Diagram.from_structure(structure, "LIG", smiles=smiles,
                                      interactions=ligand_contacts(structure, "LIG", smiles)).render(style=style)
    return build


def _solvent_sites(case):
    def build():
        m = conftest.meta_of(case)
        from bindview import Diagram, load_interactions

        dry = conftest.without_waters(conftest.structure_of(case))
        dia = Diagram.from_structure(dry, ligand=m["ligand"]["residue_name_in_cif"], smiles=m["ligand"]["smiles"],
                                     interactions=[r for r in load_interactions(conftest.TESTDATA / case / "interactions.tsv")
                                                   if r.type.value != "water_bridge"], name=m["ligand"]["name"])
        return dia.render(burial=True, solvent_sites=True)
    return build


#: Every stored figure that is not one of the single ligands in a style, by the name of its file.
FIGURES = {
    # A peptide as the ligand, a box a residue: ten residues of hirudin on thrombin (one a sulfated tyrosine),
    # semaglutide on its receptor (a modified residue and a group on a lysine), and a peptide that the file
    # holds as one residue with a SMILES.
    "chain.thrombin_hirudin.schematic": lambda: conftest.chain_diagram("thrombin_hirudin").render(style="schematic"),
    "chain.thrombin_hirudin.detailed": lambda: conftest.chain_diagram("thrombin_hirudin").render(style="detailed"),
    "chain.glp1r_semaglutide.schematic": lambda: conftest.chain_diagram("glp1r_semaglutide").render(style="schematic"),
    "chain.mhc_peptide_one_residue.schematic": _one_residue_peptide("schematic"),
    # An interface between chains: the two chains of thrombin, an antibody on its antigen, a protein on DNA
    # and a protein on RNA.
    "interface.thrombin_light_heavy.schematic": lambda: conftest.chains_interface(
        "thrombin_hirudin", "A", "B").render(style="schematic"),
    "interface.fab_lysozyme.schematic": lambda: conftest.chains_interface(
        "fab_lysozyme", ["H", "L"], "C").render(style="schematic"),
    "interface.repressor_dna.schematic": lambda: conftest.chains_interface(
        "repressor_dna", "3", ["1", "2"]).render(style="schematic"),
    "interface.u1a_rna.schematic": lambda: conftest.chains_interface("u1a_rna", "A", "P").render(style="schematic"),
    # The predicted aligned error of a contact, on three models of the AlphaFold database cut at a domain border.
    "pae.afdb_fah.schematic": lambda: conftest.pae_case("afdb_fah").render(style="schematic"),
    "pae.afdb_apobec3d.schematic": lambda: conftest.pae_case("afdb_apobec3d").render(style="schematic"),
    "pae.afdb_tectb.schematic": lambda: conftest.pae_case("afdb_tectb").render(style="schematic"),
    "pae.afdb_tectb.detailed": lambda: conftest.pae_case("afdb_tectb").render(style="detailed"),
    # The same interfaces with the layout left to the rule: boxes where the residues lie.
    "plane.thrombin_light_heavy.schematic": lambda: conftest.chains_interface(
        "thrombin_hirudin", "A", "B", layout="auto").render(style="schematic"),
    "plane.afdb_apobec3d.detailed": lambda: conftest.pae_case("afdb_apobec3d", layout="auto").render(style="detailed"),
    "plane.afdb_tectb.schematic": lambda: conftest.pae_case("afdb_tectb", layout="auto").render(style="schematic"),
    # A prediction of three chains: the PAE of a contact as the median over two draws, one of which names
    # the copies of the chain in another order; and the same interface on the plane.
    "multimer.tnfa_ab.schematic": lambda: conftest.trimer("A", "B").render(style="schematic"),
    "multimer.tnfa_ab.detailed": lambda: conftest.trimer("A", "B").render(style="detailed"),
    "multimer.tnfa_bc.schematic": lambda: conftest.trimer("B", "C").render(style="schematic"),
    "multimer.tnfa_ab_plane.schematic": lambda: conftest.trimer("A", "B", layout="auto").render(style="schematic"),
    # How much of a residue is drawn (``detail``): side chains of the residues that make directional contacts,
    # a column with its backbone, and the residues around a ligand with every atom.
    "detail.tnfa_ab.sidechain.detailed": lambda: conftest.trimer("A", "B", detail="sidechain").render(style="detailed"),
    "detail.tnfa_ab.sidechain.schematic": lambda: conftest.trimer("A", "B", detail="sidechain").render(style="schematic"),
    "detail.afdb_tectb.sidechain.detailed": lambda: conftest.pae_case(
        "afdb_tectb", detail="sidechain").render(style="detailed"),
    "detail.afdb_apobec3d.backbone.detailed": lambda: conftest.pae_case(
        "afdb_apobec3d", detail="backbone").render(style="detailed"),
    "detail.ar_bicalutamide.atoms.detailed": lambda: conftest.diagram_at("ar_bicalutamide", "atoms").render(
        style="detailed"),
    # A ligand in RNA; a strand of RNA as the ligand; the interface between two strands; and the two as
    # their secondary structures facing each other, from a dot-bracket and from an annotator's output.
    "nucleic.riboswitch_bisguanidine.detailed": lambda: conftest.nucleic_diagram().render(style="detailed"),
    "nucleic.riboswitch_bisguanidine.schematic": lambda: conftest.nucleic_diagram().render(style="schematic"),
    "nucleic.riboswitch_chain_b.detailed": lambda: conftest.nucleic_chain().render(style="detailed"),
    "nucleic.riboswitch_interface.detailed": lambda: conftest.nucleic_interface().render(style="detailed"),
    "nucleic.riboswitch_interface.schematic": lambda: conftest.nucleic_interface().render(style="schematic"),
    "nucleic.riboswitch_basepairs.light": lambda: conftest.nucleic_basepairs().render(),
    "nucleic.riboswitch_basepairs.auto": lambda: conftest.nucleic_basepairs().render(theme="auto"),
    "nucleic.riboswitch_basepairs.rnapolis": lambda: conftest.nucleic_basepairs_rnapolis().render(),
    # What a figure says of the pocket: how much of the ligand is buried, and where a water could bridge.
    "pocket.burial.dhfr_methotrexate": _case("dhfr_methotrexate", burial=True),
    "pocket.solvent.ar_r1881": _solvent_sites("ar_r1881"),
    # A metal the protein holds between it and the ligand, and a ligand bonded to a residue.
    "ions.ca2_celecoxib.detailed": lambda: conftest.fresh_diagram("ca2_celecoxib", ions=True).render(),
    "covalent.egfr_osimertinib.detailed": lambda: conftest.fresh_diagram("egfr_osimertinib", covalent=True).render(),
    # Confidence, on crystals with a pLDDT that is made up for the tests (conftest.invented_plddt).
    "confidence.kit_imatinib.detailed": lambda: conftest.predicted_diagram("kit_imatinib").render(style="detailed"),
    "confidence.gr_dexamethasone.schematic": lambda: conftest.predicted_diagram(
        "gr_dexamethasone", trusted=False, flag="grade C").render(style="schematic"),
    # Several ligands of one pocket: two side by side, a set on its common scaffold, a set without one.
    "difference.ar_ligands": _comparison("ar_ligands"),
    "difference.gr_pair": _comparison("gr_steroids", ["gr_dexamethasone", "gr_budesonide"]),
    "consensus.gr_steroids": _comparison("gr_steroids"),
    "consensus.gr_diverse": _comparison("gr_diverse", theme="dark"),
    "occupancy.gr_dexamethasone.detailed": _series_member("gr_steroids", "gr_dexamethasone"),
    # All the models of an ensemble, counted by residue: twenty sampled conformations of adenylate kinase.
    "ensemble.ake_consensus.detailed": lambda: conftest.bioemu_samples().consensus().render(bond_length=26.0),
}
for _case_, _style in ((c, s) for c in CASES for s in STYLES):
    FIGURES[f"{_case_}.{_style}"] = (lambda c=_case_, s=_style: conftest.render_of(c, s))


def contents(result) -> dict:
    """What a figure holds that does not depend on the last digits of a coordinate: its status, how many
    residues, contacts of each kind and ligand atoms it draws, and its words (numbers left out)."""
    from collections import Counter

    from bindview.scene import Text
    from scenetools import groups, walk

    kinds = Counter(g.data.get("type", "") for g in groups(result, "interaction"))
    words = sorted({re.sub(r"[-+]?\d+(?:\.\d+)?", "#", f.item.content) for f in walk(result.scene)
                    if isinstance(f.item, Text)})
    return {"status": result.status, "residues": len(groups(result, "residue")), "contacts": dict(sorted(kinds.items())),
            "ligand_atoms": len(groups(result, "atom")), "texts": len(words)}


@pytest.fixture(scope="module")
def updating(request) -> bool:
    update = bool(request.config.getoption("--update-golden"))
    if update:
        GOLDEN.mkdir(exist_ok=True)
        VERSIONS.write_text(json.dumps(golden_versions(), indent=1) + "\n")
    return update


_held: dict[str, dict] = {}


@pytest.mark.parametrize("name", sorted(FIGURES))
def test_a_figure_is_the_stored_one(name, updating):
    result = FIGURES[name]()
    svg = result.svg
    assert svg is not None
    path = GOLDEN / f"{name}.svg"
    if updating:
        path.write_text(svg, encoding="utf-8")
        _held[name] = contents(result)
        return
    stored = json.loads(CONTENTS.read_text())
    assert contents(result) == stored[name], f"{name}: the figure holds something else than the stored one"
    if not on_golden_platform():
        return                                           # (the bytes are promised in one environment only)
    assert path.exists(), f"missing golden file {path.name}; run with --update-golden"
    expected = path.read_text(encoding="utf-8")
    if svg != expected:
        got, want = svg.splitlines(), expected.splitlines()
        first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        pytest.fail(f"{path.name} differs from line {first + 1}:\n  expected: "
                    f"{want[first][:160] if first < len(want) else '<end>'}"
                    f"\n  got:      {got[first][:160] if first < len(got) else '<end>'}")


def test_the_stored_set_is_complete(updating):
    """Every figure of the list has its file and its entry, and no file is left that the list does not draw."""
    if updating:
        assert set(_held) == set(FIGURES), "run the whole module to store the figures again"
        CONTENTS.write_text(json.dumps(dict(sorted(_held.items())), indent=1) + "\n")
        return
    assert sorted(p.stem for p in GOLDEN.glob("*.svg")) == sorted(FIGURES)
    assert sorted(json.loads(CONTENTS.read_text())) == sorted(FIGURES)
