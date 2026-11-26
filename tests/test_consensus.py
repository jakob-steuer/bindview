"""All the models of an ensemble in one figure, counted by residue (``Ensemble.consensus``).

The rules are tested on tables made for the test from committed structures: bicalutamide in the androgen receptor
(``ar_bicalutamide``), whose table has a hydrogen bond to Gln711, a halogen bond to Met742 and hydrophobic contacts,
and imatinib in c-Kit (``kit_imatinib``). Nothing is concluded from those about any complex. The numbers are those
of the 20 samples of adenylate kinase that are committed (``testdata/ensemble_bioemu2``), and of the NMR bundle
where its file is on disk.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import (AP5A, CASES, TESTDATA, bioemu_samples, chain_diagram, diagram_of, fetched, invented_plddt,
                      meta_of, structure_of)
from bindview import Diagram, Ensemble, EnsembleError, ResidueShare, load_interactions, read_models
from bindview.consensus import PIECE, _Made, ligand_parts, lines_of
from bindview.ensemble import _symmetries
from bindview.interactions import InteractionType
from bindview.scene import Circle, Polyline, Text
from bindview.structure import Structure
from scenetools import groups, inside, interactions, walk, words

T = InteractionType
CASE = "ar_bicalutamide"                 # the committed structure the made-up tables are cut from
KIT = "kit_imatinib"


def table(case: str = CASE):
    return load_interactions(TESTDATA / case / "interactions.tsv")


def build(structure=None, records=None, case: str = CASE) -> Diagram:
    """A committed case, with another structure or another table in its place."""
    m = meta_of(case)
    return Diagram.from_structure(structure if structure is not None else structure_of(case),
                                  m["ligand"]["residue_name_in_cif"], m["ligand"]["smiles"],
                                  table(case) if records is None else records)


def changed(st: Structure, move) -> Structure:
    """A structure with every atom put through ``move(atom) -> atom``."""
    return Structure(atoms=[move(a) for a in st.atoms], source=st.source, chem_comp_smiles=st.chem_comp_smiles,
                     plddt_scale=st.plddt_scale, qa_metrics=st.qa_metrics)


def four(structure=None) -> Ensemble:
    """Four tables of one structure: all of it, without the hydrogen bonds, without the halogen bond too, none."""
    full = table()
    no_hbond = [r for r in full if r.type.value != "hbond"]
    bare = [r for r in no_hbond if r.type.value != "halogen"]
    return Ensemble([build(structure, t) for t in (full, no_hbond, bare, [])], ensemble="four made-up tables")


def shares(ens: Ensemble) -> dict:
    return {r.residue_label: (r.n_models, r.n_total) for r in ens.residues}


# ---------------------------------------------------------------------------
# What is counted
# ---------------------------------------------------------------------------

def test_a_residue_is_counted_in_every_model_that_has_any_contact_with_it():
    ens = four()
    got = shares(ens)
    # Gln711 has a hydrogen bond only; Leu704 has a hydrophobic contact besides it, Arg752 a water bridge.
    assert got["Gln711"] == (1, 4) and got["Met742"] == (2, 4)
    assert got["Leu704"] == got["Arg752"] == got["Leu707"] == (3, 4)
    assert all(isinstance(r, ResidueShare) for r in ens.residues)
    counts = [r.n_models for r in ens.residues]
    assert counts == sorted(counts, reverse=True)                                # the most frequent first
    met = next(r for r in ens.residues if r.residue_label == "Met742")
    assert met.models == (0, 1) and met.share == 0.5 and met.kinds == ((T.HALOGEN, 2),)
    assert met.said == "contacted in 2 of 4 models (halogen bond 2)"
    # The same numbers as the table of the ensemble has for its residues, whatever the order of the models.
    rows = {r.label: r.n_ligands for r in ens.table.consensus()}
    assert {k: n for k, (n, _) in got.items()} == rows
    full = table()
    turned = Ensemble([build(records=t) for t in ([], [r for r in full if r.type.value != "hbond"], full, full)])
    back = Ensemble([build(records=t) for t in (full, full, [r for r in full if r.type.value != "hbond"], [])])
    assert shares(turned) == shares(back)


@pytest.mark.parametrize("case", CASES)
def test_every_heavy_atom_of_a_ligand_is_in_one_part(case):
    lig = diagram_of(case).ligand
    parts = ligand_parts(lig)
    heavy = sorted(a.GetIdx() for a in lig.mol.GetAtoms() if a.GetAtomicNum() > 1)
    assert sorted(i for p in parts for i in p) == heavy
    assert ligand_parts(lig) == parts


def test_the_parts_are_the_rings_the_end_groups_and_the_long_pieces_between_them():
    lig = bioemu_samples().diagrams[0].ligand
    parts = [{lig.names[i] for i in p} for p in ligand_parts(lig)]
    assert len(parts) == 9                                                       # 2 adenines, 2 riboses, 5 phosphates
    assert sum(any(n.startswith("N9") for n in p) for p in parts) == 2
    phosphates = [p for p in parts if any(n.startswith("P") for n in p)]
    assert len(phosphates) == 5 and all(sum(n.startswith("P") for n in p) == 1 for p in phosphates)
    mtx = diagram_of("dhfr_methotrexate").ligand
    parts = [{mtx.names[i] for i in p} for p in ligand_parts(mtx)]
    assert len(parts) == 5                        # the pteridine, the ring of the benzoyl, two carboxylates, the chain
    ends = [p for p in parts if len(p) == 3 and sum(n.startswith("O") for n in p) == 2]
    assert sorted(sorted(p) for p in ends) == [["CD", "OE1", "OE2"], ["CT", "O1", "O2"]]
    assert any(len(p) == 6 for p in parts) and PIECE == 4                        # a piece of six atoms is its own


def test_a_line_goes_to_the_part_that_most_models_have_the_site_on():
    ens = bioemu_samples()
    by = {r.residue_label: r for r in ens.residues}
    parts = ligand_parts(ens.diagrams[0].ligand)
    names = ens.diagrams[0].ligand.names
    here = ens.diagrams[ens.member].ligand.names
    owner = {names[i]: k for k, p in enumerate(parts) for i in p}

    def said(label):
        return [(x.type.value, owner[here[x.ligand.atoms[0]]], round(x.record.occupancy * 20), x.record.protein_atom)
                for x in lines_of(ens, by[label], True)]
    # Gly10 is contacted in 18 models: in 14 its N is on the phosphate it has a line to, in 9 on the one beside.
    gly = by["Gly10"]
    assert sorted((len(gly.models_at(p)) for p in gly.places), reverse=True)[:2] == [14, 9]
    ((kind, _, n, atom),) = said("Gly10")                                        # one line for one site
    assert (kind, n, atom) == ("hbond", 14, "N")
    # Lys13 has two sites, its backbone N and its NZ, and two lines.
    lys = said("Lys13")
    assert [(n, atom) for _, _, n, atom in lys][0] == (16, "N") and len(lys) == 2
    assert lys[0][1] != lys[1][1]
    # A line that fewer than half of the residue's models make is not drawn: Arg123 has one, and not its 6 of 18.
    arg = by["Arg123"]
    assert [n for _, _, n, _ in said("Arg123")] == [14] and 6 in {len(arg.models_at(p)) for p in arg.places}
    for r in ens.residues:
        got = lines_of(ens, r, True)
        assert got and got[0].record.occupancy == max(len(r.models_at(p)) for p in r.places) / ens.n
        assert all(2 * round(x.record.occupancy * ens.n) >= r.n_models for x in got[1:])
        assert len(lines_of(ens, r, False)) == 1                                 # its ring stands by its best part


def test_the_ends_of_a_symmetric_ligand_are_named_so_that_the_contacts_agree():
    """Two models that differ in nothing but which of the two adenosines is called A."""
    base = bioemu_samples().diagrams[bioemu_samples().member]
    lig = base.ligand
    sym, _ = _symmetries(lig)
    flip = next(p for p in sym if lig.names[p[lig.names.index("N6A")]] == "N6B")
    other = {lig.names[i]: lig.names[int(flip[i])] for i in range(len(lig.names))}
    rows = [r.record for r in base.interactions]
    turned = [r.with_(ligand_atom=other[r.ligand_atom] if r.ligand_atom else None,
                      ligand_atoms=tuple(other[a] for a in r.ligand_atoms)) for r in rows]

    def made(records):
        return Diagram.from_structure(base.structure, base.ligand_key, AP5A, records)
    ens = Ensemble([made(rows), made(turned)], member=0)
    assert shares(ens) == {k: (2, 2) for k in shares(ens)}
    dia = ens.consensus()
    assert all(r.record.occupancy == 1.0 for r in dia.interactions)              # named as they are, half would be 0.5
    assert any("ends that are alike: in 1 model they are named the other way round" in t for t in dia.remarks)
    same = Ensemble([made(rows), made(rows)], member=0).consensus()
    assert not any("ends that are alike" in t for t in same.remarks)


# ---------------------------------------------------------------------------
# What is drawn
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", ("detailed", "schematic"))
def test_residues_stand_in_rings_and_half_of_them_are_drawn_as_atoms(style):
    ens = four()
    dia = ens.consensus()
    assert isinstance(dia, Diagram) and dia.ensemble == "four made-up tables"
    assert ens.diagrams[0].shares == {} and ens.diagrams[0].remarks == []        # the members are untouched
    res = dia.render(style=style)
    assert res.status == "ok" and res.quality.valid
    drawn = {g.data["residue"].split(":")[-1].title(): g for g in groups(res, "residue")}
    assert set(drawn) == {r.residue_label for r in ens.residues if r.share >= 0.25}
    for r in ens.residues:
        g = drawn[r.residue_label]
        assert g.data["n-models"] == f"{r.n_models}/4" and g.title == f"{r.residue_label}: {r.said}"
        text = [f.item.content for f in inside(res, g) if isinstance(f.item, Text)]
        assert r.residue_label in text and f"{r.n_models}/4" in text
        # As atoms where one of its lines has a direction: Leu704's hydrogen bond is one model's, its line hydrophobic.
        directed = any(x.type is not T.HYDROPHOBIC for x in lines_of(ens, r, True))
        assert g.data["form"] == ("fragment" if r.share >= 0.5 and directed else "bubble")
        # The ring is as long as the share.
        ring = [f.item for f in inside(res, g) if isinstance(f.item, Polyline) and f.item.paint.stroke == "text"
                and f.item.paint.width > 3.0]
        (arc,) = ring
        centre = next(f.item for f in inside(res, g) if isinstance(f.item, Circle))
        length = sum(math.dist(a, b) for a, b in zip(arc.points, arc.points[1:]))
        radius = math.dist(arc.points[0], (centre.cx, centre.cy))
        assert length / (2 * math.pi * radius) == pytest.approx(r.share, abs=0.01)
    # A line where a residue is drawn as atoms, with its share and no distance; a ring alone has none.
    assert interactions(res, "halogen", "MET742") and not interactions(res, "hbond", "GLN711")
    text = words(res)
    assert "Ring: share of four made-up tables that contact the residue" in text
    assert "Counted by residue, over all 4 models: no single one is drawn" in text
    assert "Distances" not in text and "Ligand buried" not in text and res.burial is None
    assert ens.diagrams[0].render(style=style).burial is not None
    assert "Pocket lining" not in text and "Solvent exposure" not in text
    # Every contact a ring does not draw is still no line: the same figure as rings alone.
    rings = dia.render(style=style, residues="bubbles")
    assert not groups(rings, "interaction") and {g.data["form"] for g in groups(rings, "residue")} == {"bubble"}


def test_a_line_stands_on_the_atoms_that_the_models_have_it_on():
    ens = bioemu_samples()
    for r in ens.residues:
        for x in lines_of(ens, r, True):
            assert x.method == "named" and not x.notes and x.record.distance is None
            assert all(0 <= i < ens.diagrams[ens.member].ligand.n_atoms for i in x.ligand.atoms)
    # A heme has a nitrogen called NA, which a contact table takes for a name that is not given: it is found by the
    # distance there. A line of several models has no distance, and stands on the atom all the same.
    base = diagram_of(CASE)
    st = changed(base.structure, lambda a: replace(a, name="NA") if a.residue == base.ligand_key and a.name == "N8" else a)
    dia = build(structure=st, records=[r.with_(ligand_atom="NA") if r.ligand_atom == "N8" else r for r in table()])
    both = Ensemble([dia, dia], member=0)
    gln = next(r for r in both.residues if r.residue_label == "Gln711")
    ((x,),) = [lines_of(both, gln, True)]
    assert [both.diagrams[0].ligand.names[i] for i in x.ligand.atoms] == ["NA"] and x.record.occupancy == 1.0


def test_a_line_goes_to_the_atoms_that_most_models_use_however_many_records_a_model_has():
    """Cys673 holds a nitrogen of the pyridine in two models, by one record each. The third has it on the carbon
    beside, by three records: three records are one model."""
    full = table(KIT)
    (hbond,) = [r for r in full if r.residue_number == 673]
    assert (hbond.type, hbond.protein_atom, hbond.ligand_atom) == (T.HBOND, "N", "N3")
    rest = [r for r in full if r.residue_number != 673]
    thrice = [hbond.with_(ligand_atom="C2", protein_atom=atom, distance=3.0) for atom in ("O", "CA", "C")]
    ens = Ensemble([build(records=rest + [hbond], case=KIT), build(records=rest + [hbond], case=KIT),
                    build(records=rest + thrice, case=KIT)])
    (cys,) = [r for r in ens.residues if r.residue_number == 673]
    assert cys.models == (0, 1, 2) and len(cys.places) == 1                # one part of the ligand, in all three
    (line,) = [r for r in ens.consensus().interactions if r.record.residue_number == 673]
    assert (line.record.ligand_atom, line.record.protein_atom) == ("N3", "N")
    assert line.record.occupancy == 1.0


def test_of_two_kinds_that_equally_many_models_make_a_coordination_comes_first():
    def made(*kinds):
        return ResidueShare("A", 94, "", "HIS", (0, 1), 2, (), {0: [
            _Made(k, (0,), SimpleNamespace(type=kind)) for k, kind in kinds]})
    assert made((0, T.HBOND), (1, T.HBOND), (0, T.METAL), (1, T.METAL)).kind_at(0) is T.METAL
    assert made((0, T.HBOND), (1, T.HBOND), (0, T.METAL)).kind_at(0) is T.HBOND
    assert made((0, T.METAL), (0, T.COVALENT)).kind_at(0) in (T.METAL, T.COVALENT)


@pytest.mark.parametrize("member, drawn, other", ((0, ("CYS", 2), ("Ala673", 1)), (1, ("ALA", 1), ("Cys673", 2))))
def test_a_residue_that_one_model_names_otherwise_is_counted_for_itself(member, drawn, other):
    """A point mutant among the models: cysteine 673 is an alanine in the second of three, and all three hold the
    ligand by it. Two residues are counted; the figure has one place for them, and gives it to the residue of the
    model whose ligand it draws, with that residue's own number of models."""
    model = structure_of(KIT)
    mutant = replace(model, atoms=[replace(a, resname="ALA") if (a.chain, a.resnum, a.hetero) == ("A", 673, False)
                                   else a for a in model.atoms])
    as_ala = [r.with_(residue_name="ALA") if r.residue_number == 673 else r for r in table(KIT)]
    ens = Ensemble([build(case=KIT), build(mutant, as_ala, case=KIT), build(case=KIT)], member=member)
    assert {r.residue_label: r.models for r in ens.residues if r.residue_number == 673} == {"Cys673": (0, 2), "Ala673": (1,)}
    dia = ens.consensus()
    res = dia.render()
    (node,) = [g for g in groups(res, "residue") if g.data["residue-number"] == "673"]
    name, models = drawn
    assert (node.data["residue"], node.data["n-models"]) == (f"A:{name}673", f"{models}/3")
    assert node.title.startswith(f"{name.capitalize()}673: contacted in {models} of 3 models")
    (line,) = [r for r in dia.interactions if r.record.residue_number == 673]
    assert (line.record.residue_name, line.record.occupancy) == (name, pytest.approx(models / 3))
    assert not [w for w in res.warnings if "no residue of this figure" in w]
    label, count = other
    (said,) = [t for t in dia.remarks if "another residue" in t]
    assert said == (f"Not drawn: {label} ({count} of 3), which other models have where model {member + 1} has "
                    f"another residue.")
    assert said in words(res) and any(label in w and "one place" in w for w in res.warnings)


def test_the_figure_says_which_residues_are_drawn_as_atoms_and_which_are_not():
    """A residue that enough models contact is drawn as atoms where a line of it has a direction. One whose contacts
    are hydrophobic stands in its ring alone, in every model or not: the sentence under the figure says so."""
    ens = Ensemble([build(case=KIT), build(case=KIT), build(case=KIT)])
    dia = ens.consensus()
    rings = [r.residue_label for r in ens.residues if {kind for kind, _ in r.kinds} == {T.HYDROPHOBIC}]
    # (In the table of kit_imatinib, the residues with hydrophobic contacts only are, in order, Leu595, Val603,
    # Leu644, Val654, Val668 and Leu799.)
    assert rings == ["Leu595", "Val603", "Leu644", "Val654", "Val668", "Leu799"]
    assert all(r.share == 1.0 for r in ens.residues)
    forms = {g.data["residue"].split(":")[-1]: g.data["form"] for g in groups(dia.render(), "residue")}
    assert all(forms[label.upper()] == "bubble" for label in rings) and forms["CYS673"] == "fragment"
    said = dia.remarks[0]
    assert ("A residue that half of them or more contact is drawn as atoms, unless its lines are hydrophobic contacts, "
            f"which have no direction ({len(rings)} here: Leu595; Val603; Leu644; and {len(rings) - 3} more); a line "
            "goes to the part of the ligand that most of them have it on.") in said
    # Where every such residue is drawn as atoms, the sentence has nothing to add.
    assert ("A residue that half of them or more contact is drawn as atoms; a line goes to the part of the ligand "
            "that most of them have it on.") in bioemu_samples().consensus().remarks[0]
    # The share from which a residue is drawn as atoms is a share, as ``least`` is.
    for bad in (2.0, -1.0):
        with pytest.raises(ValueError, match="atoms is a share of the models, from 0 to 1"):
            ens.consensus(atoms=bad)
    assert set(ens.consensus(atoms=1.0).forms.values()) == {"atoms", "bubbles"}
    assert set(ens.consensus(atoms=0.0).forms.values()) == {"atoms", "bubbles"}


def test_residues_that_few_models_contact_are_left_out_and_named():
    ens = four()
    dia = ens.consensus(least=0.5)
    left = [r for r in ens.residues if r.share < 0.5]
    assert [r.residue_label for r in left] == ["Gln711"]
    remark = next(t for t in dia.remarks if t.startswith("Not drawn: "))
    assert remark.startswith(f"Not drawn: {len(left)} residue") and "Gln711 (1 of 4)" in remark
    assert "GLN711" not in {g.data["residue"].split(":")[-1] for g in groups(dia.render(), "residue")}
    assert not any(t.startswith("Not drawn") for t in ens.consensus(least=0.0).remarks)
    many = ens.consensus(atoms=0.75)
    assert "A residue that 75 % of them or more contact is drawn as atoms" in many.remarks[0]
    assert next(form for name, form in many.forms.items() if name.endswith("MET742")) == "bubbles"
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="least is a share"):
            ens.consensus(least=bad)
    one = [r for r in table() if r.type.value == "halogen"]
    lone = Ensemble([build(records=t) for t in (one, [], [], [], [])])
    with pytest.raises(EnsembleError, match="no residue is contacted by 25 % of the models; the most frequent is"):
        lone.consensus()
    assert Ensemble([build(records=[]), build(records=[])]).consensus().render().status == "empty"


def test_a_figure_of_several_models_has_nothing_of_a_single_one():
    """The members are models with a pLDDT (made up for the test from the temperature factors): the figure of all
    of them has none, and what a single model's figure would draw from it is refused with a word."""
    with_plddt = invented_plddt(structure_of(CASE))
    ens = four(with_plddt)
    assert all(d.confidence is not None for d in ens.diagrams)
    dia = ens.consensus()
    assert dia.confidence is None and any("no single model" in w for w in dia.warnings)
    res = dia.render(solvent_sites=True, plddt_colors=True)
    assert any("solvent_sites was asked for, and this is a figure of several models" in w for w in res.warnings)
    assert any("plddt_colors was asked for" in w for w in res.warnings)
    assert "pLDDT" not in words(res)
    # What is asked for by hand is drawn: the lining, the halos, another form for one residue.
    lined = dia.render(lining="contacts", residues={"Met742": "bubbles", "*": "atoms"})
    forms = {g.data["residue"].split(":")[-1]: g.data["form"] for g in groups(lined, "residue")}
    assert forms["MET742"] == "bubble" and forms["ARG752"] == "fragment" and "Pocket lining" in words(lined)
    # A residue whose line is a hydrophobic contact, as atoms: its eyelash has the number beside the name.
    assert forms["PHE764"] == forms["LEU707"] == "eyelash"
    assert any(f.item.content == "Leu707 3/4" for f in walk(lined.scene) if isinstance(f.item, Text))


def test_its_graph_draws_it_again_and_says_how_many_models():
    dia = four().consensus()
    res = dia.render()
    node = next(n for n in res.graph.nodes if n.id.endswith("MET742"))
    assert (node.attrs["n_models"], node.attrs["n_total"]) == (2, 4)
    assert dia.render(graph=res.graph).svg == res.svg
    assert all("n_models" not in n.attrs for n in diagram_of(CASE).render().graph.nodes)


def test_it_answers_the_pointer():
    res = four().consensus().render(interactive=True)
    assert "lf-live" in res.svg and 'class="more' in res.svg
    assert res.svg.count("<title>Met742: contacted in 2 of 4 models (halogen bond 2)</title>") >= 1


def test_a_chain_is_refused():
    chain = chain_diagram("thrombin_hirudin")
    ens = Ensemble([chain, chain])
    with pytest.raises(EnsembleError, match="diagram\\(\\) draws one model"):
        ens.consensus()


# ---------------------------------------------------------------------------
# The numbers of the sets
# ---------------------------------------------------------------------------

def test_the_samples_of_adenylate_kinase():
    ens = bioemu_samples()
    got = {r.residue_label: r.n_models for r in ens.residues}
    assert len(got) == 34 and ens.n == 20
    assert [k for k, n in got.items() if n == 18] == ["Gly10", "Lys13", "Arg123", "Arg167"]
    # Two residues that most samples contact, and the drawn one does not: not in the figure of that sample.
    drawn = {r.record.residue_label for r in ens.diagram().interactions}
    assert {k for k, n in got.items() if n >= 10} - drawn == {"Gly12", "Lys200"}
    assert got["Gly12"] == 15 and got["Lys200"] == 13
    dia = ens.consensus()
    res = dia.render(bond_length=26.0)
    assert res.status == "ok" and res.quality.valid and res.quality.line_crossings == 0
    forms = [g.data["form"] for g in groups(res, "residue")]
    assert (forms.count("fragment"), forms.count("bubble")) == (12, 8)
    text = words(res)
    assert "Not drawn: 14 residues that fewer of the models contact (Gly32 (4 of 20); Asp84 (4 of 20);" in text
    assert "in 1 model they are named the other way round" in text


BUNDLE = TESTDATA / "ensemble" / "dhfr_mtx"


@fetched(*(f["file"].removeprefix("testdata/")
           for f in json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))["fetch"]))
def test_the_bundle_of_dihydrofolate_reductase():
    bundle = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    path = TESTDATA.parent / bundle["fetch"][0]["file"]
    ens = Ensemble.from_structures(read_models(path), bundle["ligand"], bundle["smiles"],
                                   [str(BUNDLE / m["table"]) for m in bundle["models"]], ensemble=bundle["ensemble"])
    got = {r.residue_label: r.n_models for r in ens.residues}
    assert got == {"Leu4": 21, "Asp26": 21, "Leu27": 21, "Arg57": 21, "Phe49": 19, "Thr116": 11, "His28": 9, "Arg31": 6,
                   "Phe30": 3, "Pro50": 3, "Trp5": 1, "Ala6": 1, "Glu56": 1}
    dia = ens.consensus()
    assert {r.record.residue_label: r.record.occupancy for r in dia.interactions
            if r.record.residue_label in ("Arg57", "Leu4")} == {"Arg57": 1.0, "Leu4": 1.0}
    assert dia.render().status == "ok"
    both = [r for r in ens.residues if r.residue_label in ("His28", "Arg31")]
    assert set(both[0].models) & set(both[1].models) == {16}                     # one model has both
    assert np.isclose(sum(r.share for r in both), 15 / 21)
