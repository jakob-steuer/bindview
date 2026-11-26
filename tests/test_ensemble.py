"""Several models of one complex: which contacts they make, in how many of them, and which model is drawn.

Two kinds of test, and they are kept apart.

The rules (what counts as one contact, which member is drawn, what is refused) are tested on inputs
*made for the test* from one committed structure, bicalutamide in the androgen receptor (``ar_bicalutamide``):
its table with rows taken out, its ligand moved by a known amount. Nothing is concluded from those about any
complex; they pin the arithmetic. Its table (``testdata/ar_bicalutamide/interactions.tsv``) has hydrogen bonds
to Leu704, Gln711 and Arg752, a halogen bond from F7B, a fluorine of the CF3, to Met742, a water bridge to Arg752
and six hydrophobic contacts.

The numbers are tested on real sets: twenty sampled conformations of adenylate kinase (committed), four deposited
crystal structures of MAO-B with its FAD and the 21 models of an NMR bundle. The contact tables of the last two
are committed; their structures are fetched, and the tests that read those are skipped where they are not on disk.
"""

from __future__ import annotations

import gzip
import json
import math
import shutil
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from conftest import ROOT, TESTDATA, bioemu_samples, diagram_of, fetched, meta_of, model_of, needs_plip, structure_of
from bindview import Diagram, Ensemble, EnsembleContact, EnsembleError, load_interactions
from bindview.confidence import occupancy_factors
from bindview.ensemble import CONTACT_COLUMNS, contact_key, symmetry_classes
from bindview.errors import LigandNotFoundError
from bindview.interactions import InteractionType
from bindview.scene import Text
from bindview.structure import Structure, read_models, read_structure
from scenetools import groups, inside, interactions, words

CASE = "ar_bicalutamide"                 # the committed structure the made-up inputs are cut from
SET = TESTDATA / "ensemble" / "maob_fad"
NMR = TESTDATA / "ensemble" / "dhfr_mtx"


def table():
    return load_interactions(TESTDATA / CASE / "interactions.tsv")


def named(records):
    """The records that name their ligand atom: they stay attached when the ligand is moved."""
    return [r for r in records if r.ligand_atom]


def build(structure=None, records=None) -> Diagram:
    """The committed case, with another structure or another table in its place."""
    m = meta_of(CASE)
    return Diagram.from_structure(structure if structure is not None else structure_of(CASE),
                                  m["ligand"]["residue_name_in_cif"], m["ligand"]["smiles"],
                                  table() if records is None else records)


def changed(st: Structure, move) -> Structure:
    """A structure with every atom put through ``move(atom) -> atom``."""
    return Structure(atoms=[move(a) for a in st.atoms], source=st.source, chem_comp_smiles=st.chem_comp_smiles,
                     plddt_scale=st.plddt_scale, qa_metrics=st.qa_metrics)


def ligand_shifted(dx: float) -> Structure:
    """The committed structure with its ligand moved ``dx`` angstrom along x: a pose that differs by exactly that."""
    base = diagram_of(CASE)
    return changed(base.structure, lambda a: replace(a, x=a.x + dx) if a.residue == base.ligand_key else a)


def by(ens: Ensemble) -> dict:
    return {(c.residue_label, c.type.value): c for c in ens.contacts}


def swapped(base: Diagram, one: str, other: str):
    """``move`` for :func:`changed`: two atoms of the ligand with each other's places, so each has the other's name."""
    a1, a2 = (base.structure.atom_by_name(base.ligand_key, n) for n in (one, other))

    def move(a):
        if a.residue != base.ligand_key or a.name not in (one, other):
            return a
        there = a2 if a.name == one else a1
        return replace(a, x=there.x, y=there.y, z=there.z)
    return move


# ---------------------------------------------------------------------------
# What counts, on tables made for the test
# ---------------------------------------------------------------------------

def test_occupancy_is_the_share_of_the_models_that_make_a_contact():
    full = table()
    no_hbond = [r for r in full if r.type.value != "hbond"]
    fewer = [r.with_(distance=3.8) if r.residue_number == 707 else r
             for r in no_hbond if r.type.value != "halogen"]
    ens = Ensemble([build(records=t) for t in (full, no_hbond, fewer, [])])     # an empty table is a model, too
    assert ens.n == 4 and ens.names == ["model 1", "model 2", "model 3", "model 4"]
    assert ens.ensemble == "the 4 models" and not ens.warnings
    got = by(ens)
    assert all(isinstance(c, EnsembleContact) and c.n_total == 4 for c in ens.contacts)
    assert got[("Gln711", "hbond")].occupancy == 0.25 and got[("Gln711", "hbond")].models == (0,)
    assert got[("Met742", "halogen")].occupancy == 0.5 and got[("Met742", "halogen")].models == (0, 1)
    assert got[("Arg752", "water_bridge")].occupancy == 0.75
    (row,) = [r for r in full if r.residue_number == 707]                       # Leu707, hydrophobic only
    leu = got[("Leu707", "hydrophobic")]
    assert leu.n_models == 3 and leu.distances == (row.distance, row.distance, 3.8)    # each model's own
    assert row.distance < 3.8
    assert (leu.distance_median, leu.distance_min, leu.distance_max) == (row.distance, row.distance, 3.8)
    # A contact is made through the atoms its record is resolved onto: the water bridge through the nitrile's N.
    base = diagram_of(CASE)
    bridge = next(r for r in base.interactions if r.type is InteractionType.WATER_BRIDGE)
    through = got[("Arg752", "water_bridge")].ligand_atoms
    assert through == tuple(base.ligand.names[i] for i in sorted(bridge.ligand.atoms)) == (bridge.record.ligand_atom,)
    keys = [(c.chain, c.residue_number, c.insertion_code) for c in ens.contacts]
    assert keys == sorted(keys)                                                  # a fixed order, not that of arrival
    assert ens.contacts[0].as_dict()["occupancy"] == ens.contacts[0].occupancy
    # The same pose four times: no model is nearer to the others than the first, and the first is drawn.
    assert ens.member == 0 and ens.chosen_by == "medoid" and ens.mean_rmsd == pytest.approx([0.0] * 4, abs=1e-9)
    assert all(c.in_member for c in ens.contacts)                                # it has the full table


@pytest.mark.parametrize("style", ("detailed", "schematic"))
def test_the_drawn_member_is_an_ordinary_diagram_with_the_shares_on_its_contacts(style):
    full = table()
    no_hbond = [r for r in full if r.type.value != "hbond"]
    ens = Ensemble([build(records=t) for t in (full, no_hbond, no_hbond, no_hbond)], ensemble="four made-up tables")
    dia = ens.diagram()
    assert isinstance(dia, Diagram) and dia.ensemble == "four made-up tables"
    assert dia.n_interactions == ens.diagrams[0].n_interactions
    share = {(r.record.residue_label, r.type.value): r.record.occupancy for r in dia.interactions}
    assert share[("Gln711", "hbond")] == 0.25 and share[("Met742", "halogen")] == 1.0
    assert all(r.record.occupancy is None for r in ens.diagrams[0].interactions)       # the member is untouched
    assert ens.diagrams[0].remarks == [] and ens.diagrams[0].ensemble is None
    res, plain = dia.render(style=style), ens.diagrams[0].render(style=style)
    assert res.status == plain.status == "ok" and res.quality.valid

    def strokes(result):
        (g,) = interactions(result, "hbond", "GLN711")
        return [f.item.paint for f in inside(result, g) if hasattr(f.item, "paint") and f.item.paint.stroke]
    width, opacity = occupancy_factors(0.25, steps=True)              # in steps, the default
    weak, was = strokes(res), strokes(plain)
    assert weak and all(p.width == pytest.approx(q.width * width) for p, q in zip(weak, was))
    assert all(p.opacity == pytest.approx(q.opacity * opacity) for p, q in zip(weak, was))
    (hbond,) = interactions(res, "hbond", "GLN711")
    assert hbond.data["occupancy"] == "0.250"

    # Occupancy qualifies the evidence and leaves the chemistry alone: same contacts on the same atoms.
    def claims(result):
        return sorted((g.data["residue"], g.data["type"], g.data["ligand-atoms"])
                      for g in groups(result, "interaction") + groups(result, "contact"))
    assert claims(res) == claims(plain)
    text = words(res)
    assert "Occupancy (share of four made-up tables, written on the line):" in text
    assert "Shown is model 1: of the 4 models, the one whose ligand lies nearest to the others'" in text
    assert "Distances are that model's." not in text            # the share stands where the distance would
    assert "Distances are that model's." in words(dia.render(style=style, distance_labels="all"))
    assert "No other model makes a contact that model 1 does not." in text
    assert "Shown is" not in words(plain) and "Occupancy" not in words(plain)


def test_a_contact_the_drawn_member_does_not_make_is_listed_and_named_but_not_drawn():
    full = table()
    no_hbond = [r for r in full if (r.residue_number, r.type.value) != (711, "hbond")]
    bare = [r for r in no_hbond if r.type.value != "halogen"]
    ens = Ensemble([build(records=t) for t in (full, no_hbond, bare, [])], names=["a", "b", "c", "d"], member="c")
    assert ens.member == 2 and ens.chosen_by == "given"
    missing = [(c.residue_label, c.type.value, c.n_models) for c in ens.contacts if not c.in_member]
    assert missing == [("Gln711", "hbond", 1), ("Met742", "halogen", 2)]
    dia = ens.diagram()
    assert dia.remarks == [
        "Shown is c of the 4 models, as asked. Distances are that model's.",
        "Not drawn: 2 contacts that other models make and c does not (Met742 halogen bond to F7B (2 of 4); "
        "Gln711 hydrogen bond to N8 (1 of 4))."]
    res = dia.render()
    assert res.status == "ok" and not interactions(res, "hbond", "GLN711") and not interactions(res, "halogen")
    assert "Not drawn: 2 contacts that other models make and c does not" in words(res)
    # More than three are counted, and the three that most models make are named: here the hydrophobic contacts,
    # each of which one more model makes.
    one = [[r] for r in full if r.type.value == "hydrophobic"]
    many = Ensemble([build(records=t) for t in ([], full, full, *one)], member=0)
    remark = many.diagram().remarks[1]
    assert remark.startswith(f"Not drawn: {len(many.contacts)} contacts that other models make and model 1 does not (")
    assert many.n == 3 + len(one) and len(many.contacts) == len({(r.residue_number, r.type) for r in full})
    assert remark.count(f" of {many.n})") == 3 and remark.endswith(f"; and {len(many.contacts) - 3} more).")
    assert remark.count(f"(3 of {many.n})") == 3                                 # the ones most models make
    # A member that makes no contact is an empty figure, and that figure says what it stands for.
    empty = many.diagram().render()
    assert empty.status == "empty" and "Not drawn:" in words(empty) and "Shown is model 1" in words(empty)


@pytest.mark.parametrize("style", ("detailed", "schematic"))
def test_a_contact_of_other_models_can_be_drawn_faint_with_its_share(style):
    base = diagram_of(CASE)
    full = named(table())
    no_halogen = [r for r in full if r.type.value != "halogen"]
    # A model that gives two fluorines of the CF3 each other's names: its F7C lies where the others' F7B lies,
    # and that is the fluorine its halogen bond is made through.
    through_f7c = [r.with_(ligand_atom="F7C") if r.type.value == "halogen" else r for r in full]
    ens = Ensemble([build(base.structure, no_halogen),
                    build(changed(base.structure, swapped(base, "F7B", "F7C")), through_f7c),
                    build(base.structure, full)], member=0)
    (halogen,) = [c for c in ens.contacts if not c.in_member]
    assert halogen.type is InteractionType.HALOGEN and halogen.models == (1, 2)
    assert halogen.ligand_atoms == ("F7B", "F7C")
    plain, faint = ens.diagram(), ens.diagram(absent="faint")
    assert plain.elsewhere == [] and plain.remarks[1].startswith("Not drawn: 1 contact that other models make")
    # It is put on the atom of the drawn model that lies where the other model's atom lies, not on its name.
    (there,) = faint.elsewhere
    assert there.absent and there.ligand is not None and there.record.ligand_atom == "F7B"
    assert there.record.occupancy == pytest.approx(2 / 3) and there.record.distance is None
    assert (there.record.residue_label, there.record.protein_atom) == ("Met742", "O")
    # It is not a contact of the diagram: not counted, not in its fingerprint.
    assert faint.n_interactions == plain.n_interactions == len(no_halogen)
    assert all(bit.type is not InteractionType.HALOGEN for bit in faint.fingerprint().bits)
    assert not any(r.absent for r in faint.interactions)
    res, was = faint.render(style=style), plain.render(style=style)
    assert res.status == was.status == "ok" and res.quality.valid
    assert not interactions(was, "halogen")
    (line,) = interactions(res, "halogen", "MET742")
    assert line.data["absent"] == "true" and line.data["ligand-atoms"] == "F7B" and line.data["occupancy"] == "0.667"
    assert "distance" not in line.data and "made by other models and not by the one shown" in line.title
    assert [f.item.content for f in inside(res, line, Text)] == ["67 %"]         # its share, where a distance would stand
    strokes = [f.item.paint for f in inside(res, line) if hasattr(f.item, "paint") and f.item.paint.stroke]
    assert strokes and all(p.opacity <= 0.45 for p in strokes)
    own = [g for g in groups(res, "interaction") if "absent" not in g.data]
    assert own and all(g.data["residue"] != "A:MET742" for g in own)
    assert res.graph.meta["n_interactions"] == len(no_halogen)
    assert faint.render(style=style, graph=res.graph).svg == res.svg             # and it is redrawn from its graph
    text = words(res)
    assert "Faint, with a share: made by other models, not by the one shown" in text
    assert ("Faint, with the share of the 3 models that make it: 1 contact that other models make and model 1 "
            "does not.") in text
    assert "Not drawn" not in text and "Faint" not in words(was)
    # A model that makes no contact of its own stays an empty figure, and says what it does not draw.
    alone = Ensemble([build(base.structure, []), build(base.structure, full)], member=0)
    nothing = alone.diagram(absent="faint").render(style=style)
    assert nothing.status == "empty" and "Not drawn:" in words(nothing)
    assert len(alone.diagram(absent="faint").elsewhere) == len(alone.contacts) == len(full)   # one contact a row here
    with pytest.raises(ValueError, match="absent is one of 'named', 'faint'"):
        ens.diagram(absent="dotted")


def test_atoms_that_the_symmetry_of_the_ligand_exchanges_are_one_atom():
    lig = diagram_of(CASE).ligand
    classes = symmetry_classes(lig)
    assert classes["F7A"] == classes["F7B"] == classes["F7C"]                   # the CF3
    assert classes["C16"] == classes["C20"] and classes["C17"] == classes["C19"]    # the two sides of the fluorophenyl
    assert classes["O14"] == classes["O15"]                                     # the two oxygens of the sulfone
    assert classes["C16"] != classes["C17"] and classes["N8"] != classes["N9"]
    assert len(set(classes.values())) < len(classes)
    # The halogen bond through another fluorine in each model is one contact that every model makes ...
    tables = [[r.with_(ligand_atom=f) if r.type.value == "halogen" else r for r in table()]
              for f in ("F7A", "F7B", "F7C")]
    ens = Ensemble([build(records=t) for t in tables])
    (halogen,) = [c for c in ens.contacts if c.type is InteractionType.HALOGEN]
    assert halogen.occupancy == 1.0 and halogen.in_member
    assert halogen.ligand_atoms == tuple(n for n in lig.names if n.startswith("F7"))   # all three, in its order
    # ... and without that rule it would be three that one model each makes.
    keys = {contact_key(r, d.ligand, classes, symmetry=False) for d in ens.diagrams for r in d.interactions
            if r.type is InteractionType.HALOGEN}
    assert len(keys) == 3
    # An acid's two oxygens are one atom although the molecule is written with one of them charged.
    acid = diagram_of("dhfr_methotrexate").ligand
    ends = symmetry_classes(acid)
    assert ends["O1"] == ends["O2"] and ends["OE1"] == ends["OE2"] and ends["O1"] != ends["OE1"]
    charges = {acid.names[a.GetIdx()]: a.GetFormalCharge() for a in acid.mol.GetAtoms()}
    assert {charges["O1"], charges["O2"]} == {0, -1}                             # as it is drawn: not alike


def test_a_hydrophobic_contact_is_matched_on_its_residue_and_the_protein_atom_is_no_part_of_a_contact():
    full = table()
    other_carbon = [r.with_(ligand_atom="C4", protein_atom="CD1") if r.residue_number == 707 else r for r in full]
    other_end = [r.with_(protein_atom="OE1", ligand_is_donor=True) if (r.residue_number, r.type.value) == (711, "hbond")
                 else r for r in full]
    ens = Ensemble([build(records=t) for t in (full, other_carbon, other_end)])
    got = by(ens)
    leu = got[("Leu707", "hydrophobic")]
    names = ens.diagrams[0].ligand.names
    assert leu.occupancy == 1.0                                                  # every carbon a model pairs with it
    assert leu.ligand_atoms == tuple(n for n in names if n in ("C4", "C5"))
    assert got[("Gln711", "hbond")].occupancy == 1.0                             # NE2 or OE1, donor or acceptor
    assert len(ens.contacts) == len({(r.residue_number, r.type) for r in full})
    classes = symmetry_classes(ens.diagrams[0].ligand)
    apart = {contact_key(r, d.ligand, classes, hydrophobic_by_residue=False) for d in ens.diagrams
             for r in d.interactions if r.record.residue_number == 707}
    assert len(apart) == 2                                                       # by atom it would be two contacts


# ---------------------------------------------------------------------------
# Which member is drawn, on poses made for the test
# ---------------------------------------------------------------------------

def test_the_medoid_is_drawn_whatever_the_order_of_the_models():
    rows = named(table())
    shifts = {"near": 0.0, "middle": 0.5, "far": 2.0}
    models = {name: build(ligand_shifted(dx), rows) for name, dx in shifts.items()}
    ens = Ensemble(list(models.values()), list(models))
    assert ens.rmsd == pytest.approx(np.array([[0.0, 0.5, 2.0], [0.5, 0.0, 1.5], [2.0, 1.5, 0.0]]), abs=1e-6)
    assert ens.pocket_rmsd == pytest.approx(np.zeros((3, 3)), abs=1e-6)          # one protein, three times
    assert ens.mean_rmsd == pytest.approx([1.25, 1.0, 1.75], abs=1e-6)
    assert ens.names[ens.member] == "middle" and ens.chosen_by == "medoid"
    for order in (("far", "near", "middle"), ("middle", "far", "near")):
        again = Ensemble([models[n] for n in order], list(order))
        assert again.names[again.member] == "middle"
    assert "Shown is middle: of the 3 models, the one whose ligand lies nearest to the others' (RMSD 1.00 Å" \
        in ens.diagram().remarks[0]
    # Equal means: the model that was given first. Two models have no medoid at all.
    tie = Ensemble([models[n] for n in ("far", "near", "far", "near")])
    assert tie.member == 0 and tie.chosen_by == "medoid"
    two = Ensemble([models["far"], models["near"]])
    assert two.member == 0 and two.chosen_by == "first"
    assert two.diagram().remarks[0] == "Shown is model 1, the first of the 2 models. Distances are that model's."
    # The caller may name the member; the distances are reported all the same.
    for asked in (2, "far", -1):
        given = Ensemble(list(models.values()), list(models), member=asked)
        assert given.names[given.member] == "far" and given.chosen_by == "given"
        assert given.mean_rmsd == pytest.approx([1.25, 1.0, 1.75], abs=1e-6)
    with pytest.raises(ValueError, match="no model is named 'nearest'"):
        Ensemble(list(models.values()), list(models), member="nearest")
    with pytest.raises(ValueError, match="member 3 is not one of the 3 models"):
        Ensemble(list(models.values()), list(models), member=3)


def test_a_pose_is_compared_in_the_pocket_and_up_to_the_symmetry_of_the_ligand():
    base = diagram_of(CASE)
    rows = named(table())
    # The whole complex turned and moved is the same pose: models are superposed on their pocket first.
    angle = math.radians(40.0)
    rot = np.array([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])

    def turned(a):
        x, y, z = rot @ np.array(a.xyz) + np.array([10.0, -5.0, 3.0])
        return replace(a, x=float(x), y=float(y), z=float(z))
    # Two fluorines of the CF3 with each other's names: the same pose again, under the symmetry of the ligand.
    ens = Ensemble([build(base.structure, rows), build(changed(base.structure, turned), rows),
                    build(changed(base.structure, swapped(base, "F7B", "F7C")), rows)])
    assert ens.rmsd == pytest.approx(np.zeros((3, 3)), abs=1e-6)
    assert ens.pocket_rmsd == pytest.approx(np.zeros((3, 3)), abs=1e-6)
    f7b, f7c = (base.structure.atom_by_name(base.ligand_key, n) for n in ("F7B", "F7C"))
    by_name = math.sqrt(2.0 * sum((p - q) ** 2 for p, q in zip(f7b.xyz, f7c.xyz)) / base.ligand.n_atoms)
    assert by_name > 0.3                                                         # what the two names would have cost


def test_a_model_whose_pocket_is_folded_otherwise_counts_and_is_no_candidate():
    base = diagram_of(CASE)
    rows = named(table())
    near = {k for k in base.pocket_residues(12.0)}

    def scattered(a):
        """Every second residue around the ligand eight angstrom away from where it was."""
        if a.residue in near and a.resnum % 2:
            return replace(a, x=a.x + 8.0)
        return a
    ens = Ensemble([build(base.structure, rows), build(ligand_shifted(1.0), rows),
                    build(changed(base.structure, scattered), rows)], ["a", "b", "broken"])
    assert math.isnan(ens.rmsd[0, 2]) and math.isnan(ens.rmsd[1, 2]) and ens.rmsd[0, 1] == pytest.approx(1.0)
    assert ens.pocket_rmsd[0, 2] > 3.0                                           # the fit was made, and is too poor
    assert ens.mean_rmsd[2] is None and ens.member == 0 and ens.chosen_by == "medoid"
    assert any(w.startswith("broken: the pocket does not superpose") and "Counted for occupancy" in w
               for w in ens.warnings)
    assert all(c.n_total == 3 and c.occupancy == 1.0 for c in ens.contacts)      # counted all the same


# ---------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------

def test_the_figure_of_the_member_says_what_the_ensemble_knows_of_its_choice():
    """Twenty samples of which eight are candidates for the member (the pockets of the others do not superpose on
    enough of the rest): the figure of the member carries the ensemble's warnings, and does not claim all twenty."""
    ens = bioemu_samples()
    candidates = sum(mean is not None for mean in ens.mean_rmsd)
    over = sum(not math.isnan(ens.rmsd[ens.member, k]) for k in range(ens.n)) - 1
    assert (ens.n, candidates, over, ens.chosen_by) == (20, 8, 14, "medoid")
    (apart,) = [w for w in ens.warnings if "not a candidate for the member that is drawn" in w]
    dia = ens.diagram()
    assert apart in dia.warnings and apart in dia.render().warnings
    assert set(ens.diagrams[ens.member].warnings) <= set(dia.warnings)             # with the model's own
    which = dia.remarks[0]
    assert which.startswith(f"Shown is {ens.names[ens.member]}: of the 8 of the 20 models whose pocket superposes on "
                            "that of at least half of the others, the one whose ligand lies nearest to the others' ")
    assert f"(RMSD {ens.mean_rmsd[ens.member]:.2f} Å on average, over the 14 models whose pocket superposes on its own)" in which
    assert "of the 20 models, the one" not in which
    assert which.removesuffix(" Distances are that model's.") in words(dia.render())   # (it has the shares in their place)
    # A record that is not counted is said by the figure as well.
    loose = table()[0].with_(ligand_atom="X99")
    made = Ensemble([build(), build(records=table() + [loose]), build()])
    (told,) = [w for w in made.warnings if "could not be attached to the ligand" in w]
    assert told.startswith("model 2: 1 record ") and told in made.diagram().warnings
    assert made.diagram().remarks[0].startswith("Shown is model 1: of the 3 models, the one whose ligand lies nearest")


def test_what_is_not_several_models_of_one_complex_is_refused():
    one = diagram_of(CASE)
    with pytest.raises(ValueError, match="at least two models"):
        Ensemble([one])
    with pytest.raises(ValueError, match="one name per model"):
        Ensemble([one, one], ["only one"])
    # Another ligand: nothing could be matched.
    with pytest.raises(EnsembleError, match="model 2 does not hold the ligand of model 1.*named otherwise"):
        Ensemble([one, diagram_of("ar_r1881")])
    # The same ligand in a protein that is numbered otherwise: its residues are not these residues.
    moved = changed(one.structure, lambda a: a if a.residue == one.ligand_key else replace(a, resnum=a.resnum + 1000))
    elsewhere = build(moved, [r.with_(residue_number=r.residue_number + 1000) for r in table()])
    with pytest.raises(EnsembleError, match="model 2 and model 1 have no pocket in common.*name the chains alike"):
        Ensemble([one, elsewhere])
    # ... and so are chains that are named otherwise, which is what two of the MAO-B entries do.
    other_chain = changed(one.structure, lambda a: a if a.residue == one.ligand_key else replace(a, chain="AAA"))
    with pytest.raises(EnsembleError, match="no pocket in common"):
        Ensemble([one, build(other_chain, [r.with_(chain="AAA") for r in table()])])


def test_models_from_files_one_table_each(tmp_path):
    m = meta_of(CASE)
    ligand, smiles = m["ligand"]["residue_name_in_cif"], m["ligand"]["smiles"]
    model, rows = tmp_path / "model.cif", TESTDATA / CASE / "interactions.tsv"
    model.write_bytes(gzip.decompress(model_of(CASE).read_bytes()))
    for name in ("a", "b", "c"):
        shutil.copy(model, tmp_path / f"{name}.cif")
    files = [tmp_path / f"{name}.cif" for name in ("a", "b", "c")]
    # A table may be a path, rows, or empty; a ligand one selection for all, or a list with one a model.
    ens = Ensemble.from_structures(files, ligand, smiles, [rows, table(), []], name="bicalutamide")
    assert ens.names == ["a", "b", "c"] and ens.n == 3                           # named after what tells them apart
    assert all(d.name == "bicalutamide" for d in ens.diagrams)                   # options go to every model
    assert {c.n_models for c in ens.contacts} == {2} and ens.diagrams[2].n_interactions == 0
    key = diagram_of(CASE).ligand_key
    listed = Ensemble.from_structures(files, [ligand, f"{key.chain}:{key.number}", (key.chain, key.number)], smiles,
                                      [rows, rows, rows])
    assert {c.occupancy for c in listed.contacts} == {1.0}
    same = Ensemble.from_structures([model, model], ligand, smiles, [rows, rows], names=None)
    assert same.names == ["model 1", "model 2"]                                  # the files do not tell them apart
    assert Ensemble.from_structures([model, model], ligand, smiles, [rows, rows], names=["x", "x"]).names == ["x", "x (2)"]
    with pytest.raises(ValueError, match="at least two models"):
        Ensemble.from_structures([model], ligand, smiles, [rows])
    with pytest.raises(ValueError, match="one contact table for each of the 2 models"):
        Ensemble.from_structures([model, model], ligand, smiles, [rows])
    with pytest.raises(ValueError, match="a list of tables, not the rows of one"):
        Ensemble.from_structures([model, model], ligand, smiles, table()[:2])
    with pytest.raises(ValueError, match="3 ligand selections for 2 models"):
        Ensemble.from_structures([model, model], [ligand] * 3, smiles, [rows, rows])
    with pytest.raises(LigandNotFoundError, match="^b: no residue named 'NOPE'"):   # which model, then why
        Ensemble.from_structures(files[:2], [ligand, "NOPE"], smiles, [rows, rows])


@pytest.mark.parametrize("models", (2, 5, 6))
def test_one_path_for_several_models_is_no_list_of_tables(models, tmp_path):
    """A path is a string, and a string is a sequence: of five characters for ``a.tsv``, which five models took for
    five tables called ``a``, ``.``, ``t``, ``s`` and ``v``."""
    from pathlib import Path

    path = tmp_path / "a.tsv"
    for one in ("a.tsv", str(path), Path("a.tsv")):
        with pytest.raises(ValueError, match=f"one contact table for each of the {models} models") as refused:
            Ensemble.from_structures([structure_of(CASE)] * models, meta_of(CASE)["ligand"]["residue_name_in_cif"],
                                     meta_of(CASE)["ligand"]["smiles"], interactions=one)
        assert repr(str(one)) in str(refused.value) and "one path" in str(refused.value)


def test_the_models_of_one_file_are_an_ensemble(tmp_path):
    """A file with several models, made for the test: the committed complex three times, its ligand
    moved by a known amount in the second and the third."""
    from bindview.detect import structure_to_pdb          # writes a structure as PDB text; loads no detector

    base = diagram_of(CASE)
    rows = named(table())

    def atoms(st):
        text, _, _ = structure_to_pdb(st, base.ligand_key)
        return "\n".join(ln for ln in text.splitlines() if ln[:6] in ("ATOM  ", "HETATM"))
    text = "".join(f"MODEL     {k:4d}\n{atoms(ligand_shifted(dx))}\nENDMDL\n"
                   for k, dx in ((1, 0.0), (2, 0.5), (3, 2.0))) + "END\n"
    (tmp_path / "bundle.pdb").write_text(text)
    models = read_models(tmp_path / "bundle.pdb")
    assert [m.model for m in models] == [1, 2, 3]
    smiles = meta_of(CASE)["ligand"]["smiles"]
    ligand = (base.ligand_key.chain, base.ligand_key.number)
    ens = Ensemble.from_structures(models, ligand, smiles, [rows, rows, rows])
    assert ens.names == ["model 1", "model 2", "model 3"]                        # after their numbers in the file
    assert ens.rmsd == pytest.approx(np.array([[0.0, 0.5, 2.0], [0.5, 0.0, 1.5], [2.0, 1.5, 0.0]]), abs=2e-3)
    assert ens.names[ens.member] == "model 2" and ens.chosen_by == "medoid"
    assert {c.occupancy for c in ens.contacts} == {1.0} and len(ens.contacts) == len({(r.residue_number, r.type) for r in rows})
    res = ens.diagram().render()
    assert res.status == "ok" and "Shown is model 2: of the 3 models" in words(res)
    # Two of them, in another order, keep the names the file gives them.
    assert Ensemble.from_structures([models[2], models[0]], ligand, smiles, [rows, rows]).names == ["model 3", "model 1"]


def test_the_numbers_behind_the_figure():
    full = table()
    polar = [r for r in full if r.type.value in ("halogen", "hbond")]
    (halogen,) = [r for r in full if r.type.value == "halogen"]
    tables = (full, polar, [halogen])
    ens = Ensemble([build(records=t) for t in tables], ["s0", "s1", "s2"])
    text = ens.to_csv()
    lines = text.splitlines()
    assert lines[0].split(",") == list(CONTACT_COLUMNS) and len(lines) == 1 + len(ens.contacts)
    row = dict(zip(CONTACT_COLUMNS, next(ln for ln in lines if ",halogen," in ln).split(",")))
    assert (row["residue_name"], row["residue_number"], row["ligand_atoms"]) == ("MET", "742", "F7B")
    assert (row["n_models"], row["n_total"], row["occupancy"], row["in_member"]) == ("3", "3", "1.0000", "true")
    assert row["models"] == "s0;s1;s2" and row["distance_median"] == f"{halogen.distance:.3f}"
    # The models as fingerprints, one row a model: the residue-level view, as a comparison has it for ligands.
    assert ens.table.names == ["s0", "s1", "s2"]
    assert [len(fp) for fp in ens.table.fingerprints] == [len({(r.residue_number, r.type) for r in t}) for t in tables]
    assert {r.label: r.n_ligands for r in ens.table.consensus()}["Met742"] == 3


# ---------------------------------------------------------------------------
# A set of crystal structures: four of MAO-B with its FAD
# ---------------------------------------------------------------------------

def manifest(folder) -> dict:
    """The ``manifest.json`` of a set: its models, the file of each, its ligand and its table."""
    return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def models_in(path: str) -> dict:
    return {st.model: st for st in read_models(path)}


def entry(model: dict) -> Structure:
    """One model of a set as its manifest names it: the model of its file, with the chains under the names the
    set gives them (two MAO-B entries call their chains ``AAA`` and ``BBB``, the other two ``A`` and ``B``)."""
    path = ROOT / model["file"]
    st = models_in(str(path))[model["model"]] if model.get("model") is not None else read_structure(path)
    names = model.get("chains") or {}
    return changed(st, lambda a: replace(a, chain=names.get(a.chain, a.chain))) if names else st


#: The four structures of MAO-B are fetched, not kept (``testdata/waters``).
MAOB = fetched(*(m["file"].removeprefix("testdata/") for m in manifest(SET)["models"]))
#: The NMR bundle 1AO8 is fetched, not kept.
BUNDLE = fetched(*(f["file"].removeprefix("testdata/") for f in manifest(NMR)["fetch"]))
_maob: "list[Ensemble]" = []


def maob() -> Ensemble:
    if not _maob:
        spec = manifest(SET)
        models = spec["models"]
        _maob.append(Ensemble.from_structures(
            [entry(m) for m in models], [m["ligand"] for m in models], spec["smiles"],
            [SET / m["table"] for m in models], names=[m["id"] for m in models], ensemble=spec["ensemble"],
            name=spec["name"]))
    return _maob[0]


@MAOB
def test_four_crystal_structures_of_one_cofactor():
    ens = maob()
    assert ens.n == 4 and ens.names == ["2C65", "4CRT", "7ZW3", "9FJT"] and not ens.warnings
    assert all(d.ligand.n_atoms == 53 and d.ligand.bond_source == "template" and not d.unresolved
               for d in ens.diagrams)
    assert all(d.confidence is None for d in ens.diagrams)                       # crystal structures: no pLDDT
    # One pocket and one pose, four times: everything within a tenth of an angstrom or so.
    off = ens.rmsd[~np.eye(4, dtype=bool)]
    assert 0.05 < off.min() and off.max() < 0.12 and ens.pocket_rmsd.max() < 0.12
    assert ens.names[ens.member] == "4CRT" and ens.chosen_by == "medoid"
    assert ens.mean_rmsd == pytest.approx([0.098, 0.086, 0.096, 0.096], abs=1e-3)
    counts = {n: sum(c.n_models == n for c in ens.contacts) for n in (4, 3, 2, 1)}
    assert len(ens.contacts) == 36 and counts == {4: 27, 3: 4, 2: 2, 1: 3}
    assert [(c.residue_label, c.type.value, c.ligand_atoms, c.n_models) for c in ens.contacts if not c.in_member] == [
        ("Arg38", "water_bridge", ("O3B",), 3), ("Arg42", "water_bridge", ("O2P",), 1),
        ("Lys296", "water_bridge", ("O4",), 1)]
    tyr = next(c for c in ens.contacts if c.type is InteractionType.PI_STACKING)
    assert tyr.residue_label == "Tyr393" and tyr.occupancy == 1.0
    assert tyr.distances == (5.08, 5.15, 5.17, 5.16)                             # one a model, as its table has it
    assert (tyr.distance_min, tyr.distance_median, tyr.distance_max) == (5.08, pytest.approx(5.155), 5.17)


@MAOB
def test_two_names_that_one_entry_gives_the_other_way_round_are_one_atom():
    """2C65 calls the two free oxygens of one phosphate by each other's names. Taken by name that is a
    pose half an angstrom away and three contacts that only it makes; it is neither."""
    ens = maob()
    classes = symmetry_classes(ens.diagrams[0].ligand)
    shared = sorted(sorted(n for n, c in classes.items() if c == k)
                    for k in set(classes.values()) if list(classes.values()).count(k) > 1)
    assert shared == [["O1A", "O2A"], ["O1P", "O2P"]]                            # the free oxygens of the two phosphates
    through_either = [(c.residue_label, c.type.value, c.n_models) for c in ens.contacts
                      if c.ligand_atoms == ("O1A", "O2A")]
    assert through_either == [("Ile14", "water_bridge", 4), ("Gly40", "hbond", 4), ("Arg42", "hbond", 4)]

    def n_keys(**rules):
        return len({contact_key(r, d.ligand, classes, **rules) for d in ens.diagrams for r in d.interactions})
    assert n_keys() == len(ens.contacts) == 36
    assert n_keys(symmetry=False) == 39 and n_keys(hydrophobic_by_residue=False) == 37
    assert n_keys(symmetry=False, hydrophobic_by_residue=False) == 40
    # By name, 2C65 lies half an angstrom from 4CRT; in the pocket it lies a tenth.
    a, b = ens.diagrams[0].ligand, ens.diagrams[1].ligand
    here = {n: a.coords3d[i] for i, n in enumerate(a.names)}
    there = {n: b.coords3d[i] for i, n in enumerate(b.names)}
    assert np.linalg.norm(here["O1A"] - here["O2A"]) > 2.4 and ens.rmsd[0, 1] < 0.1
    assert set(here) == set(there)


@MAOB
@pytest.mark.parametrize("style", ("detailed", "schematic"))
def test_the_figure_of_the_set(style):
    ens = maob()
    dia = ens.diagram()
    res = dia.render(style=style)
    assert res.status == "ok" and res.quality.valid
    text = words(res)
    assert "Occupancy (share of four crystal structures of MAO-B, written on the line):" in text
    assert ("Shown is 4CRT: of the 4 models, the one whose ligand lies nearest to the others' (RMSD 0.09 Å on "
            "average, pockets superposed).") in text
    assert "Distances are that model's." not in text
    assert ("Not drawn: 3 contacts that other models make and 4CRT does not (Arg38 water bridge to O3B (3 of 4); "
            "Arg42 water bridge to O2P (1 of 4); Lys296 water bridge to O4 (1 of 4)).") in text
    (half,) = interactions(res, "hbond", "GLY434")
    assert half.data["occupancy"] == "0.500"
    (full,) = interactions(res, "pi_stacking", "TYR393")                         # made in all four
    assert full.data["occupancy"] == "1.000"
    assert not [g for g in groups(res, "interaction") if g.data["residue"] in ("A:ARG38", "A:LYS296")]
    assert res.graph.ensemble == "four crystal structures of MAO-B"


@MAOB
@pytest.mark.parametrize("style", ("detailed", "schematic"))
def test_the_figure_of_the_set_with_the_other_models_contacts_drawn_faint(style):
    ens = maob()
    dia = ens.diagram(absent="faint")
    assert [(r.record.residue_label, r.record.protein_atom, r.record.ligand_atom, r.record.occupancy)
            for r in dia.elsewhere] == [("Arg38", "O", "O3B", 0.75), ("Arg42", "NH2", "O2P", 0.25),
                                        ("Lys296", "NZ", "O4", 0.25)]
    assert dia.n_interactions == ens.diagram().n_interactions == 34
    res = dia.render(style=style)
    assert res.status == "ok" and res.quality.valid
    drawn = [g for g in groups(res, "interaction") if g.data.get("absent") == "true"]
    assert sorted(g.data["residue"] for g in drawn) == ["A:ARG38", "A:ARG42", "A:LYS296"]
    assert sorted(f.item.content for g in drawn for f in inside(res, g, Text)) == ["H₂O 25 %", "H₂O 25 %",
                                                                                 "H₂O 75 %"]
    text = words(res)
    assert ("Faint, with the share of the 4 models that make it: 3 contacts that other models make and 4CRT does "
            "not.") in text and "Not drawn" not in text
    assert "Shown is 4CRT: of the 4 models" in text
    assert res.graph.meta["n_interactions"] == 34
    assert dia.render(style=style, graph=res.graph).svg == res.svg


# ---------------------------------------------------------------------------
# The models of one file: an NMR bundle (1AO8, dihydrofolate reductase with methotrexate)
# ---------------------------------------------------------------------------

#: Methotrexate's atoms by part, as the archive names them: the diaminopteridine, and the glutamate at the other end.
PTERIDINE = {"N1", "C2", "NA2", "N3", "C4", "NA4", "C4A", "N5", "C6", "C7", "N8", "C8A"}
GLUTAMATE = {"N", "CA", "CT", "O1", "O2", "CB", "CG", "CD", "OE1", "OE2"}
POLAR = (InteractionType.HBOND, InteractionType.SALTBRIDGE)
_nmr: "dict[str, object]" = {}


def nmr_models():
    spec = manifest(NMR)
    if "models" not in _nmr:
        _nmr["models"] = read_models(ROOT / spec["fetch"][0]["file"])
    return spec, _nmr["models"]


def nmr() -> Ensemble:
    spec, models = nmr_models()
    if "ensemble" not in _nmr:
        _nmr["ensemble"] = Ensemble.from_structures(models, spec["ligand"], spec["smiles"],
                                                    [NMR / m["table"] for m in spec["models"]],
                                                    ensemble=spec["ensemble"], name=spec["name"])
    return _nmr["ensemble"]


@BUNDLE
def test_the_models_of_the_bundle_are_read_one_by_one():
    """The bundle is what the archive's record of the entry says: 21 models, each with one chain of 162 residues and
    the ligand once. The first model is what a call without ``model=`` reads, and no two models are one."""
    spec, models = nmr_models()
    assert len(models) == len(spec["models"]) == 21 and [m.model for m in models] == list(range(1, 22))
    assert [m["model"] for m in spec["models"]] == [m.model for m in models]
    for st in models:
        assert len({a.chain for a in st.atoms if not a.hetero}) == 1
        assert len({a.resnum for a in st.atoms if not a.hetero and a.name == "CA"}) == 162
        assert len({a.residue for a in st.atoms if a.resname == "MTX"}) == 1          # the ligand, once in each
    first = read_structure(ROOT / spec["fetch"][0]["file"])
    assert [(a.name, a.xyz) for a in first.atoms] == [(a.name, a.xyz) for a in models[0].atoms]
    assert len({tuple(a.xyz for a in st.atoms[:50]) for st in models}) == 21


@BUNDLE
def test_how_far_the_models_of_the_bundle_lie_apart_and_which_polar_contacts_hold():
    """The 21 models lie further apart than four crystals of one complex do, and fewer than half of the contacts
    are made by every model. The polar contacts that every model makes include the diaminopteridine's salt bridge
    to Asp26; the alpha-carboxylate is held by Arg57 in every model, and the loose end is the other carboxylate."""
    ens = nmr()
    assert ens.n == 21 and ens.names == [f"model {k}" for k in range(1, 22)]
    off = ens.rmsd[~np.eye(21, dtype=bool)]
    assert 0.1 < off.min() and ens.mean_rmsd[ens.member] < 1.0
    spread = (float(off.min()), float(np.median(off)), float(off.max()))
    assert tuple(round(x, 2) for x in spread) == (0.44, 1.13, 1.82)
    assert sum(c.n_models == 21 for c in ens.contacts) < len(ens.contacts) / 2
    polar = [c for c in ens.contacts if c.type in POLAR]
    most = max(c.n_models for c in polar)
    assert any(c.residue_label == "Asp26" and set(c.ligand_atoms) & PTERIDINE for c in polar if c.n_models == most)
    assert sorted((c.residue_label, c.type.value) for c in polar if c.n_models == most) == [
        ("Arg57", "saltbridge"), ("Asp26", "saltbridge"), ("Leu4", "hbond")]
    # Of the polar contacts that fewer than a quarter of the models make, half are on the glutamate tail.
    rare = [c for c in polar if c.n_models < 21 / 4]
    assert (len(rare), sum(bool(set(c.ligand_atoms) & GLUTAMATE) for c in rare)) == (6, 3)
    held = next(c for c in polar if c.residue_label == "Arg57" and c.type is InteractionType.SALTBRIDGE)
    assert held.ligand_atoms == ("O1", "O2") and held.n_models == 21
    loose = [(c.residue_label, c.n_models) for c in polar if set(c.ligand_atoms) == {"OE1", "OE2"}]
    assert loose == [("His28", 9), ("Arg31", 5)]


@BUNDLE
def test_the_numbers_of_the_bundle():
    ens = nmr()
    assert not ens.warnings and all(not d.unresolved and d.ligand.bond_source == "template" for d in ens.diagrams)
    assert all(d.ligand.n_atoms == 33 for d in ens.diagrams)                     # heavy atoms; the file has hydrogens
    assert ens.names[ens.member] == "model 6" and ens.chosen_by == "medoid"
    assert ens.mean_rmsd[ens.member] == pytest.approx(0.955, abs=2e-3)
    assert 0.2 < ens.pocket_rmsd[~np.eye(21, dtype=bool)].min() and ens.pocket_rmsd.max() < 0.8
    counts = {n: sum(c.n_models == n for c in ens.contacts) for n in sorted({c.n_models for c in ens.contacts})}
    assert len(ens.contacts) == 17 and counts == {1: 4, 3: 3, 5: 2, 9: 2, 11: 1, 14: 1, 21: 4}
    # The model that is drawn makes six of the seventeen: most of what the bundle has to say is elsewhere.
    assert sum(c.in_member for c in ens.contacts) == 6 and sum(not c.in_member for c in ens.contacts) == 11
    classes = symmetry_classes(ens.diagrams[0].ligand)

    def n_keys(**rules):
        return len({contact_key(r, d.ligand, classes, **rules) for d in ens.diagrams for r in d.interactions})
    assert (n_keys(symmetry=False, hydrophobic_by_residue=False), n_keys(hydrophobic_by_residue=False),
            n_keys(symmetry=False), n_keys()) == (21, 20, 17, 17)
    assert [len(load_interactions(NMR / m["table"])) for m in manifest(NMR)["models"]] == [
        8, 7, 7, 9, 7, 6, 6, 7, 7, 7, 7, 7, 6, 6, 7, 7, 9, 7, 7, 10, 7]


@BUNDLE
@pytest.mark.parametrize("style", ("detailed", "schematic"))
def test_the_figure_of_the_bundle(style):
    ens = nmr()
    res = ens.diagram().render(style=style)
    assert res.status == "ok" and res.quality.valid and res.quality.n_residues == 6
    text = words(res)
    assert "Occupancy (share of the 21 models of an NMR bundle, written on the line):" in text
    assert ("Shown is model 6: of the 21 models, the one whose ligand lies nearest to the others' (RMSD 0.95 Å on "
            "average, pockets superposed).") in text
    assert ("Not drawn: 11 contacts that other models make and model 6 does not (Asp26 hydrogen bond to NA2 (9 of "
            "21); His28 salt bridge (9 of 21); Arg31 salt bridge (5 of 21); and 8 more).") in text
    # On request they are drawn: seven residues more, each line with its share.
    faint = ens.diagram(absent="faint")
    assert len(faint.elsewhere) == 11 and faint.n_interactions == ens.diagram().n_interactions
    drawn = faint.render(style=style)
    assert drawn.status == "ok" and drawn.quality.valid and drawn.quality.n_residues == 13
    shares = sorted(f.item.content for g in groups(drawn, "interaction") if g.data.get("absent") == "true"
                    for f in inside(drawn, g, Text))
    assert shares == ["14 %", "24 %", "24 %", "43 %", "43 %", "5 %", "5 %", "5 %", "5 %"]
    assert "Faint, with the share of the 21 models that make it: 11 contacts" in words(drawn)


@needs_plip
@MAOB
@BUNDLE
def test_the_committed_tables_are_what_plip_finds():
    from bindview.detect import detect_interactions

    for folder in (SET, NMR):
        for model in manifest(folder)["models"]:
            found = detect_interactions(entry(model), model["ligand"])
            assert found == load_interactions(folder / model["table"]), model["id"]
