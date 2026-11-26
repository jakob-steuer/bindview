"""``bindview.distance_table.ligand_contacts``: the contacts of a small molecule from heavy atoms alone.

The table is for when PLIP is not installed and nobody brought a table. These tests hold it against PLIP's own
tables of the cases, what it knows and does not know without bond orders, and that its rows are rows a diagram
takes.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import pytest

from conftest import CASES, TESTDATA, diagram_of, meta_of, model_of, one_residue_peptide, structure_of
from bindview import Diagram, load_interactions, read_structure
from bindview.distance_table import HBOND, SALT, as_tsv, contact_table, ligand_contacts
from bindview.interactions import parse_interactions
from bindview.structure import Atom, ResidueKey, Structure


def table_of(case: str):
    ligand = meta_of(case)["ligand"]
    return ligand_contacts(structure_of(case), ligand["residue_name_in_cif"], ligand["smiles"])


def by_residue(rows) -> "set[tuple[str, str, int]]":
    return {(str(r["interaction_type"]), str(r["chain"]), int(str(r["residue_number"]))) for r in rows}


def of_kind(rows, kind: str) -> list:
    return [r for r in rows if r["interaction_type"] == kind]


def test_the_table_has_every_contact_of_plip_that_its_criteria_reach():
    """A contact is its kind and its residue. Of PLIP's contacts in the tables of the cases, the table has every
    hydrophobic contact, ring stacking, cation over a ring and halogen bond, every hydrogen bond whose two atoms
    are 3.5 Å apart or nearer and every salt bridge at 4.0 Å or nearer: its own criteria. It has no water
    bridge."""
    reached: Counter = Counter()
    for case in CASES:
        ours = table_of(case)
        assert not of_kind(ours, "water_bridge")
        have = by_residue(ours)
        for r in load_interactions(TESTDATA / case / "interactions.tsv"):
            kind = r.type.value
            if kind == "water_bridge" or (kind, r.distance > HBOND) == ("hbond", True) or \
                    (kind, r.distance > SALT) == ("saltbridge", True):
                continue
            reached[kind] += 1
            assert (kind, r.chain, r.residue_number) in have, (case, kind, r.residue_label)
    assert set(reached) == {"hbond", "saltbridge", "hydrophobic", "pi_stacking", "pi_cation", "halogen"}


def test_the_rings_and_the_halogen_are_the_ones_plip_names():
    """Imatinib's pyrimidine on Phe811, edge on, with the atoms and the distance PLIP gives that stacking;
    bicalutamide's fluorine on the carbonyl oxygen of Met742, as PLIP has it."""
    (stack,) = of_kind(table_of("kit_imatinib"), "pi_stacking")
    (plip,) = [r for r in load_interactions(TESTDATA / "kit_imatinib" / "interactions.tsv") if r.type.value == "pi_stacking"]
    assert (stack["residue_name"], stack["residue_number"]) == (plip.residue_name, plip.residue_number)
    assert set(str(stack["ligand_atoms"]).split(",")) == set(plip.ligand_atoms)
    assert float(str(stack["distance"])) == pytest.approx(plip.distance, abs=0.02)
    assert float(str(stack["angle"])) > 60.0 and plip.angle > 60.0                       # edge on, in both
    halogens = of_kind(table_of("ar_bicalutamide"), "halogen")
    (plip,) = [r for r in load_interactions(TESTDATA / "ar_bicalutamide" / "interactions.tsv") if r.type.value == "halogen"]
    assert (plip.residue_number, plip.protein_atom, plip.ligand_atom) in {
        (r["residue_number"], r["protein_atom"], r["ligand_atom"]) for r in halogens}
    assert all(str(r["ligand_atom"]).startswith("F") for r in halogens)


def test_the_rows_are_rows_a_diagram_takes():
    ligand = meta_of("kit_imatinib")["ligand"]
    name, smiles = ligand["residue_name_in_cif"], ligand["smiles"]
    model = structure_of("kit_imatinib")
    rows = ligand_contacts(model, name, smiles)
    dia = Diagram.from_structure(model, name, smiles=smiles, interactions=rows)
    assert dia.n_interactions == len(rows) and not dia.unresolved
    assert dia.render().status == "ok"
    again = parse_interactions(as_tsv(rows))                        # and they go through a file
    assert [(r.type.value, r.residue_number) for r in again] == [(r["interaction_type"], r["residue_number"]) for r in rows]
    assert ligand_contacts(model, name, smiles) == rows             # the same table twice


def test_without_bond_orders_there_is_no_ring_to_stack_on():
    """A SMILES that the file holds is used as one that is given. Without any, the table knows no ring of the
    ligand: no stacking, and no cation over a ring of the ligand. What goes by elements and distances stays:
    the hydrophobic contacts and the halogen bonds are the same rows."""
    ligand = meta_of("kit_imatinib")["ligand"]
    filed = dataclasses.replace(structure_of("kit_imatinib"), chem_comp_smiles={ligand["residue_name_in_cif"]: ligand["smiles"]})
    assert ligand_contacts(filed, ligand["residue_name_in_cif"]) == table_of("kit_imatinib")
    stacked = 0
    for case in CASES:
        full = table_of(case)
        bare = ligand_contacts(dataclasses.replace(structure_of(case), chem_comp_smiles={}),
                               meta_of(case)["ligand"]["residue_name_in_cif"])
        stacked += len(of_kind(full, "pi_stacking"))
        assert not of_kind(bare, "pi_stacking")
        assert all(len(str(r["protein_atoms"]).split(",")) > 1 for r in of_kind(bare, "pi_cation"))   # the ring is the residue's
        for kind in ("hydrophobic", "halogen"):
            assert of_kind(bare, kind) == of_kind(full, kind)
    assert stacked > 0


def test_an_ion_is_held_by_what_lies_within_three_angstrom():
    """The zinc of carbonic anhydrase II as the ligand: its rows are metal contacts within 3 Å, with the three
    histidine nitrogens that the file's own record of its links (_struct_conn) puts on it."""
    rows = ligand_contacts(structure_of("ca2_celecoxib"), "ZN")
    assert {str(r["interaction_type"]) for r in rows} == {"metal"}
    assert {(r["residue_name"], r["residue_number"], r["protein_atom"]) for r in rows} == {
        ("HIS", 94, "NE2"), ("HIS", 96, "NE2"), ("HIS", 119, "ND1")}
    assert all(float(str(r["distance"])) <= 3.0 for r in rows)


def test_a_ligand_in_rna_has_a_salt_bridge_to_each_phosphate_it_touches():
    folder = TESTDATA / "nucleic" / "riboswitch_bisguanidine"
    smiles = json.loads((folder / "meta.json").read_text())["ligand"]["smiles"]
    model = read_structure(folder / "model.pdb")
    rows = ligand_contacts(model, "BGD", smiles)
    salt = {(r["chain"], r["residue_number"]): r for r in rows if r["interaction_type"] == "saltbridge"}
    assert set(salt) == {("A", 8), ("B", 8), ("B", 9)}
    # Each guanidinium is one group, not also a nitrogen by itself; the phosphate is named with its oxygens.
    assert all(len(str(r["ligand_atoms"]).split(",")) == 4 and r["protein_atoms"] == "P,O1P,O2P" for r in salt.values())
    assert float(str(salt["B", 9]["distance"])) == 2.84 and float(str(salt["B", 8]["distance"])) == 2.89
    # No hydrogen bond is counted between the two ends of a salt bridge: the phosphates have none of their own.
    bonds = {(r["chain"], r["residue_number"], r["protein_atom"]) for r in rows if r["interaction_type"] == "hbond"}
    assert ("B", 10, "N7") in bonds and ("B", 10, "O6") in bonds
    assert not any(atom in ("O1P", "O2P") for _, _, atom in bonds)
    assert {(r["chain"], r["residue_number"]) for r in rows if r["interaction_type"] == "pi_cation"} == {("B", 7), ("B", 8)}
    dia = Diagram.from_structure(model, "BGD", smiles=smiles, interactions=rows)
    assert dia.render().status == "ok"


def in_a_row(chain: str, number: int, name: str, atoms: "list[str]", tip: str, at: "tuple[float, float, float]",
             away: float) -> "list[tuple]":
    """A made-up residue for a table that goes by distances alone: its atom ``tip`` at ``at``, the atoms before
    it and after it in two rows behind it, 1.5 Å from one to the next, along y in the direction ``away``."""
    k = atoms.index(tip)
    return [(chain, number, name, n, n[0], at[0] + (0.7 if i < k else 0.0), at[1] + away * 1.5 * abs(i - k),
             at[2] + (0.9 if i < k else 0.0)) for i, n in enumerate(atoms)]


def made_up(*residues: "list[tuple]") -> Structure:
    rows = [row for residue in residues for row in residue]
    return Structure(atoms=[Atom(i, name, element, resname, chain, number, "", x, y, z, resname == "ACT")
                            for i, (chain, number, resname, name, element, x, y, z) in enumerate(rows)],
                     source="made up")


def test_a_salt_bridge_of_one_residue_hides_no_hydrogen_bond_of_another():
    """Acetate between an arginine and an asparagine, an oxygen 2.9 Å from a nitrogen of each. The arginine's
    nitrogens are its atoms 7, 9 and 10, and the asparagine's ND2 is its atom 7: what is one end of the salt
    bridge by its place in the arginine is no end of it in the asparagine."""
    acetate = [("L", 1, "ACT", "C1", "C", 0.0, 0.0, 0.0), ("L", 1, "ACT", "C2", "C", 1.5, 0.0, 0.0),
               ("L", 1, "ACT", "O1", "O", 2.1, 1.1, 0.0), ("L", 1, "ACT", "O2", "O", 2.1, -1.1, 0.0)]
    arg = in_a_row("A", 10, "ARG", ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"], "NH1",
                   (2.1, 4.0, 0.0), 1.0)
    asn = in_a_row("A", 20, "ASN", ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"], "ND2", (2.1, -4.0, 0.0), -1.0)

    def table(*residues):
        rows = ligand_contacts(made_up(acetate, *residues), ResidueKey("L", 1), "CC(=O)O")
        return [(r["interaction_type"], r["residue_name"], r["residue_number"], r["protein_atom"] or r["protein_atoms"],
                 r["ligand_atom"] or r["ligand_atoms"], r["distance"]) for r in rows]

    bridge, bond = ("saltbridge", "ARG", 10, "NE,NH1,NH2", "C2,O1,O2", 2.9), ("hbond", "ASN", 20, "ND2", "O2", 2.9)
    assert table(asn) == [bond] and table(arg) == [bridge]
    assert table(arg, asn) == [bridge, bond]                       # the arginine first in the file
    assert table(asn, arg) == [bridge, bond]


def test_two_atoms_that_hold_one_metal_have_no_hydrogen_bond_between_them():
    """Celecoxib on the zinc of carbonic anhydrase II. Its sulfonamide nitrogen holds the zinc, and so does a
    nitrogen of each of His94, His96 and His119, 3.2 to 3.5 Å from it and from the oxygen beside it: that is
    how far apart two atoms on one metal are, and no hydrogen bond. Thr199 holds no zinc, and its two stay."""
    ligand = meta_of("ca2_celecoxib")["ligand"]
    name, smiles = ligand["residue_name_in_cif"], ligand["smiles"]
    rows = ligand_contacts(structure_of("ca2_celecoxib"), name, smiles)
    assert [(r["residue_name"], r["residue_number"], r["protein_atom"], r["ligand_atom"], r["distance"])
            for r in rows if r["interaction_type"] == "hbond"] == [("THR", 199, "OG1", "N3", 2.84),
                                                                   ("THR", 199, "N", "O1", 2.93)]
    assert [(r["residue_name"], r["residue_number"], r["ligand_atom"], r["distance"])
            for r in rows if r["interaction_type"] == "metal"] == [("ZN", 600, "N3", 1.97), ("ZN", 600, "O2", 2.99)]
    assert len(rows) == 10
    # Drawn with the zinc as the bridge it is, the histidines have their contact with the zinc, and its length.
    bridge = Diagram.from_structure(structure_of("ca2_celecoxib"), name, smiles, rows, ions=True)
    assert sorted((r.record.residue_label, r.record.protein_atom, r.record.distance)
                  for r in bridge.interactions if r.type.value == "metal") == [
        ("His119", "ND1", 2.07), ("His94", "NE2", 2.0), ("His96", "NE2", 1.99)]
    assert {r.record.residue_label for r in bridge.interactions if r.type.value == "hbond"} == {"Thr199"}
    assert not any("left out" in w for w in bridge.warnings)       # the table had no hydrogen bond across the zinc


def test_a_metal_of_the_ligand_itself_takes_the_hydrogen_bonds_to_what_holds_it():
    """The haem of haemoglobin as the ligand: its four nitrogens and a nitrogen of His87 hold its iron, 2.9 Å
    from each other. His87 has its row with the iron, and no hydrogen bond with a nitrogen of the haem."""
    rows = ligand_contacts(structure_of("hb_carbon_monoxide"), "A:HEM:142")
    assert [(r["interaction_type"], r["protein_atom"], r["ligand_atom"], r["distance"])
            for r in rows if r["residue_number"] == 87] == [("metal", "NE2", "FE", 2.11)]
    assert not [r for r in rows if r["interaction_type"] == "hbond"]
    # The carbon monoxide on the same iron is 3.3 Å from His58, which holds no iron: that hydrogen bond stays.
    monoxide = meta_of("hb_carbon_monoxide")["ligand"]
    rows = ligand_contacts(structure_of("hb_carbon_monoxide"), monoxide["residue_name_in_cif"], monoxide["smiles"])
    assert [(r["interaction_type"], r["residue_name"], r["residue_number"], r["protein_atom"], r["ligand_atom"],
             r["distance"]) for r in rows] == [("hbond", "HIS", 58, "NE2", "O", 3.3)]


def test_a_metal_takes_only_the_hydrogen_bond_between_two_atoms_it_holds_itself():
    """Methanol with its oxygen 2.0 Å from a zinc. A histidine's nitrogen on the same zinc, 2.8 Å from the
    oxygen, has no hydrogen bond with it. A threonine's oxygen 2.8 Å from it and 3.4 Å from the zinc has one,
    and so has a serine's that is held by a second zinc, 5 Å from the methanol."""
    methanol = [("L", 1, "MOH", "C1", "C", 0.0, 0.0, 0.0), ("L", 1, "MOH", "O1", "O", 1.4, 0.0, 0.0)]
    zinc, second = [("A", 100, "ZN", "ZN", "Zn", 3.4, 0.0, 0.0)], [("A", 101, "ZN", "ZN", "Zn", 1.4, 0.0, 5.0)]
    his = in_a_row("A", 30, "HIS", ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"], "NE2",
                   (3.4, 2.0, 0.0), 1.0)
    thr = in_a_row("A", 40, "THR", ["N", "CA", "C", "O", "CB", "OG1", "CG2"], "OG1", (1.4, -2.8, 0.0), -1.0)
    ser = in_a_row("A", 50, "SER", ["N", "CA", "C", "O", "CB", "OG"], "OG", (1.4, 0.0, 2.9), -1.0)

    def table(*residues):
        rows = ligand_contacts(made_up(methanol, *residues), ResidueKey("L", 1), "CO")
        return [(r["interaction_type"], r["residue_name"], r["residue_number"], r["protein_atom"], r["distance"])
                for r in rows]

    to_his, to_thr = ("hbond", "HIS", 30, "NE2", 2.83), ("hbond", "THR", 40, "OG1", 2.8)
    to_ser, held = ("hbond", "SER", 50, "OG", 2.9), ("metal", "ZN", 100, "ZN", 2.0)
    assert table(his, thr, ser) == [to_his, to_thr, to_ser]            # no metal: three hydrogen bonds
    assert table(his, thr, ser, zinc, second) == [to_thr, to_ser, held]
    assert table(zinc, second, ser, thr, his) == [to_thr, to_ser, held]
    assert table(his, thr, ser, second) == [to_his, to_thr, to_ser]    # a zinc that the methanol does not hold


def test_a_metal_between_two_chains_takes_the_hydrogen_bond_between_the_atoms_that_hold_it():
    """The table of a chain: a histidine of chain P and one of chain A with a nitrogen each on one zinc,
    2.8 Å from each other, and beside them a serine of P 2.8 Å from a threonine of A."""
    his_p = in_a_row("P", 1, "HIS", ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"], "NE2",
                     (0.0, -2.0, 0.0), -1.0)
    ser_p = in_a_row("P", 2, "SER", ["N", "CA", "C", "O", "CB", "OG"], "OG", (9.0, -1.4, 0.0), -1.0)
    his_a = in_a_row("A", 30, "HIS", ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"], "NE2",
                     (2.0, 0.0, 0.0), 1.0)
    thr_a = in_a_row("A", 40, "THR", ["N", "CA", "C", "O", "CB", "OG1", "CG2"], "OG1", (9.0, 1.4, 0.0), 1.0)
    zinc = [("A", 100, "ZN", "ZN", "Zn", 0.0, 0.0, 0.0)]

    def bonds(*more):
        rows = contact_table(made_up(his_p, ser_p, his_a, thr_a, *more), "P")
        return [(r["residue_name"], r["residue_number"], r["protein_atom"], r["ligand_atom"], r["distance"])
                for r in rows if r["interaction_type"] == "hbond"]

    across, beside = ("HIS", 30, "NE2", "1/NE2", 2.83), ("THR", 40, "OG1", "2/OG", 2.8)
    assert bonds() == [across, beside]
    assert bonds(zinc) == [beside]


def test_a_peptide_given_as_one_residue_is_a_ligand_like_any_other():
    """The nonamer of an MHC complex rewritten as one hetero residue with its SMILES: a table of its contacts, and
    a ligand of all its heavy atoms on which every row is placed."""
    structure, smiles = one_residue_peptide("mhc_peptide")
    key = ResidueKey("Z", 1)
    rows = ligand_contacts(structure, key, smiles)
    kinds = Counter(str(r["interaction_type"]) for r in rows)
    assert kinds["hbond"] and kinds["hydrophobic"]
    chain = meta_of("mhc_peptide")["peptide_chain"]
    heavy = [a for a in structure_of("mhc_peptide").atoms if a.chain == chain and a.element not in ("H", "D")
             and not a.hetero]
    dia = Diagram.from_structure(structure, key, smiles, rows)
    assert dia.ligand.n_atoms == len(heavy) and not dia.unresolved


@pytest.mark.parametrize("case", CASES)
def test_a_figure_from_the_table_is_drawn(case):
    ligand = meta_of(case)["ligand"]
    name, smiles = ligand["residue_name_in_cif"], ligand["smiles"]
    model = structure_of(case)
    dia = Diagram.from_structure(model, name, smiles=smiles, interactions=ligand_contacts(model, name, smiles))
    assert dia.render().status == "ok" and not dia.unresolved


def test_a_diagram_can_be_asked_for_the_table_or_given_the_path_of_one(tmp_path):
    from bindview import Ensemble
    from bindview.cli import main

    case, model = TESTDATA / "kit_imatinib", model_of("kit_imatinib")
    lig = meta_of("kit_imatinib")["ligand"]
    name, smiles = lig["residue_name_in_cif"], lig["smiles"]
    asked = Diagram.from_structure(model, name, smiles, "distances")
    made = Diagram.from_structure(model, name, smiles, ligand_contacts(asked.structure, name, smiles))
    assert asked.source == "distances" and asked.render().svg == made.render().svg
    for path in (case / "interactions.tsv", str(case / "interactions.tsv")):
        given = Diagram.from_structure(model, name, smiles, path)
        assert given.source == "supplied"
        assert [r.record for r in given.interactions] == [r.record for r in diagram_of("kit_imatinib").interactions]
    ens = Ensemble.from_structures([model] * 2, name, smiles, "distances")
    assert [r.record for r in ens.diagrams[1].interactions] == [r.record for r in asked.interactions]
    # The command line says where they are from.
    out = tmp_path / "f.svg"
    assert main(["render", str(model), "--ligand", name, "--smiles", smiles, "-q", "--distances-only",
                 "-o", str(out)]) == 0
    assert out.read_text() == asked.render().svg


def test_the_angle_of_a_stacking_row_goes_through_a_file():
    """``as_tsv`` wrote no angle, and a table read back from it had lost the angle of every ring that stacks."""
    ligand = meta_of("kit_imatinib")["ligand"]
    rows = ligand_contacts(structure_of("kit_imatinib"), ligand["residue_name_in_cif"], ligand["smiles"])
    stacks = [r for r in rows if r["interaction_type"] == "pi_stacking"]
    assert stacks and all("angle" in r for r in stacks)
    again = [r for r in parse_interactions(as_tsv(rows)) if r.type.value == "pi_stacking"]
    assert [r.angle for r in again] == [float(r["angle"]) for r in stacks]


@pytest.mark.parametrize("slug, a, b", [("mhc_peptide", "A", "B"), ("mhc_peptide", "A", "P"), ("thrombin_hirudin", "A", "B"),
                                        ("fab_lysozyme", "H", "C"), ("u1a_rna", "A", "P")])
def test_the_table_of_two_chains_is_the_same_from_either_side(slug, a, b):
    """A partner residue without N, CA or C lost its contacts, and a partner's C-terminal carboxylate made no salt
    bridge, so the table depended on which chain was the ligand: HLA-A2 against beta-2-microglobulin was 29 rows one
    way and 27 the other."""
    from collections import Counter

    from bindview.distance_table import interface_contacts

    st = structure_of(slug)
    one = Counter((c.type.value, c.chain_a, c.number_a, c.chain_b, c.number_b) for c in interface_contacts(st, [a], [b]))
    two = Counter((c.type.value, c.chain_b, c.number_b, c.chain_a, c.number_a) for c in interface_contacts(st, [b], [a]))
    assert one and one == two
