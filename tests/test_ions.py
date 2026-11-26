"""An ion between a ligand and its target, drawn as the bridge it is (``bindview.ions``)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bindview import Diagram, InteractionType, read_structure
from bindview.chains import chain_smiles
from bindview.distance_table import ligand_contacts
from bindview.exposure import structure_burial
from bindview.ions import bridges, holders, with_ions
from bindview.solvent import has_waters
from bindview.structure import Atom, ResidueKey, Structure, parse_pdb
from conftest import STYLES, TESTDATA, meta_of, model_of, structure_of
from scenetools import groups

T = InteractionType
#: A methanethiolate on a zinc that two histidines and an aspartate hold; only what holds it is at its place.
ATOMS = [
    ("HETATM", "C1", "LIG", "B", 1, 3.00, 1.70, 0.00), ("HETATM", "S1", "LIG", "B", 1, 2.30, 0.00, 0.00),
    ("HETATM", "ZN", "ZN", "C", 9, 0.00, 0.00, 0.00),
    ("ATOM", "CA", "HIS", "A", 10, -4.10, 4.60, 0.00), ("ATOM", "CB", "HIS", "A", 10, -3.50, 3.20, 0.00),
    ("ATOM", "CG", "HIS", "A", 10, -2.30, 3.00, 0.00), ("ATOM", "ND1", "HIS", "A", 10, -1.20, 3.80, 0.00),
    ("ATOM", "CD2", "HIS", "A", 10, -2.10, 1.60, 0.40), ("ATOM", "CE1", "HIS", "A", 10, -0.20, 2.90, 0.00),
    ("ATOM", "NE2", "HIS", "A", 10, -0.70, 1.90, 0.00),
    ("ATOM", "CA", "HIS", "A", 12, -4.10, -4.60, 0.00), ("ATOM", "CB", "HIS", "A", 12, -3.50, -3.20, 0.00),
    ("ATOM", "CG", "HIS", "A", 12, -2.30, -3.00, 0.00), ("ATOM", "ND1", "HIS", "A", 12, -1.20, -3.80, 0.00),
    ("ATOM", "CD2", "HIS", "A", 12, -2.10, -1.60, 0.40), ("ATOM", "CE1", "HIS", "A", 12, -0.20, -2.90, 0.00),
    ("ATOM", "NE2", "HIS", "A", 12, -0.70, -1.90, 0.00),
    ("ATOM", "CA", "ASP", "A", 30, 0.00, 1.20, -4.60), ("ATOM", "CB", "ASP", "A", 30, 0.00, 0.60, -3.30),
    ("ATOM", "CG", "ASP", "A", 30, 0.00, -0.40, -2.90), ("ATOM", "OD1", "ASP", "A", 30, 0.00, 0.00, -2.00),
    ("ATOM", "OD2", "ASP", "A", 30, 0.00, -1.50, -3.40),
]
#: A table: the ion's own contact with the ligand, a hydrogen bond across it, and one of the histidine's.
TABLE = [
    {"interaction_type": "metal", "residue_name": "ZN", "residue_number": 9, "chain": "C", "protein_atom": "ZN",
     "ligand_atom": "S1", "distance": 2.3},
    {"interaction_type": "hbond", "residue_name": "ASP", "residue_number": 30, "chain": "A", "protein_atom": "OD1",
     "ligand_atom": "S1", "distance": 3.05},
    {"interaction_type": "hbond", "residue_name": "HIS", "residue_number": 10, "chain": "A", "protein_atom": "NE2",
     "ligand_atom": "S1", "distance": 3.5},
    {"interaction_type": "hydrophobic", "residue_name": "HIS", "residue_number": 12, "chain": "A", "protein_atom": "CE1",
     "ligand_atom": "C1", "distance": 3.9},
]


def pdb(atoms=ATOMS) -> str:
    lines = []
    for k, (rec, name, res, chain, number, x, y, z) in enumerate(atoms, 1):
        element = "ZN" if name == "ZN" else f" {name[0]}"
        at = f"{name:<4}" if len(element.strip()) == 2 else f" {name:<3}"
        lines.append(f"{rec:<6}{k:>5} {at} {res:>3} {chain}{number:>4}    {x:8.3f}{y:8.3f}{z:8.3f}"
                     f"  1.00 20.00          {element}")
    return "\n".join(lines) + "\nEND\n"


def diagram(ions, table=TABLE) -> Diagram:
    return Diagram.from_structure(parse_pdb(pdb()), "LIG", "CS", table, ions=ions)


def test_an_ion_that_both_hold_is_found():
    (b,) = bridges(parse_pdb(pdb()), ResidueKey("B", 1))
    assert b.name == "ZN" and b.charge == 2 and [n for n, _ in b.ligand] == ["S1"]
    assert {(a.resname, a.resnum, a.name) for a, _ in b.protein} == {("HIS", 10, "NE2"), ("HIS", 12, "NE2"), ("ASP", 30, "OD1")}
    # Without what holds it there is no bridge, nor without the ligand's donor at its distance.
    assert bridges(parse_pdb(pdb([a for a in ATOMS if a[2] in ("LIG", "ZN")])), ResidueKey("B", 1)) == []
    far = [a if a[1] != "S1" else (*a[:5], 3.4, 0.0, 0.0) for a in ATOMS]
    assert bridges(parse_pdb(pdb(far)), ResidueKey("B", 1)) == []


def between_two(name: str, metal_of: str = "MG", metal: str = "Mg") -> Structure:
    """A magnesium that an oxygen of a ligand holds, and two molecules of one oxygen called ``name``; nothing of
    a protein. ``metal_of``: the residue the magnesium is of, the ligand itself for a metal of its own.
    ``metal``: another element in its place."""
    rows = [("L", 1, "LIG", "C1", "C", 0.0, 0.0, 0.0), ("L", 1, "LIG", "O1", "O", 1.4, 0.0, 0.0),
            ("L" if metal_of == "LIG" else "M", 1, metal_of, metal.upper(), metal, 3.5, 0.0, 0.0),
            ("W", 1, name, "O", "O", 5.6, 0.0, 0.0), ("W", 2, name, "O", "O", 3.5, 2.1, 0.0)]
    return Structure(atoms=[Atom(i, atom, element, resname, chain, number, "", x, y, z, True)
                            for i, (chain, number, resname, atom, element, x, y, z) in enumerate(rows)],
                     source=f"a magnesium between a ligand and two {name}")


@pytest.mark.parametrize("name", ["HOH", "WAT", "DOD", "H2O", "D2O", "TIP3", "SOL", "sol"])
def test_a_water_is_water_under_every_name_a_file_gives_it(name):
    key = ResidueKey("L", 1)
    # Two molecules that are no water hold the ion as a residue would, and cover the ligand ...
    (bridge,) = bridges(between_two("MOH"), key)
    assert [(a.resname, a.resnum) for a, _ in bridge.protein] == [("MOH", 1), ("MOH", 2)]
    assert len(holders(between_two("MOH", metal_of="LIG"), key)) == 2
    assert not has_waters(between_two("MOH"))
    # ... and two waters do neither, whatever they are called.
    model = between_two(name)
    assert bridges(model, key) == [] and holders(between_two(name, metal_of="LIG"), key) == []
    assert has_waters(model)
    assert structure_burial(model, key).sasa_complex > structure_burial(between_two("MOH"), key).sasa_complex


@pytest.mark.parametrize("residue, element, charge", [
    ("NA", "Na", 1), ("AG", "Ag", 1), ("MG", "Mg", 2), ("ZN", "Zn", 2), ("FE2", "Fe", 2), ("FE", "Fe", 3),
    ("AL", "Al", 3), ("GA", "Ga", 3), ("TB", "Tb", 3), ("GD3", "Gd", 3), ("YT3", "Y", 3)])
def test_an_ion_has_the_charge_of_its_element_or_of_its_name(residue, element, charge):
    """Not every metal is an ion of two charges: silver has one, aluminium and the lanthanides three."""
    model, key = between_two("MOH", residue, element), ResidueKey("L", 1)
    (bridge,) = bridges(model, key)
    assert bridge.charge == charge
    _, smiles = with_ions(model, key, "CO", [bridge])
    assert smiles == f"CO.[{element}+{charge if charge > 1 else ''}]"


def test_nothing_changes_unless_asked():
    plain = diagram(False)
    assert "ZN" not in plain.ligand.names and len(plain.interactions) == len(TABLE)


@pytest.mark.parametrize("style", STYLES)
def test_the_ion_stands_on_the_ligand_and_is_held(style):
    dia = diagram(True)
    assert dia.ligand.names[-1] == "ZN" and dia.ligand.bridging == [2] and dia.ligand.coordination == [(2, 1)]
    assert dia.ligand.mol.GetAtomWithIdx(2).GetFormalCharge() == 2 and dia.ligand.mol.GetAtomWithIdx(2).GetDegree() == 0
    res = dia.render(style=style)
    assert res.status == "ok" and res.quality.valid
    held = {g.data["residue"]: g.data for g in groups(res, "interaction") if g.data["type"] == "metal"}
    assert set(held) == {"A:HIS10", "A:HIS12", "A:ASP30"}
    assert all(d["ligand-atoms"] == "ZN" for d in held.values()) and held["A:ASP30"]["distance"] == "2.00"
    # The hydrogen bonds of what holds it are none, and the ion is no residue of its own any more.
    assert not [g for g in groups(res, "interaction") if g.data["type"] == "hbond"]
    assert not [g for g in groups(res, "residue") if "ZN" in g.data["residue"]]
    assert any("2 hydrogen bonds of the table with an atom that holds an ion are left out" in w for w in res.warnings)
    assert any(w.startswith("ion bridge: Zn is between S1 of the ligand and ") for w in res.warnings)
    # It is drawn beside the sulphur, outside the ligand.
    at = {n: np.array(p) for n, p in res.atoms.items()}
    unit = res.quality.bond_length
    assert 1.1 * unit < np.linalg.norm(at["ZN"] - at["S1"]) < 1.6 * unit and np.linalg.norm(at["ZN"] - at["C1"]) > 1.3 * unit


def held_from_beyond(atoms: "list[Atom]", bound: Atom, donor: Atom, metal: str) -> "list[Atom]":
    """The atoms, an ion 2.1 Å beyond ``donor`` on the line from ``bound`` (what the donor is bound to), and an
    aspartate of a chain K whose OD1 holds the ion from the other side."""
    out = np.array(donor.xyz) - np.array(bound.xyz)
    out /= np.linalg.norm(out)
    more = [(metal.upper(), metal, metal.upper(), "M", 2.1, True)]
    more += [(name, name[0], "ASP", "K", 4.2 + 1.5 * step, False)
             for step, name in enumerate(("OD1", "CG", "CB", "CA", "N", "C", "O"))]
    atoms = list(atoms)
    for name, element, resname, chain, far, hetero in more:
        x, y, z = np.array(donor.xyz) + out * far + (np.array([0.0, 1.2, 0.0]) if name in ("N", "O") else 0.0)
        atoms.append(Atom(len(atoms), name, element, resname, chain, 1, "", float(x), float(y), float(z), hetero))
    return atoms


def beside_its_box(res, ion: str, residue: str) -> bool:
    """Does the ion stand beside the box of this residue of the chain, level with it and 1.35 bonds from it?"""
    (box,) = [n for n in res.graph.nodes if n.attrs.get("form") == "chain_residue" and n.label == residue]
    (atom,) = [n for n in res.graph.nodes if n.kind == "ligand_atom" and n.label == ion]
    off = abs(atom.x - box.x) - box.attrs["width"] / 2.0
    return abs(atom.y - box.y) < 0.01 and off == pytest.approx(1.35 * res.quality.bond_length, abs=0.01)


@pytest.mark.parametrize("style", STYLES)
def test_an_ion_that_a_chain_of_the_file_holds_stands_beside_its_residue(style):
    """A hairpin of RNA as the ligand, a magnesium on a phosphate oxygen of its A8 and an aspartate beyond it."""
    model = read_structure(TESTDATA / "nucleic" / "riboswitch_bisguanidine" / "model.pdb")
    a8 = model.find_residue("B", 8)
    atoms = held_from_beyond(model.atoms, model.atom_by_name(a8, "P"), model.atom_by_name(a8, "O1P"), "Mg")
    dia = Diagram.from_structure(Structure(atoms=atoms, source="a hairpin with a magnesium"), "chain:B",
                                 interactions=[], ions=True)
    assert dia.backend.kind == "chain" and dia.ligand.names[-1] == "MG"
    assert "ion bridge: Mg is between 8/O1P of the ligand and Asp1 OD1" in dia.warnings
    res = dia.render(style=style)
    assert res.status == "ok" and res.quality.valid
    held = [g.data for g in groups(res, "interaction") if g.data["type"] == "metal"]
    assert [(d["residue"], d["ligand-atoms"], d["distance"]) for d in held] == [("K:ASP1", "MG", "2.10")]
    assert beside_its_box(res, "MG", "A8")


def test_an_ion_that_a_peptide_of_one_residue_holds_stands_beside_its_residue():
    """Nine residues of c-Kit as a file of a co-folding model has a peptide ligand: one residue, and its SMILES.
    A zinc on the carbonyl oxygen of the fifth, and an aspartate beyond it."""
    kit = read_structure(model_of("kit_imatinib"))
    merged, key, given, _ = kit.with_residues_as_ligand(kit.polymer_residues("A")[10:19])
    peptide = merged.residue_atoms(key)
    smiles, _ = chain_smiles(peptide, given)
    carbon, oxygen = (next(a for a in peptide if a.name == f"{given[4].number}/{name}") for name in ("C", "O"))
    atoms = held_from_beyond([replace(a, index=i) for i, a in enumerate(peptide)], carbon, oxygen, "Zn")
    dia = Diagram.from_structure(Structure(atoms=atoms, source="a peptide with a zinc"), key, smiles,
                                 interactions=[], ions=True)
    assert dia.backend.kind == "chain" and not dia.backend.deposited and dia.ligand.names[-1] == "ZN"
    res = dia.render()
    assert res.status == "ok" and res.quality.valid
    held = [g.data for g in groups(res, "interaction") if g.data["type"] == "metal"]
    assert [(d["residue"], d["ligand-atoms"]) for d in held] == [("K:ASP1", "ZN")]
    fifth = dia.backend.peptide.residues[4]
    assert beside_its_box(res, "ZN", f"{fifth.code3}{fifth.shown}")


def test_a_hydrogen_bond_that_is_one_stays():
    table = TABLE + [{"interaction_type": "hbond", "residue_name": "ASP", "residue_number": 30, "chain": "A",
                      "protein_atom": "OD2", "ligand_atom": "S1", "distance": 3.3}]
    kept = [r for r in diagram(True, table).interactions if r.type is T.HBOND]
    assert [r.record.protein_atom for r in kept] == ["OD2"]        # not the oxygen that holds the ion


def test_a_model_without_one_says_so():
    ligand = meta_of("kit_imatinib")["ligand"]
    dia = Diagram.from_structure(model_of("kit_imatinib"), ligand["residue_name_in_cif"], ligand["smiles"],
                                 interactions=[], ions=True)
    assert "ions were asked for, and the model has none between the ligand and the protein" in dia.warnings


def test_a_metal_of_the_ligand_itself_is_not_moved():
    """The iron of the haem is the haem's own: no ion bridges it, and what holds it is His87's nitrogen, the atom
    that the file's own record of its links (_struct_conn, metalc1) puts on it."""
    st = structure_of("hb_carbon_monoxide")
    key = st.select_ligand("A:142")
    assert bridges(st, key) == []
    assert [(a.chain, a.resname, a.resnum, a.name) for a in holders(st, key)] == [("A", "HIS", 87, "NE2")]


def test_celecoxib_on_the_zinc_of_carbonic_anhydrase():
    """The zinc between celecoxib's sulfonamide nitrogen and three histidines (the file's _struct_conn, metalc1
    to metalc4): drawn as the bridge it is, the histidines hold the zinc and have no hydrogen bond."""
    ligand = meta_of("ca2_celecoxib")["ligand"]
    name, smiles = ligand["residue_name_in_cif"], ligand["smiles"]
    st = structure_of("ca2_celecoxib")
    dia = Diagram.from_structure(st, name, smiles, ligand_contacts(st, name, smiles), ions=True)
    held = {r.record.residue_label for r in dia.interactions if r.type is T.METAL}
    assert held == {"His94", "His96", "His119"}
    assert not [r for r in dia.interactions if r.type is T.HBOND and r.record.residue_name == "HIS"]
    assert "ion bridge: Zn is between N3 of the ligand and His96 NE2, His94 NE2, His119 ND1" in dia.warnings
    for style in STYLES:
        assert dia.render(style=style).quality.valid


def test_the_command_line_takes_it(tmp_path):
    from bindview.cli import main

    model, table, out = tmp_path / "m.pdb", tmp_path / "t.tsv", tmp_path / "f.svg"
    model.write_text(pdb())
    table.write_text("interaction_type\tresidue_name\tresidue_number\tchain\tprotein_atom\tligand_atom\tdistance\n"
                     "metal\tZN\t9\tC\tZN\tS1\t2.3\n")
    assert main(["render", str(model), "--ligand", "LIG", "--smiles", "CS", "--interactions", str(table), "--ions",
                 "-o", str(out), "-q"]) == 0
    assert out.read_text().count("Metal coordination") >= 3
