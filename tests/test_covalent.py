"""A ligand bound to its target by a covalent bond (``bindview.covalent``)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bindview import Diagram, Interaction, InteractionType, read_structure
from bindview.covalent import as_bound, covalent_bonds
from bindview.distance_table import ligand_contacts
from bindview.ligand import build_ligand
from bindview.scene import Line
from bindview.structure import ResidueKey, Structure, parse_pdb
from conftest import STYLES, TESTDATA, meta_of, model_of, structure_of
from scenetools import groups, inside

T = InteractionType

# N-methylacrylamide on a cysteine: the double bond C3-C4 of the free ligand is single in the model, and C4 is bound
# to SG. The numbers are ideal bond lengths, laid flat by hand.
ATOMS = [
    ("ATOM", "N", "CYS", "A", 12, -4.10, 3.50, 0.00), ("ATOM", "CA", "CYS", "A", 12, -2.90, 2.70, 0.00),
    ("ATOM", "C", "CYS", "A", 12, -2.90, 1.60, 1.05), ("ATOM", "O", "CYS", "A", 12, -3.85, 1.45, 1.80),
    ("ATOM", "CB", "CYS", "A", 12, -1.60, 3.50, 0.00), ("ATOM", "SG", "CYS", "A", 12, -0.10, 2.50, 0.00),
    ("ATOM", "CA", "LEU", "A", 40, 4.00, 5.80, 0.00), ("ATOM", "CB", "LEU", "A", 40, 3.10, 4.60, 0.00),
    ("ATOM", "CG", "LEU", "A", 40, 3.60, 3.30, 0.60), ("ATOM", "CD1", "LEU", "A", 40, 4.90, 3.00, 0.00),
    ("ATOM", "CD2", "LEU", "A", 40, 3.50, 3.30, 2.10), ("ATOM", "N", "LEU", "A", 40, 5.30, 5.50, -0.40),
    ("ATOM", "C", "LEU", "A", 40, 3.60, 7.10, -0.60), ("ATOM", "O", "LEU", "A", 40, 2.50, 7.30, -1.00),
    ("HETATM", "C4", "LIG", "B", 1, 0.00, 0.70, 0.00), ("HETATM", "C3", "LIG", "B", 1, 1.35, 0.00, 0.00),
    ("HETATM", "C2", "LIG", "B", 1, 1.35, -1.52, 0.00), ("HETATM", "O1", "LIG", "B", 1, 0.30, -2.15, 0.00),
    ("HETATM", "N1", "LIG", "B", 1, 2.55, -2.10, 0.00), ("HETATM", "C1", "LIG", "B", 1, 2.70, -3.55, 0.00),
]
SMILES = "CNC(=O)C=C"


def pdb(atoms=ATOMS, link=False) -> str:
    lines = []
    if link:
        lines.append("LINK         SG  CYS A  12                 C4  LIG B   1     1555   1555  1.80  ")
    for k, (rec, name, res, chain, number, x, y, z) in enumerate(atoms, 1):
        lines.append(f"{rec:<6}{k:>5}  {name:<3} {res:>3} {chain}{number:>4}    {x:8.3f}{y:8.3f}{z:8.3f}"
                     f"  1.00 20.00          {name[0]:>2}")
    return "\n".join(lines) + "\nEND\n"


def diagram(covalent, interactions=(), **options) -> Diagram:
    return Diagram.from_structure(parse_pdb(pdb(**options)), "LIG", SMILES, interactions, covalent=covalent)


def test_a_declared_bond_is_read_and_one_by_distance_is_found():
    key = ResidueKey("B", 1)
    (declared,) = covalent_bonds(parse_pdb(pdb(link=True)), key)
    (found,) = covalent_bonds(parse_pdb(pdb()), key)
    for bond, said in ((declared, True), (found, False)):
        assert (bond.ligand_atom, bond.atom, bond.resname, bond.residue) == ("C4", "SG", "CYS", ResidueKey("A", 12))
        assert bond.declared is said and bond.distance == pytest.approx(1.80, abs=0.01)
    record = declared.as_interaction()
    assert record.type is T.COVALENT and record.ligand_atom == "C4" and record.protein_atom == "SG"
    # An atom that is near and not bonded is none: 2.3 A from the sulphur.
    apart = [a if a[1] != "C4" else (*a[:5], 0.0, 0.2, 0.0) for a in ATOMS]
    assert covalent_bonds(parse_pdb(pdb(apart)), key) == []


@pytest.mark.parametrize("name", ["HOH", "TIP3", "SOL", "D2O"])
def test_a_water_at_the_length_of_a_bond_is_no_bond_under_any_name(name):
    """An oxygen 1.43 Å from C4 of the ligand: a bond where it is of a residue, none where it is a water."""
    key = ResidueKey("B", 1)
    model = parse_pdb(pdb([a for a in ATOMS if a[2] == "LIG"]))

    def beside(resname: str):
        oxygen = replace(model.atoms[0], index=len(model.atoms), name="O", element="O", resname=resname, chain="W",
                         resnum=7, x=0.0, y=2.13, z=0.0)
        return Structure(atoms=model.atoms + [oxygen], source="the ligand and one oxygen")

    (bond,) = covalent_bonds(beside("MOH"), key)
    assert (bond.ligand_atom, bond.resname, bond.atom) == ("C4", "MOH", "O")
    assert covalent_bonds(beside(name), key) == []


def test_a_water_nearer_than_the_residue_does_not_hide_the_bond():
    """A water 1.39 Å from C4, nearer than the sulphur it is bound to at 1.80 Å (a site that is modelled twice,
    with the ligand bound and with a water in its place)."""
    water = ("HETATM", "O", "HOH", "W", 7, -1.30, 0.20, 0.00)
    (bond,) = covalent_bonds(parse_pdb(pdb(ATOMS + [water])), ResidueKey("B", 1))
    assert (bond.ligand_atom, bond.resname, bond.atom) == ("C4", "CYS", "SG") and not bond.declared


def test_the_bonds_of_its_own_chain_are_none_of_a_residue():
    """A residue of a chain is bound to the one before it and the one after it, and that is the chain."""
    rna = read_structure(TESTDATA / "nucleic" / "riboswitch_bisguanidine" / "model.pdb")
    for number in (1, 10, 18):                             # the first of a hairpin, one in its loop, the last
        assert covalent_bonds(rna, ResidueKey("B", number)) == []
    protein = read_structure(model_of("kit_imatinib"))
    lysine = next(k for k in protein.residues() if protein.residue_name(k) == "LYS")
    assert covalent_bonds(protein, lysine) == []


#: An acetyl group on the nitrogen of the cysteine, its carbon called C as a file may call it.
ACETYL = [("HETATM", "CH3", "ACY", "B", 1, -6.50, 3.60, 0.00), ("HETATM", "C", "ACY", "B", 1, -5.26, 2.83, 0.00),
          ("HETATM", "O", "ACY", "B", 1, -5.30, 1.60, 0.00)]


def test_a_ligand_bound_by_an_atom_called_c_to_a_nitrogen_called_n_is_bound():
    protein = [a for a in ATOMS if a[0] == "ATOM"]
    (bond,) = covalent_bonds(parse_pdb(pdb(protein + ACETYL)), ResidueKey("B", 1))
    assert (bond.ligand_atom, bond.resname, bond.atom, bond.residue) == ("C", "CYS", "N", ResidueKey("A", 12))
    assert bond.distance == pytest.approx(1.34, abs=0.01)
    # ... also where the file has it under the chain of the protein, after it, as a deposited file has a ligand.
    filed = [(*a[:3], "A", 401, *a[5:]) for a in ACETYL]
    (bond,) = covalent_bonds(parse_pdb(pdb(protein + filed)), ResidueKey("A", 401))
    assert (bond.ligand_atom, bond.atom, bond.residue) == ("C", "N", ResidueKey("A", 12))
    # Before the cysteine in its chain it is the residue before it, and the bond is the chain's.
    cap = [(*a[:3], "A", 11, *a[5:]) for a in ACETYL]
    assert covalent_bonds(parse_pdb(pdb(cap + protein)), ResidueKey("A", 11)) == []


def test_nothing_is_looked_for_unless_asked():
    plain = diagram(False)
    assert not [r for r in plain.interactions if r.type is T.COVALENT]
    assert plain.render().svg == Diagram.from_structure(parse_pdb(pdb(link=True)), "LIG", SMILES, ()).render().svg


@pytest.mark.parametrize("style", STYLES)
def test_the_bond_is_drawn_from_atom_to_atom_on_a_highlight(style):
    dia = diagram(True, link=True)
    res = dia.render(style=style)
    assert res.status == "ok" and res.quality.valid
    (bond,) = [g for g in groups(res, "interaction") if g.data["type"] == "covalent"]
    assert bond.data["residue"] == "A:CYS12" and bond.data["ligand-atoms"] == "C4"
    assert bond.title.startswith("Covalent bond: Cys12") and "1.80" in bond.title
    highlight, line = [f.item.paint for f in inside(res, bond, Line)]
    assert highlight.stroke == "warning" and highlight.opacity < 0.6 and highlight.width > 3 * line.width
    assert line.stroke == "ligand_bond" and not line.dash
    assert "Covalent bond" in res.svg and 'class="lining"' not in res.svg          # no wall where the bond is
    assert any(w.startswith("covalent bond: C4 is bound to Cys12 SG (1.80") and "declared by the file" in w
               for w in res.warnings)


def test_the_ligand_is_drawn_as_it_is_bound():
    dia = diagram(True)
    lig = dia.ligand
    order = {frozenset((lig.names[b.GetBeginAtomIdx()], lig.names[b.GetEndAtomIdx()])): b.GetBondTypeAsDouble()
             for b in lig.mol.GetBonds()}
    assert order[frozenset(("C3", "C4"))] == 1.0 and order[frozenset(("C2", "O1"))] == 2.0
    assert any("the double bond C4–C3 of the SMILES is 1.52" in w and "drawn single" in w for w in dia.warnings)
    # ... and not where the model has the bond at the length of a double bond.
    short = [a if a[1] != "C3" else (*a[:5], 1.15, 0.0, 0.0) for a in ATOMS]
    built = build_ligand([a for a in parse_pdb(pdb(short)).atoms if a.resname == "LIG"], SMILES)
    assert as_bound(built, built.index_of("C4")) is None


def test_a_contact_of_the_bonded_atoms_is_left_out():
    near = [{"interaction_type": "hydrophobic", "residue_name": "CYS", "residue_number": 12, "chain": "A",
             "protein_atom": "CB", "ligand_atom": "C4", "distance": 2.8},
            {"interaction_type": "hydrophobic", "residue_name": "LEU", "residue_number": 40, "chain": "A",
             "protein_atom": "CB", "ligand_atom": "C3", "distance": 3.9}]
    assert [r.record.residue_name for r in diagram(False, near).interactions] == ["CYS", "LEU"]
    dia = diagram(True, near)
    assert [(r.record.residue_name, r.type) for r in dia.interactions] == [("LEU", T.HYDROPHOBIC), ("CYS", T.COVALENT)]
    assert any(w.startswith("1 contact between atoms that the covalent bond joins is left out: Cys12") for w in dia.warnings)


def test_a_table_can_bring_the_bond_itself():
    row = {"interaction_type": "covalent", "residue_name": "CYS", "residue_number": 12, "chain": "A",
           "protein_atom": "SG", "ligand_atom": "C4", "distance": 1.8}
    for covalent in (False, True):                         # brought once, found once: one bond
        dia = diagram(covalent, [row])
        assert [r.type for r in dia.interactions] == [T.COVALENT]
        assert any("drawn as bound" in w for w in dia.warnings)
    assert Interaction(type="covalent bond", residue_number=12, residue_name="CYS").type is T.COVALENT


def test_a_model_without_one_says_so():
    ligand = meta_of("kit_imatinib")["ligand"]
    dia = Diagram.from_structure(model_of("kit_imatinib"), ligand["residue_name_in_cif"], ligand["smiles"],
                                 interactions=[], covalent=True)
    assert "covalent bonds were asked for, and the model has none between the ligand and a residue" in dia.warnings


def test_the_bonds_of_an_mmcif_file_are_read():
    """6JXT declares the bond of osimertinib's C9 to the sulphur of Cys797 in its _struct_conn, 1.801 Å long."""
    st = structure_of("egfr_osimertinib")
    assert {(k.a_atom, k.b_atom) for k in st.links} == {("SG", "C9")}
    (bond,) = covalent_bonds(st, st.select_ligand(meta_of("egfr_osimertinib")["ligand"]["residue_name_in_cif"]))
    assert (bond.ligand_atom, bond.resname, bond.atom, bond.residue) == ("C9", "CYS", "SG", ResidueKey("A", 797))
    assert bond.declared and bond.distance == pytest.approx(1.801, abs=0.001)


@pytest.mark.parametrize("bound", [False, True], ids=["free", "bound"])
def test_osimertinib_on_the_cysteine_of_egfr(bound):
    """The SMILES of the free drug has a double bond where the cysteine added, and the ligand is drawn as it is
    bound; the SMILES of the bound drug has none there, and nothing is redrawn."""
    ligand = meta_of("egfr_osimertinib")["ligand"]
    name, smiles = ligand["residue_name_in_cif"], ligand["smiles"]
    if bound:
        smiles = smiles.replace("NC(=O)C=C", "NC(=O)CC")
        assert smiles != ligand["smiles"]
    st = structure_of("egfr_osimertinib")
    dia = Diagram.from_structure(st, name, smiles, ligand_contacts(st, name, smiles), covalent=True)
    (bond,) = [r for r in dia.interactions if r.type is T.COVALENT]
    assert bond.record.residue_label == "Cys797" and bond.record.distance == pytest.approx(1.80, abs=0.01)
    assert dia.ligand.names[bond.ligand.atom] == "C9"
    assert any("drawn as bound" in w for w in dia.warnings) == (not bound)
    assert dia.render().quality.valid


def test_the_command_line_takes_it(tmp_path):
    from bindview.cli import main

    model, table, out = tmp_path / "m.pdb", tmp_path / "t.tsv", tmp_path / "f.svg"
    model.write_text(pdb(link=True))
    table.write_text("interaction_type\tresidue_name\tresidue_number\tchain\tprotein_atom\tligand_atom\tdistance\n")
    assert main(["render", str(model), "--ligand", "LIG", "--smiles", SMILES, "--interactions", str(table),
                 "--covalent", "-o", str(out)]) == 0
    assert "Covalent bond" in out.read_text()
