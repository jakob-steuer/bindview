"""A ligand whose partner is RNA or DNA: what the library knows of a nucleotide, which of its atoms the
detailed style draws, and what a figure of a ligand in RNA holds.

A nucleotide is to these figures what an amino acid is: a residue with a backbone (sugar and phosphate)
and a side chain (the base). What is tested is the knowledge (templates, names, rings, charges), the rule
for the part that is drawn, and a real case: a bivalent ligand in a guanidine-II riboswitch dimer, whose
contacts PLIP found with the RNA as the receptor. A figure of a protein declares no colour for the class of a
nucleotide.
"""

import json

import pytest
from rdkit import Chem

from conftest import TESTDATA, needs_plip, nucleic_diagram, render_of
from bindview.fragments import build_fragment, fragment_atoms
from bindview.residues import (NUCLEOTIDE_BACKBONE, aromatic_rings, can_donate, canonical_atom, canonical_resname,
                             charged_groups, format_residue_label, given_atom, is_backbone_atom, is_nucleotide,
                             residue_class, template_bonds, template_for)

NUCLEOTIDES = ("A", "C", "G", "U", "DA", "DC", "DG", "DT")
CASE = TESTDATA / "nucleic" / "riboswitch_bisguanidine"


def one_piece(frag) -> bool:
    """Do the bonds of a fragment join all its atoms?"""
    seen, todo = {frag.atoms[0]}, [frag.atoms[0]]
    while todo:
        a = todo.pop()
        for x, y, _ in frag.bonds:
            for near, far in ((x, y), (y, x)):
                if near == a and far not in seen:
                    seen.add(far)
                    todo.append(far)
    return seen == set(frag.atoms)


# -- what a nucleotide is --------------------------------------------------------

@pytest.mark.parametrize("name", NUCLEOTIDES)
def test_a_template_is_its_smiles_atom_for_atom(name):
    t = template_for(name)
    mol = Chem.MolFromSmiles(t.smiles)
    assert t.kind == "nucleotide" and mol.GetNumAtoms() == len(t.atoms) == len(set(t.atoms))
    assert [a.GetSymbol().upper() for a in mol.GetAtoms()] == [n[0] for n in t.atoms]
    assert ("O2'" in t.atoms) == (not name.startswith("D"))          # a ribose, or a deoxyribose
    purine = name[-1] in "AG"
    assert [len(r) for r in aromatic_rings(name)] == ([5, 6] if purine else [6])
    in_rings = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
    assert {t.atoms[i] for i in in_rings} == {a for ring in aromatic_rings(name) for a in ring}
    bonded = {frozenset((a, b)) for a, b, _ in template_bonds(name)}
    assert frozenset(("C1'", "N9" if purine else "N1")) in bonded and frozenset(("P", "O5'")) in bonded
    assert residue_class(name) == "nucleotide" and is_nucleotide(name)
    assert charged_groups(name) == [(-1, ("P", "OP1", "OP2", "O1P", "O2P"))]


def test_the_names_that_force_fields_and_older_files_use():
    assert [canonical_resname(n) for n in ("RG5", "g3", "RA", "U5", "DA3", "DT5")] == ["G", "G", "A", "U", "DA", "DT"]
    assert [canonical_atom("G", n) for n in ("O1P", "O2P", "C1*", "N7")] == ["OP1", "OP2", "C1'", "N7"]
    assert canonical_atom("DT", "C5M") == "C7" and canonical_atom("GLN", "OE1") == "OE1"
    assert format_residue_label("G", 7, "", "B") == "G7(B)" and format_residue_label("RG5", 1) == "G1"
    assert format_residue_label("DA", 12) == "DA12"
    assert format_residue_label("GLN", 712) == "Gln712"              # an amino acid: its three letters
    assert not is_nucleotide("GLN") and residue_class("GLN") == "polar" and residue_class("XYZ") == "other"
    # A figure names a nucleotide's atoms as its contacts do: one way throughout.
    assert given_atom("OP2", ["O1P"]) == "O2P" and given_atom("OP2", ["N7"]) == "OP2"
    assert given_atom("C4'", ["O2*"]) == "C4*" and given_atom("C7", ["C5M"]) == "C5M"


def test_what_is_backbone_and_what_is_the_side_chain():
    assert all(is_backbone_atom("G", a) for a in NUCLEOTIDE_BACKBONE) and is_backbone_atom("A", "O1P")
    assert not is_backbone_atom("G", "N7") and not is_backbone_atom("C", "N4")
    # ... of an amino acid, its backbone; of a residue that is neither, the names of an amino acid's backbone.
    assert is_backbone_atom("GLN", "N") and is_backbone_atom("GLN", "OXT") and not is_backbone_atom("GLN", "OE1")
    assert is_backbone_atom("XYZ", "O") and not is_backbone_atom("XYZ", "O1P")


def test_which_atom_of_a_nucleotide_carries_a_hydrogen():
    assert can_donate("G", "N1") and can_donate("G", "N2") and can_donate("A", "N6") and can_donate("U", "N3")
    assert can_donate("A", "O2'") and can_donate("G", "N7") is False and can_donate("A", "O1P") is False
    assert can_donate("G", "O3'") is None and can_donate("G", "O5'") is None      # only where a chain ends
    assert can_donate("XYZ", "N1") is None


# -- which of its atoms are drawn ------------------------------------------------

def test_the_part_that_makes_the_contact_is_drawn_whole():
    base = ("C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4")
    assert fragment_atoms("G", ["N7"]) == base                         # half a guanine is not a guanine
    assert fragment_atoms("G", ["N9", "C8", "N7", "C5", "C4"]) == base  # a ring of it, for a contact of that ring
    assert fragment_atoms("A", ["O1P", "P"]) == ("P", "OP1", "OP2", "O5'", "C5'")
    assert fragment_atoms("U", ["O2'"]) == ("C4'", "O4'", "C1'", "C2'", "O2'", "C3'")
    assert fragment_atoms("C", ["O5'"]) == ("O5'", "C5'")
    both = fragment_atoms("A", ["O1P", "N7"])
    assert set(both) == set(template_for("A").atoms) - {"O2'", "O3'"}  # the sugar, between the two ends
    assert fragment_atoms("G", ["H1"]) is None and fragment_atoms("G", ["OP3"]) is None     # no such atom: a name


def test_a_level_of_detail_adds_to_that_part():
    assert set(fragment_atoms("C", ["P"], "sidechain")) == set(template_for("C").atoms) - {"O2'", "O3'"}
    assert set(fragment_atoms("C", ["N4"], "backbone")) == set(template_for("C").atoms)
    assert set(fragment_atoms("DG", ["N7"], "atoms")) == set(template_for("DG").atoms)


@pytest.mark.parametrize("name", NUCLEOTIDES)
@pytest.mark.parametrize("involved", [("P",), ("OP1", "O3'"), ("C2",), ("O4'",), ("C2", "OP2"), ("O5'", "C2")])
def test_a_fragment_is_one_piece_with_a_place_for_its_name(name, involved):
    frag = build_fragment(name, involved, 30.0)
    assert frag is not None and one_piece(frag) and frag.root in frag.atoms
    assert frag.root == ("C1'" if "C1'" in frag.atoms else "C5'")
    assert set(involved) <= set(frag.atoms) == set(frag.xy) == set(frag.elements)
    assert {frag.elements[a] for a in frag.atoms} <= {"C", "N", "O", "P"}


def test_atoms_are_named_as_the_figure_names_them():
    assert build_fragment("C", ["P"], 30.0).atoms == ("P", "OP1", "OP2", "O5'", "C5'")
    older = build_fragment("C", ["P"], 30.0, None, ["O1P", "N7"])       # another residue's contact says O1P
    assert older.atoms == ("P", "O1P", "O2P", "O5'", "C5'") and older.root == "C5'"
    starred = build_fragment("DT", ["C5M", "O4*"], 30.0)
    assert {"C5M", "O4*", "C1*"} <= set(starred.atoms) and starred.root == "C1*" and one_piece(starred)


# -- a ligand in RNA --------------------------------------------------------------

def residues_of(result) -> dict:
    return {n.label: n.attrs for n in result.graph.nodes if n.kind == "residue"}


def test_the_case_is_what_its_description_says():
    meta = json.loads((CASE / "meta.json").read_text())
    dia = nucleic_diagram()
    assert len(dia.interactions) == 9 and not dia.warnings
    assert {r.record.residue_name for r in dia.interactions} == {"A", "C", "G"}
    assert meta["ligand"]["residue_name_in_cif"] == "BGD" and dia.ligand.mol.GetNumHeavyAtoms() == 12


def test_the_detailed_style_draws_the_part_of_each_nucleotide_that_makes_its_contacts():
    res = nucleic_diagram().render(style="detailed")
    assert res.status == "ok" and res.quality.overlaps == 0 and res.quality.unplaced == 0
    drawn = residues_of(res)
    assert sorted(drawn) == ["A8(A)", "A8(B)", "C9(B)", "G10(B)", "G7(B)"]
    assert all(a["form"] == "fragment" and a["class"] == "nucleotide" for a in drawn.values())
    # Phosphates for the salt bridges, bases for the contacts of rings and of N7: with their names, as the file has them.
    named = nucleic_diagram().render(style="detailed", atom_labels="names").svg
    for words in (">O1P<", ">O2P<", ">N7<", ">O6<", ">N6<"):
        assert words in named and words not in res.svg, words
    assert ">OP1<" not in named and ">OP2<" not in named


def test_the_schematic_style_calls_a_nucleotide_a_nucleotide():
    res = nucleic_diagram().render(style="schematic")
    assert res.status == "ok" and all(a["class"] == "nucleotide" for a in residues_of(res).values())
    for words in (">Nucleotide<", ">H-bond (base)<", ">H-bond (backbone)<"):
        assert words in res.svg, words
    assert ">Other<" not in res.svg and "side chain" not in res.svg


def test_a_donor_that_cannot_be_one_is_said_and_drawn_without_a_direction():
    res = nucleic_diagram().render(style="detailed")
    assert any("O2P: reported as donor" in w for w in res.warnings)      # PLIP's: a phosphate oxygen has no hydrogen


def test_the_colour_of_the_class_is_declared_by_the_figures_that_draw_it_and_by_no_other():
    rna = nucleic_diagram().render(style="schematic", theme="auto").svg
    assert rna.count("--lf-class-nucleotide:") == 3                   # light, dark by the reader's setting, dark by name
    assert "var(--lf-class-nucleotide," in rna
    for case, style in (("kit_imatinib", "schematic"), ("dhfr_methotrexate", "detailed")):
        assert "class-nucleotide" not in render_of(case, style, theme="auto").svg
    dark = nucleic_diagram().render(style="schematic", theme="dark").svg
    assert "#854d0e" in dark and "#f3d477" in nucleic_diagram().render(style="schematic").svg


# -- finding the contacts ----------------------------------------------------------

def test_the_receptor_is_protein_or_nucleic_and_nothing_else():
    from bindview.detect import detect_interactions

    with pytest.raises(ValueError, match="receptor is 'protein' or 'nucleic'"):
        detect_interactions(CASE / "model.pdb", "BGD", receptor="rna")


@needs_plip
def test_plip_finds_the_contacts_once_the_rna_is_the_receptor():
    from plip.basic import config

    from bindview import load_interactions
    from bindview.detect import detect_interactions

    assert detect_interactions(CASE / "model.pdb", "BGD") == []        # protein only: there is none
    found = detect_interactions(CASE / "model.pdb", "BGD", receptor="nucleic")
    assert not config.DNARECEPTOR                                     # the switch is put back
    table = load_interactions(CASE / "interactions.tsv")
    assert [(r.type, r.chain, r.residue_number, r.distance) for r in found] == \
           [(r.type, r.chain, r.residue_number, r.distance) for r in table]
