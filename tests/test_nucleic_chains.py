"""A chain of RNA or DNA is a polymer: the ligand of a diagram, and one side of an interface.

A nucleotide is a residue of a chain by its sugar and by the phosphate that joins it to the next, as an
amino acid is by its backbone. These tests hold that on the two hairpins of the riboswitch, and with one
made-up lysine on the phosphates that have no protein; deposited proteins on DNA and RNA are in
``test_nucleic_deposited.py``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from conftest import TESTDATA, nucleic_chain, nucleic_interface, structure_of
from bindview import Diagram, read_structure
from bindview.backends import Polymer, SmallMolecule
from bindview.chains import chain_smiles
from bindview.distance_table import contact_table, interface_contacts
from bindview.errors import LigandBuildError, LigandNotFoundError
from bindview.structure import Atom, Structure

MODEL = TESTDATA / "nucleic" / "riboswitch_bisguanidine" / "model.pdb"


def only(structure: Structure, keep) -> Structure:
    """The structure with the atoms that ``keep`` says yes to, numbered anew."""
    atoms: list[Atom] = []
    for a in structure.atoms:
        if keep(a):
            atoms.append(replace(a, index=len(atoms)))
    return replace(structure, atoms=atoms, _by_residue={}, _coords=None)


def test_a_chain_of_nucleotides_is_a_polymer_and_two_loose_ones_are_not():
    model = read_structure(MODEL)
    assert [len(model.polymer_residues(c)) for c in "ABC"] == [18, 18, 0]          # C is the ligand
    assert [k.number for k in model.polymer_residues("B")] == list(range(1, 19))
    loose = only(model, lambda a: a.chain != "B" or a.resnum in (2, 9))            # two that are not joined
    assert loose.polymer_residues("B") == []
    joined = only(model, lambda a: a.chain != "B" or a.resnum in (9, 10))
    assert [k.number for k in joined.polymer_residues("B")] == [9, 10]


def test_a_chain_of_rna_is_the_ligand_as_a_chain_of_its_nucleotides():
    dia = nucleic_chain("B")
    assert isinstance(dia.backend, Polymer)
    assert dia.backend.decision.reason == "it is a chain of 18 nucleotides in the structure file"
    chain = dia.backend.peptide
    assert chain.sequence == "GGUGGGGACGACCCCACC" and chain.numbering == "deposited" and not chain.breaks
    assert {r.kind for r in chain.residues} == {"nucleotide"} and not chain.appendages
    assert dia.ligand.bond_source == "template"                     # its bonds are those of the templates
    first, second = chain.residues[:2]
    assert (dia.ligand.names[first.n], dia.ligand.names[first.c], dia.ligand.names[second.n]) == ("1/O5'", "1/O3'", "2/P")
    assert dia.backend.part_of([dia.ligand.index_of("10/N1")]) == "base"
    assert dia.backend.part_of([dia.ligand.index_of("10/P"), dia.ligand.index_of("10/C4'")]) == "backbone"
    figure = dia.render()
    assert figure.status == "ok" and figure.svg.count('class="unit"') == 18
    assert ">5′<" in figure.svg and ">3′<" in figure.svg          # its two ends, not N and C


def test_two_nucleotides_are_drawn_as_atoms():
    model = only(read_structure(MODEL), lambda a: a.chain != "B" or a.resnum in (9, 10))
    dia = Diagram.from_structure(model, "chain:B", interactions=[])
    assert isinstance(dia.backend, SmallMolecule)
    assert dia.backend.decision.reason == "its chain has 2 nucleotides in the structure file, fewer than 3"


def test_the_plain_table_finds_the_bonds_of_a_base_pair():
    model = read_structure(MODEL)
    rows = contact_table(model, "B")
    assert Counter(str(r["interaction_type"]) for r in rows) == {"hbond": 16}
    assert {r["chain"] for r in rows} == {"A"}                       # the ligand of chain C is no partner of a chain
    pair = {(r["ligand_atom"], r["protein_atom"]): r for r in rows if r["residue_number"] == 9 and r["residue_name"] == "C"
            and str(r["ligand_atom"]).startswith("10/")}
    # G10 of one hairpin on C9 of the other: the three bonds of the pair, each with its direction.
    assert pair["10/N1", "N3"]["ligand_is_donor"] == "True" and pair["10/N2", "O2"]["ligand_is_donor"] == "True"
    assert pair["10/O6", "N4"]["ligand_is_donor"] == "False"
    assert all(float(str(r["distance"])) <= 3.5 for r in rows)
    assert len(interface_contacts(model, ["A"], ["B"])) == 16


def test_the_plain_table_says_which_chain_it_cannot_be_of():
    model = read_structure(MODEL)
    with pytest.raises(LigandNotFoundError, match="no chain 'Z'.* its chains are 'A', 'B', 'C'"):
        contact_table(model, "Z")
    with pytest.raises(LigandNotFoundError, match="no chain 'Z'"):
        interface_contacts(model, ["Z"], ["A"])
    # The ligand is a chain of the file, and no chain of residues: its table is the other one.
    with pytest.raises(LigandNotFoundError, match="chain 'C' .*no residue of a protein or a nucleic acid.*ligand_contacts"):
        contact_table(model, "C")


@pytest.mark.parametrize("element", ["", "?"])
def test_an_atom_of_a_chain_whose_element_the_file_does_not_say_is_named(element):
    model = read_structure(MODEL)
    odd = replace(model, _by_residue={}, _coords=None, atoms=[
        replace(a, element=element) if (a.chain, a.resnum, a.name) == ("B", 10, "N7") else a for a in model.atoms])
    merged, key, given, _ = odd.resolve_ligand("chain:B")
    smiles, notes = chain_smiles(merged.residue_atoms(key), given)
    assert smiles is None
    assert notes == [f"G 10 has an atom N7 of no element that is known ({element!r}); bond orders of the ligand are "
                     f"not known"]
    with pytest.raises(LigandBuildError, match="unknown element .* on ligand atom 10/N7"):
        Diagram.from_structure(odd, "chain:B", interactions=[])


def test_the_interface_between_two_chains_of_rna():
    face = nucleic_interface()
    assert [k.number for k in face.site.residues] == [8, 9, 10, 11, 12, 13] and len(face.contacts) == 16
    for style in ("detailed", "schematic"):
        figure = face.render(style=style)
        assert figure.status == "ok" and 'data-class="nucleotide"' in figure.svg
    assert nucleic_interface("B", "A").render().status == "ok"       # and the other hairpin as the column


def _with_a_lysine_on(model: Structure, chain: str, number: int) -> Structure:
    """The model with one lysine of a chain ``K`` whose NZ lies 2.8 A from a phosphate oxygen."""
    key = model.find_residue(chain, number)
    assert key is not None
    p, o = model.atom_by_name(key, "P"), model.atom_by_name(key, "O1P")
    assert p is not None and o is not None
    out = np.array(o.xyz) - np.array(p.xyz)
    out /= np.linalg.norm(out)
    atoms = list(model.atoms)
    last = atoms[-1]
    for step, (name, element) in enumerate((("NZ", "N"), ("CE", "C"), ("CD", "C"), ("CG", "C"), ("CB", "C"), ("CA", "C"),
                                            ("N", "N"), ("C", "C"), ("O", "O"))):
        x, y, z = np.array(o.xyz) + out * (2.8 + 1.5 * step) + (np.array([0.0, 1.2, 0.0]) if name in ("N", "O") else 0.0)
        atoms.append(replace(last, index=len(atoms), name=name, element=element, resname="LYS", chain="K", resnum=1,
                             icode="", x=float(x), y=float(y), z=float(z), hetero=False))
    return replace(model, atoms=atoms, _by_residue={}, _coords=None)


def test_a_phosphate_is_a_charged_group_on_either_side_of_the_table():
    model = _with_a_lysine_on(read_structure(MODEL), "A", 8)
    # The RNA as the chain that is asked about: the bridge is from its phosphate, named as the file names it.
    (bridge,) = [r for r in contact_table(model, "A", partners=["K"]) if r["interaction_type"] == "saltbridge"]
    assert (bridge["residue_name"], bridge["chain"], bridge["protein_atoms"]) == ("LYS", "K", "NZ")
    assert bridge["ligand_atoms"] == "8/O1P,8/O2P" and float(str(bridge["distance"])) == pytest.approx(2.8, abs=0.01)
    # The lysine as the chain that is asked about: the nucleotide is its partner, with the same phosphate.
    bridges = {(r["chain"], r["residue_number"]): r for r in contact_table(model, "K") if r["interaction_type"] == "saltbridge"}
    other = bridges["A", 8]
    assert other["residue_name"] == "A" and float(str(other["distance"])) == pytest.approx(2.8, abs=0.01)
    assert other["protein_atoms"] == "O1P,O2P" and other["ligand_atoms"] == "1/NZ"
    assert all(str(r["protein_atoms"]) == "O1P,O2P" for r in bridges.values())          # never the phosphorus


def test_a_chain_of_amino_acids_is_its_residues_with_a_backbone_and_has_no_nucleotide():
    """The polymer of a protein chain is its residues with the atoms N, CA and C; imatinib and the waters,
    which the file puts under the same chain, are not part of it."""
    model = structure_of("kit_imatinib")
    names: dict = {}
    for a in model.atoms:
        if a.chain == "A":
            names.setdefault((a.resnum, a.icode), set()).add(a.name)
    with_backbone = sorted(k for k, atoms in names.items() if {"N", "CA", "C"} <= atoms)
    assert sorted((k.number, k.icode) for k in model.polymer_residues("A")) == with_backbone
    assert len(with_backbone) < len(names)                   # the ligand and the waters are under chain A as well
    assert model._nucleotides("A") == []
